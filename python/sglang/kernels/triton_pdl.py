# SPDX-License-Identifier: Apache-2.0
# Ported from aiueo52/sglang-rtxpro6000 (HyperBurst) python/sglang/kernels/
# triton_pdl.py (Apache-2.0), patch 0067 of the fork's series.  Deviations:
# the flag is read through sglang.srt.environ.envs so the name registered in
# environ.py is the only parser (os.environ is the fallback when the kernels
# package is used without sglang.srt on the path), and detection is deferred
# to the first instrumented launch instead of being frozen at import.
"""Programmatic Dependent Launch (PDL) for the fork's own Triton kernels.

The decode step runs ~1850 kernels inside CUDA graphs and the GPU is idle for
only ~0.15 ms of the ~11.3 ms, so the cost is not the launch gap but each small
kernel's own ramp: block scheduling plus the first dependent loads. PDL lets
kernel N+1 start scheduling blocks and run its *independent* prologue (weight,
scale and index loads that do not depend on N's output) while N drains, and
block on ``griddepcontrol.wait`` only for the loads that really do depend on N.

Two things are needed per kernel:

* the launch carries ``launch_pdl=True`` (Triton turns that into
  ``cudaLaunchAttributeProgrammaticStreamSerialization`` on the launch config);
* the body calls :func:`pdl_wait` before its first dependent load and
  :func:`pdl_trigger` once the buffers the *next* kernel may overwrite have
  been read for the last time.

Both are gated on :func:`triton_pdl_enabled` (``SGLANG_TRITON_PDL``, default
off; ``on``/``enable`` are accepted spellings, anything unparsable raises at
the launch instead of silently keeping the feature off). When it is false the
``griddepcontrol`` instructions are not traced at all and ``launch_pdl`` keeps
Triton's own default, so the generated PTX and the launch config are identical
to the un-instrumented build.

The one exception is ``ops/elementwise.fused_sigmoid_mul``, which already
PDL-launched behind the pre-existing ``is_arch_support_pdl()`` gate before this
flag existed: there ``launch_pdl`` and the kernel's ``gdc_wait`` stay driven by
that arch check, and ``SGLANG_TRITON_PDL`` gates only the added
``GDC_TRIGGER`` (``griddepcontrol.launch_dependents``, HyperBurst patch 0069).

``launch_pdl`` is omitted rather than passed as ``False`` when the flag is off:
it is declared only by Triton's CUDA backend (``CUDAOptions``), ``HIPOptions``
has no such field, and ``JITFunction._pack_args`` raises ``KeyError`` for a
launch kwarg the selected backend's options do not carry. An unconditional
``launch_pdl=`` therefore breaks every HIP launch, which is why the instrumented
sites spell it ``**({"launch_pdl": True} if pdl else {})`` like the rest of the
tree.

The JIT CUDA kernels (``kernels/jit/csrc``) that do this on their own via
``PDLWaitPrimary`` / ``PDLTriggerSecondary`` and the upstream Triton kernels
that already carry PDL (``is_arch_support_pdl`` launches) are not affected by
this flag.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from sglang.kernels.spec import DeviceType, PlatformInfo

__all__ = ["triton_pdl_enabled", "pdl_wait", "pdl_trigger"]

logger = logging.getLogger(__name__)

_FLAG = "SGLANG_TRITON_PDL"
# Keep in sync with EnvBoolCompat in sglang/srt/environ.py; _flag_set reads
# the registry when sglang.srt is importable and mirrors it otherwise. "on"
# is HyperBurst's spelling (its _detect accepts 1/true/yes/on), "enable" is
# this fork's addition so a launch script that says "enable" does not raise.
_TRUE_SPELLINGS = ("true", "1", "yes", "y", "on", "enable")
_FALSE_SPELLINGS = ("false", "0", "no", "n", "off", "disable")


def _flag_set() -> bool:
    """Read ``SGLANG_TRITON_PDL`` through the env registry, if usable.

    An invalid value raises instead of falling back to the default: an
    operator who set the knob believes PDL is on, so a half-parsed value
    must fail loudly, not launch the baseline.
    """
    try:
        from sglang.srt.environ import envs

        return envs.SGLANG_TRITON_PDL.get()
    except Exception:
        # Standalone kernels package (no sglang.srt on the path) or a
        # half-initialised sglang.srt: mirror the registry's spellings.
        raw = os.environ.get(_FLAG)
        if raw is None:
            return False
        low = raw.lower()
        if low in _TRUE_SPELLINGS:
            return True
        if low in _FALSE_SPELLINGS:
            return False
        raise ValueError(f'"{raw}" is not a valid boolean value for {_FLAG}')


_PDL_DEVICE_VERDICT: Optional[bool] = None
_PDL_PROBE_REASON: Optional[str] = None
_PDL_VERDICT_LOGGED = False
_PDL_PROBE_WARNED = False

# detect() reports these only for a torch-visible foreign accelerator, which
# never hosts a CUDA PDL launch: a False from it is a verdict, not missing
# information. The default "cpu" snapshot is neither: it is what detect()
# returns while no CUDA device has been selected in this process yet.
_FOREIGN_ACCELERATORS = frozenset(
    d.value for d in (DeviceType.HIP, DeviceType.NPU, DeviceType.MLU, DeviceType.MUSA)
)


def _fresh_platform_snapshot() -> PlatformInfo:
    """Probe the platform without publishing into the process-wide cache.

    ``PlatformInfo.detect(use_cache=False)`` re-probes *and* assigns
    ``spec._DETECTED_PLATFORM``, so the fresh snapshot a pre-device probe needs
    would also replace the cached platform that the rest of the kernels package
    reads back with ``use_cache=True``: a CPU-only probe here could leave a
    CUDA worker believing it runs on CPU. Save and restore keeps ``detect()`` as
    the single entry point and leaves the global exactly as found.
    """
    from sglang.kernels import spec as kernel_spec

    saved = kernel_spec._DETECTED_PLATFORM
    try:
        return PlatformInfo.detect(use_cache=False)
    finally:
        kernel_spec._DETECTED_PLATFORM = saved


def _probe_pdl_support() -> Optional[bool]:
    """Ask the platform once; ``None`` means "unknown -- retry next launch".

    The tree's ``is_arch_support_pdl`` is ``cache_once``'d and ``current_arch``
    caches its arch object, so either can latch a pre-device CPU verdict (a
    parent process probed before its TP/DP workers selected a device) as
    ``False`` for the life of the process -- that would let a flag-on arm
    decode all night silently without PDL (review3 B2). Conclusiveness
    therefore comes from a fresh ``PlatformInfo`` snapshot: with CUDA selected,
    ask the arch directly (it reads the live device, no cached layer); with a
    foreign accelerator ``False`` is definitive; with the default CPU snapshot
    a ``False`` is only missing information, so the probe stays retryable.
    """
    global _PDL_PROBE_REASON
    try:
        p = _fresh_platform_snapshot()
        if p.device_type == DeviceType.CUDA.value:
            from sglang.kernels.arch._cuda import CudaArch

            answer = CudaArch().supports_pdl
            _PDL_PROBE_REASON = "cuda capability verdict on the selected device"
            return answer
        from sglang.kernels.jit.utils import is_arch_support_pdl

        answer = bool(is_arch_support_pdl())
        if not answer and p.device_type not in _FOREIGN_ACCELERATORS:
            _PDL_PROBE_REASON = (
                f"platform {p.device_type!r} has no CUDA device selected yet"
            )
            return None
        _PDL_PROBE_REASON = (
            "foreign-accelerator verdict" if answer is False else "device verdict"
        )
        return answer
    except Exception as exc:
        _PDL_PROBE_REASON = f"{type(exc).__name__}: {exc}"
        return None


def _device_supports_pdl() -> bool:
    """Device gate probed on first launch, never at import.

    ``current_arch`` initialises CUDA on its first call, so probing eagerly
    would set the CUDA context before device selection/fork in whatever
    process imports the kernels package first. Only a definitive answer is
    kept: a probe that raised, or one that saw no CUDA device selected yet,
    is retried on the next launch instead of sticking as ``False`` for the
    life of the process, which would otherwise let a flag-on arm run the
    whole decode silently without PDL. The first resolved verdict is logged
    with its reason so an operator can attribute the arm from the server log.
    """
    global _PDL_DEVICE_VERDICT, _PDL_VERDICT_LOGGED, _PDL_PROBE_WARNED
    if _PDL_DEVICE_VERDICT is None:
        _PDL_DEVICE_VERDICT = _probe_pdl_support()
    if _PDL_DEVICE_VERDICT is not None:
        if not _PDL_VERDICT_LOGGED:
            _PDL_VERDICT_LOGGED = True
            logger.info(
                "SGLANG_TRITON_PDL first instrumented launch: device gate=%s "
                "(reason: %s)",
                _PDL_DEVICE_VERDICT,
                _PDL_PROBE_REASON,
            )
    elif not _PDL_PROBE_WARNED:
        # Latched while the gate stays undecided; re-armed by the reset below.
        _PDL_PROBE_WARNED = True
        logger.warning(
            "SGLANG_TRITON_PDL=1 but the device gate is undecided (%s); staying "
            "off and retrying on the next instrumented launch",
            _PDL_PROBE_REASON,
        )
    return _PDL_DEVICE_VERDICT is True


def _reset_pdl_device_gate() -> None:
    """Forget the cached verdict and the log latches; the test seams are
    swapped between cases."""
    global _PDL_DEVICE_VERDICT, _PDL_PROBE_REASON
    global _PDL_VERDICT_LOGGED, _PDL_PROBE_WARNED
    _PDL_DEVICE_VERDICT = None
    _PDL_PROBE_REASON = None
    _PDL_VERDICT_LOGGED = False
    _PDL_PROBE_WARNED = False


def triton_pdl_enabled() -> bool:
    """``True`` when ``SGLANG_TRITON_PDL`` is set and the device supports griddepcontrol.

    The three wait+trigger sites pass the result as both ``launch_pdl=`` and
    ``USE_PDL=``, so the launch attribute and the ``griddepcontrol``
    instructions can never disagree, and a value set after import (or an
    unparsable one) is honoured / reported on the next launch.
    ``ops/elementwise.fused_sigmoid_mul`` predates the flag: only its
    ``GDC_TRIGGER`` follows this read, ``launch_pdl``/``USE_PDL`` stay
    ``is_arch_support_pdl()``-driven there.
    """
    return _flag_set() and _device_supports_pdl()


@triton.jit
def pdl_wait(USE_PDL: tl.constexpr):
    """Block until the preceding kernel in the stream has made its stores visible.

    Everything *before* this in the kernel may read only memory the preceding
    kernel does not write (weights, scales, constants, its own program ids).
    """
    if USE_PDL:
        gdc_wait()


@triton.jit
def pdl_trigger(USE_PDL: tl.constexpr):
    """Allow the next kernel in the stream to start scheduling its blocks.

    Only correct after the last read of every buffer the next kernel may
    overwrite; placing it at the very end of the body is always safe and still
    hides the next kernel's scheduling ramp behind this kernel's drain.
    """
    if USE_PDL:
        gdc_launch_dependents()
