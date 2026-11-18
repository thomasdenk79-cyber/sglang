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
``griddepcontrol`` instructions are not traced at all and ``launch_pdl=False``
is Triton's own default: the generated PTX and the launch config are identical
to the un-instrumented build.

The JIT CUDA kernels (``kernels/jit/csrc``) that do this on their own via
``PDLWaitPrimary`` / ``PDLTriggerSecondary`` and the upstream Triton kernels
that already carry PDL (``is_arch_support_pdl`` launches) are not affected by
this flag.
"""

from __future__ import annotations

import functools
import os

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

__all__ = ["triton_pdl_enabled", "pdl_wait", "pdl_trigger"]

_FLAG = "SGLANG_TRITON_PDL"
# Keep in sync with EnvBoolCompat in sglang/srt/environ.py; _flag_set reads
# the registry when sglang.srt is importable and mirrors it otherwise.
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


@functools.cache
def _device_supports_pdl() -> bool:
    """Device gate probed on first launch, never at import.

    ``current_arch`` initialises CUDA on its first call, so probing eagerly
    would set the CUDA context before device selection/fork in whatever
    process imports the kernels package first.
    """
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl

        return is_arch_support_pdl()
    except Exception:
        return False


def triton_pdl_enabled() -> bool:
    """``True`` when ``SGLANG_TRITON_PDL`` is set and the device supports griddepcontrol.

    Launch sites call this per launch and pass the result as both
    ``launch_pdl=`` and ``USE_PDL=``, so the launch attribute and the
    ``griddepcontrol`` instructions can never disagree, and a value set after
    import (or an unparsable one) is honoured / reported on the next launch.
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
