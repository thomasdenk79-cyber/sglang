"""CPU tests for the fork's PDL module and the singleton-prune Triton kernel.

The kernel cases run under ``TRITON_INTERPRET=1`` (the CPU CI lane sets it);
they are skipped otherwise because a direct CPU launch has no CUDA backend.
"""

import logging
import os
import sys
import types

import pytest
import torch
import triton

from sglang.kernels import triton_pdl
from sglang.kernels.ops.attention import triton_gdn_fused_proj as gdn
from sglang.kernels.ops.elementwise import elementwise as ew
from sglang.kernels.ops.moe import prune_singleton as ps
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpret = pytest.mark.skipif(
    not INTERPRET, reason="direct CPU kernel launch needs TRITON_INTERPRET=1"
)


def test_prune_tau_canonical_and_hb_alias(monkeypatch):
    """Both spellings configure pruning; the HyperBurst name keeps working."""
    # delenv first: the alias fallback copies the canonical name into
    # os.environ, and only a recorded absence is undone at teardown.
    monkeypatch.delenv("SGLANG_MOE_SINGLETON_PRUNE_TAU", raising=False)
    monkeypatch.delenv("SGLANG_MOE_PRUNE_SINGLETON_TAU", raising=False)

    def tau():
        # get_prune_config is functools.cache'd; read it fresh each time.
        ps.get_prune_config.cache_clear()
        try:
            return ps.get_prune_config().tau
        finally:
            ps.get_prune_config.cache_clear()

    assert tau() == 0.0
    monkeypatch.setenv("SGLANG_MOE_SINGLETON_PRUNE_TAU", "0.08")
    assert tau() == 0.08
    monkeypatch.delenv("SGLANG_MOE_SINGLETON_PRUNE_TAU")
    monkeypatch.setenv("SGLANG_MOE_PRUNE_SINGLETON_TAU", "0.10")
    with pytest.warns(FutureWarning, match="SGLANG_MOE_PRUNE_SINGLETON_TAU"):
        assert tau() == 0.10
    # An unparsable value raises instead of silently falling back to tau=0, on
    # either spelling. Direct os.environ use: the alias path copies the
    # deprecated value under the canonical name outside monkeypatch's tracking.
    for name in (
        "SGLANG_MOE_PRUNE_SINGLETON_TAU",
        "SGLANG_MOE_SINGLETON_PRUNE_TAU",
    ):
        # The previous case left the canonical name set by the fallback copy;
        # start each invalid-value check from a clean environment.
        os.environ.pop("SGLANG_MOE_PRUNE_SINGLETON_TAU", None)
        os.environ.pop("SGLANG_MOE_SINGLETON_PRUNE_TAU", None)
        os.environ[name] = "0,08"
        try:
            ps.get_prune_config.cache_clear()
            with pytest.raises(ValueError, match="SGLANG_MOE_SINGLETON_PRUNE_TAU"):
                ps.get_prune_config()
        finally:
            ps.get_prune_config.cache_clear()
            os.environ.pop(name, None)
            os.environ.pop("SGLANG_MOE_SINGLETON_PRUNE_TAU", None)


def test_triton_pdl_env_registration(monkeypatch):
    """``SGLANG_TRITON_PDL`` is a registered, parsed bool, default off."""
    monkeypatch.delenv("SGLANG_TRITON_PDL", raising=False)
    assert envs.SGLANG_TRITON_PDL.get() is False
    for value, want in (("1", True), ("true", True), ("yes", True), ("0", False)):
        monkeypatch.setenv("SGLANG_TRITON_PDL", value)
        assert envs.SGLANG_TRITON_PDL.get() is want
    # Off must short-circuit before any device query: the CPU CI box has no
    # CUDA and an eager device probe would break every launch. The accessor
    # reads live state, so it stays False here without touching CUDA.
    assert triton_pdl.triton_pdl_enabled() is False


def _stub_device(monkeypatch, supports: bool) -> None:
    monkeypatch.setattr(triton_pdl, "_device_supports_pdl", lambda: supports)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("y", True),
        ("on", True),
        ("ENABLE", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("n", False),
        ("off", False),
        ("disable", False),
    ],
)
def test_pdl_enabled_flag_spellings(monkeypatch, value, want):
    """Registry, standalone fallback and the accessor accept the same words.

    "on"/"enable" are the spellings the ported HyperBurst launch scripts use.
    """
    monkeypatch.setenv("SGLANG_TRITON_PDL", value)
    _stub_device(monkeypatch, True)
    assert triton_pdl.triton_pdl_enabled() is want


@pytest.mark.parametrize("value", ["banana", "2", "0,08", ""])
def test_pdl_invalid_value_raises_on_both_paths(monkeypatch, value):
    """A half-parsed flag is an error, not a silently-off baseline."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", value)
    with pytest.raises(ValueError, match="SGLANG_TRITON_PDL"):
        triton_pdl._flag_set()
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", None)
    with pytest.raises(ValueError, match="SGLANG_TRITON_PDL"):
        triton_pdl._flag_set()


def test_pdl_enabled_requires_flag_and_device(monkeypatch):
    """Flag off wins over a supporting device, and vice versa."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", "0")
    _stub_device(monkeypatch, True)
    assert triton_pdl.triton_pdl_enabled() is False
    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _stub_device(monkeypatch, False)
    assert triton_pdl.triton_pdl_enabled() is False
    _stub_device(monkeypatch, True)
    assert triton_pdl.triton_pdl_enabled() is True


def test_pdl_device_probe_never_raises(monkeypatch):
    """A driver failure on the CUDA probe path must not escape a launch."""

    def boom_arch():
        raise RuntimeError("no driver")

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cuda")
    monkeypatch.setattr("sglang.kernels.arch._cuda.CudaArch", boom_arch)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        assert "RuntimeError" in triton_pdl._PDL_PROBE_REASON
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_pdl_device_probe_is_not_sticky_after_a_failure(monkeypatch):
    """A probe that raised must not silence PDL for the rest of the process.

    The first instrumented launch can happen before the driver is ready; if that
    answer were cached, a ``SGLANG_TRITON_PDL=1`` arm would decode all night
    with the feature silently off.
    """
    state = {"raises": True}
    calls = {"n": 0}

    def arch():
        calls["n"] += 1
        if state["raises"]:
            raise RuntimeError("driver not ready")
        return types.SimpleNamespace(supports_pdl=True)

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cuda")
    monkeypatch.setattr("sglang.kernels.arch._cuda.CudaArch", arch)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        state["raises"] = False
        assert triton_pdl.triton_pdl_enabled() is True
        # A definitive answer is cached: the arch is not probed again.
        state["raises"] = True
        assert triton_pdl.triton_pdl_enabled() is True
        assert calls["n"] == 2
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_flag_read_without_srt_falls_back_to_os_environ(monkeypatch):
    """The kernels package is importable standalone (no sglang.srt on path)."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", "yes")
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", None)
    assert triton_pdl._flag_set() is True


def _prune_case():
    # 16 is the fused shared slot (column 3, never pruned); expert 7 spans
    # rows 0-1; 10/11 tie at 0.05 to pin the column-index tie-break.
    ids = torch.tensor(
        [[7, 3, 4, 16], [7, 5, 9, 16], [10, 11, 12, 16], [-1, -1, -1, 16]],
        dtype=torch.int32,
    )
    w = torch.tensor(
        [
            [0.60, 0.35, 0.05, 0.01],
            [0.50, 0.42, 0.08, 0.01],
            [0.05, 0.05, 0.04, 0.01],
            [0.00, 0.00, 0.00, 0.01],
        ],
        dtype=torch.float32,
    )
    expected_ids = torch.tensor(
        [[7, 3, -1, 16], [7, 5, -1, 16], [10, -1, -1, 16], [-1, -1, -1, 16]],
        dtype=torch.int32,
    )
    expected_w = torch.tensor(
        [
            [0.60, 0.35, 0.0, 0.01],
            [0.50, 0.42, 0.0, 0.01],
            [0.05, 0.0, 0.0, 0.01],
            [0.0, 0.0, 0.0, 0.01],
        ],
        dtype=torch.float32,
    )
    return ids, w, expected_ids, expected_w


def _launch_prune_kernel(ids, w, *, pairwise: bool):
    m, k = ids.shape
    cnt = None if pairwise else torch.zeros(17, dtype=torch.int32)
    ps._kernel()[(1,)](
        ids,
        w,
        cnt if cnt is not None else ids,
        m,
        0.1,  # tau
        1,  # min_rank
        17,  # num_expert_slots
        k - 1,  # routed columns (last column is the fused shared expert)
        K=k,
        MP=triton.next_power_of_2(m),
        KP=triton.next_power_of_2(k),
        SENTINEL_=ps.SENTINEL,
        KEEP_IDS=False,
        PAIRWISE=pairwise,
        num_warps=4,
    )


@needs_interpret
@pytest.mark.parametrize("pairwise", [True, False], ids=["pairwise", "histogram"])
def test_prune_kernel_selection_matches_reference(pairwise):
    """In-place kernel selection equals the torch reference, pinned literals."""
    ids0, w0, exp_ids, exp_w = _prune_case()
    _launch_prune_kernel(ids0, w0, pairwise=pairwise)
    assert torch.equal(ids0, exp_ids) and torch.equal(w0, exp_w)

    ids1, w1 = _prune_case()[:2]
    mask = ps.prune_singleton_mask_reference(
        ids1, w1, tau=0.1, min_rows=1, num_expert_slots=17, num_shared_cols=1
    )
    ps.apply_prune_mask_(
        ids1, w1, mask, keep_ids=False, renorm=False, num_shared_cols=1
    )
    assert torch.equal(ids1, exp_ids) and torch.equal(w1, exp_w)


@needs_interpret
def test_gdn_contiguous_unpack_equals_torch_copy():
    """PDL-instrumented unpack is still a pure split/reshape/cat, both paths."""
    for heads_v in (4, 6):  # v-per-group 2 (pow2) and 3 (head-at-a-time)
        torch.manual_seed(0)
        tokens, heads_qk, head_qk, head_v = 3, 2, 4, 8
        tq, tv = heads_qk * head_qk, heads_v * head_v
        mixed_qkvz = torch.randn(tokens, 2 * tq + 2 * tv, dtype=torch.bfloat16)
        mixed_ba = torch.randn(tokens, 2 * heads_v, dtype=torch.bfloat16)
        qkv, z, b, a = gdn.fused_qkvzba_split_reshape_cat_contiguous(
            mixed_qkvz,
            mixed_ba,
            heads_qk,
            heads_v,
            head_qk,
            head_v,
        )
        assert torch.equal(
            qkv,
            torch.cat(
                [
                    mixed_qkvz[:, :tq],
                    mixed_qkvz[:, tq : 2 * tq],
                    mixed_qkvz[:, 2 * tq : 2 * tq + tv],
                ],
                dim=1,
            ),
        )
        assert torch.equal(
            z, mixed_qkvz[:, 2 * tq + tv :].view(tokens, heads_v, head_v)
        )
        assert torch.equal(b, mixed_ba[:, :heads_v])
        assert torch.equal(a, mixed_ba[:, heads_v:])


@needs_interpret
def test_fused_sigmoid_mul_off_path_matches_torch():
    """The PDL-instrumented gate fusion still equals ``attn * sigmoid(g)``."""
    torch.manual_seed(0)
    attn = torch.randn(3, 64, dtype=torch.float32)
    gate = torch.randn(3, 64, dtype=torch.float32)
    out = ew.fused_sigmoid_mul(attn, gate)
    ref = attn * torch.sigmoid(gate)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# Launch kwargs of the PDL-instrumented sites.  Triton validates every launch
# kwarg against the options class of the selected backend, and only the CUDA
# backend declares ``launch_pdl``: passing it unconditionally would crash the
# HIP path with the flag off, which is the shipped default.
# --------------------------------------------------------------------------


class _LaunchRecorder:
    """Stand in for a JITFunction so a launch's call can be inspected."""

    def __init__(self):
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        def _run(*args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        return _run


def _arm_pdl(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv("SGLANG_TRITON_PDL", "1" if enabled else "0")
    monkeypatch.setattr(triton_pdl, "_device_supports_pdl", lambda: enabled)


def _gemv_args(module, m: int, n: int, k: int):
    torch.manual_seed(3)
    x = torch.randn(m, k).bfloat16()
    w = (torch.randn(n, k) * 0.05).bfloat16()
    ws = torch.empty(module._WS_FLOATS)
    cnt = torch.zeros(module._WS_COUNTERS, dtype=torch.int32)
    y = torch.empty((m, n), dtype=torch.bfloat16)
    cfg = module._plan(m, n, k, False, 2, 188)
    return ws, cnt, x, w, torch.ones(1), y, m, n, k, cfg


def _recorded_launch(monkeypatch, site: str, enabled: bool, *, arch: bool = True):
    """One instrumented site's full launch call, with the kernel stubbed."""
    _arm_pdl(monkeypatch, enabled)
    rec = _LaunchRecorder()
    if site == "prune":
        ids = torch.tensor([[7, 3, 4], [7, 5, 9]], dtype=torch.int32)
        weights = torch.tensor([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]], dtype=torch.float32)
        monkeypatch.setattr(ps, "_kernel", lambda: rec)
        ps._launch_prune_kernel_(
            topk_ids=ids,
            topk_weights=weights,
            cnt=torch.zeros(17, dtype=torch.int32),
            m=2,
            k=3,
            num_expert_slots=17,
            num_shared_cols=0,
            cfg=ps.PruneConfig(0.08, 1, 1, 64, False, False),
        )
    elif site == "gemv":
        from sglang.kernels.ops.gemm import sm120_bf16_skinny_gemv as gemv

        monkeypatch.setattr(gemv, "_w8a16_gemv_kernel", rec)
        ws, cnt, x, w, s, y, m, n, k, cfg = _gemv_args(gemv, 1, 512, 2560)
        gemv._launch(ws, cnt, x, w, s, y, m, n, k, False, cfg)
    elif site == "gdn":
        mixed_qkvz = torch.randn(2, 40, dtype=torch.bfloat16)
        mixed_ba = torch.randn(2, 8, dtype=torch.bfloat16)
        monkeypatch.setattr(
            gdn, "fused_qkvzba_split_reshape_cat_contiguous_kernel", rec
        )
        gdn.fused_qkvzba_split_reshape_cat_contiguous(
            mixed_qkvz, mixed_ba, num_heads_qk=2, num_heads_v=4, head_qk=8, head_v=8
        )
    elif site == "sigmoid-mul":
        monkeypatch.setattr(ew, "is_arch_support_pdl", lambda: arch)
        monkeypatch.setattr(ew, "_fused_sigmoid_mul_kernel", rec)
        ew.fused_sigmoid_mul(
            torch.randn(2, 16, dtype=torch.bfloat16),
            torch.randn(2, 2, 8, dtype=torch.bfloat16),
        )
    else:
        raise ValueError(f"unknown instrumented site {site!r}")
    assert rec.kwargs is not None, f"{site} did not reach the launch"
    return rec


@pytest.mark.parametrize("site", ["prune", "gemv", "gdn", "sigmoid-mul"])
@pytest.mark.parametrize("enabled", [False, True])
def test_instrumented_sites_launch_kwargs(monkeypatch, site, enabled):
    """Wait+trigger sites take launch_pdl/USE_PDL from the flag; sigmoid-mul
    keeps its pre-existing arch-driven launch and gains only the trigger.

    The off arm is the HIP regression: a site that passes ``launch_pdl=False``
    unconditionally raises ``KeyError`` there, so the key must be absent.
    """
    kwargs = _recorded_launch(monkeypatch, site, enabled).kwargs
    if site == "sigmoid-mul":
        # arch is patched on here, so the pre-existing half must stay lit
        # whichever way the flag goes.
        assert kwargs["USE_PDL"] is True
        assert kwargs["launch_pdl"] is True
        assert kwargs["GDC_TRIGGER"] is enabled
        return
    assert kwargs["USE_PDL"] is enabled
    if enabled:
        assert kwargs["launch_pdl"] is True
    else:
        assert "launch_pdl" not in kwargs


def test_sigmoid_mul_flag_without_arch_adds_trigger_only(monkeypatch):
    """On a GPU without PDL launches, the flag must not light launch_pdl.

    Passing launch_pdl where the arch gate says no would either crash the
    launch or make the trigger fire with no waiting primary.
    """
    kwargs = _recorded_launch(monkeypatch, "sigmoid-mul", True, arch=False).kwargs
    assert kwargs["USE_PDL"] is False
    assert "launch_pdl" not in kwargs
    assert kwargs["GDC_TRIGGER"] is True


def test_hip_backend_rejects_the_launch_pdl_kwarg():
    """Reproduce the HIP crash that the omission above prevents.

    Triton validates each launch kwarg against the options object of the
    selected backend, so this is the code path a ROCm decode step takes. The
    CUDA side is pinned by the option-field assertion rather than re-run here:
    ``TRITON_INTERPRET=1`` replaces JITFunction with InterpretedFunction, so the
    real ``_pack_args`` is driven unbound with a signature-less kernel (the
    kwarg check does not read the signature).
    """
    import types

    from triton.backends.amd.compiler import HIPBackend, HIPOptions
    from triton.backends.nvidia.compiler import CUDAOptions
    from triton.runtime.jit import JITFunction

    # The asymmetry is the whole point: CUDA declares it, HIP does not.
    assert "launch_pdl" in CUDAOptions.__dataclass_fields__
    assert "launch_pdl" not in HIPOptions.__dataclass_fields__

    class _Hip:
        # parse_options reads only .target.arch on this input.
        target = types.SimpleNamespace(arch="gfx942")
        parse_options = HIPBackend.parse_options

    kwargs = {"num_warps": 4, "launch_pdl": False}
    hip = _Hip()
    assert "launch_pdl" not in vars(hip.parse_options(kwargs))

    class _Kernel:
        params: list = []

    with pytest.raises(KeyError, match="launch_pdl"):
        JITFunction._pack_args(_Kernel(), hip, kwargs, {}, [], None)


# --------------------------------------------------------------------------
# Device-gate probe semantics: a CPU/default PlatformInfo snapshot is
# missing information, not a verdict, and the first resolved verdict must be
# attributable from the server log.
# --------------------------------------------------------------------------


def _fake_platform(monkeypatch, device_type: str) -> None:
    monkeypatch.setattr(
        triton_pdl.PlatformInfo,
        "detect",
        staticmethod(
            lambda use_cache=True: types.SimpleNamespace(device_type=device_type)
        ),
    )


def test_cpu_snapshot_false_is_retryable_not_a_verdict(monkeypatch):
    """A pre-device CPU probe must not latch PDL off for the process lifetime.

    detect() returns the "cpu" default whenever no CUDA device is visible
    yet, so that snapshot carries no device information; a flag-on worker
    that probes too early must light up on the next instrumented launch
    instead of decoding all night silently without PDL.
    """
    import sglang.kernels.jit.utils as jit_utils

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cpu")
    # A supporting answer from the caching helper: if the probe ever
    # consults it instead of the snapshot, the cpu snapshot must not turn
    # into a PDL verdict.
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", lambda: True)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        _fake_platform(monkeypatch, "cuda")
        monkeypatch.setattr(
            "sglang.kernels.arch._cuda.CudaArch",
            lambda: types.SimpleNamespace(supports_pdl=True),
        )
        assert triton_pdl.triton_pdl_enabled() is True
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_foreign_accelerator_false_is_a_cached_verdict(monkeypatch):
    """A HIP/NPU snapshot means no CUDA launch can happen here: False sticks."""
    import sglang.kernels.jit.utils as jit_utils

    calls = {"n": 0}

    def predicate() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "hip")
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", predicate)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        assert triton_pdl.triton_pdl_enabled() is False
        # The snapshot alone says no CUDA PDL launch can happen here; the
        # caching helper must not be asked for an answer it would latch.
        assert calls["n"] == 0
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_first_resolved_verdict_is_logged_once(monkeypatch, caplog):
    """The arm's PDL state must be attributable from the server log."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cuda")
    monkeypatch.setattr(
        "sglang.kernels.arch._cuda.CudaArch",
        lambda: types.SimpleNamespace(supports_pdl=True),
    )
    triton_pdl._reset_pdl_device_gate()
    caplog.set_level(logging.INFO, logger="sglang.kernels.triton_pdl")
    try:
        assert triton_pdl.triton_pdl_enabled() is True
        assert triton_pdl.triton_pdl_enabled() is True
        lines = [
            r for r in caplog.records if "first instrumented launch" in r.getMessage()
        ]
        assert len(lines) == 1
        assert "device gate=True" in lines[0].getMessage()
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_undecided_gate_warns_once_then_lights_up(monkeypatch, caplog):
    """Operators of a flag-on arm learn about an undecided gate exactly once."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cpu")
    triton_pdl._reset_pdl_device_gate()
    caplog.set_level(logging.WARNING, logger="sglang.kernels.triton_pdl")
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        assert triton_pdl.triton_pdl_enabled() is False
        warns = [r for r in caplog.records if "undecided" in r.getMessage()]
        assert len(warns) == 1
        _fake_platform(monkeypatch, "cuda")
        monkeypatch.setattr(
            "sglang.kernels.arch._cuda.CudaArch",
            lambda: types.SimpleNamespace(supports_pdl=True),
        )
        assert triton_pdl.triton_pdl_enabled() is True
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_platform_probe_leaves_the_shared_platform_cache_untouched(monkeypatch):
    """A pre-device PDL probe must not publish into the process-wide cache.

    ``detect(use_cache=False)`` re-probes *and* writes ``spec._DETECTED_PLATFORM``,
    which the rest of the kernels package reads back with ``use_cache=True``; a
    probe here would republish another component's cached platform as whatever
    this process happens to see right now.
    """
    from sglang.kernels import spec as kernel_spec

    sentinel = kernel_spec.PlatformInfo(
        device_type="cuda", cuda_arch_major=99, cuda_arch_minor=9
    )
    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    monkeypatch.setattr(kernel_spec, "_DETECTED_PLATFORM", sentinel)
    triton_pdl._reset_pdl_device_gate()
    try:
        triton_pdl.triton_pdl_enabled()
        assert kernel_spec._DETECTED_PLATFORM is sentinel
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_pre_device_probe_never_latches_the_arch_cache(monkeypatch):
    """The probe must not freeze sglang.kernels.arch for later CUDA workers.

    current_arch() memoises into arch._DEVICE_ARCH on first call and the
    cache_once'd is_arch_support_pdl routes through it: a probe consulting
    either while no CUDA device is selected pins a plain DeviceArch for the
    life of the process, so every worker forked afterwards would launch
    without PDL even on a sm90 GPU.
    """
    import sglang.kernels.arch as kernel_arch
    import sglang.kernels.jit.utils as jit_utils

    calls = {"n": 0}

    def predicate() -> bool:
        calls["n"] += 1
        return False

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    _fake_platform(monkeypatch, "cpu")
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", predicate)
    monkeypatch.setattr(kernel_arch, "_DEVICE_ARCH", None)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        assert calls["n"] == 0
        assert kernel_arch._DEVICE_ARCH is None
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_gemv_fused_norm_requires_pdl_off(monkeypatch):
    """The PDL build preloads the norm weight, so gate+norm must conflict loudly.

    No production caller passes norm= today; a future one combining both
    would silently read the RMSNorm weight before the primary kernel wrote
    it, because the wait sits below that load.
    """
    from sglang.kernels.ops.gemm import sm120_bf16_skinny_gemv as gemv

    ws, cnt, x, w, s, y, m, n, k, cfg = _gemv_args(gemv, 1, 512, 2560)
    norm = (x, torch.ones(k), 8, 1e-5, False)
    rec = _LaunchRecorder()
    monkeypatch.setattr(gemv, "_w8a16_gemv_kernel", rec)
    _arm_pdl(monkeypatch, True)
    with pytest.raises(RuntimeError, match="SGLANG_TRITON_PDL"):
        gemv._launch(ws, cnt, x, w, s, y, m, n, k, False, cfg, norm=norm)
    assert rec.kwargs is None
    _arm_pdl(monkeypatch, False)
    gemv._launch(ws, cnt, x, w, s, y, m, n, k, False, cfg, norm=norm)
    assert rec.kwargs["NORM_G"] == 8
    assert rec.kwargs["USE_PDL"] is False
    assert "launch_pdl" not in rec.kwargs


# --------------------------------------------------------------------------
# CPU PTX proof: the griddepcontrol instructions are a compile-time
# consequence of the single flag read, taken off the host's real launch call.
# This is what licenses the value-transparency claim for the gpuq PDL arms,
# which share their HiCache namespace with their flag-off twins.
# --------------------------------------------------------------------------

needs_jit_mode = pytest.mark.skipif(
    INTERPRET,
    reason="real PTX codegen needs the non-interpreter JIT; run this file a "
    "second time with TRITON_INTERPRET unset",
)

_DTYPE_TO_PTR = {
    torch.int32: "*i32",
    torch.int64: "*i64",
    torch.float32: "*fp32",
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.uint8: "*u8",
}


def _launch_typestr(v):
    if isinstance(v, torch.Tensor):
        return _DTYPE_TO_PTR[v.dtype]
    if isinstance(v, bool):
        return "i1"
    if isinstance(v, int):
        return "i32"
    if isinstance(v, float):
        return "fp32"
    raise AssertionError(f"launch value {type(v).__name__} has no PTX type")


def _ptx_of_launch(fn, args, kwargs):
    """Compile one recorded launch to CUDA PTX on the CPU, at sm_90.

    Only reached in JIT mode (``needs_jit_mode``): the interpreter lane
    decorates every ``@triton.jit`` as an InterpretedFunction and its own
    kernel launches re-shape ``triton.language`` state, so real codegen must
    run in a process that never entered the interpreter.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature, constexprs = {}, {}
    for i, p in enumerate(fn.params):
        if i < len(args):
            v = args[i]
        elif p.name in kwargs:
            v = kwargs[p.name]
        elif p.has_default:
            continue
        else:
            raise AssertionError(f"no launch value for kernel parameter {p.name}")
        if p.is_constexpr:
            constexprs[p.name] = v
        else:
            signature[p.name] = _launch_typestr(v)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    kern = triton.compile(
        src,
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": kwargs.get("num_warps", 4)},
    )
    return kern.asm["ptx"]


@needs_jit_mode
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("site", ["prune", "gemv", "gdn"])
def test_instrumented_ptx_griddepcontrol_follows_the_flag(monkeypatch, site, enabled):
    """Flag off traces no griddepcontrol; flag on traces wait+trigger, once each.

    The generated PTX, not just the launch kwargs, must follow the single
    flag read: a stray wait with the flag off changes stream semantics for
    every consumer of the compiled kernel cache.
    """
    if site == "prune":
        fn = ps._kernel()
    elif site == "gemv":
        from sglang.kernels.ops.gemm import sm120_bf16_skinny_gemv as gemv

        fn = gemv._w8a16_gemv_kernel
    else:
        fn = gdn.fused_qkvzba_split_reshape_cat_contiguous_kernel
    rec = _recorded_launch(monkeypatch, site, enabled)
    ptx = _ptx_of_launch(fn, rec.args, rec.kwargs)
    if enabled:
        assert ptx.count("griddepcontrol.wait") == 1
        assert ptx.count("griddepcontrol.launch_dependents") == 1
    else:
        assert "griddepcontrol" not in ptx


@needs_jit_mode
@pytest.mark.parametrize(
    ("use_pdl", "trigger", "waits", "trigs"),
    [
        (True, False, 1, 0),
        (True, True, 1, 1),
        (False, False, 0, 0),
        (False, True, 0, 1),
    ],
)
def test_sigmoid_mul_ptx_splits_arch_and_flag(
    monkeypatch, use_pdl, trigger, waits, trigs
):
    """The pre-existing wait follows USE_PDL (arch); the new trigger follows GDC_TRIGGER (flag).

    The host's constexpr split is compiled directly because the quant
    pointers are dead code when QUANT is off but the signature still demands
    concrete types for them.
    """
    ptx = _ptx_of_launch(
        ew._fused_sigmoid_mul_kernel,
        (),
        {
            "output_ptr": torch.empty(2, 16, dtype=torch.bfloat16),
            "attn_output_ptr": torch.randn(2, 16).bfloat16(),
            "gate_ptr": torch.randn(2, 16).bfloat16(),
            "gate_stride_row": 16,
            "gate_stride_head": 16,
            "hidden_dim": 16,
            "HEAD_DIM": 16,
            "BLOCK_H": 16,
            "q_ptr": torch.empty(32, dtype=torch.uint8),
            "s_ptr": torch.empty(2, dtype=torch.float32),
            "QUANT": False,
            "ROUND_SIGMOID": False,
            "USE_PDL": use_pdl,
            "GDC_TRIGGER": trigger,
        },
    )
    assert ptx.count("griddepcontrol.wait") == waits
    assert ptx.count("griddepcontrol.launch_dependents") == trigs
