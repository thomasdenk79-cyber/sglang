"""CPU tests for the fork's PDL module and the singleton-prune Triton kernel.

The kernel cases run under ``TRITON_INTERPRET=1`` (the CPU CI lane sets it);
they are skipped otherwise because a direct CPU launch has no CUDA backend.
"""

import os
import sys

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
    """A driver failure in the tree-wide predicate must not escape a launch."""
    import sglang.kernels.jit.utils as jit_utils

    def boom() -> bool:
        raise RuntimeError("no driver")

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", boom)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
    finally:
        triton_pdl._reset_pdl_device_gate()


def test_pdl_device_probe_is_not_sticky_after_a_failure(monkeypatch):
    """A probe that raised must not silence PDL for the rest of the process.

    The first instrumented launch can happen before the driver is ready; if that
    answer were cached, a ``SGLANG_TRITON_PDL=1`` arm would decode all night
    with the feature silently off.
    """
    import sglang.kernels.jit.utils as jit_utils

    state = {"raises": True}

    def predicate() -> bool:
        if state["raises"]:
            raise RuntimeError("driver not ready")
        return True

    monkeypatch.setenv("SGLANG_TRITON_PDL", "1")
    monkeypatch.setattr(jit_utils, "is_arch_support_pdl", predicate)
    triton_pdl._reset_pdl_device_gate()
    try:
        assert triton_pdl.triton_pdl_enabled() is False
        state["raises"] = False
        assert triton_pdl.triton_pdl_enabled() is True
        # A definitive answer is cached: the predicate is not consulted again.
        state["raises"] = True
        assert triton_pdl.triton_pdl_enabled() is True
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
    """Stand in for a JITFunction so a launch's kwargs can be inspected."""

    def __init__(self):
        self.kwargs = None

    def __getitem__(self, grid):
        def _run(*args, **kwargs):
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


def _recorded_launch_kwargs(monkeypatch, site: str, enabled: bool) -> dict:
    """Launch kwargs of one instrumented site, with the kernel call stubbed."""
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
        monkeypatch.setattr(gdn, "fused_qkvzba_split_reshape_cat_contiguous_kernel", rec)
        gdn.fused_qkvzba_split_reshape_cat_contiguous(
            mixed_qkvz, mixed_ba, num_heads_qk=2, num_heads_v=4, head_qk=8, head_v=8
        )
    else:
        raise ValueError(f"unknown instrumented site {site!r}")
    assert rec.kwargs is not None, f"{site} did not reach the launch"
    return rec.kwargs


@pytest.mark.parametrize("site", ["prune", "gemv", "gdn"])
@pytest.mark.parametrize("enabled", [False, True])
def test_instrumented_sites_launch_kwargs(monkeypatch, site, enabled):
    """Flag off omits ``launch_pdl``; flag on sets it, both from one read.

    The off arm is the HIP regression: a site that passes ``launch_pdl=False``
    unconditionally raises ``KeyError`` there, so the key must be absent.
    """
    kwargs = _recorded_launch_kwargs(monkeypatch, site, enabled)
    assert kwargs["USE_PDL"] is enabled
    if enabled:
        assert kwargs["launch_pdl"] is True
    else:
        assert "launch_pdl" not in kwargs


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

    from triton.backends.amd.compiler import HIPOptions, HIPBackend
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
