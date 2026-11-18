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


def test_triton_pdl_env_registration(monkeypatch):
    """``SGLANG_TRITON_PDL`` is a registered, parsed bool, default off."""
    monkeypatch.delenv("SGLANG_TRITON_PDL", raising=False)
    assert envs.SGLANG_TRITON_PDL.get() is False
    for value, want in (("1", True), ("true", True), ("yes", True), ("0", False)):
        monkeypatch.setenv("SGLANG_TRITON_PDL", value)
        assert envs.SGLANG_TRITON_PDL.get() is want
    # Off must short-circuit before any device query: the CPU CI box has no
    # CUDA and a True default (or an eager device probe) breaks every import.
    assert triton_pdl.PDL is False
    assert triton_pdl.triton_pdl_enabled() is False


def _patch_cuda(monkeypatch, capability, device_raises=False):
    def current_device():
        if device_raises:
            raise RuntimeError("no CUDA device")
        return 0

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _dev: capability)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "musa", None, raising=False)


@pytest.mark.parametrize(
    "flag,capability,hip,device_raises,want",
    [
        (None, (12, 0), None, False, False),
        ("1", (12, 0), None, False, True),
        ("1", (9, 0), None, False, True),
        ("1", (8, 9), None, False, False),
        ("1", (12, 0), "6.4", False, False),
        ("1", (12, 0), None, True, False),
    ],
)
def test_detect_gate_matrix(monkeypatch, flag, capability, hip, device_raises, want):
    """PDL turns on only for flag-set, non-ROCm/CUDA, sm_90-or-later."""
    if flag is None:
        monkeypatch.delenv("SGLANG_TRITON_PDL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_TRITON_PDL", flag)
    _patch_cuda(monkeypatch, capability, device_raises)
    if hip is not None:
        monkeypatch.setattr(torch.version, "hip", hip)
    assert triton_pdl._detect() is want


def test_flag_read_without_srt_falls_back_to_os_environ(monkeypatch):
    """The kernels package is importable standalone (no sglang.srt on path)."""
    monkeypatch.setenv("SGLANG_TRITON_PDL", "yes")
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", None)
    _patch_cuda(monkeypatch, (12, 0))
    assert triton_pdl._detect() is True


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
