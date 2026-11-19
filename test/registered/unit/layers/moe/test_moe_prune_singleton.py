import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.moe import prune_singleton as ps
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fic
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(**kw):
    base = dict(
        tau=0.1, min_rank=1, min_rows=2, max_rows=64, renorm=False, keep_ids=False
    )
    base.update(kw)
    return ps.PruneConfig(**base)


def _routes():
    # Expert 7 is shared by both rows; 3, 4, 5, 9 are singletons.
    ids = torch.tensor([[7, 3, 4], [7, 5, 9]], dtype=torch.int32)
    w = torch.tensor([[0.60, 0.35, 0.05], [0.50, 0.42, 0.08]])
    return ids, w


def test_reference_prunes_only_light_singletons_below_top_rank():
    ids, w = _routes()
    mask = ps.prune_singleton_mask_reference(ids, w, tau=0.1)
    assert mask.tolist() == [[False, False, True], [False, False, True]]


def test_shared_expert_and_top1_are_never_pruned():
    ids = torch.tensor([[1, 2], [1, 3]], dtype=torch.int32)
    w = torch.tensor([[0.01, 0.009], [0.02, 0.01]])
    mask = ps.prune_singleton_mask_reference(ids, w, tau=1.0)
    # Expert 1 has two routes; rank 0 of each row stays.
    assert mask.tolist() == [[False, True], [False, True]]


def test_fused_shared_columns_and_out_of_range_ids_are_never_pruned():
    # Column 2 is a fused shared slot (id 16); id 99 is beyond the slot count.
    ids = torch.tensor([[1, 2, 16], [3, 99, 16]], dtype=torch.int32)
    w = torch.tensor([[0.9, 0.05, 0.01], [0.9, 0.05, 0.01]])
    mask = ps.prune_singleton_mask_reference(
        ids, w, tau=0.5, min_rows=1, num_expert_slots=17, num_shared_cols=1
    )
    assert mask.tolist() == [[False, True, False], [False, False, False]]
    ps.maybe_prune_singleton_routes_(
        ids, w, 17, _cfg(tau=0.5, min_rows=1), num_shared_cols=1
    )
    assert ids.tolist() == [[1, -1, 16], [3, 99, 16]]


# A fused shared column (last) carrying the LARGEST weight of the row, with all
# three routed routes singleton and below tau. If the shared column competed for
# the row's rank it would hold rank 0 and every routed route would be prunable.
_SHARED_TOP_IDS = [[10, 11, 12, 20], [13, 14, 15, 20]]
_SHARED_TOP_W = [[0.02, 0.02, 0.02, 0.94], [0.03, 0.03, 0.03, 0.91]]


def test_top_weighted_shared_column_leaves_the_routed_top_route():
    ids = torch.tensor(_SHARED_TOP_IDS, dtype=torch.int32)
    w = torch.tensor(_SHARED_TOP_W)
    mask = ps.prune_singleton_mask_reference(
        ids, w, tau=0.08, min_rows=2, num_expert_slots=21, num_shared_cols=1
    )
    # Two of the three routed routes go, never all three.
    assert mask[:, :3].sum(dim=1).tolist() == [2, 2]
    ps.maybe_prune_singleton_routes_(ids, w, 21, _cfg(tau=0.08), num_shared_cols=1)
    assert (ids != -1)[:, :3].sum(dim=1).tolist() == [1, 1]
    assert ids[:, 3].tolist() == [20, 20]
    assert w[:, 3].tolist() == pytest.approx([0.94, 0.91])


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="direct CPU kernel launch needs TRITON_INTERPRET=1",
)
def test_kernel_leaves_the_routed_top_route_with_a_top_weighted_shared_column():
    # The reference above is not enough: the kernel computes rank independently,
    # and it is the kernel that ships.
    ids = torch.tensor(_SHARED_TOP_IDS, dtype=torch.int32)
    w = torch.tensor(_SHARED_TOP_W)
    ps._launch_prune_kernel_(
        topk_ids=ids,
        topk_weights=w,
        cnt=None,
        m=2,
        k=4,
        num_expert_slots=21,
        num_shared_cols=1,
        cfg=_cfg(tau=0.08),
    )
    assert (ids != -1)[:, :3].sum(dim=1).tolist() == [1, 1]
    assert ids[:, 3].tolist() == [20, 20]
    assert w[:, 3].tolist() == pytest.approx([0.94, 0.91])


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="direct CPU kernel launch needs TRITON_INTERPRET=1",
)
def test_kernel_matches_reference_bitwise_with_pdl_off():
    """With the flag off the shipped kernel must give exactly what the
    un-instrumented reference path gives; the PTX check only proves the codegen
    differs by the gdc intrinsics.
    """
    cases = [
        (
            torch.tensor([[7, 3, 4], [7, 5, 9]], dtype=torch.int32),
            torch.tensor([[0.60, 0.35, 0.05], [0.50, 0.42, 0.08]]),
            8,
            0,
        ),
        (
            torch.tensor(_SHARED_TOP_IDS, dtype=torch.int32),
            torch.tensor(_SHARED_TOP_W),
            21,
            1,
        ),
    ]
    for ids0, w0, slots, nshared in cases:
        cfg = _cfg(tau=0.08)
        ref_ids, ref_w = ids0.clone(), w0.clone()
        mask = ps.prune_singleton_mask_reference(
            ref_ids,
            ref_w,
            tau=cfg.tau,
            min_rows=cfg.min_rows,
            num_expert_slots=slots,
            num_shared_cols=nshared,
        )
        ps.apply_prune_mask_(ref_ids, ref_w, mask, cfg.keep_ids, cfg.renorm, nshared)
        kid, kw = ids0.clone(), w0.clone()
        ps._launch_prune_kernel_(
            topk_ids=kid,
            topk_weights=kw,
            cnt=None,
            m=kid.shape[0],
            k=kid.shape[1],
            num_expert_slots=slots,
            num_shared_cols=nshared,
            cfg=cfg,
        )
        assert torch.equal(kid, ref_ids), (kid, ref_ids)
        assert torch.equal(kw, ref_w), (kw, ref_w)


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="direct CPU kernel launch needs TRITON_INTERPRET=1",
)
def test_histogram_scratch_is_left_zeroed_after_launch():
    """The histogram scratch is a cached, reused buffer, so the kernel must
    hand it back zeroed: one stale count would mis-classify the next batch's
    singletons (a pruned route stays unpruned, or vice versa).
    """
    ids0 = torch.tensor([[7, 3, 4], [7, 5, 9]], dtype=torch.int32)
    w0 = torch.tensor([[0.60, 0.35, 0.05], [0.50, 0.42, 0.08]])
    cfg = _cfg(tau=0.08)
    results = []
    for _ in range(2):
        ids, w = ids0.clone(), w0.clone()
        cnt = torch.zeros(8, dtype=torch.int32)
        ps._launch_prune_kernel_(
            topk_ids=ids,
            topk_weights=w,
            cnt=cnt,
            m=2,
            k=3,
            num_expert_slots=8,
            num_shared_cols=0,
            cfg=cfg,
        )
        assert torch.count_nonzero(cnt) == 0, cnt.tolist()
        results.append((ids.tolist(), w.tolist()))
    assert results[0][0] == [[7, 3, -1], [7, 5, 9]]
    assert results[0] == results[1], "a reused scratch changed the prune decision"


def test_padded_rows_are_ignored_and_not_counted():
    ids = torch.tensor([[7, 3], [-1, -1], [5, 9]], dtype=torch.int32)
    w = torch.tensor([[0.9, 0.05], [0.0, 0.0], [0.9, 0.05]])
    mask = ps.prune_singleton_mask_reference(ids, w, tau=0.1)
    assert mask.tolist() == [[False, True], [False, False], [False, True]]


def test_batch_bounds_disable_pruning():
    ids, w = _routes()
    assert not ps.prune_singleton_mask_reference(ids[:1], w[:1], tau=0.1).any()
    assert not ps.prune_singleton_mask_reference(ids, w, tau=0.1, max_rows=1).any()
    assert not ps.prune_singleton_mask_reference(ids, w, tau=0.0).any()


def test_rank_tie_breaks_by_column():
    w = torch.tensor([[0.5, 0.5, 0.2]])
    assert ps._within_row_rank(w).tolist() == [[0, 1, 2]]


def test_cpu_apply_in_place_with_sentinel_and_weight_zero():
    ids, w = _routes()
    ps.maybe_prune_singleton_routes_(ids, w, 16, _cfg())
    assert ids.tolist() == [[7, 3, -1], [7, 5, -1]]
    assert w[:, 2].tolist() == [0.0, 0.0]
    assert w[0, :2].tolist() == pytest.approx([0.60, 0.35])


def test_keep_ids_and_renorm():
    ids, w = _routes()
    ps.maybe_prune_singleton_routes_(ids, w, 16, _cfg(keep_ids=True, renorm=True))
    assert ids.tolist() == [[7, 3, 4], [7, 5, 9]]
    assert w[:, 2].tolist() == [0.0, 0.0]
    assert w.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_renorm_keeps_row_scale_and_leaves_unpruned_rows_alone():
    # Weights already carry a routed scaling factor (sums 2.5); the last
    # column is a fused shared slot whose weight must not move.
    ids = torch.tensor([[7, 3, 20], [7, 8, 20], [8, 9, 20]], dtype=torch.int32)
    w = torch.tensor([[2.0, 0.5, 1.0], [2.0, 0.5, 1.0], [2.4, 0.1, 1.0]])
    ps.maybe_prune_singleton_routes_(
        ids, w, 21, _cfg(tau=0.3, renorm=True), num_shared_cols=1
    )
    assert ids.tolist() == [[7, 3, 20], [7, 8, 20], [8, -1, 20]]
    assert w[:2].tolist() == [[2.0, 0.5, 1.0], [2.0, 0.5, 1.0]]
    assert w[2].tolist() == pytest.approx([2.5, 0.0, 1.0])


def test_disabled_config_is_noop():
    ids, w = _routes()
    ids0, w0 = ids.clone(), w.clone()
    ps.maybe_prune_singleton_routes_(ids, w, 16, _cfg(tau=0.0))
    assert torch.equal(ids, ids0) and torch.equal(w, w0)


@pytest.mark.parametrize(
    "env",
    [
        {"SGLANG_MOE_PRUNE_MIN_ROWS": "8", "SGLANG_MOE_PRUNE_MAX_ROWS": "4"},
        # min_rank 0 could strip a live row of every route.
        {"SGLANG_MOE_PRUNE_MIN_RANK": "0"},
        {"SGLANG_MOE_PRUNE_SINGLETON_TAU": "1.5"},
    ],
)
def test_invalid_env_settings_raise(monkeypatch, env):
    monkeypatch.delenv("SGLANG_MOE_SINGLETON_PRUNE_TAU", raising=False)
    monkeypatch.setenv("SGLANG_MOE_PRUNE_SINGLETON_TAU", "0.1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ps.get_prune_config.cache_clear()
    try:
        with pytest.raises(ValueError):
            ps.get_prune_config()
    finally:
        ps.get_prune_config.cache_clear()


def _runner(**kw):
    base = dict(
        moe_ep_size=1,
        moe_tp_size=1,
        activation="silu",
        num_experts=16,
        num_fused_shared_experts=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "runner,alltoall,expect_pruned",
    [
        (_runner(), False, True),
        (_runner(moe_ep_size=2), False, False),
        (_runner(moe_tp_size=2), False, False),
        (_runner(num_experts=None), False, False),
        (_runner(activation="relu2"), False, False),
        (_runner(), True, False),
    ],
)
def test_runner_hook_gating(runner, alltoall, expect_pruned):
    ids, w = _routes()
    with mock.patch.object(ps, "get_prune_config", return_value=_cfg()):
        fic._maybe_prune_singleton_routes(ids, w, runner, alltoall)
    assert (ids[0, 2].item() == -1) is expect_pruned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m,k", [(2, 3), (5, 10), (16, 10), (33, 8)])
@pytest.mark.parametrize("shared", [0, 1])
def test_kernel_matches_reference(m, k, shared):
    g = torch.Generator().manual_seed(m * 100 + k)
    # 64 is out of range for 64 slots; row 1 is upstream padding.
    ids = torch.randint(0, 65, (m, k), generator=g, dtype=torch.int32)
    ids[1] = -1
    w = torch.rand(m, k, generator=g).softmax(dim=-1)
    w[1] = 0.0
    expected = ps.prune_singleton_mask_reference(
        ids, w, 0.12, num_expert_slots=64, num_shared_cols=shared
    )
    ids_d, w_d = ids.cuda(), w.cuda()
    ps.maybe_prune_singleton_routes_(
        ids_d, w_d, 64, _cfg(tau=0.12), num_shared_cols=shared
    )
    want_ids = ids.masked_fill(expected, -1)
    assert torch.equal(ids_d.cpu(), want_ids)
    assert torch.equal(w_d.cpu(), w.masked_fill(expected, 0.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
