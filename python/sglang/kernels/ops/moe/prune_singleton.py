# SPDX-License-Identifier: Apache-2.0
"""Opt-in, lossy MoE route pruning for small decode/verify batches.

The routed-expert grouped GEMM of a small batch is bound by expert weight
reads, so every *distinct* expert costs a full weight read. A route that is the
only route to its expert in the current MoE call (a singleton) costs that whole
read for one row; a route to an expert another row also uses costs nothing
extra. ``SGLANG_MOE_SINGLETON_PRUNE_TAU`` (deprecated alias:
``SGLANG_MOE_PRUNE_SINGLETON_TAU``) masks every route that is

* the only route to its expert within the call,
* below ``tau`` in routing weight, and
* not among the row's top ``SGLANG_MOE_PRUNE_MIN_RANK`` (>= 1) routes, and
* not a fused shared-expert column,

for batches of ``[MIN_ROWS, MAX_ROWS]`` rows. A masked route gets expert id
``-1`` (unless ``KEEP_IDS``) and weight 0. FlashInfer's CUTLASS MoE and the
SM120 SwiGLU kernels skip expert ids outside ``[0, num_experts)``, so nothing
downstream changes; ``-1`` also cannot collide with a fused shared-expert slot.
Every row keeps its top route, so no live row loses all of its experts. Ids
outside ``[0, num_expert_slots)`` are neither counted nor pruned. Rows that
upstream masked as padding (all ids ``-1``) are ignored; unmasked stale rows of
a padded graph bucket still count. ``RENORM`` rescales the surviving routed
weights of a pruned row back to that row's routed weight sum before pruning;
rows without a pruned route are unchanged. HyperBurst's ``RENORM`` instead
divides the survivors by their own weight sum; as in HB, the default here is
off, so a pruned row keeps the surviving weights as-is and loses the dropped
routing mass.

The work is one single-CTA Triton kernel over the ``[M, k]`` top-k tensors, in
place, without host sync or allocation, so it is CUDA-graph capturable once
the histogram scratch exists. The launch honours ``SGLANG_TRITON_PDL`` (see
``sglang.kernels.triton_pdl``); with the flag off the PTX and the launch
config are identical to the un-instrumented kernel.
It changes the model output (lossy); defaults
leave it off.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.triton_pdl import PDL, pdl_trigger, pdl_wait

logger = logging.getLogger(__name__)

SENTINEL = -1


@dataclass(frozen=True)
class PruneConfig:
    tau: float
    min_rank: int
    min_rows: int
    max_rows: int
    renorm: bool
    keep_ids: bool

    @property
    def enabled(self) -> bool:
        return self.tau > 0.0


@functools.cache
def get_prune_config() -> PruneConfig:
    from sglang.srt.environ import envs

    cfg = PruneConfig(
        tau=envs.SGLANG_MOE_SINGLETON_PRUNE_TAU.get(),
        min_rank=envs.SGLANG_MOE_PRUNE_MIN_RANK.get(),
        min_rows=envs.SGLANG_MOE_PRUNE_MIN_ROWS.get(),
        max_rows=envs.SGLANG_MOE_PRUNE_MAX_ROWS.get(),
        renorm=envs.SGLANG_MOE_PRUNE_RENORM.get(),
        keep_ids=envs.SGLANG_MOE_PRUNE_KEEP_IDS.get(),
    )
    if (
        not 0.0 <= cfg.tau <= 1.0
        or cfg.min_rank < 1
        or not 1 <= cfg.min_rows <= cfg.max_rows
    ):
        raise ValueError(f"Invalid SGLANG_MOE_PRUNE_* settings: {cfg}")
    if cfg.enabled:
        logger.warning("Lossy MoE singleton route pruning enabled: %s", cfg)
    return cfg


def _within_row_rank(topk_weights: torch.Tensor) -> torch.Tensor:
    """0-based rank by weight, larger first; the column index breaks ties."""
    w = topk_weights.float()
    k = w.shape[1]
    col = torch.arange(k, device=w.device)
    gt = (w[:, None, :] > w[:, :, None]) | (
        (w[:, None, :] == w[:, :, None]) & (col[None, None, :] < col[None, :, None])
    )
    return gt.sum(dim=2)


def prune_singleton_mask_reference(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tau: float,
    min_rank: int = 1,
    min_rows: int = 2,
    max_rows: int = 64,
    num_expert_slots: Optional[int] = None,
    num_shared_cols: int = 0,
) -> torch.Tensor:
    """Pure torch decision used by the CPU tests and as a kernel reference."""
    m, k = topk_ids.shape
    if tau <= 0.0 or m < min_rows or m > max_rows:
        return torch.zeros_like(topk_ids, dtype=torch.bool)
    ids = topk_ids.long()
    valid = ids >= 0
    if num_expert_slots is not None:
        valid &= ids < num_expert_slots
    flat = ids[valid]
    counts = torch.zeros(
        int(flat.max().item()) + 1 if flat.numel() else 1,
        dtype=torch.long,
        device=ids.device,
    )
    counts.index_add_(0, flat, torch.ones_like(flat))
    cnt = torch.zeros_like(ids)
    cnt[valid] = counts[flat]
    return (
        valid
        & (cnt == 1)
        & (topk_weights.float() < tau)
        & (_within_row_rank(topk_weights) >= min_rank)
        & (torch.arange(k, device=ids.device) < k - num_shared_cols)[None, :]
    )


def _renorm_pruned_rows_(
    topk_weights: torch.Tensor, routed_sum_before: torch.Tensor, routed: int
) -> None:
    # Unpruned rows sum the same values in the same order, so their ratio is 1.
    w = topk_weights[:, :routed]
    w.mul_(routed_sum_before / w.sum(dim=-1, keepdim=True).clamp_min(1e-20))


def apply_prune_mask_(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    mask: torch.Tensor,
    keep_ids: bool = False,
    renorm: bool = False,
    num_shared_cols: int = 0,
) -> None:
    routed = topk_weights.shape[1] - num_shared_cols
    before = topk_weights[:, :routed].sum(dim=-1, keepdim=True) if renorm else None
    if not keep_ids:
        topk_ids.masked_fill_(mask, SENTINEL)
    topk_weights.masked_fill_(mask, 0.0)
    if renorm:
        _renorm_pruned_rows_(topk_weights, before, routed)


_COUNTS: Dict[Tuple[torch.device, int, int], torch.Tensor] = {}
_warned: set = set()


def _skip(reason: str) -> None:
    if reason not in _warned:
        _warned.add(reason)
        logger.warning("MoE singleton prune skipped: %s", reason)


@functools.cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _prune_singleton_kernel(
        ids_ptr,
        w_ptr,
        cnt_ptr,
        M,
        TAU,
        MIN_RANK,
        NUM_SLOTS,
        K_ROUTED,
        K: tl.constexpr,
        MP: tl.constexpr,
        KP: tl.constexpr,
        SENTINEL_: tl.constexpr,
        KEEP_IDS: tl.constexpr,
        PAIRWISE: tl.constexpr,
        USE_PDL: tl.constexpr = False,
    ):
        # ids/weights are the preceding routing kernel's output, so every load
        # below is dependent: the wait only buys the block-scheduling ramp.
        pdl_wait(USE_PDL)
        r = tl.arange(0, MP)
        c = tl.arange(0, KP)
        off = r[:, None] * K + c[None, :]
        m = (r[:, None] < M) & (c[None, :] < K)
        ids = tl.load(ids_ptr + off, mask=m, other=SENTINEL_)
        w = tl.load(w_ptr + off, mask=m, other=0.0).to(tl.float32)
        valid = m & (ids >= 0) & (ids < NUM_SLOTS)

        gt = (w[:, None, :] > w[:, :, None]) | (
            (w[:, None, :] == w[:, :, None]) & (c[None, None, :] < c[None, :, None])
        )
        rank = tl.sum((gt & m[:, None, :]).to(tl.int32), axis=2)

        if PAIRWISE:
            cnt = tl.zeros([MP, KP], tl.int32)
            for rp in tl.static_range(MP):
                orow = tl.load(
                    ids_ptr + rp * K + c, mask=(c < K) & (rp < M), other=SENTINEL_
                )
                hit = (ids[:, :, None] == orow[None, None, :]) & (
                    orow[None, None, :] >= 0
                )
                cnt += tl.sum(hit.to(tl.int32), axis=2)
        else:
            # The histogram is always left zeroed: add, barrier, read through
            # L2 (.cg) so the atomics are visible, barrier, then reset by xchg.
            tl.atomic_add(cnt_ptr + ids, 1, mask=valid)
            tl.debug_barrier()
            cnt = tl.load(cnt_ptr + ids, mask=valid, other=0, cache_modifier=".cg")
            tl.debug_barrier()
            tl.atomic_xchg(cnt_ptr + ids, 0, mask=valid)

        prune = (
            valid
            & (cnt == 1)
            & (w < TAU)
            & (rank >= MIN_RANK)
            & (c[None, :] < K_ROUTED)
        )
        tl.debug_barrier()
        if not KEEP_IDS:
            tl.store(ids_ptr + off, tl.where(prune, SENTINEL_, ids), mask=m)
        tl.store(
            w_ptr + off,
            tl.where(prune, 0.0, w).to(w_ptr.dtype.element_ty),
            mask=m,
        )
        pdl_trigger(USE_PDL)

    return _prune_singleton_kernel


def maybe_prune_singleton_routes_(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_expert_slots: int,
    config: Optional[PruneConfig] = None,
    num_shared_cols: int = 0,
) -> None:
    """Prune in place when enabled and the batch is in range; else no-op.

    ``num_expert_slots`` bounds the expert ids that are counted (fused shared
    expert slots included); the last ``num_shared_cols`` columns are fused
    shared experts and are never pruned.
    """
    cfg = config if config is not None else get_prune_config()
    if not cfg.enabled:
        return
    if topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape:
        return _skip(f"shape {tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}")
    m, k = topk_ids.shape
    if m < cfg.min_rows or m > cfg.max_rows or not 0 <= num_shared_cols < k:
        return
    if not (topk_ids.is_contiguous() and topk_weights.is_contiguous()):
        return _skip("non-contiguous top-k tensors")
    if not topk_ids.is_cuda:
        mask = prune_singleton_mask_reference(
            topk_ids,
            topk_weights,
            cfg.tau,
            cfg.min_rank,
            cfg.min_rows,
            cfg.max_rows,
            num_expert_slots,
            num_shared_cols,
        )
        apply_prune_mask_(
            topk_ids, topk_weights, mask, cfg.keep_ids, cfg.renorm, num_shared_cols
        )
        return

    import triton

    before = (
        topk_weights[:, : k - num_shared_cols].sum(dim=-1, keepdim=True)
        if cfg.renorm
        else None
    )
    # Per stream, so launches on different streams never share counters.
    key = (
        topk_ids.device,
        torch.cuda.current_stream(topk_ids.device).cuda_stream,
        num_expert_slots,
    )
    cnt = _COUNTS.get(key)
    if cnt is None:
        # Never allocate inside a graph capture (it would land in the graph's
        # private pool); the eager warmup forwards run first, pairwise is the
        # safety net.
        if torch.cuda.is_current_stream_capturing():
            _skip("histogram scratch not allocated before capture; using pairwise")
        else:
            cnt = torch.zeros(
                num_expert_slots, dtype=torch.int32, device=topk_ids.device
            )
            _COUNTS[key] = cnt
    _kernel()[(1,)](
        topk_ids,
        topk_weights,
        cnt if cnt is not None else topk_ids,
        m,
        cfg.tau,
        cfg.min_rank,
        num_expert_slots,
        k - num_shared_cols,
        K=k,
        MP=triton.next_power_of_2(m),
        KP=triton.next_power_of_2(k),
        SENTINEL_=SENTINEL,
        KEEP_IDS=cfg.keep_ids,
        PAIRWISE=cnt is None,
        USE_PDL=PDL,
        launch_pdl=PDL,
        num_warps=4,
    )
    if before is not None:
        _renorm_pruned_rows_(topk_weights, before, k - num_shared_cols)
