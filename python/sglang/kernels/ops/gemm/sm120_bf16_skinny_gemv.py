# SPDX-License-Identifier: Apache-2.0
# Kernel, tile tables and split-K fixup ported from jpezzulli/pennyroyal
# python/sglang/srt/layers/quantization/w8a16_gemv.py (Apache-2.0), itself a
# direct port of aiueo52/sglang-rtxpro6000 @5105985116eb with the per-shape tile
# table and split-K scratch of aiueo52/flash-next-rtxpro6000 (patches 0026, 0041).
# `_store_split`, `_w8a16_gemv_kernel`, `_plan`, `_fit` and the tables are
# unchanged except the counter reset, now an atomic exchange like the in-tree
# router_gemv / lora_a fixups. Deviations: only the BF16 entry point is exposed;
# PDL is wired through ``sglang.kernels.triton_pdl`` (``USE_PDL`` and
# ``launch_pdl`` both follow ``SGLANG_TRITON_PDL``, default off); the split-K
# scratch is keyed by (device, stream) instead of the donor's manual slot, and
# a scratch miss during graph capture falls back to the caller's GEMM instead
# of allocating in the graph pool.
"""Triton skinny GEMM for small decode/verify batches (M <= 16) on SM120.

y[M, N] = x[M, K] @ W[N, K]^T with bf16 x and W, fp32 accumulation and bf16
output. The split-K reduction is folded into the launch: every CTA writes an
fp32 partial and bumps a per-N-block counter, the CTA that sees the last
increment sums, writes y and resets the counter, so the kernel is graph safe
without a memset.

Scratch contract (as for router_gemv): eager launches on different streams get
different scratch, but a captured launch bakes in the scratch of its capture
stream, and all graphs of a process are captured on one stream. Graphs that
contain this kernel must therefore not be replayed concurrently; the decode
and draft graphs replay in order on the forward stream.

``SGLANG_SM120_BF16_SKINNY_GEMV`` routes unquantized BF16 linears of the shapes
measured faster than cuBLAS on RTX PRO 6000 (GDN in_proj_ba, MoE router gate,
MTP fc) here.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.triton_pdl import pdl_trigger, pdl_wait, triton_pdl_enabled


@triton.jit
def _store_split(
    acc,
    y_ptr,
    y2_ptr,
    offs_m,
    offs_n,
    stride_ym,
    stride_yn,
    stride_y2m,
    m_mask,
    n_mask,
    seg1,
    SPLIT_N: tl.constexpr,
    SEG_NB0: tl.constexpr,
):
    """Write the rounded accumulator to `y`, or to `y`/`y2` split at column SPLIT_N.

    SPLIT_N == 0 is the default single-destination store, byte-for-byte the
    original epilogue. With SPLIT_N > 0 columns [0, SPLIT_N) land in `y` and
    columns [SPLIT_N, N) land in `y2` at column `n - SPLIT_N`; both stores see
    the same rounded value, so the numerics are identical either way.

    SEG_NB0 > 0 (the re-gridded form, see BA_SPLIT_GRID) means the caller has
    already restricted this CTA's columns to one side of the split and passes
    which side in `seg1`, so exactly one store is emitted.
    """
    # Round to bf16 first even when the destination is fp32: an fp32 destination
    # is only ever the shared logits buffer, whose old contents were a bf16
    # result widened by `.copy_()`. Rounding here keeps that bit-identical.
    val = acc.to(tl.bfloat16).to(y_ptr.dtype.element_ty)
    if SEG_NB0 > 0:
        if seg1:
            tl.store(
                y2_ptr
                + offs_m[:, None] * stride_y2m
                + (offs_n - SPLIT_N)[None, :] * stride_yn,
                val,
                mask=m_mask[:, None] & n_mask[None, :],
            )
        else:
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                val,
                mask=m_mask[:, None] & n_mask[None, :],
            )
    elif SPLIT_N > 0:
        lo = offs_n < SPLIT_N
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :] & lo[None, :],
        )
        n2 = tl.maximum(offs_n - SPLIT_N, 0)
        tl.store(
            y2_ptr + offs_m[:, None] * stride_y2m + n2[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :] & (offs_n >= SPLIT_N)[None, :],
        )
    else:
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            val,
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    y2_ptr,
    ws_ptr,
    cnt_ptr,
    z_ptr,
    nw_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    stride_y2m,
    stride_zm,
    norm_eps,
    SPLIT_N: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    W_KN: tl.constexpr,
    USE_DOT: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    SEG_NB0: tl.constexpr = 0,
    NORM_G: tl.constexpr = 0,
    NORM_SIGMOID: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # SEG_NB0 > 0: the n grid is two back-to-back segments, [0, SPLIT_N) in
    # blocks [0, SEG_NB0) and [SPLIT_N, N) in the rest, so no CTA straddles the
    # two-destination split and every CTA has a single store.
    seg1 = SEG_NB0 > 0 and pid_n >= SEG_NB0
    if SEG_NB0 > 0:
        n_lo = tl.where(seg1, SPLIT_N + (pid_n - SEG_NB0) * BLOCK_N, pid_n * BLOCK_N)
        n_hi = tl.where(seg1, N, SPLIT_N)
    else:
        n_lo = pid_n * BLOCK_N
        n_hi = N
    offs_n = n_lo + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < n_hi
    m_mask = offs_m < M
    if NORM_G > 0:
        # Gated RMSNorm folded into the A-load; the group is exactly one
        # BLOCK_K tile (asserted by the planner), so the row statistics are
        # local to the tile and nothing extra is read for them.
        # Everything read before pdl_wait must be memory the preceding kernel
        # does not write: no caller passes a norm= today, and one that passes a
        # norm weight produced in-stream would read it a kernel too early.
        norm_w = tl.load(nw_ptr + offs_k).to(tl.float32)
    # x comes from the previous kernel; the weight/scale loads below are all
    # inside the reduction loop, so the wait sits at the top and what PDL buys
    # here is the block-scheduling ramp overlapping the previous kernel's drain.
    pdl_wait(USE_PDL)
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        # W_KN: the weight is [K, N] contiguous, so the [BLOCK_K, BLOCK_N] tile is the
        # contiguous one and tl.dot takes it directly instead of transposing in smem.
        # Every current caller is [N, K] contiguous and takes the else branch.
        if W_KN:
            wp = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[None, :], other=0.0)
            else:
                w = tl.load(wp, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        else:
            wp = w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[:, None], other=0.0)
            else:
                w = tl.load(wp, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        if USE_DOT:
            xp = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
            if EVEN_K:
                x = tl.load(xp, mask=m_mask[:, None], other=0.0)
            else:
                x = tl.load(xp, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            if NORM_G > 0:
                zp = z_ptr + offs_m[:, None] * stride_zm + kk[None, :] * stride_xk
                if EVEN_K:
                    zt = tl.load(zp, mask=m_mask[:, None], other=0.0).to(tl.float32)
                else:
                    zt = tl.load(
                        zp, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                    ).to(tl.float32)
                xf = x.to(tl.float32)
                var = tl.sum(xf * xf, axis=1) / NORM_G
                rstd = tl.rsqrt(var + norm_eps)
                yn = xf * rstd[:, None] * norm_w[None, :]
                if NORM_SIGMOID:
                    yn = yn * tl.sigmoid(zt)
                else:
                    yn = yn * (zt * tl.sigmoid(zt))
                x = yn.to(tl.bfloat16)
            if W_KN:
                acc += tl.dot(x, w.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        else:
            # M == 1: broadcast-multiply and reduce instead of padding to a 16-row mma.
            if EVEN_K:
                xv = tl.load(x_ptr + kk * stride_xk).to(tl.float32)
            else:
                xv = tl.load(x_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(
                    tl.float32
                )
            if NORM_G > 0:
                if EVEN_K:
                    zv = tl.load(z_ptr + kk * stride_xk).to(tl.float32)
                else:
                    zv = tl.load(z_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(
                        tl.float32
                    )
                rstd1 = tl.rsqrt(tl.sum(xv * xv, axis=0) / NORM_G + norm_eps)
                yv = xv * rstd1 * norm_w
                if NORM_SIGMOID:
                    yv = yv * tl.sigmoid(zv)
                else:
                    yv = yv * (zv * tl.sigmoid(zv))
                xv = yv.to(tl.bfloat16).to(tl.float32)
            if W_KN:
                acc += tl.sum(w.to(tl.float32) * xv[:, None], axis=0)[None, :]
            else:
                acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)[None, :]

    if SPLITS == 1:
        if PER_CHANNEL:
            acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            acc = acc * tl.load(s_ptr)
        _store_split(
            acc,
            y_ptr,
            y2_ptr,
            offs_m,
            offs_n,
            stride_ym,
            stride_yn,
            stride_y2m,
            m_mask,
            n_mask,
            seg1,
            SPLIT_N,
            SEG_NB0,
        )
        pdl_trigger(USE_PDL)
        return

    # Split-K fixup, in this same launch. `.cg` keeps the partials out of the
    # non-coherent L1 so the CTA that runs the reduction sees the other CTAs' writes.
    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * (M_PAD * BLOCK_N)
    tl.store(base + pid_k * (M_PAD * BLOCK_N) + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            tot += tl.load(base + s * (M_PAD * BLOCK_N) + slot, cache_modifier=".cg")
        if PER_CHANNEL:
            tot = tot * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            tot = tot * tl.load(s_ptr)
        _store_split(
            tot,
            y_ptr,
            y2_ptr,
            offs_m,
            offs_n,
            stride_ym,
            stride_yn,
            stride_y2m,
            m_mask,
            n_mask,
            seg1,
            SPLIT_N,
            SEG_NB0,
        )
        # Every increment for this N block has happened; leave the counter at 0
        # for the next launch / graph replay.
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="release", scope="gpu")
    pdl_trigger(USE_PDL)


# Split-K scratch, allocated once per (device, stream) and reused: the fixup CTA
# leaves the counters at 0, so nothing has to be cleared between calls. Sized
# for the largest plan _plan can emit (the planner caps the product at
# _WS_FLOATS).
_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_WS: dict = {}


def _workspace(
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Scratch (partials, counters, unit scale) for the current stream.

    None if it would have to be allocated inside a graph capture.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    ws = _WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = (
            torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
            torch.ones(1, dtype=torch.float32, device=device),
        )
        _WS[key] = ws
    return ws


_NUM_SMS = None


def _num_sms(device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


# Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) with
# bench/w8a16v2/tune_v3.py + the head-to-head re-check in bench/w8a16v2/final_v3.py;
# the metric is the CUPTI kernel-duration median over a CUDA-graph replay whose
# weight working set is several times the 128 MB L2, i.e. the number the serving
# torch profile reports. Re-run it after a kernel or hardware change.
#
# key:   (weight element bytes, N is the contiguous weight axis, N, K)
# value: (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages)
# W_KN None derives the tile orientation from the weight strides.
#
# `N is the contiguous weight axis` is False on every row: both callers pass an
# [N, K]-contiguous weight (see the module docstring). The pre-v3 table keyed the
# fp8 rows True, which no call ever matched, so the server silently ran _plan's
# fallback for all of them; the "vs" ratios below are against that fallback.
_BY_SHAPE = {
    # fp8, w8a16_gemv
    # shared-expert down: the single-split (32,128,1,...,4,4) bench winner measured 5.2us
    # in the W4 server (M=4) vs 4.0us for the split-5 fallback; M=16 keeps the winner below.
    (1, False, 2560, 640): (32, 128, 5, True, None, 4, 3),
    (1, False, 1280, 2560): (16, 128, 10, True, None, 2, 2),  # shared gate_up, 1.08x
    (1, False, 32768, 2560): (128, 256, 1, True, None, 8, 3),  # draft lm_head, 1.14x
    (1, False, 248320, 2560): (128, 256, 1, True, None, 8, 3),  # lm_head, 1.02x
    # bf16, bf16_gemv
    # GDN in_proj_ba: the bench's split-K winner (16,256,10,...,2,4) measured 9.2us in the
    # server vs 7.9us for this single-launch tile (2026-09-03 gemv3 trace), so keep it.
    (2, False, 96, 2560): (32, 512, 1, True, None, 4, 3),
    (2, False, 512, 2560): (16, 128, 10, True, None, 2, 4),  # MoE router gate, 1.04x
    # The 2560x6144 out_proj/o_proj and the 13312x2560 attention qkv are absent at
    # M > 1 because _plan's fallback already emits the fastest tile measured for them.
}
_BY_SHAPE_M16 = {
    # The _BY_SHAPE gate_up tile loses 12% at M=16; the qkvz tile below loses 1% at M=4.
    (1, False, 1280, 2560): (16, 64, 10, True, None, 4, 3),  # shared gate_up, 1.03x
    (1, False, 16384, 2560): (64, 128, 1, True, None, 4, 3),  # GDN in_proj_qkvz, 1.04x
    (1, False, 2560, 640): (
        32,
        128,
        1,
        True,
        None,
        4,
        4,
    ),  # shared-expert down (W16 server 4.1->3.8us)
    # attention qkv at M=16: both the wide (128,256,8w) tile and the (64,128) qkvz tile
    # measured 27.5us in the server vs 25.6us for the narrow single-split tile; pin it.
    (1, False, 13312, 2560): (32, 128, 1, True, None, 4, 3),
}
_BY_SHAPE_M1 = {
    # USE_DOT False (the broadcast path, M_PAD 1) is only legal at M == 1.
    (1, False, 2560, 640): (16, 256, 3, False, None, 4, 3),  # shared-expert down, 1.20x
    (1, False, 1280, 2560): (16, 128, 10, False, None, 2, 3),  # shared gate_up, 1.23x
    (1, False, 2560, 6144): (16, 256, 6, False, None, 4, 3),  # out_proj/o_proj, 1.05x
    (1, False, 13312, 2560): (32, 256, 1, False, None, 4, 3),  # attention qkv, 1.02x
    (1, False, 32768, 2560): (32, 256, 1, False, None, 4, 3),  # draft lm_head, 1.17x
    (2, False, 512, 2560): (16, 256, 10, False, None, 4, 3),  # MoE router gate, 1.15x
    (2, False, 2560, 2560): (16, 256, 5, False, None, 4, 2),  # MTP fc_*, 1.06x
}


def _expand(table, buckets):
    return {
        (wb, cn, m, N, K): cfg for (wb, cn, N, K), cfg in table.items() for m in buckets
    }


_TUNED = _expand(_BY_SHAPE, (1, 4, 16))
_TUNED.update(_expand(_BY_SHAPE_M16, (16,)))
_TUNED.update(_expand(_BY_SHAPE_M1, (1,)))


def _prev_pow2(v: int) -> int:
    return 1 << max(0, v.bit_length() - 1)


@functools.lru_cache(maxsize=512)
def _plan(M: int, N: int, K: int, contig_n: bool, w_bytes: int, sms: int):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages) for one call."""
    tuned = _TUNED.get((w_bytes, contig_n, _m_bucket(M), N, K))
    if tuned is not None:
        return _fit(M, N, tuned)
    if contig_n:
        # A [K, N]-contiguous weight wants the widest N block it can keep the grid
        # full with, since 32 fp8 columns is only one 32 B sector per k. Untuned:
        # no caller produces this layout.
        block_n = 128 if N >= 2048 else (64 if N >= 512 else 32)
        return _fit(M, N, (block_n, 128, 1, True, None, 8, 3))
    # [N, K] contiguous. When N alone fills the machine, a long BLOCK_K keeps more
    # bytes in flight: 1.15x on the 32768x2560 draft head, 1.01-1.04x on the other
    # three big shapes. The M == 4 bucket is the exception -- there the wide tile
    # measured 1-2% slower than the [32, 128] one on both 2560-K shapes.
    if triton.cdiv(N, 128) * 2 >= sms and _m_bucket(M) != 4:
        if M == 1:
            return (32, 256, 1, False, None, 4, 3)
        # A [128, 256] bf16 tile needs 144 KB of smem, over the 99 KB sm_120 limit.
        return (128 if w_bytes == 1 else 64, 256, 1, True, None, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    block_k, warps = 128, 4
    if triton.cdiv(N, 32) >= sms:
        return (block_n, block_k, 1, use_dot, None, warps, 3)
    # Aim for ~2 CTAs per SM, ~5 at M == 1 where an M_PAD 1 partial is 16x cheaper;
    # shrink BLOCK_K if K does not hold enough blocks for that.
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    block_k = min(block_k, max(64, _prev_pow2(max(1, K // want))))
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    # Prefer a split count that divides the k-block count so every CTA does equal work.
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, None, warps, 3))


def _fit(M: int, N: int, cfg, n_blocks: Optional[int] = None):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, w_kn, warps, stages = cfg
    if splits == 1:
        return cfg
    m_pad = 16 if (use_dot or M > 1) else 1
    if n_blocks is None:
        n_blocks = triton.cdiv(N, block_n)
    if (
        splits > _MAX_SPLITS
        or n_blocks > _WS_COUNTERS
        or n_blocks * splits * m_pad * block_n > _WS_FLOATS
    ):
        return (block_n, block_k, 1, use_dot, w_kn, warps, stages)
    return cfg


def _launch(
    ws, cnt, x, w, s, y, M, N, K, per_channel, cfg, y2=None, split_n=0, norm=None
):
    block_n, block_k, splits, use_dot, w_kn, num_warps, num_stages = cfg
    use_dot = use_dot or M > 1
    m_pad = 16 if use_dot else 1
    n_blocks = triton.cdiv(N, block_n)
    seg_nb0 = 0
    z, nw, norm_g, norm_eps, norm_sigmoid = (x, x, 0, 0.0, False)
    if norm is not None:
        z, nw, norm_g, norm_eps, norm_sigmoid = norm
    pdl = triton_pdl_enabled()
    _w8a16_gemv_kernel[(n_blocks, splits)](
        x,
        w,
        s,
        y,
        y if y2 is None else y2,
        ws,
        cnt,
        z,
        nw,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(0) if y2 is None else y2.stride(0),
        z.stride(0),
        norm_eps,
        SPLIT_N=split_n,
        PER_CHANNEL=per_channel,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        W_KN=(w.stride(0) == 1) if w_kn is None else w_kn,
        USE_DOT=use_dot,
        USE_PDL=pdl,
        # launch_pdl is a CUDA-only Triton launch kwarg; the HIP backend has no
        # such option and rejects unknown kwargs, so omit it rather than pass
        # False. The kernel body's USE_PDL above stays from the same read.
        **({"launch_pdl": True} if pdl else {}),
        SEG_NB0=seg_nb0,
        NORM_G=norm_g,
        NORM_SIGMOID=norm_sigmoid,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


# (N, K) of the BF16 linears whose tuned tile beat cuBLAS on sm_120 (see the
# tables above): GDN in_proj_ba, MoE router gate, MTP fc_embedding/fc_hidden.
SKINNY_BF16_SHAPES = frozenset({(96, 2560), (512, 2560), (2560, 2560)})
MAX_M = 16


def is_sm120() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 12
    )


def bf16_skinny_gemv_applies(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``bf16_skinny_linear(x, weight)`` has a tuned launch for these operands."""
    if x.dim() < 1 or weight.dim() != 2 or x.shape[-1] != weight.shape[1]:
        return False
    m = x.numel() // x.shape[-1] if x.shape[-1] else 0
    return (
        1 <= m <= MAX_M
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and weight.stride(1) == 1
        # Not differentiable; nn.Linear weights keep requires_grad under no_grad.
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and tuple(weight.shape) in SKINNY_BF16_SHAPES
    )


def bf16_skinny_gemv(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """x: [M, K] bf16 (M <= 16); w: [N, K] bf16, K contiguous.

    Returns None when the split-K scratch for this stream does not exist yet
    during a graph capture; the caller then runs its own GEMM.
    """
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError(f"bf16_skinny_gemv: x {tuple(x.shape)} w {tuple(w.shape)}")
    M, K = x.shape
    N = w.shape[0]
    if w.shape[1] != K or not 1 <= M <= MAX_M:
        raise ValueError(f"bf16_skinny_gemv: x {tuple(x.shape)} w {tuple(w.shape)}")
    scratch = _workspace(x.device)
    if scratch is None:
        return None
    ws, cnt, one = scratch
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    cfg = _plan(M, N, K, w.stride(0) == 1, w.element_size(), _num_sms(x.device))
    _launch(ws, cnt, x, w, one, y, M, N, K, False, cfg)
    return y


def bf16_skinny_linear(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """``F.linear(x, w)`` for any leading shape of ``x``; None on a scratch miss."""
    y = bf16_skinny_gemv(x.reshape(-1, x.shape[-1]), w)
    return None if y is None else y.view(*x.shape[:-1], w.shape[0])
