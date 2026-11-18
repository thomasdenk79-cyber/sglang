# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
#
# Kernel, split-K fixup and per-shape tile tables ported from
# aiueo52/sglang-rtxpro6000 @ flash-next-fast
# python/sglang/srt/layers/quantization/w8a16_gemv.py (W8A16 Triton GEMV) with
# the per-shape tiles of aiueo52/flash-next-rtxpro6000 patches 0026-0029. The
# per-stream split-K scratch of patch 0041 -- two split-K launches sharing one
# scratch and one counter array corrupt each other's partials -- is here the
# (device, stream) key of _workspace, as in the in-tree BF16 port
# sglang/kernels/ops/gemm/sm120_bf16_skinny_gemv.py, which also supplies the
# atomic-exchange counter reset. Deviations from the donor: MXFP8 (one UE8M0
# scale per 32 K elements, applied inside the K loop) instead of a per-channel
# or per-tensor FP32 scale; no PDL, no gated-norm prologue, no fused SiLU and
# no two-destination store, none of which an MXFP8 dense linear uses.
"""Triton MXFP8 skinny GEMV for decode/verify widths on RTX PRO 6000 (sm_120).

    y[M, N] (bf16) = x[M, K] (bf16) @ dequant(W)[N, K]^T, fp32 accumulation

``W`` is MXFP8: ``float8_e4m3fn`` [N, K] with one UE8M0 (power-of-two) scale
byte per 32 contiguous K elements, stored as ``torch.uint8`` [N, K // 32]. The
scale of element (n, k) is ``2 ** (e - 127)`` with ``e = scale[n, k // 32]``,
the same byte mapping as ``mxfp8_block_convert._ue8m0_to_fp32`` (so ``e == 0``
is a zero scale and ``e == 255`` is +inf, matching the in-tree dequant rather
than the OCP NaN encoding).

Why this exists: at decode batch sizes the dense linears of an MXFP8 checkpoint
are pure weight bandwidth, and the W8A8 GEMM ``Fp8LinearMethod`` runs them
through first quantizes the activation per 32-wide group. A weight-only GEMV
keeps the activations in BF16, reads each weight byte exactly once and drops
the quantization kernel; HyperBurst measured +24..40 % end-to-end on its
FP8-dense build (docs/optimizations.md §B1). What it costs instead is the FP8
-> BF16 upconvert and the per-group scale multiply in the inner loop, which is
why BLOCK_K stays a multiple of the 32-element scale group and each group is
scaled before it reaches the accumulator: no group is ever reduced under
another group's exponent.

Two properties of the decode shapes decide the geometry:

- Weight layout. The MXFP8 loader leaves ``layer.weight`` an [N, K]-CONTIGUOUS
  e4m3 tensor (stride (K, 1)) with the canonical UE8M0 bytes in
  ``weight_scale_inv``; only the FlashInfer CUTLASS / CuTe-DSL backends keep
  both (they park their swizzled copy in ``weight_scale_inv_swizzled``), and
  the fp8.py gate restricts the hook to those. So the natural weight tile is
  [BLOCK_N, BLOCK_K] and ``tl.dot`` takes it transposed.
- Grid size. One CTA per N block owning the whole K reduction launches 3..128
  CTAs for the small-N linears (shared-expert gate_up and down, k_proj,
  v_proj, the indexer) on a 188-SM part, so those calls run at a few hundred
  GB/s however well the inner loop is written. Splitting K across CTAs fixes
  that, and the reduction is folded into the same launch: every CTA writes an
  fp32 partial and bumps a per-N-block counter, and whoever observes the last
  increment sums the partials and writes bf16 y. That CTA resets the counter
  with an atomic exchange, so no memset is needed and the kernel is safe to
  capture in a CUDA graph and replay.

M == 1 uses a broadcast-multiply reduction instead of ``tl.dot`` (the donor
measured it 2-23 % faster on every shape): a 16-row tensor-core tile would pad
a single row 16x in the accumulator, in the partials and in the instructions.

The split-K scratch is per (device, stream), so work on the model's alt stream
gets its own scratch instead of corrupting the main stream's partials (patch
0041), and a launch whose scratch does not exist yet during a graph capture
returns None so the caller runs its ordinary GEMM rather than baking an
allocation into that graph's private pool. :func:`prealloc` therefore covers the
forward stream *and* the one shared stream every sglang graph is captured on: a
decode graph that silently fell back to the W8A8 GEMM would look correct and
gain nothing.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

#: One UE8M0 scale byte covers this many contiguous K elements (MXFP8 group).
GROUP_K = 32

#: Hard ceiling on the decode width this kernel is planned for; the value of
#: SGLANG_MXFP8_SKINNY_GEMV_MAX_M is clamped to it (tiles are tuned at 1/4/16).
MAX_M = 32

#: Split-K scratch, allocated once per (device, stream) and reused: the fixup
#: CTA leaves the counters at 0, so nothing has to be cleared between calls.
#: Sized for the largest plan _plan can emit (_fit caps the product).
_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_WS: dict = {}

#: The plan tables were measured on an RTX PRO 6000 Blackwell Max-Q. Off that
#: part -- and in the Triton interpreter, which has no device to query -- the
#: planner still has to pick something.
_TUNE_TARGET_SMS = 188

_NUM_SMS: Optional[int] = None


def gemv_max_m() -> int:
    """The decode width the hook serves: ``SGLANG_MXFP8_SKINNY_GEMV_MAX_M``.

    An out-of-range value is clamped, not raised: this runs on the forward
    path through :func:`mxfp8_skinny_gemv_applies`, which has to stay a
    predicate that never raises.
    """
    from sglang.srt.environ import envs

    value = int(envs.SGLANG_MXFP8_SKINNY_GEMV_MAX_M.get())
    if not 1 <= value <= MAX_M:
        logger.warning(
            "SGLANG_MXFP8_SKINNY_GEMV_MAX_M=%s is outside [1, %d]; using %d",
            value,
            MAX_M,
            min(max(value, 1), MAX_M),
        )
        value = min(max(value, 1), MAX_M)
    return value


_IS_SM120: Optional[bool] = None


def is_sm120() -> bool:
    """Whether this device is an SM120 (RTX PRO 6000 / RTX 50) part.

    Cached: the forward path asks for every dense linear, and the answer
    cannot change once the process has a CUDA context.
    """
    global _IS_SM120
    if _IS_SM120 is None:
        _IS_SM120 = bool(
            torch.cuda.is_available()
            and torch.version.hip is None
            and torch.cuda.get_device_capability()[0] == 12
        )
    return _IS_SM120


def _num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if device.type != "cuda" or not torch.cuda.is_available():
        return _TUNE_TARGET_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


@triton.jit
def _store(y_ptr, acc, offs_m, offs_n, stride_ym, stride_yn, m_mask, n_mask):
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _mxfp8_skinny_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,  # UE8M0 bytes, [N, K // GROUP_K], contiguous
    y_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # a multiple of GROUP_K
    GROUP_K: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, GROUP_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    # The weight is row-major, so one row of the scale tensor is one output
    # channel's K groups: byte (n, gk) covers columns [gk * GROUP_K, ...).
    sf_k = K // GROUP_K
    w_base = w_ptr + offs_n[:, None] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        # BLOCK_K is a multiple of GROUP_K, so every iteration covers whole
        # scale groups. Each group is multiplied by its own UE8M0 exponent
        # before it reaches the accumulator, and BLOCK_K == GROUP_K (the narrow
        # plans) collapses this inner loop to one iteration.
        for g in tl.static_range(BLOCK_K // GROUP_K):
            kb = k0 + g * GROUP_K + offs_k
            gk = k0 // GROUP_K + g  # == kb // GROUP_K, the scale column
            if EVEN_K:
                w_mask = n_mask[:, None]
                x_mask = m_mask[:, None]
                s_mask = n_mask
            else:
                k_ok = kb < K
                w_mask = n_mask[:, None] & k_ok[None, :]
                x_mask = m_mask[:, None] & k_ok[None, :]
                # A group wholly past K is reduced out by the masked weight,
                # but its scale byte is past the end of the row: never read it.
                s_mask = n_mask & (gk < sf_k)
            w = tl.load(w_base + kb[None, :] * stride_wk, mask=w_mask, other=0.0)
            scale = tl.load(s_ptr + offs_n * sf_k + gk, mask=s_mask, other=0)
            # UE8M0 -> fp32: exponent bits only, so the shift is exact and a
            # zero byte gives a zero scale (as _ue8m0_to_fp32 in
            # mxfp8_block_convert.py does it).
            s = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            if USE_DOT:
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + kb[None, :] * stride_xk,
                    mask=x_mask,
                    other=0.0,
                )
                # e4m3 -> bf16 is exact (3 stored mantissa bits, exponent range
                # well inside bf16's), so the mma sees the raw weight and the
                # group's power of two scales the fp32 accumulator.
                acc += (
                    tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
                    * s[None, :]
                )
            else:
                # M == 1: broadcast-multiply and reduce, no 16-row mma tile.
                # Row 0 is the only row there is, so x is a 1-D load.
                if EVEN_K:
                    xv = tl.load(x_ptr + kb * stride_xk).to(tl.float32)
                else:
                    xv = tl.load(x_ptr + kb * stride_xk, mask=kb < K, other=0.0).to(
                        tl.float32
                    )
                acc += (tl.sum(w.to(tl.float32) * xv[None, :], axis=1) * s)[None, :]

    if SPLITS == 1:
        _store(y_ptr, acc, offs_m, offs_n, stride_ym, stride_yn, m_mask, n_mask)
        return

    # Split-K fixup, in this same launch. `.cg` keeps the partials out of the
    # non-coherent L1 so the CTA that runs the reduction sees the other CTAs'
    # writes.
    tile = BLOCK_M * BLOCK_N
    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * tile
    tl.store(base + pid_k * tile + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for sp in tl.static_range(SPLITS):
            tot += tl.load(base + sp * tile + slot, cache_modifier=".cg")
        _store(y_ptr, tot, offs_m, offs_n, stride_ym, stride_yn, m_mask, n_mask)
        # Every increment for this N block has happened; leave the counter at 0
        # for the next launch / graph replay.
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="release", scope="gpu")


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


def _use_dot(M: int, use_dot: bool) -> bool:
    # The broadcast path reduces row 0 only, so it is legal at M == 1.
    return use_dot or M > 1


def _block_m(M: int, use_dot: bool) -> int:
    # Rows of x the CTA loads and reduces; !USE_DOT only ever runs at M == 1.
    if not _use_dot(M, use_dot):
        return 1
    return 16 if M <= 16 else 32


# Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) for the donor's
# FP8 W8A16 GEMV: bench/w8a16v2/tune_v3.py and the head-to-head re-check in
# bench/w8a16v2/final_v3.py, CUPTI kernel-duration medians over a CUDA-graph
# replay whose weight working set is several times the 128 MB L2, i.e. what the
# serving torch profile reports. Keyed (N, K) per M bucket; value
# (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages).
#
# The MXFP8 scale traffic (one byte per 32 K elements, ~3 % of the weight
# bytes) is new relative to the donor, and the shape set is this checkpoint's,
# so re-measure with scripts/bench_mxfp8_skinny_gemv.py on the target before
# trusting these over the planner's heuristic.
#
# Unlike the donor, the tables are keyed on (N, K) alone: every caller hands
# over an [N, K]-contiguous weight, and the donor's fp8 rows were keyed for a
# [K, N] layout no call ever produced, which silently ran the fallback tile
# for all of them (patches/sglang/series 0027).
_BY_SHAPE = {
    # shared-expert down: the single-split (32,128,1,4,4) bench winner measured
    # 5.2us in the W4 server (M=4) vs 4.0us for the split-5 fallback; M=16
    # keeps the winner below.
    (2560, 640): (32, 128, 5, True, 4, 3),
    # shared gate_up, 1.08x over the fallback.
    (1280, 2560): (16, 128, 10, True, 2, 2),
    # lm_head-class: N alone fills the machine, so a long BLOCK_K wins.
    (248320, 2560): (128, 256, 1, True, 8, 3),
}
_BY_SHAPE_M16 = {
    # The _BY_SHAPE gate_up tile loses 12% at M=16; the qkvz tile below loses
    # 1% at M=4.
    (1280, 2560): (16, 64, 10, True, 4, 3),
    (16384, 2560): (64, 128, 1, True, 4, 3),  # GDN in_proj_qkvz, 1.04x
    (2560, 640): (32, 128, 1, True, 4, 4),  # shared down (4.1 -> 3.8us)
    # attention qkv at M=16: both the wide (128,256,8w) tile and the (64,128)
    # qkvz tile measured 27.5us in the server vs 25.6us for the narrow
    # single-split tile; pin it.
    (13312, 2560): (32, 128, 1, True, 4, 3),
}
_BY_SHAPE_M1 = {
    # USE_DOT False (the broadcast path, BLOCK_M 1) is only legal at M == 1.
    (2560, 640): (16, 256, 3, False, 4, 3),  # shared down, 1.20x
    (1280, 2560): (16, 128, 10, False, 2, 3),  # shared gate_up, 1.23x
    (2560, 6144): (16, 256, 6, False, 4, 3),  # out_proj / o_proj, 1.05x
    (13312, 2560): (32, 256, 1, False, 4, 3),  # attention qkv, 1.02x
    (16384, 2560): (64, 128, 1, True, 4, 3),  # GDN in_proj_qkvz
    (32768, 2560): (32, 256, 1, False, 4, 3),  # draft lm_head, 1.17x
}


def _expand(table, buckets):
    return {(m, n, k): cfg for (n, k), cfg in table.items() for m in buckets}


_TUNED = _expand(_BY_SHAPE, (1, 4, 16))
_TUNED.update(_expand(_BY_SHAPE_M16, (16,)))
_TUNED.update(_expand(_BY_SHAPE_M1, (1,)))


def _prev_pow2(v: int) -> int:
    return 1 << max(0, v.bit_length() - 1)


@functools.lru_cache(maxsize=512)
def _plan(M: int, N: int, K: int, sms: int) -> Tuple[int, int, int, bool, int, int]:
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages) for one call."""
    tuned = _TUNED.get((_m_bucket(M), N, K))
    if tuned is not None:
        block_n, block_k, splits, use_dot, warps, stages = tuned
        return _fit(
            M, N, (block_n, block_k, splits, _use_dot(M, use_dot), warps, stages)
        )
    # BLOCK_K is always a multiple of GROUP_K; 128 is four scale groups, what
    # the donor's fp8 tiles used and what the A-load and mma like.
    block_k = 128
    if triton.cdiv(N, 128) * 2 >= sms and _m_bucket(M) != 4:
        if M == 1:
            return (32, block_k, 1, False, 4, 3)
        # A [128, 256] tile is over the 99 KB smem budget of sm_120 once the
        # bf16 operands are staged, so BLOCK_K stays at 128 here.
        return (128, block_k, 1, True, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    warps = 4
    if triton.cdiv(N, block_n) >= sms:
        # N alone fills the machine: no split-K, just enough K per CTA.
        return _fit(M, N, (block_n, block_k, 1, use_dot, warps, 3))
    # Aim for ~2 CTAs per SM, ~5 at M == 1 where a BLOCK_M 1 partial is 16x
    # cheaper; shrink BLOCK_K if K does not hold enough blocks for that, but
    # never below one scale group.
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    block_k = max(
        GROUP_K, min(block_k, GROUP_K * _prev_pow2(max(1, K // (want * GROUP_K))))
    )
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    # Prefer a split count that divides the k-block count so every CTA does
    # equal work.
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, warps, 3))


def _fit(M: int, N: int, cfg):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, warps, stages = cfg
    if splits == 1:
        return cfg
    block_m = _block_m(M, use_dot)
    n_blocks = triton.cdiv(N, block_n)
    if splits > _MAX_SPLITS or n_blocks > _WS_COUNTERS:
        return (block_n, block_k, 1, use_dot, warps, stages)
    if n_blocks * splits * block_m * block_n > _WS_FLOATS:
        return (block_n, block_k, 1, use_dot, warps, stages)
    return cfg


def _stream_id(device: torch.device, stream: Optional[Any] = None) -> Any:
    """The scratch key for ``stream`` (default: the current one)."""
    if device.type != "cuda":
        # CPU / Triton interpreter: one buffer, no stream to key on.
        return "cpu"
    if stream is None:
        return torch.cuda.current_stream(device).cuda_stream
    # torch.cuda.Stream and raw cudaStream_t both carry the handle directly.
    return getattr(stream, "cuda_stream", stream)


def _workspace(
    device: torch.device, stream: Optional[Any] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Split-K (partials, counters) scratch for one stream.

    None when it would have to be allocated inside a graph capture; see 0041.
    Call :func:`prealloc` so the streams that matter -- the forward stream and
    the one stream every sglang graph is captured on -- already have it.
    """
    key = (device, _stream_id(device, stream))
    ws = _WS.get(key)
    if ws is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return None
        ws = (
            torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
        )
        _WS[key] = ws
    return ws


def _graph_capture_streams(device: torch.device) -> List[Any]:
    """The streams a decode/verify forward can be captured or issued on.

    sglang captures every CUDA graph of a process on one shared stream
    (``model_executor.runner_utils.pool``), so that stream needs its own scratch
    too: a launch whose key is missing during a capture falls back to the W8A8
    GEMM, which would silently keep the decode graphs on the old path.
    """
    streams = []
    try:
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
        )

        streams.append(get_or_create_global_graph_capture_stream())
    except Exception:  # a build without the shared capture stream: eager only
        logger.debug("no shared graph capture stream to prealloc for", exc_info=True)
    return streams


def prealloc(device: Optional[torch.device] = None) -> None:
    """Materialize the split-K scratch before any CUDA graph is captured.

    Warm-up runs at prefill widths, where the GEMV gate falls through to the
    W8A8 GEMM, so without this the scratch would first be allocated inside a
    capture and land in that graph's private pool. Covers the current stream
    and the shared graph capture stream. Called from
    ``Fp8LinearMethod.process_weights_after_loading``.
    """
    if device is None:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    _workspace(device)
    if device.type == "cuda":
        for stream in _graph_capture_streams(device):
            _workspace(device, stream)


def _launch(ws, cnt, x, w, s, y, M, N, K, cfg) -> None:
    block_n, block_k, splits, use_dot, num_warps, num_stages = cfg
    use_dot = _use_dot(M, use_dot)
    _mxfp8_skinny_gemv_kernel[(triton.cdiv(N, block_n), splits)](
        x,
        w,
        s,
        y,
        ws,
        cnt,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=_block_m(M, use_dot),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_K=GROUP_K,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        USE_DOT=use_dot,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def mxfp8_skinny_gemv_applies(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_u8: Optional[torch.Tensor],
    max_m: Optional[int] = None,
) -> bool:
    """Whether :func:`mxfp8_skinny_gemv` has a launch for these operands.

    A pure predicate -- it never raises -- so a caller can use it as the gate
    in front of its own GEMM.
    """
    cap = gemv_max_m() if max_m is None else min(max_m, MAX_M)
    if (
        scale_u8 is None
        or x.dim() != 2
        or w.dim() != 2
        or w.dtype != torch.float8_e4m3fn
        or scale_u8.dtype != torch.uint8
        or x.dtype != torch.bfloat16
        or x.shape[0] > cap
        or x.shape[-1] != w.shape[1]
        or w.shape[1] % GROUP_K != 0
        or w.stride(1) != 1
        or tuple(scale_u8.shape) != (w.shape[0], w.shape[1] // GROUP_K)
        or not scale_u8.is_contiguous()
        or scale_u8.device != w.device
        or x.device != w.device
        # Not differentiable, and the caller owns the output buffer.
        or (torch.is_grad_enabled() and (x.requires_grad or w.requires_grad))
    ):
        return False
    return True


def mxfp8_skinny_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_u8: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    cfg=None,
) -> Optional[torch.Tensor]:
    """``x @ dequant(w).T`` for decode widths; None on a split-K scratch miss.

    x: [M, K] bf16 (M <= MAX_M); w: [N, K] float8_e4m3fn, K-contiguous;
    scale_u8: [N, K // 32] contiguous UE8M0 uint8. ``out``, when given, is a
    [M, N] row-contiguous bf16 CUDA buffer to write into -- how a caller keeps
    the result out of the per-call allocator and out of a graph's private pool
    -- and ``cfg`` pins the tile plan (see :func:`plan_for`).
    """
    M, K = x.shape
    N = w.shape[0]
    scratch = _workspace(x.device)
    if scratch is None:
        return None
    ws, cnt = scratch
    if out is None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    else:
        if (
            tuple(out.shape) != (M, N)
            or out.dtype != torch.bfloat16
            or out.device != x.device
            or out.stride(1) != 1
        ):
            raise ValueError(
                f"mxfp8 skinny gemv out= must be a [{M}, {N}] row-contiguous bf16 "
                f"tensor on {x.device}; got {tuple(out.shape)} {out.dtype} {out.device}"
            )
        y = out
    if cfg is None:
        cfg = plan_for(x, w)
    _launch(ws, cnt, x, w, scale_u8, y, M, N, K, cfg)
    return y


def plan_for(x: torch.Tensor, w: torch.Tensor):
    """The tile plan :func:`mxfp8_skinny_gemv` would use for these operands."""
    return _plan(x.shape[0], w.shape[0], w.shape[1], _num_sms(x.device))


def mxfp8_dequant_to_fp32(w: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """[N, K] e4m3 + [N, K // 32] UE8M0 -> fp32 [N, K], exactly as the loader
    dequantizes it (``mxfp8_block_convert.dequant_mxfp8_2d_to_bf16`` without the
    bf16 round)."""
    n, k = w.shape
    descale = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    return (
        w.to(torch.float32).view(n, k // GROUP_K, GROUP_K) * descale.unsqueeze(-1)
    ).view(n, k)


def mxfp8_skinny_gemv_reference(
    x: torch.Tensor, w: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """The fp32 reference the tests check the kernel against.

    ``x.float() @ mxfp8_dequant_to_fp32(w, scale_u8).T``: a full-precision
    dequantized-weight GEMM with no activation quantization. Device-agnostic,
    so the CPU Triton interpreter and the GPU compare against the same numbers.
    """
    return x.to(torch.float32) @ mxfp8_dequant_to_fp32(w, scale_u8).t()
