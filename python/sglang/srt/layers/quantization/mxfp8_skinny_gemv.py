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

At M == 1 the planner prefers a broadcast-multiply reduction over ``tl.dot``
(a 16-row tensor-core tile would pad a single row 16x in the accumulator, in
the partials and in the instructions), but the per-shape table below decides:
where the donor measured the mma to be the faster of the two at M == 1
(``in_proj_qkvz``, the lm_head class) the entry keeps ``USE_DOT`` True. The
tables are the authority; the heuristic only fills shapes they do not list.

``mxfp8_skinny_gemv`` covers any M: rows beyond the BLOCK_M one launch can
carry are served by further launches, so a direct caller can never get back
rows that were never written. The hook, in contrast, stops at
``MAX_TUNED_M`` -- the widest M the tables were measured at.

Scratch contract (as for the in-tree sm120_bf16_skinny_gemv). The split-K
reduction uses the (partials, counters) buffer of one (device, stream) key, so

- at most one split-K launch may be in flight per key: same-stream launches
  serialise by construction, different streams get different scratch (0041);
- a launch captured into a CUDA graph bakes that stream's scratch into the
  graph for the graph's whole life, so graphs containing this kernel must not
  replay concurrently, and an entry can never be freed while the process lives
  -- which is why :func:`_workspace` caps the registry instead of evicting
  (see :func:`prealloc` for getting the streams that matter an entry early);
- a counter that is not zero when a launch starts silently truncates that N
  block's reduction, so :func:`_launch_chunk` clears the counters on every
  *eager* split-K launch. Inside a capture that has to be the kernel's own
  atomic-exchange reset; :func:`counters_are_zero` is the host-side check.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

#: One UE8M0 scale byte covers this many contiguous K elements (MXFP8 group).
GROUP_K = 32

#: Rows one launch carries. Above it mxfp8_skinny_gemv launches again, so no
#: row of the output is ever left unwritten.
MAX_M = 32

#: Widest M the tile tables were measured at, and the value
#: SGLANG_MXFP8_SKINNY_GEMV_MAX_M is clamped to: 17..32 would ride the M=16
#: tiles with a BLOCK_M of 32, which nothing has measured, so the hook does not
#: serve it (a direct caller still can, and gets correct -- just untuned)
#: results).
MAX_TUNED_M = 16

#: Absolute sanity bound on a direct call: above it the caller means a GEMM.
_ENTRY_MAX_M = 1024

#: Split-K scratch, allocated once per (device, stream) and reused: the fixup
#: CTA leaves the counters at 0, so nothing has to be cleared between calls.
#: Sized for the largest plan _plan can emit (_fit caps the product).
_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32

#: How many (device, stream) scratch buffers one process may hold (~8 MiB each).
#: Entries are never freed -- a captured graph owns its scratch -- so a new
#: stream past the limit gets no scratch and keeps its caller's GEMM, which is
#: slow but correct, rather than an unbounded allocation.
_WS_LIMIT = 16
_WS: dict = {}
_WS_WARNED: set = set()

#: The plan tables were measured on an RTX PRO 6000 Blackwell Max-Q. Off that
#: part -- and in the Triton interpreter, which has no device to query -- the
#: planner still has to pick something.
_TUNE_TARGET_SMS = 188

#: Host-side evidence that the kernel actually ran: a launch counter and a
#: scratch-miss counter, for the benchmark and for tests (a CUDA graph capture
#: records the launch at capture time, which is exactly what needs proving).
_LAUNCHES = 0
_MISSES = 0


def stats() -> dict:
    """``{"launches": n, "scratch_misses": n}`` since process start."""
    return {"launches": _LAUNCHES, "scratch_misses": _MISSES}


def launch_count() -> int:
    """Number of GEMV launches issued so far (host side, capture-included)."""
    return _LAUNCHES


_MAX_M_WARNED: set = set()


def gemv_max_m() -> int:
    """The decode width the hook serves: ``SGLANG_MXFP8_SKINNY_GEMV_MAX_M``.

    An out-of-range value is clamped to [1, MAX_TUNED_M], not raised: this runs
    on the forward path through :func:`mxfp8_skinny_gemv_applies`, which has to
    stay a predicate that never raises. The warning is per distinct value
    because the gate asks on every dense linear of every forward.
    """
    from sglang.srt.environ import envs

    value = int(envs.SGLANG_MXFP8_SKINNY_GEMV_MAX_M.get())
    if not 1 <= value <= MAX_TUNED_M:
        if value not in _MAX_M_WARNED:
            _MAX_M_WARNED.add(value)
            logger.warning(
                "SGLANG_MXFP8_SKINNY_GEMV_MAX_M=%s is outside [1, %d]; using %d",
                value,
                MAX_TUNED_M,
                min(max(value, 1), MAX_TUNED_M),
            )
        return min(max(value, 1), MAX_TUNED_M)
    return value


def _device_index(device: Optional[Any]) -> int:
    if device is None:
        return torch.cuda.current_device()
    index = getattr(device, "index", None)
    return torch.cuda.current_device() if index is None else index


#: Per device index: a heterogeneous process must not plan for whichever GPU
#: asked first, and the answer cannot change once the context exists.
_IS_SM120_BY_INDEX: Dict[int, bool] = {}
_SMS_BY_INDEX: Dict[int, int] = {}


def is_sm120(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` is an SM120 (RTX PRO 6000 / RTX 50) part."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    index = _device_index(device)
    cached = _IS_SM120_BY_INDEX.get(index)
    if cached is None:
        cached = bool(torch.cuda.get_device_capability(index)[0] == 12)
        _IS_SM120_BY_INDEX[index] = cached
    return cached


def _num_sms(device: Optional[torch.device]) -> int:
    """Streaming multiprocessors of ``device``; the tuning target off CUDA."""
    if device is None or getattr(device, "type", None) != "cuda":
        return _TUNE_TARGET_SMS
    if not torch.cuda.is_available():
        return _TUNE_TARGET_SMS
    index = _device_index(device)
    sms = _SMS_BY_INDEX.get(index)
    if sms is None:
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        if not sms:  # a stub device with no properties
            return _TUNE_TARGET_SMS
        _SMS_BY_INDEX[index] = sms
    return sms


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
    # Nothing to reduce for an empty batch, and nothing to read: every load
    # below would be past the end of a zero-row x. mxfp8_skinny_gemv_applies
    # refuses M <= 0, this is what keeps a direct caller's bug a no-op instead
    # of an illegal access. The counters are untouched, so they stay at 0.
    if M <= 0:
        return
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
                # Row 0 is the only row there is, so x is a 1-D load -- masked
                # by M as well as by K, so a caller that bypassed the gate
                # cannot read past an x with fewer rows than that.
                xv = tl.load(
                    x_ptr + kb * stride_xk,
                    mask=(kb < K) & (M > 0),
                    other=0.0,
                ).to(tl.float32)
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
    # The broadcast path reduces row 0 only, so it is legal at M <= 1.
    return use_dot or M > 1


def _block_m(M: int, use_dot: bool) -> int:
    # Rows of x the CTA loads and reduces; !USE_DOT only ever runs at M <= 1.
    if not _use_dot(M, use_dot):
        return 1
    return 16 if M <= 16 else 32


# Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) for the donor's
# FP8 W8A16 GEMV: kernels/microbench/w8a16v2/tune_v3.py and the head-to-head
# re-check in kernels/microbench/w8a16v2/final_v3.py, CUPTI kernel-duration
# medians over a CUDA-graph replay whose weight working set is several times the
# 128 MB L2, i.e. what the serving torch profile reports. Keyed (N, K) per M
# bucket; value (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, num_warps, num_stages).
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
# M == 1. USE_DOT False is the broadcast-multiply path (BLOCK_M 1); True keeps
# the tensor-core mma, which the donor measured faster than the broadcast
# reduction on these two shapes. The entry, not the comment, is the authority:
# whatever is written here is what runs.
_BY_SHAPE_M1 = {
    (2560, 640): (16, 256, 3, False, 4, 3),  # shared down, 1.20x
    (1280, 2560): (16, 128, 10, False, 2, 3),  # shared gate_up, 1.23x
    (2560, 6144): (16, 256, 6, False, 4, 3),  # out_proj / o_proj, 1.05x
    (13312, 2560): (32, 256, 1, False, 4, 3),  # attention qkv, 1.02x
    (16384, 2560): (64, 128, 1, True, 4, 3),  # GDN in_proj_qkvz, mma wins here
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
    if N <= 0 or K <= 0:
        # The gate refuses a degenerate weight; this keeps a direct caller from
        # the division by zero below and from a grid with an empty axis.
        return (16, GROUP_K, 1, False, 4, 1)
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


def _fits(M: int, N: int, cfg) -> bool:
    """Whether the plan's partials and counters fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, warps, stages = cfg
    if not 1 <= splits <= _MAX_SPLITS:
        return False
    n_blocks = triton.cdiv(N, block_n)
    return splits == 1 or (
        n_blocks <= _WS_COUNTERS
        and n_blocks * splits * _block_m(M, use_dot) * block_n <= _WS_FLOATS
    )


def _fit(M: int, N: int, cfg):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, warps, stages = cfg
    if splits == 1:
        return cfg
    if _fits(M, N, cfg):
        return cfg
    return (block_n, block_k, 1, use_dot, warps, stages)


def _clear_counters(cnt: torch.Tensor, n_blocks: int, capturing: bool) -> None:
    """Reset the counters of an *eager* split-K launch.

    A launch aborted mid-kernel (device fault, watchdog) leaves a counter
    non-zero, and the N block that owns it then silently stops reducing: the
    fixup CTA never runs, so the caller gets a plausible-looking partial sum
    rather than an error. Nothing in the kernel can detect that from inside the
    launch, so outside a capture the counters are cleared on the stream first.
    Inside a capture this must stay a no-op -- a memset node would be replayed
    on every step and the whole point of the atomic-exchange reset is that the
    graph needs none.
    """
    if not capturing:
        cnt[:n_blocks].zero_()


def _is_capturing(device: torch.device) -> bool:
    """Whether the current stream is capturing (False off CUDA, e.g. CPU tests)."""
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return False
    return bool(torch.cuda.is_current_stream_capturing())


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
    Call :func:`prealloc` so the streams that matter -- the forward stream, the
    one stream every sglang graph is captured on, and the model's alt stream --
    already have it. Entries are never freed (a captured graph holds its
    scratch), so the registry is capped at _WS_LIMIT and a new stream past it
    falls back to the caller's GEMM; that fallback is logged once per stream.
    """
    key = (device, _stream_id(device, stream))
    ws = _WS.get(key)
    if ws is not None:
        return ws
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        if key not in _WS_WARNED:
            _WS_WARNED.add(key)
            logger.warning(
                "mxfp8 skinny GEMV: no split-K scratch for stream 0x%x of %s while a "
                "CUDA graph is being captured; this linear keeps the W8A8 GEMM. "
                "Cover it in prealloc() to get the GEMV inside the graph.",
                key[1] if isinstance(key[1], int) else 0,
                key[0],
            )
        return None
    if len(_WS) >= _WS_LIMIT:
        if "limit" not in _WS_WARNED:
            _WS_WARNED.add("limit")
            logger.warning(
                "mxfp8 skinny GEMV: %d (device, stream) scratch buffers is the limit; "
                "further streams keep the W8A8 GEMM",
                _WS_LIMIT,
            )
        return None
    ws = (
        torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
        torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
    )
    _WS[key] = ws
    return ws


def peek_scratch(
    device: Optional[torch.device] = None, stream: Optional[Any] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The scratch of one stream, or None; never allocates (tests, benchmarks)."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    return _WS.get((torch.device(device), _stream_id(torch.device(device), stream)))


def counters_are_zero(
    device: Optional[torch.device] = None, stream: Optional[Any] = None
) -> bool:
    """Whether a split-K launch could run correctly on this stream right now.

    The kernel resets its own counters, so they must read 0 between launches; a
    non-zero counter means an aborted or shared launch and silently truncated
    partials. Host-side check for tests and for the benchmark (it syncs).
    """
    scratch = peek_scratch(device, stream)
    if scratch is None:
        return True
    return bool(torch.equal(scratch[1], torch.zeros_like(scratch[1])))


def reset_workspace() -> None:
    """Forget the scratch registry. Tests only: never call while a graph holds it."""
    _WS.clear()
    _WS_WARNED.clear()


def _forward_streams(device: torch.device) -> List[Any]:
    """The streams a decode/verify forward can be captured or issued on.

    sglang captures every CUDA graph of a process on one shared stream
    (``model_executor.runner_utils.pool``), so that stream needs its own scratch
    too: a launch whose key is missing during a capture falls back to the W8A8
    GEMM, which would silently keep the decode graphs on the old path. The
    MoE block issues the shared expert on the ``alt`` named stream whenever
    ``SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP`` is on -- its default -- and the
    QSA indexer forks its overlap work onto it only inside the capture, so both
    the forward and the alt stream have to be there before the first capture
    (that half of HyperBurst 0041, which prealloc materialized both slots).
    """
    streams = []
    try:
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
        )

        streams.append(get_or_create_global_graph_capture_stream())
    except Exception:  # a build without the shared capture stream: eager only
        logger.debug("no shared graph capture stream to prealloc for", exc_info=True)
    try:
        from sglang.srt.runtime_context import get_stream

        streams.append(get_stream("alt"))
    except Exception:  # no named-stream registry (or no CUDA) to prealloc for
        logger.debug("no alt stream to prealloc for", exc_info=True)
    return streams


def prealloc(
    device: Optional[torch.device] = None, stream: Optional[Any] = None
) -> None:
    """Materialize the split-K scratch before any CUDA graph is captured.

    Warm-up runs at prefill widths, where the GEMV gate falls through to the
    W8A8 GEMM, so without this the scratch would first be allocated inside a
    capture and land in that graph's private pool. Covers the current stream,
    the shared graph capture stream, the ``alt`` named stream and ``stream``,
    for callers (a benchmark, a custom runner) that capture on their own.
    Called from ``Fp8LinearMethod.process_weights_after_loading``.
    """
    if device is None:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if stream is not None:
        _workspace(device, stream)
    _workspace(device)
    if device.type == "cuda":
        for one in _forward_streams(device):
            _workspace(device, one)


def _launch_chunk(
    ws: torch.Tensor,
    cnt: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    K: int,
    cfg,
) -> None:
    block_n, block_k, splits, use_dot, num_warps, num_stages = cfg
    use_dot = _use_dot(M, use_dot)
    block_m = _block_m(M, use_dot)
    n_blocks = triton.cdiv(N, block_n)
    if splits > 1:
        _clear_counters(cnt, n_blocks, _is_capturing(x.device))
    _mxfp8_skinny_gemv_kernel[(n_blocks, splits)](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_K=GROUP_K,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        USE_DOT=use_dot,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    global _LAUNCHES
    _LAUNCHES += 1


def _launch(ws, cnt, x, w, s, y, M, N, K, cfg) -> None:
    """One launch per MAX_M rows, so rows [MAX_M, M) are written too."""
    for m0 in range(0, M, MAX_M):
        rows = min(MAX_M, M - m0)
        _launch_chunk(
            ws, cnt, x[m0 : m0 + rows], w, s, y[m0 : m0 + rows], rows, N, K, cfg
        )


def _reject_reason(x: Any, w: Any, scale_u8: Any, cap: int) -> Optional[str]:
    """Why these operands cannot run the GEMV, or None if they can.

    Shared by the predicate and the entry point, so the gate and the launch can
    never drift apart: the kernel indexes the scale as ``offs_n * (K // 32) + gk``
    straight out of ``w.shape``, so an operand that disagrees with the weight is
    an out-of-bounds read, not an error.
    """
    if not isinstance(x, torch.Tensor):
        return "x must be a [M, K] BF16 tensor, got " + type(x).__name__
    if not isinstance(w, torch.Tensor):
        return "w must be an [N, K] e4m3 tensor, got " + type(w).__name__
    if not isinstance(scale_u8, torch.Tensor):
        return (
            "scale_u8 must be a [N, K // 32] uint8 tensor, got "
            + type(scale_u8).__name__
        )
    if x.dim() != 2:
        return f"x must be 2-D [M, K], got {tuple(x.shape)}"
    if w.dim() != 2:
        return f"w must be 2-D [N, K], got {tuple(w.shape)}"
    M, K = x.shape
    N, K_w = w.shape
    if M <= 0:
        return "empty batch: M must be >= 1"
    if M > cap:
        return f"M = {M} is above the cap {cap}"
    if N <= 0:
        return "degenerate weight: N must be >= 1"
    if K <= 0:
        return "degenerate weight: K must be >= 1"
    if K_w != K:
        return f"x has K = {K}, the weight has K = {K_w}"
    if x.dtype != torch.bfloat16:
        return f"x must be BF16, got {x.dtype}"
    if w.dtype != torch.float8_e4m3fn:
        return f"the weight must be float8_e4m3fn, got {w.dtype}"
    if scale_u8.dtype != torch.uint8:
        return f"the scale must be UE8M0 uint8, got {scale_u8.dtype}"
    if K % GROUP_K != 0:
        return f"K = {K} is not a whole number of {GROUP_K}-element scale groups"
    if tuple(scale_u8.shape) != (N, K // GROUP_K):
        return (
            f"the scale must be [{N}, {K // GROUP_K}] UE8M0 bytes, got "
            f"{tuple(scale_u8.shape)}"
        )
    if not scale_u8.is_contiguous():
        return "the scale must be contiguous [N, K // 32]"
    if w.stride(1) != 1:
        return "the weight must be K-contiguous [N, K]"
    if scale_u8.device != w.device:
        return f"the scale is on {scale_u8.device}, the weight on {w.device}"
    if x.device != w.device:
        return f"x is on {x.device}, the weight on {w.device}"
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return "grad-requiring operands need the autograd path, not this kernel"
    return None


def mxfp8_skinny_gemv_applies(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_u8: Optional[torch.Tensor],
    max_m: Optional[int] = None,
) -> bool:
    """Whether :func:`mxfp8_skinny_gemv` has a launch for these operands.

    A pure predicate -- it never raises -- so a caller can use it as the gate
    in front of its own GEMM. It refuses an empty batch and a degenerate
    weight: neither has a row to compute, and the broadcast path would read a
    zero-row ``x``.
    """
    cap = gemv_max_m() if max_m is None else min(max_m, MAX_TUNED_M)
    try:
        return _reject_reason(x, w, scale_u8, cap) is None
    except Exception:  # a gate that can raise is not a gate
        logger.debug("mxfp8 skinny gemv gate failed on these operands", exc_info=True)
        return False


def mxfp8_skinny_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_u8: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    cfg=None,
) -> Optional[torch.Tensor]:
    """``x @ dequant(w).T`` for decode widths; None on a split-K scratch miss.

    x: [M, K] bf16; w: [N, K] float8_e4m3fn, K-contiguous; scale_u8:
    [N, K // 32] contiguous UE8M0 uint8. Any M is covered -- above MAX_M the
    rows are served by further launches -- but the hook only asks up to
    MAX_TUNED_M, and anything the gate would refuse raises here rather than
    reaching the kernel with an addressing mode it does not support. ``out``,
    when given, is a [M, N] row-contiguous bf16 tensor on the weight's device to
    write into -- how a caller keeps the result out of the per-call allocator
    and out of a graph's private pool -- and ``cfg`` pins the tile plan (see
    :func:`plan_for`); a pinned plan that would not fit the scratch is demoted
    to SPLITS=1 instead of writing past it.
    """
    reason = _reject_reason(x, w, scale_u8, _ENTRY_MAX_M)
    if reason is not None:
        raise ValueError(f"mxfp8 skinny gemv: {reason}")
    M, K = x.shape
    N = w.shape[0]
    scratch = _workspace(x.device)
    if scratch is None:
        global _MISSES
        _MISSES += 1
        return None
    ws, cnt = scratch
    if out is None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    else:
        if (
            tuple(out.shape) != (M, N)
            or out.dtype != torch.bfloat16
            or out.device != x.device
            or out.dim() != 2
            or out.stride(1) != 1
        ):
            raise ValueError(
                f"mxfp8 skinny gemv out= must be a [{M}, {N}] row-contiguous bf16 "
                f"tensor on {x.device}; got {tuple(out.shape)} {out.dtype} {out.device}"
            )
        y = out
    if cfg is None:
        cfg = plan_for(x, w)
    elif not _fits(M, N, cfg):
        logger.debug(
            "mxfp8 skinny gemv: pinned plan %s does not fit, using SPLITS=1", cfg
        )
        cfg = _fit(M, N, cfg)
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
