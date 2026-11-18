# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
#
# Fused gated-RMSNorm prologue ported from aiueo52/sglang-rtxpro6000 @
# flash-next-fast python/sglang/srt/layers/quantization/w8a16_gemv.py
# ("Fused gated-RMSNorm prologue (SGLANG_NORM_INTO_GEMV=1)", Apache-2.0), where
# it folds the GDN gated RMSNorm into the W8A16 per-channel FP8 GEMV. The
# in-tree BF16 twin of that prologue is
# kernels/ops/gemm/sm120_bf16_skinny_gemv.py::_w8a16_gemv_kernel (NORM_G).
# The MXFP8 weight decode (float8_e4m3 rows, one UE8M0 byte per 32 K elements)
# mirrors sglang/srt/layers/quantization/mxfp8_skinny_gemv.py from the skinny
# MXFP8 GEMV line rather than importing it: that module lives on its own branch
# and this kernel must stand alone. Consolidate the two when both land.
"""Fused gated RMSNorm + MXFP8 skinny GEMV for the GDN output projection.

    y[M, N] (bf16) = (gated_rms_norm(x, z) @ dequant(W)^T), fp32 accumulation

``W`` is the server's MXFP8 layout: ``float8_e4m3fn`` [N, K] row-major with one
UE8M0 (power-of-two) scale byte per 32 contiguous K elements in
``weight_scale_inv`` [N, K // 32]; the scale of (n, k) is ``2 ** (e - 127)``.
The gate demands the full row-major layout (``stride(0) == K``,
``stride(1) == 1``): the kernel indexes ``w_ptr + n * K`` and a sliced view of
a wider tensor would silently read the wrong rows.

Why this exists: the GDN block normalises its attention output per
(token, v-head) -- ``RMSNormGated(head_v_dim, norm_before_gate=True)``, i.e.
``y = x * rsqrt(mean(x^2) + eps) * w * silu(z)`` -- and immediately feeds the
[tokens, num_v_heads * head_v_dim] result to the out_proj GEMV. On this
checkpoint that is 36 layers x 1 launch of ~192 one-warp CTAs moving ~150 KB of
activations, ~1.9 us each, purely to hand the same rows to the next kernel.
The GEMV already reads every A row once per N block, and with BLOCK_K ==
head_v_dim one A tile is exactly one norm group, so the row statistics are
recomputed from registers and the norm kernel disappears. The extra traffic is
the ``z`` tile -- same size as the A tile, L2-resident at these widths -- and
the [head_v_dim] norm weight.

Numerics: the fp32 sum of squares is re-associated relative to the standalone
norm kernel (different thread layout), so the result is within ~1 ulp of bf16
rather than bit-identical. Everything else is reproduced exactly: fp32 math,
the ``z * sigmoid(z)`` gate form, and the bf16 rounding of the normalised
activation before the dot.

``SGLANG_NORM_INTO_GEMV=1`` enables it; only the shapes in
:data:`_FUSED_NORM_BY_SHAPE` -- the GDN out_proj of this checkpoint, measured by
the donor -- are served, and every other call falls through to the unfused
norm + GEMM path unchanged. sm_120 only, MXFP8 dense weights only, and a
batch-invariant or deterministic run never takes it (the split-K order is not
cuBLAS's).

**Gain**: the donor measured -0.95 us mean / -1.06 us median per call against
the unfused pair, over 36 GDN layers = -36 us per decode step, and 36 fewer
kernels per step for the graph launcher.

**VRAM**: ~0. One split-K scratch of ``_WS_FLOATS`` fp32 + ``_WS_COUNTERS``
int32 (~4 MiB) per (device, stream) that actually launches it, materialized at
weight-load time by ``Fp8LinearMethod.process_weights_after_loading`` rather
than inside a capture. No weight copies.

**Numerics**: the fp32 sum of squares is re-associated relative to the
standalone norm kernel, so the out_proj input is within ~1 ulp of bf16 rather
than bit-identical; the gate form, the bf16 rounding of the activation and the
per-sub-group UE8M0 scaling are reproduced exactly.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

#: One UE8M0 scale byte covers this many contiguous K elements (MXFP8 group).
GROUP_K = 32

#: Widest decode batch the hook serves; wider calls keep the unfused path.
MAX_M = 16

#: Most split-K partials one launch may keep per N block.
_MAX_SPLITS = 16
#: Floats and counters of split-K scratch, per (device, stream) entry.
_WS_FLOATS = 1 << 20
_WS_COUNTERS = 4096

#: (M bucket, N, K, norm group) -> (BLOCK_N, SPLITS, USE_DOT, warps, stages).
#: The donor's measured fused tiles (bench/h1/h1_check.py sweep, CUPTI medians,
#: RTX PRO 6000): BLOCK_N 32 -> 64 halves how often a CTA recomputes the norm
#: for the same rows, and the split count comes down with it to keep ~2 CTAs/SM.
#: The M == 1 and M == 2 rows reuse the M == 4 tile with the cheaper broadcast
#: reduction, which nothing has measured on a GPU yet: the micro-benchmark
#: reports it next to the unfused pair so the A/B can reject it.
_FUSED_NORM_BY_SHAPE: Dict[
    Tuple[int, int, int, int], Tuple[int, int, bool, int, int]
] = {
    (4, 2560, 6144, 128): (64, 4, True, 8, 4),
    (16, 2560, 6144, 128): (64, 4, True, 8, 3),
    (1, 2560, 6144, 128): (64, 4, False, 8, 4),
}


@triton.jit
def _mxfp8_norm_gated_gemv_kernel(
    x_ptr,
    z_ptr,
    nw_ptr,  # norm weight, [G]
    w_ptr,  # e4m3 [N, K], row-major
    s_ptr,  # UE8M0 bytes [N, K // GROUP_K], row-major
    y_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,  # norm group == BLOCK_K, so one tile is one group
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    USE_DOT: tl.constexpr,
    NORM_SIGMOID: tl.constexpr,
    NORM_FUSED: tl.constexpr,  # False: plain GEMV, the benchmark's baseline
):
    if M <= 0:
        # Nothing to reduce and nothing to read; the counters stay at zero.
        return
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, G)
    n_mask = offs_n < N
    w_base = w_ptr + offs_n[:, None] * K
    sf_k = K // GROUP_K
    if NORM_FUSED:
        norm_w = tl.load(nw_ptr + offs_g).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(pid_k * G, K, SPLITS * G):
        kb = k0 + offs_g
        if USE_DOT:
            offs_m = tl.arange(0, BLOCK_M)
            m_mask = offs_m < M
            ap = x_ptr + offs_m[:, None] * K + kb[None, :]
            zp = z_ptr + offs_m[:, None] * K + kb[None, :]
            xf = tl.load(ap, mask=m_mask[:, None], other=0.0).to(tl.float32)
            if NORM_FUSED:
                zf = tl.load(zp, mask=m_mask[:, None], other=0.0).to(tl.float32)
                rstd = tl.rsqrt(tl.sum(xf * xf, axis=1) / G + eps)
                yn = xf * rstd[:, None] * norm_w[None, :]
                if NORM_SIGMOID:
                    yn = yn * tl.sigmoid(zf)
                else:
                    yn = yn * (zf * tl.sigmoid(zf))
            else:
                yn = xf
            a = yn.to(tl.bfloat16)
        else:
            # M == 1: broadcast-multiply and reduce, no 16-row mma tile.
            xf = tl.load(x_ptr + kb, mask=(kb < K) & (M > 0), other=0.0).to(tl.float32)
            if NORM_FUSED:
                zf = tl.load(z_ptr + kb, mask=(kb < K) & (M > 0), other=0.0).to(
                    tl.float32
                )
                rstd1 = tl.rsqrt(tl.sum(xf * xf, axis=0) / G + eps)
                yn = xf * rstd1 * norm_w
                if NORM_SIGMOID:
                    av = yn * tl.sigmoid(zf)
                else:
                    av = yn * (zf * tl.sigmoid(zf))
            else:
                av = xf

        # One load per norm group for the whole [BLOCK_N, G] weight tile, and
        # one UE8M0 byte per 32-element sub-group. The sub-group scale is
        # broadcast over its 32 weights and folded in BEFORE the dot: e4m3
        # carries a 3-bit mantissa and the scale is a power of two, so the
        # product is still exact in bf16, and no sub-group is ever reduced
        # under another sub-group's exponent.
        kg = k0 + offs_g
        w = tl.load(w_base + kg[None, :], mask=n_mask[:, None], other=0.0)
        sb = tl.load(
            s_ptr
            + offs_n[:, None] * sf_k
            + (k0 // GROUP_K)
            + tl.arange(0, G // GROUP_K)[None, :],
            mask=n_mask[:, None],
            other=0,
        )
        # UE8M0 -> fp32: exponent bits only, so a zero byte is a zero scale.
        sg = (sb.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        wf = tl.reshape(
            tl.reshape(w.to(tl.float32), (BLOCK_N, G // GROUP_K, GROUP_K))
            * sg[:, :, None],
            (BLOCK_N, G),
        )
        if USE_DOT:
            acc += tl.dot(a, tl.trans(wf.to(tl.bfloat16)), out_dtype=tl.float32)
        else:
            # [BLOCK_N, G] x [G] -> [BLOCK_N] -> a row of acc, which is
            # (1, BLOCK_N) because the non-dot path only serves M == 1.
            acc += tl.sum(wf * av[None, :], axis=1)[None, :]

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    if SPLITS == 1:
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )
        return

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
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            tot.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )
        # Every increment happened; leave the counter at 0 for the next launch
        # or graph replay, so no memset is needed.
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="release", scope="gpu")


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


def plan_for(M: int, N: int, K: int, group: int):
    """(BLOCK_N, SPLITS, USE_DOT, warps, stages) for this call, or None.

    Only the measured shapes are served: an unmeasured tile is a guess about
    performance, and the unfused path is the better default for a call nobody
    has benchmarked.
    """
    cfg = _FUSED_NORM_BY_SHAPE.get((_m_bucket(M), N, K, group))
    if cfg is None:
        return None
    block_n, splits, use_dot, warps, stages = cfg
    if (
        splits > _MAX_SPLITS
        or -(-N // block_n) > _WS_COUNTERS
        or -(-N // block_n) * splits * (16 if (use_dot or M > 1) else 1) * block_n
        > _WS_FLOATS
    ):
        return None
    return cfg


@functools.lru_cache(maxsize=None)
def _enabled_cache() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_NORM_INTO_GEMV.get())


def enabled() -> bool:
    """The opt-in flag, read once per process (it is a launch-time switch)."""
    return _enabled_cache()


def reset_flag_cache() -> None:
    """Drop the cached flag; tests set SGLANG_NORM_INTO_GEMV after import."""
    _enabled_cache.cache_clear()


def is_sm120() -> bool:
    """The same sm_120 test the in-tree BF16 skinny GEMV uses."""
    from sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv import is_sm120

    return is_sm120()


# --- split-K scratch, per (device, stream) --------------------------------
# Same contract as kernels/ops/gemm/sm120_bf16_skinny_gemv.py: eager launches
# on different streams get different scratch, a captured launch bakes its
# stream's scratch into the graph, and an entry is never freed.
_WORKSPACES: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
_MAX_WORKSPACES = 32


def _stream_id(device: torch.device) -> Any:
    try:
        return torch.cuda.current_stream(device).cuda_stream
    except Exception:  # pragma: no cover - CPU or a driver without streams
        return None


def _workspace(device: torch.device, capturing: bool):
    key = (device, _stream_id(device))
    ws = _WORKSPACES.get(key)
    if ws is None:
        if capturing:
            # Allocating here would put the scratch in the graph's private pool.
            return None
        if len(_WORKSPACES) >= _MAX_WORKSPACES:
            logger.warning(
                "mxfp8 fused norm GEMV: %d scratch entries already registered; "
                "keeping the unfused path for %s",
                _MAX_WORKSPACES,
                device,
            )
            return None
        ws = (
            torch.zeros(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    return ws


def prealloc(device: torch.device, streams: Tuple[int, ...] = ()) -> None:
    """Materialize the scratch for streams that a capture will bake in.

    Warm-up runs at prefill widths, where the hook never reaches this GEMV, so
    without this the first decode launch would allocate inside a capture.
    ``streams`` takes the raw handles of any extra capture streams.
    """
    try:
        handles = [torch.cuda.current_stream(device).cuda_stream, *streams]
    except Exception:  # pragma: no cover - not a CUDA device
        return
    for handle in handles:
        if (device, handle) in _WORKSPACES:
            continue
        _WORKSPACES[(device, handle)] = (
            torch.zeros(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
        )


def peek_scratch(device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    return _WORKSPACES.get((device, _stream_id(device)))


def counters_are_zero(device: torch.device) -> bool:
    ws = peek_scratch(device)
    return True if ws is None else bool(int(ws[1].abs().sum().item()) == 0)


def reset_workspaces() -> None:
    """Drop the scratch registry; for tests and for a device reset."""
    _WORKSPACES.clear()


def fused_norm_gemv_applies(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    activation: str,
    *,
    require_cuda: bool = True,
) -> bool:
    """Whether this call can take the fused gated-norm + MXFP8 GEMV path.

    ``require_cuda`` is False only for the CPU reference tests; the production
    hook uses the default and therefore never routes a CPU or cross-device
    tensor into the kernel.
    """
    if x.dim() != 2 or weight.dim() != 2:
        return False
    M, K = x.shape
    N = weight.shape[0]
    if require_cuda and (
        not x.is_cuda
        or z.device != x.device
        or weight.device != x.device
        or scale.device != x.device
        or norm_weight.device != x.device
    ):
        return False
    return bool(
        1 <= M <= MAX_M
        and group_size > 0
        and K % group_size == 0
        and x.dtype == torch.bfloat16
        and z.dtype == torch.bfloat16
        and tuple(z.shape) == (M, K)
        and x.is_contiguous()
        and z.is_contiguous()
        and norm_weight.numel() == group_size
        and norm_weight.is_contiguous()
        and norm_weight.dtype in (torch.bfloat16, torch.float32)
        and weight.dtype == torch.float8_e4m3fn
        and weight.shape == (N, K)
        # The kernel indexes w_ptr + offs_n * K, ignoring strides: an [N, K]
        # view of a wider row-major tensor would pass a stride(1) == 1-only
        # check and silently read the wrong rows. Demand the full row-major
        # layout the kernel assumes.
        and weight.stride(1) == 1
        and weight.stride(0) == K
        and scale.dtype == torch.uint8
        and scale.shape == (N, K // GROUP_K)
        and scale.is_contiguous()
        and activation in ("swish", "silu", "sigmoid")
        and plan_for(M, N, K, group_size) is not None
    )


def fused_norm_gated_gemv(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    eps: float,
    activation: str = "swish",
    out: Optional[torch.Tensor] = None,
    norm_fused: bool = True,
) -> Optional[torch.Tensor]:
    """``(gated_rms_norm(x, z, norm_weight) @ dequant(weight)^T)`` in one launch.

    ``norm_fused=False`` drops the prologue and turns z into a placeholder: the
    micro-benchmark times that twin against the standalone norm kernel plus this
    same GEMV, so the row prices the fusion rather than the GEMV.

    Returns None when the split-K scratch does not exist yet and the call is
    inside a CUDA graph capture, so the caller can run its own path instead of
    baking an allocation into the graph pool.
    """
    M, K = x.shape
    N = weight.shape[0]
    cfg = plan_for(M, N, K, group_size)
    if cfg is None:
        raise ValueError(f"no fused-norm plan for M={M} N={N} K={K} G={group_size}")
    block_n, splits, use_dot, warps, stages = cfg
    if not norm_fused:
        z = x  # unused by the kernel; keeps the signature honest
    ws_scratch = _workspace(x.device, torch.cuda.is_current_stream_capturing())
    if ws_scratch is None:
        return None
    ws, cnt = ws_scratch
    y = (
        torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        if out is None
        else out
    )
    _mxfp8_norm_gated_gemv_kernel[(-(-N // block_n), splits)](
        x,
        z,
        norm_weight,
        weight,
        scale,
        y,
        ws,
        cnt,
        M,
        float(eps),
        N=N,
        K=K,
        G=group_size,
        BLOCK_M=16 if (use_dot or M > 1) else 1,
        BLOCK_N=block_n,
        SPLITS=splits,
        USE_DOT=use_dot,
        NORM_SIGMOID=activation == "sigmoid",
        NORM_FUSED=norm_fused,
        num_warps=warps,
        num_stages=stages,
    )
    return y


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[N, K] e4m3 + [N, K // 32] UE8M0 -> fp32, the reference's view of W."""
    descale = (scale.to(torch.int32) << 23).view(torch.float32)
    return (
        weight.to(torch.float32).view(weight.shape[0], -1, GROUP_K)
        * descale[:, :, None]
    ).view(weight.shape[0], weight.shape[1])


def gated_rms_norm_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    group_size: int,
    eps: float,
    activation: str = "swish",
) -> torch.Tensor:
    """The [M, K] bf16 activation ``RMSNormGated`` would hand to out_proj."""
    M, K = x.shape
    xf = x.to(torch.float32).view(M, K // group_size, group_size)
    zf = z.to(torch.float32).view(M, K // group_size, group_size)
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    w = norm_weight.to(torch.float32)
    y = xf * rstd * w
    gate = torch.sigmoid(zf) if activation == "sigmoid" else zf * torch.sigmoid(zf)
    return (y * gate).to(torch.bfloat16).view(M, K)


def fused_norm_gated_gemv_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    eps: float,
    activation: str = "swish",
) -> torch.Tensor:
    """fp32 ``gated_rms_norm(x, z) @ dequant(W)^T``, bf16 like the kernel."""
    a = gated_rms_norm_reference(x, z, norm_weight, group_size, eps, activation)
    return (a.to(torch.float32) @ dequant_mxfp8(weight, scale).t()).to(torch.bfloat16)
