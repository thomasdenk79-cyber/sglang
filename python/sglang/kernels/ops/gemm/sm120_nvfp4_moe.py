"""Standalone donor port of the SwiGLU portion of upstream d5a4bc5a35."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


SWIGLU_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# P0.3 SM120 NVFP4 MoE tuning knobs (CPU-side parameterization only).
#   SGLANG_SM120_KUNIT: tokens per MMA unit; allowed 8 / 16 / 32, default 32.
#   SGLANG_SM120_NSTG2: GEMM2 pipeline stages; allowed 4..8, default 4.
# Both are read at import time, clamped to safe values, and forwarded as -D
# flags in extra_cuda_cflags. load_jit() stores those in BuildSpec.cuda_cflags;
# compute_build_key() hashes the generated ninja text carrying the flags, so
# each variant gets a distinct JIT build. With neither env var set the cflags
# stay exactly ["-O3", "--use_fast_math"] and the kernel keeps kUnit=32 / 4
# GEMM2 stages -- default behavior is unchanged.
#
# VALIDATION GATES -- required before ANY GPU run of a non-default variant:
#   * bitwise equality vs cutlass_fused_moe(use_fused_finalize=False), gold test
#     test/registered/kernels/ops/gemm/test_sm120_nvfp4_moe_swiglu.py;
#   * skipped experts (ids < 0 or >= E) and all-skipped tokens (zero output);
#   * CUDA-graph capture + replay with the variant's workspace sizes;
#   * one process restart per variant (_swiglu_module is @cache_once) on a
#     separate port -- never the live 8001 benchmark;
#   * one measurement method for all variants (no warm-prefix/cold-prefill mix);
#   * NSTG2=5 uses >48 KiB smem (the .cuh opts in): check launch and occupancy.
# p16 has no nvcc/GPU: only CPU syntax checks run here; t15 does GPU validation.
# ---------------------------------------------------------------------------

_SM120_KUNIT_DEFAULT = 32
_SM120_NSTG2_DEFAULT = 4
_SM120_KUNIT_CHOICES = (8, 16, 32)
_SM120_NSTG2_RANGE = (4, 8)


def _sm120_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not an integer; using default {default}", RuntimeWarning, stacklevel=2
        )
        return default


def _sm120_tuning() -> tuple[int, int, list[str]]:
    """Return (kUnit, nStg2, -D cflags), clamped to values the C++ kernel accepts."""
    kunit = _sm120_int_env("SGLANG_SM120_KUNIT", _SM120_KUNIT_DEFAULT)
    nstg2 = _sm120_int_env("SGLANG_SM120_NSTG2", _SM120_NSTG2_DEFAULT)
    if kunit not in _SM120_KUNIT_CHOICES:
        clamped = min(_SM120_KUNIT_CHOICES, key=lambda c: abs(c - kunit))
        warnings.warn(
            f"SGLANG_SM120_KUNIT={kunit} is not one of {_SM120_KUNIT_CHOICES}; clamping to {clamped}",
            RuntimeWarning,
            stacklevel=2,
        )
        kunit = clamped
    lo, hi = _SM120_NSTG2_RANGE
    if not lo <= nstg2 <= hi:
        clamped = min(max(nstg2, lo), hi)
        warnings.warn(
            f"SGLANG_SM120_NSTG2={nstg2} outside [{lo}, {hi}]; clamping to {clamped}",
            RuntimeWarning,
            stacklevel=2,
        )
        nstg2 = clamped
    cflags: list[str] = []
    if os.environ.get("SGLANG_SM120_KUNIT"):
        cflags.append(f"-DSGLANG_SM120_KUNIT={kunit}")
    if os.environ.get("SGLANG_SM120_NSTG2"):
        cflags.append(f"-DSGLANG_SM120_NSTG2={nstg2}")
    return kunit, nstg2, cflags


SM120_KUNIT, SM120_NSTG2, SM120_TUNING_CFLAGS = _sm120_tuning()


@cache_once
def _swiglu_module(num_experts: int, top_k: int, hidden: int, inter: int):
    args = make_cpp_args(num_experts, top_k, hidden, inter)
    return load_jit(
        "sm120_nvfp4_moe_swiglu",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_swiglu.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeSwiglu<{args}>::run")],
        # -D flags live in the JIT cache key, so each env variant builds separately.
        extra_cuda_cflags=["-O3", "--use_fast_math", *SM120_TUNING_CFLAGS],
        extra_dependencies=["cutlass"],
    )


def _swiglu_workspace_sizes(
    num_tokens: int, num_experts: int, top_k: int, hidden: int, inter: int
) -> tuple[int, int]:
    def align(n):
        return (n + 255) // 256 * 256

    routes = num_tokens * top_k
    # Mirror C++ Shape::max_units / u8_bytes with the same kUnit the kernel was built with.
    units = min(routes, num_experts) + (routes + SM120_KUNIT - 1) // SM120_KUNIT
    u8 = (
        align(num_tokens * hidden // 2)
        + align(num_tokens * hidden // 16)
        + align(units * SM120_KUNIT * inter // 2)
        + align(units * (inter // 128) * SM120_KUNIT * 8)
    )
    i32 = (
        2 * routes
        + 2 * units
        + 2 * num_experts
        + num_tokens * (hidden // 128)
        + num_tokens
        + 2
    )
    return u8, i32


def nvfp4_moe_swiglu(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    input_scale1: torch.Tensor,
    input_scale2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Use CUTLASS-compatible [up, gate] weights and the unfused finalize."""
    num_tokens, hidden = x.shape
    num_experts, inter = w13.shape[0], w13.shape[1] // 2
    top_k = topk_ids.shape[1]
    module = _swiglu_module(num_experts, top_k, hidden, inter)
    if out is None:
        out = torch.empty_like(x)
    u8, i32 = _swiglu_workspace_sizes(num_tokens, num_experts, top_k, hidden, inter)
    module.run(
        x,
        w13.view(torch.uint8),
        w13_scale.view(torch.uint8).view(-1),
        w2.view(torch.uint8),
        w2_scale.view(torch.uint8).view(-1),
        alpha1.view(-1),
        alpha2.view(-1),
        input_scale1.view(-1),
        input_scale2.view(-1),
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.float().contiguous(),
        torch.empty(u8, dtype=torch.uint8, device=x.device),
        torch.empty((num_tokens * top_k, hidden), dtype=torch.bfloat16, device=x.device),
        torch.empty(i32, dtype=torch.int32, device=x.device),
        out,
    )
    return out


def try_nvfp4_moe_swiglu(
    x, w13, w2, scales, ids, routing, *, output_dtype, out=None, before_launch=None
) -> Optional[torch.Tensor]:
    # Callers may hand CUTLASS's int64 packing views; shapes are checked on bytes.
    w13, w2 = w13.view(torch.uint8), w2.view(torch.uint8)
    if not (
        x.ndim == 2
        and 1 <= x.shape[0] <= SWIGLU_MAX_TOKENS
        and x.dtype == output_dtype == torch.bfloat16
        and x.is_contiguous()
        and w13.dim() == 3
        and w13.shape[0] % 32 == 0
        and x.shape[1] % 256 == 0
        and w13.shape[1] % 256 == 0
        and w13.shape[2] * 2 == x.shape[1]
        and w2.shape == (w13.shape[0], x.shape[1], w13.shape[1] // 4)
        and scales[0].numel() == 1
        and scales[3].numel() == 1
        and (out is None or out.is_contiguous())
    ):
        return None
    if before_launch is not None:
        before_launch()
    return nvfp4_moe_swiglu(
        x, w13, scales[1], w2, scales[4], scales[2], scales[5],
        scales[0], scales[3], ids, routing, out=out,
    )
