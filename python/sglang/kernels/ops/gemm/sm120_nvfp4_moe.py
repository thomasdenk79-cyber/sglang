"""Standalone donor port of the SwiGLU portion of upstream d5a4bc5a35."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


SWIGLU_MAX_TOKENS = 512


@cache_once
def _swiglu_module(num_experts: int, top_k: int, hidden: int, inter: int):
    args = make_cpp_args(num_experts, top_k, hidden, inter)
    return load_jit(
        "sm120_nvfp4_moe_swiglu",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_swiglu.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeSwiglu<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_dependencies=["cutlass"],
    )


def _swiglu_workspace_sizes(
    num_tokens: int, num_experts: int, top_k: int, hidden: int, inter: int
) -> tuple[int, int]:
    def align(n):
        return (n + 255) // 256 * 256

    routes = num_tokens * top_k
    units = min(routes, num_experts) + (routes + 31) // 32
    u8 = (
        align(num_tokens * hidden // 2)
        + align(num_tokens * hidden // 16)
        + align(units * 32 * inter // 2)
        + align(units * (inter // 128) * 32 * 8)
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
