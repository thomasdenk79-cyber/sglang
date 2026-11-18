# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Small-batch BF16 router projection for Qwen MoE decode/verify.

Port of the donor path (aiueo52/sglang-rtxpro6000 @ flash-next-fast,
``python/sglang/srt/layers/moe/router_gemv.py``, Apache-2.0). Nothing in the
kernel is changed; the docstring below records how it fits this tree.

``SGLANG_ROUTER_GEMV=1`` routes the MoE gate -- a ``ReplicatedLinear``
[num_experts, hidden] against a decode batch of 1..16 rows -- away from cuBLAS
to this Triton kernel. At those widths cuBLAS launches a general MMA GEMM whose
tile is mostly padding, while the gate is pure weight bandwidth: [512, 2560]
BF16 = 2.6 MB read once per layer per step.

Two entry points, matching the donor's split:

* :func:`router_gemv` computes into **FP32** logits and accepts a BF16 bias.
  This is the kernel the donor's ``_forward_router_logits`` reaches for when a
  gate carries a bias.
* the bias-free case -- the only one Qwen3.8-Flash-Next has -- wants the *same*
  BF16 logits the ``ReplicatedLinear`` path produced, and this tree already
  owns a tuned split-K BF16 skinny GEMV for exactly that shape
  (``kernels/ops/gemm/sm120_bf16_skinny_gemv.py``, ``(512, 2560)`` is in its
  table). The model hook therefore prefers that kernel when the gate is
  bias-free and uses this module's kernel otherwise, which is the donor's own
  arrangement with ``bf16_gemv`` replaced by the in-tree equivalent.

Graph safety: the launch is a single kernel with no workspace, no split-K
scratch and no counter, and the only allocation is the FP32 output, which a
capture bakes into the graph pool. No host sync, no ``.item()``.

**Flag**: ``SGLANG_ROUTER_GEMV=1`` (default 0), resolved once per MoE block in
``Qwen2MoeSparseMoeBlock.__init__`` and only for the Qwen MoE model families
on CUDA; every guard in :func:`_forward_router_logits` that fails sends the
call to the ``ReplicatedLinear`` unchanged.

**Gain**: the donor's record (docs/optimizations.md B3) has the cuBLAS pair --
GEMM plus splitKreduce -- at 119 us per decode step over 61 calls, and the
retuned skinny GEMV at 4.42 us per call at M == 16 (down from 6.18 us), worth
about -86 us per step and 63 fewer kernels per step. One call per MoE layer
per step, 48 layers on this checkpoint. Caveat from the same note: the
donor's *dedicated* router kernel measured 7.5 us against cuBLAS's 5.0 us, so
this module's biased FP32 entry point is a fallback for a gate that has a
bias, not a speed claim -- this checkpoint's gate is bias-free and takes the
in-tree skinny GEMV.

**VRAM**: 0. The kernel allocates nothing but its output.

**Numerics**: the bias-free path returns BF16 logits from the same kernel the
rest of this tree uses, so it is unchanged relative to cuBLAS up to the
reduction order of split-K. The biased path returns FP32 logits where
``ReplicatedLinear`` returned BF16: strictly more accurate, but not
bit-identical, so a run that needs reproducible logits should keep the flag
off or use the bias-free route.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Widest decode batch the kernel is tuned and gated for.
MAX_M = 16
#: The fixed tile geometry of the donor kernel.
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 256
NUM_WARPS = 8


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    m,
    K: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    row_mask = rows < m
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        weight = tl.load(weight_ptr + n[:, None] * K + k[None, :])
        acc += tl.dot(x, tl.trans(weight), out_dtype=tl.float32)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + n)[None, :].to(tl.float32)
    tl.store(
        out_ptr + rows[:, None] * N + n[None, :],
        acc,
        mask=row_mask[:, None],
    )


def router_gemv_shape_ok(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """The layout half of the verdict: shapes, dtypes and strides only.

    Split out of :func:`router_gemv_applies` so a CPU test can ask it about the
    real gate shape.
    """
    if x.dim() != 2 or weight.dim() != 2:
        return False
    m, k = x.shape
    n, weight_k = weight.shape
    return bool(
        1 <= m <= MAX_M
        and k == weight_k
        and k % 128 == 0
        and n % BLOCK_N == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and (
            bias is None
            or (
                bias.shape == (n,)
                and bias.dtype == torch.bfloat16
                and bias.is_contiguous()
            )
        )
    )


def router_gemv_applies(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    require_cuda: bool = True,
) -> bool:
    """Whether :func:`router_gemv` accepts these operands (no exception, just a verdict).

    ``require_cuda`` covers the device half -- same device for all three
    operands and a CUDA x -- which the CPU unit tests switch off to exercise
    the layout half.
    """
    if not router_gemv_shape_ok(x, weight, bias):
        return False
    if not require_cuda:
        return True
    return bool(
        x.is_cuda
        and weight.device == x.device
        and (bias is None or bias.device == x.device)
    )


def router_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` into an FP32 router-logit tensor."""
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("x and weight must be 2-D")
    m, k = x.shape
    n, weight_k = weight.shape
    if not 1 <= m <= MAX_M:
        raise ValueError("router GEMV requires 1 <= M <= 16")
    if k != weight_k or k % 128 != 0 or n % BLOCK_N != 0:
        raise ValueError("router GEMV requires matching K % 128 == 0 and N % 32 == 0")
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("router GEMV requires CUDA tensors")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all router GEMV tensors must be on the same device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("router GEMV supports BF16 inputs and weights only")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("router GEMV inputs and weights must be contiguous")
    if bias is not None:
        if bias.shape != (n,) or bias.dtype != torch.bfloat16:
            raise ValueError("router GEMV bias must be contiguous BF16 with shape [N]")
        if not bias.is_contiguous():
            raise ValueError("router GEMV bias must be contiguous")

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    bias_arg = bias if bias is not None else weight
    _router_gemv_kernel[(n // BLOCK_N,)](
        x,
        weight,
        bias_arg,
        out,
        m,
        K=k,
        N=n,
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
    )
    return out


def router_gemv_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32 ``x @ weight.T + bias`` the kernel is compared against."""
    y = x.to(torch.float32) @ weight.to(torch.float32).t()
    return y if bias is None else y + bias.to(torch.float32)
