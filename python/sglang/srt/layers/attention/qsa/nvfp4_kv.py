"""NVFP4 KV reads for QSA.

The NVFP4 recipe stores each full-attention layer as packed E2M1 pairs
``[slots, heads, dim/2]`` (even element in the low nibble) plus one token-linear
E4M3 block scale per 16 elements ``[slots, heads, dim/16]``; a value is
``e2m1 * block_scale * global_scale`` (flashinfer ``nvfp4_kv_quantize``).

QSA never hands the pool to an attention kernel: decode, verify and draft gather
the selected top-k rows into a scratch, and prefix chunk-prefill gathers the
whole context into a packed tensor. These kernels do the same gathers but
dequantize on the way, so the scratch keeps its BF16 (or FP8) dtype, every
downstream kernel is unchanged, and the pool needs no dequant workspace.
The global scales are 1-element device tensors, so the gathers are CUDA-graph
safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NVFP4_BLOCK_SIZE = 16


@triton.jit
def _e2m1_to_f32(nib):
    # nib: 4-bit code, bit 3 = sign, bits 2..1 = exponent, bit 0 = mantissa.
    # Magnitudes 0,.5,1,1.5,2,3,4,6 as exact piecewise-linear fp32 math
    # (tl.exp2 may lower to ex2.approx and miss the last bit).
    m = (nib & 7).to(tl.float32)
    val = tl.where(m < 4.0, 0.5 * m, tl.where(m < 6.0, m - 2.0, 2.0 * m - 8.0))
    return tl.where((nib & 8) != 0, -val, val)


@triton.jit
def _dequant_rows(
    data, sf, rows, row_mask, head, g, heads: tl.constexpr, dim: tl.constexpr, dims
):
    # rows: [R] int64 slot ids; returns [R, BLOCK_D] float32 (0 where masked).
    mask = row_mask[:, None] & (dims[None, :] < dim)
    byte = tl.load(
        data
        + rows[:, None] * (heads * (dim // 2))
        + head * (dim // 2)
        + dims[None, :] // 2,
        mask=mask,
        other=0,
    )
    nib = (byte >> ((dims[None, :] % 2) * 4).to(tl.uint8)) & 15
    scale = tl.load(
        sf
        + rows[:, None] * (heads * (dim // 16))
        + head * (dim // 16)
        + dims[None, :] // 16,
        mask=mask,
        other=0,
    )
    scale = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return _e2m1_to_f32(nib) * scale * g


@triton.jit
def _compact_kv_nvfp4(
    k,
    v,
    k_sf,
    v_sf,
    k_g,
    v_g,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    # Same row selection and layouts as sparse_attn._compact_kv; only the element
    # read differs.
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    if ZERO_FILL:
        # Strided (page-aligned) packing for the paged decode kernel: every slot in
        # [valid_count, pad_cols) is written, as zeros, never left stale.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = valid[:, None] & (dims[None, :] < dim)
    kv = _dequant_rows(k, k_sf, slots, valid, head, tl.load(k_g), heads, dim, dims)
    tl.store(out_k + dst, kv.to(out_k.dtype.element_ty), mask=store_mask)
    vv = _dequant_rows(v, v_sf, slots, valid, head, tl.load(v_g), heads, dim, dims)
    tl.store(out_v + dst, vv.to(out_v.dtype.element_ty), mask=store_mask)


def _check_layer(data: torch.Tensor, sf: torch.Tensor) -> tuple[int, int]:
    if data.ndim != 3 or not data.is_contiguous() or not sf.is_contiguous():
        raise ValueError("NVFP4 QSA KV layers must be contiguous [slots, heads, dim/2]")
    slots, heads, half = data.shape
    dim = half * 2
    if dim % NVFP4_BLOCK_SIZE or tuple(sf.shape) != (
        slots,
        heads,
        dim // NVFP4_BLOCK_SIZE,
    ):
        raise ValueError(
            f"NVFP4 block scales {tuple(sf.shape)} do not match data {tuple(data.shape)}"
        )
    return heads, dim


def qwen_sparse_kv_extraction_compact_nvfp4_triton(
    k,
    v,
    k_sf,
    v_sf,
    k_g,
    v_g,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """NVFP4 twin of ``sparse_attn.qwen_sparse_kv_extraction_compact_triton``.

    k/v: packed ``[slots, heads, dim/2]`` uint8; k_sf/v_sf: ``[slots, heads, dim/16]``
    (any 1-byte dtype); k_g/v_g: 1-element float32 device tensors;
    out_k/out_v: ``[rows, heads, dim]`` BF16 or FP8 scratch. ``zero_fill_cols``
    has the same meaning as in the FP8/BF16 kernel.
    """
    heads, dim = _check_layer(k, k_sf)
    _check_layer(v, v_sf)
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    _compact_kv_nvfp4[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k.view(torch.uint8),
        v.view(torch.uint8),
        k_sf.view(torch.uint8),
        v_sf.view(torch.uint8),
        k_g,
        v_g,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        num_warps=8,
    )


@triton.jit
def _gather_dequant(
    data,
    sf,
    g,
    index,
    out,
    n,
    heads: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid, head = tl.program_id(0), tl.program_id(1)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    row_mask = rows < n
    slots = tl.load(index + rows, mask=row_mask, other=0).to(tl.int64)
    x = _dequant_rows(data, sf, slots, row_mask, head, tl.load(g), heads, dim, dims)
    dst = rows[:, None].to(tl.int64) * heads * dim + head * dim + dims[None, :]
    tl.store(
        out + dst,
        x.to(out.dtype.element_ty),
        mask=row_mask[:, None] & (dims[None, :] < dim),
    )


def nvfp4_gather_dequant(data, sf, g, index, out_dtype):
    """``data.index_select(0, index)`` for an NVFP4 layer, dequantized to out_dtype.

    Returns ``[len(index), heads, dim]``.
    """
    heads, dim = _check_layer(data, sf)
    n = index.numel()
    out = torch.empty((n, heads, dim), dtype=out_dtype, device=data.device)
    if n == 0:
        return out
    block_n = 16
    _gather_dequant[(triton.cdiv(n, block_n), heads)](
        data.view(torch.uint8),
        sf.view(torch.uint8),
        g,
        index.contiguous(),
        out,
        n,
        heads,
        dim,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )
    return out


def pack_qsa_prefill_kv_nvfp4(
    *,
    k,
    v,
    k_sf,
    v_sf,
    k_g,
    v_g,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    total_k: int,
    output_dtype,
):
    """NVFP4 twin of ``sparse_attn.pack_qsa_prefill_kv``: each request's first
    ``seq_lens[b]`` cached rows, packed back-to-back at ``cu_k[b]`` and dequantized
    to ``output_dtype``."""
    batch = req_indices.numel()
    batch_ids = torch.repeat_interleave(
        torch.arange(batch, device=k.device), seq_lens, output_size=total_k
    )
    positions = torch.arange(total_k, device=k.device) - cu_k[batch_ids].long()
    slots = req_to_token[req_indices.long()[batch_ids], positions]
    return (
        nvfp4_gather_dequant(k, k_sf, k_g, slots, output_dtype),
        nvfp4_gather_dequant(v, v_sf, v_g, slots, output_dtype),
    )


# ---------------------------------------------------------------------------
# FP8 scratch recipe
# ---------------------------------------------------------------------------
# A dequantized row ``e2m1 * block_scale * global_scale`` overflows the E4M3
# range (448) whenever the calibrated global scale is large, which is why the
# gathers historically dequantized into a BF16 scratch.  Dividing by the
# per-layer *descale* ``6 * global_scale`` keeps the stored value inside the
# E4M3 range provably -- ``|e2m1 * block_scale| / 6 <= 6 * 448 / 6 == 448`` --
# and the downstream kernels undo it exactly like an FP8 KV pool's
# k_scale/v_scale: the true value is ``stored * descale``.
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0


def nvfp4_fp8_descale(global_scale: torch.Tensor) -> torch.Tensor:
    """E4M3-scratch descale for one K/V global scale: value = stored * descale.

    Returns a 0-dim fp32 tensor so bf16 consumers multiply without dtype
    promotion; CUDA-graph safe (derived from the method's device tensor).
    """
    return (global_scale * E2M1_MAX).reshape(())


def nvfp4_fp8_gather_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """The global scale the gather kernel must apply for an FP8 (descaled) scratch.

    ``g / descale`` is exactly ``1 / E2M1_MAX``; it stays a [1] device tensor so
    a captured gather reads it on replay. A zero (uninitialized) global scale
    maps to a zero gather scale, matching the plain dequant's all-zero output
    instead of producing NaN.
    """
    descale = global_scale * E2M1_MAX
    return torch.where(descale == 0, torch.zeros_like(descale), global_scale / descale)


def reference_dequant(data, sf, g, out_dtype=torch.float32):
    """Independent PyTorch reference:
    ``[m, heads, dim/2]`` + ``[m, heads, dim/16]`` -> ``[m, heads, dim]``."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=data.device,
    )
    b = data.view(torch.uint8)
    m, heads, half = b.shape
    codes = torch.empty((m, heads, half * 2), dtype=torch.long, device=data.device)
    codes[..., 0::2] = (b & 0x0F).long()
    codes[..., 1::2] = (b >> 4).long()
    vals = lut[codes].view(m, heads, half * 2 // 16, 16)
    scale = sf.view(torch.uint8).view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return (vals * scale * g.float().reshape(())).view(m, heads, half * 2).to(out_dtype)


__all__ = [
    "E2M1_MAX",
    "FP8_E4M3_MAX",
    "nvfp4_fp8_descale",
    "nvfp4_fp8_gather_scale",
    "nvfp4_gather_dequant",
    "pack_qsa_prefill_kv_nvfp4",
    "qwen_sparse_kv_extraction_compact_nvfp4_triton",
    "reference_dequant",
]
