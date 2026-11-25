"""QSA NVFP4 decode through FlashInfer's Cake kernels (SGLANG_QSA_NVFP4_DECODE=cake).

`flashinfer.experimental.cake_qsa_nvfp4` (FlashInfer v0.7.4) reads the NVFP4 pool
in place: packed E2M1 ``[slots, heads, 128]`` and token-linear E4M3 block scales
``[slots, heads, 16]`` are its NHD cache with ``page_size = 1``, and
``req_to_token`` is its block table. No scratch, no gather, no pool rewrite.

The package is loaded standalone (``scripts/cake_qsa/build_cake_qsa_sm120.py``
builds it from the v0.7.4 sources into ``$SGLANG_QSA_CAKE_PATH/cake_qsa_nvfp4``),
so it never touches the donor's own FlashInfer tree.

Per call the only extra work is the int32 selection ``[rows, topk + 1]``: entries
outside ``[0, seq_len)`` become -1 and the last column is the full width. The
global scales are host floats baked into the launch; they are load-time
constants, read once per layer outside any capture.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

import torch

HEAD_DIM = 256
# sigmoid(GATE_OPEN) == 1.0 exactly in fp32: the kernel's mandatory output gate
# becomes the identity (the model applies the real gate afterwards).
GATE_OPEN = 1.0e4

_module = None
_load_error: Optional[str] = None


def load_cake():
    """The standalone ``cake_qsa_nvfp4`` package, or None (reason in load_error())."""
    global _module, _load_error
    if _module is not None or _load_error is not None:
        return _module
    path = os.environ.get("SGLANG_QSA_CAKE_PATH", "")
    try:
        if path and path not in sys.path:
            sys.path.insert(0, path)
        mod = importlib.import_module("cake_qsa_nvfp4")
        backend = importlib.import_module("cake_qsa_nvfp4.cake_backend")
        _module = backend if mod is not None else None
    except Exception as exc:  # missing build, tvm_ffi, toolkit mismatch
        _load_error = f"{type(exc).__name__}: {exc}"
    return _module


def load_error() -> Optional[str]:
    return _load_error


def unsupported_reason(q, k_raw, scaling: float) -> Optional[str]:
    if q.dtype != torch.bfloat16:
        return f"q dtype {q.dtype}"
    if q.shape[-1] != HEAD_DIM or k_raw.shape[-1] * 2 != HEAD_DIM:
        return f"head dim {q.shape[-1]}"
    kv_heads = k_raw.shape[1]
    if q.shape[1] % kv_heads or q.shape[1] // kv_heads > 16:
        return f"group {q.shape[1]}/{kv_heads}"
    if abs(scaling * HEAD_DIM**0.5 - 1.0) > 1e-6:
        return f"sm_scale {scaling} != 1/sqrt({HEAD_DIM})"
    return None


def prepare_selection(topk_indices, seq_lens, out):
    """Write the Cake selection into ``out[:rows, :topk + 1]`` (graph safe)."""
    rows, topk = topk_indices.shape
    sel = out[:rows, : topk + 1]
    lens = seq_lens[:rows].to(torch.int32).unsqueeze(1)
    valid = (topk_indices >= 0) & (topk_indices < lens)
    sel[:, :topk].copy_(torch.where(valid, topk_indices, -1))
    sel[:, topk].fill_(topk)
    return sel


def cake_nvfp4_decode(
    q: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    k_scale: float,
    v_scale: float,
    req_to_token: torch.Tensor,
    row_req: torch.Tensor,
    selection: torch.Tensor,
    gate: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``q [rows, hq, 256]`` bf16 over the pool; ``selection`` from prepare_selection."""
    cake = load_cake()
    if cake is None:
        raise RuntimeError(f"cake_qsa_nvfp4 unavailable: {_load_error}")
    slots, heads, half = k_raw.shape
    rows = q.shape[0]
    return cake.qsa_sparse_paged_attention(
        q,
        k_raw.view(torch.uint8).view(slots, 1, heads, half),
        v_raw.view(torch.uint8).view(slots, 1, heads, half),
        selection,
        req_to_token,
        row_req[:rows],
        False,
        out,
        k_scale,
        v_scale,
        output_gate=gate[:rows],
        k_sf=k_sf.view(torch.uint8).view(torch.float8_e4m3fn).view(slots, 1, heads, -1),
        v_sf=v_sf.view(torch.uint8).view(torch.float8_e4m3fn).view(slots, 1, heads, -1),
        workspace_buffer=workspace,
    )


def workspace_bytes(rows: int, num_qo_heads: int, num_kv_heads: int, width: int) -> int:
    """Bytes for every call up to these sizes (largest split count the width allows)."""
    cake = load_cake()
    splits = max(1, min(-(-width // 64), int(cake.MAX_SPLITS)))
    return int(cake.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits))
