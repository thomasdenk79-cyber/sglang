# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Load-time NVFP4 packing for dense weights, with an on-disk cache.

Used by the N1 stages to move a dense FP8 weight-only path to the NVFP4 format the
experts already use (see `w4a16_nvfp4_gemv`).

Two things this module exists to get right:

**Quantise from BF16, not from FP8.** By the time the EAGLE worker reaches the draft
head, `Fp8LinearMethod.process_weights_after_loading` has already replaced the target's
`lm_head.weight` in place, so the obvious implementation would quantise the *already
FP8* rows and compound two lossy steps. Measured on the real head, that costs ~2 points
of argmax agreement against quantising from BF16 (N1_LOG.md 4). So the rows are re-read
straight from the checkpoint shard instead -- `safetensors` slices lazily, so only the
rows that are wanted are ever materialised and there is no 1.27 GB transient.

**Cache the packed result.** Packing 49152x2560 costs a few seconds and is entirely
deterministic, so it is written next to the token map under a key derived from the
model shard's identity, the tensor name, and the exact row set. Any change to the model
directory, the tensor, or the token map yields a different key, so a stale cache cannot
be picked up silently.
"""

import hashlib
import json
import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (
    FP4_SCALE_BLOCK,
    quantize_nvfp4,
)

logger = logging.getLogger(__name__)

#: Where packed weights are cached. Defaults to the user's application cache.
CACHE_DIR = os.environ.get(
    "SGLANG_NVFP4_CACHE_DIR",
    os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "flash-next-fast", "nvfp4"),
)


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a boolean switch the same way as the other fork-local flags."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


#: P0.2: large-M NVFP4 -> FP8 e4m3 prefill. Off by default: with every switch off
#: the build must stay identical to the donor FP8 baseline.
FP8_PREFILL_ENV = "SGLANG_NVFP4_DENSE_PREFILL_FP8"

#: This fork targets CUDA sm_120, where `sglang` uses e4m3fn (finite max 448), not
#: the ROCm e4m3fnuz (max 224). Keeping the dtype/max local (rather than importing
#: `fp8_dtype` / `fp8_max`) makes the scale bookkeeping testable on CPU.
_FP8_E4M3 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_FP8_SCALE_FLOOR = 1e-12


def _quantize_fp8_per_tensor(x: torch.Tensor):
    """Per-tensor e4m3 quantization: returns `(q, scale)` with `q.float() * scale ~= x`.

    `q` has the same shape as `x` and dtype `float8_e4m3fn`; `scale` is a `[1]`
    fp32 tensor. The scale is derived on-device (no `.item()` / host sync), so the
    bookkeeping is capture-safe in principle -- although large-M prefill is not
    captured today. Pure torch, hence CPU-testable.
    """
    xf = x.float()
    amax = xf.abs().amax()
    scale = (amax / _FP8_MAX).clamp(min=_FP8_SCALE_FLOOR).reshape(1)
    q = (xf / scale).clamp(min=-_FP8_MAX, max=_FP8_MAX).to(_FP8_E4M3)
    return q, scale


def _fp8_gemm_helper():
    """The fork's FP8 scaled GEMM (`fp8_utils.py:205`, used at `:1788-1963`), or None.

    Imported lazily so a CPU-only process, a switches-off build, or a stack without
    `sgl_kernel` keeps the BF16 fallback. Never let an optional import break serving.
    """
    try:
        from sglang.srt.layers.quantization.fp8_utils import fp8_scaled_mm
    except Exception as exc:  # optional backend / not a CUDA build
        logger.debug("NVFP4 dense: FP8 GEMM helper unavailable: %s", exc)
        try:
            from sgl_kernel import fp8_scaled_mm
        except Exception as exc2:
            logger.debug("NVFP4 dense: sgl_kernel FP8 GEMM unavailable: %s", exc2)
            return None
    return fp8_scaled_mm


def _index_path(model_dir: str) -> Optional[str]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    return None


def find_shard(model_dir: str, tensor_name: str) -> str:
    """Absolute path of the safetensors shard holding `tensor_name`.

    Prefers the index json; falls back to opening each shard's header, which is cheap
    because `safe_open` reads only the metadata block.
    """
    idx = _index_path(model_dir)
    if idx is not None:
        with open(idx) as f:
            wm = json.load(f).get("weight_map", {})
        if tensor_name in wm:
            return os.path.join(model_dir, wm[tensor_name])
    from safetensors import safe_open

    shards = sorted(
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    )
    for p in shards:
        with safe_open(p, framework="pt", device="cpu") as fh:
            if tensor_name in fh.keys():
                return p
    raise FileNotFoundError(f"{tensor_name} not found in any shard under {model_dir}")


def _shard_sig(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def _rows_sig(rows: Optional[torch.Tensor]) -> str:
    if rows is None:
        return "all"
    r = rows.detach().to("cpu", torch.int64).contiguous()
    h = hashlib.sha256(r.numpy().tobytes()).hexdigest()[:16]
    return f"{r.numel()}-{h}"


def cache_key(model_dir: str, tensor_name: str, rows: Optional[torch.Tensor]) -> str:
    shard = find_shard(model_dir, tensor_name)
    raw = "|".join(
        [os.path.realpath(model_dir), tensor_name, _shard_sig(shard), _rows_sig(rows)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def read_bf16_rows(
    model_dir: str, tensor_name: str, rows: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """`tensor_name`'s rows straight from the checkpoint, in its stored dtype.

    `get_slice` keeps this lazy: for a row subset only those rows are read off disk, so
    the peak transient is the subset, not the whole [248320, 2560] tensor.
    """
    from safetensors import safe_open

    path = find_shard(model_dir, tensor_name)
    with safe_open(path, framework="pt", device="cpu") as fh:
        if rows is None:
            return fh.get_tensor(tensor_name)
        sl = fh.get_slice(tensor_name)
        idx = rows.detach().to("cpu", torch.int64)
        # One contiguous read per row; the hot map is ~49k rows of 5 KB, ~250 MB total.
        return torch.cat([sl[int(i) : int(i) + 1, :] for i in idx.tolist()], dim=0)


def build_or_load(
    model_dir: str,
    tensor_name: str,
    rows: Optional[torch.Tensor] = None,
    device: str = "cuda",
    label: str = "",
):
    """-> (wq [N, K/2] uint8, bs [N, K/16] uint8 e4m3, gscale [1] fp32), all on `device`.

    Deterministic: same shard + same row set -> byte-identical output, which is what
    makes the cache safe and the offline acceptance measurement transferable.
    """
    import time

    key = cache_key(model_dir, tensor_name, rows)
    path = os.path.join(CACHE_DIR, f"{tensor_name.replace('/', '_')}.{key}.pt")
    tag = label or tensor_name
    if os.path.exists(path):
        t0 = time.time()
        blob = torch.load(path, map_location=device, weights_only=True)
        logger.info(
            "NVFP4 %s: loaded packed weight from cache in %.2fs (%s)",
            tag, time.time() - t0, os.path.basename(path),
        )
        return blob["wq"], blob["bs"], blob["gscale"]

    t0 = time.time()
    w = read_bf16_rows(model_dir, tensor_name, rows)
    t_read = time.time() - t0
    t0 = time.time()
    # Packed on the CPU, then only the *packed* tensors go to the device. The server
    # runs at mem_fraction 0.935 and reaches this with ~2 GB of VRAM free, while the
    # full target head would need 1.27 GB bf16 + an fp32 working set on top of it.
    # The packed result is 0.5625 B/element, and host RAM is not the constraint here.
    wq, bs, gscale = quantize_nvfp4(w)
    del w
    wq, bs, gscale = wq.to(device), bs.to(device), gscale.to(device)
    t_pack = time.time() - t0
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    torch.save(
        {"wq": wq.cpu(), "bs": bs.cpu(), "gscale": gscale.cpu()}, tmp
    )
    os.replace(tmp, path)  # atomic, so two concurrent starts cannot read a partial file
    logger.info(
        "NVFP4 %s: packed %s from BF16 in %.2fs (read %.2fs) -> %s",
        tag, tuple(wq.shape), t_pack, t_read, os.path.basename(path),
    )
    return wq, bs, gscale


# ---------------------------------------------------------------------------
# Serving-side linear method
# ---------------------------------------------------------------------------

#: Rows dequantised at a time in the large-M fallback, bounding its transient.
_FALLBACK_ROW_CHUNK = 16384
#: Up to this many rows, the fallback loops the GEMV instead of dequantising; this is
#: the CUDA-graph-captured draft-extend regime, where a dequant is not capture-safe.
_CHUNK_M_MAX = 128


class NVFP4DenseLinearMethod:
    """Weight-only NVFP4 for an lm_head, matching the dispatch in `logits_processor`.

    `_compute_lm_head` needs exactly three things of a quant method: the layer must have
    a `.weight` attribute, `apply(layer, x, bias)` must return logits, and the optional
    `apply_into(layer, x, out, bias)` may write straight into the shared fp32 next-token
    buffer (returning None to decline). The packed tensors live on the *layer*
    (`nvfp4_wq` / `nvfp4_bs` / `nvfp4_gscale`), following the convention the FP8 method
    uses for `weight_scale`, so one method instance can serve several layers.

    `layer.weight` is set to the packed codes. Nothing downstream interprets the draft
    head's `.weight` shape -- the FP8 path already stores it transposed as [K, H] -- but
    the attribute has to exist for `should_apply_lm_head_quant_method` to say yes.
    """

    def __init__(self, name: str = "lm_head"):
        self.name = name

    @staticmethod
    def attach(layer, wq, bs, gscale) -> None:
        layer.nvfp4_wq = wq
        layer.nvfp4_bs = bs
        layer.nvfp4_gscale = gscale

    @staticmethod
    def _ok(layer, x) -> bool:
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import _m_limit

        return (
            getattr(layer, "nvfp4_wq", None) is not None
            and x.dim() == 2
            and x.shape[0] <= _m_limit()
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.stride(1) == 1
        )

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import w4a16_nvfp4_gemv

        if self._ok(layer, x):
            y = w4a16_nvfp4_gemv(x, layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale)
        else:
            y = self._wide_m(layer, x)
        if bias is not None:
            y = y + bias
        return y

    def apply_into(
        self, layer, x: torch.Tensor, out: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import w4a16_nvfp4_gemv

        if bias is not None or not self._ok(layer, x):
            return None
        N = layer.nvfp4_wq.shape[0]
        if (
            tuple(out.shape) != (x.shape[0], N)
            or out.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device
            or out.stride(1) != 1
        ):
            return None
        return w4a16_nvfp4_gemv(
            x, layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale, out=out
        )

    def apply_into_split(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        out: torch.Tensor,
        out2: torch.Tensor,
        split_n: int,
        bias: Optional[torch.Tensor] = None,
    ) -> bool:
        """Two-destination variant of `apply_into`, mirroring Fp8LinearMethod.

        Columns [0, split_n) land in `out`, columns [split_n, N) in `out2`.
        The NVFP4 GEMV has a native SPLIT_N epilogue, so this avoids the
        separate split/reshape/cat kernel the generic fallback needs. Returns
        False when the GEMV path is unavailable (the caller must fall back).
        """
        if bias is not None or not self._ok(layer, x):
            return False
        N = layer.nvfp4_wq.shape[0]
        M = x.shape[0]
        if (
            not (0 < split_n < N)
            or tuple(out.shape) != (M, split_n)
            or tuple(out2.shape) != (M, N - split_n)
            or out.dtype not in (torch.bfloat16, torch.float32)
            or out2.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device
            or out2.device != x.device
            or out.stride(1) != 1
            or out2.stride(1) != 1
        ):
            return False
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import w4a16_nvfp4_gemv

        w4a16_nvfp4_gemv(
            x, layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale,
            out=out, out2=out2, split_n=split_n,
        )
        return True

    def _wide_m(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The M > NVFP4_GEMV_MAX_M path the decode GEMV cannot serve.

        Two regimes, because they have different constraints:

        - **Modest M (<= _CHUNK_M_MAX): loop the GEMV over knob-sized slices.** This is the
          draft-extend path, and draft-extend *is* CUDA-graph captured, so it has to be
          capture-safe -- no host->device copies, no syncs, no data-dependent shapes.
          Re-streaming the weight per slice costs bandwidth, but at these M it is a
          handful of slices.
        - **Large M: dequantise in row chunks and GEMM.** Real prefill, never captured.
          Looping the GEMV here would re-read the whole weight ~M/knob times; a GEMM
          amortises the weight read across all M rows instead. Chunking the dequant by
          output rows keeps the transient near 80 MB rather than the full BF16 weight.
          With `SGLANG_NVFP4_DENSE_PREFILL_FP8=1` the dequant goes to FP8 e4m3 and the
          GEMM is `sglang`'s own `fp8_scaled_mm`; otherwise it stays BF16 `torch.mm`.
        """
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (
            _m_limit,
            dequantize_nvfp4,
            w4a16_nvfp4_gemv,
        )

        wq, bs, gs = layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale
        M, N = x.shape[0], wq.shape[0]
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        chunk_m = _m_limit()
        if (
            M <= _CHUNK_M_MAX
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and chunk_m > 0
        ):
            for m0 in range(0, M, chunk_m):
                m1 = min(M, m0 + chunk_m)
                w4a16_nvfp4_gemv(x[m0:m1], wq, bs, gs, out=y[m0:m1])
            return y

        # P0.2 -- large-M prefill. Off by default, so the switches-off build is
        # unchanged. The FP8 branch reuses the existing GEMM helper; no new kernel.
        #
        # Scale convention: `bs` holds raw e4m3 bit patterns for 16-element blocks
        # and `gs` is the global fp32 scale. `dequantize_nvfp4` already forms
        # e2m1_code * e4m3_block_scale * gscale in fp32 -- we never touch `bs`/`gs`
        # directly, so there is no second convention to drift from the GEMV path.
        # `fp8_scaled_mm` computes (a @ b) * scales_a * scales_b, so expressing the
        # weight as `w_fp8 * w_scale` and x as `x_fp8 * x_scale` yields the right
        # product with two fp32 scalars and no host sync.
        #
        # Precision assumption: e2m1 (2 significand bits) times an e4m3 block scale
        # (4 bits) needs 6 bits -- exact in BF16 (8 bits) but not in e4m3 (4 bits),
        # so this branch loses ~2 weight mantissa bits, plus the usual bf16->e4m3
        # loss on activations. That is why it is switch-gated and must pass the
        # GSM8K/MMLU accuracy gate (<= 1% @ k=7) before it is shipped.
        use_fp8 = (
            _env_flag(FP8_PREFILL_ENV)
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
        )
        gemm = _fp8_gemm_helper() if use_fp8 else None
        x_fp8 = x_scale = None
        if gemm is not None:
            x_fp8, x_scale = _quantize_fp8_per_tensor(x)

        for r0 in range(0, N, _FALLBACK_ROW_CHUNK):
            r1 = min(N, r0 + _FALLBACK_ROW_CHUNK)
            w32 = dequantize_nvfp4(wq[r0:r1], bs[r0:r1], gs)
            # CUTLASS FP8 needs GEMM extents divisible by 16; a short tail chunk
            # silently takes the original BF16 path instead.
            if gemm is not None and (r1 - r0) % 16 == 0 and w32.shape[1] % 16 == 0:
                w_fp8, w_scale = _quantize_fp8_per_tensor(w32)
                # sgl_kernel's CUTLASS dispatch requires per-column scales_b
                # (`numel == N`); a [1] scalar raises "size of scales_b is not
                # matched". native_scalar_a_scale on SM120 keeps scales_a [1],
                # so expand only w_scale without a host sync.
                w_scale_col = w_scale.expand(r1 - r0).contiguous()
                # `.t()` matches `apply_fp8_linear`: that callsite also passes a
                # transposed, K-major [K, N] fp8 weight to `fp8_scaled_mm`.
                y[:, r0:r1] = gemm(
                    x_fp8, w_fp8.t(), x_scale, w_scale_col, out_dtype=torch.bfloat16
                )
            else:
                w = w32.to(torch.bfloat16)
                torch.mm(x.to(torch.bfloat16), w.t(), out=y[:, r0:r1])
                del w
            del w32
        return y


# ---------------------------------------------------------------------------
# N1 stage C: dense projections
# ---------------------------------------------------------------------------
#
# These are packed inside `Fp8LinearMethod.process_weights_after_loading`, *before* it
# quantises, rather than by a post-load walk. Two reasons, and the first is decisive:
#
# - **The BF16 weight is right there.** A post-load pass would have to re-read the
#   checkpoint, and these layers do not exist in it under their module names --
#   `qkv_proj` is fused from q/k/v_proj and `in_proj_qkvz` from in_proj_qkv + in_proj_z
#   by the loader (`qwen4_exp.py:1923-1932`). Packing at this point gets the true BF16
#   fused tensor for free and skips replicating that mapping.
# - It is the same hook that already preallocates the GEMV scratch, so ordering against
#   CUDA-graph capture is settled.

def _cat_flags():
    cats = []
    if _env_flag("SGLANG_LINEAR_ATTN_NVFP4"):
        cats.append("in_proj_qkvz")
    if _env_flag("SGLANG_ATTN_NVFP4"):
        cats.append("qkv_proj")
    return tuple(cats)


def maybe_pack_dense_nvfp4(layer) -> bool:
    """Pack `layer` to NVFP4 in place and install the GEMV method. True if handled.

    Called at the top of `Fp8LinearMethod.process_weights_after_loading`, where
    `layer.weight` is still the loaded BF16 tensor. Deliberately NOT applied to the GDN
    `out_proj`: measured 1.08-1.13x there for 0.22 % of the W16 step, and it is the one
    shape where the 4-bit GEMV falls to 47-52 % of roof -- below break-even (N1_LOG.md 7).
    """
    cats = _cat_flags()
    if not cats:
        return False
    prefix = getattr(layer, "prefix", "") or ""
    if not prefix:
        # Never silently skip: a layer with no prefix cannot be matched, and that is a
        # bug in the layer class rather than a reason to leave it FP8.
        logger.warning(
            "NVFP4 dense: %s has no .prefix, cannot match it against %s",
            type(layer).__name__, cats,
        )
        return False
    if not any(prefix.endswith(c) or f".{c}" in prefix for c in cats):
        return False
    w = getattr(layer, "weight", None)
    if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return False
    if w.shape[1] % FP4_SCALE_BLOCK:
        logger.warning("NVFP4 %s: K=%d not a multiple of 16, leaving FP8", prefix, w.shape[1])
        return False

    from sglang.srt.layers.quantization.w8a16_gemv import prealloc

    wq, bs, gscale = quantize_nvfp4(w.data)
    prealloc(wq.device)
    NVFP4DenseLinearMethod.attach(layer, wq, bs, gscale)
    try:
        del layer.weight
    except AttributeError:
        pass
    layer.weight = wq
    layer.weight_scale = None
    layer.input_scale = None
    layer.quant_method = NVFP4DenseLinearMethod(prefix)
    logger.info("NVFP4 dense: %s %s packed", prefix, tuple(wq.shape))
    return True

