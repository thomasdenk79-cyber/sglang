#!/usr/bin/env python3
"""CPU-only self-test for the P0.2 NVFP4 -> FP8 large-M prefill switch.

Run from the repo root with any CPU torch install, e.g.::

    CUDA_VISIBLE_DEVICES="" python tests/test_nvfp4_dense_fp8_prefill.py

It checks shape/dtype/scale bookkeeping with synthetic CPU tensors and simulates
`fp8_scaled_mm` with a float32 matmul: the real GEMM kernel needs CUDA and is
deliberately NOT validated here. If torch or the sglang modules cannot be
imported on CPU, the script prints SKIP and exits 0 -- it never claims a check
it could not run.
"""

from __future__ import annotations

import importlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "python"))


def _skip(reason: str):
    print(f"SKIP: {reason}")
    sys.exit(0)


try:
    import torch
except Exception as exc:
    _skip(f"torch not importable on CPU ({exc})")

try:
    m = importlib.import_module("sglang.srt.layers.quantization.nvfp4_dense")
    gemv = importlib.import_module("sglang.srt.layers.quantization.w4a16_nvfp4_gemv")
except Exception as exc:
    _skip(f"sglang modules not importable on CPU ({exc}); run inside a SGLang env")


def check(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"ok - {msg}")


def test_flag() -> None:
    """Default off; accepted truthy spellings; no import-time env capture."""
    key = m.FP8_PREFILL_ENV
    check(key == "SGLANG_NVFP4_DENSE_PREFILL_FP8", "flag name matches the plan")
    old = os.environ.get(key)
    try:
        for value in (None, "0", "false", "junk"):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
            check(not m._env_flag(key), f"flag off for {value!r}")
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ[key] = value
            check(m._env_flag(key), f"flag on for {value!r}")
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_nvfp4_scale_shapes() -> None:
    """Raw e4m3 block bits + global fp32 scale -> fp32 dequant, same convention."""
    torch.manual_seed(0)
    N, K = 64, 256
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    wq, bs, gs = m.quantize_nvfp4(w)
    check(wq.shape == (N, K // 2) and wq.dtype == torch.uint8, "wq [N,K/2] uint8")
    check(bs.shape == (N, K // 16) and bs.dtype == torch.uint8, "bs [N,K/16] raw e4m3 bits")
    check(gs.shape == (1,) and gs.dtype == torch.float32, "gscale [1] fp32")
    w32 = gemv.dequantize_nvfp4(wq, bs, gs)
    check(w32.shape == (N, K) and w32.dtype == torch.float32, "dequant [N,K] fp32")
    err = (w32 - w.float()).abs().max().item() / w.float().abs().max().item()
    check(err < 0.35, f"NVFP4 round-trip within 35% of amax (got {err:.4f})")


def test_fp8_quant_bookkeeping() -> None:
    """q.float() * scale ~= x; zero input still yields finite positive scale."""
    torch.manual_seed(1)
    x = torch.randn(32, 256) * 0.1
    q, s = m._quantize_fp8_per_tensor(x)
    check(q.shape == x.shape and q.dtype == torch.float8_e4m3fn, "fp8 e4m3, same shape")
    check(s.shape == (1,) and s.dtype == torch.float32 and s.item() > 0, "scale [1] fp32 > 0")
    rel = ((q.float() * s - x).abs().max() / x.abs().max()).item()
    check(rel < 0.10, f"fp8 relative error < 0.10 (got {rel:.4f})")
    qz, sz = m._quantize_fp8_per_tensor(torch.zeros(4, 32))
    check(torch.isfinite(sz).all() and sz.item() > 0, "zero input: finite positive scale")
    check(qz.float().abs().max().item() == 0, "zero input: all-zero codes")


def test_gemm_scales() -> None:
    """(xq @ wq) * x_scale * w_scale matches the fp32 reference, no missing scale."""
    torch.manual_seed(2)
    M, N, K = 24, 64, 256
    x = (torch.randn(M, K) * 0.1).to(torch.bfloat16)
    wq, bs, gs = m.quantize_nvfp4((torch.randn(N, K) * 0.02).to(torch.bfloat16))
    w32 = gemv.dequantize_nvfp4(wq, bs, gs)
    xq, xs = m._quantize_fp8_per_tensor(x)
    wq8, ws = m._quantize_fp8_per_tensor(w32)
    # Stand-in for fp8_scaled_mm(a, b, sa, sb): fp32 accumulate, then both scales.
    sim = (xq.float() @ wq8.float().t()) * xs * ws
    ref = x.float() @ w32.t()
    rel = (sim - ref).abs().mean().item() / ref.abs().mean().item()
    check(rel < 0.15, f"scaled-GEMM bookkeeping within 15% (got {rel:.4f})")


def test_fallback_on_cpu() -> None:
    """Flag must not change the BF16 fallback when CUDA/the helper is unavailable."""
    torch.manual_seed(3)
    M, N, K = 24, 64, 256
    x = (torch.randn(M, K) * 0.1).to(torch.bfloat16)
    wq, bs, gs = m.quantize_nvfp4((torch.randn(N, K) * 0.02).to(torch.bfloat16))
    helper = m._fp8_gemm_helper()  # may be importable even with no visible device
    print(f"info - FP8 GEMM helper on this host: "
          f"{'available' if helper is not None else 'unavailable'}")

    class _Layer:
        pass

    layer = _Layer()
    m.NVFP4DenseLinearMethod.attach(layer, wq, bs, gs)
    method = m.NVFP4DenseLinearMethod("self-test")
    key = m.FP8_PREFILL_ENV
    old = os.environ.get(key)
    try:
        os.environ[key] = "0"
        y0 = method.apply(layer, x)
        os.environ[key] = "1"
        y1 = method.apply(layer, x)
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
    check(y0.shape == (M, N) and y0.dtype == torch.bfloat16, "fallback shape/dtype")
    check(torch.equal(y0, y1), "flag on + no CUDA/helper == BF16 fallback (bitwise)")


if __name__ == "__main__":
    test_flag()
    test_nvfp4_scale_shapes()
    test_fp8_quant_bookkeeping()
    test_gemm_scales()
    test_fallback_on_cpu()
    print("PASS: CPU-only P0.2 bookkeeping checks (GPU GEMM not validated)")
