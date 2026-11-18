"""CPU tests for the MXFP8 skinny GEMV (mxfp8_skinny_gemv).

The kernel comparison runs the Triton interpreter in a subprocess, like
test_sm120_bf16_skinny_gemv.py does: TRITON_INTERPRET has to be set before
triton is imported. Shapes are this checkpoint's real MXFP8 dense linears
(/srv/llm-models/Qwen3.8-Flash-Next-FP4-Local/config.json, hidden_size 2560),
with N sub-sampled for the interpreter and every K covered.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from sglang.srt.layers.quantization import mxfp8_skinny_gemv as g
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

# (name, N, K, layers) as the model builds them. Spelling out the fused cases
# from the checkpoint headers of /srv/llm-models/Qwen3.8-Flash-Next-FP4-Local
# (all dense weights are F8_E4M3 [N, K] with a U8 [N, K // 32] scale) and
# text_config hidden_size=2560:
#   linear_attn.in_proj_qkvz  10240 (in_proj_qkv) + 6144 (in_proj_z) = 16384
#   self_attn.qkv_proj        12288 (q) + 512 (k) + 512 (v)          = 13312
#   shared_expert.gate_up_proj  640 (gate) + 640 (up)                =  1280
# The mtp.* copies of these are BF16 in the checkpoint (quantization_config
# ignore list), so the layer counts below are the main tower's only.
REAL_SHAPES = [
    ("linear_attn.in_proj_qkvz", 16384, 2560, 36),
    ("linear_attn.out_proj", 2560, 6144, 36),
    ("self_attn.qkv_proj", 13312, 2560, 12),
    ("self_attn.o_proj", 2560, 6144, 12),
    ("self_attn.indexer.index_qk_proj", 640, 2560, 12),
    ("shared_expert.gate_up_proj", 1280, 2560, 48),
    ("shared_expert.down_proj", 2560, 640, 48),
]
# linear_attn.in_proj_ba ([96, 2560] = in_proj_b 48 + in_proj_a 48) is also
# MXFP8 in the checkpoint, but a [96, 2560] layer is dequantized to BF16 at load
# by Fp8LinearMethod._dequantize_mxfp8_small_shape_to_bf16 (n < 128), so it
# never reaches this kernel.
M_VALUES = [1, 2, 4, 8, 16]

_ALL_KS = sorted({k for _, _, k, _ in REAL_SHAPES})
_SHAPE_BY_NAME = {name: (n, k) for name, n, k, _ in REAL_SHAPES}


def _quantize_bf16(w_fp32, scale_bytes=None, group=32):
    """A deterministic MXFP8 pair (e4m3 weight, UE8M0 scale bytes) on CPU."""
    n, k = w_fp32.shape
    assert k % group == 0
    if scale_bytes is None:
        # 2^-9 .. 2^10, the range a quantizer lands in for BF16 activations.
        exponents = torch.randint(118, 138, (n, k // group), dtype=torch.uint8)
    else:
        exponents = scale_bytes
    descale = (exponents.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    scaled = w_fp32.view(n, k // group, group) / descale
    q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k).contiguous()
    return q, exponents.contiguous()


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("name", [s[0] for s in REAL_SHAPES])
def test_plan_is_legal_for_every_real_shape(name, m):
    n, k = _SHAPE_BY_NAME[name]
    block_n, block_k, splits, use_dot, warps, stages = g._plan(m, n, k, 188)
    assert block_k % g.GROUP_K == 0, "BLOCK_K must cover whole UE8M0 groups"
    assert block_k >= g.GROUP_K and splits >= 1 and splits <= g._MAX_SPLITS
    assert 1 <= warps and 1 <= stages
    # The broadcast-multiply path (use_dot False) is legal at M == 1 only; every
    # wider M has to go through tl.dot, which is what _use_dot enforces.
    assert g._use_dot(m, use_dot) == (m > 1 or use_dot)
    if not use_dot:
        assert m == 1
    block_m = g._block_m(m, use_dot)
    assert block_m >= m and block_m & (block_m - 1) == 0
    # The split-K reduction reads and writes one fp32 tile per (n block, split).
    n_blocks = -(-n // block_n)
    assert n_blocks <= g._WS_COUNTERS
    assert n_blocks * splits * block_m * block_n <= g._WS_FLOATS


@pytest.mark.parametrize("k", _ALL_KS)
def test_scale_shape_is_one_byte_per_group(k):
    w = torch.zeros(128, k, dtype=torch.float8_e4m3fn)
    s = torch.zeros(128, k // g.GROUP_K, dtype=torch.uint8)
    x = torch.zeros(4, k, dtype=torch.bfloat16)
    assert g.mxfp8_skinny_gemv_applies(x, w, s)
    # A scale for the wrong group count, or not contiguous, is not this kernel.
    assert not g.mxfp8_skinny_gemv_applies(x, w, s[:, : k // g.GROUP_K - 1])
    assert not g.mxfp8_skinny_gemv_applies(x, w, s.t().contiguous().t())
    assert not g.mxfp8_skinny_gemv_applies(x, w, s.to(torch.float32))
    assert not g.mxfp8_skinny_gemv_applies(x.half(), w, s)


def test_applies_gates_m_and_grad_and_layout():
    w = torch.zeros(512, 2560, dtype=torch.float8_e4m3fn)
    s = torch.zeros(512, 80, dtype=torch.uint8)
    x = torch.zeros(16, 2560, dtype=torch.bfloat16)
    assert g.mxfp8_skinny_gemv_applies(x, w, s)
    assert not g.mxfp8_skinny_gemv_applies(
        torch.zeros(17, 2560, dtype=torch.bfloat16), w, s
    )
    assert not g.mxfp8_skinny_gemv_applies(x, w, s, max_m=8)
    assert not g.mxfp8_skinny_gemv_applies(x[:, :1280], w, s)  # K mismatch
    assert not g.mxfp8_skinny_gemv_applies(x.unsqueeze(0), w, s)  # 3D
    assert not g.mxfp8_skinny_gemv_applies(x, w.t().contiguous().t(), s)  # not K-contig
    # Pinned (requires_grad) operands need autograd, so the gate refuses them
    # unless a no_grad context is active.
    w.requires_grad_(True)
    assert not g.mxfp8_skinny_gemv_applies(x, w, s)
    with torch.no_grad():
        assert g.mxfp8_skinny_gemv_applies(x, w, s)


def test_reference_matches_the_loader_dequant():
    from sglang.srt.layers.quantization.mxfp8_block_convert import (
        dequant_mxfp8_2d_to_bf16,
    )

    torch.manual_seed(0)
    w, s = _quantize_bf16(torch.randn(64, 128) * 0.05)
    x = torch.randn(4, 128).bfloat16()
    ref = g.mxfp8_skinny_gemv_reference(x, w, s)
    deq = dequant_mxfp8_2d_to_bf16(w, s)  # the bf16 round the loader applies
    torch.testing.assert_close(ref, x.float() @ deq.float().t(), atol=0, rtol=2e-2)
    # UE8M0 decoding: 0x00 is a zero scale, 0x7F is 1.0, 0xFE is 2**127.
    ones = torch.full((1, 4), 0x7F, dtype=torch.uint8)
    q1 = torch.ones(1, 128, dtype=torch.float8_e4m3fn)
    torch.testing.assert_close(g.mxfp8_dequant_to_fp32(q1, ones), torch.ones(1, 128))
    torch.testing.assert_close(
        g.mxfp8_dequant_to_fp32(q1, torch.zeros_like(ones)), torch.zeros(1, 128)
    )
    assert torch.isfinite(
        g.mxfp8_dequant_to_fp32(q1, torch.full_like(ones, 0xFE))
    ).all()
    assert g.mxfp8_dequant_to_fp32(q1, torch.full_like(ones, 0xFF)).isinf().all()


# --------------------------------------------------------------------------
# Kernel vs fp32 reference, in the Triton interpreter
# --------------------------------------------------------------------------

#: N sub-sampled for the interpreter (a full 16384 rows x 2560 K would run for
#: minutes); K is always the real one, and BLOCK_K >= 32 covers every group.
_INTERPRETER_N = 96

_INTERPRETER_CHECK = textwrap.dedent(
    """
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime import interpreter as itp

    # The interpreter keeps bf16 as raw uint16 and its create_dot only widens
    # fp8, so a bf16 tl.dot multiplies bit patterns. Widen bf16 operands to
    # fp32 first; the kernel itself is unchanged.
    _dot = itp.InterpreterBuilder.create_dot

    def _widen(h):
        if h.dtype == tl.bfloat16:
            data = (h.data.astype(np.uint32) << 16).view(np.float32)
            return itp.TensorHandle(data, tl.float32)
        return h

    itp.InterpreterBuilder.create_dot = lambda self, a, b, *r: _dot(
        self, _widen(a), _widen(b), *r
    )

    from sglang.srt.layers.quantization import mxfp8_skinny_gemv as g

    SHAPES = [
        ("linear_attn.in_proj_qkvz", 16384, 2560),
        ("linear_attn.out_proj", 2560, 6144),
        ("self_attn.qkv_proj", 13312, 2560),
        ("self_attn.o_proj", 2560, 6144),
        ("self_attn.indexer.index_qk_proj", 640, 2560),
        ("shared_expert.gate_up_proj", 1280, 2560),
        ("shared_expert.down_proj", 2560, 640),
    ]
    GROUP = g.GROUP_K
    N_SUB = 96

    def quantize(w_fp32, scale_bytes=None):
        n, k = w_fp32.shape
        if scale_bytes is None:
            exponents = torch.randint(118, 138, (n, k // GROUP), dtype=torch.uint8)
        else:
            exponents = scale_bytes
        descale = (exponents.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
        q = (
            (w_fp32.view(n, k // GROUP, GROUP) / descale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .view(n, k)
            .contiguous()
        )
        return q, exponents.contiguous()

    torch.manual_seed(0)
    ws = torch.empty(g._WS_FLOATS)
    cnt = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
    failures = []

    def run(name, n, k, m, cfg, scale_bytes=None):
        cfg = cfg or g._plan(m, n, k, 188)
        x = torch.randn(m, k).bfloat16()
        w, s = quantize(torch.randn(n, k) * 0.05, scale_bytes)
        ref = g.mxfp8_skinny_gemv_reference(x, w, s)
        for attempt in range(2):  # the second launch needs the counters reset
            y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
            g._launch(ws, cnt, x, w, s, y, m, n, k, cfg)
            got = y.float()
            scale = ref.abs().amax().clamp(min=1e-6)
            abs_err = (got - ref).abs().max().item()
            rel = (got - ref).abs() / ref.abs().clamp(min=1e-3)
            max_rel = rel.max().item()
            if abs(abs_err) > 2e-2 * scale:
                failures.append((name, m, cfg, attempt, "abs", abs_err, scale.item()))
            if max_rel > 5e-2:
                failures.append((name, m, cfg, attempt, "rel", max_rel, scale.item()))
            if int(cnt.abs().sum()) != 0:
                failures.append((name, m, cfg, attempt, "counters", None, None))
        # out= writes the caller's buffer instead of allocating one.
        buf = torch.full((m, n), -1.0, dtype=torch.bfloat16)
        out = g.mxfp8_skinny_gemv(x, w, s, out=buf, cfg=cfg)
        if out is not buf or not torch.equal(out, y):
            failures.append((name, m, cfg, "out=", None, None))

    for name, n, k in SHAPES:
        for m in (1, 4, 16):
            run(name, N_SUB, k, m, None)
    # Forced tiles: a one-group BLOCK_K, a single split, and the broadcast path
    # at a shape whose tuned plan uses tl.dot.
    run("forced-broadcast", N_SUB, 2560, 1, (16, 32, 8, False, 4, 3))
    run("forced-narrow", N_SUB, 2560, 4, (32, 32, 4, True, 4, 2))
    run("forced-single", N_SUB, 6144, 16, (32, 128, 1, True, 4, 3))
    # Sub-normal (2**-126) and zero scales, and the largest finite one, on rows
    # that are all zero weight (a zero-scale group must contribute exactly 0).
    extreme = torch.full((N_SUB, 2560 // GROUP), 0x01, dtype=torch.uint8)
    extreme[: N_SUB // 2] = 0x00
    extreme[N_SUB // 4] = 0xFE
    run("extreme-scales", N_SUB, 2560, 4, (32, 128, 5, True, 4, 3), extreme)
    run("extreme-scales-m1", N_SUB, 640 // 4, 1, (16, 32, 4, False, 4, 3), extreme.clone()[:, : 160 // GROUP])
    if failures:
        raise AssertionError(failures[:12])
    print("OK")
    """
)


def test_kernel_matches_reference_in_triton_interpreter():
    root = os.path.abspath(os.path.join(os.path.dirname(g.__file__), *[".."] * 4))
    env = dict(os.environ, TRITON_INTERPRET="1")
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _INTERPRETER_CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), (
        out.stdout[-4000:] + "\n" + out.stderr[-8000:]
    )


# --------------------------------------------------------------------------
# The Fp8LinearMethod hook
# --------------------------------------------------------------------------


def _method(use_mxfp8=True, backend="flashinfer_cutlass", marlin=False):
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_mxfp8 = use_mxfp8
    method.use_marlin = marlin
    method.block_quant = use_mxfp8
    method.convert_mxfp8_to_block = False
    method.block_fp8_as_mxfp8 = False
    method.mxfp8_prefill_autotune_min_tokens = None
    method.mxfp8_dense_backend = (
        None if backend is None else Mxfp8DenseGemmBackend(backend)
    )
    return method


def _layer(n=128, k=128, dequantized=False, shuffled=False):
    from types import SimpleNamespace

    if dequantized:
        # What _dequantize_mxfp8_small_shape_to_bf16 leaves behind.
        return SimpleNamespace(
            weight=torch.zeros(n, k, dtype=torch.bfloat16),
            weight_scale_inv=None,
            dequantized_bf16=True,
        )
    layer = SimpleNamespace(
        weight=torch.zeros(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.full((n, k // g.GROUP_K), 0x7F, dtype=torch.uint8),
        weight_scale_inv_swizzled=None,
        dequantized_bf16=False,
    )
    if shuffled:
        layer.aiter_bpreshuffled = True
    return layer


@pytest.fixture
def enabled(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_mod

    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV", "1")
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV_MAX_M", "16")
    monkeypatch.setattr(g, "is_sm120", lambda: True)
    # The container has no GPU, so the import-time answer is False here.
    monkeypatch.setattr(fp8_mod, "_is_cuda", True)
    yield fp8_mod


def test_gate_accepts_mxfp8_sm120_cutlass(enabled):
    method = _method()
    x = torch.randn(4, 2560, dtype=torch.bfloat16)
    assert method._mxfp8_skinny_gemv_ok(_layer(n=512, k=2560), x, None)
    # Bias, a pre-quantized (fp8, scale) pair, and an over-wide batch all keep
    # the existing path.
    assert not method._mxfp8_skinny_gemv_ok(_layer(n=512, k=2560), x, torch.zeros(512))
    assert not method._mxfp8_skinny_gemv_ok(
        _layer(n=512, k=2560), (x, torch.ones(4, 80)), None
    )
    assert not method._mxfp8_skinny_gemv_ok(
        _layer(n=512, k=2560), torch.randn(17, 2560, dtype=torch.bfloat16), None
    )
    # A layer dequantized to BF16 (n < 128) or reshuffled by aiter is not ours.
    assert not method._mxfp8_skinny_gemv_ok(_layer(dequantized=True), x, None)
    assert not method._mxfp8_skinny_gemv_ok(
        _layer(n=512, k=2560, shuffled=True), x, None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_mxfp8", False),
        ("use_marlin", True),
        ("mxfp8_dense_backend", None),
        ("mxfp8_dense_backend", "flashinfer_trtllm"),
        ("mxfp8_dense_backend", "deep_gemm"),
        ("mxfp8_dense_backend", "unsupported"),
    ],
)
def test_gate_refuses_other_schemes_and_backends(enabled, field, value):
    from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

    method = _method()
    if field == "mxfp8_dense_backend":
        setattr(
            method,
            field,
            None if value is None else Mxfp8DenseGemmBackend(value),
        )
    else:
        setattr(method, field, value)
    assert not method._mxfp8_skinny_gemv_ok(
        _layer(n=512, k=2560), torch.randn(4, 2560, dtype=torch.bfloat16), None
    )


def test_gate_refuses_non_sm120_and_flag_off(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_mod

    method = _method()
    x = torch.randn(4, 2560, dtype=torch.bfloat16)
    layer = _layer(n=512, k=2560)
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV", "1")
    monkeypatch.setattr(fp8_mod, "_is_cuda", True)
    monkeypatch.setattr(g, "is_sm120", lambda: False)
    assert not method._mxfp8_skinny_gemv_ok(layer, x, None)
    monkeypatch.setattr(g, "is_sm120", lambda: True)
    assert method._mxfp8_skinny_gemv_ok(layer, x, None)
    monkeypatch.delitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV")
    assert not method._mxfp8_skinny_gemv_ok(layer, x, None)


def test_max_m_env_bounds_the_gate(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_mod

    method = _method()
    monkeypatch.setattr(fp8_mod, "_is_cuda", True)
    monkeypatch.setattr(g, "is_sm120", lambda: True)
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV", "1")
    layer = _layer(n=512, k=2560)
    for cap, ok in ((4, True), (3, False)):
        monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV_MAX_M", str(cap))
        assert (
            method._mxfp8_skinny_gemv_ok(
                layer, torch.randn(4, 2560, dtype=torch.bfloat16), None
            )
            is ok
        )
    # An out-of-range value is clamped, and the gate never raises.
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV_MAX_M", "4096")
    assert g.gemv_max_m() == g.MAX_M
    assert method._mxfp8_skinny_gemv_ok(
        layer, torch.randn(16, 2560, dtype=torch.bfloat16), None
    )


def test_apply_routes_to_the_gemv_and_falls_through(enabled, monkeypatch):
    method = _method()
    layer = _layer(n=128, k=128)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    sentinel = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
    fallback = object()
    calls = []
    method.w8a8_mxfp8_linear = lambda **kw: fallback

    def fake_gemv(x_, w_, s_, out=None, cfg=None):
        calls.append((tuple(x_.shape), tuple(w_.shape), tuple(s_.shape)))
        return sentinel

    monkeypatch.setattr(g, "mxfp8_skinny_gemv", fake_gemv)
    assert method.apply(layer, x) is sentinel
    assert calls == [((2, 128), (128, 128), (128, 4))]

    # A bias is not fused into the GEMV: the W8A8 GEMM runs and the GEMV is not
    # even asked.
    calls.clear()
    bias = torch.ones(128, dtype=torch.bfloat16)
    assert method.apply(layer, x, bias) is fallback
    assert calls == []

    # A scratch miss during a capture returns None and falls through unchanged.
    calls.clear()
    monkeypatch.setattr(
        g, "mxfp8_skinny_gemv", lambda *a, **kw: (calls.append(1), None)[1]
    )
    assert method.apply(layer, x) is fallback
    assert calls == [1]

    # With the flag off the GEMV is not even asked.
    def boom(*a, **kw):
        raise AssertionError("skinny GEMV must not run")

    monkeypatch.setattr(g, "mxfp8_skinny_gemv", boom)
    monkeypatch.delitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV")
    assert method.apply(layer, x) is fallback


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
