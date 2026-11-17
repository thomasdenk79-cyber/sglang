import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

from sglang.kernels.ops.gemm import sm120_bf16_skinny_gemv as g
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_SHAPES = sorted(g.SKINNY_BF16_SHAPES)

# The Triton interpreter keeps bf16 as raw uint16 and its create_dot only widens
# fp8, so a bf16 tl.dot multiplies bit patterns. Widen bf16 operands to fp32
# first; the kernel itself is unchanged.
_INTERPRETER_CHECK = textwrap.dedent(
    """
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime import interpreter as itp

    _dot = itp.InterpreterBuilder.create_dot

    def _widen(h):
        if h.dtype == tl.bfloat16:
            data = (h.data.astype(np.uint32) << 16).view(np.float32)
            return itp.TensorHandle(data, tl.float32)
        return h

    itp.InterpreterBuilder.create_dot = lambda self, a, b, *r: _dot(
        self, _widen(a), _widen(b), *r
    )

    from sglang.kernels.ops.gemm import sm120_bf16_skinny_gemv as g

    torch.manual_seed(0)
    ws = torch.empty(g._WS_FLOATS)
    cnt = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
    one = torch.ones(1)
    cases = [
        # (N, K, M, forced cfg): split-K broadcast (M=1), split-K mma, single split.
        (512, 2560, 1, None),
        (512, 2560, 3, None),
        (512, 2560, 16, None),
        (96, 2560, 16, None),
        (2560, 2560, 1, None),
        (2560, 2560, 4, None),
        # Masked K tail, both orientations of the reduction.
        (64, 2500, 3, (16, 128, 5, True, None, 4, 3)),
        (64, 2500, 1, (16, 128, 5, False, None, 4, 3)),
    ]
    for n, k, m, cfg in cases:
        cfg = cfg or g._plan(m, n, k, False, 2, 188)
        x = torch.randn(m, k).bfloat16()
        w = (torch.randn(n, k) * 0.05).bfloat16()
        ref = (x.float() @ w.float().T).bfloat16().float()
        for _ in range(2):  # the second launch needs the counters reset
            y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
            g._launch(ws, cnt, x, w, one, y, m, n, k, False, cfg)
            err = (y.float() - ref).abs().max().item()
            assert err <= 1e-2 * ref.abs().max().item(), (n, k, m, cfg, err)
            assert int(cnt.abs().sum()) == 0, (n, k, m, cfg)
    print("OK")
    """
)


@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("m", range(1, g.MAX_M + 1))
def test_plan_fits_preallocated_scratch(n, k, m):
    bn, bk, splits, use_dot, _, _, _ = g._plan(m, n, k, False, 2, 188)
    assert k % bk == 0
    if splits > 1:
        m_pad = 16 if (use_dot or m > 1) else 1
        n_blocks = -(-n // bn)
        assert splits <= g._MAX_SPLITS and n_blocks <= g._WS_COUNTERS
        assert n_blocks * splits * m_pad * bn <= g._WS_FLOATS


def test_kernel_matches_reference_in_triton_interpreter():
    root = os.path.abspath(os.path.join(os.path.dirname(g.__file__), *[".."] * 4))
    env = dict(os.environ, TRITON_INTERPRET="1")
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _INTERPRETER_CHECK],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-4000:]


def test_rejects_unsupported_operands():
    x = torch.zeros(4, 2560, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        g.bf16_skinny_gemv(
            torch.zeros(17, 2560, dtype=torch.bfloat16), torch.zeros(512, 2560)
        )
    with pytest.raises(ValueError):
        g.bf16_skinny_gemv(x, torch.zeros(512, 2048, dtype=torch.bfloat16))


def _enable_hook(monkeypatch, linear):
    from sglang.srt.layers.quantization import unquant

    monkeypatch.setattr(unquant, "_use_sm120_bf16_skinny_gemv", lambda: True)
    monkeypatch.setattr(
        unquant, "bf16_skinny_gemv_applies", lambda x, w: True, raising=False
    )
    monkeypatch.setattr(unquant, "bf16_skinny_linear", linear, raising=False)
    return unquant


def test_enabled_hook_routes_bias_free_linears(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.models.qwen4_exp_mtp import _fc

    calls = []

    def fake(x, w):
        calls.append(tuple(w.shape))
        return torch.full((*x.shape[:-1], w.shape[0]), 7.0, dtype=x.dtype)

    unquant = _enable_hook(monkeypatch, fake)
    method = unquant.UnquantizedLinearMethod()
    layer = SimpleNamespace(weight=torch.zeros(512, 2560, dtype=torch.bfloat16))
    x = torch.randn(4, 2560, dtype=torch.bfloat16)
    assert (method.apply(layer, x) == 7.0).all()
    lin = nn.Linear(2560, 2560, bias=False).bfloat16()
    assert (_fc(lin, x.view(1, 4, 2560)) == 7.0).all()
    assert calls == [(512, 2560), (2560, 2560)]

    bias = torch.ones(512, dtype=torch.bfloat16)
    torch.testing.assert_close(method.apply(layer, x, bias), bias.expand(4, 512))
    assert len(calls) == 2


@pytest.mark.parametrize("gate", ["compile", "batch_invariant"])
def test_enabled_hook_respects_compile_and_batch_invariance(monkeypatch, gate):
    def fail(x, w):
        raise AssertionError("skinny GEMV must not run")

    unquant = _enable_hook(monkeypatch, fail)
    if gate == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    else:
        monkeypatch.setattr(unquant, "is_batch_invariant_mode_enabled", lambda: True)
    w = torch.zeros(512, 2560, dtype=torch.bfloat16)
    assert unquant.maybe_sm120_bf16_skinny_linear(torch.zeros(4, 2560), w) is None


def test_applies_refuses_autograd_but_not_no_grad_nn_linear(monkeypatch):
    lin = nn.Linear(2560, 512, bias=False).bfloat16()
    x = torch.zeros(4, 2560, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda t: True))
    assert not g.bf16_skinny_gemv_applies(x, lin.weight)
    with torch.no_grad():
        assert g.bf16_skinny_gemv_applies(x, lin.weight)
        assert not g.bf16_skinny_gemv_applies(
            torch.zeros(17, 2560).bfloat16(), lin.weight
        )
        assert not g.bf16_skinny_gemv_applies(x.half(), lin.weight.half())
        assert not g.bf16_skinny_gemv_applies(
            torch.zeros(4, 2048).bfloat16(), torch.zeros(512, 2048).bfloat16()
        )


def test_disabled_hook_falls_back_to_linear():
    from sglang.srt.layers.quantization.unquant import maybe_sm120_bf16_skinny_linear
    from sglang.srt.models.qwen4_exp_mtp import _fc

    lin = nn.Linear(2560, 2560, bias=False).bfloat16()
    x = torch.randn(2, 4, 2560, dtype=torch.bfloat16)
    assert maybe_sm120_bf16_skinny_linear(x, lin.weight) is None
    torch.testing.assert_close(_fc(lin, x), lin(x))


_needs_sm120 = pytest.mark.skipif(not g.is_sm120(), reason="needs sm_120")


@_needs_sm120
@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("m", [1, 2, 4, 7, 16])
def test_matches_cublas_on_sm120(n, k, m):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    assert g.bf16_skinny_gemv_applies(x, w)
    for _ in range(2):
        y = g.bf16_skinny_linear(x, w)
        torch.testing.assert_close(y, x @ w.T, atol=2e-2, rtol=2e-2)


@_needs_sm120
def test_scratch_miss_during_capture_returns_none():
    x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            assert g.bf16_skinny_linear(x, w) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
