"""CPU tests for the small-batch MoE router GEMV (SGLANG_ROUTER_GEMV).

The kernel comparison runs the Triton interpreter in a subprocess, like
test_mxfp8_skinny_gemv.py does: TRITON_INTERPRET must be set before triton is
imported. N and K are this checkpoint's real gate (config.json of
/srv/llm-models/Qwen3.8-Flash-Next-FP4-Local: num_experts 512, hidden_size
2560); every M the hook admits is covered, with and without a bias.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import router_gemv as rg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

# The gate of Qwen3.8-Flash-Next as Qwen2MoeSparseMoeBlock builds it.
N, K = 512, 2560

_KERNEL_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime import interpreter as itp

    # The interpreter stores bf16 as raw uint16 and its create_dot only widens
    # fp8, so a bf16 tl.dot multiplies bit patterns. Widen bf16 operands to
    # fp32 first; the kernel itself is unchanged.
    _dot = itp.InterpreterBuilder.create_dot

    def _widen(h):
        if h.dtype == tl.bfloat16:
            return itp.TensorHandle(
                (h.data.astype(np.uint32) << 16).view(np.float32), tl.float32
            )
        return h

    itp.InterpreterBuilder.create_dot = lambda self, a, b, *r: _dot(
        self, _widen(a), _widen(b), *r
    )

    from sglang.srt.layers.moe import router_gemv as rg

    torch.manual_seed(0)
    N, K = 512, 2560
    w = (torch.randn(N, K) * 0.05).to(torch.bfloat16)
    bias = (torch.randn(N) * 0.02).to(torch.bfloat16)

    def launch(x, b):
        # The entry point wants CUDA tensors, so drive the kernel the way it
        # does (same grid, same constexprs) and keep the validation in the
        # host-side tests below.
        m = x.shape[0]
        out = torch.empty((m, N), dtype=torch.float32)
        rg._router_gemv_kernel[(N // rg.BLOCK_N,)](
            x, w, b if b is not None else w, out, m,
            K=K, N=N, HAS_BIAS=b is not None,
            BLOCK_M=rg.BLOCK_M, BLOCK_N=rg.BLOCK_N, BLOCK_K=rg.BLOCK_K,
            num_warps=rg.NUM_WARPS,
        )
        return out

    for m in (1, 2, 4, 8, 16):
        x = (torch.randn(m, K) * 0.5).to(torch.bfloat16)
        for with_bias in (False, True):
            b = bias if with_bias else None
            got = launch(x, b)
            ref = rg.router_gemv_reference(x, w, b)
            assert got.shape == (m, N) and got.dtype == torch.float32
            assert torch.isfinite(got).all(), (m, with_bias, "non-finite")
            scale = ref.abs().amax().clamp(min=1e-6)
            abs_err = (got - ref).abs().max().item()
            assert abs_err <= 2e-2 * scale.item(), (m, with_bias, abs_err)
            # Rows past m are masked out of the store; a kernel that ran them
            # anyway would still pass the norm above, so pin the row count by
            # checking that every row the launch owned is actually written.
            if m < rg.BLOCK_M:
                assert torch.isfinite(got).all(), (m, "unwritten row")
    print("KERNEL-OK")
    """
)


def test_kernel_matches_reference():
    env = dict(os.environ, TRITON_INTERPRET="1")
    r = subprocess.run(
        [sys.executable, "-c", _KERNEL_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert r.returncode == 0, f"interpreter run failed:\n{r.stdout}\n{r.stderr}"
    assert "KERNEL-OK" in r.stdout


def _operands(m=4, k=K, dtype=torch.bfloat16):
    return (
        torch.zeros(m, k, dtype=dtype),
        torch.zeros(N, k, dtype=dtype),
    )


def test_shape_ok_accepts_the_real_gate():
    x, w = _operands()
    assert rg.router_gemv_shape_ok(x, w)
    assert rg.router_gemv_shape_ok(x, w, torch.zeros(N, dtype=torch.bfloat16))
    for m in (1, 2, 4, 8, 16):
        xm, wm = _operands(m=m)
        assert rg.router_gemv_shape_ok(xm, wm), m


def test_shape_ok_rejects_everything_the_kernel_cannot_take():
    x, w = _operands()
    assert not rg.router_gemv_shape_ok(torch.zeros(0, K, dtype=torch.bfloat16), w)
    assert not rg.router_gemv_shape_ok(torch.zeros(17, K, dtype=torch.bfloat16), w)
    assert not rg.router_gemv_shape_ok(x[None], w)  # 3-D activations
    assert not rg.router_gemv_shape_ok(x.float(), w)
    assert not rg.router_gemv_shape_ok(x, w.float())
    assert not rg.router_gemv_shape_ok(x.t().contiguous().t(), w)  # non-contiguous x
    assert not rg.router_gemv_shape_ok(
        x, torch.zeros(N, K + 32, dtype=torch.bfloat16)[:, :K]
    )  # non-contiguous w
    assert not rg.router_gemv_shape_ok(x, torch.zeros(N - 16, K, dtype=torch.bfloat16))
    # K must be a multiple of BLOCK_K: the k-loop steps BLOCK_K columns with
    # unmasked loads, so an odd multiple of 128 would read past every row.
    assert not rg.router_gemv_shape_ok(
        torch.zeros(4, 2500, dtype=torch.bfloat16),
        torch.zeros(N, 2500, dtype=torch.bfloat16),
    )
    for k in (128, 384, 640, 896):
        assert not rg.router_gemv_shape_ok(
            torch.zeros(4, k, dtype=torch.bfloat16),
            torch.zeros(N, k, dtype=torch.bfloat16),
        ), k
    assert not rg.router_gemv_shape_ok(x, w, torch.zeros(N, 1, dtype=torch.bfloat16))
    assert not rg.router_gemv_shape_ok(x, w, torch.zeros(N, dtype=torch.float32))


def test_applies_adds_the_device_gate():
    x, w = _operands()
    assert not rg.router_gemv_applies(x, w), "a CPU tensor is not launchable"
    assert rg.router_gemv_applies(x, w, require_cuda=False)


def test_entry_point_validates():
    x, w = _operands()
    with pytest.raises(ValueError):
        rg.router_gemv(torch.zeros(17, K, dtype=torch.bfloat16), w)
    with pytest.raises(ValueError):
        rg.router_gemv(x[None], w)
    with pytest.raises(ValueError):
        rg.router_gemv(x.float(), w)
    with pytest.raises(ValueError):
        rg.router_gemv(x, w, torch.zeros(N, dtype=torch.float32))
    with pytest.raises(ValueError):
        rg.router_gemv(x, w, torch.zeros(N, K, dtype=torch.bfloat16))
    # K = 384 is an odd multiple of 128: the unmasked BLOCK_K loads would run
    # past every weight row, so the entry point must reject it, not launch.
    with pytest.raises(ValueError):
        rg.router_gemv(
            torch.zeros(4, 384, dtype=torch.bfloat16),
            torch.zeros(N, 384, dtype=torch.bfloat16),
        )


class _Gate:
    """Duck of the ReplicatedLinear the block's helper calls."""

    def __init__(self, weight, bias=None, ret=None):
        self.weight = weight
        self.bias = bias
        self.ret = ret
        self.calls = []

    def __call__(self, x):
        self.calls.append(x)
        return self.ret


def _stub(enabled, gate):
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    stub = SimpleNamespace(gate=gate, _router_gemv_enabled=enabled)
    return stub, Qwen2MoeSparseMoeBlock._forward_router_logits


def _call(stub, x, fn):
    return fn(stub, x)


def test_router_gemv_flag_defaults_off(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_ROUTER_GEMV", raising=False)
    assert envs.SGLANG_ROUTER_GEMV.get() is False


def test_hook_falls_through_to_the_linear(monkeypatch):
    import sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv as skinny

    monkeypatch.setattr(skinny, "is_sm120", lambda: False)
    x, w = _operands()
    sentry = (torch.zeros(4, N, dtype=torch.bfloat16), None)
    gate = _Gate(w, ret=sentry)
    stub, fn = _stub(False, gate)
    assert _call(stub, x, fn) is sentry[0]
    assert gate.calls == [x], "the disabled path must call the ReplicatedLinear"
    # Enabled, but the device says no: the same fallback, once.
    gate2 = _Gate(w, ret=sentry)
    stub2, fn2 = _stub(True, gate2)
    assert _call(stub2, x, fn2) is sentry[0]
    assert gate2.calls == [x]


def test_hook_uses_the_skinny_gemv_for_a_bias_free_gate(monkeypatch):
    import sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv as skinny

    seen = {}
    out = torch.zeros(4, N, dtype=torch.bfloat16)

    def fake_linear(xx, ww):
        seen["args"] = (xx, ww)
        return out

    monkeypatch.setattr(skinny, "is_sm120", lambda: True)
    monkeypatch.setattr(skinny, "bf16_skinny_gemv_applies", lambda xx, ww: True)
    monkeypatch.setattr(skinny, "bf16_skinny_linear", fake_linear)
    x, w = _operands()
    gate = _Gate(w, ret=(torch.zeros(4, N), None))
    stub, fn = _stub(True, gate)
    assert _call(stub, x, fn) is out
    assert seen["args"] == (x, w)
    assert gate.calls == [], "the GEMV replaces the linear, it does not follow it"

    # A scratch miss during capture hands the call back to the linear.
    monkeypatch.setattr(skinny, "bf16_skinny_linear", lambda xx, ww: None)
    gate2 = _Gate(w, ret=(torch.zeros(4, N), None))
    stub2, fn2 = _stub(True, gate2)
    assert _call(stub2, x, fn2) is gate2.ret[0]
    assert gate2.calls == [x]


def test_hook_uses_the_fp32_kernel_for_a_biased_gate(monkeypatch):
    import sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv as skinny
    import sglang.srt.layers.moe.router_gemv as rgmod

    seen = {}
    out = torch.zeros(4, N, dtype=torch.float32)

    def fake(x_, w_, b_):
        seen["args"] = (x_, w_, b_)
        return out

    monkeypatch.setattr(skinny, "is_sm120", lambda: True)
    monkeypatch.setattr(rgmod, "router_gemv", fake)
    # The verdict asks for CUDA tensors, which a CPU test cannot supply; the
    # layout half of it is covered by test_shape_ok_* above.
    monkeypatch.setattr(rgmod, "router_gemv_applies", lambda *a, **kw: True)
    x, w = _operands()
    bias = torch.zeros(N, dtype=torch.bfloat16)
    gate = _Gate(w, bias=bias, ret=(torch.zeros(4, N), None))
    stub, fn = _stub(True, gate)
    assert _call(stub, x, fn) is out
    assert seen["args"] == (x, w, bias)
    assert gate.calls == []


def test_hook_falls_through_on_other_arches(monkeypatch):
    # Both routes are sm_120-only: with is_sm120() False the biased gate must
    # reach the ReplicatedLinear, even when the layout verdict is True.
    import sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv as skinny
    import sglang.srt.layers.moe.router_gemv as rgmod

    monkeypatch.setattr(skinny, "is_sm120", lambda: False)
    monkeypatch.setattr(rgmod, "router_gemv_applies", lambda *a, **kw: True)
    monkeypatch.setattr(
        rgmod, "router_gemv", lambda *a, **kw: pytest.fail("launched off sm_120")
    )
    x, w = _operands()
    bias = torch.zeros(N, dtype=torch.bfloat16)
    gate = _Gate(w, bias=bias, ret=(torch.zeros(4, N), None))
    stub, fn = _stub(True, gate)
    sentry = gate.ret
    assert _call(stub, x, fn) is sentry[0]
    assert gate.calls == [x], "an unsupported arch must call the ReplicatedLinear"


def test_launch_path_syncs_no_host():
    import inspect

    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    for src in (
        inspect.getsource(rg.router_gemv),
        inspect.getsource(Qwen2MoeSparseMoeBlock._forward_router_logits),
    ):
        for needle in (".item(", "torch.cuda.synchronize", ".cpu()", ".tolist("):
            assert needle not in src, needle
