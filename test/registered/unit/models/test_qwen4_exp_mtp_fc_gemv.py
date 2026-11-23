"""CPU tests for the MTP entry-fc GEMV flag (SGLANG_MTP_FC_GEMV).

The kernel is the in-tree sm_120 BF16 skinny GEMV
(kernels/ops/gemm/sm120_bf16_skinny_gemv.py), already tested numerically in
test_sm120_bf16_skinny_gemv.py. What is new here is the switch: the MTP entry
projections must follow SGLANG_MTP_FC_GEMV on its own, so that the architect
can A/B the draft entry without also turning on the router gate and the GDN
in_proj_ba route, and must keep cuBLAS when it is off.
"""

import sys

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.qwen4_exp_mtp import _fc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# The two entry projections of this checkpoint, both [hidden, hidden] BF16 with
# no bias (config.json: hidden_size 2560, mtp_num_hidden_layers 1).
FC_SHAPE = (2560, 2560)
HIDDEN = 2560


def _linear(**over):
    lin = nn.Linear(HIDDEN, FC_SHAPE[0], bias=False)
    lin.weight = torch.nn.Parameter(torch.randn(*FC_SHAPE) * 0.05, requires_grad=False)
    for key, value in over.items():
        setattr(lin, key, value)
    return lin.to(torch.bfloat16)


@pytest.fixture(autouse=True)
def _clear_predicate_caches():
    unquant._use_mtp_fc_gemv.cache_clear()
    unquant._use_sm120_bf16_skinny_gemv.cache_clear()
    yield
    unquant._use_mtp_fc_gemv.cache_clear()
    unquant._use_sm120_bf16_skinny_gemv.cache_clear()


def _route(monkeypatch, value=7.0):
    """Replace the kernel entry point; returns the list of routed weight shapes."""
    calls = []

    def fake(x, w):
        calls.append(tuple(w.shape))
        return torch.full((*x.shape[:-1], w.shape[0]), value, dtype=x.dtype)

    monkeypatch.setattr(unquant, "_use_mtp_fc_gemv", lambda: True)
    monkeypatch.setattr(unquant, "_use_sm120_bf16_skinny_gemv", lambda: True)
    monkeypatch.setattr(
        unquant, "bf16_skinny_gemv_applies", lambda x, w: True, raising=False
    )
    monkeypatch.setattr(unquant, "bf16_skinny_linear", fake, raising=False)
    return calls


def test_flag_is_registered_and_off_by_default():
    from sglang.srt.environ import Envs, EnvBool

    field = getattr(Envs, "SGLANG_MTP_FC_GEMV")
    assert isinstance(field, EnvBool) and field.get() is False
    assert envs.SGLANG_MTP_FC_GEMV.get() is False
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    assert unquant.maybe_mtp_fc_skinny_linear(x, torch.zeros(*FC_SHAPE)) is None


def test_own_flag_routes_the_entry_projection(monkeypatch):
    calls = _route(monkeypatch)
    monkeypatch.setattr(unquant, "_use_sm120_bf16_skinny_gemv", lambda: False)
    lin = _linear()
    x = torch.randn(1, 4, HIDDEN, dtype=torch.bfloat16)
    assert (_fc(lin, x) == 7.0).all()
    assert calls == [FC_SHAPE]


def test_general_skinny_flag_also_routes_it(monkeypatch):
    calls = _route(monkeypatch)
    monkeypatch.setattr(unquant, "_use_mtp_fc_gemv", lambda: False)
    lin = _linear()
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    assert (_fc(lin, x) == 7.0).all()
    assert calls == [FC_SHAPE]


def test_both_flags_off_keeps_cublas(monkeypatch):
    def fail(x, w):
        raise AssertionError("the skinny GEMV must not run")

    monkeypatch.setattr(unquant, "_use_mtp_fc_gemv", lambda: False)
    monkeypatch.setattr(unquant, "_use_sm120_bf16_skinny_gemv", lambda: False)
    monkeypatch.setattr(unquant, "bf16_skinny_linear", fail, raising=False)
    lin = _linear()
    x = torch.randn(2, 4, HIDDEN, dtype=torch.bfloat16)
    torch.testing.assert_close(_fc(lin, x), lin(x), atol=0, rtol=0)


def test_biased_projection_never_takes_the_bias_free_kernel(monkeypatch):
    calls = _route(monkeypatch)
    lin = _linear(
        bias=nn.Parameter(
            torch.zeros(FC_SHAPE[0], dtype=torch.bfloat16), requires_grad=False
        )
    )
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    torch.testing.assert_close(_fc(lin, x), lin(x), atol=0, rtol=0)
    assert calls == []


@pytest.mark.parametrize("flag", ["mtp_fc", "skinny", "both"])
def test_unsupported_operand_falls_through(monkeypatch, flag):
    routed = []

    def fake(x, w):
        routed.append(1)
        return torch.zeros(x.shape[0], w.shape[0], dtype=x.dtype)

    monkeypatch.setattr(unquant, "_use_mtp_fc_gemv", lambda: flag in ("mtp_fc", "both"))
    monkeypatch.setattr(
        unquant, "_use_sm120_bf16_skinny_gemv", lambda: flag in ("skinny", "both")
    )
    monkeypatch.setattr(unquant, "bf16_skinny_linear", fake, raising=False)
    for applies in (False, True):
        monkeypatch.setattr(
            unquant, "bf16_skinny_gemv_applies", lambda x, w: applies, raising=False
        )
        routed.clear()
        lin = _linear()
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
        out = _fc(lin, x)
        assert len(routed) == (1 if applies else 0), (flag, applies)
        # Routed: the kernel's zeros. Not routed: what nn.Linear computes.
        if applies:
            assert (out == 0).all()
        else:
            torch.testing.assert_close(out, lin(x), atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_kernel_and_torch_agree_on_the_real_fc_shape(m):
    """The GEMV this flag turns on equals F.linear, on the real [2560, 2560]."""
    if not unquant._is_cuda or not unquant.is_sm120():
        pytest.skip("the skinny GEMV kernel needs sm_120")
    from sglang.kernels.ops.gemm.sm120_bf16_skinny_gemv import (
        bf16_skinny_gemv_applies,
        bf16_skinny_linear,
    )

    torch.manual_seed(0)
    lin = _linear().to("cuda")
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert bf16_skinny_gemv_applies(x, lin.weight)
    y = bf16_skinny_linear(x, lin.weight)
    torch.testing.assert_close(
        y, torch.nn.functional.linear(x, lin.weight), atol=2e-2, rtol=2e-2
    )


def test_flags_are_independent_switches():
    """The four ported kernels each have their own flag, all defaulting off."""
    from sglang.srt.environ import Envs

    for name in (
        "SGLANG_OPT_DRAFT_MOE_GEMV",
        "SGLANG_ROUTER_GEMV",
        "SGLANG_NORM_INTO_GEMV",
        "SGLANG_MTP_FC_GEMV",
    ):
        field = getattr(Envs, name)
        assert field.get() is False, name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
