"""CPU tests for the fused gated RMSNorm + MXFP8 GEMV (SGLANG_NORM_INTO_GEMV).

The kernel comparison runs the Triton interpreter in a subprocess, like
test_mxfp8_skinny_gemv.py does, because TRITON_INTERPRET must be set before
triton is imported. K and the norm group are this checkpoint's real GDN
out_proj values (num_v_heads 48 x head_v_dim 128 = 6144 in, hidden 2560);
only N is sub-sampled, the tile is not.
"""

import inspect
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization import mxfp8_norm_gated_gemv as g
from sglang.test.ci.ci_register import register_cpu_ci

# ~12 interpreter launches at the real K (6144): 3 s each measured, plus the
# ~30 s of imports the subprocess pays once.
register_cpu_ci(est_time=180, suite="base-a-test-cpu")

# (M bucket, N, K, G) the donor measured, from _FUSED_NORM_BY_SHAPE.
N_REAL, K_REAL, G = 2560, 6144, 128
M_VALUES = [1, 2, 4, 8, 16]


def _operands(m=4, n=N_REAL, k=K_REAL, group=G, seed=0):
    """A deterministic CPU (x, z, norm_w, w, scale) quintuple."""
    torch.manual_seed(seed)
    x = (torch.randn(m, k) * 0.4).bfloat16().contiguous()
    z = (torch.randn(m, k) * 1.5).bfloat16().contiguous()
    norm_w = (torch.randn(group) * 0.1 + 1.0).bfloat16().contiguous()
    w = (torch.randn(n, k) * 0.05).clamp(-448.0, 448.0)
    w = w.to(torch.float8_e4m3fn).contiguous()
    # 2^-7 .. 2^8: the UE8M0 range a quantizer lands in for these weights.
    scale = torch.randint(120, 137, (n, k // g.GROUP_K), dtype=torch.uint8).contiguous()
    return x, z, norm_w, w, scale


def _launch(m, n, x, z, norm_w, w, scale, cfg, activation="swish", eps=1e-6):
    """One direct interpreter launch; returns bf16 y, or NaN where unwritten."""
    block_n, splits, use_dot, warps, stages = cfg
    block_m = 16 if (use_dot or m > 1) else 1
    y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
    ws = torch.zeros(g._WS_FLOATS, dtype=torch.float32)
    cnt = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
    g._mxfp8_norm_gated_gemv_kernel[(-(-n // block_n), splits)](
        x,
        z,
        norm_w,
        w,
        scale,
        y,
        ws,
        cnt,
        m,
        float(eps),
        N=n,
        K=x.shape[1],
        G=norm_w.numel(),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLITS=splits,
        USE_DOT=use_dot,
        NORM_SIGMOID=activation == "sigmoid",
        num_warps=warps,
        num_stages=stages,
    )
    return y, ws, cnt


# The interpreter stores bf16 as raw uint16, its binary_op multiplies those bit
# patterns, and its create_dot only widens fp8. Both would make a bf16 weight
# tile nonsense, so widen bf16 to fp32 around every binary op and around every
# dot, and narrow the result back with round-to-nearest-even. Test-side only:
# the kernel under test is imported unchanged.
_SHIMS = textwrap.dedent(
    """
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as itp

    def _widen(h):
        if h.dtype == tl.bfloat16:
            return itp.TensorHandle(
                (h.data.astype(np.uint32) << 16).view(np.float32), tl.float32
            )
        return h

    def _narrow(h, scalar):
        if scalar == tl.bfloat16 and h.dtype == tl.float32:
            u = h.data.view(np.uint32)
            r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000) >> np.uint32(16)
            return itp.TensorHandle(r.astype(np.uint16), tl.bfloat16)
        return h

    _binary_op = itp.InterpreterBuilder.binary_op

    def binary_op(self, lhs, rhs, op):
        scalar = lhs.dtype.scalar
        if lhs.dtype != tl.bfloat16 and rhs.dtype != tl.bfloat16:
            return _binary_op(self, lhs, rhs, op)
        out = op(_widen(lhs).data, _widen(rhs).data)
        return _narrow(itp.TensorHandle(out, tl.float32), scalar)

    itp.InterpreterBuilder.binary_op = binary_op

    _dot = itp.InterpreterBuilder.create_dot
    itp.InterpreterBuilder.create_dot = lambda self, a, b, *r: _dot(
        self, _widen(a), _widen(b), *r
    )
    """
)

_KERNEL_SCRIPT = textwrap.dedent(
    """
    import time

    import torch

"""
    + textwrap.indent(_SHIMS, "    ")
    + """
    from sglang.srt.layers.quantization import mxfp8_norm_gated_gemv as g

    N_REAL, K, G = 2560, 6144, 128

    def operands(m, n, seed):
        torch.manual_seed(seed)
        x = (torch.randn(m, K) * 0.4).bfloat16().contiguous()
        z = (torch.randn(m, K) * 1.5).bfloat16().contiguous()
        norm_w = (torch.randn(G) * 0.1 + 1.0).bfloat16().contiguous()
        w = (torch.randn(n, K) * 0.05).clamp(-448.0, 448.0)
        w = w.to(torch.float8_e4m3fn).contiguous()
        scale = torch.randint(
            120, 137, (n, K // g.GROUP_K), dtype=torch.uint8
        ).contiguous()
        return x, z, norm_w, w, scale

    def launch(m, n, x, z, norm_w, w, scale, cfg, activation, norm_fused=True):
        block_n, splits, use_dot, warps, stages = cfg
        block_m = 16 if (use_dot or m > 1) else 1
        y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
        ws = torch.zeros(g._WS_FLOATS, dtype=torch.float32)
        cnt = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
        g._mxfp8_norm_gated_gemv_kernel[(-(-n // block_n), splits)](
            x, z, norm_w, w, scale, y, ws, cnt, m, 1e-6,
            N=n, K=K, G=G, BLOCK_M=block_m, BLOCK_N=block_n, SPLITS=splits,
            USE_DOT=use_dot, NORM_SIGMOID=activation == "sigmoid",
            NORM_FUSED=norm_fused,
            num_warps=warps, num_stages=stages,
        )
        return y, ws, cnt

    t0 = time.time()
    # (m, n, cfg, activation): the production plan per M bucket, both gate
    # activations, both reduction orientations, a no-split plan and a masked
    # N tail.
    cases = [
        (1, 128, g.plan_for(1, N_REAL, K, G), "swish"),
        (2, 128, g.plan_for(2, N_REAL, K, G), "swish"),
        (4, 128, g.plan_for(4, N_REAL, K, G), "swish"),
        (8, 128, g.plan_for(8, N_REAL, K, G), "swish"),
        (16, 128, g.plan_for(16, N_REAL, K, G), "swish"),
        (1, 128, (64, 4, True, 8, 4), "swish"),
        (4, 128, (64, 4, True, 8, 4), "sigmoid"),
        (4, 128, (64, 1, True, 8, 4), "swish"),
        (1, 128, (64, 1, False, 8, 4), "swish"),
        (4, 100, (64, 4, True, 8, 4), "swish"),
        (1, 100, (64, 2, False, 8, 4), "sigmoid"),
    ]
    for m, n, cfg, activation in cases:
        assert cfg is not None, (m, n)
        x, z, norm_w, w, scale = operands(m, n, m + n)
        y, ws, cnt = launch(m, n, x, z, norm_w, w, scale, cfg, activation)
        ref = g.fused_norm_gated_gemv_reference(
            x, z, norm_w, w, scale, G, 1e-6, activation
        )
        assert not torch.isnan(y).any(), (m, n, cfg, activation)
        err = (y.float() - ref.float()).abs().max().item()
        peak = ref.float().abs().max().item()
        rel = (y.float() - ref.float()).norm().item() / (
            ref.float().norm().item() + 1e-9
        )
        print(f"m={m} n={n} cfg={cfg} act={activation} relpeak={err / peak:.2e} "
              f"rel_l2={rel:.2e}")
        # Two bf16 ulp of the peak: the fused path re-associates the fp32 sum of
        # squares, so an activation element may round to the neighbouring bf16.
        assert err <= 2e-2 * peak, (m, n, cfg, activation, err, peak)
        assert rel <= 2e-2, (m, n, cfg, activation, rel)
        assert int(cnt.abs().sum()) == 0, (m, n, cfg, activation)

        # Determinism and counter reset: the same launch twice, no memset.
        y2, _, cnt2 = launch(m, n, x, z, norm_w, w, scale, cfg, activation)
        assert torch.equal(y, y2), (m, n, cfg, activation)
        assert int(cnt2.abs().sum()) == 0

        # The norm half alone matches the unfused RMSNormGated reference.
        a = g.gated_rms_norm_reference(x, z, norm_w, G, 1e-6, activation)
        assert not torch.isnan(a).any()
        a_ref = g.gated_rms_norm_reference(x, z, norm_w, G, 1e-6, activation)
        assert torch.equal(a, a_ref)
    # The benchmark's baseline twin: the same kernel without the prologue is a
    # plain MXFP8 GEMV over the activation the unfused norm would have written.
    for m, n, cfg, activation in [(4, 128, (64, 4, True, 8, 4), "swish"),
                                  (1, 128, (64, 2, False, 8, 4), "swish")]:
        x, z, norm_w, w, scale = operands(m, n, 100 + m)
        a = g.gated_rms_norm_reference(x, z, norm_w, G, 1e-6, activation)
        y, _, cnt = launch(m, n, a, x, norm_w, w, scale, cfg, activation, norm_fused=False)
        ref = (a.to(torch.float32) @ g.dequant_mxfp8(w, scale).t()).to(torch.bfloat16)
        err = (y.float() - ref.float()).abs().max().item()
        print(f"unfused-twin m={m} relpeak={err / ref.float().abs().max().item():.2e}")
        assert err <= 2e-2 * ref.float().abs().max().item(), (m, err)
        assert int(cnt.abs().sum()) == 0
    print(f"({time.time() - t0:.0f}s)")
    print("KERNEL-OK")
    """
)


def test_kernels_match_reference_in_triton_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1")
    python_root = os.path.abspath(
        os.path.join(os.path.dirname(g.__file__), *[".."] * 4)
    )
    env["PYTHONPATH"] = os.pathsep.join([python_root, env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _KERNEL_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert out.returncode == 0, f"interpreter run failed:\n{out.stdout}\n{out.stderr}"
    assert "KERNEL-OK" in out.stdout, out.stdout + out.stderr


# --------------------------------------------------------------------------
# Plan table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", M_VALUES)
def test_plan_covers_the_measured_out_proj(m):
    cfg = g.plan_for(m, N_REAL, K_REAL, G)
    assert cfg is not None, m
    block_n, splits, use_dot, warps, stages = cfg
    assert 1 <= splits <= g._MAX_SPLITS and warps >= 1 and stages >= 1
    n_blocks = -(-N_REAL // block_n)
    assert n_blocks <= g._WS_COUNTERS
    block_m = 16 if (use_dot or m > 1) else 1
    assert n_blocks * splits * block_m * block_n <= g._WS_FLOATS


def test_plan_rejects_unmeasured_shapes():
    assert g.plan_for(4, N_REAL, K_REAL, 64) is None
    assert g.plan_for(4, N_REAL, 2560, G) is None
    # M is bucketed: 17..32 ride the M=16 tile, and the gate (not the planner)
    # is what stops them at MAX_M.
    assert g.plan_for(17, N_REAL, K_REAL, G) == g.plan_for(16, N_REAL, K_REAL, G)


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def _good(m=4, n=N_REAL, k=K_REAL, group=G):
    return _operands(m=m, n=n, k=k, group=group)


def _applies(x, z, norm_w, w, scale, group=G, activation="swish"):
    return g.fused_norm_gemv_applies(
        x, z, norm_w, w, scale, group, activation, require_cuda=False
    )


def test_gate_accepts_the_real_call():
    x, z, norm_w, w, scale = _good()
    assert _applies(x, z, norm_w, w, scale)
    assert _applies(*_good(m=1), activation="sigmoid")
    assert _applies(*_good(m=16))


@pytest.mark.parametrize(
    "mutate,why",
    [
        (lambda t: (t[0][:0].contiguous(),) + t[1:], "M == 0"),
        (lambda t: (t[0].float(),) + t[1:], "fp32 activation"),
        (lambda t: (t[0].t().contiguous().t(),) + t[1:], "non-contiguous x"),
        (lambda t: (t[0][:, :64].contiguous(), t[1], t[2], t[3], t[4]), "K mismatch"),
        (
            lambda t: (
                (
                    t[0],
                    t[1][:1].contiguous(),
                )
                + t[2:]
            ),
            "z rows mismatch",
        ),
        (
            lambda t: (
                (
                    t[0],
                    t[1].float(),
                )
                + t[2:]
            ),
            "fp32 gate",
        ),
        (
            lambda t: (
                (
                    t[0],
                    t[1],
                    t[2][:64].contiguous(),
                )
                + t[3:]
            ),
            "short norm weight",
        ),
        (lambda t: (t[0], t[1], t[2], t[3].to(torch.bfloat16), t[4]), "not e4m3"),
        (lambda t: (t[0], t[1], t[2], t[3].t().contiguous(), t[4]), "K-major weight"),
        (lambda t: (t[0], t[1], t[2], t[3], t[4].to(torch.float32)), "fp32 scales"),
        (lambda t: (t[0], t[1], t[2], t[3], t[4][:, :7]), "short scale row"),
        (lambda t: (t[0], t[1], t[2], t[3][:128].contiguous(), t[4]), "N mismatch"),
    ],
)
def test_gate_rejects(mutate, why):
    x, z, norm_w, w, scale = _good()
    x, z, norm_w, w, scale = mutate((x, z, norm_w, w, scale))
    assert not _applies(x, z, norm_w, w, scale), why


def test_gate_rejects_too_many_rows():
    x, z, norm_w, w, scale = _operands(m=g.MAX_M + 1)
    assert not _applies(x, z, norm_w, w, scale)


def test_gate_rejects_sliced_weight_view():
    # [N, K] view of a wider row-major tensor: stride(1) == 1 but
    # stride(0) != K, and the kernel indexes w_ptr + n * K, so accepting it
    # would silently read the wrong rows (the torch reference would raise).
    x, z, norm_w, _, scale = _good()
    wide = (torch.randn(N_REAL, K_REAL + 32) * 0.05).clamp(-448.0, 448.0)
    wide = wide.to(torch.float8_e4m3fn)
    w = wide[:, :K_REAL]
    assert w.stride(1) == 1 and w.stride(0) != K_REAL  # the hazard shape
    assert not _applies(x, z, norm_w, w, scale), (
        "a sliced fp8 weight view with row stride != K must fall through "
        "to the unfused path"
    )
    # Positive control: the same data fully contiguous is still accepted.
    assert _applies(x, z, norm_w, wide[:, :K_REAL].contiguous(), scale)


def test_gate_rejects_unsupported_activation_and_group():
    x, z, norm_w, w, scale = _good()
    assert not _applies(x, z, norm_w, w, scale, activation="gelu")
    assert not _applies(x, z, norm_w, w, scale, group=0)
    assert not _applies(x, z, norm_w, w, scale, group=96)  # K % 96 != 0
    # An unmeasured shape is not served even when everything else fits.
    x2, z2, nw2, w2, s2 = _good(n=256, k=2560)
    assert not g.fused_norm_gemv_applies(
        x2, z2, nw2, w2, s2, G, "swish", require_cuda=False
    )


def test_gate_requires_cuda_and_one_device():
    x, z, norm_w, w, scale = _good()
    # On CPU the layout half passes but the device half does not.
    assert not g.fused_norm_gemv_applies(x, z, norm_w, w, scale, G, "swish")


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
def test_reference_matches_the_unfused_pair(activation):
    """gated_rms_norm_reference == the production RMSNormGated maths.

    The GDN hands ``RMSNormGated`` a [tokens * heads, head_v_dim] core and a
    [tokens, heads, head_v_dim] gate, which is the same row-wise normalisation
    the fused kernel re-derives per tile.
    """
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_ref

    m, heads = 4, 8
    x = (torch.randn(m * heads, G) * 0.4).bfloat16().contiguous()
    z = (torch.randn(m * heads, G) * 1.5).bfloat16().contiguous()
    norm_w = (torch.randn(G) * 0.1 + 1.0).bfloat16().contiguous()
    mine = g.gated_rms_norm_reference(
        x.view(m, heads * G), z.view(m, heads * G), norm_w, G, 1e-6, activation
    )
    # One row of the [tokens * heads, head_v_dim] view is one norm group, so
    # group_size=None and a 2-D gate are the production configuration.
    theirs = rms_norm_ref(
        x,
        norm_w,
        None,
        z=z,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        upcast=True,
        activation=activation,
    ).view(m, heads * G)
    assert mine.shape == theirs.shape
    err = (mine.float() - theirs.float()).abs().max().item()
    # Both are fp32 maths rounded to bf16 once; only rstd's last bit can differ.
    assert err <= 4 * 2**-9 * theirs.float().abs().max().item()


def test_dequant_mxfp8_matches_the_scale_contract():
    w = torch.full((4, 64), 1.0).to(torch.float8_e4m3fn)
    scale = torch.full((4, 2), 126, dtype=torch.uint8)  # 2 ** -1
    assert torch.equal(g.dequant_mxfp8(w, scale), torch.full((4, 64), 0.5))
    scale[:, 1] = 130  # 2 ** 3 for the second 32-wide group
    out = g.dequant_mxfp8(w, scale)
    assert torch.equal(out[:, :32], torch.full((4, 32), 0.5))
    assert torch.equal(out[:, 32:], torch.full((4, 32), 8.0))
    # An all-zero UE8M0 byte is a zero scale, as in mxfp8_block_convert.
    assert g.dequant_mxfp8(w, torch.zeros_like(scale)).abs().max().item() == 0.0


# --------------------------------------------------------------------------
# Hook
# --------------------------------------------------------------------------


def _gdn_stub(tokens=4, head_v_dim=G, heads=48, tp_size=1, **norm_over):
    norm = SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        bias=None,
        weight=torch.ones(head_v_dim, dtype=torch.bfloat16),
        eps=1e-6,
        activation="swish",
    )
    for key, value in norm_over.items():
        setattr(norm, key, value)
    seen = []

    def apply_norm(layer, x, z, norm_weight, group_size, eps, activation):
        seen.append((tuple(x.shape), tuple(z.shape), group_size, activation))
        return torch.full((x.shape[0], 7), 3.0, dtype=torch.bfloat16)

    out_proj = SimpleNamespace(
        tp_size=tp_size,
        use_decode_attn_tp=False,
        bias=None,
        quant_method=SimpleNamespace(apply_norm_gated=apply_norm),
    )
    self = SimpleNamespace(head_v_dim=head_v_dim, norm=norm, out_proj=out_proj)
    k = heads * head_v_dim
    core = torch.randn(tokens * heads, head_v_dim).bfloat16()
    z = torch.randn(tokens * heads, head_v_dim).bfloat16()
    return self, core, z, (tokens, heads, head_v_dim), seen


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setattr(g, "enabled", lambda: True)
    from sglang.srt.models import qwen3_5

    monkeypatch.setattr(qwen3_5, "_is_cuda", True)
    g.reset_flag_cache()
    yield
    g.reset_flag_cache()


def test_hook_returns_none_when_flag_off(monkeypatch):
    from sglang.srt.models import qwen3_5

    monkeypatch.setattr(g, "enabled", lambda: False)
    self, core, z, shape, seen = _gdn_stub()
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, shape) is None
    )
    assert seen == []


def test_hook_routes_the_measured_call(_flag_on):
    from sglang.srt.models import qwen3_5

    self, core, z, shape, seen = _gdn_stub(tokens=4)
    out = qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, shape)
    assert out is not None and out.shape == (4, 7)
    assert seen == [((4, 48 * G), (4, 48 * G), G, "swish")]


@pytest.mark.parametrize(
    "field,value,why",
    [
        ("tp_size", 2, "tensor-parallel out_proj"),
        ("use_decode_attn_tp", True, "decode attn-tp"),
        ("bias", torch.zeros(2560), "biased out_proj"),
    ],
)
def test_hook_rejects_parallel_or_biased_out_proj(_flag_on, field, value, why):
    from sglang.srt.models import qwen3_5

    self, core, z, shape, seen = _gdn_stub()
    setattr(self.out_proj, field, value)
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, shape) is None
    ), why
    assert seen == []


@pytest.mark.parametrize(
    "field,value,why",
    [
        ("group_size", 64, "a second grouping"),
        ("norm_before_gate", False, "gate-before-norm"),
        ("bias", torch.zeros(G), "a norm bias"),
    ],
)
def test_hook_rejects_other_norm_configurations(_flag_on, field, value, why):
    from sglang.srt.models import qwen3_5

    self, core, z, shape, seen = _gdn_stub()
    setattr(self.norm, field, value)
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, shape) is None
    ), why
    assert seen == []


def test_hook_rejects_shape_layouts(_flag_on):
    from sglang.srt.models import qwen3_5

    self, core, z, shape, seen = _gdn_stub()
    # A 3-D core_attn_out, a mismatched gate, a non-contiguous row, a wrong
    # head count and an empty batch all fall through.
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(
            self, core.view(4, 48, G), z, shape
        )
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z[:1], shape)
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, (4, 48, 64))
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(
            self, core.t().contiguous().t(), z, shape
        )
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, (0, 48, G))
        is None
    )
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, (48, G))
        is None
    )
    assert seen == []


def test_hook_declines_when_the_method_has_no_fused_path(_flag_on):
    from sglang.srt.models import qwen3_5

    self, core, z, shape, seen = _gdn_stub()
    self.out_proj.quant_method = SimpleNamespace()
    assert (
        qwen3_5.Qwen3_5GatedDeltaNet._fused_norm_out_proj(self, core, z, shape) is None
    )
    assert seen == []


# --------------------------------------------------------------------------
# CUDA-graph safety of the launch path
# --------------------------------------------------------------------------

_SYNC_CALLS = (
    ".item()",
    ".cpu()",
    ".numpy()",
    "torch.cuda.synchronize",
    "synchronize()",
    ".tolist()",
    "int(",  # a tensor-to-int conversion is a sync
    "bool(",  # ... and so is a tensor-to-bool one
)


def _no_sync(source, what):
    hit = [c for c in _SYNC_CALLS if c in source]
    assert not hit, f"{what} contains a host sync: {hit}"


def test_launch_path_has_no_host_sync():
    _no_sync(inspect.getsource(g.fused_norm_gated_gemv), "fused_norm_gated_gemv")
    _no_sync(inspect.getsource(g._mxfp8_norm_gated_gemv_kernel.fn), "the kernel")
    _no_sync(inspect.getsource(g._workspace), "_workspace")
    _no_sync(inspect.getsource(g.prealloc), "prealloc")


def test_scratch_is_reused_never_reallocated():
    g.reset_workspaces()
    try:
        device = torch.device("meta")
        first = g._workspace(device, capturing=False)
        assert first is not None
        ws, cnt = first
        assert ws.dtype == torch.float32 and cnt.dtype == torch.int32
        assert g._workspace(device, capturing=False) is first, "a second allocation"
        # A capture on a (device, stream) with no entry declines rather than
        # allocating, which is what keeps the scratch out of the graph pool.
        assert g._workspace(torch.device("cuda", 3), capturing=True) is None
        g.prealloc(device)
        assert g.peek_scratch(device) is first
    finally:
        g.reset_workspaces()


def test_prealloc_on_a_non_cuda_device_is_a_noop():
    g.reset_workspaces()
    try:
        g.prealloc(torch.device("cpu"))
        assert g.peek_scratch(torch.device("cpu")) is None
    finally:
        g.reset_workspaces()
