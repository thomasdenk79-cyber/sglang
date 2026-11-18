"""CPU tests for the draft MoE GEMV (SGLANG_OPT_DRAFT_MOE_GEMV).

The kernel comparison runs the Triton interpreter in a subprocess, like
test_mxfp8_skinny_gemv.py does: TRITON_INTERPRET must be set before triton is
imported. H, I, the tile config and the swizzle layout are the production
values for this checkpoint (hidden 2560, moe_intermediate 640, 512 experts);
only the expert count is sliced, because the kernels address experts solely
through topk_ids and unaddressed rows are never read.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import torch
from sglang.srt.layers.moe import draft_moe_gemv as dmg
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="base-a-test-cpu")

H, I, E_FULL = 2560, 640, 512


def _runner_config(**over):
    cfg = {
        "activation": "silu",
        "is_gated": True,
        "apply_router_weight_on_input": False,
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": None,
        "swiglu_limit": None,
    }
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _fake_experts(**over):
    cfg = _runner_config(**over.pop("runner", {}))
    return SimpleNamespace(
        moe_ep_size=over.get("moe_ep_size", 1),
        moe_tp_size=over.get("moe_tp_size", 1),
        moe_runner_config=cfg,
        w13_weight=torch.empty(
            (E_FULL, 2 * I, H // 2), dtype=torch.uint8, device="meta"
        ),
        layer_id=0,
    )


def _dispatch(x=None, ids=None, wts=None):
    x = torch.zeros(1, H, dtype=torch.bfloat16) if x is None else x
    R = x.shape[0]
    ids = torch.zeros(R, I // 64, dtype=torch.int64) if ids is None else ids
    wts = torch.rand(R, ids.shape[1]) if wts is None else wts
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=wts, topk_ids=ids, router_logits=torch.zeros(R, E_FULL)
        ),
    )


_KERNEL_SCRIPT = textwrap.dedent(
    """
    import time

    import torch
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as itp

    # The interpreter stores bf16 as raw uint16 and its binary_op multiplies
    # those bit patterns (only create_dot widens fp8), so the bf16 x bf16
    # product of the block scale and the decoded e2m1 weight is nonsense there.
    # Widen bf16 operands around every binary op and narrow the result back
    # with round-to-nearest-even. Test-side only; the kernel is unchanged.
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

    from sglang.srt.layers.moe import draft_moe_gemv as dmg
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    torch.manual_seed(0)
    H, I, E, R = 2560, 640, 4, 10
    _E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    _SFVALS = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)

    def rand_e4m3(*shape):
        idx = torch.randint(0, len(_SFVALS), shape, dtype=torch.long)
        return _SFVALS[idx].to(torch.float8_e4m3fn)

    def dequant(codes, sf, global_scale):
        lo = codes & 0x0F
        hi = (codes >> 4) & 0x0F
        n = codes.shape[-1] * 2
        vals = torch.zeros(codes.shape[:-1] + (n,), dtype=torch.float32)
        vals[..., 0::2] = _E2M1[lo.long()]
        vals[..., 1::2] = _E2M1[hi.long()]
        full = sf.to(torch.float32).repeat_interleave(16, dim=-1)
        return vals * full * global_scale

    codes13 = torch.randint(0, 16, (E, 2 * I, H // 2), dtype=torch.uint8)
    codes2 = torch.randint(0, 16, (E, H, I // 2), dtype=torch.uint8)
    sf13_raw = rand_e4m3(E, 2 * I, H // 16)
    sf2_raw = rand_e4m3(E, H, I // 16)
    # Gate and up carry different global scales: the gate half must read column
    # 0 (g1_alphas) and the up half column 1 (g1_alphas_up), the contract of
    # _compute_gemm1_alphas that this path replaces.
    s2_13 = torch.rand(E, 2, dtype=torch.float32) * 2e-3 + 1e-3
    s2_13[:, 1] *= 2.0
    s2_2 = torch.rand(E, dtype=torch.float32) * 2e-3 + 1e-3
    sf13 = swizzle_blockscale(sf13_raw)
    sf2 = swizzle_blockscale(sf2_raw)

    x = (torch.randn(1, H) * 0.1).to(torch.bfloat16)

    def call(ids, wts, ws):
        return dmg.draft_moe_gemv(
            hidden_states=x, topk_ids=ids, topk_weights=wts,
            w13_weight=codes13, w13_blockscale_swizzled=sf13,
            w13_weight_scale_2=s2_13, w2_weight=codes2,
            w2_blockscale_swizzled=sf2, w2_weight_scale_2=s2_2,
            workspace=ws,
        )

    _xb = x.to(torch.float32)[0]
    _expert_cache = {}

    def _expert_y(e):
        # The [H] output of expert e on x, dequantized once per expert (a route
        # that repeats an expert must not pay for the dequant twice).
        if e not in _expert_cache:
            w_up = dequant(codes13[e, :I], sf13_raw[e, :I], s2_13[e, 1])
            w_gate = dequant(codes13[e, I:], sf13_raw[e, I:], s2_13[e, 0])
            up = _xb @ w_up.T
            gate = _xb @ w_gate.T
            act = gate * torch.sigmoid(gate) * up
            w2 = dequant(codes2[e], sf2_raw[e], s2_2[e])
            _expert_cache[e] = w2 @ act
        return _expert_cache[e]

    def reference(ids, wts):
        y = torch.zeros(H, dtype=torch.float32)
        for r in range(ids.shape[1]):
            e = int(ids[0, r])
            if e < 0:
                continue  # routes with id < 0 are skipped, as in the kernels
            y += float(wts[0, r]) * _expert_y(e)
        return y.to(torch.bfloat16)

    t0 = time.time()
    ws = dmg.make_workspace(
        num_routes=R, hidden_size=H, intermediate_size=I,
        device=torch.device("cpu"),
    )
    ids_all = torch.randint(0, E, (1, R), dtype=torch.int64)
    wts = torch.rand(1, R, dtype=torch.float32)
    out = call(ids_all, wts, ws)
    rel = (out.float() - reference(ids_all, wts).float()).norm() / (
        reference(ids_all, wts).float().norm() + 1e-9
    )
    print(f"all-routes rel_l2={rel:.3e} ({time.time() - t0:.0f}s)")
    assert rel < 2e-2, f"rel_l2 {rel}"

    out2 = call(ids_all, wts, ws)
    assert torch.equal(out, out2), "determinism: outputs differ across calls"
    assert int(ws.counters.abs().sum()) == 0, "counters left dirty"

    ids_neg = ids_all.clone()
    ids_neg[0, 1] = -1
    out3 = call(ids_neg, wts, ws)
    rel3 = (out3.float() - reference(ids_neg, wts).float()).norm() / (
        reference(ids_neg, wts).float().norm() + 1e-9
    )
    print(f"skipped-route rel_l2={rel3:.3e}")
    assert rel3 < 2e-2, f"rel_l2 skipped {rel3}"
    assert int(ws.counters.abs().sum()) == 0, "counters dirty after skipped route"
    print("KERNEL-OK")
    """
)


def test_kernels_match_reference():
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


def test_unsupported_reason_shape_rules():
    good = dmg.unsupported_reason(
        hidden_size=H, intermediate_size=I, cfg=dmg.DEFAULT_CONFIG
    )
    assert good is None
    assert dmg.unsupported_reason(
        hidden_size=2500, intermediate_size=I, cfg=dmg.DEFAULT_CONFIG
    )
    assert dmg.unsupported_reason(
        hidden_size=H,
        intermediate_size=I,
        cfg=dmg.DraftMoeGemvConfig(
            block_n1=6,
            block_k1=512,
            warps1=4,
            stages1=2,
            block_n2=16,
            block_k2=128,
            splits2=10,
            warps2=2,
            stages2=3,
        ),
    )
    assert dmg.unsupported_reason(
        hidden_size=H,
        intermediate_size=I,
        cfg=dmg.DraftMoeGemvConfig(
            block_n1=8,
            block_k1=500,
            warps1=4,
            stages1=2,
            block_n2=16,
            block_k2=128,
            splits2=10,
            warps2=2,
            stages2=3,
        ),
    )


def test_layer_unsupported_reasons():
    assert dmg._layer_unsupported_reason(_fake_experts()) is None
    for over in (
        {"moe_ep_size": 2},
        {"moe_tp_size": 2},
        {"runner": {"activation": "gelu"}},
        {"runner": {"is_gated": False}},
        {"runner": {"apply_router_weight_on_input": True}},
        {"runner": {"gemm1_alpha": 1.7}},
        {"runner": {"swiglu_limit": 7.0}},
    ):
        reason = dmg._layer_unsupported_reason(_fake_experts(**over))
        assert isinstance(reason, str) and reason, over


def test_maybe_apply_falls_back():
    runner = dmg.DraftMoeGemvRunner()
    runner._workspace = dmg.make_workspace(
        num_routes=10, hidden_size=H, intermediate_size=I, device=torch.device("cpu")
    )
    # Not a StandardDispatchOutput.
    assert runner.maybe_apply(layer=None, dispatch_output=("x", None)) is None
    # Batch size != 1.
    big = _dispatch(x=torch.zeros(2, H, dtype=torch.bfloat16))
    assert runner.maybe_apply(layer=None, dispatch_output=big) is None
    # Pre-quantized input: the CUTLASS path owns it.
    q = _dispatch()
    pre = StandardDispatchOutput(
        hidden_states=q.hidden_states,
        hidden_states_scale=torch.ones(1, H // 16),
        topk_output=q.topk_output,
    )
    assert runner.maybe_apply(layer=None, dispatch_output=pre) is None
    # Non-contiguous activations.
    t = torch.zeros(H, 2, dtype=torch.bfloat16).t()
    assert runner.maybe_apply(layer=None, dispatch_output=_dispatch(x=t)) is None


def test_activation_width_mismatch_falls_back(monkeypatch):
    # k1 reads x as [1, H] with H from w13_weight; a narrower activation
    # would read past its storage. The dispatcher owns both operands, but a
    # reused runner must refuse the mismatch instead of launching an OOB read.
    runner = dmg.DraftMoeGemvRunner()
    called = {}

    def fake_gemv(**kw):
        called["yes"] = True
        return torch.zeros(1, H, dtype=torch.bfloat16)

    monkeypatch.setattr(dmg, "draft_moe_gemv", fake_gemv)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layer = SimpleNamespace(
        w13_weight=torch.empty(
            (E_FULL, 2 * I, H // 2), dtype=torch.uint8, device="meta"
        ),
        w13_blockscale_swizzled=torch.empty(
            (E_FULL, 2 * I, H // 16), dtype=torch.uint8, device="meta"
        ),
        w13_weight_scale_2=torch.ones(E_FULL, 2, device="meta"),
        w2_weight=torch.empty((E_FULL, H, I // 2), dtype=torch.uint8, device="meta"),
        w2_blockscale_swizzled=torch.empty(
            (E_FULL, H, I // 16), dtype=torch.uint8, device="meta"
        ),
        w2_weight_scale_2=torch.ones(E_FULL, device="meta"),
        layer_id=7,
    )
    narrow = _dispatch(x=torch.zeros(1, H // 2, dtype=torch.bfloat16))
    assert runner.maybe_apply(layer=layer, dispatch_output=narrow) is None, (
        "an activation narrower than the layer hidden size must keep CUTLASS"
    )
    assert not called, "the refused call must not reach the kernels"
    assert runner._workspace is None, "the refused call must not size the scratch"
    # Positive control: the matching width still takes the fast path.
    assert runner.maybe_apply(layer=layer, dispatch_output=_dispatch()) is not None
    assert called.get("yes") is True


def test_runner_workspace_and_plumbing(monkeypatch):
    runner = dmg.DraftMoeGemvRunner()
    sentry = torch.zeros(1, H, dtype=torch.bfloat16)
    seen = {}

    def fake_gemv(**kw):
        seen.update(kw)
        return sentry

    monkeypatch.setattr(dmg, "draft_moe_gemv", fake_gemv)
    layer = SimpleNamespace(
        w13_weight=torch.empty(
            (E_FULL, 2 * I, H // 2), dtype=torch.uint8, device="meta"
        ),
        w13_blockscale_swizzled=torch.empty(
            (E_FULL, 2 * I, H // 16), dtype=torch.uint8, device="meta"
        ),
        w13_weight_scale_2=torch.ones(E_FULL, 2, device="meta"),
        w2_weight=torch.empty((E_FULL, H, I // 2), dtype=torch.uint8, device="meta"),
        w2_blockscale_swizzled=torch.empty(
            (E_FULL, H, I // 16), dtype=torch.uint8, device="meta"
        ),
        w2_weight_scale_2=torch.ones(E_FULL, device="meta"),
        layer_id=3,
    )
    disp = _dispatch(ids=torch.randint(0, E_FULL, (1, 10), dtype=torch.int64))

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert runner.maybe_apply(layer=layer, dispatch_output=disp) is None
    assert runner._workspace is None, "capture must not allocate layer scratch"

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ci = runner.maybe_apply(layer=layer, dispatch_output=disp)
    assert ci.hidden_states is sentry
    ws_first = runner._workspace
    assert ws_first.act.shape == (10, I) and ws_first.counters.shape == (H // 16,)
    assert runner.maybe_apply(layer=layer, dispatch_output=disp) is not None
    assert runner._workspace is ws_first, "scratch must be reused, not reallocated"
    assert seen["cfg"] is dmg.DEFAULT_CONFIG


def test_more_routes_than_the_scratch_fall_back(monkeypatch):
    """The scratch is sized by the first call; a wider call must not overrun.

    k1 writes act[r] and k2 reads it with r < topk_ids.shape[1]. A later call
    with more routes than act has rows would write past the buffer into the
    adjacent partials, so it goes to CUTLASS instead.
    """
    runner = dmg.DraftMoeGemvRunner()
    sentry = torch.zeros(1, H, dtype=torch.bfloat16)
    seen = []
    monkeypatch.setattr(
        dmg,
        "draft_moe_gemv",
        lambda **kw: seen.append(kw["topk_ids"].shape[1]) or sentry,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    layer = SimpleNamespace(
        w13_weight=torch.empty(
            (E_FULL, 2 * I, H // 2), dtype=torch.uint8, device="meta"
        ),
        w13_blockscale_swizzled=torch.empty(
            (E_FULL, 2 * I, H // 16), dtype=torch.uint8, device="meta"
        ),
        w13_weight_scale_2=torch.ones(E_FULL, 2, device="meta"),
        w2_weight=torch.empty((E_FULL, H, I // 2), dtype=torch.uint8, device="meta"),
        w2_blockscale_swizzled=torch.empty(
            (E_FULL, H, I // 16), dtype=torch.uint8, device="meta"
        ),
        w2_weight_scale_2=torch.ones(E_FULL, device="meta"),
        layer_id=0,
    )

    d10 = _dispatch(ids=torch.zeros(1, 10, dtype=torch.int64))
    assert runner.maybe_apply(layer=layer, dispatch_output=d10) is not None
    assert seen == [10]

    d12 = _dispatch(ids=torch.zeros(1, 12, dtype=torch.int64))
    assert runner.maybe_apply(layer=layer, dispatch_output=d12) is None, (
        "a call wider than the act scratch must fall back, not overrun it"
    )
    assert seen == [10], "the falling-back call must not reach the kernels"

    # Narrower calls still take the GEMV.
    d8 = _dispatch(ids=torch.zeros(1, 8, dtype=torch.int64))
    assert runner.maybe_apply(layer=layer, dispatch_output=d8) is not None
    assert seen == [10, 8]


def test_grad_requiring_input_falls_back(monkeypatch):
    """A triton launch disconnects autograd silently: refuse it at the gate.

    Same convention as the router GEMV's layout gate -- inside grad mode a
    tensor that requires grad never takes the shortcut; under no_grad it does.
    """
    x = torch.zeros(1, H, dtype=torch.bfloat16, requires_grad=True)
    with torch.enable_grad():
        assert not dmg._is_one_token_call(_dispatch(x=x))
        # A grad-capable weight on the topk side is rejected the same way.
        wts = torch.ones(1, 10, requires_grad=True)
        assert not dmg._is_one_token_call(
            _dispatch(ids=torch.zeros(1, 10, dtype=torch.int64), wts=wts)
        )
        assert dmg._is_one_token_call(_dispatch())  # positive control
    with torch.no_grad():
        assert dmg._is_one_token_call(_dispatch(x=x))


def test_enable_flag_default_off(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_OPT_DRAFT_MOE_GEMV", raising=False)
    assert envs.SGLANG_OPT_DRAFT_MOE_GEMV.get() is False

    # enable_draft_moe_gemv refuses anything but a ModelOpt NVFP4 FusedMoE.
    assert dmg.enable_draft_moe_gemv(SimpleNamespace(quant_method=None)) is False


class _RecordingKernel:
    """Stands in for a triton.JITFunction and records grid and kwargs."""

    def __init__(self, sink, tag):
        self.sink = sink
        self.tag = tag

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.sink.append((self.tag, grid, kwargs))

        return run


def _meta_layer_tensors():
    return dict(
        w13_weight=torch.empty((8, 2 * I, H // 2), dtype=torch.uint8, device="meta"),
        w13_blockscale_swizzled=torch.empty(
            (8, 2 * I, H // 16), dtype=torch.uint8, device="meta"
        ),
        w2_weight=torch.empty((8, H, I // 2), dtype=torch.uint8, device="meta"),
        w2_blockscale_swizzled=torch.empty(
            (8, H, I // 16), dtype=torch.uint8, device="meta"
        ),
    )


def test_gate_and_up_halves_read_their_own_global_scale(monkeypatch):
    """``_compute_gemm1_alphas`` gives w1 column 0 and w3 column 1; so must k1.

    The donor pinned column 0 for both halves, which is only right for a
    shared-scale checkpoint. A [num_experts, 2] scale is the normal NVFP4
    layout, so the two halves have to be scaled apart.
    """
    calls = []
    monkeypatch.setattr(dmg, "_draft_moe_up_gate_kernel", _RecordingKernel(calls, "k1"))
    monkeypatch.setattr(dmg, "_draft_moe_down_kernel", _RecordingKernel(calls, "k2"))
    ws = dmg.make_workspace(
        num_routes=10, hidden_size=H, intermediate_size=I, device=torch.device("meta")
    )
    base = dict(
        hidden_states=torch.empty((1, H), dtype=torch.bfloat16, device="meta"),
        topk_ids=torch.zeros((1, 10), dtype=torch.int64, device="meta"),
        topk_weights=torch.ones((1, 10), dtype=torch.float32, device="meta"),
        w2_weight_scale_2=torch.ones(8, device="meta"),
        workspace=ws,
    )
    base.update(_meta_layer_tensors())

    for scale_2, expected in (
        (torch.rand(8, 2, device="meta"), 1),  # separate gate / up scales
        (torch.rand(8, 1, device="meta"), 0),  # one shared scale, second column absent
        (torch.rand(8, device="meta"), 0),  # ... and the 1-D form of it
    ):
        calls.clear()
        dmg.draft_moe_gemv(**base, w13_weight_scale_2=scale_2)
        kwargs = next(kw for tag, _, kw in calls if tag == "k1")
        assert kwargs["G_UP_OFF"] == expected, (scale_2.shape, kwargs["G_UP_OFF"])
