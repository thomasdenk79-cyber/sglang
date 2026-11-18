"""CPU tests for the MXFP8 skinny GEMV (mxfp8_skinny_gemv).

The kernel comparison runs the Triton interpreter in a subprocess, like
test_sm120_bf16_skinny_gemv.py does: TRITON_INTERPRET has to be set before
triton is imported. Shapes are this checkpoint's real MXFP8 dense linears
(/srv/llm-models/Qwen3.8-Flash-Next-FP4-Local/config.json, hidden_size 2560).
Every numerical case uses the plan the production planner picks for the REAL
(N, K): N is sub-sampled for the interpreter, the tile is not.
"""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from sglang.srt.layers.quantization import mxfp8_skinny_gemv as g
from sglang.test.ci.ci_register import register_cpu_ci

# The interpreter block below is the cost: its subprocess runs ~127 launches at
# the real K (145 s measured, the whole file 193 s on a loaded host), the rest is
# the gate, planner and hook checks.
register_cpu_ci(est_time=300, suite="base-a-test-cpu")

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
_SMS = 188  # the RTX PRO 6000 the tables were tuned on


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


def _operands(n=256, k=640, m=4, seed=0):
    torch.manual_seed(seed)
    w, s = _quantize_bf16(torch.randn(n, k) * 0.05)
    return torch.randn(m, k, dtype=torch.bfloat16), w, s


def _source_of(fn) -> str:
    return inspect.getsource(fn)


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", M_VALUES)
@pytest.mark.parametrize("name", [s[0] for s in REAL_SHAPES])
def test_plan_is_legal_for_every_real_shape(name, m):
    n, k = _SHAPE_BY_NAME[name]
    block_n, block_k, splits, use_dot, warps, stages = g._plan(m, n, k, _SMS)
    assert block_k % g.GROUP_K == 0, "BLOCK_K must cover whole UE8M0 groups"
    assert block_k >= g.GROUP_K and 1 <= splits <= g._MAX_SPLITS
    assert 1 <= warps and 1 <= stages
    # The broadcast-multiply path (use_dot False) is legal at M == 1 only; every
    # wider M has to go through tl.dot, which is what _use_dot enforces.
    assert g._use_dot(m, use_dot) == (m > 1 or use_dot)
    if not use_dot:
        assert m == 1
    block_m = g._block_m(m, use_dot)
    assert block_m >= min(m, g.MAX_M) and block_m & (block_m - 1) == 0
    # The split-K reduction reads and writes one fp32 tile per (n block, split).
    n_blocks = -(-n // block_n)
    assert n_blocks <= g._WS_COUNTERS
    assert n_blocks * splits * block_m * block_n <= g._WS_FLOATS


@pytest.mark.parametrize(
    "n,k,m",
    [
        (1, 32, 1),
        (2, 32, 16),
        (31, 96, 4),
        (512, 320, 8),
        (4095, 1024, 1),
        (248320, 20480, 16),
        (0, 2560, 4),  # degenerate: no grid axis may be empty, no division by zero
        (256, 0, 4),
    ],
)
def test_plan_is_legal_for_degenerate_and_awkward_shapes(n, k, m):
    block_n, block_k, splits, use_dot, warps, stages = g._plan(m, n, k, _SMS)
    assert block_n >= 1 and block_k % g.GROUP_K == 0 and 1 <= splits <= g._MAX_SPLITS
    block_m = g._block_m(m, use_dot)
    assert block_m & (block_m - 1) == 0, (n, k, m, block_m)
    # The rows a launch owns must be a power of two and cover this M.
    assert block_m >= (1 if m <= 1 else min(m, g.MAX_M)), (n, k, m, block_m)


def test_scale_shape_is_one_byte_per_group():
    for k in _ALL_KS:
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


def test_applies_refuses_empty_batch_and_degenerate_weights():
    """M == 0, N == 0 and K == 0 must not reach the kernel.

    At M == 0 the broadcast plan reads row 0 of a zero-row x; N == 0 and K == 0
    give an empty grid axis and a plan with SPLITS == 0.
    """
    w, s = _quantize_bf16(torch.randn(256, 2560) * 0.05)
    x = torch.randn(4, 2560, dtype=torch.bfloat16)
    assert g.mxfp8_skinny_gemv_applies(x, w, s)
    empty = torch.empty(0, 2560, dtype=torch.bfloat16)
    assert g.mxfp8_skinny_gemv_applies(empty, w, s) is False
    assert g.mxfp8_skinny_gemv_applies(x, w[:0], s[:0]) is False  # N == 0
    assert g.mxfp8_skinny_gemv_applies(x[:, :0], w[:, :0], s[:, :0]) is False  # K == 0
    # The predicate documents itself as one that cannot raise.
    for junk in (None, 3.0, "w", [1, 2], torch.zeros(2, 2, 2)):
        assert g.mxfp8_skinny_gemv_applies(x, junk, s) is False
        assert g.mxfp8_skinny_gemv_applies(junk, w, s) is False
        assert g.mxfp8_skinny_gemv_applies(x, w, junk) is False


def test_entry_validates_what_the_gate_checks():
    """The entry point checks the same invariants, so a direct caller cannot
    drive the kernel into an addressing mode it does not support."""
    x, w, s = _operands(n=256, k=640, m=4)
    good = g.mxfp8_skinny_gemv(x, w, s)
    assert good is not None and tuple(good.shape) == (4, 256)

    cases = {
        "M == 0": (torch.empty(0, 640, dtype=torch.bfloat16), w, s),
        "N == 0": (x, w[:0], s[:0]),
        "K == 0": (x[:, :0], w[:, :0], s[:, :0]),
        "M over the entry bound": (
            torch.randn(g._ENTRY_MAX_M + 1, 640, dtype=torch.bfloat16),
            w,
            s,
        ),
        "fp32 x": (x.float(), w, s),
        "e5m2 weight": (x, w.view(torch.float8_e5m2), s),
        "float scale": (x, w, s.to(torch.float32)),
        "short scale": (x, w, s[:, : s.shape[1] - 1]),
        "transposed scale": (x, w, s.t().contiguous().t()),
        "K not a multiple of 32": (
            torch.randn(4, 2590, dtype=torch.bfloat16),
            torch.zeros(256, 2590, dtype=torch.float8_e4m3fn),
            torch.zeros(256, 80, dtype=torch.uint8),
        ),
        "3-D x": (x.unsqueeze(0), w, s),
    }
    for label, (xx, ww, ss) in cases.items():
        with pytest.raises(ValueError, match="skinny gemv"):
            g.mxfp8_skinny_gemv(xx, ww, ss)
        assert g.mxfp8_skinny_gemv_applies(xx, ww, ss) is False, label


def test_out_argument_is_validated():
    x, w, s = _operands(n=256, k=640, m=4)
    bad = {
        "wrong shape": torch.zeros(4, 641, dtype=torch.bfloat16),
        "wrong dtype": torch.zeros(4, 256, dtype=torch.float32),
        "not row contiguous": torch.zeros(256, 4, dtype=torch.bfloat16).t(),
        "1-d": torch.zeros(4 * 256, dtype=torch.bfloat16),
    }
    for label, out in bad.items():
        with pytest.raises(ValueError, match="skinny gemv"):
            g.mxfp8_skinny_gemv(x, w, s, out=out)


def test_reference_matches_the_loader_dequant():
    from sglang.srt.layers.quantization.mxfp8_block_convert import (
        _ue8m0_to_fp32,
        dequant_mxfp8_2d_to_bf16,
    )

    torch.manual_seed(0)
    w, s = _quantize_bf16(torch.randn(64, 128) * 0.05)
    x = torch.randn(4, 128).bfloat16()
    ref = g.mxfp8_skinny_gemv_reference(x, w, s)
    deq = dequant_mxfp8_2d_to_bf16(w, s)  # the bf16 round the loader applies
    torch.testing.assert_close(ref, x.float() @ deq.float().t(), atol=0, rtol=2e-2)
    # UE8M0 decoding: 0x00 is a zero scale, 0x7F is 1.0, 0xFE is 2**127, and the
    # same byte mapping as the in-tree dequant for all 256 bytes.
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
    for e in range(256):
        sb = torch.full((1, 1), e, dtype=torch.uint8)
        got = g.mxfp8_dequant_to_fp32(q1[:, :32], sb)[0, 0]
        want = _ue8m0_to_fp32(sb).item()
        if want == 0:
            assert got == 0, e
        elif torch.isinf(torch.tensor(want)):
            assert torch.isinf(torch.tensor(got)) and got > 0, e
        else:
            assert got == want, (e, got, want)


# --------------------------------------------------------------------------
# Launch mechanics that do not need a kernel: chunking, counters, scratch
# --------------------------------------------------------------------------


class _FakeKernel:
    """Records the launches _launch makes instead of running Triton."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return run


def _capture_launches(monkeypatch):
    fake = _FakeKernel()
    monkeypatch.setattr(g, "_mxfp8_skinny_gemv_kernel", fake)
    return fake


def test_launch_covers_every_row_above_block_m(monkeypatch):
    """M above BLOCK_M is served by further launches, not by unwritten rows."""
    fake = _capture_launches(monkeypatch)
    x, w, s = _operands(n=256, k=640, m=40)
    out = torch.full((40, 256), float("nan"), dtype=torch.bfloat16)
    y = g.mxfp8_skinny_gemv(x, w, s, out=out)
    assert y is out
    assert len(fake.calls) == 2, [c[0] for c in fake.calls]
    rows = sorted(c[1][6] for c in fake.calls)  # the M argument of each chunk
    assert rows == [8, 32], rows
    # Each chunk reads its own rows of x and writes its own rows of the caller's
    # out buffer, and together they cover the matrix exactly once.
    x_offs, y_offs = [], []
    for _grid, args, _kw in fake.calls:
        m_chunk = args[6]
        assert args[0].shape[0] == m_chunk, "x was not sliced to the chunk"
        assert args[3].shape[0] == m_chunk, "out was not sliced to the chunk"
        x_offs.append((args[0].data_ptr() - x.data_ptr()) // x.element_size())
        y_offs.append((args[3].data_ptr() - out.data_ptr()) // out.element_size())
    assert x_offs == [0, 32 * 640], x_offs  # row-major: 32 rows of K=640
    assert y_offs == [0, 32 * 256], y_offs  # 32 rows of N=256
    # M within one tile is still exactly one launch.
    fake.calls.clear()
    g.mxfp8_skinny_gemv(*_operands(m=16))
    assert len(fake.calls) == 1


def test_launch_clears_counters_eagerly_and_not_in_a_capture():
    """A counter left non-zero by an aborted launch must not truncate the next one."""
    n_blocks = -(-256 // 32)

    cnt = torch.full((g._WS_COUNTERS,), 7, dtype=torch.int32)
    g._clear_counters(cnt, n_blocks, capturing=False)
    assert int(cnt[:n_blocks].abs().sum()) == 0
    assert int(cnt[n_blocks:].abs().sum()) == 7 * (g._WS_COUNTERS - n_blocks)
    # Inside a capture no memset may be issued: the kernel's own atomic-exchange
    # reset is what a replayed graph relies on, and a memset node would run on
    # every step.
    held = torch.full((g._WS_COUNTERS,), 7, dtype=torch.int32)
    g._clear_counters(held, n_blocks, capturing=True)
    assert int(held[:n_blocks].abs().sum()) == 7 * n_blocks, "memset inside a capture"


def test_launch_resets_counters_before_a_split_k_launch(monkeypatch):
    fake = _capture_launches(monkeypatch)
    x, w, s = _operands(n=256, k=640, m=4)
    ws = torch.zeros(g._WS_FLOATS, dtype=torch.float32)
    cnt = torch.full((g._WS_COUNTERS,), 9, dtype=torch.int32)
    y = torch.zeros(4, 256, dtype=torch.bfloat16)
    g._launch(ws, cnt, x, w, s, y, 4, 256, 640, (32, 128, 5, True, 4, 3))
    assert len(fake.calls) == 1
    assert int(cnt[:8].abs().sum()) == 0, "a split-K launch ran on dirty counters"
    # SPLITS == 1 uses no counters at all, so it does not touch them.
    fake.calls.clear()
    cnt2 = torch.full((g._WS_COUNTERS,), 5, dtype=torch.int32)
    g._launch(ws, cnt2, x, w, s, y, 4, 256, 640, (32, 128, 1, True, 4, 3))
    assert len(fake.calls) == 1
    assert int(cnt2.abs().sum()) == 5 * g._WS_COUNTERS


def test_launch_counts_itself_for_the_benchmark(monkeypatch):
    """The bench must be able to tell a captured GEMV from a silent fallback."""
    fake = _capture_launches(monkeypatch)
    x, w, s = _operands(n=256, k=640, m=4)
    before = g.launch_count()
    g.mxfp8_skinny_gemv(x, w, s)
    assert g.launch_count() == before + 1
    g.mxfp8_skinny_gemv(x, w, s, cfg=(32, 128, 2, True, 4, 3))
    assert g.launch_count() == before + 2
    assert len(fake.calls) == 2, "the counter and the launches disagree"
    assert g.stats()["launches"] == g.launch_count()
    assert g.stats()["scratch_misses"] >= 0


def test_entry_reports_a_scratch_miss(monkeypatch):
    x, w, s = _operands(n=256, k=640, m=4)
    monkeypatch.setattr(g, "_workspace", lambda device, stream=None: None)
    before = g.stats()["scratch_misses"]
    assert g.mxfp8_skinny_gemv(x, w, s) is None
    assert g.stats()["scratch_misses"] == before + 1


def test_pinned_plan_is_demoted_when_it_would_not_fit(monkeypatch):
    """cfg= is a caller's request, not a licence to write past the scratch."""
    seen = []
    monkeypatch.setattr(
        g, "_launch", lambda ws, cnt, x, w, s, y, M, N, K, cfg: seen.append(cfg)
    )
    # 130 n-blocks x 32 splits x 16 rows x 32 cols is over the 8 MiB scratch, so
    # a caller cannot pin that plan into an out-of-bounds partial store.
    x, w, s = _operands(n=4160, k=640, m=16)
    fits = (32, 128, 4, True, 4, 3)
    assert g._fits(16, 4160, fits)
    g.mxfp8_skinny_gemv(x, w, s, cfg=fits)
    assert seen[-1] == fits
    over = (32, 128, g._MAX_SPLITS, True, 4, 3)
    assert not g._fits(16, 4160, over)
    g.mxfp8_skinny_gemv(x, w, s, cfg=over)
    assert seen[-1] == (32, 128, 1, True, 4, 3), seen[-1]


def test_plan_for_uses_the_real_shape():
    """A plan keyed on a sub-sampled N would test tiles production never runs."""
    x = torch.zeros(1, 2560, dtype=torch.bfloat16)
    w = torch.zeros(16384, 2560, dtype=torch.float8_e4m3fn)
    assert g.plan_for(x, w) == g._plan(1, 16384, 2560, g._num_sms(x.device))
    # The tuned M == 1 entry, not the heuristic, is what plan_for reports.
    assert g.plan_for(x, w) == g._BY_SHAPE_M1[(16384, 2560)]


class _FakeStream:
    def __init__(self, handle):
        self.cuda_stream = handle


class _FakeDevice:
    def __init__(self, kind):
        self.type = kind


def _fake_cuda(monkeypatch, capturing=False):
    """Make _workspace believe it is on CUDA while allocating on the host."""
    real_empty, real_zeros = torch.empty, torch.zeros
    allocs = []
    state = {"capturing": capturing}

    def empty(n, dtype=None, device=None):
        allocs.append(("empty", n))
        return real_empty(n, dtype=dtype)

    def zeros(n, dtype=None, device=None):
        allocs.append(("zeros", n))
        return real_zeros(n, dtype=dtype)

    monkeypatch.setattr(g, "_WS", {})
    monkeypatch.setattr(g, "_WS_WARNED", set())
    monkeypatch.setattr(g.torch, "empty", empty)
    monkeypatch.setattr(g.torch, "zeros", zeros)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state["capturing"]
    )
    return _FakeDevice("cuda"), allocs, state


def test_workspace_is_per_stream_and_never_reallocated(monkeypatch):
    """Patch 0041: two live split-K launches must not share one counter array."""
    dev, allocs, _ = _fake_cuda(monkeypatch)
    a, b = _FakeStream(0x1000), _FakeStream(0x2000)
    ws_a = g._workspace(dev, a)
    ws_b = g._workspace(dev, b)
    assert ws_a is not None and ws_b is not None
    assert ws_a[0].data_ptr() != ws_b[0].data_ptr()
    assert ws_a[1].data_ptr() != ws_b[1].data_ptr()
    n_alloc = len(allocs)
    for _ in range(50):
        assert g._workspace(dev, a) is ws_a
        assert g._workspace(dev, b) is ws_b
    assert len(allocs) == n_alloc, "the scratch was reallocated per launch"


def test_workspace_miss_during_capture_falls_back(monkeypatch):
    dev, _allocs, state = _fake_cuda(monkeypatch, capturing=True)
    known = _FakeStream(0x1000)
    # prealloc ran before the capture for this stream: usable inside it.
    ws_known = g._workspace(dev, known)
    state["capturing"] = True
    assert g._workspace(dev, known) is ws_known
    # A stream prealloc never covered: no scratch, and no allocation baked into
    # the graph's private pool. The caller keeps its ordinary GEMM.
    assert g._workspace(dev, _FakeStream(0x2000)) is None
    assert g._workspace(dev, _FakeStream(0x3000)) is None


def test_workspace_is_bounded(monkeypatch):
    """No stream may pin another 8 MiB forever: past the cap it keeps its GEMM."""
    dev, _allocs, _state = _fake_cuda(monkeypatch)
    for i in range(g._WS_LIMIT):
        assert g._workspace(dev, _FakeStream(0x10000 + i)) is not None
    assert len(g._WS) == g._WS_LIMIT
    assert g._workspace(dev, _FakeStream(0x9999)) is None
    assert len(g._WS) == g._WS_LIMIT, "the cap was exceeded"
    # An entry already handed to a graph is never dropped to make room for one.
    assert g._workspace(dev, _FakeStream(0x10000)) is not None


def test_prealloc_covers_forward_capture_and_alt_streams(monkeypatch):
    """0041 materialized both slots; the shared expert runs on the alt stream.

    The MoE block puts the shared expert's linears on get_stream("alt") with
    SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP at its default, and the QSA indexer
    forks onto it inside the capture only, so both the capture stream and the alt
    stream need their scratch before the first capture.
    """
    calls = []
    monkeypatch.setattr(
        g, "_forward_streams", lambda device: ["capture-stream", "alt-stream"]
    )
    monkeypatch.setattr(
        g, "_workspace", lambda device, stream=None: calls.append((device, stream))
    )
    g.prealloc(torch.device("cuda", 0), stream="mine")
    seen = [c[1] for c in calls]
    assert "capture-stream" in seen and "alt-stream" in seen, calls
    assert "mine" in seen, "a caller's own capture stream was not warmed"
    assert None in seen, "the current (forward) stream was not warmed"
    assert seen.count("alt-stream") == 1


def test_forward_streams_asks_for_the_capture_and_alt_stream(monkeypatch):
    """The list is what prealloc warms; check the names it asks the runtime for."""
    asked = []

    class _Ctx:
        @staticmethod
        def get_stream(name):
            asked.append(name)
            return "stream-" + name

    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_stream", _Ctx.get_stream)
    streams = g._forward_streams(torch.device("cuda", 0))
    assert "alt" in asked, asked
    assert "stream-alt" in streams


def test_launch_path_allocates_nothing_and_syncs_no_host():
    whole = open(g.__file__, encoding="utf-8").read()
    for needle in (".item(", "torch.cuda.synchronize", ".cpu()", ".tolist("):
        assert needle not in whole, needle
    for fn in (g._launch, g._launch_chunk, g._workspace, g._plan, g._stream_id):
        src = _source_of(fn)
        for needle in (".item(", "torch.cuda.synchronize", ".cpu()", ".tolist("):
            assert needle not in src, (fn.__name__, needle)
    # The only allocation allowed in the entry point is the optional y= buffer.
    src = _source_of(g.mxfp8_skinny_gemv)
    assert src.count("torch.empty") == 1
    assert "if out is None" in src
    launch_src = _source_of(g._launch) + _source_of(g._launch_chunk)
    assert "torch.empty" not in launch_src and "torch.zeros" not in launch_src


def test_counters_are_zero_helper_detects_a_dirty_scratch(monkeypatch):
    counters = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
    scratch = (torch.zeros(g._WS_FLOATS, dtype=torch.float32), counters)
    monkeypatch.setattr(g, "peek_scratch", lambda device=None, stream=None: scratch)
    assert g.counters_are_zero()
    counters[3] = 1
    assert not g.counters_are_zero(), "a poisoned counter must not read as clean"
    monkeypatch.setattr(g, "peek_scratch", lambda device=None, stream=None: None)
    assert g.counters_are_zero(), "a stream with no scratch has nothing to poison"


# --------------------------------------------------------------------------
# Kernel vs fp32 reference, in the Triton interpreter
# --------------------------------------------------------------------------

#: N sub-sampled for the interpreter (a full 16384 rows x 2560 K would run for
#: minutes); K is always the real one and the plan is always the one the real
#: (N, K) gets, so every tuned tile and every 32-wide scale group is covered.
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
    N_SUB = 32
    SMS = 188

    def quantize(n, k, scale_bytes=None):
        if scale_bytes is None:
            scale_bytes = torch.randint(
                118, 138, (n, k // GROUP), dtype=torch.uint8
            ).contiguous()
        descale = (scale_bytes.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
        q = (
            (torch.randn(n, k) * 0.05)
            .view(n, k // GROUP, GROUP)
            .div(descale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .view(n, k)
            .contiguous()
        )
        return q, scale_bytes.contiguous()

    torch.manual_seed(0)
    ws = torch.empty(g._WS_FLOATS, dtype=torch.float32)
    cnt = torch.zeros(g._WS_COUNTERS, dtype=torch.int32)
    failures = []
    seen = {"launches": 0, "uneven_k": 0, "uneven_splits": 0, "broadcast": 0}

    def check(tag, m, cfg, got, ref):
        if not torch.isfinite(got).all():
            failures.append((tag, m, cfg, "nonfinite"))
            return
        scale = ref.abs().amax().clamp(min=1e-6)
        abs_err = (got - ref).abs().max().item()
        rel = ((got - ref).abs() / ref.abs().clamp(min=1e-3)).max().item()
        if abs_err > 2e-2 * scale:
            failures.append((tag, m, cfg, "abs", abs_err, scale.item()))
        if rel > 5e-2:
            failures.append((tag, m, cfg, "rel", rel, scale.item()))

    def launch(tag, n, k, m, cfg, w, e, x, attempts=2):
        \"\"\"The kernel against the fp32 reference, twice if asked.

        The second launch only works if the first left the split-K counters at
        zero, which is the whole CUDA-graph replay contract.
        \"\"\"
        block_n, block_k, splits, use_dot, _warps, _stages = cfg
        seen["launches"] += 1
        if k % block_k:
            seen["uneven_k"] += 1
        if splits > 1 and ((k + block_k - 1) // block_k) % splits:
            seen["uneven_splits"] += 1
        if not g._use_dot(m, use_dot):
            seen["broadcast"] += 1
        ref = g.mxfp8_skinny_gemv_reference(x, w, e)
        y = None
        for _attempt in range(attempts):
            y = torch.full((m, n), float("nan"), dtype=torch.bfloat16)
            g._launch(ws, cnt, x, w, e, y, m, n, k, cfg)
            check(tag, m, cfg, y.float(), ref)
            if int(cnt.abs().sum()) != 0:
                failures.append((tag, m, cfg, "counters-not-reset"))
        return y

    # (a) every real shape at every spec width, on the PRODUCTION plan of the
    # real (N, K): this is what runs the tuned BLOCK_N / SPLITS / USE_DOT tiles.
    for name, n, k in SHAPES:
        for m in (1, 2, 4, 8, 16):
            cfg = g._plan(m, n, k, SMS)
            w, e = quantize(N_SUB, k)
            launch(name + "|prod", N_SUB, k, m, cfg, w, e,
                   torch.randn(m, k).bfloat16())

    # (a2) out= writes the caller's buffer, bit-identically to the allocated one.
    for name, n, k in SHAPES[:3]:
        for m in (1, 16):
            cfg = g._plan(m, n, k, SMS)
            w, e = quantize(N_SUB, k)
            x = torch.randn(m, k).bfloat16()
            y = g.mxfp8_skinny_gemv(x, w, e, cfg=cfg)
            buf = torch.full((m, N_SUB), -7.0, dtype=torch.bfloat16)
            out = g.mxfp8_skinny_gemv(x, w, e, out=buf, cfg=cfg)
            if out is not buf or not torch.equal(out, buf) or not torch.equal(out, y):
                failures.append((name, m, cfg, "out-not-written-or-differs"))

    # (b) exact coverage. w == x == 1.0 and the scale is a power of two that
    # varies per group AND per row, so every partial is an exact integer (times a
    # power of two) and the result is representable in bf16 with no rounding: a
    # skipped or doubled k-block, or a scale group paired with the wrong K slice,
    # is an exact mismatch rather than a tolerance question.
    for name, n, k in SHAPES:
        for m in (1, 16):
            cfg = g._plan(m, n, k, SMS)
            e = torch.full((N_SUB, k // GROUP), 0x7F, dtype=torch.uint8)
            e[:, 1::2] = 0x7E           # 1.0 / 0.5 alternating along K
            e[1::4] = 0x7C              # a quarter of the rows at 0.25
            w = torch.ones(N_SUB, k, dtype=torch.float8_e4m3fn)
            x = torch.ones(m, k, dtype=torch.bfloat16)
            launch(name + "|exact", N_SUB, k, m, cfg, w, e, x, attempts=1)
            mult = torch.pow(
                torch.tensor(2.0), (e.to(torch.int32) - 127).to(torch.float32)
            )
            want = mult.sum(dim=1) * GROUP
            err = ((g.mxfp8_skinny_gemv_reference(x, w, e) - want.unsqueeze(0)).abs()
                   / want.unsqueeze(0).clamp(min=1e-6)).max().item()
            if err > 1e-6:
                failures.append((name, m, cfg, "reference-not-exact", err))

    # (c) uneven split-K: SPLITS does not divide the number of K blocks.
    k = 6144  # 48 blocks of 128
    for splits in (5, 3, 7, 11):
        w, e = quantize(N_SUB, k)
        launch("uneven-splits-%d" % splits, N_SUB, k, 4,
               (32, 128, splits, True, 4, 3), w, e, torch.randn(4, k).bfloat16())

    # (d) EVEN_K == False: K is a multiple of 32 but not of BLOCK_K, with and
    # without split-K, and on the broadcast path. No real shape is ragged, but the
    # pinned M=1 tile of shared_expert.down_proj (BLOCK_K 256 at K 640) reaches the
    # branch in production, so the masked scale-row guard (gk < sf_k) is live code.
    for k, cfg in ((2400, (32, 128, 4, True, 4, 3)),
                   (2400, (32, 128, 1, True, 4, 3)),
                   (6208, (16, 256, 3, True, 4, 3)),
                   (6624, (32, 128, 5, True, 4, 3)),
                   (2400, (32, 128, 7, True, 4, 3)),
                   (640, (16, 256, 3, False, 4, 3))):
        assert k % GROUP == 0 and k % cfg[1] != 0, (k, cfg)
        w, e = quantize(N_SUB, k)
        m = 1 if not cfg[3] else 4
        launch("odd-k-%d-s%d-dot%d" % (k, cfg[2], cfg[3]), N_SUB, k, m, cfg, w, e,
               torch.randn(m, k).bfloat16(), attempts=1)

    # (e) forced tiles the planner would not pick, including the broadcast path at
    # a shape whose production plan uses tl.dot.
    for tag, m, cfg, (n0, k0) in (
        ("forced-broadcast", 1, (16, 32, 8, False, 4, 3), (2560, 2560)),
        ("forced-narrow", 4, (32, 32, 4, True, 4, 2), (2560, 2560)),
        ("forced-single", 16, (32, 128, 1, True, 4, 3), (2560, 6144)),
        ("forced-dot-at-m1", 1, (32, 128, 5, True, 4, 3), (2560, 640)),
        ("broadcast-narrow", 1, (16, 32, 4, False, 4, 3), (2560, 640)),
    ):
        w, e = quantize(N_SUB, k0)
        launch(tag, N_SUB, k0, m, cfg, w, e, torch.randn(m, k0).bfloat16(),
               attempts=1)

    # (f) extreme exponents through a PRODUCTION plan, and a wholly zero-scaled
    # weight that must contribute exactly zero.
    e = torch.full((N_SUB, 2560 // GROUP), 0x01, dtype=torch.uint8)  # 2**-126
    e[: N_SUB // 2] = 0x00                                           # zero scale
    e[N_SUB // 4] = 0xFE                                             # 2**127
    w = quantize(N_SUB, 2560, e)[0]
    for m in (1, 4, 16):
        launch("extreme-prodplan-m%d" % m, N_SUB, 2560, m,
               g._plan(m, 1280, 2560, SMS), w, e, torch.randn(m, 2560).bfloat16(),
               attempts=1)
    z = torch.full((N_SUB, 2560 // GROUP), 0x00, dtype=torch.uint8)
    y = torch.full((4, N_SUB), float("nan"), dtype=torch.bfloat16)
    g._launch(ws, cnt, torch.randn(4, 2560).bfloat16(), w, z, y, 4, N_SUB, 2560,
              (32, 128, 5, True, 4, 3))
    if not torch.equal(y, torch.zeros_like(y)):
        failures.append(("zero-scale", 4, None, "not-zero"))

    # (g) the empty batch, and the widths at and above the BLOCK_M ceiling. The
    # M == 0 launch must not read x and must leave the counters alone; above the
    # ceiling every row of the output must have been written.
    for n in (2560, 1280):
        cfg = g._plan(0, n, 2560, SMS)
        x0 = torch.zeros(0, 2560, dtype=torch.bfloat16)
        y0 = torch.zeros(0, N_SUB, dtype=torch.bfloat16)
        g._launch(ws, cnt, x0, w, e, y0, 0, N_SUB, 2560, cfg)
        if int(cnt.abs().sum()) != 0:
            failures.append(("m0-n%d" % n, 0, cfg, "counters-not-reset"))
    for m in (32, 40):
        w2, e2 = quantize(N_SUB, 640)
        x2 = torch.randn(m, 640).bfloat16()
        cfg = g._plan(m, 2560, 640, SMS)
        buf = torch.full((m, N_SUB), float("nan"), dtype=torch.bfloat16)
        got = g.mxfp8_skinny_gemv(x2, w2, e2, out=buf, cfg=cfg)
        if got is not buf or not torch.isfinite(got.float()).all():
            failures.append(("m%d-over-ceiling" % m, m, cfg, "unwritten-rows"))
            continue
        check("m%d" % m, m, cfg, got.float(), g.mxfp8_skinny_gemv_reference(x2, w2, e2))
        if not g.counters_are_zero(torch.device("cpu")):
            failures.append(("m%d" % m, m, cfg, "counters-not-reset"))

    if seen["uneven_k"] < 1 or seen["uneven_splits"] < 1 or seen["broadcast"] < 1:
        failures.append(("branch-coverage", None, None, dict(seen)))
    print("cases:", seen)
    if failures:
        raise AssertionError(failures[:16])
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
        timeout=2400,
    )
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), (
        out.stdout[-6000:] + "\n" + out.stderr[-12000:]
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
    monkeypatch.setattr(g, "is_sm120", lambda device=None: True)
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
    # An empty batch is not a forward: it must never reach the kernel.
    assert not method._mxfp8_skinny_gemv_ok(
        _layer(n=512, k=2560), torch.empty(0, 2560, dtype=torch.bfloat16), None
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
    monkeypatch.setattr(g, "is_sm120", lambda device=None: False)
    assert not method._mxfp8_skinny_gemv_ok(layer, x, None)
    monkeypatch.setattr(g, "is_sm120", lambda device=None: True)
    assert method._mxfp8_skinny_gemv_ok(layer, x, None)
    monkeypatch.delitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV")
    assert not method._mxfp8_skinny_gemv_ok(layer, x, None)


def test_max_m_env_bounds_the_gate(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_mod

    method = _method()
    monkeypatch.setattr(fp8_mod, "_is_cuda", True)
    monkeypatch.setattr(g, "is_sm120", lambda device=None: True)
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
    # An out-of-range value is clamped to the widest bucket the tile tables were
    # measured at, and the gate never raises.
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV_MAX_M", "4096")
    assert g.gemv_max_m() == g.MAX_TUNED_M == 16
    assert method._mxfp8_skinny_gemv_ok(
        layer, torch.randn(16, 2560, dtype=torch.bfloat16), None
    )
    assert not method._mxfp8_skinny_gemv_ok(
        layer, torch.randn(17, 2560, dtype=torch.bfloat16), None
    )
    monkeypatch.setitem(os.environ, "SGLANG_MXFP8_SKINNY_GEMV_MAX_M", "0")
    assert g.gemv_max_m() == 1


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
