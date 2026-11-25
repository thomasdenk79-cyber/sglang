"""NVFP4 FP8-scratch launch plumbing (CPU only, no Triton launches).

Donor port of the sglang-wt-nvfp4fp8 launch test (2420da2ba9..6bdcfa37ab).

Pins the wiring the numeric fidelity tests cannot see: the per-layer descale
(6 * global_scale) reaches the attention consumers, the gathers receive the
1/6 gather scale, the split-K Triton-decode gate only opens for an NVFP4 pool
under the FP8 scratch, and an unknown scratch mode fails closed.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

HEADS, HALF_DIM, DIM = 2, 128, 256


def _module():
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

    return qb


def _backend(**attrs):
    obj = object.__new__(_module().QwenSparseAttnBackend)
    obj._fa2_scratch = {}
    obj._nvfp4_sparse_tables = {}
    obj._trtllm_sparse_tables = {}
    obj._trtllm_counters = {}
    obj._trtllm_workspace = None
    obj._cuda_graph_max_tokens = 0
    obj.compress_ratio = 1
    obj._uses_block_indices = lambda ti: False
    obj._nvfp4_scratch_dtype = torch.float8_e4m3fn
    obj._nvfp4_split_k_gate_inputs = (False, True, True, True)
    obj._decode_attn_workspace = None
    obj._nvfp4_split_k_requested = False
    obj._nvfp4_kv_plan_logged = True
    obj._nvfp4_split_k_skip_notes = set()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _topk_indices(batch, topk, rows_valid=(37, 20)):
    idx = torch.full((batch, topk), -1, dtype=torch.int32)
    for r, n in enumerate(rows_valid):
        idx[r, :n] = torch.arange(n, dtype=torch.int32)
    return idx


def _scales():
    k_sf = torch.zeros(8, HEADS, DIM // 16, dtype=torch.uint8)
    v_sf = torch.zeros(8, HEADS, DIM // 16, dtype=torch.uint8)
    k_g = torch.tensor([0.75])
    v_g = torch.tensor([1.5])
    return (k_sf, v_sf, k_g, v_g)


class TestScratchModeAndGate(CustomTestCase):
    def test_scratch_mode_parses(self):
        qb = _module()
        self.assertIs(qb._nvfp4_scratch_dtype_from_mode("fp8"), torch.float8_e4m3fn)
        self.assertIs(qb._nvfp4_scratch_dtype_from_mode("bf16"), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fp8.*bf16"):
            qb._nvfp4_scratch_dtype_from_mode("fp4")

    def test_split_k_gate_only_fp8_nvfp4(self):
        qb = _module()
        args = (True, True, True, True)
        self.assertTrue(qb._nvfp4_sparse_decode_eligible(*args))
        for i in range(4):
            broken = list(args)
            broken[i] = False
            self.assertFalse(
                qb._nvfp4_sparse_decode_eligible(*broken),
                f"requirement {i} is not enforced",
            )

    def test_gather_scales_fp8_recipe(self):
        scales_in = _scales()
        scales, k_d, v_d = _backend()._nvfp4_fp8_gather_scales(scales_in)
        self.assertIs(scales[0], scales_in[0])
        self.assertIs(scales[1], scales_in[1])
        for got in scales[2:]:
            self.assertEqual(tuple(got.shape), (1,))
            self.assertTrue(torch.allclose(got, torch.full_like(got, 1 / 6)))
        self.assertEqual(tuple(k_d.shape), ())
        self.assertEqual(float(k_d), 0.75 * 6.0)
        self.assertEqual(float(v_d), 1.5 * 6.0)

    def test_gather_scales_bf16_mode_is_passthrough(self):
        scales_in = _scales()
        scales, k_d, v_d = _backend(
            _nvfp4_scratch_dtype=torch.bfloat16
        )._nvfp4_fp8_gather_scales(scales_in)
        self.assertEqual(scales, scales_in)
        self.assertIsNone(k_d)
        self.assertIsNone(v_d)

    def test_zero_global_scale_does_not_poison_scratch(self):
        # g=0 must give a zero gather scale (plain dequant yields zeros), not 0/0.
        scales, k_d, _ = _backend()._nvfp4_fp8_gather_scales(
            (_scales()[0], _scales()[1], torch.zeros(1), torch.ones(1))
        )
        self.assertEqual(float(scales[2]), 0.0)
        self.assertEqual(float(k_d), 0.0)

    def _hook_warning(self, scratch):
        import os
        from unittest import mock

        import sglang.srt.server_args as sa

        cfg = object.__new__(sa.ServerArgs)
        cfg.kv_cache_dtype = "nvfp4"
        cfg.disaggregation_mode = "null"
        cfg.enable_lmcache = False
        cfg.attn_cp_size = 1
        cfg.enable_hierarchical_cache = False
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_OPT_TRITON_DECODE_ATTN": "1",
                    "SGLANG_QSA_NVFP4_SCRATCH": scratch,
                },
            ),
            mock.patch.object(sa.logger, "warning") as warn,
        ):
            cfg._check_qsa_kv4_compatibility(cfg)
        return [
            c.args[0] for c in warn.call_args_list if "TRITON_DECODE_ATTN" in c.args[0]
        ]

    def test_hook_warns_on_bf16_scratch_only(self):
        self.assertEqual(self._hook_warning("fp8"), [])
        warned = self._hook_warning("bf16")
        self.assertEqual(len(warned), 1)
        self.assertIn("bf16", warned[0])


class TestTrtllmLaunchFolding(CustomTestCase):
    def _run(self, scratch_dtype):
        qb = _module()
        obj = _backend(_nvfp4_scratch_dtype=scratch_dtype)
        batch, q_heads, topk = 2, 4, 37
        q = torch.randn(batch, q_heads, DIM)
        k_buffer = torch.zeros(8, HEADS, HALF_DIM, dtype=torch.uint8)
        v_buffer = torch.zeros_like(k_buffer)
        req_to_token = torch.arange(8 * 96, dtype=torch.int32).reshape(8, 96)
        obj.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        metadata = types.SimpleNamespace(
            is_cuda_graph=False,
            sequence_lengths=torch.tensor([90, 40], dtype=torch.int32),
            fa2_valid_counts=None,
            row_req_pool_indices=None,
        )
        forward_batch = types.SimpleNamespace(
            req_pool_indices=torch.tensor([3, 5], dtype=torch.int32),
            batch_size=batch,
        )
        layer = types.SimpleNamespace(scaling=0.125)
        topk_indices = torch.full((batch, topk), -1, dtype=torch.int32)
        topk_indices[0, :37] = torch.arange(37, dtype=torch.int32)
        topk_indices[1, :20] = torch.arange(20, dtype=torch.int32)
        launches = {}
        obj._trtllm_counters = {
            (batch, q_heads, q.device): torch.zeros(1, dtype=torch.uint8)
        }

        def fake_compact(*args, **kwargs):
            launches["compact"] = (args, kwargs)

        def fake_counts(seq_lens, indices, valid, b, t):
            valid.copy_((indices[:b] >= 0).sum(-1).to(torch.int32))

        def fake_decode(**kwargs):
            launches["decode"] = kwargs
            return torch.full((batch, q_heads, DIM), 2.0)

        scales = _scales()
        orig = (
            qb.qwen_sparse_kv_extraction_compact_nvfp4_triton,
            qb.qwen_sparse_valid_counts_triton,
        )
        qb.qwen_sparse_kv_extraction_compact_nvfp4_triton = fake_compact
        qb.qwen_sparse_valid_counts_triton = fake_counts
        try:
            out = obj._forward_trtllm_sparse(
                q,
                k_buffer,
                v_buffer,
                layer,
                forward_batch,
                metadata,
                topk_indices,
                fake_decode,
                nvfp4=scales,
            )
        finally:
            qb.qwen_sparse_kv_extraction_compact_nvfp4_triton = orig[0]
            qb.qwen_sparse_valid_counts_triton = orig[1]
        return out, launches, q, scales

    def test_fp8_mode_reaches_kernels(self):
        out, launches, q, scales = self._run(torch.float8_e4m3fn)
        args, _ = launches["compact"]
        k_g_got, v_g_got = args[4], args[5]
        self.assertTrue(torch.allclose(k_g_got, torch.full_like(k_g_got, 1 / 6)))
        self.assertTrue(torch.allclose(v_g_got, torch.full_like(v_g_got, 1 / 6)))
        self.assertIs(args[11].dtype, torch.float8_e4m3fn)  # out_k
        self.assertIs(args[12].dtype, torch.float8_e4m3fn)  # out_v
        decode = launches["decode"]
        self.assertTrue(
            torch.equal(decode["query"], (q * (scales[2] * 6.0)).contiguous())
        )
        self.assertTrue(
            torch.allclose(out, torch.full_like(out, 2.0 * float(scales[3] * 6.0)))
        )

    def test_bf16_mode_is_bit_identical_passthrough(self):
        out, launches, q, scales = self._run(torch.bfloat16)
        args, _ = launches["compact"]
        self.assertIs(args[2], scales[0])  # same scale tensors, untouched
        self.assertIs(args[4], scales[2])  # raw global scale, no 1/6
        self.assertIs(args[11].dtype, q.dtype)
        self.assertTrue(torch.equal(launches["decode"]["query"], q.contiguous()))
        self.assertTrue(torch.all(out == 2.0))


class TestSplitKLaunchOnScratch(CustomTestCase):
    def _run(self, scratch_dtype=torch.float8_e4m3fn, q_heads=8):
        qb = _module()
        obj = _backend(_nvfp4_scratch_dtype=scratch_dtype, _cuda_graph_max_tokens=8)
        batch, topk = 2, 37
        q = torch.randn(batch, q_heads, DIM)
        k_buffer = torch.zeros(8, HEADS, HALF_DIM, dtype=torch.uint8)
        v_buffer = torch.zeros_like(k_buffer)
        obj.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(8 * 96, dtype=torch.int32).reshape(8, 96)
        )
        metadata = types.SimpleNamespace(
            is_cuda_graph=False,
            sequence_lengths=torch.tensor([90, 40], dtype=torch.int32),
            fa2_valid_counts=None,
            row_req_pool_indices=None,
        )
        forward_batch = types.SimpleNamespace(
            req_pool_indices=torch.tensor([3, 5], dtype=torch.int32),
            batch_size=batch,
        )
        layer = types.SimpleNamespace(scaling=0.125)
        topk_indices = torch.full((batch, topk), -1, dtype=torch.int32)
        topk_indices[0, :37] = torch.arange(37, dtype=torch.int32)
        topk_indices[1, :20] = torch.arange(20, dtype=torch.int32)
        launches = {}

        def fake_compact(*args, **kwargs):
            launches["compact"] = args

        def fake_counts(seq_lens, indices, valid, b, t):
            valid.copy_((indices[:b] >= 0).sum(-1).to(torch.int32))

        def fake_decode(**kwargs):
            launches["decode"] = kwargs
            return torch.ones(batch, q_heads, DIM)

        orig = (
            qb.qwen_sparse_kv_extraction_compact_nvfp4_triton,
            qb.qwen_sparse_valid_counts_triton,
            qb.qsa_decode_attention,
        )
        qb.qwen_sparse_kv_extraction_compact_nvfp4_triton = fake_compact
        qb.qwen_sparse_valid_counts_triton = fake_counts
        qb.qsa_decode_attention = fake_decode
        try:
            out = obj._forward_nvfp4_sparse_decode(
                q,
                layer,
                forward_batch,
                metadata,
                topk_indices,
                k_buffer,
                v_buffer,
                _scales(),
            )
        finally:
            (
                qb.qwen_sparse_kv_extraction_compact_nvfp4_triton,
                qb.qwen_sparse_valid_counts_triton,
                qb.qsa_decode_attention,
            ) = orig
        return out, launches, q

    def test_identity_table_and_descale_reach_split_k(self):
        out, launches, q = self._run()
        decode = launches["decode"]
        stride = 64
        self.assertIs(decode["k_buffer"].dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(decode["k_buffer"].shape), (8 * stride, HEADS, DIM))
        self.assertTrue(decode["prefix_valid"])
        slots = decode["req_to_token"]
        rows, cols = torch.arange(8)[:, None], torch.arange(stride)[None, :]
        self.assertTrue(torch.equal(slots, rows * stride + cols))
        self.assertTrue(
            torch.equal(
                decode["topk_indices"],
                torch.arange(stride, dtype=torch.int32).expand(2, stride),
            )
        )
        self.assertTrue(
            torch.equal(
                decode["row_req_pool_indices"], torch.arange(2, dtype=torch.int32)
            )
        )
        self.assertEqual(decode["seq_lens"].tolist(), [37, 20])
        self.assertTrue(torch.allclose(decode["q"], q * (0.75 * 6.0)))
        self.assertTrue(torch.allclose(out, torch.full_like(out, 1.5 * 6.0)))

    def test_bf16_scratch_defers_to_trtllm_path(self):
        out, _, _ = self._run(scratch_dtype=torch.bfloat16)
        self.assertIsNone(out)

    def test_abi_rejects_cannot_express(self):
        out, _, _ = self._run(q_heads=8 * 17)  # q/kv ratio 17 > 16
        self.assertIsNone(out)

    def _run_graph(self, counts_buffer):
        qb = _module()
        obj = _backend(
            _nvfp4_scratch_dtype=torch.float8_e4m3fn, _cuda_graph_max_tokens=8
        )
        batch, topk = 2, 37
        q = torch.randn(batch, 8, DIM)
        k_buffer = torch.zeros(8, HEADS, HALF_DIM, dtype=torch.uint8)
        v_buffer = torch.zeros_like(k_buffer)
        obj.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(8 * 96, dtype=torch.int32).reshape(8, 96)
        )
        metadata = types.SimpleNamespace(
            is_cuda_graph=True,
            sequence_lengths=torch.tensor([90, 40], dtype=torch.int32),
            fa2_valid_counts=counts_buffer,
            row_req_pool_indices=None,
        )
        forward_batch = types.SimpleNamespace(
            req_pool_indices=torch.tensor([3, 5], dtype=torch.int32),
            batch_size=batch,
        )
        launches = {}

        def fake_compact(*args, **kwargs):
            launches["compact"] = args

        def fake_decode(**kwargs):
            launches["decode"] = kwargs
            return torch.ones(batch, 8, DIM)

        orig = (
            qb.qwen_sparse_kv_extraction_compact_nvfp4_triton,
            qb.qsa_decode_attention,
        )
        qb.qwen_sparse_kv_extraction_compact_nvfp4_triton = fake_compact
        qb.qsa_decode_attention = fake_decode
        try:
            out = obj._forward_nvfp4_sparse_decode(
                q,
                types.SimpleNamespace(scaling=0.125),
                forward_batch,
                metadata,
                _topk_indices(batch, topk),
                k_buffer,
                v_buffer,
                _scales(),
            )
        finally:
            (
                qb.qwen_sparse_kv_extraction_compact_nvfp4_triton,
                qb.qsa_decode_attention,
            ) = orig
        return out, launches

    def test_graph_mode_fill_produces_valid_counts(self):
        # Bug regression (round-1 P0): under CUDA-graph capture the split-K path
        # must launch the valid-counts fill itself; a capture that skips it
        # replays a kernel reading a stale (typically zero) buffer forever.
        counts = torch.full((4,), 999, dtype=torch.int32)  # stale sentinel
        out, launches = self._run_graph(counts)
        self.assertIsNotNone(out)
        decode = launches["decode"]
        self.assertEqual(decode["seq_lens"].tolist(), [37, 20])
        self.assertEqual(decode["seq_lens"].data_ptr(), counts.data_ptr())
        self.assertEqual(counts[:2].tolist(), [37, 20])  # filled in place

    def test_graph_mode_without_counts_buffer_defers_to_trtllm(self):
        out, launches = self._run_graph(None)
        self.assertIsNone(out)
        self.assertNotIn("decode", launches)

    def test_identity_table_cache_is_bounded(self):
        # Bug regression (round-1 P2): the table cache keyed on the eager row
        # count grew without bound under drifting batch sizes; rows must round
        # up to shared capacity mirrors and the cache must evict.
        qb = _module()
        obj = _backend(_nvfp4_scratch_dtype=torch.float8_e4m3fn)
        for rows in range(1, 40):
            slots, columns, row_index = obj._nvfp4_sparse_decode_tables(rows, 32, "cpu")
            cap = slots.shape[0]
            self.assertGreaterEqual(cap, rows)
            self.assertTrue(
                cap & (cap - 1) == 0, f"rows {rows}: cap {cap} not a bucket"
            )
            self.assertEqual(tuple(row_index.shape), (cap,))
            self.assertLessEqual(
                len(obj._nvfp4_sparse_tables), qb._NVFP4_SPARSE_TABLE_CACHE
            )
        # rows inside one bucket share a single mirror (sliced by the caller)
        small = obj._nvfp4_sparse_decode_tables(5, 32, "cpu")
        big = obj._nvfp4_sparse_decode_tables(8, 32, "cpu")
        self.assertIs(small[0], big[0])
        # distinct strides are distinct keys; eviction caps the dict
        for stride in range(64, 64 * 41, 64):
            obj._nvfp4_sparse_decode_tables(2, stride, "cpu")
        self.assertLessEqual(
            len(obj._nvfp4_sparse_tables), qb._NVFP4_SPARSE_TABLE_CACHE
        )
        self.assertEqual(len(obj._nvfp4_sparse_tables), qb._NVFP4_SPARSE_TABLE_CACHE)


class TestNVFP4SplitKPlanLog(CustomTestCase):
    def test_gate_reason_names_first_unmet_requirement(self):
        f = _module()._nvfp4_sparse_decode_gate_reason
        self.assertIsNone(f(True, True, True, True))
        needles = [
            "SGLANG_OPT_TRITON_DECODE_ATTN",
            "bf16",
            "not NVFP4",
            "CUDA",
        ]
        for i, needle in enumerate(needles):
            broken = [True] * 4
            broken[i] = False
            self.assertIn(needle, f(*broken), f"requirement {i} has no reason text")
        self.assertIn("TRITON_DECODE_ATTN", f(False, False, False, False))

    def _obj(self, gate_inputs, requested=False, logged=False):
        return _backend(
            _nvfp4_scratch_dtype=(
                torch.float8_e4m3fn if gate_inputs[1] else torch.bfloat16
            ),
            _nvfp4_split_k_gate_inputs=tuple(gate_inputs),
            _nvfp4_split_k_requested=requested,
            _nvfp4_kv_plan_logged=logged,
        )

    def test_plan_log_reports_open_gate_once(self):
        qb = _module()
        obj = self._obj((True, True, True, True), requested=True)
        with self.assertLogs(qb.__name__, level="INFO") as cm:
            obj._log_nvfp4_kv_plan()
        self.assertTrue(
            any("scratch=fp8 split_k=on" in m for m in cm.output), cm.output
        )
        self.assertTrue(obj._nvfp4_kv_plan_logged)

    def test_requested_but_closed_gate_warns(self):
        qb = _module()
        obj = self._obj((True, False, True, True), requested=True)
        with self.assertLogs(qb.__name__, level="WARNING") as cm:
            obj._log_nvfp4_kv_plan()
        self.assertTrue(
            any(
                "split_k=off (SGLANG_QSA_NVFP4_SCRATCH is bf16)" in m for m in cm.output
            ),
            cm.output,
        )

    def test_split_k_skip_note_logs_once_per_reason(self):
        qb = _module()
        obj = self._obj((True, True, True, True))
        with self.assertLogs(qb.__name__, level="INFO") as cm:
            for _ in range(3):
                obj._note_nvfp4_split_k_skip("rows 3 != batch 2 (verify)")
        self.assertEqual(sum("rows 3 != batch 2" in m for m in cm.output), 1)


if __name__ == "__main__":
    unittest.main()
