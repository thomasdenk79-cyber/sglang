"""QSA NVFP4 KV: dequantizing gathers and the QSA quant-method recipe.

The gathers are checked bit-exactly against an independent PyTorch dequant
(``reference_dequant``) on packed codes produced by a PyTorch NVFP4 quantizer;
the Triton kernels run on CPU under ``TRITON_INTERPRET=1`` (or on CUDA).
"""

import os
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    DEVICE = None

HEADS, DIM = 2, 256
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_nvfp4(x, g):
    """PyTorch NVFP4 quantizer: ``[m, h, d]`` -> packed ``[m, h, d/2]`` uint8 +
    E4M3 block scales ``[m, h, d/16]`` (uint8), even element in the low nibble."""
    m, h, d = x.shape
    blocks = x.float().view(m, h, d // 16, 16) / g
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn)
    denom = scale.float()
    scaled = torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks))
    idx = (scaled.abs().unsqueeze(-1) - E2M1).abs().argmin(-1)
    codes = (idx | ((scaled < 0).long() << 3)).view(m, h, d).to(torch.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale.view(m, h, d // 16).view(torch.uint8).contiguous()


def make_kv(m, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(m, HEADS, DIM, generator=gen)
    mag = torch.exp2(torch.empty(m, HEADS, DIM // 16, 1).uniform_(-6, 6, generator=gen))
    return (x.view(m, HEADS, DIM // 16, 16) * mag).view(m, HEADS, DIM)


@unittest.skipIf(DEVICE is None, "needs CUDA or TRITON_INTERPRET=1")
class TestQSANVFP4Gathers(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import reference_dequant

        cls.slots = 300
        cls.k_g = torch.tensor([0.75], device=DEVICE)
        cls.v_g = torch.tensor([1.5], device=DEVICE)
        k, ks = quantize_nvfp4(make_kv(cls.slots, 0), 0.75)
        v, vs = quantize_nvfp4(make_kv(cls.slots, 1), 1.5)
        cls.k, cls.k_sf = k.to(DEVICE), ks.to(DEVICE)
        cls.v, cls.v_sf = v.to(DEVICE), vs.to(DEVICE)
        cls.k_ref = reference_dequant(cls.k, cls.k_sf, cls.k_g)
        cls.v_ref = reference_dequant(cls.v, cls.v_sf, cls.v_g)

    def test_reference_layout_roundtrip(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import reference_dequant

        x = make_kv(64, 2)
        data, sf = quantize_nvfp4(x, 1.0)
        g = torch.tensor([1.0])
        rel = lambda y: ((y - x).norm() / x.norm()).item()
        self.assertLess(rel(reference_dequant(data, sf, g)), 0.15)
        swapped = ((data & 0x0F) << 4) | (data >> 4)
        self.assertGreater(rel(reference_dequant(swapped, sf, g)), 0.5)

    def test_gather_dequant_matches_reference(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import nvfp4_gather_dequant

        index = torch.tensor([5, 0, 299, 17, 17, 128], device=DEVICE)
        for dtype in (torch.float32, torch.bfloat16):
            got = nvfp4_gather_dequant(self.k, self.k_sf, self.k_g, index, dtype)
            self.assertTrue(torch.equal(got, self.k_ref[index].to(dtype)))

    def _compact(self, out_dtype, zero_fill):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import (
            qwen_sparse_kv_extraction_compact_nvfp4_triton,
        )
        from sglang.srt.layers.attention.qsa.sparse_attn import (
            qwen_sparse_valid_counts_triton,
        )

        batch, topk, stride = 3, 37, 48
        req_to_token = torch.stack(
            [torch.randperm(self.slots, device=DEVICE)[:100] for _ in range(4)]
        ).to(torch.int32)
        req_indices = torch.tensor([2, 0, 3], dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor([100, 20, 60], dtype=torch.int32, device=DEVICE)
        indices = torch.full((batch, topk), -1, dtype=torch.int32, device=DEVICE)
        indices[0, :37] = torch.arange(37, dtype=torch.int32) * 2
        indices[1, :25] = torch.arange(25, dtype=torch.int32)  # 20..24 out of range
        indices[2, :10] = torch.arange(50, 60, dtype=torch.int32)
        if zero_fill:
            counts = torch.empty(batch, dtype=torch.int32, device=DEVICE)
            qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk)
            cu_k = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * stride
            rows = batch * stride
        else:
            counts = torch.tensor([37, 20, 10], dtype=torch.int32, device=DEVICE)
            cu_k = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
            rows = int(counts.sum())
        out_k = torch.full((rows, HEADS, DIM), float("nan"), device=DEVICE).to(
            out_dtype
        )
        out_v = out_k.clone()
        qwen_sparse_kv_extraction_compact_nvfp4_triton(
            self.k,
            self.v,
            self.k_sf,
            self.v_sf,
            self.k_g,
            self.v_g,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            batch,
            topk,
            zero_fill_cols=stride if zero_fill else 0,
        )
        for b in range(batch):
            n = int(counts[b])
            start = int(cu_k[b])
            pos = indices[b, :n].long()
            slots = req_to_token[int(req_indices[b]), pos].long()
            for out, ref in ((out_k, self.k_ref), (out_v, self.v_ref)):
                got = out[start : start + n].float()
                self.assertTrue(torch.equal(got, ref[slots].to(out_dtype).float()))
                if zero_fill:
                    self.assertTrue(
                        torch.all(out[start + n : start + stride].float() == 0)
                    )
        return counts

    def test_compact_matches_reference(self):
        # The backend only dequantizes into the query dtype (bf16).
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                self._compact(dtype, zero_fill=False)

    def test_compact_zero_fills_page_stride(self):
        counts = self._compact(torch.bfloat16, zero_fill=True)
        self.assertEqual(counts.tolist(), [37, 20, 10])

    def test_prefill_pack_matches_reference(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import pack_qsa_prefill_kv_nvfp4

        req_to_token = torch.stack(
            [torch.randperm(self.slots, device=DEVICE)[:64] for _ in range(3)]
        ).to(torch.int32)
        req_indices = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)
        seq_lens = torch.tensor([7, 33], dtype=torch.int32, device=DEVICE)
        cu_k = torch.nn.functional.pad(seq_lens.cumsum(0), (1, 0)).to(torch.int32)
        pk, pv = pack_qsa_prefill_kv_nvfp4(
            k=self.k,
            v=self.v,
            k_sf=self.k_sf,
            v_sf=self.v_sf,
            k_g=self.k_g,
            v_g=self.v_g,
            req_to_token=req_to_token,
            req_indices=req_indices,
            seq_lens=seq_lens,
            cu_k=cu_k,
            total_k=40,
            output_dtype=torch.bfloat16,
        )
        slots = torch.cat([req_to_token[1, :7], req_to_token[2, :33]]).long()
        self.assertTrue(torch.equal(pk, self.k_ref[slots].to(torch.bfloat16)))
        self.assertTrue(torch.equal(pv, self.v_ref[slots].to(torch.bfloat16)))

    def test_rejects_mismatched_scales(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import nvfp4_gather_dequant

        index = torch.tensor([0], device=DEVICE)
        with self.assertRaisesRegex(ValueError, "block scales"):
            nvfp4_gather_dequant(
                self.k, self.k_sf[:, :, :8].contiguous(), self.k_g, index, torch.float32
            )


class TestQSANVFP4FP8Scratch(CustomTestCase):
    """FP8 gather scratch: the kernels receive ``g / descale`` (= 1/E2M1_MAX)
    and write ``value / descale`` into E4M3; consumers multiply the descale
    back. Pinned property: the descaled scratch stays within one E4M3 ulp of
    the true dequant, extreme block scales land exactly on +/-448 (no overflow),
    and zero block scales stay exactly zero.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import reference_dequant

        cls.slots = 160
        cls.g = torch.tensor([0.75], device=DEVICE) if DEVICE else None
        if DEVICE is None:
            raise unittest.SkipTest("needs CUDA or TRITON_INTERPRET=1")
        k, ks = quantize_nvfp4(make_kv(cls.slots, 11), 0.75)
        v, vs = quantize_nvfp4(make_kv(cls.slots, 12), 0.75)
        cls.k, cls.k_sf = k.to(DEVICE), ks.to(DEVICE)
        cls.v, cls.v_sf = v.to(DEVICE), vs.to(DEVICE)
        cls.ref = reference_dequant(cls.k, cls.k_sf, cls.g)

    def _cases(self):
        cases = {
            "random": (self.k.clone(), self.k_sf.clone()),
            # 448 (0x7E) x +/-6: stored = 448*6/6 must land exactly on +/-448.
            "max-sf": (torch.full_like(self.k, 0x7F), torch.full_like(self.k_sf, 0x7E)),
            "zero-sf": (self.k.clone(), torch.zeros_like(self.k_sf)),
            # Smallest E4M3 denormal scale (0x01 = 2^-9) with +/-0.5 codes.
            "subnormal": (
                torch.full_like(self.k, 0x01 | 0x09),
                torch.where(
                    torch.arange(self.k_sf.numel(), device=DEVICE) % self.k_sf.shape[2]
                    == 0,
                    torch.tensor(1, dtype=torch.uint8, device=DEVICE),
                    torch.tensor(0, dtype=torch.uint8, device=DEVICE),
                )
                .view_as(self.k_sf)
                .contiguous(),
            ),
            "mixed-sign": (
                torch.full_like(self.k, 0x07 | (0x08 << 4)),
                self.k_sf.clone(),
            ),
        }
        return cases

    def _fp8_gather(self, data, sf):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import (
            nvfp4_fp8_descale,
            nvfp4_fp8_gather_scale,
            nvfp4_gather_dequant,
            reference_dequant,
        )

        m = data.shape[0]
        index = torch.arange(m, device=DEVICE)
        got = nvfp4_gather_dequant(
            data,
            sf,
            nvfp4_fp8_gather_scale(self.g),
            index,
            torch.float8_e4m3fn,
        )
        descale = nvfp4_fp8_descale(self.g)
        true = reference_dequant(data, sf, self.g)
        return got, descale, true

    def test_fp8_roundtrip_error_table(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import FP8_E4M3_MAX

        rows = []
        for name, (data, sf) in self._cases().items():
            with self.subTest(case=name):
                got, descale, true = self._fp8_gather(data, sf)
                self.assertTrue(torch.isfinite(got.float()).all())
                back = got.float() * descale
                err = (back - true).abs()
                # One E4M3 ulp in the normal range, plus the smallest subnormal
                # step as headroom (the interpret-mode cast flushes stored values
                # below it to zero; the hardware cvt may round either way).
                bound = true.abs() * 2**-3 + (descale.abs() * 2**-8)
                self.assertTrue(
                    bool((err <= bound).all()),
                    f"{name}: max err {err.max().item()} exceeds one-e4m3-ulp bound",
                )
                nz = true.abs() > 0
                rel = err[nz] / true.abs()[nz]
                rows.append(
                    (
                        name,
                        err.max().item(),
                        err.mean().item(),
                        rel.max().item() if rel.numel() else 0.0,
                        rel.mean().item() if rel.numel() else 0.0,
                        int((got.float() == 0).sum()),
                    )
                )
                if name == "max-sf":
                    self.assertTrue(
                        bool((got.float().abs() == FP8_E4M3_MAX).all()),
                        "saturated values escaped the descale recipe",
                    )
                if name == "zero-sf":
                    self.assertTrue(bool((got.float() == 0).all()))
        table = (
            "\nFP8 scratch error table "
            "(case, max_abs, mean_abs, max_rel, mean_rel, zero_rows):\n"
        )
        self.maxDiff = None
        print(
            table
            + "\n".join(
                f"  {r[0]:>10}  {r[1]:.3e}  {r[2]:.3e}  {r[3]:.3e}  {r[4]:.3e}  {r[5]}"
                for r in rows
            )
        )

    def test_fp8_compact_and_prefill_descale_roundtrip(self):
        from sglang.srt.layers.attention.qsa.nvfp4_kv import (
            nvfp4_fp8_descale,
            nvfp4_fp8_gather_scale,
            pack_qsa_prefill_kv_nvfp4,
            qwen_sparse_kv_extraction_compact_nvfp4_triton,
            reference_dequant,
        )
        from sglang.srt.layers.attention.qsa.sparse_attn import (
            qwen_sparse_valid_counts_triton,
        )

        scale = nvfp4_fp8_gather_scale(self.g)
        descale = nvfp4_fp8_descale(self.g)
        batch, topk, stride = 2, 40, 64
        req_to_token = torch.stack(
            [torch.randperm(self.slots, device=DEVICE)[:80] for _ in range(3)]
        ).to(torch.int32)
        req_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor([80, 45], dtype=torch.int32, device=DEVICE)
        indices = torch.full((batch, topk), -1, dtype=torch.int32, device=DEVICE)
        indices[0, :40] = torch.arange(40, dtype=torch.int32)
        indices[1, :30] = torch.arange(30, dtype=torch.int32)
        counts = torch.empty(batch, dtype=torch.int32, device=DEVICE)
        qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk)
        cu_k = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * stride
        out_k = torch.zeros(
            batch * stride, HEADS, DIM, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        out_v = torch.zeros_like(out_k)
        qwen_sparse_kv_extraction_compact_nvfp4_triton(
            self.k,
            self.v,
            self.k_sf,
            self.v_sf,
            scale,
            scale,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            batch,
            topk,
            zero_fill_cols=stride,
        )
        for b in range(batch):
            n, start = int(counts[b]), b * stride
            slots = req_to_token[int(req_indices[b]), indices[b, :n].long()].long()
            true = reference_dequant(self.k, self.k_sf, self.g)[slots]
            err = (out_k[start : start + n].float() * descale - true).abs()
            self.assertTrue(bool((err <= true.abs() * 2**-3 + descale * 2**-8).all()))
            self.assertTrue(
                bool((out_k[start + n : start + stride].float() == 0).all())
            )
        pk, pv = pack_qsa_prefill_kv_nvfp4(
            k=self.k,
            v=self.v,
            k_sf=self.k_sf,
            v_sf=self.v_sf,
            k_g=scale,
            v_g=scale,
            req_to_token=req_to_token,
            req_indices=req_indices,
            seq_lens=seq_lens,
            cu_k=torch.nn.functional.pad(seq_lens, (1, 0)).to(torch.int32),
            total_k=125,
            output_dtype=torch.float8_e4m3fn,
        )
        slots = torch.cat([req_to_token[1, :80], req_to_token[2, :45]]).long()
        true = reference_dequant(self.k, self.k_sf, self.g)[slots]
        err = (pk.float() * descale - true).abs()
        self.assertEqual(int(pk.shape[0]), 125)
        self.assertTrue(bool((err <= true.abs() * 2**-3 + descale * 2**-8).all()))
        true_v = reference_dequant(self.v, self.v_sf, self.g)[slots]
        err_v = (pv.float() * descale - true_v).abs()
        self.assertTrue(bool((err_v <= true_v.abs() * 2**-3 + descale * 2**-8).all()))

    def test_fp8_scratch_attention_matches_bf16_scratch(self):
        # Bug regression (round-1 review): which consumer sees the FP8 scratch
        # and how the descales are folded is wiring, not math. The real
        # chunk-prefill kernel receives the backend's exact pairing (q * k_d
        # before, out * v_d after); a dropped fold leaves a one-descale-sized
        # error and a swapped pair changes the softmax shape (scores scale by
        # v_d instead of k_d), both far outside FP8 quantization noise.
        from sglang.srt.layers.attention.qsa.nvfp4_kv import (
            FP8_E4M3_MAX,
            nvfp4_fp8_descale,
            nvfp4_fp8_gather_scale,
            pack_qsa_prefill_kv_nvfp4,
        )
        from sglang.srt.layers.attention.qsa.sparse_attn import (
            sparse_gqa_fwd_interface_triton_ck,
        )

        gen = torch.Generator().manual_seed(21)
        q_heads, topk = 4, 16
        seq_lens = torch.tensor([40, 30], dtype=torch.int32, device=DEVICE)
        cu = torch.tensor([0, 40, 70], dtype=torch.int32, device=DEVICE)
        total_q = total_k = 70
        k_g = torch.tensor([0.75], device=DEVICE)
        v_g = torch.tensor([1.5], device=DEVICE)
        k, k_sf = quantize_nvfp4(make_kv(self.slots, 31), 0.75)
        v, v_sf = quantize_nvfp4(make_kv(self.slots, 32), 1.5)
        k, k_sf, v, v_sf = (t.to(DEVICE) for t in (k, k_sf, v, v_sf))
        req_to_token = (
            ((torch.arange(3 * 70) % (self.slots - 1) + 1).view(3, 70))
            .to(torch.int32)
            .to(DEVICE)
        )
        req_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
        # fp32 query: the kernel folds scale*log2e in the query dtype and the
        # Triton 3.8 interpreter has no bf16 scalar builder; the arms compared
        # are the K/V scratch storages, not the query.
        q = (
            torch.randn(total_q, q_heads, DIM, generator=gen)
            .to(torch.float32)
            .to(DEVICE)
        )
        indices = torch.randint(
            0, 30, (total_q, topk), generator=gen, device=DEVICE
        ).to(torch.int32)

        def pack(k_g_, v_g_, dt):
            return pack_qsa_prefill_kv_nvfp4(
                k=k,
                v=v,
                k_sf=k_sf,
                v_sf=v_sf,
                k_g=k_g_,
                v_g=v_g_,
                req_to_token=req_to_token,
                req_indices=req_indices,
                seq_lens=seq_lens,
                cu_k=cu,
                total_k=total_k,
                output_dtype=dt,
            )

        pk, pv = pack(k_g, v_g, torch.bfloat16)
        # The host-side block-config table reads the CUDA device name; pin the
        # table entry this size selects (total_q=70 -> (64, 4, 2)).
        with mock.patch(
            "sglang.srt.layers.attention.qsa.sparse_attn._get_best_config",
            return_value=(64, 4, 2),
        ):
            out_bf16 = sparse_gqa_fwd_interface_triton_ck(
                q, pk, pv, indices, cu, cu, seq_lens, 0.125, max_q=40
            )
            k8 = nvfp4_fp8_gather_scale(k_g)
            v8 = nvfp4_fp8_gather_scale(v_g)
            pk8, pv8 = pack(k8, v8, torch.float8_e4m3fn)
            self.assertTrue(bool((pk8.float().abs() <= FP8_E4M3_MAX).all()))
            k_d = nvfp4_fp8_descale(k_g)
            v_d = nvfp4_fp8_descale(v_g)
            out_fp8 = (
                sparse_gqa_fwd_interface_triton_ck(
                    q * k_d, pk8, pv8, indices, cu, cu, seq_lens, 0.125, max_q=40
                )
                * v_d
            )
            self.assertIs(out_fp8.dtype, out_bf16.dtype)
            ref = out_bf16.float()
            rel = ((out_fp8.float() - ref).norm() / ref.norm()).item()
            rel_swap = (
                (
                    sparse_gqa_fwd_interface_triton_ck(
                        q * v_d, pk8, pv8, indices, cu, cu, seq_lens, 0.125, max_q=40
                    )
                    * k_d
                ).float()
                - ref
            ).norm().item() / ref.norm().item()
        print(
            f"\nfp8-vs-bf16 attention: rel={rel:.3e} swapped-folds rel={rel_swap:.3e}"
        )
        self.assertLess(rel, 0.1, f"FP8 scratch attention drifted from BF16: {rel}")
        self.assertGreater(rel_swap, 0.25)
        self.assertGreater(rel_swap, 10 * rel)


class TestQSANVFP4Recipe(CustomTestCase):
    def _method(self, num_layers=12):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            get_qsa_kv_cache_quant_method,
        )

        return get_qsa_kv_cache_quant_method(
            "nvfp4", num_layers=num_layers, device="cpu", page_size=64
        )

    def test_buffers_have_linear_scales_and_no_workspace(self):
        method = self._method()
        method.configure_attention_backends("flashinfer", "flashinfer")
        buf = method.create_buffers(128, HEADS, DIM, 3, "cpu")
        self.assertEqual(tuple(buf["k_buffer"][0].shape), (128, HEADS, DIM // 2))
        self.assertEqual(tuple(buf["v_scale_buffer"][2].shape), (128, HEADS, DIM // 16))
        self.assertIsNone(buf["dq_k_buffer"])
        self.assertIsNone(buf["dq_v_buffer"])
        self.assertIsNone(buf.get("native_k_scale_buffer"))

    def test_cell_size_counts_codes_and_scales_only(self):
        # 2 heads x (128 B codes + 16 B scales) x K/V = 576 B per layer vs 1024 B FP8.
        self.assertEqual(self._method().compute_cell_size(HEADS, DIM, 12, 1), 576 * 12)

    def test_plain_reads_fail_loudly(self):
        with self.assertRaises(NotImplementedError):
            self._method().dequantize_kv_tensor(None, None, 0)

    def test_global_scale_rows_cover_model_layer_ids(self):
        method = self._method(num_layers=12)
        method.k_scales_gpu[3] = 2.0
        method.ensure_global_scale_rows(48)
        self.assertEqual(method.k_scales_gpu.numel(), 48)
        self.assertEqual(len(method.v_scales_float), 48)
        self.assertEqual(float(method.k_scales_gpu[3]), 2.0)
        self.assertTrue(torch.all(method.v_scales_gpu[12:] == 1.0))

    def test_rejects_other_fp4_recipes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            get_qsa_kv_cache_quant_method,
        )

        with self.assertRaisesRegex(ValueError, "not supported for Qwen QSA"):
            get_qsa_kv_cache_quant_method(
                "fp4_mx_block16", num_layers=1, device="cpu", page_size=64
            )

    def test_block_format_constants_match_the_quantizer(self):
        # The gather/dequant math hardcodes the NVFP4 block-format constants; the
        # quantization package is the external source of truth for them, and a
        # silent drift would skew every scratch without any local symptom.
        from sglang.srt.layers.attention.qsa.nvfp4_kv import E2M1_MAX, FP8_E4M3_MAX
        from sglang.srt.layers.quantization.kvfp4_tensor import (
            E2M1_MAX as QUANT_E2M1_MAX,
            MAX_BLOCK_SCALE_FP8,
        )

        self.assertEqual(E2M1_MAX, QUANT_E2M1_MAX)
        self.assertEqual(FP8_E4M3_MAX, MAX_BLOCK_SCALE_FP8)


if __name__ == "__main__":
    unittest.main()
