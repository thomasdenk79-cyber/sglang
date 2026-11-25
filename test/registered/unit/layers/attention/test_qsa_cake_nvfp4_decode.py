"""QSA NVFP4 decode via FlashInfer Cake vs the donor's PyTorch dequant reference.

Needs CUDA SM120 and a standalone Cake build (``SGLANG_QSA_CAKE_PATH``, see
``scripts/cake_qsa/build_cake_qsa_sm120.py``); skipped otherwise. Covers the
donor pool layout in place (page_size 1, req_to_token as block table), -1 holes,
indices past seq_len, MTP-verify style multi-row requests, non-unit global
scales, and a CUDA-graph replay with a caller workspace.
"""

import math
import unittest

import torch

from sglang.srt.layers.attention.qsa import cake_decode
from sglang.srt.layers.attention.qsa.nvfp4_kv import reference_dequant

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
HQ, HKV, DIM = 16, 2, 256


def quantize_nvfp4(x, g):
    m, h, d = x.shape
    blocks = x.float().view(m, h, d // 16, 16) / g
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn)
    denom = scale.float()
    scaled = torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks))
    idx = (scaled.abs().unsqueeze(-1) - E2M1.to(x.device)).abs().argmin(-1)
    codes = (idx | ((scaled < 0).long() << 3)).view(m, h, d).to(torch.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale.view(m, h, d // 16).view(torch.uint8).contiguous()


def reference(q, k, v, req_to_token, row_req, topk, seq_lens):
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    group = HQ // HKV
    for r in range(q.shape[0]):
        pos = topk[r].long()
        pos = pos[(pos >= 0) & (pos < int(seq_lens[r]))]
        if pos.numel() == 0:
            continue
        slots = req_to_token[int(row_req[r]), pos].long()
        kr = k[slots].repeat_interleave(group, dim=1)  # [n, hq, d]
        vr = v[slots].repeat_interleave(group, dim=1)
        s = torch.einsum("hd,nhd->hn", q[r].float(), kr) / math.sqrt(DIM)
        p = s.softmax(-1)
        out[r] = torch.einsum("hn,nhd->hd", p, vr)
    return out


def _available():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (12, 0):
        return False
    return cake_decode.load_cake() is not None


@unittest.skipUnless(_available(), "needs SM120 and SGLANG_QSA_CAKE_PATH build")
class TestQSACakeNVFP4Decode(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        dev = "cuda"
        self.slots, self.reqs, self.ctx = 16384, 3, 4000
        self.k_g, self.v_g = 0.75, 1.5
        x = torch.randn(self.slots, HKV, DIM, device=dev)
        y = torch.randn(self.slots, HKV, DIM, device=dev)
        self.k_raw, self.k_sf = quantize_nvfp4(x, self.k_g)
        self.v_raw, self.v_sf = quantize_nvfp4(y, self.v_g)
        g = lambda s: torch.tensor([s], device=dev)
        self.k = reference_dequant(self.k_raw, self.k_sf, g(self.k_g))
        self.v = reference_dequant(self.v_raw, self.v_sf, g(self.v_g))
        perm = torch.randperm(self.slots - 1, device=dev)[: self.reqs * self.ctx] + 1
        self.req_to_token = torch.zeros(self.reqs + 1, 4096, dtype=torch.int32, device=dev)
        self.req_to_token[1:, : self.ctx] = perm.view(self.reqs, self.ctx).int()

    def make_batch(self, rows_per_req, topk):
        dev = "cuda"
        row_req, seq_lens = [], []
        for i in range(self.reqs):
            base = 2500 + 100 * i
            for j in range(rows_per_req):  # MTP verify: consecutive positions
                row_req.append(i + 1)
                seq_lens.append(base + j + 1)
        rows = len(row_req)
        row_req = torch.tensor(row_req, dtype=torch.int32, device=dev)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        idx = torch.stack(
            [torch.randperm(int(seq_lens[r]) + 200, device=dev)[:topk] for r in range(rows)]
        ).int()  # some positions >= seq_len (must be masked)
        idx[:, -5:] = -1  # holes
        idx[0, 3] = -1
        q = torch.randn(rows, HQ, DIM, device=dev, dtype=torch.bfloat16)
        return q, row_req, seq_lens, idx

    def run_cake(self, q, row_req, seq_lens, idx, workspace=None):
        rows, topk = idx.shape
        sel_buf = torch.empty(rows, topk + 1, dtype=torch.int32, device=q.device)
        gate = torch.full_like(q, cake_decode.GATE_OPEN)
        sel = cake_decode.prepare_selection(idx, seq_lens, sel_buf)
        return cake_decode.cake_nvfp4_decode(
            q, self.k_raw, self.v_raw, self.k_sf, self.v_sf, self.k_g, self.v_g,
            self.req_to_token, row_req, sel, gate, workspace=workspace,
        )

    def check(self, rows_per_req, topk):
        q, row_req, seq_lens, idx = self.make_batch(rows_per_req, topk)
        out = self.run_cake(q, row_req, seq_lens, idx)
        ref = reference(q, self.k, self.v, self.req_to_token, row_req, idx, seq_lens)
        err = (out.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        print(f"rows={q.shape[0]} topk={topk} max_abs_err={err:.3e} ref_max={scale:.3e}")
        self.assertLessEqual(err, 2**-6 * max(scale, 1.0))

    def test_decode(self):
        self.check(1, 2048)

    def test_decode_small_topk(self):
        self.check(1, 64)

    def test_mtp_verify_rows(self):
        self.check(4, 2048)

    def test_cuda_graph(self):
        q, row_req, seq_lens, idx = self.make_batch(4, 2048)
        rows, topk = idx.shape
        ws = torch.zeros(
            cake_decode.workspace_bytes(rows, HQ, HKV, topk), dtype=torch.uint8, device="cuda"
        )
        sel_buf = torch.empty(rows, topk + 1, dtype=torch.int32, device="cuda")
        gate = torch.full_like(q, cake_decode.GATE_OPEN)
        out = torch.empty_like(q)

        def step():
            sel = cake_decode.prepare_selection(idx, seq_lens, sel_buf)
            cake_decode.cake_nvfp4_decode(
                q, self.k_raw, self.v_raw, self.k_sf, self.v_sf, self.k_g, self.v_g,
                self.req_to_token, row_req, sel, gate, out=out, workspace=ws,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        eager = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, eager))
        # replay picks up new inputs in place
        q.copy_(torch.randn_like(q))
        graph.replay()
        ref = reference(q, self.k, self.v, self.req_to_token, row_req, idx, seq_lens)
        err = (out.float() - ref).abs().max().item()
        self.assertLessEqual(err, 2**-6 * max(ref.abs().max().item(), 1.0))

    def test_backend_method(self):
        """The backend's flag-gated entry on a stand-in backend object."""
        from types import SimpleNamespace

        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseAttnBackend,
        )

        q, row_req, seq_lens, idx = self.make_batch(2, 2048)
        fake = SimpleNamespace(
            _cake_scales={},
            _cake_buffers=None,
            _cuda_graph_max_tokens=8,
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
        )
        metadata = SimpleNamespace(row_req_pool_indices=row_req.long(), sequence_lengths=seq_lens)
        g = lambda s: torch.tensor([s], device="cuda")
        out = QwenSparseAttnBackend._forward_nvfp4_cake_decode(
            fake, q, SimpleNamespace(layer_id=0, scaling=1 / 16), None, metadata, idx,
            self.k_raw, self.v_raw, (self.k_sf, self.v_sf, g(self.k_g), g(self.v_g)),
        )
        ref = reference(q, self.k, self.v, self.req_to_token, row_req, idx, seq_lens)
        err = (out.view_as(ref).float() - ref).abs().max().item()
        self.assertLessEqual(err, 2**-6 * max(ref.abs().max().item(), 1.0))


if __name__ == "__main__":
    unittest.main()
