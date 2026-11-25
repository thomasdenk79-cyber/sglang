"""CPU oracle and GPU stable-buffer check for the donor LRU adapter."""

import collections
import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.draft_lru_vocab import LRUSlots, _MixedLRUHead


class DonorLRUTest(unittest.TestCase):
    def test_batch_matches_sequential(self):
        rng = random.Random(12)
        seed = list(range(32))
        device_map = LRUSlots(torch.tensor(seed), 200, metrics=True)
        oracle = collections.OrderedDict.fromkeys(reversed(seed))
        for _ in range(512):
            batch = [rng.randrange(-1, 202) for _ in range(rng.randint(1, 32))]
            device_map.observe(torch.tensor(batch))
            for token in batch:
                if 0 <= token < 200:
                    oracle.pop(token, None)
                    oracle[token] = None
                    if len(oracle) > len(seed):
                        oracle.popitem(last=False)
            ids = device_map.ids[:-1]
            self.assertEqual(set(ids.tolist()), set(oracle))
            self.assertEqual(device_map.slots[ids].tolist(), list(range(len(seed))))
        self.assertEqual(device_map.demoted.item(), len(seed))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph update test")
    def test_mixed_rows_replay_without_rebinding(self):
        device = "cuda"
        vocab, hidden = 1024, 256
        values = torch.arange(vocab, device=device)
        bank = (values[:, None] + torch.arange(hidden, device=device))
        owner = SimpleNamespace(
            source_head=(bank % 23).to(torch.float8_e4m3fn).t(),
            source_scale=values.float().add(1),
            packed=(
                bank[:, :hidden // 2].to(torch.uint8).contiguous(),
                bank[:, :hidden // 16].to(torch.uint8).contiguous(),
                torch.ones(1, device=device),
            ),
            metrics=True,
        )
        layer = torch.nn.Linear(hidden, 512, bias=False, device=device)
        state = _MixedLRUHead(owner, layer, values[:512], 256)
        buffers = (state.hot_token_id, layer.nvfp4_wq, layer.nvfp4_bs, state.tail_weight, state.tail_scale)
        pointers = [x.data_ptr() for x in buffers]
        graph = torch.cuda.CUDAGraph()
        snapshots = [torch.empty_like(x) for x in buffers]
        with torch.cuda.graph(graph):
            for out, src in zip(snapshots, buffers):
                out.copy_(src)
        state.observe(values[512:])
        graph.replay()
        ids, packed, scales, tail, tail_scales = snapshots
        self.assertEqual(set(ids.tolist()), set(range(512, 1024)))
        self.assertEqual(state.lru.demoted.item(), 512)
        self.assertEqual([x.data_ptr() for x in buffers], pointers)
        torch.testing.assert_close(packed, owner.packed[0][ids[:256]])
        torch.testing.assert_close(scales, owner.packed[1][ids[:256]])
        torch.testing.assert_close(tail.t().view(torch.uint8), state.fp8_bank[ids[256:]])
        torch.testing.assert_close(tail_scales[:, 0], owner.source_scale[ids[256:]])


if __name__ == "__main__":
    unittest.main()
