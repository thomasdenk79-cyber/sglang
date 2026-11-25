# SPDX-License-Identifier: Apache-2.0
"""Opt-in, fully replaceable draft vocabulary for the donor's mixed head."""

import logging
from types import SimpleNamespace

import torch

from sglang.srt.speculative.draft_context_vocab import _ContextVocabHeadMethod

logger = logging.getLogger(__name__)


class LRUSlots:
    """Fixed-size map, updated on device. The last slot and token are sinks."""

    def __init__(self, seed, vocab_size, metrics=False):
        self.size = seed.numel()
        self.vocab_size = vocab_size
        self.ids = torch.cat((seed, seed.new_tensor([vocab_size])))
        self.slots = seed.new_full((vocab_size + 1,), -1)
        self.slots[seed] = torch.arange(self.size, device=seed.device)
        self.age = seed.new_zeros(self.size + 1)
        self.age[:self.size] = torch.arange(self.size, 0, -1, device=seed.device)
        self.last = seed.new_zeros(vocab_size + 1)
        self.clock = self.size + 1
        self.origin = torch.ones(self.size + 1, device=seed.device, dtype=torch.bool) if metrics else None
        self.demoted = seed.new_zeros(()) if metrics else None

    @torch.no_grad()
    def observe(self, tokens):
        n = tokens.numel()
        if not 0 < n <= self.size:
            raise ValueError("LRU updates must be bounded by slot count")
        sink = self.vocab_size
        valid = (tokens >= 0) & (tokens < sink)
        ids = torch.where(valid, tokens, sink)
        slot = self.slots[ids]
        clock = torch.arange(self.clock, self.clock + n, device=ids.device)
        self.clock += n
        self.age.scatter_reduce_(0, torch.where(slot >= 0, slot, self.size), clock, reduce="amax")
        self.last.scatter_reduce_(0, ids, clock, reduce="amax")
        new = valid & (slot < 0) & (self.last[ids] == clock)
        order = new.cumsum(0) - 1
        oldest = torch.topk(self.age[:self.size], n, largest=False).indices
        victim = oldest[order.clamp(min=0)]
        admit = new & (clock > self.age[victim])
        dest = torch.where(admit, victim, self.size)
        old_ids = self.ids[dest]
        self.slots.index_fill_(0, old_ids, -1)
        src = torch.where(admit, ids, sink)
        self.slots.index_copy_(0, src, dest.masked_fill(~admit, -1))
        self.ids.index_copy_(0, dest, src)
        self.age.index_copy_(0, dest, clock.masked_fill(~admit, 0))
        if self.origin is not None:
            self.demoted.add_((self.origin[dest] & admit).sum())
            self.origin.index_fill_(0, dest, False)
        return dest, src


class _MixedLRUHead:
    def __init__(self, owner, layer, seed, tail_count):
        from sglang.srt.layers.quantization.nvfp4_dense import NVFP4DenseLinearMethod

        self.source_head = owner.source_head
        self.source_scale = owner.source_scale
        self._torch = torch
        self.capacity = tail_count
        self.prefix_count = seed.numel() - tail_count
        self.lru = LRUSlots(seed, self.source_scale.numel(), owner.metrics)
        self.hot_token_id = self.lru.ids[:-1]
        self.bank_wq, self.bank_scale, global_scale = owner.packed
        self.wq = self._initialize(self.bank_wq, seed[:self.prefix_count])
        self.scales = self._initialize(self.bank_scale, seed[:self.prefix_count])
        self.fp8_bank = self.source_head.t().view(torch.uint8)
        self.fp8 = self._initialize(self.fp8_bank, seed[self.prefix_count:])
        self.fp8_scale = self._initialize(self.source_scale[:, None], seed[self.prefix_count:])
        self.tail_weight = self.fp8[:-1].view(self.source_head.dtype).t()
        self.tail_scale = self.fp8_scale[:-1]

        weight = self.wq[:-1]
        NVFP4DenseLinearMethod.attach(layer, weight, self.scales[:-1], global_scale)
        del layer.weight
        layer.weight = weight
        layer.weight_scale = None
        layer.input_scale = None
        view = SimpleNamespace(owner=self, prefix_count=self.prefix_count, hot_token_id=self.hot_token_id)
        layer.quant_method = _ContextVocabHeadMethod(NVFP4DenseLinearMethod("LRU draft lm_head"), view)
        layer._rnd_ctx_vocab = self

    @staticmethod
    def _initialize(bank, ids):
        return bank[torch.cat((ids, ids.new_zeros(1)))].contiguous()

    def mask_logits(self, logits):
        # All rows are occupied and unique, including the FP8 tail.
        return logits

    def observe(self, ids):
        dest, src = self.lru.observe(ids)
        prefix = dest < self.prefix_count
        tail = (dest >= self.prefix_count) & (dest < self.lru.size)
        packed_dst = dest.masked_fill(~prefix, self.prefix_count)
        tail_dst = (dest - self.prefix_count).masked_fill(~tail, self.capacity)
        packed_src = src.masked_fill(~prefix, 0)
        tail_src = src.masked_fill(~tail, 0)
        self.wq.index_copy_(0, packed_dst, self.bank_wq[packed_src])
        self.scales.index_copy_(0, packed_dst, self.bank_scale[packed_src])
        self.fp8.index_copy_(0, tail_dst, self.fp8_bank[tail_src])
        self.fp8_scale.index_copy_(0, tail_dst, self.source_scale[tail_src, None])


class DonorLRUVocab:
    """Single observer with a separate stable map for each deployed draft head."""

    target_topk = 0

    def __init__(self, source_head, source_scale, hot_ids, model_dir, metrics=False):
        from sglang.srt.layers.quantization.nvfp4_dense import build_or_load
        from sglang.srt.layers.quantization.w8a16_gemv import prealloc

        if hot_ids.numel() != 49152 or source_head.dtype != torch.float8_e4m3fn:
            raise ValueError("donor LRU requires 49152 seed IDs and the per-row FP8 target head")
        self.seed = hot_ids
        self.hot_ids = hot_ids[:-256]
        self.source_head = source_head
        self.source_scale = source_scale
        self.metrics = metrics
        self.packed = build_or_load(model_dir, "lm_head.weight", device=hot_ids.device, label="LRU draft source bank")
        self.heads = []
        self._verify_updates = 0
        prealloc(hot_ids.device)

    def attach_head(self, layer, token_ids):
        if not self.heads:
            seed = self.seed
        else:
            # Keep the donor recurrent width and an FP8 tail, initially seeded
            # from distinct IDs in the first-head map.
            import itertools

            recurrent = set(token_ids.tolist())
            primary = self.seed.flip(0).tolist()
            tail = list(itertools.islice((i for i in primary if i not in recurrent), 256))
            if len(tail) != 256:
                raise ValueError("not enough distinct primary IDs for recurrent LRU tail")
            seed = torch.cat((token_ids, token_ids.new_tensor(tail)))
        if seed.unique().numel() != seed.numel():
            raise ValueError("LRU seed must contain unique token IDs")
        state = _MixedLRUHead(self, layer, seed, 256)
        self.heads.append(state)
        logger.info("Donor draft vocabulary (lru): %d replaceable rows, 0 pinned (%d NVFP4 + 256 FP8)", seed.numel(), state.prefix_count)
        return state.hot_token_id

    def _observe(self, ids):
        for chunk in ids.split(4096):
            if chunk.numel():
                for head in self.heads:
                    head.observe(chunk)

    def update_prefill(self, token_ids):
        self._observe(torch.as_tensor(token_ids, device=self.seed.device, dtype=torch.int64).reshape(-1))

    def update_committed(self, ids, accept_lens=None, width=1):
        ids = ids.to(torch.int64).reshape(-1)
        if accept_lens is not None:
            valid = torch.arange(width, device=ids.device)[None, :] < accept_lens.reshape(-1, 1)
            ids = ids.masked_fill(~valid.reshape(-1), -1)
        self._observe(ids)
        if self.metrics:
            self._verify_updates += 1
            if self._verify_updates % 32 == 0:
                counts = [h.lru.demoted.item() for h in self.heads]
                logger.info("Donor LRU seed demotions: %s", counts)
                if all(counts):
                    for head in self.heads:
                        head.lru.origin = None
                    self.metrics = False
                    logger.info("Donor LRU validation counters disabled; policy remains active")
