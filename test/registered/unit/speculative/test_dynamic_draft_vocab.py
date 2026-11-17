import unittest

import torch

from sglang.srt.speculative.dynamic_draft_vocab import DynamicDraftVocab
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 64
HIDDEN = 8


def _make(hot, slots, **kwargs):
    head = torch.randn(VOCAB + 4, HIDDEN)  # padded like ParallelLMHead
    dv = DynamicDraftVocab(torch.tensor(hot), head, slots, VOCAB, **kwargs)
    return dv, head


class TestDynamicDraftVocab(CustomTestCase):
    def assert_consistent(self, dv, head, hot):
        ids = dv.hot_token_id
        self.assertEqual(ids.numel(), len(hot) + dv.num_slots)
        self.assertEqual(torch.unique(ids).numel(), ids.numel())
        self.assertEqual(ids[: len(hot)].tolist(), hot)
        torch.testing.assert_close(dv.head, head[ids], rtol=0, atol=0)
        ring = ids[len(hot) :]
        self.assertEqual(
            dv.slot_of[ring].tolist(), list(range(dv.num_slots)), "ring/slot_of"
        )
        self.assertTrue(bool((dv.slot_of[torch.tensor(hot)] == -3).all()))
        in_map = torch.zeros(VOCAB, dtype=torch.bool)
        in_map[ids] = True
        self.assertTrue(bool((dv.slot_of[:VOCAB][~in_map] == -1).all()))

    def test_prefill_is_unique_and_matches_target_rows(self):
        hot = [0, 1, 2, 3, 10, 11, 12, 13]
        dv, head = _make(hot, 4)
        self.assert_consistent(dv, head, hot)

    def test_observe_inserts_unseen_tokens(self):
        hot = [0, 1, 2, 3, 10, 11, 12, 13]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([40, 0, 41, 40, 11]))
        self.assert_consistent(dv, head, hot)
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertTrue({40, 41} <= ring)

    def test_invalid_and_static_ids_are_ignored(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        before = dv.hot_token_id.clone()
        dv.observe(torch.tensor([-1, VOCAB, VOCAB + 3, 0, 3]))
        self.assertTrue(torch.equal(dv.hot_token_id, before))
        self.assert_consistent(dv, head, hot)

    def test_fifo_eviction_keeps_recent_tokens(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        for step in [[40, 41], [42, 43], [44, 45]]:
            dv.observe(torch.tensor(step))
            self.assert_consistent(dv, head, hot)
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {42, 43, 44, 45})

    def test_long_observe_keeps_most_recent_new_tokens(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([30, 31, 32, 33, 34, 35, 36, 0, 36]))
        self.assert_consistent(dv, head, hot)
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {33, 34, 35, 36})

    def test_fifo_evicts_oldest_within_a_batch(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([40, 41, 42, 43]))
        dv.observe(torch.tensor([44]))
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {41, 42, 43, 44})
        dv.observe(torch.tensor([45]))
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {42, 43, 44, 45})
        self.assert_consistent(dv, head, hot)

    def test_duplicates_count_at_their_last_occurrence(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 2)
        # 50 is the oldest token by first occurrence but recent by its last.
        dv.observe(torch.tensor([50, 51, 52, 50]))
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {52, 50})
        dv.observe(torch.tensor([53]))
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {50, 53})
        self.assert_consistent(dv, head, hot)

    def test_stage_defers_until_commit_and_honors_mask(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        before = dv.hot_token_id.clone()
        buf = torch.tensor([[40, 41, 60], [42, 61, 62]])
        dv.stage(buf, torch.tensor([[True, True, False], [True, False, False]]))
        dv.stage(torch.tensor([43]))
        buf.fill_(0)  # the caller may reuse its buffer before the commit
        self.assertTrue(torch.equal(dv.hot_token_id, before))
        dv.commit()
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertEqual(ring, {40, 41, 42, 43})
        dv.commit()  # nothing staged
        self.assertEqual(set(dv.hot_token_id[len(hot) :].tolist()), ring)
        self.assert_consistent(dv, head, hot)

    def test_max_observe_bounds_the_tail(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4, max_observe=2)
        dv.stage(torch.tensor([50]))
        dv.stage(torch.tensor([51, 52]))
        dv.commit()
        ring = set(dv.hot_token_id[len(hot) :].tolist())
        self.assertIn(51, ring)
        self.assertIn(52, ring)
        self.assertNotIn(50, ring)
        self.assert_consistent(dv, head, hot)

    def test_rejects_invalid_configs(self):
        head = torch.randn(VOCAB, HIDDEN)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 1]), head[:32], 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 0]), head, 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, VOCAB]), head, 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 1]), head, VOCAB - 1, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 1]), head, 0, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 1]), head.to(torch.int8), 4, VOCAB)


if __name__ == "__main__":
    unittest.main()
