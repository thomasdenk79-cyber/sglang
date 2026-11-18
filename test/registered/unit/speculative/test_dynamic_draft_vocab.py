import os
import random
import tempfile
import unittest

import torch

from sglang.srt.speculative.dynamic_draft_vocab import (
    DynamicDraftVocab,
    DynamicVocabConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 64
HIDDEN = 8


def _make(hot, slots, **kwargs):
    head = torch.randn(VOCAB + 4, HIDDEN)  # padded like ParallelLMHead
    dv = DynamicDraftVocab(torch.tensor(hot), head, slots, VOCAB, **kwargs)
    return dv, head


def _cfg(**kwargs):
    return DynamicVocabConfig(**kwargs)


def _ring(dv):
    return set(dv.hot_token_id[dv.num_hot : dv.num_hot + dv.num_ring].tolist())


def _pool(dv):
    return set(dv.hot_token_id[dv.num_hot + dv.num_ring :].tolist())


class _ReferenceFifo:
    """The FIFO ring before policies were added, for bit-identity checks."""

    def __init__(self, hot, head, num_slots, vocab_size):
        hot = torch.tensor(hot)
        self.vocab_size, self.num_hot, self.num_slots = vocab_size, len(hot), num_slots
        slot_of = torch.full((vocab_size + 1,), -1, dtype=torch.int64)
        slot_of[hot] = -3
        slot_of[vocab_size] = -3
        fill = torch.nonzero(slot_of[:vocab_size] == -1).flatten()[:num_slots]
        slot_of[fill] = torch.arange(num_slots)
        self.slot_of = slot_of
        ids = torch.empty(self.num_hot + num_slots + 1, dtype=torch.int64)
        ids[: self.num_hot] = hot
        ids[self.num_hot : -1] = fill
        ids[-1] = ids[0]
        self._ids, self._head = ids, head
        self._rows = head[ids]
        self._cursor = torch.zeros(1, dtype=torch.int64)
        self._last_seen = torch.full((vocab_size + 1,), -1, dtype=torch.int64)
        self._stamp = 0

    def observe(self, ids):
        n, sink = ids.numel(), self.vocab_size
        tok = torch.where((ids >= 0) & (ids < sink), ids, sink)
        new = self.slot_of[tok] == -1
        stamp = torch.arange(self._stamp, self._stamp + n)
        self._stamp += n
        self._last_seen.scatter_reduce_(
            0, torch.where(new, tok, sink), stamp, reduce="amax"
        )
        win = new & (self._last_seen[tok] == stamp)
        order = torch.cumsum(win, 0) - 1
        drop = (win.sum() - self.num_slots).clamp(min=0)
        win = win & (order >= drop)
        pos = torch.where(
            win, (self._cursor + order - drop) % self.num_slots, self.num_slots
        )
        row = self.num_hot + pos
        prev = self._ids[row]
        self.slot_of.index_put_(
            (torch.where(win, prev, sink),), torch.where(win, -1, -3)
        )
        self.slot_of.index_put_(
            (torch.where(win, tok, sink),), torch.where(win, pos, -3)
        )
        src = torch.where(win, tok, self._ids[-1])
        self._ids.index_copy_(0, row, src)
        self._rows.index_copy_(0, row, self._head[src])
        self._cursor.copy_((self._cursor + win.sum()) % self.num_slots)


class TestDynamicDraftVocab(CustomTestCase):
    def assert_consistent(self, dv, head, hot=None):
        ids = dv.hot_token_id
        self.assertEqual(ids.numel(), dv.num_hot + dv.num_slots)
        self.assertEqual(torch.unique(ids).numel(), ids.numel())
        if hot is not None:
            self.assertEqual(ids[: dv.num_hot].tolist(), hot[: dv.num_hot])
        torch.testing.assert_close(dv.head, head[ids], rtol=0, atol=0)
        dyn = ids[dv.num_hot :]
        self.assertEqual(
            dv.slot_of[dyn].tolist(), list(range(dv.num_slots)), "slots/slot_of"
        )
        self.assertTrue(bool((dv.slot_of[ids[: dv.num_hot]] == -3).all()))
        in_map = torch.zeros(VOCAB, dtype=torch.bool)
        in_map[ids] = True
        self.assertTrue(bool((dv.slot_of[:VOCAB][~in_map] == -1).all()))
        self.assertEqual(int(dv.slot_of[VOCAB]), -3)

    def test_prefill_is_unique_and_matches_target_rows(self):
        hot = [0, 1, 2, 3, 10, 11, 12, 13]
        dv, head = _make(hot, 4)
        self.assert_consistent(dv, head, hot)

    def test_observe_inserts_unseen_tokens(self):
        hot = [0, 1, 2, 3, 10, 11, 12, 13]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([40, 0, 41, 40, 11]))
        self.assert_consistent(dv, head, hot)
        self.assertTrue({40, 41} <= _ring(dv))

    def test_invalid_and_static_ids_are_ignored(self):
        hot = [0, 1, 2, 3]
        for policy in ("fifo", "lru", "lfu"):
            dv, head = _make(hot, 4, config=_cfg(policy=policy))
            before = dv.hot_token_id.clone()
            dv.observe(torch.tensor([-1, VOCAB, VOCAB + 3, 0, 3]), output=True)
            self.assertTrue(torch.equal(dv.hot_token_id, before), policy)
            self.assert_consistent(dv, head, hot)

    def test_fifo_eviction_keeps_recent_tokens(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        for step in [[40, 41], [42, 43], [44, 45]]:
            dv.observe(torch.tensor(step))
            self.assert_consistent(dv, head, hot)
        self.assertEqual(_ring(dv), {42, 43, 44, 45})

    def test_long_observe_keeps_most_recent_new_tokens(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([30, 31, 32, 33, 34, 35, 36, 0, 36]))
        self.assert_consistent(dv, head, hot)
        self.assertEqual(_ring(dv), {33, 34, 35, 36})

    def test_fifo_evicts_oldest_within_a_batch(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4)
        dv.observe(torch.tensor([40, 41, 42, 43]))
        dv.observe(torch.tensor([44]))
        self.assertEqual(_ring(dv), {41, 42, 43, 44})
        dv.observe(torch.tensor([45]))
        self.assertEqual(_ring(dv), {42, 43, 44, 45})
        self.assert_consistent(dv, head, hot)

    def test_duplicates_count_at_their_last_occurrence(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 2)
        # 50 is the oldest token by first occurrence but recent by its last.
        dv.observe(torch.tensor([50, 51, 52, 50]))
        self.assertEqual(_ring(dv), {52, 50})
        dv.observe(torch.tensor([53]))
        self.assertEqual(_ring(dv), {50, 53})
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
        ring = _ring(dv)
        self.assertEqual(ring, {40, 41, 42, 43})
        dv.commit()  # nothing staged
        self.assertEqual(_ring(dv), ring)
        self.assert_consistent(dv, head, hot)

    def test_max_observe_bounds_the_tail(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4, max_observe=2)
        dv.stage(torch.tensor([50]))
        dv.stage(torch.tensor([51, 52]))
        dv.commit()
        ring = _ring(dv)
        self.assertIn(51, ring)
        self.assertIn(52, ring)
        self.assertNotIn(50, ring)
        self.assert_consistent(dv, head, hot)

    def test_rejects_invalid_configs(self):
        head = torch.randn(VOCAB, HIDDEN)
        hot = torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            DynamicDraftVocab(hot, head[:32], 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, 0]), head, 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(torch.tensor([0, VOCAB]), head, 4, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(hot, head, VOCAB - 1, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(hot, head, 0, VOCAB)
        with self.assertRaises(ValueError):
            DynamicDraftVocab(hot, head.to(torch.int8), 4, VOCAB)
        for cfg in (
            _cfg(policy="mru"),
            _cfg(policy="tiered"),
            _cfg(policy="fifo", adaptive_static=True),
            _cfg(policy="fifo", ttl_s=5.0),
            _cfg(policy="tiered", proven_slots=4, proven_score="x"),
            _cfg(policy="lru", commit_every=0),
        ):
            with self.assertRaises(ValueError, msg=str(cfg)):
                DynamicDraftVocab(hot, head, 4, VOCAB, config=cfg)

    # ------------------------------------------------------------- policies

    def test_fifo_is_bit_identical_to_reference(self):
        rng = random.Random(0)
        hot = [0, 1, 2, 3, 7, 9, 20, 33]
        head = torch.randn(VOCAB + 4, HIDDEN)
        dv = DynamicDraftVocab(torch.tensor(hot), head, 6, VOCAB)
        ref = _ReferenceFifo(hot, head, 6, VOCAB)
        for _ in range(200):
            batch = torch.tensor(
                [rng.randrange(-2, VOCAB + 2) for _ in range(rng.randrange(1, 7))]
            )
            dv.observe(batch, output=rng.random() < 0.5)
            ref.observe(batch)
            self.assertTrue(torch.equal(dv._ids, ref._ids))
            self.assertTrue(torch.equal(dv.slot_of, ref.slot_of))
            self.assertTrue(torch.equal(dv._rows, ref._rows))
            self.assertTrue(torch.equal(dv._cursor, ref._cursor))

    def test_buffers_are_updated_in_place(self):
        hot = [0, 1, 2, 3]
        configs = [
            _cfg(),
            _cfg(policy="lru"),
            _cfg(policy="lfu"),
            _cfg(policy="tiered", proven_slots=3),
            _cfg(policy="lru", adaptive_static=True, static_core=1, ttl_s=1.0),
        ]
        rng = random.Random(1)
        for cfg in configs:
            dv, head = _make(hot, 5, config=cfg)
            ptrs = (
                dv.head.data_ptr(),
                dv.hot_token_id.data_ptr(),
                dv.slot_of.data_ptr(),
            )
            shapes = (dv.head.shape, dv.hot_token_id.shape)
            for _ in range(100):
                n = rng.choice([1, 3, 8, 40])
                batch = torch.tensor([rng.randrange(VOCAB) for _ in range(n)])
                dv.stage(batch, output=rng.random() < 0.6)
                dv.commit()
                self.assert_consistent(dv, head, hot)
            self.assertEqual(
                ptrs,
                (dv.head.data_ptr(), dv.hot_token_id.data_ptr(), dv.slot_of.data_ptr()),
                cfg.policy,
            )
            self.assertEqual(shapes, (dv.head.shape, dv.hot_token_id.shape))

    def test_lru_reuse_refreshes_recency(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 4, config=_cfg(policy="lru"))
        dv.observe(torch.tensor([40, 41, 42, 43]))
        dv.observe(torch.tensor([40]))  # fifo would evict 40 next
        dv.observe(torch.tensor([44]))
        self.assertEqual(_pool(dv), {40, 42, 43, 44})
        dv.observe(torch.tensor([45, 46]))
        self.assertEqual(_pool(dv), {40, 44, 45, 46})
        self.assert_consistent(dv, head, hot)

    def test_lru_overflowing_batch_keeps_last_tokens(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 3, config=_cfg(policy="lru"))
        dv.observe(torch.tensor([30, 31, 32, 33, 34, 30]))
        self.assertEqual(_pool(dv), {33, 34, 30})
        self.assert_consistent(dv, head, hot)

    def test_lfu_keeps_frequent_tokens(self):
        hot = [0, 1, 2, 3]
        cfg = _cfg(policy="lfu", half_life=1e6, output_weight=4.0)
        dv, head = _make(hot, 3, config=cfg)
        dv.observe(torch.tensor([40, 40, 40, 41, 42]))
        dv.observe(torch.tensor([43]))  # evicts 41 (count 1, oldest)
        self.assertEqual(_pool(dv), {40, 42, 43})
        dv.observe(torch.tensor([42]), output=True)  # 42 -> weight 5
        # 44 takes 43's slot; 45 (count 1) does not beat 40 (count 3).
        dv.observe(torch.tensor([44, 45]))
        self.assertEqual(_pool(dv), {40, 42, 44})
        self.assert_consistent(dv, head, hot)

    def test_lfu_decay_lets_new_tokens_win(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 2, config=_cfg(policy="lfu", half_life=4.0))
        dv.observe(torch.tensor([40, 40, 40, 41]))
        dv.observe(torch.tensor([60] * 64))  # 64 tokens = 16 half-lives later
        dv.observe(torch.tensor([42]))
        self.assertEqual(_pool(dv), {60, 42})
        self.assert_consistent(dv, head, hot)

    def test_lfu_rescale_keeps_order(self):
        hot = [0, 1]
        dv, head = _make(hot, 2, config=_cfg(policy="lfu", half_life=1.0))
        dv.observe(torch.tensor([40, 40, 41]))
        for _ in range(5):  # exponent passes the rescale limit
            dv.observe(torch.tensor([VOCAB] * 300))
        self.assertTrue(bool(torch.isfinite(dv._score).all()))
        dv.observe(torch.tensor([40]))
        dv.observe(torch.tensor([42]))
        self.assertEqual(_pool(dv), {40, 42})
        self.assert_consistent(dv, head, hot)

    def test_tiered_promotes_output_tokens(self):
        hot = [0, 1, 2, 3]
        cfg = _cfg(policy="tiered", proven_slots=2, metrics=True)
        dv, head = _make(hot, 3, config=cfg)
        dv.observe(torch.tensor([40, 41, 42]))  # prompt -> ring only
        self.assertEqual(_ring(dv), {40, 41, 42})
        self.assertFalse(_pool(dv) & {40, 41, 42})
        dv.observe(torch.tensor([41]), output=True)  # promoted, swaps a filler in
        self.assertIn(41, _pool(dv))
        self.assertNotIn(41, _ring(dv))
        dv.observe(torch.tensor([50]), output=True)  # absent output -> pool
        self.assertEqual(_pool(dv), {41, 50})
        dv.observe(torch.tensor([41]), output=True)  # refresh 41
        dv.observe(torch.tensor([40]), output=True)  # 50 is LRU -> into ring
        self.assertEqual(_pool(dv), {41, 40})
        self.assertIn(50, _ring(dv))
        c = dv.counters()
        self.assertEqual(c["promote"], 2)
        self.assertEqual(c["out"], 4)
        self.assertEqual(c["hit_ring"], 2)
        self.assertEqual(c["hit_pool"], 1)
        self.assertEqual(c["miss"], 1)
        self.assert_consistent(dv, head, hot)

    def test_tiered_prompt_tokens_do_not_refresh_pool(self):
        hot = [0, 1]
        dv, head = _make(hot, 2, config=_cfg(policy="tiered", proven_slots=2))
        dv.observe(torch.tensor([40, 41]), output=True)
        dv.observe(torch.tensor([40]))  # prompt use: no refresh
        dv.observe(torch.tensor([42]), output=True)
        self.assertEqual(_pool(dv), {41, 42})
        self.assert_consistent(dv, head, hot)

    def test_adaptive_static_demotes_unused_and_protects_core(self):
        hot = [0, 1, 2, 3, 4, 5]
        cfg = _cfg(
            policy="lru", adaptive_static=True, static_core=2, static_horizon=20.0
        )
        dv, head = _make(hot, 2, config=cfg)
        self.assertEqual(dv.num_hot, 2)
        self.assertEqual(dv.num_pool, 6)
        self.assertTrue({2, 3, 4, 5} <= _pool(dv))
        dv.observe(torch.tensor([40, 41, 42, 43]))  # within horizon: statics stay
        self.assertTrue({2, 3, 4, 5} <= _pool(dv))
        self.assertEqual(len(_pool(dv) & {40, 41, 42, 43}), 2)
        dv.observe(torch.tensor([VOCAB] * 30))  # past the horizon
        dv.observe(torch.tensor([3]))
        dv.observe(torch.tensor([50, 51, 52, 53, 54]))
        pool = _pool(dv)
        self.assertIn(3, pool)
        self.assertEqual(len(pool & {2, 4, 5}), 0)
        self.assertEqual(dv.hot_token_id[:2].tolist(), [0, 1])
        self.assert_consistent(dv, head, hot)

    def test_fully_adaptive_core_zero(self):
        hot = [0, 1, 2, 3]
        cfg = _cfg(
            policy="lfu",
            adaptive_static=True,
            static_core=0,
            static_horizon=4.0,
            half_life=2.0,
        )
        dv, head = _make(hot, 1, config=cfg)
        self.assertEqual(dv.num_hot, 0)
        self.assertEqual(_pool(dv), {0, 1, 2, 3, 4})
        for _ in range(10):
            dv.observe(torch.tensor([40, 41, 42, 43, 44]), output=True)
        self.assertEqual(_pool(dv), {40, 41, 42, 43, 44})
        self.assert_consistent(dv, head)

    def test_static_rank_picks_the_core(self):
        hot = [0, 1, 2, 3, 4, 5]
        with tempfile.TemporaryDirectory() as d:
            rank_path = os.path.join(d, "rank.pt")
            torch.save([5, 60, 3], rank_path)
            cfg = _cfg(
                policy="lru", adaptive_static=True, static_core=2, static_rank=rank_path
            )
            dv, head = _make(hot, 1, config=cfg)
        self.assertEqual(dv.hot_token_id[:2].tolist(), [5, 3])
        self.assertTrue({0, 1, 2, 4} <= _pool(dv))
        self.assert_consistent(dv, head)

    def test_export_ranked_map_puts_learned_tokens_first(self):
        from sglang.srt.speculative.dynamic_draft_vocab import main

        hot = [0, 1, 2, 3]
        with tempfile.TemporaryDirectory() as d:
            dv, _ = _make(hot, 2, config=_cfg(policy="lru", state_dir=d))
            dv.observe(torch.tensor([40, 41, 41]), output=True)
            dv.save_state()
            map_path, out = os.path.join(d, "map.pt"), os.path.join(d, "out.pt")
            torch.save(hot, map_path)
            main(
                [
                    "--token-map",
                    map_path,
                    "--k",
                    "3",
                    "--out",
                    out,
                    "--state",
                    cfg_path(d),
                ]
            )
            self.assertEqual(torch.load(out), [41, 40, 0])

    def test_ttl_expired_tokens_go_first_with_floor(self):
        hot = [0, 1]
        clock = [0.0]
        cfg = _cfg(policy="lfu", ttl_s=10.0, min_rows=4, half_life=1e9, metrics=True)
        dv, head = _make(hot, 4, config=cfg)
        dv._now = lambda: clock[0]
        dv.observe(torch.tensor([40, 40, 40, 41, 41, 41]), output=True)
        clock[0] = 5.0
        dv.observe(torch.tensor([42, 43]), output=True)
        clock[0] = 14.0  # 40, 41 expired; floor 4 rows = 2 static + 2 live
        dv.observe(torch.tensor([42]), output=True)
        dv.observe(torch.tensor([44]))
        # 44 (count 1) replaces the expired 40/41 despite their higher counts.
        pool = _pool(dv)
        self.assertIn(44, pool)
        self.assertEqual(len(pool & {40, 41}), 1)
        self.assertTrue({42, 43} <= pool)
        self.assert_consistent(dv, head, hot)

        # A floor at the full size disables expiry.
        cfg.min_rows = dv.num_rows
        clock[0] = 100.0
        dv.observe(torch.tensor([45]))
        self.assertEqual(len(_pool(dv) & {42, 43, 44}), 2)  # lfu victim, not ttl
        self.assert_consistent(dv, head, hot)

    def test_commit_every_batches_updates(self):
        hot = [0, 1]
        dv, head = _make(hot, 4, config=_cfg(policy="lru", commit_every=2))
        dv.stage(torch.tensor([40]))
        dv.commit()
        self.assertNotIn(40, _pool(dv))
        dv.stage(torch.tensor([41]))
        dv.commit()
        self.assertTrue({40, 41} <= _pool(dv))
        self.assert_consistent(dv, head, hot)

    def test_metrics_coverage_and_log_suffix(self):
        hot = [0, 1, 2, 3]
        dv, head = _make(hot, 2, config=_cfg(metrics=True))
        self.assertEqual(dv.log_suffix(), "")  # first call starts the snapshot
        dv.stage(torch.tensor([40, 41]))  # prompt: not counted
        dv.stage(
            torch.tensor([[0, 40, 50]]), torch.tensor([[True, True, True]]), output=True
        )
        dv.commit()
        c = dv.counters()
        # 40 was staged in the same commit, so it was not yet in the vocab.
        self.assertEqual((c["out"], c["hit_static"], c["miss"]), (3, 1, 2))
        dv.stage(torch.tensor([50, 1]), output=True)
        dv.commit()
        msg = dv.log_suffix()
        self.assertIn("draft vocab: cov=60.0%", msg)
        self.assertAlmostEqual(dv.last_coverage, 0.6)
        self.assertEqual(dv.log_suffix(), "")  # nothing new

    # ---------------------------------------------------------- persistence

    def test_persistence_round_trip(self):
        hot = [0, 1, 2, 3]
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(policy="tiered", proven_slots=3, state_dir=d, profile="py")
            dv, head = _make(hot, 3, config=cfg)
            dv.observe(torch.tensor([40, 41, 42]))
            dv.observe(torch.tensor([50, 51]), output=True)
            dv.observe(torch.tensor([51]), output=True)
            dv.save_state()
            self.assertTrue(os.path.exists(os.path.join(d, "dvocab-py.pt")))
            self.assertEqual(os.listdir(d), ["dvocab-py.pt"])

            dv2, head2 = _make(hot, 3, config=cfg)
            self.assertEqual(_ring(dv2), {40, 41, 42})
            self.assertTrue({50, 51} <= _pool(dv2))
            self.assert_consistent(dv2, head2, hot)
            # Learned order survives: 50 is older than 51 in the pool.
            dv2.observe(torch.tensor([60, 61]), output=True)
            self.assertEqual(_pool(dv2) & {50, 51}, {51})

            # A smaller config keeps the best entries.
            small = _cfg(policy="lru", state_dir=d, profile="py")
            dv3, head3 = _make(hot, 1, config=small)
            self.assertEqual(_pool(dv3), {51})
            self.assert_consistent(dv3, head3, hot)

    def test_persistence_rejects_incompatible_files(self):
        hot = [0, 1, 2, 3]
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(policy="lru", state_dir=d)
            dv, _ = _make(hot, 2, config=cfg)
            dv.observe(torch.tensor([40, 41]), output=True)
            dv.save_state()
            cold = _make(hot, 2, config=_cfg(policy="lru"))[0]
            # A different static map changes the map hash.
            other = _make([0, 1, 2, 5], 2, config=cfg)[0]
            self.assertFalse(_pool(other) & {40, 41})
            # A different vocab size.
            head = torch.randn(VOCAB + 8, HIDDEN)
            dv_v = DynamicDraftVocab(torch.tensor(hot), head, 2, VOCAB + 4, config=cfg)
            self.assertFalse(_pool(dv_v) & {40, 41})
            # A corrupt file.
            with open(cfg_path(d), "wb") as f:
                f.write(b"not a checkpoint")
            dv_c, head_c = _make(hot, 2, config=cfg)
            self.assertEqual(_pool(dv_c), _pool(cold))
            self.assert_consistent(dv_c, head_c, hot)

    def test_periodic_save_runs_off_thread(self):
        hot = [0, 1]
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(policy="lfu", state_dir=d, save_interval=0.0)
            dv, _ = _make(hot, 2, config=cfg)
            dv.observe(torch.tensor([40]), output=True)
            dv._save_thread.join(timeout=10)
            state = torch.load(cfg_path(d), weights_only=True)
            self.assertEqual(state["pool_ids"].tolist(), [40])
            self.assertEqual(state["map_hash"], dv.map_hash)


def cfg_path(d):
    return os.path.join(d, "dvocab-default.pt")


if __name__ == "__main__":
    unittest.main()
