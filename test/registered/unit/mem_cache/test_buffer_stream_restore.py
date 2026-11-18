"""Unit coverage for the buffer-only streamed restore and oversize backup split
(SGLANG_HICACHE_BUFFER_STREAM). CPU only: the CUDA copy path is exercised
through fakes; the real H2D needs a GPU (see the module docstring of
buffer_mode/stream.py)."""

import threading
import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    _StagedPrefetch,
)
from sglang.srt.mem_cache.buffer_mode.stream import (
    resolve_stream_device,
    BufferStreamPlan,
    StreamChunk,
    StreamGate,
    StreamStats,
    chunk_batches,
    drive_stream,
    h2d_quota_tokens,
    plan_stream_chunks,
    ring_slice,
    ring_tokens,
    split_piece_ranges,
    split_span,
    stream_chunk_tokens,
    streamed_page_transfer,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache import fixtures as _fixtures
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestStreamPlanning(unittest.TestCase):
    def test_chunks_cover_hit_page_aligned_and_cycle_slots(self):
        chunks = plan_stream_chunks(64 * 10, 64 * 3 + 5, 64, inflight=2)
        self.assertEqual(
            [(c.start, c.end, c.slot) for c in chunks],
            [(0, 192, 0), (192, 384, 1), (384, 576, 0), (576, 640, 1)],
        )
        self.assertEqual(ring_tokens(chunks), 2 * 192)
        self.assertEqual(ring_slice(chunks[3], 192), slice(192, 256))

    def test_slots_never_exceed_chunks_and_unaligned_hit_rejected(self):
        chunks = plan_stream_chunks(128, 1024, 64, inflight=3)
        self.assertEqual([(c.start, c.end, c.slot) for c in chunks], [(0, 128, 0)])
        self.assertEqual(ring_tokens(chunks), 128)
        with self.assertRaises(ValueError):
            plan_stream_chunks(100, 64, 64, inflight=2)

    def test_chunk_tokens_env_is_page_aligned(self):
        with envs.SGLANG_HICACHE_BUFFER_STREAM_CHUNK_TOKENS.override(1000):
            self.assertEqual(stream_chunk_tokens(64), 960)
        with envs.SGLANG_HICACHE_BUFFER_STREAM_CHUNK_TOKENS.override(10):
            self.assertEqual(stream_chunk_tokens(64), 64)

    def test_chunk_batches_follow_storage_batches(self):
        chunk = StreamChunk(1, 64 * 300, 64 * 600, 1)
        self.assertEqual(
            chunk_batches(chunk, 64, 128), [(300, 428), (428, 556), (556, 600)]
        )

    def test_h2d_quota_and_pieces(self):
        with envs.SGLANG_HICACHE_BUFFER_STREAM_H2D_MB_PER_ITER.override(1):
            self.assertEqual(h2d_quota_tokens(1000, 64), 1024)  # 1048 -> page
            self.assertEqual(h2d_quota_tokens(1 << 30, 64), 64)
            self.assertEqual(h2d_quota_tokens(0, 64), 0)
        with envs.SGLANG_HICACHE_BUFFER_STREAM_H2D_MB_PER_ITER.override(0):
            self.assertEqual(h2d_quota_tokens(1000, 64), 0)
        self.assertEqual(split_piece_ranges(0, 10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(split_piece_ranges(0, 10, 0), [(0, 10)])

    def test_split_span_keeps_order_and_alignment(self):
        self.assertEqual(split_span(1000, 256, 64), [256, 256, 256, 232])
        self.assertEqual(split_span(512, 256, 64), [256, 256])
        self.assertEqual(split_span(100, 1000, 64), [100])


class TestDriveStream(unittest.TestCase):
    def _run(self, num_chunks, inflight, *, fail_read_at=None, deny_grant_at=None):
        chunks = plan_stream_chunks(64 * num_chunks, 64, 64, inflight)
        log = []

        def read_chunk(c):
            log.append(("read", c.index))
            return c.index != fail_read_at

        def acquire(c):
            return None if c.index == deny_grant_at else ("rows", c.index)

        def submit(c, grant):
            log.append(("h2d", c.index))
            return c.index, True

        def wait(h):
            log.append(("wait", h))

        stats = StreamStats(tokens=64 * num_chunks, chunks=num_chunks, inflight=inflight)
        ok = drive_stream(
            chunks,
            read_chunk=read_chunk,
            acquire=acquire,
            submit_h2d=submit,
            wait_h2d=wait,
            h2d_seconds=lambda h: 0.0,
            stats=stats,
        )
        return ok, log, stats

    def test_slot_reused_only_after_its_previous_copy(self):
        ok, log, stats = self._run(5, 2)
        self.assertTrue(ok and stats.ok)
        for k in range(2, 5):
            self.assertLess(log.index(("wait", k - 2)), log.index(("read", k)))
        # Reads run ahead of the copies of the other slot (pipelined).
        self.assertLess(log.index(("read", 1)), log.index(("wait", 0)))
        self.assertEqual(sorted(h for op, h in log if op == "wait"), list(range(5)))

    def test_failed_read_drains_every_copy_and_reports_failure(self):
        ok, log, stats = self._run(5, 3, fail_read_at=3)
        self.assertFalse(ok or stats.ok)
        self.assertNotIn(("h2d", 3), log)
        self.assertEqual(sorted(h for op, h in log if op == "wait"), [0, 1, 2])

    def test_denied_grant_aborts(self):
        ok, log, _ = self._run(4, 2, deny_grant_at=1)
        self.assertFalse(ok)
        self.assertEqual([e for e in log if e[0] == "h2d"], [("h2d", 0)])

    def test_overlap_ratio(self):
        stats = StreamStats(tokens=1, chunks=1, inflight=2, read_s=4, h2d_s=2, wall_s=4.5)
        self.assertAlmostEqual(stats.overlap_ratio(), 0.75)
        stats.wall_s = 6
        self.assertEqual(stats.overlap_ratio(), 0.0)
        self.assertIn("overlap=0.00", stats.format("r"))


class TestStreamGate(unittest.TestCase):
    def test_wait_returns_grant_from_scheduler_thread(self):
        gate = StreamGate()
        threading.Timer(0.02, gate.grant, args=(0, "rows")).start()
        self.assertEqual(gate.wait(0, lambda: False), "rows")
        self.assertEqual(gate.taken, 1)

    def test_credit_is_capped_and_spent(self):
        gate = StreamGate()
        for _ in range(5):
            gate.deposit(4, cap=8)
        self.assertTrue(gate.spend(8, lambda: False))
        # Idle iterations do not bank more than one quota.
        self.assertFalse(gate.spend(4, lambda: True))
        threading.Timer(0.02, gate.deposit, args=(4, 8)).start()
        self.assertTrue(gate.spend(4, lambda: False))
        gate.close()
        self.assertFalse(gate.spend(4, lambda: False))

    def test_close_and_cancel_end_the_wait(self):
        gate = StreamGate()
        gate.close()
        self.assertIsNone(gate.wait(0, lambda: False))
        self.assertIsNone(StreamGate().wait(0, lambda: True))


def _plan(num_chunks, inflight=2, page=4):
    chunks = plan_stream_chunks(page * num_chunks, page, page, inflight)
    slots = ring_tokens(chunks)
    return BufferStreamPlan(
        request=SimpleNamespace(rid="r"),
        chunks=chunks,
        slot_tokens=chunks[0].tokens,
        ring_host_indices=torch.arange(100, 100 + slots),
        ring_units=slots,
        engine=None,
        stats=StreamStats(tokens=chunks[-1].end, chunks=num_chunks, inflight=inflight),
    )


class TestStreamedPageTransfer(unittest.TestCase):
    def _controller(self, plan, *, hit_pages_at=None):
        reads = []
        submits = []

        def kv_batch(op, hashes, host, extra_info, kv_derived):
            reads.append((list(hashes), host.tolist(), [t.name for t in kv_derived]))
            if hit_pages_at is not None and hashes[0] == hit_pages_at:
                return 0
            return len(hashes)

        completion = SimpleNamespace(
            finish_event=SimpleNamespace(synchronize=lambda: None),
            timing_enabled=False,
        )
        plan.engine = SimpleNamespace(
            host_to_device_stream=MagicMock(),
            submit_host_to_device=lambda transfers, **kw: (
                submits.append(transfers) or completion
            ),
        )
        cc = SimpleNamespace(
            page_size=4,
            device="cpu",
            transfer_layer_id_max=1,
            prefetch_sync_queue=Queue(),
            storage_stop_event=threading.Event(),
            _page_transfer_kv_batch=kv_batch,
            _move_op_indices=lambda op: (
                op.host_indices,
                op.device_indices,
                op.pool_transfers,
            ),
            _l2_load_transfers=lambda h, d, p: [(h.tolist(), d.tolist(), p)],
        )
        return cc, reads, submits

    def _operation(self, plan, pages):
        return SimpleNamespace(
            buffer_stream=plan,
            hash_value=[f"h{i}" for i in range(pages)],
            pool_transfers=[
                PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ],
            prefix_keys=None,
            request_id="r",
            is_terminated=lambda: False,
        )

    def _acks(self, cc):
        acks = []
        while not cc.prefetch_sync_queue.empty():
            acks.append(cc.prefetch_sync_queue.get().completed_tokens)
        return acks

    def test_all_chunks_land_then_one_final_ack(self):
        plan = _plan(3)
        for c in plan.chunks:
            rows = torch.arange(c.start, c.end) + 1000
            plan.device_parts[c.index] = rows
            plan.gate.grant(c.index, (rows, None))
        cc, reads, submits = self._controller(plan)
        pages = streamed_page_transfer(cc, self._operation(plan, 3))
        self.assertEqual(pages, 3)
        self.assertEqual(self._acks(cc), [0, 0, 12])
        # Chunk 2 reuses ring slot 0; KV-derived sidecars ride the KV rows.
        self.assertEqual(reads[2], (["h2"], [100, 101, 102, 103], [PoolName.INDEXER]))
        host, device, derived = submits[1][0]
        self.assertEqual((host, device), ([104, 105, 106, 107], [1004, 1005, 1006, 1007]))
        self.assertEqual(derived[0].device_indices.tolist(), device)
        self.assertEqual(plan.device_rows().tolist(), list(range(1000, 1012)))

    def test_quota_splits_copies_and_cancel_stops_them(self):
        plan = _plan(2)
        plan.quota_tokens = 2
        for c in plan.chunks:
            plan.gate.grant(c.index, (torch.arange(c.start, c.end), None))
        plan.gate.deposit(6, cap=6)  # three pieces of credit, four needed
        cc, _, submits = self._controller(plan)
        op = self._operation(plan, 2)
        timer = threading.Timer(0.1, cc.storage_stop_event.set)
        timer.start()
        pages = streamed_page_transfer(cc, op)
        timer.cancel()
        self.assertEqual(pages, 0)
        self.assertEqual([s[0][1] for s in submits], [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(self._acks(cc), [0, 0])

    def test_short_read_reports_nothing_with_same_ack_count(self):
        plan = _plan(3)
        for c in plan.chunks:
            plan.gate.grant(c.index, (torch.arange(c.start, c.end), None))
        cc, _, submits = self._controller(plan, hit_pages_at="h1")
        pages = streamed_page_transfer(cc, self._operation(plan, 3))
        self.assertEqual(pages, 0)
        self.assertEqual(self._acks(cc), [0, 0, 0])
        self.assertEqual(len(submits), 1)


def _pipeline():
    pipeline = BufferModePipeline.__new__(BufferModePipeline)
    allocator = MagicMock(spec=["alloc", "free", "size"])
    allocator.alloc.side_effect = lambda n: torch.arange(n)
    cc = SimpleNamespace(
        load_fence_stream=None,
        prefetch_tokens_occupied=10,
        append_host_mem_release=MagicMock(),
    )
    pipeline._cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator, cache_controller=cc, page_size=4
    )
    pipeline._streams = {}
    pipeline._stream_last_tick = None
    pipeline._stream_baseline_step_s = 0.0
    pipeline._ensure_full_capacity = MagicMock(return_value=True)
    return pipeline, allocator, cc


class TestTickStreams(unittest.TestCase):
    def test_inflight_caps_reservations(self):
        pipeline, allocator, _ = _pipeline()
        plan = _plan(6, inflight=2)
        pipeline._streams[1] = plan
        with envs.SGLANG_HICACHE_BUFFER_STREAM_DEVICE_HEADROOM.override(0):
            for _ in range(3):
                pipeline.tick_streams()
            # The IO worker took nothing: at most `inflight` grants ahead.
            self.assertEqual(plan.next_grant, 2)
            self.assertEqual(plan.granted_tokens(), 8)
            plan.gate.wait(0, lambda: False)
            pipeline.tick_streams()
            self.assertEqual(plan.next_grant, 3)

    def test_quota_deposit_is_shared_between_restores(self):
        pipeline, _, _ = _pipeline()
        plans = [_plan(2), _plan(2)]
        for i, plan in enumerate(plans):
            plan.quota_tokens = 16
            pipeline._streams[i] = plan
        pipeline.tick_streams()
        for plan in plans:
            self.assertTrue(plan.gate.spend(8, lambda: False))
            self.assertFalse(plan.gate.spend(4, lambda: True))

    def test_no_reservation_without_headroom(self):
        pipeline, allocator, _ = _pipeline()
        plan = _plan(2)
        pipeline._streams[1] = plan
        pipeline._ensure_full_capacity.return_value = False
        with envs.SGLANG_HICACHE_BUFFER_STREAM_DEVICE_HEADROOM.override(8):
            pipeline.tick_streams()
        pipeline._ensure_full_capacity.assert_called_with(4 + 8)
        allocator.alloc.assert_not_called()
        self.assertEqual(plan.next_grant, 0)

    def test_finish_frees_unstaged_rows_and_ring(self):
        pipeline, allocator, cc = _pipeline()
        plan = _plan(2)
        plan.device_parts = {0: torch.arange(4), 1: torch.arange(4, 8)}
        pipeline._streams[7] = plan
        op = SimpleNamespace(id=7, pool_transfers=None, pool_transfers_done=True)
        pipeline.finish_stream_operation(op)
        self.assertEqual(allocator.free.call_args.args[0].tolist(), list(range(8)))
        self.assertEqual(cc.prefetch_tokens_occupied, 10 - plan.ring_units)
        self.assertIs(cc.append_host_mem_release.call_args.args[0], plan.ring_host_indices)

        allocator.free.reset_mock()
        staged = _plan(1)
        staged.device_parts = {0: torch.arange(4)}
        staged.staged = True
        pipeline._streams[8] = staged
        pipeline.finish_stream_operation(SimpleNamespace(id=8, pool_transfers=None, pool_transfers_done=True))
        allocator.free.assert_not_called()


def _staged(rows):
    return _StagedPrefetch(
        request="h",
        key_tokens=array("q"),
        extra_key=None,
        cache_salt=None,
        matched_len=0,
        num_tokens=len(rows),
        occupied_tokens=3,
        host_indices=torch.empty(0, dtype=torch.int64),
        aux_xfers=[],
        hash_values=[],
        operation_id=1,
        device_indices=rows,
    )


class TestStreamedPublish(unittest.TestCase):
    def _pipeline(self, queued):
        pipeline, allocator, cc = _pipeline()
        cc.load = MagicMock(return_value=queued)
        mamba = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7]))
        context = MagicMock(load_xfers=[mamba])
        component = MagicMock()
        component.prepare_buffer_load_back.return_value = context
        pipeline._cache.components = {"mamba": component}
        return pipeline, allocator, cc, context

    def test_publish_binds_streamed_rows_and_drops_kv_copy(self):
        pipeline, allocator, cc, context = self._pipeline(queued=1)
        rows = torch.arange(100, 116)
        f = _staged(rows)
        plan = SimpleNamespace(device_prefix_len=4, full_tokens=12)
        load_xfers = [PoolTransfer(name=PoolName.KV, indices_from_pool=PoolName.KV)]
        state, reason = pipeline._queue_streamed_load_back(
            SimpleNamespace(rid="r"), f, plan, -2, load_xfers
        )
        self.assertEqual(reason, "")
        kwargs = cc.load.call_args.kwargs
        self.assertEqual(kwargs["host_indices"].numel(), 0)
        self.assertEqual(kwargs["anchor_indices"].tolist(), list(range(104, 116)))
        self.assertEqual([t.name for t in kwargs["extra_pools"]], [PoolName.MAMBA])
        # The head the device prefix grew over is freed; the rest is published.
        self.assertEqual(allocator.free.call_args.args[0].tolist(), [100, 101, 102, 103])
        self.assertEqual(state.device_indices.tolist(), list(range(104, 116)))
        self.assertIsNone(f.device_indices)
        context.finalize_allocation.assert_called_once_with(success=True)

    def test_failed_publish_keeps_rows_for_retry(self):
        pipeline, allocator, _, context = self._pipeline(queued=None)
        rows = torch.arange(8)
        f = _staged(rows)
        state, reason = pipeline._queue_streamed_load_back(
            SimpleNamespace(rid="r"),
            f,
            SimpleNamespace(device_prefix_len=0, full_tokens=8),
            -2,
            [],
        )
        self.assertIsNone(state)
        self.assertEqual(reason, "full_or_aux")
        self.assertIs(f.device_indices, rows)
        allocator.free.assert_not_called()
        context.finalize_allocation.assert_called_once_with(success=False)

    def test_abort_of_staged_stream_frees_device_rows(self):
        pipeline, allocator, cc = _pipeline()
        rows = torch.arange(8)
        pipeline.staged_prefetches = {"h": _staged(rows)}
        pipeline._staged_admission_defers = {}
        pipeline.release_anchor_lock = MagicMock()
        pipeline._free_staging_now = MagicMock()
        pipeline._cache._finish_storage_prefetch = MagicMock()
        self.assertTrue(pipeline.release_staged_hold("h", reason="abort"))
        self.assertIs(allocator.free.call_args.args[0], rows)
        self.assertEqual(cc.prefetch_tokens_occupied, 10 - 3)


class TestFlagOff(unittest.TestCase):
    def test_oversize_prefetch_still_declined_without_stream(self):
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        cc = SimpleNamespace(can_fit_prefetch_host_buffers=MagicMock(return_value=False))
        pipeline._cache = SimpleNamespace(cache_controller=cc)
        pipeline._try_stream_prefetch = MagicMock()
        pipeline._reject_oversized_prefetch = MagicMock(return_value=True)
        with envs.SGLANG_HICACHE_BUFFER_STREAM.override(False):
            self.assertTrue(
                pipeline.allocate_and_queue_prefetch(MagicMock(), MagicMock(), 1024, 0)
            )
        pipeline._try_stream_prefetch.assert_not_called()
        pipeline._reject_oversized_prefetch.assert_called_once()

    def test_unsupported_stream_falls_back_to_decline(self):
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        cc = SimpleNamespace(can_fit_prefetch_host_buffers=MagicMock(return_value=False))
        pipeline._cache = SimpleNamespace(cache_controller=cc)
        pipeline._try_stream_prefetch = MagicMock(return_value=None)
        pipeline._reject_oversized_prefetch = MagicMock(return_value=True)
        with envs.SGLANG_HICACHE_BUFFER_STREAM.override(True):
            pipeline.allocate_and_queue_prefetch(MagicMock(), MagicMock(), 1024, 0)
        pipeline._reject_oversized_prefetch.assert_called_once()

    def test_oversize_backup_split_deferred_only_with_flag(self):
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = SimpleNamespace(
            tree_core=SimpleNamespace(split_for_backup=None), page_size=4
        )
        pipeline._oversize_splits = {}
        pipeline._backup_oversize = MagicMock(return_value=True)
        intents = [(PoolName.KV, ["a", "b"]), (PoolName.MAMBA, ["b"])]
        self.assertTrue(pipeline._defer_oversize_split(5, intents))
        self.assertEqual(list(pipeline._oversize_splits), [5])
        pipeline._backup_oversize.return_value = False
        self.assertFalse(pipeline._defer_oversize_split(6, intents))

    def test_oversize_backup_without_split_support_warns_once(self):
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = SimpleNamespace(tree_core=SimpleNamespace(), page_size=4)
        pipeline._oversize_splits = {}
        pipeline._warned_no_split = False
        pipeline._backup_oversize = MagicMock(return_value=True)
        intents = [(PoolName.KV, ["a", "b"])]
        with self.assertLogs(
            "sglang.srt.mem_cache.buffer_mode.pipeline", level="WARNING"
        ) as logs:
            self.assertFalse(pipeline._defer_oversize_split(5, intents))
            self.assertFalse(pipeline._defer_oversize_split(6, intents))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(pipeline._oversize_splits, {})


class TestSplitForBackup(unittest.TestCase):
    def test_split_keeps_tail_id_and_hash_chain(self):
        cfg = CacheConfig(
            page_size=4, components=(ComponentType.FULL,), kv_size=128, max_context_len=128
        )
        with mock.patch.object(_fixtures, "get_device", return_value="cpu"):
            cache, allocator, _ = build_fixture(cfg)
        tokens = array("q", range(1, 41))
        key = RadixKey(tokens)
        cache.insert(InsertParams(key=key, value=allocator.alloc(len(tokens))))
        leaf = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        core = cache.tree_core
        arena = core._node_arena
        # Hashes only exist with storage; seed one per page to follow them.
        arena[leaf].hash_value = [f"h{i}" for i in range(10)]

        pieces, actions = core.split_for_backup(leaf, 16)
        for action in actions:
            cache._apply_cache_action(action)

        self.assertEqual(pieces[-1], leaf)
        self.assertEqual([len(arena[p].key) for p in pieces], [16, 16, 8])
        for parent, child in zip(pieces, pieces[1:]):
            self.assertIs(arena[child].parent, arena[parent])
        self.assertEqual(
            [h for p in pieces for h in arena[p].hash_value],
            [f"h{i}" for i in range(10)],
        )
        snap = core.snapshot_buffer_backup(leaf, True)
        self.assertEqual(snap.prefix_keys, [f"h{i}" for i in range(8)])
        self.assertEqual(
            cache.match_prefix(MatchPrefixParams(key=key)).last_device_node, leaf
        )
        cache.sanity_check()


class TestResolveStreamDevice(unittest.TestCase):
    def test_bare_cuda_takes_the_stream_index(self):
        engine = SimpleNamespace(
            host_to_device_stream=SimpleNamespace(device=torch.device("cuda", 3))
        )
        self.assertEqual(resolve_stream_device("cuda", engine), torch.device("cuda", 3))

    def test_explicit_index_and_cpu_are_kept(self):
        engine = SimpleNamespace(
            host_to_device_stream=SimpleNamespace(device=torch.device("cuda", 3))
        )
        self.assertEqual(
            resolve_stream_device("cuda:1", engine), torch.device("cuda", 1)
        )
        self.assertEqual(resolve_stream_device("cpu", engine), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
