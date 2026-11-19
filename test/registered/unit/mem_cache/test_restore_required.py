"""SGLANG_HICACHE_RESTORE_REQUIRED: a confirmed storage hit is fully restored
or ends as a typed restore_timeout (Task 184, commit C2); CPU only, no sleeps."""

import asyncio
import importlib.util
import json
import logging
import os
import re
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import orjson
import torch
from test_storage_prefetch_lifecycle import (
    _REQ,
    _arm_tick,
    _hit_drain_fixture,
    _staged_fixture,
    _timeout_fixture,
)

from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager, TypedAbortError
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage_prefetch import (
    RESTORE_TIMEOUT_ERR_TYPE,
    RestorePromises,
    RestoreState,
    StoragePrefetchRetries,
    restore_wait_s,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.mem_cache.storage_prefetch"
_OUTCOME = re.compile(
    r"HiCache restore outcome req=(\S+) generation=(\d+) "
    r"outcome=(restored|restore_timeout) wait_ms=(\d+) "
    r"restored_tokens=(\d+) required_tokens=(\d+)"
)
_MAX = torch.distributed.ReduceOp.MAX


def _restore_fixture(policy="wait_complete"):
    """One live buffer-mode prefetch of _REQ (8 tokens, span [2, 8))."""
    cache, operation = _timeout_fixture(policy=policy)
    cache.restore_required = True
    cache.restore_min_wait_s = 120.0
    _arm_tick(cache)
    operation.storage_start = 2
    operation.storage_hit_count = 6
    return cache, operation


def _confirm(cache, operation, now=100.0):
    with patch("time.perf_counter", return_value=now):
        cache._confirm_restore_promise(operation)
    return cache.storage_prefetch_retries.promises.get(_REQ.rid)


def _tick(cache, now):
    reduces = []
    cache._all_reduce = lambda tensor, op: reduces.append((tensor.tolist(), op))
    with patch("time.perf_counter", return_value=now):
        cache.check_hicache_events()
    return reduces


def _abort_timeouts(cache, reqs):
    scheduler = SimpleNamespace(
        tree_cache=cache, waiting_queue=list(reqs), request_retirer=Mock()
    )
    Scheduler._abort_restore_timeouts(scheduler)
    return scheduler.request_retirer.abort


def _queued_req(rid="r"):
    return SimpleNamespace(
        rid=rid,
        time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=Mock())),
    )


def _outcomes(records):
    return [m for m in (_OUTCOME.search(r.getMessage()) for r in records) if m]



def _flag_off_trace():
    """Admission decisions of the flag-off paths on fixed traces, using only
    APIs present at tag speedy-rc2 (golden recorded on C1 == tag for these
    files)."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from test_storage_prefetch_lifecycle import (
        _REQ,
        _arm_tick,
        _staged_fixture,
        _timeout_fixture,
    )

    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries

    trace = []
    # pop_ready: refetch / poll_miss below and at the cap.
    for attempts in (3, 8):
        for kind in ("refetch", "poll_miss"):
            retries = StoragePrefetchRetries()
            req = SimpleNamespace(rid="q", storage_prefetch_retry_attempts=attempts)
            getattr(retries, kind)("q", 8)
            for step in range(3):
                ready = retries.pop_ready([req], 2, 8)
                trace.append(
                    ("pop", attempts, kind, step, [(r.rid, e) for r, e in ready])
                )
    # Deadline policies: polls and the tick vote.
    for policy in ("timeout", "wait_complete", "best_effort"):
        cache, operation = _timeout_fixture(policy=policy)
        other = CacheRequestHandle("other", 0)
        cache.prefetch_from_storage(other, 0, list(range(8)))
        _arm_tick(cache)
        reduces = []
        cache._all_reduce = lambda tensor, op: reduces.append(tensor.tolist())
        for now in (100.0, 100.5, 200.0, 200.0):
            with patch("time.perf_counter", return_value=now):
                polled = cache.check_prefetch_progress(_REQ)
                cache.check_hicache_events()
            trace.append(
                (
                    "tick",
                    policy,
                    now,
                    polled,
                    list(reduces),
                    operation.is_terminated(),
                    _REQ in cache._storage_prefetch_wait
                    and cache._storage_prefetch_wait[_REQ].expired,
                    sorted(h.rid for h in cache.ongoing_prefetch),
                    sorted(cache.storage_prefetch_retries._pending),
                )
            )
    # Staged admission: 40 capacity defers drop the hold at the cap.
    cache, pipeline, req = _staged_fixture()
    req.staged_prefetch_plan = None
    for defer in range(40):
        pipeline.defer_staged_admission(req, "full")
        trace.append(("defer", defer, pipeline.has_staged(_REQ)))
    trace.append(("defer_prepare", pipeline.prepare_staged_prefetch(req)))
    # Staged admission plan and a released hold.
    cache, pipeline, req = _staged_fixture()
    trace.append(
        (
            "prepare",
            pipeline.prepare_staged_prefetch(req),
            req.host_hit_length,
            req.storage_hit_length,
            req.storage_hit_start,
        )
    )
    pipeline.release_staged_hold(_REQ)
    trace.append(
        (
            "released",
            pipeline.prepare_staged_prefetch(req),
            sorted(cache.storage_prefetch_retries._pending),
        )
    )
    return trace


# Recorded on sglang-wt-c1 (identical to speedy-rc2 for every file C2 touches).
_TAG_TRACE = [('pop', 3, 'refetch', 0, [('q', 8)]), ('pop', 3, 'refetch', 1, []),
 ('pop', 3, 'refetch', 2, []), ('pop', 3, 'poll_miss', 0, []),
 ('pop', 3, 'poll_miss', 1, []), ('pop', 3, 'poll_miss', 2, []),
 ('pop', 8, 'refetch', 0, []), ('pop', 8, 'refetch', 1, []),
 ('pop', 8, 'refetch', 2, []), ('pop', 8, 'poll_miss', 0, []),
 ('pop', 8, 'poll_miss', 1, []), ('pop', 8, 'poll_miss', 2, []),
 ('tick', 'timeout', 100.0, False, [[0, 0]], False, False, ['other', 'r'], []),
 ('tick', 'timeout', 100.5, False, [[0, 0], [0, 0]], False, False, ['other', 'r'],
  []),
 ('tick', 'timeout', 200.0, False, [[0, 0], [0, 0], [1, 0]], True, True,
  ['other', 'r'], []),
 ('tick', 'timeout', 200.0, True, [[0, 0], [0, 0], [1, 0], [0]], True, True,
  ['other'], []),
 ('tick', 'wait_complete', 100.0, False, [], False, False, ['other', 'r'], []),
 ('tick', 'wait_complete', 100.5, False, [], False, False, ['other', 'r'], []),
 ('tick', 'wait_complete', 200.0, False, [], False, False, ['other', 'r'], []),
 ('tick', 'wait_complete', 200.0, False, [], False, False, ['other', 'r'], []),
 ('tick', 'best_effort', 100.0, True, [], True, False, ['other'], ['r']),
 ('tick', 'best_effort', 100.5, True, [], True, False, ['other'], ['r']),
 ('tick', 'best_effort', 200.0, True, [], True, False, ['other'], ['r']),
 ('tick', 'best_effort', 200.0, True, [], True, False, ['other'], ['r']),
 ('defer', 0, True), ('defer', 1, True), ('defer', 2, True), ('defer', 3, True),
 ('defer', 4, True), ('defer', 5, True), ('defer', 6, True), ('defer', 7, True),
 ('defer', 8, True), ('defer', 9, True), ('defer', 10, True), ('defer', 11, True),
 ('defer', 12, True), ('defer', 13, True), ('defer', 14, True), ('defer', 15, True),
 ('defer', 16, True), ('defer', 17, True), ('defer', 18, True), ('defer', 19, True),
 ('defer', 20, True), ('defer', 21, True), ('defer', 22, True), ('defer', 23, True),
 ('defer', 24, True), ('defer', 25, True), ('defer', 26, True), ('defer', 27, True),
 ('defer', 28, True), ('defer', 29, True), ('defer', 30, True),
 ('defer', 31, False), ('defer', 32, False), ('defer', 33, False),
 ('defer', 34, False), ('defer', 35, False), ('defer', 36, False),
 ('defer', 37, False), ('defer', 38, False), ('defer', 39, False),
 ('defer_prepare', True), ('prepare', True, 6, 6, 2), ('released', True, [])]


class TestRestoreWait(unittest.TestCase):
    def test_deadline_scales_with_the_confirmed_missing_span(self):
        self.assertEqual(restore_wait_s(0, 120.0), 120.0)
        self.assertEqual(restore_wait_s(6, 120.0), 120.0)
        # 1 s + 0.25 s per started 1000 tokens, floored at MIN_WAIT_S.
        self.assertEqual(restore_wait_s(1_000_000, 120.0), 251.0)
        self.assertEqual(restore_wait_s(1_001, 0.0), 1.5)


class TestPopReadyCap(unittest.TestCase):
    def _pop(self, kind, **kwargs):
        retries = StoragePrefetchRetries()
        req = SimpleNamespace(rid="q", storage_prefetch_retry_attempts=8)
        getattr(retries, kind)("q", 8)
        return [(r.rid, end) for r, end in retries.pop_ready([req], 2, 8, **kwargs)]

    def test_a_confirmed_refetch_passes_the_cap_only_when_uncapped(self):
        self.assertEqual(self._pop("refetch"), [])
        self.assertEqual(self._pop("refetch", uncapped_immediate=True), [("q", 8)])

    def test_b_poll_miss_keeps_the_stock_cap(self):
        self.assertEqual(self._pop("poll_miss"), [])
        self.assertEqual(self._pop("poll_miss", uncapped_immediate=True), [])

    def test_t5_unconfirmed_shrink_keeps_cap_and_confirmed_rereads(self):
        cache, _ = _restore_fixture()
        req = SimpleNamespace(
            rid="r",
            cache_request_handle=_REQ,
            storage_prefetch_last_match_len=8,
            prefix_len=2,
            host_hit_length=0,
            storage_prefetch_retry_attempts=8,
        )
        scheduler = SimpleNamespace(tree_cache=cache, _retry_storage_prefetch=Mock())
        memory = SimpleNamespace(hicache_storage_prefetch_retry_max_attempts=8)
        with patch("sglang.srt.managers.scheduler.get_memory", return_value=memory):
            # No promise: past the cap the shorter match is admitted (stock).
            self.assertFalse(Scheduler._prefetch_after_device_hit_loss(scheduler, req))
            scheduler._retry_storage_prefetch.assert_not_called()
            self.assertEqual(req.storage_prefetch_last_match_len, 2)

            req.storage_prefetch_last_match_len = 8
            _confirm(cache, cache.ongoing_prefetch[_REQ].operation)
            # Confirmed: re-read the promised span; defer while I/O is open.
            self.assertTrue(Scheduler._prefetch_after_device_hit_loss(scheduler, req))
            scheduler._retry_storage_prefetch.assert_called_once_with(req, 8)
            # Nothing open: no defer; the staged gate still refuses the prefix.
            del cache.ongoing_prefetch[_REQ]
            self.assertFalse(Scheduler._prefetch_after_device_hit_loss(scheduler, req))


class TestDeadlinePolicy(unittest.TestCase):
    def test_c_flag_off_keeps_the_tag_policies(self):
        self.assertNotIn("SGLANG_HICACHE_RESTORE_REQUIRED", os.environ)
        for policy, active in (
            ("wait_complete", False),
            ("timeout", True),
            ("best_effort", False),
        ):
            cache, _ = _timeout_fixture(policy=policy)
            self.assertFalse(cache.restore_required)
            self.assertEqual(cache._prefetch_deadline_active(), active, policy)
            self.assertIsNone(cache.pending_restore_promise("r"))

    def test_flag_on_wait_complete_uses_the_vote(self):
        cache, _ = _restore_fixture("wait_complete")
        self.assertTrue(cache._prefetch_deadline_active())
        self.assertFalse(cache._short_prefetch_timeout())


class TestRestoreTimeout(unittest.TestCase):
    def test_d_j_expiry_is_one_typed_abort_and_one_outcome_log(self):
        cache, operation = _restore_fixture("wait_complete")
        with patch("time.perf_counter", return_value=100.0):
            self.assertFalse(cache.check_prefetch_progress(_REQ))
        with self.assertLogs(_LOGGER, logging.INFO) as logs:
            promise = _confirm(cache, operation, now=100.0)
            self.assertEqual(promise.deadline, 220.0)
            self.assertEqual(_tick(cache, 219.0), [([0, 0], _MAX)])
            self.assertFalse(promise.expired)
            self.assertEqual(_tick(cache, 221.0), [([0, 1], _MAX)])
            self.assertTrue(promise.expired)
            self.assertIs(promise.state, RestoreState.TERMINAL)
            self.assertTrue(operation.is_terminated())
            self.assertFalse(cache.storage_prefetch_retries.is_pending("r"))
            # Never a partial admit, however often it is polled.
            for now in (221.0, 500.0):
                with patch("time.perf_counter", return_value=now):
                    self.assertFalse(cache.check_prefetch_progress(_REQ))

            req = _queued_req()
            abort = _abort_timeouts(cache, [req])
            abort.assert_called_once()
            (aborted,), kwargs = abort.call_args
            self.assertIs(aborted, req)
            self.assertEqual(kwargs["reason"].err_type, RESTORE_TIMEOUT_ERR_TYPE)
            self.assertEqual(
                kwargs["reason"].status_code, HTTPStatus.SERVICE_UNAVAILABLE
            )
            self.assertEqual(kwargs["finished_reason"]["err_type"], "restore_timeout")
            self.assertEqual(kwargs["finished_reason"]["status_code"], 503)
            # The abort releases the attempt: the generation ends.
            cache._abandon_attempt(_REQ)
            self.assertIsNone(cache.storage_prefetch_retries.promises.get("r"))
            _tick(cache, 600.0)
            _abort_timeouts(cache, [req]).assert_not_called()

        outcomes = _outcomes(logs.records)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(
            outcomes[0].groups(), ("r", "0", "restore_timeout", "121000", "0", "6")
        )
        message = logs.records[-1].getMessage()
        self.assertIsNotNone(_OUTCOME.fullmatch(message), message)

    def test_t6_deadline_with_a_full_batch_needs_no_poll(self):
        cache, operation = _restore_fixture("wait_complete")
        promise = _confirm(cache, operation, now=100.0)
        # Admission never reaches the request (batch full): the tick decides.
        self.assertEqual(_tick(cache, 221.0), [([0, 1], _MAX)])
        self.assertTrue(promise.expired)
        self.assertEqual(
            [p.rid for p in cache.storage_prefetch_retries.promises.take_timeouts()],
            ["r"],
        )

    def test_timeout_policy_promise_supersedes_the_short_budget(self):
        cache, operation = _restore_fixture("timeout")
        with patch("time.perf_counter", return_value=100.0):
            self.assertFalse(cache.check_prefetch_progress(_REQ))
        promise = _confirm(cache, operation, now=100.0)
        # The page budget (2 s) is spent; the confirmed hit keeps loading.
        self.assertEqual(_tick(cache, 150.0), [([0, 0], _MAX)])
        self.assertFalse(operation.is_terminated())
        self.assertEqual(_tick(cache, 221.0), [([0, 1], _MAX)])
        self.assertTrue(promise.expired)
        self.assertTrue(operation.is_terminated())

    def test_t8_sidecar_without_ack_expires_at_the_same_deadline(self):
        cache, operation = _restore_fixture("wait_complete")
        # KV landed, sidecar ACK never comes: the stock cut skips such an op.
        operation.host_indices = torch.arange(6)
        operation.completed_tokens = len(operation.hash_value) * cache.page_size
        with patch("time.perf_counter", return_value=100.0):
            self.assertFalse(cache.check_prefetch_progress(_REQ))
        promise = _confirm(cache, operation, now=100.0)
        self.assertEqual(_tick(cache, 221.0), [([0, 1], _MAX)])
        self.assertTrue(promise.expired)
        self.assertTrue(operation.is_terminated())


def _staged_restore_fixture(promised_end=8):
    cache, pipeline, req = _staged_fixture()
    cache.restore_required = True
    cache.restore_min_wait_s = 120.0
    cache.prefetch_stop_policy = "wait_complete"
    cache.pp_rank = 0
    _arm_tick(cache)
    req.staged_prefetch_plan = None
    promise = cache.storage_prefetch_retries.promises.confirm(
        rid="r",
        operation_id=1,
        promised_end=promised_end,
        missing_tokens=6,
        now=100.0,
        min_wait_s=120.0,
    )
    return cache, pipeline, req, promise


class TestStagedRestore(unittest.TestCase):
    def test_e_forty_defers_keep_the_promise_then_type_the_timeout(self):
        cache, pipeline, req, promise = _staged_restore_fixture()
        retries = cache.storage_prefetch_retries
        for defer in range(40):
            pipeline.defer_staged_admission(req, "full")
            self.assertEqual(
                pipeline.has_staged(_REQ),
                defer + 1 < pipeline.max_staged_admission_defers,
            )
        # The host bounce is freed; the promise keeps span and deadline.
        self.assertIs(cache.pending_restore_promise("r"), promise)
        self.assertIs(promise.state, RestoreState.PROMISED)
        self.assertEqual((promise.promised_end, promise.deadline), (8, 220.0))
        # No recompute: the gate re-reads the span instead.
        self.assertFalse(pipeline.prepare_staged_prefetch(req))
        self.assertEqual(retries._pending["r"].storage_hit_end, 8)
        self.assertTrue(retries._pending["r"].immediate)
        self.assertFalse(pipeline.prepare_staged_prefetch(req))
        self.assertEqual(promise.rearms, 1)

        _tick(cache, 221.0)
        self.assertTrue(promise.expired)
        self.assertFalse(retries.is_pending("r"))
        self.assertFalse(pipeline.prepare_staged_prefetch(req))
        abort = _abort_timeouts(cache, [_queued_req()])
        self.assertEqual(
            abort.call_args.kwargs["reason"].err_type, RESTORE_TIMEOUT_ERR_TYPE
        )

    def test_full_staged_span_admits(self):
        cache, pipeline, req, promise = _staged_restore_fixture(promised_end=8)
        self.assertTrue(pipeline.prepare_staged_prefetch(req))
        self.assertEqual(req.storage_hit_length, 6)

    def test_partial_staged_span_is_reread_not_started(self):
        cache, pipeline, req, promise = _staged_restore_fixture(promised_end=10)
        self.assertFalse(pipeline.prepare_staged_prefetch(req))
        self.assertFalse(pipeline.has_staged(_REQ))
        self.assertEqual(cache.storage_prefetch_retries._pending["r"].storage_hit_end, 10)

    def test_t9_zero_load_is_not_recomputed(self):
        cache, pipeline, req, promise = _staged_restore_fixture()
        pipeline.staged_prefetches.clear()
        self.assertFalse(pipeline.prepare_staged_prefetch(req))
        self.assertTrue(cache.storage_prefetch_retries.is_pending("r"))
        # The device match alone covering the span admits.
        req.prefix_len = 8
        self.assertTrue(pipeline.prepare_staged_prefetch(req))


class TestGenerations(unittest.TestCase):
    def _confirm(self, promises, operation_id, now, promised_end=8, missing=6):
        return promises.confirm(
            rid="r",
            operation_id=operation_id,
            promised_end=promised_end,
            missing_tokens=missing,
            now=now,
            min_wait_s=120.0,
        )

    def test_retries_keep_span_and_deadline(self):
        promises = RestorePromises()
        first = self._confirm(promises, 1, now=100.0)
        again = self._confirm(promises, 2, now=150.0, promised_end=20, missing=50_000)
        self.assertIs(again, first)
        self.assertEqual((first.promised_end, first.required_tokens), (8, 6))
        self.assertEqual((first.t0, first.deadline), (100.0, 220.0))
        self.assertEqual(first.operation_ids, [1, 2])

    def test_f_t7_stale_completion_of_a_retired_generation_is_ignored(self):
        promises = RestorePromises()
        old = self._confirm(promises, 1, now=100.0)
        self.assertTrue(promises.advance(operation_id=1, state=RestoreState.STAGING))
        # Admitted, then retracted and queued again as a new attempt.
        promises.admit("r", covered_end=8, now=101.0)
        new = self._confirm(promises, 2, now=200.0)
        self.assertNotEqual(new.generation, old.generation)
        self.assertFalse(promises.advance(operation_id=1, state=RestoreState.STAGED))
        self.assertIs(new.state, RestoreState.PROMISED)
        self.assertTrue(promises.advance(operation_id=2, state=RestoreState.STAGED))
        # An expired generation ignores a late completion too.
        promises.apply_verdicts([1], now=400.0)
        self.assertFalse(promises.advance(operation_id=2, state=RestoreState.LOADING))
        self.assertIs(new.state, RestoreState.TERMINAL)
        self.assertEqual([p.rid for p in promises.take_timeouts()], ["r"])
        self.assertEqual(promises.take_timeouts(), [])

    def test_admit_logs_one_restored_outcome(self):
        promises = RestorePromises()
        self._confirm(promises, 1, now=100.0)
        with self.assertLogs(_LOGGER, logging.INFO) as logs:
            promises.admit("r", covered_end=8, now=100.25)
            promises.admit("r", covered_end=8, now=101.0)
        (outcome,) = _outcomes(logs.records)
        self.assertEqual(outcome.groups(), ("r", "0", "restored", "250", "6", "6"))
        self.assertIsNone(promises.get("r"))


class TestHitDrainConfirm(unittest.TestCase):
    def _drain(self, cache, operation, now):
        from array import array

        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch

        cache.ongoing_prefetch[_REQ] = _OngoingPrefetch(
            0, RadixKey(array("q", range(8))), None, operation, None, {}
        )
        cache.cache_controller.prefetch_hit_queue.put(operation)
        with patch("time.perf_counter", return_value=now):
            cache._drain_storage_control_queues_impl(
                n_storage_hit=1,
                n_ack_prefetch=0,
                n_backup=0,
                n_release=0,
                extra_release_counts={},
                log_metrics=False,
            )

    def _fixture(self, restore_required):
        cache = _hit_drain_fixture()
        cache.restore_required = restore_required
        cache.restore_min_wait_s = 120.0
        # Staging saturated: the confirmed hit parks (no allocation here).
        cache.cache_controller.prefetch_rate_limited = lambda: True
        return cache

    @staticmethod
    def _operation(operation_id):
        return SimpleNamespace(
            id=operation_id,
            request_id="r",
            handle=_REQ,
            storage_start=2,
            storage_hit_count=6,
            stats_requested_tokens=8,
            all_hash_values=None,
            deadline_expired=False,
            is_terminated=lambda: False,
        )

    def test_first_confirmed_hit_pins_the_promise(self):
        cache = self._fixture(restore_required=True)
        self._drain(cache, self._operation(7), now=50.0)
        promise = cache.storage_prefetch_retries.promises.get("r")
        self.assertEqual(
            (promise.promised_end, promise.required_tokens, promise.deadline),
            (8, 6, 170.0),
        )
        self._drain(cache, self._operation(8), now=90.0)
        self.assertIs(cache.storage_prefetch_retries.promises.get("r"), promise)
        self.assertEqual((promise.deadline, promise.operation_ids), (170.0, [7, 8]))

    def test_flag_off_creates_nothing(self):
        cache = self._fixture(restore_required=False)
        self._drain(cache, self._operation(7), now=50.0)
        self.assertEqual(len(cache.storage_prefetch_retries.promises), 0)


class TestFlagOffParity(unittest.TestCase):
    def test_g_t10_flag_off_matches_the_tag_on_recorded_traces(self):
        self.assertEqual(_flag_off_trace(), _TAG_TRACE)


class TestTypedAbortWire(unittest.TestCase):
    def _handle(self, finish_reason):
        manager = SimpleNamespace(rid_to_state={}, _release_lora_once=lambda s: None)
        state = SimpleNamespace(obj=SimpleNamespace(rid="r"))
        out = {"meta_info": {"finish_reason": finish_reason}}
        return asyncio.run(
            TokenizerManager._handle_abort_finish_reason(
                manager, out, state, is_stream=False
            )
        )

    def test_h_tokenizer_raises_the_typed_error_only_for_restore_timeout(self):
        typed = FINISH_ABORT("late", HTTPStatus.SERVICE_UNAVAILABLE, "restore_timeout")
        with self.assertRaises(TypedAbortError) as ctx:
            self._handle(typed.to_json())
        self.assertEqual(
            (ctx.exception.status_code, ctx.exception.err_type),
            (503, "restore_timeout"),
        )
        plain = FINISH_ABORT("busy", HTTPStatus.SERVICE_UNAVAILABLE)
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            self._handle(plain.to_json())
        self.assertNotIsInstance(ctx.exception, TypedAbortError)


def _serving_chat_case():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "entrypoints",
        "openai",
        "test_serving_chat.py",
    )
    spec = importlib.util.spec_from_file_location("_c2_serving_chat_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SC = _serving_chat_case()


class TestTypedAbortServing(CustomTestCase):
    # Borrow the serving-chat fixture only (not its tests).
    setUp = _SC.ServingChatTestCase.setUp
    _close_event_loops = staticmethod(_SC.ServingChatTestCase._close_event_loops)

    def test_h_non_stream_is_http_503_with_typed_error(self):
        async def _raise():
            raise TypedAbortError(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="late",
                err_type="restore_timeout",
            )
            yield

        self.tm.generate_request = Mock(return_value=_raise())
        response = asyncio.run(
            self.chat._chat_completion_response(None, self.basic_req, None)
        )
        self.assertEqual(response.status_code, 503)
        body = orjson.loads(response.body)
        self.assertEqual(body["type"], "restore_timeout")
        self.assertEqual(body["code"], 503)
        self.assertIs(type(body["code"]), int)

    def _stream(self, finish_reason):
        def _out(text, reason):
            return {
                "text": text,
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": reason,
                    "output_token_logprobs": [],
                    "output_token_logprobs_length": 0,
                    "output_top_logprobs": [],
                },
                "index": 0,
            }

        async def _generate():
            yield _out("Hi", None)
            yield _out("Hi", finish_reason)

        self.tm.generate_request = Mock(return_value=_generate())
        req = _SC.ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}], stream=True
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_mock.return_value = Mock(get_prompt=Mock(return_value="p"))
            adapted, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

        async def run():
            return [
                chunk
                async for chunk in self.chat._generate_chat_stream(
                    adapted, req, self.fastapi_request
                )
            ]

        chunks = asyncio.run(run())
        errors = [
            json.loads(c[len("data: ") :])["error"] for c in chunks if '"error"' in c
        ]
        self.assertEqual(len(errors), 1, chunks)
        # The error follows a content chunk: the stream had started.
        self.assertIn('"content"', "".join(chunks[: chunks.index(next(
            c for c in chunks if '"error"' in c))]))
        return errors[0]

    def test_i_started_stream_carries_the_typed_sse_error(self):
        typed = FINISH_ABORT("late", HTTPStatus.SERVICE_UNAVAILABLE, "restore_timeout")
        error = self._stream(typed.to_json())
        self.assertEqual((error["type"], error["code"]), ("restore_timeout", 503))

    def test_i_other_aborts_keep_their_status_derived_type(self):
        plain = FINISH_ABORT("busy", HTTPStatus.SERVICE_UNAVAILABLE)
        error = self._stream(plain.to_json())
        self.assertEqual((error["type"], error["code"]), ("SERVICE_UNAVAILABLE", 503))


if __name__ == "__main__":
    unittest.main()
