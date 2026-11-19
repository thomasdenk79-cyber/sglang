import logging
import re
import time
import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.admission_log import admission_fields, format_admission
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The line the incident parser keys on: one record per admitted request.
LINE = re.compile(
    r"^HiCache admission rid=(\S+) prompt=(\d+) device=(\d+) host=(\d+)"
    r" storage=(\d+) recompute=(\d+) attempts=(\d+) wait_ms=(-?\d+)$"
)
LOGGER_NAME = "sglang.srt.managers.schedule_policy"


class _TreeCacheStub:
    """Only the two admission hooks the accounting method calls."""

    def __init__(self):
        self.finished = []
        self.discarded = []

    def discard_storage_prefetch_accounting(self, handle):
        self.discarded.append(handle)

    def finish_storage_prefetch_admission(self, handle, fulfilled_tokens, reason):
        self.finished.append((handle, fulfilled_tokens, reason))


def _req(prompt_len=1000, rid="r-wire", wait_s=None):
    """A Req with just the fields admission reads; time stamp unset by default."""
    return SimpleNamespace(
        rid=rid,
        retracted_stain=False,
        cache_request_handle="h-1",
        full_untruncated_fill_ids=array("q", range(prompt_len)),
        materialized_host_hit_len=lambda: 0,
        storage_hit_length=0,
        storage_hit_start=None,
        host_hit_is_storage=False,
        fulfilled_storage_hit_len=lambda prefix_len: 0,
        needs_host_load_back=lambda: False,
        host_loaded_length=0,
        host_hit_length=0,
        storage_prefetch_retry_attempts=2,
        time_stats=SimpleNamespace(
            wait_queue_entry_time=time.perf_counter() if wait_s is None else wait_s
        ),
    )


def _adder():
    return SimpleNamespace(
        tree_cache=_TreeCacheStub(),
        log_device_hit_tokens=0,
        log_host_hit_tokens=0,
        log_storage_hit_tokens=0,
    )


class TestAdmissionFields(unittest.TestCase):
    def test_recompute_is_prompt_minus_prefix(self):
        """600 of 1000 prompt tokens were cached, so admission must price 400
        tokens as new prefill work; the tier split comes from the cache lookup."""
        fields = admission_fields(
            prompt_len=1000,
            prefix_len=600,
            device_hit=500,
            host_hit=0,
            storage_hit=100,
            attempts=2,
            wait_ms=37,
        )
        self.assertEqual(fields["recompute"], 400)
        self.assertEqual(
            fields,
            {
                "prompt": 1000,
                "device": 500,
                "host": 0,
                "storage": 100,
                "recompute": 400,
                "attempts": 2,
                "wait_ms": 37,
            },
        )

    def test_recompute_never_goes_negative(self):
        """A prefix longer than the prompt (session replay reports the full
        history) must not log a negative recompute: the parser casts to int
        and a negative share would hide rather than show a bug."""
        fields = admission_fields(
            prompt_len=100,
            prefix_len=180,
            device_hit=100,
            host_hit=80,
            storage_hit=0,
            attempts=0,
            wait_ms=-1,
        )
        self.assertEqual(fields["recompute"], 0)

    def test_wait_ms_minus_one_is_kept(self):
        """An unstamped queue entry reports -1, not 0: 0 would read as an
        instant admission and hide the missing stamp."""
        fields = admission_fields(
            prompt_len=10,
            prefix_len=0,
            device_hit=0,
            host_hit=0,
            storage_hit=0,
            attempts=0,
            wait_ms=-1,
        )
        self.assertEqual(fields["wait_ms"], -1)


class TestAdmissionFormat(unittest.TestCase):
    def test_line_matches_the_documented_shape(self):
        """The incident parser splits on single spaces, so the exact shape is
        part of the contract."""
        fields = admission_fields(
            prompt_len=1000,
            prefix_len=600,
            device_hit=500,
            host_hit=0,
            storage_hit=100,
            attempts=2,
            wait_ms=37,
        )
        self.assertEqual(
            format_admission("r1", fields),
            "HiCache admission rid=r1 prompt=1000 device=500 host=0"
            " storage=100 recompute=400 attempts=2 wait_ms=37",
        )

    def test_line_matches_the_parser_regex(self):
        """Every field must be present and numeric, and wait_ms may be -1."""
        line = format_admission(
            "6f9d1c2a-0",
            admission_fields(
                prompt_len=7,
                prefix_len=7,
                device_hit=3,
                host_hit=2,
                storage_hit=2,
                attempts=0,
                wait_ms=-1,
            ),
        )
        match = LINE.match(line)
        self.assertIsNotNone(match, line)
        self.assertEqual(match.group(1), "6f9d1c2a-0")
        self.assertEqual(match.group(2), "7")
        self.assertEqual(match.group(8), "-1")


class TestAdmissionWiring(unittest.TestCase):
    """The log line must come out of the real admission path, not only of the
    helpers: a typo in one field name would keep every helper test green."""

    def _capture(self):
        """Records of the scheduler-policy logger, INFO forced on: an unset
        level falls back to the root WARNING and would swallow the INFO line."""
        logger = logging.getLogger(LOGGER_NAME)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.setLevel, previous_level)
        messages = []

        class _Handler(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = _Handler()
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        return messages

    def test_no_line_while_the_flag_is_off(self):
        """Default off means the tag's code path: accounting runs, nothing is
        logged. A log line in a production run would cost one line per request."""
        messages = self._capture()
        adder = _adder()
        with envs.SGLANG_HICACHE_ADMISSION_LOG.override(False):
            PrefillAdder._account_prefill_cache_admission(adder, _req(), 600)
        self.assertEqual(messages, [])
        self.assertEqual(adder.tree_cache.finished, [("h-1", 0, None)])
        self.assertEqual(adder.log_device_hit_tokens, 600)

    def test_one_line_per_admitted_request(self):
        """With the flag on, each admitted request logs exactly one parsable
        line whose split matches what the cache reported."""
        messages = self._capture()
        adder = _adder()
        with envs.SGLANG_HICACHE_ADMISSION_LOG.override(True):
            PrefillAdder._account_prefill_cache_admission(adder, _req(), 600)
        self.assertEqual(len(messages), 1, messages)
        match = LINE.match(messages[0])
        self.assertIsNotNone(match, messages[0])
        self.assertEqual(match.group(1), "r-wire")
        self.assertEqual(match.group(2), "1000")  # prompt
        self.assertEqual(match.group(3), "600")  # device
        self.assertEqual(match.group(4), "0")  # host
        self.assertEqual(match.group(5), "0")  # storage
        self.assertEqual(match.group(6), "400")  # recompute
        self.assertEqual(match.group(7), "2")  # attempts
        self.assertGreaterEqual(int(match.group(8)), 0)  # wait_ms
        self.assertEqual(adder.log_device_hit_tokens, 600)
        # The log line is additive: accounting must have run exactly as before.
        self.assertEqual(adder.tree_cache.finished, [("h-1", 0, None)])

    def test_unstamped_entry_logs_wait_minus_one(self):
        """A request that never entered the waiting queue must report -1 rather
        than an invented queue time."""
        messages = self._capture()
        with envs.SGLANG_HICACHE_ADMISSION_LOG.override(True):
            PrefillAdder._account_prefill_cache_admission(
                _adder(), _req(wait_s=0.0), 600
            )
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("wait_ms=-1", messages[0])

    def test_retracted_request_logs_nothing(self):
        """The retraction branch returns early by design; logging it would
        double-count a request that is prefilled again."""
        messages = self._capture()
        adder = _adder()
        req = _req()
        req.retracted_stain = True
        with envs.SGLANG_HICACHE_ADMISSION_LOG.override(True):
            PrefillAdder._account_prefill_cache_admission(adder, req, 600)
        self.assertEqual(messages, [])
        self.assertEqual(adder.tree_cache.discarded, ["h-1"])


if __name__ == "__main__":
    unittest.main()
