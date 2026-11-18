"""SGLANG_STARTUP_AUTOTUNE_REPLAY skips profiling only for a cleanly loaded cache."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    should_replay_autotune_cache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAutotuneReplayGate(CustomTestCase):
    def _replay(self, *, reuse_cache=True, loaded=True, sync_group=None):
        return should_replay_autotune_cache(
            reuse_cache=reuse_cache, loaded=loaded, sync_group=sync_group
        )

    def test_off_by_default(self):
        self.assertFalse(self._replay())

    def test_replays_a_loaded_single_rank_cache(self):
        with envs.SGLANG_STARTUP_AUTOTUNE_REPLAY.override(True):
            self.assertTrue(self._replay())

    def test_tunes_when_the_cache_is_missing_or_rejected(self):
        with envs.SGLANG_STARTUP_AUTOTUNE_REPLAY.override(True):
            self.assertFalse(self._replay(loaded=False))
            self.assertFalse(self._replay(reuse_cache=False))

    def test_tunes_across_ranks(self):
        with envs.SGLANG_STARTUP_AUTOTUNE_REPLAY.override(True):
            self.assertFalse(self._replay(sync_group=MagicMock()))


if __name__ == "__main__":
    unittest.main()
