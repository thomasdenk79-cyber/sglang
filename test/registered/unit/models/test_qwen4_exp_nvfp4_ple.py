"""Load-completeness check for NVFP4-packed Qwen4-Exp PLE tables.

The offloaded host table starts uninitialized, so every shard's packed codes
and e4m3 block scales, and the fp32 global scale, must have been loaded.
"""

import unittest

from sglang.srt.models.qwen4_exp import check_nvfp4_ple_complete
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "model.layers.1.ple.ple_embedding"


def _check(vocab, num_shards, weight, scale, has_global_scale=True):
    check_nvfp4_ple_complete(
        PREFIX,
        vocab,
        num_shards,
        {"weight": set(weight), "weight_scale": set(scale)},
        has_global_scale,
    )


class TestCheckNvfp4PleComplete(CustomTestCase):
    def test_complete_checkpoint_passes(self):
        _check(1024, 128, range(128), range(128))

    def test_uneven_vocab_expects_only_non_empty_shards(self):
        # A vocab of 9 in 4 shards of ceil(9 / 4) = 3 rows fills only shards
        # 0..2; a vocab of 10 needs all four.
        _check(9, 4, range(3), range(3))
        with self.assertRaisesRegex(ValueError, r"1/4 weight shards missing"):
            _check(10, 4, range(3), range(4))

    def test_missing_weight_shard_raises(self):
        with self.assertRaisesRegex(
            ValueError, r"ple_embedding: 1/128 weight shards missing \(first: \[5\]\)"
        ):
            _check(1024, 128, set(range(128)) - {5}, range(128))

    def test_missing_block_scale_shards_raise(self):
        with self.assertRaisesRegex(
            ValueError, r"2/128 weight_scale shards missing \(first: \[0, 127\]\)"
        ):
            _check(1024, 128, range(128), range(1, 127))

    def test_missing_global_scale_raises(self):
        with self.assertRaisesRegex(ValueError, r"weight_scale_2 missing"):
            _check(1024, 128, range(128), range(128), has_global_scale=False)

    def test_reports_all_problems_together(self):
        with self.assertRaises(ValueError) as ctx:
            _check(1024, 128, [], [], has_global_scale=False)
        msg = str(ctx.exception)
        self.assertIn("128/128 weight shards missing", msg)
        self.assertIn("128/128 weight_scale shards missing", msg)
        self.assertIn("weight_scale_2 missing", msg)


if __name__ == "__main__":
    unittest.main()
