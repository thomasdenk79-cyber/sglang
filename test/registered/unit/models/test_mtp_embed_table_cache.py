"""SGLANG_STARTUP_MTP_EMBED_TABLE_CACHE keys, stores and validates the table."""

import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _embed_table_cache_path,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMtpEmbedTableCache(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = [torch.randn(8, 4, dtype=torch.bfloat16), torch.ones(4)]

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, weights):
        return _embed_table_cache_path(weights, 8, 4, torch.bfloat16)

    def test_off_by_default(self):
        self.assertIsNone(self._path(self.w))

    def test_key_tracks_tensor_bytes(self):
        with envs.SGLANG_STARTUP_MTP_EMBED_TABLE_CACHE.override(
            True
        ), envs.SGLANG_CACHE_DIR.override(self.tmp.name):
            a = self._path(self.w)
            self.assertEqual(a, self._path([t.clone() for t in self.w]))
            changed = [self.w[0].clone(), self.w[1].clone()]
            changed[1][0] = 2.0
            self.assertNotEqual(a, self._path(changed))
            self.assertTrue(a.startswith(self.tmp.name))

    def _load(self, path, check):
        fake = MagicMock()
        fake._check_fc_embed_table.return_value = check
        ok = Qwen4ExpForCausalLMMTP._load_cached_fc_embed_table(
            fake, path, None, 8, 4, torch.bfloat16, torch.device("cpu")
        )
        return ok, fake

    def test_load_validates_shape_and_sample(self):
        path = f"{self.tmp.name}/t.pt"
        self.assertFalse(self._load(path, (4, 4, 0.0))[0])  # missing
        torch.save(torch.zeros(8, 4, dtype=torch.bfloat16), path)
        ok, fake = self._load(path, (4, 4, 0.0))
        self.assertTrue(ok)
        self.assertEqual(tuple(fake.fc_embed_table.shape), (8, 4))
        self.assertFalse(self._load(path, (4, 3, 1.0))[0])  # check fails
        torch.save(torch.zeros(7, 4, dtype=torch.bfloat16), path)
        self.assertFalse(self._load(path, (4, 4, 0.0))[0])  # wrong shape


if __name__ == "__main__":
    unittest.main()
