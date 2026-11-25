"""SGLANG_STARTUP_WEIGHT_SHARD_FILTER keeps every shard a model consumes."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    filter_safetensors_files_by_weight_name,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INDEX_NAME = "model.safetensors.index.json"
WEIGHT_MAP = {
    "model.language_model.layers.0.mlp.weight": "a.safetensors",
    "mtp.fc.weight": "b.safetensors",
    "mtp.layers.0.mlp.experts.0.down_proj.weight": "c.safetensors",
    "lm_head.weight": "c.safetensors",
}


class TestWeightShardFilter(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        with open(os.path.join(self.folder, INDEX_NAME), "w") as f:
            json.dump({"weight_map": WEIGHT_MAP}, f)
        self.files = [
            os.path.join(self.folder, n)
            for n in ("a.safetensors", "b.safetensors", "c.safetensors", "x.safetensors")
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _names(self, files):
        return sorted(os.path.basename(f) for f in files)

    def test_keeps_needed_and_unindexed_shards(self):
        kept = filter_safetensors_files_by_weight_name(
            self.files,
            self.folder,
            INDEX_NAME,
            Qwen3_5ForCausalLMMTP.checkpoint_weight_name_filter,
        )
        self.assertEqual(
            self._names(kept), ["b.safetensors", "c.safetensors", "x.safetensors"]
        )

    def test_no_index_keeps_everything(self):
        os.remove(os.path.join(self.folder, INDEX_NAME))
        kept = filter_safetensors_files_by_weight_name(
            self.files, self.folder, INDEX_NAME, lambda n: False
        )
        self.assertEqual(kept, self.files)

    def _source(self, name_filter):
        source = MagicMock()
        source.weight_name_filter = name_filter
        return source

    def test_loader_gate_off_by_default(self):
        source = self._source(Qwen3_5ForCausalLMMTP.checkpoint_weight_name_filter)
        kept = DefaultModelLoader._maybe_filter_shards_by_name(
            source, self.folder, self.files, True
        )
        self.assertEqual(kept, self.files)

    def test_loader_gate_on(self):
        source = self._source(Qwen3_5ForCausalLMMTP.checkpoint_weight_name_filter)
        with envs.SGLANG_STARTUP_WEIGHT_SHARD_FILTER.override(True):
            kept = DefaultModelLoader._maybe_filter_shards_by_name(
                source, self.folder, self.files, True
            )
            self.assertEqual(len(kept), 3)
            # Models without a filter (the target) still read every shard.
            self.assertEqual(
                DefaultModelLoader._maybe_filter_shards_by_name(
                    self._source(None), self.folder, self.files, True
                ),
                self.files,
            )


if __name__ == "__main__":
    unittest.main()
