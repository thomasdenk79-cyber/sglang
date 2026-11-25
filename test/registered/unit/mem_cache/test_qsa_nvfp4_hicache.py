"""HiCache stays fail-closed for QSA NVFP4 KV on the donor (cf. a22a95b12b).

The donor host mirror sizes rows from the logical head_dim and has no copy of
the token-linear block scales, so the assembler must refuse NVFP4 device pools.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQSANVFP4HiCacheFailClosed(CustomTestCase):
    def _nvfp4_pool(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            QSANVFP4KVCacheMethod,
        )

        return SimpleNamespace(
            quant_method=QSANVFP4KVCacheMethod(num_layers=1, device="cpu")
        )

    def _build(self, **kwargs):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_kv_host_pool,
        )

        return build_kv_host_pool(
            page_size=16, server_args=None, use_mla=False, **kwargs
        )

    def test_rejects_nvfp4_target_pool(self):
        with self.assertRaisesRegex(ValueError, "QSA NVFP4"):
            self._build(kv_pool=self._nvfp4_pool())

    def test_rejects_nvfp4_draft_pool_behind_wrapper(self):
        plain = SimpleNamespace(quant_method=None)
        draft = SimpleNamespace(full_kv_pool=self._nvfp4_pool())
        with self.assertRaisesRegex(ValueError, "QSA NVFP4"):
            self._build(kv_pool=plain, mtp_draft_device_pools=(draft,))


if __name__ == "__main__":
    unittest.main()
