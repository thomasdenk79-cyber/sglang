"""JIT caches that must follow SGLANG_CACHE_DIR, so a mounted cache survives restarts."""

import os
import sys
import unittest

from sglang.srt.environ import envs, third_party_cache_defaults
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStartupCacheDirs(CustomTestCase):
    def test_jit_caches_follow_sglang_cache_dir(self):
        base = os.path.join(os.getcwd(), "mounted-cache")
        with envs.SGLANG_CACHE_DIR.override(base):
            defaults = third_party_cache_defaults()
            self.assertEqual(envs.SGLANG_JIT_CACHE_DIR.get(), os.path.join(base, "jit"))
        self.assertEqual(defaults["CUTE_DSL_CACHE_DIR"], os.path.join(base, "cute_dsl"))
        torch_ext = defaults["TORCH_EXTENSIONS_DIR"]
        self.assertEqual(
            os.path.dirname(torch_ext), os.path.join(base, "torch_extensions")
        )
        # Per-python/torch ABI tag, like the one torch appends to its own root.
        py_tag = f"py{sys.version_info.major}{sys.version_info.minor}_"
        self.assertTrue(os.path.basename(torch_ext).startswith(py_tag), torch_ext)

    def test_explicit_jit_cache_dir_wins(self):
        with envs.SGLANG_JIT_CACHE_DIR.override("/explicit/jit"):
            self.assertEqual(envs.SGLANG_JIT_CACHE_DIR.get(), "/explicit/jit")


if __name__ == "__main__":
    unittest.main()
