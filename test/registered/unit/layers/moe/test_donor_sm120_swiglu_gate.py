import logging
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as m

KERNEL = "sglang.kernels.ops.gemm.sm120_nvfp4_moe"


def _probe(env):
    code = (
        "import sys; from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as m; "
        f"print(m._DONOR_SM120_SWIGLU, m._DONOR_SM120_TRACE, {KERNEL!r} in sys.modules)"
    )
    full = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_DONOR_SM120_")}
    full.update(env, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, check=True)
    return out.stdout.split()[-3:]


class DonorSm120GateTest(unittest.TestCase):
    def test_switch_off_keeps_original_path(self):
        self.assertEqual(_probe({}), ["False", "False", "False"])

    def test_switch_on_parses_without_eager_import(self):
        env = {"SGLANG_DONOR_SM120_SWIGLU": "1", "SGLANG_DONOR_SM120_TRACE": "1"}
        self.assertEqual(_probe(env), ["True", "True", "False"])

    def test_unavailable_kernel_falls_back_and_warns_once(self):
        with mock.patch.object(m, "_DONOR_SM120_SWIGLU", True), mock.patch.dict(
            sys.modules, {KERNEL: None}
        ):
            with self.assertLogs(m.logger, logging.WARNING) as logs:
                self.assertIsNone(m._try_donor_sm120(output_dtype=torch.bfloat16))
            self.assertFalse(m._DONOR_SM120_SWIGLU)
        self.assertEqual(len(logs.records), 1)

    def test_unsupported_shape_returns_none_without_jit(self):
        from sglang.kernels.ops.gemm import sm120_nvfp4_moe as k

        with mock.patch.object(k, "nvfp4_moe_swiglu") as run:
            x = torch.zeros(4, 100, dtype=torch.bfloat16)
            self.assertIsNone(
                k.try_nvfp4_moe_swiglu(
                    x, torch.zeros(1), torch.zeros(1), [None] * 6, None, None,
                    output_dtype=torch.bfloat16,
                )
            )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
