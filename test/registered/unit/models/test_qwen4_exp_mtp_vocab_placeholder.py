"""Placeholder vocab weights of the Qwen4-Exp MTP draft (SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS).

CPU only: the gate needs the flag, a worker that shares embedding and head, a
single target pipeline stage and untied embeddings; the placeholder keeps the meta-built weight's attributes but holds
one row; the loader filter drops exactly the vocab tensors; and the draft refuses
to run on placeholders when the worker cannot share both target tensors.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models import qwen4_exp_mtp
from sglang.srt.models.qwen4_exp_mtp import (
    Qwen4ExpForCausalLMMTP,
    _build_with_placeholder_vocab_weight,
    _draft_vocab_weights_are_shared,
    _is_draft_vocab_weight,
    _is_mtp_draft_weight_without_vocab,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VocabModule(nn.Module):
    def __init__(self, rows=1000, cols=8, with_buffer=False):
        super().__init__()
        weight = nn.Parameter(torch.empty(rows, cols), requires_grad=False)
        weight.output_dim = 0
        weight.weight_loader = self.weight_loader
        self.register_parameter("weight", weight)
        if with_buffer:
            self.register_buffer("table", torch.zeros(rows))

    def weight_loader(self, param, loaded):
        param.data.copy_(loaded)


class TestDraftVocabGate(CustomTestCase):
    def _gate(self, flag=True, algorithm="NEXTN", pp_size=1, tied=False):
        spec = SimpleNamespace(speculative_algorithm=algorithm)
        server_args = SimpleNamespace(pp_size=pp_size)
        config = SimpleNamespace(tie_word_embeddings=tied)
        with (
            envs.SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS.override(flag),
            mock.patch.object(qwen4_exp_mtp, "get_spec", return_value=spec),
            mock.patch.object(
                qwen4_exp_mtp, "get_server_args", return_value=server_args
            ),
        ):
            return _draft_vocab_weights_are_shared(config)

    def test_needs_flag_algorithm_single_stage_and_untied(self):
        self.assertTrue(self._gate())
        self.assertFalse(self._gate(flag=False))
        self.assertFalse(self._gate(algorithm=None))
        self.assertFalse(self._gate(pp_size=2))
        self.assertFalse(self._gate(tied=True))

    def test_only_workers_that_share_embed_and_head(self):
        self.assertTrue(self._gate(algorithm="EAGLE"))
        self.assertTrue(self._gate(algorithm="FROZEN_KV_MTP"))
        self.assertTrue(self._gate(algorithm=SpeculativeAlgorithm.EAGLE))
        for algorithm in ("EAGLE3", "STANDALONE", "NGRAM", "DFLASH"):
            with self.subTest(algorithm=algorithm):
                self.assertFalse(self._gate(algorithm=algorithm))

    def test_reads_target_pp_size_not_patched_draft_group(self):
        # The draft group always reports one stage; the server args do not.
        with mock.patch.object(
            qwen4_exp_mtp, "get_parallel", side_effect=AssertionError
        ):
            self.assertFalse(self._gate(pp_size=4))


class TestPlaceholderVocabWeight(CustomTestCase):
    def test_one_row_with_the_built_attributes(self):
        module = _build_with_placeholder_vocab_weight(lambda: _VocabModule(1000, 8))
        weight = module.weight
        self.assertFalse(weight.is_meta)
        self.assertEqual(tuple(weight.shape), (1, 8))
        self.assertEqual(weight.dtype, torch.float32)
        self.assertEqual(weight.output_dim, 0)
        self.assertEqual(weight.weight_loader.__self__, module)
        self.assertEqual(module.skipped_weight_nbytes, 1000 * 8 * 4)

    def test_rejects_modules_that_keep_meta_tensors(self):
        with self.assertRaisesRegex(ValueError, "meta tensors"):
            _build_with_placeholder_vocab_weight(
                lambda: _VocabModule(1000, 8, with_buffer=True)
            )


class TestDraftVocabWeightFilter(CustomTestCase):
    def test_drops_only_vocab_tensors(self):
        dropped = [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "mtp.embed_tokens.weight",
            "mtp.shared_head.head.weight",
            "mtp.lm_head.weight",
        ]
        kept = [
            "mtp.fc_embedding.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.self_attn.qkv_proj.weight",
        ]
        for name in dropped:
            self.assertTrue(_is_draft_vocab_weight(name), name)
            self.assertFalse(_is_mtp_draft_weight_without_vocab(name), name)
        for name in kept:
            self.assertFalse(_is_draft_vocab_weight(name), name)
            self.assertTrue(_is_mtp_draft_weight_without_vocab(name), name)
        self.assertFalse(_is_mtp_draft_weight_without_vocab("lm_head.weight"))

    def test_load_weights_skips_vocab_tensors(self):
        draft = Qwen4ExpForCausalLMMTP.__new__(Qwen4ExpForCausalLMMTP)
        nn.Module.__init__(draft)
        draft.skip_vocab_weights = True
        weights = [
            ("model.language_model.embed_tokens.weight", torch.zeros(1)),
            ("mtp.fc_embedding.weight", torch.zeros(1)),
        ]
        with mock.patch.object(
            qwen4_exp_mtp.Qwen3_5ForCausalLMMTP,
            "load_weights",
            side_effect=lambda w, is_mtp: [name for name, _ in w],
        ):
            loaded = draft.load_weights(iter(weights))
        self.assertEqual(loaded, ["mtp.fc_embedding.weight"])


class TestSetEmbedAndHead(CustomTestCase):
    def test_placeholders_need_both_target_tensors(self):
        draft = Qwen4ExpForCausalLMMTP.__new__(Qwen4ExpForCausalLMMTP)
        nn.Module.__init__(draft)
        draft.skip_vocab_weights = True
        tensor = torch.zeros(2, 2)
        for embed, head in ((None, tensor), (tensor, None)):
            with self.subTest(embed=embed is None, head=head is None):
                with self.assertRaisesRegex(ValueError, "placeholders"):
                    draft.set_embed_and_head(embed, head)

    def test_forward_refuses_placeholders_until_shared(self):
        draft = Qwen4ExpForCausalLMMTP.__new__(Qwen4ExpForCausalLMMTP)
        nn.Module.__init__(draft)
        draft.skip_vocab_weights = True
        draft.vocab_weights_shared = False
        with self.assertRaisesRegex(RuntimeError, "shared"):
            draft.forward(None, None, None)
        with mock.patch.object(
            qwen4_exp_mtp.Qwen3_5ForCausalLMMTP, "set_embed_and_head"
        ):
            draft.set_embed_and_head(torch.zeros(2, 2), torch.zeros(2, 2))
        self.assertTrue(draft.vocab_weights_shared)


if __name__ == "__main__":
    unittest.main()
