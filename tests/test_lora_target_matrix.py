#!/usr/bin/env python3
"""Tests for LoRA target matrix registry and matching."""

from __future__ import annotations

import unittest

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.lora import LoRAConfig, configure_lora, module_matches_target, parameter_counts
from scripts.lora_target_registry import TARGET_SPECS, get_target


class TestLoRATargetMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AlphaFold(model_config("seq_model_esm1b_ptm", train=True)).float()

    def test_prefix_targets_match_expected_counts(self):
        for slug, spec in TARGET_SPECS.items():
            with self.subTest(slug=slug):
                model = AlphaFold(model_config("seq_model_esm1b_ptm", train=True)).float()
                replaced = configure_lora(
                    model,
                    LoRAConfig(rank=4, alpha=8.0, dropout=0.0, target=spec.target),
                )
                counts = parameter_counts(model)
                self.assertEqual(len(replaced), spec.expected_modules, slug)
                self.assertEqual(int(counts["trainable"]), spec.expected_trainable, slug)

    def test_comma_prefixes_are_valid(self):
        self.assertTrue(
            module_matches_target(
                "structure_module.ipa.linear_q",
                get_target("T0").target,
            )
        )
        self.assertTrue(
            module_matches_target(
                "evoformer.blocks.47.msa_att_row.mha.linear_q",
                get_target("T3").target,
            )
        )
        self.assertFalse(
            module_matches_target(
                "evoformer.blocks.40.msa_att_row.mha.linear_q",
                get_target("T3").target,
            )
        )

    def test_bad_target_fail_fast(self):
        model = AlphaFold(model_config("seq_model_esm1b_ptm", train=True)).float()
        with self.assertRaises(ValueError):
            configure_lora(
                model,
                LoRAConfig(rank=4, alpha=8.0, target="does_not_exist_module"),
            )


if __name__ == "__main__":
    unittest.main()
