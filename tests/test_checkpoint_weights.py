import unittest

import torch

from openfold.utils.training_utils import (
    DEFAULT_INIT_WEIGHTS_SOURCE,
    INFERENCE_AUTO_ORDER,
    extract_alphafold_weights,
)


class TestCheckpointWeightSources(unittest.TestCase):
    def setUp(self):
        self.module = torch.tensor([1.0])
        self.state_dict = torch.tensor([2.0])
        self.ema = torch.tensor([3.0])
        self.checkpoint = {
            "module": {"model.proj.weight": self.module},
            "state_dict": {"model.proj.weight": self.state_dict},
            "ema": {"params": {"proj.weight": self.ema}},
        }

    def test_explicit_ema_and_module_are_distinct(self):
        ema, ema_source = extract_alphafold_weights(
            self.checkpoint,
            source="ema",
            return_source=True,
        )
        module, module_source = extract_alphafold_weights(
            self.checkpoint,
            source="module",
            return_source=True,
        )
        self.assertIs(ema["proj.weight"], self.ema)
        self.assertIs(module["proj.weight"], self.module)
        self.assertEqual(ema_source, "ema")
        self.assertEqual(module_source, "module")

    def test_weights_only_cli_default_contract_is_ema(self):
        self.assertEqual(DEFAULT_INIT_WEIGHTS_SOURCE, "ema")

    def test_explicit_missing_source_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "requested weight source"):
            extract_alphafold_weights(
                {"module": {"model.x": self.module}},
                source="ema",
            )

    def test_training_auto_preserves_legacy_raw_priority(self):
        weights, source = extract_alphafold_weights(
            self.checkpoint,
            source="auto",
            return_source=True,
        )
        self.assertIs(weights["proj.weight"], self.module)
        self.assertEqual(source, "module")

    def test_inference_auto_prefers_public_ema(self):
        weights, source = extract_alphafold_weights(
            self.checkpoint,
            source="auto",
            auto_order=INFERENCE_AUTO_ORDER,
            return_source=True,
        )
        self.assertIs(weights["proj.weight"], self.ema)
        self.assertEqual(source, "ema")

    def test_inference_auto_loads_plain_merged_state_dict(self):
        plain = {"proj.weight": torch.tensor([4.0])}
        weights, source = extract_alphafold_weights(
            plain,
            source="auto",
            auto_order=INFERENCE_AUTO_ORDER,
            return_source=True,
        )
        self.assertIs(weights["proj.weight"], plain["proj.weight"])
        self.assertEqual(source, "state_dict")


if __name__ == "__main__":
    unittest.main()
