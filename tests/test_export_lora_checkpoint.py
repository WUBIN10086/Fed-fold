import unittest

import torch

from scripts.export_lora_checkpoint import (
    compare_export_to_base,
    extract_adapter_weights,
    merge_adapters_into_base,
    validate_merged_state_dict,
)


class TestFP32LoRAExport(unittest.TestCase):
    def setUp(self):
        self.base = {
            "target.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "target.bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "untouched.weight": torch.ones(1, 2, dtype=torch.float32),
        }
        self.model_adapters = {
            "target.lora_A": torch.ones(1, 3, dtype=torch.bfloat16),
            "target.lora_B": torch.ones(2, 1, dtype=torch.bfloat16),
        }
        self.ema_adapters = {
            "target.lora_A": torch.full(
                (1, 3), 2.0, dtype=torch.bfloat16
            ),
            "target.lora_B": torch.ones(2, 1, dtype=torch.bfloat16),
        }
        self.checkpoint = {
            "state_dict": {
                "model." + key: value
                for key, value in self.model_adapters.items()
            },
            "ema": {"params": self.ema_adapters},
        }

    def test_model_and_ema_adapter_sources_are_distinct(self):
        model = extract_adapter_weights(self.checkpoint, "model")
        ema = extract_adapter_weights(self.checkpoint, "ema")
        self.assertTrue(torch.equal(
            model["target.lora_A"],
            self.model_adapters["target.lora_A"],
        ))
        self.assertTrue(torch.equal(
            ema["target.lora_A"],
            self.ema_adapters["target.lora_A"],
        ))
        self.assertFalse(torch.equal(
            model["target.lora_A"],
            ema["target.lora_A"],
        ))

    def test_fp32_merge_only_changes_target_weight(self):
        merged, target_keys = merge_adapters_into_base(
            self.base,
            self.model_adapters,
            alpha=2,
            rank=1,
        )
        validate_merged_state_dict(
            merged,
            self.base,
            base=self.base,
            target_keys=target_keys,
        )
        self.assertTrue(all(
            tensor.dtype == torch.float32 for tensor in merged.values()
        ))
        self.assertTrue(torch.equal(
            merged["untouched.weight"],
            self.base["untouched.weight"],
        ))
        self.assertTrue(torch.equal(
            merged["target.bias"],
            self.base["target.bias"],
        ))
        self.assertEqual(
            compare_export_to_base(
                merged,
                self.base,
                target_keys,
            )["changed_keys"],
            ["target.weight"],
        )

    def test_scale_zero_is_all_tensor_bitwise_identity(self):
        merged, target_keys = merge_adapters_into_base(
            self.base,
            self.ema_adapters,
            alpha=2,
            rank=1,
            lora_scale=0,
        )
        report = compare_export_to_base(merged, self.base, target_keys)
        self.assertTrue(report["bitwise_identity"])
        self.assertEqual(report["changed_keys"], [])

    def test_unpaired_and_bad_shapes_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "no LoRA adapters"):
            extract_adapter_weights(
                {"state_dict": {"model.x": torch.ones(1)}},
                "model",
            )
        with self.assertRaisesRegex(ValueError, "Invalid LoRA shapes"):
            merge_adapters_into_base(
                self.base,
                {
                    "target.lora_A": torch.ones(2, 3),
                    "target.lora_B": torch.ones(2, 2),
                },
                alpha=2,
                rank=1,
            )


if __name__ == "__main__":
    unittest.main()
