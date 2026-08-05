import argparse
import unittest

import torch
import torch.nn as nn

from openfold.model.primitives import Linear
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.lora import LoRAConfig, configure_lora
from openfold.utils.training_utils import (
    build_trainer_kwargs,
    extract_alphafold_weights,
    reset_ema_from_model,
    resolve_learning_rate,
    resolve_lora_config,
    should_reset_ema,
    validate_lora_runtime,
)


class TinyWrapper:
    def __init__(self):
        self.model = nn.Sequential(Linear(3, 2))
        self.ema = ExponentialMovingAverage(self.model, decay=0.999)


class TestTrainingConfiguration(unittest.TestCase):
    def test_accumulate_grad_batches_enters_trainer_kwargs(self):
        def fake_trainer_init(
            self,
            max_epochs=1,
            accumulate_grad_batches=1,
        ):
            pass

        args = argparse.Namespace(
            max_epochs=3,
            accumulate_grad_batches=4,
            flush_logs_every_n_steps=5,
        )
        kwargs, ignored = build_trainer_kwargs(args, fake_trainer_init)
        self.assertEqual(kwargs["accumulate_grad_batches"], 4)
        self.assertEqual(kwargs["max_epochs"], 3)
        self.assertIn("flush_logs_every_n_steps", ignored)

    def test_weights_only_reset_synchronizes_model_and_ema(self):
        wrapper = TinyWrapper()
        with torch.no_grad():
            wrapper.model[0].weight.add_(1.0)
        self.assertFalse(torch.equal(
            wrapper.model.state_dict()["0.weight"],
            wrapper.ema.params["0.weight"],
        ))
        reset_ema_from_model(wrapper, decay=0.999)
        for key, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(wrapper.ema.params[key], value)

    def test_weights_only_prefers_raw_model_and_falls_back_to_public_ema(self):
        raw = torch.ones(2, 3)
        ema = torch.full((2, 3), 2.0)
        checkpoint = {
            "state_dict": {"model.linear.weight": raw},
            "ema": {"params": {"linear.weight": ema}},
        }
        self.assertIs(
            extract_alphafold_weights(checkpoint)["linear.weight"],
            raw,
        )
        public_weights = {
            "ema": {"params": {"linear.weight": ema}},
        }
        self.assertIs(
            extract_alphafold_weights(public_weights)["linear.weight"],
            ema,
        )

    def test_lora_ema_contains_adapters(self):
        wrapper = TinyWrapper()
        wrapper.model.structure_module = nn.Module()
        wrapper.model.structure_module.proj = Linear(3, 2)
        configure_lora(
            wrapper.model,
            LoRAConfig(rank=2, target="structure_module"),
        )
        reset_ema_from_model(wrapper, decay=0.999)
        self.assertIn(
            "structure_module.proj.lora_A",
            wrapper.ema.params,
        )
        self.assertIn(
            "structure_module.proj.lora_B",
            wrapper.ema.params,
        )

    def test_full_resume_uses_checkpoint_config_and_does_not_reset_ema(self):
        restored = resolve_lora_config(
            rank=None,
            alpha=None,
            dropout=None,
            target=None,
            checkpoint_config={
                "rank": 4,
                "alpha": 8.0,
                "dropout": 0.05,
                "target": "structure_module",
            },
            full_checkpoint_resume=True,
        )
        self.assertEqual(restored.rank, 4)
        self.assertFalse(should_reset_ema(full_checkpoint_resume=True))
        self.assertTrue(should_reset_ema(full_checkpoint_resume=False))

    def test_weights_only_lora_checkpoint_uses_stored_config(self):
        restored = resolve_lora_config(
            rank=None,
            alpha=None,
            dropout=None,
            target=None,
            checkpoint_config={
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target": "structure_module",
            },
            full_checkpoint_resume=False,
        )
        self.assertEqual(restored, LoRAConfig(rank=2, alpha=4.0))

    def test_resume_config_mismatch_fails_before_load(self):
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            resolve_lora_config(
                rank=8,
                alpha=None,
                dropout=None,
                target=None,
                checkpoint_config={
                    "rank": 4,
                    "alpha": 8.0,
                    "dropout": 0.05,
                    "target": "structure_module",
                },
                full_checkpoint_resume=True,
            )
        with self.assertRaisesRegex(ValueError, "without lora_config"):
            resolve_lora_config(
                rank=4,
                alpha=8,
                dropout=0.05,
                target="structure_module",
                checkpoint_config=None,
                full_checkpoint_resume=True,
            )

    def test_lora_torchscript_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "not supported with LoRA"):
            validate_lora_runtime(LoRAConfig(rank=4), script_modules=True)
        validate_lora_runtime(LoRAConfig(rank=0), script_modules=True)

    def test_learning_rate_defaults_preserve_old_mode(self):
        self.assertEqual(resolve_learning_rate(None, lora_enabled=False), 1e-3)
        self.assertEqual(resolve_learning_rate(None, lora_enabled=True), 1e-4)
        self.assertEqual(resolve_learning_rate(2e-4, lora_enabled=True), 2e-4)


if __name__ == "__main__":
    unittest.main()
