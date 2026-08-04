import copy
import unittest

import torch
import torch.nn as nn

from openfold.model.primitives import Linear
from openfold.utils.lora import (
    LoRAConfig,
    LoRALinear,
    configure_lora,
    inject_lora,
    merge_lora_modules,
    merge_lora_state_dict,
    module_matches_target,
)
from scripts.export_lora_checkpoint import (
    extract_checkpoint_weights,
    resolve_export_lora_config,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure_module = nn.Module()
        self.structure_module.proj = Linear(6, 5)
        self.evoformer = nn.Module()
        self.evoformer.blocks = nn.ModuleList([nn.Module()])
        block = self.evoformer.blocks[0]
        block.msa_att_row = nn.Module()
        block.msa_att_row.mha = nn.Module()
        block.msa_att_row.mha.linear_q = Linear(6, 6, bias=False)
        block.msa_att_row.mha.linear_g = Linear(6, 6)
        block.msa_transition = nn.Module()
        block.msa_transition.linear_1 = Linear(6, 12)


class TestLoRA(unittest.TestCase):
    def test_rank_zero_changes_nothing(self):
        model = ToyModel()
        original = model.structure_module.proj
        inputs = torch.randn(3, 6)
        expected = original(inputs)
        names = configure_lora(model, LoRAConfig(rank=0))
        self.assertEqual(names, [])
        self.assertIs(model.structure_module.proj, original)
        torch.testing.assert_close(model.structure_module.proj(inputs), expected)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_initial_output_and_dtype_match(self):
        base = Linear(6, 5)
        layer = LoRALinear(base, rank=2, alpha=4)
        inputs = torch.randn(3, 6)
        torch.testing.assert_close(layer(inputs), base(inputs), rtol=0, atol=0)
        self.assertEqual(layer(inputs).dtype, inputs.dtype)

        bf16_inputs = inputs.to(torch.bfloat16)
        bf16_layer = copy.deepcopy(layer).to(torch.bfloat16)
        output = bf16_layer(bf16_inputs)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(
            output,
            copy.deepcopy(base).to(torch.bfloat16)(bf16_inputs),
            rtol=0,
            atol=0,
        )

    def test_only_adapters_are_trainable(self):
        model = ToyModel()
        configure_lora(
            model,
            LoRAConfig(rank=2, alpha=4, target="structure_module"),
        )
        trainable = [name for name, p in model.named_parameters()
                     if p.requires_grad]
        self.assertEqual(
            trainable,
            [
                "structure_module.proj.lora_A",
                "structure_module.proj.lora_B",
            ],
        )

    def test_merge_preserves_eval_output(self):
        model = ToyModel()
        configure_lora(
            model,
            LoRAConfig(rank=2, alpha=4, dropout=0.3),
        )
        with torch.no_grad():
            model.structure_module.proj.lora_B.normal_()
        model.eval()
        inputs = torch.randn(4, 6)
        expected = model.structure_module.proj(inputs)
        merged = merge_lora_modules(model)
        self.assertEqual(merged, ["structure_module.proj"])
        self.assertIsInstance(model.structure_module.proj, Linear)
        self.assertNotIsInstance(model.structure_module.proj, LoRALinear)
        torch.testing.assert_close(
            model.structure_module.proj(inputs),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_state_dict_save_and_restore(self):
        config = LoRAConfig(rank=2, alpha=4)
        source = ToyModel()
        configure_lora(source, config)
        with torch.no_grad():
            source.structure_module.proj.lora_B.normal_()
        restored = ToyModel()
        configure_lora(restored, config)
        restored.load_state_dict(source.state_dict())
        inputs = torch.randn(2, 6)
        torch.testing.assert_close(restored.structure_module.proj(inputs),
                                   source.structure_module.proj(inputs))

    def test_state_dict_merge_has_plain_keys(self):
        model = ToyModel()
        base_keys = set(model.state_dict())
        configure_lora(model, LoRAConfig(rank=2, alpha=4))
        merged = merge_lora_state_dict(model.state_dict(), alpha=4, rank=2)
        self.assertEqual(set(merged), base_keys)
        self.assertFalse(any(".lora_" in key for key in merged))

    def test_export_can_select_ema_or_raw_model(self):
        model = ToyModel()
        configure_lora(model, LoRAConfig(rank=2, alpha=4))
        state = model.state_dict()
        checkpoint = {
            "ema": {"params": state, "decay": 0.999},
            "state_dict": {"model." + key: value for key, value in state.items()},
            "lora_config": {
                "rank": 2,
                "alpha": 4.0,
                "dropout": 0.0,
                "target": "structure_module",
            },
        }
        self.assertEqual(
            set(extract_checkpoint_weights(checkpoint, "ema")),
            set(state),
        )
        self.assertEqual(
            set(extract_checkpoint_weights(checkpoint, "model")),
            set(state),
        )
        self.assertEqual(resolve_export_lora_config(checkpoint).rank, 2)

    def test_evoformer_attention_target_is_restricted(self):
        self.assertTrue(module_matches_target(
            "evoformer.blocks.0.msa_att_row.mha.linear_q",
            "evoformer_attention",
        ))
        self.assertFalse(module_matches_target(
            "evoformer.blocks.0.msa_att_row.mha.linear_g",
            "evoformer_attention",
        ))
        self.assertFalse(module_matches_target(
            "evoformer.blocks.0.msa_transition.linear_1",
            "evoformer_attention",
        ))
        self.assertFalse(module_matches_target(
            "structure_module.ipa.linear_q",
            "evoformer_attention",
        ))

    def test_fail_fast_for_zero_match_and_duplicate_injection(self):
        model = ToyModel()
        with self.assertRaisesRegex(ValueError, "matched zero"):
            inject_lora(
                model,
                LoRAConfig(rank=2, target="re:^does_not_exist"),
            )
        inject_lora(model, LoRAConfig(rank=2))
        with self.assertRaisesRegex(ValueError, "already injected"):
            inject_lora(model, LoRAConfig(rank=2))

    def test_fail_fast_for_unpaired_or_bad_shape(self):
        model = ToyModel()
        configure_lora(model, LoRAConfig(rank=2))
        state = dict(model.state_dict())
        del state["structure_module.proj.lora_B"]
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            merge_lora_state_dict(state, alpha=4, rank=2)

        state = dict(model.state_dict())
        state["structure_module.proj.lora_B"] = torch.zeros(5, 3)
        with self.assertRaisesRegex(ValueError, "rank"):
            merge_lora_state_dict(state, alpha=4, rank=2)


if __name__ == "__main__":
    unittest.main()
