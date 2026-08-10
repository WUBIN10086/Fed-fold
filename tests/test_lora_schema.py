#!/usr/bin/env python3
"""Tests for LoRA schema fingerprints and FedLoRA reject-on-mismatch."""

from __future__ import annotations

import unittest

import torch

from scripts.lora_schema import (
    assert_same_schema,
    fingerprint_from_adapters,
    lora_schema_fingerprint,
    schemas_compatible,
)


class TestLoRASchema(unittest.TestCase):
    def test_same_schema_compatible(self):
        adapters = {
            "input_embedder.linear.lora_A": torch.zeros(4, 8),
            "input_embedder.linear.lora_B": torch.zeros(8, 4),
        }
        fp1 = fingerprint_from_adapters(
            adapters, target="input_embedder", rank=4, alpha=8.0, parent_sha256="abc"
        )
        fp2 = fingerprint_from_adapters(
            adapters, target="input_embedder", rank=4, alpha=8.0, parent_sha256="abc"
        )
        self.assertTrue(schemas_compatible(fp1, fp2))
        assert_same_schema([("c0", fp1), ("c1", fp2)])

    def test_t5a_t5b_fingerprints_differ(self):
        a = {
            "input_embedder.linear.lora_A": torch.zeros(4, 8),
            "input_embedder.linear.lora_B": torch.zeros(8, 4),
        }
        b = {
            "input_embedder.linear.lora_A": torch.zeros(4, 8),
            "input_embedder.linear.lora_B": torch.zeros(8, 4),
            "structure_module.proj.lora_A": torch.zeros(4, 6),
            "structure_module.proj.lora_B": torch.zeros(5, 4),
        }
        fp_a = fingerprint_from_adapters(
            a, target="input_embedder", rank=4, alpha=8.0, parent_sha256="p"
        )
        fp_b = fingerprint_from_adapters(
            b,
            target="structure_module,input_embedder",
            rank=4,
            alpha=8.0,
            parent_sha256="p",
        )
        self.assertFalse(schemas_compatible(fp_a, fp_b))
        with self.assertRaises(ValueError):
            assert_same_schema([("c0", fp_a), ("c1", fp_b)])

    def test_actual_adapter_dtype_changes_fingerprint(self):
        fp32 = {
            "x.lora_A": torch.zeros(4, 8, dtype=torch.float32),
            "x.lora_B": torch.zeros(8, 4, dtype=torch.float32),
        }
        bf16 = {
            key: value.to(torch.bfloat16) for key, value in fp32.items()
        }
        fp_a = fingerprint_from_adapters(
            fp32, target="input_embedder", rank=4, alpha=8.0, parent_sha256="p"
        )
        fp_b = fingerprint_from_adapters(
            bf16, target="input_embedder", rank=4, alpha=8.0, parent_sha256="p"
        )
        self.assertNotEqual(fp_a, fp_b)

    def test_target_string_changes_fingerprint(self):
        shapes = {"x.lora_A": [4, 8], "x.lora_B": [8, 4]}
        fp1 = lora_schema_fingerprint(
            target="input_embedder",
            rank=4,
            alpha=8.0,
            parent_sha256="p",
            adapter_shapes=shapes,
        )
        fp2 = lora_schema_fingerprint(
            target="structure_module",
            rank=4,
            alpha=8.0,
            parent_sha256="p",
            adapter_shapes=shapes,
        )
        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
