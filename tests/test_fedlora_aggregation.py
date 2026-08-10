"""Unit tests for FedLoRA effective-delta aggregation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.fedlora_aggregate import (
    aggregate_effective_deltas,
    compute_effective_deltas,
    non_target_max_abs_diff,
    clip_deltas,
)


class TestFedLoRAAggregation(unittest.TestCase):
    def test_effective_delta_matches_ba(self):
        rank = 2
        alpha = 8.0
        a = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
        adapters = {
            "layer.lora_A": a,
            "layer.lora_B": b,
        }
        deltas = compute_effective_deltas(adapters, alpha=alpha, rank=rank)
        expected = (alpha / rank) * (b @ a)
        self.assertTrue(torch.allclose(deltas["layer.weight"], expected))

    def test_average_ab_not_equal_average_ba(self):
        rank = 1
        alpha = 1.0
        a1 = torch.tensor([[1.0, 0.0]])
        b1 = torch.tensor([[1.0], [0.0]])
        a2 = torch.tensor([[0.0, 1.0]])
        b2 = torch.tensor([[0.0], [1.0]])
        d1 = compute_effective_deltas(
            {"w.lora_A": a1, "w.lora_B": b1}, alpha=alpha, rank=rank
        )
        d2 = compute_effective_deltas(
            {"w.lora_A": a2, "w.lora_B": b2}, alpha=alpha, rank=rank
        )
        avg_delta = 0.5 * (d1["w.weight"] + d2["w.weight"])
        avg_a = 0.5 * (a1 + a2)
        avg_b = 0.5 * (b1 + b2)
        avg_ab = (alpha / rank) * (avg_b @ avg_a)
        self.assertFalse(torch.allclose(avg_delta, avg_ab))

    def test_sample_weighted_aggregation_and_nontarget_identity(self):
        parent = {
            "layer.weight": torch.zeros(2, 2),
            "other.bias": torch.tensor([3.0, 4.0]),
            "buffer": torch.tensor([1], dtype=torch.int64),
        }
        d1 = {"layer.weight": torch.tensor([[2.0, 0.0], [0.0, 0.0]])}
        d2 = {"layer.weight": torch.tensor([[0.0, 0.0], [0.0, 4.0]])}
        child, target_keys = aggregate_effective_deltas(
            parent, [d1, d2], weights=[1.0, 3.0]
        )
        # (1*d1 + 3*d2)/4
        expected = torch.tensor([[0.5, 0.0], [0.0, 3.0]])
        self.assertTrue(torch.allclose(child["layer.weight"], expected))
        self.assertEqual(target_keys, ["layer.weight"])
        self.assertEqual(non_target_max_abs_diff(parent, child, target_keys), 0.0)
        self.assertTrue(torch.equal(child["other.bias"], parent["other.bias"]))
        self.assertTrue(torch.equal(child["buffer"], parent["buffer"]))

    def test_openfold_aliases_resolve_to_canonical_parent_keys(self):
        aliases = {
            "evoformer.blocks.44.msa_transition.linear_1.weight":
                "evoformer.blocks.44.core.msa_transition.linear_1.weight",
            "evoformer.blocks.44.pair_stack.tri_mul_in.linear_a_p.weight":
                "evoformer.blocks.44.core.tri_mul_in.linear_a_p.weight",
            "structure_module.ipa.linear_q_points.linear.weight":
                "structure_module.ipa.linear_q_points.weight",
            "structure_module.ipa.linear_kv_points.linear.weight":
                "structure_module.ipa.linear_kv_points.weight",
        }
        parent = {
            canonical: torch.zeros(2, 2)
            for canonical in aliases.values()
        }
        deltas = {
            adapter: torch.full((2, 2), float(index))
            for index, adapter in enumerate(aliases, start=1)
        }
        child, target_keys = aggregate_effective_deltas(
            parent, [deltas], weights=[1.0]
        )
        self.assertEqual(target_keys, sorted(aliases.values()))
        for index, canonical in enumerate(aliases.values(), start=1):
            self.assertTrue(
                torch.equal(child[canonical], torch.full((2, 2), float(index)))
            )
        self.assertEqual(non_target_max_abs_diff(parent, child, target_keys), 0.0)

    def test_alias_shape_mismatch_is_rejected(self):
        parent = {
            "evoformer.blocks.44.core.msa_transition.linear_1.weight":
                torch.zeros(2, 3)
        }
        deltas = {
            "evoformer.blocks.44.msa_transition.linear_1.weight":
                torch.zeros(3, 2)
        }
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            aggregate_effective_deltas(parent, [deltas], weights=[1.0])

    def test_clipping(self):
        deltas = {"w": torch.tensor([[3.0, 4.0]])}  # norm 5
        clipped, norm, was = clip_deltas(deltas, max_norm=1.0)
        self.assertTrue(was)
        self.assertAlmostEqual(norm, 1.0, places=5)
        self.assertTrue(
            torch.allclose(clipped["w"], torch.tensor([[0.6, 0.8]]))
        )

    def test_reject_nonunit_scale(self):
        with self.assertRaises(ValueError):
            compute_effective_deltas(
                {
                    "x.lora_A": torch.ones(1, 2),
                    "x.lora_B": torch.ones(3, 1),
                },
                alpha=1.0,
                rank=1,
                scale=0.5,
            )


if __name__ == "__main__":
    unittest.main()
