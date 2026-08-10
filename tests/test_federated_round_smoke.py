"""Integration-style smoke tests for federated round lineage helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.fedlora_aggregate import aggregate_round, load_pure_weights, sha256_file
from scripts.select_global_round import select_global_best
from scripts.evaluate_hardcase_metrics import (
    macro_from_server_summaries,
    server_summary_from_paired,
)


def _fake_lora_checkpoint(path: Path, a: torch.Tensor, b: torch.Tensor) -> None:
    # Minimal DeepSpeed-like / Lightning-like payload accepted by extract_adapter_weights.
    state = {
        "model.layer.weight": torch.zeros(b.shape[0], a.shape[1]),
        "model.layer.lora_A": a,
        "model.layer.lora_B": b,
        "model.other.bias": torch.tensor([1.0, 2.0, 3.0][: b.shape[0]]),
    }
    # Pad other.bias to stay independent of shapes in parent.
    payload = {
        "state_dict": state,
        "lora_config": {
            "rank": int(a.shape[0]),
            "alpha": 2.0,
            "dropout": 0.0,
            "target": "layer",
        },
    }
    torch.save(payload, str(path))


class TestFederatedRoundSmoke(unittest.TestCase):
    def test_two_client_round_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = {
                "layer.weight": torch.zeros(2, 2),
                "other.bias": torch.tensor([9.0, 8.0]),
            }
            parent_path = root / "parent.pt"
            torch.save(parent, str(parent_path))

            c0 = root / "c0.pt"
            c1 = root / "c1.pt"
            _fake_lora_checkpoint(
                c0,
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [0.0]]),
            )
            _fake_lora_checkpoint(
                c1,
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[0.0], [1.0]]),
            )

            child, manifest = aggregate_round(
                parent_path=parent_path,
                client_checkpoints=[c0, c1],
                weights=[1.0, 1.0],
                round_id=1,
                client_ids=["client_0", "client_1"],
                rank=1,
                alpha=2.0,
            )
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["non_target_max_abs_diff"], 0.0)
            self.assertTrue(torch.equal(child["other.bias"], parent["other.bias"]))

            child_path = root / "child.pt"
            torch.save(child, str(child_path))
            reloaded = load_pure_weights(child_path)
            self.assertIn("layer.weight", reloaded)
            self.assertEqual(sha256_file(parent_path), manifest["parent_global_sha"])

            paired = [
                {
                    "label": "A",
                    "cluster_id": "c1",
                    "difficulty": "hard",
                    "tm_baseline": 0.2,
                    "tm_model": 0.3,
                    "delta_tm": 0.1,
                    "delta_lddt_ca": 0.01,
                },
                {
                    "label": "B",
                    "cluster_id": "c2",
                    "difficulty": "easy",
                    "tm_baseline": 0.9,
                    "tm_model": 0.9,
                    "delta_tm": 0.0,
                    "delta_lddt_ca": 0.0,
                },
            ]
            s0 = server_summary_from_paired("client_0", 1, "sha", paired)
            s1 = server_summary_from_paired("client_1", 1, "sha", paired)
            # Ensure server summary has no label fields.
            self.assertNotIn("label", s0)
            self.assertNotIn("native", json.dumps(s0))
            macro = macro_from_server_summaries([s0, s1])
            macro["round_id"] = 1
            baseline = {
                "round_id": 0,
                "macro_mean_delta_tm_hard": 0.0,
                "macro_mean_delta_tm_nonhard": 0.0,
                "min_hard_clusters": 1,
            }
            decision = select_global_best([baseline, macro])
            self.assertEqual(decision["selected_round_id"], 1)

    def test_incomplete_client_summary_blocks_macro_and_selection(self):
        complete = {
            "client_id": "client_0",
            "hard_count": 1,
            "hard_cluster_count": 1,
            "sum_delta_tm_hard": 0.1,
            "nonhard_count": 1,
            "sum_delta_tm_nonhard": 0.0,
            "metric_status": "ok",
        }
        missing = dict(complete)
        missing["client_id"] = "client_1"
        missing["metric_status"] = "missing_private_paired_csv"
        macro = macro_from_server_summaries([complete, missing])
        self.assertEqual(macro["metric_status"], "incomplete")
        self.assertIsNone(macro["macro_mean_delta_tm_hard"])
        macro["round_id"] = 1
        with self.assertRaisesRegex(ValueError, "No complete round summaries"):
            select_global_best([macro])

    def test_mismatch_parent_sha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_path = root / "parent.pt"
            torch.save({"layer.weight": torch.zeros(2, 2)}, str(parent_path))
            ckpt = root / "c0.pt"
            _fake_lora_checkpoint(
                ckpt,
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [0.0]]),
            )
            with self.assertRaises(ValueError):
                aggregate_round(
                    parent_path=parent_path,
                    client_checkpoints=[ckpt],
                    weights=[1.0],
                    expected_parent_sha="deadbeef",
                    rank=1,
                    alpha=2.0,
                )


if __name__ == "__main__":
    unittest.main()
