#!/usr/bin/env python3
"""Tests for client0 overfit8 subset construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_client0_overfit_subset import build_overfit_subset, select_overfit8
from scripts.lora_target_registry import OVERFIT8_CLUSTERS, OVERFIT8_LABELS

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs" / "fed_lora_hardcase_fed_v1"
SOURCE = REPO / "outputs" / "fed_lora_fp32"


@unittest.skipUnless(
    (RUN / "clients/client_0/private/splits/train_labels.txt").exists(),
    "client0 hardcase split not prepared",
)
class TestClient0OverfitSubset(unittest.TestCase):
    def test_deterministic_selection_matches_registry(self):
        import csv

        private = RUN / "clients/client_0/private"
        train = [
            line.strip()
            for line in (private / "splits/train_labels.txt").read_text().splitlines()
            if line.strip()
        ]
        with (private / "difficulty/baseline_difficulty.csv").open() as handle:
            difficulty = {
                row["label"].upper(): row
                for row in csv.DictReader(handle)
            }
        selected = select_overfit8(train, difficulty)
        self.assertEqual([lab for lab, _, _ in selected], list(OVERFIT8_LABELS))
        self.assertEqual([c for _, c, _ in selected], list(OVERFIT8_CLUSTERS))
        self.assertTrue(all(tm < 0.5 for _, _, tm in selected))

    def test_build_manifest_no_leakage(self):
        # Build into the real ablation path; idempotent and read-only on splits.
        manifest = build_overfit_subset(run_root=RUN, source_run=SOURCE, link=True)
        self.assertEqual(manifest["n_labels"], 8)
        self.assertEqual(manifest["n_clusters"], 8)
        self.assertTrue(manifest["all_hard"])
        self.assertTrue(manifest["train_only"])
        self.assertFalse(manifest["cluster_leakage_vs_validation"])
        self.assertFalse(manifest["cluster_leakage_vs_development"])
        self.assertEqual(manifest["labels"], list(OVERFIT8_LABELS))
        cache = json.loads(
            Path(manifest["paths"]["chain_cache"]).read_text(encoding="utf-8")
        )
        self.assertEqual(set(cache), set(OVERFIT8_LABELS))


if __name__ == "__main__":
    unittest.main()
