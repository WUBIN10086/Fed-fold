#!/usr/bin/env python3
"""End-to-end smoke for target ablation report/gate wiring (no GPU training)."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_target_gate import (
    capacity_pass_from_rows,
    promotion_gate_from_rows,
    should_schedule_overfit_targets,
)
from scripts.lora_target_registry import PRIMARY_OVERFIT_TARGETS, PRIMARY_VALIDATION_TARGETS
from scripts.summarize_client0_target_ablation import summarize


def _write_paired(path: Path, deltas_hard, deltas_nonhard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, d in enumerate(deltas_hard):
        rows.append(
            {
                "label": f"h{i}",
                "client": "client_0",
                "cluster_id": f"c{i}",
                "difficulty": "hard",
                "tm_baseline": 0.3,
                "tm_model": 0.3 + d,
                "delta_tm": d,
                "lddt_baseline": 0.4,
                "lddt_model": 0.4,
                "delta_lddt_ca": 0.0,
            }
        )
    for i, d in enumerate(deltas_nonhard):
        rows.append(
            {
                "label": f"n{i}",
                "client": "client_0",
                "cluster_id": f"n{i}",
                "difficulty": "medium",
                "tm_baseline": 0.7,
                "tm_model": 0.7 + d,
                "delta_tm": d,
                "lddt_baseline": 0.7,
                "lddt_model": 0.7,
                "delta_lddt_ca": 0.0,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestTargetAblationSmoke(unittest.TestCase):
    def test_smoke_report_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "clients/client_0/private/target_ablation_v1"
            private.mkdir(parents=True)
            eval_root = root / "evaluation/client0_target_ablation_v1"

            # Fabricate overfit capacity pass on T4 only; all targets scheduled.
            scheduled = should_schedule_overfit_targets({})
            self.assertEqual(scheduled, list(PRIMARY_OVERFIT_TARGETS))

            details = {}
            statuses = {}
            for slug in PRIMARY_OVERFIT_TARGETS:
                paired = private / slug / "overfit8/seed_42/eval/step_200/scale_1p0/paired_deltas.csv"
                if slug == "T4":
                    _write_paired(paired, [0.02] * 5 + [0.0] * 3, [])
                else:
                    _write_paired(paired, [0.0] * 8, [])
                with paired.open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row["delta_tm"] = float(row["delta_tm"])
                    row["tm_baseline"] = float(row["tm_baseline"])
                    row["tm_model"] = float(row["tm_model"])
                cap = capacity_pass_from_rows(rows)
                statuses[slug] = "capacity_passed" if cap["pass"] else "capacity_failed"
                details[slug] = {
                    "status": statuses[slug],
                    "first_pass_step": 200 if cap["pass"] else None,
                    "completed_steps": [40, 80, 120, 200],
                    "steps": {200: cap},
                }

            (private / "overfit_summary.json").write_text(
                json.dumps({"statuses": statuses, "details": details})
            )
            (private / "capacity_diagnostics.json").write_text(
                json.dumps(
                    {
                        "targets": {},
                        "failure_classification": {
                            "failure_type": None,
                            "action": "proceed_to_full_train_validation_matrix",
                        },
                    }
                )
            )

            results = {}
            for slug in PRIMARY_VALIDATION_TARGETS:
                paired = private / slug / "uniform/seed_42/validation/epoch_5/scale_1p0/paired_deltas.csv"
                # Give T4 a promotion pass; others fail promotion but still executed.
                if slug == "T4":
                    _write_paired(paired, [0.02, 0.01, 0.01], [0.0] * 7)
                else:
                    _write_paired(paired, [0.0, 0.0, 0.0], [0.0] * 7)
                with paired.open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row["delta_tm"] = float(row["delta_tm"])
                    row["tm_baseline"] = float(row["tm_baseline"])
                    row["tm_model"] = float(row["tm_model"])
                    row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
                gate = promotion_gate_from_rows(rows)
                results[slug] = gate

            (private / "validation_summary.json").write_text(
                json.dumps(
                    {
                        "blocked": False,
                        "results": results,
                        "paired_target_deltas": {
                            "T3_minus_T1_hard_mean": 0.0,
                            "T4_minus_T3_hard_mean": 0.02,
                        },
                        "promotion_blocks_execution": False,
                    }
                )
            )
            (private / "seed_confirmation.json").write_text(
                json.dumps(
                    {
                        "blocks_development": True,
                        "blocks_fedlora": True,
                        "unique_target": None,
                        "per_target": {},
                    }
                )
            )
            (private / "development_summary.json").write_text(
                json.dumps(
                    {
                        "blocked": True,
                        "reason": "client0_target_ablation_no_stable_generalization_signal",
                    }
                )
            )

            payload = summarize(run_root=root, eval_root=eval_root)
            self.assertTrue((eval_root / "overfit8_target_summary.csv").exists())
            self.assertTrue((eval_root / "validation_target_summary.csv").exists())
            self.assertTrue((eval_root / "target_ablation_report.md").exists())
            report = (eval_root / "target_ablation_report.md").read_text(encoding="utf-8")
            self.assertIn("Train memorization", report)
            self.assertIn("Validation generalization", report)
            self.assertIn("Development confirmation", report)
            self.assertEqual(statuses["T4"], "capacity_passed")
            self.assertTrue(results["T4"]["promotion_pass"])
            self.assertFalse(results["T0"]["promotion_pass"])
            self.assertIsNone(payload["unique_target"])


if __name__ == "__main__":
    unittest.main()
