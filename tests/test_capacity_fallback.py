#!/usr/bin/env python3
"""Tests for capacity fallback routing when all primary overfit targets fail."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch.nn as nn

from scripts.evaluate_target_gate import (
    classify_capacity_failure,
    should_run_capacity_fallback,
)
from scripts.lora_target_registry import PRIMARY_OVERFIT_TARGETS
from scripts.positive_control_structure_module import (
    positive_control_status,
    selective_unfreeze_structure_module,
    should_run_real_positive_control,
)
from scripts.summarize_client0_target_ablation import summarize


class TestCapacityFallback(unittest.TestCase):
    def test_all_failed_triggers_fallback_flag(self):
        statuses = {slug: "capacity_failed" for slug in PRIMARY_OVERFIT_TARGETS}
        self.assertTrue(should_run_capacity_fallback(statuses))

    def test_selective_control_only_unfreezes_structure_module(self):
        model = nn.Module()
        model.upstream = nn.Linear(3, 3)
        model.structure_module = nn.Sequential(nn.Linear(3, 2))
        names = selective_unfreeze_structure_module(model)
        self.assertTrue(names)
        self.assertFalse(model.upstream.weight.requires_grad)
        self.assertTrue(model.structure_module[0].weight.requires_grad)

    def test_setup_only_positive_control_not_classified_as_pass(self):
        status = positive_control_status(setup={"ok": True}, executed=False)
        self.assertEqual(status["state"], "setup_only")
        self.assertFalse(status["pass"])
        self.assertFalse(status["executed"])

    def test_should_run_real_positive_control(self):
        self.assertTrue(
            should_run_real_positive_control(
                t5_capacity_v2_any_pass=False,
                diagnostic_validation_pass=False,
                diagnostic_validation_executed=True,
                audit_pass=True,
            )
        )
        # A standard T5 capacity pass followed by validation failure also triggers it.
        self.assertTrue(
            should_run_real_positive_control(
                t5_capacity_v2_any_pass=True,
                diagnostic_validation_pass=False,
                diagnostic_validation_executed=True,
                audit_pass=True,
            )
        )
        self.assertFalse(
            should_run_real_positive_control(
                t5_capacity_v2_any_pass=False,
                diagnostic_validation_pass=False,
                diagnostic_validation_executed=False,
                audit_pass=True,
            )
        )
        self.assertFalse(
            should_run_real_positive_control(
                t5_capacity_v2_any_pass=False,
                diagnostic_validation_pass=False,
                diagnostic_validation_executed=True,
                audit_pass=False,
            )
        )

    def test_unknown_loss_trend_has_explicit_state(self):
        status = positive_control_status(
            setup={"ok": True},
            executed=True,
            capacity_pass=False,
            loss_decreased=None,
            tm_unchanged=True,
        )
        self.assertEqual(status["state"], "executed_fail_loss_trend_unknown")

    def test_sparse_tm_response_has_explicit_state(self):
        status = positive_control_status(
            setup={"ok": True},
            executed=True,
            capacity_pass=False,
            loss_decreased=True,
            tm_unchanged=False,
            tm_sparse_response=True,
        )
        self.assertEqual(status["state"], "executed_fail_sparse_tm_response")

    def test_report_includes_explicit_failure_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "clients/client_0/private/target_ablation_v1"
            private.mkdir(parents=True)
            eval_root = root / "evaluation/client0_target_ablation_v1"
            classification = classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=False,
                loss_decreased=True,
                tm_unchanged=True,
            )
            (private / "overfit_summary.json").write_text(
                json.dumps(
                    {
                        "statuses": {
                            slug: "capacity_failed" for slug in PRIMARY_OVERFIT_TARGETS
                        },
                        "details": {},
                    }
                )
            )
            (private / "capacity_diagnostics.json").write_text(
                json.dumps(
                    {
                        "targets": {},
                        "fallback": {
                            "t5_results": {"T5A": {"pass": False}, "T5B": {"pass": False}},
                            "positive_control": {
                                "state": "setup_only",
                                "executed": False,
                                "pass": False,
                            },
                        },
                        "failure_classification": classification,
                        "blocks_other_clients": True,
                        "blocks_fedlora": True,
                        "test_accessed": False,
                    }
                )
            )
            (private / "validation_summary.json").write_text(
                json.dumps({"results": {}, "blocked": True, "status": "blocked"})
            )
            (private / "seed_confirmation.json").write_text(
                json.dumps(
                    {
                        "blocks_development": True,
                        "blocks_fedlora": True,
                        "blocks_other_clients": True,
                    }
                )
            )
            (private / "development_summary.json").write_text(
                json.dumps({"blocked": True})
            )
            summarize(run_root=root, eval_root=eval_root)
            report = (eval_root / "target_ablation_report.md").read_text(encoding="utf-8")
            self.assertIn(
                "primary_and_t5_capacity_v2_failed_positive_control_not_executed",
                report,
            )
            self.assertIn("setup_only", report)
            self.assertIn("blocks_fedlora", report.lower() + report)


if __name__ == "__main__":
    unittest.main()
