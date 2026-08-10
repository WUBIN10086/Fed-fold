#!/usr/bin/env python3
"""Tests for capacity / promotion / seed gates and validation routing."""

from __future__ import annotations

import unittest

from scripts.evaluate_target_gate import (
    assert_split_isolation,
    capacity_pass_from_rows,
    capacity_pass_v2_from_rows,
    capacity_status_for_target,
    classify_capacity_failure,
    promotion_does_not_block_targets,
    promotion_gate_from_rows,
    promotion_gate_v2_from_rows,
    resolve_validation_branch,
    seed_confirmation_pass,
    select_seed_confirmation_targets,
    should_enter_full_train_validation,
    should_run_capacity_fallback,
    should_schedule_overfit_targets,
)
from scripts.lora_target_registry import (
    FALLBACK_VALIDATION_TARGETS,
    PRIMARY_OVERFIT_TARGETS,
    PRIMARY_VALIDATION_TARGETS,
)


def _row(
    delta_tm: float,
    difficulty: str = "hard",
    tm_b: float = 0.3,
    label: str = "x",
    cluster_id: str = "c1",
) -> dict:
    return {
        "label": label,
        "difficulty": difficulty,
        "tm_baseline": tm_b,
        "tm_model": tm_b + delta_tm,
        "delta_tm": delta_tm,
        "delta_lddt_ca": 0.0,
        "cluster_id": cluster_id,
    }


def _overfit8(deltas):
    assert len(deltas) == 8
    return [
        _row(d, label=f"L{i}", cluster_id=f"c{i}")
        for i, d in enumerate(deltas)
    ]


class TestTargetGate(unittest.TestCase):
    def test_overfit_schedule_covers_t0_t1_t3_t4(self):
        self.assertEqual(
            should_schedule_overfit_targets({}),
            list(PRIMARY_OVERFIT_TARGETS),
        )
        remaining = should_schedule_overfit_targets(
            {"T0": "capacity_failed", "T1": "capacity_failed"}
        )
        self.assertEqual(remaining, ["T3", "T4"])

    def test_intermediate_capacity_fail_is_nonterminal(self):
        step_results = {
            40: {"pass": False, "mean_delta_tm": 0.0},
            80: {"pass": False, "mean_delta_tm": 0.001},
        }
        status = capacity_status_for_target(step_results, max_steps=200)
        self.assertEqual(status["status"], "in_progress")
        step_results[200] = {"pass": False, "mean_delta_tm": 0.002}
        status = capacity_status_for_target(step_results, max_steps=200)
        self.assertEqual(status["status"], "capacity_failed")

    def test_t3_fail_still_schedules_t4(self):
        scheduled = should_schedule_overfit_targets({"T3": "capacity_failed"})
        self.assertIn("T4", scheduled)

    def test_capacity_v1_criteria(self):
        rows = _overfit8([0.02] * 4 + [0.0] * 4)
        self.assertTrue(capacity_pass_from_rows(rows)["pass"])

    def test_capacity_v2_rejects_single_outlier(self):
        # Mean dominated by one +0.65; four tiny rises keep v1 alive.
        deltas = [0.001, 0.001, 0.001, 0.001, -0.05, -0.05, -0.05, 0.65]
        rows = _overfit8(deltas)
        self.assertTrue(capacity_pass_from_rows(rows)["pass"])  # v1 still passes
        v2 = capacity_pass_v2_from_rows(rows)
        self.assertFalse(v2["pass"])
        self.assertEqual(v2["bootstrap_mean_delta_tm_ci"]["n_bootstrap"], 2000)
        self.assertFalse(v2["criteria"]["leave_max_out_mean_gt_0"])
        self.assertFalse(v2["criteria"]["delta_tm_ge_0p01_at_least_3"])

    def test_capacity_v2_passes_dispersed_gains(self):
        deltas = [0.02, 0.015, 0.014, 0.012, 0.011, 0.01, 0.008, 0.006]
        rows = _overfit8(deltas)
        v2 = capacity_pass_v2_from_rows(rows)
        self.assertTrue(v2["pass"], v2["criteria"])

    def test_promotion_v2_rejects_single_hard_outlier(self):
        rows = (
            [_row(0.65, "hard", label="h0", cluster_id="ch0")]
            + [_row(-0.001, "hard", label=f"h{i}", cluster_id=f"ch{i}") for i in range(1, 4)]
            + [_row(0.0, "medium", tm_b=0.7, label=f"n{i}", cluster_id=f"n{i}") for i in range(6)]
        )
        v1 = promotion_gate_from_rows(rows)
        v2 = promotion_gate_v2_from_rows(rows)
        # v1 may pass on mean; v2 must fail leave-max-out / ge count.
        self.assertFalse(v2["promotion_pass"])
        self.assertIn("promotion_v1", v2)

    def test_promotion_gate_does_not_block_execution(self):
        self.assertTrue(promotion_does_not_block_targets())
        self.assertEqual(list(PRIMARY_VALIDATION_TARGETS), ["T0", "T1", "T2", "T3", "T4"])
        self.assertEqual(list(FALLBACK_VALIDATION_TARGETS), ["T5A", "T5B"])

    def test_seed_failure_blocks_development_signal(self):
        summaries = [
            {
                "hard_mean_delta_tm": 0.001,
                "hard_median_delta_tm": 0.001,
                "nonhard_mean_delta_tm": 0.0,
                "all_mean_delta_lddt": 0.0,
            }
        ] * 3
        self.assertFalse(seed_confirmation_pass(summaries)["pass"])

    def test_select_top_two_seed_candidates(self):
        passers = [
            {
                "target_slug": "T4",
                "hard_mean_delta_tm": 0.02,
                "hard_median_delta_tm": 0.01,
                "all_mean_delta_lddt": 0.0,
                "trainable_params": 273024,
            },
            {
                "target_slug": "T1",
                "hard_mean_delta_tm": 0.015,
                "hard_median_delta_tm": 0.01,
                "all_mean_delta_lddt": 0.0,
                "trainable_params": 43904,
            },
            {
                "target_slug": "T3",
                "hard_mean_delta_tm": 0.018,
                "hard_median_delta_tm": 0.01,
                "all_mean_delta_lddt": 0.0,
                "trainable_params": 103104,
            },
        ]
        selected = select_seed_confirmation_targets(passers, max_targets=2)
        self.assertEqual([r["target_slug"] for r in selected], ["T4", "T3"])

    def test_near_tie_prefers_fewer_params(self):
        passers = [
            {
                "target_slug": "T5B",
                "hard_mean_delta_tm": 0.010,
                "hard_median_delta_tm": 0.01,
                "all_mean_delta_lddt": 0.0,
                "trainable_params": 63196,
            },
            {
                "target_slug": "T5A",
                "hard_mean_delta_tm": 0.009,
                "hard_median_delta_tm": 0.01,
                "all_mean_delta_lddt": 0.0,
                "trainable_params": 19292,
            },
        ]
        selected = select_seed_confirmation_targets(passers, max_targets=2)
        self.assertEqual(selected[0]["target_slug"], "T5A")

    def test_fallback_routing_flags(self):
        failed = {slug: "capacity_failed" for slug in PRIMARY_OVERFIT_TARGETS}
        self.assertTrue(should_run_capacity_fallback(failed))
        self.assertFalse(should_enter_full_train_validation(failed))

    def test_resolve_validation_branch_matrix(self):
        primary_fail = {s: "capacity_failed" for s in PRIMARY_OVERFIT_TARGETS}
        audit_ok = {
            "audit_integrity_pass": True,
            "audit_complete": True,
            "metric_reproducible": True,
            "inference_reproducible": True,
        }
        # audit fail
        r = resolve_validation_branch(primary_fail, {}, {"audit_integrity_pass": False})
        self.assertEqual(r["branch"], "blocked_audit_integrity")
        # integrity alone is insufficient without independent inference replay
        r = resolve_validation_branch(
            primary_fail, {}, {"audit_integrity_pass": True}
        )
        self.assertEqual(r["branch"], "blocked_audit_incomplete")
        # primary pass
        primary = dict(primary_fail)
        primary["T0"] = "capacity_passed"
        r = resolve_validation_branch(primary, {}, audit_ok)
        self.assertEqual(r["branch"], "primary")
        self.assertEqual(r["scheduled_targets"], list(PRIMARY_VALIDATION_TARGETS))
        # t5 standard
        r = resolve_validation_branch(
            primary_fail, {"T5A": {"pass": True}, "T5B": {"pass": False}}, audit_ok
        )
        self.assertEqual(r["branch"], "t5_standard")
        self.assertEqual(r["scheduled_targets"], ["T5A", "T5B"])
        # blocked
        r = resolve_validation_branch(
            primary_fail, {"T5A": {"pass": False}, "T5B": {"pass": False}}, audit_ok
        )
        self.assertEqual(r["branch"], "blocked_no_capacity_v2")
        # diagnostic-only
        r = resolve_validation_branch(
            primary_fail,
            {"T5A": {"pass": False}, "T5B": {"pass": False}},
            audit_ok,
            diagnostic_only=True,
        )
        self.assertEqual(r["branch"], "t5_diagnostic_only")
        self.assertEqual(r["scheduled_targets"], ["T5A"])
        self.assertTrue(r["capacity_v2_failed"])

    def test_split_isolation(self):
        with self.assertRaises(AssertionError):
            assert_split_isolation(
                train_labels=["a"],
                validation_labels=["b"],
                development_labels=["c"],
                accessed_split="development",
                allowing_selection=True,
            )

    def test_promotion_allows_manifested_duplicate_clusters(self):
        rows = [
            _row(0.02, "hard", label="h0", cluster_id="shared"),
            _row(0.015, "hard", label="h1", cluster_id="shared"),
            _row(0.012, "hard", label="h2", cluster_id="c2"),
            _row(0.011, "hard", label="h3", cluster_id="c3"),
            _row(0.0, "medium", label="n0", cluster_id="c4"),
        ]
        gate = promotion_gate_v2_from_rows(
            rows,
            expected_labels=[row["label"] for row in rows],
            expected_clusters=[row["cluster_id"] for row in rows],
        )
        self.assertTrue(gate["checks"]["validation_clusters_match_manifest"])
        self.assertTrue(gate["promotion_pass"], gate["checks"])

    def test_manifest_identity_errors_fail_robust_gates(self):
        rows = _overfit8([0.02] * 8)
        capacity = capacity_pass_v2_from_rows(
            rows,
            expected_labels=[f"L{i}" for i in range(7)] + ["wrong"],
            expected_clusters=[f"c{i}" for i in range(8)],
        )
        self.assertFalse(capacity["pass"])
        self.assertIn("labels_do_not_match_manifest", capacity["validation_errors"])

        validation = rows[:4]
        for row in validation:
            row["difficulty"] = "hard"
        promotion = promotion_gate_v2_from_rows(
            validation,
            expected_labels=["wrong"],
            expected_clusters=[row["cluster_id"] for row in validation],
        )
        self.assertFalse(promotion["promotion_pass"])
        self.assertFalse(promotion["checks"]["validation_labels_match_manifest"])


class TestCapacityFallback(unittest.TestCase):
    def test_failure_classification_routes(self):
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=True,
                positive_control_pass=False,
                positive_control_executed=False,
                loss_decreased=True,
                tm_unchanged=True,
            )["failure_type"],
            "primary_targets_failed_but_t5_fallback_viable",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=False,
                loss_decreased=True,
                tm_unchanged=True,
            )["failure_type"],
            "primary_and_t5_capacity_v2_failed_positive_control_not_executed",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=True,
                positive_control_executed=True,
                loss_decreased=True,
                tm_unchanged=True,
            )["failure_type"],
            "lora_parameterization_or_capacity_insufficient",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=True,
                loss_decreased=None,
                tm_unchanged=True,
            )["failure_type"],
            "positive_control_loss_trend_unavailable",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=True,
                loss_decreased=False,
                tm_unchanged=True,
            )["failure_type"],
            "optimization_backprop_or_data_pipeline_issue",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=True,
                loss_decreased=True,
                tm_unchanged=True,
            )["failure_type"],
            "train_loss_tm_objective_mismatch",
        )
        self.assertEqual(
            classify_capacity_failure(
                t5_any_pass=False,
                positive_control_pass=False,
                positive_control_executed=True,
                loss_decreased=True,
                tm_unchanged=False,
                tm_sparse_response=True,
            )["failure_type"],
            "positive_control_sparse_tm_response_not_robust",
        )


if __name__ == "__main__":
    unittest.main()
