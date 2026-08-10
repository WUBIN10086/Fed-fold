#!/usr/bin/env python3
"""Capacity, promotion, and seed-confirmation gates for target ablation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scripts.evaluate_hardcase_metrics import summarize_group
from scripts.lora_target_registry import (
    CAPACITY_CHECKPOINTS,
    FALLBACK_TARGETS,
    FALLBACK_VALIDATION_TARGETS,
    PRIMARY_OVERFIT_TARGETS,
    PRIMARY_VALIDATION_TARGETS,
)


def read_paired(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    return rows


def _median(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.median(vals))


def leave_max_out_mean(deltas: Sequence[float]) -> Optional[float]:
    if len(deltas) < 2:
        return None
    ordered = sorted(float(x) for x in deltas)
    trimmed = ordered[:-1]  # drop maximum
    return sum(trimmed) / len(trimmed)


def leave_one_out_means(deltas: Sequence[float]) -> List[dict]:
    vals = [float(x) for x in deltas]
    out = []
    for i, dropped in enumerate(vals):
        keep = vals[:i] + vals[i + 1 :]
        out.append(
            {
                "dropped_index": i,
                "dropped_delta_tm": dropped,
                "mean_delta_tm": sum(keep) / len(keep) if keep else None,
            }
        )
    return out


def cluster_bootstrap_mean_ci(
    rows: Sequence[dict],
    *,
    value_key: str = "delta_tm",
    cluster_key: str = "cluster_id",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Deterministic cluster bootstrap, reported for context only."""
    grouped: Dict[str, List[float]] = {}
    for index, row in enumerate(rows):
        cluster = str(row.get(cluster_key) or f"missing_{index}")
        grouped.setdefault(cluster, []).append(float(row[value_key]))
    clusters = sorted(grouped)
    if not clusters:
        return {"low": None, "high": None, "n_bootstrap": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [value for cluster in sampled for value in grouped[cluster]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    low_index = int(0.025 * (n_bootstrap - 1))
    high_index = int(0.975 * (n_bootstrap - 1))
    return {
        "low": estimates[low_index],
        "high": estimates[high_index],
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "method": "cluster_bootstrap_mean_informational_only",
    }


def max_point_contribution(deltas: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in deltas]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if mean <= 1e-12:
        return None
    return max(vals) / (mean * len(vals))


def capacity_pass_from_rows(rows: Sequence[dict]) -> dict:
    """Legacy capacity_v1 gate (kept for traceability; do not overwrite)."""
    summary = summarize_group(list(rows), "")
    mean_dtm = summary["mean_delta_tm"]
    ge_count = summary["delta_tm_ge_0p01_count"]
    rise_count = summary["tm_rise_count"]
    passed = (
        mean_dtm is not None
        and mean_dtm >= 0.01
        and ge_count >= 1
        and rise_count >= 4
    )
    return {
        "pass": bool(passed),
        "gate_version": "capacity_v1",
        "mean_delta_tm": mean_dtm,
        "median_delta_tm": summary.get("median_delta_tm"),
        "delta_tm_ge_0p01_count": ge_count,
        "tm_rise_count": rise_count,
        "mean_delta_lddt": summary["mean_delta_lddt"],
        "rescue_rate": summary["rescue_rate"],
        "criteria": {
            "mean_delta_tm_ge_0p01": bool(mean_dtm is not None and mean_dtm >= 0.01),
            "at_least_one_delta_tm_ge_0p01": ge_count >= 1,
            "tm_rise_at_least_4_of_8": rise_count >= 4,
        },
    }


def capacity_pass_v2_from_rows(
    rows: Sequence[dict],
    *,
    expected_labels: Optional[Sequence[str]] = None,
    expected_clusters: Optional[Sequence[str]] = None,
) -> dict:
    """Robust overfit capacity gate; single extreme sample cannot dominate."""
    labels = [r.get("label") for r in rows]
    clusters = [r.get("cluster_id") for r in rows]
    errors = []
    if len(rows) != 8:
        errors.append(f"expected_8_rows_got_{len(rows)}")
    if len(set(labels)) != len(labels):
        errors.append("duplicate_labels")
    if any(label in (None, "") for label in labels):
        errors.append("missing_label")
    if any(c in (None, "") for c in clusters):
        errors.append("missing_cluster_id")
    elif len(set(clusters)) != len(clusters):
        errors.append("duplicate_clusters")
    if expected_labels is not None and sorted(labels) != sorted(expected_labels):
        errors.append("labels_do_not_match_manifest")
    if expected_clusters is not None and sorted(clusters) != sorted(expected_clusters):
        errors.append("clusters_do_not_match_manifest")
    for row in rows:
        if row.get("delta_tm") is None:
            errors.append(f"missing_delta_tm:{row.get('label')}")

    deltas = [float(r["delta_tm"]) for r in rows]
    summary = summarize_group(list(rows), "")
    mean_dtm = summary["mean_delta_tm"]
    median_dtm = summary["median_delta_tm"]
    lmo_mean = leave_max_out_mean(deltas)
    ge_count = summary["delta_tm_ge_0p01_count"]
    rise_count = summary["tm_rise_count"]

    criteria = {
        "n_rows_eq_8": len(rows) == 8 and not errors,
        "mean_delta_tm_ge_0p01": bool(mean_dtm is not None and mean_dtm >= 0.01),
        "median_delta_tm_gt_0": bool(median_dtm is not None and median_dtm > 0),
        "leave_max_out_mean_gt_0": bool(lmo_mean is not None and lmo_mean > 0),
        "delta_tm_ge_0p01_at_least_3": ge_count >= 3,
        "tm_rise_at_least_4_of_8": rise_count >= 4,
    }
    passed = all(criteria.values()) and not errors
    return {
        "pass": bool(passed),
        "gate_version": "capacity_v2",
        "mean_delta_tm": mean_dtm,
        "median_delta_tm": median_dtm,
        "leave_max_out_mean_delta_tm": lmo_mean,
        "delta_tm_ge_0p01_count": ge_count,
        "tm_rise_count": rise_count,
        "mean_delta_lddt": summary["mean_delta_lddt"],
        "rescue_rate": summary["rescue_rate"],
        "bootstrap_mean_delta_tm_ci": cluster_bootstrap_mean_ci(rows),
        "max_point_contribution": max_point_contribution(deltas),
        "leave_one_out": leave_one_out_means(deltas),
        "validation_errors": errors,
        "criteria": criteria,
    }


def capacity_status_for_target(
    step_results: Dict[int, dict],
    max_steps: int = 200,
    pass_key: str = "pass",
) -> dict:
    ordered = sorted(int(s) for s in step_results)
    first_pass_step = None
    for step in ordered:
        if step_results[step].get(pass_key):
            first_pass_step = step
            break
    final = step_results.get(max_steps)
    if first_pass_step is not None:
        status = "capacity_passed"
    elif max_steps in step_results and not step_results[max_steps].get(pass_key):
        status = "capacity_failed"
    else:
        status = "in_progress"
    if status == "capacity_failed" and max_steps not in step_results:
        status = "in_progress"
    return {
        "status": status,
        "first_pass_step": first_pass_step,
        "completed_steps": ordered,
        "final_step_result": final,
        "pass_key": pass_key,
        "note": (
            "intermediate step fails are non-terminal; only max_steps can "
            "finalize capacity_failed"
        ),
    }


def should_schedule_overfit_targets(
    completed: Optional[Dict[str, str]] = None,
) -> List[str]:
    completed = completed or {}
    scheduled = []
    for slug in PRIMARY_OVERFIT_TARGETS:
        status = completed.get(slug)
        if status in {"capacity_passed", "capacity_failed", "skipped_early_pass"}:
            continue
        scheduled.append(slug)
    return scheduled


def should_run_capacity_fallback(statuses: Dict[str, str]) -> bool:
    return all(
        statuses.get(slug) == "capacity_failed" for slug in PRIMARY_OVERFIT_TARGETS
    )


def should_enter_full_train_validation(statuses: Dict[str, str]) -> bool:
    """Legacy helper: primary capacity only. Prefer resolve_validation_branch."""
    return any(
        statuses.get(slug) == "capacity_passed" for slug in PRIMARY_OVERFIT_TARGETS
    )


def resolve_validation_branch(
    primary_statuses: Dict[str, str],
    fallback_capacity_v2: Dict[str, dict],
    audit_summary: Optional[dict] = None,
    diagnostic_only: bool = False,
) -> dict:
    """
    Explicit validation routing state machine.

    Returns branch, scheduled_targets, promotion_eligible, blocked_reason.
    """
    audit_summary = audit_summary or {}
    if not audit_summary:
        return {
            "branch": "blocked_audit_incomplete",
            "scheduled_targets": [],
            "promotion_eligible": False,
            "blocked_reason": "audit_summary_missing",
            "capacity_v2_failed": True,
        }
    if audit_summary.get("audit_integrity_pass") is not True:
        return {
            "branch": "blocked_audit_integrity",
            "scheduled_targets": [],
            "promotion_eligible": False,
            "blocked_reason": "audit_integrity_not_passed",
            "capacity_v2_failed": True,
        }
    audit_complete = audit_summary.get("audit_complete") is True
    metric_ok = audit_summary.get("metric_reproducible") is True
    inference_ok = audit_summary.get("inference_reproducible") is True
    if not (audit_complete and metric_ok and inference_ok):
        return {
            "branch": "blocked_audit_incomplete",
            "scheduled_targets": [],
            "promotion_eligible": False,
            "blocked_reason": "independent_inference_audit_not_complete_or_not_reproducible",
            "capacity_v2_failed": True,
        }

    if any(primary_statuses.get(s) == "capacity_passed" for s in PRIMARY_OVERFIT_TARGETS):
        return {
            "branch": "primary",
            "scheduled_targets": list(PRIMARY_VALIDATION_TARGETS),
            "promotion_eligible": True,
            "blocked_reason": None,
            "capacity_v2_failed": False,
        }

    t5_pass = {
        slug: bool((fallback_capacity_v2.get(slug) or {}).get("pass"))
        for slug in FALLBACK_VALIDATION_TARGETS
    }
    if any(t5_pass.values()):
        return {
            "branch": "t5_standard",
            "scheduled_targets": list(FALLBACK_VALIDATION_TARGETS),
            "promotion_eligible": True,
            "blocked_reason": None,
            "capacity_v2_failed": False,
            "t5_capacity_v2": t5_pass,
        }

    if diagnostic_only:
        return {
            "branch": "t5_diagnostic_only",
            "scheduled_targets": ["T5A"],
            "promotion_eligible": True,
            "blocked_reason": None,
            "capacity_v2_failed": True,
            "note": (
                "diagnostic-only is not capacity success; keep capacity_v2_failed=true"
            ),
            "t5_capacity_v2": t5_pass,
        }

    return {
        "branch": "blocked_no_capacity_v2",
        "scheduled_targets": [],
        "promotion_eligible": False,
        "blocked_reason": "primary_and_t5_capacity_v2_failed",
        "capacity_v2_failed": True,
        "t5_capacity_v2": t5_pass,
        "hint": "use validate-t5 --diagnostic-only for one pre-registered T5A run",
    }


def promotion_gate_from_rows(rows: Sequence[dict]) -> dict:
    hard = [r for r in rows if r["difficulty"] == "hard"]
    nonhard = [r for r in rows if r["difficulty"] != "hard"]
    hard_s = summarize_group(hard, "hard_")
    nonhard_s = summarize_group(nonhard, "nonhard_")
    all_s = summarize_group(list(rows), "all_")
    checks = {
        "hard_mean_delta_tm_ge_0p005": (
            hard_s["hard_mean_delta_tm"] is not None
            and hard_s["hard_mean_delta_tm"] >= 0.005
        ),
        "hard_median_delta_tm_gt_0": (
            hard_s["hard_median_delta_tm"] is not None
            and hard_s["hard_median_delta_tm"] > 0
        ),
        "nonhard_mean_delta_tm_ge_m0p005": (
            nonhard_s["nonhard_mean_delta_tm"] is not None
            and nonhard_s["nonhard_mean_delta_tm"] >= -0.005
        ),
        "all_mean_delta_lddt_ge_m0p002": (
            all_s["all_mean_delta_lddt"] is not None
            and all_s["all_mean_delta_lddt"] >= -0.002
        ),
        "at_least_one_hard_delta_tm_ge_0p01": (
            hard_s["hard_delta_tm_ge_0p01_count"] >= 1
        ),
    }
    return {
        "promotion_pass": all(checks.values()),
        "gate_version": "promotion_v1",
        "checks": checks,
        **hard_s,
        **nonhard_s,
        **all_s,
    }


def promotion_gate_v2_from_rows(
    rows: Sequence[dict],
    *,
    expected_labels: Optional[Sequence[str]] = None,
    expected_clusters: Optional[Sequence[str]] = None,
) -> dict:
    v1 = promotion_gate_from_rows(rows)
    hard = [r for r in rows if r["difficulty"] == "hard"]
    hard_deltas = [float(r["delta_tm"]) for r in hard]
    lmo = leave_max_out_mean(hard_deltas) if hard_deltas else None
    ge_count = sum(d >= 0.01 for d in hard_deltas)
    low_n = len(hard) < 4
    if low_n:
        ge_ok = ge_count >= 1
    else:
        ge_ok = ge_count >= 2

    labels = [r.get("label") for r in rows]
    clusters = [r.get("cluster_id") for r in rows]
    label_ok = bool(labels) and all(labels) and len(labels) == len(set(labels))
    # Validation may legitimately contain multiple labels from one frozen cluster.
    # Exact manifest multiset matching below is the integrity check; uniqueness is
    # only an overfit8 capacity requirement.
    cluster_ok = bool(clusters) and all(clusters)
    if expected_labels is not None:
        got = sorted(labels)
        label_ok = label_ok and got == sorted(expected_labels)
    if expected_clusters is not None:
        got_c = sorted(clusters)
        cluster_ok = cluster_ok and got_c == sorted(expected_clusters)

    checks = dict(v1["checks"])
    checks.update(
        {
            "hard_leave_max_out_mean_gt_0": bool(lmo is not None and lmo > 0),
            "hard_delta_tm_ge_0p01_count_ok": ge_ok,
            "validation_labels_match_manifest": label_ok,
            "validation_clusters_match_manifest": cluster_ok,
        }
    )
    return {
        "promotion_pass": all(checks.values()),
        "gate_version": "promotion_v2",
        "checks": checks,
        "promotion_v1": {
            "promotion_pass": v1["promotion_pass"],
            "checks": v1["checks"],
        },
        "hard_leave_max_out_mean_delta_tm": lmo,
        "hard_delta_tm_ge_0p01_count": ge_count,
        "hard_count": len(hard),
        "low_hard_sample_count": low_n,
        "hard_bootstrap_mean_delta_tm_ci": cluster_bootstrap_mean_ci(hard),
        "max_point_contribution": max_point_contribution(hard_deltas),
        "hard_mean_delta_tm": v1.get("hard_mean_delta_tm"),
        "hard_median_delta_tm": v1.get("hard_median_delta_tm"),
        "nonhard_mean_delta_tm": v1.get("nonhard_mean_delta_tm"),
        "all_mean_delta_lddt": v1.get("all_mean_delta_lddt"),
        "hard_delta_tm_ge_0p01_count_v1": v1.get("hard_delta_tm_ge_0p01_count"),
    }


def promotion_does_not_block_targets() -> bool:
    return True


def rank_promotion_candidates(candidates: Sequence[dict]) -> List[dict]:
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["hard_mean_delta_tm"]),
            -float(row["hard_median_delta_tm"]),
            -float(row.get("all_mean_delta_lddt") or -1e9),
            int(row["trainable_params"]),
            row["target_slug"],
        ),
    )


def select_seed_confirmation_targets(
    promotion_passers: Sequence[dict],
    max_targets: int = 2,
    near_tie_eps: float = 0.002,
) -> List[dict]:
    """
    Prefer true performance; if top-2 hard means differ by < near_tie_eps,
    prefer fewer params (typically T5A over T5B).
    """
    ranked = rank_promotion_candidates(promotion_passers)
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        if abs(float(a["hard_mean_delta_tm"]) - float(b["hard_mean_delta_tm"])) < near_tie_eps:
            ranked[:2] = sorted(
                ranked[:2],
                key=lambda row: (
                    int(row["trainable_params"]),
                    row["target_slug"],
                ),
            )
    return ranked[:max_targets]


def seed_confirmation_pass(seed_summaries: Sequence[dict]) -> dict:
    if len(seed_summaries) != 3:
        return {
            "pass": False,
            "reason": f"expected 3 seeds, got {len(seed_summaries)}",
        }
    hard_means = [float(s["hard_mean_delta_tm"]) for s in seed_summaries]
    hard_medians = [float(s["hard_median_delta_tm"]) for s in seed_summaries]
    lddt_means = [float(s["all_mean_delta_lddt"]) for s in seed_summaries]
    mean_of_means = sum(hard_means) / 3.0
    positive_seeds = sum(v > 0 for v in hard_means)
    nonhard_ok = all(float(s["nonhard_mean_delta_tm"]) >= -0.005 for s in seed_summaries)
    lddt_ok = all(float(s["all_mean_delta_lddt"]) >= -0.002 for s in seed_summaries)
    passed = (
        mean_of_means >= 0.005
        and positive_seeds >= 2
        and nonhard_ok
        and lddt_ok
    )
    return {
        "pass": passed,
        "mean_hard_mean_delta_tm": mean_of_means,
        "mean_hard_median_delta_tm": sum(hard_medians) / 3.0,
        "mean_all_delta_lddt": sum(lddt_means) / 3.0,
        "positive_seed_count": positive_seeds,
        "nonhard_ok": nonhard_ok,
        "lddt_ok": lddt_ok,
        "seed_hard_means": hard_means,
    }


def choose_unique_target(confirmed: Sequence[dict]) -> Optional[dict]:
    if not confirmed:
        return None
    # Near-tie prefers fewer params.
    selected = select_seed_confirmation_targets(confirmed, max_targets=1)
    return selected[0] if selected else None


def classify_capacity_failure(
    *,
    t5_any_pass: bool,
    positive_control_pass: bool,
    positive_control_executed: bool,
    loss_decreased: Optional[bool],
    tm_unchanged: bool,
    tm_sparse_response: bool = False,
) -> dict:
    if t5_any_pass:
        return {
            "failure_type": "primary_targets_failed_but_t5_fallback_viable",
            "action": "run_t5_full_train_validation_then_promotion_gate",
        }
    if not positive_control_executed:
        return {
            "failure_type": "primary_and_t5_capacity_v2_failed_positive_control_not_executed",
            "action": "run_real_positive_control_or_stop",
            "note": "setup-only positive control must not classify pass/fail",
        }
    if positive_control_pass:
        return {
            "failure_type": "lora_parameterization_or_capacity_insufficient",
            "action": "stop_lora_full_train_promotion_do_not_rank_positive_control",
        }
    if loss_decreased is None:
        return {
            "failure_type": "positive_control_loss_trend_unavailable",
            "action": "repair_loss_logging_then_reclassify",
        }
    if not loss_decreased:
        return {
            "failure_type": "optimization_backprop_or_data_pipeline_issue",
            "action": "debug_training_graph_and_data_path",
        }
    if tm_unchanged:
        return {
            "failure_type": "train_loss_tm_objective_mismatch",
            "action": "investigate_loss_vs_tm_alignment",
        }
    if tm_sparse_response:
        return {
            "failure_type": "positive_control_sparse_tm_response_not_robust",
            "action": (
                "redesign_training_signal_or_data_before_more_lora_search"
            ),
            "note": (
                "backprop changes TM, but gains are dominated by too few samples"
            ),
        }
    return {
        "failure_type": "positive_control_partial_tm_response_not_robust",
        "action": "review_per_sample_response_and_training_objective",
    }


def assert_split_isolation(
    *,
    train_labels: Sequence[str],
    validation_labels: Sequence[str],
    development_labels: Sequence[str],
    accessed_split: str,
    allowing_selection: bool,
) -> None:
    """Fail if target-selection stage accesses development/test."""
    if allowing_selection and accessed_split in {"development", "development_test", "test"}:
        raise AssertionError(
            f"target selection must not read split={accessed_split}; "
            f"dev_n={len(development_labels)} train_n={len(train_labels)} "
            f"val_n={len(validation_labels)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "capacity",
            "capacity-v2",
            "capacity-status",
            "promotion",
            "promotion-v2",
            "seed-confirm",
            "schedule-overfit",
            "fallback-route",
            "resolve-validation",
        ],
        required=True,
    )
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--step-results-json", type=Path, default=None)
    parser.add_argument("--statuses-json", type=Path, default=None)
    parser.add_argument("--fallback-v2-json", type=Path, default=None)
    parser.add_argument("--audit-summary-json", type=Path, default=None)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--seed-summaries-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "capacity":
        payload = capacity_pass_from_rows(read_paired(args.paired_csv))
    elif args.mode == "capacity-v2":
        payload = capacity_pass_v2_from_rows(read_paired(args.paired_csv))
    elif args.mode == "capacity-status":
        step_results = json.loads(args.step_results_json.read_text(encoding="utf-8"))
        payload = capacity_status_for_target(
            {int(k): v for k, v in step_results.items()}
        )
    elif args.mode == "promotion":
        payload = promotion_gate_from_rows(read_paired(args.paired_csv))
    elif args.mode == "promotion-v2":
        payload = promotion_gate_v2_from_rows(read_paired(args.paired_csv))
    elif args.mode == "seed-confirm":
        summaries = json.loads(args.seed_summaries_json.read_text(encoding="utf-8"))
        payload = seed_confirmation_pass(summaries)
    elif args.mode == "schedule-overfit":
        completed = (
            json.loads(args.statuses_json.read_text(encoding="utf-8"))
            if args.statuses_json
            else {}
        )
        payload = {
            "required_targets": list(PRIMARY_OVERFIT_TARGETS),
            "scheduled": should_schedule_overfit_targets(completed),
            "validation_matrix": list(PRIMARY_VALIDATION_TARGETS),
            "fallback_validation": list(FALLBACK_VALIDATION_TARGETS),
            "promotion_blocks_targets": False,
            "capacity_eval_steps": list(CAPACITY_CHECKPOINTS),
        }
    elif args.mode == "resolve-validation":
        primary = json.loads(args.statuses_json.read_text(encoding="utf-8"))
        fallback = (
            json.loads(args.fallback_v2_json.read_text(encoding="utf-8"))
            if args.fallback_v2_json
            else {}
        )
        audit = (
            json.loads(args.audit_summary_json.read_text(encoding="utf-8"))
            if args.audit_summary_json
            else {"audit_integrity_pass": True}
        )
        payload = resolve_validation_branch(
            primary, fallback, audit, diagnostic_only=args.diagnostic_only
        )
    else:
        statuses = json.loads(args.statuses_json.read_text(encoding="utf-8"))
        payload = {
            "run_fallback": should_run_capacity_fallback(statuses),
            "enter_full_train": should_enter_full_train_validation(statuses),
            "required_overfit_targets": list(PRIMARY_OVERFIT_TARGETS),
            "fallback_targets": list(FALLBACK_TARGETS),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
