#!/usr/bin/env python3
"""Summarize client0 target ablation outputs into CSV/Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

from scripts.lora_target_registry import (
    PRIMARY_OVERFIT_TARGETS,
    PRIMARY_VALIDATION_TARGETS,
    get_target,
)


DEFAULT_RUN = REPO / "outputs" / "fed_lora_hardcase_fed_v1"
DEFAULT_EVAL = DEFAULT_RUN / "evaluation" / "client0_target_ablation_v1"
NAMESPACE = "target_ablation_v1"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def summarize(run_root: Path, eval_root: Path) -> dict:
    private = run_root / "clients" / "client_0" / "private" / NAMESPACE
    eval_root.mkdir(parents=True, exist_ok=True)

    overfit = load_json(private / "overfit_summary.json")
    diagnostics = load_json(private / "capacity_diagnostics.json")
    validation = load_json(private / "validation_summary.json")
    seeds = load_json(private / "seed_confirmation.json")
    development = load_json(private / "development_summary.json")
    status = load_json(private / "status.json")
    trajectory = load_json(private / "t5_trajectory_summary.json")
    t5_seeds = load_json(private / "t5a_seed_stability.json")
    positive_summary = load_json(private / "positive_control_summary.json")

    overfit_rows = []
    capacity_rows = []
    details = overfit.get("details") or {}
    for slug in PRIMARY_OVERFIT_TARGETS:
        detail = details.get(slug, {})
        steps = detail.get("steps") or {}
        overfit_rows.append(
            {
                "target_slug": slug,
                "status": (status.get("overfit_statuses_v2") or {}).get(slug)
                or detail.get("status")
                or overfit.get("statuses", {}).get(slug, "not_run"),
                "first_pass_step": detail.get("first_pass_step", ""),
                "completed_steps": ",".join(str(s) for s in detail.get("completed_steps", [])),
                "trainable_params": get_target(slug).expected_trainable,
                "modules": get_target(slug).expected_modules,
            }
        )
        for step, result in sorted(steps.items(), key=lambda kv: int(kv[0])):
            capacity_rows.append(
                {
                    "target_slug": slug,
                    "step": step,
                    "pass": result.get("pass", ""),
                    "pass_v2": (result.get("capacity_v2") or {}).get("pass", ""),
                    "mean_delta_tm": result.get("mean_delta_tm", ""),
                    "median_delta_tm_v2": (result.get("capacity_v2") or {}).get("median_delta_tm", ""),
                    "leave_max_out_mean_delta_tm_v2": (result.get("capacity_v2") or {}).get("leave_max_out_mean_delta_tm", ""),
                    "delta_tm_ge_0p01_count": result.get("delta_tm_ge_0p01_count", ""),
                    "tm_rise_count": result.get("tm_rise_count", ""),
                    "mean_delta_lddt": result.get("mean_delta_lddt", ""),
                    "rescue_rate": result.get("rescue_rate", ""),
                    "A_norm": (
                        diagnostics.get("targets", {})
                        .get(slug, {})
                        .get("norms", {})
                        .get("mean_A_norm", "")
                    ),
                    "B_norm": (
                        diagnostics.get("targets", {})
                        .get(slug, {})
                        .get("norms", {})
                        .get("mean_B_norm", "")
                    ),
                    "delta_W_norm": (
                        diagnostics.get("targets", {})
                        .get(slug, {})
                        .get("norms", {})
                        .get("mean_delta_W_norm", "")
                    ),
                    "lora_key_count": (
                        diagnostics.get("targets", {})
                        .get(slug, {})
                        .get("norms", {})
                        .get("lora_key_count", "")
                    ),
                    "changed_target_keys": (
                        diagnostics.get("targets", {})
                        .get(slug, {})
                        .get("norms", {})
                        .get("changed_target_keys", "")
                    ),
                    "first_pass_step": detail.get("first_pass_step", ""),
                }
            )

    validation_rows = []
    results = validation.get("results") or {}
    scheduled = validation.get("scheduled_targets") or []
    for slug in scheduled:
        row = results.get(slug, {})
        v2 = row.get("promotion_v2") or {}
        validation_rows.append(
            {
                "target_slug": slug,
                "promotion_pass": row.get("promotion_pass", ""),
                "promotion_pass_v2": v2.get("promotion_pass", row.get("promotion_pass_v2", "")),
                "hard_mean_delta_tm": row.get("hard_mean_delta_tm", ""),
                "hard_median_delta_tm": row.get("hard_median_delta_tm", ""),
                "nonhard_mean_delta_tm": row.get("nonhard_mean_delta_tm", ""),
                "all_mean_delta_lddt": row.get("all_mean_delta_lddt", ""),
                "hard_delta_tm_ge_0p01_count": row.get("hard_delta_tm_ge_0p01_count", ""),
                "trainable_params": get_target(slug).expected_trainable,
            }
        )

    seed_rows = []
    for slug, payload in (seeds.get("per_target") or {}).items():
        seed_rows.append(
            {
                "target_slug": slug,
                "pass": payload.get("pass", ""),
                "mean_hard_mean_delta_tm": payload.get("mean_hard_mean_delta_tm", ""),
                "positive_seed_count": payload.get("positive_seed_count", ""),
                "nonhard_ok": payload.get("nonhard_ok", ""),
                "lddt_ok": payload.get("lddt_ok", ""),
                "seed_hard_means": ",".join(
                    str(x) for x in payload.get("seed_hard_means", [])
                ),
            }
        )

    paired_rows = [
        {"comparison": k, "delta": v}
        for k, v in (validation.get("paired_target_deltas") or {}).items()
    ]

    write_csv(
        eval_root / "overfit8_target_summary.csv",
        overfit_rows,
        [
            "target_slug",
            "status",
            "first_pass_step",
            "completed_steps",
            "trainable_params",
            "modules",
        ],
    )
    write_csv(
        eval_root / "capacity_diagnostics.csv",
        capacity_rows,
        [
            "target_slug",
            "step",
            "pass",
            "pass_v2",
            "mean_delta_tm",
            "median_delta_tm_v2",
            "leave_max_out_mean_delta_tm_v2",
            "delta_tm_ge_0p01_count",
            "tm_rise_count",
            "mean_delta_lddt",
            "rescue_rate",
            "A_norm",
            "B_norm",
            "delta_W_norm",
            "lora_key_count",
            "changed_target_keys",
            "first_pass_step",
        ],
    )
    write_csv(
        eval_root / "validation_target_summary.csv",
        validation_rows,
        [
            "target_slug",
            "promotion_pass",
            "promotion_pass_v2",
            "hard_mean_delta_tm",
            "hard_median_delta_tm",
            "nonhard_mean_delta_tm",
            "all_mean_delta_lddt",
            "hard_delta_tm_ge_0p01_count",
            "trainable_params",
        ],
    )
    write_csv(
        eval_root / "seed_confirmation.csv",
        seed_rows,
        [
            "target_slug",
            "pass",
            "mean_hard_mean_delta_tm",
            "positive_seed_count",
            "nonhard_ok",
            "lddt_ok",
            "seed_hard_means",
        ],
    )
    write_csv(
        eval_root / "paired_deltas.csv",
        paired_rows,
        ["comparison", "delta"],
    )

    unique = (seeds.get("unique_target") or {}).get("target_slug")
    failure = diagnostics.get("failure_classification") or {}
    fallback = diagnostics.get("fallback") or {}
    pc = positive_summary or fallback.get("positive_control") or {}
    audit = load_json(private / "diagnostics" / "9u78_A" / "audit_summary.json")
    route = load_json(private / "validation_route.json") or validation.get("route") or {}
    schema_fingerprint = None
    if unique:
        schema_manifest = (
            private
            / unique
            / "uniform"
            / "seed_42"
            / "development"
            / "epoch_5"
            / "scale_1p0"
            / "model_scale_1.0.export_manifest.json"
        )
        schema_fingerprint = load_json(schema_manifest).get("lora_schema_fingerprint")

    if audit.get("audit_pass") is not True:
        stop_reason = "9u78_audit_incomplete_or_failed"
    elif validation.get("status") != "complete":
        stop_reason = validation.get("reason") or route.get("blocked_reason")
    elif not unique:
        stop_reason = seeds.get("reason") or "no_three_seed_stable_target"
    elif development.get("development_pass") is not True:
        stop_reason = development.get("reason") or "development_confirmation_failed"
    else:
        stop_reason = None

    lines = [
        "# Client0 Target Ablation Report",
        "",
        "## Scope separation",
        "",
        "- **Train memorization**: overfit8 capacity (v1 legacy + capacity_v2 robust).",
        "- **Validation generalization**: fixed epoch-5 / scale-1.0; branch-routed targets.",
        "- **Development confirmation**: only after 3-seed confirmation of one unique target.",
        "",
        "## 9u78_A audit",
        "",
        f"- audit_complete: `{audit.get('audit_complete')}`",
        f"- audit_pass: `{audit.get('audit_pass')}`",
        f"- audit_integrity_pass: `{audit.get('audit_integrity_pass')}`",
        f"- metric_reproducible: `{audit.get('metric_reproducible')}`",
        f"- inference_reproducible: `{audit.get('inference_reproducible')}`",
        f"- mapping_errors: `{audit.get('mapping_errors')}`",
        "",
        "## Overfit8 capacity",
        "",
    ]
    for row in overfit_rows:
        lines.append(
            f"- `{row['target_slug']}`: status=`{row['status']}`, "
            f"first_pass_step=`{row['first_pass_step']}`"
        )
    lines.extend(
        [
            "",
            "## Capacity diagnostics / fallback",
            "",
            f"- failure_type: `{failure.get('failure_type')}`",
            f"- action: `{failure.get('action')}`",
            f"- t5_capacity_v2: `{fallback.get('t5_capacity_v2')}`",
            f"- positive_control state: `{pc.get('state', 'setup_only' if not pc.get('executed') else pc.get('state'))}`",
            f"- positive_control executed: `{pc.get('executed', False)}`",
            f"- T5 trajectory status: `{trajectory.get('status', 'not_run')}`",
            f"- T5A seed audit status: `{t5_seeds.get('status', 'not_run')}`",
            f"- T5A 9u78 stable 2/3: `{t5_seeds.get('large_9u78_stable_2_of_3')}`",
            "",
            "## Validation route",
            "",
            f"- branch: `{validation.get('branch') or route.get('branch')}`",
            f"- status: `{validation.get('status')}`",
            f"- blocked: `{validation.get('blocked')}`",
            f"- capacity_v2_failed: `{validation.get('capacity_v2_failed')}`",
            f"- scheduled_targets: `{validation.get('scheduled_targets')}`",
            "",
            "## Validation (epoch 5, scale 1.0)",
            "",
        ]
    )
    if not validation_rows:
        lines.append("- (no validation results; see blocked reason above)")
    for row in validation_rows:
        lines.append(
            f"- `{row['target_slug']}`: promotion_v1=`{row['promotion_pass']}`, "
            f"promotion_v2=`{row['promotion_pass_v2']}`, "
            f"hard_mean_ΔTM=`{row['hard_mean_delta_tm']}`"
        )
    lines.extend(
        [
            "",
            "## Seed confirmation",
            "",
            f"- status: `{seeds.get('status')}`",
            f"- unique_target: `{unique}`",
            f"- blocks_development: `{seeds.get('blocks_development')}`",
            f"- blocks_other_clients: `{seeds.get('blocks_other_clients', True)}`",
            f"- blocks_fedlora: `{seeds.get('blocks_fedlora')}`",
            "",
            "## Development",
            "",
            f"- status: `{development.get('status', 'not_run')}`",
            f"- blocked: `{development.get('blocked')}`",
            f"- development_pass: `{development.get('development_pass')}`",
            f"- reason: `{development.get('reason', '')}`",
            f"- target: `{development.get('target_slug', '')}`",
            f"- hard_mean_ΔTM: `{development.get('hard_mean_delta_tm', '')}`",
            "",
            "## Final decision",
            "",
            f"- final_target: `{unique}`",
            f"- lora_schema_fingerprint: `{schema_fingerprint}`",
            f"- stop_reason: `{stop_reason}`",
            "",
            "## Freeze / isolation",
            "",
            f"- blocks_other_clients: `{status.get('blocks_other_clients', diagnostics.get('blocks_other_clients', True))}`",
            f"- blocks_fedlora: `{status.get('blocks_fedlora', diagnostics.get('blocks_fedlora', True))}`",
            f"- test_accessed: `{status.get('test_accessed', diagnostics.get('test_accessed', False))}`",
            "",
            "## Notes",
            "",
            "- capacity_v2 does not overwrite capacity_v1 artifacts.",
            "- positive control setup_only must not be treated as executed pass/fail.",
            "- FedLoRA requires one common lora_schema_fingerprint across clients.",
            "- Existing `local_only/` and baseline splits remain untouched.",
            "",
        ]
    )
    report_path = eval_root / "target_ablation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (private / "target_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "eval_root": str(eval_root),
        "report": str(report_path),
        "unique_target": unique,
        "overfit_statuses": overfit.get("statuses"),
        "validation_blocked": validation.get("blocked"),
        "validation_branch": validation.get("branch") or route.get("branch"),
        "development_blocked": development.get("blocked"),
        "development_pass": development.get("development_pass"),
        "blocks_other_clients": status.get("blocks_other_clients", True),
        "blocks_fedlora": status.get("blocks_fedlora", True),
        "test_accessed": status.get("test_accessed", False),
        "positive_control_state": pc.get("state", "not_run"),
        "audit_pass": audit.get("audit_pass"),
        "t5_trajectory_status": trajectory.get("status", "not_run"),
        "t5_seed_audit_status": t5_seeds.get("status", "not_run"),
        "lora_schema_fingerprint": schema_fingerprint,
        "stop_reason": stop_reason,
    }
    (eval_root / "report_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root, args.eval_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
