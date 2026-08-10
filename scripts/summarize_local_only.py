#!/usr/bin/env python3
"""Summarize local-only development results by arm and seed."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.evaluate_hardcase_metrics import cluster_bootstrap_ci, summarize_group


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def local_result_root(
    run_root: Path,
    client: str,
    arm: str,
    seed: int,
    *,
    target_slug: str = "",
    local_output_namespace: str = "",
) -> Path:
    base = run_root / "clients" / client / "private"
    if local_output_namespace:
        base = base / local_output_namespace
        if target_slug:
            base = base / target_slug
    else:
        base = base / "local_only"
        if target_slug:
            base = base / target_slug
    return base / arm / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-slug", default="")
    parser.add_argument("--local-output-namespace", default="")
    args = parser.parse_args()

    arms = [value for value in args.arms.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    report_rows = []
    all_paired = []

    for arm in arms:
        for seed in seeds:
            client_summaries = []
            grouped_rows = []
            selected = []
            for client_idx in range(args.num_clients):
                client = f"client_{client_idx}"
                root = local_result_root(
                    args.run_root,
                    client,
                    arm,
                    seed,
                    target_slug=args.target_slug,
                    local_output_namespace=args.local_output_namespace,
                )
                paired_path = root / "development" / "paired_deltas.csv"
                selection_path = root / "selection.json"
                if not paired_path.exists() or not selection_path.exists():
                    continue
                rows = read_rows(paired_path)
                grouped_rows.extend(rows)
                for row in rows:
                    row = dict(row)
                    row.update({"arm": arm, "seed": seed})
                    all_paired.append(row)
                hard = [row for row in rows if row["difficulty"] == "hard"]
                nonhard = [row for row in rows if row["difficulty"] != "hard"]
                client_summaries.append(
                    {
                        "client": client,
                        **summarize_group(hard, "hard_"),
                        **summarize_group(nonhard, "nonhard_"),
                    }
                )
                selected.append(
                    json.loads(selection_path.read_text(encoding="utf-8"))
                )

            hard_means = [
                row["hard_mean_delta_tm"]
                for row in client_summaries
                if row["hard_mean_delta_tm"] is not None
            ]
            hard_medians = [
                row["hard_median_delta_tm"]
                for row in client_summaries
                if row["hard_median_delta_tm"] is not None
            ]
            nonhard_means = [
                row["nonhard_mean_delta_tm"]
                for row in client_summaries
                if row["nonhard_mean_delta_tm"] is not None
            ]
            bootstrap = cluster_bootstrap_ci(grouped_rows, "hard", seed=seed)
            report_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "client_count": len(client_summaries),
                    "client_macro_hard_mean_delta_tm": mean(hard_means),
                    "client_macro_hard_median_delta_tm": mean(hard_medians),
                    "client_macro_nonhard_mean_delta_tm": mean(nonhard_means),
                    "pooled_hard_ci_low": bootstrap["ci_low"],
                    "pooled_hard_ci_high": bootstrap["ci_high"],
                    "baseline_fallback_clients": sum(
                        bool(row["fallback_to_baseline"]) for row in selected
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "local_only_summary.csv"
    summary_fields = [
        "arm",
        "seed",
        "client_count",
        "client_macro_hard_mean_delta_tm",
        "client_macro_hard_median_delta_tm",
        "client_macro_nonhard_mean_delta_tm",
        "pooled_hard_ci_low",
        "pooled_hard_ci_high",
        "baseline_fallback_clients",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(report_rows)

    seed_summary_csv = args.out_dir / "local_only_seed_summary.csv"
    by_arm = defaultdict(list)
    for row in report_rows:
        by_arm[row["arm"]].append(row)
    seed_fields = [
        "arm",
        "seed_count",
        "hard_macro_mean",
        "hard_macro_std",
        "nonhard_macro_mean",
        "nonhard_macro_std",
    ]
    seed_rows = []
    for arm, rows in sorted(by_arm.items()):
        hard = [
            float(row["client_macro_hard_mean_delta_tm"])
            for row in rows
            if row["client_macro_hard_mean_delta_tm"] is not None
        ]
        nonhard = [
            float(row["client_macro_nonhard_mean_delta_tm"])
            for row in rows
            if row["client_macro_nonhard_mean_delta_tm"] is not None
        ]
        seed_rows.append(
            {
                "arm": arm,
                "seed_count": len(rows),
                "hard_macro_mean": mean(hard),
                "hard_macro_std": statistics.stdev(hard) if len(hard) > 1 else 0.0,
                "nonhard_macro_mean": mean(nonhard),
                "nonhard_macro_std": (
                    statistics.stdev(nonhard) if len(nonhard) > 1 else 0.0
                ),
            }
        )
    with seed_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(seed_rows)

    paired_csv = args.out_dir / "local_only_paired_deltas.csv"
    paired_fields = [
        "arm",
        "seed",
        "label",
        "client",
        "cluster_id",
        "difficulty",
        "tm_baseline",
        "tm_model",
        "delta_tm",
        "lddt_baseline",
        "lddt_model",
        "delta_lddt_ca",
    ]
    with paired_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in paired_fields}
            for row in all_paired
        )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {seed_summary_csv}")
    print(f"Wrote {paired_csv}")


if __name__ == "__main__":
    main()
