#!/usr/bin/env python3
"""Hard-case metrics: private paired tables and server-visible summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric_map(rows: List[dict], label_key: str, value_key: str) -> Dict[str, float]:
    out = {}
    value_candidates = {
        "tm_selected": ("tm_selected", "baseline_tm", "tm_score", "tm"),
        "lddt_ca": ("lddt_ca", "baseline_lddt_ca"),
        "mean_plddt": ("mean_plddt", "baseline_plddt"),
    }.get(value_key, (value_key,))
    for row in rows:
        label = (row.get(label_key) or row.get("label") or "").strip()
        if not label:
            continue
        raw = next(
            (row.get(name) for name in value_candidates if row.get(name) not in (None, "")),
            None,
        )
        if raw in (None, ""):
            continue
        out[label.upper()] = float(raw)
    return out


def load_difficulty(path: Path) -> Dict[str, dict]:
    rows = {}
    for row in read_csv(path):
        rows[row["label"].upper()] = row
    return rows


def paired_rows(
    labels: Sequence[str],
    difficulty: Dict[str, dict],
    baseline_tm: Dict[str, float],
    model_tm: Dict[str, float],
    baseline_lddt: Optional[Dict[str, float]] = None,
    model_lddt: Optional[Dict[str, float]] = None,
    baseline_plddt: Optional[Dict[str, float]] = None,
    model_plddt: Optional[Dict[str, float]] = None,
    client: str = "",
) -> List[dict]:
    baseline_lddt = baseline_lddt or {}
    model_lddt = model_lddt or {}
    baseline_plddt = baseline_plddt or {}
    model_plddt = model_plddt or {}
    rows = []
    for label in labels:
        key = label.upper()
        if key not in baseline_tm or key not in model_tm:
            continue
        diff = difficulty.get(key, {})
        tm_b = baseline_tm[key]
        tm_m = model_tm[key]
        row = {
            "label": label,
            "client": client or diff.get("client", ""),
            "cluster_id": diff.get("cluster_id", ""),
            "difficulty": diff.get("difficulty")
            or (
                "hard"
                if tm_b < 0.5
                else ("medium" if tm_b < 0.8 else "easy")
            ),
            "tm_baseline": tm_b,
            "tm_model": tm_m,
            "delta_tm": tm_m - tm_b,
            "lddt_baseline": baseline_lddt.get(key, ""),
            "lddt_model": model_lddt.get(key, ""),
            "delta_lddt_ca": (
                ""
                if key not in baseline_lddt or key not in model_lddt
                else model_lddt[key] - baseline_lddt[key]
            ),
            "baseline_plddt": baseline_plddt.get(key, ""),
            "model_plddt": model_plddt.get(key, ""),
            "delta_plddt": (
                ""
                if key not in baseline_plddt or key not in model_plddt
                else model_plddt[key] - baseline_plddt[key]
            ),
        }
        rows.append(row)
    return rows


def summarize_group(rows: Sequence[dict], prefix: str = "") -> dict:
    if not rows:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean_delta_tm": None,
            f"{prefix}median_delta_tm": None,
            f"{prefix}improved_fraction": None,
            f"{prefix}tm_rise_count": 0,
            f"{prefix}delta_tm_ge_0p01_count": 0,
            f"{prefix}delta_tm_ge_0p01_rate": None,
            f"{prefix}large_improve_rate": None,
            f"{prefix}large_degrade_rate": None,
            f"{prefix}rescue_rate": None,
            f"{prefix}mean_delta_lddt": None,
        }
    deltas = [float(r["delta_tm"]) for r in rows]
    deltas_sorted = sorted(deltas)
    mid = len(deltas_sorted) // 2
    if len(deltas_sorted) % 2:
        median = deltas_sorted[mid]
    else:
        median = 0.5 * (deltas_sorted[mid - 1] + deltas_sorted[mid])
    hard_like = rows
    rescue = 0
    rescue_den = 0
    for row in hard_like:
        if float(row["tm_baseline"]) < 0.5:
            rescue_den += 1
            if float(row["tm_model"]) >= 0.5:
                rescue += 1
    lddt_vals = [
        float(r["delta_lddt_ca"])
        for r in rows
        if r.get("delta_lddt_ca") not in ("", None)
    ]
    rise_count = sum(d > 0 for d in deltas)
    ge_0p01 = sum(d >= 0.01 for d in deltas)
    return {
        f"{prefix}count": len(rows),
        f"{prefix}mean_delta_tm": sum(deltas) / len(deltas),
        f"{prefix}median_delta_tm": median,
        f"{prefix}improved_fraction": rise_count / len(deltas),
        f"{prefix}tm_rise_count": rise_count,
        f"{prefix}delta_tm_ge_0p01_count": ge_0p01,
        f"{prefix}delta_tm_ge_0p01_rate": ge_0p01 / len(deltas),
        f"{prefix}large_improve_rate": sum(d >= 0.05 for d in deltas) / len(deltas),
        f"{prefix}large_degrade_rate": sum(d <= -0.05 for d in deltas) / len(deltas),
        f"{prefix}rescue_rate": (rescue / rescue_den) if rescue_den else None,
        f"{prefix}mean_delta_lddt": (
            sum(lddt_vals) / len(lddt_vals) if lddt_vals else None
        ),
    }


def server_summary_from_paired(
    client_id: str,
    round_id: int,
    global_model_sha: str,
    rows: Sequence[dict],
) -> dict:
    hard = [r for r in rows if r["difficulty"] == "hard"]
    nonhard = [r for r in rows if r["difficulty"] != "hard"]
    hard_clusters = {r["cluster_id"] for r in hard if r.get("cluster_id")}
    return {
        "client_id": client_id,
        "round_id": round_id,
        "global_model_sha": global_model_sha,
        "hard_count": len(hard),
        "hard_cluster_count": len(hard_clusters),
        "sum_delta_tm_hard": sum(float(r["delta_tm"]) for r in hard),
        "hard_improved_count": sum(float(r["delta_tm"]) > 0 for r in hard),
        "hard_rescue_count": sum(
            float(r["tm_baseline"]) < 0.5 and float(r["tm_model"]) >= 0.5
            for r in hard
        ),
        "hard_large_improve_count": sum(float(r["delta_tm"]) >= 0.05 for r in hard),
        "hard_large_degrade_count": sum(float(r["delta_tm"]) <= -0.05 for r in hard),
        "nonhard_count": len(nonhard),
        "sum_delta_tm_nonhard": sum(float(r["delta_tm"]) for r in nonhard),
        "sum_delta_lddt": sum(
            float(r["delta_lddt_ca"])
            for r in rows
            if r.get("delta_lddt_ca") not in ("", None)
        ),
        "metric_status": "ok" if rows else "empty",
    }


def macro_from_server_summaries(summaries: Sequence[dict]) -> dict:
    if not summaries:
        return {
            "macro_mean_delta_tm_hard": None,
            "macro_mean_delta_tm_nonhard": None,
            "n_clients": 0,
            "metric_status": "missing_client_summaries",
        }
    bad_statuses = [
        str(row.get("metric_status", "missing"))
        for row in summaries
        if row.get("metric_status") != "ok"
    ]
    if bad_statuses:
        return {
            "macro_mean_delta_tm_hard": None,
            "macro_mean_delta_tm_nonhard": None,
            "n_clients": len(summaries),
            "min_hard_clusters": min(
                int(row.get("hard_cluster_count", 0)) for row in summaries
            ),
            "metric_status": "incomplete",
            "client_metric_statuses": [
                str(row.get("metric_status", "missing")) for row in summaries
            ],
        }
    hard_means = []
    nonhard_means = []
    for row in summaries:
        if row["hard_count"]:
            hard_means.append(row["sum_delta_tm_hard"] / row["hard_count"])
        if row["nonhard_count"]:
            nonhard_means.append(
                row["sum_delta_tm_nonhard"] / row["nonhard_count"]
            )
    return {
        "macro_mean_delta_tm_hard": (
            sum(hard_means) / len(hard_means) if hard_means else None
        ),
        "macro_mean_delta_tm_nonhard": (
            sum(nonhard_means) / len(nonhard_means) if nonhard_means else None
        ),
        "n_clients": len(summaries),
        "min_hard_clusters": min(r["hard_cluster_count"] for r in summaries),
        "metric_status": "ok",
    }


def cluster_bootstrap_ci(
    rows: Sequence[dict],
    difficulty: str = "hard",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    subset = [r for r in rows if r["difficulty"] == difficulty]
    by_cluster = defaultdict(list)
    for row in subset:
        by_cluster[row.get("cluster_id") or row["label"]].append(
            float(row["delta_tm"])
        )
    clusters = list(by_cluster.keys())
    if not clusters:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_clusters": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        drawn = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        vals = [v for c in drawn for v in by_cluster[c]]
        means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    point = sum(float(r["delta_tm"]) for r in subset) / len(subset)
    return {
        "mean": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_clusters": len(clusters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pair", "server-summary", "macro"], required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--difficulty-csv", type=Path, default=None)
    parser.add_argument("--baseline-tm-csv", type=Path, default=None)
    parser.add_argument("--model-tm-csv", type=Path, default=None)
    parser.add_argument("--baseline-lddt-csv", type=Path, default=None)
    parser.add_argument("--model-lddt-csv", type=Path, default=None)
    parser.add_argument("--baseline-plddt-csv", type=Path, default=None)
    parser.add_argument("--model-plddt-csv", type=Path, default=None)
    parser.add_argument("--client-id", default="")
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--global-model-sha", default="")
    parser.add_argument("--paired-csv", type=Path, default=None)
    parser.add_argument("--server-summaries", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "pair":
        labels = [
            line.strip()
            for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        difficulty = load_difficulty(args.difficulty_csv)
        baseline_tm = metric_map(read_csv(args.baseline_tm_csv), "label", "tm_selected")
        model_tm = metric_map(read_csv(args.model_tm_csv), "label", "tm_selected")
        baseline_lddt = (
            metric_map(read_csv(args.baseline_lddt_csv), "label", "lddt_ca")
            if args.baseline_lddt_csv
            else {}
        )
        model_lddt = (
            metric_map(read_csv(args.model_lddt_csv), "label", "lddt_ca")
            if args.model_lddt_csv
            else {}
        )
        baseline_plddt = (
            metric_map(read_csv(args.baseline_plddt_csv), "label", "mean_plddt")
            if args.baseline_plddt_csv
            else {}
        )
        model_plddt = (
            metric_map(read_csv(args.model_plddt_csv), "protein_name", "mean_plddt")
            if args.model_plddt_csv
            else {}
        )
        rows = paired_rows(
            labels,
            difficulty,
            baseline_tm,
            model_tm,
            baseline_lddt,
            model_lddt,
            baseline_plddt,
            model_plddt,
            client=args.client_id,
        )
        write_csv(
            args.out,
            rows,
            [
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
                "baseline_plddt",
                "model_plddt",
                "delta_plddt",
            ],
        )
        hard = [r for r in rows if r["difficulty"] == "hard"]
        nonhard = [r for r in rows if r["difficulty"] != "hard"]
        summary = {}
        summary.update(summarize_group(hard, "hard_"))
        summary.update(summarize_group(nonhard, "nonhard_"))
        summary.update(summarize_group(rows, "all_"))
        # pLDDT is diagnostic-only when callers attach it to paired rows.
        for key in ("baseline_plddt", "model_plddt", "delta_plddt"):
            vals = [
                float(r[key])
                for r in rows
                if r.get(key) not in ("", None) and str(r.get(key)).strip() != ""
            ]
            summary[f"all_mean_{key}"] = (
                sum(vals) / len(vals) if vals else None
            )
        summary["bootstrap_hard"] = cluster_bootstrap_ci(rows, "hard")
        args.out.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return

    if args.mode == "server-summary":
        rows = read_csv(args.paired_csv)
        # Ensure numeric types
        for row in rows:
            row["delta_tm"] = float(row["delta_tm"])
            row["tm_baseline"] = float(row["tm_baseline"])
            row["tm_model"] = float(row["tm_model"])
        summary = server_summary_from_paired(
            args.client_id,
            args.round_id,
            args.global_model_sha,
            rows,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return

    summaries = []
    for path in args.server_summaries or []:
        summaries.append(json.loads(Path(path).read_text(encoding="utf-8")))
    out = macro_from_server_summaries(summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
