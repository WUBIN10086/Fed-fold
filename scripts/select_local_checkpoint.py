#!/usr/bin/env python3
"""Select one local-only epoch/scale using validation paired metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluate_hardcase_metrics import summarize_group


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_candidate(path: Path) -> dict:
    rows = read_rows(path)
    hard = [row for row in rows if row["difficulty"] == "hard"]
    nonhard = [row for row in rows if row["difficulty"] != "hard"]
    hard_summary = summarize_group(hard, "hard_")
    nonhard_summary = summarize_group(nonhard, "nonhard_")
    metadata_path = path.parent / "candidate.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return {
        "paired_csv": str(path),
        "model_path": metadata.get("model_path", ""),
        "epoch": int(metadata.get("epoch", -1)),
        "scale": float(metadata.get("scale", 1.0)),
        **hard_summary,
        **nonhard_summary,
    }


def select_candidate(
    candidates: list[dict],
    nonhard_constraint: float = -0.005,
) -> dict:
    eligible = [
        row
        for row in candidates
        if row["hard_count"] > 0
        and row["nonhard_count"] > 0
        and row["nonhard_mean_delta_tm"] is not None
        and row["nonhard_mean_delta_tm"] >= nonhard_constraint
    ]
    if not eligible:
        return {
            "selected_epoch": 0,
            "selected_scale": 0.0,
            "model_path": "",
            "fallback_to_baseline": True,
            "reason": "no_candidate_satisfied_nonhard_constraint",
            "candidates": candidates,
        }
    best = max(
        eligible,
        key=lambda row: (
            float(row["hard_mean_delta_tm"]),
            float(row["hard_median_delta_tm"]),
            -float(row["hard_large_degrade_rate"]),
            -int(row["epoch"]),
            -float(row["scale"]),
        ),
    )
    if float(best["hard_mean_delta_tm"]) <= 0:
        return {
            "selected_epoch": 0,
            "selected_scale": 0.0,
            "model_path": "",
            "fallback_to_baseline": True,
            "reason": "best_candidate_has_nonpositive_hard_gain",
            "best_candidate": best,
            "candidates": candidates,
        }
    return {
        "selected_epoch": best["epoch"],
        "selected_scale": best["scale"],
        "model_path": best["model_path"],
        "paired_csv": best["paired_csv"],
        "fallback_to_baseline": False,
        "reason": "max_hard_mean_under_nonhard_constraint",
        "best_candidate": best,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nonhard-constraint", type=float, default=-0.005)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates = [score_candidate(path) for path in args.paired_csvs]
    decision = select_candidate(candidates, args.nonhard_constraint)
    decision.update(
        {
            "client_id": args.client_id,
            "arm": args.arm,
            "seed": args.seed,
            "nonhard_constraint": args.nonhard_constraint,
            "selection_split": "validation",
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.client_id} arm={args.arm} seed={args.seed}: "
        f"epoch={decision['selected_epoch']} scale={decision['selected_scale']} "
        f"fallback={decision['fallback_to_baseline']}"
    )


if __name__ == "__main__":
    main()
