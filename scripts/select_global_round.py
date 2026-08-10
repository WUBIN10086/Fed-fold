#!/usr/bin/env python3
"""Select the global best communication round from server validation summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_round(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_global_best(
    round_summaries: list[dict],
    nonhard_constraint: float = -0.005,
    min_hard_clusters_warn: int = 3,
) -> dict:
    if not round_summaries:
        raise ValueError("No round summaries provided")

    warnings = []
    candidates = []
    for row in round_summaries:
        if row.get("metric_status") != "ok":
            warnings.append(
                f"round {row.get('round_id')}: metric_status="
                f"{row.get('metric_status', 'missing')}"
            )
            continue
        hard = row.get("macro_mean_delta_tm_hard")
        nonhard = row.get("macro_mean_delta_tm_nonhard")
        min_hard = row.get("min_hard_clusters")
        if min_hard is not None and min_hard < min_hard_clusters_warn:
            warnings.append(
                f"round {row.get('round_id')}: min_hard_clusters={min_hard} "
                f"< {min_hard_clusters_warn}"
            )
        if hard is None:
            continue
        if nonhard is None or nonhard < nonhard_constraint:
            continue
        candidates.append(row)

    if not candidates:
        # Fall back to round 0 / baseline if present, else lowest round_id.
        complete = [r for r in round_summaries if r.get("metric_status") == "ok"]
        if not complete:
            raise ValueError("No complete round summaries; run validation inference first")
        baseline = min(complete, key=lambda r: int(r.get("round_id", 10**9)))
        return {
            "selected_round_id": int(baseline.get("round_id", 0)),
            "reason": "no_candidate_satisfied_constraint_or_no_hard_gain",
            "selected": baseline,
            "fallback_to_baseline": True,
            "warnings": warnings,
        }

    # Prefer positive hard gain; if all non-positive, still pick best under constraint
    # but mark fallback if best hard gain is not stable/positive.
    best = max(
        candidates,
        key=lambda r: (
            float(r["macro_mean_delta_tm_hard"]),
            float(r.get("macro_mean_delta_tm_nonhard") or -1e9),
            -int(r.get("round_id", 0)),
        ),
    )
    fallback = float(best["macro_mean_delta_tm_hard"]) <= 0
    return {
        "selected_round_id": int(best["round_id"]),
        "reason": (
            "max_macro_hard_under_nonhard_constraint"
            if not fallback
            else "constraint_ok_but_nonpositive_hard_gain"
        ),
        "selected": best,
        "fallback_to_baseline": fallback and int(best.get("round_id", 0)) == 0,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-summaries", nargs="+", required=True)
    parser.add_argument("--nonhard-constraint", type=float, default=-0.005)
    parser.add_argument("--min-hard-clusters-warn", type=int, default=3)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copy-best-to", type=Path, default=None)
    args = parser.parse_args()

    rows = [load_round(Path(p)) for p in args.round_summaries]
    decision = select_global_best(
        rows,
        nonhard_constraint=args.nonhard_constraint,
        min_hard_clusters_warn=args.min_hard_clusters_warn,
    )
    round_id = decision["selected_round_id"]
    src = args.models_root / f"round_{round_id:03d}" / "global_model.pt"
    if not src.exists() and round_id == 0:
        # Fall back to shared round_000 parent used by all arms.
        alt = args.models_root.parent.parent / "round_000" / "global_model.pt"
        if alt.exists():
            src = alt
        else:
            src = args.models_root / "round_000" / "global_model.pt"
    decision["global_model_path"] = str(src)
    if args.copy_best_to is not None and src.exists():
        args.copy_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, args.copy_best_to)
        decision["copied_to"] = str(args.copy_best_to)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
