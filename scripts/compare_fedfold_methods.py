#!/usr/bin/env python3
"""Compare baseline / local-only / FedLoRA hard-case metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "method",
    "arm",
    "seed",
    "hard_macro_mean_delta_tm",
    "hard_macro_median_delta_tm",
    "nonhard_macro_mean_delta_tm",
    "hard_improved_fraction",
    "hard_large_improve_rate",
    "hard_large_degrade_rate",
    "hard_rescue_rate",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "selected_round",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON summary files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            {
                "method": data.get("method", Path(path).stem),
                "arm": data.get("arm", ""),
                "seed": data.get("seed", ""),
                "hard_macro_mean_delta_tm": data.get("hard_macro_mean_delta_tm", ""),
                "hard_macro_median_delta_tm": data.get(
                    "hard_macro_median_delta_tm", ""
                ),
                "nonhard_macro_mean_delta_tm": data.get(
                    "nonhard_macro_mean_delta_tm", ""
                ),
                "hard_improved_fraction": data.get("hard_improved_fraction", ""),
                "hard_large_improve_rate": data.get("hard_large_improve_rate", ""),
                "hard_large_degrade_rate": data.get("hard_large_degrade_rate", ""),
                "hard_rescue_rate": data.get("hard_rescue_rate", ""),
                "bootstrap_ci_low": data.get("bootstrap_ci_low", ""),
                "bootstrap_ci_high": data.get("bootstrap_ci_high", ""),
                "selected_round": data.get("selected_round", ""),
                "notes": data.get("notes", ""),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
