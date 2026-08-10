#!/usr/bin/env python3
"""Verify LoRA target strings match expected module/parameter counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.lora import LoRAConfig, configure_lora, parameter_counts
from scripts.lora_target_registry import TARGET_SPECS, get_target, list_slugs


def verify_one(slug: str, rank: int = 4, alpha: float = 8.0) -> dict:
    spec = get_target(slug)
    model = AlphaFold(model_config("seq_model_esm1b_ptm", train=True)).float()
    replaced = configure_lora(
        model,
        LoRAConfig(rank=rank, alpha=alpha, dropout=0.0, target=spec.target),
    )
    counts = parameter_counts(model)
    result = {
        "slug": slug,
        "target": spec.target,
        "modules": len(replaced),
        "trainable": int(counts["trainable"]),
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "module_names": replaced,
        "ok": (
            len(replaced) == spec.expected_modules
            and int(counts["trainable"]) == spec.expected_trainable
        ),
    }
    if not result["ok"]:
        raise AssertionError(
            f"{slug} mismatch: modules={result['modules']} "
            f"(expected {spec.expected_modules}), "
            f"trainable={result['trainable']} "
            f"(expected {spec.expected_trainable})"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slugs",
        default=",".join(list_slugs()),
        help="Comma-separated target slugs to verify",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for slug in [s.strip() for s in args.slugs.split(",") if s.strip()]:
        if slug not in TARGET_SPECS:
            raise SystemExit(f"Unknown slug: {slug}")
        print(f"Verifying {slug} ...")
        result = verify_one(slug, rank=args.rank, alpha=args.alpha)
        print(
            f"  OK modules={result['modules']} "
            f"trainable={result['trainable']}"
        )
        results.append(result)

    payload = {"rank": args.rank, "alpha": args.alpha, "targets": results}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.out}")
    print(f"Verified {len(results)} targets")


if __name__ == "__main__":
    main()
