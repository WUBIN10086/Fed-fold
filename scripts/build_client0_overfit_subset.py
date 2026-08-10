#!/usr/bin/env python3
"""Build the client0 overfit8 hard capacity-check subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_fed_test_set import write_reclustered_chain_cache
from scripts.build_hardcase_splits import copy_assets_for_labels
from scripts.lora_target_registry import OVERFIT8_CLUSTERS, OVERFIT8_LABELS


DEFAULT_RUN = REPO / "outputs" / "fed_lora_hardcase_fed_v1"
DEFAULT_SOURCE = REPO / "outputs" / "fed_lora_fp32"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_labels(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_labels(path: Path, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{lab}\n" for lab in labels), encoding="utf-8")


def load_difficulty(path: Path) -> Dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["label"].strip().upper()] = row
    return rows


def select_overfit8(
    train_labels: Sequence[str],
    difficulty: Dict[str, dict],
) -> List[Tuple[str, str, float]]:
    """Return [(label, cluster, baseline_tm), ...] for 8 hardest clusters."""
    best_by_cluster: Dict[str, Tuple[str, float]] = {}
    for label in train_labels:
        row = difficulty[label.upper()]
        tm = float(row["baseline_tm"])
        if tm >= 0.5:
            continue
        cluster = row["cluster_id"]
        prev = best_by_cluster.get(cluster)
        if prev is None or tm < prev[1]:
            best_by_cluster[cluster] = (label, tm)
    ranked = sorted(best_by_cluster.items(), key=lambda item: item[1][1])
    selected = [
        (label, cluster, tm) for cluster, (label, tm) in ranked[:8]
    ]
    return selected


def assert_no_leakage(
    selected: Sequence[Tuple[str, str, float]],
    val_labels: Sequence[str],
    dev_labels: Sequence[str],
    difficulty: Dict[str, dict],
) -> None:
    selected_labels = {lab.upper() for lab, _, _ in selected}
    selected_clusters = {cluster for _, cluster, _ in selected}
    for split_name, labels in (("validation", val_labels), ("development", dev_labels)):
        other_labels = {lab.upper() for lab in labels}
        other_clusters = {
            difficulty[lab.upper()]["cluster_id"]
            for lab in labels
            if lab.upper() in difficulty
        }
        label_overlap = selected_labels & other_labels
        cluster_overlap = selected_clusters & other_clusters
        if label_overlap:
            raise AssertionError(f"label leakage vs {split_name}: {sorted(label_overlap)}")
        if cluster_overlap:
            raise AssertionError(
                f"cluster leakage vs {split_name}: {sorted(cluster_overlap)}"
            )


def build_overfit_subset(
    run_root: Path,
    source_run: Path,
    link: bool = True,
    expected_labels: Sequence[str] = OVERFIT8_LABELS,
    expected_clusters: Sequence[str] = OVERFIT8_CLUSTERS,
) -> dict:
    private = run_root / "clients" / "client_0" / "private"
    split_root = private / "splits"
    difficulty_csv = private / "difficulty" / "baseline_difficulty.csv"
    out_root = private / "target_ablation_v1" / "overfit8"
    difficulty = load_difficulty(difficulty_csv)

    train_labels = read_labels(split_root / "train_labels.txt")
    val_labels = read_labels(split_root / "validation_labels.txt")
    dev_labels = read_labels(split_root / "development_test_labels.txt")

    selected = select_overfit8(train_labels, difficulty)
    if len(selected) != 8:
        raise AssertionError(f"expected 8 overfit labels, got {len(selected)}")
    labels = [lab for lab, _, _ in selected]
    clusters = [cluster for _, cluster, _ in selected]
    if tuple(labels) != tuple(expected_labels):
        raise AssertionError(
            f"overfit8 labels mismatch:\n  got={labels}\n  expected={list(expected_labels)}"
        )
    if tuple(clusters) != tuple(expected_clusters):
        raise AssertionError(
            f"overfit8 clusters mismatch:\n  got={clusters}\n  expected={list(expected_clusters)}"
        )
    if any(tm >= 0.5 for _, _, tm in selected):
        raise AssertionError("overfit8 contains non-hard samples")
    if len({c for _, c, _ in selected}) != 8:
        raise AssertionError("overfit8 clusters are not disjoint")
    assert_no_leakage(selected, val_labels, dev_labels, difficulty)

    out_root.mkdir(parents=True, exist_ok=True)
    labels_path = out_root / "overfit8_labels.txt"
    write_labels(labels_path, labels)

    label2cluster = {lab.upper(): cluster for lab, cluster, _ in selected}
    cache_src = split_root / "train_chain_data_cache.json"
    cache_dst = out_root / "overfit8_chain_data_cache.json"
    write_reclustered_chain_cache(cache_src, cache_dst, labels, label2cluster)

    client_src = source_run / "clients" / "client_0"
    copy_assets_for_labels(labels, client_src, out_root / "assets", link=link)

    # Convenience aliases used by inference helpers that expect split naming.
    assets_alias = out_root / "assets_overfit8"
    if not assets_alias.exists():
        assets_alias.symlink_to((out_root / "assets").resolve())
    labels_alias = out_root / "overfit8_split_labels.txt"
    if not labels_alias.exists():
        labels_alias.symlink_to(labels_path.resolve())

    rows = []
    for lab, cluster, tm in selected:
        row = difficulty[lab.upper()]
        rows.append(
            {
                "label": lab,
                "cluster_id": cluster,
                "baseline_tm": tm,
                "baseline_lddt_ca": float(row.get("baseline_lddt_ca") or 0.0),
                "difficulty": row.get("difficulty", "hard"),
                "in_train": True,
            }
        )

    manifest = {
        "client": "client_0",
        "n_labels": 8,
        "n_clusters": 8,
        "labels": labels,
        "clusters": clusters,
        "rows": rows,
        "all_hard": True,
        "train_only": True,
        "cluster_leakage_vs_validation": False,
        "cluster_leakage_vs_development": False,
        "labels_sha256": sha256_file(labels_path),
        "train_labels_sha256": sha256_file(split_root / "train_labels.txt"),
        "validation_labels_sha256": sha256_file(split_root / "validation_labels.txt"),
        "development_labels_sha256": sha256_file(
            split_root / "development_test_labels.txt"
        ),
        "chain_cache_sha256": sha256_file(cache_dst),
        "paths": {
            "labels": str(labels_path),
            "chain_cache": str(cache_dst),
            "assets": str(out_root / "assets"),
        },
        "capacity_budget": {
            "max_optimizer_steps": 200,
            "epoch_len": 8,
            "max_epochs": 25,
            "eval_steps": [40, 80, 120, 200],
            "warmup_steps": 5,
        },
    }
    manifest_path = out_root / "overfit8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--copy", action="store_true", help="Copy assets instead of symlink")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_overfit_subset(
        run_root=args.run_root,
        source_run=args.source_run,
        link=not args.copy,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
