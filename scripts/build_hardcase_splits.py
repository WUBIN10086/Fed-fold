#!/usr/bin/env python3
"""Build hard-case train/val/development_test splits and external final test.

Keeps existing client ownership and locks the current development_test labels.
Carves validation from the previous train pools by sequence cluster.
Builds an external final set from leftover quality-kept, cluster-disjoint chains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_fed_test_set import (  # noqa: E402
    build_label_to_cluster,
    difficulty_band,
    difficulty_distribution,
    group_by_cluster,
    labels_sha256,
    read_labels,
    split_by_cluster,
    write_reclustered_chain_cache,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_labels(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{lab}\n" for lab in labels), encoding="utf-8")


def read_difficulty_csv(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip()
            rows[label.upper()] = row
    return rows


def tm_map_from_difficulty(rows: dict[str, dict]) -> dict[str, float]:
    out = {}
    for key, row in rows.items():
        raw = row.get("baseline_tm", "")
        if raw == "" or row.get("tm_status") != "ok":
            continue
        out[key] = float(raw)
    return out


def cluster_ids(labels: list[str], label2cluster: dict) -> list[str]:
    clusters = group_by_cluster(labels, label2cluster)
    return sorted(clusters.keys())


def assert_disjoint(*cluster_sets: set[str], name: str) -> None:
    for i, left in enumerate(cluster_sets):
        for j, right in enumerate(cluster_sets):
            if j <= i:
                continue
            overlap = left & right
            if overlap:
                raise AssertionError(f"{name} cluster overlap: {sorted(overlap)[:10]}")


def copy_assets_for_labels(
    labels: list[str],
    client_src: Path,
    dest: Path,
    link: bool,
) -> None:
    fasta_dirs = [
        client_src / "solo_fasta_dir_finetune",
        client_src / "solo_fasta_dir",
    ]
    emb_dirs = [
        client_src / "solo_alignment",
        client_src / "solo_alignment_dir",
    ]
    cif_dirs = [
        client_src / "mmcif_files_finetune",
        client_src / "mmcif_files",
    ]
    out_fasta = dest / "solo_fasta_dir"
    out_emb = dest / "solo_alignment_dir"
    out_cif = dest / "mmcif_files"
    for path in (out_fasta, out_emb, out_cif):
        path.mkdir(parents=True, exist_ok=True)

    for lab in labels:
        for fasta_dir in fasta_dirs:
            src = fasta_dir / f"{lab}.fasta"
            if src.exists():
                _copy_or_link(src, out_fasta / f"{lab}.fasta", link)
                break
        for emb_dir in emb_dirs:
            src = emb_dir / lab
            if src.exists():
                _copy_or_link(src, out_emb / lab, link)
                break
        for cif_dir in cif_dirs:
            # labels are PDBID_CHAIN; cif is usually PDBID.cif
            pdb_id = lab.split("_")[0].lower()
            candidates = [
                cif_dir / f"{pdb_id}.cif",
                cif_dir / f"{lab}.cif",
                cif_dir / f"{pdb_id.upper()}.cif",
            ]
            for src in candidates:
                if src.exists():
                    _copy_or_link(src, out_cif / src.name, link)
                    break


def _copy_or_link(src: Path, dst: Path, link: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        dst.symlink_to(src.resolve())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def leftover_labels(
    source_run: Path,
    used_clusters: set[str],
    label2cluster: dict,
    num_clients: int,
) -> list[tuple[str, str]]:
    """Return (label, client) leftover kept chains with unused clusters."""
    out = []
    for idx in range(num_clients):
        client = f"client_{idx}"
        # All quality-kept / finetune candidates for this client live under reports
        # or solo_fasta; use prescreen prediction stems as available leftovers.
        pred_dir = source_run / "clients" / client / "prescreen" / "predictions"
        finetune = set(
            read_labels(
                source_run
                / "clients"
                / client
                / "solo_fasta_dir_finetune"
                / "_reports"
                / "kept_labels.txt"
            )
        )
        if not pred_dir.exists():
            continue
        for pdb in pred_dir.glob("*_unrelaxed.pdb"):
            name = pdb.name
            label = name.split("_seq_model_esm1b_ptm_unrelaxed.pdb")[0]
            if label in finetune:
                continue
            key = label.upper()
            cid = label2cluster.get(key)
            cluster = f"c{cid}" if cid is not None else f"singleton:{key}"
            if cluster in used_clusters:
                continue
            out.append((label, client))
    return out


def build_novelty_rows(
    difficulty_rows: dict[str, dict],
    base_cutoff: str,
) -> list[dict]:
    rows = []
    for key, row in sorted(difficulty_rows.items()):
        release = row.get("release_date") or ""
        after = ""
        if release:
            after = str(release[:10] > base_cutoff[:10])
        rows.append(
            {
                "label": row["label"],
                "client": row["client"],
                "release_date": release,
                "after_base_cutoff": after,
                "nearest_base_train_identity": "unknown",
                "nearest_base_cluster": "unknown",
                "exact_sequence_seen": "unknown",
                "novelty_status": (
                    "after_cutoff_identity_unknown"
                    if after == "True"
                    else ("before_or_unknown_cutoff" if release else "unknown")
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=REPO / "outputs" / "fed_lora_fp32",
    )
    parser.add_argument("--cluster-file", type=Path, required=True)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--min-val-clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-size", type=int, default=80)
    parser.add_argument("--base-cutoff-date", default="2021-09-30")
    parser.add_argument("--link", action="store_true", default=True)
    args = parser.parse_args()

    label2cluster = build_label_to_cluster(args.cluster_file)
    source = args.source_run
    run_root = args.run_root
    splits_root = run_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    union_diff = run_root / "difficulty" / "baseline_difficulty.csv"
    if not union_diff.exists():
        raise SystemExit(
            f"Missing {union_diff}; run build_baseline_difficulty.py first"
        )
    all_diff = read_difficulty_csv(union_diff)
    tm_difficulty = tm_map_from_difficulty(all_diff)

    used_clusters: set[str] = set()
    used_labels: set[str] = set()
    client_manifests = []
    all_dev_test: list[tuple[str, str]] = []

    for idx in range(args.num_clients):
        client = f"client_{idx}"
        split_src = source / "split" / client
        client_src = source / "clients" / client
        private = run_root / "clients" / client / "private" / "splits"
        private.mkdir(parents=True, exist_ok=True)

        # Lock development_test to the previously published test holdout.
        development_test = read_labels(split_src / "test_labels.txt")
        train_pool = read_labels(split_src / "train_labels.txt")

        train_labels, val_labels, train_clusters, val_clusters = split_by_cluster(
            train_pool,
            label2cluster,
            args.val_frac,
            random.Random(args.seed + 1000 + idx),
            min_test_clusters=args.min_val_clusters,
            difficulty=tm_difficulty,
        )
        dev_clusters = cluster_ids(development_test, label2cluster)
        assert_disjoint(
            set(train_clusters),
            set(val_clusters),
            set(dev_clusters),
            name=client,
        )
        if set(map(str.upper, train_labels)) & set(map(str.upper, val_labels)):
            raise AssertionError(f"{client} train/val label overlap")
        if set(map(str.upper, train_labels)) & set(
            map(str.upper, development_test)
        ):
            raise AssertionError(f"{client} train/dev label overlap")

        write_labels(private / "train_labels.txt", train_labels)
        write_labels(private / "validation_labels.txt", val_labels)
        write_labels(private / "development_test_labels.txt", development_test)
        # Also mirror under splits/ for convenience in single-machine simulation.
        public_client = splits_root / client
        write_labels(public_client / "train_labels.txt", train_labels)
        write_labels(public_client / "validation_labels.txt", val_labels)
        write_labels(public_client / "development_test_labels.txt", development_test)

        source_cache = split_src / "train_chain_data_cache.json"
        if not source_cache.exists():
            source_cache = client_src / "chain_data_cache_finetune.json"
        write_reclustered_chain_cache(
            source_cache,
            private / "train_chain_data_cache.json",
            train_labels,
            label2cluster,
        )
        shutil.copy2(
            private / "train_chain_data_cache.json",
            public_client / "train_chain_data_cache.json",
        )

        copy_assets_for_labels(
            train_labels,
            client_src,
            private / "assets_train",
            link=args.link,
        )
        copy_assets_for_labels(
            val_labels,
            client_src,
            private / "assets_validation",
            link=args.link,
        )
        copy_assets_for_labels(
            development_test,
            client_src,
            private / "assets_development_test",
            link=args.link,
        )

        hard_val_clusters = 0
        val_grouped = group_by_cluster(val_labels, label2cluster)
        for members in val_grouped.values():
            values = [
                tm_difficulty[m.upper()]
                for m in members
                if m.upper() in tm_difficulty
            ]
            if values and difficulty_band(sum(values) / len(values)) == "hard":
                hard_val_clusters += 1

        client_manifest = {
            "client": client,
            "train_label_count": len(train_labels),
            "validation_label_count": len(val_labels),
            "development_test_label_count": len(development_test),
            "train_cluster_count": len(train_clusters),
            "validation_cluster_count": len(val_clusters),
            "development_test_cluster_count": len(dev_clusters),
            "hard_validation_clusters": hard_val_clusters,
            "train_difficulty": difficulty_distribution(train_labels, tm_difficulty),
            "validation_difficulty": difficulty_distribution(
                val_labels, tm_difficulty
            ),
            "development_test_difficulty": difficulty_distribution(
                development_test, tm_difficulty
            ),
            "train_labels_sha256": labels_sha256(train_labels),
            "validation_labels_sha256": labels_sha256(val_labels),
            "development_test_labels_sha256": labels_sha256(development_test),
            "cluster_leakage": False,
            "paths": {
                "private_splits": str(private),
                "difficulty_csv": str(
                    run_root
                    / "clients"
                    / client
                    / "private"
                    / "difficulty"
                    / "baseline_difficulty.csv"
                ),
            },
        }
        (private / "split_manifest.json").write_text(
            json.dumps(client_manifest, indent=2, sort_keys=True) + "\n"
        )
        client_manifests.append(client_manifest)

        used_clusters.update(train_clusters)
        used_clusters.update(val_clusters)
        used_clusters.update(dev_clusters)
        used_labels.update(lab.upper() for lab in train_labels + val_labels + development_test)
        all_dev_test.extend((lab, client) for lab in development_test)
        print(
            f"[{client}] train={len(train_labels)} val={len(val_labels)} "
            f"dev={len(development_test)} hard_val_clusters={hard_val_clusters}"
        )

    def _client_all_clusters(client_name: str) -> set[str]:
        clusters = set()
        base = run_root / "clients" / client_name / "private" / "splits"
        for name in (
            "train_labels.txt",
            "validation_labels.txt",
            "development_test_labels.txt",
        ):
            clusters.update(cluster_ids(read_labels(base / name), label2cluster))
        return clusters

    # Cross-client cluster overlap check
    for i, left in enumerate(client_manifests):
        left_clusters = _client_all_clusters(left["client"])
        for right in client_manifests[i + 1 :]:
            overlap = left_clusters & _client_all_clusters(right["client"])
            if overlap:
                raise AssertionError(
                    f"Unexpected client cluster overlap "
                    f"{left['client']}/{right['client']}: {sorted(overlap)[:10]}"
                )

    leftovers = leftover_labels(
        source, used_clusters, label2cluster, args.num_clients
    )
    rng = random.Random(args.seed + 999)
    # Stratify leftover by difficulty when available (often unknown until scored).
    by_band = {"hard": [], "medium": [], "easy": [], "unknown": []}
    for label, client in leftovers:
        key = label.upper()
        band = difficulty_band(tm_difficulty.get(key))
        by_band[band].append((label, client))
    for band in by_band:
        rng.shuffle(by_band[band])
    external = []
    bands = ["hard", "medium", "easy", "unknown"]
    while len(external) < args.external_size and any(by_band.values()):
        for band in bands:
            if len(external) >= args.external_size:
                break
            if by_band[band]:
                external.append(by_band[band].pop())
    if len(external) < args.external_size:
        print(
            f"[warn] external final only {len(external)} < requested "
            f"{args.external_size}"
        )

    ext_dir = splits_root / "external_final"
    ext_dir.mkdir(parents=True, exist_ok=True)
    ext_labels = [lab for lab, _ in external]
    write_labels(ext_dir / "labels.txt", ext_labels)
    with (ext_dir / "label_client_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "source_client"])
        writer.writeheader()
        for lab, client in external:
            writer.writerow({"label": lab, "source_client": client})

    ext_clusters = set(cluster_ids(ext_labels, label2cluster))
    if ext_clusters & used_clusters:
        raise AssertionError("external final overlaps development clusters")

    # Development union map
    dev_dir = splits_root / "development_test"
    dev_dir.mkdir(parents=True, exist_ok=True)
    write_labels(dev_dir / "labels.txt", [lab for lab, _ in all_dev_test])
    with (dev_dir / "label_client_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "client"])
        writer.writeheader()
        for lab, client in all_dev_test:
            writer.writerow({"label": lab, "client": client})

    novelty_rows = build_novelty_rows(all_diff, args.base_cutoff_date)
    novelty_path = run_root / "reports" / "novelty_audit.csv"
    novelty_path.parent.mkdir(parents=True, exist_ok=True)
    with novelty_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "client",
                "release_date",
                "after_base_cutoff",
                "nearest_base_train_identity",
                "nearest_base_cluster",
                "exact_sequence_seen",
                "novelty_status",
            ],
        )
        writer.writeheader()
        for row in novelty_rows:
            writer.writerow(row)

    min_hard = min(m["hard_validation_clusters"] for m in client_manifests)
    split_manifest = {
        "method": "locked_dev_test_plus_cluster_val_from_train",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "min_val_clusters": args.min_val_clusters,
        "hard_threshold": 0.5,
        "baseline_difficulty_csv": str(union_diff),
        "baseline_difficulty_sha256": sha256_file(union_diff),
        "cluster_file": str(args.cluster_file.resolve()),
        "cluster_file_sha256": sha256_file(args.cluster_file),
        "source_run": str(source.resolve()),
        "clients": client_manifests,
        "min_hard_validation_clusters": min_hard,
        "use_macro_validation_for_shared_hparams": min_hard < 3,
        "external_final": {
            "label_count": len(ext_labels),
            "cluster_count": len(ext_clusters),
            "labels_sha256": labels_sha256(ext_labels),
            "locked": False,
            "path": str(ext_dir),
        },
        "development_test": {
            "label_count": len(all_dev_test),
            "labels_sha256": labels_sha256([lab for lab, _ in all_dev_test]),
            "path": str(dev_dir),
            "note": "Already analyzed; not for final statistical claims",
        },
        "novelty_audit_csv": str(novelty_path),
        "final_test_lock_state": "unlocked",
    }
    manifest_path = splits_root / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {manifest_path}")
    print(
        f"macro_validation_required={min_hard < 3} "
        f"(min_hard_val_clusters={min_hard})"
    )


if __name__ == "__main__":
    main()
