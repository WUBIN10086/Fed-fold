"""Create a locked cluster-aware validation split inside fed_test_v2 train."""

import argparse
import json
import os
import random
from pathlib import Path

from scripts.build_fed_test_set import (
    build_label_to_cluster,
    labels_sha256,
    read_labels,
    split_by_cluster,
)


REPO = Path(__file__).resolve().parents[1]


def link(src, dst):
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        os.symlink(src.resolve(), dst)


def prepare(
    output_root,
    seed=42,
    validation_fraction=0.2,
    min_validation_clusters=5,
):
    source_labels = (
        REPO / "data/all_pdb_1y/fed_test_v2/client_1/train_labels.txt"
    )
    cluster_file = REPO / "data/all_pdb_1y/clusters_30.txt"
    labels = read_labels(source_labels)
    label_to_cluster = build_label_to_cluster(cluster_file)
    train, validation, train_clusters, validation_clusters = split_by_cluster(
        labels,
        label_to_cluster,
        validation_fraction,
        random.Random(seed),
        min_test_clusters=min_validation_clusters,
    )
    data = output_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "search_train_labels.txt").write_text(
        "".join(f"{label}\n" for label in train),
        encoding="utf-8",
    )
    (data / "validation_labels.txt").write_text(
        "".join(f"{label}\n" for label in validation),
        encoding="utf-8",
    )
    source = REPO / "data/all_pdb_1y/fed_split/client_1"
    validation_root = data / "validation"
    for label in validation:
        pdb_id = label.rsplit("_", 1)[0].lower()
        link(
            source / "solo_fasta_dir" / f"{label}.fasta",
            validation_root / "solo_fasta_dir" / f"{label}.fasta",
        )
        link(
            source / "solo_alignment" / label,
            validation_root / "solo_alignment_dir" / label,
        )
        link(
            source / "mmcif_files" / f"{pdb_id}.cif",
            validation_root / "mmcif_files" / f"{pdb_id}.cif",
        )
    manifest = {
        "seed": seed,
        "source_labels": str(source_labels),
        "source_labels_sha256": labels_sha256(labels),
        "search_train_count": len(train),
        "validation_count": len(validation),
        "search_train_cluster_ids": train_clusters,
        "validation_cluster_ids": validation_clusters,
        "search_train_labels_sha256": labels_sha256(train),
        "validation_labels_sha256": labels_sha256(validation),
        "cluster_leakage": bool(
            set(train_clusters) & set(validation_clusters)
        ),
    }
    (data / "validation_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO
            / "data/all_pdb_1y/client1_res/lora_search_ema_fp32_v2"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-validation-clusters", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.output_root,
        args.seed,
        args.validation_fraction,
        args.min_validation_clusters,
    ), indent=2))


if __name__ == "__main__":
    main()
