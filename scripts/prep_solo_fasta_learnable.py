import argparse
import csv
import logging
import os
from collections import Counter
from pathlib import Path

from openfold.data import mmcif_parsing


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_target_labels(path: str) -> set[str]:
    """
    支持两种格式：
    1) 纯文本：每行一个 label，如 1abc_A
    2) csv：优先读取 protein_name / label / chain_id 字段
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Target labels file not found: {path}")

    labels: set[str] = set()
    if p.suffix.lower() == ".csv":
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            candidate_cols = ["protein_name", "label", "chain_id"]
            selected_col = None
            for c in candidate_cols:
                if c in fieldnames:
                    selected_col = c
                    break
            if selected_col is None:
                raise ValueError(
                    f"CSV missing expected columns {candidate_cols}: {path}"
                )
            for row in reader:
                v = (row.get(selected_col) or "").strip()
                if v:
                    labels.add(v)
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                v = line.strip()
                if v:
                    labels.add(v)

    return labels


def analyze_sequence(seq: str) -> dict:
    length = len(seq)
    counts = Counter(seq)
    x_count = counts.get("X", 0)
    canonical_count = sum(counts.get(a, 0) for a in CANONICAL_AA)
    noncanonical_count = length - canonical_count
    x_ratio = (x_count / length) if length > 0 else 1.0
    noncanonical_ratio = (noncanonical_count / length) if length > 0 else 1.0
    return {
        "length": length,
        "x_count": x_count,
        "x_ratio": x_ratio,
        "noncanonical_count": noncanonical_count,
        "noncanonical_ratio": noncanonical_ratio,
    }


def should_keep(
    label: str,
    seq: str,
    resolution,
    args,
    target_labels: set[str] | None,
    seen_sequences: set[str],
) -> tuple[bool, str]:
    if target_labels is not None and label not in target_labels:
        return False, "not_in_target_labels"

    stats = analyze_sequence(seq)
    n = stats["length"]
    if n < args.min_len:
        return False, "too_short"
    if n > args.max_len:
        return False, "too_long_for_esm1b"

    if stats["x_ratio"] > args.max_x_ratio:
        return False, "x_ratio_too_high"
    if stats["noncanonical_ratio"] > args.max_noncanonical_ratio:
        return False, "noncanonical_ratio_too_high"

    # 过滤掉几乎全是未知残基的样本
    if stats["x_count"] == n:
        return False, "all_unknown_x"

    if args.max_resolution is not None:
        try:
            r = float(resolution)
            if r > args.max_resolution:
                return False, "resolution_too_low_quality"
        except Exception:
            return False, "invalid_resolution"

    if args.dedup_by_sequence:
        if seq in seen_sequences:
            return False, "duplicate_sequence"
        seen_sequences.add(seq)

    return True, "kept"


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_labels = None
    if args.target_labels_file:
        target_labels = read_target_labels(args.target_labels_file)
        logging.info("Loaded %d target labels", len(target_labels))

    seen_sequences: set[str] = set()
    kept_rows = []
    dropped_rows = []

    cif_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".cif"])
    logging.info("Found %d cif files in %s", len(cif_files), input_dir)

    for cif_path in cif_files:
        file_id = cif_path.stem
        with cif_path.open("r", encoding="utf-8") as fp:
            mmcif_str = fp.read()

        parsed = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_str)
        if parsed.mmcif_object is None:
            dropped_rows.append(
                {
                    "label": file_id,
                    "reason": "parse_failed",
                    "length": "",
                    "x_ratio": "",
                    "noncanonical_ratio": "",
                    "resolution": "",
                    "source_file": str(cif_path),
                }
            )
            continue

        obj = parsed.mmcif_object
        resolution = obj.header.get("resolution", None)

        for chain_id, seq in obj.chain_to_seqres.items():
            label = f"{file_id}_{chain_id}"
            seq = (seq or "").strip().upper()
            stats = analyze_sequence(seq)
            keep, reason = should_keep(
                label=label,
                seq=seq,
                resolution=resolution,
                args=args,
                target_labels=target_labels,
                seen_sequences=seen_sequences,
            )

            row = {
                "label": label,
                "reason": reason,
                "length": stats["length"],
                "x_ratio": f"{stats['x_ratio']:.4f}",
                "noncanonical_ratio": f"{stats['noncanonical_ratio']:.4f}",
                "resolution": resolution,
                "source_file": str(cif_path),
            }

            if keep:
                fasta_path = output_dir / f"{label}.fasta"
                with fasta_path.open("w", encoding="utf-8") as out:
                    out.write(f">{label}\n")
                    out.write(seq + "\n")
                kept_rows.append(row)
            else:
                dropped_rows.append(row)

    # 写汇总报告
    kept_csv = report_dir / "kept_samples.csv"
    dropped_csv = report_dir / "dropped_samples.csv"
    summary_txt = report_dir / "summary.txt"
    kept_labels_txt = report_dir / "kept_labels.txt"
    dropped_labels_txt = report_dir / "dropped_labels.txt"

    fieldnames = [
        "label",
        "reason",
        "length",
        "x_ratio",
        "noncanonical_ratio",
        "resolution",
        "source_file",
    ]

    with kept_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    with dropped_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped_rows)

    with kept_labels_txt.open("w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(r["label"] + "\n")

    with dropped_labels_txt.open("w", encoding="utf-8") as f:
        for r in dropped_rows:
            f.write(f'{r["label"]}\t{r["reason"]}\n')

    reason_counts = Counter([r["reason"] for r in dropped_rows])
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"input_dir: {input_dir}\n")
        f.write(f"output_dir: {output_dir}\n")
        f.write(f"total_cif_files: {len(cif_files)}\n")
        f.write(f"kept_fasta_files: {len(kept_rows)}\n")
        f.write(f"dropped_samples: {len(dropped_rows)}\n")
        f.write("\n[dropped_reason_counts]\n")
        for reason, count in reason_counts.most_common():
            f.write(f"{reason}: {count}\n")

    logging.info("Done. Kept=%d Dropped=%d", len(kept_rows), len(dropped_rows))
    logging.info("Summary: %s", summary_txt)
    logging.info("Kept list: %s", kept_labels_txt)
    logging.info("Dropped list: %s", dropped_labels_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate learnable SOLO FASTA samples from mmCIF with quality filters."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Input directory containing mmCIF files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for filtered FASTA files.",
    )
    parser.add_argument(
        "--target_labels_file",
        type=str,
        default=None,
        help=(
            "Optional label whitelist (txt/csv). "
            "If provided, only labels in this file are kept."
        ),
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=30,
        help="Minimum chain length to keep.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1022,
        help="Maximum chain length (ESM-1b limit is 1022).",
    )
    parser.add_argument(
        "--max_x_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed X ratio in sequence.",
    )
    parser.add_argument(
        "--max_noncanonical_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed ratio of non-canonical residues.",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=4.0,
        help="Maximum allowed resolution (Angstrom). Use <=0 to disable.",
    )
    parser.add_argument(
        "--dedup_by_sequence",
        action="store_true",
        help="If enabled, remove duplicate sequences and keep first occurrence.",
    )

    args = parser.parse_args()
    if args.max_resolution is not None and args.max_resolution <= 0:
        args.max_resolution = None
    main(args)
