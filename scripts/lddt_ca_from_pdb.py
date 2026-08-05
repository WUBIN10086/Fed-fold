#!/usr/bin/env python3
"""Compute sequence-aligned C-alpha lDDT for predicted/native PDB pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner


PREDICTION_SUFFIX = "_seq_model_esm1b_ptm_unrelaxed"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


@dataclass
class Structure:
    sequence: str
    ca: np.ndarray


def read_ca_structure(path: Path) -> Structure:
    residues: list[str] = []
    coordinates: list[list[float]] = []
    seen = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            key = (line[21:22], line[22:26], line[26:27])
            if key in seen:
                continue
            seen.add(key)
            residues.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    if not coordinates:
        raise ValueError(f"No C-alpha atoms found in {path}")
    return Structure("".join(residues), np.asarray(coordinates, dtype=np.float64))


def aligned_indices(pred_sequence: str, native_sequence: str) -> tuple[np.ndarray, np.ndarray]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(pred_sequence, native_sequence)[0]

    pred_indices: list[int] = []
    native_indices: list[int] = []
    for (pred_start, pred_end), (native_start, native_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        length = min(pred_end - pred_start, native_end - native_start)
        for offset in range(length):
            pred_index = int(pred_start + offset)
            native_index = int(native_start + offset)
            pred_indices.append(pred_index)
            native_indices.append(native_index)
    return np.asarray(pred_indices), np.asarray(native_indices)


def compute_lddt_ca(
    pred: Structure,
    native: Structure,
    cutoff: float = 15.0,
) -> tuple[float, int, int]:
    pred_indices, native_indices = aligned_indices(pred.sequence, native.sequence)
    if len(pred_indices) < 2:
        raise ValueError("Fewer than two sequence-aligned C-alpha atoms")

    pred_ca = pred.ca[pred_indices]
    native_ca = native.ca[native_indices]
    pred_dist = np.linalg.norm(pred_ca[:, None] - pred_ca[None, :], axis=-1)
    native_dist = np.linalg.norm(
        native_ca[:, None] - native_ca[None, :], axis=-1
    )
    pair_mask = (native_dist < cutoff) & ~np.eye(len(native_ca), dtype=bool)
    scored_pairs = int(np.count_nonzero(pair_mask))
    if scored_pairs == 0:
        raise ValueError("No native C-alpha pairs within the lDDT cutoff")

    errors = np.abs(pred_dist - native_dist)[pair_mask]
    score = np.stack(
        [
            errors < 0.5,
            errors < 1.0,
            errors < 2.0,
            errors < 4.0,
        ]
    ).mean()
    return float(score), len(pred_indices), scored_pairs


def prediction_label(path: Path) -> str:
    stem = path.stem
    if stem.endswith(PREDICTION_SUFFIX):
        return stem[: -len(PREDICTION_SUFFIX)]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--missing-log", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=15.0)
    args = parser.parse_args()

    predictions = sorted(args.pred_dir.rglob("*_unrelaxed.pdb"))
    if not predictions:
        raise FileNotFoundError(f"No predicted PDB files found in {args.pred_dir}")

    rows = []
    issues = []
    for pred_path in predictions:
        label = prediction_label(pred_path)
        native_path = args.native_dir / f"{label}.pdb"
        if not native_path.exists():
            issues.append((label, "missing native"))
            continue
        try:
            score, aligned_ca, scored_pairs = compute_lddt_ca(
                read_ca_structure(pred_path),
                read_ca_structure(native_path),
                cutoff=args.cutoff,
            )
            rows.append(
                {
                    "label": label,
                    "pred_pdb": str(pred_path),
                    "native_pdb": str(native_path),
                    "lddt_ca": score,
                    "lddt_ca_percent": score * 100.0,
                    "aligned_ca": aligned_ca,
                    "scored_pairs": scored_pairs,
                    "status": "ok",
                }
            )
        except Exception as exc:
            issues.append((label, str(exc)))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label", "pred_pdb", "native_pdb", "lddt_ca",
                "lddt_ca_percent", "aligned_ca", "scored_pairs", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    args.missing_log.write_text(
        "".join(f"{label}\t{issue}\n" for label, issue in issues),
        encoding="utf-8",
    )
    if issues:
        raise RuntimeError(
            f"{len(issues)} lDDT-Ca evaluations failed; see {args.missing_log}"
        )
    mean_score = sum(row["lddt_ca"] for row in rows) / len(rows)
    print(f"lDDT-Ca: n={len(rows)}, mean={mean_score:.6f}")


if __name__ == "__main__":
    main()
