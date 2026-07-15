#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
except ImportError as e:
    raise SystemExit(
        "Biopython is required for this script. Install with: pip install biopython"
    ) from e


DEFAULT_PRED_SUFFIX = "_seq_model_esm1b_ptm_unrelaxed"


class _SingleChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):  # type: ignore[override]
        return chain.id == self.chain_id


def _iter_pred_pdbs(pred_inputs: List[Path]) -> Iterable[Path]:
    files: List[Path] = []
    for p in pred_inputs:
        if p.is_file() and p.suffix.lower() == ".pdb":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.pdb")))
        else:
            raise FileNotFoundError(f"Prediction input not found or invalid: {p}")

    seen: Set[Path] = set()
    for f in files:
        rf = f.resolve()
        if rf in seen:
            continue
        seen.add(rf)
        yield f


def _load_labels_from_csv(csv_path: Path) -> Set[str]:
    labels: Set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "protein_name" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain column 'protein_name'")
        for row in reader:
            v = (row.get("protein_name") or "").strip()
            if v:
                labels.add(v)
    return labels


def _load_labels_from_txt(txt_path: Path) -> Set[str]:
    labels: Set[str] = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                labels.add(v)
    return labels


def _label_from_pred_stem(stem: str, strip_suffix: str) -> str:
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def _parse_label(label: str) -> Tuple[str, str]:
    if "_" not in label:
        raise ValueError(f"Label has no chain separator '_': {label}")
    pdb_id, chain_id = label.split("_", 1)
    if not pdb_id or not chain_id:
        raise ValueError(f"Invalid label format: {label}")
    return pdb_id.lower(), chain_id


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract native single-chain PDBs from mmCIF files by labels "
            "(e.g., 9abc_A)."
        )
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        required=True,
        help="Directory containing source mmCIF files (named like <pdb_id>.cif).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for extracted single-chain PDB files.",
    )
    parser.add_argument(
        "--pred",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Prediction PDB file(s)/dir(s). Labels are inferred from filenames. "
            "Use this OR --labels-csv/--labels-txt."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with column 'protein_name' as labels.",
    )
    parser.add_argument(
        "--labels-txt",
        type=Path,
        default=None,
        help="Optional txt labels file, one label per line.",
    )
    parser.add_argument(
        "--strip-suffix",
        type=str,
        default=DEFAULT_PRED_SUFFIX,
        help="Suffix to strip from prediction filename stems.",
    )
    parser.add_argument(
        "--strict-chain-match",
        action="store_true",
        help="Fail if requested chain ID is not found in mmCIF model 0.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("docs/selectedPDB/extract_native_chain_report.csv"),
        help="CSV report for extraction status.",
    )
    parser.add_argument(
        "--fail-log",
        type=Path,
        default=Path("docs/selectedPDB/extract_native_chain_failures.txt"),
        help="Text log for failed labels.",
    )

    args = parser.parse_args(argv)

    if not args.cif_dir.exists():
        raise SystemExit(f"cif dir not found: {args.cif_dir}")

    labels: Set[str] = set()
    if args.pred:
        for p in _iter_pred_pdbs(args.pred):
            labels.add(_label_from_pred_stem(p.stem, args.strip_suffix))
    if args.labels_csv is not None:
        labels.update(_load_labels_from_csv(args.labels_csv))
    if args.labels_txt is not None:
        labels.update(_load_labels_from_txt(args.labels_txt))

    if not labels:
        raise SystemExit(
            "No labels found. Provide at least one of: --pred / --labels-csv / --labels-txt"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    args.fail_log.parent.mkdir(parents=True, exist_ok=True)

    parser_cif = MMCIFParser(QUIET=True)
    io = PDBIO()

    report_rows: List[List[str]] = []
    failures: List[Tuple[str, str]] = []

    for label in sorted(labels):
        try:
            pdb_id, chain_id = _parse_label(label)
            cif_path = args.cif_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                msg = f"missing cif: {cif_path}"
                failures.append((label, msg))
                report_rows.append([label, "", "", "", "fail", msg])
                continue

            structure = parser_cif.get_structure(label, str(cif_path))
            model = structure[0]
            if chain_id not in model.child_dict:
                msg = f"chain not found in cif model0: {chain_id}"
                failures.append((label, msg))
                report_rows.append([label, str(cif_path), chain_id, "", "fail", msg])
                if args.strict_chain_match:
                    continue
                else:
                    continue

            out_path = args.out_dir / f"{label}.pdb"
            io.set_structure(structure)
            io.save(str(out_path), select=_SingleChainSelect(chain_id))
            report_rows.append(
                [label, str(cif_path), chain_id, str(out_path), "ok", ""]
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            failures.append((label, msg))
            report_rows.append([label, "", "", "", "fail", msg])

    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label", "cif_path", "chain_id", "out_pdb", "status", "message"]
        )
        writer.writerows(report_rows)

    with args.fail_log.open("w", encoding="utf-8") as f:
        f.write("label\treason\n")
        for label, reason in failures:
            f.write(f"{label}\t{reason}\n")

    ok_count = sum(1 for r in report_rows if r[4] == "ok")
    fail_count = len(report_rows) - ok_count
    print(f"Done. extracted={ok_count}, failed={fail_count}")
    print(f"Report: {args.report_csv}")
    print(f"Fail log: {args.fail_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
