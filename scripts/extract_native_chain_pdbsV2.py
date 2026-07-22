#!/usr/bin/env python3
"""
从 mmCIF 提取单链 native PDB,坐标与编号严格对齐到完整 SEQRES(1..N)。

关键区别(相比原 Biopython 版):
  - 用 OpenFold 的 mmcif_parsing + get_atom_coords,得到按 SEQRES 对齐的
    [N,37,3] 坐标和 [N,37] mask;缺失残基自动 mask=0。
  - 用 OpenFold 的 protein.to_pdb 写出,残基编号 = SEQRES 位置(1..N),
    与 SoloSeq 预测(同样 1..N 全长)逐位对应 => TMscore 能正确配对。
  - 链命名与 prep 脚本用同一套 chain_to_seqres 的 key,消除 auth/label 错位
    (原来的 9vak_B 之类问题)。
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from openfold.data import mmcif_parsing
from openfold.np import protein as protein_lib
from openfold.np import residue_constants as rc


DEFAULT_PRED_SUFFIX = "_seq_model_esm1b_ptm_unrelaxed"


# ----------------------- label / 输入解析(沿用原逻辑) -----------------------
def _iter_pred_labels(pred_inputs: List[Path], strip_suffix: str) -> Set[str]:
    labels: Set[str] = set()
    for p in pred_inputs:
        files: List[Path] = []
        if p.is_file() and p.suffix.lower() == ".pdb":
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob("*.pdb"))
        else:
            raise FileNotFoundError(f"Prediction input not found or invalid: {p}")
        for f in files:
            stem = f.stem
            if strip_suffix and stem.endswith(strip_suffix):
                stem = stem[: -len(strip_suffix)]
            labels.add(stem)
    return labels


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


def _parse_label(label: str) -> Tuple[str, str]:
    if "_" not in label:
        raise ValueError(f"Label has no chain separator '_': {label}")
    pdb_id, chain_id = label.split("_", 1)
    if not pdb_id or not chain_id:
        raise ValueError(f"Invalid label format: {label}")
    return pdb_id.lower(), chain_id


# ----------------------- 核心:按 SEQRES 对齐写 native -----------------------
def _seq_to_aatype(seq: str) -> np.ndarray:
    order = getattr(rc, "restype_order_with_x", rc.restype_order)
    x_idx = getattr(rc, "restype_num", 20)
    return np.array([order.get(aa, x_idx) for aa in seq], dtype=np.int64)


def _build_protein(atom_positions, atom_mask, aatype, n):
    """兼容不同 OpenFold 版本的 Protein 签名(有无 chain_index)。"""
    kwargs = dict(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(1, n + 1, dtype=np.int64),  # -> 输出 1..N
        b_factors=np.zeros_like(atom_mask),
    )
    field_names = {f.name for f in dataclasses.fields(protein_lib.Protein)}
    if "chain_index" in field_names:
        kwargs["chain_index"] = np.zeros(n, dtype=np.int64)
    return protein_lib.Protein(**kwargs)


def extract_one(cif_path: Path, chain_id: str, out_path: Path) -> Tuple[bool, str]:
    file_id = cif_path.stem
    mmcif_str = cif_path.read_text(encoding="utf-8")
    parsed = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_str)
    if parsed.mmcif_object is None:
        return False, "parse_failed"

    obj = parsed.mmcif_object
    if chain_id not in obj.chain_to_seqres:
        return False, f"chain_not_in_seqres:{chain_id};have={list(obj.chain_to_seqres)}"

    seq = obj.chain_to_seqres[chain_id]
    n = len(seq)
    if n == 0:
        return False, "empty_seqres"

    # 按 SEQRES 对齐的坐标 [N,37,3] 与 mask [N,37];缺失残基 mask 全 0
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=obj, chain_id=chain_id
    )
    if all_atom_mask.sum() == 0:
        return False, "no_resolved_atoms"

    aatype = _seq_to_aatype(seq)
    prot = _build_protein(all_atom_positions, all_atom_mask, aatype, n)
    out_path.write_text(protein_lib.to_pdb(prot))
    n_resolved = int((all_atom_mask[:, 1] > 0.5).sum())  # 有 CA 的残基数
    return True, f"ok;seqres_len={n};resolved_ca={n_resolved}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract native single-chain PDBs aligned to full SEQRES (1..N)."
    )
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pred", nargs="*", type=Path, default=None,
                        help="Prediction PDB file(s)/dir(s); labels inferred from names.")
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--labels-txt", type=Path, default=None)
    parser.add_argument("--strip-suffix", type=str, default=DEFAULT_PRED_SUFFIX)
    parser.add_argument("--report-csv", type=Path,
                        default=Path("docs/selectedPDB/extract_native_chain_report.csv"))
    parser.add_argument("--fail-log", type=Path,
                        default=Path("docs/selectedPDB/extract_native_chain_failures.txt"))
    args = parser.parse_args(argv)

    if not args.cif_dir.exists():
        raise SystemExit(f"cif dir not found: {args.cif_dir}")

    labels: Set[str] = set()
    if args.pred:
        labels |= _iter_pred_labels(args.pred, args.strip_suffix)
    if args.labels_csv is not None:
        labels |= _load_labels_from_csv(args.labels_csv)
    if args.labels_txt is not None:
        labels |= _load_labels_from_txt(args.labels_txt)
    if not labels:
        raise SystemExit("No labels found. Provide --pred / --labels-csv / --labels-txt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    args.fail_log.parent.mkdir(parents=True, exist_ok=True)

    report_rows: List[List[str]] = []
    failures: List[Tuple[str, str]] = []

    for label in sorted(labels):
        try:
            pdb_id, chain_id = _parse_label(label)
            cif_path = args.cif_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                msg = f"missing cif: {cif_path}"
                failures.append((label, msg))
                report_rows.append([label, "", chain_id, "", "fail", msg])
                continue

            out_path = args.out_dir / f"{label}.pdb"
            ok, msg = extract_one(cif_path, chain_id, out_path)
            if ok:
                report_rows.append([label, str(cif_path), chain_id, str(out_path), "ok", msg])
            else:
                failures.append((label, msg))
                report_rows.append([label, str(cif_path), chain_id, "", "fail", msg])
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            failures.append((label, msg))
            report_rows.append([label, "", "", "", "fail", msg])

    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "cif_path", "chain_id", "out_pdb", "status", "message"])
        writer.writerows(report_rows)
    with args.fail_log.open("w", encoding="utf-8") as f:
        f.write("label\treason\n")
        for label, reason in failures:
            f.write(f"{label}\t{reason}\n")

    ok_count = sum(1 for r in report_rows if r[4] == "ok")
    print(f"Done. extracted={ok_count}, failed={len(report_rows) - ok_count}")
    print(f"Report: {args.report_csv}")
    print(f"Fail log: {args.fail_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
