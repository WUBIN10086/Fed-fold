#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

NAME_SUFFIX_TO_STRIP = "_seq_model_esm1b_ptm_unrelaxed"


@dataclass
class TMResult:
    label: str
    pred_pdb: Path
    native_pdb: Path
    tm_norm_chain1: float
    tm_norm_chain2: float
    tm_selected: float
    rmsd: Optional[float]
    aligned_len: Optional[int]
    status: str


def parse_tmscore_output(text: str) -> Tuple[float, float, Optional[float], Optional[int]]:
    """
    解析 TMscore 输出：
    - TM-score 通常会出现两次（按两条链长度归一化）
    - RMSD / Aligned length 为可选解析
    """
    # 兼容不同 TMscore/USalign 版本输出：
    # - 常见: "TM-score = ..."
    # - 少数: 仅输出一个 TM-score
    # - 变体: "TMscore = ..."
    tm_matches = re.findall(r"TM-?score\s*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if len(tm_matches) >= 2:
        tm1 = float(tm_matches[0])
        tm2 = float(tm_matches[1])
    elif len(tm_matches) == 1:
        tm1 = float(tm_matches[0])
        tm2 = tm1
    else:
        snippet = text[:400].replace("\n", "\\n")
        raise ValueError(
            "Unable to parse TM-score values from TMscore output. "
            f"Output snippet: {snippet}"
        )

    rmsd_match = re.search(r"RMSD of  the common residues=\s*([0-9]*\.?[0-9]+)", text)
    rmsd = float(rmsd_match.group(1)) if rmsd_match else None

    aligned_match = re.search(r"Aligned length=\s*([0-9]+)", text)
    aligned_len = int(aligned_match.group(1)) if aligned_match else None

    return tm1, tm2, rmsd, aligned_len


def resolve_label_from_pred(stem: str, strip_suffix: str) -> str:
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def read_selected_labels(selected_csv: Path) -> set[str]:
    labels: set[str] = set()
    with selected_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "protein_name" not in (reader.fieldnames or []):
            raise ValueError(f"CSV {selected_csv} must contain 'protein_name' column.")
        for row in reader:
            v = (row.get("protein_name") or "").strip()
            if v:
                labels.add(v)
    return labels


def iter_pred_pdbs(inputs: List[Path]) -> Iterable[Path]:
    files: List[Path] = []
    for inp in inputs:
        if inp.is_file() and inp.suffix.lower() == ".pdb":
            files.append(inp)
        elif inp.is_dir():
            files.extend(sorted(inp.rglob("*.pdb")))
        else:
            raise FileNotFoundError(f"Input path not found or unsupported: {inp}")

    # 去重并保持顺序
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        yield p


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    q1 = vals_sorted[int(0.25 * (len(vals_sorted) - 1))]
    q2 = vals_sorted[int(0.50 * (len(vals_sorted) - 1))]
    q3 = vals_sorted[int(0.75 * (len(vals_sorted) - 1))]
    return {
        "count": len(vals_sorted),
        "mean": statistics.fmean(vals_sorted),
        "std": statistics.pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0,
        "min": vals_sorted[0],
        "q1": q1,
        "median": q2,
        "q3": q3,
        "max": vals_sorted[-1],
    }


def pick_tm_score(tm1: float, tm2: float, mode: str) -> float:
    if mode == "chain1":
        return tm1
    if mode == "chain2":
        return tm2
    if mode == "max":
        return max(tm1, tm2)
    if mode == "min":
        return min(tm1, tm2)
    raise ValueError(f"Unknown tm_select mode: {mode}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch compute TM-score between predicted PDBs and native single-chain PDBs."
    )
    parser.add_argument(
        "pred",
        nargs="+",
        type=Path,
        help="Predicted PDB file(s) and/or directory(ies).",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        required=True,
        help="Directory containing native single-chain PDB files named as <label>.pdb.",
    )
    parser.add_argument(
        "--tm-exec",
        type=Path,
        required=True,
        help="Path to TMscore executable.",
    )
    parser.add_argument(
        "--strip-suffix",
        type=str,
        default=NAME_SUFFIX_TO_STRIP,
        help="Suffix stripped from prediction filename stem to derive label.",
    )
    parser.add_argument(
        "--native-ext",
        type=str,
        default=".pdb",
        help="Native structure extension (default: .pdb).",
    )
    parser.add_argument(
        "--tm-select",
        choices=("chain1", "chain2", "max", "min"),
        default="max",
        help="How to pick one TM-score from two normalized scores.",
    )
    parser.add_argument(
        "--select-threshold",
        type=float,
        default=0.7,
        help="Export label list where selected TM-score < threshold (default: 0.7).",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("docs/selectedPDB/low_tmscore.csv"),
        help="Output CSV path for samples below threshold.",
    )
    parser.add_argument(
        "--selected-labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with column 'protein_name' to evaluate only selected labels.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/selectedPDB/tmscore_results.csv"),
        help="Output full per-sample TM-score CSV.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=Path("docs/selectedPDB/tmscore_missing_native.txt"),
        help="Log file for missing native/failed cases.",
    )

    args = parser.parse_args(argv)

    if not args.tm_exec.exists():
        raise SystemExit(f"TMscore executable not found: {args.tm_exec}")
    if not args.native_dir.exists():
        raise SystemExit(f"Native dir not found: {args.native_dir}")

    eval_whitelist: Optional[set[str]] = None
    if args.selected_labels_csv is not None:
        eval_whitelist = read_selected_labels(args.selected_labels_csv)

    results: List[TMResult] = []
    issues: List[Tuple[str, str]] = []

    for pred_pdb in iter_pred_pdbs(args.pred):
        label = resolve_label_from_pred(pred_pdb.stem, args.strip_suffix)
        if eval_whitelist is not None and label not in eval_whitelist:
            continue

        native_pdb = args.native_dir / f"{label}{args.native_ext}"
        if not native_pdb.exists():
            issues.append((str(pred_pdb), f"missing native: {native_pdb}"))
            continue

        try:
            out = subprocess.check_output(
                [str(args.tm_exec), str(pred_pdb), str(native_pdb)],
                text=True,
                errors="replace",
            )
            tm1, tm2, rmsd, aligned_len = parse_tmscore_output(out)
            tm_selected = pick_tm_score(tm1, tm2, args.tm_select)
            results.append(
                TMResult(
                    label=label,
                    pred_pdb=pred_pdb,
                    native_pdb=native_pdb,
                    tm_norm_chain1=tm1,
                    tm_norm_chain2=tm2,
                    tm_selected=tm_selected,
                    rmsd=rmsd,
                    aligned_len=aligned_len,
                    status="ok",
                )
            )
        except Exception as e:
            issues.append((str(pred_pdb), f"TMscore failed: {e}"))

    # 写完整结果
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "pred_pdb",
                "native_pdb",
                "tm_norm_chain1",
                "tm_norm_chain2",
                "tm_selected",
                "rmsd",
                "aligned_len",
                "status",
            ]
        )
        for r in sorted(results, key=lambda x: x.tm_selected):
            writer.writerow(
                [
                    r.label,
                    str(r.pred_pdb),
                    str(r.native_pdb),
                    f"{r.tm_norm_chain1:.5f}",
                    f"{r.tm_norm_chain2:.5f}",
                    f"{r.tm_selected:.5f}",
                    "" if r.rmsd is None else f"{r.rmsd:.5f}",
                    "" if r.aligned_len is None else str(r.aligned_len),
                    r.status,
                ]
            )

    # 写低分集合
    low = [r for r in results if r.tm_selected < args.select_threshold]
    args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.selected_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_name", "tm_score"])
        for r in sorted(low, key=lambda x: x.tm_selected):
            writer.writerow([r.label, f"{r.tm_selected:.5f}"])

    # 写缺失/失败日志
    args.missing_log.parent.mkdir(parents=True, exist_ok=True)
    with args.missing_log.open("w", encoding="utf-8") as f:
        f.write("pred_pdb\treason\n")
        for p, reason in issues:
            f.write(f"{p}\t{reason}\n")

    # 控制台汇总
    vals = [r.tm_selected for r in results]
    s = summarize(vals)
    print(f"Evaluated: {len(results)}")
    print(f"Issues    : {len(issues)} (see {args.missing_log})")
    if s:
        print(
            "TM-score summary "
            f"(mode={args.tm_select}): "
            f"mean={s['mean']:.4f}, std={s['std']:.4f}, "
            f"min={s['min']:.4f}, q1={s['q1']:.4f}, median={s['median']:.4f}, "
            f"q3={s['q3']:.4f}, max={s['max']:.4f}"
        )
    print(f"Full results: {args.out_csv}")
    print(f"Low TM list (<{args.select_threshold:.2f}): {args.selected_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
