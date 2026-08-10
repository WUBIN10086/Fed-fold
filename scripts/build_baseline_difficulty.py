#!/usr/bin/env python3
"""Build client-local baseline difficulty tables from existing predictions.

Reuses baseline/prescreen PDBs when available. Does not overwrite source runs.
Writes per-client private CSVs plus an optional union CSV under RUN_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_fed_test_set import (  # noqa: E402
    build_label_to_cluster,
    difficulty_band,
    difficulty_weight,
    read_labels,
)


DIFFICULTY_FIELDS = [
    "label",
    "client",
    "split",
    "cluster_id",
    "release_date",
    "baseline_tm",
    "baseline_lddt_ca",
    "baseline_plddt",
    "sequence_length",
    "difficulty",
    "difficulty_weight",
    "pred_pdb",
    "native_pdb",
    "tm_status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metric_map(path: Path | None, label_key: str, value_key: str) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = (row.get(label_key) or row.get("label") or "").strip()
            if not label:
                continue
            raw = row.get(value_key)
            if raw is None or raw == "":
                continue
            out[label.upper()] = float(raw)
    return out


def find_pred_pdb(pred_dirs: list[Path], label: str) -> Path | None:
    patterns = [
        f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb",
        f"{label}_unrelaxed.pdb",
        f"{label}.pdb",
    ]
    for pred_dir in pred_dirs:
        if pred_dir is None or not pred_dir.exists():
            continue
        for name in patterns:
            path = pred_dir / name
            if path.exists():
                return path
        matches = sorted(pred_dir.glob(f"{label}*_unrelaxed.pdb"))
        if matches:
            return matches[0]
    return None


def load_release_and_length(cache_paths: list[Path]) -> tuple[dict[str, str], dict[str, int]]:
    release = {}
    length = {}
    for path in cache_paths:
        if path is None or not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in data.items():
            label = key.upper()
            if isinstance(entry, dict):
                if "release_date" in entry and label not in release:
                    release[label] = entry["release_date"]
                seq = entry.get("seq")
                if seq is None and isinstance(entry.get("seqs"), list) and entry["seqs"]:
                    seq = entry["seqs"][0]
                if seq and label not in length:
                    length[label] = len(seq)
    return release, length


def ensure_natives(
    labels: list[str],
    cif_dirs: list[Path],
    native_dir: Path,
    python_bin: str,
) -> None:
    missing = [lab for lab in labels if not (native_dir / f"{lab}.pdb").exists()]
    if not missing:
        return
    native_dir.mkdir(parents=True, exist_ok=True)
    labels_txt = native_dir / "_pending_labels.txt"
    labels_txt.write_text("".join(f"{lab}\n" for lab in missing), encoding="utf-8")
    for cif_dir in cif_dirs:
        if cif_dir is None or not cif_dir.exists():
            continue
        cmd = [
            python_bin,
            str(REPO / "scripts" / "extract_native_chain_pdbsV2.py"),
            "--cif-dir",
            str(cif_dir),
            "--out-dir",
            str(native_dir),
            "--labels-txt",
            str(labels_txt),
            "--report-csv",
            str(native_dir / "native_report.csv"),
            "--fail-log",
            str(native_dir / "native_failures.txt"),
        ]
        subprocess.run(cmd, check=False)
    still = [lab for lab in missing if not (native_dir / f"{lab}.pdb").exists()]
    if still:
        print(f"[warn] native still missing for {len(still)} labels")


def run_tmscore(
    pred_paths: dict[str, Path],
    native_dir: Path,
    out_csv: Path,
    tm_exec: Path,
    python_bin: str,
) -> dict[str, float]:
    if not pred_paths:
        return {}
    staging = out_csv.parent / "_pred_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for label, src in pred_paths.items():
        dst = staging / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except FileExistsError:
                pass
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(REPO / "scripts" / "tmscore_from_pdb.py"),
        str(staging),
        "--native-dir",
        str(native_dir),
        "--tm-exec",
        str(tm_exec),
        "--out-csv",
        str(out_csv),
        "--missing-log",
        str(out_csv.with_suffix(".missing.txt")),
    ]
    subprocess.run(cmd, check=False)
    return load_metric_map(out_csv, "label", "tm_selected")


def run_lddt(
    pred_dir: Path,
    native_dir: Path,
    out_csv: Path,
    python_bin: str,
) -> dict[str, float]:
    if not pred_dir.exists():
        return {}
    cmd = [
        python_bin,
        str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
        str(pred_dir),
        "--native-dir",
        str(native_dir),
        "--out-csv",
        str(out_csv),
        "--missing-log",
        str(out_csv.with_suffix(".missing.txt")),
    ]
    subprocess.run(cmd, check=False)
    return load_metric_map(out_csv, "label", "lddt_ca")


def run_plddt(pred_dir: Path, out_csv: Path, python_bin: str) -> dict[str, float]:
    if not pred_dir.exists():
        return {}
    cmd = [
        python_bin,
        str(REPO / "scripts" / "plddt_from_pdb.py"),
        str(pred_dir),
        "--no-extremes",
        "--select-threshold",
        "101",
        "--selected-csv",
        str(out_csv),
        "--bad-log",
        str(out_csv.with_suffix(".bad.txt")),
    ]
    subprocess.run(cmd, check=False)
    return load_metric_map(out_csv, "protein_name", "mean_plddt")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFFICULTY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in DIFFICULTY_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=REPO / "outputs" / "fed_lora_fp32",
    )
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--cluster-file", type=Path, required=True)
    parser.add_argument(
        "--tm-exec",
        type=Path,
        default=REPO / "tmscore" / "TMscore",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--reuse-existing-metrics",
        action="store_true",
        default=True,
        help="Reuse evaluation/baseline metrics for development_test labels",
    )
    parser.add_argument("--base-cutoff-date", default="2021-09-30")
    args = parser.parse_args()

    label2cluster = build_label_to_cluster(args.cluster_file)
    source = args.source_run
    out_root = args.run_root / "difficulty"
    out_root.mkdir(parents=True, exist_ok=True)

    existing_tm = load_metric_map(
        source / "evaluation" / "baseline" / "metrics" / "tm_score.csv",
        "label",
        "tm_selected",
    )
    existing_lddt = load_metric_map(
        source / "evaluation" / "baseline" / "metrics" / "lddt_ca.csv",
        "label",
        "lddt_ca",
    )
    existing_plddt = load_metric_map(
        source / "evaluation" / "baseline" / "metrics" / "plddt.csv",
        "protein_name",
        "mean_plddt",
    )

    all_rows: list[dict] = []
    failures: list[dict] = []
    summary = {"clients": {}}

    for idx in range(args.num_clients):
        client = f"client_{idx}"
        client_src = source / "clients" / client
        split_src = source / "split" / client
        private = args.run_root / "clients" / client / "private" / "difficulty"
        private.mkdir(parents=True, exist_ok=True)

        train_labels = read_labels(split_src / "train_labels.txt")
        test_labels = read_labels(split_src / "test_labels.txt")
        label_split = {lab.upper(): "train_pool" for lab in train_labels}
        label_split.update({lab.upper(): "development_test" for lab in test_labels})
        labels = sorted(set(train_labels) | set(test_labels), key=str.upper)

        cache_paths = [
            client_src / "mmcif_cache_finetune.json",
            client_src / "chain_data_cache_finetune.json",
            split_src / "train_chain_data_cache.json",
        ]
        release, length = load_release_and_length(cache_paths)

        pred_dirs = [
            source / "evaluation" / "baseline" / "predictions",
            client_src / "prescreen" / "predictions",
        ]
        cif_dirs = [
            client_src / "mmcif_files_finetune",
            client_src / "mmcif_files",
            source / "split" / "all" / "mmcif_files",
        ]
        native_dir = private / "native"
        ensure_natives(labels, cif_dirs, native_dir, args.python_bin)

        need_tm: dict[str, Path] = {}
        tm_map = dict(existing_tm)
        lddt_map = dict(existing_lddt)
        plddt_map = dict(existing_plddt)
        pred_map: dict[str, Path] = {}

        for lab in labels:
            pred = find_pred_pdb(pred_dirs, lab)
            if pred is not None:
                pred_map[lab.upper()] = pred
            if lab.upper() not in tm_map:
                if pred is None:
                    failures.append(
                        {
                            "label": lab,
                            "client": client,
                            "reason": "missing_pred",
                        }
                    )
                else:
                    need_tm[lab] = pred

        if need_tm:
            computed = run_tmscore(
                need_tm,
                native_dir,
                private / "computed_tm_score.csv",
                args.tm_exec,
                args.python_bin,
            )
            tm_map.update(computed)
            staging = private / "computed_tm_score.csv"
            # also compute lddt/plddt for newly scored preds if missing
            staging_pred = private / "_pred_staging"
            if staging_pred.exists():
                if any(lab.upper() not in lddt_map for lab in need_tm):
                    lddt_map.update(
                        run_lddt(
                            staging_pred,
                            native_dir,
                            private / "computed_lddt_ca.csv",
                            args.python_bin,
                        )
                    )
                if any(lab.upper() not in plddt_map for lab in need_tm):
                    plddt_map.update(
                        run_plddt(
                            staging_pred,
                            private / "computed_plddt.csv",
                            args.python_bin,
                        )
                    )

        rows = []
        band_counts = {"hard": 0, "medium": 0, "easy": 0, "unknown": 0}
        for lab in labels:
            key = lab.upper()
            tm = tm_map.get(key)
            status = "ok"
            if tm is None:
                native = native_dir / f"{lab}.pdb"
                if key not in pred_map:
                    status = "missing_pred"
                elif not native.exists():
                    status = "missing_native"
                else:
                    status = "tm_failed"
                failures.append(
                    {"label": lab, "client": client, "reason": status}
                )
            band = difficulty_band(tm)
            band_counts[band] += 1
            cid = label2cluster.get(key)
            cluster_id = f"c{cid}" if cid is not None else f"singleton:{key}"
            row = {
                "label": lab,
                "client": client,
                "split": label_split.get(key, "unknown"),
                "cluster_id": cluster_id,
                "release_date": release.get(key, ""),
                "baseline_tm": "" if tm is None else f"{tm:.6f}",
                "baseline_lddt_ca": (
                    "" if key not in lddt_map else f"{lddt_map[key]:.6f}"
                ),
                "baseline_plddt": (
                    "" if key not in plddt_map else f"{plddt_map[key]:.6f}"
                ),
                "sequence_length": length.get(key, ""),
                "difficulty": band,
                "difficulty_weight": (
                    "" if tm is None else f"{difficulty_weight(tm):.6f}"
                ),
                "pred_pdb": str(pred_map.get(key, "")),
                "native_pdb": str(native_dir / f"{lab}.pdb"),
                "tm_status": status,
            }
            rows.append(row)
            all_rows.append(row)

        out_csv = private / "baseline_difficulty.csv"
        write_csv(out_csv, rows)
        summary["clients"][client] = {
            "csv": str(out_csv),
            "sha256": sha256_file(out_csv),
            "label_count": len(rows),
            "difficulty": band_counts,
            "ok_count": sum(1 for r in rows if r["tm_status"] == "ok"),
        }
        print(f"[{client}] wrote {out_csv} ok={summary['clients'][client]['ok_count']}")

    union_csv = out_root / "baseline_difficulty.csv"
    write_csv(union_csv, all_rows)
    fail_csv = out_root / "failures.csv"
    with fail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "client", "reason"])
        writer.writeheader()
        for row in failures:
            writer.writerow(row)

    summary.update(
        {
            "union_csv": str(union_csv),
            "union_sha256": sha256_file(union_csv),
            "failures_csv": str(fail_csv),
            "hard_threshold": 0.5,
            "medium_threshold": 0.8,
            "base_cutoff_date": args.base_cutoff_date,
            "source_run": str(source.resolve()),
        }
    )
    summary_path = out_root / "difficulty_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {union_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
