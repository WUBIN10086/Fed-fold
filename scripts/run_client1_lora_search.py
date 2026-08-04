#!/usr/bin/env python3
"""Resumable staged LoRA search for the client1 OpenFold experiment."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.lora_search_metrics import (
    cluster_macro_metrics,
    scale_grid,
    select_model_with_baseline,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/all_pdb_1y/client1_res/lora_search_ema_fp32_v2"
MANIFEST = ROOT / "search_manifest.csv"
SUMMARY = ROOT / "comparison/validation_summary.csv"
PYTHON = Path("/home/test/miniconda3/envs/Fedfold/bin/python")
BASE_CHECKPOINT = (
    REPO
    / "openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
)
LORA_SCALES = scale_grid()
CLUSTER_FILE = REPO / "data/all_pdb_1y/clusters_30.txt"
TARGETS = {
    "ipa": "structure_module.ipa",
    "ipa_bb": "structure_module.ipa,structure_module.bb_update",
    "core": (
        "structure_module.ipa,structure_module.transition,"
        "structure_module.bb_update"
    ),
    "all": "structure_module",
}
FIELDS = [
    "candidate_id",
    "candidate_type",
    "stages",
    "target_id",
    "target",
    "rank",
    "alpha",
    "dropout",
    "max_epochs",
    "status",
    "error",
]


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "DS_IGNORE_CUDA_DETECTION": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO),
    })
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in command) + "\n")
        log.flush()
        result = subprocess.run(
            [str(x) for x in command],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def read_manifest():
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(MANIFEST)


def candidate_id(target_id, rank, scaling, dropout):
    return (
        f"{target_id}_r{rank}_s{scaling:g}_"
        f"d{round(dropout * 1000):03d}"
    )


def add_candidate(rows, stage, target_id, rank, scaling, dropout):
    cid = candidate_id(target_id, rank, scaling, dropout)
    existing = next((row for row in rows if row["candidate_id"] == cid), None)
    if existing is not None:
        stages = set(existing["stages"].split(";"))
        stages.add(stage)
        existing["stages"] = ";".join(sorted(stages))
        return existing
    row = {
        "candidate_id": cid,
        "candidate_type": "lora",
        "stages": stage,
        "target_id": target_id,
        "target": TARGETS[target_id],
        "rank": str(rank),
        "alpha": f"{rank * scaling:g}",
        "dropout": f"{dropout:g}",
        "max_epochs": "2",
        "status": "pending",
        "error": "",
    }
    rows.append(row)
    return row


def add_baseline_candidate(rows):
    existing = next(
        (row for row in rows if row["candidate_id"] == "baseline"),
        None,
    )
    if existing is not None:
        return existing
    row = {
        "candidate_id": "baseline",
        "candidate_type": "baseline",
        "stages": "baseline",
        "target_id": "none",
        "target": "",
        "rank": "0",
        "alpha": "0",
        "dropout": "0",
        "max_epochs": "0",
        "status": "pending",
        "error": "",
    }
    rows.insert(0, row)
    return row


def checkpoint_path(run_dir, epoch, epoch_len):
    return run_dir / "training/checkpoints" / f"{epoch - 1}-{epoch * epoch_len}.ckpt"


def load_label_to_cluster(path):
    mapping = {}
    with Path(path).open(encoding="utf-8") as handle:
        for cluster_id, line in enumerate(handle):
            for label in line.split():
                mapping[label.upper()] = str(cluster_id)
    return mapping


def load_baseline_tm():
    path = (
        ROOT
        / "baseline_validation/evaluations/baseline/metrics/tm_score.csv"
    )
    if not path.exists():
        raise RuntimeError(
            "Validation baseline must be evaluated before LoRA candidates"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["label"]: float(row["tm_selected"])
            for row in csv.DictReader(handle)
        }


def train_command(row, output_dir, labels, epoch_len, max_epochs, resume=None):
    output_dir = Path(output_dir)
    command = [
        PYTHON,
        REPO / "train_openfold.py",
        REPO / "data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune",
        REPO / "data/all_pdb_1y/fed_split/client_1/solo_alignment",
        REPO / "data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune",
        output_dir,
        "2026-01-01",
        "--train_filter_path",
        labels,
        "--use_single_seq_mode",
        "True",
        "--config_preset",
        "seq_model_esm1b_ptm",
        "--experiment_config_json",
        REPO / "seq_model_esm1b_ptm_finetune_override.json",
        "--resume_from_ckpt",
        resume
        if resume is not None
        else BASE_CHECKPOINT,
        "--template_release_dates_cache_path",
        REPO / "data/all_pdb_1y/fed_split/client_1/solo_mmcif_cache_finetune.json",
        "--train_chain_data_cache_path",
        REPO / "data/all_pdb_1y/fed_test_v2/client_1/train_chain_data_cache.json",
        "--lora_rank",
        row["rank"],
        "--lora_alpha",
        row["alpha"],
        "--lora_dropout",
        row["dropout"],
        "--lora_target",
        row["target"],
        "--learning_rate",
        "1e-4",
        "--lr_warmup_steps",
        "20",
        "--accumulate_grad_batches",
        "1",
        "--train_epoch_len",
        str(epoch_len),
        "--sampling_audit_path",
        output_dir / "sampling_audit.jsonl",
        "--sampling_cluster_file",
        CLUSTER_FILE,
        "--max_epochs",
        str(max_epochs),
        "--checkpoint_every_epoch",
        "--precision",
        "bf16-mixed",
        "--gpus",
        "1",
        "--seed",
        "42",
        "--deepspeed_config_path",
        REPO / "deepspeed_config.json",
    ]
    if resume is None:
        command.extend([
            "--resume_model_weights_only",
            "True",
            "--init_weights_source",
            "ema",
        ])
    return command


def ensure_training(row, run_dir, epoch_len, max_epochs, labels, resume=None):
    final_checkpoint = checkpoint_path(run_dir, max_epochs, epoch_len)
    if final_checkpoint.exists():
        return final_checkpoint
    run_logged(
        train_command(
            row,
            run_dir / "training",
            labels,
            epoch_len,
            max_epochs,
            resume=resume,
        ),
        run_dir / "training" / (
            "continue.log" if resume is not None else "train.log"
        ),
    )
    if not final_checkpoint.exists():
        raise FileNotFoundError(final_checkpoint)
    return final_checkpoint


def export_command(checkpoint, output, lora_scale):
    return [
        PYTHON,
        REPO / "scripts/export_lora_checkpoint.py",
        "--input",
        checkpoint,
        "--output",
        output,
        "--base-checkpoint",
        BASE_CHECKPOINT,
        "--base-weights-source",
        "ema",
        "--adapter-weights-source",
        "model",
        "--lora-scale",
        str(lora_scale),
        "--config-preset",
        "seq_model_esm1b_ptm",
    ]


def inference_command(
    fasta_dir,
    cif_dir,
    alignment_dir,
    predictions,
    checkpoint,
    checkpoint_source,
):
    return [
        PYTHON,
        REPO / "run_pretrained_openfold.py",
        fasta_dir,
        cif_dir,
        "--use_precomputed_alignments",
        alignment_dir,
        "--output_dir",
        predictions,
        "--model_device",
        "cuda:0",
        "--skip_relaxation",
        "--config_preset",
        "seq_model_esm1b_ptm",
        "--openfold_checkpoint_path",
        checkpoint,
        "--checkpoint_weights_source",
        checkpoint_source,
        "--data_random_seed",
        "42",
    ]


def ensure_evaluation(
    row,
    run_dir,
    checkpoint,
    epoch,
    fasta_dir,
    cif_dir,
    alignment_dir,
    native_dir,
    expected_count,
    lora_scale=1.0,
):
    is_baseline = row.get("candidate_type") == "baseline"
    evaluation_name = (
        "baseline"
        if is_baseline
        else f"epoch_{epoch:03d}/scale_{float(lora_scale):g}"
    )
    evaluation = run_dir / "evaluations" / evaluation_name
    merged = BASE_CHECKPOINT if is_baseline else evaluation / "merged_model.pt"
    predictions = evaluation / "predictions"
    metrics = evaluation / "metrics"
    if not is_baseline and not merged.exists():
        run_logged(
            export_command(checkpoint, merged, lora_scale),
            evaluation / "export.log",
        )
    pdbs = list((predictions / "predictions").glob("*_unrelaxed.pdb"))
    if len(pdbs) != expected_count:
        run_logged(
            inference_command(
                fasta_dir,
                cif_dir,
                alignment_dir,
                predictions,
                merged,
                "ema" if is_baseline else "auto",
            ),
            evaluation / "inference.log",
        )
    metrics.mkdir(parents=True, exist_ok=True)
    tm_csv = metrics / "tm_score.csv"
    plddt_csv = metrics / "plddt.csv"
    if not tm_csv.exists():
        run_logged(
            [
                PYTHON,
                REPO / "scripts/tmscore_from_pdb.py",
                predictions / "predictions",
                "--native-dir",
                native_dir,
                "--tm-exec",
                REPO / "tmscore/TMscore",
                "--out-csv",
                tm_csv,
                "--selected-csv",
                metrics / "low_tm.csv",
                "--missing-log",
                metrics / "tm_missing.txt",
            ],
            metrics / "tmscore.log",
        )
    if not plddt_csv.exists():
        run_logged(
            [
                PYTHON,
                REPO / "scripts/plddt_from_pdb.py",
                predictions / "predictions",
                "--no-extremes",
                "--select-threshold",
                "101",
                "--selected-csv",
                plddt_csv,
                "--bad-log",
                metrics / "plddt_bad.txt",
            ],
            metrics / "plddt.log",
        )
    with tm_csv.open(newline="", encoding="utf-8") as handle:
        tm_rows = list(csv.DictReader(handle))
    with plddt_csv.open(newline="", encoding="utf-8") as handle:
        plddt_rows = list(csv.DictReader(handle))
    if len(tm_rows) != expected_count or len(plddt_rows) != expected_count:
        raise ValueError(
            f"{row['candidate_id']} epoch {epoch}: expected {expected_count} "
            f"metrics, got TM={len(tm_rows)}, pLDDT={len(plddt_rows)}"
        )
    mean_tm = sum(float(item["tm_selected"]) for item in tm_rows) / expected_count
    mean_plddt = (
        sum(float(item["mean_plddt"]) for item in plddt_rows) / expected_count
    )
    tm_by_label = {
        item["label"]: float(item["tm_selected"])
        for item in tm_rows
    }
    cluster_metrics = cluster_macro_metrics(
        tm_by_label,
        load_label_to_cluster(CLUSTER_FILE),
        None if is_baseline else load_baseline_tm(),
    )
    result = {
        "candidate_id": (
            "baseline"
            if is_baseline
            else f"{row['candidate_id']}@scale={float(lora_scale):g}"
        ),
        "base_candidate_id": row["candidate_id"],
        "candidate_type": row.get("candidate_type", "lora"),
        "lora_scale": float(lora_scale),
        "stages": row["stages"],
        "target_id": row["target_id"],
        "target": row["target"],
        "rank": int(row["rank"]),
        "alpha": float(row["alpha"]),
        "dropout": float(row["dropout"]),
        "epoch": epoch,
        "global_step": epoch * (64 if run_dir.parent.name == "final_refit" else 52),
        "n_validation": expected_count,
        "mean_tm": mean_tm,
        "mean_plddt": mean_plddt,
        **cluster_metrics,
    }
    (evaluation / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def validation_paths():
    data = ROOT / "data/validation"
    return (
        data / "solo_fasta_dir",
        data / "mmcif_files",
        data / "solo_alignment_dir",
        data / "native",
    )


def collect_summary():
    results = []
    metric_paths = list(ROOT.glob(
        "search_runs/*/evaluations/epoch_*/scale_*/metrics.json"
    ))
    metric_paths.extend(ROOT.glob(
        "baseline_validation/evaluations/baseline/metrics.json"
    ))
    for metrics_json in metric_paths:
        results.append(json.loads(metrics_json.read_text()))
    results.sort(
        key=lambda row: (row["candidate_id"], row["epoch"])
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(dict.fromkeys(
            key for result in results for key in result
        ))
        with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in result.items()
                }
                for result in results
            ])
    return results


def best_results(stage, epoch=2, limit=None):
    rows_by_id = {row["candidate_id"]: row for row in read_manifest()}
    results = [
        result
        for result in collect_summary()
        if result["epoch"] == epoch
        and stage in rows_by_id[
            result.get("base_candidate_id", result["candidate_id"])
        ]["stages"].split(";")
    ]
    results.sort(
        key=lambda row: (
            -row["cluster_macro_tm"],
            -row["mean_tm"],
            row["candidate_id"],
        )
    )
    return results if limit is None else results[:limit]


def best_unique_results(stage, epoch=2, limit=None):
    unique = []
    seen = set()
    for result in best_results(stage, epoch=epoch):
        base_id = result.get("base_candidate_id", result["candidate_id"])
        if base_id in seen:
            continue
        seen.add(base_id)
        unique.append(result)
    return unique if limit is None else unique[:limit]


def init_a():
    rows = read_manifest()
    add_baseline_candidate(rows)
    for target_id in TARGETS:
        add_candidate(rows, "A", target_id, 4, 2, 0.05)
    write_manifest(rows)


def expand_b():
    rows = read_manifest()
    top_targets = [
        item["target_id"] for item in best_unique_results("A", limit=2)
    ]
    if len(top_targets) != 2:
        raise RuntimeError("Stage A must be complete before expanding stage B")
    for target_id in top_targets:
        for rank in (2, 4, 8):
            for scaling in (1, 2):
                add_candidate(rows, "B", target_id, rank, scaling, 0.05)
    write_manifest(rows)
    (ROOT / "comparison/stage_b_targets.json").write_text(
        json.dumps(top_targets, indent=2) + "\n"
    )


def expand_c():
    rows = read_manifest()
    top = best_unique_results("B", limit=3)
    if len(top) != 3:
        raise RuntimeError("Stage B must be complete before expanding stage C")
    for result in top:
        rank = int(result["rank"])
        scaling = float(result["alpha"]) / rank
        for dropout in (0, 0.05, 0.1, 0.2):
            add_candidate(
                rows,
                "C",
                result["target_id"],
                rank,
                scaling,
                dropout,
            )
    write_manifest(rows)
    (ROOT / "comparison/stage_c_seeds.json").write_text(
        json.dumps([item["candidate_id"] for item in top], indent=2) + "\n"
    )


def run_pending():
    rows = read_manifest()
    labels = ROOT / "data/search_train_labels.txt"
    fasta, cif, alignments, native = validation_paths()
    baseline_row = next(
        (row for row in rows if row.get("candidate_type") == "baseline"),
        None,
    )
    if baseline_row is None:
        baseline_row = add_baseline_candidate(rows)
    ensure_evaluation(
        baseline_row,
        ROOT / "baseline_validation",
        checkpoint=None,
        epoch=0,
        fasta_dir=fasta,
        cif_dir=cif,
        alignment_dir=alignments,
        native_dir=native,
        expected_count=12,
        lora_scale=0,
    )
    baseline_row["status"] = "completed"
    write_manifest(rows)
    had_error = False
    for row in rows:
        if row.get("candidate_type") == "baseline":
            continue
        if row["status"] == "completed":
            continue
        run_dir = ROOT / "search_runs" / row["candidate_id"]
        try:
            checkpoint = ensure_training(
                row,
                run_dir,
                epoch_len=52,
                max_epochs=2,
                labels=labels,
            )
            for lora_scale in LORA_SCALES:
                ensure_evaluation(
                    row,
                    run_dir,
                    checkpoint,
                    epoch=2,
                    fasta_dir=fasta,
                    cif_dir=cif,
                    alignment_dir=alignments,
                    native_dir=native,
                    expected_count=12,
                    lora_scale=lora_scale,
                )
            row["status"] = "completed"
            row["error"] = ""
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            had_error = True
        write_manifest(rows)
        collect_summary()
    if had_error:
        raise RuntimeError("One or more candidates failed; inspect manifest")


def run_finalists():
    rows = read_manifest()
    by_id = {row["candidate_id"]: row for row in rows}
    top = best_unique_results("C", limit=3)
    if len(top) != 3:
        raise RuntimeError("Stage C must be complete before finalists")
    fasta, cif, alignments, native = validation_paths()
    for result in top:
        row = by_id[result["base_candidate_id"]]
        stages = set(row["stages"].split(";"))
        stages.add("F")
        row["stages"] = ";".join(sorted(stages))
        run_dir = ROOT / "search_runs" / row["candidate_id"]
        resume = checkpoint_path(run_dir, 2, 52)
        ensure_training(
            row,
            run_dir,
            epoch_len=52,
            max_epochs=5,
            labels=ROOT / "data/search_train_labels.txt",
            resume=resume,
        )
        for epoch in (3, 4, 5):
            for lora_scale in LORA_SCALES:
                ensure_evaluation(
                    row,
                    run_dir,
                    checkpoint_path(run_dir, epoch, 52),
                    epoch,
                    fasta,
                    cif,
                    alignments,
                    native,
                    12,
                    lora_scale=lora_scale,
                )
        write_manifest(rows)
        collect_summary()
    (ROOT / "comparison/finalist_ids.json").write_text(
        json.dumps([item["candidate_id"] for item in top], indent=2) + "\n"
    )


def run_final_refit(min_delta=0.0):
    rows = read_manifest()
    by_id = {row["candidate_id"]: row for row in rows}
    finalists = {
        row["candidate_id"]
        for row in rows
        if "F" in row["stages"].split(";")
    }
    results = [
        result
        for result in collect_summary()
        if result.get("base_candidate_id") in finalists
    ]
    baseline = next(
        result
        for result in collect_summary()
        if result.get("candidate_type") == "baseline"
    )
    best = select_model_with_baseline(
        results,
        baseline,
        min_delta=min_delta,
    )
    selection_path = ROOT / "comparison/selected_config.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    if best["selected_model"] == "baseline":
        selection_path.write_text(
            json.dumps(
                {
                    "selected_model": "baseline",
                    "min_delta": min_delta,
                    "validation": best,
                    "final_refit_skipped": True,
                },
                indent=2,
            )
            + "\n"
        )
        return

    row = by_id[best["base_candidate_id"]]
    run_dir = ROOT / "final_refit" / best["candidate_id"]
    epoch = int(best["epoch"])
    checkpoint = ensure_training(
        row,
        run_dir,
        epoch_len=64,
        max_epochs=epoch,
        labels=REPO / "data/all_pdb_1y/fed_test_v2/client_1/train_labels.txt",
    )
    result = ensure_evaluation(
        row,
        run_dir,
        checkpoint,
        epoch,
        REPO / "data/all_pdb_1y/fed_test/all/solo_fasta_dir",
        REPO / "data/all_pdb_1y/fed_test/all/mmcif_files",
        REPO / "data/all_pdb_1y/fed_test/all/solo_alignment_dir",
        REPO / "data/all_pdb_1y/fed_test/all/native",
        20,
        lora_scale=float(best["lora_scale"]),
    )
    result["n_test"] = result.pop("n_validation")
    result["selected_validation_tm"] = best["mean_tm"]
    result["selected_validation_cluster_macro_tm"] = best["cluster_macro_tm"]
    result["selected_validation_plddt"] = best["mean_plddt"]
    output = ROOT / "comparison/final_test_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    selection_path.write_text(
        json.dumps(
            {
                "selected_model": best["candidate_id"],
                "min_delta": min_delta,
                "config": row,
                "validation": best,
                "test": result,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "init-a",
            "run-pending",
            "expand-b",
            "expand-c",
            "run-finalists",
            "run-final-refit",
            "summarize",
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation cluster-macro TM gain over baseline.",
    )
    args = parser.parse_args()
    action = {
        "init-a": init_a,
        "run-pending": run_pending,
        "expand-b": expand_b,
        "expand-c": expand_c,
        "run-finalists": run_finalists,
        "run-final-refit": lambda: run_final_refit(args.min_delta),
        "summarize": collect_summary,
    }[args.mode]
    action()


if __name__ == "__main__":
    main()
