#!/usr/bin/env python3
"""Run the fixed-schema client0 T4 LoRA capacity test with unclamped FAPE."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_HINT = Path(__file__).resolve().parents[1]
if str(REPO_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_HINT))

from scripts.build_fed_test_set import labels_sha256
from scripts.evaluate_hardcase_metrics import summarize_group
from scripts.evaluate_target_gate import (
    capacity_pass_from_rows,
    capacity_pass_v2_from_rows,
    promotion_gate_v2_from_rows,
    seed_confirmation_pass,
)
from scripts.lora_target_registry import TARGET_SPECS
from scripts.run_client0_target_ablation import (
    DEFAULT_EVAL,
    DEFAULT_RUN,
    DEFAULT_SOURCE,
    OVERFIT_EPOCH_LEN,
    OVERFIT_MAX_EPOCHS,
    OVERFIT_WARMUP_STEPS,
    PYTHON,
    REPO,
    ablation_root,
    checkpoint_for_epochs,
    ensure_lora_export,
    evaluate_full_validation,
    full_train_root,
    frozen_split_identity,
    get_target,
    prepare_run_manifest,
    private_root,
    read_labels,
    run_cmd,
    sha256_file,
    sha256_path,
    update_status,
    verify_one,
    write_json,
)


EXPERIMENT_SLUG = "T4_UNCLAMPED"
SELECTIVE_EXPERIMENT_SLUG = "T4_SELECTIVE_CLAMP"
ANCHOR_EXPERIMENT_SLUG = "T4_ANCHOR_BALANCED"
ANCHOR_RATIOS = "0.4,0.35,0.25"
PRESERVATION_EXPERIMENT_SLUG = "T4_BASELINE_PRESERVE"
PRESERVATION_WEIGHT = 1.0
STRONG_PRESERVATION_EXPERIMENT_SLUG = "T4_BASELINE_PRESERVE_W1P5"
STRONG_PRESERVATION_WEIGHT = 1.5
HARD_CORRECTION_EXPERIMENT_SLUG = (
    "T4_BASELINE_PRESERVE_W1P5_HARD_CORRECTION"
)
HARD_CORRECTION_WEIGHT = 1.0
HARD_CORRECTION_MARGIN = 0.5
HARD_CORRECTION_MIN_BASELINE_ERROR = 2.0
HARD_CORRECTION_MIN_SEQUENCE_SEPARATION = 12
SOFT_TM_EXPERIMENT_SLUG = "T4_BASELINE_PRESERVE_W1P5_SOFT_TM"
SOFT_TM_WEIGHT = 5.0
SOFT_TM_MARGIN = 0.02
SOFT_TM_SHRINK_SCALE = 0.85
TARGET_SLUG = "T4"
CHECKPOINT_EPOCHS = (5, 10, 15, 20, 25)
VALIDATION_EPOCHS = 5
SHRINKAGE_SCALE = 0.5
CONFIG_PATH = REPO / "seq_model_esm1b_ptm_hardcase_unclamped_override.json"


def experiment_root(run_root: Path, seed: int = 42) -> Path:
    return (
        ablation_root(run_root)
        / EXPERIMENT_SLUG
        / "overfit8"
        / f"seed_{seed}"
    )


def train(
    run_root: Path,
    source_run: Path,
    gpu_id: str,
    seed: int = 42,
) -> Path:
    spec = get_target(TARGET_SLUG)
    overfit = ablation_root(run_root) / "overfit8"
    out = experiment_root(run_root, seed)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = overfit / "overfit8_labels.txt"
    cluster_file = REPO / "data" / "all_pdb_1y" / "clusters_30.txt"
    final_ckpt = checkpoint_for_epochs(
        train_dir, OVERFIT_MAX_EPOCHS, OVERFIT_EPOCH_LEN
    )
    for required in (parent, labels, CONFIG_PATH, cluster_file):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = {
        "client_id": "client_0",
        "mode": "t4_unclamped_overfit8",
        "experiment_slug": EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "seed": seed,
        "learning_rate": 1e-4,
        "lr_warmup_steps": OVERFIT_WARMUP_STEPS,
        "max_epochs": OVERFIT_MAX_EPOCHS,
        "train_epoch_len": OVERFIT_EPOCH_LEN,
        "checkpoint_every_n_train_steps": 40,
        "sampling_mode": "uniform",
        "sampling_audit": True,
        "csv_loss_metrics": True,
        "fape_clamp_prob": 0.0,
        "experiment_config_path": str(CONFIG_PATH),
        "experiment_config_sha256": sha256_file(CONFIG_PATH),
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "labels_sha256": sha256_file(labels),
        "precision": "32",
        "aggregation": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    verify_one(TARGET_SLUG)
    prepare_run_manifest(
        out / "local_run.json",
        manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context=f"{EXPERIMENT_SLUG} overfit seed={seed}",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] manifest verified: {final_ckpt}")
        return final_ckpt

    data_src = source_run / "clients" / "client_0"
    cmd = [
        PYTHON,
        str(REPO / "train_openfold.py"),
        str(data_src / "mmcif_files_finetune"),
        str(data_src / "solo_alignment"),
        str(data_src / "mmcif_files_finetune"),
        str(train_dir),
        "2026-01-01",
        "--train_filter_path", str(labels),
        "--val_data_dir", str(overfit / "assets" / "mmcif_files"),
        "--val_alignment_dir", str(overfit / "assets" / "solo_alignment_dir"),
        "--use_single_seq_mode", "True",
        "--config_preset", "seq_model_esm1b_ptm",
        "--experiment_config_json", str(CONFIG_PATH),
        "--resume_from_ckpt", str(parent),
        "--resume_model_weights_only", "True",
        "--init_weights_source", "auto",
        "--template_release_dates_cache_path",
        str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path",
        str(overfit / "overfit8_chain_data_cache.json"),
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0",
        "--lora_target", spec.target,
        "--learning_rate", "1e-4",
        "--lr_warmup_steps", str(OVERFIT_WARMUP_STEPS),
        "--accumulate_grad_batches", "1",
        "--train_epoch_len", str(OVERFIT_EPOCH_LEN),
        "--max_epochs", str(OVERFIT_MAX_EPOCHS),
        "--checkpoint_every_n_train_steps", "40",
        "--sampling_mode", "uniform",
        "--sampling_audit_path", str(out / "sampling_audit.jsonl"),
        "--sampling_cluster_file", str(cluster_file),
        "--csv_log_metrics",
        "--log_every_n_steps", "1",
        "--precision", "32",
        "--gpus", "1",
        "--seed", str(seed),
        "--deepspeed_config_path", str(REPO / "deepspeed_config_fp32.json"),
    ]
    with (out / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "PYTHONPATH": str(REPO),
            },
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return checkpoint_for_epochs(
        train_dir, OVERFIT_MAX_EPOCHS, OVERFIT_EPOCH_LEN
    )


def evaluate_checkpoint(
    run_root: Path,
    epochs: int,
    gpu_id: str,
    seed: int = 42,
) -> dict:
    spec = get_target(TARGET_SLUG)
    private = private_root(run_root)
    overfit = ablation_root(run_root) / "overfit8"
    out = experiment_root(run_root, seed)
    checkpoint = checkpoint_for_epochs(
        out / "training", epochs, OVERFIT_EPOCH_LEN
    )
    if not (checkpoint.exists() or checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)
    step = epochs * OVERFIT_EPOCH_LEN
    cand = out / "eval" / f"step_{step}" / "scale_1p0"
    cand.mkdir(parents=True, exist_ok=True)
    model = cand / "model_scale_1.0.pt"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = overfit / "overfit8_labels.txt"
    overfit_manifest = __import__("json").loads(
        (overfit / "overfit8_manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = {
        "experiment_slug": EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "target": spec.target,
        "rank": 4,
        "alpha": 8.0,
        "scale": 1.0,
        "seed": seed,
        "step": step,
        "fape_clamp_prob": 0.0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "parent_sha256": sha256_file(parent),
        "labels_sha256": sha256_file(labels),
        "experiment_config_sha256": sha256_file(CONFIG_PATH),
    }
    prepare_run_manifest(
        cand / "evaluation_run.json",
        evaluation_manifest,
        artifact_exists=model.exists() or (cand / "predictions").exists(),
        context=f"{EXPERIMENT_SLUG} evaluation step={step}",
    )
    ensure_lora_export(
        checkpoint=checkpoint,
        model=model,
        parent=parent,
        target=spec.target,
        scale=1.0,
    )

    assets = overfit / "assets"
    pred_dir = cand / "predictions"
    expected = len(read_labels(labels))
    existing = list(pred_dir.glob("*_unrelaxed.pdb")) if pred_dir.exists() else []
    if len(existing) != expected:
        if existing:
            raise RuntimeError(
                f"Partial predictions at {cand}: {len(existing)}/{expected}"
            )
        run_cmd(
            [
                PYTHON,
                str(REPO / "run_pretrained_openfold.py"),
                str(assets / "solo_fasta_dir"),
                str(assets / "mmcif_files"),
                "--use_precomputed_alignments",
                str(assets / "solo_alignment_dir"),
                "--use_single_seq_mode",
                "--output_dir", str(cand),
                "--model_device", "cuda:0",
                "--skip_relaxation",
                "--config_preset", "seq_model_esm1b_ptm",
                "--openfold_checkpoint_path", str(model),
                "--checkpoint_weights_source", "auto",
                "--data_random_seed", str(seed),
                "--precision", "fp32",
            ],
            env={"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)},
        )

    metrics = cand / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    run_cmd(
        [
            PYTHON,
            str(REPO / "scripts" / "tmscore_from_pdb.py"),
            str(pred_dir),
            "--native-dir", str(native),
            "--tm-exec", str(REPO / "tmscore" / "TMscore"),
            "--out-csv", str(metrics / "tm_score.csv"),
            "--selected-csv", str(metrics / "low_tm.csv"),
            "--missing-log", str(metrics / "tm_missing.txt"),
        ]
    )
    run_cmd(
        [
            PYTHON,
            str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
            str(pred_dir),
            "--native-dir", str(native),
            "--out-csv", str(metrics / "lddt_ca.csv"),
            "--missing-log", str(metrics / "lddt_missing.txt"),
        ]
    )
    paired = cand / "paired_deltas.csv"
    run_cmd(
        [
            PYTHON,
            str(REPO / "scripts" / "evaluate_hardcase_metrics.py"),
            "--mode", "pair",
            "--labels", str(labels),
            "--difficulty-csv", str(difficulty),
            "--baseline-tm-csv", str(difficulty),
            "--model-tm-csv", str(metrics / "tm_score.csv"),
            "--baseline-lddt-csv", str(difficulty),
            "--model-lddt-csv", str(metrics / "lddt_ca.csv"),
            "--client-id", "client_0",
            "--out", str(paired),
        ]
    )
    rows = list(csv.DictReader(paired.open(encoding="utf-8")))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    v1 = capacity_pass_from_rows(rows)
    v2 = capacity_pass_v2_from_rows(
        rows,
        expected_labels=overfit_manifest.get("labels"),
        expected_clusters=overfit_manifest.get("clusters"),
    )
    write_json(cand / "capacity.json", {"step": step, **v1})
    write_json(cand / "capacity_v2.json", {"step": step, **v2})
    return {
        "step": step,
        "paired_csv": str(paired),
        "capacity_v1": v1,
        "capacity_v2": v2,
    }


def run(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    train(run_root, source_run, gpu_id, seed=seed)
    rows = []
    for epochs in CHECKPOINT_EPOCHS:
        result = evaluate_checkpoint(
            run_root, epochs, gpu_id, seed=seed
        )
        paired_rows = list(
            csv.DictReader(Path(result["paired_csv"]).open(encoding="utf-8"))
        )
        by_label = {
            row["label"]: float(row["delta_tm"])
            for row in paired_rows
        }
        v2 = result["capacity_v2"]
        row = {
            "step": result["step"],
            "mean_delta_tm": v2.get("mean_delta_tm"),
            "median_delta_tm": v2.get("median_delta_tm"),
            "leave_max_out_mean_delta_tm": v2.get(
                "leave_max_out_mean_delta_tm"
            ),
            "delta_tm_ge_0p01_count": v2.get("delta_tm_ge_0p01_count"),
            "tm_rise_count": v2.get("tm_rise_count"),
            "mean_delta_lddt": v2.get("mean_delta_lddt"),
            "max_point_contribution": v2.get("max_point_contribution"),
            "capacity_v2_pass": bool(v2.get("pass")),
        }
        for label in sorted(by_label):
            row[f"delta_tm_{label}"] = by_label[label]
        rows.append(row)

    eval_root.mkdir(parents=True, exist_ok=True)
    csv_path = eval_root / "t4_unclamped_capacity_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    any_pass = any(row["capacity_v2_pass"] for row in rows)
    payload = {
        "stage": "run-unclamped-t4-capacity",
        "status": "complete",
        "experiment_slug": EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "fape_clamp_prob": 0.0,
        "capacity_v2_any_pass": any_pass,
        "rows": rows,
        "csv": str(csv_path),
        "test_accessed": False,
    }
    write_json(eval_root / "t4_unclamped_capacity_summary.json", payload)
    write_json(
        ablation_root(run_root) / "t4_unclamped_capacity_summary.json",
        payload,
    )
    update_status(
        run_root,
        run_unclamped_t4_capacity="complete",
        t4_unclamped_capacity_v2_any_pass=any_pass,
    )
    return payload



def train_frozen_validation(
    run_root: Path,
    source_run: Path,
    gpu_id: str,
    capacity_step: int,
    seed: int = 42,
    epochs: int = VALIDATION_EPOCHS,
    experiment_slug: str = EXPERIMENT_SLUG,
    difficulty_conditioned_fape: bool = False,
    sampling_mode: str = "uniform",
    hard_aware_ratios: str | None = None,
    baseline_preservation_weight: float = 0.0,
    hard_correction_weight: float = 0.0,
    hard_correction_margin: float = HARD_CORRECTION_MARGIN,
    hard_correction_min_baseline_error: float = (
        HARD_CORRECTION_MIN_BASELINE_ERROR
    ),
    hard_correction_min_sequence_separation: int = (
        HARD_CORRECTION_MIN_SEQUENCE_SEPARATION
    ),
    hard_soft_tm_weight: float = 0.0,
    hard_soft_tm_margin: float = SOFT_TM_MARGIN,
) -> Path:
    """Train a locked T4 configuration on client0 train only."""
    spec = get_target(TARGET_SLUG)
    private = private_root(run_root)
    split_root = private / "splits"
    out = full_train_root(run_root, experiment_slug, seed)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = split_root / "train_labels.txt"
    epoch_len = len(read_labels(labels))
    final_ckpt = checkpoint_for_epochs(train_dir, epochs, epoch_len)
    data_src = source_run / "clients" / "client_0"
    baseline_prediction_dir = data_src / "prescreen" / "predictions"
    cluster_file = REPO / "data" / "all_pdb_1y" / "clusters_30.txt"

    if epoch_len != 57:
        raise AssertionError(f"Unexpected client0 train size: {epoch_len}")
    for required in (
        parent,
        labels,
        CONFIG_PATH,
        cluster_file,
        split_root / "split_manifest.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = {
        "client_id": "client_0",
        "mode": (
            "t4_selective_clamp_frozen_validation"
            if difficulty_conditioned_fape
            else "t4_unclamped_frozen_validation"
        ),
        "experiment_slug": experiment_slug,
        "target_slug": TARGET_SLUG,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "seed": seed,
        "learning_rate": 1e-4,
        "lr_warmup_steps": 20,
        "sampling_mode": sampling_mode,
        "local_epochs": epochs,
        "train_epoch_len": epoch_len,
        "primary_eval_epoch": epochs,
        "primary_eval_scale": 1.0,
        "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
        "capacity_configuration_locked_at_step": capacity_step,
        "fape_clamp_prob": 0.0,
        "experiment_config_path": str(CONFIG_PATH),
        "experiment_config_sha256": sha256_file(CONFIG_PATH),
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "split_train_sha256": sha256_file(labels),
        "precision": "32",
        "aggregation": False,
        "baseline_preservation_weight": baseline_preservation_weight,
        "hard_correction_weight": hard_correction_weight,
        "hard_correction_margin": hard_correction_margin,
        "hard_correction_min_baseline_error": (
            hard_correction_min_baseline_error
        ),
        "hard_correction_min_sequence_separation": (
            hard_correction_min_sequence_separation
        ),
        "hard_soft_tm_weight": hard_soft_tm_weight,
        "hard_soft_tm_margin": hard_soft_tm_margin,
        "test_accessed": False,
    }
    if difficulty_conditioned_fape:
        manifest.update({
            "difficulty_conditioned_fape": True,
            "difficulty_clamp_policy": {
                "hard": 0.0,
                "medium": 1.0,
                "easy": 1.0,
            },
        })
    if hard_aware_ratios is not None:
        manifest["hard_aware_ratios"] = hard_aware_ratios
    if (
        baseline_preservation_weight > 0.0
        or hard_correction_weight > 0.0
        or hard_soft_tm_weight > 0.0
    ):
        manifest["baseline_preservation_dir"] = str(
            baseline_prediction_dir
        )
    if baseline_preservation_weight > 0.0:
        manifest.update({
            "baseline_preservation_scope": ["medium", "easy"],
            "baseline_preservation_geometry": "all_pair_ca_distances_huber",
        })
    if hard_correction_weight > 0.0:
        manifest.update({
            "hard_correction_scope": ["hard"],
            "hard_correction_geometry": (
                "baseline_relative_long_range_ca_distance_margin"
            ),
        })
    if hard_soft_tm_weight > 0.0:
        manifest.update({
            "hard_soft_tm_scope": ["hard"],
            "hard_soft_tm_geometry": "kabsch_aligned_soft_tm_margin",
        })
    out.mkdir(parents=True, exist_ok=True)
    verify_one(TARGET_SLUG)
    prepare_run_manifest(
        out / "local_run.json",
        manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context=f"{experiment_slug} frozen validation seed={seed}",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] frozen-validation manifest verified: {final_ckpt}")
        return final_ckpt

    cmd = [
        PYTHON,
        str(REPO / "train_openfold.py"),
        str(data_src / "mmcif_files_finetune"),
        str(data_src / "solo_alignment"),
        str(data_src / "mmcif_files_finetune"),
        str(train_dir),
        "2026-01-01",
        "--train_filter_path", str(labels),
        "--val_data_dir", str(split_root / "assets_validation" / "mmcif_files"),
        "--val_alignment_dir",
        str(split_root / "assets_validation" / "solo_alignment_dir"),
        "--use_single_seq_mode", "True",
        "--config_preset", "seq_model_esm1b_ptm",
        "--experiment_config_json", str(CONFIG_PATH),
        "--resume_from_ckpt", str(parent),
        "--resume_model_weights_only", "True",
        "--init_weights_source", "auto",
        "--template_release_dates_cache_path",
        str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path",
        str(split_root / "train_chain_data_cache.json"),
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0",
        "--lora_target", spec.target,
        "--learning_rate", "1e-4",
        "--lr_warmup_steps", "20",
        "--accumulate_grad_batches", "1",
        "--train_epoch_len", str(epoch_len),
        "--max_epochs", str(epochs),
        "--checkpoint_every_epoch",
        "--sampling_mode", sampling_mode,
        "--difficulty_csv",
        str(private / "difficulty" / "baseline_difficulty.csv"),
        "--sampling_audit_path", str(out / "sampling_audit.jsonl"),
        "--sampling_cluster_file", str(cluster_file),
        "--csv_log_metrics",
        "--log_every_n_steps", "1",
        "--precision", "32",
        "--gpus", "1",
        "--seed", str(seed),
        "--deepspeed_config_path", str(REPO / "deepspeed_config_fp32.json"),
    ]
    if difficulty_conditioned_fape:
        cmd.append("--difficulty_conditioned_fape")
    if hard_aware_ratios is not None:
        cmd.extend(["--hard_aware_ratios", hard_aware_ratios])
    if (
        baseline_preservation_weight > 0.0
        or hard_correction_weight > 0.0
        or hard_soft_tm_weight > 0.0
    ):
        cmd.extend([
            "--baseline_preservation_dir", str(baseline_prediction_dir),
        ])
    if baseline_preservation_weight > 0.0:
        cmd.extend([
            "--baseline_preservation_weight",
            str(baseline_preservation_weight),
        ])
    if hard_correction_weight > 0.0:
        cmd.extend([
            "--hard_correction_weight", str(hard_correction_weight),
            "--hard_correction_margin", str(hard_correction_margin),
            "--hard_correction_min_baseline_error",
            str(hard_correction_min_baseline_error),
            "--hard_correction_min_sequence_separation",
            str(hard_correction_min_sequence_separation),
        ])
    if hard_soft_tm_weight > 0.0:
        cmd.extend([
            "--hard_soft_tm_weight", str(hard_soft_tm_weight),
            "--hard_soft_tm_margin", str(hard_soft_tm_margin),
        ])
    with (out / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "PYTHONPATH": str(REPO),
            },
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    final_ckpt = checkpoint_for_epochs(train_dir, epochs, epoch_len)
    if not (final_ckpt.exists() or final_ckpt.is_dir()):
        raise FileNotFoundError(final_ckpt)
    return final_ckpt


def run_frozen_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Run exactly one pre-registered validation evaluation."""
    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    if not capacity_path.exists():
        raise FileNotFoundError(
            f"Capacity summary missing; run capacity first: {capacity_path}"
        )
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    split_root = private_root(run_root) / "splits"
    train_n = len(read_labels(split_root / "train_labels.txt"))
    validation_n = len(read_labels(split_root / "validation_labels.txt"))
    if (train_n, validation_n) != (57, 10):
        raise AssertionError(
            f"Unexpected frozen split sizes: train={train_n}, "
            f"validation={validation_n}"
        )

    # Process-local alias lets the existing strict evaluator use a unique output
    # namespace while preserving the canonical T4 adapter target and schema.
    TARGET_SPECS[EXPERIMENT_SLUG] = get_target(TARGET_SLUG)
    train_frozen_validation(
        run_root,
        source_run,
        gpu_id,
        capacity_step,
        seed=seed,
        epochs=VALIDATION_EPOCHS,
    )
    result = evaluate_full_validation(
        run_root=run_root,
        slug=EXPERIMENT_SLUG,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=1.0,
        smoke=False,
        gpu_id=gpu_id,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-unclamped-t4",
        "status": "complete",
        "experiment_slug": EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "fape_clamp_prob": 0.0,
        "capacity_configuration_locked_at_step": capacity_step,
        "primary_comparison": {
            "epoch": VALIDATION_EPOCHS,
            "scale": 1.0,
            "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
        },
        "train_label_count": train_n,
        "validation_label_count": validation_n,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "test_accessed": False,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    write_json(eval_root / "t4_unclamped_validation_summary.json", payload)
    write_json(
        ablation_root(run_root) / "t4_unclamped_validation_summary.json",
        payload,
    )
    update_status(
        run_root,
        validate_unclamped_t4="complete",
        t4_unclamped_validation_pass_v2=promotion_pass,
        t4_unclamped_validation_seed=seed,
        t4_unclamped_validation_epoch=VALIDATION_EPOCHS,
        test_accessed=False,
    )
    return payload



def run_shrinkage_validation(
    run_root: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Evaluate the single pre-registered LoRA shrinkage candidate."""
    prior_path = eval_root / "t4_unclamped_validation_summary.json"
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Primary scale=1.0 validation is required first: {prior_path}"
        )
    prior = __import__("json").loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("promotion_pass_v2") is True:
        raise RuntimeError("Primary scale=1.0 already passed; shrinkage is unnecessary")

    split_root = private_root(run_root) / "splits"
    train_n = len(read_labels(split_root / "train_labels.txt"))
    checkpoint = checkpoint_for_epochs(
        full_train_root(run_root, EXPERIMENT_SLUG, seed) / "training",
        VALIDATION_EPOCHS,
        train_n,
    )
    if not (checkpoint.exists() or checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)

    candidate_root = (
        full_train_root(run_root, EXPERIMENT_SLUG, seed)
        / "validation"
        / f"epoch_{VALIDATION_EPOCHS}"
        / "scale_0p5"
    )
    preregistration = {
        "stage": "preregister-unclamped-t4-shrinkage",
        "reason": (
            "scale_1p0_hard_passed_but_nonhard_mean_delta_tm_missed_floor"
        ),
        "candidate_scales": [SHRINKAGE_SCALE],
        "selection_policy": "single_followup_no_scale_grid",
        "epoch": VALIDATION_EPOCHS,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "prior_validation_summary_sha256": sha256_file(prior_path),
        "test_accessed": False,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    prereg_path = eval_root / "t4_unclamped_shrinkage_preregister.json"
    prepare_run_manifest(
        prereg_path,
        preregistration,
        artifact_exists=candidate_root.exists(),
        context="T4 unclamped single shrinkage candidate",
    )

    TARGET_SPECS[EXPERIMENT_SLUG] = get_target(TARGET_SLUG)
    result = evaluate_full_validation(
        run_root=run_root,
        slug=EXPERIMENT_SLUG,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=SHRINKAGE_SCALE,
        smoke=False,
        gpu_id=gpu_id,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-unclamped-t4-shrinkage",
        "status": "complete",
        "experiment_slug": EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "fape_clamp_prob": 0.0,
        "epoch": VALIDATION_EPOCHS,
        "scale": SHRINKAGE_SCALE,
        "selection_policy": "single_followup_no_scale_grid",
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "test_accessed": False,
    }
    summary_path = eval_root / "t4_unclamped_shrinkage_summary.json"
    write_json(summary_path, payload)
    write_json(
        ablation_root(run_root) / "t4_unclamped_shrinkage_summary.json",
        payload,
    )
    update_status(
        run_root,
        validate_unclamped_t4_shrinkage="complete",
        t4_unclamped_shrinkage_scale=SHRINKAGE_SCALE,
        t4_unclamped_shrinkage_pass_v2=promotion_pass,
        test_accessed=False,
    )
    return payload


def run_selective_clamp_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Train and evaluate the single selective-clamp follow-up."""
    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    primary_path = eval_root / "t4_unclamped_validation_summary.json"
    shrinkage_path = eval_root / "t4_unclamped_shrinkage_summary.json"
    for required in (capacity_path, primary_path, shrinkage_path):
        if not required.exists():
            raise FileNotFoundError(required)

    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    split_root = private_root(run_root) / "splits"
    train_n = len(read_labels(split_root / "train_labels.txt"))
    validation_n = len(read_labels(split_root / "validation_labels.txt"))
    if (train_n, validation_n) != (57, 10):
        raise AssertionError(
            f"Unexpected frozen split sizes: train={train_n}, "
            f"validation={validation_n}"
        )

    experiment_root = full_train_root(
        run_root, SELECTIVE_EXPERIMENT_SLUG, seed
    )
    preregistration = {
        "stage": "preregister-t4-selective-clamp",
        "single_changed_variable": "difficulty_conditioned_fape",
        "hard_fape_clamped": False,
        "medium_fape_clamped": True,
        "easy_fape_clamped": True,
        "sampling_mode": "uniform",
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "capacity_configuration_locked_at_step": capacity_step,
        "primary_validation_summary_sha256": sha256_file(primary_path),
        "shrinkage_summary_sha256": sha256_file(shrinkage_path),
        "selection_policy": "single_followup_no_sampling_or_hparam_change",
        "development_test_accessed": False,
        "test_accessed": False,
    }
    eval_root.mkdir(parents=True, exist_ok=True)
    prepare_run_manifest(
        eval_root / "t4_selective_clamp_preregister.json",
        preregistration,
        artifact_exists=experiment_root.exists(),
        context="T4 selective-clamp follow-up",
    )

    TARGET_SPECS[SELECTIVE_EXPERIMENT_SLUG] = get_target(TARGET_SLUG)
    train_frozen_validation(
        run_root,
        source_run,
        gpu_id,
        capacity_step,
        seed=seed,
        epochs=VALIDATION_EPOCHS,
        experiment_slug=SELECTIVE_EXPERIMENT_SLUG,
        difficulty_conditioned_fape=True,
    )
    result = evaluate_full_validation(
        run_root=run_root,
        slug=SELECTIVE_EXPERIMENT_SLUG,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=1.0,
        smoke=False,
        gpu_id=gpu_id,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-t4-selective-clamp",
        "status": "complete",
        "experiment_slug": SELECTIVE_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "difficulty_conditioned_fape": True,
        "clamp_policy": {
            "hard": 0.0,
            "medium": 1.0,
            "easy": 1.0,
        },
        "sampling_mode": "uniform",
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    write_json(eval_root / "t4_selective_clamp_summary.json", payload)
    write_json(
        ablation_root(run_root) / "t4_selective_clamp_summary.json",
        payload,
    )
    update_status(
        run_root,
        validate_t4_selective_clamp="complete",
        t4_selective_clamp_pass_v2=promotion_pass,
        test_accessed=False,
    )
    return payload


def run_anchor_balanced_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Train selective clamp with one pre-registered anchor-balanced ratio."""
    selective_path = eval_root / "t4_selective_clamp_summary.json"
    if not selective_path.exists():
        raise FileNotFoundError(selective_path)
    selective = __import__("json").loads(
        selective_path.read_text(encoding="utf-8")
    )
    if selective.get("promotion_pass_v2") is True:
        raise RuntimeError(
            "Selective-clamp uniform already passed; anchor balancing is unnecessary"
        )

    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    split_root = private_root(run_root) / "splits"
    train_n = len(read_labels(split_root / "train_labels.txt"))
    validation_n = len(read_labels(split_root / "validation_labels.txt"))
    if (train_n, validation_n) != (57, 10):
        raise AssertionError(
            f"Unexpected frozen split sizes: train={train_n}, "
            f"validation={validation_n}"
        )

    experiment_root = full_train_root(
        run_root, ANCHOR_EXPERIMENT_SLUG, seed
    )
    preregistration = {
        "stage": "preregister-t4-anchor-balanced",
        "single_changed_variable": "sampling_ratios",
        "parent_experiment": SELECTIVE_EXPERIMENT_SLUG,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "hard_aware",
        "hard_aware_ratios": ANCHOR_RATIOS,
        "ratio_rationale": (
            "increase_easy_anchors_without_hard_oversampling"
        ),
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "capacity_configuration_locked_at_step": capacity_step,
        "selective_summary_sha256": sha256_file(selective_path),
        "selection_policy": "single_ratio_no_grid",
        "development_test_accessed": False,
        "test_accessed": False,
    }
    prepare_run_manifest(
        eval_root / "t4_anchor_balanced_preregister.json",
        preregistration,
        artifact_exists=experiment_root.exists(),
        context="T4 anchor-balanced follow-up",
    )

    TARGET_SPECS[ANCHOR_EXPERIMENT_SLUG] = get_target(TARGET_SLUG)
    train_frozen_validation(
        run_root,
        source_run,
        gpu_id,
        capacity_step,
        seed=seed,
        epochs=VALIDATION_EPOCHS,
        experiment_slug=ANCHOR_EXPERIMENT_SLUG,
        difficulty_conditioned_fape=True,
        sampling_mode="hard_aware",
        hard_aware_ratios=ANCHOR_RATIOS,
    )
    result = evaluate_full_validation(
        run_root=run_root,
        slug=ANCHOR_EXPERIMENT_SLUG,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=1.0,
        smoke=False,
        gpu_id=gpu_id,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-t4-anchor-balanced",
        "status": "complete",
        "experiment_slug": ANCHOR_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "hard_aware",
        "hard_aware_ratios": ANCHOR_RATIOS,
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    write_json(eval_root / "t4_anchor_balanced_summary.json", payload)
    write_json(
        ablation_root(run_root) / "t4_anchor_balanced_summary.json",
        payload,
    )
    update_status(
        run_root,
        validate_t4_anchor_balanced="complete",
        t4_anchor_balanced_pass_v2=promotion_pass,
        test_accessed=False,
    )
    return payload


def train_and_evaluate_preservation_candidate(
    run_root: Path,
    source_run: Path,
    gpu_id: str,
    seed: int,
    experiment_slug: str,
    preservation_weight: float,
    capacity_step: int,
    hard_correction_weight: float = 0.0,
    hard_correction_margin: float = HARD_CORRECTION_MARGIN,
    hard_correction_min_baseline_error: float = (
        HARD_CORRECTION_MIN_BASELINE_ERROR
    ),
    hard_correction_min_sequence_separation: int = (
        HARD_CORRECTION_MIN_SEQUENCE_SEPARATION
    ),
    hard_soft_tm_weight: float = 0.0,
    hard_soft_tm_margin: float = SOFT_TM_MARGIN,
) -> dict:
    """Train/evaluate one fixed-schema selective-clamp preservation candidate."""
    TARGET_SPECS[experiment_slug] = get_target(TARGET_SLUG)
    train_frozen_validation(
        run_root,
        source_run,
        gpu_id,
        capacity_step,
        seed=seed,
        epochs=VALIDATION_EPOCHS,
        experiment_slug=experiment_slug,
        difficulty_conditioned_fape=True,
        sampling_mode="uniform",
        baseline_preservation_weight=preservation_weight,
        hard_correction_weight=hard_correction_weight,
        hard_correction_margin=hard_correction_margin,
        hard_correction_min_baseline_error=(
            hard_correction_min_baseline_error
        ),
        hard_correction_min_sequence_separation=(
            hard_correction_min_sequence_separation
        ),
        hard_soft_tm_weight=hard_soft_tm_weight,
        hard_soft_tm_margin=hard_soft_tm_margin,
    )
    return evaluate_full_validation(
        run_root=run_root,
        slug=experiment_slug,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=1.0,
        smoke=False,
        gpu_id=gpu_id,
    )


def run_baseline_preservation_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Train selective-clamp T4 with non-hard baseline geometry anchors."""
    selective_path = eval_root / "t4_selective_clamp_summary.json"
    anchor_path = eval_root / "t4_anchor_balanced_summary.json"
    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    for required in (selective_path, anchor_path, capacity_path):
        if not required.exists():
            raise FileNotFoundError(required)
    selective = __import__("json").loads(
        selective_path.read_text(encoding="utf-8")
    )
    anchor = __import__("json").loads(
        anchor_path.read_text(encoding="utf-8")
    )
    if selective.get("promotion_pass_v2") is True:
        raise RuntimeError(
            "Selective-clamp uniform already passed; preservation is unnecessary"
        )
    if anchor.get("promotion_pass_v2") is True:
        raise RuntimeError(
            "Anchor-balanced already passed; preservation is unnecessary"
        )

    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    private = private_root(run_root)
    split_root = private / "splits"
    train_labels = read_labels(split_root / "train_labels.txt")
    validation_labels = read_labels(split_root / "validation_labels.txt")
    if (len(train_labels), len(validation_labels)) != (57, 10):
        raise AssertionError(
            f"Unexpected frozen split sizes: train={len(train_labels)}, "
            f"validation={len(validation_labels)}"
        )
    baseline_dir = (
        source_run / "clients" / "client_0" / "prescreen" / "predictions"
    )
    missing = [
        label for label in train_labels
        if not (
            baseline_dir
            / f"{label}_seq_model_esm1b_ptm_unrelaxed.pdb"
        ).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} local baseline predictions: {missing[:5]}"
        )

    experiment_root = full_train_root(
        run_root, PRESERVATION_EXPERIMENT_SLUG, seed
    )
    preregistration = {
        "stage": "preregister-t4-baseline-preservation",
        "parent_experiment": SELECTIVE_EXPERIMENT_SLUG,
        "single_changed_variable": "nonhard_baseline_distance_preservation",
        "difficulty_conditioned_fape": True,
        "clamp_policy": {"hard": 0.0, "medium": 1.0, "easy": 1.0},
        "sampling_mode": "uniform",
        "baseline_preservation_scope": ["medium", "easy"],
        "baseline_preservation_weight": PRESERVATION_WEIGHT,
        "baseline_preservation_geometry": "all_pair_ca_distances_huber",
        "baseline_predictions_are_client_local": True,
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "capacity_configuration_locked_at_step": capacity_step,
        "selective_summary_sha256": sha256_file(selective_path),
        "anchor_balanced_summary_sha256": sha256_file(anchor_path),
        "selection_policy": "single_fixed_weight_no_grid",
        "development_test_accessed": False,
        "test_accessed": False,
    }
    prepare_run_manifest(
        eval_root / "t4_baseline_preservation_preregister.json",
        preregistration,
        artifact_exists=experiment_root.exists(),
        context="T4 baseline-preservation follow-up",
    )

    result = train_and_evaluate_preservation_candidate(
        run_root=run_root,
        source_run=source_run,
        gpu_id=gpu_id,
        seed=seed,
        experiment_slug=PRESERVATION_EXPERIMENT_SLUG,
        preservation_weight=PRESERVATION_WEIGHT,
        capacity_step=capacity_step,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-t4-baseline-preservation",
        "status": "complete",
        "experiment_slug": PRESERVATION_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "rank": 4,
        "alpha": 8.0,
        "seed": seed,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "uniform",
        "baseline_preservation_scope": ["medium", "easy"],
        "baseline_preservation_weight": PRESERVATION_WEIGHT,
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    summary_name = (
        "t4_baseline_preservation_summary.json"
        if seed == 42
        else f"t4_baseline_preservation_seed_{seed}_summary.json"
    )
    write_json(eval_root / summary_name, payload)
    write_json(ablation_root(run_root) / summary_name, payload)
    status_fields = {
        "validate_t4_baseline_preservation": "complete",
        "t4_baseline_preservation_pass_v2": promotion_pass,
    } if seed == 42 else {
        f"validate_t4_baseline_preservation_seed_{seed}": "complete",
        f"t4_baseline_preservation_seed_{seed}_pass_v2": promotion_pass,
    }
    update_status(
        run_root,
        **status_fields,
        test_accessed=False,
    )
    return payload


def run_preservation_seed_confirmation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Confirm the passing preservation method with seeds 42, 43, and 44."""
    primary_path = eval_root / "t4_baseline_preservation_summary.json"
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    primary = __import__("json").loads(
        primary_path.read_text(encoding="utf-8")
    )
    if primary.get("promotion_pass_v2") is not True:
        raise RuntimeError(
            "Seed 42 baseline-preservation run did not pass promotion-v2"
        )

    seeds = (42, 43, 44)
    confirmation_path = (
        eval_root / "t4_baseline_preservation_seed_confirmation.json"
    )
    preregistration = {
        "stage": "preregister-t4-baseline-preservation-seeds",
        "experiment_slug": PRESERVATION_EXPERIMENT_SLUG,
        "seeds": list(seeds),
        "fixed_epoch": VALIDATION_EPOCHS,
        "fixed_scale": 1.0,
        "fixed_preservation_weight": PRESERVATION_WEIGHT,
        "seed_gate": (
            "repository_seed_confirmation_pass: mean hard delta TM >= 0.005; "
            "at least 2/3 hard means positive; every seed nonhard mean delta "
            "TM >= -0.005 and all mean delta lDDT >= -0.002"
        ),
        "primary_seed_42_summary_sha256": sha256_file(primary_path),
        "development_test_accessed": False,
        "test_accessed": False,
    }
    prepare_run_manifest(
        eval_root / "t4_baseline_preservation_seeds_preregister.json",
        preregistration,
        artifact_exists=confirmation_path.exists(),
        context="T4 baseline-preservation seed confirmation",
    )

    seed_rows = []
    for seed in seeds:
        if seed == 42:
            summary = primary
        else:
            summary = run_baseline_preservation_validation(
                run_root,
                source_run,
                eval_root,
                gpu_id,
                seed,
            )
        metrics = summary["result"]
        row = {
            "seed": seed,
            "hard_mean_delta_tm": metrics["hard_mean_delta_tm"],
            "hard_median_delta_tm": metrics["hard_median_delta_tm"],
            "nonhard_mean_delta_tm": metrics["nonhard_mean_delta_tm"],
            "all_mean_delta_lddt": metrics["all_mean_delta_lddt"],
            "promotion_v2_pass": bool(
                (metrics.get("promotion_v2") or {}).get("promotion_pass")
            ),
        }
        seed_rows.append(row)

    decision = seed_confirmation_pass(seed_rows)
    payload = {
        "stage": "confirm-t4-baseline-preservation-seeds",
        "status": "complete" if decision["pass"] else "failed",
        "experiment_slug": PRESERVATION_EXPERIMENT_SLUG,
        "seeds": seed_rows,
        "decision": decision,
        "seed_confirmation_pass": bool(decision["pass"]),
        "blocks_development": not bool(decision["pass"]),
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    write_json(confirmation_path, payload)
    write_json(
        ablation_root(run_root)
        / "t4_baseline_preservation_seed_confirmation.json",
        payload,
    )
    update_status(
        run_root,
        confirm_t4_baseline_preservation_seeds=payload["status"],
        t4_baseline_preservation_seed_confirmation_pass=bool(
            decision["pass"]
        ),
        blocks_other_clients=True,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def run_stronger_preservation_seed43(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Test one pre-registered 1.5 preservation weight on failing seed 43."""
    confirmation_path = (
        eval_root / "t4_baseline_preservation_seed_confirmation.json"
    )
    if not confirmation_path.exists():
        raise FileNotFoundError(confirmation_path)
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    decision = confirmation.get("decision") or {}
    if confirmation.get("seed_confirmation_pass") is True:
        raise RuntimeError("Weight 1.0 already passed seed confirmation")
    if not (
        decision.get("nonhard_ok") is False
        and decision.get("lddt_ok") is True
        and int(decision.get("positive_seed_count", 0)) >= 2
        and float(decision.get("mean_hard_mean_delta_tm", -1.0)) >= 0.005
    ):
        raise RuntimeError(
            "Weight increase is only authorized for isolated non-hard failure"
        )

    seed = 43
    experiment_root = full_train_root(
        run_root, STRONG_PRESERVATION_EXPERIMENT_SLUG, seed
    )
    preregistration = {
        "stage": "preregister-t4-baseline-preservation-w1p5-seed43",
        "parent_experiment": PRESERVATION_EXPERIMENT_SLUG,
        "single_changed_variable": "baseline_preservation_weight",
        "baseline_preservation_weight_before": PRESERVATION_WEIGHT,
        "baseline_preservation_weight_after": STRONG_PRESERVATION_WEIGHT,
        "selected_seed": seed,
        "selection_reason": (
            "only seed failing the prior confirmation, by nonhard safety "
            "margin 0.000342857"
        ),
        "difficulty_conditioned_fape": True,
        "sampling_mode": "uniform",
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "selection_policy": "single_minimal_weight_increase_no_grid",
        "prior_confirmation_sha256": sha256_file(confirmation_path),
        "development_test_accessed": False,
        "test_accessed": False,
    }
    prereg_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed43_preregister.json"
    )
    prepare_run_manifest(
        prereg_path,
        preregistration,
        artifact_exists=experiment_root.exists(),
        context="T4 stronger baseline-preservation seed43",
    )

    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    result = train_and_evaluate_preservation_candidate(
        run_root=run_root,
        source_run=source_run,
        gpu_id=gpu_id,
        seed=seed,
        experiment_slug=STRONG_PRESERVATION_EXPERIMENT_SLUG,
        preservation_weight=STRONG_PRESERVATION_WEIGHT,
        capacity_step=min(passing_steps),
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": "validate-t4-baseline-preservation-w1p5-seed43",
        "status": "complete",
        "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "parent_experiment": PRESERVATION_EXPERIMENT_SLUG,
        "seed": seed,
        "rank": 4,
        "alpha": 8.0,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "uniform",
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_strong_multiseed": not promotion_pass,
        "blocks_development": True,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    summary_name = (
        "t4_baseline_preservation_w1p5_seed_43_summary.json"
    )
    write_json(eval_root / summary_name, payload)
    write_json(ablation_root(run_root) / summary_name, payload)
    update_status(
        run_root,
        validate_t4_baseline_preservation_w1p5_seed43="complete",
        t4_baseline_preservation_w1p5_seed43_pass_v2=promotion_pass,
        blocks_other_clients=True,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def run_strong_preservation_seed_confirmation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Confirm weight-1.5 preservation with fixed seeds 42, 43, and 44."""
    seed43_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed_43_summary.json"
    )
    if not seed43_path.exists():
        raise FileNotFoundError(seed43_path)
    seed43_summary = __import__("json").loads(
        seed43_path.read_text(encoding="utf-8")
    )
    if seed43_summary.get("promotion_pass_v2") is not True:
        raise RuntimeError("Weight-1.5 seed 43 did not pass promotion-v2")

    seeds = (42, 43, 44)
    confirmation_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed_confirmation.json"
    )
    preregistration = {
        "stage": "preregister-t4-baseline-preservation-w1p5-seeds",
        "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "seeds": list(seeds),
        "fixed_epoch": VALIDATION_EPOCHS,
        "fixed_scale": 1.0,
        "fixed_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "seed_gate": (
            "repository_seed_confirmation_pass: mean hard delta TM >= 0.005; "
            "at least 2/3 hard means positive; every seed nonhard mean delta "
            "TM >= -0.005 and all mean delta lDDT >= -0.002"
        ),
        "seed43_summary_sha256": sha256_file(seed43_path),
        "selection_policy": "confirm_fixed_w1p5_no_further_tuning",
        "development_test_accessed": False,
        "test_accessed": False,
    }
    prepare_run_manifest(
        eval_root
        / "t4_baseline_preservation_w1p5_seeds_preregister.json",
        preregistration,
        artifact_exists=confirmation_path.exists(),
        context="T4 weight-1.5 baseline-preservation seed confirmation",
    )

    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    seed_rows = []
    for seed in seeds:
        if seed == 43:
            summary = seed43_summary
            metrics = summary["result"]
        else:
            metrics = train_and_evaluate_preservation_candidate(
                run_root=run_root,
                source_run=source_run,
                gpu_id=gpu_id,
                seed=seed,
                experiment_slug=STRONG_PRESERVATION_EXPERIMENT_SLUG,
                preservation_weight=STRONG_PRESERVATION_WEIGHT,
                capacity_step=capacity_step,
            )
            promotion_v2 = metrics.get("promotion_v2") or {}
            promotion_pass = bool(
                promotion_v2.get(
                    "promotion_pass", metrics.get("promotion_pass")
                )
            )
            summary = {
                "stage": (
                    "validate-t4-baseline-preservation-w1p5-seed"
                    f"{seed}"
                ),
                "status": "complete",
                "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
                "seed": seed,
                "rank": 4,
                "alpha": 8.0,
                "difficulty_conditioned_fape": True,
                "sampling_mode": "uniform",
                "baseline_preservation_weight": (
                    STRONG_PRESERVATION_WEIGHT
                ),
                "epoch": VALIDATION_EPOCHS,
                "scale": 1.0,
                "result": metrics,
                "promotion_pass_v2": promotion_pass,
                "development_test_accessed": False,
                "test_accessed": False,
            }
            summary_name = (
                "t4_baseline_preservation_w1p5_seed_"
                f"{seed}_summary.json"
            )
            write_json(eval_root / summary_name, summary)
            write_json(ablation_root(run_root) / summary_name, summary)

        seed_rows.append({
            "seed": seed,
            "hard_mean_delta_tm": metrics["hard_mean_delta_tm"],
            "hard_median_delta_tm": metrics["hard_median_delta_tm"],
            "nonhard_mean_delta_tm": metrics["nonhard_mean_delta_tm"],
            "all_mean_delta_lddt": metrics["all_mean_delta_lddt"],
            "promotion_v2_pass": bool(
                (metrics.get("promotion_v2") or {}).get("promotion_pass")
            ),
        })

    decision = seed_confirmation_pass(seed_rows)
    payload = {
        "stage": "confirm-t4-baseline-preservation-w1p5-seeds",
        "status": "complete" if decision["pass"] else "failed",
        "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "seeds": seed_rows,
        "decision": decision,
        "seed_confirmation_pass": bool(decision["pass"]),
        "blocks_development": not bool(decision["pass"]),
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_accessed": False,
        "test_accessed": False,
    }
    write_json(confirmation_path, payload)
    write_json(
        ablation_root(run_root)
        / "t4_baseline_preservation_w1p5_seed_confirmation.json",
        payload,
    )
    update_status(
        run_root,
        confirm_t4_baseline_preservation_w1p5_seeds=payload["status"],
        t4_baseline_preservation_w1p5_seed_confirmation_pass=bool(
            decision["pass"]
        ),
        blocks_other_clients=True,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def run_hard_correction_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Develop the hard-only geometry margin on frozen validation only."""
    development_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_development_summary.json"
    )
    confirmation_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed_confirmation.json"
    )
    for required in (development_path, confirmation_path):
        if not required.exists():
            raise FileNotFoundError(required)
    development = __import__("json").loads(
        development_path.read_text(encoding="utf-8")
    )
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if development.get("development_pass") is not False:
        raise RuntimeError(
            "Hard correction is only authorized after development failure"
        )
    if development.get("external_final_test_accessed") is not False:
        raise RuntimeError("external_final was already accessed")
    if confirmation.get("seed_confirmation_pass") is not True:
        raise RuntimeError("Parent weight-1.5 validation was not confirmed")

    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    experiment_root = full_train_root(
        run_root, HARD_CORRECTION_EXPERIMENT_SLUG, seed
    )
    summary_name = (
        "t4_hard_correction_summary.json"
        if seed == 42
        else f"t4_hard_correction_seed_{seed}_summary.json"
    )
    preregistration = {
        "stage": "preregister-t4-hard-correction",
        "experiment_slug": HARD_CORRECTION_EXPERIMENT_SLUG,
        "parent_experiment": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "new_method": (
            "hard-only baseline-relative long-range C-alpha "
            "distance improvement margin"
        ),
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_correction_weight": HARD_CORRECTION_WEIGHT,
        "hard_correction_margin_angstrom": HARD_CORRECTION_MARGIN,
        "hard_correction_min_baseline_error_angstrom": (
            HARD_CORRECTION_MIN_BASELINE_ERROR
        ),
        "hard_correction_min_sequence_separation": (
            HARD_CORRECTION_MIN_SEQUENCE_SEPARATION
        ),
        "pair_weight_cap": 4.0,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "uniform",
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "selection_policy": "single_mechanistic_candidate_no_grid",
        "parent_confirmation_sha256": sha256_file(confirmation_path),
        "failed_development_sha256": sha256_file(development_path),
        "development_informed_method_design": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    prereg_path = (
        eval_root
        / f"t4_hard_correction_seed_{seed}_preregister.json"
    )
    prepare_run_manifest(
        prereg_path,
        preregistration,
        artifact_exists=experiment_root.exists(),
        context=f"T4 hard correction seed={seed}",
    )

    result = train_and_evaluate_preservation_candidate(
        run_root=run_root,
        source_run=source_run,
        gpu_id=gpu_id,
        seed=seed,
        experiment_slug=HARD_CORRECTION_EXPERIMENT_SLUG,
        preservation_weight=STRONG_PRESERVATION_WEIGHT,
        capacity_step=capacity_step,
        hard_correction_weight=HARD_CORRECTION_WEIGHT,
        hard_correction_margin=HARD_CORRECTION_MARGIN,
        hard_correction_min_baseline_error=(
            HARD_CORRECTION_MIN_BASELINE_ERROR
        ),
        hard_correction_min_sequence_separation=(
            HARD_CORRECTION_MIN_SEQUENCE_SEPARATION
        ),
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": f"validate-t4-hard-correction-seed-{seed}",
        "status": "complete",
        "experiment_slug": HARD_CORRECTION_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "seed": seed,
        "rank": 4,
        "alpha": 8.0,
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_correction_weight": HARD_CORRECTION_WEIGHT,
        "hard_correction_margin_angstrom": HARD_CORRECTION_MARGIN,
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_external_final": True,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    write_json(eval_root / summary_name, payload)
    write_json(ablation_root(run_root) / summary_name, payload)
    update_status(
        run_root,
        **{
            f"validate_t4_hard_correction_seed_{seed}": "complete",
            f"t4_hard_correction_seed_{seed}_pass_v2": promotion_pass,
        },
        blocks_other_clients=True,
        blocks_fedlora=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def run_soft_tm_validation(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str,
    seed: int,
) -> dict:
    """Develop a hard-only aligned soft-TM margin on validation."""
    hard_summary_path = eval_root / "t4_hard_correction_summary.json"
    development_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_development_summary.json"
    )
    confirmation_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed_confirmation.json"
    )
    for required in (
        hard_summary_path,
        development_path,
        confirmation_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    hard_summary = __import__("json").loads(
        hard_summary_path.read_text(encoding="utf-8")
    )
    development = __import__("json").loads(
        development_path.read_text(encoding="utf-8")
    )
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if hard_summary.get("promotion_pass_v2") is not False:
        raise RuntimeError("Pair-distance hard correction already passed")
    if development.get("external_final_test_accessed") is not False:
        raise RuntimeError("external_final was already accessed")
    if confirmation.get("seed_confirmation_pass") is not True:
        raise RuntimeError("Parent weight-1.5 validation was not confirmed")

    capacity_path = eval_root / "t4_unclamped_capacity_summary.json"
    capacity = __import__("json").loads(
        capacity_path.read_text(encoding="utf-8")
    )
    passing_steps = [
        int(row["step"])
        for row in capacity.get("rows", [])
        if row.get("capacity_v2_pass")
    ]
    if not passing_steps:
        raise RuntimeError("T4 unclamped hard-case capacity-v2 did not pass")
    capacity_step = min(passing_steps)

    experiment_root = full_train_root(
        run_root, SOFT_TM_EXPERIMENT_SLUG, seed
    )
    summary_name = (
        "t4_soft_tm_summary.json"
        if seed == 42
        else f"t4_soft_tm_seed_{seed}_summary.json"
    )
    preregistration = {
        "stage": "preregister-t4-soft-tm",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "parent_experiment": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "replaces_failed_experiment": HARD_CORRECTION_EXPERIMENT_SLUG,
        "new_method": (
            "hard-only Kabsch-aligned baseline-relative soft-TM margin"
        ),
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "hard_soft_tm_score_margin": SOFT_TM_MARGIN,
        "hard_pair_correction_weight": 0.0,
        "difficulty_conditioned_fape": True,
        "sampling_mode": "uniform",
        "rank": 4,
        "alpha": 8.0,
        "learning_rate": 1e-4,
        "epochs": VALIDATION_EPOCHS,
        "evaluation_scale": 1.0,
        "selection_policy": "single_tm_aligned_candidate_no_grid",
        "parent_confirmation_sha256": sha256_file(confirmation_path),
        "failed_pair_correction_sha256": sha256_file(hard_summary_path),
        "failed_development_sha256": sha256_file(development_path),
        "development_informed_method_design": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    prereg_path = (
        eval_root / f"t4_soft_tm_seed_{seed}_preregister.json"
    )
    prepare_run_manifest(
        prereg_path,
        preregistration,
        artifact_exists=experiment_root.exists(),
        context=f"T4 soft-TM seed={seed}",
    )

    result = train_and_evaluate_preservation_candidate(
        run_root=run_root,
        source_run=source_run,
        gpu_id=gpu_id,
        seed=seed,
        experiment_slug=SOFT_TM_EXPERIMENT_SLUG,
        preservation_weight=STRONG_PRESERVATION_WEIGHT,
        capacity_step=capacity_step,
        hard_soft_tm_weight=SOFT_TM_WEIGHT,
        hard_soft_tm_margin=SOFT_TM_MARGIN,
    )
    promotion_v2 = result.get("promotion_v2") or {}
    promotion_pass = bool(
        promotion_v2.get("promotion_pass", result.get("promotion_pass"))
    )
    payload = {
        "stage": f"validate-t4-soft-tm-seed-{seed}",
        "status": "complete",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "seed": seed,
        "rank": 4,
        "alpha": 8.0,
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "hard_soft_tm_score_margin": SOFT_TM_MARGIN,
        "epoch": VALIDATION_EPOCHS,
        "scale": 1.0,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_multiseed": not promotion_pass,
        "blocks_external_final": True,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    write_json(eval_root / summary_name, payload)
    write_json(ablation_root(run_root) / summary_name, payload)
    update_status(
        run_root,
        **{
            f"validate_t4_soft_tm_seed_{seed}": "complete",
            f"t4_soft_tm_seed_{seed}_pass_v2": promotion_pass,
        },
        blocks_other_clients=True,
        blocks_fedlora=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def preregister_soft_tm_seed_confirmation(
    run_root: Path,
    eval_root: Path,
) -> dict:
    """Lock soft-TM seeds and aggregate gate before seeing extra seeds."""
    primary_path = eval_root / "t4_soft_tm_summary.json"
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    primary = __import__("json").loads(
        primary_path.read_text(encoding="utf-8")
    )
    if primary.get("promotion_pass_v2") is not True:
        raise RuntimeError("Soft-TM seed 42 did not pass promotion-v2")

    confirmation_path = eval_root / "t4_soft_tm_seed_confirmation.json"
    payload = {
        "stage": "preregister-t4-soft-tm-seeds",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "seeds": [42, 43, 44],
        "fixed_epoch": VALIDATION_EPOCHS,
        "fixed_scale": 1.0,
        "fixed_baseline_preservation_weight": (
            STRONG_PRESERVATION_WEIGHT
        ),
        "fixed_hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "fixed_hard_soft_tm_margin": SOFT_TM_MARGIN,
        "seed_gate": (
            "repository_seed_confirmation_pass: mean hard delta TM >= 0.005; "
            "at least 2/3 hard means positive; every seed nonhard mean delta "
            "TM >= -0.005 and all mean delta lDDT >= -0.002"
        ),
        "selection_policy": "fixed_method_no_seed_specific_tuning",
        "primary_seed_42_summary_sha256": sha256_file(primary_path),
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    prereg_path = eval_root / "t4_soft_tm_seeds_preregister.json"
    prepare_run_manifest(
        prereg_path,
        payload,
        artifact_exists=confirmation_path.exists(),
        context="T4 soft-TM seed confirmation",
    )
    write_json(
        ablation_root(run_root) / "t4_soft_tm_seeds_preregister.json",
        payload,
    )
    return payload


def confirm_soft_tm_seeds(
    run_root: Path,
    eval_root: Path,
) -> dict:
    """Aggregate the pre-registered soft-TM seeds without new evaluation."""
    prereg_path = eval_root / "t4_soft_tm_seeds_preregister.json"
    if not prereg_path.exists():
        raise FileNotFoundError(
            "Run preregister-soft-tm-seeds before extra seed results"
        )
    preregistration = __import__("json").loads(
        prereg_path.read_text(encoding="utf-8")
    )
    seeds = tuple(int(seed) for seed in preregistration["seeds"])
    seed_rows = []
    summary_hashes = {}
    for seed in seeds:
        summary_name = (
            "t4_soft_tm_summary.json"
            if seed == 42
            else f"t4_soft_tm_seed_{seed}_summary.json"
        )
        summary_path = eval_root / summary_name
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = __import__("json").loads(
            summary_path.read_text(encoding="utf-8")
        )
        metrics = summary["result"]
        seed_rows.append({
            "seed": seed,
            "hard_mean_delta_tm": metrics["hard_mean_delta_tm"],
            "hard_median_delta_tm": metrics["hard_median_delta_tm"],
            "nonhard_mean_delta_tm": metrics["nonhard_mean_delta_tm"],
            "all_mean_delta_lddt": metrics["all_mean_delta_lddt"],
            "promotion_v2_pass": bool(
                (metrics.get("promotion_v2") or {}).get("promotion_pass")
            ),
        })
        summary_hashes[str(seed)] = sha256_file(summary_path)

    decision = seed_confirmation_pass(seed_rows)
    passed = bool(decision["pass"])
    payload = {
        "stage": "confirm-t4-soft-tm-seeds",
        "status": "complete" if passed else "failed",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "seeds": seed_rows,
        "decision": decision,
        "summary_sha256_by_seed": summary_hashes,
        "preregistration_sha256": sha256_file(prereg_path),
        "seed_confirmation_pass": passed,
        "blocks_external_final": not passed,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    confirmation_path = eval_root / "t4_soft_tm_seed_confirmation.json"
    write_json(confirmation_path, payload)
    write_json(
        ablation_root(run_root)
        / "t4_soft_tm_seed_confirmation.json",
        payload,
    )
    update_status(
        run_root,
        confirm_t4_soft_tm_seeds=payload["status"],
        t4_soft_tm_seed_confirmation_pass=passed,
        blocks_other_clients=True,
        blocks_fedlora=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def run_soft_tm_shrink_seed43(
    run_root: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Test one pre-registered global 0.85 scale on failing seed 43."""
    confirmation_path = eval_root / "t4_soft_tm_seed_confirmation.json"
    if not confirmation_path.exists():
        raise FileNotFoundError(confirmation_path)
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    decision = confirmation.get("decision") or {}
    if confirmation.get("seed_confirmation_pass") is True:
        raise RuntimeError("Soft-TM scale 1.0 already passed seed confirmation")
    if not (
        decision.get("nonhard_ok") is False
        and decision.get("lddt_ok") is True
        and int(decision.get("positive_seed_count", 0)) == 3
        and float(decision.get("mean_hard_mean_delta_tm", -1.0)) >= 0.005
    ):
        raise RuntimeError(
            "Shrinkage is only authorized for isolated non-hard failure"
        )

    seed = 43
    preregistration = {
        "stage": "preregister-t4-soft-tm-scale0p85",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "parent_scale": 1.0,
        "fixed_scale": SOFT_TM_SHRINK_SCALE,
        "seeds": [42, 43, 44],
        "first_evaluation_seed": seed,
        "first_evaluation_reason": (
            "only seed violating non-hard floor, by 0.0006"
        ),
        "selection_rationale": (
            "minimal fixed shrinkage expected to preserve hard mean above "
            "0.005 while returning seed43 non-hard mean above -0.005"
        ),
        "fixed_epoch": VALIDATION_EPOCHS,
        "fixed_baseline_preservation_weight": (
            STRONG_PRESERVATION_WEIGHT
        ),
        "fixed_hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "fixed_hard_soft_tm_margin": SOFT_TM_MARGIN,
        "selection_policy": "single_global_scale_no_grid",
        "scale1_confirmation_sha256": sha256_file(confirmation_path),
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    prereg_path = eval_root / "t4_soft_tm_scale0p85_preregister.json"
    seed43_summary_path = (
        eval_root / "t4_soft_tm_scale0p85_seed_43_summary.json"
    )
    prepare_run_manifest(
        prereg_path,
        preregistration,
        artifact_exists=seed43_summary_path.exists(),
        context="T4 soft-TM scale0.85 seed43",
    )
    write_json(
        ablation_root(run_root) / prereg_path.name,
        preregistration,
    )

    TARGET_SPECS[SOFT_TM_EXPERIMENT_SLUG] = get_target(TARGET_SLUG)
    result = evaluate_full_validation(
        run_root=run_root,
        slug=SOFT_TM_EXPERIMENT_SLUG,
        seed=seed,
        epoch=VALIDATION_EPOCHS,
        scale=SOFT_TM_SHRINK_SCALE,
        smoke=False,
        gpu_id=gpu_id,
    )
    promotion_pass = bool(
        (result.get("promotion_v2") or {}).get(
            "promotion_pass", result.get("promotion_pass")
        )
    )
    payload = {
        "stage": "validate-t4-soft-tm-scale0p85-seed43",
        "status": "complete",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "seed": seed,
        "epoch": VALIDATION_EPOCHS,
        "scale": SOFT_TM_SHRINK_SCALE,
        "result": result,
        "promotion_pass_v2": promotion_pass,
        "blocks_shrink_confirmation": not promotion_pass,
        "blocks_external_final": True,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    write_json(seed43_summary_path, payload)
    write_json(ablation_root(run_root) / seed43_summary_path.name, payload)
    update_status(
        run_root,
        validate_t4_soft_tm_scale0p85_seed43="complete",
        t4_soft_tm_scale0p85_seed43_pass_v2=promotion_pass,
        blocks_other_clients=True,
        blocks_fedlora=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def confirm_soft_tm_shrink(
    run_root: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Evaluate and aggregate the fixed 0.85 scale for all three seeds."""
    prereg_path = eval_root / "t4_soft_tm_scale0p85_preregister.json"
    seed43_path = eval_root / "t4_soft_tm_scale0p85_seed_43_summary.json"
    for required in (prereg_path, seed43_path):
        if not required.exists():
            raise FileNotFoundError(required)
    seed43_summary = __import__("json").loads(
        seed43_path.read_text(encoding="utf-8")
    )
    if seed43_summary.get("promotion_pass_v2") is not True:
        raise RuntimeError("Scale0.85 seed43 did not pass promotion-v2")
    preregistration = __import__("json").loads(
        prereg_path.read_text(encoding="utf-8")
    )
    seeds = tuple(int(seed) for seed in preregistration["seeds"])
    TARGET_SPECS[SOFT_TM_EXPERIMENT_SLUG] = get_target(TARGET_SLUG)

    seed_rows = []
    summary_hashes = {}
    for seed in seeds:
        if seed == 43:
            summary = seed43_summary
            metrics = summary["result"]
            summary_path = seed43_path
        else:
            metrics = evaluate_full_validation(
                run_root=run_root,
                slug=SOFT_TM_EXPERIMENT_SLUG,
                seed=seed,
                epoch=VALIDATION_EPOCHS,
                scale=SOFT_TM_SHRINK_SCALE,
                smoke=False,
                gpu_id=gpu_id,
            )
            promotion_pass = bool(
                (metrics.get("promotion_v2") or {}).get(
                    "promotion_pass", metrics.get("promotion_pass")
                )
            )
            summary = {
                "stage": (
                    f"validate-t4-soft-tm-scale0p85-seed{seed}"
                ),
                "status": "complete",
                "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
                "seed": seed,
                "epoch": VALIDATION_EPOCHS,
                "scale": SOFT_TM_SHRINK_SCALE,
                "result": metrics,
                "promotion_pass_v2": promotion_pass,
                "development_test_reaccessed": False,
                "external_final_test_accessed": False,
                "test_accessed": False,
            }
            summary_path = (
                eval_root
                / f"t4_soft_tm_scale0p85_seed_{seed}_summary.json"
            )
            write_json(summary_path, summary)
            write_json(
                ablation_root(run_root) / summary_path.name,
                summary,
            )
        seed_rows.append({
            "seed": seed,
            "hard_mean_delta_tm": metrics["hard_mean_delta_tm"],
            "hard_median_delta_tm": metrics["hard_median_delta_tm"],
            "nonhard_mean_delta_tm": metrics["nonhard_mean_delta_tm"],
            "all_mean_delta_lddt": metrics["all_mean_delta_lddt"],
            "promotion_v2_pass": bool(
                (metrics.get("promotion_v2") or {}).get("promotion_pass")
            ),
        })
        summary_hashes[str(seed)] = sha256_file(summary_path)

    decision = seed_confirmation_pass(seed_rows)
    passed = bool(decision["pass"])
    payload = {
        "stage": "confirm-t4-soft-tm-scale0p85-seeds",
        "status": "complete" if passed else "failed",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "scale": SOFT_TM_SHRINK_SCALE,
        "seeds": seed_rows,
        "decision": decision,
        "summary_sha256_by_seed": summary_hashes,
        "preregistration_sha256": sha256_file(prereg_path),
        "seed_confirmation_pass": passed,
        "blocks_external_final": not passed,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "test_accessed": False,
    }
    confirmation_path = (
        eval_root / "t4_soft_tm_scale0p85_seed_confirmation.json"
    )
    write_json(confirmation_path, payload)
    write_json(
        ablation_root(run_root) / confirmation_path.name,
        payload,
    )
    update_status(
        run_root,
        confirm_t4_soft_tm_scale0p85_seeds=payload["status"],
        t4_soft_tm_scale0p85_seed_confirmation_pass=passed,
        blocks_other_clients=True,
        blocks_fedlora=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def freeze_soft_tm_candidate(
    run_root: Path,
    eval_root: Path,
) -> dict:
    """Freeze the validated client0 method without opening shared final."""
    confirmation_path = (
        eval_root / "t4_soft_tm_scale0p85_seed_confirmation.json"
    )
    prereg_path = eval_root / "t4_soft_tm_scale0p85_preregister.json"
    for required in (confirmation_path, prereg_path):
        if not required.exists():
            raise FileNotFoundError(required)
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if confirmation.get("seed_confirmation_pass") is not True:
        raise RuntimeError("Scale0.85 multi-seed confirmation did not pass")
    seed_rows = confirmation.get("seeds", [])
    if sorted(int(row.get("seed", -1)) for row in seed_rows) != [42, 43, 44]:
        raise RuntimeError("Frozen confirmation must contain seeds 42/43/44")
    if any(row.get("promotion_v2_pass") is not True for row in seed_rows):
        raise RuntimeError("At least one frozen validation seed did not pass")
    if confirmation.get("external_final_test_accessed") is not False:
        raise RuntimeError("Shared external_final was already accessed")

    split_manifest_path = run_root / "splits" / "split_manifest.json"
    run_manifest_path = run_root / "run_manifest.json"
    for required in (split_manifest_path, run_manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)
    split_manifest = __import__("json").loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    run_manifest = __import__("json").loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    external = split_manifest.get("external_final") or {}
    if (
        split_manifest.get("final_test_lock_state") != "unlocked"
        or run_manifest.get("final_test_lock_state") != "unlocked"
        or external.get("locked") is not False
    ):
        raise RuntimeError(
            "Shared external_final must remain untouched and unlocked "
            "during client replication"
        )

    seed = 42
    epoch = VALIDATION_EPOCHS
    spec = get_target(TARGET_SLUG)
    root = full_train_root(run_root, SOFT_TM_EXPERIMENT_SLUG, seed)
    epoch_len = len(
        read_labels(private_root(run_root) / "splits" / "train_labels.txt")
    )
    checkpoint = checkpoint_for_epochs(root / "training", epoch, epoch_len)
    training_manifest = root / "local_run.json"
    export_root = (
        root / "validation" / f"epoch_{epoch}" / "scale_0p85"
    )
    model = export_root / "model_base_global_adapter_raw_scale_0.85.pt"
    export_manifest = model.with_suffix(".export_manifest.json")
    for required in (checkpoint, training_manifest, model, export_manifest):
        if not (required.exists() or required.is_dir()):
            raise FileNotFoundError(required)
    training_data = __import__("json").loads(
        training_manifest.read_text(encoding="utf-8")
    )
    expected_training = {
        "target_slug": TARGET_SLUG,
        "target": spec.target,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "learning_rate": 1e-4,
        "sampling_mode": "uniform",
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "hard_soft_tm_margin": SOFT_TM_MARGIN,
    }
    training_mismatches = {
        key: {"expected": value, "actual": training_data.get(key)}
        for key, value in expected_training.items()
        if training_data.get(key) != value
    }
    if training_mismatches:
        raise RuntimeError(f"Frozen training config mismatch: {training_mismatches}")
    export_data = __import__("json").loads(
        export_manifest.read_text(encoding="utf-8")
    )
    export_expected = {
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_scale": SOFT_TM_SHRINK_SCALE,
    }
    export_mismatches = {
        key: {"expected": value, "actual": export_data.get(key)}
        for key, value in export_expected.items()
        if export_data.get(key) != value
    }
    if export_mismatches:
        raise RuntimeError(f"Frozen export config mismatch: {export_mismatches}")
    schema_fingerprint = export_data.get("lora_schema_fingerprint")
    if schema_fingerprint != (
        "358441d4d7d301e89f0afa3fb09c0ffad899616bb79f5b4e1721109c4f1e6cb8"
    ):
        raise RuntimeError(
            f"Unexpected T4 schema fingerprint: {schema_fingerprint}"
        )

    payload = {
        "stage": "freeze-client0-soft-tm-candidate",
        "status": "frozen_for_client_replication",
        "claim_scope": (
            "client0 internal cluster-held-out validation; not final-test evidence"
        ),
        "client_id": "client_0",
        "experiment_slug": SOFT_TM_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "lora_schema_fingerprint": schema_fingerprint,
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.0,
        "learning_rate": 1e-4,
        "sampling_mode": "uniform",
        "baseline_preservation_weight": STRONG_PRESERVATION_WEIGHT,
        "hard_soft_tm_weight": SOFT_TM_WEIGHT,
        "hard_soft_tm_margin": SOFT_TM_MARGIN,
        "selected_seed": seed,
        "selected_epoch": epoch,
        "inference_adapter_scale": SOFT_TM_SHRINK_SCALE,
        "selection_policy": (
            "validation-only, one fixed shrink after isolated non-hard failure"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "validated_model": str(model),
        "validated_model_sha256": sha256_file(model),
        "export_manifest": str(export_manifest),
        "export_manifest_sha256": sha256_file(export_manifest),
        "validation_preregistration_sha256": sha256_file(prereg_path),
        "validation_confirmation_sha256": sha256_file(confirmation_path),
        "validation_decision": confirmation.get("decision"),
        "validation_seeds": confirmation.get("seeds"),
        "ready_for_other_client_replication": True,
        "ready_for_fedlora_aggregation": False,
        "fedlora_block_reason": (
            "client1-client4 must independently pass before aggregation"
        ),
        "development_test_reaccessed": False,
        "external_final_test_accessed": False,
        "external_final_reserved_for_frozen_fedfold_comparison": True,
        "final_test_lock_state": "unlocked",
        "test_accessed": False,
    }
    candidate_path = eval_root / "client0_soft_tm_frozen_candidate.json"
    prepare_run_manifest(
        candidate_path,
        payload,
        artifact_exists=False,
        context="frozen client0 soft-TM candidate",
    )
    private_path = (
        ablation_root(run_root) / "client0_soft_tm_frozen_candidate.json"
    )
    prepare_run_manifest(
        private_path,
        payload,
        artifact_exists=False,
        context="private frozen client0 soft-TM candidate",
    )
    update_status(
        run_root,
        freeze_client0_soft_tm_candidate="complete",
        client0_candidate=SOFT_TM_EXPERIMENT_SLUG,
        client0_candidate_scale=SOFT_TM_SHRINK_SCALE,
        client0_candidate_ready_for_replication=True,
        blocks_other_clients=False,
        blocks_fedlora=True,
        external_final_reserved=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def run_strong_preservation_development(
    run_root: Path,
    eval_root: Path,
    gpu_id: str,
) -> dict:
    """Evaluate confirmed weight-1.5 seed-42 model once on development."""
    confirmation_path = (
        eval_root
        / "t4_baseline_preservation_w1p5_seed_confirmation.json"
    )
    if not confirmation_path.exists():
        raise FileNotFoundError(confirmation_path)
    confirmation = __import__("json").loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    if confirmation.get("seed_confirmation_pass") is not True:
        raise RuntimeError("Weight-1.5 seed confirmation did not pass")

    seed = 42
    epoch = VALIDATION_EPOCHS
    TARGET_SPECS[STRONG_PRESERVATION_EXPERIMENT_SLUG] = get_target(
        TARGET_SLUG
    )
    private = private_root(run_root)
    split_root = private / "splits"
    out = full_train_root(
        run_root, STRONG_PRESERVATION_EXPERIMENT_SLUG, seed
    )
    epoch_len = len(read_labels(split_root / "train_labels.txt"))
    checkpoint = checkpoint_for_epochs(
        out / "training", epoch, epoch_len
    )
    if not (checkpoint.exists() or checkpoint.is_dir()):
        raise FileNotFoundError(checkpoint)

    candidate = out / "development" / f"epoch_{epoch}" / "scale_1p0"
    candidate.mkdir(parents=True, exist_ok=True)
    model = candidate / "model_scale_1.0.pt"
    parent = (
        run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    )
    ensure_lora_export(
        checkpoint=checkpoint,
        model=model,
        parent=parent,
        target=get_target(TARGET_SLUG).target,
        scale=1.0,
    )

    labels = split_root / "development_test_labels.txt"
    assets = split_root / "assets_development_test"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    expected_labels, expected_clusters = frozen_split_identity(
        split_root, "development_test", difficulty
    )
    development_manifest = {
        "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "target": get_target(TARGET_SLUG).target,
        "seed": seed,
        "epoch": epoch,
        "scale": 1.0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "parent_sha256": sha256_file(parent),
        "development_labels_sha256": labels_sha256(expected_labels),
        "development_clusters": sorted(expected_clusters),
        "seed_confirmation_sha256": sha256_file(confirmation_path),
        "evaluation_policy": "single_access_no_checkpoint_selection",
        "external_final_test_accessed": False,
    }
    prepare_run_manifest(
        candidate / "evaluation_run.json",
        development_manifest,
        artifact_exists=(candidate / "predictions").exists(),
        context=(
            "development T4_BASELINE_PRESERVE_W1P5 "
            "seed=42 epoch=5"
        ),
    )

    expected = len(expected_labels)
    prediction_dir = candidate / "predictions"
    existing = (
        list(prediction_dir.glob("*_unrelaxed.pdb"))
        if prediction_dir.exists()
        else []
    )
    if len(existing) != expected:
        if existing:
            raise RuntimeError(
                f"Partial development predictions at {candidate}: "
                f"{len(existing)}/{expected}"
            )
        run_cmd(
            [
                PYTHON,
                str(REPO / "run_pretrained_openfold.py"),
                str(assets / "solo_fasta_dir"),
                str(assets / "mmcif_files"),
                "--use_precomputed_alignments",
                str(assets / "solo_alignment_dir"),
                "--use_single_seq_mode",
                "--output_dir",
                str(candidate),
                "--model_device",
                "cuda:0",
                "--skip_relaxation",
                "--config_preset",
                "seq_model_esm1b_ptm",
                "--openfold_checkpoint_path",
                str(model),
                "--checkpoint_weights_source",
                "auto",
                "--data_random_seed",
                str(seed),
                "--precision",
                "fp32",
            ],
            env={
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "PYTHONPATH": str(REPO),
            },
        )

    metrics_dir = candidate / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    if not (metrics_dir / "tm_score.csv").exists():
        run_cmd([
            PYTHON,
            str(REPO / "scripts" / "tmscore_from_pdb.py"),
            str(prediction_dir),
            "--native-dir",
            str(native),
            "--tm-exec",
            str(REPO / "tmscore" / "TMscore"),
            "--out-csv",
            str(metrics_dir / "tm_score.csv"),
            "--selected-csv",
            str(metrics_dir / "low_tm.csv"),
            "--missing-log",
            str(metrics_dir / "tm_missing.txt"),
        ])
    if not (metrics_dir / "lddt_ca.csv").exists():
        run_cmd([
            PYTHON,
            str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
            str(prediction_dir),
            "--native-dir",
            str(native),
            "--out-csv",
            str(metrics_dir / "lddt_ca.csv"),
            "--missing-log",
            str(metrics_dir / "lddt_missing.txt"),
        ])

    paired = candidate / "paired_deltas.csv"
    run_cmd([
        PYTHON,
        str(REPO / "scripts" / "evaluate_hardcase_metrics.py"),
        "--mode",
        "pair",
        "--labels",
        str(labels),
        "--difficulty-csv",
        str(difficulty),
        "--baseline-tm-csv",
        str(difficulty),
        "--model-tm-csv",
        str(metrics_dir / "tm_score.csv"),
        "--baseline-lddt-csv",
        str(difficulty),
        "--model-lddt-csv",
        str(metrics_dir / "lddt_ca.csv"),
        "--client-id",
        "client_0",
        "--out",
        str(paired),
    ])
    rows = list(csv.DictReader(paired.open()))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])

    hard = [row for row in rows if row["difficulty"] == "hard"]
    summary = summarize_group(hard, "hard_")
    summary.update(summarize_group(rows, "all_"))
    development_gate = promotion_gate_v2_from_rows(
        rows,
        expected_labels=expected_labels,
        expected_clusters=expected_clusters,
    )
    development_pass = bool(development_gate.get("promotion_pass"))
    write_json(candidate / "development_gate_v2.json", development_gate)
    payload = {
        "stage": "evaluate-t4-baseline-preservation-w1p5-development",
        "status": "complete" if development_pass else "failed",
        "development_pass": development_pass,
        "experiment_slug": STRONG_PRESERVATION_EXPERIMENT_SLUG,
        "target_slug": TARGET_SLUG,
        "seed": seed,
        "epoch": epoch,
        "scale": 1.0,
        "paired_csv": str(paired),
        "development_gate_v2": development_gate,
        "blocks_other_clients": not development_pass,
        "blocks_fedlora": True,
        "development_test_accessed": True,
        "external_final_test_accessed": False,
        "test_accessed": False,
        **summary,
    }
    summary_name = (
        "t4_baseline_preservation_w1p5_development_summary.json"
    )
    write_json(eval_root / summary_name, payload)
    write_json(ablation_root(run_root) / summary_name, payload)
    update_status(
        run_root,
        evaluate_t4_baseline_preservation_w1p5_development=(
            "complete" if development_pass else "failed"
        ),
        t4_baseline_preservation_w1p5_development_pass=development_pass,
        blocks_other_clients=not development_pass,
        blocks_fedlora=True,
        development_test_accessed=True,
        external_final_test_accessed=False,
        test_accessed=False,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=(
            "capacity",
            "validate",
            "shrinkage",
            "selective",
            "anchor-balanced",
            "preserve",
            "confirm-preserve",
            "stronger-preserve",
            "confirm-stronger",
            "evaluate-stronger-dev",
            "hard-correct",
            "soft-tm",
            "preregister-soft-tm-seeds",
            "confirm-soft-tm",
            "shrink-soft-tm",
            "confirm-shrunk-soft-tm",
            "freeze-soft-tm-candidate",
        ),
        default="capacity",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage == "validate":
        payload = run_frozen_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "shrinkage":
        payload = run_shrinkage_validation(
            args.run_root,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "selective":
        payload = run_selective_clamp_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "anchor-balanced":
        payload = run_anchor_balanced_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "preserve":
        payload = run_baseline_preservation_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "confirm-preserve":
        payload = run_preservation_seed_confirmation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "stronger-preserve":
        payload = run_stronger_preservation_seed43(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "confirm-stronger":
        payload = run_strong_preservation_seed_confirmation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "evaluate-stronger-dev":
        payload = run_strong_preservation_development(
            args.run_root,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "hard-correct":
        payload = run_hard_correction_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "soft-tm":
        payload = run_soft_tm_validation(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    elif args.stage == "preregister-soft-tm-seeds":
        payload = preregister_soft_tm_seed_confirmation(
            args.run_root,
            args.eval_root,
        )
    elif args.stage == "confirm-soft-tm":
        payload = confirm_soft_tm_seeds(
            args.run_root,
            args.eval_root,
        )
    elif args.stage == "shrink-soft-tm":
        payload = run_soft_tm_shrink_seed43(
            args.run_root,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "confirm-shrunk-soft-tm":
        payload = confirm_soft_tm_shrink(
            args.run_root,
            args.eval_root,
            args.gpu_id,
        )
    elif args.stage == "freeze-soft-tm-candidate":
        payload = freeze_soft_tm_candidate(
            args.run_root,
            args.eval_root,
        )
    else:
        payload = run(
            args.run_root,
            args.source_run,
            args.eval_root,
            args.gpu_id,
            args.seed,
        )
    print(__import__("json").dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
