#!/usr/bin/env python3
"""Orchestrate client0 LoRA target ablation stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_client0_overfit_subset import build_overfit_subset
from scripts.build_fed_test_set import labels_sha256
from scripts.evaluate_hardcase_metrics import summarize_group
from scripts.evaluate_target_gate import (
    capacity_pass_from_rows,
    capacity_pass_v2_from_rows,
    capacity_status_for_target,
    choose_unique_target,
    classify_capacity_failure,
    promotion_gate_from_rows,
    promotion_gate_v2_from_rows,
    resolve_validation_branch,
    seed_confirmation_pass,
    select_seed_confirmation_targets,
    should_enter_full_train_validation,
    should_run_capacity_fallback,
    should_schedule_overfit_targets,
)
from scripts.lora_target_registry import (
    CAPACITY_CHECKPOINTS,
    FALLBACK_TARGETS,
    FALLBACK_VALIDATION_TARGETS,
    OVERFIT_EPOCH_LEN,
    OVERFIT_MAX_EPOCHS,
    OVERFIT_WARMUP_STEPS,
    PRIMARY_OVERFIT_TARGETS,
    PRIMARY_VALIDATION_TARGETS,
    TARGET_SPECS,
    get_target,
)
from scripts.verify_lora_target_matrix import verify_one


DEFAULT_RUN = REPO / "outputs" / "fed_lora_hardcase_fed_v1"
DEFAULT_SOURCE = REPO / "outputs" / "fed_lora_fp32"
DEFAULT_EVAL = DEFAULT_RUN / "evaluation" / "client0_target_ablation_v1"
NAMESPACE = "target_ablation_v1"
POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE = "state_dict"
PYTHON = os.environ.get(
    "PYTHON_BIN",
    str(Path.home() / "miniconda3" / "envs" / "fedfold" / "bin" / "python"),
)
if not Path(PYTHON).exists():
    PYTHON = sys.executable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot fingerprint empty artifact directory: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_labels(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_manifest_compatible(path: Path, expected: dict, context: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{context}: artifact exists but manifest is missing: {path}")
    actual = load_json(path)
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"{context}: existing artifact manifest/config mismatch; "
            f"use a new namespace: {mismatches}"
        )


def prepare_run_manifest(path: Path, expected: dict, *, artifact_exists: bool, context: str) -> None:
    if path.exists():
        assert_manifest_compatible(path, expected, context)
    elif artifact_exists:
        raise RuntimeError(f"{context}: refusing to reuse artifact without {path.name}")
    else:
        write_json(path, expected)


def difficulty_clusters_for_labels(difficulty_csv: Path, labels: Sequence[str]) -> List[str]:
    with difficulty_csv.open(newline="", encoding="utf-8") as handle:
        mapping = {
            row["label"].strip().upper(): row.get("cluster_id", "").strip()
            for row in csv.DictReader(handle)
        }
    clusters = [mapping.get(label.upper(), "") for label in labels]
    if any(not cluster for cluster in clusters):
        missing = [label for label, cluster in zip(labels, clusters) if not cluster]
        raise RuntimeError(f"Missing cluster IDs for labels: {missing}")
    return clusters


def frozen_split_identity(split_root: Path, role: str, difficulty_csv: Path) -> tuple[List[str], List[str]]:
    labels_path = split_root / f"{role}_labels.txt"
    labels = read_labels(labels_path)
    manifest = load_json(split_root / "split_manifest.json")
    expected_count = manifest.get(f"{role}_label_count")
    expected_hash = manifest.get(f"{role}_labels_sha256")
    actual_hash = labels_sha256(labels)
    if expected_count != len(labels) or expected_hash != actual_hash:
        raise RuntimeError(
            f"Frozen {role} split mismatch: count {len(labels)} vs {expected_count}, "
            f"hash {actual_hash} vs {expected_hash}"
        )
    return labels, difficulty_clusters_for_labels(difficulty_csv, labels)


def private_root(run_root: Path) -> Path:
    return run_root / "clients" / "client_0" / "private"


def ablation_root(run_root: Path) -> Path:
    return private_root(run_root) / NAMESPACE


def status_path(run_root: Path) -> Path:
    return ablation_root(run_root) / "status.json"


def update_status(run_root: Path, **kwargs) -> dict:
    path = status_path(run_root)
    data = load_json(path) if path.exists() else {}
    data.update(kwargs)
    write_json(path, data)
    return data


def run_cmd(cmd: Sequence[str], env: Optional[dict] = None, cwd: Optional[Path] = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(list(cmd), check=True, env=merged, cwd=str(cwd or REPO))


def export_manifest_path(model: Path) -> Path:
    return model.with_suffix(".export_manifest.json")


def lora_export_is_compatible(
    *, checkpoint: Path, model: Path, parent: Path, target: str, scale: float
) -> tuple[bool, List[str]]:
    manifest_path = export_manifest_path(model)
    errors: List[str] = []
    if not model.exists():
        errors.append("model_missing")
    if not manifest_path.exists():
        errors.append("manifest_missing")
        return False, errors
    manifest = load_json(manifest_path)
    report = manifest.get("report") or {}
    checks = {
        "output": str(model),
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_scale": float(scale),
        "lora_target": target,
        "base_checkpoint_sha256": sha256_path(parent),
        "adapter_checkpoint_sha256": sha256_path(checkpoint),
        "scale_semantics": "unit_raw_delta",
    }
    for key, expected in checks.items():
        actual = manifest.get(key)
        if key == "output":
            try:
                if Path(str(actual)).resolve() != model.resolve():
                    errors.append(f"{key}_mismatch")
            except (TypeError, ValueError):
                errors.append(f"{key}_mismatch")
        elif actual != expected:
            errors.append(f"{key}_mismatch")
    if not manifest.get("lora_schema_fingerprint"):
        errors.append("schema_fingerprint_missing")
    if not manifest.get("adapter_dtypes"):
        errors.append("adapter_dtypes_missing")
    if not manifest.get("adapter_shapes") or not manifest.get("adapter_key_count"):
        errors.append("adapter_schema_details_missing")
    expected_spec = next(
        (spec for spec in TARGET_SPECS.values() if spec.target == target), None
    )
    if expected_spec and manifest.get("adapter_key_count") != 2 * expected_spec.expected_modules:
        errors.append("adapter_key_count_registry_mismatch")
    if model.exists():
        output_sha = sha256_file(model)
        if manifest.get("output_sha256") != output_sha:
            errors.append("output_sha256_mismatch")
    if report.get("lora_schema_fingerprint") != manifest.get("lora_schema_fingerprint"):
        errors.append("report_schema_fingerprint_mismatch")
    return not errors, errors


def ensure_lora_export(
    *, checkpoint: Path, model: Path, parent: Path, target: str, scale: float
) -> None:
    compatible, errors = lora_export_is_compatible(
        checkpoint=checkpoint, model=model, parent=parent, target=target, scale=scale
    )
    if compatible:
        return
    manifest_path = export_manifest_path(model)
    if model.exists() or manifest_path.exists():
        raise RuntimeError(
            f"Existing export is incompatible for {model}: {errors}; "
            "refusing to overwrite it—use a new namespace"
        )
    cmd = [
        PYTHON,
        str(REPO / "scripts" / "export_lora_checkpoint.py"),
        "--input", str(checkpoint),
        "--output", str(model),
        "--base-checkpoint", str(parent),
        "--base-weights-source", "auto",
        "--adapter-weights-source", "model",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--lora-scale", str(scale),
        "--config-preset", "seq_model_esm1b_ptm",
    ]
    run_cmd(cmd)
    compatible, errors = lora_export_is_compatible(
        checkpoint=checkpoint, model=model, parent=parent, target=target, scale=scale
    )
    if not compatible:
        raise RuntimeError(f"Export manifest verification failed for {model}: {errors}")


def stage_verify(run_root: Path, eval_root: Path, slugs: Optional[Sequence[str]] = None) -> dict:
    targets = list(slugs or TARGET_SPECS.keys())
    results = []
    for slug in targets:
        print(f"[verify] {slug}")
        results.append(verify_one(slug))
    out = {
        "stage": "verify",
        "rank": 4,
        "alpha": 8.0,
        "targets": results,
        "ok": all(r["ok"] for r in results),
    }
    write_json(ablation_root(run_root) / "verify_targets.json", out)
    write_json(eval_root / "verify_targets.json", out)
    update_status(run_root, verify="complete", verify_sha256=sha256_file(
        ablation_root(run_root) / "verify_targets.json"
    ))
    return out


def stage_build_overfit(run_root: Path, source_run: Path, eval_root: Path) -> dict:
    manifest = build_overfit_subset(run_root=run_root, source_run=source_run, link=True)
    write_json(eval_root / "overfit8_manifest.json", manifest)
    update_status(
        run_root,
        build_overfit="complete",
        overfit8_labels_sha256=manifest["labels_sha256"],
    )
    return manifest


def overfit_run_root(run_root: Path, slug: str, seed: int = 42) -> Path:
    return ablation_root(run_root) / slug / "overfit8" / f"seed_{seed}"


def full_train_root(run_root: Path, slug: str, seed: int = 42, arm: str = "uniform") -> Path:
    return ablation_root(run_root) / slug / arm / f"seed_{seed}"


def checkpoint_for_epochs(train_dir: Path, epochs: int, epoch_len: int) -> Path:
    step = epochs * epoch_len
    epoch_index = epochs - 1
    name = f"{epoch_index}-{step}.ckpt"
    primary = train_dir / "checkpoints" / name
    # With a local Lightning logger, ModelCheckpoint defaults below the
    # logger's log directory. Preserve the historical path while resolving
    # diagnostic runs that enable CSV metrics.
    csv_logger_path = train_dir / "csv_logs" / "checkpoints" / name
    if not primary.exists() and csv_logger_path.exists():
        return csv_logger_path
    return primary


def train_overfit_segment(
    *,
    run_root: Path,
    source_run: Path,
    slug: str,
    seed: int,
    max_epochs: int,
    smoke: bool = False,
    gpu_id: str = "0",
) -> Path:
    spec = get_target(slug)
    private = private_root(run_root)
    overfit = ablation_root(run_root) / "overfit8"
    out = overfit_run_root(run_root, slug, seed)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = overfit / "overfit8_labels.txt"
    if smoke:
        smoke_labels = out / "smoke_labels.txt"
        labs = read_labels(labels)[:2]
        smoke_labels.write_text("".join(f"{x}\n" for x in labs))
        labels = smoke_labels
        epoch_len = len(labs)
        max_epochs = 1
    else:
        epoch_len = OVERFIT_EPOCH_LEN

    if not parent.exists():
        raise FileNotFoundError(parent)
    final_ckpt = checkpoint_for_epochs(train_dir, max_epochs, epoch_len)
    run_manifest = {
        "client_id": "client_0",
        "mode": "overfit8",
        "target_slug": slug,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "seed": seed,
        "rank": 4,
        "alpha": 8,
        "dropout": 0,
        "learning_rate": 1e-4,
        "lr_warmup_steps": OVERFIT_WARMUP_STEPS,
        "max_epochs": max_epochs if smoke else OVERFIT_MAX_EPOCHS,
        "train_epoch_len": epoch_len,
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "labels_sha256": sha256_file(labels),
        "smoke": smoke,
        "primary_eval_scale": 1.0,
        "aggregation": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "local_run.json").exists():
        verify_one(slug)
    prepare_run_manifest(
        out / "local_run.json",
        run_manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context=f"{slug} overfit seed={seed}",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] {slug} overfit manifest verified: {final_ckpt}")
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
        "--train_filter_path",
        str(labels),
        "--val_data_dir",
        str(overfit / "assets" / "mmcif_files"),
        "--val_alignment_dir",
        str(overfit / "assets" / "solo_alignment_dir"),
        "--use_single_seq_mode",
        "True",
        "--config_preset",
        "seq_model_esm1b_ptm",
        "--experiment_config_json",
        str(REPO / "seq_model_esm1b_ptm_finetune_override.json"),
        "--resume_from_ckpt",
        str(parent),
        "--resume_model_weights_only",
        "True",
        "--init_weights_source",
        "auto",
        "--template_release_dates_cache_path",
        str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path",
        str(overfit / "overfit8_chain_data_cache.json"),
        "--lora_rank",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0",
        "--lora_target",
        spec.target,
        "--learning_rate",
        "1e-4",
        "--lr_warmup_steps",
        str(OVERFIT_WARMUP_STEPS),
        "--accumulate_grad_batches",
        "1",
        "--train_epoch_len",
        str(epoch_len),
        "--max_epochs",
        str(max_epochs),
        "--checkpoint_every_epoch",
        "--sampling_mode",
        "uniform",
        "--precision",
        "32",
        "--gpus",
        "1",
        "--seed",
        str(seed),
        "--deepspeed_config_path",
        str(REPO / "deepspeed_config_fp32.json"),
    ]
    # Continue from the latest completed registered segment.
    if not smoke:
        for prev_epochs in sorted(
            (e for e in (5, 10, 15, 25) if e < max_epochs), reverse=True
        ):
            prev_ckpt = checkpoint_for_epochs(train_dir, prev_epochs, epoch_len)
            if prev_ckpt.exists() or prev_ckpt.is_dir():
                idx = cmd.index("--resume_from_ckpt")
                cmd[idx + 1] = str(prev_ckpt)
                widx = cmd.index("--resume_model_weights_only")
                cmd[widx + 1] = "False"
                break

    env = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)}
    log_path = out / f"train_to_epoch_{max_epochs}.log"
    print(f"[overfit-train] {slug} -> epoch {max_epochs}")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, env={**os.environ, **env}, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    return final_ckpt


def export_and_score_overfit(
    *,
    run_root: Path,
    slug: str,
    seed: int,
    epochs: int,
    epoch_len: int,
    scale: float = 1.0,
    gpu_id: str = "0",
    labels_path: Optional[Path] = None,
) -> dict:
    private = private_root(run_root)
    overfit = ablation_root(run_root) / "overfit8"
    out_root = overfit_run_root(run_root, slug, seed)
    ckpt = checkpoint_for_epochs(out_root / "training", epochs, epoch_len)
    step = epochs * epoch_len
    cand = out_root / "eval" / f"step_{step}" / f"scale_{str(scale).replace('.', 'p')}"
    cand.mkdir(parents=True, exist_ok=True)
    model = cand / f"model_scale_{scale}.pt"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = labels_path or (overfit / "overfit8_labels.txt")

    ensure_lora_export(
        checkpoint=ckpt,
        model=model,
        parent=parent,
        target=get_target(slug).target,
        scale=scale,
    )

    assets = overfit / "assets"
    pred_dir = cand / "predictions"
    expected = len(read_labels(labels))
    existing = list(pred_dir.glob("*_unrelaxed.pdb")) if pred_dir.exists() else []
    if len(existing) != expected:
        if existing:
            raise RuntimeError(f"Partial predictions at {cand}: {len(existing)}/{expected}")
        # Build a temporary fasta dir filtered to labels if smoke uses subset.
        fasta_dir = assets / "solo_fasta_dir"
        aln_dir = assets / "solo_alignment_dir"
        cif_dir = assets / "mmcif_files"
        env = {"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)}
        run_cmd(
            [
                PYTHON,
                str(REPO / "run_pretrained_openfold.py"),
                str(fasta_dir),
                str(cif_dir),
                "--use_precomputed_alignments",
                str(aln_dir),
                "--use_single_seq_mode",
                "--output_dir",
                str(cand),
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
            env=env,
        )

    metrics = cand / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    if not (metrics / "tm_score.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "tmscore_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--tm-exec",
                str(REPO / "tmscore" / "TMscore"),
                "--out-csv",
                str(metrics / "tm_score.csv"),
                "--selected-csv",
                str(metrics / "low_tm.csv"),
                "--missing-log",
                str(metrics / "tm_missing.txt"),
            ]
        )
    if not (metrics / "lddt_ca.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--out-csv",
                str(metrics / "lddt_ca.csv"),
                "--missing-log",
                str(metrics / "lddt_missing.txt"),
            ]
        )
    paired = cand / "paired_deltas.csv"
    run_cmd(
        [
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
            str(metrics / "tm_score.csv"),
            "--baseline-lddt-csv",
            str(difficulty),
            "--model-lddt-csv",
            str(metrics / "lddt_ca.csv"),
            "--client-id",
            "client_0",
            "--out",
            str(paired),
        ]
    )
    rows = list(csv.DictReader(paired.open()))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    cap = capacity_pass_from_rows(rows)
    overfit_manifest = load_json(overfit / "overfit8_manifest.json")
    expected_labels = overfit_manifest.get("labels") or read_labels(labels)
    expected_clusters = overfit_manifest.get("clusters")
    cap_v2 = capacity_pass_v2_from_rows(
        rows,
        expected_labels=expected_labels,
        expected_clusters=expected_clusters,
    )
    write_json(cand / "capacity.json", {"step": step, "epochs": epochs, **cap})
    write_json(cand / "capacity_v2.json", {"step": step, "epochs": epochs, **cap_v2})
    return {
        "step": step,
        "epochs": epochs,
        "paired_csv": str(paired),
        **cap,
        "capacity_v2": cap_v2,
    }


def stage_overfit(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
    seed: int = 42,
) -> dict:
    statuses: Dict[str, str] = {}
    details: Dict[str, dict] = {}
    scheduled = should_schedule_overfit_targets({})
    assert scheduled == list(PRIMARY_OVERFIT_TARGETS), (
        "overfit must schedule T0/T1/T3/T4 regardless of prior outcomes"
    )

    for slug in PRIMARY_OVERFIT_TARGETS:
        step_results = {}
        # Segmented 5-epoch chunks -> 40/80/120/200 steps
        for epochs in (5, 10, 15, 25):
            if smoke:
                epochs = 1
            train_overfit_segment(
                run_root=run_root,
                source_run=source_run,
                slug=slug,
                seed=seed,
                max_epochs=epochs,
                smoke=smoke,
                gpu_id=gpu_id,
            )
            epoch_len = 2 if smoke else OVERFIT_EPOCH_LEN
            labels_path = None
            if smoke:
                labels_path = overfit_run_root(run_root, slug, seed) / "smoke_labels.txt"
            result = export_and_score_overfit(
                run_root=run_root,
                slug=slug,
                seed=seed,
                epochs=epochs,
                epoch_len=epoch_len,
                gpu_id=gpu_id,
                labels_path=labels_path,
            )
            step_results[result["step"]] = result
            if (result.get("capacity_v2") or {}).get("pass") and not smoke:
                break
            if smoke:
                break
        robust_step_results = {
            step: result.get("capacity_v2") or {}
            for step, result in step_results.items()
        }
        status = capacity_status_for_target(
            robust_step_results, max_steps=200 if not smoke else 2
        )
        # For smoke, map to non-terminal diagnostic only.
        if smoke:
            statuses[slug] = "smoke_complete"
        else:
            statuses[slug] = status["status"]
        details[slug] = {"status": statuses[slug], "steps": step_results, **status}
        write_json(overfit_run_root(run_root, slug, seed) / "capacity_status.json", details[slug])

    # Failures must not skip later targets; assert all present.
    for slug in PRIMARY_OVERFIT_TARGETS:
        if slug not in statuses:
            raise RuntimeError(f"overfit missing required target {slug}")

    payload = {
        "stage": "overfit",
        "statuses": statuses,
        "details": details,
        "enter_full_train": should_enter_full_train_validation(statuses) if not smoke else False,
        "run_fallback": should_run_capacity_fallback(statuses) if not smoke else False,
        "smoke": smoke,
    }
    write_json(ablation_root(run_root) / "overfit_summary.json", payload)
    write_json(eval_root / "overfit_summary.json", payload)
    update_status(run_root, overfit="complete", overfit_statuses=statuses)
    return payload


def diagnose_checkpoint_norms(ckpt: Path, target: str, rank: int = 4, alpha: float = 8.0) -> dict:
    from scripts.export_lora_checkpoint import extract_checkpoint_weights, load_checkpoint

    checkpoint = load_checkpoint(ckpt)
    weights = extract_checkpoint_weights(checkpoint, "model")
    a_norms = []
    b_norms = []
    dw_norms = []
    for key, tensor in weights.items():
        if key.endswith(".lora_A"):
            a_norms.append(float(tensor.detach().float().norm().item()))
        elif key.endswith(".lora_B"):
            b_norms.append(float(tensor.detach().float().norm().item()))
        elif key.endswith(".lora_A") or key.endswith(".lora_B"):
            pass
    # Approximate effective ΔW norms from paired A/B when shapes allow.
    prefixes = {
        key[: -len(".lora_A")]
        for key in weights
        if key.endswith(".lora_A")
    }
    for prefix in prefixes:
        a = weights.get(prefix + ".lora_A")
        b = weights.get(prefix + ".lora_B")
        if a is None or b is None:
            continue
        # ΔW ~= (alpha/rank) * B @ A
        scale = float(alpha) / float(rank)
        delta = scale * (b.detach().float() @ a.detach().float())
        dw_norms.append(float(delta.norm().item()))
    lora_keys = [k for k in weights if ".lora_" in k]
    return {
        "checkpoint": str(ckpt),
        "target": target,
        "lora_key_count": len(lora_keys),
        "mean_A_norm": sum(a_norms) / len(a_norms) if a_norms else 0.0,
        "mean_B_norm": sum(b_norms) / len(b_norms) if b_norms else 0.0,
        "mean_delta_W_norm": sum(dw_norms) / len(dw_norms) if dw_norms else 0.0,
        "max_A_norm": max(a_norms) if a_norms else 0.0,
        "max_B_norm": max(b_norms) if b_norms else 0.0,
        "max_delta_W_norm": max(dw_norms) if dw_norms else 0.0,
        "n_A": len(a_norms),
        "n_B": len(b_norms),
        "changed_target_keys": len(dw_norms),
    }


def _infer_loss_decreased(run_root: Path) -> bool:
    """Best-effort: look for decreasing train loss lines in overfit logs."""
    import re

    pattern = re.compile(r"train[_\s-]*loss[\"'\\s:=]+([0-9.eE+-]+)", re.I)
    losses = []
    root = ablation_root(run_root)
    for log in root.glob("*/overfit8/seed_42/train*.log"):
        text = log.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            try:
                losses.append(float(match.group(1)))
            except ValueError:
                continue
    if len(losses) < 2:
        # Unknown; do not claim loss failed to decrease.
        return True
    return losses[-1] < losses[0]


def stage_diagnose_capacity(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
) -> dict:
    status = load_json(status_path(run_root)) if status_path(run_root).exists() else {}
    statuses = status.get("overfit_statuses_v2") or status.get("overfit_statuses", {})
    diagnostics = {"targets": {}, "fallback": None, "failure_classification": None}

    for slug in PRIMARY_OVERFIT_TARGETS:
        out = overfit_run_root(run_root, slug)
        cap_path = out / "capacity_status.json"
        if not cap_path.exists():
            continue
        cap = load_json(cap_path)
        ckpt = checkpoint_for_epochs(out / "training", OVERFIT_MAX_EPOCHS, OVERFIT_EPOCH_LEN)
        if not (ckpt.exists() or ckpt.is_dir()):
            # pick latest available
            for epochs in (25, 15, 10, 5, 1):
                cand = checkpoint_for_epochs(out / "training", epochs, OVERFIT_EPOCH_LEN if not smoke else 2)
                if cand.exists() or cand.is_dir():
                    ckpt = cand
                    break
        norms = {}
        if ckpt.exists() or ckpt.is_dir():
            norms = diagnose_checkpoint_norms(ckpt, get_target(slug).target)
        diagnostics["targets"][slug] = {
            "capacity_status": cap.get("status"),
            "first_pass_step": cap.get("first_pass_step"),
            "norms": norms,
        }

    if should_run_capacity_fallback(statuses) and not smoke:
        fallback_results = {}
        for slug in FALLBACK_TARGETS:
            train_overfit_segment(
                run_root=run_root,
                source_run=source_run,
                slug=slug,
                seed=42,
                max_epochs=OVERFIT_MAX_EPOCHS,
                smoke=False,
                gpu_id=gpu_id,
            )
            # Evaluate final step only for fallback
            result = export_and_score_overfit(
                run_root=run_root,
                slug=slug,
                seed=42,
                epochs=OVERFIT_MAX_EPOCHS,
                epoch_len=OVERFIT_EPOCH_LEN,
                gpu_id=gpu_id,
            )
            fallback_results[slug] = result
        from scripts.positive_control_structure_module import (
            positive_control_status,
            verify_positive_control_setup,
        )

        positive_setup = verify_positive_control_setup()
        positive_control = positive_control_status(setup=positive_setup, executed=False)
        # Prefer capacity_v2 for routing decisions when available.
        t5_v2 = {
            slug: bool((r.get("capacity_v2") or {}).get("pass"))
            for slug, r in fallback_results.items()
        }
        # Recompute v2 from paired CSVs if present.
        for slug, r in fallback_results.items():
            paired_path = Path(r.get("paired_csv") or "")
            if paired_path.exists():
                rows = list(csv.DictReader(paired_path.open()))
                for row in rows:
                    row["delta_tm"] = float(row["delta_tm"])
                    row["tm_baseline"] = float(row["tm_baseline"])
                    row["tm_model"] = float(row["tm_model"])
                    if row.get("delta_lddt_ca") not in ("", None):
                        row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
                overfit_manifest = load_json(
                    ablation_root(run_root) / "overfit8" / "overfit8_manifest.json"
                )
                v2 = capacity_pass_v2_from_rows(
                    rows,
                    expected_labels=overfit_manifest.get("labels"),
                    expected_clusters=overfit_manifest.get("clusters"),
                )
                fallback_results[slug]["capacity_v2"] = v2
                t5_v2[slug] = bool(v2.get("pass"))
                write_json(paired_path.parent / "capacity_v2.json", v2)
        t5_any_v2 = any(t5_v2.values())
        t5_any_v1 = any(r.get("pass") for r in fallback_results.values())
        loss_decreased = _infer_loss_decreased(run_root)
        tm_unchanged = all(
            (diagnostics["targets"].get(s, {}).get("capacity_status") == "capacity_failed")
            for s in PRIMARY_OVERFIT_TARGETS
        )
        classification = classify_capacity_failure(
            t5_any_pass=t5_any_v2,
            positive_control_pass=bool(positive_control.get("pass")),
            positive_control_executed=bool(positive_control.get("executed")),
            loss_decreased=loss_decreased,
            tm_unchanged=tm_unchanged,
        )
        diagnostics["fallback"] = {
            "t5_results": fallback_results,
            "t5_capacity_v1_any_pass": t5_any_v1,
            "t5_capacity_v2": t5_v2,
            "t5_capacity_v2_any_pass": t5_any_v2,
            "positive_control": positive_control,
        }
        diagnostics["failure_classification"] = classification
        diagnostics["blocks_other_clients"] = True
        diagnostics["blocks_fedlora"] = True
        diagnostics["test_accessed"] = False
    elif should_enter_full_train_validation(statuses):
        diagnostics["failure_classification"] = {
            "failure_type": None,
            "action": "proceed_to_full_train_validation_matrix",
        }
    else:
        diagnostics["failure_classification"] = {
            "failure_type": "overfit_incomplete_or_smoke",
            "action": "complete_overfit_before_validation",
        }

    write_json(ablation_root(run_root) / "capacity_diagnostics.json", diagnostics)
    write_json(eval_root / "capacity_diagnostics.json", diagnostics)
    update_status(run_root, diagnose_capacity="complete")
    return diagnostics


def train_full_target(
    *,
    run_root: Path,
    source_run: Path,
    slug: str,
    seed: int,
    epochs: int = 5,
    smoke: bool = False,
    gpu_id: str = "0",
) -> Path:
    spec = get_target(slug)
    private = private_root(run_root)
    split_root = private / "splits"
    out = full_train_root(run_root, slug, seed)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = split_root / "train_labels.txt"
    epoch_len = len(read_labels(labels))
    if smoke:
        smoke_labels = out / "smoke_train_labels.txt"
        labs = read_labels(labels)[:2]
        smoke_labels.write_text("".join(f"{x}\n" for x in labs))
        labels = smoke_labels
        epoch_len = len(labs)
        epochs = 1

    if not parent.exists():
        raise FileNotFoundError(parent)
    final_ckpt = checkpoint_for_epochs(train_dir, epochs, epoch_len)
    run_manifest = {
        "client_id": "client_0",
        "mode": "full_train_validation",
        "target_slug": slug,
        "target": spec.target,
        "expected_modules": spec.expected_modules,
        "expected_trainable": spec.expected_trainable,
        "seed": seed,
        "rank": 4,
        "alpha": 8,
        "dropout": 0,
        "learning_rate": 1e-4,
        "lr_warmup_steps": 20,
        "sampling_mode": "uniform",
        "precision": "32",
        "local_epochs": epochs,
        "train_epoch_len": epoch_len,
        "primary_eval_epoch": epochs,
        "primary_eval_scale": 1.0,
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "split_train_sha256": sha256_file(split_root / "train_labels.txt"),
        "aggregation": False,
        "smoke": smoke,
    }
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "local_run.json").exists():
        verify_one(slug)
    prepare_run_manifest(
        out / "local_run.json",
        run_manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context=f"full-train {slug} seed={seed}",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] full-train manifest verified {slug} seed={seed}")
        return final_ckpt
    data_src = source_run / "clients" / "client_0"
    env = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": str(REPO),
        "LOCAL_OUTPUT_NAMESPACE": NAMESPACE,
        "TARGET_SLUG": slug,
        "LORA_TARGET": spec.target,
        "SEEDS": str(seed),
        "ARMS": "uniform",
        "LOCAL_ONLY_EPOCHS": str(epochs),
        "LOCAL_CLIENTS": "0",
        "RUN_ROOT": str(run_root),
        "SOURCE_RUN": str(source_run),
    }
    # Prefer direct train_openfold for path control with smoke subsets.
    cmd = [
        PYTHON,
        str(REPO / "train_openfold.py"),
        str(data_src / "mmcif_files_finetune"),
        str(data_src / "solo_alignment"),
        str(data_src / "mmcif_files_finetune"),
        str(train_dir),
        "2026-01-01",
        "--train_filter_path",
        str(labels),
        "--val_data_dir",
        str(split_root / "assets_validation" / "mmcif_files"),
        "--val_alignment_dir",
        str(split_root / "assets_validation" / "solo_alignment_dir"),
        "--use_single_seq_mode",
        "True",
        "--config_preset",
        "seq_model_esm1b_ptm",
        "--experiment_config_json",
        str(REPO / "seq_model_esm1b_ptm_finetune_override.json"),
        "--resume_from_ckpt",
        str(parent),
        "--resume_model_weights_only",
        "True",
        "--init_weights_source",
        "auto",
        "--template_release_dates_cache_path",
        str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path",
        str(split_root / "train_chain_data_cache.json"),
        "--lora_rank",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0",
        "--lora_target",
        spec.target,
        "--learning_rate",
        "1e-4",
        "--lr_warmup_steps",
        "20",
        "--accumulate_grad_batches",
        "1",
        "--train_epoch_len",
        str(epoch_len),
        "--max_epochs",
        str(epochs),
        "--checkpoint_every_epoch",
        "--sampling_mode",
        "uniform",
        "--difficulty_csv",
        str(private / "difficulty" / "baseline_difficulty.csv"),
        "--precision",
        "32",
        "--gpus",
        "1",
        "--seed",
        str(seed),
        "--deepspeed_config_path",
        str(REPO / "deepspeed_config_fp32.json"),
    ]
    log_path = out / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            check=True,
            env={**os.environ, **env},
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return final_ckpt


def evaluate_full_validation(
    *,
    run_root: Path,
    slug: str,
    seed: int,
    epoch: int = 5,
    scale: float = 1.0,
    smoke: bool = False,
    gpu_id: str = "0",
) -> dict:
    private = private_root(run_root)
    split_root = private / "splits"
    out = full_train_root(run_root, slug, seed)
    labels = split_root / "validation_labels.txt"
    epoch_len = len(read_labels(split_root / "train_labels.txt"))
    if smoke:
        epoch_len = 2
        epoch = 1
    ckpt = checkpoint_for_epochs(out / "training", epoch, epoch_len)
    if not (ckpt.exists() or ckpt.is_dir()):
        raise FileNotFoundError(ckpt)
    scale_slug = str(scale).replace(".", "p")
    cand = out / "validation" / f"epoch_{epoch}" / f"scale_{scale_slug}"
    cand.mkdir(parents=True, exist_ok=True)
    model = cand / f"model_base_global_adapter_raw_scale_{scale}.pt"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    expected_labels, expected_clusters = frozen_split_identity(
        split_root, "validation", difficulty
    )
    evaluation_manifest = {
        "target_slug": slug,
        "target": get_target(slug).target,
        "seed": seed,
        "epoch": epoch,
        "scale": float(scale),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_path(ckpt),
        "parent_sha256": sha256_file(parent),
        "validation_labels_sha256": labels_sha256(expected_labels),
        "validation_clusters": sorted(expected_clusters),
    }
    prepare_run_manifest(
        cand / "evaluation_run.json",
        evaluation_manifest,
        artifact_exists=model.exists() or (cand / "predictions").exists(),
        context=f"validation {slug} seed={seed} epoch={epoch} scale={scale}",
    )
    ensure_lora_export(
        checkpoint=ckpt,
        model=model,
        parent=parent,
        target=get_target(slug).target,
        scale=scale,
    )
    assets = split_root / "assets_validation"
    expected = len(read_labels(labels))
    pred_dir = cand / "predictions"
    existing = list(pred_dir.glob("*_unrelaxed.pdb")) if pred_dir.exists() else []
    if len(existing) != expected:
        if existing:
            raise RuntimeError(f"Partial validation predictions at {cand}")
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
                str(cand),
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
            env={"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)},
        )
    metrics = cand / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    if not (metrics / "tm_score.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "tmscore_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--tm-exec",
                str(REPO / "tmscore" / "TMscore"),
                "--out-csv",
                str(metrics / "tm_score.csv"),
                "--selected-csv",
                str(metrics / "low_tm.csv"),
                "--missing-log",
                str(metrics / "tm_missing.txt"),
            ]
        )
    if not (metrics / "lddt_ca.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--out-csv",
                str(metrics / "lddt_ca.csv"),
                "--missing-log",
                str(metrics / "lddt_missing.txt"),
            ]
        )
    paired = cand / "paired_deltas.csv"
    run_cmd(
        [
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
            str(metrics / "tm_score.csv"),
            "--baseline-lddt-csv",
            str(difficulty),
            "--model-lddt-csv",
            str(metrics / "lddt_ca.csv"),
            "--client-id",
            "client_0",
            "--out",
            str(paired),
        ]
    )
    rows = list(csv.DictReader(paired.open()))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    gate = promotion_gate_from_rows(rows)
    gate_v2 = promotion_gate_v2_from_rows(
        rows,
        expected_labels=expected_labels,
        expected_clusters=expected_clusters,
    )
    write_json(
        cand / "candidate.json",
        {
            "model_path": str(model),
            "epoch": epoch,
            "scale": scale,
            "target_slug": slug,
            "selection_split": "validation",
            "primary_comparison": True,
            "test_accessed": False,
        },
    )
    write_json(cand / "promotion_gate.json", gate)
    write_json(cand / "promotion_gate_v2.json", gate_v2)
    return {
        "target_slug": slug,
        "seed": seed,
        "epoch": epoch,
        "scale": scale,
        "paired_csv": str(paired),
        **gate,
        "promotion_v2": gate_v2,
        "promotion_pass_v2": gate_v2.get("promotion_pass"),
    }



def stage_audit_9u78(
    run_root: Path,
    eval_root: Path,
    gpu_id: str = "0",
    skip_inference: bool = False,
) -> dict:
    from scripts.audit_overfit_outlier import (
        audit_static_and_mapping,
        build_audit_summary,
        independent_inference_replay,
        metric_rescore,
        write_json as audit_write,
    )

    out_dir = ablation_root(run_root) / "diagnostics" / "9u78_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    static = audit_static_and_mapping(run_root=run_root)
    audit_write(out_dir / "static_integrity.json", static)
    private = private_root(run_root)
    pred = (
        private
        / NAMESPACE
        / "T5A"
        / "overfit8"
        / "seed_42"
        / "eval"
        / "step_200"
        / "scale_1p0"
        / "predictions"
        / "9u78_A_seq_model_esm1b_ptm_unrelaxed.pdb"
    )
    native = private / "difficulty" / "native" / "9u78_A.pdb"
    rescore = metric_rescore(
        pred=pred,
        native=native,
        tm_exec=REPO / "tmscore" / "TMscore",
        out_dir=out_dir,
        repeats=2,
    )
    audit_write(out_dir / "tm_rescore.json", rescore)
    replay = None
    if not skip_inference and pred.exists():
        replay = independent_inference_replay(
            run_root=run_root,
            source_run=DEFAULT_SOURCE,
            out_dir=out_dir / "independent_inference",
            gpu_id=gpu_id,
            python_bin=PYTHON,
        )
        audit_write(out_dir / "independent_inference.json", replay)
    summary = build_audit_summary(static, rescore, replay)
    audit_write(out_dir / "audit_summary.json", summary)
    write_json(eval_root / "audit_9u78_summary.json", summary)
    if summary.get("audit_pass"):
        audit_state = "complete"
    elif summary.get("audit_integrity_pass") and not summary.get("audit_complete"):
        audit_state = "incomplete"
    else:
        audit_state = "failed"
    update_status(
        run_root,
        audit_9u78=audit_state,
        audit_integrity_pass=summary.get("audit_integrity_pass"),
        audit_complete=summary.get("audit_complete"),
        audit_pass=summary.get("audit_pass"),
    )
    return summary


def stage_evaluate_t5_trajectory(
    run_root: Path,
    eval_root: Path,
    gpu_id: str = "0",
) -> dict:
    """Score existing T5 checkpoints at 40/80/120/160/200 without retraining seed42."""
    audit = load_json(
        ablation_root(run_root) / "diagnostics" / "9u78_A" / "audit_summary.json"
    )
    if audit.get("audit_pass") is not True:
        payload = {
            "stage": "evaluate-t5-trajectory",
            "status": "blocked",
            "reason": "complete_9u78_independent_inference_audit_first",
            "rows": [],
        }
        write_json(ablation_root(run_root) / "t5_trajectory_summary.json", payload)
        write_json(eval_root / "t5_trajectory_summary.json", payload)
        update_status(run_root, evaluate_t5_trajectory="blocked")
        return payload
    steps = (40, 80, 120, 160, 200)
    rows = []
    for slug in FALLBACK_TARGETS:
        for step in steps:
            epochs = step // OVERFIT_EPOCH_LEN
            try:
                result = export_and_score_overfit(
                    run_root=run_root,
                    slug=slug,
                    seed=42,
                    epochs=epochs,
                    epoch_len=OVERFIT_EPOCH_LEN,
                    gpu_id=gpu_id,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"target_slug": slug, "step": step, "error": str(exc)})
                continue
            v2 = result.get("capacity_v2") or {}
            paired = Path(result["paired_csv"])
            paired_rows = list(csv.DictReader(paired.open()))
            by_label = {r["label"]: float(r["delta_tm"]) for r in paired_rows}
            rows.append(
                {
                    "target_slug": slug,
                    "step": step,
                    "mean_delta_tm": result.get("mean_delta_tm"),
                    "median_delta_tm": v2.get("median_delta_tm"),
                    "leave_max_out_mean_delta_tm": v2.get("leave_max_out_mean_delta_tm"),
                    "delta_tm_ge_0p01_count": v2.get("delta_tm_ge_0p01_count"),
                    "tm_rise_count": v2.get("tm_rise_count"),
                    "capacity_v1_pass": result.get("pass"),
                    "capacity_v2_pass": v2.get("pass"),
                    "delta_9u78_A": by_label.get("9u78_A"),
                    "delta_9xpd_A": by_label.get("9xpd_A"),
                }
            )
    out_csv = eval_root / "t5_overfit_trajectory.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    errors = [row for row in rows if row.get("error")]
    complete = len(rows) == len(FALLBACK_TARGETS) * len(steps) and not errors
    payload = {
        "stage": "evaluate-t5-trajectory",
        "status": "complete" if complete else "failed",
        "rows": rows,
        "errors": errors,
        "csv": str(out_csv),
    }
    write_json(ablation_root(run_root) / "t5_trajectory_summary.json", payload)
    write_json(eval_root / "t5_trajectory_summary.json", payload)
    update_status(
        run_root, evaluate_t5_trajectory="complete" if complete else "failed"
    )
    return payload


def stage_audit_t5_seeds(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str = "0",
) -> dict:
    """Train/eval T5A overfit8 with seeds 43/44 on all 8 labels."""
    audit = load_json(
        ablation_root(run_root) / "diagnostics" / "9u78_A" / "audit_summary.json"
    )
    if audit.get("audit_pass") is not True:
        payload = {
            "stage": "audit-t5-seeds",
            "status": "blocked",
            "reason": "complete_9u78_independent_inference_audit_first",
            "seeds": [],
        }
        write_json(ablation_root(run_root) / "t5a_seed_stability.json", payload)
        write_json(eval_root / "t5a_seed_stability.json", payload)
        update_status(run_root, audit_t5_seeds="blocked")
        return payload
    overfit_manifest = load_json(
        ablation_root(run_root) / "overfit8" / "overfit8_manifest.json"
    )
    seed_rows = []
    for seed in (42, 43, 44):
        if seed != 42:
            train_overfit_segment(
                run_root=run_root,
                source_run=source_run,
                slug="T5A",
                seed=seed,
                max_epochs=OVERFIT_MAX_EPOCHS,
                smoke=False,
                gpu_id=gpu_id,
            )
        result = export_and_score_overfit(
            run_root=run_root,
            slug="T5A",
            seed=seed,
            epochs=OVERFIT_MAX_EPOCHS,
            epoch_len=OVERFIT_EPOCH_LEN,
            gpu_id=gpu_id,
        )
        v2 = result.get("capacity_v2") or capacity_pass_v2_from_rows(
            list(csv.DictReader(Path(result["paired_csv"]).open())),
            expected_labels=overfit_manifest.get("labels"),
            expected_clusters=overfit_manifest.get("clusters"),
        )
        paired_rows = list(csv.DictReader(Path(result["paired_csv"]).open()))
        for row in paired_rows:
            row["delta_tm"] = float(row["delta_tm"])
        by_label = {r["label"]: float(r["delta_tm"]) for r in paired_rows}
        max_label = max(by_label, key=by_label.get)
        seed_rows.append(
            {
                "seed": seed,
                "capacity_v1_pass": result.get("pass"),
                "capacity_v2_pass": v2.get("pass"),
                "mean_delta_tm": result.get("mean_delta_tm"),
                "median_delta_tm": v2.get("median_delta_tm"),
                "leave_max_out_mean_delta_tm": v2.get("leave_max_out_mean_delta_tm"),
                "delta_9u78_A": by_label.get("9u78_A"),
                "max_delta_label": max_label,
                "max_delta_tm": by_label[max_label],
                "large_9u78": bool(by_label.get("9u78_A", 0) >= 0.1),
            }
        )
    large_count = sum(1 for r in seed_rows if r["large_9u78"])
    payload = {
        "stage": "audit-t5-seeds",
        "seeds": seed_rows,
        "large_9u78_seed_count": large_count,
        "large_9u78_stable_2_of_3": large_count >= 2,
    }
    # CSV
    csv_path = eval_root / "t5a_seed_stability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    payload["csv"] = str(csv_path)
    write_json(ablation_root(run_root) / "t5a_seed_stability.json", payload)
    write_json(eval_root / "t5a_seed_stability.json", payload)
    update_status(run_root, audit_t5_seeds="complete")
    return payload


def stage_reeval_capacity(run_root: Path, eval_root: Path) -> dict:
    """Offline capacity_v2 reevaluation; does not overwrite capacity_v1 files."""
    rows = []
    slugs = list(PRIMARY_OVERFIT_TARGETS) + list(FALLBACK_TARGETS)
    for slug in slugs:
        for step in (40, 80, 120, 200):
            paired = (
                overfit_run_root(run_root, slug)
                / "eval"
                / f"step_{step}"
                / "scale_1p0"
                / "paired_deltas.csv"
            )
            if not paired.exists():
                continue
            paired_rows = list(csv.DictReader(paired.open()))
            for row in paired_rows:
                row["delta_tm"] = float(row["delta_tm"])
                row["tm_baseline"] = float(row["tm_baseline"])
                row["tm_model"] = float(row["tm_model"])
                if row.get("delta_lddt_ca") not in ("", None):
                    row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
            v1 = capacity_pass_from_rows(paired_rows)
            overfit_manifest = load_json(ablation_root(run_root) / "overfit8" / "overfit8_manifest.json")
            v2 = capacity_pass_v2_from_rows(
                paired_rows,
                expected_labels=overfit_manifest.get("labels"),
                expected_clusters=overfit_manifest.get("clusters"),
            )
            write_json(paired.parent / "capacity_v2.json", {"step": step, **v2})
            # Also write status_v2 beside existing capacity_status when final step.
            rows.append(
                {
                    "target_slug": slug,
                    "step": step,
                    "capacity_v1_pass": v1.get("pass"),
                    "capacity_v2_pass": v2.get("pass"),
                    "mean_delta_tm": v2.get("mean_delta_tm"),
                    "median_delta_tm": v2.get("median_delta_tm"),
                    "leave_max_out_mean_delta_tm": v2.get("leave_max_out_mean_delta_tm"),
                    "delta_tm_ge_0p01_count": v2.get("delta_tm_ge_0p01_count"),
                    "tm_rise_count": v2.get("tm_rise_count"),
                }
            )
        # Final status_v2 from step map if available
        step_map = {}
        for step in (40, 80, 120, 200):
            path = (
                overfit_run_root(run_root, slug)
                / "eval"
                / f"step_{step}"
                / "scale_1p0"
                / "capacity_v2.json"
            )
            if path.exists():
                step_map[step] = load_json(path)
        if step_map:
            status_v2 = capacity_status_for_target(step_map, max_steps=200, pass_key="pass")
            write_json(overfit_run_root(run_root, slug) / "capacity_status_v2.json", status_v2)

    # Refresh diagnostics fallback v2 flags from T5 final steps.
    diagnostics = load_json(ablation_root(run_root) / "capacity_diagnostics.json")
    if diagnostics.get("fallback"):
        t5_v2 = {}
        for slug in FALLBACK_TARGETS:
            path = (
                overfit_run_root(run_root, slug)
                / "eval"
                / "step_200"
                / "scale_1p0"
                / "capacity_v2.json"
            )
            if path.exists():
                t5_v2[slug] = load_json(path)
                if diagnostics["fallback"].get("t5_results", {}).get(slug):
                    diagnostics["fallback"]["t5_results"][slug]["capacity_v2"] = t5_v2[slug]
        diagnostics["fallback"]["t5_capacity_v2"] = {
            s: bool(v.get("pass")) for s, v in t5_v2.items()
        }
        diagnostics["fallback"]["t5_capacity_v2_any_pass"] = any(
            diagnostics["fallback"]["t5_capacity_v2"].values()
        )
        diagnostics["failure_classification"] = classify_capacity_failure(
            t5_any_pass=diagnostics["fallback"]["t5_capacity_v2_any_pass"],
            positive_control_pass=bool(
                (diagnostics["fallback"].get("positive_control") or {}).get("pass")
            ),
            positive_control_executed=bool(
                (diagnostics["fallback"].get("positive_control") or {}).get("executed")
            ),
            loss_decreased=True,
            tm_unchanged=True,
        )
        write_json(ablation_root(run_root) / "capacity_diagnostics.json", diagnostics)
        write_json(eval_root / "capacity_diagnostics.json", diagnostics)

    csv_path = eval_root / "capacity_v2_reeval.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    primary_v2_statuses = {}
    for slug in PRIMARY_OVERFIT_TARGETS:
        status_v2_path = overfit_run_root(run_root, slug) / "capacity_status_v2.json"
        if status_v2_path.exists():
            primary_v2_statuses[slug] = load_json(status_v2_path).get("status")
    payload = {"stage": "reeval-capacity", "rows": rows, "csv": str(csv_path)}
    write_json(eval_root / "capacity_v2_reeval.json", payload)
    update_status(
        run_root,
        reeval_capacity="complete",
        overfit_statuses_v2=primary_v2_statuses,
    )
    return payload


def positive_control_root(run_root: Path) -> Path:
    return ablation_root(run_root) / "PC_STRUCTURE" / "overfit8" / "seed_42"


def unclamped_positive_control_root(run_root: Path) -> Path:
    return (
        ablation_root(run_root)
        / "PC_UNCLAMPED_STRUCTURE"
        / "overfit8"
        / "seed_42"
    )


def infer_loss_trend(log_path: Path) -> Optional[bool]:
    import re

    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    values = []
    for match in re.finditer(
        r"train[/_]loss(?:_(?:step|epoch))?[\s:=]+([0-9.eE+-]+)", text
    ):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    if len(values) < 2:
        return None
    window = min(8, max(1, len(values) // 4))
    first = sum(values[:window]) / window
    last = sum(values[-window:]) / window
    return last < first


def train_real_positive_control(
    run_root: Path,
    source_run: Path,
    gpu_id: str,
) -> Path:
    private = private_root(run_root)
    overfit = ablation_root(run_root) / "overfit8"
    out = positive_control_root(run_root)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = overfit / "overfit8_labels.txt"
    final_ckpt = checkpoint_for_epochs(train_dir, OVERFIT_MAX_EPOCHS, OVERFIT_EPOCH_LEN)
    if not parent.exists():
        raise FileNotFoundError(parent)
    manifest = {
        "client_id": "client_0",
        "mode": "positive_control_overfit8",
        "control": "selective_non_lora_unfreeze",
        "trainable_scope": "structure_module",
        "lora_rank": 0,
        "seed": 42,
        "learning_rate": 1e-4,
        "lr_warmup_steps": OVERFIT_WARMUP_STEPS,
        "max_epochs": OVERFIT_MAX_EPOCHS,
        "train_epoch_len": OVERFIT_EPOCH_LEN,
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "labels_sha256": sha256_file(labels),
        "precision": "32",
        "aggregation": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    prepare_run_manifest(
        out / "local_run.json",
        manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context="real positive control",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] positive-control manifest verified: {final_ckpt}")
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
        "--experiment_config_json", str(REPO / "seq_model_esm1b_ptm_finetune_override.json"),
        "--resume_from_ckpt", str(parent),
        "--resume_model_weights_only", "True",
        "--init_weights_source", "auto",
        "--template_release_dates_cache_path", str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path", str(overfit / "overfit8_chain_data_cache.json"),
        "--lora_rank", "0",
        "--trainable_scope", "structure_module",
        "--learning_rate", "1e-4",
        "--lr_warmup_steps", str(OVERFIT_WARMUP_STEPS),
        "--accumulate_grad_batches", "1",
        "--train_epoch_len", str(OVERFIT_EPOCH_LEN),
        "--max_epochs", str(OVERFIT_MAX_EPOCHS),
        "--checkpoint_every_epoch",
        "--sampling_mode", "uniform",
        "--precision", "32",
        "--gpus", "1",
        "--seed", "42",
        "--deepspeed_config_path", str(REPO / "deepspeed_config_fp32.json"),
    ]
    log_path = out / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)},
            cwd=str(REPO),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return final_ckpt


def train_unclamped_positive_control(
    run_root: Path,
    source_run: Path,
    gpu_id: str,
) -> Path:
    """Single-variable capacity control: structure module with unclamped FAPE."""
    overfit = ablation_root(run_root) / "overfit8"
    out = unclamped_positive_control_root(run_root)
    train_dir = out / "training"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    labels = overfit / "overfit8_labels.txt"
    config_path = REPO / "seq_model_esm1b_ptm_hardcase_unclamped_override.json"
    cluster_file = REPO / "data" / "all_pdb_1y" / "clusters_30.txt"
    final_ckpt = checkpoint_for_epochs(
        train_dir, OVERFIT_MAX_EPOCHS, OVERFIT_EPOCH_LEN
    )
    for required in (parent, labels, config_path, cluster_file):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = {
        "client_id": "client_0",
        "mode": "unclamped_positive_control_overfit8",
        "control": "selective_non_lora_unfreeze_unclamped_fape",
        "trainable_scope": "structure_module",
        "lora_rank": 0,
        "seed": 42,
        "learning_rate": 1e-4,
        "lr_warmup_steps": OVERFIT_WARMUP_STEPS,
        "max_epochs": OVERFIT_MAX_EPOCHS,
        "train_epoch_len": OVERFIT_EPOCH_LEN,
        "checkpoint_every_n_train_steps": 40,
        "sampling_mode": "uniform",
        "sampling_audit": True,
        "csv_loss_metrics": True,
        "fape_clamp_prob": 0.0,
        "experiment_config_path": str(config_path),
        "experiment_config_sha256": sha256_file(config_path),
        "parent_path": str(parent),
        "parent_sha256": sha256_file(parent),
        "labels_sha256": sha256_file(labels),
        "precision": "32",
        "aggregation": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    prepare_run_manifest(
        out / "local_run.json",
        manifest,
        artifact_exists=final_ckpt.exists() or final_ckpt.is_dir(),
        context="unclamped-FAPE positive control",
    )
    if final_ckpt.exists() or final_ckpt.is_dir():
        print(f"[skip] unclamped positive-control manifest verified: {final_ckpt}")
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
        "--experiment_config_json", str(config_path),
        "--resume_from_ckpt", str(parent),
        "--resume_model_weights_only", "True",
        "--init_weights_source", "auto",
        "--template_release_dates_cache_path",
        str(data_src / "mmcif_cache_finetune.json"),
        "--train_chain_data_cache_path",
        str(overfit / "overfit8_chain_data_cache.json"),
        "--lora_rank", "0",
        "--trainable_scope", "structure_module",
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
        "--seed", "42",
        "--deepspeed_config_path", str(REPO / "deepspeed_config_fp32.json"),
    ]
    log_path = out / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
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
    return final_ckpt


def evaluate_real_positive_control(
    run_root: Path,
    checkpoint: Path,
    gpu_id: str,
    step: int = 200,
    weights_source: str = POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE,
    control_root: Optional[Path] = None,
    control_name: str = "selective_non_lora_structure_module",
) -> dict:
    private = private_root(run_root)
    overfit = ablation_root(run_root) / "overfit8"
    out = control_root or positive_control_root(run_root)
    if weights_source not in {"state_dict", "ema"}:
        raise ValueError(f"Unsupported positive-control weights source: {weights_source}")
    # Preserve the original raw-weight artifact layout for backward compatibility.
    cand = (
        out / "eval" / f"step_{step}"
        if weights_source == POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE
        else out / "eval" / weights_source / f"step_{step}"
    )
    cand.mkdir(parents=True, exist_ok=True)
    labels = overfit / "overfit8_labels.txt"
    expected = len(read_labels(labels))
    manifest = load_json(overfit / "overfit8_manifest.json")
    evaluation_manifest = {
        "control": control_name,
        "seed": 42,
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "labels_sha256": manifest.get("labels_sha256"),
        "clusters": sorted(manifest.get("clusters") or []),
        "precision": "fp32",
        "checkpoint_weights_source": weights_source,
    }
    prepare_run_manifest(
        cand / "evaluation_run.json",
        evaluation_manifest,
        artifact_exists=(cand / "predictions").exists(),
        context="positive-control evaluation",
    )
    pred_dir = cand / "predictions"
    existing = list(pred_dir.glob("*_unrelaxed.pdb")) if pred_dir.exists() else []
    if len(existing) != expected:
        if existing:
            raise RuntimeError(f"Partial positive-control predictions: {len(existing)}/{expected}")
        run_cmd(
            [
                PYTHON,
                str(REPO / "run_pretrained_openfold.py"),
                str(overfit / "assets" / "solo_fasta_dir"),
                str(overfit / "assets" / "mmcif_files"),
                "--use_precomputed_alignments", str(overfit / "assets" / "solo_alignment_dir"),
                "--use_single_seq_mode",
                "--output_dir", str(cand),
                "--model_device", "cuda:0",
                "--skip_relaxation",
                "--config_preset", "seq_model_esm1b_ptm",
                "--openfold_checkpoint_path", str(checkpoint),
                "--checkpoint_weights_source",
                weights_source,
                "--data_random_seed", "42",
                "--precision", "fp32",
            ],
            env={"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)},
        )
    metrics = cand / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    run_cmd([
        PYTHON, str(REPO / "scripts" / "tmscore_from_pdb.py"), str(pred_dir),
        "--native-dir", str(native), "--tm-exec", str(REPO / "tmscore" / "TMscore"),
        "--out-csv", str(metrics / "tm_score.csv"),
        "--selected-csv", str(metrics / "low_tm.csv"),
        "--missing-log", str(metrics / "tm_missing.txt"),
    ])
    run_cmd([
        PYTHON, str(REPO / "scripts" / "lddt_ca_from_pdb.py"), str(pred_dir),
        "--native-dir", str(native), "--out-csv", str(metrics / "lddt_ca.csv"),
        "--missing-log", str(metrics / "lddt_missing.txt"),
    ])
    paired = cand / "paired_deltas.csv"
    run_cmd([
        PYTHON, str(REPO / "scripts" / "evaluate_hardcase_metrics.py"),
        "--mode", "pair", "--labels", str(labels),
        "--difficulty-csv", str(difficulty),
        "--baseline-tm-csv", str(difficulty), "--model-tm-csv", str(metrics / "tm_score.csv"),
        "--baseline-lddt-csv", str(difficulty), "--model-lddt-csv", str(metrics / "lddt_ca.csv"),
        "--client-id", "client_0", "--out", str(paired),
    ])
    rows = list(csv.DictReader(paired.open()))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    v1 = capacity_pass_from_rows(rows)
    v2 = capacity_pass_v2_from_rows(
        rows,
        expected_labels=manifest.get("labels"),
        expected_clusters=manifest.get("clusters"),
    )
    write_json(cand / "capacity.json", {"step": step, **v1})
    write_json(cand / "capacity_v2.json", {"step": step, **v2})
    return {
        "step": step,
        "checkpoint_weights_source": weights_source,
        "paired_csv": str(paired),
        "capacity_v1": v1,
        "capacity_v2": v2,
    }


def stage_evaluate_positive_control_trajectory(
    run_root: Path,
    eval_root: Path,
    gpu_id: str = "0",
    weights_source: str = POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE,
    control_root: Optional[Path] = None,
    control_name: str = "selective_non_lora_structure_module",
    artifact_stem: str = "positive_control_trajectory",
    status_stem: str = "positive_control",
) -> dict:
    steps = (40, 80, 120, 160, 200)
    out = control_root or positive_control_root(run_root)
    train_dir = out / "training"
    rows = []
    for step in steps:
        epochs = step // OVERFIT_EPOCH_LEN
        checkpoint = checkpoint_for_epochs(train_dir, epochs, OVERFIT_EPOCH_LEN)
        if not (checkpoint.exists() or checkpoint.is_dir()):
            raise FileNotFoundError(checkpoint)
        result = evaluate_real_positive_control(
            run_root,
            checkpoint,
            gpu_id,
            step=step,
            weights_source=weights_source,
            control_root=out,
            control_name=control_name,
        )
        paired_rows = list(csv.DictReader(Path(result["paired_csv"]).open()))
        by_label = {row["label"]: float(row["delta_tm"]) for row in paired_rows}
        v2 = result["capacity_v2"]
        row = {
            "step": step,
            "checkpoint_weights_source": weights_source,
            "mean_delta_tm": v2.get("mean_delta_tm"),
            "median_delta_tm": v2.get("median_delta_tm"),
            "leave_max_out_mean_delta_tm": v2.get("leave_max_out_mean_delta_tm"),
            "delta_tm_ge_0p01_count": v2.get("delta_tm_ge_0p01_count"),
            "tm_rise_count": v2.get("tm_rise_count"),
            "mean_delta_lddt": v2.get("mean_delta_lddt"),
            "max_point_contribution": v2.get("max_point_contribution"),
            "capacity_v2_pass": v2.get("pass"),
        }
        for label in sorted(by_label):
            row[f"delta_tm_{label}"] = by_label[label]
        rows.append(row)

    artifact_suffix = (
        ""
        if weights_source == POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE
        else f"_{weights_source}"
    )
    csv_path = eval_root / f"{artifact_stem}{artifact_suffix}.csv"
    fieldnames = list(rows[0]) if rows else []
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    stage_name = (
        f"evaluate-positive-control-{weights_source}-trajectory"
        if artifact_stem == "positive_control_trajectory"
        else f"evaluate-{artifact_stem}-{weights_source}"
    )
    payload = {
        "stage": stage_name,
        "status": "complete",
        "checkpoint_weights_source": weights_source,
        "steps": list(steps),
        "rows": rows,
        "csv": str(csv_path),
    }
    json_name = f"{artifact_stem}{artifact_suffix}.json"
    write_json(ablation_root(run_root) / json_name, payload)
    write_json(eval_root / json_name, payload)
    status_key = (
        f"{status_stem}_trajectory"
        if weights_source == POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE
        else f"{status_stem}_{weights_source}_trajectory"
    )
    update_status(run_root, **{status_key: "complete"})
    return payload


def stage_run_unclamped_positive_control(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str = "0",
) -> dict:
    checkpoint = train_unclamped_positive_control(run_root, source_run, gpu_id)
    trajectory = stage_evaluate_positive_control_trajectory(
        run_root,
        eval_root,
        gpu_id=gpu_id,
        weights_source=POSITIVE_CONTROL_CHECKPOINT_WEIGHTS_SOURCE,
        control_root=unclamped_positive_control_root(run_root),
        control_name="selective_non_lora_structure_module_unclamped_fape",
        artifact_stem="unclamped_positive_control_trajectory",
        status_stem="unclamped_positive_control",
    )
    payload = {
        "stage": "run-unclamped-positive-control",
        "status": "complete",
        "checkpoint": str(checkpoint),
        "capacity_v2_any_pass": any(
            bool(row.get("capacity_v2_pass"))
            for row in trajectory.get("rows", [])
        ),
        "trajectory": trajectory,
        "loss_metrics_csv": str(
            unclamped_positive_control_root(run_root)
            / "training"
            / "csv_logs"
            / "metrics.csv"
        ),
        "sampling_audit": str(
            unclamped_positive_control_root(run_root) / "sampling_audit.jsonl"
        ),
    }
    write_json(eval_root / "unclamped_positive_control_summary.json", payload)
    write_json(
        ablation_root(run_root) / "unclamped_positive_control_summary.json",
        payload,
    )
    update_status(run_root, run_unclamped_positive_control="complete")
    return payload


def stage_run_positive_control(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    gpu_id: str = "0",
    force: bool = False,
) -> dict:
    from scripts.positive_control_structure_module import (
        positive_control_status,
        should_run_real_positive_control,
        verify_positive_control_setup,
    )

    diagnostics_path = ablation_root(run_root) / "capacity_diagnostics.json"
    diagnostics = load_json(diagnostics_path)
    fallback = diagnostics.get("fallback") or {}
    audit = load_json(ablation_root(run_root) / "diagnostics" / "9u78_A" / "audit_summary.json")
    validation_path = ablation_root(run_root) / "validation_summary.json"
    validation = load_json(validation_path) if validation_path.exists() else {}
    diagnostic_executed = (
        validation.get("branch") in {"t5_standard", "t5_diagnostic_only"}
        and validation.get("status") == "complete"
    )
    diagnostic_pass = bool(validation.get("promotion_passers"))
    should_run = should_run_real_positive_control(
        t5_capacity_v2_any_pass=bool(fallback.get("t5_capacity_v2_any_pass")),
        diagnostic_validation_pass=diagnostic_pass,
        diagnostic_validation_executed=diagnostic_executed,
        audit_pass=bool(audit.get("audit_pass")),
    )
    audit_pass = bool(audit.get("audit_pass"))
    if not audit_pass or not (should_run or force):
        reason = (
            "requires_complete_9u78_audit"
            if not audit_pass
            else "requires_executed_failed_t5_validation_or_explicit_force"
        )
        payload = {
            "stage": "run-positive-control",
            "status": "blocked",
            "blocked": True,
            "reason": reason,
        }
        write_json(ablation_root(run_root) / "positive_control_summary.json", payload)
        write_json(eval_root / "positive_control_summary.json", payload)
        update_status(run_root, positive_control="blocked")
        return payload

    setup = verify_positive_control_setup()
    checkpoint = train_real_positive_control(run_root, source_run, gpu_id)
    result = evaluate_real_positive_control(run_root, checkpoint, gpu_id)
    loss_decreased = infer_loss_trend(positive_control_root(run_root) / "train.log")
    capacity_v2 = result["capacity_v2"]
    mean_delta = capacity_v2.get("mean_delta_tm")
    tm_unchanged = mean_delta is not None and abs(float(mean_delta)) < 0.005
    max_contribution = capacity_v2.get("max_point_contribution")
    median_delta = capacity_v2.get("median_delta_tm")
    tm_sparse_response = bool(
        not capacity_v2.get("pass")
        and (
            (max_contribution is not None and float(max_contribution) > 0.5)
            or (median_delta is not None and float(median_delta) <= 0)
        )
    )
    pc = positive_control_status(
        setup=setup,
        executed=True,
        capacity_pass=bool(capacity_v2.get("pass")),
        loss_decreased=loss_decreased,
        tm_unchanged=tm_unchanged,
        tm_sparse_response=tm_sparse_response,
    )
    pc["result"] = result
    fallback["positive_control"] = pc
    diagnostics["fallback"] = fallback
    diagnostics["failure_classification"] = classify_capacity_failure(
        t5_any_pass=bool(fallback.get("t5_capacity_v2_any_pass")),
        positive_control_pass=bool(pc.get("pass")),
        positive_control_executed=True,
        loss_decreased=loss_decreased,
        tm_unchanged=bool(tm_unchanged),
        tm_sparse_response=tm_sparse_response,
    )
    write_json(diagnostics_path, diagnostics)
    write_json(eval_root / "capacity_diagnostics.json", diagnostics)
    payload = {"stage": "run-positive-control", "status": "complete", "blocked": False, **pc}
    write_json(ablation_root(run_root) / "positive_control_summary.json", payload)
    write_json(eval_root / "positive_control_summary.json", payload)
    update_status(run_root, positive_control="complete")
    return payload


def stage_validate(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
    diagnostic_only: bool = False,
) -> dict:
    return _stage_validate_impl(
        run_root, source_run, eval_root, smoke, gpu_id, diagnostic_only
    )


# Keep implementation under a stable name after replacing the old function body.
stage_validate.__doc__ = "Route-aware validation using resolve_validation_branch."


def _stage_validate_impl(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
    diagnostic_only: bool = False,
) -> dict:
    status = load_json(status_path(run_root)) if status_path(run_root).exists() else {}
    statuses = status.get("overfit_statuses_v2") or status.get("overfit_statuses", {})
    diagnostics = load_json(ablation_root(run_root) / "capacity_diagnostics.json")
    fallback_v2 = {}
    if diagnostics.get("fallback"):
        fallback_v2 = {
            slug: {"pass": bool(pass_)}
            for slug, pass_ in (diagnostics["fallback"].get("t5_capacity_v2") or {}).items()
        }
        for slug, result in (diagnostics["fallback"].get("t5_results") or {}).items():
            if isinstance(result.get("capacity_v2"), dict):
                fallback_v2[slug] = result["capacity_v2"]

    audit_path = ablation_root(run_root) / "diagnostics" / "9u78_A" / "audit_summary.json"
    if audit_path.exists():
        audit_summary = load_json(audit_path)
    elif smoke:
        audit_summary = {
            "audit_integrity_pass": True,
            "metric_reproducible": True,
            "inference_reproducible": True,
            "audit_complete": True,
            "audit_pass": True,
        }
    else:
        audit_summary = {}

    route = resolve_validation_branch(
        statuses, fallback_v2, audit_summary, diagnostic_only=diagnostic_only
    )
    write_json(ablation_root(run_root) / "validation_route.json", route)
    write_json(eval_root / "validation_route.json", route)

    if not smoke and not route["scheduled_targets"]:
        payload = {
            "stage": "validate",
            "blocked": True,
            "status": "blocked",
            "reason": route.get("blocked_reason"),
            "branch": route.get("branch"),
            "route": route,
            "blocks_other_clients": True,
            "blocks_fedlora": True,
            "test_accessed": False,
        }
        write_json(ablation_root(run_root) / "validation_summary.json", payload)
        write_json(eval_root / "validation_summary.json", payload)
        update_status(run_root, validate="blocked", validation_branch=route.get("branch"))
        return payload

    private = private_root(run_root)
    split_root = private / "splits"
    train_n = len(read_labels(split_root / "train_labels.txt"))
    val_n = len(read_labels(split_root / "validation_labels.txt"))
    if not smoke and (train_n != 57 or val_n != 10):
        raise AssertionError(
            f"Unexpected client0 split sizes train={train_n} val={val_n}"
        )

    scheduled = route["scheduled_targets"]
    results = {}
    for slug in scheduled:
        print(
            f"[validate] branch={route['branch']} target={slug} seed=42 "
            f"out={full_train_root(run_root, slug, 42)}"
        )
        train_full_target(
            run_root=run_root,
            source_run=source_run,
            slug=slug,
            seed=42,
            epochs=5,
            smoke=smoke,
            gpu_id=gpu_id,
        )
        results[slug] = evaluate_full_validation(
            run_root=run_root,
            slug=slug,
            seed=42,
            epoch=5,
            scale=1.0,
            smoke=smoke,
            gpu_id=gpu_id,
        )

    passers = []
    for slug in scheduled:
        row = results[slug]
        v2 = row.get("promotion_v2") or {}
        if bool(v2.get("promotion_pass", row.get("promotion_pass"))):
            passers.append(
                {
                    "target_slug": slug,
                    "hard_mean_delta_tm": v2.get(
                        "hard_mean_delta_tm", row.get("hard_mean_delta_tm")
                    ),
                    "hard_median_delta_tm": v2.get(
                        "hard_median_delta_tm", row.get("hard_median_delta_tm")
                    ),
                    "all_mean_delta_lddt": v2.get(
                        "all_mean_delta_lddt", row.get("all_mean_delta_lddt")
                    ),
                    "trainable_params": get_target(slug).expected_trainable,
                    "promotion_pass": True,
                }
            )
    seed_candidates = select_seed_confirmation_targets(passers, max_targets=2)
    payload = {
        "stage": "validate",
        "blocked": False,
        "status": "complete",
        "branch": route.get("branch"),
        "route": route,
        "scheduled_targets": scheduled,
        "results": results,
        "promotion_passers": passers,
        "seed_confirmation_candidates": seed_candidates,
        "capacity_v2_failed": route.get("capacity_v2_failed"),
        "primary_comparison": {"epoch": 5, "scale": 1.0},
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "test_accessed": False,
        "train_label_count": train_n,
        "validation_label_count": val_n,
    }
    write_json(ablation_root(run_root) / "validation_summary.json", payload)
    write_json(eval_root / "validation_summary.json", payload)
    update_status(
        run_root,
        validate="complete",
        validation_branch=route.get("branch"),
        seed_confirmation_candidates=[c["target_slug"] for c in seed_candidates],
        blocks_other_clients=True,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def stage_confirm_seeds(
    run_root: Path,
    source_run: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
) -> dict:
    val = load_json(ablation_root(run_root) / "validation_summary.json")
    candidates = val.get("seed_confirmation_candidates") or []
    if not candidates:
        payload = {
            "stage": "confirm-seeds",
            "blocked": True,
            "status": "blocked",
            "reason": "no_promotion_passers",
            "blocks_development": True,
            "blocks_other_clients": True,
            "blocks_fedlora": True,
            "test_accessed": False,
        }
        write_json(ablation_root(run_root) / "seed_confirmation.json", payload)
        update_status(run_root, confirm_seeds="blocked")
        return payload

    confirmed = []
    per_target = {}
    for cand in candidates:
        slug = cand["target_slug"]
        seed_rows = []
        for seed in (42, 43, 44):
            if seed != 42:
                train_full_target(
                    run_root=run_root,
                    source_run=source_run,
                    slug=slug,
                    seed=seed,
                    epochs=5,
                    smoke=smoke,
                    gpu_id=gpu_id,
                )
            metrics = evaluate_full_validation(
                run_root=run_root,
                slug=slug,
                seed=seed,
                epoch=1 if smoke else 5,
                scale=1.0,
                smoke=smoke,
                gpu_id=gpu_id,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "hard_mean_delta_tm": metrics["hard_mean_delta_tm"],
                    "hard_median_delta_tm": metrics["hard_median_delta_tm"],
                    "nonhard_mean_delta_tm": metrics["nonhard_mean_delta_tm"],
                    "all_mean_delta_lddt": metrics["all_mean_delta_lddt"],
                    "promotion_v2_pass": bool(
                        (metrics.get("promotion_v2") or {}).get("promotion_pass")
                    ),
                }
            )
        decision = seed_confirmation_pass(seed_rows)
        aggregate_candidate = {
            **cand,
            **decision,
            "hard_mean_delta_tm": decision.get("mean_hard_mean_delta_tm"),
            "hard_median_delta_tm": decision.get("mean_hard_median_delta_tm"),
            "all_mean_delta_lddt": decision.get("mean_all_delta_lddt"),
        }
        per_target[slug] = {"seeds": seed_rows, **aggregate_candidate}
        if decision["pass"]:
            confirmed.append(aggregate_candidate)

    unique = choose_unique_target(confirmed)
    payload = {
        "stage": "confirm-seeds",
        "blocked": False,
        "status": "complete" if unique else "failed",
        "per_target": per_target,
        "confirmed": confirmed,
        "unique_target": unique,
        "blocks_development": unique is None,
        "blocks_other_clients": True,
        "blocks_fedlora": True,
        "test_accessed": False,
    }
    write_json(ablation_root(run_root) / "seed_confirmation.json", payload)
    write_json(eval_root / "seed_confirmation.json", payload)
    update_status(
        run_root,
        confirm_seeds="complete" if unique else "failed",
        unique_target=(unique or {}).get("target_slug"),
        blocks_other_clients=True,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def stage_evaluate_dev(
    run_root: Path,
    eval_root: Path,
    smoke: bool = False,
    gpu_id: str = "0",
) -> dict:
    conf = load_json(ablation_root(run_root) / "seed_confirmation.json")
    unique = conf.get("unique_target")
    if not unique:
        payload = {
            "stage": "evaluate-dev",
            "blocked": True,
            "status": "blocked",
            "reason": "client0_target_ablation_no_stable_generalization_signal",
            "blocks_client2": True,
            "blocks_other_clients": True,
            "blocks_fedlora": True,
            "test_accessed": False,
        }
        write_json(ablation_root(run_root) / "development_summary.json", payload)
        update_status(run_root, evaluate_dev="blocked")
        return payload

    slug = unique["target_slug"]
    # Development uses seed 42 model from unique target.
    private = private_root(run_root)
    split_root = private / "splits"
    out = full_train_root(run_root, slug, 42)
    epoch = 1 if smoke else 5
    epoch_len = 2 if smoke else len(read_labels(split_root / "train_labels.txt"))
    ckpt = checkpoint_for_epochs(out / "training", epoch, epoch_len)
    cand = out / "development" / f"epoch_{epoch}" / "scale_1p0"
    cand.mkdir(parents=True, exist_ok=True)
    model = cand / "model_scale_1.0.pt"
    parent = run_root / "server" / "rounds" / "round_000" / "global_model.pt"
    if not (ckpt.exists() or ckpt.is_dir()):
        raise FileNotFoundError(ckpt)
    ensure_lora_export(
        checkpoint=ckpt,
        model=model,
        parent=parent,
        target=get_target(slug).target,
        scale=1.0,
    )
    labels = split_root / "development_test_labels.txt"
    assets = split_root / "assets_development_test"
    difficulty = private / "difficulty" / "baseline_difficulty.csv"
    expected_labels, expected_clusters = frozen_split_identity(
        split_root, "development_test", difficulty
    )
    development_manifest = {
        "target_slug": slug,
        "target": get_target(slug).target,
        "seed": 42,
        "epoch": epoch,
        "scale": 1.0,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_path(ckpt),
        "parent_sha256": sha256_file(parent),
        "development_labels_sha256": labels_sha256(expected_labels),
        "development_clusters": sorted(expected_clusters),
    }
    prepare_run_manifest(
        cand / "evaluation_run.json",
        development_manifest,
        artifact_exists=(cand / "predictions").exists(),
        context=f"development {slug} seed=42 epoch={epoch}",
    )
    expected = len(expected_labels)
    pred_dir = cand / "predictions"
    existing = list(pred_dir.glob("*_unrelaxed.pdb")) if pred_dir.exists() else []
    if len(existing) != expected:
        if existing:
            raise RuntimeError(
                f"Partial development predictions at {cand}: {len(existing)}/{expected}"
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
                str(cand),
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
                "42",
                "--precision",
                "fp32",
            ],
            env={"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONPATH": str(REPO)},
        )
    metrics = cand / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    native = private / "difficulty" / "native"
    if not (metrics / "tm_score.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "tmscore_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--tm-exec",
                str(REPO / "tmscore" / "TMscore"),
                "--out-csv",
                str(metrics / "tm_score.csv"),
                "--selected-csv",
                str(metrics / "low_tm.csv"),
                "--missing-log",
                str(metrics / "tm_missing.txt"),
            ]
        )
    if not (metrics / "lddt_ca.csv").exists():
        run_cmd(
            [
                PYTHON,
                str(REPO / "scripts" / "lddt_ca_from_pdb.py"),
                str(cand / "predictions"),
                "--native-dir",
                str(native),
                "--out-csv",
                str(metrics / "lddt_ca.csv"),
                "--missing-log",
                str(metrics / "lddt_missing.txt"),
            ]
        )
    paired = cand / "paired_deltas.csv"
    run_cmd(
        [
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
            str(metrics / "tm_score.csv"),
            "--baseline-lddt-csv",
            str(difficulty),
            "--model-lddt-csv",
            str(metrics / "lddt_ca.csv"),
            "--client-id",
            "client_0",
            "--out",
            str(paired),
        ]
    )
    rows = list(csv.DictReader(paired.open()))
    for row in rows:
        row["delta_tm"] = float(row["delta_tm"])
        row["tm_baseline"] = float(row["tm_baseline"])
        row["tm_model"] = float(row["tm_model"])
        if row.get("delta_lddt_ca") not in ("", None):
            row["delta_lddt_ca"] = float(row["delta_lddt_ca"])
    hard = [r for r in rows if r["difficulty"] == "hard"]
    summary = summarize_group(hard, "hard_")
    summary.update(summarize_group(rows, "all_"))
    development_gate = promotion_gate_v2_from_rows(
        rows,
        expected_labels=expected_labels,
        expected_clusters=expected_clusters,
    )
    development_pass = bool(development_gate.get("promotion_pass"))
    write_json(cand / "development_gate_v2.json", development_gate)
    payload = {
        "stage": "evaluate-dev",
        "blocked": False,
        "status": "complete" if development_pass else "failed",
        "development_pass": development_pass,
        "target_slug": slug,
        "paired_csv": str(paired),
        "development_gate_v2": development_gate,
        "blocks_other_clients": not development_pass,
        "blocks_fedlora": True,
        "test_accessed": False,
        "external_final_test_accessed": False,
        **summary,
    }
    write_json(ablation_root(run_root) / "development_summary.json", payload)
    write_json(eval_root / "development_summary.json", payload)
    update_status(
        run_root,
        evaluate_dev="complete" if development_pass else "failed",
        development_pass=development_pass,
        blocks_other_clients=not development_pass,
        blocks_fedlora=True,
        test_accessed=False,
    )
    return payload


def stage_report(run_root: Path, eval_root: Path) -> dict:
    from scripts.summarize_client0_target_ablation import summarize

    return summarize(run_root=run_root, eval_root=eval_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "verify",
            "build-overfit",
            "overfit",
            "diagnose-capacity",
            "audit-9u78",
            "evaluate-t5-trajectory",
            "audit-t5-seeds",
            "reeval-capacity",
            "run-positive-control",
            "evaluate-positive-control-trajectory",
            "evaluate-positive-control-ema-trajectory",
            "run-unclamped-positive-control",
            "validate",
            "validate-t5",
            "confirm-seeds",
            "evaluate-dev",
            "report",
        ],
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--force-positive-control", action="store_true")
    args = parser.parse_args()

    args.eval_root.mkdir(parents=True, exist_ok=True)
    ablation_root(args.run_root).mkdir(parents=True, exist_ok=True)

    if args.stage == "verify":
        out = stage_verify(args.run_root, args.eval_root)
    elif args.stage == "build-overfit":
        out = stage_build_overfit(args.run_root, args.source_run, args.eval_root)
    elif args.stage == "overfit":
        out = stage_overfit(
            args.run_root, args.source_run, args.eval_root, smoke=args.smoke, gpu_id=args.gpu_id
        )
    elif args.stage == "diagnose-capacity":
        out = stage_diagnose_capacity(
            args.run_root, args.source_run, args.eval_root, smoke=args.smoke, gpu_id=args.gpu_id
        )
    elif args.stage == "audit-9u78":
        out = stage_audit_9u78(
            args.run_root,
            args.eval_root,
            gpu_id=args.gpu_id,
            skip_inference=args.skip_inference,
        )
    elif args.stage == "evaluate-t5-trajectory":
        out = stage_evaluate_t5_trajectory(
            args.run_root, args.eval_root, gpu_id=args.gpu_id
        )
    elif args.stage == "audit-t5-seeds":
        out = stage_audit_t5_seeds(
            args.run_root, args.source_run, args.eval_root, gpu_id=args.gpu_id
        )
    elif args.stage == "reeval-capacity":
        out = stage_reeval_capacity(args.run_root, args.eval_root)
    elif args.stage == "evaluate-positive-control-trajectory":
        out = stage_evaluate_positive_control_trajectory(
            args.run_root, args.eval_root, gpu_id=args.gpu_id
        )
    elif args.stage == "evaluate-positive-control-ema-trajectory":
        out = stage_evaluate_positive_control_trajectory(
            args.run_root,
            args.eval_root,
            gpu_id=args.gpu_id,
            weights_source="ema",
        )
    elif args.stage == "run-unclamped-positive-control":
        out = stage_run_unclamped_positive_control(
            args.run_root,
            args.source_run,
            args.eval_root,
            gpu_id=args.gpu_id,
        )
    elif args.stage == "run-positive-control":
        out = stage_run_positive_control(
            args.run_root,
            args.source_run,
            args.eval_root,
            gpu_id=args.gpu_id,
            force=args.force_positive_control,
        )
    elif args.stage in {"validate", "validate-t5"}:
        diagnostic_only = args.diagnostic_only or args.stage == "validate-t5" and args.diagnostic_only
        if args.stage == "validate-t5" and not args.diagnostic_only:
            # validate-t5 without flag still uses resolver; with --diagnostic-only forces branch.
            diagnostic_only = False
        if args.stage == "validate-t5" and args.diagnostic_only:
            diagnostic_only = True
        out = stage_validate(
            args.run_root,
            args.source_run,
            args.eval_root,
            smoke=args.smoke,
            gpu_id=args.gpu_id,
            diagnostic_only=diagnostic_only,
        )
    elif args.stage == "confirm-seeds":
        out = stage_confirm_seeds(
            args.run_root, args.source_run, args.eval_root, smoke=args.smoke, gpu_id=args.gpu_id
        )
    elif args.stage == "evaluate-dev":
        out = stage_evaluate_dev(
            args.run_root, args.eval_root, smoke=args.smoke, gpu_id=args.gpu_id
        )
    else:
        out = stage_report(args.run_root, args.eval_root)

    print(json.dumps(out if isinstance(out, dict) else {"ok": True}, indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
