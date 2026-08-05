"""Generate, but do not execute, the reproducible client1 LoRA v2 matrix."""

import csv
import shlex
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data/all_pdb_1y/client1_res/lora_search_ema_fp32_v2"
PYTHON = Path("/home/test/miniconda3/envs/Fedfold/bin/python")
BASE = REPO / "openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
TARGET = (
    "structure_module.ipa,structure_module.transition,"
    "structure_module.bb_update"
)
LEARNING_RATES = (3e-6, 1e-5, 3e-5)
STEPS = (5, 10, 20, 40, 80)
SEEDS = (42, 43, 44)
SCALES = (0, 0.1, 0.25, 0.5, 0.75, 1)


def shell_join(command):
    return " ".join(shlex.quote(str(value)) for value in command)


def build_train_command(run_dir, learning_rate, steps, seed):
    return [
        PYTHON,
        REPO / "train_openfold.py",
        REPO / "data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune",
        REPO / "data/all_pdb_1y/fed_split/client_1/solo_alignment",
        REPO / "data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune",
        run_dir,
        "2026-01-01",
        "--train_filter_path", ROOT / "data/search_train_labels.txt",
        "--use_single_seq_mode", "True",
        "--config_preset", "seq_model_esm1b_ptm",
        "--experiment_config_json",
        REPO / "seq_model_esm1b_ptm_finetune_override.json",
        "--resume_from_ckpt", BASE,
        "--resume_model_weights_only", "True",
        "--init_weights_source", "ema",
        "--template_release_dates_cache_path",
        REPO / "data/all_pdb_1y/fed_split/client_1/solo_mmcif_cache_finetune.json",
        "--train_chain_data_cache_path",
        REPO / "data/all_pdb_1y/fed_test_v2/client_1/train_chain_data_cache.json",
        "--lora_rank", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0",
        "--lora_target", TARGET,
        "--learning_rate", str(learning_rate),
        "--lr_warmup_steps", str(min(5, steps)),
        "--accumulate_grad_batches", "1",
        "--train_epoch_len", str(steps),
        "--max_epochs", "1",
        "--max_steps", str(steps),
        "--checkpoint_every_n_train_steps", str(steps),
        "--sampling_audit_path", run_dir / "sampling_audit.jsonl",
        "--sampling_cluster_file", REPO / "data/all_pdb_1y/clusters_30.txt",
        "--precision", "bf16-mixed",
        "--gpus", "1",
        "--seed", str(seed),
        "--deepspeed_config_path", REPO / "deepspeed_config.json",
    ]


def build_export_command(checkpoint, output, scale):
    return [
        PYTHON,
        REPO / "scripts/export_lora_checkpoint.py",
        "--input", checkpoint,
        "--output", output,
        "--base-checkpoint", BASE,
        "--base-weights-source", "ema",
        "--adapter-weights-source", "model",
        "--lora-scale", str(scale),
        "--config-preset", "seq_model_esm1b_ptm",
    ]


def generate(output_root=ROOT):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [{
        "run_id": "baseline",
        "candidate_type": "baseline",
        "learning_rate": "",
        "step": 0,
        "seed": 42,
        "lora_scale": 0,
        "status": "pending",
        "train_command": "",
        "export_command": "",
    }]
    train_commands = []
    for learning_rate in LEARNING_RATES:
        lr_name = f"{learning_rate:.0e}".replace("-", "m")
        for steps in STEPS:
            for seed in SEEDS:
                run_id = f"core_r4_a8_lr{lr_name}_step{steps}_seed{seed}"
                run_dir = output_root / "runs" / run_id
                checkpoint = run_dir / "checkpoints" / f"0-{steps}.ckpt"
                train = build_train_command(
                    run_dir,
                    learning_rate,
                    steps,
                    seed,
                )
                train_commands.append(
                    f'test -d {shlex.quote(str(checkpoint))} || '
                    + shell_join(train)
                )
                for scale in SCALES:
                    output = (
                        output_root
                        / "exports"
                        / run_id
                        / f"scale_{scale:g}.pt"
                    )
                    rows.append({
                        "run_id": run_id,
                        "candidate_type": "lora",
                        "learning_rate": learning_rate,
                        "step": steps,
                        "seed": seed,
                        "lora_scale": scale,
                        "status": "pending",
                        "train_command": shell_join(train),
                        "export_command": shell_join(
                            build_export_command(checkpoint, output, scale)
                        ),
                    })
    manifest = output_root / "experiment_matrix.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    commands = output_root / "train_commands.sh"
    commands.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n".join(train_commands)
        + "\n",
        encoding="utf-8",
    )
    return manifest, commands, rows


if __name__ == "__main__":
    manifest, commands, rows = generate()
    print(f"Wrote {len(rows)} matrix rows to {manifest}")
    print(f"Wrote {len(LEARNING_RATES) * len(STEPS) * len(SEEDS)} "
          f"resumable training commands to {commands}")
