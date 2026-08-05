#!/usr/bin/env bash
# Reproducible five-client FP32 LoRA pipeline.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-$REPO/data/all_pdb_1y}"
INPUT_ROOT="${INPUT_ROOT:-$DATA_ROOT/fed_split}"
RUN_ROOT="${RUN_ROOT:-$REPO/outputs/fed_lora_fp32_v1}"
CLIENTS_ROOT="$RUN_ROOT/clients"
SPLIT_ROOT="$RUN_ROOT/split"
MODELS_ROOT="$RUN_ROOT/models"
EVAL_ROOT="$RUN_ROOT/evaluation"
BASE_CKPT="${BASE_CKPT:-$REPO/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt}"
CLUSTERS="${CLUSTERS:-$DATA_ROOT/clusters_30.txt}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
LR="${LR:-1e-4}"
LORA_SCALE="${LORA_SCALE:-1.0}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

check_common() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
  test -f "$CLUSTERS" || die "Missing cluster file: $CLUSTERS"
  mkdir -p "$RUN_ROOT"
  if [[ ! -f "$BASE_CKPT" ]]; then
    echo "Downloading public SoloSeq checkpoint (about 605 MB)"
    mkdir -p "$(dirname "$BASE_CKPT")"
    "$PYTHON_BIN" - "$BASE_CKPT" <<'PY'
import sys
import urllib.request
from pathlib import Path

output = Path(sys.argv[1])
temporary = output.with_suffix(output.suffix + ".part")
url = (
    "https://openfold.s3.amazonaws.com/openfold_soloseq_params/"
    "seq_model_esm1b_ptm.pt"
)
urllib.request.urlretrieve(url, temporary)
temporary.replace(output)
PY
  fi
  test -f "$REPO/deepspeed_config_fp32.json" ||
    die "Missing deepspeed_config_fp32.json"
  test -x "$REPO/tmscore/TMscore" || die "Missing executable tmscore/TMscore"
  if [[ ! -f "$RUN_ROOT/run_manifest.json" ]]; then
    "$PYTHON_BIN" - \
      "$RUN_ROOT/run_manifest.json" "$REPO" "$DATA_ROOT" "$INPUT_ROOT" \
      "$CLUSTERS" "$SEED" "$MAX_EPOCHS" "$LR" "$LORA_SCALE" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, repo, data_root, input_root, clusters,
    seed, max_epochs, learning_rate, lora_scale,
) = sys.argv[1:]
try:
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    commit = None
cluster_path = Path(clusters)
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": commit,
    "data_root": str(Path(data_root).resolve()),
    "input_root": str(Path(input_root).resolve()),
    "cluster_file": str(cluster_path.resolve()),
    "cluster_file_sha256": hashlib.sha256(cluster_path.read_bytes()).hexdigest(),
    "split": {"method": "fixed_cluster_holdout", "seed": int(seed)},
    "training": {
        "precision": "fp32",
        "max_epochs": int(max_epochs),
        "learning_rate": float(learning_rate),
        "lora_scale": float(lora_scale),
        "rank": 4,
        "alpha": 8,
        "dropout": 0,
        "target": [
            "structure_module.ipa",
            "structure_module.transition",
            "structure_module.bb_update",
        ],
    },
}
Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
  fi
}

client_ids() {
  local requested="${1:-all}"
  if [[ "$requested" == "all" ]]; then
    seq 0 $((NUM_CLIENTS - 1))
  elif [[ "$requested" =~ ^[0-9]+$ ]] &&
       (( requested >= 0 && requested < NUM_CLIENTS )); then
    echo "$requested"
  else
    die "Client must be one of 0..$((NUM_CLIENTS - 1)) or all"
  fi
}

prepare_client() {
  local id="$1"
  local client="client_$id"
  local input_fasta="$INPUT_ROOT/$client/solo_fasta_dir"
  local root="$CLIENTS_ROOT/$client"
  local mmcif="$root/mmcif_files"
  local alignment="$root/solo_alignment"
  local prescreen="$root/prescreen"
  local reports="$root/reports"
  local fasta_count embedding_count

  test -d "$input_fasta" || die "Missing Git FASTA directory: $input_fasta"
  mkdir -p "$root" "$reports"
  if [[ ! -e "$root/solo_fasta_dir" ]]; then
    ln -s "$(realpath "$input_fasta")" "$root/solo_fasta_dir"
  fi

  echo "[$client] download mmCIF referenced by FASTA"
  "$PYTHON_BIN" "$REPO/scripts/download_mmcif_from_fasta.py" \
    "$input_fasta" "$mmcif"

  fasta_count="$(find "$input_fasta" -maxdepth 1 -name '*.fasta' | wc -l)"
  if [[ -d "$alignment" ]]; then
    embedding_count="$(find "$alignment" -mindepth 2 -maxdepth 2 \
      -name '*.pt' | wc -l)"
  else
    embedding_count=0
  fi
  if (( embedding_count != fasta_count )); then
    echo "[$client] precompute ESM-1b embeddings"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
      "$REPO/scripts/precompute_embeddings.py" "$input_fasta" "$alignment"
  fi

  if [[ ! -f "$reports/low_plddt.csv" ]]; then
    echo "[$client] FP32 baseline prescreen"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
      "$REPO/run_pretrained_openfold.py" \
      "$input_fasta" "$mmcif" \
      --use_precomputed_alignments "$alignment" \
      --use_single_seq_mode \
      --output_dir "$prescreen" \
      --model_device cuda:0 \
      --skip_relaxation \
      --config_preset seq_model_esm1b_ptm \
      --openfold_checkpoint_path "$BASE_CKPT" \
      --checkpoint_weights_source ema \
      --data_random_seed "$SEED" \
      --precision fp32

    "$PYTHON_BIN" "$REPO/scripts/plddt_from_pdb.py" \
      "$prescreen/predictions" \
      --no-extremes \
      --select-threshold 80 \
      --selected-csv "$reports/low_plddt.csv" \
      --bad-log "$reports/prescreen_bad_pdb.txt"
  fi

  echo "[$client] build fine-tune inputs and caches"
  "$PYTHON_BIN" "$REPO/scripts/extract_selected_mmcif.py" \
    --csv-path "$reports/low_plddt.csv" \
    --source-dir "$mmcif" \
    --dst-dir "$root/mmcif_files_finetune"

  "$PYTHON_BIN" "$REPO/scripts/prep_solo_fasta_learnable.py" \
    "$root/mmcif_files_finetune" \
    "$root/solo_fasta_dir_finetune" \
    --target_labels_file "$reports/low_plddt.csv" \
    --max_len 1022 \
    --dedup_by_sequence

  "$PYTHON_BIN" "$REPO/scripts/generate_mmcif_cache.py" \
    "$root/mmcif_files_finetune" "$root/mmcif_cache_finetune.json" \
    --no_workers 8
  "$PYTHON_BIN" "$REPO/scripts/generate_chain_data_cache.py" \
    "$root/mmcif_files_finetune" "$root/chain_data_cache_finetune.json" \
    --cluster_file "$CLUSTERS" \
    --no_workers 8
}

prepare() {
  check_common
  local id
  while read -r id; do
    prepare_client "$id"
  done < <(client_ids "${1:-all}")
}

build_split() {
  check_common
  local id root
  for id in $(client_ids all); do
    root="$CLIENTS_ROOT/client_$id"
    test -f "$root/solo_fasta_dir_finetune/_reports/kept_labels.txt" ||
      die "Prepare client_$id first"
    test -f "$root/chain_data_cache_finetune.json" ||
      die "Missing client_$id chain cache"
  done

  if [[ -e "$SPLIT_ROOT" ]]; then
    test -f "$SPLIT_ROOT/split_manifest.json" ||
      die "Incomplete split directory exists: $SPLIT_ROOT"
    echo "Split already exists: $SPLIT_ROOT"
    return
  fi

  "$PYTHON_BIN" "$REPO/scripts/build_fed_test_set.py" \
    --fed-split-dir "$CLIENTS_ROOT" \
    --num-clients "$NUM_CLIENTS" \
    --cluster-file "$CLUSTERS" \
    --out-dir "$SPLIT_ROOT" \
    --test-frac 0.2 \
    --seed "$SEED" \
    --min-test-clusters 5 \
    --fasta-subdir solo_fasta_dir \
    --emb-subdir solo_alignment \
    --cif-subdir mmcif_files \
    --chain-cache-subpath chain_data_cache_finetune.json

  "$PYTHON_BIN" "$REPO/scripts/extract_native_chain_pdbsV2.py" \
    --cif-dir "$SPLIT_ROOT/all/mmcif_files" \
    --out-dir "$SPLIT_ROOT/all/native" \
    --labels-txt "$SPLIT_ROOT/all/test_labels.txt" \
    --report-csv "$SPLIT_ROOT/all/native_report.csv" \
    --fail-log "$SPLIT_ROOT/all/native_failures.txt"
}

train_client() {
  local id="$1"
  local client="client_$id"
  local data="$CLIENTS_ROOT/$client"
  local model="$MODELS_ROOT/$client"
  local training="$model/training"
  local labels="$SPLIT_ROOT/$client/train_labels.txt"
  local epoch_len total_steps adapter merged verify

  test -f "$labels" || die "Run split before training"
  epoch_len="$(awk 'NF {count++} END {print count+0}' "$labels")"
  (( epoch_len > 0 )) || die "Empty train split for $client"
  total_steps=$((epoch_len * MAX_EPOCHS))
  adapter="$training/checkpoints/$((MAX_EPOCHS - 1))-$total_steps.ckpt"
  merged="$model/merged_ema_fp32.pt"
  verify="$model/roundtrip.json"

  if [[ ! -d "$adapter" ]]; then
    if [[ -e "$training" ]]; then
      die "Partial training exists for $client; inspect or move $training"
    fi
    mkdir -p "$training"
    echo "[$client] FP32 LoRA training: $MAX_EPOCHS epochs, $epoch_len draws/epoch"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
      "$REPO/train_openfold.py" \
      "$data/mmcif_files_finetune" \
      "$data/solo_alignment" \
      "$data/mmcif_files_finetune" \
      "$training" \
      2026-01-01 \
      --train_filter_path "$labels" \
      --use_single_seq_mode True \
      --config_preset seq_model_esm1b_ptm \
      --experiment_config_json "$REPO/seq_model_esm1b_ptm_finetune_override.json" \
      --resume_from_ckpt "$BASE_CKPT" \
      --resume_model_weights_only True \
      --init_weights_source ema \
      --template_release_dates_cache_path "$data/mmcif_cache_finetune.json" \
      --train_chain_data_cache_path "$SPLIT_ROOT/$client/train_chain_data_cache.json" \
      --lora_rank 4 \
      --lora_alpha 8 \
      --lora_dropout 0 \
      --lora_target \
        structure_module.ipa,structure_module.transition,structure_module.bb_update \
      --learning_rate "$LR" \
      --lr_warmup_steps 20 \
      --accumulate_grad_batches 1 \
      --train_epoch_len "$epoch_len" \
      --max_epochs "$MAX_EPOCHS" \
      --checkpoint_every_epoch \
      --sampling_audit_path "$model/sampling_audit.jsonl" \
      --sampling_cluster_file "$CLUSTERS" \
      --precision 32 \
      --gpus 1 \
      --seed "$SEED" \
      --deepspeed_config_path "$REPO/deepspeed_config_fp32.json" \
      2>&1 | tee "$model/train.log"
  fi

  if [[ ! -f "$merged" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/verify_lora_roundtrip.py" \
      --adapter-checkpoint "$adapter" \
      --base-checkpoint "$BASE_CKPT" \
      --base-weights-source ema \
      --adapter-weights-source model \
      --lora-scale "$LORA_SCALE" \
      --config-preset seq_model_esm1b_ptm \
      --output-json "$verify"

    "$PYTHON_BIN" "$REPO/scripts/export_lora_checkpoint.py" \
      --input "$adapter" \
      --output "$merged" \
      --base-checkpoint "$BASE_CKPT" \
      --base-weights-source ema \
      --adapter-weights-source model \
      --lora-scale "$LORA_SCALE" \
      --config-preset seq_model_esm1b_ptm
  fi
}

train_models() {
  check_common
  local id
  while read -r id; do
    train_client "$id"
  done < <(client_ids "${1:-all}")
}

infer_model() {
  local name="$1"
  local checkpoint="$2"
  local source="$3"
  local output="$EVAL_ROOT/$name"
  local expected actual
  expected="$(awk 'NF {count++} END {print count+0}' \
    "$SPLIT_ROOT/all/test_labels.txt")"
  if [[ -d "$output/predictions" ]]; then
    actual="$(find "$output/predictions" -maxdepth 1 \
      -name '*_unrelaxed.pdb' | wc -l)"
  else
    actual=0
  fi
  if (( actual != expected )); then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
      "$REPO/run_pretrained_openfold.py" \
      "$SPLIT_ROOT/all/solo_fasta_dir" \
      "$SPLIT_ROOT/all/mmcif_files" \
      --use_precomputed_alignments "$SPLIT_ROOT/all/solo_alignment_dir" \
      --use_single_seq_mode \
      --output_dir "$output" \
      --model_device cuda:0 \
      --skip_relaxation \
      --config_preset seq_model_esm1b_ptm \
      --openfold_checkpoint_path "$checkpoint" \
      --checkpoint_weights_source "$source" \
      --data_random_seed "$SEED" \
      --precision fp32
  fi

  mkdir -p "$output/metrics"
  "$PYTHON_BIN" "$REPO/scripts/tmscore_from_pdb.py" \
    "$output/predictions" \
    --native-dir "$SPLIT_ROOT/all/native" \
    --tm-exec "$REPO/tmscore/TMscore" \
    --out-csv "$output/metrics/tm_score.csv" \
    --selected-csv "$output/metrics/low_tm.csv" \
    --missing-log "$output/metrics/tm_missing.txt"
  "$PYTHON_BIN" "$REPO/scripts/plddt_from_pdb.py" \
    "$output/predictions" \
    --no-extremes \
    --select-threshold 101 \
    --selected-csv "$output/metrics/plddt.csv" \
    --bad-log "$output/metrics/plddt_bad.txt"
  "$PYTHON_BIN" "$REPO/scripts/lddt_ca_from_pdb.py" \
    "$output/predictions" \
    --native-dir "$SPLIT_ROOT/all/native" \
    --out-csv "$output/metrics/lddt_ca.csv" \
    --missing-log "$output/metrics/lddt_ca_missing.txt"
}

evaluate() {
  check_common
  test -f "$SPLIT_ROOT/all/test_labels.txt" || die "Run split first"
  infer_model baseline "$BASE_CKPT" ema

  local id checkpoint
  for id in $(client_ids all); do
    checkpoint="$MODELS_ROOT/client_$id/merged_ema_fp32.pt"
    test -f "$checkpoint" || die "Train client_$id first"
    infer_model "client_$id" "$checkpoint" auto
  done

  "$PYTHON_BIN" - "$EVAL_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    tm_path = model_dir / "metrics" / "tm_score.csv"
    plddt_path = model_dir / "metrics" / "plddt.csv"
    lddt_path = model_dir / "metrics" / "lddt_ca.csv"
    if not tm_path.exists() or not plddt_path.exists() or not lddt_path.exists():
        continue
    tm = list(csv.DictReader(tm_path.open(encoding="utf-8")))
    plddt = list(csv.DictReader(plddt_path.open(encoding="utf-8")))
    lddt = list(csv.DictReader(lddt_path.open(encoding="utf-8")))
    if not (len(tm) == len(plddt) == len(lddt)):
        raise ValueError(f"{model_dir.name}: metric row counts differ")
    rows.append({
        "model": model_dir.name,
        "n": len(tm),
        "mean_tm": sum(float(row["tm_selected"]) for row in tm) / len(tm),
        "mean_lddt_ca": sum(float(row["lddt_ca"]) for row in lddt) / len(lddt),
        "mean_plddt": sum(float(row["mean_plddt"]) for row in plddt) / len(plddt),
    })
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(root / "summary.csv")
PY
}

usage() {
  cat <<EOF
Usage:
  bash scripts/fed_lora_fp32.sh prepare [0..4|all]
  bash scripts/fed_lora_fp32.sh split
  bash scripts/fed_lora_fp32.sh train [0..4|all]
  bash scripts/fed_lora_fp32.sh evaluate
  bash scripts/fed_lora_fp32.sh all

Optional environment variables:
  PYTHON_BIN=python3 GPU_ID=0 RUN_ROOT=/path/to/output
EOF
}

command="${1:-}"
case "$command" in
  prepare) prepare "${2:-all}" ;;
  split) build_split ;;
  train) train_models "${2:-all}" ;;
  evaluate) evaluate ;;
  all)
    prepare all
    build_split
    train_models all
    evaluate
    ;;
  *) usage; exit 2 ;;
esac
