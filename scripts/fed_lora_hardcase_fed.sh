#!/usr/bin/env bash
# Hard-aware FedFold / FedLoRA orchestration (single-machine simulation).
#
# Output root: outputs/fed_lora_hardcase_fed_v1/
# Never writes into outputs/fed_lora_fp32/ or outputs/fed_lora_hardcase_v1/.
#
# Usage examples:
#   bash scripts/fed_lora_hardcase_fed.sh prepare
#   bash scripts/fed_lora_hardcase_fed.sh smoke
#   SEEDS=42 ARMS=uniform,hard_aware_70 ROUNDS=5 \
#     bash scripts/fed_lora_hardcase_fed.sh train
#   bash scripts/fed_lora_hardcase_fed.sh validate
#   bash scripts/fed_lora_hardcase_fed.sh select_global
#   bash scripts/fed_lora_hardcase_fed.sh evaluate_dev
#   bash scripts/fed_lora_hardcase_fed.sh lock_final
#   bash scripts/fed_lora_hardcase_fed.sh evaluate_final
#   bash scripts/fed_lora_hardcase_fed.sh report

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-$SEED}"
ARMS="${ARMS:-uniform,hard_aware_70}"
ROUNDS="${ROUNDS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LOCAL_ONLY_EPOCHS="${LOCAL_ONLY_EPOCHS:-5}"
LOCAL_EVAL_EPOCHS="${LOCAL_EVAL_EPOCHS:-all}"
LOCAL_SCALES="${LOCAL_SCALES:-0.25,0.5,0.75,1.0}"
PRIMARY_EVAL_SCALE="${PRIMARY_EVAL_SCALE:-1.0}"
LOCAL_CLIENTS="${LOCAL_CLIENTS:-all}"
FED_VALIDATION_CLIENTS="${FED_VALIDATION_CLIENTS:-all}"
FED_VALIDATION_ROUNDS="${FED_VALIDATION_ROUNDS:-all}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"
LR="${LR:-1e-4}"
RANK="${RANK:-4}"
ALPHA="${ALPHA:-8}"
DROPOUT="${DROPOUT:-0}"
LORA_TARGET="${LORA_TARGET:-structure_module.ipa,structure_module.transition,structure_module.bb_update}"
EXPERIMENT_CONFIG_JSON="${EXPERIMENT_CONFIG_JSON:-$REPO/seq_model_esm1b_ptm_finetune_override.json}"
BASELINE_PRESERVATION_WEIGHT="${BASELINE_PRESERVATION_WEIGHT:-0}"
HARD_SOFT_TM_WEIGHT="${HARD_SOFT_TM_WEIGHT:-0}"
HARD_SOFT_TM_MARGIN="${HARD_SOFT_TM_MARGIN:-0.02}"
TARGET_SLUG="${TARGET_SLUG:-}"
LOCAL_OUTPUT_NAMESPACE="${LOCAL_OUTPUT_NAMESPACE:-}"
MAX_UPDATE_NORM="${MAX_UPDATE_NORM:-}"
BASE_CKPT="${BASE_CKPT:-$REPO/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt}"
CLUSTERS="${CLUSTERS:-$REPO/data/all_pdb_1y/clusters_30.txt}"
SOURCE_RUN="${SOURCE_RUN:-$REPO/outputs/fed_lora_fp32}"
RUN_ROOT="${RUN_ROOT:-$REPO/outputs/fed_lora_hardcase_fed_v1}"
SERVER_ROOT="$RUN_ROOT/server"
CLIENTS_ROOT="$RUN_ROOT/clients"
SPLITS_ROOT="$RUN_ROOT/splits"
LOG_ROOT="$RUN_ROOT/logs"
REPORT_ROOT="$RUN_ROOT/reports"
EVAL_ROOT="$RUN_ROOT/evaluation"
THREAT_MODEL="${THREAT_MODEL:-simulation}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

positive_float() {
  awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

check_common() {
  test -d "$REPO" || die "Missing repo $REPO"
  test -f "$BASE_CKPT" || die "Missing base checkpoint $BASE_CKPT"
  test -f "$CLUSTERS" || die "Missing cluster file $CLUSTERS"
  test -f "$EXPERIMENT_CONFIG_JSON" || \
    die "Missing experiment config $EXPERIMENT_CONFIG_JSON"
  test -x "$REPO/tmscore/TMscore" || die "Missing tmscore/TMscore"
  mkdir -p "$RUN_ROOT" "$SERVER_ROOT" "$CLIENTS_ROOT" "$SPLITS_ROOT" \
    "$LOG_ROOT" "$REPORT_ROOT" "$EVAL_ROOT"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

write_run_manifest() {
  local git_commit dirty_py
  git_commit="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  if git -C "$REPO" status --porcelain 2>/dev/null | grep -q .; then
    dirty_py=True
  else
    dirty_py=False
  fi
  "$PYTHON_BIN" - "$RUN_ROOT/run_manifest.json" <<PY
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
manifest = {
  "git_commit": "$git_commit",
  "git_dirty": $dirty_py,
  "run_root": "$RUN_ROOT",
  "source_run": "$(readlink -f "$SOURCE_RUN" 2>/dev/null || echo "$SOURCE_RUN")",
  "base_checkpoint": "$(readlink -f "$BASE_CKPT")",
  "base_checkpoint_sha256": "$(sha256_file "$BASE_CKPT")",
  "cluster_file": "$(readlink -f "$CLUSTERS")",
  "cluster_file_sha256": "$(sha256_file "$CLUSTERS")",
  "threat_model": "$THREAT_MODEL",
  "shared_config": {
    "rank": int("$RANK"),
    "alpha": float("$ALPHA"),
    "dropout": float("$DROPOUT"),
    "learning_rate": float("$LR"),
    "target": [x for x in "$LORA_TARGET".split(",") if x],
    "local_epochs_per_round": int("$LOCAL_EPOCHS"),
    "experiment_config_json": "$(readlink -f "$EXPERIMENT_CONFIG_JSON")",
    "experiment_config_sha256": "$(sha256_file "$EXPERIMENT_CONFIG_JSON")",
    "baseline_preservation_weight": float("$BASELINE_PRESERVATION_WEIGHT"),
    "hard_soft_tm_weight": float("$HARD_SOFT_TM_WEIGHT"),
    "hard_soft_tm_margin": float("$HARD_SOFT_TM_MARGIN"),
    "precision": "fp32",
    "aggregation": "sample_weighted_effective_delta",
    "adapter_source": "model",
    "aggregation_scale": 1.0,
  },
  "arms": [x for x in "$ARMS".split(",") if x],
  "seeds": [int(x) for x in "$SEEDS".split(",") if x],
  "rounds": int("$ROUNDS"),
  "final_test_lock_state": "unlocked",
  "privacy_notes": [
    "simulation only unless threat_model overridden",
    "server must not read client-private paired CSVs",
    "secure aggregation interface exists but plaintext average is used in MVP",
  ],
}
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
print(path)
PY
}

prepare() {
  check_common
  write_run_manifest
  echo "[prepare] building baseline difficulty"
  "$PYTHON_BIN" "$REPO/scripts/build_baseline_difficulty.py" \
    --run-root "$RUN_ROOT" \
    --source-run "$SOURCE_RUN" \
    --cluster-file "$CLUSTERS" \
    --num-clients "$NUM_CLIENTS" \
    --python-bin "$PYTHON_BIN" \
    --tm-exec "$REPO/tmscore/TMscore"

  echo "[prepare] building hardcase splits"
  "$PYTHON_BIN" "$REPO/scripts/build_hardcase_splits.py" \
    --run-root "$RUN_ROOT" \
    --source-run "$SOURCE_RUN" \
    --cluster-file "$CLUSTERS" \
    --num-clients "$NUM_CLIENTS" \
    --seed "$SEED" \
    --link

  # Initialize round_000 global model from public base (pure weights).
  local round0="$SERVER_ROOT/rounds/round_000"
  mkdir -p "$round0"
  if [[ ! -f "$round0/global_model.pt" ]]; then
    "$PYTHON_BIN" - <<PY
import torch, pathlib
src = pathlib.Path("$BASE_CKPT")
dst = pathlib.Path("$round0/global_model.pt")
obj = torch.load(str(src), map_location="cpu", weights_only=False)
# Official OpenFold packs may be nested; keep tensor mapping only.
if isinstance(obj, dict) and "ema" in obj and isinstance(obj["ema"], dict) and "params" in obj["ema"]:
    params = obj["ema"]["params"]
elif isinstance(obj, dict) and all(hasattr(v, "shape") for v in obj.values() if True):
    params = {k: v for k, v in obj.items() if hasattr(v, "detach")}
else:
    params = obj
torch.save(params, str(dst))
print("initialized", dst)
PY
  fi
  cat > "$SERVER_ROOT/config.json" <<EOF
{
  "rank": $RANK,
  "alpha": $ALPHA,
  "dropout": $DROPOUT,
  "learning_rate": $LR,
  "local_epochs_per_round": $LOCAL_EPOCHS,
  "target": "$LORA_TARGET",
  "experiment_config_json": "$(readlink -f "$EXPERIMENT_CONFIG_JSON")",
  "experiment_config_sha256": "$(sha256_file "$EXPERIMENT_CONFIG_JSON")",
  "baseline_preservation_weight": $BASELINE_PRESERVATION_WEIGHT,
  "hard_soft_tm_weight": $HARD_SOFT_TM_WEIGHT,
  "hard_soft_tm_margin": $HARD_SOFT_TM_MARGIN,
  "aggregation": "sample_weighted_effective_delta",
  "adapter_source": "model",
  "threat_model": "$THREAT_MODEL"
}
EOF
  write_privacy_audit_stub
  echo "[prepare] done -> $RUN_ROOT"
}

write_privacy_audit_stub() {
  mkdir -p "$REPORT_ROOT"
  cat > "$REPORT_ROOT/privacy_audit.md" <<EOF
# Privacy audit (MVP)

- Threat model: \`$THREAT_MODEL\`
- Single-machine layout separates \`server/\` and \`clients/*/private/\`.
- Server selection code must consume only validation summary JSON (no labels/native/paired CSV).
- Uploaded objects are raw LoRA updates / round-end checkpoints, not sequences or native PDBs.
- Update clipping supported via \`MAX_UPDATE_NORM\`.
- \`scripts/fedlora_aggregate.py:secure_aggregate_placeholder\` is the secure-aggregation hook; MVP uses trusted weighted average.
- Formal differential privacy is out of scope for this small-data MVP.
EOF
}

client_ids() {
  local mode="${1:-all}"
  if [[ "$mode" == "all" ]]; then
    seq 0 $((NUM_CLIENTS - 1))
  else
    local -a ids=()
    local id
    IFS=',' read -r -a ids <<< "$mode"
    for id in "${ids[@]}"; do
      [[ "$id" =~ ^[0-9]+$ ]] || die "Invalid client id: $id"
      (( id >= 0 && id < NUM_CLIENTS )) || \
        die "Client id $id outside 0..$((NUM_CLIENTS - 1))"
      echo "$id"
    done
  fi
}

federated_validation_round_ids() {
  local mode="${FED_VALIDATION_ROUNDS:-all}"
  if [[ "$mode" == "all" ]]; then
    seq 0 "$ROUNDS"
  else
    local -a rounds=()
    local round
    IFS=',' read -r -a rounds <<< "$mode"
    for round in "${rounds[@]}"; do
      [[ "$round" =~ ^[0-9]+$ ]] || die "Invalid validation round: $round"
      (( round >= 0 && round <= ROUNDS )) || \
        die "Validation round $round outside 0..$ROUNDS"
      echo "$round"
    done
  fi
}

train_epoch_len_for_client() {
  local id="$1"
  awk 'NF {c++} END {print c+0}' \
    "$CLIENTS_ROOT/client_$id/private/splits/train_labels.txt"
}

local_only_root() {
  local id="$1"
  local arm="$2"
  local seed="$3"
  local base="$CLIENTS_ROOT/client_$id/private"
  # Default keeps the original local_only layout. Target ablation sets
  # LOCAL_OUTPUT_NAMESPACE=target_ablation_v1 and TARGET_SLUG=T0..T4.
  if [[ -n "$LOCAL_OUTPUT_NAMESPACE" ]]; then
    if [[ -n "$TARGET_SLUG" ]]; then
      echo "$base/$LOCAL_OUTPUT_NAMESPACE/$TARGET_SLUG/$arm/seed_$seed"
    else
      echo "$base/$LOCAL_OUTPUT_NAMESPACE/$arm/seed_$seed"
    fi
  elif [[ -n "$TARGET_SLUG" ]]; then
    echo "$base/local_only/$TARGET_SLUG/$arm/seed_$seed"
  else
    echo "$base/local_only/$arm/seed_$seed"
  fi
}

local_checkpoint_path() {
  local id="$1"
  local arm="$2"
  local seed="$3"
  local epoch="$4"
  local epoch_len step epoch_index root
  epoch_len="$(train_epoch_len_for_client "$id")"
  step=$((epoch_len * epoch))
  epoch_index=$((epoch - 1))
  root="$(local_only_root "$id" "$arm" "$seed")"
  echo "$root/training/checkpoints/$epoch_index-$step.ckpt"
}

train_local_client() {
  local arm="$1"
  local seed="$2"
  local id="$3"
  local client="client_$id"
  local private="$CLIENTS_ROOT/$client/private"
  local split_root="$private/splits"
  local data_src="$SOURCE_RUN/clients/$client"
  local parent="$SERVER_ROOT/rounds/round_000/global_model.pt"
  local out epoch_len final_checkpoint baseline_prediction_dir
  local -a extra_train_args=()
  out="$(local_only_root "$id" "$arm" "$seed")"
  epoch_len="$(train_epoch_len_for_client "$id")"
  final_checkpoint="$(local_checkpoint_path "$id" "$arm" "$seed" "$LOCAL_ONLY_EPOCHS")"
  baseline_prediction_dir="$data_src/prescreen/predictions"

  test -f "$split_root/train_labels.txt" || die "Missing local train split for $client"
  test -d "$split_root/assets_validation/mmcif_files" || \
    die "Missing validation assets for $client; run prepare"
  test -f "$EXPERIMENT_CONFIG_JSON" || \
    die "Missing experiment config $EXPERIMENT_CONFIG_JSON"
  if positive_float "$BASELINE_PRESERVATION_WEIGHT" || \
      positive_float "$HARD_SOFT_TM_WEIGHT"; then
    test -d "$baseline_prediction_dir" || \
      die "Missing baseline predictions for $client: $baseline_prediction_dir"
    extra_train_args+=(--baseline_preservation_dir "$baseline_prediction_dir")
  fi
  if positive_float "$BASELINE_PRESERVATION_WEIGHT"; then
    extra_train_args+=(--baseline_preservation_weight "$BASELINE_PRESERVATION_WEIGHT")
  fi
  if positive_float "$HARD_SOFT_TM_WEIGHT"; then
    extra_train_args+=(
      --hard_soft_tm_weight "$HARD_SOFT_TM_WEIGHT"
      --hard_soft_tm_margin "$HARD_SOFT_TM_MARGIN"
    )
  fi
  (( epoch_len > 0 )) || die "Empty local train split for $client"
  if [[ -d "$final_checkpoint" || -f "$final_checkpoint" ]]; then
    echo "[skip local_train] $client arm=$arm seed=$seed already complete"
    return 0
  fi
  if [[ -e "$out/training" ]]; then
    die "Partial local training exists at $out/training; inspect or move it"
  fi
  mkdir -p "$out"
  cat > "$out/local_run.json" <<EOF
{
  "client_id": "$client",
  "arm": "$arm",
  "seed": $seed,
  "parent_path": "$parent",
  "parent_sha256": "$(sha256_file "$parent")",
  "local_epochs": $LOCAL_ONLY_EPOCHS,
  "train_epoch_len": $epoch_len,
  "rank": $RANK,
  "alpha": $ALPHA,
  "dropout": $DROPOUT,
  "learning_rate": $LR,
  "target": "$LORA_TARGET",
  "target_slug": "${TARGET_SLUG}",
  "local_output_namespace": "${LOCAL_OUTPUT_NAMESPACE}",
  "primary_eval_epoch": $LOCAL_ONLY_EPOCHS,
  "primary_eval_scale": $PRIMARY_EVAL_SCALE,
  "eval_epochs": "$LOCAL_EVAL_EPOCHS",
  "experiment_config_json": "$EXPERIMENT_CONFIG_JSON",
  "experiment_config_sha256": "$(sha256_file "$EXPERIMENT_CONFIG_JSON")",
  "baseline_preservation_weight": $BASELINE_PRESERVATION_WEIGHT,
  "hard_soft_tm_weight": $HARD_SOFT_TM_WEIGHT,
  "hard_soft_tm_margin": $HARD_SOFT_TM_MARGIN,
  "baseline_predictions_are_client_local": true,
  "adapter_source": "model",
  "aggregation": false
}
EOF

  echo "[local_train] $client arm=$arm seed=$seed epochs=$LOCAL_ONLY_EPOCHS"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    "$REPO/train_openfold.py" \
    "$data_src/mmcif_files_finetune" \
    "$data_src/solo_alignment" \
    "$data_src/mmcif_files_finetune" \
    "$out/training" \
    2026-01-01 \
    --train_filter_path "$split_root/train_labels.txt" \
    --val_data_dir "$split_root/assets_validation/mmcif_files" \
    --val_alignment_dir "$split_root/assets_validation/solo_alignment_dir" \
    --use_single_seq_mode True \
    --config_preset seq_model_esm1b_ptm \
    --experiment_config_json "$EXPERIMENT_CONFIG_JSON" \
    --resume_from_ckpt "$parent" \
    --resume_model_weights_only True \
    --init_weights_source auto \
    --template_release_dates_cache_path "$data_src/mmcif_cache_finetune.json" \
    --train_chain_data_cache_path "$split_root/train_chain_data_cache.json" \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_dropout "$DROPOUT" \
    --lora_target "$LORA_TARGET" \
    --learning_rate "$LR" \
    --lr_warmup_steps 20 \
    --accumulate_grad_batches 1 \
    --train_epoch_len "$epoch_len" \
    --max_epochs "$LOCAL_ONLY_EPOCHS" \
    --checkpoint_every_epoch \
    --sampling_audit_path "$out/sampling_audit.jsonl" \
    --sampling_cluster_file "$CLUSTERS" \
    --sampling_mode "$arm" \
    --difficulty_csv "$private/difficulty/baseline_difficulty.csv" \
    --hard_aware_ratios "$(hard_aware_ratios_for_arm "$arm")" \
    "${extra_train_args[@]}" \
    --precision 32 \
    --gpus 1 \
    --seed "$seed" \
    --deepspeed_config_path "$REPO/deepspeed_config_fp32.json" \
    2>&1 | tee "$out/train.log"
}

local_train() {
  check_common
  test -f "$SPLITS_ROOT/split_manifest.json" || die "Run prepare first"
  local arm seed id
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for id in $(client_ids "$LOCAL_CLIENTS"); do
        train_local_client "$arm" "$seed" "$id"
      done
    done
  done
}

local_candidate_dir() {
  local id="$1"
  local arm="$2"
  local seed="$3"
  local split="$4"
  local epoch="$5"
  local scale="$6"
  local scale_slug="${scale//./p}"
  echo "$(local_only_root "$id" "$arm" "$seed")/$split/epoch_$epoch/scale_$scale_slug"
}

export_local_candidate() {
  local id="$1"
  local arm="$2"
  local seed="$3"
  local epoch="$4"
  local scale="$5"
  local checkpoint out model
  checkpoint="$(local_checkpoint_path "$id" "$arm" "$seed" "$epoch")"
  out="$(local_candidate_dir "$id" "$arm" "$seed" validation "$epoch" "$scale")"
  model="$out/model_base_global_adapter_raw_scale_${scale}.pt"
  test -d "$checkpoint" || test -f "$checkpoint" || \
    die "Missing local checkpoint $checkpoint; run local_train"
  mkdir -p "$out"
  if [[ ! -f "$model" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/export_lora_checkpoint.py" \
      --input "$checkpoint" \
      --output "$model" \
      --base-checkpoint "$SERVER_ROOT/rounds/round_000/global_model.pt" \
      --base-weights-source auto \
      --adapter-weights-source model \
      --lora-rank "$RANK" \
      --lora-alpha "$ALPHA" \
      --lora-scale "$scale" \
      --config-preset seq_model_esm1b_ptm
  fi
}

run_local_inference() {
  local id="$1"
  local model="$2"
  local split="$3"
  local out="$4"
  local seed="$5"
  local force="${6:-0}"
  local private="$CLIENTS_ROOT/client_$id/private"
  local assets="$private/splits/assets_$split"
  local labels="$private/splits/${split}_labels.txt"
  local expected actual
  test -d "$assets/solo_fasta_dir" || die "Missing $split FASTA assets for client_$id"
  test -d "$assets/solo_alignment_dir" || die "Missing $split alignments for client_$id"
  test -f "$labels" || die "Missing $split labels for client_$id"
  test -f "$model" || die "Missing inference model $model"
  expected="$(awk 'NF {c++} END {print c+0}' "$labels")"
  if [[ -d "$out/predictions" ]]; then
    shopt -s nullglob
    local predictions=("$out"/predictions/*_unrelaxed.pdb)
    actual="${#predictions[@]}"
    shopt -u nullglob
  else
    actual=0
  fi
  if [[ "$force" == "1" ]] || (( actual != expected )); then
    if (( actual != 0 )); then
      echo "[inference] regenerating $out (expected=$expected existing=$actual)"
    fi
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
      "$REPO/run_pretrained_openfold.py" \
      "$assets/solo_fasta_dir" \
      "$assets/mmcif_files" \
      --use_precomputed_alignments "$assets/solo_alignment_dir" \
      --use_single_seq_mode \
      --output_dir "$out" \
      --model_device cuda:0 \
      --skip_relaxation \
      --config_preset seq_model_esm1b_ptm \
      --experiment_config_json "$EXPERIMENT_CONFIG_JSON" \
      --openfold_checkpoint_path "$model" \
      --checkpoint_weights_source auto \
      --data_random_seed "$seed" \
      --precision fp32
  fi
  shopt -s nullglob
  local final_predictions=("$out"/predictions/*_unrelaxed.pdb)
  actual="${#final_predictions[@]}"
  shopt -u nullglob
  (( actual == expected )) || \
    die "Inference incomplete at $out: expected=$expected actual=$actual"
}

score_local_predictions() {
  local id="$1"
  local split="$2"
  local out="$3"
  local force="${4:-0}"
  local private="$CLIENTS_ROOT/client_$id/private"
  local labels="$private/splits/${split}_labels.txt"
  local difficulty="$private/difficulty/baseline_difficulty.csv"
  local native="$private/difficulty/native"
  local metrics="$out/metrics"
  mkdir -p "$metrics"
  if [[ "$force" == "1" || ! -f "$metrics/tm_score.csv" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/tmscore_from_pdb.py" \
      "$out/predictions" \
      --native-dir "$native" \
      --tm-exec "$REPO/tmscore/TMscore" \
      --out-csv "$metrics/tm_score.csv" \
      --selected-csv "$metrics/low_tm.csv" \
      --missing-log "$metrics/tm_missing.txt"
  fi
  if [[ "$force" == "1" || ! -f "$metrics/lddt_ca.csv" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/lddt_ca_from_pdb.py" \
      "$out/predictions" \
      --native-dir "$native" \
      --out-csv "$metrics/lddt_ca.csv" \
      --missing-log "$metrics/lddt_missing.txt"
  fi
  if [[ "$force" == "1" || ! -f "$metrics/plddt.csv" ]]; then
    "$PYTHON_BIN" "$REPO/scripts/plddt_from_pdb.py" \
      "$out/predictions" \
      --no-extremes \
      --select-threshold 101 \
      --selected-csv "$metrics/plddt.csv" \
      --bad-log "$metrics/plddt_bad.txt"
  fi
  "$PYTHON_BIN" "$REPO/scripts/evaluate_hardcase_metrics.py" \
    --mode pair \
    --labels "$labels" \
    --difficulty-csv "$difficulty" \
    --baseline-tm-csv "$difficulty" \
    --model-tm-csv "$metrics/tm_score.csv" \
    --baseline-lddt-csv "$difficulty" \
    --model-lddt-csv "$metrics/lddt_ca.csv" \
    --baseline-plddt-csv "$difficulty" \
    --model-plddt-csv "$metrics/plddt.csv" \
    --client-id "client_$id" \
    --out "$out/paired_deltas.csv"
  "$PYTHON_BIN" - "$labels" "$out/paired_deltas.csv" <<'PY'
import csv, sys
labels = [line.strip() for line in open(sys.argv[1]) if line.strip()]
rows = list(csv.DictReader(open(sys.argv[2])))
if len(rows) != len(labels):
    raise SystemExit(
        f"paired row mismatch: labels={len(labels)} metrics={len(rows)}"
    )
PY
}

validate_local_candidate() {
  local arm="$1"
  local seed="$2"
  local id="$3"
  local epoch="$4"
  local scale="$5"
  local out model
  out="$(local_candidate_dir "$id" "$arm" "$seed" validation "$epoch" "$scale")"
  model="$out/model_base_global_adapter_raw_scale_${scale}.pt"
  export_local_candidate "$id" "$arm" "$seed" "$epoch" "$scale"
  run_local_inference "$id" "$model" validation "$out" "$seed"
  score_local_predictions "$id" validation "$out"
  "$PYTHON_BIN" - "$out/candidate.json" "$model" "$epoch" "$scale" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "model_path": sys.argv[2],
    "epoch": int(sys.argv[3]),
    "scale": float(sys.argv[4]),
    "adapter_source": "model",
    "base_source": "global_round_000",
    "selection_split": "validation",
}, indent=2, sort_keys=True) + "\n")
PY
}

local_validate() {
  check_common
  local arm seed id epoch scale
  local -a epoch_arr=()
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  IFS=',' read -r -a scale_arr <<< "$LOCAL_SCALES"
  if [[ "$LOCAL_EVAL_EPOCHS" == "all" ]]; then
    mapfile -t epoch_arr < <(seq 1 "$LOCAL_ONLY_EPOCHS")
  else
    IFS=',' read -r -a epoch_arr <<< "$LOCAL_EVAL_EPOCHS"
  fi
  for epoch in "${epoch_arr[@]}"; do
    [[ "$epoch" =~ ^[0-9]+$ ]] || die "Invalid eval epoch: $epoch"
    (( epoch >= 1 && epoch <= LOCAL_ONLY_EPOCHS )) || \
      die "Eval epoch $epoch outside 1..$LOCAL_ONLY_EPOCHS"
  done
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for id in $(client_ids "$LOCAL_CLIENTS"); do
        for epoch in "${epoch_arr[@]}"; do
          for scale in "${scale_arr[@]}"; do
            echo "[local_validate] client_$id arm=$arm seed=$seed epoch=$epoch scale=$scale"
            validate_local_candidate "$arm" "$seed" "$id" "$epoch" "$scale"
          done
        done
      done
    done
  done
}

select_local_client() {
  local arm="$1"
  local seed="$2"
  local id="$3"
  local root
  root="$(local_only_root "$id" "$arm" "$seed")"
  shopt -s nullglob
  local paired=("$root"/validation/epoch_*/scale_*/paired_deltas.csv)
  shopt -u nullglob
  (( ${#paired[@]} > 0 )) || \
    die "No validation metrics for client_$id arm=$arm seed=$seed; run local_validate"
  "$PYTHON_BIN" "$REPO/scripts/select_local_checkpoint.py" \
    --paired-csvs "${paired[@]}" \
    --client-id "client_$id" \
    --arm "$arm" \
    --seed "$seed" \
    --nonhard-constraint -0.005 \
    --out "$root/selection.json"
}

local_select() {
  check_common
  local arm seed id
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for id in $(client_ids "$LOCAL_CLIENTS"); do
        select_local_client "$arm" "$seed" "$id"
      done
    done
  done
}

evaluate_local_development_client() {
  local arm="$1"
  local seed="$2"
  local id="$3"
  local root selection out fallback model
  root="$(local_only_root "$id" "$arm" "$seed")"
  selection="$root/selection.json"
  out="$root/development"
  test -f "$selection" || die "Missing $selection; run local_select"
  mkdir -p "$out"
  fallback="$("$PYTHON_BIN" -c \
    "import json; print(str(json.load(open('$selection'))['fallback_to_baseline']).lower())")"
  if [[ "$fallback" == "true" ]]; then
    local private="$CLIENTS_ROOT/client_$id/private"
    local difficulty="$private/difficulty/baseline_difficulty.csv"
    "$PYTHON_BIN" "$REPO/scripts/evaluate_hardcase_metrics.py" \
      --mode pair \
      --labels "$private/splits/development_test_labels.txt" \
      --difficulty-csv "$difficulty" \
      --baseline-tm-csv "$difficulty" \
      --model-tm-csv "$difficulty" \
      --baseline-lddt-csv "$difficulty" \
      --model-lddt-csv "$difficulty" \
      --client-id "client_$id" \
      --out "$out/paired_deltas.csv"
    echo "[local_dev] client_$id arm=$arm seed=$seed uses baseline fallback"
  else
    model="$("$PYTHON_BIN" -c \
      "import json; print(json.load(open('$selection'))['model_path'])")"
    test -f "$model" || die "Selected model missing: $model"
    run_local_inference "$id" "$model" development_test "$out" "$seed"
    score_local_predictions "$id" development_test "$out"
  fi
  cp "$selection" "$out/selection_used.json"
}

local_evaluate_dev() {
  check_common
  local arm seed id
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for id in $(client_ids "$LOCAL_CLIENTS"); do
        evaluate_local_development_client "$arm" "$seed" "$id"
      done
    done
  done
}

local_report() {
  check_common
  local -a path_args=()
  if [[ -n "$TARGET_SLUG" ]]; then
    path_args+=(--target-slug "$TARGET_SLUG")
  fi
  if [[ -n "$LOCAL_OUTPUT_NAMESPACE" ]]; then
    path_args+=(--local-output-namespace "$LOCAL_OUTPUT_NAMESPACE")
  fi
  "$PYTHON_BIN" "$REPO/scripts/summarize_local_only.py" \
    --run-root "$RUN_ROOT" \
    --arms "$ARMS" \
    --seeds "$SEEDS" \
    --num-clients "$NUM_CLIENTS" \
    --out-dir "$EVAL_ROOT/local_only" \
    "${path_args[@]}"
  cat > "$REPORT_ROOT/local_only_report.md" <<EOF
# Local-only hard-case LoRA report

This report compares independent client adapters before any federated
aggregation. Hyperparameters and checkpoints are selected only on each
client's cluster-disjoint validation split. The already-inspected 80-chain
development test is used only after selection.

Outputs:

- \`$EVAL_ROOT/local_only/local_only_summary.csv\`
- \`$EVAL_ROOT/local_only/local_only_seed_summary.csv\`
- \`$EVAL_ROOT/local_only/local_only_paired_deltas.csv\`

Selection constraint: non-hard mean delta TM >= -0.005, then maximize hard
mean delta TM. If no candidate passes or the best hard gain is non-positive,
the client falls back to scale 0 (public baseline).

The external final test remains untouched.
EOF
  echo "Wrote $REPORT_ROOT/local_only_report.md"
}

local_status() {
  check_common
  test -f "$SPLITS_ROOT/split_manifest.json" || die "Run prepare first"
  local arm_count seed_count scale_count eval_epoch_count client_count train_jobs candidates
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  IFS=',' read -r -a scale_arr <<< "$LOCAL_SCALES"
  arm_count="${#arm_arr[@]}"
  seed_count="${#seed_arr[@]}"
  scale_count="${#scale_arr[@]}"
  if [[ "$LOCAL_EVAL_EPOCHS" == "all" ]]; then
    eval_epoch_count="$LOCAL_ONLY_EPOCHS"
  else
    IFS=',' read -r -a eval_epoch_arr <<< "$LOCAL_EVAL_EPOCHS"
    eval_epoch_count="${#eval_epoch_arr[@]}"
  fi
  if [[ "$LOCAL_CLIENTS" == "all" ]]; then
    client_count="$NUM_CLIENTS"
  else
    client_count=1
  fi
  train_jobs=$((arm_count * seed_count * client_count))
  candidates=$((train_jobs * eval_epoch_count * scale_count))
  echo "Local-only configuration"
  echo "  clients=$LOCAL_CLIENTS (count=$client_count)"
  echo "  arms=$ARMS"
  echo "  seeds=$SEEDS"
  echo "  train_epochs=$LOCAL_ONLY_EPOCHS"
  echo "  validation_epochs=$LOCAL_EVAL_EPOCHS"
  echo "  validation_scales=$LOCAL_SCALES"
  echo "  independent_train_jobs=$train_jobs"
  echo "  validation_candidates=$candidates"
  echo "  aggregation=disabled"
  local id
  for id in $(client_ids "$LOCAL_CLIENTS"); do
    local private="$CLIENTS_ROOT/client_$id/private"
    test -f "$private/splits/train_labels.txt" || die "Missing client_$id train labels"
    test -f "$private/splits/validation_labels.txt" || die "Missing client_$id validation labels"
    test -f "$private/splits/development_test_labels.txt" || die "Missing client_$id development labels"
    test -d "$private/splits/assets_validation/solo_fasta_dir" || \
      die "Missing client_$id validation FASTA assets"
    test -d "$private/splits/assets_development_test/solo_fasta_dir" || \
      die "Missing client_$id development FASTA assets"
    echo "  client_$id: train=$(train_epoch_len_for_client "$id") " \
      "val=$(awk 'NF {c++} END {print c+0}' "$private/splits/validation_labels.txt") " \
      "dev=$(awk 'NF {c++} END {print c+0}' "$private/splits/development_test_labels.txt")"
  done
}

local_all() {
  local_train
  local_validate
  local_select
  local_evaluate_dev
  local_report
}

replicate_soft_tm_client() {
  local id="${1:-}"
  [[ "$id" =~ ^[1-4]$ ]] || \
    die "replicate_soft_tm requires one client id from 1 to 4"

  LOCAL_CLIENTS="$id"
  ARMS="uniform"
  SEED="42"
  SEEDS="42"
  LOCAL_ONLY_EPOCHS="5"
  LOCAL_EVAL_EPOCHS="5"
  LOCAL_SCALES="0.85"
  PRIMARY_EVAL_SCALE="0.85"
  LR="1e-4"
  RANK="4"
  ALPHA="8"
  DROPOUT="0"
  TARGET_SLUG="T4"
  LOCAL_OUTPUT_NAMESPACE="frozen_soft_tm_v1"
  LORA_TARGET="structure_module,evoformer.linear,evoformer.blocks.44,evoformer.blocks.45,evoformer.blocks.46,evoformer.blocks.47"
  EXPERIMENT_CONFIG_JSON="$REPO/seq_model_esm1b_ptm_hardcase_unclamped_override.json"
  BASELINE_PRESERVATION_WEIGHT="1.5"
  HARD_SOFT_TM_WEIGHT="5.0"
  HARD_SOFT_TM_MARGIN="0.02"

  echo "=== Replicate frozen client0 soft-TM method on client_$id ==="
  local_status
  local_train
  local_validate
  local_select

  local root paired gate selection summary paired_summary
  root="$(local_only_root "$id" uniform 42)"
  paired="$root/validation/epoch_5/scale_0p85/paired_deltas.csv"
  paired_summary="${paired%.csv}.summary.json"
  gate="$root/validation/epoch_5/scale_0p85/promotion_v2.json"
  selection="$root/selection.json"
  summary="$root/frozen_client_test_summary.json"
  "$PYTHON_BIN" "$REPO/scripts/evaluate_target_gate.py" \
    --mode promotion-v2 --paired-csv "$paired" --out "$gate"
  "$PYTHON_BIN" - "$summary" "$id" "$gate" "$paired_summary" "$selection" <<'PY'
import json
import pathlib
import sys

out, client_id, gate_path, metrics_path, selection_path = sys.argv[1:]
with open(gate_path) as handle:
    gate = json.load(handle)
with open(metrics_path) as handle:
    metrics = json.load(handle)
with open(selection_path) as handle:
    selection = json.load(handle)
payload = {
    "stage": "replicate-frozen-client0-soft-tm-method",
    "status": "complete",
    "client_id": f"client_{client_id}",
    "configuration": {
        "target_slug": "T4", "rank": 4, "alpha": 8.0, "dropout": 0.0,
        "learning_rate": 1e-4, "epoch": 5, "adapter_scale": 0.85,
        "baseline_preservation_weight": 1.5,
        "hard_soft_tm_weight": 5.0, "hard_soft_tm_margin": 0.02,
        "fape_clamp_prob": 0.0,
    },
    "promotion_v2": gate,
    "validation_metrics": metrics,
    "selection": selection,
    "development_test_accessed": False,
    "external_final_test_accessed": False,
    "ready_for_seed_confirmation": bool(gate.get("promotion_pass")),
}
path = pathlib.Path(out)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  echo "Wrote $summary"
}


train_client_round() {
  local arm="$1"
  local seed="$2"
  local round="$3"
  local id="$4"
  local client="client_$id"
  local private="$CLIENTS_ROOT/$client/private"
  local data_src="$SOURCE_RUN/clients/$client"
  local parent="$SERVER_ROOT/rounds/round_$(printf '%03d' "$round")/global_model.pt"
  local out="$private/rounds/$arm/seed_$seed/round_$(printf '%03d' "$round")"
  local epoch_len total_steps adapter baseline_prediction_dir
  local -a extra_train_args=()
  epoch_len="$(train_epoch_len_for_client "$id")"
  (( epoch_len > 0 )) || die "Empty train split for $client"
  total_steps=$((epoch_len * LOCAL_EPOCHS))
  adapter="$out/training/checkpoints/$((LOCAL_EPOCHS - 1))-$total_steps.ckpt"
  baseline_prediction_dir="$data_src/prescreen/predictions"

  if positive_float "$BASELINE_PRESERVATION_WEIGHT" || \
      positive_float "$HARD_SOFT_TM_WEIGHT"; then
    test -d "$baseline_prediction_dir" || \
      die "Missing baseline predictions for $client: $baseline_prediction_dir"
    extra_train_args+=(--baseline_preservation_dir "$baseline_prediction_dir")
  fi
  if positive_float "$BASELINE_PRESERVATION_WEIGHT"; then
    extra_train_args+=(
      --baseline_preservation_weight "$BASELINE_PRESERVATION_WEIGHT"
    )
  fi
  if positive_float "$HARD_SOFT_TM_WEIGHT"; then
    extra_train_args+=(
      --hard_soft_tm_weight "$HARD_SOFT_TM_WEIGHT"
      --hard_soft_tm_margin "$HARD_SOFT_TM_MARGIN"
    )
  fi

  mkdir -p "$out"
  if [[ -d "$adapter" || -f "$adapter" ]]; then
    echo "[skip] $client arm=$arm seed=$seed round=$round already trained"
    return 0
  fi

  # Lineage sidecar (server-visible metadata only).
  cat > "$out/lineage.json" <<EOF
{
  "client_id": "$client",
  "arm": "$arm",
  "seed": $seed,
  "round_id": $round,
  "parent_global_path": "$parent",
  "parent_global_sha256": "$(sha256_file "$parent")",
  "sampling_mode": "$arm",
  "local_epochs": $LOCAL_EPOCHS,
  "train_epoch_len": $epoch_len,
  "rank": $RANK,
  "alpha": $ALPHA,
  "dropout": $DROPOUT,
  "learning_rate": $LR,
  "target": "$LORA_TARGET",
  "experiment_config_json": "$(readlink -f "$EXPERIMENT_CONFIG_JSON")",
  "experiment_config_sha256": "$(sha256_file "$EXPERIMENT_CONFIG_JSON")",
  "baseline_preservation_weight": $BASELINE_PRESERVATION_WEIGHT,
  "hard_soft_tm_weight": $HARD_SOFT_TM_WEIGHT,
  "hard_soft_tm_margin": $HARD_SOFT_TM_MARGIN
}
EOF

  echo "[train] $client arm=$arm seed=$seed round=$round epoch_len=$epoch_len"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    "$REPO/train_openfold.py" \
    "$data_src/mmcif_files_finetune" \
    "$data_src/solo_alignment" \
    "$data_src/mmcif_files_finetune" \
    "$out/training" \
    2026-01-01 \
    --train_filter_path "$private/splits/train_labels.txt" \
    --val_data_dir "$private/splits/assets_validation/mmcif_files" \
    --val_alignment_dir "$private/splits/assets_validation/solo_alignment_dir" \
    --use_single_seq_mode True \
    --config_preset seq_model_esm1b_ptm \
    --experiment_config_json "$EXPERIMENT_CONFIG_JSON" \
    --resume_from_ckpt "$parent" \
    --resume_model_weights_only True \
    --init_weights_source auto \
    --template_release_dates_cache_path "$data_src/mmcif_cache_finetune.json" \
    --train_chain_data_cache_path "$private/splits/train_chain_data_cache.json" \
    --lora_rank "$RANK" \
    --lora_alpha "$ALPHA" \
    --lora_dropout "$DROPOUT" \
    --lora_target "$LORA_TARGET" \
    --learning_rate "$LR" \
    --lr_warmup_steps 5 \
    --accumulate_grad_batches 1 \
    --train_epoch_len "$epoch_len" \
    --max_epochs "$LOCAL_EPOCHS" \
    --checkpoint_every_epoch \
    --sampling_audit_path "$out/sampling_audit.jsonl" \
    --sampling_cluster_file "$CLUSTERS" \
    --sampling_mode "$arm" \
    --difficulty_csv "$private/difficulty/baseline_difficulty.csv" \
    --hard_aware_ratios "$(hard_aware_ratios_for_arm "$arm")" \
    "${extra_train_args[@]}" \
    --precision 32 \
    --gpus 1 \
    --seed "$seed" \
    --deepspeed_config_path "$REPO/deepspeed_config_fp32.json" \
    2>&1 | tee "$out/train.log"

  # Upload pointer (simulation): round-end raw adapter checkpoint path.
  cat > "$out/upload.json" <<EOF
{
  "client_id": "$client",
  "round_id": $round,
  "adapter_checkpoint": "$adapter",
  "adapter_source": "model",
  "n_k": $epoch_len,
  "parent_global_sha256": "$(sha256_file "$parent")"
}
EOF
}


verify_federated_config() {
  local manifest="$RUN_ROOT/run_manifest.json"
  local server_config="$SERVER_ROOT/config.json"
  test -f "$manifest" || die "Missing $manifest; run prepare in this RUN_ROOT"
  test -f "$server_config" || \
    die "Missing $server_config; run prepare in this RUN_ROOT"
  "$PYTHON_BIN" - \
    "$manifest" "$server_config" "$RANK" "$ALPHA" "$DROPOUT" "$LR" \
    "$LOCAL_EPOCHS" "$LORA_TARGET" "$EXPERIMENT_CONFIG_JSON" \
    "$BASELINE_PRESERVATION_WEIGHT" "$HARD_SOFT_TM_WEIGHT" \
    "$HARD_SOFT_TM_MARGIN" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    manifest_path,
    server_path,
    rank,
    alpha,
    dropout,
    learning_rate,
    local_epochs,
    target,
    experiment_config,
    preservation_weight,
    soft_tm_weight,
    soft_tm_margin,
) = sys.argv[1:]
manifest = json.loads(pathlib.Path(manifest_path).read_text())
server = json.loads(pathlib.Path(server_path).read_text())
shared = manifest.get("shared_config", {})
config_path = pathlib.Path(experiment_config).resolve()
config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
expected = {
    "rank": int(rank),
    "alpha": float(alpha),
    "dropout": float(dropout),
    "learning_rate": float(learning_rate),
    "local_epochs_per_round": int(local_epochs),
    "target": target,
    "experiment_config_json": str(config_path),
    "experiment_config_sha256": config_sha,
    "baseline_preservation_weight": float(preservation_weight),
    "hard_soft_tm_weight": float(soft_tm_weight),
    "hard_soft_tm_margin": float(soft_tm_margin),
}
errors = []
for key, value in expected.items():
    manifest_value = shared.get(key)
    if key == "target":
        manifest_value = ",".join(manifest_value or [])
    if manifest_value != value:
        errors.append(
            f"run_manifest shared_config.{key}: "
            f"expected {value!r}, found {manifest_value!r}"
        )
    if server.get(key) != value:
        errors.append(
            f"server config {key}: expected {value!r}, "
            f"found {server.get(key)!r}"
        )
if errors:
    raise SystemExit("Federated configuration mismatch:\n- " + "\n- ".join(errors))
print("Federated configuration verified")
PY
}

hard_aware_ratios_for_arm() {
  case "$1" in
    hard_aware_50) echo "0.5,0.25,0.25" ;;
    hard_aware_70|hard_aware) echo "0.7,0.15,0.15" ;;
    *) echo "0.7,0.15,0.15" ;;
  esac
}

aggregate_round() {
  local arm="$1"
  local seed="$2"
  local round="$3"
  local next=$((round + 1))
  local parent="$SERVER_ROOT/rounds/round_$(printf '%03d' "$round")/global_model.pt"
  local out_dir="$SERVER_ROOT/rounds/$arm/seed_$seed/round_$(printf '%03d' "$next")"
  mkdir -p "$out_dir"

  local ckpts=()
  local weights=()
  local ids=()
  local id
  for id in $(client_ids all); do
    local upload="$CLIENTS_ROOT/client_$id/private/rounds/$arm/seed_$seed/round_$(printf '%03d' "$round")/upload.json"
    test -f "$upload" || die "Missing upload $upload"
    local ckpt n_k
    ckpt="$("$PYTHON_BIN" -c "import json;print(json.load(open('$upload'))['adapter_checkpoint'])")"
    n_k="$("$PYTHON_BIN" -c "import json;print(json.load(open('$upload'))['n_k'])")"
    ckpts+=("$ckpt")
    weights+=("$n_k")
    ids+=("client_$id")
  done

  local clip_args=()
  if [[ -n "$MAX_UPDATE_NORM" ]]; then
    clip_args+=(--max-update-norm "$MAX_UPDATE_NORM")
  fi

  "$PYTHON_BIN" "$REPO/scripts/fedlora_aggregate.py" \
    --parent "$parent" \
    --checkpoints "${ckpts[@]}" \
    --weights "${weights[@]}" \
    --client-ids "${ids[@]}" \
    --output "$out_dir/global_model.pt" \
    --manifest "$out_dir/aggregation.json" \
    --round-id "$next" \
    --lora-rank "$RANK" \
    --lora-alpha "$ALPHA" \
    --expected-parent-sha "$(sha256_file "$parent")" \
    "${clip_args[@]}"

  # Convenience alias for next round parent lookup used by train_client_round
  # when arm/seed namespacing is active: also mirror into arm path round index.
  mkdir -p "$SERVER_ROOT/rounds/round_$(printf '%03d' "$next")"
}

# For multi-arm runs, keep global models under server/rounds/<arm>/seed_*/...
# and point train_client_round at the correct parent via symlink helper.
parent_for() {
  local arm="$1"
  local seed="$2"
  local round="$3"
  if (( round == 0 )); then
    echo "$SERVER_ROOT/rounds/round_000/global_model.pt"
  else
    echo "$SERVER_ROOT/rounds/$arm/seed_$seed/round_$(printf '%03d' "$round")/global_model.pt"
  fi
}

train_federated_arm() {
  local arm="$1"
  local seed="$2"
  local r id
  for ((r=0; r<ROUNDS; r++)); do
    local parent
    parent="$(parent_for "$arm" "$seed" "$r")"
    test -f "$parent" || die "Missing parent $parent"
    # Expose current parent at the path train_client_round expects for round r
    mkdir -p "$SERVER_ROOT/rounds/round_$(printf '%03d' "$r")"
    if (( r > 0 )); then
      ln -sfn "$parent" "$SERVER_ROOT/rounds/round_$(printf '%03d' "$r")/global_model.pt"
    fi
    for id in $(client_ids all); do
      train_client_round "$arm" "$seed" "$r" "$id"
    done
    aggregate_round "$arm" "$seed" "$r"
  done
}

train() {
  check_common
  test -f "$SPLITS_ROOT/split_manifest.json" || die "Run prepare first"
  verify_federated_config
  local arm seed
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      echo "=== FedLoRA train arm=$arm seed=$seed rounds=$ROUNDS ==="
      train_federated_arm "$arm" "$seed"
    done
  done
}

smoke() {
  check_common
  # Do not re-run prepare with a reduced client count (would overwrite splits).
  # Verify existing artifacts when present; always run unit tests.
  if [[ -f "$SPLITS_ROOT/split_manifest.json" && -f "$SERVER_ROOT/rounds/round_000/global_model.pt" ]]; then
    echo "[smoke] reusing existing prepare artifacts under $RUN_ROOT"
  else
    echo "[smoke] missing prepare artifacts; running full prepare"
    prepare
  fi
  test -f "$SPLITS_ROOT/split_manifest.json" || die "smoke split missing"
  test -f "$REPORT_ROOT/privacy_audit.md" || die "privacy audit missing"
  test -f "$SERVER_ROOT/rounds/round_000/global_model.pt" || die "round0 missing"
  echo "[smoke] prepare artifacts OK"
  "$PYTHON_BIN" -m unittest \
    tests.test_fedlora_aggregation \
    tests.test_hardcase_sampling \
    tests.test_federated_round_smoke
  echo "[smoke] unit/integration tests OK"
}

federated_validation_dir() {
  local id="$1"
  local arm="$2"
  local seed="$3"
  local round="$4"
  echo "$CLIENTS_ROOT/client_$id/private/validation/$arm/seed_$seed/round_$(printf "%03d" "$round")"
}

write_federated_validation_manifest() {
  local out="$1"
  local model="$2"
  local model_sha="$3"
  local arm="$4"
  local seed="$5"
  local round="$6"
  local id="$7"
  local status="$8"
  mkdir -p "$out"
  "$PYTHON_BIN" - "$out/inference_manifest.json" "$model" "$model_sha" \
    "$arm" "$seed" "$round" "$id" "$status" "$EXPERIMENT_CONFIG_JSON" <<"PY"
import datetime
import json
import os
import pathlib
import sys

(path_s, model_s, model_sha, arm, seed, round_id, client_id, status,
 config_s) = sys.argv[1:]
path = pathlib.Path(path_s)
payload = {
    "arm": arm,
    "client_id": f"client_{client_id}",
    "experiment_config_json": str(pathlib.Path(config_s).resolve()),
    "global_model_path": str(pathlib.Path(model_s).resolve()),
    "global_model_sha256": model_sha,
    "round_id": int(round_id),
    "seed": int(seed),
    "split": "validation",
    "status": status,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

federated_validation_job_complete() {
  local id="$1"
  local out="$2"
  local model_sha="$3"
  local round="$4"
  local labels="$CLIENTS_ROOT/client_$id/private/splits/validation_labels.txt"
  "$PYTHON_BIN" - "$labels" "$out/paired_deltas.csv" \
    "$out/inference_manifest.json" "$model_sha" "$round" <<"PY"
import csv
import json
import pathlib
import sys

labels_path = pathlib.Path(sys.argv[1])
paired_path = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])
model_sha = sys.argv[4]
round_id = int(sys.argv[5])
try:
    labels = [x.strip().upper() for x in labels_path.read_text().splitlines() if x.strip()]
    with paired_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text())
    assert [row["label"].strip().upper() for row in rows] == labels
    if model_sha != "-":
        assert manifest["global_model_sha256"] == model_sha
    assert int(manifest["round_id"]) == round_id
    assert manifest["status"] == "complete"
except (AssertionError, FileNotFoundError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

federated_validation_predictions_ready() {
  local id="$1"
  local out="$2"
  local model_sha="$3"
  local round="$4"
  local labels="$CLIENTS_ROOT/client_$id/private/splits/validation_labels.txt"
  "$PYTHON_BIN" - "$labels" "$out/predictions" \
    "$out/inference_manifest.json" "$model_sha" "$round" <<"PY"
import json
import pathlib
import sys

labels_path = pathlib.Path(sys.argv[1])
predictions = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])
model_sha = sys.argv[4]
round_id = int(sys.argv[5])
try:
    expected = sum(bool(x.strip()) for x in labels_path.read_text().splitlines())
    actual = len(list(predictions.glob("*_unrelaxed.pdb")))
    manifest = json.loads(manifest_path.read_text())
    assert actual == expected
    assert manifest["global_model_sha256"] == model_sha
    assert int(manifest["round_id"]) == round_id
    assert manifest["status"] in {"inference_complete", "complete"}
except (AssertionError, FileNotFoundError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

write_round0_validation_client() {
  local arm="$1"
  local seed="$2"
  local id="$3"
  local model="$4"
  local model_sha="$5"
  local out private labels difficulty
  out="$(federated_validation_dir "$id" "$arm" "$seed" 0)"
  if federated_validation_job_complete "$id" "$out" "$model_sha" 0; then
    echo "[validate_infer] skip complete baseline client_$id"
    return
  fi
  private="$CLIENTS_ROOT/client_$id/private"
  labels="$private/splits/validation_labels.txt"
  difficulty="$private/difficulty/baseline_difficulty.csv"
  mkdir -p "$out"
  "$PYTHON_BIN" "$REPO/scripts/evaluate_hardcase_metrics.py" \
    --mode pair \
    --labels "$labels" \
    --difficulty-csv "$difficulty" \
    --baseline-tm-csv "$difficulty" \
    --model-tm-csv "$difficulty" \
    --baseline-lddt-csv "$difficulty" \
    --model-lddt-csv "$difficulty" \
    --client-id "client_$id" \
    --out "$out/paired_deltas.csv"
  write_federated_validation_manifest "$out" "$model" "$model_sha" \
    "$arm" "$seed" 0 "$id" complete
  federated_validation_job_complete "$id" "$out" "$model_sha" 0 || \
    die "Failed to create complete round-0 validation for client_$id"
}

validate_global_round_client() {
  local arm="$1"
  local seed="$2"
  local round="$3"
  local id="$4"
  local model="$5"
  local model_sha="$6"
  local out
  out="$(federated_validation_dir "$id" "$arm" "$seed" "$round")"
  if federated_validation_job_complete "$id" "$out" "$model_sha" "$round"; then
    echo "[validate_infer] skip complete arm=$arm seed=$seed round=$round client_$id"
    return
  fi
  if ! federated_validation_predictions_ready "$id" "$out" "$model_sha" "$round"; then
    write_federated_validation_manifest "$out" "$model" "$model_sha" \
      "$arm" "$seed" "$round" "$id" running
    run_local_inference "$id" "$model" validation "$out" "$seed" 1
    write_federated_validation_manifest "$out" "$model" "$model_sha" \
      "$arm" "$seed" "$round" "$id" inference_complete
  else
    echo "[validate_infer] reuse predictions arm=$arm seed=$seed round=$round client_$id"
  fi
  score_local_predictions "$id" validation "$out" 1
  write_federated_validation_manifest "$out" "$model" "$model_sha" \
    "$arm" "$seed" "$round" "$id" complete
  federated_validation_job_complete "$id" "$out" "$model_sha" "$round" || \
    die "Validation integrity check failed: arm=$arm seed=$seed round=$round client_$id"
}

validate_infer() {
  check_common
  test -f "$SPLITS_ROOT/split_manifest.json" || die "Run prepare first"
  local arm seed round id model model_sha
  IFS=, read -r -a arm_arr <<< "$ARMS"
  IFS=, read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for round in $(federated_validation_round_ids); do
        model="$(parent_for "$arm" "$seed" "$round")"
        test -f "$model" || die "Missing global model $model; finish train first"
        model_sha="$(sha256_file "$model")"
        for id in $(client_ids "$FED_VALIDATION_CLIENTS"); do
          echo "=== Validation arm=$arm seed=$seed round=$round client_$id ==="
          if (( round == 0 )); then
            write_round0_validation_client "$arm" "$seed" "$id" "$model" "$model_sha"
          else
            validate_global_round_client \
              "$arm" "$seed" "$round" "$id" "$model" "$model_sha"
          fi
        done
      done
    done
  done
  validate
  echo "[validate_infer] selected jobs complete; summaries refreshed"
  echo "Next: bash scripts/fed_lora_hardcase_fed.sh select_global"
}

validation_status() {
  check_common
  local arm seed round id out complete total
  complete=0
  total=0
  IFS=, read -r -a arm_arr <<< "$ARMS"
  IFS=, read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for round in $(federated_validation_round_ids); do
        local round_complete=0
        local round_total=0
        for id in $(client_ids "$FED_VALIDATION_CLIENTS"); do
          total=$((total + 1))
          round_total=$((round_total + 1))
          out="$(federated_validation_dir "$id" "$arm" "$seed" "$round")"
          if federated_validation_job_complete "$id" "$out" - "$round"; then
            complete=$((complete + 1))
            round_complete=$((round_complete + 1))
          fi
        done
        echo "  arm=$arm seed=$seed round=$round complete=$round_complete/$round_total"
      done
    done
  done
  echo "Federated validation complete=$complete/$total"
  echo "Active inference processes:"
  pgrep -af "run_pretrained_openfold.py" || true
}

validate_round_model() {
  # Aggregate private paired metrics into server-visible client and macro summaries.
  # PDB inference itself is performed by validate_infer.
  local arm="$1"
  local seed="$2"
  local round="$3"
  local model model_sha
  model="$(parent_for "$arm" "$seed" "$round")"
  model_sha="$(sha256_file "$model")"
  local out="$SERVER_ROOT/rounds/$arm/seed_$seed/round_$(printf '%03d' "$round")"
  mkdir -p "$out"
  local summaries=()
  local id
  for id in $(client_ids all); do
    local client="client_$id"
    local paired="$CLIENTS_ROOT/$client/private/validation/$arm/seed_$seed/round_$(printf '%03d' "$round")/paired_deltas.csv"
    local summary="$out/validation_client_$id.json"
    if [[ -f "$paired" ]]; then
      "$PYTHON_BIN" "$REPO/scripts/evaluate_hardcase_metrics.py" \
        --mode server-summary \
        --paired-csv "$paired" \
        --client-id "$client" \
        --round-id "$round" \
        --global-model-sha "$model_sha" \
        --out "$summary"
    else
      cat > "$summary" <<EOF
{
  "client_id": "$client",
  "round_id": $round,
  "global_model_sha": "$model_sha",
  "hard_count": 0,
  "hard_cluster_count": 0,
  "sum_delta_tm_hard": 0.0,
  "hard_improved_count": 0,
  "hard_rescue_count": 0,
  "hard_large_improve_count": 0,
  "hard_large_degrade_count": 0,
  "nonhard_count": 0,
  "sum_delta_tm_nonhard": 0.0,
  "sum_delta_lddt": 0.0,
  "metric_status": "missing_private_paired_csv"
}
EOF
    fi
    summaries+=("$summary")
  done
  "$PYTHON_BIN" "$REPO/scripts/evaluate_hardcase_metrics.py" \
    --mode macro \
    --server-summaries "${summaries[@]}" \
    --out "$out/validation_summary.json"
  # Attach round_id for selector.
  "$PYTHON_BIN" - "$out/validation_summary.json" "$round" <<'PY'
import json, sys
path, round_id = sys.argv[1], int(sys.argv[2])
data = json.loads(open(path).read())
data["round_id"] = round_id
open(path, "w").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

validate() {
  check_common
  local arm seed r
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      for ((r=0; r<=ROUNDS; r++)); do
        local model
        model="$(parent_for "$arm" "$seed" "$r")"
        if [[ -f "$model" ]]; then
          validate_round_model "$arm" "$seed" "$r"
        fi
      done
    done
  done
}

select_global() {
  check_common
  local arm seed
  IFS=',' read -r -a arm_arr <<< "$ARMS"
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for arm in "${arm_arr[@]}"; do
    for seed in "${seed_arr[@]}"; do
      local files=()
      local r
      for ((r=0; r<=ROUNDS; r++)); do
        local f="$SERVER_ROOT/rounds/$arm/seed_$seed/round_$(printf '%03d' "$r")/validation_summary.json"
        if [[ -f "$f" ]]; then
          files+=("$f")
        fi
      done
      (( ${#files[@]} == ROUNDS + 1 )) || \
        die "Validation summaries incomplete for $arm seed=$seed: expected=$((ROUNDS + 1)) actual=${#files[@]}; run validate_infer"
      "$PYTHON_BIN" - "$NUM_CLIENTS" "${files[@]}" <<"PY"
import json
import pathlib
import sys

expected_clients = int(sys.argv[1])
for path_s in sys.argv[2:]:
    path = pathlib.Path(path_s)
    row = json.loads(path.read_text())
    if row.get("metric_status") != "ok" or int(row.get("n_clients", -1)) != expected_clients:
        raise SystemExit(
            f"Incomplete validation summary {path}: "
            f"status={row.get('metric_status')} n_clients={row.get('n_clients')}; "
            "run validate_infer for all clients/rounds"
        )
PY
      if [[ ! -f "$SERVER_ROOT/rounds/$arm/seed_$seed/round_000/global_model.pt" ]]; then
        mkdir -p "$SERVER_ROOT/rounds/$arm/seed_$seed/round_000"
        ln -sfn "$SERVER_ROOT/rounds/round_000/global_model.pt" \
          "$SERVER_ROOT/rounds/$arm/seed_$seed/round_000/global_model.pt"
      fi
      "$PYTHON_BIN" "$REPO/scripts/select_global_round.py" \
        --round-summaries "${files[@]}" \
        --models-root "$SERVER_ROOT/rounds/$arm/seed_$seed" \
        --out "$SERVER_ROOT/rounds/$arm/seed_$seed/global_best_selection.json" \
        --copy-best-to "$SERVER_ROOT/rounds/$arm/seed_$seed/global_best.pt"
    done
  done
}

evaluate_dev() {
  check_common
  echo "[evaluate_dev] Fill private paired CSVs via inference, then re-run validate/select."
  echo "Development labels: $SPLITS_ROOT/development_test/labels.txt"
  echo "Do not use development_test for round selection."
}

lock_final() {
  check_common
  "$PYTHON_BIN" - <<PY
import json, pathlib
manifest = pathlib.Path("$SPLITS_ROOT/split_manifest.json")
data = json.loads(manifest.read_text())
data["final_test_lock_state"] = "locked"
manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
run = pathlib.Path("$RUN_ROOT/run_manifest.json")
if run.exists():
    rd = json.loads(run.read_text())
    rd["final_test_lock_state"] = "locked"
    run.write_text(json.dumps(rd, indent=2, sort_keys=True) + "\n")
print("final test locked")
PY
}

evaluate_final() {
  check_common
  local state
  state="$("$PYTHON_BIN" -c "import json;print(json.load(open('$SPLITS_ROOT/split_manifest.json'))['final_test_lock_state'])")"
  [[ "$state" == "locked" ]] || die "Final test is not locked; run lock_final after freezing configs"
  echo "[evaluate_final] external labels: $SPLITS_ROOT/external_final/labels.txt"
  echo "Run inference once per frozen method and write $EVAL_ROOT/external_final/"
}

report() {
  check_common
  mkdir -p "$REPORT_ROOT"
  cat > "$REPORT_ROOT/federated_run_report.md" <<EOF
# FedFold / FedLoRA hard-case report

## Success criteria (pre-registered)

- hard client-macro mean delta TM >= +0.02
- hard median delta TM > 0
- cluster-bootstrap 95% CI lower bound > 0
- large improve rate > large degrade rate
- non-hard client-macro mean delta TM >= -0.005

## Methods

- M0 public pretrained baseline
- M1 local-only uniform LoRA
- M2 local-only hard-aware LoRA
- M3 uniform FedLoRA
- M4 hard-aware FedLoRA
- M5 optional personalization after global_best (private only)

## Stage budgets

- Stage 1A: 2 arms x 5 rounds x 5 clients = 50 local one-epoch jobs (~3.5-5 GPU h)
- Stage 1B: x3 seeds => ~10-15 GPU h plus validation inference

## Notes

- Aggregation uses raw effective LoRA deltas only.
- Global best round selected by client-macro hard delta under non-hard constraint.
- External final is locked behind \`lock_final\`.
EOF
  echo "Wrote $REPORT_ROOT/federated_run_report.md"
}

usage() {
  cat <<EOF
Usage: bash scripts/fed_lora_hardcase_fed.sh <command>

Commands:
  prepare         difficulty + splits + round0 global
  smoke           prepare subset + unit tests
  local_train     independent client LoRA training; never aggregates
  replicate_soft_tm <1..4>
                  run frozen client0 soft-TM config on one other client
  local_validate  evaluate every local epoch/scale on validation
  local_select    select local epoch/scale with baseline fallback
  local_evaluate_dev
                  evaluate selected local models on development_test
  local_report    summarize local-only client-macro results
  local_status    validate inputs and print the local-only task matrix
  local_all       local_train + validate + select + dev + report
  train           multi-round FedLoRA for ARMS/SEEDS
  validate_infer  infer/score global rounds on private validation; resumable
  validation_status
                  show completed client/round validation jobs and active inference
  validate        rebuild server summaries only (no PDB inference)
  select_global   constrained global best round; refuses incomplete validation
  evaluate_dev    development-test instructions
  lock_final      freeze configs before external final
  evaluate_final  external final (requires lock)
  report          write report stub / success criteria
  all             prepare + train + validate_infer + select_global + report
EOF
}

cmd="${1:-}"
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
case "$cmd" in
  prepare) prepare ;;
  smoke) smoke ;;
  local_train) local_train ;;
  local_validate) local_validate ;;
  local_select) local_select ;;
  local_evaluate_dev) local_evaluate_dev ;;
  local_report) local_report ;;
  local_status) local_status ;;
  local_all) local_all ;;
  replicate_soft_tm) replicate_soft_tm_client "${2:-}" ;;
  train) train ;;
  validate_infer|federated_validate) validate_infer ;;
  validation_status) validation_status ;;
  validate) validate ;;
  select_global) select_global ;;
  evaluate_dev) evaluate_dev ;;
  lock_final) lock_final ;;
  evaluate_final) evaluate_final ;;
  report) report ;;
  all)
    prepare
    train
    validate_infer
    select_global
    report
    ;;
  ""|-h|--help) usage ;;
  *) die "Unknown command: $cmd" ;;
esac
fi
