#!/bin/bash
# =============================================================================
# FedFold — SoloSeq (single-sequence, ESM-1b) finetuning with cluster-balanced
# sampling.  PORTABLE TEMPLATE — edit the CONFIG block for your own data.
#
# What it does (see finetune_openfold_clustered.md for the full write-up):
#   [1] merge all training chains into one FASTA
#   [2] MMseqs2-cluster them at SEQ_ID identity      -> clusters.txt
#   [3] rebuild the chain-data cache WITH cluster_size (from the cluster file)
#   [4] finetune train_openfold.py, sampling each chain ~ 1/cluster_size so
#       over-represented families stop dominating the gradient.
#
# SoloSeq has NO MSA.  The per-target input is an ESM-1b embedding (a
# [L, 1280] feature matrix per chain) produced by scripts/precompute_embeddings.py.
# That embedding directory is what you pass where an MSA "alignment dir" would go.
#
# MMseqs2 clusters ONLY your own sequences against each other — no reference /
# MSA database is needed, just the `mmseqs` binary.
#
# Prereqs you must already have produced for YOUR data (see the .md, "Data prep"):
#   - mmCIF files for the training chains          (CIF_DIR)
#   - ESM-1b embeddings, one <label>/<label>.pt    (EMB_DIR)  <- replaces MSA
#   - template release-dates cache json            (MMCIF_CACHE)
#   - chain-data cache json (pre-clustering)       (CHAIN_CACHE)
#   - train-chain include list                     (TRAIN_FILTER)
#   - a held-out validation split                  (VAL_CIF_DIR, VAL_EMB_DIR)
#   - the SoloSeq base weights .pt                  (BASE_CKPT)
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")"

# ============================== CONFIG (EDIT ME) =============================
# --- your data root; every path below is relative to it unless absolute ---
BASE=${BASE:-data/pdb_recent}

# --- inputs you generated during data prep ---
CIF_DIR=${CIF_DIR:-$BASE/selected_sequences}                 # training mmCIFs
EMB_DIR=${EMB_DIR:-$BASE/embeddings_output_dir}              # ESM-1b embeddings (replaces MSA)
TEMPLATE_MMCIF_DIR=${TEMPLATE_MMCIF_DIR:-$BASE/mmcif_files}  # mmCIFs for template search
MMCIF_CACHE=${MMCIF_CACHE:-$BASE/solo_mmcif_cache.json}      # template release-dates cache
CHAIN_CACHE=${CHAIN_CACHE:-$BASE/solo_chain_data_cache.json} # chain-data cache (pre-clustering)
TRAIN_FILTER=${TRAIN_FILTER:-$BASE/train_filter.txt}         # train-chain include list
VAL_CIF_DIR=${VAL_CIF_DIR:-$BASE/val_cifs}                   # held-out val structures
VAL_EMB_DIR=${VAL_EMB_DIR:-$BASE/val_embeddings}            # held-out val embeddings

# --- model / training knobs ---
BASE_CKPT=${BASE_CKPT:-openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt}
CONFIG_PRESET=${CONFIG_PRESET:-seq_model_esm1b_ptm}
MAX_TEMPLATE_DATE=${MAX_TEMPLATE_DATE:-2026-01-18}  # templates released after this are ignored
MAX_EPOCHS=${MAX_EPOCHS:-3}
GPUS=${GPUS:-2}
SEED=${SEED:-42}
PRECISION=${PRECISION:-bf16-mixed}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-deepspeed_config.json}

# --- clustering knobs ---
MMSEQS_BIN=${MMSEQS_BIN:-mmseqs}   # `conda activate <env>` puts it on PATH, or set an abs path
SEQ_ID=${SEQ_ID:-0.4}              # 0.4 = 40% identity, matches RCSB PDB's official clusters

# --- derived (usually leave as-is) ---
MERGED_FASTA=$BASE/all_chains_for_clustering.fasta
CLUSTERS=$BASE/clusters_seqid${SEQ_ID}.txt
CLUSTERED_CACHE=$BASE/solo_chain_data_cache_clustered.json
# ============================================================================

# ------------------------------ sanity checks -------------------------------
[ -f "$CHAIN_CACHE" ] || { echo "ERROR: missing chain-data cache: $CHAIN_CACHE"; exit 1; }
[ -d "$CIF_DIR" ]     || { echo "ERROR: missing training mmCIF dir: $CIF_DIR"; exit 1; }
[ -d "$EMB_DIR" ]     || { echo "ERROR: missing ESM-1b embedding dir: $EMB_DIR"; exit 1; }
[ -f "$BASE_CKPT" ]   || { echo "ERROR: missing base checkpoint: $BASE_CKPT"; exit 1; }

# ----------------- prep (steps 1-3): build clustered cache ------------------
# One-time CPU work (safe to run while a GPU eval is going). Auto-skipped if the
# clustered cache already exists; force a rebuild with FORCE_PREP=1.
if [ -f "$CLUSTERED_CACHE" ] && [ -z "${FORCE_PREP:-}" ]; then
  echo "[prep] reusing existing $CLUSTERED_CACHE  (set FORCE_PREP=1 to rebuild)"
else
  command -v "$MMSEQS_BIN" >/dev/null 2>&1 || {
    echo "ERROR: mmseqs not found ('$MMSEQS_BIN'). Install (conda install -c bioconda mmseqs2)" >&2
    echo "       or run with MMSEQS_BIN=/abs/path/to/mmseqs" >&2
    exit 1
  }

  echo "[1/4] building merged FASTA from $CHAIN_CACHE"
  # Source sequences from the chain-data cache so IDs match the cache keys exactly.
  python - "$CHAIN_CACHE" "$MERGED_FASTA" <<'PY'
import json, sys
cache = json.load(open(sys.argv[1]))
n = 0
with open(sys.argv[2], "w") as f:
    for name, entry in cache.items():
        seq = (entry.get("seq") or "").strip()
        if seq:
            f.write(f">{name}\n{seq}\n"); n += 1
print(f"    wrote {n} sequences -> {sys.argv[2]}")
PY

  echo "[2/4] clustering at ${SEQ_ID} identity -> $CLUSTERS"
  python scripts/fasta_to_clusterfile.py "$MERGED_FASTA" "$CLUSTERS" "$MMSEQS_BIN" --seq-id "$SEQ_ID"
  echo "    $(wc -l < "$CLUSTERS") clusters for $(grep -c '^>' "$MERGED_FASTA") chains"

  echo "[3/4] rebuilding chain-data cache with cluster_size -> $CLUSTERED_CACHE"
  python scripts/generate_chain_data_cache.py "$CIF_DIR" "$CLUSTERED_CACHE" \
    --cluster_file "$CLUSTERS" --no_workers 8
  python - "$CLUSTERED_CACHE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
pos = sum(1 for v in d.values() if isinstance(v.get("cluster_size"), int) and v["cluster_size"] > 0)
print(f"    chains={len(d)}  with cluster_size>0={pos}")
assert pos > 0, "cluster_size not populated — FASTA headers must match cache keys (check name casing!)"
PY
fi   # end prep

# --------------------------- [4/4] finetune ---------------------------------
RUN_DATE=$(date +%y%m%d_%H%M%S)
OUTPUT_DIR=$BASE/soloseq_finetuning_outputs/run_${RUN_DATE}_clustered
mkdir -p "${OUTPUT_DIR}"
echo "[4/4] finetuning -> ${OUTPUT_DIR}"

# Base weights are loaded weights-only and never written back (read-only lineage).
python train_openfold.py "$CIF_DIR/" "$EMB_DIR/" "$TEMPLATE_MMCIF_DIR/" "${OUTPUT_DIR}" "$MAX_TEMPLATE_DATE" \
  --use_single_seq_mode True \
  --config_preset "$CONFIG_PRESET" \
  --resume_from_ckpt "$BASE_CKPT" \
  --resume_model_weights_only True \
  --template_release_dates_cache_path "$MMCIF_CACHE" \
  --train_chain_data_cache_path "$CLUSTERED_CACHE" \
  --train_filter_path "$TRAIN_FILTER" \
  --val_data_dir "$VAL_CIF_DIR/" \
  --val_alignment_dir "$VAL_EMB_DIR/" \
  --val_check_interval 0.25 \
  --num_sanity_val_steps 2 \
  --max_epochs "$MAX_EPOCHS" \
  --checkpoint_every_epoch \
  --precision "$PRECISION" \
  --gpus "$GPUS" \
  --seed "$SEED" \
  --deepspeed_config_path "$DEEPSPEED_CFG"

echo "done -> ${OUTPUT_DIR}"
echo "checkpoints in ${OUTPUT_DIR}/checkpoints ; metrics in ${OUTPUT_DIR}/lightning_logs/.../metrics.csv"
