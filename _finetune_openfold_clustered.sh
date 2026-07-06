#!/bin/bash
# SoloSeq finetuning WITH sequence-identity cluster-balanced sampling.
#
# Difference vs _finetune_openfold.sh: before training we (1) MMseqs2-cluster the
# training chains at 40% identity and (2) rebuild the chain-data cache so every
# chain carries a `cluster_size`. OpenFold's data module then samples each chain
# with probability ~ 1/cluster_size (openfold/data/data_modules.py:616-618), so
# over-represented families stop dominating the gradient. No training-code change.
#
# MMseqs2 here clusters ONLY our own sequences against each other — it needs the
# `mmseqs` binary but NO reference/MSA database (uniref/bfd/etc. are irrelevant to
# the SoloSeq route).
#
# Install mmseqs once if missing:  conda install -n openfold_env -c bioconda mmseqs2
# Then either put it on PATH or pass its path:  MMSEQS_BIN=/path/to/mmseqs bash _finetune_openfold_clustered.sh
set -eo pipefail
cd "$(dirname "$0")"

# ---- knobs ----------------------------------------------------------------
BASE=data/pdb_recent
MMSEQS_BIN=${MMSEQS_BIN:-mmseqs}   # override with MMSEQS_BIN=/abs/path/mmseqs
SEQ_ID=${SEQ_ID:-0.4}              # 40% identity — matches RCSB PDB's official clusters
CIF_DIR=$BASE/selected_sequences                 # training mmCIFs (same as base script)
SRC_CACHE=$BASE/solo_chain_data_cache.json       # existing cache (no cluster_size yet)
CLUSTERS=$BASE/clusters_seqid${SEQ_ID}.txt       # one cluster per line -> generated
CLUSTERED_CACHE=$BASE/solo_chain_data_cache_clustered.json   # new cache WITH cluster_size
MERGED_FASTA=$BASE/all_chains_for_clustering.fasta
# ---------------------------------------------------------------------------

# ---- sanity checks --------------------------------------------------------
command -v "$MMSEQS_BIN" >/dev/null 2>&1 || {
  echo "ERROR: mmseqs binary not found ('$MMSEQS_BIN')." >&2
  echo "       install it (conda install -n openfold_env -c bioconda mmseqs2)" >&2
  echo "       or run with MMSEQS_BIN=/abs/path/to/mmseqs bash $0" >&2
  exit 1
}
[ -f "$SRC_CACHE" ] || { echo "ERROR: missing $SRC_CACHE (run generate_chain_data_cache.py first)"; exit 1; }
[ -d "$CIF_DIR" ]   || { echo "ERROR: missing $CIF_DIR"; exit 1; }

# ---- prep (steps 1-3): build cluster file + clustered cache ----------------
# Skipped automatically if the clustered cache already exists (prep is a one-time
# CPU step, safe to run while a GPU eval is going). Force a rebuild with FORCE_PREP=1.
if [ -f "$CLUSTERED_CACHE" ] && [ -z "${FORCE_PREP:-}" ]; then
  echo "[prep] reusing existing $CLUSTERED_CACHE  (set FORCE_PREP=1 to rebuild)"
else

# ---- [1/4] build a single FASTA of all chains, headers >PDBID_CHAIN --------
# Source the sequences from the existing chain-data cache so the IDs are exactly
# the cache keys (generate_chain_data_cache.py uppercases both sides when it maps
# clusters -> chains, so lower/upper case does not matter here).
echo "[1/4] building merged FASTA from $SRC_CACHE"
python - "$SRC_CACHE" "$MERGED_FASTA" <<'PY'
import json, sys
cache_path, out_path = sys.argv[1], sys.argv[2]
cache = json.load(open(cache_path))
n = 0
with open(out_path, "w") as f:
    for name, entry in cache.items():
        seq = (entry.get("seq") or "").strip()
        if seq:
            f.write(f">{name}\n{seq}\n")
            n += 1
print(f"    wrote {n} sequences -> {out_path}")
PY

# ---- [2/4] cluster with MMseqs2 (PDB-identical flags, via the repo script) --
echo "[2/4] clustering at ${SEQ_ID} identity -> $CLUSTERS"
python scripts/fasta_to_clusterfile.py "$MERGED_FASTA" "$CLUSTERS" "$MMSEQS_BIN" --seq-id "$SEQ_ID"
echo "    $(wc -l < "$CLUSTERS") clusters for $(grep -c '^>' "$MERGED_FASTA") chains"

# ---- [3/4] rebuild chain-data cache WITH cluster_size ----------------------
echo "[3/4] regenerating chain-data cache with cluster sizes -> $CLUSTERED_CACHE"
python scripts/generate_chain_data_cache.py "$CIF_DIR" "$CLUSTERED_CACHE" \
  --cluster_file "$CLUSTERS" --no_workers 8
# verify cluster_size actually landed (should be > 0 for the vast majority)
python - "$CLUSTERED_CACHE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
have = [v.get("cluster_size") for v in d.values()]
pos  = sum(1 for c in have if isinstance(c, int) and c > 0)
neg  = sum(1 for c in have if c == -1)
print(f"    chains: {len(d)}  with cluster_size>0: {pos}  unmatched(-1): {neg}")
assert pos > 0, "cluster_size not populated — check that FASTA headers match cache keys"
PY

fi   # end prep

# ---- [4/4] finetune (identical to _finetune_openfold.sh, clustered cache) --
RUN_DATE=$(date +%y%m%d_%H%M%S)
OUTPUT_DIR=$BASE/soloseq_finetuning_outputs/run_${RUN_DATE}_clustered
mkdir -p "${OUTPUT_DIR}"
echo "[4/4] finetuning -> ${OUTPUT_DIR}"

# Start from the SAME stock weights as the baseline (read-only, weights-only load),
# so before/after numbers stay on one lineage. Only --train_chain_data_cache_path
# changed vs the non-clustered run.
python train_openfold.py "$CIF_DIR/" "$BASE/embeddings_output_dir/" "$BASE/mmcif_files/" "${OUTPUT_DIR}" 2026-01-18 \
  --use_single_seq_mode True \
  --config_preset seq_model_esm1b_ptm \
  --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
  --resume_model_weights_only True \
  --template_release_dates_cache_path "$BASE/solo_mmcif_cache.json" \
  --train_chain_data_cache_path "$CLUSTERED_CACHE" \
  --train_filter_path "$BASE/train_filter.txt" \
  --val_data_dir "$BASE/val_cifs/" \
  --val_alignment_dir "$BASE/val_embeddings/" \
  --val_check_interval 0.25 \
  --num_sanity_val_steps 2 \
  --max_epochs 3 \
  --checkpoint_every_epoch \
  --precision bf16-mixed \
  --gpus 2 \
  --seed 42 \
  --deepspeed_config_path deepspeed_config.json

echo "done -> ${OUTPUT_DIR}"
echo "evaluate with: bash _eval_all_epochs.sh   (point CKDIR at ${OUTPUT_DIR}/checkpoints)"
