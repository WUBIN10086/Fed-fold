#!/bin/bash
# Evaluate one finetuned checkpoint: run inference over all chains, then compute
# pLDDT (self-confidence) and lDDT-Ca (accuracy vs experimental structures).
#
# Usage:  bash _eval_checkpoint.sh <checkpoint_dir_or_pt> <out_tag>
#   <checkpoint_dir_or_pt>  e.g. .../checkpoints/epoch=0-step=10000.ckpt  (DeepSpeed dir or .pt)
#   <out_tag>               e.g. ep1   -> outputs under data/pdb_recent/ckpt_eval_ep1/
#
# Prereqs (already created): data/pdb_recent/ft_infer_fastas/ and ft_infer_aln/
# (uppercase-tag symlinks bridging to the lowercase embedding dirs).
set -eo pipefail
source /home/j-wang/miniconda3/etc/profile.d/conda.sh
conda activate openfold_env

CKPT="$1"; TAG="$2"
[ -z "$CKPT" ] || [ -z "$TAG" ] && { echo "usage: $0 <ckpt> <out_tag>"; exit 1; }

BASE=data/pdb_recent
OUT=$BASE/ckpt_eval_${TAG}
PRED=$OUT/predictions
mkdir -p "$OUT"

echo "[1/3] inference with $CKPT -> $PRED"
NFASTA=$(ls "$BASE/ft_infer_fastas" 2>/dev/null | wc -l)
# Guard: find on a missing dir exits non-zero, and with `pipefail` that kills
# the whole script (set -e) at this assignment. Only count if $PRED exists.
if [ -d "$PRED" ]; then NPRED=$(find "$PRED" -name '*unrelaxed.pdb' | wc -l); else NPRED=0; fi
if [ "$NPRED" -ge "$NFASTA" ] && [ "$NFASTA" -gt 0 ]; then
  echo "    skipping inference: $NPRED/$NFASTA predictions already present in $PRED"
else
  python run_pretrained_openfold.py $BASE/ft_infer_fastas $BASE/mmcif_files \
    --use_precomputed_alignments $BASE/ft_infer_aln \
    --output_dir "$OUT" \
    --model_device cuda:0 --config_preset seq_model_esm1b_ptm \
    --openfold_checkpoint_path "$CKPT" \
    --skip_relaxation > "$OUT/inference.log" 2>&1
fi
echo "    predictions: $(find $PRED -name '*unrelaxed.pdb' | wc -l)"

echo "[2/3] pLDDT (self-confidence)"
: > "$OUT/plddt_per_chain.csv"
NSKIP=0
for f in "$PRED"/*unrelaxed.pdb; do
  # `|| m=""` is essential: plddt_from_pdb.py exits non-zero on a degenerate/
  # empty PDB (e.g. 8ZOE_g), and with `pipefail` the failing pipeline would kill
  # the script at THIS assignment (set -e) before the guard below runs.
  m=$(python scripts/plddt_from_pdb.py "$f" 2>/dev/null | awk '/^mean_pLDDT/{print $2}') || m=""
  t=$(basename "$f" | sed 's/_seq_model.*//')
  # Use if/then, not `[ -n "$m" ] && echo` — the latter returns non-zero when
  # $m is empty and `set -e` then kills the eval. Skip unscoreable preds instead.
  if [ -n "$m" ]; then
    echo "$t,$m" >> "$OUT/plddt_per_chain.csv"
  else
    NSKIP=$((NSKIP + 1))
    echo "    [warn] no pLDDT for $t (empty/degenerate PDB) — skipped" >&2
  fi
done
[ "$NSKIP" -gt 0 ] && echo "    skipped $NSKIP unscoreable prediction(s)"
echo "    wrote $OUT/plddt_per_chain.csv ($(wc -l < "$OUT/plddt_per_chain.csv") chains)"

echo "[3/3] lDDT-Ca (accuracy vs experiment)"
CUDA_VISIBLE_DEVICES="" python scripts/lddt_from_predictions.py \
  "$PRED" $BASE/mmcif_files -o "$OUT/lddt_per_chain.csv"

echo "done. Compare to baseline:"
echo "  pLDDT baseline: $BASE/soloseq_inference_outputs/plddt_compute.log"
echo "  lDDT  baseline: $BASE/baseline_lddt.csv"
