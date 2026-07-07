#!/bin/bash
# Driver: evaluate the 3 epoch-end checkpoints of run_260629_152451 in sequence.
# Each calls _eval_checkpoint.sh (inference over all 3210 chains -> pLDDT -> lDDT-Ca).
# Runs on tornade's local GPUs (cuda:0 inside _eval_checkpoint.sh). Logs progress.
set -eo pipefail
cd "$(dirname "$0")"

CKDIR=data/pdb_recent/soloseq_finetuning_outputs/run_260629_152451/checkpoints
declare -A CKPTS=( [ep1]="$CKDIR/0-10000.ckpt" [ep2]="$CKDIR/1-20000.ckpt" [ep3]="$CKDIR/2-30000.ckpt" )

for TAG in ep1 ep2 ep3; do
  echo "======== $(date '+%F %T')  START $TAG  (${CKPTS[$TAG]}) ========"
  # Don't let a single failing checkpoint abort the remaining ones.
  if bash _eval_checkpoint.sh "${CKPTS[$TAG]}" "$TAG"; then
    echo "======== $(date '+%F %T')  DONE  $TAG ========"
  else
    echo "======== $(date '+%F %T')  FAILED $TAG (rc=$?) — continuing ========"
  fi
done
echo "ALL EPOCH EVALS COMPLETE"
