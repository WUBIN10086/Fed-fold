#!/bin/bash

# Generated the cache files and moved them to ./data/pdb_recent/
# python scripts/generate_mmcif_cache.py train_subset/ solo_mmcif_cache.json
# python scripts/generate_chain_data_cache.py train_subset/ solo_chain_data_cache.json


# python train_openfold.py solo_train_subset/ solo_alignment_dir/ data/pdb_mmcif/mmcif_files/ output_dir 2026-01-18 --use_single_seq_mode True --config_preset seq_model_esm1b_ptm --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt --resume_model_weights_only True --template_release_dates_cache_path solo_mmcif_cache.json --train_chain_data_cache_path solo_chain_data_cache.json --precision bf16-mixed --gpus 1 --deepspeed_config_path deepspeed_config.json

# Each run writes to its own dated output dir so runs never overwrite each other.
# The original weights in --resume_from_ckpt are read-only (loaded weights-only, never written back).
RUN_DATE=$(date +%y%m%d_%H%M%S)
OUTPUT_DIR=data/pdb_recent/soloseq_finetuning_outputs/run_${RUN_DATE}
mkdir -p "${OUTPUT_DIR}"

# Validation split is created (once) by scripts/make_soloseq_val_split.py, which produces:
#   data/pdb_recent/val_cifs/        held-out structures' .cif (symlinks)
#   data/pdb_recent/val_embeddings/  held-out chains' ESM embeddings (symlinks)
#   data/pdb_recent/train_filter.txt include-list of TRAIN chains (val held out, no leakage)
# --val_check_interval 0.25 -> validate 4x per 10k-step epoch; metrics -> ${OUTPUT_DIR}/lightning_logs/.../metrics.csv
# Start finetuning from the SAME weights the baseline inference used
# (seq_model_esm1b_ptm.pt) so the existing pLDDT CSV is a true "before" and the
# finetuned checkpoint is a clean "after" of the same model lineage.
# (Was seq_model_esm1b_ptm_finetuning_260331.pt, a prior finetune — different lineage.)
python train_openfold.py data/pdb_recent/selected_sequences/ data/pdb_recent/embeddings_output_dir/ data/pdb_recent/mmcif_files/ "${OUTPUT_DIR}" 2026-01-18 --use_single_seq_mode True --config_preset seq_model_esm1b_ptm --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt --resume_model_weights_only True --template_release_dates_cache_path data/pdb_recent/solo_mmcif_cache.json --train_chain_data_cache_path data/pdb_recent/solo_chain_data_cache.json --train_filter_path data/pdb_recent/train_filter.txt --val_data_dir data/pdb_recent/val_cifs/ --val_alignment_dir data/pdb_recent/val_embeddings/ --val_check_interval 0.25 --num_sanity_val_steps 2 --max_epochs 3 --checkpoint_every_epoch --precision bf16-mixed --gpus 2 --seed 42 --deepspeed_config_path deepspeed_config.json