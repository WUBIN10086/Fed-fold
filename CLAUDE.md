# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FedFold is a fork of [OpenFold](https://github.com/aqlaboratory/openfold) (a trainable PyTorch reproduction of AlphaFold 2). The upstream `openfold/` package and `scripts/` are largely unchanged; the FedFold-specific work lives in:

- **Root shell scripts** (`_finetune_openfold.sh`, `_run_all_fasta_in_folders.sh`, `_compute_plddt.sh`, `_download_mmseq.sh`) — wrappers encoding the team's exact training/inference commands and data paths under `data/pdb_recent/`.
- **`docs/source/SoloSeq_Reasoning_Train_Finetune_Repro.md`** and **`docs/source/数据筛选.md`** — the authoritative, project-specific workflow and data-curation docs (in Chinese). Read these before changing pipeline behavior; the upstream `docs/source/*.md` describe generic OpenFold.
- **`download_cif_from_pdb_bank.py`**, **`scripts/plddt_from_pdb.py`** — custom data-download and pLDDT-scoring helpers added for this project.

The project goal (per `数据筛选.md`): curate recent PDB structures (2025–2026) with mean pLDDT < 80 and finetune the SoloSeq single-sequence model on them. Team members split the work by date range and record selections in `docs/selectedPDB/*.csv`.

## Three modeling routes (README)

1. **SoloSeq** — single-sequence, no MSA; uses ESM-1b embeddings. This is the active route (see the SoloSeq doc and `_finetune_openfold.sh`).
2. Standard OpenFold — generate MSAs the DeepMind way.
3. Direct RODA data — fastest path to large-scale training.

## Environment & build

This is a CUDA project; the `attn_core_inplace_cuda` extension is compiled from `openfold/utils/kernel/csrc/` at install time. `setup.py` auto-detects the GPU compute capability via `scripts/utils.get_nvidia_cc` and only emits matching `-gencode` flags, so **build on the target GPU**.

**This machine:** 2× NVIDIA RTX A6000 (49 GB each, compute capability 8.6 / sm_86). This matches `--gpus 2` in `_finetune_openfold.sh`. The installed/active conda env is **`openfold_env`** — use `conda activate openfold_env` (note: `environment.yml` names it `openfold-env` and the SoloSeq doc says `Fedfold`; the real env on this box is `openfold_env`).

```bash
conda activate openfold_env   # existing env on this machine
# To recreate from scratch (pins torch 2.5 + pytorch-cuda 12.4, python 3.10):
conda env create -f environment.yml
python setup.py install   # or: pip install -e .  (rebuilds the CUDA kernel for sm_86)
```

`cutlass/` is a vendored submodule used by the CUDA kernels — do not edit it.

## Common commands

```bash
# Run the full unit test suite (forces CUDA_VISIBLE_DEVICES=0)
bash scripts/run_unit_tests.sh
# Run a single test module / case
python -m unittest tests.test_model            # one module
python -m unittest tests.test_model.TestClass.test_method

# SoloSeq: precompute ESM-1b embeddings, then run pretrained inference
python scripts/precompute_embeddings.py examples/monomer/fasta_dir/ embeddings_output_dir/
python run_pretrained_openfold.py examples/monomer/fasta_dir data/pdb_mmcif/mmcif_files/ \
    --use_precomputed_alignments embeddings_output_dir --output_dir ./ \
    --model_device "cuda:0" --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt

# SoloSeq finetuning on curated recent-PDB data (see _finetune_openfold.sh for exact paths)
bash _finetune_openfold.sh

# Batch-fold every FASTA folder under data/pdb_recent/fasta_files/
bash _run_all_fasta_in_folders.sh

# Score predicted structures by mean pLDDT
bash _compute_plddt.sh    # wraps scripts/plddt_from_pdb.py over prediction outputs
```

Training/finetuning uses DeepSpeed (`deepspeed_config.json`, ZeRO stage 2 + CPU optimizer offload, bf16) via `--deepspeed_config_path`. The SoloSeq finetune runs `train_openfold.py` with `--use_single_seq_mode True --config_preset seq_model_esm1b_ptm --precision bf16-mixed`.

## Training data prep pipeline (SoloSeq route)

Caches must be generated before training. Order matters:

1. Download recent CIFs → `data/pdb_recent/mmcif_files/` (`download_cif_from_pdb_bank.py`).
2. `python scripts/generate_mmcif_cache.py <mmcif_dir> <out>.json` → template release-dates cache.
3. `python scripts/generate_chain_data_cache.py <mmcif_dir> <out>.json` → chain data cache.
4. Generate FASTA, then ESM-1b embeddings (the SoloSeq "alignment" dir).

These cache JSONs are passed to `train_openfold.py` via `--template_release_dates_cache_path` and `--train_chain_data_cache_path`.

## Architecture (upstream OpenFold)

- `openfold/config.py` — single source of truth for all hyperparameters and `--config_preset` definitions (e.g. `seq_model_esm1b_ptm` for SoloSeq, multimer presets, etc.). Most behavior changes route through here.
- `openfold/model/` — the network: `model.py` (AlphaFold top level) → `embedders.py`, `evoformer.py` (+ `msa.py`, `triangular_*`, `outer_product_mean.py`, `pair_transition.py`), `template.py`, `structure_module.py`, `heads.py`. `primitives.py` holds the attention/layernorm building blocks and dispatches to the custom CUDA kernel / DeepSpeed Evoformer attention / FlashAttention.
- `openfold/data/` — feature pipeline (`data_pipeline.py`, `feature_pipeline.py`), MSA/template tools (`data/tools/`), and dataset/datamodule plumbing used by `train_openfold.py`.
- `openfold/np/` — NumPy-side protein representation, relaxation (OpenMM/Amber), and residue constants.
- `openfold/utils/` — losses, rigid/frame math (`rigid_utils.py`), feature transforms, and `kernel/csrc/` (the compiled CUDA attention).

Entry points: `run_pretrained_openfold.py` (inference), `train_openfold.py` (training/finetuning), `thread_sequence.py`.

## Conventions

- Outputs and large data live under `data/pdb_recent/` (mmcif_files, fasta_files, embeddings_output_dir, prediction/finetuning output dirs). Generated PDB files and downloaded DBs are **not** committed (see `数据筛选.md` and `.gitignore`).
- Per-member curated selections go in `docs/selectedPDB/*.csv`, split by date range.
- Bug logs the team has hit are kept in `docs/encountered_bugs/`.
