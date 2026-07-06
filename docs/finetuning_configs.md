# SoloSeq Fine-tuning Configuration

Reference for the current SoloSeq (single-sequence, ESM-1b) fine-tuning run, for sharing with the team.
Source of truth: [`_finetune_openfold.sh`](../_finetune_openfold.sh), [`scripts/prep_solo_fasta_learnable.py`](../scripts/prep_solo_fasta_learnable.py), [`deepspeed_config.json`](../deepspeed_config.json).

**Two fine-tuning scripts:**
- [`_finetune_openfold.sh`](../_finetune_openfold.sh) — the baseline run (uniform per-chain sampling).
- [`_finetune_openfold_clustered.sh`](../_finetune_openfold_clustered.sh) — same config **+ sequence-identity cluster-balanced sampling** (see §4). Recommended.

The run currently being evaluated is **`run_260629_152451`** (baseline, 3 epochs, checkpoints per epoch — see [`_eval_all_epochs.sh`](../_eval_all_epochs.sh)).

---

## 1. Base model (starting weights)

| Item | Value |
|------|-------|
| **Checkpoint** | **`openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt`** |
| Lineage | Stock OpenFold SoloSeq weights (ESM-1b + pTM head), dated **2023-11-30** |
| Config preset | `seq_model_esm1b_ptm` (defined in [`openfold/config.py`](../openfold/config.py)) |
| Sequence embedder | **ESM-1b** single-sequence embeddings (`.pt`), **no MSA** |
| How it's loaded | `--resume_from_ckpt … --resume_model_weights_only True` → weights only, optimizer/step state **not** restored. The `.pt` is read-only and never written back. |

> **Why this file and not `seq_model_esm1b_ptm_finetuning_260331.pt`?**
> We start from the *same* stock weights the baseline inference used, so the existing baseline pLDDT/lDDT numbers are a true "before" and this checkpoint is a clean "after" of the same lineage. `…_finetuning_260331.pt` is a *prior* fine-tune (different lineage) and is intentionally not used here.
> Also present but unused: `seq_model_esm1b_noptm.pt` (no pTM head).

---

## 2. Data & sequence filtering

Data source: **recent PDB structures (2025–2026)** downloaded as mmCIF into `data/pdb_recent/mmcif_files/`
(see [`download_cif_from_pdb_bank.py`](../download_cif_from_pdb_bank.py)). Team members split the download by date range and record selections in `docs/selectedPDB/*.csv`.

### Filtering — `scripts/prep_solo_fasta_learnable.py`

The script reads each `.cif`, splits it per chain, and writes one `PDBID_chain.fasta` per surviving chain (label = `1abc_A`). See [`scripts/prep_solo_fasta_learnable.md`](../scripts/prep_solo_fasta_learnable.md) for the full write-up. Filters applied (in order):

| Filter | Flag | Value used | Meaning |
|--------|------|-----------|---------|
| Target whitelist | `--target_labels_file` | `sequence_plddt_lower_than_80.csv` | **Only chains whose baseline mean pLDDT < 80** are eligible — i.e. we fine-tune on the cases the stock model is *least* confident on. |
| Min length | `--min_len` | `30` | Drop very short chains. |
| Max length | `--max_len` | `1022` | ESM-1b hard limit. |
| Max `X` ratio | `--max_x_ratio` | `0.05` | ≤ 5 % unknown residues. |
| Max non-canonical ratio | `--max_noncanonical_ratio` | `0.05` | ≤ 5 % residues outside the 20 standard AAs. |
| Max resolution | `--max_resolution` | `4.0` Å | Drop structures worse than 4.0 Å (and any with unparseable resolution). |
| Dedup | `--dedup_by_sequence` | on | Keep only the first occurrence of each **exact** sequence. |

Reports land in `<output>/_reports/`: `kept_samples.csv`, `dropped_samples.csv` (with reason), `kept_labels.txt`, `dropped_labels.txt`, `summary.txt`.

**`kept_labels.txt` → `--train_filter_path`.** This whitelist is what keeps the training set aligned with the chains that actually have a FASTA + precomputed embedding, so training never reads a chain with no embedding.

### Pipeline order

1. Download recent CIFs → `data/pdb_recent/mmcif_files/`.
2. `generate_mmcif_cache.py` → `solo_mmcif_cache.json` (template release-dates cache).
3. `generate_chain_data_cache.py` → `solo_chain_data_cache.json` (chain data cache).
4. `prep_solo_fasta_learnable.py` → filtered FASTAs + `kept_labels.txt`.
5. `precompute_embeddings.py` → ESM-1b embeddings (the SoloSeq "alignment" dir).
6. `make_soloseq_val_split.py` → held-out `val_cifs/`, `val_embeddings/`, and `train_filter.txt` (no train/val leakage).

### Approximate data volume (this run)

| Set | Count |
|-----|-------|
| Chains in `sequence_plddt_lower_than_80.csv` (pLDDT<80 candidates) | ~4,443 |
| Chains in chain-data cache | ~2,673 |
| Precomputed embeddings (`embeddings_output_dir/`) | ~3,212 |
| **Train** chains (`train_filter.txt`) | **~2,479** |
| **Validation** — structures / chains | **34 / ~121** |

> Numbers are from the current working tree; regenerate the `_reports/summary.txt` for exact per-run figures.

---

## 3. Training hyper-parameters

Command: [`_finetune_openfold.sh`](../_finetune_openfold.sh) → `train_openfold.py`.

| Parameter | Value |
|-----------|-------|
| Mode | `--use_single_seq_mode True` (SoloSeq, no MSA) |
| Config preset | `seq_model_esm1b_ptm` |
| Precision | `bf16-mixed` |
| GPUs | `2` × NVIDIA RTX A6000 (49 GB, sm_86) |
| Max epochs | `3` (≈ 10,000 steps/epoch → checkpoints `0-10000`, `1-20000`, `2-30000`) |
| Checkpointing | `--checkpoint_every_epoch` |
| Validation | `--val_check_interval 0.25` (4× per epoch); `--num_sanity_val_steps 2` |
| Seed | `42` |
| Caches | `--template_release_dates_cache_path solo_mmcif_cache.json`, `--train_chain_data_cache_path solo_chain_data_cache.json` |
| Template cutoff date | `2026-01-18` |
| Distributed backend | DeepSpeed (`--deepspeed_config_path deepspeed_config.json`) |

**DeepSpeed** (`deepspeed_config.json`): ZeRO **stage 2**, optimizer offload to **CPU**, `bfloat16` enabled (fp16/amp off), gradient clipping **0.1**, activation checkpointing with `partition_activations`.

Each run writes to its own dated dir: `data/pdb_recent/soloseq_finetuning_outputs/run_<date>/`. Lightning metrics → `…/lightning_logs/.../metrics.csv`.

### Evaluation

[`_eval_all_epochs.sh`](../_eval_all_epochs.sh) → [`_eval_checkpoint.sh`](../_eval_checkpoint.sh) runs, per epoch-end checkpoint:
1. Inference over all chains (`run_pretrained_openfold.py`, `--skip_relaxation`).
2. **pLDDT** (self-confidence) per chain → `plddt_per_chain.csv`.
3. **lDDT-Cα** (accuracy vs. experimental structure) → `lddt_per_chain.csv`.

Baselines to compare against: `data/pdb_recent/soloseq_inference_outputs/plddt_compute.log` (pLDDT) and `data/pdb_recent/baseline_lddt.csv` (lDDT).

---

## 4. Grouping the data to improve finetuning — cluster-balanced sampling

**Yes, grouping helps. The highest-value grouping is sequence-identity clustering with inverse-cluster-size sampling, and it is now implemented in [`_finetune_openfold_clustered.sh`](../_finetune_openfold_clustered.sh).**

### The problem it fixes

The base run (`_finetune_openfold.sh`) only does **exact-sequence** dedup (`--dedup_by_sequence`). Near-identical homologs — point mutants, the same protein re-crystallized, homomer chains — still enter as separate chains, and every chain is sampled with **equal probability**. Over-represented families then dominate the gradient while rare folds are undertrained. This is the same redundancy bias AlphaFold-2 / OpenFold's pipeline was built to remove.

### How OpenFold supports the fix (no training-code change)

The data module already does **inverse-cluster-size sampling** — [`openfold/data/data_modules.py:616-618`](../openfold/data/data_modules.py#L616-L618):

```python
cluster_size = cache_entry.get("cluster_size", None)
if cluster_size is not None and cluster_size > 0:
    probabilities.append(1 / cluster_size)
```

A chain in a cluster of size *N* is sampled with weight *1/N*, so each cluster contributes roughly equally regardless of how many PDB entries it has. All that's needed is a chain-data cache whose entries carry `cluster_size`. The default `solo_chain_data_cache.json` has it on **0** chains (uniform sampling); the clustered pipeline produces one that has it on all of them.

### What `_finetune_openfold_clustered.sh` does

Identical to `_finetune_openfold.sh` except it runs three CPU-only prep steps first, then points `--train_chain_data_cache_path` at the clustered cache:

1. **Merge FASTA** — build one FASTA of all chains from `solo_chain_data_cache.json` (headers `>PDBID_CHAIN`, IDs guaranteed to match the cache).
2. **Cluster** — `scripts/fasta_to_clusterfile.py … --seq-id 0.4` runs MMseqs2 `easy-cluster` at **40 % identity / 90 % coverage** (flags identical to RCSB PDB's official sequence clusters). This clusters **our own sequences against each other — no MSA / reference database needed**, only the `mmseqs` binary. → `data/pdb_recent/clusters_seqid0.4.txt`.
3. **Rebuild cache** — `scripts/generate_chain_data_cache.py … --cluster_file` writes `data/pdb_recent/solo_chain_data_cache_clustered.json` with a `cluster_size` on every chain (asserts it was populated).

Prep is idempotent and **auto-skipped** if the clustered cache already exists (`FORCE_PREP=1` forces a rebuild). Training then runs unchanged.

**Requirements / gotchas:**
- `mmseqs` must be on PATH. It lives in `openfold_env` (`/home/j-wang/miniconda3/envs/openfold_env/bin/mmseqs`), so `conda activate openfold_env` is enough; otherwise pass `MMSEQS_BIN=/abs/path/mmseqs`.
- Training uses `--gpus 2`. **Do not launch while a GPU eval is running** — it will contend for both GPUs. The prep steps (1–3) are CPU-only and safe to run alongside an eval.

```bash
conda activate openfold_env      # puts mmseqs on PATH
bash _finetune_openfold_clustered.sh
```

### Result on the current dataset (2026-07-06)

| Metric | Value |
|--------|-------|
| Chains clustered | 2,673 |
| Clusters at 40 % identity | **930** (avg ≈ 2.9 chains/cluster) |
| Largest clusters | 57, 52, 29, 28 chains |
| Chains with valid `cluster_size` | 2,673 / 2,673 (0 unmatched) |

The largest family had **57 near-identical chains** — previously sampled ~57× more often than a unique protein; now down-weighted to contribute like a single one. This confirms real redundancy was present to correct.

### Other groupings — and whether they're worth it

| Grouping | Helpful? | Why |
|----------|----------|-----|
| **Sequence-identity clusters (40 %) + 1/N sampling** | **Done** | Implemented in `_finetune_openfold_clustered.sh`; standard AF2/OpenFold practice. |
| Cluster-aware **train/val split** | **Recommended next** | `make_soloseq_val_split.py` splits by whole structure but not by *cluster*. A val chain that is a 40 %+ homolog of a train chain makes val lDDT optimistic (leakage). Splitting along cluster boundaries would close this. Not yet done. |
| **Structure-based clustering** (Foldseek on the CIFs) | Optional A/B | Sequence clustering ignores 3D and misses *remote* homologs (same fold, <25 % identity). Foldseek clusters by structure and plugs into the *same* pipeline (only step 2 changes). Worth trying if results plateau — we already have all the CIFs. |
| **Length bins** for batching | Minor | Reduces padding waste; a throughput win, not accuracy. Low priority with `--gpus 2` + CPU offload. |
| **Difficulty / curriculum** (by baseline pLDDT) | Maybe | Data is already restricted to pLDDT<80; staging easy→hard adds complexity with unclear payoff at this scale. |

**Bottom line:** sequence-identity clustering is in place and validated. The most valuable remaining step is making the **train/val split cluster-aware** so the before/after numbers aren't inflated by homolog leakage. Foldseek (structure-based) is a cheap follow-up experiment since the pipeline is method-agnostic.
