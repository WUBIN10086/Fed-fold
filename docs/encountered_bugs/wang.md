# Bugs encountered during Openfold finetuning

### Prerequisites
- Inputted soloseq MSA output into funetuning. But the formats are different, so used an AI written functions, which should be incorrect, to parse the contents in xxx.pt files.

### Finetuing command
\$ python train_openfold.py data/pdb_recent/selected_sequences/ data/pdb_recent/embeddings_output_dir/ data/pdb_recent/mmcif_files/ data/pdb_recent/soloseq_finetuning_outputs 2026-01-18 --use_single_seq_mode True --config_preset seq_model_esm1b_ptm --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm_finetuning_260331.pt --resume_model_weights_only True --template_release_dates_cache_path data/pdb_recent/solo_mmcif_cache.json --train_chain_data_cache_path data/pdb_recent/solo_chain_data_cache.json --precision bf16-mixed --gpus 2 --seed 42 --deepspeed_config_path deepspeed_config.json

### Root cause and solution by Claude (the "format" issue was a red herring)
Two misconceptions, one real bug:

1. **SoloSeq does not take an MSA at all.** In `--use_single_seq_mode True`, the data pipeline never parses an MSA. It fabricates a dummy MSA from the single query sequence and reads an **ESM-1b per-residue embedding** from a `.pt` file. See `_process_seqemb_features` (openfold/data/data_pipeline.py:849-862): it globs the alignment subdir for any `.pt` and pulls `seqemb_data["representations"][33]` (a `[L, 1280]` tensor). So feeding/reformatting an LLM-generated MSA can never work — the model has no MSA input port. The AI-written `.pt` parser was solving the wrong problem.

2. **The actual failure was a filename CASE mismatch**, which produced a silently empty dataset. The dataset filters chains to those present in `chain_data_cache` (openfold/data/data_modules.py:164-167), then resolves the cif via `file_id = name.rsplit('_',1)[0]`. These four things must share identical names, and they didn't:
   - embedding subdirs were UPPERCASE PDB id: `embeddings_output_dir/7IAU_A/`
   - `solo_chain_data_cache.json` keys were lowercase: `7iau_A`
   - `solo_mmcif_cache.json` keys were lowercase: `7iau`
   - cif files were lowercase: `selected_sequences/7iau.cif`

   Result: `embeddings ∩ chain_data_cache = 0` → 0 training chains.

### Fix
Renamed every embedding subdir to the **lowercase-PDB, uppercase-chain** form used everywhere else (`7IAU_A` → `7iau_A`); the chain letter stays uppercase. The `.pt` filename *inside* the subdir does not matter (it is globbed), so only the directory names were changed — no `.pt`, cache, or cif files were touched.

```python
import os
base = 'data/pdb_recent/embeddings_output_dir'
def lc(name):
    if '_' in name:
        a, b = name.rsplit('_', 1)
        return a.lower() + '_' + b   # lowercase PDB id, keep chain letter
    return name.lower()
for e in os.listdir(base):
    src = os.path.join(base, e)
    if os.path.isdir(src):
        dst = os.path.join(base, lc(e))
        if src != dst and not os.path.exists(dst):
            os.rename(src, dst)
```

After the rename: 2600 chains match across all four sources, and every one resolves to a valid `.cif`, a `.pt`, and a template release-date entry. The finetune command above then runs as-is.

### Prevention
`scripts/precompute_embeddings.py` derives each embedding subdir name from the FASTA **filename**. Generate the FASTAs with lowercase PDB ids (`7iau_A.fasta`) so the embeddings stay consistent with the caches and cif files. Do **not** attempt to feed an MSA into the SoloSeq route.

# Finetuning results (run_260616_180217, 2026-06-16)

### Setting
- **Command**: `_finetune_openfold.sh` (resume *weights-only* from `seq_model_esm1b_ptm_finetuning_260331.pt`, an already-finetuned SoloSeq checkpoint).
- **Model**: AlphaFold SoloSeq, 75.3 M trainable params, `--config_preset seq_model_esm1b_ptm`, `--use_single_seq_mode True`.
- **Hardware / precision**: 2× NVIDIA RTX A6000 (48 GB), `bf16-mixed`, DeepSpeed ZeRO-2 with CPU optimizer offload, gradient clipping 0.1.
- **Schedule**: `--max_epochs 1` with `--train_epoch_len 10000` ⇒ **10,000 optimizer steps**. An "epoch" here is *not* one pass over the data — the pipeline samples chains stochastically with replacement, so 1 epoch is just a bookkeeping unit of `train_epoch_len` steps. One epoch is the standard OpenFold convention; raise `train_epoch_len`/`max_epochs` for more training.
- **Data**: 2600 matched chains / 280 structures (intersection of embeddings ∩ chain_data_cache ∩ cif). No validation set. No logger configured.
- **Runtime**: ~15 h wall-clock, ~0.19 it/s. Completed cleanly, checkpoint at `run_260616_180217/checkpoints/epoch=0-step=10000.ckpt` (434 MB model + 863 MB optimizer states). No NaN/Inf, no crash.

### Observation (train/loss, only metric captured)
Mean `train/loss` per 1000-step bin:

| steps | mean | std |
|---|---|---|
| 1–1000 | 24.86 | 8.41 |
| 5001–6000 | 23.31 | 8.32 |
| 9001–10000 | 22.59 | 8.04 |

- First-500 vs last-500 mean: **24.65 → 22.30**. Total drop over the whole run ≈ **9% (24.9 → 22.6)**, while per-step std (~8.4) is ~4× the improvement.
- 243/10000 steps had loss < 5 (short/easy chains); no exact-zero losses.

### Results / interpretation
- Training is **nearly flat** — a weak downward trend buried in high per-sample variance. Plausible causes: (1) resuming from an *already-finetuned* checkpoint leaves little to gain; (2) tiny effective batch (~2) over only 10k steps; (3) FAPE-dominated loss is inherently noisy across variable-length chains.
- **Cannot judge generalization**: no validation (`num_sanity_val_steps=0`, zero `val/` lines) and **no logger** (dozens of "no logger configured" warnings) — only the scalar `train/loss` from the progress bar survived; component losses (FAPE, lddt_ca, distogram, …) were not persisted.
- **Next step to actually know if it helped**: evaluate this checkpoint vs. the base `..._260331.pt` on held-out structures (lDDT/TM). Flat train loss alone is inconclusive.

### The "Removing 612 alignment entries" warning — benign, expected
Cause: **the chain_data_cache covers fewer structures than the embeddings dir** — not a casing bug this time.
- `embeddings_output_dir/` has **3211 chain dirs across 506 distinct PDBs**.
- `solo_chain_data_cache.json` has **2673 keys across only 280 PDBs** (built from the smaller `train_subset/`).
- The dataset keeps only chains present in the cache (`data_modules.py:160-167`), so the **611 embedding dirs whose PDB is absent from the cache are dropped** (warning rounds to 612). All 611 fail at the *whole-structure* level (226 PDBs, 225 of which still have a cif present) — i.e. those structures were simply never added to the cache, not corrupted.
- Verified **not** a case issue: 0 of the dropped names match a cache key case-insensitively.
- One stray non-PDB embedding dir exists: `p62068(canonical)` (a UniProt accession) — minor hygiene, harmless.
- **To recover those chains**: regenerate `solo_chain_data_cache.json` (and `solo_mmcif_cache.json`) over the full `selected_sequences/` set rather than `train_subset/`. Otherwise the drop is harmless — training ran on the 2600 intersection as intended.

### Added: validation + logging (for the next run)
- `scripts/make_soloseq_val_split.py` carves a held-out val split **by whole structure** (no homomer/identical-chain leakage), via symlinks (originals untouched): 121 chains / 34 structures held out, 2479 train chains. Writes `val_cifs/`, `val_embeddings/`, and `train_filter.txt` (train include-list).
- `train_openfold.py` changes: (1) **CSVLogger fallback** when `--wandb` is off, so train/val metrics persist to `<output_dir>/lightning_logs/version_*/metrics.csv` (fixes the silent "no logger" problem); (2) new `--val_check_interval` arg (validate N times within an epoch instead of only at the end).
- `_finetune_openfold.sh` now passes `--train_filter_path`, `--val_data_dir`, `--val_alignment_dir`, `--val_check_interval 0.25` (validate 4×/epoch), `--num_sanity_val_steps 2` (baseline val before training). Early stopping monitors `val/lddt_ca` (mode=max) if `--early_stopping` is enabled.

# Inference pLDDT summary (soloseq_inference_outputs, 2026-06-25)

Computed exactly as `_compute_plddt.sh` (per-residue CA pLDDT from the PDB B-factor column via `scripts/plddt_from_pdb.py`) over every `*unrelaxed.pdb` in `data/pdb_recent/soloseq_inference_outputs/predictions/`.
- **Per-file log**: `data/pdb_recent/soloseq_inference_outputs/plddt_compute.log` (full per-prediction global/bin/per-chain/extremes output).
- **Low-confidence list**: `data/pdb_recent/sequence_plddt_lower_than_80.csv` (chain id + mean pLDDT for every prediction with mean < 80). Prior CSV backed up to `*.bak_260625_173017` before this run (the script appends, so it was truncated first to avoid duplicate rows).

### Results
- **3210** unrelaxed predictions processed; **3202** scored. **8 were empty (0-byte) PDB files** — failed/incomplete inference outputs, not a pLDDT bug: `8ZOE_g, 9KY7_L, 9PF1_v, 9PZR_E/F/G/H, 9V27_sH`.
- Distribution of **per-prediction mean pLDDT** (n=3202): mean **74.03**, std 15.47, median 77.17, range 26.36–98.05 (p10=51.3, p25=64.8, p75=85.8, p90=91.4).

| mean-pLDDT bin | predictions | % |
|---|---|---|
| <50 | 282 | 8.8% |
| 50–70 | 783 | 24.5% |
| 70–80 | 764 | 23.9% |
| 80–90 | 912 | 28.5% |
| ≥90 | 461 | 14.4% |

- **57.1%** of predictions have mean pLDDT < 80 (1829 chains, the CSV); only **14.4%** are high-confidence (≥90). Consistent with single-sequence (MSA-free) SoloSeq prediction, which is expected to be lower-confidence than MSA-based OpenFold.
- Note: these predictions predate the finetuning run analyzed above — they are a baseline of the inference model, not an evaluation of `epoch=0-step=10000.ckpt`.

