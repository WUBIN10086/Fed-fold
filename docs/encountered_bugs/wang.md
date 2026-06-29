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

# Finetuned vs baseline pLDDT comparison (2026-06-25, PARTIAL — short chains only)

Re-ran inference with the finetuned checkpoint `run_260616_180217/checkpoints/epoch=0-step=10000.ckpt` (loaded via `run_pretrained_openfold.py`, which auto-consolidates the DeepSpeed dir and uses the EMA weights) and compared per-chain mean pLDDT against the baseline (`seq_model_esm1b_ptm.pt`, in `plddt_compute.log`).

### Setup / how it was run
- Embeddings had been renamed to lowercase on Jun 16, but FASTA tags are uppercase (`13SI_A`). Bridged with uppercase symlinks: `data/pdb_recent/ft_infer_aln/<TAG>` → lowercase embedding dir, and a flat `ft_infer_fastas/`. Ran once (model loads once) with `--skip_relaxation` (pLDDT lives in the unrelaxed PDB, so relaxation is irrelevant and skipping it is ~faithful + far faster).
- Output: `data/pdb_recent/soloseq_inference_finetuned/predictions/`. Compare script: `scripts/compare_plddt_finetuned_vs_baseline.py` → `data/pdb_recent/plddt_finetuned_vs_baseline.csv`.
- **Bug fixed to make this run**: `openfold/np/protein.py:to_pdb` raised `UnboundLocalError: chain_tag` on degenerate predictions (all atoms of a residue masked). Hoisted the `chain_tag` assignment above the inner atom loop.

### ⚠️ Incomplete + biased sample
Inference was **killed at 938/3210 chains (29%)**; 934 scored. Targets are processed **shortest-sequence-first**, so the completed set is the **short proteins only**: mean length **88** residues (median 95, max 139) vs **264** (median 215, max 985) across the full set. The numbers below do **not** generalize to longer chains.

### Results (n=934, short chains)
- Mean pLDDT **76.89 → 74.98** (mean delta **−1.90**, median **−4.40**, std 9.77, range −20.6 … +49.0).
- **67.9% of chains worsened**, 32.1% improved. |delta|≥5: down 45.8%, up 16.4%.
- Confidence-band (80) migration: stayed ≥80: 269; **dropped ≥80→<80: 213**; rose <80→≥80: 73; stayed <80: 379 → net **+140 chains pushed below 80**.
- Extremes: biggest drops `9DNI_A/B 86.6→66.0 (−20.6)`, `8ZP8_B5 78.9→62.5`. Biggest gains are baseline near-failures **rescued**: `7IPI_A/7ICX_A/7IBY_A 31.5→80.5 (+49)`, `7IEF_A/7IDX_A 31.6→80.5`.

### Interpretation
On short chains, this finetuning **lowered the model's self-confidence on average** (most chains down a few pLDDT points) while **rescuing a handful of catastrophic baseline failures** (~31 → ~80). Caveats: (1) pLDDT is confidence, not accuracy — a real verdict needs lDDT/TM vs ground truth; (2) lineages differ (baseline = original `seq_model_esm1b_ptm.pt`; finetune started from `..._finetuning_260331.pt`); (3) sample is short-chain-biased and only 29% complete. **To finish**: resume inference on the remaining 2272 chains, then rerun the compare script.

# 3-epoch finetuning experiment, clean lineage (job 40, submitted 2026-06-29) — IN PROGRESS

Addresses the lineage + epoch-count + accuracy-metric gaps above. Confirmed the previous setup used `--max_epochs` default = **1** (one "epoch" = `train_epoch_len` 10000 stochastically-sampled chains, ≈4 passes over the 2479-chain train set, NOT one traversal).

### Configuration changes (`_finetune_openfold.sh`)
1. **Clean lineage**: `--resume_from_ckpt` → `seq_model_esm1b_ptm.pt` (the SAME weights the baseline pLDDT/lDDT CSVs were generated from), so baseline = true "before".
2. **3 epochs, checkpoint each**: `--max_epochs 3 --checkpoint_every_epoch` (`ModelCheckpoint(every_n_epochs=1, save_top_k=-1)` keeps all 3). Pick the best epoch by `val/lddt_ca`.
3. Validation (added earlier) confirmed working at startup — logs `val/lddt_ca`, `val/gdt_ts`, `val/gdt_ha`, `val/drmsd_ca` per epoch to `lightning_logs/.../metrics.csv` on the 121 held-out chains.
- Submitted via `sbatch _finetune_openfold.slurm` → **job 40** on `tornade` (2× RTX 6000 Ada). ETA ~16 h/epoch → **~48 h total**. Partition `normal` time limit = infinite (no wall-time kill).

### Accuracy-metric tooling (built + validated)
- `scripts/lddt_from_predictions.py` — global **lDDT-Cα** of predicted `*unrelaxed.pdb` vs experimental mmCIF, reusing OpenFold's `lddt_ca` + `mmcif_parsing.get_atom_coords` (position-aligned to seqres; length mismatches skipped + reported, no silent truncation). Validated on baseline: 13SI homomer chains share pLDDT 85.1 but resolve to distinct true lDDT 85.8–96.5.
- Baseline lDDT for the full set → `data/pdb_recent/baseline_lddt.csv` (the lDDT "before"): **3170 chains scored** (40 skipped: 8 empty preds, 32 pred/GT length mismatches — reported, not silently dropped), **mean lDDT-Cα 76.21, median 80.61** (std 16.7; <50: 9.4%, 50–70: 20.6%, 70–90: 49.2%, ≥90: 20.9%). **pLDDT↔lDDT correlation r=0.839** — pLDDT is a strong accuracy proxy here, so the confidence drop seen above likely tracks a real accuracy change.
- `_eval_checkpoint.sh <ckpt_dir> <tag>` — one command per checkpoint: inference over all 3210 chains (reusing `ft_infer_fastas/` + `ft_infer_aln/`, `--skip_relaxation`) → per-chain pLDDT → per-chain lDDT-Cα. Outputs under `data/pdb_recent/ckpt_eval_<tag>/`.

### Post-training plan (run when job 40 finishes)
For each of the 3 epoch checkpoints: `bash _eval_checkpoint.sh <epoch_ckpt> ep{1,2,3}`, then compare per-chain **pLDDT** (vs `plddt_compute.log`) and **lDDT-Cα** (vs `baseline_lddt.csv`) — both confidence and accuracy, across epochs, against the same-lineage baseline. Also read `val/lddt_ca` per epoch from `metrics.csv` for model selection.

