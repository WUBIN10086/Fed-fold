# SoloSeq finetuning with cluster-balanced sampling — usage guide

**Audience:** FedFold team members (and their AI agents) who want to finetune the
OpenFold **SoloSeq** single-sequence model on *their own* curated protein data.

**Script:** [`finetune_openfold_clustered.sh`](finetune_openfold_clustered.sh) — a
portable template. Every input path is a variable in the `CONFIG` block at the top;
set them for your data and run. Nothing below is specific to one machine.

> This is the generalized copy of the repo-root `_finetune_openfold_clustered.sh`
> (which is hard-wired to `data/pdb_recent/`). Use *this* one for new datasets.

---

## 1. What SoloSeq is, and what replaces the MSA

Standard AlphaFold-2 / OpenFold feeds the network a **multiple sequence alignment
(MSA)** for each target: you search big sequence databases (UniRef, BFD, MGnify…),
build an alignment of homologs, and the Evoformer reads the evolutionary signal
from that alignment. It is accurate but slow and database-heavy.

**SoloSeq removes the MSA entirely.** Instead, each single sequence is passed
through the **ESM-1b protein language model** (`esm1b_t33_650M_UR50S`, 650 M params,
33 transformer layers), and the model's **per-residue hidden states** become the
network input. So:

| Standard OpenFold | SoloSeq (this workflow) |
|-------------------|--------------------------|
| Per target: an MSA built from database search | Per target: **one ESM-1b embedding**, no search |
| Input lives in an "alignment directory" | Input lives in an **embedding directory** (same slot in the pipeline) |
| Needs UniRef/BFD/MGnify databases | Needs **no sequence database** at inference/training time |

**These ESM-1b embeddings ARE the "feature vectors that replace MSA data."** They
are the single most important input to get right — everything else is bookkeeping.

### 1.1 How the feature vectors are produced

Built by [`scripts/precompute_embeddings.py`](../../scripts/precompute_embeddings.py):

```bash
python scripts/precompute_embeddings.py <fasta_dir>/ <embedding_output_dir>/
# useful flags: --nogpu, --toks_per_batch 4096, --use_local_esm <path-to-esm-repo>
```

What it does, per chain:
1. Reads the chain's FASTA (one sequence).
2. Runs ESM-1b and extracts the **layer-33 representation** (`repr_layers=[33]`).
3. Drops the BOS/EOS special tokens so the embedding aligns 1:1 with residues.
4. Saves a tensor of shape **`[L, 1280]`** (L = sequence length, 1280 = ESM-1b
   embedding dim) as a torch dict.

**Output layout — memorize this, it's what training reads:**

```
<embedding_output_dir>/
  <label>/
    <label>.pt          # {"label": <label>, "representations": {33: Tensor[L, 1280]}}
```

- One **sub-directory per chain**, named exactly by the chain `<label>` (e.g. `1abc_A`).
- Inside it, a single `<label>.pt`. This directory is passed to `train_openfold.py`
  as the **alignment/embedding directory** (positional arg 2, and `--val_alignment_dir`).
- **Length limit:** ESM-1b handles ≤ 1022 residues; longer chains are truncated.
  That is why the FASTA filter uses `--max_len 1022` (see §2).

### 1.2 The base model weights

Download once (AWS S3, no credentials):

```bash
bash scripts/download_openfold_soloseq_params.sh <dest_dir>
# -> <dest_dir>/openfold_soloseq_params/seq_model_esm1b_ptm.pt   (the pTM variant we finetune)
```

Point `BASE_CKPT` at `seq_model_esm1b_ptm.pt`. It is loaded **weights-only**
(`--resume_model_weights_only True`) and never written back, so your finetune starts
from a clean, known lineage and the original file stays intact.

---

## 2. Full data-prep pipeline (do this for YOUR data first)

The finetuning script assumes these artifacts already exist. Produce them in order.
`<PDBID>` casing matters — see the **Critical: naming** box at the end.

1. **Collect structures** → `CIF_DIR/` (mmCIF `.cif` files for your training chains).
   In this project they come from recent PDB (`download_cif_from_pdb_bank.py`), but
   any mmCIF source works.

2. **Filter sequences + emit per-chain FASTA** —
   [`scripts/prep_solo_fasta_learnable.py`](../../scripts/prep_solo_fasta_learnable.py)
   (detailed doc: [`scripts/prep_solo_fasta_learnable.md`](../../scripts/prep_solo_fasta_learnable.md)):

   ```bash
   python scripts/prep_solo_fasta_learnable.py <CIF_DIR>/ <fasta_dir>/ \
     --target_labels_file <your_target_labels.csv> \
     --min_len 30 --max_len 1022 \
     --max_x_ratio 0.05 --max_noncanonical_ratio 0.05 \
     --max_resolution 4.0 --dedup_by_sequence
   ```

   Filters: keep only whitelisted labels (`--target_labels_file`; in this project =
   chains with baseline mean pLDDT < 80), length 30–1022 (1022 = ESM-1b limit),
   ≤ 5 % unknown `X`, ≤ 5 % non-canonical residues, resolution ≤ 4.0 Å, exact-sequence
   dedup. Writes `<fasta_dir>/_reports/kept_labels.txt` → use it as `TRAIN_FILTER`.
   **Your target-labels file and thresholds will differ — that's expected.**

3. **ESM-1b embeddings** (§1.1) → `EMB_DIR/`. Run over the kept FASTAs.

4. **Caches** (order matters):
   ```bash
   python scripts/generate_mmcif_cache.py      <CIF_DIR>/ <MMCIF_CACHE>   # release-dates
   python scripts/generate_chain_data_cache.py <CIF_DIR>/ <CHAIN_CACHE>   # chain data
   ```
   (The script re-generates `CHAIN_CACHE` *with* cluster sizes in step 3; you still
   need this plain one first, as it is the source of sequences for clustering.)

5. **Validation split** — hold out whole structures (never split a homomer across
   train/val). This repo uses `scripts/make_soloseq_val_split.py`, which emits
   `VAL_CIF_DIR/`, `VAL_EMB_DIR/`, and `TRAIN_FILTER`. Adapt paths for your data.

---

## 3. Why cluster-balanced sampling (step [1]–[3] of the script)

Exact-sequence dedup still lets **near-identical homologs** (point mutants, the same
protein re-crystallized, homomer chains) enter as separate chains. With uniform
sampling, a family of 50 near-duplicates gets pulled ~50× more than a unique protein,
so the model over-fits crowded families and under-trains rare folds.

OpenFold's data loader already fixes this **if** each chain carries a `cluster_size`:
it samples each chain with probability ∝ `1 / cluster_size`
([`openfold/data/data_modules.py:616-618`](../../openfold/data/data_modules.py#L616-L618)),
so every cluster contributes about equally. The script provides that by:

1. Merging all chains into one FASTA.
2. Clustering with **MMseqs2 `easy-cluster` at 40 % identity / 90 % coverage** — the
   same flags RCSB PDB uses for its official clusters. **This clusters your own
   sequences against each other; no MSA/reference database is needed**, only the
   `mmseqs` binary.
3. Rebuilding the chain-data cache with `--cluster_file`, stamping `cluster_size` on
   every chain, then pointing training at that clustered cache.

No change to any training code — just a richer cache.

> **Reference numbers from this project's run:** 2,673 chains → 930 clusters at 40 %
> identity; the largest family had 57 near-identical chains (previously sampled ~57×
> more than a singleton). Your numbers will differ.

---

## 4. Run it

```bash
conda activate <your_openfold_env>     # must expose train_openfold.py deps AND mmseqs

# simplest: defaults assume data/pdb_recent/ layout
bash finetune_openfold_clustered.sh

# for your own data, override the CONFIG variables inline:
BASE=/data/myproj \
CIF_DIR=/data/myproj/cifs EMB_DIR=/data/myproj/esm_embeddings \
TEMPLATE_MMCIF_DIR=/data/myproj/cifs \
MMCIF_CACHE=/data/myproj/mmcif_cache.json \
CHAIN_CACHE=/data/myproj/chain_cache.json \
TRAIN_FILTER=/data/myproj/fasta/_reports/kept_labels.txt \
VAL_CIF_DIR=/data/myproj/val_cifs VAL_EMB_DIR=/data/myproj/val_emb \
BASE_CKPT=/opt/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
MAX_TEMPLATE_DATE=2024-01-01 GPUS=2 SEQ_ID=0.4 \
bash finetune_openfold_clustered.sh
```

Behavior:
- Steps 1–3 are **CPU-only** and safe to run while a GPU job is going. They are
  **auto-skipped** if the clustered cache already exists (`FORCE_PREP=1` rebuilds).
- Step 4 (training) uses `GPUS` GPUs via DeepSpeed (ZeRO-2 + CPU offload, bf16).
  **Do not start it while another job is using those GPUs** — it will contend/OOM.
- Output: `<BASE>/soloseq_finetuning_outputs/run_<timestamp>_clustered/`
  (checkpoints per epoch; metrics in `lightning_logs/.../metrics.csv`).

### Key config variables

| Variable | Meaning |
|----------|---------|
| `BASE` | your data root; other paths default relative to it |
| `CIF_DIR` | training mmCIFs |
| `EMB_DIR` | **ESM-1b embeddings** (the MSA replacement) — one `<label>/<label>.pt` each |
| `TEMPLATE_MMCIF_DIR` | mmCIFs used for template search |
| `MMCIF_CACHE` / `CHAIN_CACHE` | template release-dates cache / chain-data cache (pre-clustering) |
| `TRAIN_FILTER` | include-list of train chains (val held out) |
| `VAL_CIF_DIR` / `VAL_EMB_DIR` | held-out validation structures / embeddings |
| `BASE_CKPT` | SoloSeq base weights (`seq_model_esm1b_ptm.pt`) |
| `MAX_TEMPLATE_DATE` | ignore templates released after this date |
| `SEQ_ID` | clustering identity threshold (0.4 = 40 %) |
| `MMSEQS_BIN`, `GPUS`, `MAX_EPOCHS`, `SEED` | mmseqs path, GPU count, epochs, seed |

---

## Critical: naming / casing consistency

> **The chain `<label>` must be spelled identically across the FASTA header, the
> embedding sub-directory `<label>/<label>.pt`, the mmCIF filename stem, and the
> cache keys.** If the embedding dir uses one casing and the CIFs/caches use another,
> the data loader silently matches **0 chains** and training does nothing useful.
> The convention in this project is **lowercase PDB IDs** (e.g. `1abc_A`).
>
> The clustering step is case-robust internally (it upper-cases both sides when
> mapping clusters → chains), so the `assert cluster_size>0` in the script is a good
> early tripwire: if it fires, your FASTA headers don't match your cache keys —
> fix the casing before going further.

## Requirements checklist

- [ ] Conda env with OpenFold installed (built for your GPU) **and** `mmseqs` on PATH
      (`conda install -c bioconda mmseqs2` — a small standalone binary, no databases).
- [ ] SoloSeq base weights downloaded (`download_openfold_soloseq_params.sh`).
- [ ] ESM-1b reachable by `precompute_embeddings.py` (torch.hub download, or
      `--use_local_esm <path>` for offline nodes).
- [ ] Data-prep artifacts from §2 present and consistently named.
- [ ] GPUs free for the training step.

See also: [`../finetuning_configs.md`](../finetuning_configs.md) for the full
hyper-parameter reference and the grouping/clustering rationale.
