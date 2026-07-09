# Computing TM-score for SoloSeq predictions (`tm_from_predictions.py`)

A small, self-contained script that measures the **accuracy** of predicted
structures against experimental ground truth, using **TM-score**. Use it after
running inference to see how good the predictions actually are (not just how
confident the model is).

> Canonical copy lives at `scripts/tm_from_predictions.py` in the repo root; a
> copy sits next to this doc for convenience. Prefer the repo copy so it stays
> in sync with `openfold/`.

---

## What it does

For every predicted `*unrelaxed.pdb` in a directory, it:

1. Reads the predicted Cα coordinates.
2. Loads the matching **experimental structure** from an mmCIF file.
3. Superimposes prediction onto ground truth and computes global **TM-score**:

   ```
   TM = (1/L) · Σ_i 1/(1 + (d_i/d0)²),   d0 = 1.24·(L−15)^(1/3) − 1.8
   ```
   where `L` = number of valid Cα residues in the ground truth and `d_i` is the
   per-residue Cα distance after superposition.

Output is a per-chain CSV plus a printed summary (mean/median/min/max, skip
reasons).

### Two things this is NOT
- **Not pTM / `val/tm`.** The `val/tm` you see in training `metrics.csv` is the
  model's *self-estimated* pTM (a confidence, from the TM head). This script
  computes the *true* TM-score against the deposited structure.
- **Not pLDDT.** pLDDT is confidence; TM-score (and lDDT) is accuracy. On SoloSeq
  finetuning these can diverge — judge model quality by TM/lDDT, not pLDDT.

### Superposition caveat
It uses OpenFold's `superimpose()` (SVD/Kabsch, **RMSD-minimizing**) — the same
superposition the repo uses for GDT — **not** TMalign's TM-optimal rotation.
Absolute TM values are therefore a slight underestimate of a TMalign score. This
is fine for *relative* comparisons (baseline vs finetuned, epoch vs epoch) because
the same method is applied to everything. If you need publication-grade absolute
TM, run TMalign instead.

---

## Requirements

- Run inside the project's conda env (`openfold_env`) — it needs `torch`,
  `numpy`, `Biopython` (for `Bio.SVDSuperimposer`), and the in-repo `openfold`
  package.
- **Run from the repo root** (`Fed-fold/`). The script does
  `sys.path.insert(0, os.getcwd())` to import `openfold`, so the current
  directory must contain `openfold/`.
- **No GPU needed** — it's pure CPU. Prefix with `CUDA_VISIBLE_DEVICES=""` so it
  doesn't grab a GPU another job may be using.

---

## Usage

```bash
cd /path/to/Fed-fold
conda activate openfold_env

CUDA_VISIBLE_DEVICES="" python scripts/tm_from_predictions.py \
    <predictions_dir> <mmcif_dir> -o <out.csv>
```

Example (evaluating one checkpoint's predictions):

```bash
CUDA_VISIBLE_DEVICES="" python scripts/tm_from_predictions.py \
    data/pdb_recent/ckpt_eval_ep3/predictions \
    data/pdb_recent/mmcif_files \
    -o data/pdb_recent/ckpt_eval_ep3/tm_per_chain.csv
```

Arguments:
| arg | meaning |
|---|---|
| `predictions_dir` | dir containing predicted `*unrelaxed.pdb` files |
| `mmcif_dir` | dir of experimental ground-truth `<pdb_id>.cif` files |
| `-o / --out_csv` | output path (required) |

Output CSV: `chain,tm_score` — one row per successfully scored chain, e.g.
```
chain,tm_score
13SI_A,0.6124
13SI_B,0.7815
```
Console summary also reports `scored N chains, skipped M` and a breakdown of
skip reasons.

---

## ⚠️ Naming / matching conventions — READ THIS if your data differs

The script does **no sequence alignment and no fuzzy name matching**. It relies
entirely on filename conventions to pair a prediction with its ground truth.
If your training/inference data uses different names, it will silently skip
everything (or mis-pair). The three rules:

### 1. Prediction filename → `(pdb_id, chain_id)`
The tag is extracted by **splitting the filename on the literal string
`_seq_model`**, then splitting the remainder on the last `_`:

```
13SI_A_seq_model_esm1b_ptm_unrelaxed.pdb
└──┬─┘ └──────── stripped ───────────┘
 tag = "13SI_A"  →  pdb_id="13si" (lowercased), chain_id="A"
```

- Your prediction files **must** be named `<PDBID>_<CHAIN>_seq_model…_unrelaxed.pdb`.
  This is exactly what `run_pretrained_openfold.py` emits with
  `--config_preset seq_model_esm1b_ptm`.
- **If you use a different `config_preset`** (so the filename infix is not
  `_seq_model…`), `tag_of()` won't strip correctly and every chain will fail to
  match. Fix: change the split token in `tag_of()` (line ~34,
  `.split("_seq_model")`) to whatever your filenames use, or rename your outputs.

### 2. mmCIF ground-truth files must be lowercase-PDB-id
The cif is looked up as `<mmcif_dir>/<pdb_id>.cif` with `pdb_id` **lowercased**
(e.g. tag `7IAU_A` → `7iau.cif`). Your `mmcif_dir` must contain lowercase-named
cif files. (This matches the wider FedFold convention: embeddings/caches/cifs
all share lowercase-PDB names — see `docs/encountered_bugs/wang.md`.)

### 3. Chain id must exist in the cif
`chain_id` (e.g. `A`) is passed to `mmcif_parsing.get_atom_coords`, which expects
the mmCIF **author chain id**. If your chain letters don't match the cif's, that
chain is skipped (`chain <id>: <Error>`).

---

## Alignment assumption (may break on other data)

The script assumes the prediction and the ground truth are **position-aligned and
equal length**. This holds for SoloSeq because it predicts the full input
**seqres**, and `get_atom_coords` returns GT aligned to that same seqres. So
residue *i* of the prediction corresponds to residue *i* of the GT.

If your pipeline violates this — e.g. you predict a cropped/truncated sequence, a
different construct than the deposited seqres, or your cif residue numbering
differs — you will get `len mismatch pred=X gt=Y` skips (and, worse, silently
wrong scores if lengths happen to coincide but residues don't correspond). The
script deliberately **skips length mismatches rather than truncating**, so a high
skip count is your signal that the alignment assumption is being violated for
your data.

---

## Interpreting skips

The summary prints skip reasons, e.g. `{'empty prediction': 8, 'len mismatch': 32}`:

| reason | meaning | typical fix |
|---|---|---|
| `empty prediction` | the `*unrelaxed.pdb` has no atoms (failed inference) | re-run inference for that chain |
| `pred parse …` | the PDB couldn't be parsed | check the prediction file |
| `no cif <id>` | no `<pdb_id>.cif` in `mmcif_dir` | check cif naming/casing (rule 2) |
| `unparsable cif <id>` | mmCIF failed to parse | check the cif file |
| `chain <id>: …` | chain absent/ambiguous in the cif | check chain id (rule 3) |
| `len mismatch pred=X gt=Y` | prediction and GT lengths differ | alignment assumption violated (see above) |
| `no valid CA` | GT has zero resolved Cα for that chain | nothing scoreable |

A few skips are normal; **lots of `no cif` or `len mismatch` means a
naming/alignment mismatch with your data — fix that before trusting the numbers.**

---

## Companion tools

- `scripts/lddt_from_predictions.py` — same interface, computes **lDDT-Cα**
  (superposition-free accuracy). Good to run alongside TM.
- `scripts/compare_metrics_epochs.py` — paired per-chain comparison of
  pLDDT / lDDT / TM across checkpoints vs a baseline; consumes the
  `tm_per_chain.csv` / `lddt_per_chain.csv` this and the lDDT script produce.

## Quick recipe: evaluate a checkpoint end-to-end

```bash
cd /path/to/Fed-fold && conda activate openfold_env
PRED=path/to/predictions          # your *unrelaxed.pdb dir
CIF=path/to/mmcif_files           # your lowercase-named .cif dir

CUDA_VISIBLE_DEVICES="" python scripts/tm_from_predictions.py   $PRED $CIF -o tm_per_chain.csv
CUDA_VISIBLE_DEVICES="" python scripts/lddt_from_predictions.py $PRED $CIF -o lddt_per_chain.csv
```
Read the printed `mean TM` / `mean lDDT-Ca` and the per-chain CSVs.
