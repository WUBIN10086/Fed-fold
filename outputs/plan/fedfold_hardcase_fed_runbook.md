# FedFold / FedLoRA MVP Runbook

All new artifacts go to `outputs/fed_lora_hardcase_fed_v1/`.
Do not modify `outputs/fed_lora_fp32/`.

## Environment

```bash
conda activate fedfold
export PYTHONPATH=/home/wu/Desktop/Fed-fold
export PYTHON_BIN=/home/wu/miniconda3/envs/fedfold/bin/python
```

## Stage 0 — prepare + smoke

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh prepare
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh smoke
```

`prepare` builds:

- client-private baseline difficulty CSVs (`hard: TM < 0.5`)
- train / validation / development_test splits
- external final candidate labels
- novelty audit stub
- `server/rounds/round_000/global_model.pt`

## Stage 1A — single-seed FedLoRA

```bash
SEEDS=42 ARMS=uniform,hard_aware_70 ROUNDS=5 LOCAL_EPOCHS=1 \
  PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh train

PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh validate
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh select_global
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh report
```

Budget: 2 arms × 5 rounds × 5 clients = 50 local one-epoch jobs (~3.5–5 GPU h) plus validation inference.

## Stage 1B — multi-seed

```bash
SEEDS=42,43,44 ARMS=uniform,hard_aware_70 ROUNDS=5 \
  PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh train
```

## Method comparison (after metrics exist)

Methods:

- M0 baseline
- M1 local-only uniform
- M2 local-only hard-aware
- M3 FedLoRA uniform
- M4 FedLoRA hard-aware
- M5 optional private personalization after `global_best.pt`

```bash
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh lock_final
PYTHON_BIN=$PYTHON_BIN bash scripts/fed_lora_hardcase_fed.sh evaluate_final
```

## Pre-registered success criteria

- hard client-macro mean ΔTM ≥ +0.02
- hard median ΔTM > 0
- cluster-bootstrap 95% CI lower bound > 0
- large improve rate > large degrade rate
- non-hard client-macro mean ΔTM ≥ −0.005

## Privacy boundary

- Threat model default: `simulation`
- Server may read only validation summary JSON
- Client private paired CSVs / natives / difficulty remain under `clients/*/private/`
- Aggregation supports update clipping (`MAX_UPDATE_NORM`) and a secure-aggregation hook
