# prep_solo_fasta_learnable.py 使用记录

`scripts/prep_solo_fasta_learnable.py` 用于从一批 mmCIF 文件中生成可用于 SOLO/ESM 单序列训练流程的 FASTA 文件。脚本会逐个读取输入目录中的 `.cif` 文件，解析其中的链序列，并按链生成 `PDBID_chain.fasta` 格式的单序列 FASTA。

## 主要功能

脚本会对每条链做基础质量过滤：

- 只保留 `--target_labels_file` 中指定的样本标签，例如 `1abc_A`。
- 过滤长度小于 `--min_len` 或大于 `--max_len` 的序列。
- 过滤 `X` 残基比例超过 `--max_x_ratio` 的序列。
- 过滤非标准氨基酸比例超过 `--max_noncanonical_ratio` 的序列。
- 过滤分辨率差于 `--max_resolution` 的结构。
- 如果加上 `--dedup_by_sequence`，会按序列去重，只保留第一次出现的重复序列。

输出目录中除 FASTA 文件外，还会生成 `_reports/` 报告目录，包含：

- `kept_samples.csv`：最终保留下来的样本及其统计信息。
- `dropped_samples.csv`：被过滤掉的样本及过滤原因。
- `kept_labels.txt`：最终保留下来的样本 label，每行一个。
- `dropped_labels.txt`：被过滤掉的样本 label 及原因。
- `summary.txt`：整体统计摘要。

## 运行 FASTA 生成

下面是按照当前训练使用的参数整理后的通用写法，路径改为一般目录形式：

```bash
python3 scripts/prep_solo_fasta_learnable.py \
  <mmcif_dir>/ \
  <output_fasta_dir>/ \
  --target_labels_file <target_labels_csv> \
  --min_len 30 \
  --max_len 1022 \
  --max_x_ratio 0.05 \
  --max_noncanonical_ratio 0.05 \
  --max_resolution 4.0 \
  --dedup_by_sequence
```

参数说明：

- `<mmcif_dir>/`：输入 mmCIF 文件目录。
- `<output_fasta_dir>/`：输出 FASTA 文件目录。
- `<target_labels_csv>`：pLDDT 跑分小于 80 的样本列表 csv 文件。脚本会优先读取 `protein_name`、`label` 或 `chain_id` 字段作为 label，只保留这些低 pLDDT 样本中通过后续质量过滤的链。
- `--min_len 30`：保留长度至少为 30 的链。
- `--max_len 1022`：保留长度不超过 1022 的链，适配 ESM-1b 的长度限制。
- `--max_x_ratio 0.05`：允许最多 5% 的 `X` 未知残基。
- `--max_noncanonical_ratio 0.05`：允许最多 5% 的非标准氨基酸。
- `--max_resolution 4.0`：只保留分辨率不差于 4.0 Angstrom 的结构。
- `--dedup_by_sequence`：按序列去重。

## 预计算 embedding

FASTA 生成后，需要对保留下来的 FASTA 计算单序列 embedding：

```bash
python3 scripts/precompute_embeddings.py \
  <output_fasta_dir>/ \
  <output_embedding_dir>/
```

这里的 `<output_fasta_dir>/` 应该和上一步的 FASTA 输出目录一致，`<output_embedding_dir>/` 是后续训练传给 `train_openfold.py` 的 alignment/embedding 目录。

## 训练命令

训练时使用单序列 embedding，并从 SOLO sequence model 的参数继续 finetune：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python3 train_openfold.py \
  <train_mmcif_dir>/ \
  <output_embedding_dir>/ \
  <template_mmcif_dir>/ \
  <output_dir>/ \
  <max_template_date> \
  --train_filter_path <output_fasta_dir>/_reports/kept_labels.txt \
  --use_single_seq_mode True \
  --config_preset seq_model_esm1b_ptm \
  --experiment_config_json seq_model_esm1b_ptm_finetune_override.json \
  --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
  --resume_model_weights_only True \
  --template_release_dates_cache_path <template_release_dates_cache.json> \
  --train_chain_data_cache_path <train_chain_data_cache.json> \
  --precision bf16-mixed \
  --gpus 1 \
  --deepspeed_config_path deepspeed_config.json
```

关键参数说明：

- `<train_mmcif_dir>/`：训练使用的 mmCIF 目录。
- `<output_embedding_dir>/`：由 `scripts/precompute_embeddings.py` 生成的单序列 embedding 目录。
- `<template_mmcif_dir>/`：模板搜索使用的 mmCIF 目录。
- `<output_dir>/`：训练 checkpoint、日志等输出目录。
- `<max_template_date>`：模板日期截断，例如 `2026-04-22`。
- `--use_single_seq_mode True`：使用单序列 embedding，而不是传统 MSA。
- `--config_preset seq_model_esm1b_ptm`：使用 ESM-1b 单序列 PTM 配置。
- `--experiment_config_json seq_model_esm1b_ptm_finetune_override.json`：加载 finetune 覆盖配置。
- `--resume_from_ckpt .../seq_model_esm1b_ptm.pt`：从 SOLO sequence model 权重初始化。
- `--resume_model_weights_only True`：只加载模型权重，不恢复优化器等训练状态。
- `--precision bf16-mixed`：使用 bf16 mixed precision。
- `--deepspeed_config_path deepspeed_config.json`：使用 DeepSpeed 配置。

## kept_labels.txt 的作用

`kept_labels.txt` 是 `prep_solo_fasta_learnable.py` 在 `_reports/` 目录下生成的最终样本白名单。它记录了所有通过目标 label、长度、未知残基比例、非标准氨基酸比例、分辨率和去重过滤后的样本 label。

训练命令中加入：

```bash
--train_filter_path <output_fasta_dir>/_reports/kept_labels.txt
```

目的是让 `train_openfold.py` 的训练集只包含这些通过 FASTA 生成阶段过滤、并且已经预计算 embedding 的样本。这样可以避免训练阶段读到未生成 FASTA/embedding 的链，也能保证训练样本和 embedding 目录中的样本范围一致。

如果不加这个参数，训练集可能会从 mmCIF 目录或 cache 中看到更多样本，其中一部分没有通过 `prep_solo_fasta_learnable.py` 的过滤，或者没有对应的单序列 embedding，容易导致训练数据和 embedding 数据不匹配。
