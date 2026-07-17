# 联邦学习 SoloSeq 微调统合流程（数据准备 → pLDDT 筛选 → train/test 划分 → 逐 client 微调 → FedAvg 聚合 → TM-score 评测矩阵）

> **本版本说明**：本文档是把「本项目的联邦方案」与「团队集中式方案」取长补短后的**统合版**，并新增了**独立测试集 + 评测矩阵**，以核心目标——**联邦学习（5 个 client 各自本地微调，再聚合成全局模型）**——为骨架。
>
> - **保留（本项目）**：按聚类分 5 个 client、pLDDT<80 难样本筛选、官方 `TMscore` 标准绝对值评测、`extract_native_chain_pdbsV2.py` 的 SEQRES 对齐、**FedAvg 全局聚合**。
> - **吸收（集中式方案）**：`cluster_size` 均衡采样、**独立验证/测试集**、**命名/大小写一致性**纪律、**pTM vs 真实 TM-score vs pLDDT** 概念澄清、epoch/早停停止策略。
> - **新增（本次）**：按 cluster 划 train/test，产出**评测矩阵**——每个 client 微调前/后、以及 FedAvg 全局模型，在「各自 client 测试集」与「全部 client 并集测试集」上的 TM-score，一张表做对比。
>
> 路径可按需修改，下文默认以 `client_1` 为例，其余 client 把 `client_1` 换成对应编号。

> **执行环境**：所有命令建议在**项目根目录**下、有 GPU 的远程服务器上执行。`CUDA_VISIBLE_DEVICES=2` 表示用 2 号卡，按你机器实际情况改。

---

## 0. 前置准备

| 依赖                                       | 用途                  | 检查方式                                                                                                         |
| ------------------------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Python 环境（OpenFold 依赖）               | 运行所有脚本          | `python -c "import openfold"`                                                                                  |
| `mmseqs` 可执行文件                      | 序列聚类              | `which mmseqs` 有输出                                                                                          |
| ESM-1b 权重（可联网自动下载）              | 生成 embedding        | 首次运行`torch.hub` 下载；离线见下方说明                                                                       |
| SoloSeq 初始权重`seq_model_esm1b_ptm.pt` | 预筛选预测 + 微调起点 | 位于`openfold/resources/openfold_soloseq_params/`，缺失用 `scripts/download_openfold_soloseq_params.sh` 下载 |
| `TMscore` 可执行文件                     | 算 TM-score           | 项目已带编译好的`tmscore/TMscore`                                                                              |

- 离线无法下载 ESM-1b：给 `precompute_embeddings.py` 传 `--use_local_esm <本地esm仓库路径>`。

### 命名一致性纪律（务必遵守）

> **同一条链的 `<label>`（如 `9vak_A`）必须在这些地方拼写完全一致**：FASTA 头、embedding 子目录 `<label>/<label>.pt`、mmCIF 文件名 `<pdb_id>.cif`、各类缓存 json 的 key。**本项目统一用小写 PDB ID**（如 `9vak_A`）。
>
> 若大小写不一致，数据加载器会**静默匹配 0 条链**、训练看似在跑实则什么都没学。步骤 12 生成的 `cluster_size` 是很好的"早期报警"：几乎没有链拿到 `cluster_size>0` 多半就是命名对不上（聚类文件是大写、cif/缓存是小写属正常，脚本内部已做大小写归一，不用担心）。

### 流程总览

```
下载 cif ─▶ 抽序列(all.fasta) ─▶ MMseqs2 聚类(30%) ─▶ 按 cluster 分 5 个 client
                                                            │
                    (阶段三~五：每个 client 各自做，无训练)   ▼
   全量 fasta ─▶ ESM embedding ─▶ 全量预测(原始权重) ─▶ 选 pLDDT<80 子集 ─▶ 子集缓存
                                                            │
                          (所有 client 都完成后，做一次)     ▼
             阶段六：按 cluster 把子集切 train / test（同一 cluster 不跨两边）
                                                            │
                    (阶段七：每个 client 只用 train 微调)      ▼
                                                    每个 client 一个 checkpoint
                                                            │
                                     阶段八：FedAvg 聚合     ▼
                                                    全局模型 global_model.pt
                                                            │
                    阶段九：在"并集测试集"上给每个模型各预测一次 ▼
   before(基线) / 各 client after / global  ──▶ 官方 TMscore ──▶ 评测矩阵
        (行=模型，列=各client测试集 + 全部并集测试集)
```

---

## 阶段一：下载数据

### 步骤 1 — 从 RCSB 下载 mmCIF

```bash
python download_cif_from_pdb_bankV2.py
```

- **作用**：按脚本内检索条件（近一年、X-ray、高分辨率、单蛋白单链、长度 30–300）从 RCSB 下载 `.cif`。
- **产出**：`data/all_pdb_1y/all_mmcif_files/*.cif`、`data/all_pdb_1y/entry_ids.txt`。

---

## 阶段二：序列抽取 → 聚类去冗余 → 分 client

### 步骤 2 — 从 cif 抽取序列，汇总成一个 FASTA

```bash
python scripts/data_dir_to_fasta.py \
    data/all_pdb_1y/all_mmcif_files/ \
    data/all_pdb_1y/all_fasta_files/all.fasta
```

- **产出**：`data/all_pdb_1y/all_fasta_files/all.fasta`（记录名 `>{PDBID}_{CHAIN}`）。

### 步骤 3 — 用 MMseqs2 按 30% identity 聚类去冗余

```bash
python scripts/fasta_to_clusterfile.py \
    data/all_pdb_1y/all_fasta_files/all.fasta \
    data/all_pdb_1y/clusters_30.txt \
    $(which mmseqs) \
    --seq-id 0.3
```

- **产出**：`data/all_pdb_1y/clusters_30.txt`，每行一个 cluster。检查：`wc -l`。
- **注意**：PowerShell 里把 `$(which mmseqs)` 换成绝对路径，注意要有mmseqs依赖，没有的话装一下，conda之类的。
- **阈值说明**：本项目用 **30%**（聚类一物两用：分 client + 训练内均衡采样 + train/test 划分）。要对齐 RCSB 官方口径可改 `--seq-id 0.4`，但**整条流程必须共用同一个聚类文件**。

### 步骤 4 — 以 cluster 为单位均分给 5 个 client

```bash
python scripts/split_clusters_to_clients.py \
    data/all_pdb_1y/clusters_30.txt \
    data/all_pdb_1y/all_mmcif_files/ \
    data/all_pdb_1y/fed_split \
    --num_clients 5
```

- **作用**：以 cluster 为最小单位（同一 cluster 不跨 client，防泄漏），贪心均衡各 client 样本量。
- **产出**：`data/all_pdb_1y/fed_split/client_0..4/mmcif_files/`、`pdb_ids.txt`、`split_manifest.json`。
- **检查**：`split_manifest.json` 里 `num_pdbs` 是否均衡、`cif_missing` 是否为空。磁盘紧张加 `--link`。

> 阶段三～五是**每个 client 各跑一遍**（示例 `client_1`）。**5 个 client 都完成阶段三～五后**，再统一做阶段六的 train/test 划分。

---

## 阶段三：单 client 全量数据预处理

### 步骤 5 — cif → 单序列 FASTA（全量，带质量过滤）

```bash
python3 scripts/prep_solo_fasta_learnable.py \
    data/all_pdb_1y/fed_split/client_1/mmcif_files/ \
    data/all_pdb_1y/fed_split/client_1/solo_fasta_dir \
    --max_len 1022 \
    --dedup_by_sequence
```

- **产出**：`.../client_1/solo_fasta_dir/*.fasta` 及 `_reports/`。

### 步骤 6 — 生成 ESM-1b embedding（全量，SoloSeq 的 "alignment"）

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/precompute_embeddings.py \
    data/all_pdb_1y/fed_split/client_1/solo_fasta_dir/ \
    data/all_pdb_1y/fed_split/client_1/solo_alignment_dir
```

- **产出**：`.../client_1/solo_alignment_dir/{label}/{label}.pt`（后续训练/预测全程复用）。离线加 `--use_local_esm <路径>`。

---

## 阶段四：pLDDT 预筛选，构建微调子集

### 步骤 7 — 全量预测（原始权重，仅用于筛选）

```bash
python3 run_pretrained_openfold.py \
    data/all_pdb_1y/fed_split/client_1/solo_fasta_dir/ \
    data/all_pdb_1y/fed_split/client_1/mmcif_files/ \
    --use_precomputed_alignments data/all_pdb_1y/fed_split/client_1/solo_alignment_dir/ \
    --output_dir data/all_pdb_1y/fed_split/client_1/pred_prescreen \
    --model_device "cuda:0" \
    --skip_relaxation \
    --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
```

- **产出**：`.../client_1/pred_prescreen/*.pdb`（pLDDT 写在 B-factor 列）。

### 步骤 8 — 计算 pLDDT，选出 mean pLDDT < 80 的样本

```bash
python3 scripts/plddt_from_pdb.py \
    data/all_pdb_1y/fed_split/client_1/pred_prescreen/ \
    --selected-csv docs/selectedPDB/client_1_lowplddt.csv \
    --bad-log docs/selectedPDB/client_1_bad_pdb.txt \
    --no-extremes
```

- **产出**：`docs/selectedPDB/client_1_lowplddt.csv`（`protein_name,mean_plddt`）。阈值可加 `--select-threshold <值>`。

### 步骤 9 — 抽取低 pLDDT 子集的 cif

```bash
python3 scripts/extract_selected_mmcif.py \
    --csv-path docs/selectedPDB/client_1_lowplddt.csv \
    --source-dir data/all_pdb_1y/fed_split/client_1/mmcif_files \
    --dst-dir data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune
```

- **产出**：`.../client_1/mmcif_files_finetune/*.cif`（微调数据来源子集）。

---

## 阶段五：微调子集预处理

### 步骤 10 — 子集 cif → 单序列 FASTA（得到候选清单 kept_labels.txt）

```bash
python3 scripts/prep_solo_fasta_learnable.py \
    data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune/ \
    data/all_pdb_1y/fed_split/client_1/solo_fasta_dir_finetune \
    --max_len 1022 \
    --dedup_by_sequence
```

- **产出**：`.../client_1/solo_fasta_dir_finetune/_reports/kept_labels.txt`（这就是阶段六 train/test 划分的**候选池**）。

### 步骤 11 — 生成 mmCIF 结构信息缓存（子集）

```bash
python3 scripts/generate_mmcif_cache.py \
    data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune/ \
    data/all_pdb_1y/fed_split/client_1/solo_mmcif_cache_finetune.json \
    --no_workers 8
```

### 步骤 12 — 生成链数据缓存（子集，带 cluster_size 均衡采样）

```bash
python3 scripts/generate_chain_data_cache.py \
    data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune/ \
    data/all_pdb_1y/fed_split/client_1/solo_chain_data_cache_finetune.json \
    --cluster_file data/all_pdb_1y/clusters_30.txt \
    --no_workers 8
```

- **为什么要 `--cluster_file`**：序列去重后仍有近乎相同的同源链（点突变、同蛋白多次结晶、同源多聚体多链）。均匀采样会让大家族被过度采样、稀有折叠欠训。带 `cluster_size` 后，加载器按 `1/cluster_size` 采样（见 `openfold/data/data_modules.py`），各 cluster 贡献大致相等——不改训练代码，只让缓存更"聪明"。
- **缓存可为超集**：即使某些链后面被划到 test，这两个缓存仍可包含它们；训练实际用哪些链由阶段七的 `--train_filter_path` 决定，所以**划分 test 后无需重建缓存**。

---

## 阶段六：按 cluster 划分 train / test（所有 client 完成阶段三~五后，做一次）

> 这是"加测试集"的关键一步。测试集必须**从训练里剔除**，且**同一 cluster 不能同时出现在 train 和 test**，否则近重复同源链会造成泄漏、TM 虚高。

### 步骤 13 — 划分 train/test 并组装"并集测试集"

```bash
python3 scripts/build_fed_test_set.py \
    --fed-split-dir data/all_pdb_1y/fed_split \
    --num-clients 5 \
    --cluster-file data/all_pdb_1y/clusters_30.txt \
    --out-dir data/all_pdb_1y/fed_test \
    --test-frac 0.2 --seed 42
```

- **作用**：对每个 client，把其低 pLDDT 候选池（步骤 10 的 `kept_labels.txt`）按整簇切成 train/test；再把所有 client 的 test 组装成一个**并集测试集**（后续每个模型只在并集上预测一次，即可切出「各自」与「全部」两种口径）。
- **产出**：
  - 每个 client：`data/all_pdb_1y/fed_test/client_i/train_labels.txt`、`test_labels.txt`
  - 并集：`data/all_pdb_1y/fed_test/all/`，内含 `solo_fasta_dir/`、`solo_alignment_dir/`、`mmcif_files/`、`test_labels.txt`、`label_client_map.csv`
- **参数**：`--test-frac` 测试比例（默认 0.2）；embedding/cif 想省磁盘加 `--link` 用软链接。
- **检查**：终端会打印每个 client 的 `候选/train/test` 数量以及并集链数；如有大量 `缺失项` 警告，多半是命名不一致（见开头纪律）。

---

## 阶段七：本地微调（每个 client 各跑一次，只用 train）

### 步骤 14 — 在 train 子集上做 SoloSeq 微调

```bash
CUDA_VISIBLE_DEVICES=2 python3 train_openfold.py \
    data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune/ \
    data/all_pdb_1y/fed_split/client_1/solo_alignment_dir/ \
    data/all_pdb_1y/fed_split/client_1/mmcif_files_finetune/ \
    data/all_pdb_1y/fed_split/client_1/output_dir \
    2026-01-01 \
    --train_filter_path data/all_pdb_1y/fed_test/client_1/train_labels.txt \
    --use_single_seq_mode True \
    --config_preset seq_model_esm1b_ptm \
    --experiment_config_json seq_model_esm1b_ptm_finetune_override.json \
    --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
    --resume_model_weights_only True \
    --template_release_dates_cache_path data/all_pdb_1y/fed_split/client_1/solo_mmcif_cache_finetune.json \
    --train_chain_data_cache_path data/all_pdb_1y/fed_split/client_1/solo_chain_data_cache_finetune.json \
    --precision bf16-mixed \
    --gpus 1 \
    --deepspeed_config_path deepspeed_config.json
```

> **⚠️ 与旧版的唯一区别**：`--train_filter_path` 从"整个低 pLDDT 子集 kept_labels"改成了**阶段六划出的 `train_labels.txt`**（把 test 排除在训练外）。如果你之前已经用整个子集训过，为了得到干净的测试指标，请用新的 `train_labels.txt` **重训**一遍。

**位置参数（前 5 个，顺序不能乱）**：`train_data_dir`=子集 cif、`train_alignment_dir`=全量 embedding、`template_mmcif_dir`=子集 cif、`output_dir`、`max_template_date`（≥数据发布日期）。

**产出**：`.../client_1/output_dir/checkpoints/` 下的 checkpoint（DeepSpeed 是**目录**，形如 `epoch=0-step=10000.ckpt/`）。训练完 `ls` 确认名字。

> **停止策略（二选一）**：默认按 `--train_epoch_len`（默认 10000，约 10000 step）。想用 epoch/早停加 `--max_epochs 3 --checkpoint_every_epoch`；要早停再加 `--early_stopping True`（它监控 `val/lddt_ca`，须同时配置下方验证集）。
> ⚠️ 集中式方案的 `--val_check_interval` 在本仓库**不是有效参数**，请勿照抄。

### （可选）步骤 14b — 训练中用验证集监控真实精度

本仓库支持 `--val_data_dir / --val_alignment_dir / --val_mmcif_data_cache_path / --num_sanity_val_steps`。你可以从 `train_labels.txt` 里再切一小部分做验证（同样按整簇），用 `extract_selected_mmcif.py` + `generate_mmcif_cache.py` 搭出 `val_mmcif_files/` 与 `val_mmcif_cache.json`（embedding 复用全量 `solo_alignment_dir`），然后在步骤 14 追加：

```
--val_data_dir .../client_1/val_mmcif_files/ \
--val_alignment_dir .../client_1/solo_alignment_dir/ \
--val_mmcif_data_cache_path .../client_1/val_mmcif_cache.json \
--num_sanity_val_steps 2
```

> **概念澄清**：日志里的 `val/tm` 是模型自估的 **pTM（置信度）**，pLDDT 也是置信度；判断好坏要看**真实 TM-score / lDDT**（对着实验结构算，见阶段九）。

---

## 阶段八：FedAvg 全局聚合

### 步骤 15 — 聚合 5 个 client 的权重

```bash
python3 scripts/fedavg_aggregate.py \
    --checkpoints \
        data/all_pdb_1y/fed_split/client_0/output_dir/checkpoints/epoch=0-step=10000.ckpt \
        data/all_pdb_1y/fed_split/client_1/output_dir/checkpoints/epoch=0-step=10000.ckpt \
        data/all_pdb_1y/fed_split/client_2/output_dir/checkpoints/epoch=0-step=10000.ckpt \
        data/all_pdb_1y/fed_split/client_3/output_dir/checkpoints/epoch=0-step=10000.ckpt \
        data/all_pdb_1y/fed_split/client_4/output_dir/checkpoints/epoch=0-step=10000.ckpt \
    --output data/all_pdb_1y/fed_global/round1/global_model.pt
```

- **作用**：提取每个 client 的 **EMA 参数**（与推理实际加载一致）做加权平均，输出 AlphaFold 层级纯参数 `.pt`（与官方 base 权重同构，既能当训练初始权重、也能直接推理）。
- **加权**：默认等权；标准 FedAvg 用样本数：加 `--weights n0 n1 n2 n3 n4`（`n_k` = 各 client `train_labels.txt` 行数）。
- **只聚合健康的 client**：先看阶段九各 client 的 after-TM 是否相对 before 没有崩/回退，把崩掉的 client 从 `--checkpoints` 里剔除再聚合。
- **多轮**：把 `global_model.pt` 作为下一轮各 client 步骤 14 的 `--resume_from_ckpt`，循环"本地训练→聚合"。

---

## 阶段九：测试集评测矩阵（正式对比）

思路：**每个模型都只在"并集测试集"上预测一次**，再按 `label_client_map.csv` 切片，即可同时得到「各自 client 测试集」和「全部并集测试集」两种口径。需要预测的模型共 **7 个**：`before`（原始权重，作为所有 client 的微调前基线）、`client_0..4_after`（各 client 自己的 checkpoint）、`global`（FedAvg 全局模型）。

> 测试集输入统一用阶段六产出的 `fed_test/all/`：fasta=`solo_fasta_dir/`、embedding=`solo_alignment_dir/`、模板 cif=`mmcif_files/`。

### 步骤 16 — before（原始权重）在并集测试集上预测

```bash
python3 run_pretrained_openfold.py \
    data/all_pdb_1y/fed_test/all/solo_fasta_dir/ \
    data/all_pdb_1y/fed_test/all/mmcif_files/ \
    --use_precomputed_alignments data/all_pdb_1y/fed_test/all/solo_alignment_dir/ \
    --output_dir data/all_pdb_1y/fed_test/pred_before \
    --model_device "cuda:0" --skip_relaxation --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
```

### 步骤 17 — 每个 client 的 after 模型在并集测试集上预测（循环 5 次）

```bash
# 以 client_1 为例；client_0/2/3/4 同理，换 checkpoint 路径与 output_dir
python3 run_pretrained_openfold.py \
    data/all_pdb_1y/fed_test/all/solo_fasta_dir/ \
    data/all_pdb_1y/fed_test/all/mmcif_files/ \
    --use_precomputed_alignments data/all_pdb_1y/fed_test/all/solo_alignment_dir/ \
    --output_dir data/all_pdb_1y/fed_test/pred_client_1_after \
    --model_device "cuda:0" --skip_relaxation --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path data/all_pdb_1y/fed_split/client_1/output_dir/checkpoints/epoch=0-step=10000.ckpt
```

### 步骤 18 — global（FedAvg）在并集测试集上预测

```bash
python3 run_pretrained_openfold.py \
    data/all_pdb_1y/fed_test/all/solo_fasta_dir/ \
    data/all_pdb_1y/fed_test/all/mmcif_files/ \
    --use_precomputed_alignments data/all_pdb_1y/fed_test/all/solo_alignment_dir/ \
    --output_dir data/all_pdb_1y/fed_test/pred_global \
    --model_device "cuda:0" --skip_relaxation --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path data/all_pdb_1y/fed_global/round1/global_model.pt
```

### 步骤 19 — 抽取并集测试集的 native（一次，所有模型共用）

```bash
python3 scripts/extract_native_chain_pdbsV2.py \
    --cif-dir data/all_pdb_1y/fed_test/all/mmcif_files/ \
    --out-dir data/all_pdb_1y/fed_test/all/native \
    --labels-txt data/all_pdb_1y/fed_test/all/test_labels.txt
```

- **作用**：坐标按 SEQRES 对齐、残基编号 `1..N`，与 SoloSeq 全长预测逐位对应。

### 步骤 20 — 对每个模型算 per-chain TM-score（官方 TMscore）

```bash
# 对 7 个预测目录各跑一次；下面给 before / client_1_after / global 三个示例
python3 scripts/tmscore_from_pdb.py data/all_pdb_1y/fed_test/pred_before/ \
    --native-dir data/all_pdb_1y/fed_test/all/native/ --tm-exec tmscore/TMscore \
    --out-csv data/all_pdb_1y/fed_test/tm_before.csv

python3 scripts/tmscore_from_pdb.py data/all_pdb_1y/fed_test/pred_client_1_after/ \
    --native-dir data/all_pdb_1y/fed_test/all/native/ --tm-exec tmscore/TMscore \
    --out-csv data/all_pdb_1y/fed_test/tm_client_1_after.csv

python3 scripts/tmscore_from_pdb.py data/all_pdb_1y/fed_test/pred_global/ \
    --native-dir data/all_pdb_1y/fed_test/all/native/ --tm-exec tmscore/TMscore \
    --out-csv data/all_pdb_1y/fed_test/tm_global.csv
```

### 步骤 21 — 汇总成评测矩阵

```bash
python3 scripts/eval_tm_matrix.py \
    --label-map data/all_pdb_1y/fed_test/all/label_client_map.csv \
    --scores \
        before=data/all_pdb_1y/fed_test/tm_before.csv \
        client_0_after=data/all_pdb_1y/fed_test/tm_client_0_after.csv \
        client_1_after=data/all_pdb_1y/fed_test/tm_client_1_after.csv \
        client_2_after=data/all_pdb_1y/fed_test/tm_client_2_after.csv \
        client_3_after=data/all_pdb_1y/fed_test/tm_client_3_after.csv \
        client_4_after=data/all_pdb_1y/fed_test/tm_client_4_after.csv \
        global=data/all_pdb_1y/fed_test/tm_global.csv \
    --out-prefix data/all_pdb_1y/fed_test/tm_matrix
```

- **产出**：
  - `tm_matrix_mean.csv`：**行=模型，列=各 client 测试集 + `all`（并集）**，格=平均 TM；终端也打印这张表。
  - `tm_matrix_long.csv`：逐项明细（model, testset, count, mean_tm, median_tm）。

**怎么读这张矩阵**（正是你要的对比）：

| 你想比较的                       | 看矩阵里的                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| client_i 微调前后（自己测试集）  | 行`before` vs 行 `client_i_after`，都取列 `client_i`                                           |
| client_i 微调前后（全部测试集）  | 行`before` vs 行 `client_i_after`，都取列 `all`                                                |
| 聚合后模型（各自 client 测试集） | 行`global`，逐列看 `client_0..4`                                                                 |
| 聚合后模型（全部测试集）         | 行`global`，列 `all`                                                                             |
| 联邦 vs 单打独斗（泛化）         | 同一列下，`global` vs 各 `client_i_after`；`client_i_after` 在 `client_j`(j≠i) 列即跨域泛化 |

### 评测口径统一（重要）

> 团队里两套算 TM 的方法测的是同一个量（真实 TM-score、残基按 SEQRES 逐位对齐），但**叠合算法不同**：本项目用官方 `TMscore`（TM 最优旋转，标准绝对值）；集中式方案用 in-repo `superimpose()`（RMSD 最优，绝对值系统性偏低）。相对结论一致但**绝对值不可互比**。**本统合流程统一用官方 `TMscore`**，团队汇报请都用这一套重算。

### 结果解读与注意

- `after`/`global` 相对 `before` 在同一列上升即说明微调/聚合有效。
- 测试集是"低 pLDDT 难样本"里划出、且**未参与训练**，因此这是较可信的泛化指标（相比在训练子集上评的旧做法）。
- 两次以上对比务必用**同一 native、同一 `--tm-select`（默认 `max`）**。

---

## 复用到其它 client & 联邦轮次

- **每个 client**：阶段三～五各跑一遍（`client_1`→`client_0/2/3/4`，`docs/selectedPDB/` 下 CSV/日志名也换掉避免覆盖）；阶段七各自用自己的 `train_labels.txt` 训练。数据不出本地、只有权重参与聚合——这就是"联邦"。
- **阶段六/八/九是全局一次性步骤**：阶段六在所有 client 完成预处理后做一次；阶段八聚合；阶段九出矩阵。
- **多轮**：全局模型回灌为下一轮 `--resume_from_ckpt`，重复"本地训练→聚合→评测"直到测试 TM 收敛。

---

## 附：关键路径与产物速查

| 步骤     | 主要产物                                                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1        | `data/all_pdb_1y/all_mmcif_files/*.cif`                                                                                           |
| 2        | `.../all_fasta_files/all.fasta`                                                                                                   |
| 3        | `.../clusters_30.txt`                                                                                                             |
| 4        | `.../fed_split/client_*/`、`split_manifest.json`                                                                                |
| 5        | `.../client_1/solo_fasta_dir/`（全量 fasta）                                                                                      |
| 6        | `.../client_1/solo_alignment_dir/{label}/{label}.pt`（全量 embedding）                                                            |
| 7        | `.../client_1/pred_prescreen/*.pdb`                                                                                               |
| 8        | `docs/selectedPDB/client_1_lowplddt.csv`                                                                                          |
| 9        | `.../client_1/mmcif_files_finetune/*.cif`                                                                                         |
| 10       | `.../client_1/solo_fasta_dir_finetune/_reports/kept_labels.txt`（划分候选池）                                                     |
| 11/12    | `.../client_1/solo_mmcif_cache_finetune.json`、`solo_chain_data_cache_finetune.json`                                            |
| 13       | `.../fed_test/client_*/train_labels.txt`+`test_labels.txt`；`.../fed_test/all/`（fasta/emb/cif/test_labels/label_client_map） |
| 14       | `.../client_1/output_dir/checkpoints/*.ckpt/`（每个 client 一份）                                                                 |
| 15       | `.../fed_global/round1/global_model.pt`                                                                                           |
| 16/17/18 | `.../fed_test/pred_before/`、`pred_client_*_after/`、`pred_global/`                                                           |
| 19       | `.../fed_test/all/native/{label}.pdb`                                                                                             |
| 20       | `.../fed_test/tm_before.csv`、`tm_client_*_after.csv`、`tm_global.csv`                                                        |
| 21       | `.../fed_test/tm_matrix_mean.csv`、`tm_matrix_long.csv`                                                                         |
