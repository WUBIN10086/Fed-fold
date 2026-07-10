# 联邦学习数据准备 → pLDDT 筛选 → SoloSeq 微调 → TM-score 评测 完整流程

本文档记录从**下载 PDB 数据**，到**按聚类分成 5 个联邦 client**，再到**用 pLDDT 预筛选微调数据**，对单个 client 做 **SoloSeq(ESM-1b) 微调**，最后**对比微调前后 TM-score** 的完整可复现步骤。跟着从头到尾执行即可，路径可按需修改（下文默认以 `client_1` 为例，其余 client 把命令里的 `client_1` 换成对应编号即可）。

> **关于 pLDDT 筛选（本版本新增）**：并不是拿 client 的全部数据去微调，而是**先用原始 SoloSeq 权重预测全部序列，算每个结构的平均 pLDDT，只把 pLDDT < 80（即模型当前"没预测好"的难样本）挑出来作为微调数据**；pLDDT ≥ 80 的样本模型本来就预测得不错，不参与微调。这样微调更聚焦在模型薄弱的样本上。

> 说明：以下所有命令建议在**项目根目录**下、且在有 GPU 的远程服务器上执行。示例里的 `CUDA_VISIBLE_DEVICES=2` 表示用 2 号卡，按你机器实际情况改。

---

## 0. 前置准备

在开始前，确认以下依赖已就绪：

| 依赖 | 用途 | 检查方式 |
|---|---|---|
| Python 环境（OpenFold 依赖） | 运行所有脚本 | `python -c "import openfold"` |
| `mmseqs` 可执行文件 | 序列聚类 | `which mmseqs` 有输出 |
| ESM-1b 权重（可联网自动下载） | 生成 embedding | 首次运行会 `torch.hub` 下载；不能联网见下方说明 |
| SoloSeq 初始权重 `seq_model_esm1b_ptm.pt` | 预筛选预测 + 微调的起点 | 位于 `openfold/resources/openfold_soloseq_params/`，缺失则用 `scripts/download_openfold_soloseq_params.sh` 下载 |
| `TMscore` 可执行文件 | 算 TM-score | 项目已带编译好的 `tmscore/TMscore` |

若服务器**不能联网**下载 ESM-1b，可给 `precompute_embeddings.py` 传 `--use_local_esm <本地esm仓库路径>`。

### 流程总览

```
下载 cif ─▶ 抽序列(all.fasta) ─▶ MMseqs2 聚类(30%) ─▶ 分 5 个 client
                                                            │
                                          (以下针对单个 client, 例如 client_1)
                                                            ▼
   全量: 单序列 fasta ─▶ ESM embedding
                                    │
                                    ▼
   全量预测(原始权重) ─▶ 算 pLDDT ─▶ 选 mean pLDDT<80 ─▶ 抽取子集 cif  ← 微调数据
                                                            │
                                                            ▼
   子集: kept_labels + 两个缓存 ─▶ 微调训练(10000 step)
                                                            │
                                                            ▼
   子集微调前预测 ┐
   子集微调后预测 ┴─▶ 抽 native(按SEQRES对齐) ─▶ 分别算 TM-score ─▶ 对比 before/after
```

---

## 阶段一：下载数据

### 步骤 1 — 从 RCSB 下载 mmCIF

```bash
python download_cif_from_pdb_bankV2.py
```

- **作用**：按脚本内设定的检索条件（近一年、X-ray、高分辨率、单蛋白单链、长度 30–300）从 RCSB 检索并下载 `.cif`。
- **可调参数**：脚本顶部的 `DATE_START/DATE_END`、`RES_MAX`、`LEN_MIN/LEN_MAX`、`MAX_ENTRIES` 等，按需修改。
- **产出**：
  - `data/sha_pdb_0703/sha_mmcif_files/*.cif`
  - `data/sha_pdb_0703/entry_ids.txt`（下载到的 PDB ID 列表）

---

## 阶段二：序列抽取 → 聚类去冗余 → 分 client

### 步骤 2 — 从 cif 抽取序列，汇总成一个 FASTA

```bash
python scripts/data_dir_to_fasta.py \
    data/sha_pdb_0703/sha_mmcif_files/ \
    data/sha_pdb_0703/sha_fasta_files/all.fasta
```

- **作用**：遍历所有 `.cif`，把每条链序列写进一个 `all.fasta`，记录名格式为 `>{PDBID}_{CHAIN}`（正是后续聚类需要的格式）。
- **产出**：`data/sha_pdb_0703/sha_fasta_files/all.fasta`

### 步骤 3 — 用 MMseqs2 按 30% identity 聚类去冗余

```bash
python scripts/fasta_to_clusterfile.py \
    data/sha_pdb_0703/sha_fasta_files/all.fasta \
    data/sha_pdb_0703/clusters_30.txt \
    $(which mmseqs) \
    --seq-id 0.3
```

- **作用**：用与 PDB 官方一致的 MMseqs2 参数，把相似序列归到同一 cluster。`--seq-id 0.3` 即 30% identity 阈值。
- **产出**：`data/sha_pdb_0703/clusters_30.txt`，**每行一个 cluster**（空格分隔的 `PDBID_CHAIN`）。
- **检查**：`wc -l data/sha_pdb_0703/clusters_30.txt` 即聚类后的 cluster 数量。
- **注意**：`$(which mmseqs)` 是 bash 写法；若在 PowerShell 里运行，请把它替换成 mmseqs 的绝对路径。

### 步骤 4 — 以 cluster 为单位均分给 5 个 client

```bash
python scripts/split_clusters_to_clients.py \
    data/sha_pdb_0703/clusters_30.txt \
    data/sha_pdb_0703/sha_mmcif_files/ \
    data/sha_pdb_0703/fed_split \
    --num_clients 5
```

- **作用**：以 cluster 为最小单位（**保证同一 cluster 不跨 client**，避免 client 间高相似度数据泄漏），用贪心让各 client 样本量尽量均衡，并为每个 client 建立独立的 cif 子目录。
- **产出**：
  ```
  data/sha_pdb_0703/fed_split/
    client_0/mmcif_files/*.cif   client_0/pdb_ids.txt
    client_1/mmcif_files/*.cif   client_1/pdb_ids.txt
    ... client_4/ ...
    split_manifest.json          # 每个 client 的 cluster数 / 链数 / PDB数
  ```
- **检查**：看 `split_manifest.json` 里各 client 的 `num_pdbs` 是否大致均衡、`cif_missing` 是否为空。
- **可选**：磁盘紧张可加 `--link`，用软链接代替拷贝。

> 从这里开始，下面所有步骤都是**针对单个 client** 的。示例用 `client_1`，其它 client 把命令里的 `client_1` 全部替换即可。

---

## 阶段三：单 client 全量数据预处理

这一阶段先对 client 的**全部**数据做预处理，产出后续「全量预测」所需的 fasta 与 embedding。

### 步骤 5 — cif → 单序列 FASTA（全量，带质量过滤）

```bash
python3 scripts/prep_solo_fasta_learnable.py \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files/ \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir \
    --max_len 1022 \
    --dedup_by_sequence
```

- **作用**：把每条链导成单条 `{label}.fasta`，并做质量过滤（长度、X 比例、非标准残基、分辨率），`--dedup_by_sequence` 按序列去重。`--max_len 1022` 对应 ESM-1b 的长度上限。
- **产出**：`.../client_1/solo_fasta_dir/*.fasta` 及 `_reports/`（含 `kept_labels.txt`、`summary.txt` 等）。

### 步骤 6 — 生成 ESM-1b embedding（全量，当作 SoloSeq 的 "alignment"）

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/precompute_embeddings.py \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir/ \
    data/sha_pdb_0703/fed_split/client_1/solo_alignment_dir
```

- **作用**：为**每条**序列预计算 ESM-1b 表征，SoloSeq 用它代替 MSA。
- **产出**：`.../client_1/solo_alignment_dir/{label}/{label}.pt`。
- **注意**：这份全量 embedding 后面「全量预测」「微调训练」「微调前/后预测」都会复用，无需重复生成。超过 1022 残基的序列会被跳过并记录在 `skipped_sequences.txt`。不能联网时加 `--use_local_esm <本地esm仓库路径>`。

---

## 阶段四：pLDDT 预筛选，构建微调子集

用**原始 SoloSeq 权重**把全量数据预测一遍，算 pLDDT，挑出 mean pLDDT < 80 的难样本，作为微调数据。

### 步骤 7 — 全量预测（原始权重，仅用于筛选）

```bash
python3 run_pretrained_openfold.py \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir/ \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files/ \
    --use_precomputed_alignments data/sha_pdb_0703/fed_split/client_1/solo_alignment_dir/ \
    --output_dir data/sha_pdb_0703/fed_split/client_1/pred_prescreen \
    --model_device "cuda:0" \
    --skip_relaxation \
    --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
```

- **作用**：对全部序列用原始权重预测。SoloSeq 预测会把每个残基的 pLDDT 写进输出 PDB 的 B-factor 列，供下一步统计。
- **产出**：`.../client_1/pred_prescreen/*.pdb`（文件名形如 `{label}_seq_model_esm1b_ptm_unrelaxed.pdb`）。

### 步骤 8 — 计算 pLDDT，选出 mean pLDDT < 80 的样本

```bash
python3 scripts/plddt_from_pdb.py \
    data/sha_pdb_0703/fed_split/client_1/pred_prescreen/ \
    --selected-csv docs/selectedPDB/client_1_lowplddt.csv \
    --bad-log docs/selectedPDB/client_1_bad_pdb.txt \
    --no-extremes
```

- **作用**：逐个读取预测 PDB 的 B-factor（pLDDT），计算每个结构的平均 pLDDT。`--select-threshold` 默认 **80**，会把 **mean pLDDT < 80** 的结构写入 `--selected-csv`。`--no-extremes` 关闭逐残基的最高/最低明细打印。
- **产出**：
  - `docs/selectedPDB/client_1_lowplddt.csv`：列 `protein_name,mean_plddt`，即微调子集的标签清单（`protein_name` 形如 `9vak_A`）。
  - `docs/selectedPDB/client_1_bad_pdb.txt`：解析失败/空文件的记录。
- **阈值可调**：想改成别的阈值加 `--select-threshold <值>`（例如 `--select-threshold 70`）。

### 步骤 9 — 抽取低 pLDDT 子集的 cif，作为微调数据目录

```bash
python3 scripts/extract_selected_mmcif.py \
    --csv-path docs/selectedPDB/client_1_lowplddt.csv \
    --source-dir data/sha_pdb_0703/fed_split/client_1/mmcif_files \
    --dst-dir data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune
```

- **作用**：读 `client_1_lowplddt.csv` 里的 `protein_name`，取出 `pdb_id`，把对应 `.cif` 从全量目录拷到微调子集目录。**这个子集就是真正拿去微调的数据。**
- **产出**：
  - `.../client_1/mmcif_files_finetune/*.cif`（pLDDT<80 的子集）
  - 该目录下还有 `copied_manifest.csv`（拷贝清单）、`missing_ids.txt`（在源目录没找到的 id）。
- **可选**：加 `--dry-run` 只看统计不实际拷贝。

---

## 阶段五：微调子集预处理

在**子集** cif 上生成训练所需的 fasta 白名单和两个缓存（embedding 直接复用步骤 6 的全量 `solo_alignment_dir`，无需重算）。

### 步骤 10 — 子集 cif → 单序列 FASTA（得到微调白名单 kept_labels.txt）

```bash
python3 scripts/prep_solo_fasta_learnable.py \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir_finetune \
    --max_len 1022 \
    --dedup_by_sequence
```

- **作用**：对子集重新生成单序列 fasta，其 `_reports/kept_labels.txt` 即微调训练的样本白名单（`--train_filter_path` 用）。这个 fasta 目录同时也是后面「微调前/后预测」的输入。
- **产出**：`.../client_1/solo_fasta_dir_finetune/*.fasta` 及 `_reports/kept_labels.txt`。

### 步骤 11 — 生成 mmCIF 结构信息缓存（子集）

```bash
python3 scripts/generate_mmcif_cache.py \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/solo_mmcif_cache_finetune.json \
    --no_workers 8
```

- **产出**：`.../client_1/solo_mmcif_cache_finetune.json`（训练时兼作 `--template_release_dates_cache_path`）。

### 步骤 12 — 生成链数据缓存（子集）

```bash
python3 scripts/generate_chain_data_cache.py \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/solo_chain_data_cache_finetune.json \
    --cluster_file data/sha_pdb_0703/clusters_30.txt \
    --no_workers 8
```

- **产出**：`.../client_1/solo_chain_data_cache_finetune.json`（`--cluster_file` 仍用全量聚类表给出 `cluster_size`）。

---

## 阶段六：微调训练

### 步骤 13 — 在低 pLDDT 子集上做 SoloSeq 微调

```bash
CUDA_VISIBLE_DEVICES=2 python3 train_openfold.py \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/solo_alignment_dir/ \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/output_dir \
    2026-01-01 \
    --train_filter_path data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir_finetune/_reports/kept_labels.txt \
    --use_single_seq_mode True \
    --config_preset seq_model_esm1b_ptm \
    --experiment_config_json seq_model_esm1b_ptm_finetune_override.json \
    --resume_from_ckpt openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
    --resume_model_weights_only True \
    --template_release_dates_cache_path data/sha_pdb_0703/fed_split/client_1/solo_mmcif_cache_finetune.json \
    --train_chain_data_cache_path data/sha_pdb_0703/fed_split/client_1/solo_chain_data_cache_finetune.json \
    --precision bf16-mixed \
    --gpus 1 \
    --deepspeed_config_path deepspeed_config.json
```

**位置参数说明（前 5 个是位置参数，顺序不能乱）**：

| 位置 | 值 | 含义 |
|---|---|---|
| 1 `train_data_dir` | `client_1/mmcif_files_finetune/` | 训练用 mmCIF（**低 pLDDT 子集**）|
| 2 `train_alignment_dir` | `client_1/solo_alignment_dir/` | 步骤 6 的全量 ESM embedding（含子集，直接复用）|
| 3 `template_mmcif_dir` | `client_1/mmcif_files_finetune/` | 模板搜索目录（SoloSeq 不真用模板，指向同目录即可）|
| 4 `output_dir` | `client_1/output_dir` | 输出（checkpoint / 日志）|
| 5 `max_template_date` | `2026-01-01` | 必须 ≥ 数据发布日期（本例数据为 2025 年），设成之后的日期即可 |

**关键可选参数**：

- `--train_filter_path .../solo_fasta_dir_finetune/_reports/kept_labels.txt`：只在子集白名单上训练。
- `--resume_from_ckpt ... --resume_model_weights_only True`：以官方 SoloSeq 权重为起点，只载入模型权重。
- `--use_single_seq_mode True` / `--config_preset seq_model_esm1b_ptm`：启用单序列(ESM)模式。
- `--experiment_config_json seq_model_esm1b_ptm_finetune_override.json`：微调超参覆盖。
- `--deepspeed_config_path deepspeed_config.json` + `--precision bf16-mixed`：DeepSpeed + 混合精度。

**产出**：`.../client_1/output_dir/checkpoints/` 下的 checkpoint。因为用了 DeepSpeed，checkpoint 是一个**目录**（形如 `epoch=0-step=10000.ckpt/`，内含 zero 分片），后续预测直接把这个目录路径传进去即可。

> 训练结束后先 `ls data/sha_pdb_0703/fed_split/client_1/output_dir/checkpoints/` 确认实际的 checkpoint 名字，下一步要用。

---

## 阶段七：预测（微调前 / 微调后）

在**微调子集**上，分别用「原始权重」和「微调后权重」各预测一次。两条命令只有 `--openfold_checkpoint_path` 和 `--output_dir` 不同。

> 说明：`--config_preset seq_model_esm1b_ptm` 会自动开启单序列模式，因此这里**不需要**再加 `--use_single_seq_mode`。评测输入统一用**子集** fasta（`solo_fasta_dir_finetune`），embedding 复用全量 `solo_alignment_dir`。

### 步骤 14 — 微调前预测（原始 SoloSeq 权重，baseline）

```bash
python3 run_pretrained_openfold.py \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    --use_precomputed_alignments data/sha_pdb_0703/fed_split/client_1/solo_alignment_dir/ \
    --output_dir data/sha_pdb_0703/fed_split/client_1/pred_before \
    --model_device "cuda:0" \
    --skip_relaxation \
    --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
```

- **产出**：`.../client_1/pred_before/*.pdb`。
- **省算力提示**：这批预测和步骤 7 用的是同样的原始权重、同样的序列（只是子集），你也可以直接从 `pred_prescreen/` 里挑出子集对应的 `.pdb` 复用，跳过本步。

### 步骤 15 — 微调后预测（训练得到的 checkpoint）

```bash
python3 run_pretrained_openfold.py \
    data/sha_pdb_0703/fed_split/client_1/solo_fasta_dir_finetune/ \
    data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    --use_precomputed_alignments data/sha_pdb_0703/fed_split/client_1/solo_alignment_dir/ \
    --output_dir data/sha_pdb_0703/fed_split/client_1/pred_after \
    --model_device "cuda:0" \
    --skip_relaxation \
    --config_preset "seq_model_esm1b_ptm" \
    --openfold_checkpoint_path data/sha_pdb_0703/fed_split/client_1/output_dir/checkpoints/epoch\=0-step\=10000.ckpt/
```

- **注意**：`--openfold_checkpoint_path` 直接传 DeepSpeed 的 `.ckpt` **目录**即可，脚本会自动把 zero 分片转换成 fp32 权重再加载（用其中的 EMA 权重）。checkpoint 名字按步骤 13 实际产出替换（本例 `epoch=0-step=10000.ckpt`，命令里的 `\=` 是对 `=` 的转义）。
- **产出**：`.../client_1/pred_after/*.pdb`。

---

## 阶段八：提取 native + 计算 TM-score

### 步骤 16 — 从 cif 提取 native 单链结构（按 SEQRES 对齐）

```bash
python3 scripts/extract_native_chain_pdbsV2.py \
    --cif-dir data/sha_pdb_0703/fed_split/client_1/mmcif_files_finetune/ \
    --out-dir data/sha_pdb_0703/fed_split/client_1/native_pdb_v2 \
    --pred data/sha_pdb_0703/fed_split/client_1/pred_after/
```

- **作用**：把真值结构从 cif 抽成单链 `.pdb`。V2 版用 OpenFold 解析链路，坐标**按 SEQRES 对齐、残基编号 `1..N`**，与 SoloSeq 全长预测逐位对应；链命名与预处理同源，避免 auth/label 错位。
- **`--pred`**：从预测结果反推 label，保证 native 集合与实际预测出来的样本完全一致。
- **产出**：`.../client_1/native_pdb_v2/{label}.pdb`，以及 `docs/selectedPDB/` 下的抽取报告。

### 步骤 17 — 计算微调前 TM-score

```bash
python3 scripts/tmscore_from_pdb.py \
    data/sha_pdb_0703/fed_split/client_1/pred_before/ \
    --native-dir data/sha_pdb_0703/fed_split/client_1/native_pdb_v2/ \
    --tm-exec tmscore/TMscore \
    --out-csv data/sha_pdb_0703/fed_split/client_1/tmscore_before.csv
```

### 步骤 18 — 计算微调后 TM-score

```bash
python3 scripts/tmscore_from_pdb.py \
    data/sha_pdb_0703/fed_split/client_1/pred_after/ \
    --native-dir data/sha_pdb_0703/fed_split/client_1/native_pdb_v2/ \
    --tm-exec tmscore/TMscore \
    --out-csv data/sha_pdb_0703/fed_split/client_1/tmscore_after.csv
```

- **作用**：批量用 `TMscore` 把每个预测 `.pdb` 与同名 native `.pdb` 比对，输出逐样本 TM-score 及汇总。默认 `--strip-suffix _seq_model_esm1b_ptm_unrelaxed` 正好把预测文件名映射回 `{label}`，从而找到 native 里的 `{label}.pdb`。
- **产出**：`tmscore_before.csv` / `tmscore_after.csv`（含 `label`、`tm_selected`、`rmsd` 等列），终端还会打印 `TM-score summary (mean/median/...)`。

### 结果解读

- 对比两个 CSV（或终端汇总）里 `tm_selected` 的 **mean / median**：`after` 高于 `before` 即说明微调让这批**原本 pLDDT 低（难预测）的样本**结构精度提升了。
- **务必保证两次评测用的是同一个 `--native-dir` 和同一个 `--tm-select`（默认 `max`），对比才公平。**
- 因为微调集是"低 pLDDT 难样本"，这批样本本身 baseline TM-score 就偏低，正好用来观察微调是否把短板补上；这也是在**训练集自身**上评测（结果偏乐观），若需更严谨结论应另留不参与训练的样本单独评测。

---

## 复用到其它 client 与后续联邦聚合

- **其它 client**：把阶段三～八所有命令里的 `client_1` 替换为 `client_0` / `client_2` / … / `client_4`（`docs/selectedPDB/` 下的 CSV/日志名也换成对应 client），逐个重复即可（各 client 的数据、embedding、pLDDT 筛选、缓存、checkpoint、预测、评测均相互独立）。
- **联邦聚合（FedAvg）**：当 5 个 client 各自微调完成后，可把 5 份 checkpoint 的权重做平均，作为下一轮的全局初始权重，再分发回各 client 继续训练，循环若干轮。该聚合脚本尚未在本流程内，需要时再单独实现。

---

## 附：关键路径与产物速查

| 步骤 | 主要产物 |
|---|---|
| 1 | `data/sha_pdb_0703/sha_mmcif_files/*.cif` |
| 2 | `.../sha_fasta_files/all.fasta` |
| 3 | `.../clusters_30.txt` |
| 4 | `.../fed_split/client_*/`（含 `mmcif_files/`、`pdb_ids.txt`）、`split_manifest.json` |
| 5 | `.../client_1/solo_fasta_dir/`（全量 fasta）|
| 6 | `.../client_1/solo_alignment_dir/{label}/{label}.pt`（全量 embedding，全程复用）|
| 7 | `.../client_1/pred_prescreen/*.pdb`（全量预测，仅用于筛选）|
| 8 | `docs/selectedPDB/client_1_lowplddt.csv`（mean pLDDT<80 的样本）|
| 9 | `.../client_1/mmcif_files_finetune/*.cif`（微调子集）|
| 10 | `.../client_1/solo_fasta_dir_finetune/`（含 `_reports/kept_labels.txt`）|
| 11/12 | `.../client_1/solo_mmcif_cache_finetune.json`、`solo_chain_data_cache_finetune.json` |
| 13 | `.../client_1/output_dir/checkpoints/*.ckpt/` |
| 14/15 | `.../client_1/pred_before/`、`pred_after/`（均为子集）|
| 16 | `.../client_1/native_pdb_v2/{label}.pdb` |
| 17/18 | `.../client_1/tmscore_before.csv`、`tmscore_after.csv` |
