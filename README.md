header
*Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B.*

# FedFold 重写版文档

FedFold 是一个面向蛋白质结构预测的联邦学习 SoloSeq 微调仓库，以 5 个 client
本地微调、FedAvg 全局聚合和独立测试集评测为核心流程，支持从数据准备、pLDDT
难样本筛选、train/test 划分到 TM-score 评测矩阵的完整实验闭环。

本仓库主要参考了 [OpenFold](https://github.com/aqlaboratory/openfold) 和
[AlphaFold 2](https://github.com/deepmind/alphafold)。

# 使用方法：

请参考Sha_log[pipeline](docs/sha_log/fed_finetune_pipeline.md)

# 任务进度：

## 2026.07:

1. 下载了从2025-01-01到2026-07-01的数据(符合结构的总数4334)
2. 修改了一下data下载的文件夹名字，对应需要修改的路径也一起改了(一年的数据在data/all_pdb_1y)
3. data_dir_to_fasta脚本进行了修改，避免一次写入(我的内存不够ORZ)
4. 数据聚类划分已提交，5个client，mmcif文件没有上传，上传了处理好的fasta序列。可以按照姓氏排名分配：

- He -> client 0
- HU -> client 1
- Sha -> client 2
- Wu -> client 3
- Wang -> client 4

## 2026.08:

1. 对比之前的全量微调，增加了LoRA微调
2. LoRA微调最佳配置检索， 23 个唯一短训候选组合

- 4 种 Structure Module target
- rank 2/4/8，alpha/rank 1/2
- dropout 0 / 0.05 / 0.1 / 0.2
  最佳（局部最佳）配置：
- Target：structure_module.ipa, structure_module.transition, structure_module.bb_update（即 core）
- Rank：4
- Alpha：8
- Dropout：0
- 学习率：1e-4
- 最佳验证 epoch：5

1. 发现TM score结果非常接近，开始寻找问题
2. 尝试不同学习率发现并没有屌用
3. epoch在client1的64条训练集数据的时候第五轮最佳，往后边开始过拟合
4. 各种尝试最终锁定是模型精度带来的偏差（LoRA微调以及之前的全量微调BF16，原始模型精度FP32）
5. client1 上用最佳 LoRA 配置做 **FP32 训练** 5 epoch（lr=1e-4）后，正式 20 条测试
   mean TM=0.79334，略高于 baseline 0.79305，并明显好于同配置 BF16 训练的 0.787705。
   因此正式流程改为训练/导出/推理全链路 FP32，不再使用 bf16-mixed 训练。
6. 需要注意的是，EMA和raw model的问题，由于训练轮次和数据量都较小，如果是EMA加载的模型其实EMA会严重滞后，比如使用epoch5导出的模型实际上EMA跟epoch1差不多。
7. 对于EMA的问题可以有两种选择，加载EMA，影响也不大；或者直接使用raw model模型不加载EMA权重。
8. 原始的训练文件会保存EMA版本和raw model两种。导出时只取 raw model 的 LoRA A/B，
   base 始终重新取 public EMA 的 FP32 权重。

### Client0 hard-case 定向 LoRA 优化（2026-08-09）

这一轮的目标不是只降低训练 loss，而是改善 baseline 在 client0 上预测偏差较大的hard case，同时限制 medium/easy 样本的退化。当前配置已经通过 3 个训练 seed 的cluster-held-out validation，并冻结为可向其他 client 复制的候选。

#### 与之前配置相比

| 项目                   | 之前                                       | 当前冻结配置                                                          |
| ---------------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| LoRA target            | 只训练 Structure Module core               | T4：Structure Module、`evoformer.linear` 和 Evoformer blocks 44–47 |
| target 规模            | 上游表示基本冻结                           | 167 个模块、273,024 个 LoRA 参数                                      |
| rank / alpha / dropout | 4 / 8 / 0                                  | 保持 4 / 8 / 0，兼容同构 FedLoRA 聚合                                 |
| FAPE                   | 使用随机 clamp                             | `fape_clamp_prob=0`，避免 hard case 梯度被截断                      |
| medium/easy 约束       | 没有专门的 baseline 保持项                 | baseline Cα 几何保持，weight=1.5                                     |
| hard-case 目标         | 通用 OpenFold loss，与 TM-score 不完全对齐 | Kabsch 对齐的 soft-TM margin loss，weight=5.0、margin=0.02            |
| 训练 / 推理            | epoch=5、FP32、scale=1.0                   | epoch=5、FP32，固定 adapter scale=0.85                                |

数据加载会读取本地 baseline 预测结构：medium/easy 样本使用 baseline preservation
抑制遗忘；只有 hard 样本启用 soft-TM 定向损失。这些 baseline 结构和 loss 只留在
client 本地，不需要上传到联邦服务器。LoRA target、rank 和 tensor schema 没有按
样本或 seed 改变，因此后续仍可使用同构 FedLoRA 聚合。

#### 简要分析过程

1. Target ablation 表明只调 Structure Module 的容量和对上游表示的修正能力不足；
   随机 FAPE clamp 也会削弱偏差较大样本的梯度，因此扩展为 T4 并关闭 clamp。
2. 单独提高 epoch、学习率或 rank 没有稳定解决问题；单纯 baseline preservation
   虽能保护普通样本，但对 hard case 的提升泛化不足。
3. 尝试过 hard pair-distance correction，但它只是 TM-score 的间接代理，关键样本
   的改善不如 soft-TM，因此没有作为最终方案。
4. 改为 Kabsch 对齐的可微 soft-TM hard loss 后，3 个 hard validation 样本在每个
   seed 中均为正提升。scale=1.0 时 seed43 的 non-hard 均值为 -0.0056，略越过
   -0.005 下限；预注册一次全局缩放到 0.85 后，hard 提升保留且副作用回到门槛内。

#### 冻结结果

| Seed       | hard TM 均值变化   | hard TM 中位数变化 | non-hard TM 均值变化 | 全集 lDDT 变化     |
| ---------- | ------------------ | ------------------ | -------------------- | ------------------ |
| 42         | +0.00923           | +0.00610           | -0.00239             | +0.00274           |
| 43         | +0.00840           | +0.00770           | -0.00471             | +0.00068           |
| 44         | +0.00860           | +0.00550           | -0.00354             | +0.00142           |
| 三种子汇总 | **+0.00874** | **+0.00643** | 均高于 -0.005 下限   | **+0.00161** |

结论：当前结果证明 client0 的 hard-case TM-score 在内部 cluster-held-out validation
上有稳定但幅度较小的提升，不能表述为已经完成全 FedFold 最终测试。冻结候选记录在
`outputs/fed_lora_hardcase_fed_v1/evaluation/client0_target_ablation_v1/client0_soft_tm_frozen_candidate.json`，
三种子确认记录在同目录的 `t4_soft_tm_scale0p85_seed_confirmation.json`。

共享的 `external_final` 仍为 `unlocked` 且未访问；应先将同一方法复制到 client1–4，
冻结各客户端和 FedLoRA 配置后，再对所有方法统一进行一次最终评测。不要继续用
client0 或 `external_final` 搜索超参数。

## FP32 LoRA 五客户端复现 - LoRA优化和loss优化版

这一节是唯一推荐入口。Git 中只保存各 client 的 FASTA，不保存体积较大的 mmCIF；
脚本会按 FASTA 文件名从 RCSB 自动下载所需 mmCIF，并把所有新文件放进同一个
输出根目录，不会再把中间结果散落到 `fed_split`、`client1_res` 等目录。

### 1.环境准备 直接运行

先进入已经安装好 OpenFold 依赖的任意 Python 环境。Conda 只是可选方案，
virtualenv、venv 或系统 Python 都可以。然后在仓库根目录复制执行：

```bash
# 使用当前环境的 Python；也可以改成 python3 或 Python 的绝对路径
export PYTHON_BIN=python
export GPU_ID=0

# 所有下载、预处理、划分、模型和评测结果统一写到这里
export RUN_ROOT="$PWD/outputs/fed_lora_fp32_v1"
```

脚本会在缺失时自动下载约 605 MB 的 public SoloSeq checkpoint。第一次生成
ESM-1b embedding 时也需要联网。

### 2. 开始训练

```bash
# 准备client数据集 建议跑一下 因为当前的仓库不包含mmcif
# 我把路径改了所以会放在outpus文件夹里面，跟之前文档切割开了
# 所以按照这个直接运行就行
bash scripts/fed_lora_fp32.sh prepare all
# 进行分集 这样每个人电脑上都有全部数据方便使用
bash scripts/fed_lora_fp32.sh split
# 选择自己对应的client进行训练 训练参数是默认调好的
# 如果有问题再说
# 万一结果不好首先考虑：LoRA注入位置？LoRA参数最适化？等等
# 只处理单个 client 时，写成 `0`～`4`，例如：
# 可以写成all 代表一键训练五个client
bash scripts/fed_lora_fp32.sh train 1
bash scripts/fed_lora_fp32.sh evaluate
```

```bash
# （全 5 client一键运行 不推荐）自动执行：下载 mmCIF → 预处理 → 固定划分 → 5 个 client FP32 训练 → 统一评测
bash scripts/fed_lora_fp32.sh all
```

上面的 `scripts/fed_lora_fp32.sh` 保留用于复现早期的 Structure Module core
FP32 baseline，不包含当前冻结的 T4、unclamped FAPE、baseline preservation 和
soft-TM hard loss。将 client0 的冻结方案原样复制到其他 client 时，使用：

```bash
cd /home/wu/Desktop/Fed-fold
export PYTHONPATH="$PWD"
export PYTHON_BIN=/home/wu/miniconda3/envs/fedfold/bin/python
export GPU_ID=0

# client id 仅允许 1、2、3、4；例如复现 client1
bash scripts/fed_lora_hardcase_fed.sh replicate_soft_tm 1
```

该命令固定使用 T4、rank=4、alpha=8、dropout=0、lr=1e-4、FP32、5 epoch、
seed=42、soft-TM weight=5、preservation weight=1.5 和 adapter scale=0.85。
它只在对应 client 的 train/validation 上运行，不会访问 development test 或共享的
`external_final`。结果写入：

```text
outputs/fed_lora_hardcase_fed_v1/clients/client_<id>/private/
└── frozen_soft_tm_v1/T4/uniform/seed_42/
```

client1 的首次迁移复现结果（11 个 cluster-held-out validation 样本）为：

| 指标     | Baseline |     LoRA |      变化 |
| -------- | -------: | -------: | --------: |
| TM-score | 0.529918 | 0.530855 | +0.000936 |
| lDDT-Cα | 0.568700 | 0.570211 | +0.001511 |
| pLDDT    |  54.0473 |  51.5691 |   -2.4782 |

其中 TM-score 为 7/11 上升；hard 子集均值变化为 +0.00236，2/5 上升且
2 个样本达到至少 +0.01，但没有通过 client0 预注册的 promotion-v2 门槛。因此，
这次结果说明冻结方案在 client1 上有轻微正迁移，尚不能说明它对所有 client 都有
稳定提升，也不应把 client1 validation 用来反向修改这套共享配置。

### 统一输出目录

```text
outputs/fed_lora_fp32_v1/
├── run_manifest.json           # Git commit、cluster hash、划分与训练参数
├── clients/client_0..4/        # 下载的 mmCIF、embedding、筛选结果和 cache
├── split/                      # seed=42 的固定 train/test 划分及 native PDB
├── models/client_0..4/         # FP32 训练 checkpoint、日志和合并模型
└── evaluation/
    ├── baseline/
    ├── client_0..4/
    └── summary.csv             # 最终 TM-score / lDDT-Cα / pLDDT 汇总
```

默认实验配置已经固定为：Structure Module core、rank=4、alpha=8、dropout=0、
lr=1e-4、warmup=20、5 epoch、seed=42、FP32 训练与 FP32 推理。train/test 是按
30% sequence cluster 隔离的固定 hold-out，不是交叉验证。

评测会为每个模型生成逐样本的 `tm_score.csv`、`lddt_ca.csv` 和 `plddt.csv`。
其中 lDDT-Cα 和 TM-score 都使用 native 结构；pLDDT 是模型自身的置信度。

需要改路径或运行参数时使用环境变量，不需要修改脚本：

```bash
PYTHON_BIN=/path/to/python \
DATA_ROOT=/path/to/data/all_pdb_1y \
RUN_ROOT=/path/to/output \
GPU_ID=0 \
bash scripts/fed_lora_fp32.sh all
```

```bash
cd /home/wu/Desktop/Fed-fold
export PYTHONPATH="$PWD"
export PYTHON_BIN=/home/wu/miniconda3/envs/fedfold/bin/python
export GPU_ID=0
export RUN_ROOT="$PWD/outputs/fedfold_lora_hardcase_loss_opt"

# 冻结 soft-TM 配置
export TARGET_SLUG=T4
export ARMS=uniform
export SEEDS=42
export LOCAL_ONLY_EPOCHS=5
export LOCAL_EVAL_EPOCHS=5
export LOCAL_SCALES=0.85
export PRIMARY_EVAL_SCALE=0.85
export LR=1e-4
export RANK=4
export ALPHA=8
export DROPOUT=0
export LORA_TARGET="structure_module,evoformer.linear,evoformer.blocks.44,evoformer.blocks.45,evoformer.blocks.46,evoformer.blocks.47"
export EXPERIMENT_CONFIG_JSON="$PWD/seq_model_esm1b_ptm_hardcase_unclamped_override.json"
export BASELINE_PRESERVATION_WEIGHT=1.5
export HARD_SOFT_TM_WEIGHT=5.0
export HARD_SOFT_TM_MARGIN=0.02
# all = client_0..4；只要 1–4 可改成循环
export LOCAL_CLIENTS=all
bash scripts/fed_lora_hardcase_fed.sh local_train
bash scripts/fed_lora_hardcase_fed.sh local_validate
bash scripts/fed_lora_hardcase_fed.sh local_select
```

## T4 soft-TM FedLoRA 五客户端联邦训练（当前推荐）

联邦训练使用独立的 `RUN_ROOT`，不会把前面的 5 epoch local-only
检查点直接拿去聚合。每一轮的实际顺序是：

1. 五个 client 都从当前同一个 global model 开始。
2. 每个 client 在自己的 train split 上重新微调 1 epoch。
3. 服务器只读取 raw-model LoRA A/B，计算
   `delta_W=(alpha/rank)*(B@A)`。
4. 按 client 训练样本数加权聚合，得到下一轮 global model。

T4 的 167 个 LoRA 目标包含 Structure Module、`evoformer.linear` 和
Evoformer blocks 44–47。当前实现已处理冻结输入下激活检查点的反向传播；
梯度冒烟检查已确认 167/167 个目标都会产生非零更新。

### 完整运行指令

```bash
cd /home/wu/Desktop/Fed-fold
export PYTHONPATH="$PWD"
export PYTHON_BIN=/home/wu/miniconda3/envs/fedfold/bin/python
export GPU_ID=0

# 与 local-only 输出隔离，避免混用检查点
export RUN_ROOT="$PWD/outputs/fedfold_lora_hardcase_loss_opt_fed"

# 冻结的 T4 soft-TM 配置
export TARGET_SLUG=T4
export ARMS=uniform
export SEEDS=42
export ROUNDS=5
export LOCAL_EPOCHS=1
export LR=1e-4
export RANK=4
export ALPHA=8
export DROPOUT=0
export LORA_TARGET="structure_module,evoformer.linear,evoformer.blocks.44,evoformer.blocks.45,evoformer.blocks.46,evoformer.blocks.47"
export EXPERIMENT_CONFIG_JSON="$PWD/seq_model_esm1b_ptm_hardcase_unclamped_override.json"
export BASELINE_PRESERVATION_WEIGHT=1.5
export HARD_SOFT_TM_WEIGHT=5.0
export HARD_SOFT_TM_MARGIN=0.02

# 新 RUN_ROOT 只需准备一次：生成难度表、固定划分和 round_000
bash scripts/fed_lora_hardcase_fed.sh prepare

# 5 轮联邦训练；中断后用同一组环境变量重跑即可续训
bash scripts/fed_lora_hardcase_fed.sh train
```

`train` 会跳过已存在最终检查点的 client/round，因此可以安全重入。
联邦聚合固定使用未缩放的 raw effective delta（scale=1.0）；
local-only 的 `LOCAL_SCALES=0.85` 不应传入联邦聚合。

每轮成功后应出现：

```text
outputs/fedfold_lora_hardcase_loss_opt_fed/server/rounds/uniform/seed_42/
├── round_001/global_model.pt
├── round_001/aggregation.json
├── ...
└── round_005/global_model.pt
```

`aggregation.json` 必须满足 `status="ok"`、`non_target_max_abs_diff=0.0`，
并且包含 5 个 `participating_clients` 和 167 个 `target_keys`。

### 联邦验证、监控与全局轮次选择

训练完成后，必须在每个 client 的私有 validation split 上评估
`round_000`–`round_005`，才能选最佳全局轮次。脚本现已补齐这一步：

- `round_000` 直接从冻结的 baseline difficulty 表生成零增益配对结果，
  不重复跑基线 PDB 推理。
- `round_001`–`round_005` 分别用对应 global model 推理 5 个 client。
- 每个 client/round 都写入 `inference_manifest.json`、PDB 指标和
  `paired_deltas.csv`。
- 已完整结束且模型 SHA 一致的任务会自动跳过；推理完成但打分中断的
  任务会复用 PDB。可以用同一指令安全续跑。

保持上一节的环境变量不变，再设置：

```bash
export FED_VALIDATION_CLIENTS=all  # 也可用 0,1 或单个 2 分批执行
export FED_VALIDATION_ROUNDS=all   # 也可用 0,1,2 或单个 5 分批执行
```

前台运行：

```bash
bash scripts/fed_lora_hardcase_fed.sh validate_infer
```

需要关闭终端后继续运行时：

```bash
mkdir -p "$RUN_ROOT/logs"
nohup bash scripts/fed_lora_hardcase_fed.sh validate_infer \
  > "$RUN_ROOT/logs/validate_infer.log" 2>&1 &
echo $!
```

本配置共有 30 个 client/round 验证单元：round 0 的 5 个基线单元，
以及 round 1–5 的 25 个推理单元；后者总计生成 260 个 validation PDB。
单张 GPU 上不要同时启动多个 `validate_infer` 进程。

#### 查看进度

```bash
# 完整性进度：最终应为每轮 5/5、总计 30/30
bash scripts/fed_lora_hardcase_fed.sh validation_status

# 每 30 秒刷新一次
watch -n 30 bash scripts/fed_lora_hardcase_fed.sh validation_status

# 后台运行时查看实时日志
 tail -f "$RUN_ROOT/logs/validate_infer.log"

# 查看 GPU
nvidia-smi
```

如果中断，原样重新执行 `validate_infer` 即可。也可以分批续跑，例如：

```bash
FED_VALIDATION_ROUNDS=1,2 FED_VALIDATION_CLIENTS=0,1,2,3,4 \
  bash scripts/fed_lora_hardcase_fed.sh validate_infer
```

`validate_infer` 结束时会自动刷新服务器侧汇总。确认进度为 `30/30` 后：

```bash
# 可选：只重建 JSON 汇总，不跑 PDB 推理
bash scripts/fed_lora_hardcase_fed.sh validate

# 只有 0–5 轮且每轮 5 个 client 均为 metric_status=ok 时才允许选轮
bash scripts/fed_lora_hardcase_fed.sh select_global
bash scripts/fed_lora_hardcase_fed.sh report

# 查看最终选择
$PYTHON_BIN -m json.tool \
  "$RUN_ROOT/server/rounds/uniform/seed_42/global_best_selection.json"
```

最佳模型输出为：

```text
$RUN_ROOT/server/rounds/uniform/seed_42/global_best.pt
```

`select_global` 采用“非 hard 平均 TM 下降不低于 `-0.005`，再最大化
hard 平均 TM 增益”的约束；现在会拒绝缺轮次、缺 client 或
`metric_status != ok` 的汇总，避免把未完成验证误当成有效结果。

# openfold原始文档引导

See our new home for docs at [openfold.readthedocs.io](https://openfold.readthedocs.io/en/latest/), with instructions for installation and model inference/training.

Much of the content from this page may be found [here.](https://github.com/aqlaboratory/openfold/blob/main/docs/source/original_readme.md)

## Copyright Notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters
fall under the CC BY 4.0 license, a copy of which is downloaded to
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Contributing

If you encounter problems using OpenFold, feel free to create an issue! We also
welcome pull requests from the community.

## Citing this Work

Please cite our paper:

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

If you use OpenProteinSet, please also cite:

```bibtex
@misc{ahdritz2023openproteinset,
      title={{O}pen{P}rotein{S}et: {T}raining data for structural biology at scale}, 
      author={Gustaf Ahdritz and Nazim Bouatta and Sachin Kadyan and Lukas Jarosch and Daniel Berenberg and Ian Fisk and Andrew M. Watkins and Stephen Ra and Richard Bonneau and Mohammed AlQuraishi},
      year={2023},
      eprint={2308.05326},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

Any work that cites OpenFold should also cite [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) if applicable.
