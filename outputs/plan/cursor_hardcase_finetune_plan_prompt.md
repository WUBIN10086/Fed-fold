# Cursor Plan 模式提示词：按客户端进行 baseline 困难样本纠偏

你现在处于 **Plan 模式**。请先对仓库进行只读检查并输出一份可以直接执行的详细实施计划；本阶段不要修改代码、不要启动训练或全量推理、不要覆盖任何已有结果。计划获批后才进入实现阶段。

## 一、项目背景

仓库路径为：

```text
/home/wu/Desktop/Fed-fold
```

请优先检查这些现有文件和目录，并基于真实实现制定计划，不要凭空假设接口：

```text
scripts/fed_lora_fp32.sh
train_openfold.py
scripts/export_lora_checkpoint.py
scripts/verify_lora_roundtrip.py
scripts/tmscore_from_pdb.py
scripts/lddt_ca_from_pdb.py
scripts/plddt_from_pdb.py
openfold/data/data_modules.py
outputs/fed_lora_fp32/
outputs/fed_lora_fp32/run_manifest.json
outputs/fed_lora_fp32/split/
outputs/fed_lora_fp32/evaluation/
outputs/fed_lora_fp32/models/
```

IDE 中曾出现 `outputs/fed_lora_fp32_v1`，但不要假定它一定存在。请先确认实际目录，并在计划中说明 `_v1` 与非 `_v1` 路径是否存在不一致。

现有实验配置大致为：

```text
5 个 client
FP32 LoRA
rank = 4
alpha = 8
learning rate = 1e-4
max_epochs = 5
LoRA target = structure_module.ipa,
              structure_module.transition,
              structure_module.bb_update
```

现有结果表明，所有模型相对 baseline 的总体 TM-score 基本持平，没有统计可靠的整体提升。更重要的是，我们真正的研究目标并不是提高所有样本的 pooled mean，而是：

> 对每个已经划分好的 client，重点改善该 client 中 baseline 预测偏差较大的困难样本，同时尽量不破坏 baseline 已经预测较好的样本。

当前结果中，以 `baseline TM-score < 0.5` 定义困难样本，共有约 26 条；现有 LoRA 在该子集上的平均增益仍接近 0。因此需要把训练采样、validation 选择和评估口径改成 hard-case correction，而不是继续盲目增大 epoch、学习率或 rank。

## 二、本阶段的明确范围

当前阶段先设计 **5 个独立的 personalized/local LoRA**，不是立即实现 FedAvg/FedLoRA 聚合：

```text
client_0 -> local LoRA 0
client_1 -> local LoRA 1
...
client_4 -> local LoRA 4
```

困难度标注、采样和训练必须在各 client 内部完成，不得混合或移动不同 client 的原始训练数据。

请把真正的联邦聚合设计作为后续可选阶段简要列出，但不要让它扩大当前实现范围。

## 三、数据划分与防泄漏要求

请检查现有 split 是如何按 30% sequence cluster 划分的，以及 train/test cluster 是否严格无泄漏。

目标数据流程为：

```text
每个 client 的本地数据
├── local train
├── local validation
└── local final test
```

要求：

1. 必须按 sequence cluster/group 划分，不能随机按单条 chain 划分。
2. 必须先完成 cluster split，再分别计算各 split 的 baseline 难度。
3. validation 用于选择 epoch、LoRA scale、采样策略和 adapter 权重来源。
4. final test 不得参与超参数选择。
5. 当前 80 条 test 已经被查看和分析，应在计划中明确它只能继续作为 development test，不能再被当作完全未见的最终统计检验集。
6. 请先检查仓库中是否还有未使用且 cluster-disjoint 的数据，可以构建新的 external/final test。
7. 如果无法构建新的 final test，请规划基于 cluster/group 的 nested cross-validation 备选方案，例如 outer 3-fold + inner 3-fold，并说明算力成本。
8. 由于每个 client 数据量较小，请估计新增 validation 后每个 client 的 train/validation/test label 数和 hard cluster 数。如果单个 client 的 hard validation cluster 太少，请规划最小数量保护或 group cross-validation，不要静默地在 1～2 个 hard 样本上选择模型。

## 四、baseline 难度表

请规划一个可复现的 baseline difficulty 生成步骤。对于所有拥有 native structure 的 train/validation/test 样本，使用同一个固定 baseline、相同 checkpoint source、相同 FP32 推理设置和固定 seed，输出类似：

```csv
label,client,split,cluster_id,baseline_tm,baseline_lddt_ca,baseline_plddt,sequence_length,difficulty
```

默认难度定义为：

```text
hard:   baseline_tm < 0.5
medium: 0.5 <= baseline_tm < 0.8
easy:   baseline_tm >= 0.8
```

同时保留连续难度，用于采样权重：

```text
difficulty_weight = max(0, (0.5 - baseline_tm) / 0.5)
```

计划中需要说明：

- 如何复用现有 baseline prediction，避免重复推理；
- 哪些 split 缺少 baseline prediction/native/alignment；
- 如何检测缺失、重复 label 和 TM-score 失败；
- 如何把 difficulty CSV 的路径和 SHA256 写入 run manifest；
- 如何替换当前 split manifest 中 `baseline_difficulty_csv: null` 和全部 difficulty 为 `unknown` 的状态。

## 五、困难样本感知训练

请设计并比较以下控制组，所有组必须保持相同的 optimizer step 数、相同 seed 集合和相同基础配置，避免把“采样策略”和“训练更久”混为一谈：

```text
A. baseline / LoRA scale 0
B. uniform LoRA（现有均匀或 cluster-balanced 采样，作为控制组）
C. hard-aware 50%：50% hard + 25% medium + 25% easy
D. hard-aware 70%：70% hard + 15% medium + 15% easy
```

如果某个 client 某一难度层样本不足，计划中必须定义明确的 fallback 和重归一化规则。

优先通过 difficulty-aware sampler/采样概率实现，不要第一步就修改 AlphaFold loss。当前 batch size/epoch sampler 如果是随机有放回抽样，请核对其真实语义，并规划记录：

```text
每个 epoch 的 hard/medium/easy draw 数
unique label 数
unique cluster 数
cluster sampling ESS
累计样本覆盖率
```

可考虑连续权重：

```text
sampling_weight_i = min(4, 1 + 3 * difficulty_weight_i)
```

但请先比较分层采样与连续权重的复杂度，选一个最小且可验证的实现作为第一版。

不要只训练 hard 样本；必须保留 medium/easy anchor，避免 catastrophic forgetting 或 easy-case regression。

## 六、分阶段实验矩阵

为控制算力，计划应采用 staged search，而不是所有参数做全笛卡尔积。

### Stage 1：固定容量，验证 hard-aware sampling 是否有信号

固定：

```text
rank = 4
alpha = 8
learning_rate = 1e-4
dropout = 0
LoRA target = 当前 structure module target
```

比较：

```text
sampling strategy: uniform、hard-aware 50%、hard-aware 70%
epoch checkpoint:  1、2、3、5
LoRA scale:         0、0.25、0.5、0.75、1.0
training seed:      至少 3 个
```

请利用每个 epoch 已保存的 checkpoint，避免为不同 epoch 重复训练。`scale=0` 必须复用 baseline 结果，不要重复执行相同推理。

请评估当前导出流程中的：

```text
base weights source = EMA
adapter weights source = raw model
```

现有文件名 `merged_ema_fp32.pt` 可能造成误解。计划中应要求输出名称和 manifest 准确区分：

```text
base_ema_adapter_raw
base_ema_adapter_ema
checkpoint_average（若采用）
```

如果比较 adapter EMA，请检查当前 `ema.decay=0.999` 在约 300～500 个 step 下是否过度偏向初始化，并提出最小、可控的 raw/EMA/checkpoint-average 对照，而不是直接假定 EMA 一定更好。

### Stage 2：只有 Stage 1 在 hard validation 上出现稳定信号后才执行

候选：

```text
learning rate: 3e-5、1e-4、3e-4
rank:          2、4、8
dropout:       0、0.05
```

不要同时全组合。先选择 sampling、epoch、scale，再搜索 learning rate，最后才搜索 rank。

### Stage 3：诊断 LoRA target 是否限制纠错能力

只有当 hard 样本主要是全局拓扑/结构域排列错误、且当前 structure-only LoRA 无法改善时，才规划 target ablation：

```text
T1: 当前 structure module
T2: 最后若干 trunk/Evoformer block + structure module
T3: 更宽的 trunk attention + structure module
```

请先检查本仓库模型模块名称、内存和 trainable parameter 数，再给出准确 target，不要在计划中杜撰模块路径。

## 七、validation 模型选择目标

当前训练日志显示 validation loader 可能为空，且现有流程直接导出最后一个 epoch。计划必须修复为非空的 cluster-disjoint validation，并明确 checkpoint selection。

主要指标：

```text
delta_tm_hard = tm_lora - tm_baseline
```

对于 personalized/local LoRA，优先在各 client 的 local validation 上选择模型；但如果 hard validation cluster 数小于计划中设定的最低值，则使用所有 client 等权的 macro validation 选择共享训练超参数，只允许 LoRA scale 或 baseline fallback 做有限个性化。

建议采用约束式选择：

```text
先要求：mean delta_tm_non_hard >= -0.005
再最大化：client-macro mean delta_tm_hard
如果没有候选满足约束或 hard gain 不稳定，则选择 scale=0，回退 baseline
```

请规划输出以下 validation 指标：

```text
hard mean delta TM
hard median delta TM
hard improved fraction
hard delta TM >= +0.05 的明显改善率
baseline TM < 0.5 -> LoRA TM >= 0.5 的 rescue rate
hard delta TM <= -0.05 的明显恶化率
easy/non-hard mean delta TM
mean delta lDDT-CA
pLDDT 与真实 per-residue lDDT 的校准指标（如现有数据足以支持）
```

不要用 mean pLDDT 代替真实结构精度。

## 八、最终统计与成功标准

所有比较必须在相同 label 上做 paired comparison。置信区间应按 sequence cluster/group bootstrap，而不是把同一 cluster 内的 chain 当作完全独立样本。

至少报告：

```text
每个 client 的 hard/non-hard 指标
client-macro 指标（每个 client 等权）
pooled 指标（仅作为补充）
3 个或更多训练 seed 的均值、标准差和分层置信区间
逐样本 delta 表
逐 cluster delta 表
最大正/负离群样本
leave-one-out 或去最大离群样本敏感性分析
```

请在看 final test 前预注册成功标准。第一版可把以下值作为待确认的建议，而不是宣称领域标准：

```text
hard-set mean delta TM >= +0.02
hard-set median delta TM > 0
cluster-bootstrap 95% CI 下界 > 0
明显改善率 > 明显恶化率
non-hard/easy mean delta TM >= -0.005
```

## 九、可选的安全路由方案

因为目标只针对 baseline 失败样本，计划中请加入一个后续可选的 baseline/LoRA gating 设计：

```text
普通或高置信度样本 -> baseline
预计为困难且 LoRA 有正收益的样本 -> hard-case LoRA expert
```

部署时不能使用 native TM-score 作为 gate。候选输入只能是推理时可获得的信息，例如：

```text
baseline pLDDT/pTM
多 seed/多模型预测分歧
recycling 稳定性
sequence length/domain 特征
与训练 cluster 的相似度或 novelty
```

gate 只能在 validation 上拟合或选择阈值，final test 只运行一次。该 gating 属于后续阶段，不要阻塞第一版 hard-aware sampler。

## 十、输出目录与不可覆盖约束

所有后续实现产生的新 split、baseline difficulty、checkpoint、prediction、metric、日志和报告必须位于：

```text
outputs/fed_lora_hardcase_v1/
```

禁止覆盖或修改：

```text
outputs/fed_lora_fp32/
outputs/fed_lora_fp32_v1/（如果实际存在）
```

建议的产物结构如下；请根据仓库现状调整并解释：

```text
outputs/fed_lora_hardcase_v1/
├── run_manifest.json
├── splits/
│   ├── split_manifest.json
│   └── client_0 ... client_4/
├── difficulty/
│   ├── baseline_difficulty.csv
│   └── difficulty_summary.csv
├── models/
│   └── client_0 ... client_4/
├── evaluation/
│   ├── validation/
│   ├── final_test/
│   ├── candidate_metrics.csv
│   ├── paired_deltas.csv
│   ├── hard_summary.csv
│   ├── difficulty_summary.csv
│   └── bootstrap_ci.csv
├── logs/
└── reports/
    └── experiment_report.md
```

临时文件也必须在该输出根目录或 `/tmp`，不能散落到数据目录。

## 十一、复现性与审计要求

`run_manifest.json` 至少记录：

```text
git commit 和 dirty status
输入路径与输出路径
cluster 文件路径和 SHA256
train/validation/test label 文件 SHA256
baseline checkpoint、weight source 和 SHA256
difficulty CSV 路径和 SHA256
hard/medium/easy 阈值
采样策略和实际采样审计
所有训练超参数
seed
epoch/global step
LoRA rank/alpha/target/scale
base 和 adapter 的 weights source
推理精度与 data_random_seed
指标脚本和 TMscore executable 路径
```

必须包含以下检查：

```text
train/validation/test cluster overlap = 0
client 间不应出现的 label/cluster overlap = 0
baseline 与 LoRA 评估 label 完全一致
scale=0 与 baseline 权重/输出的一致性检查
validation loader 非空
每个 candidate 的预测和 metric 行数一致
缺失/失败样本被显式记录，不能静默丢弃
```

## 十二、请输出的 Plan 内容

请先完成只读检查，然后给出：

1. 当前实现和目标设计之间的差距清单，引用准确文件和行号。
2. 推荐的数据流和训练/评估流程图。
3. 需要新增或修改的文件列表，每个文件说明具体职责。
4. difficulty CSV、manifest、sampling audit 和 metric CSV 的字段设计。
5. staged experiment matrix，以及最小可行版本和严谨版本各自的训练/推理任务数量与大致算力成本。
6. checkpoint、LoRA scale、raw/EMA adapter 的导出和复用方案。
7. 防止 test leakage、cluster leakage、重复推理和离群样本误导的具体措施。
8. 单元测试、集成测试和端到端 smoke test 计划。
9. 实现后的建议命令行入口及示例命令，但本阶段不要执行。
10. 明确列出需要用户决定的问题；如果仓库已有信息能够确定，请自行检查，不要把可发现的问题反问用户。

计划应优先复用现有脚本和数据结构，只在必要时增加最小代码。不要为了实验管理引入新的大型依赖。最终计划必须足够具体，使后续实现阶段可以按步骤完成而不需要重新设计。
