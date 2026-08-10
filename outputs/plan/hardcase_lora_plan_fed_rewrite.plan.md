---
name: Hard-aware FedFold / FedLoRA Plan
overview: 在 5 个 client 内使用 baseline 难度驱动的 hard-aware 本地训练，并通过同步通信轮聚合 LoRA 有效权重更新，产出可下发的 global FedFold 模型；所有客户端共享 LoRA 结构与本地训练规则，由分布式 validation 选择 global best round。当前 80 条仅作 development test，external final 保留到 baseline、local-only 与 FedLoRA 全部冻结后统一评估。全部新产物写入 outputs/fed_lora_hardcase_fed_v1/，不覆盖旧实验。
todos:
  - id: difficulty-and-splits
    content: 实现各 client 本地 baseline difficulty CSV、cluster-disjoint train/val/dev 与 external final；增加时间和同源性审计
    status: pending
  - id: hard-aware-local-training
    content: 扩展 sampler/train_openfold：所有 client 使用统一 hard-aware 比例和统一 local-epoch 语义，输出本地采样审计
    status: pending
  - id: fedlora-round-orchestration
    content: 实现同步通信轮：下发同一 global parent、各 client 固定本地训练、上传 round-end raw LoRA update、校验 lineage
    status: pending
  - id: effective-delta-aggregation
    content: 实现 FedLoRA 有效权重更新聚合；禁止聚合 local best、不同 scale 或 EMA adapter；输出 aggregation manifest
    status: pending
  - id: distributed-validation-selection
    content: 每轮聚合模型在各 client 本地 validation 评估，只汇总 sufficient statistics，由 client-macro hard 指标选择 global best round
    status: pending
  - id: privacy-audit-and-tests
    content: 明确单机模拟与真实部署边界，加入更新裁剪和 secure-aggregation 接口、泄漏审计、单测与 smoke
    status: pending
  - id: federated-mvp-runbook
    content: 准备 baseline/local-only/FedLoRA 对照、分阶段命令、算力预算与 final-test 成功标准
    status: pending
isProject: false
---

# Hard-aware FedFold / FedLoRA 实施计划

## 0. 目标和范围

最终目标不是生成 5 个可任意平均的 personalized best checkpoint，而是实现可多轮下发和聚合的 hard-aware FedLoRA/FedFold：

    public pretrained SoloSeq base + global state G_t
                           |
                    下发同一个 G_t
                           |
        5 个 client 分别执行 hard-aware LoRA 本地训练
                           |
           上传本轮结束时的 raw LoRA update
                           |
                    服务器安全聚合
                           |
                       G_(t+1)
                           |
          各 client 本地 validation 同一全局模型
                           |
              选择 global best communication round

当前范围包括：

1. 每个 client 内部的 baseline 难度标注和 hard-aware sampler。
2. local-only 对照。
3. 多轮同步 FedLoRA。
4. 分布式 validation 和 global best round。
5. external final 上一次性统一比较。
6. 可选的联邦训练后个性化；个性化权重不再回传聚合。

所有新产物写入：

    outputs/fed_lora_hardcase_fed_v1/

禁止覆盖：

    outputs/fed_lora_fp32/
    outputs/fed_lora_fp32_v1/（若存在）
    outputs/fed_lora_hardcase_v1/（若存在）

## 1. 仓库现状和差距

可复用：

- scripts/fed_lora_fp32.sh：数据准备、cluster holdout、本地 FP32 LoRA、逐 epoch checkpoint 和推理。
- train_openfold.py：LoRA 注入、weights-only 初始化、EMA reset 和 sampling audit。
- scripts/export_lora_checkpoint.py：从 raw/EMA adapter 计算有效更新并合入指定 base。
- scripts/verify_lora_roundtrip.py：scale=0、非 target 参数和导出校验。
- scripts/fedavg_aggregate.py：完整参数字典加权平均，仅作为数值实现参考。
- TM-score、lDDT-Cα、pLDDT 和 split 工具。

必须修复：

1. 现有流程只从 public base 启动 5 个独立 local run，没有 communication round 和 global-parent lineage。
2. 当前没有 global validation round selection。
3. 旧 fedavg_aggregate.py 读取训练 checkpoint 时优先 EMA；短训 EMA decay=0.999 明显滞后，不能直接作为 MVP 的 round-end 聚合源。
4. 不允许平均各 client 独立选择的 epoch、scale、raw/EMA 或不同 rank 的 checkpoint。
5. 分别平均 LoRA A/B 不严格等于平均有效 BA 更新。
6. difficulty 全为 unknown，采样不关注 baseline 失败样本。
7. 当前 80 条 test 已分析，只能作为 development test。
8. 单机输出必须区分 client-private 和 server-visible。
9. 新一年结构不自动等于模型没见过同源序列，需增加时间和同源性审计。

实现前重新核对 outputs/fed_lora_fp32 与 outputs/fed_lora_fp32_v1 的实际存在状态，manifest 记录解析后的绝对路径。

## 2. 数据、难度和防泄漏

### 2.1 每个 client 本地三路划分

保持已有 5 client 数据归属，不跨 client 移动样本。每个 client 内按 MMseqs 30% sequence cluster 划分：

    client_k/private/
    ├── train
    ├── validation
    └── development_test

要求：

- 30% 是 sequence identity threshold，不是 30% holdout。
- 在原 train clusters 中切约 20% clusters 为 validation。
- 当前 80 条固定为 development test，不参与 sampling、epoch、round、scale 或 aggregator 选择。
- 从 leftover quality-kept 且与全部已用 cluster 无交的池构建 external final。
- external final 在所有方法和配置冻结前不可运行。
- external 不足时采用 outer/inner group cross-validation，不能随机 chain split。

强制断言：

    train、validation、development_test 的 cluster 两两无交
    external_final 与全部开发 cluster 无交
    client 间不应出现的 label/cluster overlap 为 0

### 2.2 client-local baseline difficulty

每个 client 使用相同 public baseline、FP32、checkpoint source 和 inference seed，在本地输出：

    label,client,split,cluster_id,release_date,
    baseline_tm,baseline_lddt_ca,baseline_plddt,
    sequence_length,difficulty,difficulty_weight,tm_status

定义：

    hard: baseline_tm < 0.5
    medium: 0.5 <= baseline_tm < 0.8
    easy: baseline_tm >= 0.8
    difficulty_weight = max(0, (0.5 - baseline_tm) / 0.5)

训练/validation 的 difficulty CSV 只保存在 client 私有目录。真实服务器只能接收聚合后的 band counts 和指标 sufficient statistics，不能收到 label、native 路径、PDB 路径或逐样本 delta。

单机模拟允许把 client-private 文件放到统一 RUN_ROOT，但服务器代码不得读取其逐样本内容。

### 2.3 时间与同源性审计

新增 novelty_audit.csv：

    label,client,release_date,after_base_cutoff,
    nearest_base_train_identity,nearest_base_cluster,
    exact_sequence_seen,novelty_status

缺少 baseline 训练集映射时写 unknown，不得仅凭年份宣称模型完全没见过。报告表述限定为：结构发布时间晚于 baseline cutoff，并完成当前可获得范围内的 sequence/cluster novelty audit。

## 3. 联邦模型不变量

同一个 federated run 内所有 client 必须共享：

    global parent SHA
    config preset
    LoRA target
    rank
    alpha
    dropout
    optimizer
    learning rate（MVP）
    effective batch size
    local epoch/draw 语义
    hard-aware target ratio
    precision
    round id
    aggregation method

允许不同：

    本地数据量
    hard/medium/easy 实际数量
    相同 local epoch 下的实际 optimizer step 数
    本地抽样 seed 和运行时间

禁止进入同一次聚合：

    不同 rank/target/alpha
    不同 global parent 或 round id
    不同 adapter source
    不同 inference LoRA scale
    client 自行选择的 local best

MVP 统一配置：

    rank = 4
    alpha = 8
    dropout = 0
    learning_rate = 1e-4
    precision = FP32
    target = structure_module.ipa,
             structure_module.transition,
             structure_module.bb_update
    local_epochs_per_round = 1
    train_epoch_len_k = client_k 当前 train label 数
    aggregation adapter source = raw model
    aggregation scale = 1.0

每轮 train_epoch_len_k 等于 n_k，表示每个 client 约一次本地数据曝光。不同 client 实际 step 可以不同，但轮内不能用 local validation 提前停止。

## 4. Hard-aware 本地训练

同一个 federated run 中所有 client 使用相同 sampling arm：

    U: uniform / 当前 cluster-balanced 控制组
    H50: 50% hard + 25% medium + 25% easy
    H70: 70% hard + 15% medium + 15% easy

MVP 先比较 U 与 H70。优先实现分层有放回采样，不修改 AlphaFold loss。

每轮 audit：

    round_id,client_id,global_parent_sha,draw_count,
    hard/medium/easy draws,realized_ratios,
    unique labels/clusters,unique hard labels/clusters,
    cluster sampling ESS,cumulative coverage,fallback_applied

统一 fallback：

1. 某层为空：按目标比例把配额重分配到其余非空层。
2. hard 为空：退回 uniform 并报告 no_hard_local_data。
3. 实际比例偏离超过 10%：记录 warning，不静默修改全局配置。

### 每轮初始化

每轮所有 client 必须从同一个 global_round_t.pt 开始：

1. 加载并校验 global SHA 和 round lineage。
2. 注入统一 LoRA。
3. optimizer state 在每轮重置。
4. EMA 从 global parent 同步重置，但 EMA adapter 不用于 MVP 聚合。
5. 固定训练 1 local epoch。
6. 上传 round-end raw LoRA update，不上传 local best。

实现前验证 train_openfold.py 从服务器 pure parameter dict 做 weights-only 初始化时应使用的 init_weights_source，不能把 pure dict 错当成包含 ema.params 的训练 checkpoint。

## 5. FedLoRA 聚合

### 5.1 MVP：有效权重更新聚合

client k 的 raw LoRA 有效更新：

    delta_W_k = (alpha / rank) * B_k * A_k

服务器：

    delta_W_global = sum_k weight_k * delta_W_k
    G_(t+1) = G_t + delta_W_global

要求：

- 非 LoRA target 参数与 G_t bitwise identical。
- target 参数只应用一次聚合更新。
- 不聚合 optimizer state、EMA adapter 或 client-specific scale。
- 所有 update 必须引用同一个 parent SHA。

本方案不需要分别平均 A/B，也不要求每轮 SVD：服务器把聚合更新吸收到 global full-model target weights，下一轮从新的 global model 重新注入 LoRA。global 多轮累计更新可以高于单一 rank，但客户端每轮上传仍是 rank-r 更新。

### 5.2 新增主聚合器

新增 scripts/fedlora_aggregate.py：

1. 读取 round parent pure weights。
2. 从各 client raw checkpoint 只提取 LoRA A/B。
3. 校验 rank/alpha/target/parent SHA/round id。
4. 计算并聚合有效 delta_W。
5. 可选执行 client update norm clipping。
6. 写 global child pure weights。
7. 写 aggregation manifest。
8. 验证与“各 raw adapter scale1 merge 到同一 base 后，对完整模型 FedAvg”数值等价。

旧 scripts/fedavg_aggregate.py 只作参考，不能直接接收 DeepSpeed client checkpoint，因为它默认优先 EMA。应增加显式 weights-source 或误用警告。

### 5.3 聚合权重

MVP 使用：

    weight_k = n_k / sum_j n_j

训练按 sample count 聚合；模型选择和公平性报告按 client macro 等权，不得混淆。manifest 同时记录 n_k、draw_count 和实际采样比例。

可选消融：

- equal-client aggregation。
- direct A/B FedAvg。
- FedAdam/FedYogi。
- local step 异质时的 FedNova。
- 将累计 delta 相对 public base 做 rank-r SVD。

## 6. 分布式 validation 与 global best

每轮得到 global_round_(t+1).pt 后下发至全部 client。每个 client 在本地 validation 上评估，不上传逐样本数据，只返回：

    client_id,round_id,global_model_sha,
    hard_count,hard_cluster_count,sum_delta_tm_hard,
    hard_improved_count,hard_rescue_count,
    hard_large_improve_count,hard_large_degrade_count,
    nonhard_count,sum_delta_tm_nonhard,
    sum_delta_lddt,metric_status

单机模拟可以保留 private paired CSV 做审计，但 server selection 代码只能读取 summary schema。

服务器模型选择：

    primary = 5-client equal-weight macro mean delta_TM_hard
    constraint = macro mean delta_TM_nonhard >= -0.005

规则：

1. 过滤不满足 non-hard constraint 的 round。
2. 在剩余 round 最大化 primary。
3. hard validation cluster 太少时输出 minimum-count warning 和 cluster bootstrap。
4. 如果所有 round 无稳定 hard gain，则 global_best 选择 round_0，即 public baseline。
5. 最终只有一个 global_best.pt。

严禁：

    每个 client 选不同 round 后再聚合
    用 development_test 或 external_final 选通信轮
    上传 local best.pt

MVP 全局训练和聚合固定 scale=1。若搜索 shrinkage，只能在 macro validation 上选择一个全局共享值；每客户端不同 scale 仅允许在最终 personalization 中使用。

## 7. 实验阶段

### Stage 0：smoke

    clients = 2
    rounds = 1
    train_epoch_len = 4
    local_epochs = 1
    sampling = uniform
    rank = 4

验证 parent/child lineage、有效更新、非 target identity、下一轮初始化和 validation summary。

### Stage 1A：单 seed MVP

    clients = 5
    seed = 42
    rounds = 5
    local_epochs_per_round = 1
    arms = FedLoRA-U,FedLoRA-H70
    rank = 4
    alpha = 8
    lr = 1e-4
    aggregation = sample-weighted effective-delta FedAvg

任务数：2 arms × 5 rounds × 5 clients = 50 个 local one-epoch 任务。按现有约 4–6 min/local epoch，训练约 3.5–5 GPU 小时，另加 validation 推理。

### Stage 1B：严谨复现

Stage 1A 有信号后：

    seeds = 42,43,44
    arms = U,H70
    rounds = 5

共 150 个 local one-epoch 任务，约 10–15 GPU 小时，另加 validation 推理。

### 最终对照

    M0 public pretrained baseline
    M1 local-only uniform LoRA
    M2 local-only hard-aware LoRA
    M3 uniform FedLoRA
    M4 hard-aware FedLoRA
    M5 hard-aware FedLoRA + optional personalization

local-only 只作基线，不参与 global aggregation；其总本地曝光量与 federated rounds × local_epochs 对齐。

### Stage 2

Stage 1 稳定后按顺序单因素搜索共享联邦参数：

1. H50/H70。
2. local epochs 1/2。
3. learning rate。
4. rank。
5. aggregation。
6. target ablation。

每个候选必须作为完整 federated run 运行，不能把 local-only 最佳参数直接当成 federated 最佳参数。

## 8. External final 和统计

external final 只有在以下项目冻结后才能运行：

    split,difficulty threshold,sampling arm,
    rank/alpha/target/LR,local epochs,
    global-best rule,aggregation method/weights,
    global scale,personalization policy

最终报告：

- 每 client hard/non-hard paired delta。
- client-macro 与 pooled。
- hard mean/median delta TM。
- improved fraction。
- delta TM >= 0.05 明显改善率。
- baseline TM < 0.5 到 FedLoRA TM >= 0.5 的 rescue rate。
- delta TM <= -0.05 明显恶化率。
- easy retention 和 delta lDDT-Cα。
- cluster bootstrap 95% CI。
- 多 seed 方差。
- 最大离群与 leave-one-cluster-out。
- baseline/local-only/FedLoRA 统一比较。

预注册建议：

    hard client-macro mean delta TM >= +0.02
    hard median delta TM > 0
    cluster-bootstrap 95% CI 下界 > 0
    明显改善率 > 明显恶化率
    non-hard client-macro mean delta TM >= -0.005

## 9. 隐私、安全和威胁模型

FedLoRA 只保证原始序列/native 不直接上传，不自动提供形式化隐私。LoRA update 仍可能被 membership inference、属性推断或更新反演。

manifest 声明：

    simulation / trusted_server / honest_but_curious_server
    malicious_client 是否在范围内

单机 MVP：

- client-private 与 server-visible 分开。
- server orchestration 不读取 client label/native/paired CSV。
- 上传前记录 update norm，并支持 clipping。
- 日志不输出序列/native。
- 聚合接口设计为可替换 secure aggregation。

真实部署最低要求：

- TLS。
- secure aggregation。
- client update clipping。
- 最低参与 client 数和 dropout 处理。
- access control、审计日志。
- 不上传逐样本 validation 指标。

client-level differential privacy 作为后续隐私—效用实验，不在小数据 MVP 中默认启用。

## 10. 新增和修改文件

新增：

| 文件 | 职责 |
|---|---|
| scripts/build_baseline_difficulty.py | client-local difficulty 与 failures |
| scripts/build_hardcase_splits.py | train/val/dev/external 与 novelty audit |
| scripts/fedlora_aggregate.py | raw A/B 到有效 delta，再到 global child |
| scripts/fed_lora_hardcase_fed.sh | 同步轮编排入口 |
| scripts/evaluate_hardcase_metrics.py | private paired metrics 和 server summary |
| scripts/select_global_round.py | constrained global best |
| scripts/compare_fedfold_methods.py | final method comparison |
| tests/test_fedlora_aggregation.py | 聚合数值、target identity、lineage |
| tests/test_hardcase_sampling.py | difficulty、ratio、fallback、audit |
| tests/test_federated_round_smoke.py | 两 client 端到端 smoke |

最小修改：

| 文件 | 改动 |
|---|---|
| openfold/data/data_modules.py | optional hard-aware sampler 和 audit |
| train_openfold.py | difficulty/sampling CLI、pure-global 初始化 |
| scripts/export_lora_checkpoint.py | round/parent/source sidecar 和准确命名 |
| scripts/fedavg_aggregate.py | 非主聚合器；增加 weights-source 或 EMA 误用警告 |
| scripts/build_fed_test_set.py | hard<0.5、medium<0.8 |

优先复用现有函数，不引入大型联邦框架。

## 11. 输出结构

    outputs/fed_lora_hardcase_fed_v1/
    ├── run_manifest.json
    ├── server/
    │   ├── config.json
    │   ├── rounds/
    │   │   ├── round_000/global_model.pt
    │   │   ├── round_001/
    │   │   │   ├── aggregation.json
    │   │   │   ├── global_model.pt
    │   │   │   └── validation_summary.json
    │   │   └── ...
    │   └── global_best.pt
    ├── clients/client_0..4/private/
    │   ├── splits/
    │   ├── difficulty/
    │   ├── rounds/
    │   └── validation/
    ├── evaluation/
    │   ├── development/
    │   ├── external_final/
    │   ├── method_comparison.csv
    │   ├── hard_summary.csv
    │   └── bootstrap_ci.csv
    ├── logs/
    └── reports/
        ├── federated_run_report.md
        ├── privacy_audit.md
        └── novelty_audit.md

真实部署时 client private 目录分别位于各客户端机器；单机模拟仅为了复现。

## 12. Manifest 和 lineage

run manifest 至少记录：

    git commit / dirty status
    input/output roots
    cluster/base checkpoint hashes
    difficulty schema
    client split hashes
    shared LoRA config
    sampling/local epoch semantics
    rounds/aggregation/update clipping
    seeds/precision/inference seed
    final-test lock state

每轮 aggregation manifest：

    round_id,parent_global_sha,participating_clients,
    client update SHA,n_k,draw_count,
    update norms before/after clipping,
    normalized weights,target keys,rank/alpha,
    raw adapter source,aggregation scale,
    child_global_sha,non_target_max_abs_diff,status

服务器拒绝：

    parent/round/config 不一致
    参数 key/shape 不一致
    EMA 或 scaled adapter
    NaN/Inf
    client 数不足
    duplicate update

## 13. 测试和验收

单元测试：

- difficulty 边界和 sampling fallback。
- synthetic A/B 的有效 delta。
- sample-weighted aggregation。
- 非 target bitwise identity。
- parent/round/config mismatch。
- raw/EMA/scale 混用失败。
- global-round selection constraint。
- SHA/manifest 稳定。

数值等价：

    有效 delta 聚合
    约等于
    同一 base 上 raw scale1 merged models 的完整 FedAvg

并构造反例证明 average(A/B) 不等于 average(BA)。

集成测试：

- 两 client、两轮。
- 同一 parent。
- round1 child 可初始化 round2。
- server summary 不含 label/native。
- best 只由 validation 选择。
- final lock 在配置冻结前阻止执行。

Smoke 验收：

- parent SHA 相同。
- aggregation 成功。
- non-target 不变。
- global model 可推理。
- validation 非空。
- metrics 行数一致。
- 仅写新 RUN_ROOT。

## 14. 建议命令（实现后）

    bash scripts/fed_lora_hardcase_fed.sh prepare
    bash scripts/fed_lora_hardcase_fed.sh smoke

    SEEDS=42 ARMS=uniform,hard_aware_70 ROUNDS=5 LOCAL_EPOCHS=1 \
      bash scripts/fed_lora_hardcase_fed.sh train

    bash scripts/fed_lora_hardcase_fed.sh validate
    bash scripts/fed_lora_hardcase_fed.sh select_global
    bash scripts/fed_lora_hardcase_fed.sh evaluate_dev

    bash scripts/fed_lora_hardcase_fed.sh lock_final
    bash scripts/fed_lora_hardcase_fed.sh evaluate_final
    bash scripts/fed_lora_hardcase_fed.sh report

Plan 阶段不运行训练。实现时先核对现有 CLI 和实际数据路径。

## 15. 可选个性化

唯一 global_best.pt 产生后，每个 client 可从它出发做少量个性化：

    global_best
    ├── client0 personalized adapter
    ├── client1 personalized adapter
    └── ...

个性化阶段可以本地选择 epoch/scale，产物只留在 client 私有目录，禁止再上传。报告分开标注 global FedLoRA 和 personalized FedLoRA。

## 16. 已拍板事项

- 主目标是 hard-aware FedLoRA/FedFold，不是 5 个独立 local best。
- 所有 client 使用统一 LoRA 结构、训练规则和 sampling arm。
- 每轮固定 1 local epoch，不允许上传不同 local best。
- MVP 聚合 raw LoRA 的有效 delta，不聚合 EMA 或不同 scale。
- 服务器按 sample count 聚合，按 client-macro validation 选 global best round。
- 当前 80 条只作 development test。
- external final 保留到全部方法与超参数冻结后一次评估。
- local-only 只作对照，不参与聚合。
- personalization 仅在 global training 后产生，不回传。
- 单机输出区分 server-visible 与 client-private。
- 真实隐私需要 secure aggregation，FL 本身不等于形式化隐私。
- 不引入大型联邦框架，优先最小扩展现有 OpenFold/LoRA/aggregation 代码。
