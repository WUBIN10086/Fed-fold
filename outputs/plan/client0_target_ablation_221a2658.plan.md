---
name: Client0 Target Ablation
overview: 为 client0 实现不聚合的 LoRA target 分级实验：T0/T1/T3/T4 均作为必须执行的容量诊断，不允许前序 target 失败而阻断更深上游 target；overfit8 使用最多 200 optimizer steps 排除训练不足，T0–T4 再以固定 epoch-5、scale-1.0 做统一 validation 比较。Gate 只决定多 seed、development、其他 client 与 FedLoRA 的推广，不决定基本消融是否执行。现有 split、baseline difficulty 与旧结果保持只读。
todos:
  - id: verify-targets
    content: 实现 T0–T5B target 注册表、模块/参数计数核验与隔离输出路径
    status: pending
  - id: build-overfit8
    content: 生成 client0 八个 cluster-disjoint hard 训练子集、资产和防泄漏 manifest
    status: pending
  - id: run-overfit-diagnostic
    content: 实现 T0/T1/T3/T4 overfit8 分段训练、同集推理、容量诊断与 T5/positive-control 失败回退
    status: pending
  - id: run-target-validation
    content: 实现固定 epoch5/scale1 的 T0–T4 完整 validation 矩阵；gate 仅生成推广候选，不阻断 target 执行
    status: pending
  - id: confirm-seeds-dev
    content: 实现 seed43/44 稳定性确认和唯一入选 target 的 development 评估
    status: pending
  - id: target-reports-tests
    content: 实现跨 target 汇总报告、单元测试与端到端 smoke
    status: pending
isProject: false
---

# Client0 LoRA Target 分级实验

## 实验不变量与产物隔离

- 复用现有 [`client_0` split](outputs/fed_lora_hardcase_fed_v1/clients/client_0/private/splits/split_manifest.json)：train 57（26 hard、37 clusters）、validation 10（3 hard clusters）、development 13，不重新划分。
- 固定 `uniform / rank=4 / alpha=8 / lr=1e-4 / dropout=0 / epochs=5 / FP32 / seed=42`；主比较固定 **epoch 5、LoRA scale 1.0**，不做 per-target checkpoint/scale 搜索。
- overfit8 是单独的容量诊断，不受主实验 `epochs=5` 限制：最多 200 optimizer steps、warmup 5 steps，在 40/80/120/200 steps 做同集推理；达到容量标准可提前停止该 target。
- 新产物只写入 `outputs/fed_lora_hardcase_fed_v1/clients/client_0/private/target_ablation_v1/`，汇总写入 `evaluation/client0_target_ablation_v1/`；不覆盖已有 `local_only/`、旧 split 或 baseline。
- T3/T4 使用用户给出的 comma-prefix target。当前 [`module_matches_target`](openfold/utils/lora.py) L180–196 会逐 prefix 匹配，这些字符串有效；执行前仍强制核验模块数和 trainable 参数量。

## Target 矩阵

- T0：`structure_module.ipa,structure_module.transition,structure_module.bb_update` → 10 / 32,072
- T1：`structure_module` → 18 / 43,904
- T2：`structure_module,evoformer.linear` → 19 / 46,464
- T3：`structure_module,evoformer.linear,evoformer.blocks.47` → 56 / 103,104
- T4：`structure_module,evoformer.linear,evoformer.blocks.44,evoformer.blocks.45,evoformer.blocks.46,evoformer.blocks.47` → 167 / 273,024
- T5A（失败回退，不进入主矩阵）：`input_embedder` → 5 / 19,292，用于入口映射诊断。
- T5B（可选失败回退）：`structure_module,input_embedder` → 23 / 63,196，用于判断输入映射与结构头联合适配是否具备训练集纠偏能力。

T0/T1 是 structure-only；T2 已进入 Evoformer 输出接口；T3/T4 分别覆盖最后 1/4 个 Evoformer block。T3 失败不能作为不运行 T4 的理由。T5 只在 T0/T1/T3/T4 经过充分容量检查仍全部失败时运行，不参与正常 target 排名。

## 实施

### 1. Target 与路径安全

- 新增 [`scripts/verify_lora_target_matrix.py`](scripts/verify_lora_target_matrix.py)：CPU 初始化 `seq_model_esm1b_ptm`，调用 `configure_lora` / `parameter_counts`，逐项断言 T0–T5B 的 target、模块数、参数量；零匹配或计数偏差立即失败。T5 只验证注册，不代表正常流程一定运行。
- 扩展 [`scripts/fed_lora_hardcase_fed.sh`](scripts/fed_lora_hardcase_fed.sh)：加入 `TARGET_SLUG`/`LOCAL_OUTPUT_NAMESPACE`，使 target 路径进入 `target_ablation_v1/<Tn>/...`；保留默认空 namespace，从而不破坏已有 local-only 路径。
- `local_run.json`、candidate/selection/report 全部记录 target 字符串、模块数、参数量、split SHA、checkpoint SHA 和固定主评估口径。

### 2. 8-cluster hard 容量检查（不得提前阻断上游 target）

- 新增 [`scripts/build_client0_overfit_subset.py`](scripts/build_client0_overfit_subset.py)：从 client0 train 中按 `baseline_tm` 升序，每个 cluster 仅取 1 条，确定性选择 8 个最困难 cluster：`9fei_A(c50), 9wye_A(c772), 9b3e_A(c207), 9ftf_A(c343), 9i2o_A(c496), 9t2d_A(c167), 9xpd_A(c752), 9u78_A(c935)`。
- 生成 `overfit8_labels.txt`、仅含这 8 条的 chain cache、FASTA/alignment/mmCIF 软链接和 manifest；断言 8 labels、8 clusters，且全部 `baseline_tm<0.5`、属于 train、不与 val/dev cluster 相交。
- T0、T1、T3、T4 都必须在 overfit8 上运行；T0/T1/T3 的失败状态不得阻止 T4。T2 被 T3 包含且只多一个输出映射，本阶段可省略，仍必须参加后续 full-train validation。
- 每个 target 最多训练 25 epochs（8 draws/epoch，共 200 optimizer steps），`lr_warmup_steps=5`。按 5/10/15/25 epochs 对应的 40/80/120/200 steps 分段断点续跑，每段结束后在同一 8 条上推理并生成 paired 指标。
- 容量通过标准固定为：`mean ΔTM >= 0.01`、至少 1 条 `ΔTM>=0.01`、TM 上升样本数至少 4/8；lDDT 与 rescue 同时报告但不单独否决容量判断。首次通过后记录 `first_pass_step`，允许停止该 target 的后续 overfit 分段。
- 40 steps 未通过只表示“当前步数未通过”，不能标记 target 最终失败；只有完成 200 steps 仍未通过才记为 `capacity_failed`。
- 无论 T0/T1 是否通过，T3/T4 都执行。至少一个 T0/T1/T3/T4 通过后，进入完整 T0–T4 full-train validation 矩阵。
- 若 T0/T1/T3/T4 在 200 steps 后全部失败，不直接宣告 LoRA 无效，也不悄悄进入 full-train sweep；先运行失败回退诊断：
  - 核验每个 checkpoint 的 LoRA A/B norm、有效 `ΔW` norm、合并后的 changed target keys 和 non-target identity；
  - 核验 train loss 是否随 step 下降，并区分“loss 不下降”和“loss 下降但 TM 不变”；
  - 运行 T5A；资源允许时再运行 T5B；
  - 增加一个 positive control：在 overfit8 上对完整 `structure_module` 做选择性非 LoRA 解冻，验证数据、loss、反向传播和 TM 评估链路是否具备记忆能力。
- 回退分支必须给出互斥结论和后续动作：
  - T5A/T5B 中至少一个通过：将通过的 T5 作为独立 fallback target 做固定 epoch-5/scale-1.0 full-train validation；只有通过同一推广 gate 后才进入 seed 确认。
  - T5A/T5B 失败但非 LoRA positive control 通过：判为“当前 LoRA 参数化/容量不足”，停止 LoRA full-train 推广，不把 positive control 混入 FedLoRA target 排名。
  - T5A/T5B 与 positive control 全部失败且 loss 不下降：优先判为优化、反向传播或数据链路问题。
  - loss 明显下降但 TM 始终不变：优先判为训练 loss 与 TM 目标错配。
- 报告必须按上述证据区分失败类型，不得笼统归因为“structure target 无效”或“LoRA 无效”。

### 3. Full-train validation 完整矩阵与推广 Gate

- 至少一个主矩阵 target 通过容量检查后，T0/T1/T2/T3/T4 各自在完整 57 条 train 上独立训练 5 epochs；即使某个 target 在 overfit8 未通过，也保留其固定预算 validation 结果作为容量与泛化对照。只以 epoch-5、scale-1.0 作为预注册主比较，保证唯一实验变量为 target。
- 可以额外导出 epoch1–4 的诊断指标用于绘制轨迹，但不得据此为不同 target 选择不同 epoch，也不得改变 epoch-5 主结论。
- 扩展 [`scripts/evaluate_hardcase_metrics.py`](scripts/evaluate_hardcase_metrics.py)：增加 `ΔTM>=0.01` 计数/比例、TM 上升数、all-validation mean ΔlDDT、baseline/model/delta pLDDT（仅诊断）。
- 新增 [`scripts/evaluate_target_gate.py`](scripts/evaluate_target_gate.py)，按预注册条件判定：
  - hard validation mean ΔTM ≥ +0.005；
  - hard median ΔTM > 0；
  - non-hard mean ΔTM ≥ −0.005；
  - all-validation mean ΔlDDT-Cα ≥ −0.002；
  - 至少 1 个 hard 样本 ΔTM ≥ +0.01。
- 上述五条是“推广 gate”，不是“target 执行 gate”。T0–T4 全部完成后分别标记 `promotion_pass/fail`；删除“T3 必须通过且比 T1 高 0.002 才解锁 T4”的条件。
- 报告同时给出 T3−T1、T4−T3 的 paired 差异，用于回答“最后 1/4 个 Evoformer block 是否带来增益”，但不以单个 3-hard validation 的 `+0.002` 差异作为执行前置条件。
- Validation 仅用于 target 晋级；development 不参与 target、epoch 或 scale 选择。

### 4. Seed 确认与 development

- seed 42 所有通过推广 gate 的 target 都进入候选集；候选超过 2 个时，按 hard mean ΔTM、hard median、all-validation ΔlDDT、较少 trainable 参数依次排序，最多保留前 2 个运行 seed 43、44，避免只确认单 seed winner 而产生 winner's curse。
- 三 seed 确认标准：3-seed hard mean ΔTM 的均值 ≥ +0.005、至少 2/3 seed 的 hard mean > 0，并且每个 seed 均满足 non-hard ≥ −0.005 与 all-validation ΔlDDT ≥ −0.002。
- 在 seed 42/43/44 全部结束后再选唯一 target；按 3-seed hard mean、hard median、all-validation ΔlDDT、参数量依次排序。只有通过 seed 确认后，才对该唯一 target 在 client0 development 上运行一次；若未通过，报告“client0 target ablation 无稳定泛化信号”，不扩展 client2，也不启动 FedLoRA。

### 5. 编排与报告

- 新增 [`scripts/run_client0_target_ablation.py`](scripts/run_client0_target_ablation.py)，提供 `verify / build-overfit / overfit / diagnose-capacity / validate / confirm-seeds / evaluate-dev / report` 阶段；每阶段检查产物 SHA 和阶段状态。`overfit` 必须调度完 T0/T1/T3/T4 或明确记录提前通过，不能因前序 target 失败跳过后序 target；推广 gate 只阻止 `confirm-seeds/evaluate-dev`。
- 新增 [`scripts/summarize_client0_target_ablation.py`](scripts/summarize_client0_target_ablation.py)，输出：
  - `overfit8_target_summary.csv`；
  - `capacity_diagnostics.csv`（40/80/120/200 steps、loss、A/B norm、ΔW norm、changed keys、first_pass_step）；
  - `validation_target_summary.csv`；
  - `seed_confirmation.csv`；
  - `paired_deltas.csv`；
  - `target_ablation_report.md`，明确区分 train memorization、validation generalization 与 development confirmation。
- 现有 [`select_local_checkpoint.py`](scripts/select_local_checkpoint.py) 不用于主 target 比较，避免不同 target 选择不同 epoch 造成混杂；仅复用其指标读取辅助函数。

## 测试与验收

- `tests/test_lora_target_matrix.py`：T0–T5B 路径匹配、模块/参数计数、错误 target fail-fast。
- `tests/test_client0_overfit_subset.py`：8 条、8 clusters、全 hard、train-only、与 val/dev 零泄漏、固定 seed/排序可复现。
- `tests/test_target_ablation_paths.py`：T0–T4 输出互不冲突，已有 local-only 路径保持不变。
- `tests/test_target_gate.py`：容量检查必须覆盖 T0/T1/T3/T4；40-step 失败不会终止、200-step 才可判失败；T3 失败仍调度 T4；五条 validation 推广条件不阻止 T0–T4 执行；三 seed 确认失败时阻止 development/FedLoRA。
- `tests/test_capacity_fallback.py`：四个主 target 全部 capacity_failed 时触发 norm/loss 诊断、T5 与 positive-control 路由，并产生明确失败类型。
- Smoke：单 target、2 个 overfit labels、1 epoch，仅验证 checkpoint、推理、paired metrics、gate 和报告链路；不运行正式实验。

## 实现后的命令入口

```bash
python scripts/run_client0_target_ablation.py verify
python scripts/run_client0_target_ablation.py build-overfit
python scripts/run_client0_target_ablation.py overfit
python scripts/run_client0_target_ablation.py diagnose-capacity
python scripts/run_client0_target_ablation.py validate
python scripts/run_client0_target_ablation.py confirm-seeds
python scripts/run_client0_target_ablation.py evaluate-dev
python scripts/run_client0_target_ablation.py report
```

正式执行严格按上述顺序。T0–T4 是预注册诊断矩阵，不由前序性能 gate 裁剪；机器可读 gate 只控制多 seed、development、其他 client 和 FedLoRA。T5/positive control 仅在主矩阵容量检查全部失败时作为诊断回退，其他 client 和 FedLoRA 均不在本轮实施范围内。
