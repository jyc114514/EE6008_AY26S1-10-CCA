> Archived study summary for `CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z` (source snapshot `20261015`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V5 冻结 state/video 水平—形状组合实验

**状态：complete（开发实验；负结果按预注册固定矩阵收口）。**

V5 生成并验收了 18/18 个固定组合预测 bundle、90 个 task views；没有新增训练或校准。主候选 H_SV 的 Safe-FD 与 V 由公式恒等继承，但它没有降低 Position MAE：五任务等权的 H_SV MAE 为 **0.159740 ± 0.000788**，高于 V、BIAS_VIDEO、S 与 BIAS_STATE。H_SV 的 seam error 为 **1.566279**，明显高于 V 的 **0.795761**。因此本轮结论是否定这条固定水平—形状组合在当前开发集上的 MAE 改善假设。

## 绝对结果

指标采用 V2 接受的 primary-train `target_sigma` 与 `safe_fd_scale`。MAE 和 Safe-FD 均为标准化误差；不表示物理世界精度。±值是三个固定模型 seed 间的样本 SD，Persistence 是共享参考，没有 seed SD。

| Method | Position MAE, mean ± seed SD | Safe-FD L1, mean ± seed SD | Seam Safe-FD L1, mean ± seed SD |
|---|---:|---:|---:|
| S | 0.157555 ± 0.001035 | 0.830807 ± 0.003094 | 0.766943 ± 0.005265 |
| V | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| BIAS_STATE | 0.157301 ± 0.001047 | 0.830807 ± 0.003094 | 0.819232 ± 0.006349 |
| BIAS_VIDEO | 0.158066 ± 0.001482 | 0.819754 ± 0.002994 | 0.847943 ± 0.021419 |
| H_SV | 0.159740 ± 0.000788 | 0.819754 ± 0.002994 | 1.566279 ± 0.026016 |
| H_AV | 0.161138 ± 0.001176 | 0.819754 ± 0.002994 | 1.457222 ± 0.034795 |
| H_SC | 0.159946 ± 0.000371 | 0.829653 ± 0.001748 | 1.414292 ± 0.024350 |
| H_PV | 0.222943 ± 0.000424 | 0.819754 ± 0.002994 | 4.476856 ± 0.105647 |
| MIX50 | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |
| PERSISTENCE | 0.222262 (shared) | 0.854193 (shared) | 0.802417 (shared) |

## H_SV 配对对比

统一定义 `gain = L(reference) − L(candidate)`，正值才表示 H_SV 误差更低。置信区间为固定公式/固定模型下、5,000 次 within-task source-group paired bootstrap 的条件 95% 区间；它不涵盖跨训练 seed 总体推断或未观测 session 相关性。

| 对比（reference − H_SV） | Position MAE gain, mean ± seed SD | Position MAE conditional 95% CI | Safe-FD gain | Seam gain |
|---|---:|---:|---:|---:|
| LEVEL_TRANSFER_GAIN | -0.001401 ± 0.001512 | [-0.003848, +0.001403] | +0.000000 | -0.770518 |
| HYBRID_VS_OPT_STATE | -0.002184 ± 0.000284 | [-0.003305, -0.001133] | +0.011053 | -0.799336 |
| HYBRID_VS_BIAS_STATE | -0.002438 ± 0.000294 | [-0.003584, -0.001390] | +0.011053 | -0.747047 |
| HYBRID_VS_BIAS_VIDEO | -0.001673 ± 0.001515 | [-0.004143, +0.001160] | +0.000000 | -0.718335 |
| MATCHED_SHAPE_GAIN | +0.000206 ± 0.000514 | [-0.000997, +0.001303] | +0.009898 | -0.151986 |
| OPT_VS_COMMON_LEVEL_GAIN | +0.001399 ± 0.000486 | [+0.000074, +0.002807] | -0.000000 | -0.109057 |
| HYBRID_VS_ANCHOR_LEVEL | +0.063203 ± 0.000831 | [+0.051319, +0.075433] | +0.000000 | +2.910578 |
| HYBRID_VS_MIX50 | -0.004268 ± 0.000694 | [-0.005733, -0.002775] | -0.000781 | -0.801056 |

H_SV 对 V 的 Position MAE gain 为 −0.001401，条件区间包含 0；对 BIAS_VIDEO 为 −0.001673，区间也包含 0。对 S 与 BIAS_STATE 的点估计更差，条件区间均低于 0。普通 MIX50 的 MAE 比 H_SV 低 0.004268；H_SV 不优于这一固定平均参考。

逐 task、跨三个 seed 的配对 MAE gain 如下。负值表示 H_SV 更差。

| Task | Dev episodes | Anchors | V − H_SV | S − H_SV | BIAS_VIDEO − H_SV |
|---:|---:|---:|---:|---:|---:|
| 119 | 12 | 48 | -0.000923 | -0.003867 | -0.001054 |
| 94 | 12 | 48 | -0.002686 | -0.003849 | -0.002749 |
| 90 | 11 | 41 | +0.000017 | -0.000006 | -0.000953 |
| 141 | 5 | 20 | -0.003493 | -0.003201 | -0.003570 |
| 230 | 7 | 28 | +0.000079 | +0.000002 | -0.000041 |

任务 141 和 94 的差值最大；119 也略差于 V，但 seed 间有方向变化；90 与 230 接近 0。Leave-one-seed-out 后，H_SV 对 V 的五任务 MAE gain 三个结果仍全为负（−0.002166、−0.001384、−0.000654）。在 15 个 task×seed 的 leave-one-task-out 汇总中，H_SV 对 S 和 BIAS_STATE 全部为负，对 V 只有 4/15 为正，对 BIAS_VIDEO 2/15 为正。所有 15 个 task×seed 绝对结果、120 条配对效应、seed LOO、task LOO 和条件区间见结果 CSV。

## FD、水平与形状诊断

五个组合的 29 步内部差分恒等式通过。最大 native 差为 **2.22×10⁻¹⁶**，最大按 accepted Safe-FD scale 归一后的差为 **3.26×10⁻¹⁴**，低于锁定的 `atol=1e-12, rtol=1e-10`。因此 H_SV 的内部 FD 与 V 完全重合是公式直接继承的性质，不能算作新学到的 FD 优势；相对 S 的 FD 变化也只是 V 相对 S 的既有差异。

seam 没有不变性。H_SV 相对 V 的 seam error 增加 **0.770518**；相对 S 增加 **0.799336**，相对 BIAS_VIDEO 增加 **0.718335**。因此只检查未来内部 FD 会遗漏明显的 anchor→first-step 接缝代价。

train-scale L2 诊断中，S 的 level MSE 是 **0.067307**，低于 V 的 **0.071411**；A 为 **0.070900**。H_SV 的 level MSE **0.067307** 与 S 一致，demeaned-shape MSE **0.049747** 与 V 一致，二者之和 0.117055。逐 anchor 恒等式最大误差为 `6.38e-16`。这说明预测 state 水平在此 L2 诊断上更接近目标，但不推出 L1 MAE 一定下降；真实 H_SV MAE 仍高于上述主要来源。H_SV 优于 H_AV 的 MAE gain 为 +0.001399，支持优化后的 S recipe 相对 A common recipe 有水平优势；H_SV 对 H_SC 的差异很小且条件区间跨 0，不能将其归因为纯视觉动态的因果效应。H_PV 明显更差（MAE 0.222943），说明当前 command anchor 不是足够的预测水平参考。

## 计算代价

V5 没有使用 GPU、没有 watcher job、没有新 encoder extraction 或模型 fit。H_SV 需要 S 与 V 两个冻结模型。V2 接受推理记录显示，每个 185-window bundle 的 S 平均 elapsed 为 **0.692481 s**，V 为 **0.758281 s**；两项历史 job elapsed 的和平均为 **1.450761 child-seconds**，三个 seed 的和为 1.957953、1.324986、1.069345 s。它是既有分开 job 的 elapsed 合计，不是 V5 测得的串行/并行端到端延迟。V5 在 CPU 上生成并序列化全部 18 个组合 bundle 用时约 **0.338 s**。组合可由冻结输出构造，不代表已集成到机器人执行接口，也不与单模型推理等成本。

## 暴露与限制

- 预测生成阶段 target access 为 0；18 个非空 bundle seal 通过后才打开 dev target。final/heldout payload access 为 **0**。
- 实际 target archive 共加载 **4 次**，覆盖同一 185 个唯一 dev windows，共计 740 个 window-row loads。三次失败/恢复尝试已保留：第一次载入 185 行后在前缀角色解析处停止；第二次完成 3,330 条损失计算并保存，随后发现 V4 primary 表没有 seam 列；第三次载入后在恢复 CSV seed 类型核验处停止。最终一次读取复用已保存损失表，完成 L2 和轨迹数据；没有重复 3,330 条派生损失计算。失败记录见 `qa/SCORER_ATTEMPT_0*_FAILURE.json`。
- V4 回传中的历史计数为 `cumulative_dev_future_action_row_reads=35,523`；其 grain 与 V5 的 bundle-window loss evaluation 尚未核实，故保留原值并单列，不做加总。V5 完成 3,330 个 primary window-loss evaluations，另有 2,775 个 L2 诊断 evaluations。dev target archive 总加载 4 次、累计加载 740 条 receiver-window rows，unique windows 为 185；图表没有再读 archive。
- V4 per-window source 文件被读取三次做预检/恢复，累计解析 24,975 行；三次各筛选 3,885 个接受参考行，所有 B_D/C_D donor-map 行均被排除，实际独特参考为 3,885 行。source MAE/FD 与 V4 primary 表 210 个可比单元对账最大差为 0；V4 primary 表没有 seam 列，因此 seam 直接沿用已接受 per-window 表。
- 结果只适用于这个已重复暴露的开发集；source-group CI 条件于固定模型和公式，不覆盖 seed 总体不确定性及未知 session 依赖。数值 command 的物理单位未解决。

## 验收产物

- 公式/对比锁：[FORMULA_LOCK.json](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/protocol/FORMULA_LOCK.json>)、[CONTRAST_LOCK.json](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/protocol/CONTRAST_LOCK.json>)；锁定 hash `fdc1a4b971ad60ef69299ab4b499b8c28c6724ae28782a62444b3b1ffee48662`。预测 seal：[PREDICTION_SEAL.json](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/qa/PREDICTION_SEAL.json>)，hash `38dbed76a3cb9dbc06c05457469ab5f4ff31dcc1d77f1a4c0a24613900c65ebe`。
- 主结果：[PRIMARY_TASK_SEED.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/PRIMARY_TASK_SEED.csv>)、[PAIRED_EFFECTS.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/PAIRED_EFFECTS.csv>)、[PAIRED_GROUP_CI.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/PAIRED_GROUP_CI.csv>)、[SEED_LOO.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/SEED_LOO.csv>)、[TASK_LOO.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/TASK_LOO.csv>)。
- 诊断与记录：[LEVEL_SHAPE_L2.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/LEVEL_SHAPE_L2.csv>)、[RESOURCE_LEDGER.csv](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/results/RESOURCE_LEDGER.csv>)、[FINAL_ACCEPTANCE.json](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/qa/FINAL_ACCEPTANCE.json>)、[OUTCOME_EXPOSURE.json](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/inventory/OUTCOME_EXPOSURE.json>)。
- 图：[双指标](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/figures/V5_DUAL_METRIC.png>)、[按 task 配对 gain](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/figures/V5_PAIRED_GAINS_BY_TASK.png>)、[L2 level/shape](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/figures/V5_LEVEL_SHAPE_L2.png>)、[轨迹](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/figures/V5_TRAJECTORY_EXAMPLES.png>). 自动/本轮视觉检查通过；人工图审仍为 pending，不是数字完成门。

校验清单：[CHECKSUMS.sha256](<server-only://project-artifacts/reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z/CHECKSUMS.sha256>)。

V5 固定矩阵已收口；本轮未追加 alpha、路由、seed 或训练。

校验历史：早先的 238 项 manifest（SHA-256 `a98f6cf183c81f5e1ad673a890875ca7a78edb1c2130d7d50632de192b1d0323`）及其 PASS receipt 已归档到 `archive/`。之后对回传、figure QA 和 acceptance 的一致性修正使旧 manifest 不再覆盖当前正文；本次重新封存的 manifest/receipt 覆盖最终文件。
