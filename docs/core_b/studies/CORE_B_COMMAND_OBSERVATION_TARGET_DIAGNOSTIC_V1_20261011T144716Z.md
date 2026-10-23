> Archived study summary for `CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z` (source snapshot `20261011`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B：OBS43 记录状态转移与未来 command 信息诊断

## 结论

这项新任务预测 raw Parquet 中分开的 43D observation 记录，保留旧的 28D command benchmark，不替代旧结果。全部 canonical primary anchors 均满足新任务的数据和时钟资格：train 17,757 行、2,712 source groups；dev 2,322 行、603 groups；本轮没有剔除 anchor。所有五种对照使用同一 support。

同一个线性一步模型做 H30 开放 rollout 时，没有超过 observation persistence。把实际未来 command 喂给同一模型后，Full43 位置和 safe-FD 都优于 hold-command rollout，但仍没有超过 persistence。收益集中于前两组源字段；第三组较弱。Teacher forcing 优于局部 persistence，略逊于局部 constant velocity；它是 GT-state reset 后的一步诊断，不是 H30 预测。

这些结果只描述本数据、当前线性模型与记录时间对齐。未来 command 的收益表示该模型可利用记录中的未来 command（其中可能包含控制器/操作者反馈），不证明物理因果或独立 intent 信息；teacher 与 rollout 的差异提示递推误差/增长可能相关，但不能确认唯一根因。它们也不能证明未来 observation 或 command 理论上不可预测，不能解释旧 command28 Ridge 失败的唯一原因。

## 字段、时钟与输入可用性

Producer 和 Parquet 列核验通过：s 来自 `observation.arm_joints` 14 + `observation.hand_joints` 14 + `observation.leg_joints` 15；u 来自另一个 `action` 28 列。这里是分开的记录字段，不表示统计独立；没有给 28 个 command 与 43 个 observation 强配对。关节解剖绑定与物理单位未确认，只报告来源切片及 numeric native values。

raw schema 没有 issue/availability timestamp。u[0] 的 prefix 可用性沿用已接受 benchmark 的 known-anchor command 合同，属于 recorded-clock 假设，不是对真实执行时序的独立证明。u[0] 从 raw action[anchor_frame] 补回，未把旧 H30 target[0] 错当成 u[0]。oracle 使用 u[0..29]，不读取未来 observation；teacher 在每步使用记录 s[j]、v[j] 和 u[j]。

名义步长为 1/30 秒，clock tolerance 为 1e-5 秒。622,449 个相邻间隔已检查，最大偏差 7.1208e-6 秒；H30 实际平均时距 0.99999997 秒。H={1,3,6,10,15,30} 对应约 0.03333、0.10000、0.20000、0.33333、0.50000、1.00000 秒。未做插值、padding 或未来时间条件化。

## 模型与训练

每个 eligible train anchor 只使用 s[0]→s[1] 一条训练转移。输入为 [s[0], v[0], u[0]] 共 114 维，输出为按 observation train-only sigma 标准化的 next velocity，共 43 维。输入 scaler、sigma_obs 和 safe-FD_obs 均按 episode-equal 规则只在 fit groups 计算；系数目标仍只有每 anchor 一步。常数/近零 observation 输出通道为 0，常数输入列为 0。

复用 Ridge source-group 三折 assignment。固定 λ 网格 {0.001, 0.01, 0.1, 1.0} 以 episode-equal OOF 一步位置 MAE 选择一个全局 λ；取 λ=0.1（OOF standardized MAE 0.016745）。执行 12 个正式 CV fit + 1 次全 train refit，模型参数含截距共 4,945。另有 1 次真实 512-anchor train-only smoke fit；总监督 CPU fit 数为 14，其中正式 CV/refit 为 13。

Full-train augmented 86×86 state transition 的谱半径为 1.000448，仅 1 个 eigenvalue 的模超过 1；||A^30||₂=12.22。它是模型递推增长信号，不等于真实机器人动力学不稳定。

## Dev H30 四个 forecast views

MAE 使用 train-fit observation scales；native 值是来源记录数值的均值误差，单位未知。Persistence skill 定义为 1 − model loss / persistence loss。TV 为 episode-equal 总变差比 TVpred/TVGT。H1 的 future-internal FD 为 NA。

| View | Position MAE (σ; native) | Position skill | Safe-FD error (native) | FD skill | TVpred/TVGT | Active direction accuracy |
|---|---:|---:|---:|---:|---:|---:|
| OBS persistence | 0.16470 (0.03557) | 0.0% | 1.06845 (0.003209) | 0.0% | 0.000 | 0.000 |
| OBS constant velocity | 0.35892 (0.05024) | −117.9% | 1.55841 (0.004438) | −45.9% | 0.986 | 0.549 |
| ARX hold command | 0.19832 (0.03472) | −20.4% | 1.23854 (0.003421) | −15.9% | 0.385 | 0.566 |
| ARX oracle command | 0.17230 (0.02696) | −4.6% | 1.13935 (0.002949) | −6.6% | 0.600 | 0.648 |

Horizon position skill for HOLD declines from +16.0% at H1 to +11.0% at H3 and +2.6% at H6, then is negative from H10 (−4.8%) through H30 (−20.4%). ORACLE follows the same decline and reaches −4.6% at H30. The OOF-selected one-step objective therefore does not imply a good long rollout.

同一 exposed dev 的 paired source-group bootstrap（seed 20261011、10,000 draws；条件于固定模型，不计算 p 值）给出：

- G_HOLD = L_persistence − L_hold：position −0.03362，95% CI [−0.03681, −0.03047]；safe-FD −0.17010，CI [−0.17731, −0.16299]。负值表示 HOLD 劣于 persistence。
- G_DRIVER = L_hold − L_oracle：position +0.02602，95% CI [+0.02496, +0.02709]；safe-FD +0.09920，CI [+0.09516, +0.10331]。正值表示实际未来 command 在该 ARX 下改善该损失。
- 配套 G_CV = L_constant-velocity − L_hold：position +0.16060，95% CI [+0.13697, +0.18624]；safe-FD +0.31987，CI [+0.27118, +0.37071]。这表示本数据上 anchor constant-velocity rollout 误差高于 HOLD；仍按支持性对照解释。

Full43 H30 有 2,360,114 个 active GT direction components（正向 1,185,157；负向 1,174,957），603 个 source groups 均同时有正负方向支持。Persistence 没有活动方向；ORACLE direction accuracy 为 0.648。更高 TV 或 direction accuracy 不代表整体轨迹误差更优。

## Raw source blocks 的 H30 persistence-relative skill

单元格为“position skill / safe-FD skill”，百分数；每一组分别用该组自己的 train-fit scale。

| Source block | OBS persistence | Constant velocity | ARX HOLD | ARX ORACLE |
|---|---:|---:|---:|---:|
| Full43 | 0.0 / 0.0 | −117.9 / −45.9 | −20.4 / −15.9 | −4.6 / −6.6 |
| `arm_joints` 切片 | 0.0 / 0.0 | +17.6 / −5.0 | +19.8 / +7.3 | +32.8 / +22.6 |
| `hand_joints` 切片 | 0.0 / 0.0 | −98.7 / −58.1 | −12.8 / −15.2 | +23.7 / +0.4 |
| `leg_joints` 切片 | 0.0 / 0.0 | −232.0 / −69.9 | −55.9 / −35.5 | −55.6 / −36.1 |

这些名称只是原始列切片别名，不是已验证的人体关节语义。ORACLE 的 H30 收益由 arm/hand 块贡献，leg 块仍差于 persistence。

## Teacher-forced local 诊断

跨 30 个单步 transition、episode-equal 的 Full43 position MAE 为：teacher 0.01715σ，LOCAL persistence 0.02020σ，LOCAL constant velocity 0.01613σ。局部 safe increment error 分别为 0.847、1.070、0.772。teacher 相对局部 persistence 较好，但整体略逊于局部 CV；arm/hand 改善更清楚，leg 与局部 persistence 接近。teacher 每步从记录真值状态重置，不能把这些点拼成一个 forecast 或排在 H30 主表。

## 资源、顺序与审计

- 仅 CPU：taskset CPU 0–7，OpenBLAS 8 threads；无 GPU/CUDA/NVML 命令或 API，未运行新神经网络、H60、D5、final/held-out。
- 已计时的 7 个计算/验收阶段合计 175.86 process CPU 秒、155.42 墙钟秒；最高已记录 RSS 2,232,428 KiB（2.13 GiB）。完整 raw extractor 在写 summary 时发生 `dev_in`/`dv_in` 名称错误；所有数据数组/清单已写完，恢复验收通过且未重读 raw。这个 extractor 没保存 CPU/RSS receipt；session wall 约 31–32 秒，精确 CPU 时间与该阶段 peak RSS 为 UNKNOWN。
- 抽取 recovery 验证 3,315 个 raw Parquet 文件读取前后 size/mtime 未变；source code、canonical input、fold、query manifest 和旧 checksum manifest 5 项来源哈希全部与已接受值一致。旧 Ridge/query 根写入 0。
- 四个 forecast arrays/hash 在任何 dev score 读取前 PASS。teacher 首次 privileged dev-state label 读取：2026-10-11 15:04:03.024379Z；首次独立 scorer dev label 读取：15:05:49.873186Z。最初 raw dev label 在首次 extraction 中已构造，但精确 first-read 时间没有写入 receipt，记录为 UNKNOWN，文件 mtime 上界为 14:58:18.105377Z。病例图首次读取 label 也未单独记录精确时刻，时间界限为 15:07:10.165683Z 至 15:07:14.386715Z；后续 Type42 重导出读取记录时间 15:15:56.779925Z。所有这些图表访问均发生在评分/模型选择之后，没有再选模型或指标。
- Dev split 此前已多次暴露；CI 只条件于固定模型与当前 dev，不能当作独立确认。无 p 值；human review pending，submission_ready=false。

## 交付索引

- 主报告：`FINAL_REPORT.zh-CN.md`
- 回传：`returns/RETURN_TO_CHATGPT.zh-CN.md` 与 `.json`
- 数据/时钟：`data/ELIGIBILITY_SUMMARY.json`、`data/TARGET_TIME_GRID.csv`、`protocol/FIELD_MAP.json`、`protocol/CLOCK_AND_INPUT_ROLES.json`
- 模型/fit：`models/MODEL_LOCK.json`、`results/LAMBDA_SELECTION.csv`、`results/CV_FIT_RECEIPTS.csv`、`models/CV_SCALER_RECEIPTS.json`
- 主指标/CI：`results/FORECAST_METRICS_BY_HORIZON.csv`、`results/EPISODE_FORECAST_METRICS.csv`、`results/PAIRED_GROUP_BOOTSTRAP.csv`、`results/TEACHER_LOCAL_METRICS.csv`
- arrays、图表和验收：`predictions/`、`figures/FIGURE_MANIFEST.json`、`qa/FINAL_ACCEPTANCE.json`
- checksum：`CHECKSUMS.sha256`（checksum-last）。
