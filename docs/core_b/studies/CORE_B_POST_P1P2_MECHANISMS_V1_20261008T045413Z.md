> Archived study summary for `CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z` (source snapshot `20261008`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# P1/P2 机制研究与 N1 query 接续 — 最终回传

**状态：完成。** NEXT_ROOT：`server-only://project-artifacts/reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z`；N1_ROOT：`server-only://project-artifacts/reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z`。未重建输出根，未新增 H60、B_REP、B6、D5 或 query sweep。`FINAL_TEST_ACCESS=0`。

## P1/P2 sealed reference

P1 的三个 H30 R3D primary/dense bundles 和训练均已接受。H30 episode-equal Full28 MAE：Persistence 0.223135，S-HIST 0.194640，F-POOL-D 0.194216，ALT-POOL-R3D18 0.194952。ALT−S-HIST 为 +0.000313（95% CI [−0.000256,+0.000863]，跨零）；ALT−F-POOL-D 为 +0.000736（CI [+0.000093,+0.001383]），低于锁定显示阈值 0.0022313517。当前协议没有建立稳定的 R3D pooled 增量。

H60 gate 为 `NOT_PASS`；未启动 H60。`B_REP=false`，新 B6=false。P2 支持门通过，但 seed 交互方向混合；D4 三个 ALT DeltaU 均为负；这不支持稳定的视觉互补结论。完整锁定参考见 [P1/P2 交接](server-only://CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z/RETURN_TO_CHATGPT.zh-CN.md)。

## N0：固定 checkpoint 的离线视觉内容敏感性

**接受：** 6/6 REAL strict replays 和 12/12 ZERO/PERM interventions；2322 dev anchors、603 source groups。新训练 0、encoder extraction 0、optimizer steps 0。评分阶段在接受后读取 train/dev target arrays；未读 final/heldout。CI 与 p-value 均未计算。

| 模型 | Seed | REAL | HIST | ZERO | ΔL ZERO−REAL | PERM | ΔL PERM−REAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-POOL-D | 20260950 | 0.193412 | 0.193424 | 0.199618 | 0.006206 | 0.203065 | 0.009653 |
| F-POOL-D | 20260951 | 0.194672 | 0.195527 | 0.198469 | 0.003797 | 0.202029 | 0.007358 |
| F-POOL-D | 20260952 | 0.194565 | 0.194968 | 0.197731 | 0.003166 | 0.200157 | 0.005593 |
| ALT-POOL-R3D18 | 20260950 | 0.194332 | 0.193424 | 0.196057 | 0.001725 | 0.200356 | 0.006024 |
| ALT-POOL-R3D18 | 20260951 | 0.195665 | 0.195527 | 0.196734 | 0.001069 | 0.198923 | 0.003259 |
| ALT-POOL-R3D18 | 20260952 | 0.194860 | 0.194968 | 0.196341 | 0.001481 | 0.198245 | 0.003385 |

**固定三 seed 均值的函数响应：**

| 模型 | Probe | D_out | D_level | D_shape |
|---|---|---:|---:|---:|
| F-POOL-D | ZERO | 0.021450 | 0.020880 | 0.012438 |
| F-POOL-D | PERM | 0.032351 | 0.031502 | 0.018297 |
| ALT-POOL-R3D18 | ZERO | 0.013844 | 0.013432 | 0.008307 |
| ALT-POOL-R3D18 | PERM | 0.023179 | 0.022521 | 0.013508 |

两模型对 ZERO 与跨组 PERM 都有可测响应，D_level 大于 D_shape，且 REAL MAE 在全部 seed 下低于两种 probe。它只说明当前固定模型对指定扰动敏感；ZERO/PERM 是离开自然联合分布的输入，不能解释为视频的因果或独立信息价值。D_level/D_shape 是描述量，不是可加分解。

## N1：既有 H30 full-data query study 接续

**动作：** 接续原 N1 root；6 个 corrected-scheduler Q0 controls 复用，6 个 Hadamard Q1 cells 接受，旧 control repair 0。Formal training `PASS_6_OF_6`，每个新 cell 11,120 updates。Train/dev predictions `12/12` / `12/12`，总计 `24/24`；fresh-process replay `PASS_6_OF_6_FRESH_PROCESS_REPLAY`。

### Development co-primary cell summaries

| System | Query | Full28 position MAE (mean ± 3-seed SD) | safe-FD error (mean ± 3-seed SD) | TV_pred/TV_GT | Active-step direction accuracy |
|---|---|---:|---:|---:|---:|
| TS11-D | old | 0.199408 ± 0.009118 | 0.961396 ± 0.013074 | 0.092927 | 0.455500 |
| TS11-D | had | 0.198287 ± 0.003661 | 0.956054 ± 0.002089 | 0.123143 | 0.561571 |
| STATE-CTRL | old | 0.195021 ± 0.000839 | 0.953524 ± 0.001697 | 0.133039 | 0.565038 |
| STATE-CTRL | had | 0.197823 ± 0.001080 | 0.952543 ± 0.003312 | 0.124557 | 0.561985 |

### Per-seed paired point estimates

| Metric | Seed | G_TS | G_STATE | V_old | V_Had | I |
|---|---:|---:|---:|---:|---:|---:|
| Full28 position | 20261001 | -0.008257 | -0.003951 | 0.000907 | -0.003399 | -0.004306 |
| Full28 position | 20261002 | 0.014806 | -0.003611 | -0.015717 | 0.002699 | 0.018417 |
| Full28 position | 20261003 | -0.003185 | -0.000847 | 0.001649 | -0.000689 | -0.002338 |
| safe-FD | 20261001 | -0.003783 | 0.001337 | -0.000198 | -0.005318 | -0.005121 |
| safe-FD | 20261002 | 0.019887 | -0.004186 | -0.024798 | -0.000724 | 0.024074 |
| safe-FD | 20261003 | -0.000077 | 0.005790 | 0.001379 | -0.004488 | -0.005867 |

### Fixed-model paired source-group bootstrap (10,000 draws, 603 dev groups)

| Metric | Estimand | Mean over fixed seeds | 3-seed SD | 95% group CI | Crossed seed×group CI (low precision) |
|---|---|---:|---:|---:|---:|
| Full28 position | G_TS | 0.001121 | 0.012119 | [0.000375,0.001843] | [-0.008058,0.014255] |
| Full28 position | G_STATE | -0.002803 | 0.001703 | [-0.003886,-0.001750] | [-0.004611,-0.000700] |
| Full28 position | V_old | -0.004387 | 0.009819 | [-0.005050,-0.003710] | [-0.015238,0.001889] |
| Full28 position | V_Had | -0.000463 | 0.003055 | [-0.001119,0.000183] | [-0.003295,0.002388] |
| Full28 position | I | 0.003924 | 0.012589 | [0.003028,0.004821] | [-0.004446,0.017810] |
| safe-FD | G_TS | 0.005342 | 0.012732 | [0.004097,0.006554] | [-0.003910,0.018672] |
| safe-FD | G_STATE | 0.000980 | 0.004998 | [-0.000199,0.002189] | [-0.003775,0.005739] |
| safe-FD | V_old | -0.007872 | 0.014679 | [-0.009319,-0.006410] | [-0.023698,0.001711] |
| safe-FD | V_Had | -0.003510 | 0.002449 | [-0.004496,-0.002538] | [-0.006009,-0.000893] |
| safe-FD | I | 0.004362 | 0.017075 | [0.002801,0.005882] | [-0.007004,0.022875] |

`G=old−Had` 为正表示 Hadamard 在对应系统下误差更低；`V=STATE−TS` 为正表示 TS11-D 绝对误差更低；`I=G_TS−G_STATE` 是系统交互。两项 co-primary 的 I 点估计和固定模型 group CI 为正，但 crossed seed×group sensitivity CI 均跨零；V_Had 两项点估计均为负，position CI 跨零，safe-FD 则 STATE-CTRL 更低。因此不能声称 Hadamard fusion 有绝对性能优势，也不能把 I 当作视频因果效应。

N1 scores are on a previously exposed development population; no model selection or independent confirmation was claimed. P-values were not computed. The locked query changes temporal basis/spacing/smoothness/conditioning together, so the result does not isolate rank. TV ratios below 1 indicate less aggregate predicted motion than ground truth; direction accuracy alone is not evidence of better dynamics.

### Execution deviation

Pre-output resolution requested CPU FP32 Q1 inference to match Q0. Accepted Q0 sidecars are CPU; all 12 new Q1 train/dev bundles report CUDA:0 on physical GPU 0. Watcher receipts record CUDA_VISIBLE_DEVICES=0 and UUID GPU-3c8b646d-d1dc-1bc4-6e82-d96422d9f36b, while Q1 prediction sidecars omit the visible-device field. Accepted Q1 predictions and exact fresh-process replays were retained rather than rerun. The paired Q0/Q1 effect therefore includes a backend-class difference as a limitation.

## QA、资源与非干扰

N0 visual QA：14 个 case + 3 summary figures，AI pass / human pending。N1：11 PNG + 11 PDF、11 captions。AMENDMENT_006 修复 case 图标题/图例裁切；AMENDMENT_007 把 dev 主图轴范围改为包含 mean±seed SD，把 ECDF 改为每个 source group 跨 3 seed 先平均后作图（每模型 603 点），并为 12 个锁定 dev case × 2 个 action index × 30 步新增 720 行确定性 anchor-hold persistence overlay。未改训练、预测、主分析表或 estimands，也未读取 target arrays。11 张 PNG 原像素预览通过；人工审阅 pending，submission_ready=false。
N1 root `CHECKSUMS.sha256` 已覆盖并验证 1,245 个普通文件（清单自身排除）；N1 final acceptance 记录 SHA-256 清单。

N0 watcher lease 用量 0.024337 GPU-h，N1 1.128549 GPU-h，合计 1.152886 GPU-h。N0 forward timer 0.003996 GPU-h 单独记录、不与 lease time 重复相加。物理设备：GPU0/1/2/3；全部 N1 watcher receipts 显示 lease 已释放，N1 controller/session 已结束。

磁盘快照（AMENDMENT_007 与 N1 checksum 已完成、NEXT checksum 前；含既有 N1 文件和临时 staging，不是增量）：NEXT_ROOT 0.153 GiB / 164,297,121 bytes；N1_ROOT 3.039 GiB / 3,263,407,733 bytes。

旧根非干扰：N1 启动快照登记的 317/317 old-source bytes 均保持不变，包含 MECH 178/178 和 SCHED 139/139（SCHED 有一项预先存在的 expected-hash 漂移，但 observed bytes 从启动快照起未变）。另一个 NEXT prelock registry 的 31/32 计数差异是 N1_ROOT 内可变 `inventory/SOURCE_ACCEPTANCE.json` 元数据记录，不是 sealed old-root input。N0 reused checkpoint/REAL 输入 12/12、N1 Q0 prediction 12/12 与 checkpoint 6/6 均 SHA 匹配；没有旧根写入，也未对全部旧根递归重哈希。

没有新增 H60、B_REP、B6、D5；`FINAL_TEST_ACCESS=0`。项目下一步无 GPU 或 scorer 工作。

## 文件

- 综合回传：`RETURN_TO_CHATGPT.zh-CN.md`
- 最终接受：`qa/FINAL_ACCEPTANCE.json`
- 旧根非干扰：`qa/OLD_ROOT_NONINTERFERENCE.json`
- 资源摘要：`coordination/RESOURCE_SUMMARY.json`
- N1 报告：`../CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z/RETURN_TO_CHATGPT.zh-CN.md`
- N1 artifact manifest：`n1_refs/N1_ARTIFACT_MANIFEST.csv`
- 最终 checksum manifest：`CHECKSUMS.sha256`（覆盖 NEXT_ROOT 普通文件，清单自身除外）
