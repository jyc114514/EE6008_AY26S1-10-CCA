> Archived study summary for `CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z` (source snapshot `20261008`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B H30 query-geometry 完整数据实验回传

状态：本轮完成。实验根目录：`server-only://project-artifacts/reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z`。旧根非干扰重哈希通过；human review 仍待人工签核，`submission_ready=false`。

## 接续与执行
- 接续同一 RUN_ROOT；复用 6 个 corrected-scheduler Q0 controls，训练 6 个 Q1 Hadamard H30 单元，旧 control 修复 0 个。
- 3 个固定 seed：20261001/20261002/20261003；每个 Q1 单元 40 epochs、11,120 updates、890,780 parameters。
- Train/dev bundle 分别 12/12、12/12，共 24/24；fresh-process replay 6/6。train 17,757 anchors；此前已暴露的 dev 2,322 anchors、603 个 source groups。
- 前一轮请求放宽 watcher 被自动审批拒绝；本次保留原共享 watcher、动态可见 GPU 池和最多 2 个 project GPU children。共享 watcher 未改，本地放宽副本未用于执行；child overlap 实测最大值为 2。

## Dev 主指标
误差越低越好。`seed SD` 是三个固定 seed 组均值的样本标准差。

| 系统 | Query | Full28 standardized position MAE | safe-FD normalized error | TV_pred/TV_GT | active-step direction accuracy |
|---|---|---:|---:|---:|---:|
| TS11-D | Q0 old | 0.199408 ± 0.009118 | 0.961396 ± 0.013074 | 0.092927 | 0.455500 |
| TS11-D | Q1 Hadamard | 0.198287 ± 0.003661 | 0.956054 ± 0.002089 | 0.123143 | 0.561571 |
| STATE-CTRL | Q0 old | 0.195021 ± 0.000839 | 0.953524 ± 0.001697 | 0.133039 | 0.565038 |
| STATE-CTRL | Q1 Hadamard | 0.197823 ± 0.001080 | 0.952543 ± 0.003312 | 0.124557 | 0.561985 |

## Paired query and system contrasts
正 `G` 表示 Q1 Hadamard 误差较低；正 `V` 表示 TS11-D 误差低于 STATE-CTRL。`I=G_TS−G_STATE=V_Had−V_old` 是 query×system 相对交互。区间为固定模型、当前已暴露 dev 上的 source-group bootstrap 95% percentile CI，10,000 draws；无 p-values。

| Metric | Estimand | Mean over fixed seeds | Seed SD | 95% group CI | Crossed seed×group CI (low precision) |
|---|---|---:|---:|---:|---:|
| position_standardized_mae | G_TS | 0.001121 | 0.012119 | [0.000375, 0.001843] | [-0.008058, 0.014255] |
| position_standardized_mae | G_STATE | -0.002803 | 0.001703 | [-0.003886, -0.001750] | [-0.004611, -0.000700] |
| position_standardized_mae | V_old | -0.004387 | 0.009819 | [-0.005050, -0.003710] | [-0.015238, 0.001889] |
| position_standardized_mae | V_Had | -0.000463 | 0.003055 | [-0.001119, 0.000183] | [-0.003295, 0.002388] |
| position_standardized_mae | I | 0.003924 | 0.012589 | [0.003028, 0.004821] | [-0.004446, 0.017810] |
| safe_fd_normalized | G_TS | 0.005342 | 0.012732 | [0.004097, 0.006554] | [-0.003910, 0.018672] |
| safe_fd_normalized | G_STATE | 0.000980 | 0.004998 | [-0.000199, 0.002189] | [-0.003775, 0.005739] |
| safe_fd_normalized | V_old | -0.007872 | 0.014679 | [-0.009319, -0.006410] | [-0.023698, 0.001711] |
| safe_fd_normalized | V_Had | -0.003510 | 0.002449 | [-0.004496, -0.002538] | [-0.006009, -0.000893] |
| safe_fd_normalized | I | 0.004362 | 0.017075 | [0.002801, 0.005882] | [-0.007004, 0.022875] |

Per-seed paired estimates:

| Metric | Seed | G_TS | G_STATE | V_old | V_Had | I |
|---|---:|---:|---:|---:|---:|---:|
| position_standardized_mae | 20261001 | -0.008257 | -0.003951 | 0.000907 | -0.003399 | -0.004306 |
| position_standardized_mae | 20261002 | 0.014806 | -0.003611 | -0.015717 | 0.002699 | 0.018417 |
| position_standardized_mae | 20261003 | -0.003185 | -0.000847 | 0.001649 | -0.000689 | -0.002338 |
| safe_fd_normalized | 20261001 | -0.003783 | 0.001337 | -0.000198 | -0.005318 | -0.005121 |
| safe_fd_normalized | 20261002 | 0.019887 | -0.004186 | -0.024798 | -0.000724 | 0.024074 |
| safe_fd_normalized | 20261003 | -0.000077 | 0.005790 | 0.001379 | -0.004488 | -0.005867 |

Interpretation: TS11-D 的 position `G_TS` 均值为 +0.001121、seed SD 为 0.012119；safe-FD `G_TS` 为 +0.005342、seed SD 为 0.012732。STATE-CTRL 的 position `G_STATE` 为 −0.002803，safe-FD 为 +0.000980。`I` 两指标均值为正，但 crossed seed×group 低精度区间都跨 0。`V_Had` 为负：safe-FD CI 全负；position CI 跨 0。因此该结果显示相对 query×system 交互，未建立 TS11-D 的绝对优势，也不支持 video 因果解释。
TV ratios 四个 dev cells 均低于 1（0.092927, 0.123143, 0.133039, 0.124557），聚合预测运动幅值仍偏低。Persistence anchor-hold 是确定性未训练参考：dev position MAE 0.223135，safe-FD 0.976491。

## GPU-hours、UUID 与失败记录
Watcher lease-held 总时间 4062.777 s = 1.128549 GPU-hours；child active runtime 4062.556 s = 1.128488 GPU-hours。21 个 watcher receipts：16 complete、2 failed child、3 launching/no-child；所有 lease 已释放。正式 Q1 训练 child runtime 1.089957 GPU-hours。实际最大并发 2。物理设备/UUID：GPU 0 `GPU-3c8b646d-d1dc-1bc4-6e82-d96422d9f36b`；GPU 2 `GPU-00465dc6-e96e-67f1-285a-84ed61565ac2`；GPU 3 `GPU-b9c07d60-9a04-0693-af83-6976ba6d4dfb`。美元费用不可得：没有账单费率或权威计费记录。
- 两个 Stage0 初次 child 在 optimizer 前因 manifest `saved_artifacts` 键报错，累计分配 16.177 s，0 optimizer steps；修复后两 canaries 通过。三次 formal watcher fallback 校验失败均在 child spawn 前，随后仅移除 runtime queue 的非绑定 GPU preference，未改共享 watcher。
- 分析器第一次因 `byseed` 顺序错误失败，Amendment 004 仅移动已有汇总块；attempt 004 全数据评分通过并提交 37 个文件。两个重复/并行分析尝试无第二份结果提交，退出码未知，均保留 reconciliation。图件的缺失 SEEDS、病例布局、主轴/ECDF/缺失 persistence 问题均有独立 amendment 和归档记录。完整 failure ledger：`coordination/POST_LOCK_FAILURES.jsonl`。

## 来源与边界
`inventory/SOURCE_ACCEPTANCE.json`：MECH 178/178 match；SCHED 138/139 match。唯一 SCHED mismatch 是预先记录的旧 `generate_figures.py`，不属于本轮训练/推理/scoring 依赖。启动快照中的 317 个注册来源随后全部重新哈希且 byte-identical：MECH 178/178、SCHED 139/139（包括其预先存在的 mismatch 观察值），无 old-root writes。
Dev 曾暴露且用于早期研究；本轮没有新 dev model selection、p-values 或 independent confirmation。Q0 推理为 CPU，Q1 为 CUDA logical `cuda:0`/physical GPU 0；Q1 watcher 收据记录 `CUDA_VISIBLE_DEVICES=0`，Q1 prediction sidecar 未写该字段。查询同时改变 basis、frequency、smoothness 与 conditioning，不能把交互归因于 rank。H60、D5、更多 seeds/模型均未运行。`FINAL_TEST_ACCESS=0`，`HELDOUT_ACCESS=0`。

## 图件与交付文件
11 PNG + 11 PDF，11 captions/alt-text sections；四张 case figure 现在含 GT、Q0、Q1、anchor-hold persistence。`figure_data/PERSISTENCE_ANCHOR_HOLD.csv` 有 720 行；ECDF 为每模型 603 个 source-group 均值（每组先跨 3 seeds 平均）。所有 11 PNG 已由 agent 以原像素维度预览检查；human review 仍 pending，`submission_ready=false`，目标期刊规范未指定/未核验。

- `qa/FINAL_ACCEPTANCE.json`
- `qa/ANALYSIS_ACCEPTANCE.json`、`qa/RESULT_TABLE_CONSISTENCY.json`
- `qa/FIGURE_AMENDMENT_007_ACCEPTANCE.json`、`qa/VISUAL_QA.json`、`qa/AGENT_VISUAL_REVIEW.json`
- `coordination/OLD_ROOT_NONINTERFERENCE.json`、`coordination/RESOURCE_USAGE_SUMMARY.json`
- `figures/FIGURE_MANIFEST.json`、`captions/FIGURE_CAPTIONS_AND_ALT.zh-CN.md`
- `RETURN_TO_CHATGPT.zh-CN.json`、`CHECKSUMS.sha256`

Checksum file is generated after all other final writes; `CHECKSUMS.sha256` excludes itself and is verified with `sha256sum -c`.
