> Archived study summary for `D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z` (source snapshot `20261006`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# D3-EQ 当前回传

状态：`internal_analysis_complete_human_review_pending`。详细结论见 [D3_EQ_SCIENTIFIC_RESULTS_REVIEWED_V3.zh-CN.md](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/D3_EQ_SCIENTIFIC_RESULTS_REVIEWED_V3.zh-CN.md)。未做新训练、optimizer step=0、GPU inference=0h、D5 未训练、final/held-out 访问=0。

- 已对现有 36 个 epoch-40 head 做 CPU train-only inference，36/36 × 17,757 train anchors 通过；dev 的 36 个 accepted predictions 复用。训练 target 未在推理进程载入。
- 新评分独立复现旧 dev MAE/FD 216 项，最大绝对差 `1.11e-9`；train 2,712 groups、dev 603 groups，source-group overlap=0。统计口径保持锁定 group bootstrap/primary Holm；跨 split 无配对 bootstrap。
- TS11-D Full28 的标准化 MAE / Persistence / model gain：H30 train `0.1861 / 0.2242 / +0.0381`，dev `0.1963 / 0.2231 / +0.0269`；H60 train `0.2453 / 0.3343 / +0.0890`，dev `0.2676 / 0.3286 / +0.0610`。
- 但预测 TV/GT 仅约 `13.1% / 13.2%`（H30 train/dev）与 `14.2% / 13.5%`（H60）。同一 under-amplitude 在 train 和 dev 都存在；不是纯 dev-only 失效。SLOPE 同尺度分数明显差于 Persistence（Full28 std MAE H30 `0.488/0.485`、H60 `0.927/0.924`）。
- action[0:14] 与 action[14:28] 的变化幅度不同，但来源代码对 Hand/Arm 顺序冲突，故本包只用 raw slices；不作 anatomy/physical-unit 解释。
- 更正：TS11-D−TS10-D 表示加入空间信息；TS11-D−TS01-D 表示加入时间信息；TS11-S 是 state+anchor 条件下的静态视频 control，不是 video-only。raw-static 源实现已验证，但 extraction run manifest 未绑定代码 SHA。
- 72 条旧 claims 中 70 条已 agent 检查；C007/C059 的 E023 D5 来源未打开，仍 `not_checked`；human review 仍 pending。范围内 EQ hashes 41/41 通过，范围外 E023 条目排除、非失败。

唯一建议后续是 TS11-D/STATE-CTRL × H30/H60 × 3 seeds 的 12 格 corrected-scheduler matched control，完整提案见 [NEXT_SINGLE_EXPERIMENT_PROPOSAL.md](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/NEXT_SINGLE_EXPERIMENT_PROPOSAL.md)。它仅生成、未执行，也不能证明 scheduler 是当前根因。请先查看不超过 8 项的 [人工审阅清单](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/HUMAN_REVIEW_CHECKLIST.zh-CN.md)。

可复核关键表：[`TRAIN_DEV_SIX_CONFIG_SUMMARY.csv`](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/TRAIN_DEV_SIX_CONFIG_SUMMARY.csv)、[`PERSISTENCE_SLOPE_BASELINES.csv`](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/PERSISTENCE_SLOPE_BASELINES.csv)、[`AMPLITUDE_TV_DIAGNOSTICS.csv`](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/AMPLITUDE_TV_DIAGNOSTICS.csv)、[`SEED_SENSITIVITY_PRIMARY.csv`](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/SEED_SENSITIVITY_PRIMARY.csv)、[`D3_EQ_OBJECTIVE_SENSITIVITY_ALL_CONFIGS.csv`](server-only://D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z/D3_EQ_OBJECTIVE_SENSITIVITY_ALL_CONFIGS.csv)。
