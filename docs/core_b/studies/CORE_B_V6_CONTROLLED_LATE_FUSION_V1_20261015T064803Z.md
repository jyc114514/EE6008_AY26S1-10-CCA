> Archived study summary for `CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z` (source snapshot `20261015`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V6 有对照的晚期融合与 inner-tune 标量校准

**状态：complete；开发实验。** 固定矩阵已评分。正收益和最低样本量不是完成门；final/heldout payload access 为 0。

## V5 MIX50 基线补全

| 指标 | macro mean | 三个固定 seed 的 sample SD |
|---|---:|---:|
| Position MAE | 0.155471482 | 0.001066242 |
| Safe-FD L1 | 0.818973919 | 0.002633799 |
| Seam Safe-FD L1 | 0.765222186 | 0.013308299 |

V5 的 15 行 task×seed 双指标与 seam、相对 S/V/BIAS_STATE/BIAS_VIDEO 的每行 gain、以及 task/seed/macro 条件区间分别在 `results/V5_MIX50_TASK_SEED.csv`、`results/V5_MIX50_GAIN_TASK_SEED.csv` 与 `results/V5_MIX50_PAIRED_GROUP_CI.csv`。以下是五任务等权、三个固定 seed 均值的 gain=reference−MIX50。

| V5 reference − MIX50 | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
|---|---:|---:|---:|
| S | +0.002084 [+0.000676, +0.003430] | +0.011833 [+0.008659, +0.015374] | +0.001720 [-0.013329, +0.012246] |
| V | +0.002867 [+0.001395, +0.004551] | +0.000781 [-0.002582, +0.003883] | +0.030539 [+0.017081, +0.047615] |
| BIAS_STATE | +0.001830 [+0.000414, +0.003198] | +0.011833 [+0.008659, +0.015374] | +0.054010 [+0.036896, +0.067470] |
| BIAS_VIDEO | +0.002595 [+0.001114, +0.004303] | +0.000781 [-0.002582, +0.003883] | +0.082721 [+0.066866, +0.101480] |

## Inner-tune 系数

| 家族 | alpha | inner-tune Position MAE | inner-tune Safe-FD L1 | seam | J |
|---|---:|---:|---:|---:|---:|
| SV | 0.50 | 0.174647057 | 1.053357472 | 1.070822122 | 0.279982804 |
| SS | 0.45 | 0.175329965 | 1.055553059 | 1.066553297 | 0.280885271 |

每族固定 21 点，使用 screen seed 202610130 的 S/A/V inner-fit-only checkpoint predictions；46 episodes、270 anchors 按 task 等权、episode 等权选择 `Position MAE + 0.1×Safe-FD L1`。inner-tune 曾用于 V2 recipe 选择，因此这里是开发元校准。`alpha_SV=0.50` 使 `SV_CAL` 与 V5 `SV50/MIX50` 共享 payload。

## 四个程序的正式 dev 结果

| 方法 | Position MAE mean ± seed SD | Safe-FD L1 mean ± seed SD | Seam Safe-FD L1 mean ± seed SD |
|---|---:|---:|---:|
| S | 0.157555430 ± 0.001035352 | 0.830807296 ± 0.003093777 | 0.766942532 ± 0.005265167 |
| A | 0.159870915 ± 0.001342864 | 0.829451281 ± 0.002901399 | 0.776169872 ± 0.015822087 |
| V | 0.158338282 ± 0.001486070 | 0.819754475 ± 0.002993723 | 0.795760936 ± 0.021558371 |
| BIAS_STATE | 0.157301432 ± 0.001046684 | 0.830807296 ± 0.003093777 | 0.819231899 ± 0.006348953 |
| BIAS_VIDEO | 0.158066240 ± 0.001481867 | 0.819754475 ± 0.002993723 | 0.847943220 ± 0.021419398 |
| PERSISTENCE | 0.222262188 (shared) | 0.854192669 (shared) | 0.802416508 (shared) |
| SV50 | 0.155471482 ± 0.001066242 | 0.818973919 ± 0.002633799 | 0.765222186 ± 0.013308299 |
| SS50 | 0.157003371 ± 0.001060122 | 0.825176925 ± 0.000320478 | 0.761043120 ± 0.008338075 |
| SV_CAL | 0.155471482 ± 0.001066242 | 0.818973919 ± 0.002633799 | 0.765222186 ± 0.013308299 |
| SS_CAL | 0.156907223 ± 0.001048333 | 0.825295522 ± 0.000409550 | 0.760652011 ± 0.007726950 |

## 锁定 paired gains

正值表示 candidate loss 更低。95% 区间为 5000 次 within-task source-group paired bootstrap；对固定 bases、alpha 与已暴露 dev 条件化，不是跨 seed 总体推断。

| reference − candidate | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
|---|---:|---:|---:|
| VIDEO_SUBSTITUTION_50 | +0.001532 [+0.000261, +0.002744] | +0.006203 [+0.002931, +0.010112] | -0.004179 [-0.023131, +0.009914] |
| VIDEO_SUBSTITUTION_CAL | +0.001436 [+0.000174, +0.002642] | +0.006322 [+0.003089, +0.010157] | -0.004570 [-0.023044, +0.009167] |
| SV_CALIBRATION_GAIN | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| SS_CALIBRATION_GAIN | +0.000096 [+0.000018, +0.000179] | -0.000119 [-0.000272, +0.000037] | +0.000391 [-0.000295, +0.001146] |
| SV_CAL_VS_S | +0.002084 [+0.000676, +0.003430] | +0.011833 [+0.008659, +0.015374] | +0.001720 [-0.013329, +0.012246] |
| SV_CAL_VS_V | +0.002867 [+0.001395, +0.004551] | +0.000781 [-0.002582, +0.003883] | +0.030539 [+0.017081, +0.047615] |
| SV_CAL_VS_BIAS_STATE | +0.001830 [+0.000414, +0.003198] | +0.011833 [+0.008659, +0.015374] | +0.054010 [+0.036896, +0.067470] |
| SV_CAL_VS_BIAS_VIDEO | +0.002595 [+0.001114, +0.004303] | +0.000781 [-0.002582, +0.003883] | +0.082721 [+0.066866, +0.101480] |

Position seed-LOO gains remain positive for VIDEO_SUBSTITUTION_50 (202610131: 0.001877, 202610132: 0.001491, 202610133: 0.001227) and VIDEO_SUBSTITUTION_CAL (202610131: 0.001780, 202610132: 0.001403, 202610133: 0.001124). Across all 15 task-LOO rows, VIDEO_SUBSTITUTION_50 is positive in 15; range +0.000518 to +0.002690.
附加固定 comparisons `SV50_VS_S/V/BIAS_STATE/BIAS_VIDEO` 与 `SS50_VS_S/A` 的三指标 gains 和条件区间在 `results/PAIRED_GROUP_CI.csv` 与 `results/PAIRED_EFFECTS.csv`。完整 seed/task LOO 文件包含全部 14 contrasts。`SS_CAL` 相对 `SS50` 的 Position gain 较小，Safe-FD 点估计为负且区间跨 0，指标呈轻微 trade-off。

## 资源与 target exposure

- screen-seed inner-fit models reused: 3；new inner-fit/formal neural fits: 0；CPU scalar calibrations: 2；inner-tune GPU replay attempts: 4，接受 3 个 tune bundles。
- GPU watcher job 共 4 次，使用物理 GPU 2（进程内 `cuda:0`），最大 observed reserved 为 134 MiB；全部 lease 已释放。V_FIT 首次登记 102 MiB、实测 reserved 134 MiB，且首次 run-local lease-dir 指向 `coordination/watcher_status`；该输出保留但不用于 alpha。更正重放使用 `coordination/gpu_lease` 和 134 MiB 登记，payload 逐元素相同。
- V6 预测矩阵 12 个逻辑 bundles：9 个新增逻辑程序×seed、6 个唯一新 tensor、3 个 SV_CAL alias、3 个 V5 SV50 复用。
- V5 dev archive 加载 4 次；V6 在 seal 后加载 2 次，共 6 次；V6 读入 370 个 target-window rows，唯一仍为 185。第一轮读取后在打分前停止；第二轮计算 1110 条新窗口损失，随后图表由保存的损失/图数据恢复。V5 已有 3330 条 primary loss；V5+V6 known-grain 合计 4440。先前报告的 35523 计数单位未核实，保持单列未加总。
- final/heldout payload access: **0**；source roots V2–V5 writes: 0。session/trial 独立性与物理 command 单位未知。

## 文件

- V5 MIX50 补全：`results/V5_MIX50_TASK_SEED.csv`, `results/V5_MIX50_GAIN_TASK_SEED.csv`, `results/V5_MIX50_PAIRED_GROUP_CI.csv`
- V6 结果：`results/PRIMARY_TASK_SEED.csv`, `results/PAIRED_EFFECTS.csv`, `results/PAIRED_GROUP_CI.csv`, `results/SEED_LOO.csv`, `results/TASK_LOO.csv`, `results/CONVEXITY_CHECK.csv`
- QA 与资源：`qa/FINAL_ACCEPTANCE.json`, `inventory/OUTCOME_EXPOSURE.json`, `results/RESOURCE_LEDGER.csv`, `CHECKSUMS.sha256`
- 图与数据：`figures/`, `figure_data/`; agent visual review passed on alpha, dual metric/seam, and paired contrasts. Full manual review remains pending for trajectory details.

固定矩阵到此停止；未基于 dev 追加系数、模型家族、seed 或训练。
