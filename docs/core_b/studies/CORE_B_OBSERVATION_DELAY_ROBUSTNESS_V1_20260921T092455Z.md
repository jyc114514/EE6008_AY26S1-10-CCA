> Archived study summary for `CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z` (source snapshot `20260921`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B：Observation 延迟鲁棒性收菜报告

## 完成状态

STUDY_STATUS=complete；Stage0 6/6，训练 18/18，预测 72/72。每 checkpoint 为 epoch 40、11,120 steps；所有 replay、ID、finite、cutoff、参数 hash 检查通过。FINAL_TEST_ACCESS=0。

## Persistence baseline

Arm standardized MAE 0.363856，native FD 0.02020820；Hand 0.293306，FD 0.00696523；Full 0.328581，FD 0.01358671。Persistence 始终使用未延迟 anchor command，四个 delay 下保持不变。

## 六模型 × 四个 delay

MAE 为 standardized episode-equal；FD 为 native units。Skill 是相对 Persistence 的描述性 1−模型 MAE/Persistence MAE；表为三 seed 均值。

| Train cell | δ(s) | Arm MAE | Hand MAE | Full MAE | Arm FD | Hand FD | Full FD | Arm skill | Hand skill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN-N | 0.0 | 0.30915 | 0.23931 | 0.27423 | 0.0208640 | 0.0067052 | 0.0137846 | 0.150 | 0.184 |
| CLEAN-N | 0.25 | 0.30944 | 0.23962 | 0.27453 | 0.0208457 | 0.0067077 | 0.0137767 | 0.150 | 0.183 |
| CLEAN-N | 0.5 | 0.31057 | 0.24045 | 0.27551 | 0.0208344 | 0.0067158 | 0.0137751 | 0.146 | 0.180 |
| CLEAN-N | 1.0 | 0.31910 | 0.24593 | 0.28251 | 0.0208269 | 0.0067687 | 0.0137978 | 0.123 | 0.162 |
| CLEAN-D | 0.0 | 0.31759 | 0.24306 | 0.28033 | 0.0207240 | 0.0067417 | 0.0137329 | 0.127 | 0.171 |
| CLEAN-D | 0.25 | 0.31770 | 0.24335 | 0.28052 | 0.0207205 | 0.0067447 | 0.0137326 | 0.127 | 0.170 |
| CLEAN-D | 0.5 | 0.31827 | 0.24403 | 0.28115 | 0.0207246 | 0.0067528 | 0.0137387 | 0.125 | 0.168 |
| CLEAN-D | 1.0 | 0.32297 | 0.24783 | 0.28540 | 0.0207256 | 0.0067868 | 0.0137562 | 0.112 | 0.155 |
| CLEAN-S | 0.0 | 0.30463 | 0.23124 | 0.26793 | 0.0209743 | 0.0066399 | 0.0138071 | 0.163 | 0.212 |
| CLEAN-S | 0.25 | 0.30498 | 0.23172 | 0.26835 | 0.0209569 | 0.0066438 | 0.0138003 | 0.162 | 0.210 |
| CLEAN-S | 0.5 | 0.30668 | 0.23299 | 0.26984 | 0.0209426 | 0.0066547 | 0.0137987 | 0.157 | 0.206 |
| CLEAN-S | 1.0 | 0.31744 | 0.24038 | 0.27891 | 0.0209283 | 0.0067111 | 0.0138197 | 0.128 | 0.180 |
| ROBUST-N | 0.0 | 0.31494 | 0.24591 | 0.28042 | 0.0207231 | 0.0067495 | 0.0137363 | 0.134 | 0.162 |
| ROBUST-N | 0.25 | 0.31498 | 0.24593 | 0.28046 | 0.0207241 | 0.0067501 | 0.0137371 | 0.134 | 0.162 |
| ROBUST-N | 0.5 | 0.31495 | 0.24595 | 0.28045 | 0.0207263 | 0.0067507 | 0.0137385 | 0.134 | 0.161 |
| ROBUST-N | 1.0 | 0.31573 | 0.24653 | 0.28113 | 0.0207262 | 0.0067558 | 0.0137410 | 0.132 | 0.159 |
| ROBUST-D | 0.0 | 0.30563 | 0.23180 | 0.26872 | 0.0209682 | 0.0066379 | 0.0138030 | 0.160 | 0.210 |
| ROBUST-D | 0.25 | 0.30556 | 0.23178 | 0.26867 | 0.0209685 | 0.0066386 | 0.0138036 | 0.160 | 0.210 |
| ROBUST-D | 0.5 | 0.30553 | 0.23175 | 0.26864 | 0.0209709 | 0.0066388 | 0.0138049 | 0.160 | 0.210 |
| ROBUST-D | 1.0 | 0.30628 | 0.23217 | 0.26922 | 0.0209734 | 0.0066415 | 0.0138075 | 0.158 | 0.208 |
| ROBUST-S | 0.0 | 0.30586 | 0.23087 | 0.26837 | 0.0209682 | 0.0066356 | 0.0138019 | 0.159 | 0.213 |
| ROBUST-S | 0.25 | 0.30580 | 0.23083 | 0.26832 | 0.0209688 | 0.0066364 | 0.0138026 | 0.160 | 0.213 |
| ROBUST-S | 0.5 | 0.30586 | 0.23085 | 0.26836 | 0.0209733 | 0.0066376 | 0.0138054 | 0.159 | 0.213 |
| ROBUST-S | 1.0 | 0.30629 | 0.23155 | 0.26892 | 0.0209751 | 0.0066418 | 0.0138085 | 0.158 | 0.211 |

## Primary robustness contrasts

Stress outcome first averages δ=0.25/0.50/1.00 equally within episode, then resamples seed and episode. Six Arm/Hand contrasts form one Holm family; positive effect favors candidate.

| Contrast | Group | Effect | Crossed 95% CI | p | Holm p | Conditional episode sign-flip p |
|---|---|---:|---|---:|---:|---:|
| ROBUST-D_vs_ROBUST-N | arm_mae_std | 0.009430 | [-0.003888, 0.032214] | 0.57957 | 1 | 9.999e-05 |
| ROBUST-D_vs_ROBUST-N | hand_mae_std | 0.014239 | [0.000675, 0.037810] | 0.0012999 | 0.0064997 | 9.999e-05 |
| ROBUST-D_vs_ROBUST-S | arm_mae_std | 0.000192 | [-0.002485, 0.002830] | 0.90705 | 1 | 0.78802 |
| ROBUST-D_vs_ROBUST-S | hand_mae_std | -0.000823 | [-0.001925, 0.000066] | 0.077496 | 0.23249 | 0.026497 |
| ROBUST-D_vs_CLEAN-D | arm_mae_std | 0.013854 | [0.000080, 0.037491] | 0.044298 | 0.17719 | 9.999e-05 |
| ROBUST-D_vs_CLEAN-D | hand_mae_std | 0.013173 | [0.001121, 0.035995] | 9.9995e-05 | 0.00059997 | 9.999e-05 |

Holm 后 2/6 通过：ROBUST-D 相对 ROBUST-N 的 Hand MAE 改善 0.014239（CI [0.000675, 0.037810]，Holm p=0.00650）；相对 CLEAN-D 的 Hand 改善 0.013173（CI [0.001121, 0.035995]，Holm p=0.00060）。Arm 的 D 对 N 未通过；D 对 S 的 Arm/Hand 均未通过。

ROBUST-D−ROBUST-N 的 Hand seed 效应为 0.003868、0.000568、0.038280，三 seed 方向均为正但效应大小差异明显；请同时看 PRIMARY_ROBUSTNESS_BY_SEED.csv。

## FD、clean-input guardrail 与 clean-video

FD 的 6 项 secondary family 中 2 项 Holm 后通过，均为 Hand FD：ROBUST-D 优于 ROBUST-N 和 CLEAN-D。Arm FD 的三项比较均未通过。

| δ=0 guardrail：ROBUST-D−CLEAN-D | 相对变化 | crossed 95% CI | +1% margin |
|---|---:|---|---|
| arm_mae_std | -3.77% | [-11.04%, +0.87%] | CI 上界在 +1% 内 |
| hand_mae_std | -4.63% | [-13.13%, +0.64%] | CI 上界在 +1% 内 |
| arm_fd_native | +1.18% | [-0.10%, +3.69%] | CI 仍可能超过 +1% |
| hand_fd_native | -1.54% | [-4.45%, +0.10%] | CI 上界在 +1% 内 |

δ=0 的 Arm/Hand MAE 区间上界在 +1% 内；Arm FD 相对区间上界约 +3.69%，所以 Arm normal-input fidelity guardrail 仍不确定。CLEAN-D 对 CLEAN-N/S 的 clean-input 비교均未建立可靠增量。

## Video × delay 与 observation age

ROBUST-D 对 ROBUST-N 的 Hand gain 在 δ=0/0.25/0.50/1 秒约为 0.01411/0.01416/0.01420/0.01436。G(delayed)−G(0) 的 Hand 效应仅 0.000134（CI [0.000004,0.000295]，未对该诊断作额外 Holm）；Arm 为 0.000117 且 CI 跨 0。D 相对 S 未建立优势。因此 Hand 有 video 相对 no-video 增量，但动态内容并未显著胜过静态，且 video gain 随 delay 增长的额外幅度很小。

| 请求 δ(s) | episode-equal mean age(s) | window p95(s) | max age(s) |
|---:|---:|---:|---:|
| 0 | 0.0132 | 0.0250 | 0.0333 |
| 0.25 | 0.0257 | 0.1417 | 0.2667 |
| 0.5 | 0.0507 | 0.3750 | 0.5000 |
| 1 | 0.1511 | 0.8750 | 1.0333 |

1 秒 requested delay 的最大实际 age 为约 1.033 秒，来自历史时间采样量化；72/72 bundle 均满足所选 observation timestamp 不晚于 cutoff。

## Training、资源和边界

- Fresh seeds 20260956/57/58；matched initializations 3/3；ROBUST N/D/S 每 window/epoch schedule 一致，120/120 seed-epoch checks 通过。
- Source registry 与 frozen video cache SHA-256：11/11 匹配。
- 实际 GPU-hours 1.161075；GPU 1,2,3；最大并发 child 2；失败/OOM 0。
- 本轮只模拟 observation feedback delay；command history、anchor、video 和 targets 保持及时。不是实测网络延迟，也不是闭环控制安全证明。
- 结果仍为 development-only；encoder 未更新、未重提视频、未读 final/held-out、未联系教授、旧 roots 未修改。
- 下一步建议：冻结当前结论。若要真实延迟 claim，再单独制定并获准独立 held-out 或实机延迟验证；此处不自动启动。

详细数据见 analysis 目录；六组图及其 source CSV/caption 在 figures、figure_data、figure_manifests。

## Seed-level primary effects

| Contrast | Group | Seed | Effect | Episode bootstrap 95% CI |
|---|---|---:|---:|---|
| ROBUST-D_vs_ROBUST-N | arm_mae_std | 20260956 | -0.000483 | [-0.002742, 0.001764] |
| ROBUST-D_vs_ROBUST-N | arm_mae_std | 20260957 | -0.003920 | [-0.006124, -0.001763] |
| ROBUST-D_vs_ROBUST-N | arm_mae_std | 20260958 | 0.032695 | [0.028664, 0.036798] |
| ROBUST-D_vs_ROBUST-N | hand_mae_std | 20260956 | 0.003868 | [0.003002, 0.004752] |
| ROBUST-D_vs_ROBUST-N | hand_mae_std | 20260957 | 0.000568 | [-0.000406, 0.001545] |
| ROBUST-D_vs_ROBUST-N | hand_mae_std | 20260958 | 0.038280 | [0.035280, 0.041437] |
| ROBUST-D_vs_ROBUST-S | arm_mae_std | 20260956 | -0.002422 | [-0.004433, -0.000385] |
| ROBUST-D_vs_ROBUST-S | arm_mae_std | 20260957 | 0.000293 | [-0.001924, 0.002498] |
| ROBUST-D_vs_ROBUST-S | arm_mae_std | 20260958 | 0.002706 | [0.000346, 0.005001] |
| ROBUST-D_vs_ROBUST-S | hand_mae_std | 20260956 | -0.000549 | [-0.001430, 0.000303] |
| ROBUST-D_vs_ROBUST-S | hand_mae_std | 20260957 | -0.000132 | [-0.001052, 0.000797] |
| ROBUST-D_vs_ROBUST-S | hand_mae_std | 20260958 | -0.001787 | [-0.003132, -0.000389] |
| ROBUST-D_vs_CLEAN-D | arm_mae_std | 20260956 | 0.003253 | [0.001389, 0.005168] |
| ROBUST-D_vs_CLEAN-D | arm_mae_std | 20260957 | -0.000033 | [-0.002023, 0.001901] |
| ROBUST-D_vs_CLEAN-D | arm_mae_std | 20260958 | 0.038342 | [0.033403, 0.043262] |
| ROBUST-D_vs_CLEAN-D | hand_mae_std | 20260956 | 0.001240 | [0.000446, 0.002042] |
| ROBUST-D_vs_CLEAN-D | hand_mae_std | 20260957 | 0.001831 | [0.000923, 0.002754] |
| ROBUST-D_vs_CLEAN-D | hand_mae_std | 20260958 | 0.036449 | [0.033601, 0.039408] |

## Clean-input paired comparisons

| Contrast | Group | Effect | Crossed 95% CI | p |
|---|---|---:|---|---:|
| CLEAN-D_vs_CLEAN-N | arm_mae_std | -0.008440 | [-0.022481, 0.000405] | 0.092795 |
| CLEAN-D_vs_CLEAN-N | hand_mae_std | -0.003747 | [-0.016434, 0.004297] | 0.58347 |
| CLEAN-D_vs_CLEAN-S | arm_mae_std | -0.012967 | [-0.034936, 0.001138] | 0.11499 |
| CLEAN-D_vs_CLEAN-S | hand_mae_std | -0.011823 | [-0.037814, 0.002489] | 0.57557 |
