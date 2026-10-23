> Archived study summary for `HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z` (source snapshot `20260917`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# 历史信息 × 视频信息 2×2 实验决策

## 完成状态

四个正式 cell 已全部完成：每格 40 epochs、1720 optimizer steps、last-epoch checkpoint、固定 CPU dev inference 均通过验收。真实 Stage0 为 4/4；正式训练实际使用物理 GPU [0, 2]，最多同时两个本项目 child。

本轮是 single-seed development exploration，不能称 final 或跨 seed 确认。

## 主要结果

| Cell | Hand standardized MAE | Full-28 standardized MAE | Hand native FD |
|---|---:|---:|---:|
| Persistence | 0.257422 | 0.192195 | 0.018316 |
| Fixed hybrid | 0.257422 | 0.180836 | 0.018316 |
| HI-S-LAST | 0.246817 | 0.175534 | 0.018437 |
| HI-S-HIST | 0.250291 | 0.177270 | 0.018478 |
| HI-F-LAST | 0.257358 | 0.180804 | 0.018318 |
| **HI-F-HIST** | **0.252800** | **0.178525** | **0.018398** |

预注册候选优先级为 F-HIST、S-HIST、F-LAST、S-LAST，因此当前 eligible development proposal 是 `HI-F-HIST`。它相对 Persistence 的 Hand improvement 为 `0.004622`，95% CI `[0.001187, 0.008306]`；相对固定 Arm-learned + Hand-Persistence hybrid 的 Full improvement 为 `0.002311`，95% CI `[0.000593, 0.004153]`。native FD 相对 Persistence 变化为 `0.445%`，95% CI `[0.274%, 0.641%]`，位于当前 +1% operating margin 内。

## 两个科学问题

1. `S-HIST − S-LAST` 的 level effect 为负且 CI 不跨 0：没有视频时，加入真实历史反而没有改善该模型。
2. `F-HIST − F-LAST` 的 level effect 为正且 CI 不跨 0：有固定视频时，历史带来 development level 增量；但 FD effect 为负，表示 temporal fidelity 有小幅代价。
3. `F-HIST − S-HIST` 的 level CI 跨 0：不能声称视频在历史之外提高 level accuracy；其 native FD effect 为正，说明视频增量主要体现在当前 FD 指标。
4. `F-LAST − S-LAST` 也表现为 level 变差、FD 改善，说明视频作用依赖所测指标，不能简单称为全面视频收益。

因此当前最准确的表述是：**历史信息的 level 价值是 video-conditional；固定视频对 native FD 有 development 增量，但 level 增量不确定。** 本轮没有单独证明顺序、因果机制或 dynamics understanding。

## 为什么上一轮 router 不继续

上一轮 fixed-q oracle 仍比 Persistence 差 `2.2886%` native FD，旧 router 距离该 oracle 仅 `0.2593%`，所以那条结论只说明固定两组件选择空间不足；本轮更换为直接监督的输入信息问题，旧 router gate 不再阻断。

旧 `ARI=1` 已撤回：它没有在 bootstrap 中重新拟合 centers；正确 refit 后 K=2 ARI median 约 `-0.0007` 且区间很宽。Base OOF 完整也不等于 selector OOF 完整，旧 fold whitening 的 all-train 统计共享仍保留为限制。

## 边界

所有结果来自反复使用的 development set；物理 command units、URDF/FK、session/source 泛化仍 unknown。`FINAL_TEST_ACCESS=0`，没有联系教授，没有执行下一轮 multi-seed Prompt，也没有修改旧 roots。
