> Archived study summary for `HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z` (source snapshot `20260915`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B Hand DCT × Temporal Dual-Loss 决策

## 结果边界

本轮是单个 fresh training seed `20260933` 的 development-only 2×2 实验。四个 formal cell 均完成，3-fold train-only OOF 均为 `2741/2741`、overlap=`0`，final/held-out access=`0`。旧 Gate A、旧 Gate B、冻结候选和 Benchmark v2 均未修改。

此前无训练 DCT counterfactual 的 FD-harm reduction 最高为 `45.15%`，低于事前锁定的 `50%`，所以旧 Gate B 必须继续保持 FAIL。本轮只是基于该机制信号锁定的新训练假设。

## 四格结果

| Cell | Hand standardized MAE | Hand native FD MAE | Full-28 standardized MAE | 相对 Persistence Hand | 相对 Persistence FD |
|---|---:|---:|---:|---:|---:|
| `HD-D0-L0` | 0.235928 | 0.020088 | 0.170089 | 8.35% | +9.67% |
| `HD-D0-L1` | 0.240891 | 0.018657 | 0.172570 | 6.42% | +1.86% |
| `HD-D1-L0` | 0.236478 | 0.019397 | 0.170364 | 8.14% | +5.90% |
| `HD-D1-L1` | 0.238725 | 0.018620 | 0.171487 | 7.26% | +1.66% |

固定 Arm-learned/Hand-Persistence hybrid 的 Full-28 MAE 为 `0.180836`。四格 Full-28 均优于该 comparator；但没有一格同时满足全部 temporal gate。

## Factorial 解释

- Decoder main effect：Hand level error improvement `+0.000808`，native FD improvement `+0.000364`。
- Loss main effect：Hand level error `-0.003605`，native FD improvement `+0.001104`。
- Decoder × Loss interaction：Hand level `-0.001358`，native FD `+0.000327`。

因此 dual loss 确实改善了 first-difference，DCT 也有时间表示信号；但 level accuracy 与相对 Persistence 的 1% FD 门禁没有同时满足。

## Candidate gate

按事前顺序 `HD-D1-L1 → HD-D1-L0 → HD-D0-L1` 检查：

- `HD-D1-L1`：level-gain retention=`86.99%`，相对 Persistence native FD 退化=`1.66%`，超过 `1%`，FAIL。
- `HD-D1-L0`：level-gain retention=`97.44%`，native FD 退化=`5.90%`，FAIL。
- `HD-D0-L1`：level-gain retention=`76.91%` 且 native FD 退化=`1.86%`，FAIL。

`HD-D1-L1` 相对 `HD-D0-L0` 的 native FD improvement 为 `0.001468`，95% episode-bootstrap CI=`[0.001320, 0.001617]`；这说明 dual loss 有明确的 matched-baseline temporal improvement，但仍不足以把 FD 拉到 Persistence 的 1% practical bound 内。故 candidate gate=`FAIL`，不启动 multi-seed。

## 结论与下一步

本轮支持：DCT parameterization 与 native FD-aware loss 都能改变时间误差，且结果不是单一 joint/task 贡献；D1-L1 的高运动、late 和 endpoint 相对 Persistence 仍分别改善约 `7.9%`、`10.8%` 和 `11.8%`。

本轮不支持：DCT × dual-loss 已经成为可确认的 multi-seed candidate，也不支持 physical/Cartesian trajectory claim。下一步唯一生成的分支是 conditional Hand trajectory prototype/mixture/action-phase proposal；该 Prompt 仅为 `generated_not_executed`，不构成执行授权。不要追加 DCT K、lambda、decoder 或 smoothness sweep。
