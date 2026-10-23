> Archived study summary for `V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z` (source snapshot `20260923`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B 固定模型视频对应关系实验与延迟结论复核

## 结论摘要

状态：complete_with_caveats。6 个既有 checkpoint、15 个固定模型上下文、120/120 development bundle 均通过接受门；matched-D 全量 source replay 逐元素一致。无训练、无 optimizer step、无 encoder forward/提取、final/held-out access=0、教授联系=0。GPU 2,3，总 GPU-hours 0.06030，最大并发 2。

主要发现：在 ANCHOR-D 与 ROBUST-D 延迟压力两情境中，正确 matched dynamic 相对 same-task donor 与 matched static 的 Arm/Hand MAE 效应均为正，8/8 primary crossed 95% CI 均不跨 0，条件 sign-flip Holm 均通过。它支持固定模型在该 development population 使用 episode-specific、multi-frame 视觉信息；不证明物理因果或原始帧顺序理解。

## 延迟结果复核

全部 6×4 MAE/FD mean±seed SD、12 个 primary/FD effect、Holm、观测 age、normal-input guardrail 与旧状态字段更正在 [DELAY_RESULT_RECONCILIATION.zh-CN.md](server-only://V3_VIDEO_CORRESPONDENCE_AUDIT_V1_20260923T055927Z/DELAY_RESULT_RECONCILIATION.zh-CN.md)。重要限制：δ=0 下 ROBUST-D Arm FD 相对 CLEAN-D 为 +1.18%，95% relative CI [-0.10%, +3.69%]，上界超过 +1% margin；不能称鲁棒训练没有正常输入代价。sealed source FINAL_STATUS 未改。

## Recipient 与 donor 质量

总体 603 episodes/2,322 windows。25 个单 episode task 无法构造同-task derangement，按协议排除；共同子集 578 episodes/2,222 windows/93 tasks。三套 same-task 与三套 cross-task maps 均 episode-level 一对一、跨 episode/duplicate group；donor frames 只到其自身 anchor 为止，无 future target。

| Donor type | Token distance mean / p95 | Donor-window reuse | Pairwise maps differ |
|---|---:|---:|---:|
| Same-task | 0.1147 / 0.1782 | 30.8%–31.6% | 79.1%–81.3% |
| Cross-task | 0.1943 / 0.2469 | 54.4%–55.9% | 99.8%–99.8% |

Cross-task feature distance高于 same-task，且 donor 来自有限 dev 池并重复使用；cross-task 损害的 OOD/场景分布解释不能排除，推断条件于锁定 maps。

## MAE、FD、TV 与 Persistence skill

表为三训练 seed 的 mean±SD。ROBUST_STRESS 先在每 seed/episode 内对 delay 0.25/0.50/1.00 等权平均。Persistence 重复未改变的 recipient anchor command。

| Context | Condition | Arm MAE | Hand MAE | Full MAE | Arm FD | Hand FD | Full FD | TV ratio | Persistence skill A/H/F |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ANCHOR | matched dynamic | 0.31590 ± 0.00807 | 0.25009 ± 0.00543 | 0.28299 ± 0.00654 | 0.02093473 ± 0.00009949 | 0.00679130 ± 0.00001949 | 0.01386301 ± 0.00004091 | 0.095 ± 0.015 | 13.4/15.1/14.2% |
| ANCHOR | same-episode static token swap | 0.35838 ± 0.00104 | 0.29153 ± 0.00147 | 0.32495 ± 0.00125 | 0.02070464 ± 0.00003926 | 0.00700581 ± 0.00001460 | 0.01385523 ± 0.00001542 | 0.040 ± 0.006 | 1.8/1.1/1.5% |
| ANCHOR | same-task donor (3 maps averaged) | 0.35356 ± 0.00309 | 0.27692 ± 0.00202 | 0.31524 ± 0.00199 | 0.02101371 ± 0.00011735 | 0.00700044 ± 0.00000607 | 0.01400707 ± 0.00005635 | 0.089 ± 0.015 | 3.1/6.0/4.4% |
| ANCHOR | cross-task donor (3 maps averaged) | 0.38942 ± 0.00842 | 0.31205 ± 0.00505 | 0.35074 ± 0.00673 | 0.02093153 ± 0.00013439 | 0.00723930 ± 0.00004282 | 0.01408541 ± 0.00008806 | 0.076 ± 0.019 | -6.7/-5.9/-6.4% |
| ROBUST_STRESS | matched dynamic | 0.30385 ± 0.00231 | 0.23118 ± 0.00241 | 0.26751 ± 0.00110 | 0.02118072 ± 0.00000342 | 0.00666721 ± 0.00001227 | 0.01392396 ± 0.00000602 | 0.147 ± 0.005 | 16.7/21.5/18.9% |
| ROBUST_STRESS | same-episode static token swap | 0.33059 ± 0.00990 | 0.25757 ± 0.01437 | 0.29408 ± 0.01140 | 0.02116230 ± 0.00021324 | 0.00686075 ± 0.00013971 | 0.01401153 ± 0.00014958 | 0.129 ± 0.041 | 9.4/12.6/10.8% |
| ROBUST_STRESS | same-task donor (3 maps averaged) | 0.31247 ± 0.00085 | 0.23898 ± 0.00532 | 0.27573 ± 0.00260 | 0.02117656 ± 0.00000504 | 0.00672573 ± 0.00003633 | 0.01395115 ± 0.00001927 | 0.138 ± 0.006 | 14.4/18.9/16.4% |
| ROBUST_STRESS | cross-task donor (3 maps averaged) | 0.33281 ± 0.00521 | 0.25520 ± 0.00952 | 0.29400 ± 0.00736 | 0.02113105 ± 0.00001170 | 0.00684063 ± 0.00006587 | 0.01398584 ± 0.00003291 | 0.126 ± 0.007 | 8.8/13.4/10.8% |

Persistence subset Arm/Hand/Full standardized MAE = 0.36487 / 0.29466 / 0.32976。TV ratios远低于 1，预测变化总量仍明显小于 target；物理单位未标定，TV 仅作命令空间描述。

## Primary MAE effects（8 项 Holm）

Effect = alternative loss − matched-D loss，正值表示正确 episode-matched video 更好。episode CI 10,000 次；crossed seed×episode CI 20,000 次；Holm 调整 conditional-on-seeds episode sign-flip p。

| Context | Alternative | Group | Effect | Relative | Episode CI 95% | Crossed CI 95% | Sign-flip p | Holm p |
|---|---|---|---:|---:|---|---|---:|---:|
| ANCHOR | SAME-TASK | Arm | 0.037657 | 13.41% | [0.033353, 0.042063] | [0.028835, 0.045789] | 9.999e-05 | 0.00079992 |
| ANCHOR | SAME-TASK | Hand | 0.026837 | 12.46% | [0.024277, 0.029537] | [0.022460, 0.031130] | 9.999e-05 | 0.00079992 |
| ANCHOR | MATCHED-S | Arm | 0.042482 | 15.05% | [0.037835, 0.047257] | [0.033092, 0.051193] | 9.999e-05 | 0.00079992 |
| ANCHOR | MATCHED-S | Hand | 0.041440 | 18.43% | [0.038196, 0.044805] | [0.034566, 0.047806] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | SAME-TASK | Arm | 0.008626 | 3.32% | [0.006825, 0.010513] | [0.006209, 0.011297] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | SAME-TASK | Hand | 0.007806 | 3.85% | [0.006864, 0.008795] | [0.005289, 0.011014] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | MATCHED-S | Arm | 0.026745 | 9.86% | [0.023861, 0.029659] | [0.015756, 0.037396] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | MATCHED-S | Hand | 0.026388 | 12.79% | [0.024080, 0.028738] | [0.010819, 0.036838] | 9.999e-05 | 0.00079992 |

primary HOLM: 8/8 pass。Seed/map sensitivity：same-task Arm/Hand effects 9/9 seed×map rows positive in each context; static effects 3/3 seeds positive. Task support is descriptive (not per-task tests): same-task contrasts favor matched dynamic in about 79–92/93 tasks depending on group/context; all 14 Arm and all 14 Hand joints show positive pooled effects. Map/window are not independent replicates。

## Secondary native FD effects（8 项 Holm）

| Context | Alternative | Group | Effect | Crossed CI 95% | Sign-flip p | Holm p |
|---|---|---|---:|---|---:|---:|
| ANCHOR | SAME-TASK | Arm | 0.00007898 | [0.00003922, 0.00012136] | 9.999e-05 | 0.00079992 |
| ANCHOR | SAME-TASK | Hand | 0.00020914 | [0.00017914, 0.00023827] | 9.999e-05 | 0.00079992 |
| ANCHOR | MATCHED-S | Arm | -0.00023008 | [-0.00029449, -0.00015974] | 9.999e-05 | 0.00079992 |
| ANCHOR | MATCHED-S | Hand | 0.00021451 | [0.00018356, 0.00024702] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | SAME-TASK | Arm | -0.00000416 | [-0.00002756, 0.00001927] | 0.70893 | 0.70893 |
| ROBUST_STRESS | SAME-TASK | Hand | 0.00005852 | [0.00003634, 0.00008672] | 9.999e-05 | 0.00079992 |
| ROBUST_STRESS | MATCHED-S | Arm | -0.00001841 | [-0.00023853, 0.00016776] | 0.19398 | 0.38796 |
| ROBUST_STRESS | MATCHED-S | Hand | 0.00019355 | [0.00006708, 0.00033373] | 9.999e-05 | 0.00079992 |

FD HOLM: 6/8 pass。native FD 与 MAE 不是同一目标；具体差异须按上表逐组解释。

## Cross-task 与解释边界

Cross-task donor 的 Arm/Hand MAE 平均损害相对 matched-D：ANCHOR +0.0735/+0.0620，ROBUST_STRESS +0.0290/+0.0240；同时 token 距离更大，因此该诊断不分离 task semantics 与 OOD feature shift。

支持：固定模型依赖正确对应的当前 episode dynamic video；static swap 与 same-task donor 均造成 MAE 退化。不能推出物理因果、raw-frame temporal-order understanding、网络故障恢复或 final 泛化。MATCHED-S 是 D-trained checkpoint 的 test-time static-token swap，不是 S-trained control。

唯一后续建议：若需进一步检验 temporal order，另行预注册 raw-frame order intervention；本轮不启动训练、视频重提取、更多 delay 或 final evaluation。

六组图（PNG/PDF）、caption 与底层 CSV 位于 figures/、figures/captions/、figure_data/。逐 episode、seed/map、task、joint 表位于 analysis/。
