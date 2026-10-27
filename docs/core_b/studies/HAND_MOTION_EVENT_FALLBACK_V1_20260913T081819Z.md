> Archived study summary for `HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z` (source snapshot `20260913`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Hand motion-event fallback exploratory decision

状态：`single_seed_exploratory_complete`；Stage0=`6/6`；matrix=`6/6`；seed=`20260925`。

| Cell | Event | Arm MAE | Hand MAE | Full-28 MAE | primary AUPRC | candidate gate |
|---|---:|---:|---:|---:|---:|---:|
| H-S-U | U | 0.109245 | 0.279601 | 0.194423 | 0.3537 | FAIL |
| H-S-W | W | 0.110172 | 0.274714 | 0.192443 | 0.5008 | FAIL |
| H-S-T | T | 0.109351 | 0.277664 | 0.193507 | 0.4925 | FAIL |
| H-F-U | U | 0.104007 | 0.268507 | 0.186257 | 0.3581 | FAIL |
| H-F-W | W | 0.104059 | 0.273145 | 0.188602 | 0.5560 | FAIL |
| H-F-T | T | 0.103898 | 0.275775 | 0.189836 | 0.5201 | FAIL |

## 结论

候选门禁：`FAIL`。当前没有通过完整门禁的 W/T cell 时，不能生成或执行 multi-seed follow-up；本轮结果仅是单 seed、development、outcome-exposed 的机制诊断。

Phase A 的原始“Hand 更低运动”解释没有得到支持；它只通过 C 路径进入本矩阵。所有新 cell 都必须同时超过 Persistence、同输入 U、以及零训练 `HYB-A_NN-H_P` 才能宣称 learned event fallback 有增量价值。

Single-seed hierarchical bootstrap 的 seed 层是退化的：它没有估计 training-seed uncertainty。不能把本轮 AUPRC 或 MAE 解释成跨 seed 稳定性，也不能替代 held-out/final evaluation。

final-test access=`0`；冻结候选未改变；final protocol 未改变；没有 LoRA、encoder、raw-video 或后续自动训练。
