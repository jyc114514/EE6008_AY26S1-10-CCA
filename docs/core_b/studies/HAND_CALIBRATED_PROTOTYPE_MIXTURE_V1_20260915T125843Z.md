> Archived study summary for `HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z` (source snapshot `20260915`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B Hand Constrained Calibration + Conditional Prototype 决策

## 总体状态

本轮是 development-only、单 fresh seed `20260938` 的 CAL/PROTO exploratory screen。CAL、corrected train-only mode audit、真实 GPU Stage0 `2/2`、formal matrix `2/2`、OOF、10,000 次 paired episode bootstrap、16 类图和 QA 均完成。Final/held-out access=`0`；旧 Gate A/B、旧 DCT candidate gate、冻结候选和 Benchmark v2 均未改写。

第一版 mode audit 因错误聚类了 D1-L1 proposal residual 而被判 invalid，完整保存在 `invalid_attempts/`；所有 Route C 和 prototype 结论只使用 corrected audit 的 ground-truth future residual。

## CAL 结果

在 `HD-D1-L1` 的 `2741` 条 train OOF 上固定扫描 `alpha=0.000,...,1.000`。没有任何 `alpha>0` 同时满足 native FD 相对 Persistence 不增加和 level-gain retention 至少 `80%`。因此 `alpha=0` 仅作为 Persistence sanity control，CAL gate=`FAIL`，没有 calibrated comparator。

## Corrected mode audit

使用 ground-truth `Hand future residual to Persistence` 的 DCT K=16 系数：

- `K_mode=2`：silhouette=`0.2920`，bootstrap ARI median=`1.0000`，最小 cluster prevalence=`6.35%`，FUSION/state/prefix macro-F1=`0.7295`，majority=`0.4836`，task-only=`0.6502`，28 个 task 有多个 mode，mode oracle native-FD headroom=`0.000179`，通过。
- `K_mode=4`：最小 cluster prevalence=`1.28%`，失败。
- `K_mode=8`：最小 cluster prevalence=`0.04%` 且 single-cluster task concentration=`100%`，失败。

按 smallest-passing-K 规则选择 `K_mode=2`，Route C=`PROTO`。这说明 train-only mode signal 稳定且能由 past context 部分预测，但不保证 prototype 在 dev 上可部署。

## Prototype screen

| 输出 | Hand MAE | Native FD MAE | Full-28 MAE | FD 相对 Persistence |
|---|---:|---:|---:|---:|
| fixed hybrid Persistence | 0.257422 | 0.018316 | 0.180836 | 0.00% |
| HP-SINGLE | 0.238900 | 0.018616 | 0.171575 | +1.64% |
| HP-PROTO-MAP | 0.238557 | 0.018784 | 0.171404 | +2.55% |
| HP-PROTO-MEAN | 0.238244 | 0.018779 | 0.171247 | +2.52% |

MAP/MEAN 的 Hand level improvement 相对 Persistence 分别为 `0.018865` 和 `0.019178`；相对 fixed hybrid 的 Full-28 improvement 分别为 `0.009432` 和 `0.009589`。但相对 HP-SINGLE 的 Hand improvement CI 均跨 0，native FD 均显著恶化。

Prototype dev mode output 还出现明显 collapse：argmax prevalence=`585/614=95.28%`，posterior max>0.95 的 episode fraction=`91.53%`。正因如此，MAP 和 MEAN 都未通过 mode-collapse、mode-support、FD-vs-SINGLE 与 co-primary incremental gates。

## 科学结论

本轮支持：Hand future residual 中存在稳定的 train-only 二模式结构，past FUSION/state 能预测其一部分；prototype oracle 有正的 FD headroom；DCT/dual-loss/prototype 不是完全无效。

本轮不支持：当前 single shared-refinement prototype 能在实际 deterministic MAP/MEAN 输出中同时击败 Persistence、SINGLE 并满足 native FD `<=1%` 门禁。不能把 mode oracle 当部署结果，也不能声称 physical/Cartesian trajectory fidelity。

Candidate gate=`FAIL`。不启动 multi-seed，不访问 final/held-out，不增加 mode K、cluster seed、loss weight、refinement head 或 decode rule sweep。下一步唯一生成的是 action-phase conditioning review Prompt，标记 `generated_not_executed`。
