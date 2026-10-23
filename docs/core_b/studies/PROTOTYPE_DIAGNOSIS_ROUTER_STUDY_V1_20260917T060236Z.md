> Archived study summary for `PROTOTYPE_DIAGNOSIS_ROUTER_STUDY_V1_20260917T060236Z` (source snapshot `20260917`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B prototype 诊断与 router 对照：实验决策

## 结论

本轮 CPU-only 诊断完成，审计本身通过，但预注册的 router-entry gate 为
`false`。因此 `R-NAT` 与 `R-BAL` 均保持 `NOT_RUN`，不是训练失败。

阻断原因是固定-q、完整 chunk 的 FD oracle 相对 Persistence 的 native FD
仍为 `+2.2886%`，超过锁定的 `+1%` 上限；旧 prototype router 相对同一
oracle 的 FD gap 只有 `0.2593%`，低于锁定的 `0.5%` 最低差距。虽然该
oracle 的 Hand level gain 为正，且保留 shadow-MAP gain 的 `91.48%`，但这
两项不能覆盖前述阻断条件。

## 已完成

- Hand/Full MAE、native 29-step future-to-future FD、anchor-to-first、TV
  和 native/standardized identity audit；
- base OOF、selector OOF 限制和 normalization caveat audit；
- train/dev mode imbalance、minority recall、可靠性曲线；
- 修正 bootstrap refit 后的 K=2/4/8 ARI audit；
- loss、gradient、codebook/shared-refinement 和 fixed-q component oracle；
- 6 组诊断图、底层数据表、captions 和视觉 QA；
- CPU contract tests：`6 passed`。

## 重要边界

Prototype 的旧 `ARI=1` 是 fixed-center bootstrap 实现造成的非诊断性结果。
重新拟合中心后，K=2 的 ARI median 为 `-0.000677`，95%区间为
`[-0.0120216, 0.971835]`；K=4 为 `0.597605`
`[0.002448, 0.905718]`；K=8 为 `0.723035`
`[0.426393, 0.945854]`。这不支持把旧 K=2 结果表述为稳定模式发现。

Base OOF 覆盖为 `2741/2741`、overlap=`0`，但没有完整的 final-system
selector OOF；prototype fold whitening 还复用了 all-train coefficient
统计量。因此相关结果只能作 development diagnostic，不能升级为严格的
无泄漏 final-system OOF 或泛化结论。

## 资源与停止边界

资源策略保持动态枚举全部可见 GPU、允许共置、利用率仅排序、按真实 peak
加安全余量检查显存、最多两个本项目 child，并且不触碰其他用户进程。本轮
因 CPU gate 已失败，没有必要执行 GPU Stage0，实际 GPU child 数为 `0`。

`FINAL_TEST_ACCESS=0`，没有启动 final/held-out inference，没有联系教授，
没有修改旧实验根目录，也没有启动新的多 seed 或 base decoder 训练。
