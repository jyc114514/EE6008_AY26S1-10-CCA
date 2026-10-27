> Archived study summary for `SEL_U_PROSPECTIVE_CONFIRMATION_V1_20260914T142514Z` (source snapshot `20260914`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# SEL-U 实验决策

状态：`ineligible`；本轮资格复核已完成，Gate A 未通过。

SEL-U 在 development split 的回顾性结果为：Hand MAE `0.237144`
（Persistence `0.257422`，改善 `7.88%`），Full-28 MAE `0.170697`
（固定 Arm-learned/Hand-Persistence hybrid `0.180836`，改善 `5.61%`）。
Hand 改善的 paired 95% CI 为 `[0.013746, 0.027213]`，Full-28 对固定
hybrid 的 CI 为 `[0.006900, 0.013585]`；oracle headroom closure 为
`52.27%`，native-TV relative change 为 `-15.71%`。

但 Hand first-difference relative change 为 `+9.26%`，超过预先锁定的
`+1%` practical limit；其 paired effect（Persistence 减 SEL-U）为
`-0.006604`，95% CI 为 `[-0.007541, -0.005735]`。因此不能启动
3-seed prospective confirmation，也不能因为 GPU1 可用而放宽科学门禁。

本轮没有新训练、GPU Stage0、final/held-out 访问或教授联系。SEL-U 仍是
outcome-exposed 的 single-seed development evidence，不是 final model，
也没有替换冻结的 `JY02-F-R-D`。下一步若获单独批准，应改进 Hand expert
（如 conditional trajectory prototypes、mixture decoder 或 action-phase
conditioning），并重新建立 protocol lock。
