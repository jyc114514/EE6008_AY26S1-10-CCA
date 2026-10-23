> Archived study summary for `HAND_TEMPORAL_FIDELITY_V1_20260914T152845Z` (source snapshot `20260914`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Hand temporal-fidelity 决策

状态：`gate_b_fail`，Route=`NONE`。

SEL-U 的 level accuracy 优势在 counterfactual 中保持：D0-G0 Hand MAE
`0.237144`，D1-G1 为 `0.236936`；对应 Full-28 MAE 为 `0.170697` 和
`0.170593`。但 temporal fidelity 的修复不足：原始 D0-G0 native FD harm
为 `0.00193377`，固定 DCT D1-G0 只减少 `44.84%`，DCT+smoothing D1-G1
只减少 `45.15%`，低于预先锁定的 `50%` 门禁。固定 gate smoothing 单独
只减少约 `0.34%`。

signed decomposition 在 native command space 精确成立；term energy 中
residual-motion 约 `92.85%`，gate-switch 约 `7.15%`。`Delta q` 的
sign-change 和 extreme-to-extreme switch rate 均为 `0`。这些证据支持
“raw residual temporal structure 是主要问题”，但不证明任意 raw residual
decoder 都能修复轨迹，也不把 term energy 当作因果 attribution。

因此不启动 single-seed screen、GPU Stage0 或 multi-seed。下一步唯一建议
是另行预注册并设计 Hand expert 的 conditional trajectory prototypes、
mixture decoder 或 action-phase conditioning；本 root 不再追加 kernel、K、
lambda、loss 或 seed sweep。所有结果仍是 development-only、
outcome-exposed，不是 final/held-out 结论。
