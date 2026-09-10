# Core A 工程验证

工程验证与模型性能分开报告。

## 代码与 import closure

`src/ee6008/` 中的 project-authored package 覆盖 config、dataset
inventory/selection/split、clip preprocessing、feature cache、model loader
seam、pooled/attentive head、metrics、probe、LoRA 和 top-block forward helper。
Core A 脚本通过 `PYTHONPATH=src:scripts/core_a` 导入这些模块。V-JEPA 实现
本身是外部依赖，本仓库不 vendoring。

公开副本的 `data/clips.py` 将 decord 作为 optional dependency；当其 C++ 扩展
无法导入时使用项目已记录的 OpenCV fallback。这个改动只是 portability guard，
在 decord 可用时不改变 decoder 选择。

LoRA 只向 blocks 10/11 的 `qkv` 与 `proj` 注入 adapter，冻结 base parameters，
并将 `lora_B` 零初始化。聚焦测试覆盖 zero-init 等价性、frozen boundary、
top-block forward、token shape guard、early-stopping tie-break 和 final-test
split guard。

## Checkpoint 与 merge seam

只读 replay 证据记录：

- FP64 algebraic merge 最大绝对差：`1.687538997430238e-14`；
- best checkpoint 的 U32/M32 最大绝对差：`4.053115844726562e-06`，超 tolerance
  数为 0；
- last checkpoint 的 U32/M32 最大绝对差：`6.794929504394531e-06`，超 tolerance
  数为 0；
- BF16 merged 与 unmerged 最大绝对差：`0.0625`，best/last 分别有 4/5 个值超
  tolerance；
- BF16 merged/unmerged argmax disagreement 记录为 0，但由于 value-level
  equivalence 失败，merged export 仍不支持。

因此 D-P-L 使用 `unmerged_base_plus_adapter` inference。checkpoint 和
prediction hashes 只作为 server-only artifact metadata 记录，文件不随仓库
分发。

## Writer、schema 与 metric 检查

源 acceptance 检查了 finite logits、唯一 episode IDs、labels、predictions、
argmax 一致性、strict model/head load 和 development-only selection。unified
analysis 又独立复算 aggregate metrics，并明确记录 D-P-L 缺少 source
`RUN_METRICS.json`，没有用占位 0 补写。

源 analysis QA 的 cross-file checks 已通过，包括 9 个 current cells、row
counts、bootstrap shape/range、Stage1 acceptance、no-new-training、Stage2
未启动、图表链接、visual QA、final-test lock 和 state reconciliation。本
clone 的 figure QA 会再次执行。

## 解释边界

工程检查通过只说明 seam 或 artifact relationship 一致；不等于单 seed
development result 已成为 final result，不会修复 BF16 merged-export failure，
也不构成因果 dynamics understanding 证据。
