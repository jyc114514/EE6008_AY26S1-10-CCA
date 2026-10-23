# Core A 结果摘要

Full-120 Stage1 源 acceptance 为 9/9：617 个 development/validation
episodes、8 个 video cells 加 `STATE-CTRL`。当前 dynamic 中按 Top-1 最强为
D-P-L（96.110%）；绝对 video 分数最高为 S-P-L（Top-1 97.083%、macro-F1
92.992%）。D-P-L 的 BF16 merged export 未通过 value-level equivalence，
因此使用 accepted unmerged derived inference。

匹配 effect 不能证明 LoRA 带来 dynamic-specific benefit：pooled Top-1
interaction 为 -0.810 个百分点，attentive 为 -2.269 个百分点。Stage2
未启动。以上历史 Stage1 结果只来自 development/validation，这些 runs 记录
`final_test_access=0`。后续单独的 V2 amended analysis 在披露先前标签访问后，
对 602 个同源 final labels 评分；这不是未触碰标签的盲测确认。见
[V2 分析](../../docs/core_a/FINAL_V2_AMENDED_ANALYSIS.zh-CN.md)。

详见[完整结论](../../docs/core_a/CURRENT_CONCLUSIONS.zh-CN.md)、
[run-level results](../../experiments/core_a/RUN_LEVEL_RESULTS.csv) 和
[figure manifest](figure_manifest.json)。
