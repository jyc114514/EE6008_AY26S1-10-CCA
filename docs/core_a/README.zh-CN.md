# Core A 证据发布包

本目录整理 Core A：G1 Humanoid Everyday 的 partial-observation task
recognition。本文档版本截至 2026-09-10，主要依据已封存的
`CORE_A_UNIFIED_ANALYSIS_20260909T052330Z` 以及更早、带 hash 的 Core A
记录。

公开科学任务是根据四秒 observed prefix 进行 task-ID recognition。它不是
action anticipation 结果，也不代表最终泛化：所有模型指标都只来自
development/validation，且 `core_a_final_test_access=0`。

建议入口：

- [当前结论](CURRENT_CONCLUSIONS.zh-CN.md)
- [实验脉络与覆盖](EXPERIMENTS_AND_RATIONALE.zh-CN.md)
- [数据与评估协议](DATA_AND_EVALUATION_PROTOCOL.md)
- [复现指南](REPRODUCTION_GUIDE.md)
- [工程验证](ENGINEERING_VALIDATION.md)
- [局限](LIMITATIONS.zh-CN.md)
- [公开清单与排除项](publication/PUBLICATION_MANIFEST.md)

源码在 `src/ee6008/`，可移植 launcher 和审计脚本在 `scripts/core_a/`，
小型 aggregate 表格和图在 `results/core_a/`。表格中的 server-only artifact
ID 是稳定的定位指针，不是随仓库分发的文件。
