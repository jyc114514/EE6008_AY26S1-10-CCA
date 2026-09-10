# Core A 局限与剩余证据缺口

- 当前 Full-120 Stage1 只有一个训练 seed。617 个 development/validation
  episodes 的 episode bootstrap 区间只量化配对不确定性，不能证明跨 seed
  稳定性。
- 恢复运行使用 effective batch 32（physical batch 8 × accumulation 4），
  与锁定的 batch 48 协议不同。
- D-P-L 的 source run 未通过 BF16 merged-export equivalence diagnostic，且
  缺少 source run-level metrics 文件；公开科学数字来自独立复算的 unmerged
  derived path。
- dynamic/static 结果随 head 混合，不能证明因果 dynamics understanding。
  static repeated-first-frame 是 shortcut control，不是对所有视觉信息的
  完整 counterfactual。
- state-only 显示明显 shortcut signal，但历史 state-only 与当前 STATE-CTRL
  的 prediction stream 不同，不能进行 paired 比较。
- 12-task frozen/LoRA pilot 只属于工程/探索证据，不能外推到 120 类。
- Batch-4/32/48、R3D-18、pooled、attentive 和 nonparametric 行属于不同
  protocol，不能合并成一个排行榜。
- 没有纳入或使用 final-test IDs、labels、predictions 或 metrics。
- raw data、features、checkpoints、完整 logs、环境目录和第三方源码树均被
  有意排除，因此不声称公众可以端到端重训。
- 数据、模型和第三方源码许可需要分别治理；本分支不为未确认再分发权利的
  材料新增 blanket license。

## 剩余工作

下一步科学决策是是否授权锁定协议的 multi-seed confirmation。本发布包没有
启动 Stage2；不能因为仓库中存在计划或脚本就推断已经产生新训练结果。
