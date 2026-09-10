# Core A 实验脉络与研究理由

## 证据边界

本 catalog 依据 project source package、v3 split/data audit、带 hash 的
run manifest、Full-120 Stage1 acceptance 和封存的 unified analysis 整理，
版本截至 2026-09-10。公开文件只保留 aggregate 数字或稳定的
`server-only://` 指针，不代表所有服务器 artifact 都可再分发。

Core A 的问题是：四秒 observed RGB prefix 能否支持 G1 子集的 task-ID
recognition。它不评估 future action prediction。比较不同 family 时必须
同时保留 task 数、representation、head、adaptation、split 和重复单位。

## Family 覆盖

| Family | 研究问题 | 计划 / 执行 / 验收 | 边界 |
| --- | --- | ---: | --- |
| G1 data 与 duplicate-safe v3 | 能否在不使用 final-test 指标的情况下审计 G1 population 和 split？ | 1 / 1 / 1 | 所有模型的共同前置；2,757 train、617 development-validation、120 个 primary classes；final-test 未触碰。 |
| State-only control | 只使用 state 能达到多少 task recognition？ | 1 / 1 / 1 | shortcut control，不是 video understanding。 |
| Frozen R3D-18 reference | 常规 frozen video reference 是什么？ | 1 / 1 / 1 | 历史 development/validation reference；不能分离 dynamic/static。 |
| Frozen pooled V-JEPA 与 feature geometry | frozen feature 是否已有 task structure，dynamic 是否有帮助？ | 1 / 1 / 1 | 历史 pooled dynamic/static 与 1-NN/centroid reference；不是 encoder 的因果证据。 |
| Frozen attentive dynamic/static | attentive head 是否优于 frozen reference？ | 2 / 2 / 2 | 历史多 seed aggregate，与 pooled 结果分开。 |
| 12-task frozen 与 Top-2 LoRA pilot | LoRA seam 和训练路径能否在小 pilot 上运行？ | 2 / 2 / 2 | 仅 exploratory，不能推广为 120-class 证据。 |
| Batch-4/32/48 screening | optimization 与单因素 head/loss 选择会如何影响 frozen 结果？ | 6 / 6 / 6 | 历史 aggregate，误差为 seed-level SD；不做统一排行榜。 |
| Full-120 LoRA Stage1 | 在匹配的 pooled/attentive、dynamic/static 条件下，LoRA 是否带来 dynamic-specific gain？ | 9 / 9 / 9 | 单 seed；恢复后 effective batch=32；D-P-L 只在 unmerged derived path 上验收。 |
| Full-120 LoRA Stage2 | 是否值得进行 gated multi-seed confirmation？ | 12 / 0 / 0 | 未启动，因为锁定的 dynamic-specific gate 未满足。 |
| Engineering validation | checkpoint、merge、writer、schema、ID 和 metric seam 是否可审计？ | 1 / 1 / 1 | QA 证据，不是模型性能结果。 |

## Full-120 Stage1 矩阵

8 个 video cells 交叉比较：

- dynamic 与 repeated-first-frame static input；
- pooled 与 attentive head；
- frozen 与 top-block LoRA adaptation。

`STATE-CTRL` 是独立的 state-only control。源 acceptance 记录为
`accepted=9`、`required=9`、`final_test_access=0`。公开 run-level 表保留
cell 指标、source run status、可用的 selected epoch、precision、effective
batch 和 artifact hashes。请以
[RUN_LEVEL_RESULTS.csv](../../experiments/core_a/RUN_LEVEL_RESULTS.csv) 与
[ACCEPTANCE_SUMMARY.csv](../../experiments/core_a/ACCEPTANCE_SUMMARY.csv)
为准，不要根据文件名重建排行榜。

## Family 之间的关系

data/QA family 固定 population 与 duplicate-safe v3 contract；历史 frozen
与 geometry family 提供 reference；12-task pilot 只验证 seam，不能作为
Full-120 factorial 证据；Batch-size screening 说明 optimization 会改变
dynamic/static gap；Full-120 Stage1 因而作为单 seed matched screen 报告，
Stage2 仍是 proposal，不是执行结果。

## 指标与不确定性

主要指标为 Top-1、Top-5、macro-F1；NLL、ECE、Brier 和 mean class recall
为次要指标。Stage1 effect 表使用 10,000 次 paired episode bootstrap，
并按源协议用 exact two-sided McNemar 与 Holm 校正。重采样单位是 episode，
不是 time step 或 trajectory dimension。历史 baseline 的误差线是源文件中的
seed-level sample SD，不能和 Stage1 episode bootstrap CI 混用。
