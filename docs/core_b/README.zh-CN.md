# Core B 研究与结果

[English](README.md) | 简体中文 · [项目介绍](../../README.zh-CN.md)

本目录汇集 Core B 的独立 study 摘要、当前 V2 结果边界和便携复现说明。不同实验线路必须分别解读，不能把某一组的访问或结果声明推广到所有 study。

## 研究线路

已编目的线路包括 B1/B2/B3、history×video、JY-02、V3 multi-anchor、V3 state×video、observation delay、fixed-model video correspondence、D3、V4–V9、VideoMAE handoff 与 CURRENT_PROFILE_V2。完整日期化记录见[实验线路索引](EXPERIMENT_LINE_INDEX.md)及[study 摘要目录](studies/)；聚合表见 [experiments/core_b](../../experiments/core_b/)。

先读[当前结果与局限](CURRENT_RESULTS_AND_LIMITATIONS.md)。其中记录 CURRENT_PROFILE_V2 的 development 和 final scoring 使用此前已训练的 checkpoint，没有新增训练。每个 split 记录了 22 个 prediction bundles；final targets 仅在 final prediction seal 通过后读取。

| Split | Episode / anchor 数 | V0 position MAE | Persistence position MAE |
| --- | ---: | ---: | ---: |
| Development | 47 / 185 | 0.158397 | 0.222262 |
| Final recovery | 41 / 162 | 0.178383 | 0.238057 |

这些是标准化指标，不代表机器人实际精度。41 个 final episodes 与 Core A 编目视频群体重叠，session 独立性未得到证明；这不是独立确认。原始 Q0 bridge 仍为 `FAIL_UNRESOLVED`，原因是 `UNKNOWN`。

## 访问与数据边界

B1/B2/B3、V3、D3、V4–V9 及其他早期 study 分别保留其对应的 development-only 范围记录。Core A V2 与 Core B CURRENT_PROFILE_V2 后续 final scoring 的标签访问事件另有披露；不要把较早 study 中的 `final_test_access=0` 当作后续 study 的整体结论。

服务器保留原始视频、源 Parquet、特征 tensor、checkpoint、完整预测/目标数组、完整训练输出和日志。这些文件未包含在公开目录中；受限工件类别与索引见定期发布材料。公开聚合表和图不能替代原始数据或运行工件。

## 复现边界与链接

公开源代码包含训练和推理入口，但本次发布没有执行这些入口。完整实验需具体 study 的获授权输入、checkpoint、运行锁、外部运行环境和服务器工件；脚本本身不表示全流程能从公开 clone 独立复现。

- [便携代码路径与入口要求](PORTABLE_REPRODUCTION.md)
- [当前结果与局限](CURRENT_RESULTS_AND_LIMITATIONS.md)
- [日期化 study 摘要](studies/)
- [Core B 实验线路索引](EXPERIMENT_LINE_INDEX.md)
- [发布结果和来源索引](../releases/server-sync-20261022T082418Z/EXPERIMENT_RESULT_INDEX.csv)
- [服务器同步发布摘要](../releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md)
