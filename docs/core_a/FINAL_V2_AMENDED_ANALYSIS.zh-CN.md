# Core A V2 修正后分析

**状态：** 同一来源上的修正分析已完成。该评分所用标签此前已经被访问，因此不是未触碰标签的盲测确认。

V2 修正了 120 类映射，并在同一 602 行 primary final population 上重新评分六个 frozen/LoRA、dynamic/static 预测 cell。本次评分没有启动训练、feature extraction 或 model forward。第 1 次评分在产生指标前停止；第 2 次评分完成六个 cell 和 5,000 次共享配对 bootstrap 抽样。独立 CPU 算术/来源核查通过。

## 结果

以下数值是同一组三个固定 seed 的均值。主 estimand 为 `A-FULL602`。由于模型类别集合与 final 类别集合相同，class-set-matched 替代口径不适用。

| Program | Top-1 accuracy | Top-5 accuracy | 120 个真实类别上的 Macro-F1 |
| --- | ---: | ---: | ---: |
| A_DYNAMIC | 0.950166 | 0.988372 | 0.861032 |
| A_STATIC | 0.931894 | 0.996124 | 0.878978 |
| Dynamic 减 static | +0.018272 | -0.007752 | -0.017947 |

配对 grouped-bootstrap 区间分别为：Top-1 `[+0.002769, +0.034330]`；Top-5 `[-0.015504, -0.001107]`；Macro-F1 `[-0.027277, +0.022766]`。这些区间以指定同源 population 与固定 checkpoint 为条件。不同指标的方向不一致，不支持宽泛的 dynamic-video 改善结论。

## 暴露与来源

- Final 标签访问早于本次修正。V2 两次评分都读取同一 602 个 primary labels；第一次未产出指标或 bootstrap 抽样。
- 映射修正来自已接受的训练来源和 train/development targets。预测、ID、权重及训练 seed 未改变。
- 原始实验运行时 Git commit 为 `UNKNOWN`：服务器代码目录不是 Git checkout。本次发布记录导出代码的 SHA-256 文件哈希。
- 逐 seed、聚合和配对结果见 [per-seed 表](../../experiments/core_a/FINAL_V2_PER_SEED.csv)、[汇总表](../../experiments/core_a/FINAL_V2_SUMMARY.csv)、[配对对比](../../experiments/core_a/FINAL_V2_PAIRED_CONTRASTS.csv)。

原 Core A Stage1 仍是独立的 development/validation 屏幕结果，见 [历史结论](CURRENT_CONCLUSIONS.zh-CN.md)。本次 V2 amended score 不替换或改写这些历史记录。
