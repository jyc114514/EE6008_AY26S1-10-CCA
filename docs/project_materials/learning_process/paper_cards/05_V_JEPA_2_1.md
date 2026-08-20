# V-JEPA 2.1：密集特征增强，但先做小模型验证

## 核心动机

全局语义强并不保证每个时空 token 都足够有用。V-JEPA 2.1 通过更密集的学习信号、深层监督与多模态 tokenizer 等设计，让局部/密集特征更可用，同时保留 JEPA 的表征预测范式。

## 项目相关结果

论文报告 EK100 action anticipation 的 R@5 达到 40.8（应回到论文表格核对模型、split 与 protocol）。这是选择 2.1 作为 JFAA backbone 的理由之一，但不是我们无需复现实验设置就能声明达到的成绩。

## 工程入口

- 训练：`repositories/vjepa2/app/vjepa_2_1/`。
- 评估配置：`configs/eval_2_1/{vitb-384,vitl-384,vitG-384}/ek100.yaml`。
- 当前仓库的权重 key、模型名和大小写目录需按具体 YAML 检查。

## 风险

ViT-G/384 计算和显存压力最高；JFAA 默认依赖它，不适合作为第一天 smoke test。先用官方更小模型/更短数据清单确认数据、张量、指标和 checkpoint 加载，再决定是否上 ViT-G。Windows 检出的 `vitG`/`vitg` 大小写冲突进一步要求 Linux 上重新克隆。

## 自测

1. “dense feature 更强”为什么可能帮助 action anticipation probe？
2. 若 ViT-B smoke test 正常、ViT-G 失败，如何区分代码错误与资源问题？

