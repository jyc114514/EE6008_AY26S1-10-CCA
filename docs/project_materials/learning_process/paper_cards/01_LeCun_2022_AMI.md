# LeCun 2022：A Path Towards Autonomous Machine Intelligence

## 一句话

这是一份研究纲领：智能体不应把世界建模成像素级逐帧生成，而应在抽象表征空间中学习可预测的世界状态，并用代价、记忆与规划模块选择行动。

## 阅读抓手

- **World model**：输入当前状态表征与候选动作，预测未来状态表征。
- **JEPA**：上下文 encoder 与目标 encoder 产生 embedding，predictor 预测目标 embedding；预测可以是集合/能量，而非唯一像素答案。
- **Energy-based model**：用标量能量表示变量组合的相容性；低能量代表更相容。
- **为什么不只做生成**：未来存在不可预测细节，像素损失会迫使模型花容量拟合纹理、噪声和多模态平均。

## 对 EE6008 的意义

它提供“为何预测表征”的动机，但不是 EK100 工程说明。组内解释 JEPA 时先用它建立直觉，再用 I-JEPA/V-JEPA 的具体目标和代码消除抽象性。若导师没有要求控制/planning，不要把整套自主智能体架构扩张为本学期交付范围。

## 证据状态

官方 OpenReview 页面：<https://openreview.net/forum?id=BZ5a1r-kVsf>。本次官方下载被浏览器验证阻挡，没有在 `papers/` 中伪造或以第三方镜像替换。讲义中的概念性解释与具体实验结果严格分开。

## 自测

1. 为什么“预测 embedding”不等于“把视频压缩后再做像素回归”？
2. 哪些任务真的需要动作条件 world model，哪些只需要视觉动作预期？

