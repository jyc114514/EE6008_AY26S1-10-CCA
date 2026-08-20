# JFAA：EK100 竞赛增强方案

## 身份与边界

JFAA 是论文作者的第三方仓库，不是 Meta 官方项目。README 明确称其为 EgoVis 2026 EK100 action anticipation 冠军方案的 “code scaffold”。仓库给出 probe 训练、评估、提交导出与 ensemble 脚本，但不包含 V-JEPA 2.1 ViT-G/384 权重、EK100 annotations 或 RGB frames。

## 方法结构

1. 冻结 V-JEPA 2.1 encoder/predictor，抽取当前上下文与多个未来位置的 token。
2. 用 attentive pooling/probe 将 token 聚合为 verb、noun、action 相关 logits。
3. 训练多个 epoch/field 的 probe。
4. 以 field-aware 方式融合预测并导出挑战格式。

这种路线把风险集中在数据对齐、权重加载、probe 训练和 ensemble 规则，而不是重新预训练 1B backbone。

## 本项目最小实现

先复用 `vjepa2` 官方 EK100 单 probe 得到可复现基线；再把 `JFAA/train_jfaa_probe.py` 的多头/多时刻设计逐项移植。每加入一个组件都保留消融：base probe → +future token → +field heads → +ensemble。不要一开始就把最终 ensemble 当成单模型改进。

## 指标

仓库写明官方 `Score` 是 action Mean Top-5 Recall，并报告 open test 第一。该成绩是挑战提交结果，不能由本地仓库存在性自动复现；复现必须匹配数据版本、checkpoint、epoch 选择和融合规则。

## 自测

1. 为什么 code scaffold 不等于 incomplete/unusable，也不等于 turnkey？
2. ensemble 提升是否能证明 backbone 更强？
3. 如何设计消融隔离 future tokens 与 field-aware ensemble 的贡献？

