# V-JEPA：从二维块预测到时空潜变量预测

## 核心改变

V-JEPA 将输入变成视频 clip，并在时空 token 网格上采样 masks。在线 encoder 看可见时空上下文，predictor 预测被遮挡位置的 target-encoder features。模型仍不重建 RGB，因此学习压力集中在跨时间稳定、可预测的语义与运动信息。

## 训练闭环

1. 视频增强后形成 `[B,C,T,H,W]`。
2. 3D patch/tubelet embedding 形成时空 token。
3. target encoder 产生完整目标特征并选取 mask 位置。
4. context encoder 只编码可见 token。
5. predictor 用 context + mask tokens 预测目标表征。
6. 表征损失训练在线 encoder/predictor；target encoder 用 EMA 更新。

## 代码锚点

- `repositories/jepa/app/vjepa/train.py:222`：target encoder 副本。
- `app/vjepa/train.py:227-236`：multi-block 3D 或 tube mask collator。
- `app/vjepa/train.py:419-456`：target、context、prediction loss。
- `app/vjepa/train.py:483-488`：EMA。

## 对项目的角色

它是理解视频 JEPA 的最佳过渡，但官方 `vjepa2` 已成为新工程主体。项目不需要先完整复现 V-JEPA 预训练；先理解张量与 mask，再进入 V-JEPA 2 EK100 probe。

## 自测

1. `T`、tubelet size 与 token 时间长度如何换算？
2. 为什么高 mask 比例可能迫使模型学习更长程语义？
3. frozen evaluation 测的是表征可读性还是端到端最优能力？

