# I-JEPA：图像中的经典 JEPA 实例

## 问题与方法

I-JEPA 从同一图像构造 context blocks 与 target blocks。在线 encoder 只看上下文，predictor 接收上下文 token 和目标位置编码，预测目标 encoder 在目标区域产生的表征。target encoder 不反向传播，由在线 encoder 的指数滑动平均（EMA）更新。

令上下文为 `x`，目标区域为 `y`，在线 encoder 为 `f_theta`，目标 encoder 为 `f_bar_theta`，预测器为 `g_phi`，则核心目标可写成：

`L = mean || g_phi(f_theta(x), pos_y) - stopgrad(f_bar_theta(y)) ||_1`。

这里的关键不是 L1 本身，而是：目标在表示空间；目标分支 stop-gradient；EMA 使目标缓慢变化；多块 mask 迫使模型利用语义和空间上下文。

## 代码锚点

- `repositories/ijepa/src/train.py:169`：复制在线 encoder 得到 target encoder。
- `src/train.py:224-228`：冻结目标参数并建立 momentum schedule。
- `src/train.py:295-319`：target/context 前向与表征损失。
- `src/train.py:332-336`：EMA 更新。
- `src/masks/multiblock.py`：多块 context/target mask 采样。

## 不要误用

官方仓库已归档，根许可证为 CC BY-NC 4.0。它适合解释“两个 encoder + predictor + EMA + mask”的经典骨架，不适合作为新视频项目底座，也没有 EK100 流程。

## 自测

1. 删除 stop-gradient 或 EMA 会改变哪个学习闭环？
2. predictor 为什么需要目标位置编码？
3. I-JEPA 的图像 block 与 V-JEPA 的时空 tube 有何对应关系？

