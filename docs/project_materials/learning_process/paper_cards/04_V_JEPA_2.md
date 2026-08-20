# V-JEPA 2：本项目的工程主体

## 论文贡献分层

V-JEPA 2 把大规模视频自监督表征、未来状态预测和动作条件规划放在一个体系中。对 EE6008 最直接的是视觉 encoder/predictor 与 EK100 action anticipation frozen-probe；2-AC/DROID 与机器人 planning 是可选扩展，不应自动等同于课程最低要求。

## EK100 官方路径

- 配置：`repositories/vjepa2/configs/eval/vitl/ek100.yaml`、`configs/eval/vitg-384/ek100.yaml`。
- 入口：`evals/main.py` / `evals/main_distributed.py`。
- 数据：`evals/action_anticipation_frozen/epickitchens.py` 和 `dataloader.py`。
- 模型：`models.py` 的 attentive classifier，以及 `modelcustom/vit_encoder_predictor_concat_ar.py` 的 encoder–predictor wrapper。
- 训练/指标：`eval.py`、`losses.py`、`metrics.py`。

## 当前配置读法

以 V-JEPA 2.1 ViT-L/384 EK100 配置为例：`frames_per_clip=32`、`frames_per_second=8`、`resolution=384`、`num_output_frames=2`、`use_v2_1=true`，encoder checkpoint key 为 `ema_encoder`，predictor key 为 `predictor`。这些字段决定实际输入、时间跨度、权重字段和未来 token 数，不能只改路径就开跑。

## 项目策略

首个里程碑是“官方配置 + 小样本 + frozen encoder/predictor + 单一 attentive probe + 指标输出”。在这个闭环通过前，不进行全量特征缓存、多模型 ensemble 或 encoder 微调。论文报告的 EK100 数字只作为外部参考，项目结果必须用相同 split、metric 和 inference protocol 才能比较。

## 自测

1. 32 帧、8 fps 表示多长观察窗口？
2. 为什么 encoder 与 predictor 都可能需要 checkpoint？
3. `anticipation_time_sec` 如何进入未来 token 的位置计算？

