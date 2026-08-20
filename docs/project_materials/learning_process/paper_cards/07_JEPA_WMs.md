# JEPA-WMs：动作条件世界模型（条件阅读）

## 任务变化

普通视频 JEPA 从观察预测被遮挡/未来表征；JEPA world model 进一步把动作或控制序列作为条件，预测执行候选动作后的潜在状态。规划器可以在潜空间滚动候选动作，按目标代价选动作。

形式上可写为 `z_{t+1}=F(z_t,a_t)`，多步 rollout 为 `z_{t+H}=F(...F(z_t,a_t),...,a_{t+H-1})`。规划优化的是目标代价与动作正则，而非要求生成可观看 RGB。

## 代码锚点

- `repositories/jepa-wms/src/models/ac_predictor.py`：action-conditioned predictor。
- `app/vjepa_wm/train.py`：world-model 训练闭环。
- `evals/simu_env_planning/`：仿真环境 planning 评估。
- `app/plan_common/datasets/droid_dset.py`：DROID 数据接入。

## 是否进入 EE6008 主线

只有当导师明确要求 humanoid/robot action-conditioned future prediction、control 或 planning，且提供/认可带动作与状态的数据，才深入本仓库。若任务只是从第一视角人类视频预测下一动作，JEPA-WMs 会增加动作空间、动力学数据、规划 horizon 与闭环评估，属于明显扩项。

## 自测

1. action anticipation 标签为何不能直接充当低层机器人控制 action？
2. open-loop 表征预测与 closed-loop planning 的评估有何本质差异？

