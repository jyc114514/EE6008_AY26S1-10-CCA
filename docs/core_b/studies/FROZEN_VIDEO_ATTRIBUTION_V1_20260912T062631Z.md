> Archived study summary for `FROZEN_VIDEO_ATTRIBUTION_V1_20260912T062631Z` (source snapshot `20260912`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Frozen-only 视频归因决定

状态：`COMPLETE`，但仍是 development-only、single Frozen checkpoint pair、outcome-informed exploratory evidence。

Fusion chronological matched 的 Arm-14 full MAE 为 `0.17325`。将视频 feature 置零后为 `0.25254`，damage `+0.07929`；cross-task mismatch 为 `0.32065`，damage `+0.14740`；same-task mismatch 为 `0.17696`，damage `+0.00887`（N=589，单例 task recipient 按 NA 处理）。这支持“Fusion 确实使用视频表示”，但不支持“视频让总体预测优于 state/persistence”：`B-STATE=0.16561`，`PERSISTENCE=0.12697`。

Video-only 也表现出明显控制敏感性：chronological `0.29809`，repeated-first `0.80452`，repeated-last `0.74493`，reversed `0.42224`，shuffle `0.52587`。Fusion 中 repeated-first/last 的损失大于 reverse/shuffle；因此证据更符合“最新/多帧视觉状态 + temporal order 都被使用”，而不是单纯 latest-frame 或完全 order-invariant。

Same-task mismatch 接近 matched，而 cross-task mismatch 明显恶化，说明 task/scene shortcut-like dependence 仍然存在；但 same-task mismatch 的小幅正 damage 表明不能把它简化为纯 task identity。所有机制表述均为 evidence-consistent interpretation，不是 causal proof。

High-motion 中 Fusion matched full Arm-14 MAE `0.22760`，B-STATE `0.22601`，Persistence `0.23405`；endpoint 分别为 `0.29544`、`0.30741`、`0.40872`。这是小的描述性 matched increment，不能升级为 seed-robust high-motion claim。Fusion matched 的 native TV ratio `6.99`，zero `7.79`，cross-task `11.19`；first-difference 和 TV 仍显示轨迹振荡/变化过多的代价。

```ini
VIDEO_ATTRIBUTION=useful
TEMPORAL_ORDER_SIGNAL=present
VISUAL_FRESHNESS_SIGNAL=present
TASK_SCENE_SHORTCUT_SIGNAL=present
HIGH_MOTION_VIDEO_INCREMENT=small_descriptive_not_seed_robust
ENDPOINT_VIDEO_INCREMENT=matched_better_than_state_persistence_descriptive
FIRST_DIFFERENCE_EFFECT=matched_better_than_repeat_mismatch_but_variation_remains_high
CARTESIAN_TRAJECTORY_STATUS=not_applicable_unverified_kinematics
```

Frozen Mainline 已完成 `13/13`：2/2 exact reuse、11/11 new inference。旧 Route C 仍 `blocked`，没有 partial 统计被移入旧矩阵。LoRA 不影响本结论。
