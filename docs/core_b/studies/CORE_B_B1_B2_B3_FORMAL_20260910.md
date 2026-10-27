> Historical B1/B2/B3 analysis snapshot (2026-09-10). Its final-test-access statement applies to those earlier development runs only.

# Core B 完整收菜与结论（中文）

分析版本：`CORE_B_UNIFIED_ANALYSIS_20260909T163929Z`；生成时间：2026-09-10T07:01:43Z。正式分析范围为 G1 train=2741、development=614、target=(30,28)，final-test access=0。

## 验收状态

- B1 canonical inference：7/7 accepted，未重训；
- true-pixel B2：6/6 accepted；旧 contextual-token sparse B2 单独标为 proxy，不进入正式选择；
- corrected B3：8/8 accepted，统一使用 dense 0.125s/32 pixel observations、pooled head、seed 20260903 和 corrected train-only feature normalization；
- Core B 的 native command-value 单位仍未被物理语义/URDF 审计确认，因此 native 数值只作为 source-native command-value error 报告。

## 主要结果

- B1 最低 standardized arm14 MAE：B-STATE = 0.165614；persistence = 0.126969。
- true-pixel B2 最低 standardized arm14 MAE：B2-FUS-0125 = 0.173250；正式选择仍为 dense 0.125s/32，稀疏组没有形成合格效率/精度候选。
- B3 八格中最低 standardized arm14 MAE：B3-FUS-D-L = 0.169159。
- B3-FUS-D-L 相对 STATE 的 improvement（state MAE−fusion LoRA MAE）为 -0.003545，95% CI [-0.006132, -0.001041]；相对 persistence 为 -0.042191，95% CI [-0.048655, -0.035710]。

## 解释边界

B1/B2/B3 的 learned trajectory error 必须与 persistence、STATE、video-only 和 fusion 同时阅读。若 learned 误差仍高于 persistence，只能说明当前 head/target parameterization 没有超过强 naive control；不能据此否定视觉 encoder 的表示能力。

B2 的 sparse 结论只来自真正像素级采样；旧 proxy 只说明 dense contextual representation 的下游抽稀/聚合行为，不能说明 encoder 只观察了 16/8 帧。

B3 是 single-seed exploratory matrix；episode bootstrap 不是 multi-seed confirmation。B3 candidate gate、若未通过，不启动 fresh-seed confirmation。

## 失败与工程保留

早期 manifest、module-count、缺 feature normalization 和纯 writer 错误均保留在 failed-attempt evidence，不计入 formal matrix。LoRA canonical inference 使用 unmerged base+adapter；没有把工程成功、GPU 利用率或单个 child 完成当作科学完成证明。

详细的 B1/B2/B3 指标、per-time/per-joint/per-task、paired bootstrap、Holm、轨迹和图表底层数据均位于本版本目录。
