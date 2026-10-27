> Archived study summary for `CORE_B_V8_CONTROLLED_REPLAY_V1_20261016T045154Z` (source snapshot `20261016`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V8 同 runner 硬标签对照与训练路径核验：最终交付

**状态：完成。** 分支 **B_SYNCHRONOUS_VIDEO_9** 在任何新 dev target 读取前锁定：`GT_RETRAIN (Z=Y)`、`SELF_SOFT (Z=0.5Y+0.5V)`、`SV_SOFT (Z=0.5Y+0.25S+0.25V)` × seeds 202610131/132/133。9/9 fits 完成 epoch 160、每 fit 3,200 updates；train/dev model-only bundles 均 9/9。

## 结论

V8 `GT_RETRAIN` 的三个 hard-video checkpoint 与旧 `V_MATCHED` checkpoint 的 model-file SHA 和 final state hash **完全相同**；train prediction 完全相同，dev prediction 均通过原 `atol=rtol=1e-6` replay tolerance。因此，VIDEO 硬标签下的新旧 baseline 路径等价已闭合。paired `HARD_VS_OLD_V = L(old_V)-L(GT_RETRAIN)` 的 Position/Safe-FD/Seam 点估计分别为 `-4.39e-10 / +5.21e-09 / -4.05e-08`；95% CI 分别为 `[-1.15e-09, +2.78e-10] / [-1.15e-09, +1.17e-08] / [-7.59e-08, -7.75e-09]`，最大绝对差小于 `4.1e-8`。

V8 同 runner 下，SELF/SV soft targets 没有重现 V7 的巨大共同退化，但相对 hard 的 Position/Safe-FD 小幅变差，Seam 改善；SV 相对 SELF 没有稳定全指标优势。固定 beta 不是 V7 巨大软学生塌陷的充分解释。V7 as-run source hash/RNG/batch stream 不完整，确切旧路径差异仍未知；没有发现具体 V7 训练或 scoring bug。当前结果只覆盖 VIDEO，没有 SS 或 STATE 同步组。

## 零 optimizer 数据与训练路径核验

- 1,280 个 outer-train anchor 的 Y/T/Z、identity、prefix/known-anchor 和 accepted frame mapping 检查覆盖 7,680 个 VIDEO program/seed 行；共享输入字段 11/11 精确匹配，Z native mixture 最大差 `0`，固定 normalizer native→model→native 最大差 `2.22e-16`。真实 frame 构造由 V2 接受的 `anchor+1 … anchor+30` 源实现支持；采集时钟/物理单位仍未知。
- SELF 教师 `T` 与对应同构 student adapter 的三 seed full-train replay、V7 SELF 学生回放各 3,840 rows，最大 abs diff `0`，沿原 `atol=rtol=1e-6` 通过，state hash 不变。存储 native T 上 `L(T,T)=0`、`L(T,Z)=0.5L(T,Y)` 的 FP64 position 与 29-difference identities 均通过 `atol=1e-12, rtol=1e-10`；Z loss 非零符合公式。
- 五任务真实 batch（64 rows、cross-episode、不等 sample weights）上，独立 weight 公式和 runner 差 0；helper 对 native reference 的最大 per-sample Position 差 `1.49e-08`、29-difference 差 `1.19e-07`、gradient abs 差 `1.86e-08`、两步临时 AdamW 参数 abs 差 `1.80e-07`。真实 batch、dropout RNG、clip norm 和 scheduler post-step 位置均核验。

## Five-task dev macro（mean ± sample SD）

| Method | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |
|---|---:|---:|---:|
| GT_RETRAIN | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| SELF_SOFT | 0.159634 ± 0.000831 | 0.824238 ± 0.002065 | 0.787120 ± 0.011804 |
| SV_SOFT | 0.159148 ± 0.000449 | 0.824962 ± 0.001039 | 0.783478 ± 0.018068 |
| V | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| A | 0.159871 ± 0.001343 | 0.829451 ± 0.002901 | 0.776170 ± 0.015822 |
| S | 0.157555 ± 0.001035 | 0.830807 ± 0.003094 | 0.766943 ± 0.005265 |
| SV50 | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |
| SS50 | 0.157003 ± 0.001060 | 0.825177 ± 0.000320 | 0.761043 ± 0.008338 |
| PERSISTENCE | 0.222262 ± 0.000000 | 0.854193 ± 0.000000 | 0.802417 ± 0.000000 |

## 每 seed 新模型

| Program | Seed | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |
|---|---:|---:|---:|---:|
| GT_RETRAIN | 202610131 | 0.159213 | 0.820959 | 0.818169 |
| GT_RETRAIN | 202610132 | 0.159179 | 0.821959 | 0.775166 |
| GT_RETRAIN | 202610133 | 0.156622 | 0.816346 | 0.793948 |
| SELF_SOFT | 202610131 | 0.159597 | 0.825272 | 0.799802 |
| SELF_SOFT | 202610132 | 0.160484 | 0.825581 | 0.776453 |
| SELF_SOFT | 202610133 | 0.158823 | 0.821860 | 0.785104 |
| SV_SOFT | 202610131 | 0.159651 | 0.825521 | 0.804301 |
| SV_SOFT | 202610132 | 0.159002 | 0.825603 | 0.771942 |
| SV_SOFT | 202610133 | 0.158790 | 0.823763 | 0.774191 |

## Paired target gains 与 95% CI

正值表示 candidate loss 更低。区间是固定模型条件下的 5,000 次 within-task source-group bootstrap（seed 20261016）。

| Contrast (reference − candidate) | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |
|---|---:|---:|---:|
| SELF_TARGET_GAIN | -0.001296 [-0.002539, -0.000134] | -0.004483 [-0.006820, -0.002160] | +0.008641 [+0.000045, +0.018069] |
| SV_TARGET_GAIN | -0.000809 [-0.002104, +0.000547] | -0.005208 [-0.009057, -0.001920] | +0.012283 [+0.001725, +0.023577] |
| SV_VS_SELF | +0.000487 [-0.000721, +0.001710] | -0.000724 [-0.004407, +0.002738] | +0.003642 [-0.004129, +0.012032] |

LOO 方向与区间范围：

| Contrast | Metric | Seed-LOO positive | Seed-LOO range | Task-LOO positive | Task-LOO range |
|---|---|---:|---:|---:|---:|
| SELF_TARGET_GAIN | Position MAE | 0/3 | [-0.001752, -0.000844] | 0/5 | [-0.001722, -0.000212] |
| SELF_TARGET_GAIN | Safe-FD L1 | 0/3 | [-0.004914, -0.003968] | 0/5 | [-0.005611, -0.000904] |
| SELF_TARGET_GAIN | Seam Safe-FD L1 | 3/3 | [+0.003779, +0.013605] | 5/5 | [+0.005005, +0.011674] |
| SV_TARGET_GAIN | Position MAE | 0/3 | [-0.001302, -0.000130] | 0/5 | [-0.001019, -0.000423] |
| SV_TARGET_GAIN | Safe-FD L1 | 0/3 | [-0.005990, -0.004103] | 0/5 | [-0.006520, -0.003337] |
| SV_TARGET_GAIN | Seam Safe-FD L1 | 3/3 | [+0.008546, +0.016812] | 5/5 | [+0.005357, +0.014927] |
| SV_VS_SELF | Position MAE | 2/3 | [-0.000010, +0.000757] | 4/5 | [-0.000210, +0.000734] |
| SV_VS_SELF | Safe-FD L1 | 0/3 | [-0.001076, -0.000135] | 1/5 | [-0.003241, +0.002142] |
| SV_VS_SELF | Seam Safe-FD L1 | 3/3 | [+0.000006, +0.007712] | 4/5 | [-0.001224, +0.006961] |

## Outer-train Y/Z 与 teacher distance

| Program | Train Y pos / FD / seam | Train target pos / FD | Teacher distance pos / FD |
|---|---:|---:|---:|
| GT_RETRAIN | 0.151703 / 0.971162 / 0.910888 | 0.151703 / 0.971162 | 0.000000 / 0.000000 |
| SELF_SOFT | 0.154104 / 0.976166 / 0.921062 | 0.079043 / 0.494366 | 0.009800 / 0.040825 |
| SV_SOFT | 0.153511 / 0.976197 / 0.915685 | 0.080579 / 0.496645 | 0.012270 / 0.044595 |

软目标 Z train loss 较低，软学生的真实 Y train 误差仍略高于 hard。Teacher distance 较小但不是 task loss 替代量。V7 历史 VIDEO SELF/SV Position 为 `0.175800/0.175330`、Safe-FD 为 `1.001883/1.002610`；V8 同 runner soft 明显未复现那一量级的退化。

## Exposure 与资源

V8 新 dev logical loss evaluations **1,665**，唯一 dev windows 185，dev target archive content **恰好读取 1 次**；Final/heldout payload 和逐任务 final 汇总访问均为 0。V5+V6+V7 known-grain 7,770，加 V8 后 9,435；legacy 35,523 unit-unverified、排除。

物理 GPU 0/1/2（三张 RTX 3090），watcher 并发 cap 2，leases 全释放。9 fits elapsed 合计 1274.1s，watcher child wall 合计 1339.1s；formal fit 峰值 allocated/reserved 138.14/178 MiB，Stage0 154.74/182 MiB。9 个正式配置与 runner/recipe/batch sequence hashes 均有收据。V2/V4/V6/V7 固定源资产及 shared watcher 的 SHA 检查 12/12 通过，旧 root 写入 0。

CI 条件于固定开发集、模型、recipe 和已重复曝光的 split；不代表 seed 总体不确定性或独立 final-test。没有 p 值。Session/trial 独立性、物理 command 单位和机器人执行可行性仍 UNKNOWN。

## 主要文件

- 主报告：`RETURN_TO_CHATGPT.zh-CN.md`；结构化结果：`RETURN_TO_CHATGPT.json`
- 分支/recipe：`protocol/BRANCH_LOCK.json`、`protocol/RECIPE_LOCK.json`
- 训练、paired、LOO、CI：`results/PRIMARY_TASK_SEED.csv`、`results/PAIRED_EFFECTS.csv`、`results/SEED_LOO.csv`、`results/TASK_LOO.csv`、`results/PAIRED_GROUP_CI.csv`
- Stage0 审计：`audit/FULL_ROW_TARGET_ROUNDTRIP.csv`、`audit/TEACHER_CLONE_REPLAY.csv`、`audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv`
- 图：`figures/`；图注：`figures/CAPTIONS.zh-CN.md`
- Exposure/resources/failures：`inventory/OUTCOME_EXPOSURE.json`、`results/RESOURCE_LEDGER.csv`、`qa/FAILED_ATTEMPTS.jsonl`
- SHA-256：`CHECKSUMS.sha256`
