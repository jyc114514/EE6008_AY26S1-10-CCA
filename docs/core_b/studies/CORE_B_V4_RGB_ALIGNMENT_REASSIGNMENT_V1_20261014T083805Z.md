> Archived study summary for `CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z` (source snapshot `20261014`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V4 RGB 配对训练与 state 近邻重分配：结果回传

**状态：complete（post-V3 开发实验）**。输出目录：`server-only://project-artifacts/reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z`。

## 主要结果

| 方法 | Position MAE | Safe-FD L1 |
|---|---:|---:|
| BASE_STATE | 0.157555 ± 0.001035 | 0.830807 ± 0.003094 |
| BIAS_STATE | 0.157301 ± 0.001047 | 0.830807 ± 0.003094 |
| BIAS_VIDEO | 0.158066 ± 0.001482 | 0.819754 ± 0.002994 |
| PERSISTENCE | 0.222262 ± 0.000000 | 0.854193 ± 0.000000 |
| A_M | 0.159871 ± 0.001343 | 0.829451 ± 0.002901 |
| B_M | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 |
| C_M | 0.160244 ± 0.000789 | 0.829653 ± 0.001748 |

以上是三 seed 的五任务 episode-equal macro 均值 ± sample SD；15 行 task×seed 原始表见 `results/PRIMARY_TASK_SEED.csv`。本轮没有正 gain 或 minimum support 完成门，负结果按计划报告并完成。`RGB_COMMON_GAIN=L(A_M)-L(B_M)` 正值表示 B_M 误差更低；`PAIRED_TRAIN_GAIN=L(C_M)-L(B_M)` 正值表示 C_M 误差更高、负值表示 C_M 更好；input-sensitivity 正值表示 map 替换使该模型变差；`C_D_vs_B_D` 正值表示 C_D 误差更大；interaction 正值表示 B 的 map penalty 更大。完整方向见 `protocol/RESULT_SIGN_GUIDE.json`。

B_MATCHED 对旧 optimized STATE 的 position gain 为负（-0.000783），safe-FD gain 为正（+0.011053）；因此 B 提升 FD，但没有在 Position MAE 上超过旧 state 基线。

| B_MATCHED 对旧参考的 gain = L(reference) − L(B_M) | Position MAE | Safe-FD L1 |
|---|---:|---:|
| V2 optimized STATE | -0.000783 | 0.011053 |
| V3 BIAS_STATE | -0.001037 | 0.011053 |
| V3 BIAS_VIDEO | -0.000272 | 0.000000 |
| Persistence | 0.063924 | 0.034438 |

| 配对量 | Position MAE gain | Safe-FD gain | 95% paired source-group CI（Position / Safe-FD） |
|---|---:|---:|---:|
| RGB_COMMON_GAIN | 0.001533 | 0.009697 | [-0.001128, 0.004113] / [0.003391, 0.016890] |
| PAIRED_TRAIN_GAIN | 0.001906 | 0.009898 | [-0.001066, 0.004742] / [0.002752, 0.017042] |
| B_INPUT_SENSITIVITY | 0.011495 | 0.023009 | [0.006449, 0.016835] / [0.012514, 0.034433] |
| C_INPUT_SENSITIVITY | 0.000073 | 0.000278 | [-0.000127, 0.000263] / [-0.000175, 0.000744] |
| C_D_vs_B_D | -0.009517 | -0.012833 | [-0.014225, -0.005051] / [-0.021380, -0.004214] |
| TRAIN_TEST_INTERACTION | 0.011423 | 0.022731 | [0.006452, 0.016730] / [0.012219, 0.034162] |

同一配对 source-group 抽样用于所有方法、seed 和 map；CI 条件于已训练模型、固定 recipe 和已暴露 dev。没有 p 值。地图平均的是 loss，不平均预测；maps 不是独立重复样本。

## V3 数值精度复核

实质性复核中，非零旧效果的符号变化数为 0。P_CAPTURE 对 P_REF 的最大 native 差为 1.3113022e-06；R=0 与 capture 的最大 native 差为 0。非零 V3 bias/residual 效应没有符号翻转。30 个原本精确为零的 bias safe-FD 对比只有 float32 级漂移（最大 3.81e-8，低于 1e-6 replay tolerance）；非零旧效应最大相对变化为 0.087%。

任务×seed 的原生与 standardized 差及 contrast 量级在 `audit/V3_CANONICAL_ZERO_R_REPLAY.csv`、`audit/V3_CANONICAL_NATIVE_DIFFERENCES.csv`、原始 `audit/V3_METRIC_PRECISION_IMPACT.csv` 与 `audit/V3_METRIC_PRECISION_SENSITIVITY_CORRECTED.csv`。

## 训练与 donor 证据

B_MATCHED 复用 V2 G-V 三个 formal 模型：同一 1280-anchor outer-train split、同一 normalizer、LR 5e-4、160 epochs、batch 64、3200 updates、AdamW/update-cosine；每 seed 的 fresh INIT 与配对 G-S INIT 逐字节相同。新训练为 A×3 与 C×3，未从训练后 B 热启动。

State-neighbor donor scaler 只在 outer-train prefix 上按 task→episode→anchor 等权拟合；same-task 查询排除接收 episode/source group，每个 source episode 取最近 anchor，最多 K=5。dev map 只从 train donor bank 按预注册 SHA-256 规则选。所有 receiver state/anchor/task/target 保持不变；future label 不选 donor。匹配距离、V3 donor 对照、轮换复用率见 `results/DONOR_MATCH_QUALITY.csv`。

任务变化分层使用 train 的 M_fd q25/q75，在 seal 后按 dev future target 作描述性分组；分层误差和 support 见 `results/MOTION_STRATA_RESULTS.csv`、`results/MOTION_STRATA_SUPPORT_SUMMARY.csv` 与 `figures/V4_MOTION_STRATA_SUPPORT.png`。M_fd 表示录制 command change，不是物理速度或运动。

## 资源、访问与限制

接受新 fits=6/6；新预测 jobs=24/24；复用 B_MATCHED predictions=3。GPU lease 全部释放=True；V4 child jobs 使用 GPU 0, 1, 2；最大 allocated/reserved 为 137.53076171875/176.0 MiB；child elapsed 合计 1027.9 秒。Watcher 并发上限为现场硬限制 2。完整资源与 admission snapshots 见 `results/RESOURCE_LEDGER.csv` 和 `results/GPU_ADMISSION_SNAPSHOTS.csv`。

V2/V3 既有累计 dev future-window reads=27750；V4 metric audit=3330；V4 新预测评分=4440；本轮新增逻辑 window reads=7773；累计=35523。scorer archive load=1；同一预选 trajectory anchor 的三次图表构建又读取该 target archive，故总 archive loads=4、图表 row accesses=3，unique dev windows 仍为185。final/heldout payload access=0。V2/V3 旧 roots 未写入。

缓存历史 extractor 与 session/trial 绑定仍不完整；session/trial 独立性未验证。近邻重分配是 state-conditioned RGB 输入诊断，不是保持 RGB 多重集合的精确 conditional-randomization test；不估计 CMI 或物理因果效应。人审图状态为 pending，不改变数字完成状态。

## 文件

- `results/PRIMARY_TASK_SEED.csv`：5 task × 3 seed 的 A/B/C、旧 STATE/BIAS/Persistence 指标与 primary gain。
- `results/PAIRED_EFFECTS.csv`、`results/SEED_LOO.csv`、`results/TASK_LOO.csv`、`results/PAIRED_GROUP_CI.csv`：配对效果、两种 LOO 与条件区间；符号按 `protocol/RESULT_SIGN_GUIDE.json`。
- `figures/`、`figure_data/`：实际训练曲线、donor 匹配、变化分层、双指标、V3 选定 trajectory case。视觉接触表见 `qa/V4_FIGURE_CONTACT_SHEET.jpg`。
- `qa/FINAL_ACCEPTANCE.json`、`CHECKSUMS.sha256`：验收和最终校验清单。
- `qa/FAILED_ATTEMPTS.jsonl`：保留并说明的准备阶段失败与修复；没有 fit 失败。
- `qa/POSTPROCESSING_SIGN_CORRECTION.json` 与 `archive/PAIRED_TRAIN_GAIN_PRE_FORMULA_CORRECTION_*/`：保留锁定 C−B 公式的修正和更正前衍生表；无新 target read/refit。
- `qa/CHECKSUM_HISTORY.json`：保留前两份被替代的 pre-final 清单。
- `qa/OLD_ROOT_READONLY_AUDIT.json`：18 项来源 hash 匹配，旧 V2/V3 文件在 V4 开始后无 mtime 变化。
- `qa/FIGURE_VISUAL_REVIEW.md`：图表已做视觉检查；人工复核仍 pending。
