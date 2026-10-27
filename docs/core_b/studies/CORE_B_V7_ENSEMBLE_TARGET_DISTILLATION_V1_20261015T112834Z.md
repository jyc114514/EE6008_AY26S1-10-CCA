> Archived study summary for `CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z` (source snapshot `20261015`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V7 集成教师向单模型轨迹目标蒸馏

状态：**complete**；开发集实验。输出根目录：`server-only://project-artifacts/reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z`。

- 新学生正式 fits：18/18（均 epoch 160）；额外 hard-only baseline fits：0。
- 冻结教师 outer-train source 回放：11520 forward rows；9 个 source bundles。教师 train predictions 是 in-sample，不是 OOF。
- 学生 train/dev 独立推理 bundles：18/18、18/18；dev target content reads：1；唯一 dev windows：185；V7 新的学生 dev loss evaluations：3330；final/heldout access：0。
- watcher jobs：66；物理 GPU：0, 1, 2；峰值 allocated/reserved：138.14013671875 / 178.0 MiB；首个作业准入时 free-memory margin：14946 MiB；child 累计 GPU wall time：5918.6s；全 pipeline wall time：4573.6s；并发 cap：2；每个 child 的物理卡映射进程内 `cuda:0`。
- GPU pipeline session 已结束：`coreb-v7-score-recovery-20261015t1305z`；日志保留在 `server-only://project-artifacts/reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/logs/RECOVERY_PIPELINE.log`（另含初始 `PIPELINE.log`）；session 运行时重连命令为 `tmux attach -t coreb-v7-score-recovery-20261015t1305z`，安全停止新任务命令为 `touch "server-only://project-artifacts/reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/coordination/STOP_REQUESTED"`。
- 恢复记录：首轮两项 pilot standalone inference 因配置缺少 model/manifest SHA 字段失败；失败状态与日志已归档，补齐哈希后两项均 PASS。pilot fit 复用，未重训。
- bootstrap：5,000 within-task source-group paired draws，seed 20261015；CI 条件于固定模型/teacher/beta/recipe 与已反复暴露 dev。无 p 值或事后非劣阈值。
- V6 旧 TASK_LOO.csv grain：每个 metric 有 15 行（3 seeds x 5 omitted tasks）；VIDEO_SUBSTITUTION_50 的 Position、Safe-FD 各 15/15 正，Seam 仅 5/15 正。

## 五任务 macro dev 结果

| 方法 | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |
|---|---:|---:|---:|
| S | 0.157555 ± 0.001035 | 0.830807 ± 0.003094 | 0.766943 ± 0.005265 |
| A | 0.159871 ± 0.001343 | 0.829451 ± 0.002901 | 0.776170 ± 0.015822 |
| V | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| SS50 | 0.157003 ± 0.001060 | 0.825177 ± 0.000320 | 0.761043 ± 0.008338 |
| SV50 | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |
| BIAS_STATE | 0.157301 ± 0.001047 | 0.830807 ± 0.003094 | 0.819232 ± 0.006349 |
| BIAS_VIDEO | 0.158066 ± 0.001482 | 0.819754 ± 0.002994 | 0.847943 ± 0.021419 |
| PERSISTENCE | 0.222262 ± 0.000000 | 0.854193 ± 0.000000 | 0.802417 ± 0.000000 |
| ST_SELF | 0.176524 ± 0.000996 | 1.006985 ± 0.001755 | 0.940597 ± 0.011793 |
| ST_SS | 0.176849 ± 0.001243 | 1.007701 ± 0.003263 | 0.941890 ± 0.012715 |
| ST_SV | 0.176337 ± 0.000904 | 1.006825 ± 0.002581 | 0.939399 ± 0.007425 |
| VD_SELF | 0.175800 ± 0.000821 | 1.001883 ± 0.001705 | 0.956603 ± 0.009749 |
| VD_SS | 0.175505 ± 0.001426 | 1.003167 ± 0.003613 | 0.949929 ± 0.004729 |
| VD_SV | 0.175330 ± 0.000466 | 1.002610 ± 0.000915 | 0.953820 ± 0.014785 |

### 每个 fixed seed 的五任务 macro

| 方法 | 指标 | seed 202610131 | seed 202610132 | seed 202610133 |
|---|---|---:|---:|---:|
| S | Position MAE | 0.156833 | 0.158742 | 0.157091 |
| S | Safe-FD L1 | 0.829130 | 0.834378 | 0.828915 |
| S | Seam Safe-FD L1 | 0.772502 | 0.766293 | 0.762032 |
| A | Position MAE | 0.159078 | 0.161421 | 0.159113 |
| A | Safe-FD L1 | 0.832258 | 0.826464 | 0.829631 |
| A | Seam Safe-FD L1 | 0.778637 | 0.790614 | 0.759259 |
| V | Position MAE | 0.159213 | 0.159179 | 0.156622 |
| V | Safe-FD L1 | 0.820959 | 0.821959 | 0.816346 |
| V | Seam Safe-FD L1 | 0.818169 | 0.775166 | 0.793948 |
| SS50 | Position MAE | 0.156197 | 0.158204 | 0.156609 |
| SS50 | Safe-FD L1 | 0.825466 | 0.825232 | 0.824833 |
| SS50 | Seam Safe-FD L1 | 0.763800 | 0.767654 | 0.751676 |
| SV50 | Position MAE | 0.155356 | 0.156591 | 0.154468 |
| SV50 | Safe-FD L1 | 0.818737 | 0.821718 | 0.816467 |
| SV50 | Seam Safe-FD L1 | 0.780409 | 0.755594 | 0.759664 |
| BIAS_STATE | Position MAE | 0.156573 | 0.158501 | 0.156831 |
| BIAS_STATE | Safe-FD L1 | 0.829130 | 0.834378 | 0.828915 |
| BIAS_STATE | Seam Safe-FD L1 | 0.825625 | 0.819142 | 0.812928 |
| BIAS_VIDEO | Position MAE | 0.158952 | 0.158891 | 0.156355 |
| BIAS_VIDEO | Safe-FD L1 | 0.820959 | 0.821959 | 0.816346 |
| BIAS_VIDEO | Seam Safe-FD L1 | 0.870519 | 0.827906 | 0.845405 |
| PERSISTENCE | Position MAE | 0.222262 | 0.222262 | 0.222262 |
| PERSISTENCE | Safe-FD L1 | 0.854193 | 0.854193 | 0.854193 |
| PERSISTENCE | Seam Safe-FD L1 | 0.802417 | 0.802417 | 0.802417 |
| ST_SELF | Position MAE | 0.176640 | 0.177457 | 0.175475 |
| ST_SELF | Safe-FD L1 | 1.008435 | 1.007486 | 1.005034 |
| ST_SELF | Seam Safe-FD L1 | 0.942501 | 0.951323 | 0.927968 |
| ST_SS | Position MAE | 0.177358 | 0.177755 | 0.175432 |
| ST_SS | Safe-FD L1 | 1.010754 | 1.008085 | 1.004263 |
| ST_SS | Seam Safe-FD L1 | 0.943669 | 0.953622 | 0.928380 |
| ST_SV | Position MAE | 0.176803 | 0.176913 | 0.175296 |
| ST_SV | Safe-FD L1 | 1.009361 | 1.006911 | 1.004201 |
| ST_SV | Seam Safe-FD L1 | 0.933491 | 0.947734 | 0.936973 |
| VD_SELF | Position MAE | 0.175911 | 0.176560 | 0.174929 |
| VD_SELF | Safe-FD L1 | 1.002638 | 1.003080 | 0.999931 |
| VD_SELF | Seam Safe-FD L1 | 0.967300 | 0.948217 | 0.954292 |
| VD_SS | Position MAE | 0.176950 | 0.175466 | 0.174099 |
| VD_SS | Safe-FD L1 | 1.006908 | 1.002898 | 0.999696 |
| VD_SS | Seam Safe-FD L1 | 0.954602 | 0.945147 | 0.950039 |
| VD_SV | Position MAE | 0.175853 | 0.175177 | 0.174960 |
| VD_SV | Safe-FD L1 | 1.003040 | 1.003231 | 1.001560 |
| VD_SV | Seam Safe-FD L1 | 0.970854 | 0.944328 | 0.946277 |

## 锁定 paired gains

正值表示 candidate loss 更低；每个 contrast 均在 `PAIRED_EFFECTS.csv`、`PAIRED_GROUP_CI.csv` 保存逐指标值。

| Contrast | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
|---|---:|---:|---:|
| VD_KD_VS_HARD | -0.016992 [-0.020047, -0.013797] | -0.182856 [-0.205369, -0.161851] | -0.158059 [-0.212891, -0.108476] |
| VD_SV_VS_SELF | +0.000470 [-0.000568, +0.001485] | -0.000727 [-0.003763, +0.002063] | +0.002783 [-0.003748, +0.009534] |
| VD_VISUAL_TEACHER | +0.000175 [-0.000595, +0.000930] | +0.000557 [-0.001575, +0.002652] | -0.003891 [-0.010778, +0.002660] |
| ST_KD_VS_HARD | -0.016466 [-0.019480, -0.013410] | -0.177373 [-0.199459, -0.156892] | -0.163230 [-0.216169, -0.114536] |
| ST_SV_VS_SELF | +0.000187 [-0.000428, +0.000785] | +0.000161 [-0.001457, +0.001792] | +0.001198 [-0.003989, +0.006563] |
| ST_VISUAL_TEACHER | +0.000511 [-0.000426, +0.001444] | +0.000876 [-0.001401, +0.003004] | +0.002491 [-0.003561, +0.008289] |
| STUDENT_INPUT_GAIN | +0.001007 [-0.001628, +0.003530] | +0.004214 [-0.001312, +0.009294] | -0.014421 [-0.038226, +0.000860] |
| VD_VS_TEACHER | -0.019859 [-0.022986, -0.016622] | -0.183636 [-0.206166, -0.162494] | -0.188598 [-0.241813, -0.139885] |
| ST_VS_TEACHER | -0.020866 [-0.024050, -0.017521] | -0.187851 [-0.210425, -0.167097] | -0.174177 [-0.229339, -0.123375] |
| ST_SV_VS_S | -0.018782 [-0.021967, -0.015556] | -0.176017 [-0.197844, -0.155590] | -0.172457 [-0.226855, -0.122728] |
| ST_SV_VS_BIAS_STATE | -0.019036 [-0.022188, -0.015803] | -0.176017 [-0.197844, -0.155590] | -0.120167 [-0.177205, -0.067659] |
| VD_SV_VS_S | -0.017775 [-0.021405, -0.014108] | -0.171803 [-0.194424, -0.150297] | -0.186877 [-0.242812, -0.134160] |
| VD_SV_VS_BIAS_VIDEO | -0.017264 [-0.020309, -0.014072] | -0.182856 [-0.205369, -0.161851] | -0.105877 [-0.164308, -0.053563] |

## 解释边界

- ST_SV 推理时只使用 state/history/task/known anchor；训练监督来自 SV50 视觉派生教师，因此它不是全流程无视觉。
- 顺序偏差：pilot 已在真实 batch QA 代码补入前开始并完成；为避免丢弃正式槽，未中断或重训。四条真实 outer-train receiver 的 Y/T/Z、梯度、ID/history 与 forward 分离 QA 随后通过，并在扩展剩余 17 fits 前完成。该检查通过，但不是首个 optimizer step 前的 gate。
- 如视频学生只胜 hard-only 而未胜 SELF/SS，结果支持一般目标混合/自蒸馏解释，不能归因于视觉 ensemble。若固定 beta/recipe 未迁移收益，只能说明本轮迁移未成功。
- 教师在 outer-train 样本上训练，train targets 为 in-sample；教师 train MAE/FD/seam 仅描述，不用于选择。Session/trial 独立性、numeric command 的物理单位仍 UNKNOWN。
- 结果是固定开发集比较，不是独立 final-test、真实机器人成功或预先设计的 non-inferiority 检验。

## 逐项审计

- 最终验收：`qa/FINAL_ACCEPTANCE.json`；prediction seal：`qa/PREDICTION_SEAL.json`；source map：`inventory/SOURCE_MAP.json`。
- 训练目标、seam、LOO、imitation：`results/PRIMARY_TASK_SEED.csv`、`results/PAIRED_EFFECTS.csv`、`results/PAIRED_GROUP_CI.csv`、`results/TASK_LOO.csv`、`results/SEED_LOO.csv`、`results/IMITATION_ERROR.csv`。
- GPU、模型成本与 exposure：`results/RESOURCE_LEDGER.csv`、`results/MODEL_COST.csv`、`inventory/OUTCOME_EXPOSURE.json`。
- 图：`figures/`；人审记录：`qa/FIGURE_VISUAL_REVIEW.md`。人审不作为数值完成门。
- checksum manifest：`CHECKSUMS.sha256`。Git state：UNKNOWN（当前 shell 没有 git executable，V4 locator 记录无 `.git` metadata）。
