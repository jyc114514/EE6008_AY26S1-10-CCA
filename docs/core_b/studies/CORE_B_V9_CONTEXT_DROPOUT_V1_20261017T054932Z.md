> Archived study summary for `CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z` (source snapshot `20261017`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V9 训练期随机数值上下文屏蔽：实验交付

**状态：完成。** 范围为固定开发集与固定 3 个 seed 的受控实验；final/heldout 未访问。

## 主要结论

在完整数值上下文主场景下，VD 没有优于 V0 的单模型总体误差：V0−VD 的 Position MAE 和 Safe-FD L1 为负，Seam 指标为正。AD 对 A0 的 Position MAE 也变差。交互项没有支持数值上下文屏蔽专门提升视频作用的一致证据；Safe-FD 交互为负且 95% source-group bootstrap 区间不跨 0。

固定 mixture 对照中，MIXD 相对 MIX0 只有 seam 指标有正向点估计和区间；Position 与 Safe-FD 没有改善。MIXD 仍优于历史 S 单模型，但这不等于 VD 单 head 优于 V0。强制屏蔽推理仅作为次要诊断，不替代完整输入主结果。

V0 三个 seed 的 checkpoint 与 V8 hard-video checkpoints 文件 SHA/state hash 完全一致；train/dev V0 预测数组与 V8 accepted hard-video 输出逐字节一致，最大绝对差为 0。

## 完整上下文主结果（五任务 macro，mean ± sample SD across 3 seeds）

| Program | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |
| --- | --- | --- | --- |
| A0 | 0.159871 ± 0.001343 | 0.829451 ± 0.002901 | 0.776170 ± 0.015822 |
| AD | 0.164050 ± 0.000740 | 0.828216 ± 0.001285 | 0.767115 ± 0.014540 |
| V0 | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| VD | 0.161149 ± 0.001384 | 0.826934 ± 0.003999 | 0.776202 ± 0.012066 |
| MIX0 | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |
| MIXD | 0.155725 ± 0.000209 | 0.819831 ± 0.001981 | 0.755074 ± 0.007449 |

## 每 seed 主指标

每个单元格顺序为 Position MAE / Safe-FD L1 / Seam Safe-FD L1。

| Program | 202610131 | 202610132 | 202610133 |
| --- | --- | --- | --- |
| A0 | 0.159078 / 0.832258 / 0.778637 | 0.161421 / 0.826464 / 0.790614 | 0.159113 / 0.829631 / 0.759259 |
| AD | 0.163283 / 0.827228 / 0.766094 | 0.164761 / 0.827751 / 0.782139 | 0.164105 / 0.829668 / 0.753113 |
| V0 | 0.159213 / 0.820959 / 0.818169 | 0.159179 / 0.821959 / 0.775166 | 0.156622 / 0.816346 / 0.793948 |
| VD | 0.162463 / 0.831470 / 0.765781 | 0.159705 / 0.825416 / 0.789421 | 0.161279 / 0.823917 / 0.773403 |
| MIX0 | 0.155356 / 0.818737 / 0.780409 | 0.156591 / 0.821718 / 0.755594 | 0.154468 / 0.816467 / 0.759664 |
| MIXD | 0.155926 / 0.821035 / 0.749911 | 0.155509 / 0.820914 / 0.763613 | 0.155739 / 0.817545 / 0.751698 |

## 预注册 paired contrasts

下面均为 reference − candidate，正值表示 candidate 误差更低；区间为 5,000 次 within-task paired source-group bootstrap（seed 20261016），条件于固定模型和重复曝光的 dev 集。

| Contrast | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
| --- | --- | --- | --- |
| NUMCTX_DROP_VIDEO | -0.002810 [-0.005209, -0.000482] | -0.007180 [-0.013860, -0.001393] | 0.019559 [0.000594, 0.042446] |
| NUMCTX_DROP_STATE | -0.004179 [-0.006201, -0.001975] | 0.001235 [-0.003107, 0.005642] | 0.009054 [-0.004114, 0.024459] |
| VIDEO_GAIN_0 | 0.001533 [-0.001123, 0.004069] | 0.009697 [0.003506, 0.016860] | -0.019591 [-0.057440, 0.007736] |
| VIDEO_GAIN_D | 0.002901 [-0.000554, 0.005942] | 0.001282 [-0.005908, 0.008304] | -0.009086 [-0.027218, 0.007235] |
| INTERACTION | 0.001368 [-0.002294, 0.004596] | -0.008415 [-0.015531, -0.001270] | 0.010505 [-0.007750, 0.033033] |
| MIX_GAIN | -0.000253 [-0.001437, 0.000904] | -0.000857 [-0.004461, 0.002473] | 0.010148 [0.000427, 0.021970] |

INTERACTION 定义为 (E(AD)-E(VD)) - (E(A0)-E(V0))。V0−VD 的 Position/FD 都是负值；VD 的 seam 改善不构成视频单 head 总体优化成功。

## 固定比较与历史参考

| Contrast | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
| --- | --- | --- | --- |
| VD_VS_MIX0 | -0.005677 [-0.007979, -0.003485] | -0.007960 [-0.013814, -0.002432] | -0.010980 [-0.024117, 0.001939] |
| VD_VS_S | -0.003593 [-0.006519, -0.000631] | 0.003873 [-0.002628, 0.010586] | -0.009259 [-0.025742, 0.005910] |
| VD_VS_PERSISTENCE | 0.061113 [0.049380, 0.073136] | 0.027258 [0.016722, 0.036857] | 0.026215 [0.003007, 0.047241] |
| MIXD_VS_S | 0.001831 [0.000281, 0.003468] | 0.010976 [0.007537, 0.014714] | 0.011869 [0.004483, 0.019110] |
| MIXD_VS_V0 | 0.002614 [0.000423, 0.004872] | -0.000077 [-0.006041, 0.004970] | 0.040687 [0.019564, 0.067628] |
| MIXD_VS_OLD_SV50 | -0.000253 [-0.001437, 0.000904] | -0.000857 [-0.004461, 0.002473] | 0.010148 [0.000427, 0.021970] |
| V0_VS_OLD_V | -0.000000 [-0.000000, 0.000000] | 0.000000 [-0.000000, 0.000000] | -0.000000 [-0.000000, -0.000000] |

V0/V8 hard-video replay：

- train seed 202610131: arrays exact=True, max abs diff=0, SHA equal=True
- dev seed 202610131: arrays exact=True, max abs diff=0, SHA equal=True
- train seed 202610132: arrays exact=True, max abs diff=0, SHA equal=True
- dev seed 202610132: arrays exact=True, max abs diff=0, SHA equal=True
- train seed 202610133: arrays exact=True, max abs diff=0, SHA equal=True
- dev seed 202610133: arrays exact=True, max abs diff=0, SHA equal=True

## 强制屏蔽输入的次要诊断

单元格为 E(force-mask) − E(full)，mean ± sample SD across seeds；正值表示强制屏蔽时误差更高。

| Program | Position Δ | Safe-FD Δ | Seam Δ |
| --- | --- | --- | --- |
| A0 | 0.064278 ± 0.002119 | 0.037288 ± 0.014865 | 0.059738 ± 0.014463 |
| AD | 0.015816 ± 0.004013 | 0.005148 ± 0.003210 | 0.015146 ± 0.017225 |
| V0 | 0.050663 ± 0.006992 | 0.052849 ± 0.014621 | 0.006382 ± 0.036231 |
| VD | 0.000578 ± 0.001101 | 0.000416 ± 0.002269 | -0.010391 ± 0.010533 |

这项输入干预诊断不等同于条件互信息或物理因果检验。

## 训练、输入与随机数值计划

- 12 个新正式 fits 均使用真实 Y；没有 teacher/soft target 或 encoder 训练/feature extraction。
- 固定 LR 5e-4、AdamW、batch 64、160 epochs、3,200 updates、last epoch；没有 p/LR/epoch 搜索。
- 屏蔽位置为 normalizer 后、state projection 前，整个 [B,32,71] receiver state block out-of-place 置零。history_mask、relative_delta、独立 anchor_command、task 与视频输入保持。
- AD/VD 每 seed 共用 SHA-256 receiver×epoch mask bank；三 seed 计划 drop 率分别为 0.499692、0.500088、0.499644，训练日志合计观测为 0.499808。未重抽以凑精确 50%；最大连续 drop 分别为 23、20、20 个 epoch。每个 seed/task/epoch、episode coverage 和 receiver run 长度见 mask_banks/MASK_TASK_EPOCH_STATS.csv、MASK_EPISODE_STATS.csv、MASK_RECEIVER_STATS.csv。
- Stage0 的真实跨任务输入 sentinel、p=0 output/loss/gradient equality、force-drop block-zero、Y/RNG isolation 均通过；另做 2 个 Stage0 optimizer updates，不计入 12 个 formal fits。

## 预测、暴露与资源

| 项 | 实际结果 | 说明 |
| --- | --- | --- |
| Formal fits | 12/12 | 160 epochs × 20 updates = 3,200 updates each |
| Training targets | true Y only | No teacher/soft target; 2,457,600 cumulative fit presentations |
| Neural predictions | 36 bundles | 12 train full + 12 dev full + 12 dev force-mask |
| Derived mixtures | 6 bundles | MIX0=0.5(S_OPT+V0), MIXD=0.5(S_OPT+VD), native float64 |
| Primary dev evaluation | 18 bundles / 3,330 evaluations | 4 single-head arms × 3 seeds + 6 fixed mixtures |
| Force-mask diagnostic | 12 bundles / 2,220 evaluations | All receiver numeric-context blocks forced to zero |
| New dev total | 30 bundles / 5,550 evaluations | 185 unique windows; 150 task views |
| Dev archive read | 1 | After prediction seal; one archive read / one in-memory numpy load |
| Final/heldout | 0 | Payload and task-level final aggregates remained closed |
| Cumulative known-grain evaluations | 14,985 | 9,435 inherited + 5,550 V9; legacy 35,523 unit-unverified excluded |

| 资源项 | 数值 | 说明 |
| --- | --- | --- |
| Formal fits | 12 | 0 failures / 0 OOM; all watcher exit 0 |
| GPU pool | RTX 3090 physical GPUs 0 and 1 | dynamic_all_visible; watcher cap 2; leases all released |
| Measured resource gate | Stage0 reserved 508 MiB | Admission required free memory ≥1,020 MiB |
| Formal peak reserved | 168 MiB A0/AD; 176 MiB V0/VD | Allocated peak 124.14/137.33 MiB by model family |
| GPU fit elapsed sum | 1194.5 s | Sum of 12 per-fit process elapsed times |
| Inference elapsed sum | 70.6 s | 36 fresh-process label-free bundles |
| Environment | Python 3.12.14; PyTorch 2.12.1+cu130; CUDA build 13.0 | RTX 3090; child logical device cuda:0 |

正式 fit queue 按 seed 轮换 arm，watcher 动态选择物理 GPU 0/1，最多两个 child 并发；显存门来自 Stage0 实测，没有增加小时数/GiB/正收益/最低 N 完成门。完整 ledger 在 results/RESOURCE_LEDGER.csv。

## 可复现边界与异常

- 置信区间条件于固定模型/recipe 和反复曝光的 185-window 开发集；不代表训练 seed 总体不确定性。session/trial 独立性仍未知，无 p 值。
- 标准化误差与原生数值轨迹不是物理世界精度；采集单位/机器人执行可行性仍未知。
- 首次 Stage0 因未使用的模块级路径常量失败；retry 的 Stage0 通过。封存与评分出现的 schema/post-processing 错误均保留在失败账本中；paired 表从已保存 loss rows 恢复，没有重读 dev archive。
- 无 formal fit 失败或 OOM；12 fits、36 neural predictions、6 derived mixtures 完成；final/heldout access 为 0。

## 交付路径

- 主要报告：RETURN_TO_CHATGPT.zh-CN.md；结构化摘要：RETURN_TO_CHATGPT.json
- 锁与计划：protocol/STUDY_LOCK.json、protocol/MASK_FIELD_MAP.json、protocol/MASK_SCHEDULE.json、protocol/RECIPE_LOCK.json、protocol/RESOURCE_LOCK.json
- 主要结果：results/PRIMARY_TASK_SEED.csv、results/MASKED_DIAGNOSTIC.csv、results/PAIRED_EFFECTS.csv、results/PAIRED_GROUP_CI.csv、results/SEED_LOO.csv、results/TASK_LOO.csv
- 预测封存与暴露：qa/PREDICTION_SEAL.json、qa/DEV_TARGET_SINGLE_READ.json、inventory/OUTCOME_EXPOSURE.json
- 图：figures/；SHA-256 manifest: CHECKSUMS.sha256
