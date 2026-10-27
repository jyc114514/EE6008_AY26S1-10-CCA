> Archived study summary for `CORE_B_V8_1_EVIDENCE_COMPLETION_V1_20261017T054525Z` (source snapshot `20261017`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V8.1 固定开发集证据补全

状态：complete_with_limits。该包补齐 V8 已锁定证据，未开展新模型实验。

## 三个核查问题

1. **学生是否更好地拟合实际训练目标 Z？** 否。V8 的差值定义为 J(G,Z)−J(student,Z)，三 seed 都为负，表示 G 的同目标评估损失更低。Z_M 上 M 的均值差为 -0.002499 ± 0.000541；Z_U 上 U 为 -0.004041 ± 0.000684。这只说明当前固定目标、固定尺度和 dropout-off objective 的比较结果。

2. **输出是否朝教师 T 靠近？** V8 的 train/dev 距离表显示 G 比 U、M 更接近 T；这些是输出间距离，不等同于真实 Y 准确度。V8.1 复用完整 seed 级表的三 seed 均值和 sample SD：

| Split | Method | Position distance to T | Safe-FD distance to T | Seam distance to T |
|---|---|---:|---:|---:|
| train | G | 0.010673 ± 0.000322 | 0.036858 ± 0.000863 | 0.086166 ± 0.004289 |
| train | U | 0.013901 ± 0.000525 | 0.053825 ± 0.001980 | 0.119422 ± 0.004667 |
| train | M | 0.012874 ± 0.000216 | 0.052297 ± 0.001648 | 0.109849 ± 0.002932 |
| dev | G | 0.019022 ± 0.000065 | 0.067520 ± 0.001627 | 0.106275 ± 0.009315 |
| dev | U | 0.024620 ± 0.001630 | 0.091901 ± 0.008366 | 0.146612 ± 0.008023 |
| dev | M | 0.025062 ± 0.002433 | 0.092728 ± 0.007475 | 0.138593 ± 0.018077 |

V8.1 轨迹图使用同一批已封存预测。

3. **MAE/FD 退化与 seam 改善在哪里？** 完整 30-step、29-transition、28-axis 表已生成。对 M 相对 G，正收益定义为 G error−M error；三 seed 宏观均值为 position -0.000850 ± 0.001198，FD -0.004274 ± 0.001355，seam +0.020449 ± 0.019151。主要平均退化落在任务 141、119、94；seam 的 task macro 在五个任务均改善，增益集中于 119 和 141。M 的最负三 seed 平均单元是 task 141 position step 7 (−0.010552) 与 FD transition 9→10 (−0.091063)；坐标上 position 的最负单元为 task 141 axis 6 (−0.015775)，FD 为 task 119 axis 20 (−0.063199)。Seam 在 task 119 axis 4 有最大正单元 (+0.290707)，task 141 axis 20 仍有局部负单元 (−0.188403)。这些是完整网格上的描述值，不是逐格显著性检验。

## 宏观结果（mean ± sample SD, n=3 seeds）

| 方法 | Position | Safe FD | Seam safe FD |
|---|---:|---:|---:|
| G | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| U | 0.159380 ± 0.001048 | 0.822409 ± 0.002542 | 0.794948 ± 0.012296 |
| M | 0.159188 ± 0.001340 | 0.824029 ± 0.003250 | 0.775311 ± 0.008490 |
| T | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |

M 相对 G 的分任务收益（正值有利于 M）：

| Task | Position gain | Safe-FD gain | Seam gain |
|---:|---:|---:|---:|
| 119 | -0.000731886 | -0.007582835 | +0.060522782 |
| 94 | -0.000406675 | -0.004121787 | +0.005491820 |
| 90 | -0.000027274 | +0.000049513 | +0.005302973 |
| 141 | -0.003100671 | -0.009753305 | +0.030055079 |
| 230 | +0.000018720 | +0.000038154 | +0.000874713 |

V7 已接受的条件 source-group 95% CI（5,000 draws；本轮 0 次重抽样）仍为 position [−0.002180, +0.000579]、FD [−0.008554, −0.000481]、seam [+0.007785, +0.035567]。对账表的最大直接/step/axis差为 4.44e−16，低于 1e−12 abs / 1e−10 rel 容差。

## 覆盖与访问账本

- HORIZON_ERRORS 4,248 行：position 2,160，safe-FD 2,088。AXIS_ERRORS 6,048 行：position、internal-FD、seam 各 2,016。MACRO_RECONCILIATION 216 行全部 PASS；PER_WINDOW_RECONCILIATION 2,220 行全部 PASS。
- 任务支持：119=48、94=48、90=41、141=20、230=28 anchors；共 47 episodes/source groups。
- V8.1 原始 dev-target archive 只 load 1 次，185 windows；精确读取五个白名单字段，cache 先写入和读回后才归约。scorer-cache package load 2 次（初次 readback + 从已封存缓存恢复）。错误组装重试未重开 target。
- 12 个既有预测 bundle 在 cache 计算和恢复各 decode 一次（24 次），行映射 36 次含 preflight；2220 是方法×seed×window 诊断单元，不是新增样本。
- 新 fit/update/forward/dev prediction/encoder/bootstrap/search/GPU 均为 0；final/heldout payload access=0；旧 root writes=0。V7/V8 manifest SHA 保持 58c50a9928106c2cf95fe78c5d3a0bdaa05c050a271803d01a6ef8e011810a29 / 7778803b295f0bd5c3563fa69b940e740ced7b77a7e77c5650f4a0790b88d1c0。V8 原 status complete_with_limits 和 4 个 UNAVAILABLE 行保持封存；补全仅存在 V8.1 新目录。
- 图：14 张总体/task误差与收益图，15 张固定 seed/anchor、固定轴 2/20 的轨迹图；四张 V8 图 byte-identical 复用。Agent visual review PASS；human review PENDING。trajectory_data.csv 含已经暴露的 dev Y，留在私有根目录；不作为独立 pool。

输出根：server-only://project-artifacts/reports/benchmarks/CORE_B_V8_1_EVIDENCE_COMPLETION_V1_20261017T054525Z
关键文件：results/HORIZON_ERRORS.csv、AXIS_ERRORS.csv、MACRO_RECONCILIATION.csv、CONCLUSION_LEDGER.csv、RESOURCE_AND_EXPOSURE.csv；图与 captions 均在 figures/、figure_data/、captions/。最终 manifest SHA 由 qa/FINAL_CHECKSUM_VERIFICATION.json 记录，且在最终聊天回执中报告。
