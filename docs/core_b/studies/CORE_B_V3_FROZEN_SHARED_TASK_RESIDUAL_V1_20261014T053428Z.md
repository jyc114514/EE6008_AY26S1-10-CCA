> Archived study summary for `CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z` (source snapshot `20261014`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# EE6008 V3 冻结共享模型与输出残差适配：执行回传

**状态：完成（开发实验）**。V3 复用 V2 corrected run 的 split、冻结权重、normalizer、Q0 cache 和预测；未重训 V2 的 78 fits，也未训练/提取 encoder。最终/heldout payload 访问为 0。

## 数据、模型与训练

- 固定任务：119、94、90、141、230。训练集 220 episodes / 1,280 anchors；直接复用 inner-fit 174 / 1,010 与 inner-tune 46 / 270；dev 为 185 anchors，逐任务 support 为 48、48、41、20、28。无最小样本或正收益门。
- base hidden 为 `[B,30,128]`；共享线性 residual 3,612 参数，任务五头总计 18,060，每条样本激活 3,612。base 全程冻结，optimizer 只含 residual heads；future labels 只进入训练 loss/scorer，没有进入 forward。
- 8 个 screening fits 全部接受，12 个 formal fits 全部接受，另有 6 个 CPU train-only weighted-median bias calibrations。Stage0 两个 smoke 通过，readout 更新不计入 fits。
- V3 LR 固定为 1e-3；只在 inner-tune 选择 λ∈{0,1} 与 epoch∈{20,40,80}。四程序选择如下：

|程序|λ|epoch|tune J|
|---|---:|---:|---:|
|SHARED_RES_STATE|1|20|0.247627409|
|TASK_RES_STATE|1|20|0.247714589|
|SHARED_RES_VIDEO|1|40|0.247704468|
|TASK_RES_VIDEO|1|40|0.247859097|

冻结 V2 base 的实际来源（selection/manifest 锁定；V3 未更新 base）：

|阶段|模型|seed(s)|V2 LR|epochs|模式|
|---|---|---|---:|---:|---|
|formal|G-S|202610131;202610132;202610133|0.001|160|STATE-CTRL|
|formal|G-V|202610131;202610132;202610133|0.0005|160|TS11-D|
|rgb_control|S-V-90|202610131;202610132;202610133|0.0005|160|TS11-D|
|screening|G-S|202610130|0.001|160|STATE-CTRL|
|screening|G-V|202610130|0.0005|160|TS11-D|

## V2 轻量复核

V2 saved-loss 复算与接受表核对通过：336 个 per-seed contrast 行最大绝对差 `0.0e+00`；432 个 summary 字段在四位小数匹配（最大差 `5.00e-16`）。旧 `ALL_SEED_LOO.csv` 实际是 leave-one-task-out；V3 分别生成 true seed-LOO（留 2 seeds、5 tasks 全保留）与 task-LOO（每 seed 留 4 tasks）。

position/safe-FD 使用 V2 accepted canonical metric scales（lock SHA `dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32`）；完整 28 轴 `target_sigma` 与 `safe_fd_scale` 在 `RETURN_TO_CHATGPT.json`。旧 V2 macro 点估计：

|V2 contrast|position MAE mean±sample SD|safe-FD mean±sample SD|
|---|---:|---:|
|D_S|-0.003717 ± 0.000070|-0.009081 ± 0.001852|
|D_V|0.000392 ± 0.000755|-0.008059 ± 0.005584|
|V_G|-0.000783 ± 0.001456|0.011053 ± 0.002497|
|V_S|0.003326 ± 0.001776|0.012075 ± 0.009007|
|I|0.004109 ± 0.000686|0.001022 ± 0.006807|

## V3 五任务结果

下表为每任务三个 formal seeds 的均值；每格是 `position MAE / safe-FD L1`，两个指标分别报告、越低越好。全量 per-seed 数值和 sample SD 见 `results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv` 与 `SUMMARY_STATISTICS.csv`。

|Task|Persistence|Base state|Bias state|Shared-R state|Task-R state|Base video|Bias video|Shared-R video|Task-R video|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|119|0.324670/1.064909|0.202165/1.070958|0.202081/1.070958|0.202454/1.076106|0.202703/1.078331|0.205109/1.041719|0.204978/1.041719|0.204850/1.045344|0.205372/1.047960|
|94|0.232628/0.754663|0.143957/0.749523|0.143926/0.749523|0.143870/0.753087|0.144134/0.753514|0.145120/0.741606|0.145056/0.741606|0.145059/0.745644|0.145195/0.745150|
|90|0.162914/0.571054|0.162563/0.571248|0.161592/0.571248|0.162412/0.571263|0.160988/0.570795|0.162587/0.571223|0.161617/0.571223|0.162823/0.571244|0.161453/0.570827|
|141|0.237868/0.927019|0.126042/0.808944|0.125978/0.808944|0.126788/0.813365|0.126850/0.813574|0.125749/0.790843|0.125673/0.790843|0.126555/0.795640|0.127907/0.795143|
|230|0.153231/0.953318|0.153050/0.953363|0.152931/0.953363|0.153095/0.953376|0.153022/0.953261|0.153127/0.953382|0.153007/0.953382|0.153347/0.953388|0.153538/0.953059|

五任务 macro（先 episode 等权、再 task 等权；数值为三个 seed 的 mean±sample SD）：

|方法|position MAE|safe-FD L1|
|---|---:|---:|
|PERSISTENCE|0.222262 ± 0.000000|0.854193 ± 0.000000|
|BASE_STATE|0.157555 ± 0.001035|0.830807 ± 0.003094|
|BIAS_STATE|0.157301 ± 0.001047|0.830807 ± 0.003094|
|SHARED_RES_STATE|0.157724 ± 0.001176|0.833440 ± 0.003662|
|TASK_RES_STATE|0.157540 ± 0.001202|0.833895 ± 0.003781|
|BASE_VIDEO|0.158338 ± 0.001486|0.819754 ± 0.002994|
|BIAS_VIDEO|0.158066 ± 0.001482|0.819754 ± 0.002994|
|SHARED_RES_VIDEO|0.158527 ± 0.001515|0.822252 ± 0.003094|
|TASK_RES_VIDEO|0.158693 ± 0.001381|0.822428 ± 0.003423|

### 配对 contrast 与条件区间

定义统一为 `L(left)-L(right)`；正数表示 right 误差更低。区间为 5,000 次 paired source-group bootstrap，seed 202610134，同一组 draws 共用于方法、seed、指标；区间条件于已训练 seeds 与 tune 选择，不是独立确认，不给 p 值。

|Contrast|Δposition mean±SD [95% group CI]|Δsafe-FD mean±SD [95% group CI]|
|---|---:|---:|
|G_BIAS_STATE|0.000254 ± 0.000011 [0.000119, 0.000381]|0.000000 ± 0.000000 [0.000000, 0.000000]|
|G_BIAS_VIDEO|0.000272 ± 0.000014 [0.000139, 0.000398]|0.000000 ± 0.000000 [-0.000000, 0.000000]|
|G_R_shared_STATE|-0.000168 ± 0.000149 [-0.000408, 0.000086]|-0.002632 ± 0.000577 [-0.003268, -0.001946]|
|G_R_shared_VIDEO|-0.000188 ± 0.000049 [-0.000573, 0.000204]|-0.002497 ± 0.000105 [-0.003097, -0.001881]|
|G_R_task_STATE|0.000016 ± 0.000195 [-0.000377, 0.000409]|-0.003088 ± 0.000705 [-0.004097, -0.002091]|
|G_R_task_VIDEO|-0.000355 ± 0.000420 [-0.001034, 0.000262]|-0.002673 ± 0.000555 [-0.003681, -0.001716]|
|R_vs_BIAS_shared_STATE|-0.000422 ± 0.000138 [-0.000672, -0.000160]|-0.002632 ± 0.000577 [-0.003268, -0.001946]|
|R_vs_BIAS_shared_VIDEO|-0.000460 ± 0.000062 [-0.000851, -0.000068]|-0.002497 ± 0.000105 [-0.003097, -0.001881]|
|R_vs_BIAS_task_STATE|-0.000238 ± 0.000185 [-0.000620, 0.000138]|-0.003088 ± 0.000705 [-0.004097, -0.002091]|
|R_vs_BIAS_task_VIDEO|-0.000627 ± 0.000431 [-0.001302, -0.000001]|-0.002673 ± 0.000555 [-0.003681, -0.001716]|
|T_R_STATE|0.000184 ± 0.000063 [-0.000045, 0.000409]|-0.000455 ± 0.000133 [-0.000912, -0.000045]|
|T_R_VIDEO|-0.000167 ± 0.000396 [-0.000580, 0.000209]|-0.000176 ± 0.000509 [-0.000792, 0.000481]|
|V_program_BASE|-0.000783 ± 0.001456 [-0.003692, 0.002059]|0.011053 ± 0.002497 [0.004894, 0.017791]|
|V_program_BIAS|-0.000765 ± 0.001464 [-0.003684, 0.002091]|0.011053 ± 0.002497 [0.004894, 0.017791]|
|V_program_SHARED_RES|-0.000803 ± 0.001452 [-0.003665, 0.002066]|0.011188 ± 0.002936 [0.004840, 0.018050]|
|V_program_TASK_RES|-0.001153 ± 0.001028 [-0.004142, 0.001735]|0.011467 ± 0.002760 [0.005199, 0.018301]|

**解释：**全五任务上，task residual 没有稳定地同时优于 shared residual 的 position 与 safe-FD。state 的 `T_R` position 为正但区间跨 0，safe-FD 为负且区间低于 0；video 的两项 `T_R` 区间跨 0。train-only bias 的 macro position 小幅改善，safe-FD 恒等于 base；residual 的 `R_vs_BIAS` 多数为负，整体未超过简单 bias 校准。任务 90 上 residual 有局部改善，不能外推为五任务一致优势。

真正 seed-LOO/task-LOO 的完整表在 `LEAVE_ONE_SEED_OUT.csv` 与 `LEAVE_ONE_TASK_OUT.csv`。seed-LOO 排除一个 seed、保留另两个 seed 和全部五任务；task-LOO 每个 seed 排除一个 task、保留其余四任务。主要 paired contrast 的范围：

|Contrast|指标|seed-LOO range|task-LOO all-seed mean range|
|---|---|---:|---:|
|G_R_shared_STATE|position_mae|[-0.000227, -0.000085]|[-0.000248, -0.000024]|
|G_R_shared_STATE|safe_fd_l1|[-0.002842, -0.002303]|[-0.003287, -0.002003]|
|G_R_task_STATE|position_mae|[-0.000079, 0.000116]|[-0.000374, 0.000222]|
|G_R_task_STATE|safe_fd_l1|[-0.003361, -0.002690]|[-0.003973, -0.002016]|
|T_R_STATE|position_mae|[0.000148, 0.000204]|[-0.000125, 0.000296]|
|T_R_STATE|safe_fd_l1|[-0.000519, -0.000386]|[-0.000686, -0.000013]|
|G_R_shared_VIDEO|position_mae|[-0.000205, -0.000160]|[-0.000300, -0.000034]|
|G_R_shared_VIDEO|safe_fd_l1|[-0.002557, -0.002461]|[-0.003120, -0.001923]|
|G_R_task_VIDEO|position_mae|[-0.000595, -0.000205]|[-0.000727, 0.000096]|
|G_R_task_VIDEO|safe_fd_l1|[-0.002870, -0.002356]|[-0.003441, -0.001781]|
|T_R_VIDEO|position_mae|[-0.000395, -0.000045]|[-0.000551, 0.000130]|
|T_R_VIDEO|safe_fd_l1|[-0.000333, 0.000117]|[-0.000344, 0.000434]|

## RGB 诊断

固定 checkpoint RGB 探针为 12 个新 job / 36 个 task-condition-seed views；另复用 18 个 matched V2 references，其中 15 个 G-V views 与主比较的 30 个旧 base views 重叠，另有 3 个 S-V-90 views。G-V 上零输入与 same-task donor 的平均 `condition−matched` position/F-D 变化分别为 `0.046559/0.024284` 与 `0.030291/0.032248`；正数代表扰动使误差上升。S-V-90 position 也上升，safe-FD 变化跨 seed。
state-only QA 覆盖 5 个任务；base output、h、非零 shared-R 与 task-R 的最大变化都为 `0.0`。这说明这些 state-only 输入路径对所测 RGB block 扰动不敏感；video 对照显示输入扰动会改变 h/output 和误差，但不能推出视频信息有因果效用。详见 `results/scoring/attempt_02/RGB_PROBE_RESULTS.csv`。

## GPU、执行与访问审计

宿主 `restricted-compute` 在 admission 快照有 4 张 RTX 3090（24,576 MiB/card）；GPU0/1 分别空闲 24,555/24,560 MiB，GPU2 当时利用率 93%、空闲 10,968 MiB，GPU3 空闲 2,266 MiB。按共享 watcher 的 `dynamic_all_visible` 与 hard cap 2 并行，实际用物理 GPU 0, 1（子进程逻辑设备 `cuda:0`）；57 个 watcher attempts 中 43 complete、14 failed，全部 lease 与 child slot 都释放，未向非 child 进程发信号。
成功 workload 的最大应用峰值为 `224.1 MiB allocated / 276 MiB reserved`（RGB probe）；readout fit 最大 `24.7/46 MiB`，feature extraction 最大 `61.8/88 MiB`。20 个 fit 记录的 GPU elapsed 合计 `168.4s`；43 个成功 watcher job 的逐 job wall-time 合计 `347.3s`（并行总和，不是端到端墙钟；失败尝试另计）。
watcher provenance：当前源码与 V2 source acceptance/later phase receipt 都是 `b703ef125ae481819db4af20c66a2a13906098e6522fe24fe2ce1735164355a5`；`ENVIRONMENT_PRECHECK.json` 原记录为 `49fcf49e…`，保留该 hash 不一致且未修改 watcher。
seal 前 dev target rows 为 0。V3 scorer 的 attempt_01 在 seal 后读了 5,550 个 future action rows 后因 V2 `seed=shared` 汇总行解析失败；attempt_02 再读 5,550 行并评分通过。V3 实际共读 11,100 行，V2 历史 16,650 行保留，累计 27,750；final/heldout 一直为 0。示例轨迹复用已有 V2 derived plot data 的 300 个 target trace points，单独记账；没有再打开 scorer-only target artifact。

## 保留的失败与限制

`qa/FAILED_ATTEMPTS.jsonl` 保留 43 条首轮失败。12 个 formal attempt_01 在 optimizer update 前失败，修正后 12/12 通过；18 个主 CPU prediction attempt_01 修复 shape/key 后 18/18 通过；两次 seal 失败均在读标签前；score attempt_01 是唯一造成额外 postseal dev reads 的失败。RGB figure attempt_01 的空条件分组图也已备份并修正，评分结果未受影响。
缓存 extractor/session/trial 绑定仍有既有证据限制；dev 已在 V2 多次访问，本轮是 post-V2 exploratory，不是首次 dev 暴露、独立确认、OOD/ICL 或真实机器人结论。当前结果保留负向与 seed-sensitive 结果。人工/教授复审仍 pending。

## 文件索引

- 完整机器可读回传：`RETURN_TO_CHATGPT.json`。
- per-task/per-seed 双指标：`results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv`。
- paired contrasts、LOO、条件区间：同目录 `CONTRASTS_BY_TASK_SEED.csv`、`LEAVE_ONE_SEED_OUT.csv`、`LEAVE_ONE_TASK_OUT.csv`、`PAIRED_SOURCE_GROUP_CI.csv`。
- 预测封存、访问、验收：`protocol/PREDICTION_BATCH_SEAL.json`、`qa/DEV_TARGET_SCORING_QA.json`、`qa/FINAL_ACCEPTANCE.json`。
- figures 与 captions：`figures/`；绘图数据：`figure_data/`。Checksum-last 清单：`CHECKSUMS.sha256`。
