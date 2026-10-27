> Archived study summary for `CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z` (source snapshot `20261012`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# N4 固定 command 模型误差预算：执行回传

状态：**完成**。结果根：server-only://project-artifacts/reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z

## 执行与验收

- 固定矩阵 12 models × train/dev，24/24 bundles 接受：12 个 dev 直接复用 N2 CUDA_COMMON，12 个 profile 不匹配的 train bundle 通过 N2 已接受 safetensors 做 target-blind CUDA forward。没有重训、optimizer step、encoder extraction 或新 dev formal forward。
- 复用 N1 scorer/normalizer，24 条原 co-primary dev 对照逐项复现，最大绝对差 0（容差 1e-12）。position MAE 与 safe-FD L1 保持 co-primary；safe-FD 是未来内部 29 个差分，anchor-to-first-frame seam 单独报告。
- inference 使用物理 GPU2，UUID GPU-00465dc6-e96e-67f1-285a-84ed61565ac2，CVD=2 / logical cuda:0，FP32、batch64、TF32 off、deterministic、math-only SDPA。实测峰值 71 MiB，watcher exit 0，lease/child slot/tmux 均释放，GPU2 回到 20 MiB used；lease 区间 116.37 秒（0.03232 GPU-hours），新训练成本 0。
- target gate 在 24/24 prediction acceptance 后开启。5 次 scoring attempts 中 4 次因代码错误在提交结果前停止；日志与 staging 均保留。成功 attempt 计分 primary-train 17,757 rows 和 exposed-dev 2,322 rows；V3 NPZ 的 targets_native key 每次物理物化其 20,079-row canonical train+dev 内容，但只有锁定 split slice 进入评分。case 图另外选取 14 个预锁定 dev anchors。FINAL/heldout access=0。
- dev interval 使用 N2 的 10,000 次 paired source-group 与 seed draws（seed 20261011）；区间条件于固定模型和已暴露 dev。没有 p 值；train 为 in-sample，不提供泛化 CI。

## A. seed 20261002：level 与 time-centered

TS11-D 的 Q0→Q1 position standardized MAE 从 0.209936 降到 0.195131（G=Q0−Q1=+0.014806）；safe-FD 从 0.976491 降到 0.956604（G=+0.019888）。
position L2 MSE gain=+0.026511: level component +0.012870, time-centered component +0.013642. 两部分都下降，time-centered 略大，因此 S02 的变化不只是整段路径的高低平移。独立 future-GT oracle MAE ledger 中，removable 项 gain=+0.004658, oracle-corrected remaining 项 gain=+0.010148. 剩余项改善更大；这不是主要由可移除 level offset 驱动。oracle 只作诊断，不能作为部署校正。
该变化高度 seed-sensitive：TS11-D 的 position G 三 seed 为 -0.008257, +0.014806, -0.003185, safe-FD G 为 -0.003783, +0.019888, -0.000078. 去掉 S02 后，position G 均值为 -0.005721, safe-FD 为 -0.001931.

## B. 动态幅度与零时滞对齐

TS11-D/S02/Q0：T=6.025032, P=1.806e-09, C=4.461e-06, r=1.73e-05, c=0.0428, skill=1.48e-06. P 非零但极小，故没有套用 P=0 规则；c 在近零 P 下应谨慎看待，不设新阈值。
切换到 Q1 后 T 不变，P=0.397155, C=0.412184, r=0.257, c=0.266, skill=0.071. 预测变化 RMS 约为目标的 25.7%, zero-lag cosine 为正但中等。归一化 FD-MSE 的 norm 项=0.552, alignment 项=0.377. FD-MSE G=+0.427204, 分解为同一 S02 的 G_P=-0.397155 与 −2G_C=+0.824358. c 是 zero-lag cosine，不是 Pearson 或时间滞后检测；skill 只相对零导数 persistence，不替代 position MAE 或 safe-FD L1。

## C. G/V/I 与绝对优势

三 seed TS11-D G 的 position 均值=+0.001121, sample SD=0.012119, median=-0.003185. safe-FD 均值=+0.005342, sample SD=0.012733, median=-0.000078. 对应 position G 的 95% group-bootstrap interval 为 [-0.008074, +0.014395]；safe-FD 为 [-0.003915, +0.018774]，两者都跨 0。
Q1 下 V_Q1=STATE−TS：position 均值=-0.000463, 95% interval [-0.003293, +0.002424]；safe-FD 均值=-0.003511, interval [-0.006008, -0.000864]。负值表示 TS11-D 误差更高；当前结果没有建立 TS 的原 co-primary 绝对优势。该 interval 是已暴露 dev、固定模型的探索性组 bootstrap，不是独立确认或显著性检验。

完整逐 seed G/V/I、三 seed mean/SD/median/LOO、T/P/C component CI 与 per-axis moments 见 results/PAIRED_COMPONENT_GAINS.csv、results/ALL_SEED_LOO.csv、results/GROUP_BOOTSTRAP_CI.csv、results/PER_WINDOW_MOMENTS/ 和 results/N4_EFFECT_SUPPORT_SUMMARY.json。

## N3 与其他窗口

N3 仅直接引用 18 accepted predictions 与 162-row reconciliation，未重评分或重训。此前 FULL g≈+0.000631（区间跨 0）、OBS g≈−0.003568、ANCHOR g≈+0.004195，呈非单调关系。
HORIZON root 的 RUN_STATUS/FINAL_ACCEPTANCE/return 标为 complete/pass，但 OWNER_STATE 仍为 LOCKED_READY_FOR_CPU_FIT，本 N4 记录为 stale/contradictory 且未重启。OBS43 closeout root 为 sealed/pass；另一同名 root 是空目录（0 files，完成状态 UNKNOWN）。两者均未修改或重跑。

## 图件与边界

交付 5 张 summary figure、14 个锁定 case page、1 个 contact sheet；PNG/PDF/caption/data file QA 与 agent visual review 通过，human review pending。case 数据来自 14 个预先锁定的不同 dev groups；时间标 future step 1–30，raw axes 0/14，物理单位未核验。
下一轮独立确认条件只写入 NEXT_RESEARCH_CONDITIONS.md，未执行。QUERY_LINE_CLOSED.md 关闭本 query 扩张。独立确认=false、D5=false、final/test=0、submission-ready=false。

目标访问记录：5 次 scoring attempts，4 次 pre-commit code failures；每次 unique scoring support 未变。NPZ target key 的 20,079-row canonical train+dev 内容因压缩格式物化 6 次（5 次 scoring +1 次 case target extraction），实际评分仍只用 train 17,757 / dev 2,322，case page 只索引 14 个 dev anchors；final/heldout=0。
N2/N3 manifest SHA 保持 21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f 与 60701b7b357e5282222c60b15994a1408fcc1398618025dbebe5388fc69f1a75；N2 本轮 selected source 文件 111 项复验通过，旧根写入 0，共享 watcher 未改。最终文件清单见 CHECKSUMS.sha256，封存 QA 见 qa/FINAL_ACCEPTANCE.json。
