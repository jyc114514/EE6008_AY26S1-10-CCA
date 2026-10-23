> Archived study summary for `CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z` (source snapshot `20261016`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V8 固定教师转移诊断回传

STATUS=complete_with_limits
STUDY=post_V7_frozen_transfer_diagnosis_development
POST_RESULT_INFORMED_DIAGNOSTIC=true
INDEPENDENT_CONFIRMATION=false

## 1. 学生是否更好地拟合规定训练目标 Z

按最终 epoch 160 的冻结 eval 输出，以相同 Z 比较 G=GT_RETRAIN 与软目标学生。J 使用 V7 runner 的 position L1 + 0.1 internal-FD L1、原 Y 拟合尺度及 task/episode 权重；正的 G−student 表示软目标学生更好。
- MIX 对 Z_M：mean J gain = -0.002499（seed SD 0.000541）；三 seed = 202610131:-0.002268, 202610132:-0.002113, 202610133:-0.003117.
- SELF 对 Z_U：mean J gain = -0.004041（seed SD 0.000684）；三 seed = 202610131:-0.004050, 202610132:-0.003353, 202610133:-0.004720.
- 两项均三 seed 为负：在相同规定 Z 上，软目标学生的最终 eval loss 没有胜过硬监督 G。该结果不证明容量上限或未收敛。
- 最终 train 对真实 Y：MIX position MAE mean 0.135593, safe-FD 0.784323, seam 0.758811. 均为锁定尺度下的标准化误差，不是物理单位。详情见 TRAIN_TARGET_AND_FIT.csv。
- V7 epoch 曲线是 dropout-on、pre-update 汇总。跨三 seed 的 epoch160 objective 均值 G/U/M=0.262412/0.144083/0.146343；last20 min/median/max 等 seed 平均分别 G 0.262304/0.262594/0.262865、U 0.143953/0.144198/0.144472、M 0.146131/0.146395/0.146748。不同 Z 的 objective 不作准确率横比；有限值检查通过，梯度范数未记录。

## 2. 输出是否朝教师 T 移动，train 与 dev 的差异

- Train 到 T 的距离 G→M：position 0.010673→0.012874；safe-FD 0.036858→0.052297；seam 0.086166→0.109849.
- Dev 到 T 的距离 G→M：position 0.019022→0.025062；safe-FD 0.067520→0.092728；seam 0.106275→0.138593.
- M 在这六个距离上都比 G 更远；均为三 seed 均值，样本 SD 在 RETURN_TO_CHATGPT.json 与 TRAIN_DEV_IMITATION.csv。
- correction geometry 部分对齐但不能抵消距离增长：A/C position cosine/k train 0.408/0.388、dev 0.350/0.436；internal-FD train 0.373/0.509、dev 0.376/0.494。E/C position cosine/k train 0.329/0.257、dev 0.042/0.047；internal-FD train 0.181/0.211、dev 0.049/0.062。k 是描述性投影，不是部署系数或因果证据。
- C/A/E RMS：position train 0.0350/0.0333/0.0274、dev 0.0648/0.0808/0.0722；internal-FD train 0.1296/0.1766/0.1506、dev 0.2860/0.3756/0.3654；seam train 0.2839/0.3801/0.3492、dev 0.4161/0.6586/0.5425。
- whole-seed/task LOO 保持全部 G/U/M/S/V/T 行并重新池化。例：A/C position k 的 train seed-LOO 范围 0.364–0.401、task-LOO 0.382–0.398；E/C position k 的 dev seed-LOO -0.024–0.128、task-LOO -0.014–0.101。完整 position/FD、cos/k/orthogonal-energy 范围见 CORRECTION_GEOMETRY_LOO.csv；投影正交和能量恒等式误差通过锁定容差。

## 3. MAE、FD、seam 的任务局部结果，以及 step/coordinate 限制

- V7 已接受 macro：G position/FD/seam = 0.158338/0.819754/0.795761; M = 0.159188/0.824029/0.775311. G loss−M loss gains 为 -0.000850 / -0.004274 / +0.020449；正 gain 有利于 M。
- 原 V7 source-group 95% CI：position [-0.002180,+0.000579]，safe-FD [-0.008554,-0.000481]，seam [+0.007785,+0.035567]；仅复用原 CI，本轮无 bootstrap/p 值。
- 按 task 平均三 seed 的 G loss−M loss gain，位置 MAE：119 -0.000732、94 -0.000407、90 -0.000027、141 -0.003101、230 +0.000019；FD：119 -0.007583、94 -0.004122、90 +0.000050、141 -0.009753、230 +0.000038；seam：119 +0.060523、94 +0.005492、90 +0.005303、141 +0.030055、230 +0.000875。
- 退化主要见 task 141 的 position/FD，另有 task 119/94 的 FD；seam gain 最大在 task 119 与 141。每个 task/seed/contrast 见 DECISION_TABLE.csv。
- V7 每窗口 help/harm 表显示三 seed 合计 555 个窗口/metric：position teacher-help 345、harm 210；FD 326/229；seam 381/174。窗口数不是独立 N；加权 contribution 与 M 对 G gain 分层见 TEACHER_HELP_HARM.csv。
- 30 帧 position step、29 个 internal-FD transition、28 个坐标误差以及带 Y 的例子轨迹在本轮标为 UNAVAILABLE。第二次提取因向量聚合错误在唯一 dev archive load 后中止；已按锁定规则未重开 archive。V7 无可复用的 per-step/per-axis 明细。

## 执行、资源与建议

- 本轮新 fit=0，optimizer update=0，teacher forward=0，新 dev prediction=0；仅 6 个 U/M final train-prefix frozen forwards，均在物理 GPU 1（逻辑 cuda:0），状态哈希不变。
- 实际 V8 读取累计：train Y NPY 3 次、9 个 Z 各 3 次；dev target archive 1 次/185 IDs；dev prefix 1 次；prediction_native tensor arrays 100 次（含 1 次 safe_open schema probe、两次失败 CPU extraction、一次 salvage；所有文件均只读）。V8_RESOURCE_AND_EXPOSURE.csv 分列成功/失败成本。
- V7_RESOURCE_SUPPLEMENT.csv 从九份接受 model manifest 补列，9 行；formal peak allocated 均 139.799 MiB、reserved 均 182 MiB；host RSS UNKNOWN。V8 新推理实际 peak allocated 36.318 MiB、reserved 50 MiB。
- Final heldout payload access=0；旧 V7 root writes=0；V7 manifest SHA-256 58c50a9928106c2cf95fe78c5d3a0bdaa05c050a271803d01a6ef8e011810a29.
- 建议关闭当前固定 0.5 soft-target 方案；MIX50 只保留双 head 开发精度参考，G/V 保留单 head 参考。下一问题：NONE。

V8 root: server-only://project-artifacts/reports/benchmarks/CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z
最终 manifest SHA-256 与文件核验数见 qa/FINAL_CHECKSUM_VERIFICATION.json。
Figures were visually inspected in reduced previews; human review remains pending.
