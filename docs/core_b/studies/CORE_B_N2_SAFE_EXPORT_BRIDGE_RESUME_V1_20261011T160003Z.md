> Archived study summary for `CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z` (source snapshot `20261011`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# N2 安全模型导出与后端桥接续跑报告

## 执行状态

- 状态：COMPLETE_N2_BRIDGE_SAFE_EXPORTS_ACCEPTED_N3_DIRECT_REFERENCE。
- 新结果根：server-only://project-artifacts/reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z
- Safe model-only 导出：12/12 safetensors；fresh-process safetensors 读取和 strict model load 12/12 通过，参数数 890,780。
- N2 矩阵：24/24 development views 通过，其中 CPU_COMMON 12 个、CUDA_COMMON 12 个；每个 2,322 行 × 30 × 28，603 source groups；推理未读取 target arrays。
- 额外 source-profile QA（不计入 N2_MATRIX24）：Q0 CPU 6/6、Q1 CUDA physical GPU 0 6/6 与接受的 N1 stored dev predictions 逐元素完全一致，max absolute difference=0。
- 新 common-profile 评分与已接受 N1 stored 评分逐项完全复现：24 个 co-primary 分数 max absolute difference=0。

## N2 common-profile 效应

定义：G = L(Q0) − L(Q1)，正值代表 Q1 误差更低；V = L(STATE-CTRL) − L(TS11-D)；I = G_TS11-D − G_STATE-CTRL。区间为 10,000 次 paired source-group percentile bootstrap，固定 3 个 seed、603 groups、seed 20261011；未计算 p 值。

| 对比 | Position MAE：CPU_COMMON | Position MAE：CUDA_COMMON | Safe-FD：CPU_COMMON | Safe-FD：CUDA_COMMON |
|---|---:|---:|---:|---:|
| G_TS11-D | +0.001121 [+0.000379, +0.001833] | +0.001121 [+0.000379, +0.001833] | +0.005342 [+0.004092, +0.006568] | +0.005342 [+0.004092, +0.006568] |
| G_STATE-CTRL | -0.002803 [-0.003891, -0.001764] | -0.002803 [-0.003891, -0.001764] | +0.000981 [-0.000200, +0.002159] | +0.000981 [-0.000200, +0.002159] |
| I | +0.003924 [+0.003053, +0.004799] | +0.003924 [+0.003053, +0.004799] | +0.004362 [+0.002819, +0.005904] | +0.004362 [+0.002819, +0.005904] |
| V_Q0 | -0.004387 [-0.005067, -0.003739] | -0.004387 [-0.005067, -0.003739] | -0.007872 [-0.009363, -0.006426] | -0.007872 [-0.009363, -0.006426] |
| V_Q1 | -0.000463 [-0.001136, +0.000190] | -0.000463 [-0.001136, +0.000190] | -0.003511 [-0.004518, -0.002529] | -0.003511 [-0.004518, -0.002529] |

common backend 对比：各 view 的 position B = L_CUDA − L_CPU 平均值最大绝对值约 4×10⁻¹⁰；safe-FD B 最大绝对值约 6.8×10⁻⁹。12 对预测的平均绝对差均值为 7.464e-09，跨对最大元素绝对差为 2.861e-06. 这些差值接近 float32 数值差；查询结论在两个 common backend 下保持一致。

存储 profile 漂移单独列出：Q0 common CPU 与历史 CPU 的 co-primary group score 漂移为 0；Q1 common CUDA 与历史 CUDA 的 position 漂移均值 TS11-D 2.522e-08、STATE-CTRL -1.279e-07；历史 Q1 inference 在物理 GPU 0，新 CUDA_COMMON 使用 GPU 2（同为 RTX 3090）。详见 BACKEND_PROFILE_DRIFT.csv 与 PROFILE_DRIFT_DECOMPOSITION.csv。

固定 seed 的不稳定性保留：TS11-D position G 的三个 seed 为 -0.008257, +0.014806, -0.003185；排除 seed 20261002 后两 seed 均值为 -0.005721. 因此不能称为所有 seed 一致获益。跨 seed group CI 是 low precision，只作参考。

## N3 直接复用

N3 未重评分、未训练：直接引用 sealed prior root 的 18 个 accepted predictions 与 162-row score reconciliation。此前结果：FULL g=+0.000631（CI 跨 0）、OBS g=−0.003568（CI 低于 0）、ANCHOR g=+0.004195（CI 高于 0）；h_command=+0.001696、h_observation=+0.017611、J_OBS=−0.004200、J_ANCHOR=+0.003563。不呈现“history 越完整，视觉增益越小”的单调关系；ANCHOR 仍含当前 action/skip。它们是固定 seed、已暴露 dev 的结果，不是独立确认。

## 资源、安全与边界

- 受限恢复通过实际 require_escalated 工具执行。首个尝试在 checkpoint load 前因计划缺 query table SHA 退出；第二次在精确临时 NumPy safe_globals 下完成 12/12 提取并生成 staged safetensors，随后独立进程 safe-only reload、tensor-catalogue 对照和 strict model load 均通过。调用记录见 export_review/APPROVAL_DECISION_RECORD.json；平台工具没有返回独立 decision ID，未伪造批准记录。
- 未使用 weights_only=False、永久 safe-global、任意类、环境绕过或自制 pickle。没有新训练、optimizer steps 或 encoder extraction。
- CUDA_COMMON 使用物理 GPU 2（UUID GPU-00465dc6-e96e-67f1-285a-84ed61565ac2，CVD=2，logical cuda:0），最大 allocated 71 MiB；原 profile Q1 source replay QA 使用 GPU 0。两次 watcher 作业 exit 0；tmux 和 child/watcher PID 均已结束，使用后显存恢复。
- Scoring target-access log 有 2 次同一已暴露 dev 切片访问：第一次在写表前遇到代码键错误，第二次 scoring 通过；唯一使用的数据是 2,322 dev rows，最终 test/heldout access=0。
- 新训练=0，optimizer steps=0，encoder extraction=0，N3 rerun/rescore=0；旧根 checksums SHA 仍为 60701b7b…，旧根写入=0，共享 watcher 未修改。
- 两张图和图件元数据/图像 QA 已通过 in-session review；独立人工审阅和期刊规范审查未做。图件：figures/N2_QUERY_EFFECTS.png、figures/N2_BACKEND_PROFILE_DRIFT.png。

## 关键产物

- 24-view acceptance：qa/N2_PREDICTION_ACCEPTANCE.json
- N2 效应、seed/LOO、bootstrap 与 drift：results/N2_BY_SEED.csv、results/ALL_SEED_LOO.csv、results/BOOTSTRAP_CI.csv、results/BACKEND_PROFILE_DRIFT.csv、results/PROFILE_DRIFT_DECOMPOSITION.csv
- Safe exports 与 strict reload：safe_models/、qa/SAFE_EXPORT_ACCEPTANCE.json、qa/STRICT_MODEL_REPLAY.json
- N3 direct-reference manifest：n3_refs/ACCEPTED_N3_ARTIFACT_MANIFEST.csv
- Final acceptance 与 checksum：qa/FINAL_ACCEPTANCE.json、CHECKSUMS.sha256（checksum 在最后生成并复核）。

