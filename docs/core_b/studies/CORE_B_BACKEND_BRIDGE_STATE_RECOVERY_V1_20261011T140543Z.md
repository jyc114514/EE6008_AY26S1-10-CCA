> Archived study summary for `CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z` (source snapshot `20261011`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# N2/N3 后端桥接与 state-information 结果回收

**总状态：partial。** N3 的 18 个已接受 H30 dev 预测已完成重评分、原分数复现、paired/group 统计和图件；N2 在固定 checkpoint 的安全加载门被阻断，未产生 CPU_COMMON/CUDA_COMMON 视图。没有追加训练、推理或 encoder extraction。

结果根：`server-only://project-artifacts/reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z`

## N0 既有 function-sensitivity 结果（本轮未重跑）

先前接受的 N0 有 6/6 REAL strict replay 与 12/12 ZERO/PERM interventions，2322 dev anchors、603 source groups。三 seed 上，两模型的 REAL MAE 均低于 ZERO/PERM probe。平均 `ΔL= L_probe−L_REAL`：F-POOL-D 为 ZERO `+0.004390`、PERM `+0.007534`；ALT-POOL-R3D18 为 ZERO `+0.001425`、PERM `+0.004223`。跨固定 seed 的平均 D_out/D_level/D_shape 也显示对扰动有响应：F-POOL-D ZERO `0.021450/0.020880/0.012438`、PERM `0.032351/0.031502/0.018297`；ALT-POOL-R3D18 ZERO `0.013844/0.013432/0.008307`、PERM `0.023179/0.022521/0.013508`。这证明指定扰动下存在 function sensitivity；ZERO/PERM 离开自然联合分布，不能解释为视频的因果或独立信息价值。详见既有 [N0/N1 交接报告](server-only://CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z/RETURN_TO_CHATGPT.zh-CN.md)。

## N2：24 个 common-profile 视图未运行

已从 N1 接受清单核对 12 个固定 checkpoint 的 SHA；12/12 均无法由项目 PyTorch 2.12.1+cu130 的 `torch.load(weights_only=True)` 安全加载，报错指向不允许的 `numpy._core.multiarray._reconstruct`。检查到的 checkpoint 目录没有 tensor-only 或 safetensors 模型副本。没有使用 `weights_only=False`，没有添加 safe-global allowlist，也没有提交新的 CUDA 子进程。N2 矩阵保留了全部 12 checkpoint × CPU_COMMON/CUDA_COMMON 计划，当前 accepted=0/24；backend/profile drift、G/I/V common-profile 分解因此仍为未知。

历史 N1 sidecar 确认 Q0 用 CPU；Q1 用 CUDA，物理 GPU 0、logical `cuda:0`。Q1 使用 FP32、未启用 AMP/compile、TF32 matmul 关闭，但 TF32 cuDNN 开启、deterministic algorithms 关闭，flash/math/memory-efficient SDPA 均开启。Q0 的部分 CPU/BLAS 线程字段未记录。这些旧结果仍是 AS_RUN_REPLAY 参考，不能代替共同 profile 桥接。N1 的 `V_Had` 两系统都来自 CUDA；不能把它的负值全部归因于旧 Q0 CPU 后端。

单独的 VideoMAE handoff acceptance 记录了此前对本地 Core B checkpoint 使用 `torch.load(weights_only=False)` 的自动审核拒绝，理由是任意 pickle 代码执行风险，并指出安全 weights-only/state-dict 路径。该拒绝属于既有 handoff 证据；本轮没有请求或尝试该不安全加载。

## 宿主资源与作业门

预检时间 2026-10-11 14:04:29 UTC，主机 `restricted-compute`，未发现 Slurm allocation，scheduler 身份未知；项目 Python CUDA 初始化成功，Torch 2.12.1+cu130 / CUDA build 13.0，4 张 RTX 3090 可见。GPU 0/1/2 当时各约 24.1 GiB free；GPU 3 约 1.8 GiB free（约 22.3 GiB used），未查询或打印其他用户 PID。`/usr1` 当时约 5.8 TiB free；quota CLI 不可用，用户 quota 未知。

STATE 根的旧 lease 文件指向 2026-09-20 的 PID，复核时均不存在。D3EQ remaining24 的精确 tmux session 未找到；它的日志于 2026-09-25 写入 `formal_phase_complete_or_partial`、36 个 formal cells，且该根没有 gpu_lease 目录。没有停止或修改这些旧状态。此次 N2/N3 新 GPU children=0，GPU-hours=0，共享 watcher/policy 未改。

## N3：18/18 已接受视图复评分通过

FULL/OBS/ANCHOR × N/D × seeds 20260953/54/55 的 18 个 `primary.npz` 均逐项通过旧 root checksum、prediction marker、checkpoint SHA、epoch 40/11,120 steps 及 final-test denial 检查后，复制到新根。形状均为 2322×30×28，ID 与 canonical dev 顺序一致，有限值通过。没有新推理。

重评分与原 N3 接受表的 162 项指标比较全部通过，最大绝对差为 `5.55e-17`。原聚合是 window→episode→等权 source group；603 episode 与 603 source group 一一对应。position 主指标使用 N3 train-fitted `target_sigma`。原分析的 `full_fd_scaled` 继续按 `target_sigma` 计算；另列 `full_fd_safe_normalized`，其分母是 N3 train-fitted `safe_fd_scale`。N3 原分析没有 direction 指标，本轮未另造该指标。

| State | L(N), mean±seed SD | L(D), mean±seed SD | g=L(N)−L(D) | 95% paired group CI |
|---|---:|---:|---:|---:|
| FULL | 0.195032 ± 0.001294 | 0.194401 ± 0.001243 | +0.000631 | [−0.000045, +0.001278] |
| OBS | 0.196728 ± 0.001158 | 0.200296 ± 0.008688 | −0.003568 | [−0.004262, −0.002889] |
| ANCHOR | 0.214339 ± 0.001019 | 0.210144 ± 0.007512 | +0.004195 | [+0.002648, +0.005766] |

每个 CI 用 10,000 次 source-group paired percentile bootstrap、固定三个训练 seed、PRNG seed 20261011。正 g 表示该状态下动态视频误差更低。FULL 接近零且 CI 跨零；OBS 下动态视频误差更高；ANCHOR 下动态视频误差更低。OBS 的 g 受 seed 20260955 影响明显：该 seed 的 g 为 −0.012315，排除它的两 seed 均值转为 +0.000805。ANCHOR 的三个 g 是 +0.008889、+0.006947、−0.003252，三个 leave-one-seed-out 均值仍为正。

| Contrast | Mean | 95% paired group CI | 解释 |
|---|---:|---:|---|
| h_command at N = L(OBS,N)−L(FULL,N) | +0.001696 | [+0.001092, +0.002271] | 去掉历史 action 列后误差上升 |
| h_observation at N = L(ANCHOR,N)−L(OBS,N) | +0.017611 | [+0.015599, +0.019676] | 再去掉 observation history 后误差明显上升 |
| J_OBS = g_OBS−g_FULL | −0.004200 | [−0.005003, −0.003407] | OBS 的视频增益比 FULL 更负 |
| J_ANCHOR = g_ANCHOR−g_FULL | +0.003563 | [+0.001967, +0.005193] | ANCHOR 的视频增益相对 FULL 更正向 |

因此，这批模型没有呈现“history 越弱，视觉增益越大”的单调关系：OBS 的视频条件更差，ANCHOR 的视频条件更好，FULL 的差异接近零。结果支持当前模型/recipe 对 state 条件敏感；它不证明视觉输入对完整任务有普遍因果增益，也不代表独立确认。`J>0` 只描述相对交互，单独并不等于绝对视频收益。ANCHOR 仍保留独立的 28D 当前命令 anchor token 和 native residual skip，不是 strict RGB-only；OBS 仍保留当前命令 anchor token。原始输入构造明确将 43 个 observation 列后接 28 个 action 列，N3 mask 的 action 起点为 43，故 OBS 屏蔽索引 43–70；没有按维数猜列位置。

## N1 接受结果与训练表恢复

从已接受的 `TRAIN_DEV_BY_SEED.csv` 取回 12 个固定模型的 train 行（另保留一个 deterministic persistence reference）。表中是 in-sample train 表现，均值±三个 seed 的样本 SD，不用于泛化比较。

| Model | Query | Position MAE / train σ | safe-FD normalized |
|---|---|---:|---:|
| TS11-D | Q0 old | 0.190247 ± 0.013744 | 0.953429 ± 0.022506 |
| TS11-D | Q1 Had | 0.189481 ± 0.005941 | 0.945242 ± 0.006148 |
| STATE-CTRL | Q0 old | 0.184829 ± 0.000713 | 0.943067 ± 0.001060 |
| STATE-CTRL | Q1 Had | 0.189515 ± 0.001516 | 0.941136 ± 0.003619 |

N1 已接受 dev 的 as-run 混合后端结果仍显示：position 的 `G_TS=+0.001121`（95% group CI `[+0.000375,+0.001843]`，seed SD 0.012119），`G_STATE=−0.002803`（`[−0.003886,−0.001750]`），`V_old=−0.004387`（`[−0.005050,−0.003710]`），`V_Had=−0.000463`（`[−0.001119,+0.000183]`），`I=+0.003924`（`[+0.003028,+0.004821]`）。safe-FD 的同一顺序为 `+0.005342`、`+0.000980`、`−0.007872`、`−0.003510`、`+0.004362`；对应 CI 与 per-seed/LOO 值见 `N1_AS_RUN_REFERENCE.csv`、`N1_SEED_MEDIAN_ALL_LOO.csv`。TS11-D position G 的 seed 值为 −0.008257/+0.014806/−0.003185，seed 20261002 拉高均值，故不能称为稳定 query 收益。N2 未能测出该混合结果中的后端/profile 漂移。

## VideoMAE background（不并入 leaderboard）

本地保存的 VideoMAE summary 表中，H30 fusion 为 `0.195368 ± 0.001749`、其 STATE-CTRL 为 `0.194846 ± 0.001107`；H60 分别为 `0.268999 ± 0.001557`、`0.268134 ± 0.000466`。handoff 的 top-level acceptance 为 partial，且该方案的 backbone、feature layout 和 projection 与 N1/N3 不同；只作为背景，不并入本轮匹配对比或因果结论，也没有加载其 checkpoint。

## QA 与续跑

- N2 safe checkpoint blocker：`qa/CHECKPOINT_SAFE_LOAD.json`；24-view 计划：`protocol/N2_MATRIX24.csv`；历史与受控 profile 状态：`protocol/AS_RUN_PROFILE_REGISTRY.csv`、`results/N2_PROFILE_DRIFT_DECOMPOSITION.csv`。
- N3 18-view acceptance：`qa/PREDICTION_ACCEPTANCE.json`；完整分数、每 seed/LOO、bootstrap 与 figure data 均在 `results/`、`figure_data/`。
- 最终图件：`figures/FIG01_N3_VIDEO_GAIN_BY_STATE.png/.pdf`、`figures/FIG02_N3_INFORMATION_INTERACTIONS.png/.pdf`；PNG 全图预览及 PDF 页尺寸/嵌入字体检查通过。human review 仍待签核，`submission_ready=false`。
- N2 无需重训；安全 tensor-only/safetensors 副本与其源 checkpoint hash、参数/buffer identity 到位后，才可恢复 24-view shared-watcher bridge。见 `coordination/RESUME_COMMANDS.md`。
- 旧根核验限于 source registry 的 121 个唯一文件，所有来源与新根副本 SHA 匹配；这不是对旧根全量重验收。此次对旧根写入动作数为 0。

```ini
STATUS=partial
N2_STATUS=source_blocked
N2_DEV_VIEWS_EXPECTED=24
N2_DEV_VIEWS_ACCEPTED=0
N2_NEW_INFERENCE_VIEWS=0
N3_STATUS=complete
N3_H30_DEV_VIEWS_EXPECTED=18
N3_H30_DEV_VIEWS_ACCEPTED=18
N3_NEW_TRAINING=0
N3_NEW_INFERENCE=0
NEW_TRAINING=0
OPTIMIZER_STEPS=0
NEW_ENCODER_EXTRACTION=0
FINAL_TEST_ACCESS=0
OLD_ROOT_WRITES=0
SHARED_WATCHER_MODIFIED=false
MAX_PROJECT_GPU_CHILDREN=2
ACTUAL_NEW_GPU_CHILDREN=0
INDEPENDENT_CONFIRMATION=false
N3_REUSE_STATUS=legacy_descriptive
CHECKSUM_ACCEPTANCE=PASS_CHECKSUMS_SHA256_GENERATED_LAST_AND_VERIFIED
```
