> Archived study summary for `CORE_B_BENCHMARK_LAB_V1_20260920T065538Z` (source snapshot `20260920`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B 独立 Benchmark 当前交付报告

生成时间：`2026-09-22T05:03:49.067966Z`

## 总状态

`BENCHMARK_STATUS=complete`。本根是独立 development benchmark；原 V3 producer、旧 Core B 根和 raw 数据未被修改，final/held-out access=`0`。

- 实验家族：`22`；登记模型/基线行：`124`；artifact inventory：见 `inventory/ARTIFACT_INVENTORY.csv`。
- V3：训练 `36/36`，sealed prediction `36/36`。
- Recovery B：历史 6/6 development predictions 已验收，但 candidate gate 仍 FAIL；不晋级为本轮新模型。
- D1：CPU neighbor baseline 完成，V3 learned sparse 已完成；dense/新 D3 扩展由状态字段决定。
- D2：事件 registry、sparse baseline/learned 已完成；dense learned 状态见 `d2/D2_DENSE_LEARNED_SUMMARY.json`。
- D3：Stage0 grid 已 PASS；full cache 和 36 格训练按 `BENCHMARK_STATE.json` 实时状态。
- D4/D5：`prepared_not_executed`，没有真实滚动 utility 或概率榜单。

## D1：状态邻域分歧与视觉消歧

训练 bank=`17757` rows / `2712` episodes；dev=`2322` rows / `603` episodes；K=`20`，支持阈值按锁定 train calibration 规则。2s 高/低分歧支持窗口=`796/199`，自然总体支持=`2085`。

D1 NN median 是合法的 state-neighborhood baseline，不是条件熵或不可约不确定性证明。它的结果在 `d1/D1_RESULTS.csv`，10,000 cluster bootstrap 在 `d1/D1_BOOTSTRAP_CLUSTER_STATS.csv`。V3 learned sparse 分层表在 `d1/D1_LEARNED_RESULTS.csv`，paired interaction 在 `d1/D1_LEARNED_INTERACTIONS.csv`；任何 high-minus-low gain 都只作 exploratory development interpretation，不当作 final generalization。

## D2：command-change onset 提前预警

事件定义是 source-native command-change onset：按 Hand/Arm 分组，train-only 阈值，连续 6 个 quiet steps arm、连续 3 个 high steps 触发，初始 unknown/ongoing/边界事件单列。registry 有 `588` 条 dev event 记录。固定 dense panel=`118` episodes、`3718` queries，暴露时间以真实 timestamp 计算。

Sparse/Dense 的 Persistence 和 command-slope baseline 不换算部署 prevalence；learned sparse 的 bundle 数见 `36`。dense learned 当前为 `DENSE_LEARNED_COMPLETE`，不得以 sparse 分数冒充 dense 完成。

## D3：时间×空间 factor matrix

D3 是从原始 frozen encoder 新提取的 `T'=16 × 2×2 spatial bins × D=768` cache，并非给旧 V3 temporal mean cache 换 evaluator。Stage0 bridge 的最大绝对差为约 `3.8e-6`；当前 full cache status=`PASS`，covered=`20079/20079`。

锁定矩阵：`TS00-D/TS10-D/TS01-D/TS11-D/TS11-S/STATE-CTRL × H30/H60 × seeds 20261001/02/03 = 36`。当前 formal training=`36/36`，prediction acceptance=`36/36`，analysis=`D3_ANALYSIS_COMPLETE`；各阶段独立计数。

## D4/D5 准备

D4 已交付固定 4s rolling protocol、absolute-time intersection、revision utility、stale/fresh/latency/missing-frame controls、coverage CSV、接口和 synthetic tests；状态 `prepared_not_executed`。

D5 已交付 `(N,K,H,28)` samples interface、Energy Score、FD distribution、coverage/risk-coverage/credible-horizon 规则、capability audit 和 synthetic tests；状态 `prepared_not_executed`。

## QA、资源与边界

- CPU tests：D4/D5 synthetic `8 passed`；D1/D2 ID/shape/finite 与 10,000 bootstrap artifacts 已写入。
- Stage0 grid：`shape=(16,2,16,4,768)`，finite/ID/bridge/hash 通过。
- project-wide GPU child ceiling=`2`，one child per physical GPU；所有 GPU 任务通过现有 watcher。
- 本根绝对路径：`server-only://project-artifacts/reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z`。
- 关键状态：`BENCHMARK_STATE.json`、`JOB_LEDGER.jsonl`、`RECONCILIATION_SNAPSHOT.json`。
- 未解问题：D3 full cache/36 training 若尚在运行，整体状态必须保持 `running`；任何 V3/D3 missing row 都不能被压成零分或伪装 complete。

## Evidence map

- Context: `context/CONTEXT_RESTORED.zh-CN.md` / `context/CONTEXT_CONFLICTS.md`
- Inventory: `inventory/SOURCE_REGISTRY.csv`, `MODEL_REGISTRY.csv`, `ARTIFACT_INVENTORY.csv`, `EVALUATION_ELIGIBILITY.csv`
- Figures: `figures/FIGURE_MANIFEST.json` and paired PNG/PDF files
- Return machine state: `RETURN_TO_CHATGPT.json`
