> Archived study summary for `CORE_B_TASK_SPECIALIZATION_WAYPOINT_V1_20261013T060007Z` (source snapshot `20261013`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# 本轮回传：五类任务专门化 / predicted-waypoint

**状态：`insufficient_task_support`，在任务锁定前按预设门停止。**

我读取了接受的 Core B primary-train 元数据，核对 H30 前缀帧与冻结特征缓存行的对应关系，并按训练演示数排序。有效 primary-train 支持为 **2,712 个 episode、17,757 个 H30 anchor、120 个任务**。每个任务至少需要 50 个有效训练 source cluster，但只有两个任务过门；五类比较无法构成。

| 排名 | Task ID | 原始任务名 | Train episodes / observed source clusters | H30 anchors |
|---:|---:|---|---:|---:|
| 1 | 119 | deformable/fold_towel | 56 / 56 | 362 |
| 2 | 94 | Basic/put_dumpling_into_plate_g1 | 56 / 56 | 345 |
| 3 | 90 | Basic/push_duck_g1 | 48 / 48 | 203 |
| 4 | 141 | HRI/hold_a_lunch_bag_and_hand_it_back_to_human | 30 / 30 | 232 |
| 5 | 230 | Tool_use/click_a_mouse | 30 / 30 | 138 |

因此 **36 次内部筛选、36 次正式重训和最多 6 次 waypoint 拟合均为 0 次**。没有模型预测、dev 评分、Persistence 对比、配对统计、区间或图表。fit/tune 划分和逐任务 dev 支持没有展开；没有降低 50 episode 门槛，也没有借用旧 dev 结果。V-JEPA encoder 训练/提取均为 0。最终 heldout payload 访问为 0。

GPU 预检在 `restricted-compute` 看到 4 张 RTX 3090：GPU 0/1/2 分别使用 21/16/20 MiB，GPU 3 使用 22,310 MiB。共享 watcher 的动态 GPU 池没有启动作业，未检查其他用户 PID，也没有租约或训练 GPU 时间。

**范围说明：**训练 cluster 使用接受的 `split_assignment_key` 和 source 路径；该 train 元数据未含 session/trial/独立 alias ID，所以采集独立性为 `unverified_episode_source_only`，计数可能因后续身份合并降低。`DATA_ALLOWLIST`、`ANCHOR_MANIFEST` 和输入 NPZ 含 train/dev 混合角色元数据；筛选 train 行时 dev 元数据短暂进入内存，但只用 train 行做 census/ranking，没有生成逐任务 dev support 表，也没有读取 dev target、预测或误差。接受的审计摘要还显示全局 dev 支持为 603 episodes、2,322 windows。预检时我打开了接受的 split summary，输出了按任务聚合的 train/validation/final-test 数量；这是聚合 metadata 暴露，已记入 `inventory/OUTCOME_EXPOSURE.json`。没有读取 heldout 逐条 ID、label、视频、target、预测或指标。缓存回执含 encoder/cache 哈希和帧规则，但没有历史 extractor 源码哈希；本轮补做了只读源码与行映射审计，精确的历史脚本到数组字节绑定仍未证实。旧项目根目录没有本轮写入。

证据在 `server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V1_20261013T060007Z`。主要文件：

- `inventory/TRAIN_TASK_CENSUS.csv`：247 个 source catalogue task 的 train-only census。
- `inventory/CLUSTER_IDENTITY_EVIDENCE.csv`：2,712 个 accepted train source identity 条目及身份限制。
- `protocol/TASK_SELECTION_LOCK.json`：固定门槛、排名与停止原因。
- `inventory/OUTCOME_EXPOSURE.json`：开发/heldout 暴露边界及本轮聚合 metadata 偏差。
- `qa/FINAL_ACCEPTANCE.json`、`coordination/JOB_LEDGER.json`：未训练、未预测、未取得 GPU lease。

恢复需要新的接受数据/缓存证明至少五个任务各自达到 50 个合法 H30 source clusters，并能通过后续 fit/tune 身份阈值。现有数据下重跑不会改变结果。人工图审为 pending；确认性研究、机器人执行和投稿结论均未进行。
