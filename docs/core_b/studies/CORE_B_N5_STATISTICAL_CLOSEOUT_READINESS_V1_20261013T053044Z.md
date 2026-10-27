> Archived study summary for `CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z` (source snapshot `20261013`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# N5 统计收口与独立评价就绪检查

状态：**完成统计收口；独立评价数据尚未就绪**。输出根：`server-only://project-artifacts/reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z`。

N2 主区间已核实为固定三个 seed 的 source-group bootstrap；N4 已报告区间已核实为 crossed paired seed×group。原 N2/N4 的 30 个 co-primary/contrast 端点全部从接受的组级损失和原 draw indices 重放通过，最大端点差为 2.29e-16。N5 新建 30 行 group-only、精确 seed-only 和 crossed 区间；全体 seed、sample SD、median、三个 LOO 与配对代数 QA 均通过。

N5 G_TS 点估计为 position MAE +0.001121、safe-FD +0.005342。固定-seed group-only 两个区间为正；seed-only/crossed 区间覆盖零，反映结果对 seed 选择的敏感性。N4 的 S20261002 机制中 level 与 time-centered error 都改善，centered 项略大；Q1 动态能量和对齐仍有限。完整逐项结果见 `results/CO_PRIMARY_SCOPE_TABLE30.csv`、`results/BY_SEED_AND_LOO.csv` 与 `results/HISTORICAL_INTERVAL_REPRODUCTION.csv`。

N2/N3/N4 老根保持只读。HORIZON 的结果回执完整且 owner 字段陈旧；OBS43 主根封存完成、独立确认未执行；第二个 OBS43 根为空 scaffold、状态 UNKNOWN。N5 未重启或修改这些窗口。

限定 metadata 中没有独立候选 population，当前 readiness=`NOT_READY`；未来 12 个固定模型加 Persistence 共 13 个预测槽位已列出为 `prepared_not_executed`。完整协议、字段模板和数据缺口见 `readiness/`。

本轮原始 target array、新候选数值/评分、模型 forward、训练、GPU 与 final/heldout 访问均为0。读取了14,472条既有 GT-derived dev group-loss 值。边界记录说明：一个既有 OBS43 return JSON 含历史 dev scores，该文件在状态核验时被打开；数值未用于 N5 结果。

人工图像审阅 pending，physical units 未核验，`submission_ready=false`。Pre-seal QA 24/24 通过；`CHECKSUMS.sha256` 已最后生成并逐文件只读复核。
