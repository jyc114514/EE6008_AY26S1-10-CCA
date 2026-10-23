> Archived study summary for `D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z` (source snapshot `20261006`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# D3-EQ corrected-scheduler 交接

**状态：开发集任务完成。** 唯一运行根：`reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z`。

已完成 Stage0 4/4、corrected-scheduler 正式矩阵 12/12、train 预测 12/12、dev 预测 12/12、独立评分/主分析 PASS；另完成固定 D4 dense rolling 次级分析：精确 matched old-as-run 12 格 + corrected 12 格，共 24/24 dense predictions，每格 3,718 queries，评分支持 H30/H60 各 3,600 pairs、118 source groups。所有推理均 target-blind；targets 只在 prediction acceptance 后由独立 scorer 读取。final/held-out=0，D5 training=false。

核心结论：scheduler 更新时机修正后没有出现普遍收益。Dev Full28 sMAE gain（old−corrected）在 TS11-D/H30 为 −0.003134（95% group-bootstrap CI [−0.003831,−0.002435]），STATE-CTRL/H30 为 +0.001662（[+0.001073,+0.002247]）；H60 两配置均不确定。H30 视频特异 sMAE interaction 为 −0.004796，Holm p=0.000400，方向偏向 STATE-CTRL。Train/dev TV ratio 多数仍远低于 1；TS11-D/H30/seed 20261002 在 train 与 dev 都近似静止。Dense H30 Full28 中 TS11-D 的 `L_new` scheduler effect 为 +0.001965（[+0.000913,+0.002917]），即新锚点绝对误差变差；滚动 `U` 变化接近零。不能据此认定 scheduler 是唯一根因或视频获益。

训练总 lease 为 1.849114 GPU-hours（cap 12），全部任务 GPU lease 已释放；本任务自身的 GPU queue、CPU inference、CPU evaluation tmux sessions 均已结束。8 次早期 Stage0 UUID guard 失败保留在记录中，发生于 trainer/optimizer 前；修复后 Stage0 通过。旧 EQ/DIAG roots 写入 0。Dense 分析第一次 acceptance-path pre-gate 失败，发生在加载 scoring targets 前；修复与 lock amendment 已记录，24 个预测数组 hash 未变。

另有 3 个早期失败日志以 `.json` 命名但尾随字面 `\n`（hex `5c6e`）；原件保留、非验收 manifest，已登记于 [FAILURE_LOG_FORMAT_AUDIT.json](server-only://D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/FAILURE_LOG_FORMAT_AUDIT.json)。

主结果报告：[02_SCIENTIFIC_RESULTS.zh-CN.md](server-only://D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/02_SCIENTIFIC_RESULTS.zh-CN.md)。最终验收和全文件 checksums 位于本目录 `FINAL_ACCEPTANCE.json`、`CHECKSUMS.sha256`。4 张图已生成，预览尺寸检查通过；完整尺寸视觉审阅与人工签字仍 pending，因此 `submission_ready=false`。
