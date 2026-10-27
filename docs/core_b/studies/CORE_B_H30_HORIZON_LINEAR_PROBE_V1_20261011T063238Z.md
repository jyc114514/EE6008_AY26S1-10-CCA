> Archived study summary for `CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z` (source snapshot `20261011`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B H30 CPU Ridge 与时距诊断回传

已执行完毕。完整报告见 FINAL_REPORT.zh-CN.md；新输出根为：
server-only://project-artifacts/reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z

## 直接结论

- Query 研究收口：Hadamard 的 TS11-D 平均 gain 依赖 seed 02，seed 01/03 无改善；STATE-CTRL position 三 seed 均负。没有建立跨 seed 的普遍 query 收益或融合绝对优势。
- 新 CPU residual Ridge：source-group 三折 CV 12 fit 后全 train refit 1 次；λ=1.0，是固定网格上界。H30 Ridge 未优于 Persistence 或 Q0 STATE-CTRL。
- H30 dev：Ridge prefix position MAE 0.230365、skill −3.24%；safe-FD error 1.044448、skill −6.96%；TV 比 0.283，direction active accuracy 0.548。Ridge 比 Q0 STATE-CTRL 三 seed 平均更差；两项 group-bootstrap contrast 的 95% CI 均全负。
- H=1/3/6/10/15/30 的 Ridge position 和内部 FD skill 表明短/长时距都未超过 Persistence。H=1 的内部 FD 为 NA。
- 旧 12 模型 H30 position/safe-FD 48/48 复现通过，最大差 3.40e-9；query gain 12/12 复现通过，最大差 4.63e-9。

dev 是已反复暴露的探索性 split，不是独立确认。没有 p 值、GPU、H60、D5、新神经网络 seed/query 或 held-out。

## 验收状态

- 12 source-group CV fits + 1 full-train refit；lambda grid 未扩展。
- Ridge train/dev predictions 接受：17,757 / 2,322 行，shape 分别为 (17757,30,28) / (2322,30,28)。
- H30 时距：实际 raw timestamp 元数据只读 frame_index/timestamp，没有读取 action 列。
- 67/67 本轮登记来源 hash 仍匹配，旧 query checksum manifest 未变，旧根写入 0。
- GPU API/命令 0，final/heldout access 0。
- 最终图集 agent visual QA pass；human review pending；submission_ready=false。

重要文件：qa/FINAL_ACCEPTANCE.json、qa/FINAL_SOURCE_NONINTERFERENCE.json、results/HORIZON_SCORE_SUMMARY.json、figures/FIGURE_MANIFEST_FINAL.json、CHECKSUMS.sha256。
