> Archived study summary for `CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z` (source snapshot `20261007`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B 后续诊断交付回传

状态：**已完成（development-only 训练记忆诊断）**。没有启动全数据 benchmark 候选、没有做 dev 选参，D5 未训练，final/held-out 未访问。

## 关键结论

- Phase A：16 个 train 样本（low/mid/high=5/5/6）。9 个 checkpoint 完成 CPU 前向；6 个有相应 accepted train endpoint 的逐 ID 输出比较最大差为 0，并且重复 replay 逐位一致。异常 corrected TS11-D/H30/seed20261002 不是 anchor-copy，而是样本间 offset 明显、未来时间变化几乎为零；epoch10 已近静态，具体发生时刻早于 epoch10 的部分未知。
- Query：没有精确重复 query，但相邻 cosine 约 0.999965，FP32 尺度有效秩 H30=4/30、H60=5/60。这支持 conditioning 风险，不等于证明模型无法表达动态输出。
- Head tiny-fit：8/8 格均完成 2000 updates。P1 只有 STATE-CTRL/FULL_LOSS 满足严格的逐样本 strong-fit；P8 没有 head cell 达到全 8 样本强拟合。STATE-CTRL/FULL_LOSS 在 P8 有 6/8 单样本通过，但不能用均值掩盖其余样本。
- Free-path：P1 未触发；P8 四个 head/loss 格全失败后触发，按协议运行仅 2 个独立输出参数控制。FULL_LOSS、POS_ONLY 均在 8/8 轨迹达标。它们不是预测模型；这表明相同目标和优化器可以优化独立输出，不唯一证明 head 的表达能力或某个 loss 是根因。
- 资源：全部在 CPU 单线程完成；head+free-path 正式优化约 7.3 分钟，保守预算估计 73.7 分钟，低于 120 分钟上限。GPU 未查询、未使用，当前 GPU 状态未知。

更正：早期进度文字把 accepted train-bundle 比较数说成 8；审计后的正确数是 6 个 endpoint。另有 3 个 epoch10/20/30 checkpoint 没有对应的 accepted final bundle。更正及原始失败记录保存在 `logs/`。

## 交付文件

- [Phase A 科学报告](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/A_FORENSIC_REPORT.md)
- [Phase B 结果报告](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/B_FINAL_REPORT.zh-CN.md)
- [结构化最终结果](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/B_FINAL_RESULTS.json)
- [最终 QA](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/FINAL_QA.json) 与 [图像 QA](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/FIGURE_QA.json)
- [逐样本指标](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/TINYFIT_FINAL_SAMPLE_METRICS.csv) 和 [训练曲线](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/TINYFIT_LEARNING_CURVES.csv)
- [校验和](server-only://CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/CHECKSUMS.sha256)

下一步只建议一个动作：只读审计 TS11-D 动态视频 token 到 decoder 的实际尺度/梯度路径，并冻结独立评价前的数据、目标和指标合同；不建议再加训练格或 seed/query/LR sweep。SHA-256 清单已核验通过（520 个文件，清单自排除）。Human review 仍 pending。
