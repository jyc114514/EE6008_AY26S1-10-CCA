> Archived study summary for `VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z` (source snapshot `20261018`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# 回传摘要

- **状态：partial；审计工作完成：是；handoff acceptance：否。** Audit root：`server-only://project-artifacts/reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z`。
- 十二份safe model strict reload 12/12；V2注册文件538/538；原handoff169/169；V1 checksum81/81。
- GPU replay：Stage0 4/4，正式12/12；train12/12、dev12/12，共24份。全部通过预声明工程门，且24份新GPU预测与历史GPU预测逐位相同（max abs=0）。
- 评分：20,736个metric diff行；V1独立scorer QA PASS，144项比较最大差3.41060513164848e-13（门限1e-10），旧GPU面板重算最大差0，新GPU对历史GPU面板最大差0。
- 本轮使用current profile：物理GPU2、UUID `GPU-00465dc6-e96e-67f1-285a-84ed61565ac2`、RTX 3090、CVD=2、逻辑`cuda:0`；历史只存物理index、无UUID/name/architecture/cuDNN，因此原profile匹配**无法证实**。
- 首次全量验证失败原因是runner把dev行元数据写入train预测bundle；原失败证据和原bundle均已保留。修复只改三项元数据，预测数组变化数为0；重验证24/24通过后封存并评分。
- 历史CPU zero-tolerance FAIL保留：最大train预测差0.003096103668、dev0.002043843269。CPU最大指标差字段是`STATE-CTRL_H60_S20261003 dev action_14_28 STATE-CTRL.tv_ratio_window_p50`，abs delta 0.00011648237705230713。
- 新训练、optimizer updates、encoder extraction、unrestricted pickle、final/held-out访问、旧根写入均为0；scope为`frozen_feature_predictor_replay`，raw-video端到端未测试。

剩余owner动作：Team erratum链接；V-JEPA历史extractor SHA/64-row bridge；若要升级安全权重来源等级，补独立导出前tensor catalogue与exporter执行回执。详见`ERRATA_AND_REMAINING_GAPS.json`。
