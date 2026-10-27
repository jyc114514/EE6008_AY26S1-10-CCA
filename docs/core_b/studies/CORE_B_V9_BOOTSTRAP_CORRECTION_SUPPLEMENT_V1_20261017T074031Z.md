> Archived study summary for `CORE_B_V9_BOOTSTRAP_CORRECTION_SUPPLEMENT_V1_20261017T074031Z` (source snapshot `20261017`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V9 bootstrap-seed and checksum supplement

生成时间 UTC：2026-10-17T07:47:17.658540Z

## 范围

既有 V9 包已记录 12/12 个 fresh fits、36 份 neural prediction bundles、6 份固定混合输出和 5,550 次 dev loss evaluation。本补充不新增 fit、optimizer update、预测或 dev target 读取，只从封存的 per-window 分数 CSV 按本轮锁定 seed 2026101709 重算 paired source-group bootstrap 区间。

原 V9 包保持不变。其已封存区间使用 seed 20261016；本目录 results/PAIRED_GROUP_CI.csv 给出按指定 seed 2026101709 生成的 42 行修正版区间。同一组 5,000 次 within-task source-group 抽样用于所有 contrasts、metrics、seeds、policies 和 mixtures。

## 指定比较

gain = reference error − candidate error；正数表示 candidate 误差更低。区间条件于固定模型和重复使用的 dev 集，未计算 p 值，也未作多重比较校正。

| Contrast | Metric | Gain | 95% CI |
| --- | --- | ---: | ---: |
| NUMCTX_DROP_VIDEO | Position MAE | -0.002810 | [-0.005162, -0.000463] |
| NUMCTX_DROP_VIDEO | Safe-FD L1 | -0.007180 | [-0.013657, -0.001422] |
| NUMCTX_DROP_VIDEO | Seam Safe-FD L1 | 0.019559 | [0.000476, 0.041936] |
| NUMCTX_DROP_STATE | Position MAE | -0.004179 | [-0.006191, -0.001980] |
| NUMCTX_DROP_STATE | Safe-FD L1 | 0.001235 | [-0.003106, 0.005460] |
| NUMCTX_DROP_STATE | Seam Safe-FD L1 | 0.009054 | [-0.003940, 0.025137] |
| INTERACTION | Position MAE | 0.001368 | [-0.002144, 0.004677] |
| INTERACTION | Safe-FD L1 | -0.008415 | [-0.015414, -0.001370] |
| INTERACTION | Seam Safe-FD L1 | 0.010505 | [-0.008372, 0.034228] |
| MIX_GAIN | Position MAE | -0.000253 | [-0.001401, 0.000920] |
| MIX_GAIN | Safe-FD L1 | -0.000857 | [-0.004461, 0.002372] |
| MIX_GAIN | Seam Safe-FD L1 | 0.010148 | [0.000126, 0.021987] |
| VD_VS_MIX0 | Position MAE | -0.005677 | [-0.008016, -0.003425] |
| VD_VS_MIX0 | Safe-FD L1 | -0.007960 | [-0.013760, -0.002322] |
| VD_VS_MIX0 | Seam Safe-FD L1 | -0.010980 | [-0.024017, 0.001868] |
| VD_VS_S | Position MAE | -0.003593 | [-0.006478, -0.000645] |
| VD_VS_S | Safe-FD L1 | 0.003873 | [-0.002621, 0.010485] |
| VD_VS_S | Seam Safe-FD L1 | -0.009259 | [-0.026001, 0.006232] |

## 结果解读

完整数值上下文下，VD 没有优于 V0：Position MAE 和 Safe-FD 的 gain 为负，seam 改善。AD 的 Position MAE 也比 A0 更差。Safe-FD interaction 为负，不支持数值上下文 dropout 增强视频作用。MIXD 相对 MIX0 仅 seam 有正向 gain；Position MAE 与 Safe-FD 未改善。按指定 bootstrap seed 修正后，核心判断不变。

## 审计记录

- 原 V9 checksum manifest：758/758 项已验证，missing=0，mismatch=0；manifest SHA-256 为 c966007d5f737eb2222c000b51066fd9783542ad21a78c862c0ce3b5dc448f0c。
- 原 V9 根目录缺少 qa/FINAL_CHECKSUM_VERIFICATION.json；本补充的 qa/ORIGINAL_V9_PACKAGE_VERIFICATION.json 保存本次只读 readback 收据。
- V8.1 scorer-only cache 与 V9 dev archive 和 ordered IDs 元数据一致（compatible=True），但 V9 实际记录为 seal 后读取 archive 一次，没有复用 cache。该流程偏差保存在 qa/SCORER_CACHE_USE_AUDIT.json；本补充没有重新读取 target。
- final/heldout payload 与 task-level final aggregates 访问数为 0。
- 原 V9 figures/FIGURE_VISUAL_REVIEW.json 记录的图像审阅状态为 PASS；其 paired-effects 图使用原 seed。修正区间以本目录表格为准。

## 重现方式

使用项目环境 Python 运行 code/recompute_bootstrap_ci.py。该脚本仅读取 V9 per-window score CSV 和 receipt 元数据，不导入 PyTorch，也不打开 target payload。
