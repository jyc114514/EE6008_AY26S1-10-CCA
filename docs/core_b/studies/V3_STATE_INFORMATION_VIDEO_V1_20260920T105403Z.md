> Archived study summary for `V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z` (source snapshot `20260920`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B v3：状态信息量 × 视频内容 54 格收菜报告

## 终态

```ini
STUDY_STATUS=complete
STAGE0_ACCEPTED=18/18
TRAINING_ACCEPTED=54/54
PREDICTIONS_ACCEPTED=54/54
FINAL_TEST_ACCESS=0
ENCODER_UPDATED=false
NEW_ENCODER_EXTRACTION_STARTED=false
```

54 格均使用固定 40 epochs、11,120 optimizer steps、epoch-40 last checkpoint；没有按首个 seed 或 dev 结果筛掉配置。所有 formal/inference watcher 均返回 0。

## V3 基线复核

Persistence 是从冻结的 primary-dev targets/anchor 独立复算的，episode-equal 聚合；与旧 V3 保存表一致。

| Horizon | Arm standardized MAE | Hand standardized MAE | Full standardized MAE | Full native FD |
|---:|---:|---:|---:|---:|
| 30 | 0.265956 | 0.180314 | 0.223135 | 0.01329258 |
| 60 | 0.363856 | 0.293306 | 0.328581 | 0.01358671 |

primary dev 为 603 episodes/2322 windows；supplementary 为 64 episodes/64 windows。二者交集 53，supplementary-only 11，union 614，因此不能写成 667 个独立 episode。

## 54 格主结果

表中 skill 是相对于同 horizon Persistence 的 standardized Full-MAE 描述性 skill；结果仍是 development-only。

| Horizon | Cell | Arm MAE | Hand MAE | Full MAE | Full native FD | Full skill |
|---:|---|---:|---:|---:|---:|---:|
| 30 | FULL-N | 0.248734 | 0.141331 | 0.195032 | 0.01329285 | 0.1259 |
| 30 | FULL-D | 0.248324 | 0.140478 | 0.194401 | 0.01330424 | 0.1288 |
| 30 | FULL-S | 0.248246 | 0.141325 | 0.194785 | 0.01329021 | 0.1271 |
| 30 | OBS-N | 0.248588 | 0.144868 | 0.196728 | 0.01329956 | 0.1183 |
| 30 | OBS-D | 0.251925 | 0.148668 | 0.200296 | 0.01329877 | 0.1024 |
| 30 | OBS-S | 0.248867 | 0.145184 | 0.197025 | 0.01330879 | 0.1170 |
| 30 | ANCHOR-N | 0.258492 | 0.170185 | 0.214339 | 0.01330072 | 0.0394 |
| 30 | ANCHOR-D | 0.257673 | 0.162615 | 0.210144 | 0.01329513 | 0.0582 |
| 30 | ANCHOR-S | 0.262621 | 0.173460 | 0.218041 | 0.01329256 | 0.0228 |
| 60 | FULL-N | 0.305644 | 0.231833 | 0.268738 | 0.01380461 | 0.1821 |
| 60 | FULL-D | 0.306446 | 0.228875 | 0.267661 | 0.01379453 | 0.1854 |
| 60 | FULL-S | 0.305956 | 0.229510 | 0.267733 | 0.01380548 | 0.1852 |
| 60 | OBS-N | 0.310403 | 0.234700 | 0.272552 | 0.01378548 | 0.1705 |
| 60 | OBS-D | 0.308921 | 0.236946 | 0.272933 | 0.01380981 | 0.1694 |
| 60 | OBS-S | 0.308223 | 0.234527 | 0.271375 | 0.01380294 | 0.1741 |
| 60 | ANCHOR-N | 0.328638 | 0.273665 | 0.301151 | 0.01377617 | 0.0835 |
| 60 | ANCHOR-D | 0.317268 | 0.250488 | 0.283878 | 0.01374142 | 0.1360 |
| 60 | ANCHOR-S | 0.327140 | 0.267173 | 0.297156 | 0.01372153 | 0.0956 |

## 视频增量与信息交互

24 项 primary MAE family 中 6 项 Holm 校正后小于 0.05；24 项 FD secondary family 中 5 项通过同一预声明 secondary Holm；8 项 interaction family 中 4 项 Holm 校正后小于 0.05。完整数值见 CSV。

在完整 FULL 条件下，视频净增量总体较小且未形成普遍显著的 D>N/D>S 主结论；ANCHOR 条件下 2 秒 D 相对 N/S 的 Arm、Hand 视频增量显著，说明视觉信号在状态信息被人为减少时更容易显现。该结果不能包装为完整状态任务已被视频改善。

interaction 中，ANCHOR−OBS 的视频 gain 在 1 秒 Arm/Hand 和 2 秒 Arm/Hand 均为正，其中 Holm 后支持较强；OBS−FULL 没有形成同样一致的证据。这个结果支持“视频价值依赖可用 state 信息”的限定表述，不证明 command history 的物理冗余。

## seed、工程和资源

Fresh seeds：20260953、20260954、20260955。54/54 checkpoint strict reload、zero-residual Persistence、prediction shape/ID/finite 检查通过；旧 V3 的 outlier audit 仍为 training variation supported、无 confirmed bug。
实际 GPU-hours：2.879588；物理 GPU：1, 2, 3；Stage0 实际 GPU：2/3；最多两个本项目 child，未 signal/kill 其他用户进程。

## 解释边界

- ANCHOR 仍保留独立 last-command anchor token，不是 strict video-only。
- 删除 state 后出现的视频收益只说明 reduced-information condition 下的条件增量。
- FULL/OBS/ANCHOR 的差异也可能包含优化、正则化和输入融合稳定性，不能直接等价为物理信息冗余。
- 所有结果都来自 development split；不替换旧冻结 candidate，不访问 final/held-out，不联系教授。

完整证据：`analysis/RESULTS_BY_SEED.csv`、`analysis/AGGREGATE_RESULTS.csv`、`analysis/PRIMARY_VIDEO_EFFECTS.csv`、`analysis/FD_RESULTS.csv`、`analysis/INFORMATION_INTERACTIONS.csv`、`analysis/SUPPLEMENTARY_ID_UNION.json`。
