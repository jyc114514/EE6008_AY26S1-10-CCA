> Archived study summary for `CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z` (source snapshot `20261007`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# P1/P2 续跑结果与交接

## 执行状态

锁定的 `B_CORE` H30 主任务与 P1/P2 分析已完成。三个 `ALT-POOL-R3D18` H30 seed、三个 primary prediction、三个 D4 dense prediction 均通过接受。所有工作使用物理 GPU2；GPU3 未使用。`FINAL_TEST_ACCESS=0`。

锁定的 H60 门结果为 `NOT_PASS`，因此没有启动 H60 训练。`B_REP`、新 B6 训练继续关闭；`LEGACY_ACTUAL_RECIPE` 只作匹配描述性对照，未更改 loss 或 scheduler。

## H30 主结果

下表为 episode-equal Full28 MAE，三个 seed 的样本标准差；Persistence 为单一基线。

| 模型 | Full28 MAE | seed SD |
|---|---:|---:|
| Persistence | 0.223135 | — |
| S-HIST | 0.194640 | 0.001089 |
| F-POOL-D | 0.194216 | 0.000699 |
| ALT-POOL-R3D18 | 0.194952 | 0.000671 |

正的 `ALT − control` 表示 R3D 的 MAE 更高。ALT−S-HIST 为 `+0.000313`，source-group bootstrap 95% CI `[-0.000256, +0.000863]`，区间跨零。ALT−F-POOL-D 为 `+0.000736`，CI `[+0.000093, +0.001383]`；点估计虽为正，但绝对效应低于锁定显示阈值 `0.002231`。这组结果没有显示 R3D pooled 路径优于两个 H30 对照。

## P2 效应与支持

- `D1_A_H30` 高/低支持分别有 421/143 个 source groups、89/30 个 tasks；`M_H30` large/stable 分别有 449/482 个 groups、109/116 个 tasks。样本支持门通过。
- ALT 的高 A 高低差（gain）按三个 seed 为 `-0.001350, +0.001200, -0.000386`；三个 seed 的 interaction CI 均跨零。
- ALT 的 large-M 与 stable-M gain 差按 seed 为 `-0.001792, +0.001211, +0.002672`；方向不一致，前两个 CI 跨零，第三个下界仅略高于零。不能据此声称稳定的互补增益。
- H60 增益门在 high-A 与 large-M 支持上都无 ALT 候选通过。ALT pooled gain 分别为 `-0.000093`（95% CI `[-0.001477, +0.001302]`）和 `-0.000185`（CI `[-0.001776, +0.001442]`）；seed 方向也不全为正。两项均低于 `0.002231` 的显示阈值。
- D4 same-future-time 包含 3,600 个 pairs、118 个 source groups/tasks。ALT `DeltaU` 相对 S-HIST 的三个 seed 值均为负：`-0.000033, -0.000395, -0.000381`；三组 95% CI 下界均未高于零。`L_new` 差值在显示阈值内，但方向/CI 条件未通过。

因此 H60 的两个锁定数据分支均未通过，H60 训练保持关闭。

## D2 事件 onset 次分析

锁定 V3 规则在同一 118-group、3,718-query development panel 上完成，状态 `PASS_EXPLORATORY_SECONDARY`。ALT Hand recall 按 seed 为 `2/238`、`1/238`、`0/238`（0–0.84%），无 false alarms。ALT Arm recall 为 `15/174`、`18/174`、`17/174`（8.62–10.34%）；false-alarm rate 为每小时约 409、425、485。此模块是探索性次分析，不升级成主终点结论。

## 可视化 QA

14 张锁定 case 图全部通过 contact-sheet 检查；案例 01、04、14 另做全尺寸检查。最终图像为 2880×2080 RGBA、无空白或透明像素问题。修正版 figure-data 仍为 55,482 行，SHA-256 与首次渲染相同：`6bef595b…1260c3a9f`。42 条所选帧均为 anchor 及其之前的合法历史帧，没有 post-anchor 帧。模型线型和 channel 色彩图例移入空白 subplot cell，检查的轨迹无图例遮挡。

总览图：[CONTACT_SHEET_CASES_01_14.png](server-only://CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/CONTACT_SHEET_CASES_01_14.png)；完整 QA 记录：[VISUAL_QA.json](server-only://CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/VISUAL_QA.json)。原始量纲仍未解析，图中标注为 development-only 的 standardized absolute error Full28。

## 资源、封存与边界

- 总 GPU 用量估计 `2.2584 / 16` GPU-hours，剩余约 `13.7416` GPU-hours。encoder Stage0 实测 reserved 峰值 1,988 MiB；完整 feature bank 用时 2.0854 GPU-hours。
- 实验根实际占用约 `0.27 / 20 GiB`。最终资源快照（2026-10-08 03:37:41 UTC）：GPU2 20 MiB used、24,108 MiB free、0% util；没有 GPU2 compute process。GPU3 有外部进程且仅 1,817 MiB free，未被触碰。GPU2/GPU3 advisory leases 与两个 project-child slots 均为 free，P1/P2 tmux 已结束。
- 旧根 pre-existing checksum manifests 仍记录为 PASS。本次续跑所有写入都落在本 EXP_ROOT；未重新对全部旧根做全量哈希审计。
- 根内最终 SHA-256 清单：[CHECKSUMS.sha256](server-only://CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/CHECKSUMS.sha256)，覆盖本 EXP_ROOT 的全部普通文件（清单自身除外），生成后已验证。

## 复现记录与边界

所有正式训练、推理、特征缓存、分析、日志、失败 attempt 与 source-fix backup 均保留在本 EXP_ROOT。包括 extractor 的 Sequential Conv3d 属性修正、inference sigma tensor device 修正、D2 锁定路径/阈值注入修正、D4 控制流缩进修正和纯布局图例修正；没有改变科学矩阵。首次推理失败、门脚本 system-Python 缺 NumPy，以及本次 CPU QA 工具的小失败都保留了记录并使用既有项目环境/受支持路径重试，未安装依赖。

结果只支持当前锁定的 development 分析。没有访问 final-test target arrays；没有开启 `B_REP`、新 B6 或 H60 training。
