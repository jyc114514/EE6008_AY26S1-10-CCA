# EE6008 AY26S1-10-CCA

[English](README.md) | 简体中文

**面向 Humanoid Everyday Action Anticipation 的联合嵌入预测建模**

这是 NTU EE6008 AY26S1-10-CCA 课程项目的公开研究仓库，包含 Core A 与 Core B 的精选代码和证据，并分别记录实验状态、数据暴露、来源和服务器保留材料。

## 项目目标

项目拟研究 joint-embedding video representation 在 humanoid everyday action understanding 与 action anticipation 中的作用。当前暂定流程如下：

V-JEPA / V-JEPA 2 baseline<br>
→ 选择并审核公开数据的 metadata 与小规模数据子集<br>
→ feature extraction 或 frozen representation<br>
→ current-action recognition baseline<br>
→ action anticipation / 其他下游任务<br>
→ 轻量模型或训练策略修改<br>
→ ablation 与失败案例分析

Core A 包含历史 development/validation task-recognition 屏幕结果，以及后续 V2 同源 final-label 修正分析。V2 标签访问已披露，不是未触碰标签的盲测确认。Core B prediction/forecasting 证据单独记录在 [docs/core_b/](docs/core_b/)。

## 当前状态

- 历史 Core A Full-120 Stage1：源证据中 9/9 通过验收，仅 development/validation；D-P-L 使用 unmerged derived inference。
- Core A V2：在披露映射修正与先前标签访问后，对 602 个同源 final labels 完成六格重评分；没有新训练，不是独立盲测确认。
- Core B：B1/B2/B3、V3、D3、query/ablation 和 V2 CURRENT_PROFILE dev/final 结果分别编目。
- Core B V2 final targets 在 22 个预测包封存后读取；41 个 final episode 与 Core A 视频索引重叠，session 独立性未知。
- 结题证据仍标记团队/学术审阅、视频录制和课程提交待完成。
- 许可：本证据导出不新增许可证声明；第三方源码、数据和模型许可仍需分别核对。

## GitHub 与 GPU Server 的分工

GitHub 中保存：

- 源代码、配置文件和小型工具；
- 数据 metadata、manifest、label map 和 split 定义；
- 项目范围、会议记录、周报和学习笔记；
- 小型、经过整理并带有 provenance 的最终结果。

GPU Server 中保存：

- raw videos、frames 和其他原始数据；
- V-JEPA checkpoint；
- extracted feature tensors；
- training outputs、logs、cache 和临时文件。

`data_metadata/` 不是 `data_raw/`，`results/` 不是 `outputs/`。不要把原始数据、模型权重或大规模训练产物提交到 GitHub。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `configs/` | 可移植的实验配置骨架，使用环境变量表示机器路径。 |
| `docs/` | 项目范围、协作规范、服务器布局、会议/周报模板和整理后的公开材料。 |
| `data_metadata/` | 小型 manifest、label map 和 split 定义。 |
| `src/` | project-authored Core A package 与原有骨架模块。 |
| `scripts/` | 可移植 Core A runner、审计和图表 QA 入口。 |
| `experiments/` | Core A catalog、run-level aggregate、effects 和 acceptance summary。 |
| `results/` | 小型 Core A 汇总、plot data 和图，不存放原始训练输出。 |
| `tests/` | Core A 单元和 contract tests。 |

## 协作流程

`main` 作为相对稳定分支。分支名称按工作内容命名，例如：

- `feature/vjepa2-loader`
- `experiment/g1-recognition-baseline`
- `fix/dataset-path`
- `docs/weekly-update`

不要为每个人创建永久个人分支。

```bash
git pull origin main
git switch -c feature/<short-description>
git add -- <confirmed-paths>
git commit -m "<type>: <short description>"
git push -u origin feature/<short-description>
```

准备好后通过 pull request 合并。小型文档修正是否直接合并到 `main`，由团队协商决定。任何提交都必须避免 raw data、credential、checkpoint 和未经审核的大文件。

## 实验可复现性

每次真实实验至少记录：

- dataset revision 与 split；
- config 路径；
- Git commit hash；
- run ID、seed、model/checkpoint；
- server 与 output path；
- owner、日期、status；
- primary metric、secondary metric 和备注。

记录位置为 `experiments/runs.csv`。run registry 分开记录 development、validation 和 final/heldout scoring；历史访问状态只适用于对应 run。

## 项目材料与版权边界

原始本地资料目录继续保留在本机，没有被移动或删除。公开仓库只保留经过筛选的项目说明、学习卡片和公开来源链接。版权许可不明确的论文 PDF、登录态文件、私有表单、原始数据、checkpoint 和大规模审计产物不上传到 Public repository。

论文、数据集和模型权重的许可是相互独立的。引用链接不等于获得再分发许可；实际使用前应回到官方 source、revision 和 license 核对。

## 项目成员

这是一个五人课程项目，当前仓库 owner 为 `jyc114514`。其他成员的 GitHub collaborator 邀请会在确认准确 username 后逐一处理，不猜测用户名。

## 本次服务器同步发布

上次发布以来的代码与结果索引见[发布摘要](docs/releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md)、[实验结果索引](docs/releases/server-sync-20261022T082418Z/EXPERIMENT_RESULT_INDEX.csv)和[Core B 研究索引](docs/core_b/EXPERIMENT_LINE_INDEX.md)。发布材料区分历史 development-only 访问记录与后来已披露的同源 final-label 分析。

## 发布后修补补充（2026-10-24）

本补充接续 PR head 348469cb8c3d5ba468b32740e8c23684b9f2dcc4 的独立验收。此前 18 个排除候选中，15 项现有明确标记且绑定源哈希的修复副本；原历史文件保持不变。另有 3 个主机管理/行政包装器继续排除。逐文件来源和导出哈希见 docs/releases/server-sync-20261022T082418Z/REPAIR_SOURCE_MAP_20261024.csv。 ENTRYPOINT_RESULT_MAP.csv 中非公开服务器配置/输入引用以稳定 artifact_id 占位符表示；输入类型和访问条件仍可见，公开仓库相对路径保持原样。

本补充公开了已有的 120 个聚合工件：91 个结果表/JSON 和 29 个绘图数据文件；没有重算指标。来源/导出哈希、大小、路径脱敏说明与 CSV 行数见 RESULT_EXPORT_PROVENANCE_20261024.csv。RELEASE_REPAIR_MANIFEST_20261024.csv 与 RELEASE_REPAIR_CHECKSUMS_20261024.sha256 仅覆盖列明的本轮修补载荷；原发布 manifest/checksum 仍仅适用于原快照。

V7 single-head scorer 与 V7 ensemble scorer 分开；V9 context-dropout、V9 cached-Q0 和 V9 bootstrap correction supplement 也分别编目。检查范围内没有 V10 运行目录或已保存聚合结果。

请使用上方 Core A 与 Core B CPU 命令执行公开合约检查。修复后的手工校准分析、R3D 特征提取和 temporal-variation replay 仍需要对应服务器工件。R3D 必须显式提供 --project-root、--study-root、--primary-inputs、--dense-inputs、--raw-root 和 --weights；源码不内置服务器数据根路径，输出元数据使用不透明 artifact_id。完整提取仍需授权后执行 GPU；脚本不下载权重。历史脚本不应统一视为开箱可运行工具。

Core A 历史 Stage1 源码绑定为部分可证：V-JEPA2 源码 revision 已核实，自定义 runner 文件身份仍为 UNKNOWN。既有披露继续有效：Core A prior label exposure/amended analysis；Core B same-source overlap 与 session independence UNKNOWN；Q0 bridge FAIL_UNRESOLVED；standardized metrics 不等于 physical robot accuracy。

## 独立验收补充（2026-10-26）

已从 GitHub 独立下载并核对 PR head `d3c2dd144d498b9f5aa656f8a61f371fc28974de`；本次结果与未解决边界见[验收报告](docs/releases/server-sync-20261022T082418Z/POST_RELEASE_ACCEPTANCE_20261026.md)。版本化的[验收 manifest](docs/releases/server-sync-20261022T082418Z/POST_RELEASE_ACCEPTANCE_MANIFEST_20261026.csv)和[校验和](docs/releases/server-sync-20261022T082418Z/POST_RELEASE_ACCEPTANCE_CHECKSUMS_20261026.sha256)覆盖原 149 个修补载荷文件及该验收报告。两个 sidecar 不对自身计算哈希。2026-10-24 的修补 manifest/checksum 仅适用于后续修改前的历史快照。

公开树中的历史 B1 runner 不是自包含工具：`checkpoint_utils.py`、`fresh_process_batch8_d_a_l_fixed.py` 与 `src.models.attentive_pooler` 模块闭包不可用。B1 表格保留为历史 development 证据，公开包不宣称能够独立重跑 B1。
