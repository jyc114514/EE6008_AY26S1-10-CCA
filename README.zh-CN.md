# EE6008 AY26S1-10-CCA

[English](README.md) | 简体中文

**面向 Humanoid Everyday Action Anticipation 的联合嵌入预测建模**

这是 NTU EE6008 课程项目的公开仓库，收录两条相关研究线的代码、精选实验依据和复现说明：Core A 视频任务识别，以及 Core B 状态/动作轨迹预测研究。

> **快照范围：** 本说明依据 `main` 分支在 2026-10-27 的公开版本 `3343154b65dc98a3758d131c059e5806d2bc58ab`，以及下方链接的定期发布记录和 B1 修补记录。它不代表该快照之后尚未公开的项目进展。

## 研究方向

### Core A：部分观测下的任务识别

Core A 研究如何根据 G1 视频中四秒的观测前缀识别 task ID。公开代码包含 V-JEPA 2.1 特征提取流程、任务识别基线、probe 与模型适配工具、评估代码和整理后的实验依据。

历史 Full-120 Stage1 的 9/9 工件/运行验收通过。该批次使用一个 seed 和 development/validation 数据；9/9 并不表示所有科学假设都成立。D-P-L 结果标记为 `accepted_derived_unmerged`：合并导出路径失败后，依据未合并的 base-plus-adapter 预测重新计算。

Core A V2 在事先披露标签访问与类别映射修正后，对六组已有预测在 602 个同源 final labels 上重新评分。它没有新增训练或模型 forward pass；这是修正后的同源分析，不是未触碰标签的盲测确认。结果在不同指标上的方向并不一致，不能据此证明 action anticipation、独立的运动理解或机器人部署表现。

可从[当前结论](docs/core_a/CURRENT_CONCLUSIONS.md)、[V2 修正分析](docs/core_a/FINAL_V2_AMENDED_ANALYSIS.zh-CN.md)和 [Core A 指南](docs/core_a/README.zh-CN.md)开始阅读。

### Core B：状态/动作轨迹预测研究

Core B 收录针对不同模型、输入和时间预测设置的独立研究及聚合结果。最新编目的 CURRENT_PROFILE_V2 流程使用已训练的 checkpoint 对 development 与 final split 评分，没有新增训练。读取 final targets 之前，每个 split 的 22 个 prediction bundles 均已封存。

| Split | Episode / anchor 数 | V0 position MAE | Persistence position MAE |
| --- | ---: | ---: | ---: |
| Development | 47 / 185 | 0.158397 | 0.222262 |
| Final recovery | 41 / 162 | 0.178383 | 0.238057 |

这些是标准化指标，不代表机器人实际精度。41 个 final episodes 与 Core A 编目视频群体重叠，session 独立性未得到证明。原始 Q0 bridge 仍为 `FAIL_UNRESOLVED`，原因是 `UNKNOWN`。详见[当前结果与局限](docs/core_b/CURRENT_RESULTS_AND_LIMITATIONS.md)、[研究索引](docs/core_b/EXPERIMENT_LINE_INDEX.md)和 [Core B 指南](docs/core_b/README.zh-CN.md)。

## 仓库导航

| 路径 | 内容 |
| --- | --- |
| `src/ee6008/` | 当前 project-authored Core A package，包含模型加载、配置、基线、probe 和工具。 |
| `src/models/`、`src/training/`、`src/evaluation/` | 旧 scaffold 目录；其中 README 描述的是约定，并非当前实现。 |
| `scripts/core_a/`、`scripts/core_b/`、`scripts/final_evaluation/`、`scripts/audits/` | Core A/B runner、历史 study 入口、评估和审计工具。各入口依赖的数据与环境不同。 |
| `configs/`、`data_metadata/` | 公开配置摘要和 metadata，不含原始数据集。 |
| `docs/core_a/`、`docs/core_b/` | 研究范围、结论、局限和复现边界。 |
| `experiments/core_a/`、`experiments/core_b/` | 精选聚合表和实验索引。 |
| `results/core_a/`、`results/core_b/` | 选定图表和绘图输入，不含完整训练输出。 |
| `tests/` | 有针对性的单元与 contract tests。 |

## 环境与公开 CPU 检查

项目环境面向 Python 3.11–3.12。请查看 [environment.yml](environment.yml) 与 [requirements.txt](requirements.txt)；PyTorch 和视频依赖可能需要换成适配已授权运行环境的构建版本。公开仓库不包含原始视频、feature tensor、真实 checkpoint、逐样本预测/目标数组或执行日志。完整训练或重放需要相应的授权数据、权重、运行环境和服务器工件。

在仓库根目录使用 Bash 兼容的 shell，可执行已发布的 CPU contract checks：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts/core_a/final_recovery_v2:scripts/final_evaluation/v1_historical" python3 -B -m pytest -q -p no:cacheprovider scripts/core_a/final_recovery_v2/test_core_a_v2_scorer.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' EE6008_PROJECT_ROOT=. PYTHONPATH="scripts/core_b/final_recovery_v2:scripts/final_evaluation/v1_historical" python3 -B -m pytest -q -p no:cacheprovider scripts/core_b/final_recovery_v2/test_core_b_v2_contracts.py -k 'not anchor_frame_schedule_stops_at_anchor'
```

Core B 的 schedule 测试因读取仅存在于服务器的 development/final input manifests 而排除。这些检查覆盖公开 scorer 和 contract fixtures，不会执行训练、完整推理、视频特征提取或历史实验重放。

B1 公开依赖闭包另有合成 CPU/import 检查：

```bash
B1_DIR=scripts/core_b/legacy_archives/code/b1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$B1_DIR:$B1_DIR/public_support/analysis_scripts" \
  python3 -B -m pytest -q -p no:cacheprovider "$B1_DIR/public_support/tests/test_b1_public_dependency_closure.py"
```

它会检查公开导入、路径保护和合成 CPU contract，其中包括随机初始化 pooler 与合成 checkpoint payload。它不会重放历史 B1 结果，也不会运行 B1 训练。历史 helper 的源码身份仅得到部分证实，完整 B1 数值重放未执行。物理 GPU 3 是历史 B1 runner 对应配置的要求，并非所有项目流程都必须使用 GPU 3。详见 [B1 修补记录](docs/releases/server-sync-20261022T082418Z/B1_PUBLIC_DEPENDENCY_REPAIR_20261026.md)和 [entrypoint/result map](docs/releases/server-sync-20261022T082418Z/ENTRYPOINT_RESULT_MAP.csv)。

Core A 图表可依据[脚本说明](scripts/core_a/README.md)，从整理后的绘图输入重新生成。重新绘图不等于重复实验。

## 数据、来源与许可

原始视频与帧、checkpoint、feature cache、完整预测和目标值，以及运行日志都保留在获授权的研究环境中。公开 artifact ID 是引用标识，不是可下载的数据。请参阅定期发布的[服务器同步摘要](docs/releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md)和[实验/结果索引](docs/releases/server-sync-20261022T082418Z/EXPERIMENT_RESULT_INDEX.csv)，了解导出来源及历史研究边界。早期 development-only 访问记录仅适用于对应研究；后续 V2 final-label 访问情况另有披露。

根目录的 [SHA256SUMS](SHA256SUMS) 和 [Core A publication manifest](docs/core_a/publication/PUBLICATION_MANIFEST.csv) 保留 2026-09-10 Core A 发布快照的校验值。两份 README 的对应哈希与 [f43c07b 提交](https://github.com/jyc114514/EE6008_AY26S1-10-CCA/tree/f43c07b6055e993986ee1fd6a952261ec6634b71)上的文件一致；它们不能用于校验当前 `main` 全树。

本证据导出没有新增统一开源许可证。项目、第三方代码、数据集和模型权重可能各自适用不同条款；引用或来源链接并不自动授予再分发权。
