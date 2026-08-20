# EE6008 AY26S1-10-CCA

[English](README.md) | 简体中文

**面向 Humanoid Everyday Action Anticipation 的联合嵌入预测建模**

这是 NTU EE6008 AY26S1-10-CCA 五人课程项目的公开研究仓库。当前版本是一个可复现的项目骨架；V-JEPA/V-JEPA 2 的具体适配方式、下游任务和第一版数据划分仍需团队与导师确认。

## 项目目标

项目拟研究 joint-embedding video representation 在 humanoid everyday action understanding 与 action anticipation 中的作用。当前暂定流程如下：

V-JEPA / V-JEPA 2 baseline  
→ 选择并审核公开数据的 metadata 与小规模数据子集  
→ feature extraction 或 frozen representation  
→ current-action recognition baseline  
→ action anticipation / 其他下游任务  
→ 轻量模型或训练策略修改  
→ ablation 与失败案例分析

首次初始化不代表已经完成 baseline，也不代表已经取得任何实验指标。

## 当前状态

- 项目仓库骨架：已初始化。
- 数据集、robot 子集和最终任务：暂定，等待团队确认。
- baseline 实现：尚未声称完成。
- 实验结果：当前没有已记录的结果。

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
| `src/` | 后续加入 dataset、model、feature、training、evaluation 和 utility 代码。 |
| `scripts/` | 在服务器环境和官方命令确认前，先记录 pipeline 说明。 |
| `experiments/` | 实验命名规则和版本控制的实验记录表。 |
| `results/` | 小型最终汇总结果与图片，不存放原始训练输出。 |
| `tests/` | 后续加入单元测试和 smoke tests。 |

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

记录位置为 `experiments/runs.csv`。初始化仓库不填入虚构实验。

## 项目材料与版权边界

原始本地资料目录继续保留在本机，没有被移动或删除。公开仓库只保留经过筛选的项目说明、学习卡片和公开来源链接。版权许可不明确的论文 PDF、登录态文件、私有表单、原始数据、checkpoint 和大规模审计产物不上传到 Public repository。

论文、数据集和模型权重的许可是相互独立的。引用链接不等于获得再分发许可；实际使用前应回到官方 source、revision 和 license 核对。

## 项目成员

这是一个五人课程项目，当前仓库 owner 为 `jyc114514`。其他成员的 GitHub collaborator 邀请会在确认准确 username 后逐一处理，不猜测用户名。
