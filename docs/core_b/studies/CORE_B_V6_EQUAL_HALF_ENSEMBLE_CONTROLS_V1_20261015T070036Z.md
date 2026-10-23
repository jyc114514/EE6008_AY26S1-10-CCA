> Archived study summary for `CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z` (source snapshot `20261015`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# EE6008 V6 固定等权平均对照结果

- 状态：complete；输出根目录：server-only://project-artifacts/reports/benchmarks/CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z
- 研究类型：post-V5 development；主候选固定为 V5 三个同 seed MIX50 pair。
- 统计区间条件于 9 个固定模型和已暴露 development 窗口。bootstrap：47 个 task-nested source groups，5,000 draws，seed 2026101506。

## MIX50 对锁定参考的直接配对结果

gain = reference loss − MIX50 loss，正值支持 MIX50。四个非 Persistence 参考使用同一 source-group draw 计算 95% percentile CI；Persistence 按锁定计划报告点值。

| 指标 | 参考 | seed 131 | seed 132 | seed 133 | 三对均值 | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| position MAE | S | +0.001477 | +0.002151 | +0.002624 | +0.002084 | [+0.000668, +0.003434] |
| position MAE | V | +0.003857 | +0.002588 | +0.002155 | +0.002867 | [+0.001388, +0.004554] |
| position MAE | BIAS_STATE | +0.001216 | +0.001910 | +0.002363 | +0.001830 | [+0.000417, +0.003211] |
| position MAE | BIAS_VIDEO | +0.003596 | +0.002301 | +0.001888 | +0.002595 | [+0.001084, +0.004312] |
| position MAE | PERSISTENCE | +0.066906 | +0.065671 | +0.067795 | +0.066791 | 未计算 |
| safe-FD L1 | S | +0.010393 | +0.012659 | +0.012448 | +0.011833 | [+0.008675, +0.015295] |
| safe-FD L1 | V | +0.002222 | +0.000240 | -0.000120 | +0.000781 | [-0.002465, +0.003863] |
| safe-FD L1 | BIAS_STATE | +0.010393 | +0.012659 | +0.012448 | +0.011833 | [+0.008675, +0.015295] |
| safe-FD L1 | BIAS_VIDEO | +0.002222 | +0.000240 | -0.000120 | +0.000781 | [-0.002465, +0.003863] |
| safe-FD L1 | PERSISTENCE | +0.035456 | +0.032474 | +0.037726 | +0.035219 | 未计算 |
| seam safe-FD L1 | S | -0.007906 | +0.010700 | +0.002368 | +0.001720 | [-0.013245, +0.012330] |
| seam safe-FD L1 | V | +0.037760 | +0.019572 | +0.034284 | +0.030539 | [+0.017089, +0.047568] |
| seam safe-FD L1 | BIAS_STATE | +0.045217 | +0.063548 | +0.053264 | +0.054010 | [+0.037502, +0.067032] |
| seam safe-FD L1 | BIAS_VIDEO | +0.090110 | +0.072312 | +0.085741 | +0.082721 | [+0.067126, +0.101953] |
| seam safe-FD L1 | PERSISTENCE | +0.022008 | +0.046823 | +0.042752 | +0.037194 | 未计算 |

三对 MIX50 的绝对宏平均：position MAE 0.155471（seed sample SD 0.001066），safe-FD 0.818974（SD 0.002634），seam 0.765222（SD 0.013308）。

Position MAE 对 S、V、BIAS_STATE、BIAS_VIDEO 的四个 CI 下界均大于零，三对固定模型的宏平均 gain 也全为正。Safe-FD 对 S 与 BIAS_STATE 的区间支持正 gain；对 V 与 BIAS_VIDEO 的区间跨零，且 seed 202610133 的 gain 为 −0.000120。按 15 个 task×seed 单元计，position gain 为正的数量：S 12/15、V 13/15、BIAS_STATE 9/15、BIAS_VIDEO 9/15；safe-FD：12/15、9/15、12/15、9/15。Persistence 的三项平均 gain 为 position 0.066791、safe-FD 0.035219、seam 0.037194。

## 六类 family 均值

各 family 对固定 pair loss 等权平均；括号是有限 pair 的 sample SD，表示当前成员离散度，不表示独立 seed 不确定性。

| Family | Pair 数 | Position MAE | Safe-FD L1 | Seam safe-FD L1 |
|---|---:|---:|---:|---:|
| SS | 3 | 0.155838 (0.000471) | 0.825341 (0.001192) | 0.755558 (0.001490) |
| AA | 3 | 0.158702 (0.000430) | 0.825886 (0.001988) | 0.767337 (0.005664) |
| VV | 3 | 0.156758 (0.000799) | 0.815624 (0.001574) | 0.782452 (0.008106) |
| SA | 9 | 0.156954 (0.000681) | 0.825270 (0.001738) | 0.760800 (0.006774) |
| SV | 9 | 0.155476 (0.000779) | 0.818867 (0.001677) | 0.765530 (0.009646) |
| AV | 9 | 0.157270 (0.001011) | 0.819387 (0.001836) | 0.772207 (0.010990) |

SV 的 finite-family position MAE 均值最低；safe-FD 低于 SS、AA、SA、AV，但高于 VV；seam 高于 SS、AA、VV。

## 六个固定 family contrasts

每格为 point estimate [95% source-group CI]，正值支持 contrast 名称中的候选。

| Contrast | Position MAE | Safe-FD L1 | Seam safe-FD L1 |
|---|---:|---:|---:|
| SV_vs_SS | +0.000362 [-0.001063, +0.001698] | +0.006474 [+0.003221, +0.009921] | -0.009972 [-0.025839, +0.001444] |
| SV_vs_VV | +0.001281 [-0.000235, +0.003024] | -0.003243 [-0.006311, -0.000370] | +0.016922 [+0.005202, +0.032367] |
| SV_vs_SA | +0.001478 [+0.000249, +0.002676] | +0.006404 [+0.003193, +0.010187] | -0.004730 [-0.023500, +0.009366] |
| AV_vs_AA | +0.001432 [+0.000192, +0.002629] | +0.006500 [+0.003119, +0.010554] | -0.004869 [-0.023423, +0.009138] |
| AV_vs_VV | -0.000512 [-0.001843, +0.000970] | -0.003763 [-0.006975, -0.001013] | +0.010245 [-0.003639, +0.028300] |
| SV_pairing_effect | +0.000007 [-0.000098, +0.000114] | -0.000161 [-0.000360, +0.000042] | +0.000462 [-0.000750, +0.001628] |

SV 相对 SA 在 position MAE 与 safe-FD 的区间为正；SV 相对 SS 的 safe-FD 区间为正，position MAE 区间跨零；SV 相对 VV 的 safe-FD 为负，position MAE 跨零。相同 RGB recipe 的 AV 相对 AA 在 position MAE 和 safe-FD 为正，相对 VV 的 safe-FD 为负。seam 排序与前两项不同。

SV pairing effect 使用 L_offdiag−L_diag。三项区间均包含零，point estimates 为 position +0.000007、safe-FD −0.000161、seam +0.000462。

## Seed-block 与 task LOO

Seed-block 每次共同删除 S_k、A_k、V_k 及所有含这些节点的 pair。position 下 SV 对 SS/VV/SA 与 AV 对 AA 的 gain 三次均为正，AV 对 VV 三次均为负；SV offdiag−diag 在零附近变号。safe-FD 下 SV 对 SS/SA 与 AV 对 AA 三次均为正，SV/AV 对 VV 三次均为负；offdiag−diag 三次均为微小负值。seam 下 SV 对 SS 三次均为负、对 VV 三次均为正，其余对照有变号。

Task LOO 每次去掉一个 task 后重算等 task macro。Position 下 SV 对 SS/VV/SA 与 AV 对 AA 五次均为正；AV 对 VV 只有一次为正且最大正值 +0.000077；pairing effect 近零并变号。Safe-FD 下 SV 对 SS/SA 与 AV 对 AA 五次均为正，SV/AV 对 VV 五次均为负，pairing effect 五次均为负。Seam 下 SV 对 SS 五次均为负、对 VV 五次均为正；其他对照随任务变化。所有 LOO 数值在 results/SEED_BLOCK_LOO.csv 与 results/TASK_LOO.csv。

## L1 误差抵消

J_gap = 0.5*(L_X+L_Y)−L_E 描述 pair loss 相对两成员平均 loss 的凸性收益，不代表超过较好成员。36 个固定 pair 的宏平均 J_gap / 异号误差比例 / 相对较好成员 gain：

| Metric | J_gap | Opposite-sign fraction | Gain vs better member | Max element residual | Max aggregated J_gap residual |
|---|---:|---:|---:|---:|---:|
| position MAE | 0.001941 | 6.03% | 0.001031 | 2.22e-16 | 7.37e-17 |
| safe-FD L1 | 0.005369 | 7.84% | 0.002020 | 1.55e-15 | 3.41e-16 |
| seam safe-FD L1 | 0.012685 | 10.79% | 0.000727 | 8.88e-16 | 3.38e-16 |

全部 648 个 pair×task/macro ledger checks 通过。凸性抵消量与具体 pair 超过较好成员的 gain 分列；该机制本身不识别视频特有信息。

## 执行与数据暴露

- 9 个节点、36 个固定 pair：SS/AA/VV 各 3，SA/SV/AV 各 9；33 个新 bundle，3 个 V5 MIX50 原 bundle 复用。
- 新 pair loss evaluations 6,105；555 条 V5 MIX50 per-window loss 复用并做 555 条 reconciliation。首轮 reducer 类型错误在完整 6,660 行评分表写盘后发生；错误记录保存在 qa/FAILURE.json 与 qa/FAILED_ATTEMPTS.jsonl。恢复复用该表、没有重算损失；为 elementwise L1 ledger 增加一次 DEV archive 读取。
- V6 实际 DEV archive load 2 次、总 window-row loads 370，unique windows 185；figure generation 没有重新打开 archive。Final/heldout access 0，V2–V5 writes 0。
- New neural fit、optimizer、calibration、weight search、encoder extraction 全部为 0。V2 已接受的 185-window S/V child-seconds 均值分别为 0.692481 / 0.758281，两者 child sum 1.450761；这些不是端到端延迟。A 时长不在已接受的 V5 cost summary 中。每个 pair 用两个冻结模型输出，不假设 FLOPs 相等。
- Assistant 图表目视检查 PASS；external human review pending。source-group CI 条件于固定 9 个模型和开发 exposure；session/trial 关联未知，结论是开发线索而非独立确认。

固定结果不支持启动额外训练或权重搜索。

STATUS=complete；NEW_NEURAL_FITS=0；FINAL_HELDOUT_PAYLOAD_ACCESS=0；OLD_ROOT_WRITES=0；INDEPENDENT_CONFIRMATION=false；CHECKSUM_RECEIPT=qa/FINAL_CHECKSUM_VERIFICATION.json（报告完成后生成）。
