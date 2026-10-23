> Archived study summary for `HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z` (source snapshot `20260914`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B Expert-Advantage Selector 决策

Phase 1 Gate 1=`PASS`；OOF=`2741/2741`；selector matrix=`4/4`；selector seed=`20260932`。

- Primary `N_RAW`：E-F-SP-U 的纯 ungated Hand proposal。
- STEP_GROUP oracle Hand MAE：`0.211665`；相对 Persistence oracle gain=`17.77%`。
- Fixed Arm-learned/Hand-Persistence comparator Full-28 MAE：`0.180836`。
- Preordered candidate：`none`；完整 candidate gate=`FAIL`。

SEL-AS 先判、SEL-AH 后判，未按结果重排：

### SEL-AS

Gate: `FAIL`. Failed checks: `hand_first_difference_not_worse, false_activation_cost_le_sel_m_and_sel_u`.
### SEL-AH

Gate: `FAIL`. Failed checks: `hand_first_difference_not_worse, oracle_headroom_closed_ge_25pct, false_activation_cost_le_sel_m_and_sel_u`.

当前结果仍是 development、outcome-exposed、single selector seed。SEL-U/M/AH/AS 的 Hand q 只能使用 past context 与 frozen OOF/all-train base proposal；没有 future target、future motion 或 oracle error 作为输入。若没有候选通过，不生成或执行 multi-seed selector prompt。

Final-test access=`0`；教授联系=`false`；冻结候选/Benchmark v2 均未改变；GPU1 未使用。

## 解释与边界

1. 旧 motion-event gate 的分数不等于 expert advantage。针对真实
   `N_RAW` advantage label，旧 H-F-W/H-F-T 的 AUPRC 约为 `0.3675/0.3887`，
   motion-event label 本身约为 `0.3890`，而 advantage label prevalence 为
   `0.3077`。它们能检测运动相关性，却不能直接回答 neural expert 是否比
   Persistence 更准确。
2. 当前 all-train Hand expert 的 `STEP_GROUP` oracle Hand MAE 为 `0.211665`，
   theoretical headroom 为 `17.77%`，NN-better step prevalence 为 `30.63%`。
   因此 selector 在理论上值得测试；oracle 使用 future target，是不可部署
   upper bound。
3. motion magnitude 与 expert advantage 不是同一变量；散点、按 time/task/joint
   的表格和 correlation diagnostics 均只作描述性证据，不能把 high-motion
   解释成 neural 一定更准确。
4. SEL-AS 的 Hand MAE=`0.243030`、Full-28=`0.173640`，相对 Persistence
   Hand improvement=`5.59%`，相对 fixed hybrid Full improvement=`3.98%`，
   oracle closure=`31.45%`；SEL-U 的点估计更低，但它是 reconstruction control。
5. fixed `HYB-A_NN-H_P` 仍是锁定的、不可替代的 development comparator；虽然
   SEL-U/SEL-AS 点估计低于它，SEL-AS/SEL-AH 均未通过完整 gate，故没有被提升为
   新 comparator 或 frozen candidate。
6. SEL-AS 失败项为 Hand first-difference CI effect=`-0.014973` 及
   false-activation cost=`0.027405` 高于 U；SEL-AH 还只关闭约 `23.19%`
   oracle headroom。Arm composite 对四格保持 bitwise identical。
7. 所有 selector 使用 train-only 3-fold OOF（`2741/2741`、0 overlap）；但
   selector 只有一个 seed，且本轮 Stage0/formal 是 CPU-only，因为实际项目
   watcher allowlist 为 GPU2/3，GPU1 未被绕过使用。因此状态标为 `partial`，
   而不是 GPU 完整验收。
8. 唯一建议：停止当前 selector/motion-event gating 路线，转向新的 Hand expert
   结构假设（conditional prototypes、mixture decoder 或 action-phase
   conditioning）。该方向需要另行锁协议和授权；不要自动 multi-seed 或 final。
