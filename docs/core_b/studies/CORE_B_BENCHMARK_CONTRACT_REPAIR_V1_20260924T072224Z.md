> Archived study summary for `CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z` (source snapshot `20260924`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B 下一阶段审计交付

总体状态：`partial`。P2 与 D2_V3 的开发审计已完成；P0 确认旧 D3 代码不等价于新任务附件中的 episode-equal 公式，但原 `MODEL_LOSS_CONTRACT.md` / `D3_PROTOCOL_LOCKED.json` 没有写入 episode 等权。附件又明确要求目标不一致时暂停，因此没有擅自修约或启动 D3-R。

## 关键结果

- P0：train-only 数据为 17,757 windows / 2,712 episodes；实际每个64样本 batch 都混合 episode。真实首批 legacy loss `0.285918`，附件目标 `0.288485`，梯度差 `-0.001421`；32+32正确加权梯度逐参数一致，29样本尾批除以2会缩放梯度。旧 D3 有效训练/ckpt/预测 36/36，其中一格保留了 KeyboardInterrupt 失败和 RETRY1 成功。
- P2：24个对比格按 anchor ID 重算，与封存总体 D1 MAE/FD 表最大差 `7.2e-10`，checkpoint hashes 和 ID/frame/timestamp 对齐均通过。seed 20260951/H30 的 F-TOK-D 高分歧 Arm/Hand gain `-0.029802/-0.027453`，而 F-TOK-S/F-POOL-D 不同样退化；对应 F-TOK-D train loss也高。seed 20260952/H60 的 S-HIST train loss偏高、预测幅度收缩，所有视频类模型相近改善。与训练不稳定/seed×condition 效应相符，少数任务不是主要支配因素；因每格单 seed且dev已反复使用，因果归因仍未确定。
- D2_V3：先用 train-only 的 source-group 等权移动Q75冻结阈值，再锁0.2s簇与一对一匹配并评估已有36个dense bundles。Hand H30/H60 有238/259个完整机会（95/99 tasks），满足预注册支持门；Arm有174/190（106/107 tasks）。但 F-TOK-D Hand recall仍接近0，Arm recall约11–16%而误报率约200–990/hour，0.2496小时的短暴露不代表部署率。无需扩展panel。
- D5：保持 `redesign_no_training`。GT minority train/dev为174/2741与41/614，旧argmax预测多数mode为585/614；类别不平衡和minority underprediction一致，但 mode collapse 未证实。CAL/candidate gate fail、router entry gate=false、router fits=0；fixed-q FD oracle比Persistence差2.2886%。

## 需要用户裁决

请确认：是否把附件的 episode-equal 逐样本损失视为对历史 D3 锁的正式修订？如确认，下一步再按全36格 D3-R 协议推进并单独申请宿主GPU审批；如不确认，请指定以哪份历史损失合同为准。当前没有GPU子进程，GPU状态未查询。

旧实验根写入数为0，final/held-out访问为0；V3_STATE旧manifest的1258条列出hash通过，但有14个既有生成pyc未列入。详见 `BENCHMARK_VALIDITY_LEDGER.csv`、`p0_loss_audit/`、`p2_seed_audit/`、`d2_v3_measurement/` 和 `d5_decision/`。
