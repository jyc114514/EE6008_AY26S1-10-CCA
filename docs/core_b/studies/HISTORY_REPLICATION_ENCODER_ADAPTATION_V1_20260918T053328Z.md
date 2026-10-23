> Archived study summary for `HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z` (source snapshot `20260918`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B 历史×视频复现决策

## 结论

Phase A 的 12 个 fresh-seed run 全部完成并通过工程验收，但 `HI-F-HIST` 未通过预先锁定的科学 gate，因此 Phase B 按协议未启动。

- fresh seeds：`20260940 / 20260941 / 20260942`；discovery seed `20260939` 未纳入聚合。
- Hand 相对 Persistence 的 crossed seed×episode effect 为 `0.01272362`，95% CI `[0.00848481, 0.01685358]`。
- Full-28 相对 fixed hybrid 的 effect 为 `0.00636181`，95% CI `[0.00422322, 0.00842160]`。
- native FD 相对 Persistence 为 `1.0342%`，95% CI `[0.8207%, 1.2457%]`；超过预锁定 +1% 上界。
- 每 seed FD 点估计约为 `1.0678% / 0.8748% / 1.1601%`，其中两 seed 已超过 1%。
- low-motion Hand MAE guardrail 的 crossed CI 上界约 `2.4646%`，也未通过。

因此这是一轮完整的 multi-seed development negative gate result：它支持历史×视频模型有位置误差收益，但 temporal-fidelity margin 跨 seed 不够稳定。不能称 final、held-out 泛化或 physical control success。

## 资源与边界

Stage0 为 `4/4`，正式训练为 `12/12`；动态 watcher 实际使用物理 GPU2/3，最多两个本项目 child，未 signal 或清理其他用户进程。GPU3 未被清空。最终测试访问为 0，未联系教授，旧冻结候选和旧 root 未修改。
