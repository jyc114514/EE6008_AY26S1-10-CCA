# 当前 Core A 结论

## 范围

以下结论只适用于 G1 duplicate-safe v3 的 development/validation
population：2,757 个 train episodes、617 个 development/validation episodes、
120 个 primary classes，`final_test_access=0`。Full-120 Stage1 使用单一 seed
`20260825`，恢复设置为 physical batch 8、gradient accumulation 4，即
effective batch 32；锁定协议原本是 batch 48、accumulation 1。

## Full-120 Stage1 结果

| Cell | Input | Head | Adaptation | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | NLL | ECE | Acceptance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D-P-F | dynamic | pooled | frozen | 94.814 | 99.676 | 90.211 | 0.3315 | 0.1384 | accepted_complete |
| D-P-L | dynamic | pooled | LoRA | 96.110 | 99.676 | 92.200 | 0.1451 | 0.0245 | accepted_derived_unmerged |
| D-A-F | dynamic | attentive | frozen | 93.517 | 98.217 | 83.475 | 0.3977 | 0.0430 | accepted_complete |
| D-A-L | dynamic | attentive | LoRA | 93.841 | 98.541 | 82.128 | 0.3450 | 0.0382 | accepted_complete |
| S-P-F | static repeated-first-frame | pooled | frozen | 94.976 | 99.676 | 91.035 | 0.3500 | 0.1455 | accepted_complete |
| S-P-L | static repeated-first-frame | pooled | LoRA | 97.083 | 99.676 | 92.992 | 0.1241 | 0.0221 | accepted_complete |
| S-A-F | static repeated-first-frame | attentive | frozen | 92.382 | 98.541 | 82.915 | 0.3836 | 0.0500 | accepted_complete |
| S-A-L | static repeated-first-frame | attentive | LoRA | 94.976 | 99.028 | 88.655 | 0.2716 | 0.0353 | accepted_complete |
| STATE-CTRL | state-only | MLP | head-only | 63.371 | 87.034 | 49.080 | 1.9805 | 0.3566 | accepted_complete |

源 acceptance 是 9/9，但这是 artifact/run gate，不代表每个科学假设都
通过。D-P-L 没有 source `RUN_METRICS.json`；公开 aggregate 是从 canonical
unmerged base-plus-adapter prediction 独立复算得到的。原 BF16 merged-export
路径仍是 failed diagnostic。

## 当前证据支持的解释

- 当前 dynamic video 中按 Top-1 最强的是 D-P-L（96.110%）；绝对分数最高的
  video cell 是 S-P-L（Top-1 97.083%、macro-F1 92.992%）。两者都是单 seed
  development/validation 观察。
- 目前不能证明 LoRA 的增益是 dynamic-specific。pooled head 的 dynamic
  LoRA−frozen 为 Top-1 +1.297 个百分点、macro-F1 +1.988 个百分点；static
  对应 +2.107 和 +1.957。Top-1 interaction 为 -0.810 个百分点，paired
  bootstrap 95% CI 为 [-2.917, +1.297] 个百分点。
- attentive head 的 dynamic LoRA−frozen 为 Top-1 +0.324、macro-F1 -1.347；
  static 对应 +2.593、+5.739。interaction 为 Top-1 -2.269、macro-F1 -7.086。
  dynamic gain 没有满足锁定 gate 中“至少不低于 static gain”的组件。
- dynamic/static 证据随 head 和 protocol 混合：pooled frozen pair 的
  dynamic−static Top-1 为 -0.162 个百分点；attentive frozen pair 为 +1.135
  个百分点。这是 matched screen 结果，不是已隔离 dynamics understanding
  的证明。
- state control 显示 state 中存在大量 task 信息，因此高 task-recognition
  分数本身不能证明模型使用了 video motion。历史 state-only（86.494% /
  73.657%）与当前 STATE-CTRL（63.371% / 49.080%）属于不同 protocol，不能
  做 paired 或互换使用。

## 当前证据不支持的结论

本发布包不支持 action anticipation、因果 dynamic understanding、部署性能、
final-test 泛化或跨 seed 稳定性结论。12-task pilot 不是 120-class 证据。
Stage2 未启动；本次发布没有运行新实验。

精确 effect 和区间见
[COMPARISON_EFFECTS.csv](../../experiments/core_a/COMPARISON_EFFECTS.csv)，
checkpoint/merge 边界见 [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md)。
