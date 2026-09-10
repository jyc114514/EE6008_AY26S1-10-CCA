# Current Core A conclusions

## Scope

These conclusions are limited to the G1 duplicate-safe v3 development/validation
population: 2,757 train episodes, 617 development/validation episodes, and 120
primary classes. `final_test_access=0`. The Full-120 Stage1 screen used one seed
(`20260825`) and recovery settings of physical batch 8 with gradient accumulation
4, i.e. effective batch 32, while the locked protocol specified batch 48 with
accumulation 1.

## Full-120 Stage1 results

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

The source acceptance is 9/9, but that is an artifact/run gate, not a claim
that every hypothesis passed. D-P-L has no source `RUN_METRICS.json`; its
published aggregate is an independent recomputation from the canonical
unmerged base-plus-adapter prediction. The original BF16 merged-export path
remains a failed diagnostic.

## Supported interpretations

- The strongest current dynamic video cell by Top-1 is D-P-L at 96.110%, and
  the strongest absolute video cell is S-P-L at 97.083% Top-1 and 92.992%
  macro-F1. Both are single-seed development/validation observations.
- LoRA gains are not demonstrably dynamic-specific. For the pooled head,
  dynamic LoRA-minus-frozen is +1.297 percentage points Top-1 and +1.988
  macro-F1, while static is +2.107 and +1.957. The Top-1 interaction is
  -0.810 points; its paired bootstrap 95% CI is [-2.917, +1.297] points.
- For the attentive head, dynamic LoRA-minus-frozen is +0.324 Top-1 and
  -1.347 macro-F1, while static is +2.593 and +5.739. The interaction is
  -2.269 Top-1 and -7.086 macro-F1. The dynamic gain does not meet the locked
  gate component requiring it to be at least the static gain.
- Dynamic/static evidence is mixed across heads and protocols. The pooled
  frozen pair has dynamic minus static Top-1 -0.162 points; the attentive
  frozen pair has +1.135 points. These are matched screen results, not proof
  that the model has isolated dynamics understanding.
- The state control confirms substantial task information in state, so a high
  task-recognition score alone cannot establish that video motion was used.
  Historical state-only (86.494% / 73.657%) and current STATE-CTRL (63.371% /
  49.080%) are different protocols and are not paired or interchangeable.

## Not supported

This release does not support claims of action anticipation, causal dynamic
understanding, deployment performance, final-test generalization, or
cross-seed stability. The 12-task pilot is not 120-class evidence. Stage2 was
not launched; no new experiment was run for this release.

See [COMPARISON_EFFECTS.csv](../../experiments/core_a/COMPARISON_EFFECTS.csv)
for the exact effects and intervals, and [ENGINEERING_VALIDATION.md](ENGINEERING_VALIDATION.md)
for the checkpoint/merge boundary.
