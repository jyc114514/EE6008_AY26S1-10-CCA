# Core A result summary

The Full-120 Stage1 source acceptance is 9/9 on 617 development/validation
episodes across eight video cells plus `STATE-CTRL`. The strongest dynamic
cell by Top-1 is D-P-L (96.110%); the strongest absolute video cell is S-P-L
(97.083% Top-1, 92.992% macro-F1). D-P-L is an accepted unmerged derived
inference because its BF16 merged export failed value-level equivalence.

The matched effect tables do not establish dynamic-specific LoRA benefit:
pooled Top-1 interaction is -0.810 percentage points and attentive Top-1
interaction is -2.269 points. Stage2 was not launched. These historical Stage1
results are development/validation-only and recorded `final_test_access=0` for
those runs. A separate later V2 amended analysis scores existing predictions on
602 same-source final labels after prior label access was disclosed; it is not
an untouched blind confirmation. See
[the V2 analysis](../../docs/core_a/FINAL_V2_AMENDED_ANALYSIS.md).

See [the full conclusions](../../docs/core_a/CURRENT_CONCLUSIONS.md),
[run-level results](../../experiments/core_a/RUN_LEVEL_RESULTS.csv), and
[figure manifest](figure_manifest.json).
