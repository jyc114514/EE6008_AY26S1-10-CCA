# Core A curated results

All rows in this directory are aggregate or plot-data records. They do not
contain raw frames, per-sample predictions, logits, targets, checkpoints,
features, or complete logs.

- `CORE_A_UNIFIED_RESULTS.csv`: nine accepted Full-120 Stage1 cells, including
  the state control.
- `CORE_A_FACTORIAL_EFFECTS.csv`: matched LoRA gains and interactions.
- `CORE_A_PAIRWISE_COMPARISONS.csv`: paired development/validation comparisons.
- `BASELINE_PANORAMA.csv` and `BATCH_SIZE_EFFECTS.csv`: historical references.
- `CORE_A_12_TASK_PILOT_SUMMARY.json`: frozen and LoRA pilot summaries; these
  are exploratory and cannot be promoted to Full-120 evidence.
- `figure_manifest.json`: figure, plot-data, caption, and script links.

The authoritative interpretation is in
[`docs/core_a/CURRENT_CONCLUSIONS.md`](../../docs/core_a/CURRENT_CONCLUSIONS.md).
