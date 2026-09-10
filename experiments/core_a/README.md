# Core A experiment registry

`EXPERIMENT_CATALOG.csv` describes the research question, treatment/control,
population, planned/executed/accepted counts, and evidence boundary for each
family. `RUN_LEVEL_RESULTS.csv` is the nine-cell Full-120 Stage1 aggregate
table. `ACCEPTANCE_SUMMARY.csv` carries sanitized artifact IDs and source
hashes. `COMPARISON_EFFECTS.csv` contains paired and factorial effects.

Counts are not a ranking across incompatible protocols. In particular, the
12-task pilot, historical frozen baselines, Batch-4/32/48 screens, and
Full-120 LoRA Stage1 remain separate families.
