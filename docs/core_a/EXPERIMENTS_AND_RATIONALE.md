# Core A experiments and rationale

## Evidence boundary

This catalog was assembled from the project source package, the v3 split and
data audits, hashed run manifests, the Full-120 Stage1 acceptance record, and
the sealed unified analysis. The release date is 2026-09-10. A row is not a
claim that every private artifact is distributable; the public files carry
aggregate values or stable `server-only://` pointers.

Core A asks whether a four-second observed RGB prefix supports task-ID
recognition on a G1 subset. It does not evaluate future action prediction.
The task count, representation, head, adaptation, split, and replication unit
must remain visible when comparing families.

## Family coverage

| Family | Research question | Planned / executed / accepted | Boundary |
| --- | --- | ---: | --- |
| G1 data and duplicate-safe v3 | Can the eligible G1 population and split be audited without using final-test metrics? | 1 / 1 / 1 | Shared prerequisite; 2,757 train, 617 development-validation, 120 primary classes; final-test untouched. |
| State-only control | How much task recognition is available from state alone? | 1 / 1 / 1 | Shortcut control, not video understanding. |
| Frozen R3D-18 reference | What conventional frozen-video reference is available? | 1 / 1 / 1 | Historical development/validation reference; no dynamic/static isolation. |
| Frozen pooled V-JEPA and feature geometry | Is task structure already present in frozen features, and does dynamic input help? | 1 / 1 / 1 | Historical pooled dynamic/static and 1-NN/centroid references; not causal encoder evidence. |
| Frozen attentive dynamic/static | Does the attentive head improve the frozen reference? | 2 / 2 / 2 | Historical multi-seed source aggregates; kept separate from pooled results. |
| 12-task frozen and Top-2 LoRA pilots | Can the LoRA seam and training path run on a small pilot? | 2 / 2 / 2 | Exploratory only; never promoted to 120-class evidence. |
| Batch-4/32/48 screening | How much does optimization and single-factor head/loss choice affect the frozen result? | 6 / 6 / 6 | Historical source aggregates with seed-level SD; not a unified leaderboard. |
| Full-120 LoRA Stage1 | Does LoRA add a dynamic-specific gain under matched pooled/attentive and dynamic/static conditions? | 9 / 9 / 9 | Single seed; effective batch 32 after recovery; D-P-L is accepted only on the unmerged derived path. |
| Full-120 LoRA Stage2 | Would a gated multi-seed confirmation be justified? | 12 / 0 / 0 | Not launched because the locked dynamic-specific gate was not satisfied. |
| Engineering validation | Are checkpoint, merge, writer, schema, ID, and metric seams auditable? | 1 / 1 / 1 | QA evidence, not a model-performance result. |

## Full-120 Stage1 matrix

The eight video cells cross:

- dynamic versus repeated-first-frame static input;
- pooled versus attentive head; and
- frozen versus top-block LoRA adaptation.

`STATE-CTRL` is a separate state-only control. The source acceptance record
reports `accepted=9`, `required=9`, and `final_test_access=0`. The public
run-level table retains the exact cell-level metrics, source run status,
selected epoch where available, precision, effective batch, and artifact
hashes. Use [RUN_LEVEL_RESULTS.csv](../../experiments/core_a/RUN_LEVEL_RESULTS.csv)
and [ACCEPTANCE_SUMMARY.csv](../../experiments/core_a/ACCEPTANCE_SUMMARY.csv)
instead of reconstructing a ranking from filenames.

## Relationship between families

The data/QA family establishes the population and duplicate-safe v3 contract.
The historical frozen and geometry families establish references. The
12-task pilot validates a seam but is intentionally not evidence for the
Full-120 factorial. Batch-size screening shows that optimization choices can
change the observed dynamic/static gap. Full-120 Stage1 is therefore reported
as a matched single-seed screen, and Stage2 remains a proposal rather than an
executed result.

## Metrics and uncertainty

Primary reported metrics are Top-1, Top-5, and macro-F1; NLL, ECE, Brier, and
mean class recall are secondary. The Stage1 effect tables use 10,000 paired
episode bootstrap resamples and exact two-sided McNemar tests with the source
Holm correction. Episodes—not time steps or trajectory dimensions—are the
resampling unit. Historical baseline error bars are source-declared seed-level
sample SD and are not interchangeable with the Stage1 episode bootstrap CI.
