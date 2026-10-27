> Archived study summary for `CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z` (source snapshot `20261015`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# EE6008 Core B V7: fixed teacher soft targets for one fusion head

## Execution

- Accepted new head fits: 9/9; formal optimizer updates: 28800.
- Student dev prediction jobs: 9/9; student primary window-loss evaluations: 1665.
- Six frozen S/V outer-train teacher bundles were replayed; three fixed T_i predictions were averaged in native float64 before the Y train read.
- The synchronous GT_RETRAIN control reproduced the accepted V_i checkpoint byte-identically for all three seeds.
- Train teacher outputs are in-sample supervision, not OOF. The student forward uses one TS11-D head and reads no teacher output or dev GT.
- Final/heldout payload access: 0.
- Preserved earlier technical attempts: 4 (details in the final acceptance record).

## Absolute macro losses

| Method | Position MAE | Safe-FD L1 | Seam safe-FD L1 |
|---|---:|---:|---:|
| GT_RETRAIN | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| SELF_SOFT | 0.159380 ± 0.001048 | 0.822409 ± 0.002542 | 0.794948 ± 0.012296 |
| MIX_SOFT | 0.159188 ± 0.001340 | 0.824029 ± 0.003250 | 0.775311 ± 0.008490 |
| S | 0.157555 ± 0.001035 | 0.830807 ± 0.003094 | 0.766943 ± 0.005265 |
| V | 0.158338 ± 0.001486 | 0.819754 ± 0.002994 | 0.795761 ± 0.021558 |
| T | 0.155471 ± 0.001066 | 0.818974 ± 0.002634 | 0.765222 ± 0.013308 |
| BIAS_STATE | 0.157301 ± 0.001047 | 0.830807 ± 0.003094 | 0.819232 ± 0.006349 |
| BIAS_VIDEO | 0.158066 ± 0.001482 | 0.819754 ± 0.002994 | 0.847943 ± 0.021419 |
| PERSISTENCE | shared | shared | shared |

Positive gain means reference loss minus candidate loss; positive values favor MIX_SOFT.

- MIX_VS_GT: MAE gain -0.000850 ± 0.001198 (95% CI [-0.002180, +0.000579]); safe-FD gain -0.004274 ± 0.001355 (95% CI [-0.008554, -0.000481]); positive seeds MAE 1/3, FD 0/3; POINT_JOINT_GAIN=False; CONDITIONAL_CI_JOINT_GAIN=False.
- MIX_VS_SELF: MAE gain +0.000192 ± 0.001083 (95% CI [-0.000865, +0.001267]); safe-FD gain -0.001620 ± 0.001463 (95% CI [-0.005218, +0.001437]); positive seeds MAE 1/3, FD 0/3; POINT_JOINT_GAIN=False; CONDITIONAL_CI_JOINT_GAIN=False.

### Other paired references

| Contrast | Position MAE gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |
|---|---:|---:|---:|
| SELF_VS_GT | -0.001041 [-0.002269, +0.000216] | -0.002654 [-0.004963, -0.000462] | +0.000813 [-0.013159, +0.011430] |
| MIX_VS_OLD_V | -0.000850 [-0.002180, +0.000579] | -0.004274 [-0.008554, -0.000481] | +0.020449 [+0.007785, +0.035567] |
| MIX_VS_TEACHER | -0.003716 [-0.005420, -0.002029] | -0.005055 [-0.008413, -0.001826] | -0.010089 [-0.017746, -0.002906] |
| MIX_VS_BIAS_STATE | -0.001886 [-0.004639, +0.000873] | +0.006779 [+0.001779, +0.011942] | +0.043920 [+0.024654, +0.060681] |
| MIX_VS_BIAS_VIDEO | -0.001122 [-0.002455, +0.000319] | -0.004274 [-0.008554, -0.000481] | +0.072632 [+0.057615, +0.089329] |

The intervals are unadjusted exploratory source-group bootstrap intervals (5,000 draws, seed 2026101507), conditional on the three paired source instances and exposed dev groups. They are not p-values, equivalence intervals, or independent confirmation.

## MAE teacher-gain retention

- seed 202610131: R=-0.3890935423206241; numerator=-0.001501; denominator=0.003857; epsilon=1.000e-12.
- seed 202610132: R=0.20572026098363846; numerator=0.000532; denominator=0.002588; epsilon=1.000e-12.
- seed 202610133: R=-0.733386496614437; numerator=-0.001580; denominator=0.002155; epsilon=1.000e-12.
- overall_equal_seed_losses macro: R=-0.2963433871841275; numerator=-0.000850; denominator=0.002867; epsilon=1.000e-12.

R is a descriptive MAE point estimate with no ratio CI. R=1 means the MIX_SOFT point loss reaches the teacher's MAE gain; it does not prove ensemble equivalence.

## Evidence files

- Training target perturbations: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/TRAIN_TARGET_SIGNAL.csv
- Last-epoch train fit and imitation diagnostics: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/TRAIN_FIT_AND_IMITATION.csv
- Full training curves: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/TRAINING_CURVES.csv
- Task/seed rows: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/PRIMARY_TASK_SEED.csv
- Paired effects and intervals: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/PAIRED_EFFECTS.csv and server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/PAIRED_GROUP_CI.csv
- Seed/task LOO: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/SEED_BLOCK_LOO.csv and server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/TASK_LOO.csv
- Cached-input timing: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/INFERENCE_PROFILE.csv
- Resource/exposure ledgers: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/results/RESOURCE_LEDGER.csv and server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/inventory/OUTCOME_EXPOSURE.json; CPU/RAM metadata: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/inventory/RUNTIME_ENV.json.
- Host process peak RSS was not instrumented; system availability and materialized array payload sizes are recorded separately.
- V7 root: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z
- Final checksum receipt: server-only://project-artifacts/reports/benchmarks/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/qa/FINAL_CHECKSUM_VERIFICATION.json

The conclusion is bounded to the fixed 0.5 soft-target ratio, this fusion-head architecture, recipe, and three paired source instances. It does not establish that video-only information caused a gain, that averaging is irreplaceable, or that development results transfer to an independent test set.


Execution status: complete. The checksum read-back receipt is written after the manifest at qa/FINAL_CHECKSUM_VERIFICATION.json.
