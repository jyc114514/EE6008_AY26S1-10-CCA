# Portable code paths

The exported Python sources replace private server roots with environment-based paths. For a local reproduction, set only the paths required by the selected study:

- `EE6008_PROJECT_ROOT`: project-local metadata, code, and study tree;
- `EE6008_DATA_ROOT`: approved source data mount;
- `EE6008_CHECKPOINT_ROOT`: approved pretrained or trained checkpoints;
- `EE6008_CACHE_ROOT`: derived feature/cache root;
- `EE6008_OUTPUT_ROOT`: run outputs for the selected user/run;
- `EE6008_RUN_ROOT`: optional exact run root for a recovered study;
- `EE6008_LOG_ROOT`: user/run-isolated logs;
- `EE6008_HOST_LABEL`: optional portable host label for run receipts;
- `EE6008_CUDA_VISIBLE_DEVICES`: optional logical-device visibility for a separately authorized run.

The published sources include training and inference entrypoints but this synchronization did not execute them. Full datasets, checkpoints, feature tensors, predictions, targets, and per-run logs remain server-only; the artifact registry records the retained categories and the recovery identifiers. Follow the project's GPU policy and study-specific lock before executing a workload.

The original experiment-time commit is `UNKNOWN` when it could not be recovered from a local Git checkout. `PUBLIC_FILE_MANIFEST.csv` provides the exact exported file hashes; the internal source-to-export map retains original source hashes and path-redaction changes.

## Repaired historical entrypoints and execution boundary

These public copies are source-hash-bound repairs, not the exact files proven to have produced historical runs. Full runs still require study-specific server-only inputs, locks, receipts, features, predictions or checkpoints. Non-public server path references are stable artifact_id values; public repository-relative entrypoints and results remain directly indexed.

| Study | Public entrypoint | Explicit path contract | Check boundary |
| --- | --- | --- | --- |
| Hand-calibrated prototype mixture | scripts/core_b/studies/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z/code/analyze_cal_proto_repaired_public.py | EE6008_PROJECT_ROOT and EE6008_HAND_CALIBRATED_STUDY_ROOT | Pure CPU summary functions; full analysis needs server-only prediction and lock artifacts. |
| P1/P2 R3D visual complementarity | scripts/core_b/studies/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z/code/extract_r3d_features_repaired_public.py | Required --project-root, --study-root, --primary-inputs, --dense-inputs, --raw-root, --weights | Same-study experiment_contracts.py and model.py are public. Output records use opaque video artifact IDs. CPU CLI help and synthetic ID check passed; no decode, weights load, feature extraction or model forward. |
| V7 single-head soft-target | scripts/core_b/studies/CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z/code/v7_score_repaired_public.py | EE6008_PROJECT_ROOT and EE6008_V7_SINGLE_HEAD_STUDY_ROOT via v7_common.py | Python 3.10/3.12 syntax and synthetic scorer contract only; prediction bundles/dev targets are server-only. |
| Collapse temporal variation | scripts/core_b/studies/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z/code/replay_temporal_variation_repaired_public.py | EE6008_PROJECT_ROOT, EE6008_COLLAPSE_STUDY_ROOT, EE6008_D3_EQ_STUDY_ROOT, EE6008_D3_SCHEDULER_STUDY_ROOT | Synthetic query/summary functions only; no checkpoint, replay or training payload was loaded. |
| Query/freeze/protocol lock and preflight tools | See artifact-level paths in REPAIR_SOURCE_MAP_20261024.csv | Explicit EE6008 project/study roots and attachment paths are documented in each copy | Parse/path-contract checks only; server locks were not rebuilt. |

Only the two V2 contract commands in the repository README are verified as general CPU checks. Research scripts require exact study artifacts and protocol gates. Do not pass real final/heldout artifacts to development scripts or infer runnable status from AST parsing alone.
