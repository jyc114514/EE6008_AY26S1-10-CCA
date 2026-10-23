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
