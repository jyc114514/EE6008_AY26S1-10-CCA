# Excluded artifacts and reasons

The following classes were found during the project-wide discovery and are not
committed to the public repository:

- Raw G1/H1 data, video, frames, Parquet files, and data archives: size,
  privacy/licensing, and final-test boundary.
- Checkpoints, model weights, optimizer/RNG state, and adapter payloads:
  large or restricted artifacts; represented by server-only IDs and hashes.
- Feature tensors, logits, targets, per-sample predictions, bootstrap index
  arrays, and episode-level sensitive arrays: not needed for the curated public
  summaries and may expose protected evaluation material.
- Complete training logs, TensorBoard/W&B state, caches, temporary files,
  queue/lease traces, and environment directories: server/runtime state.
- `.env`, credential, login-state, private-key, and private-form material:
  security boundary.
- Third-party source trees and paper PDFs with separate redistribution
  obligations: not vendored; official source/revision/license must be checked
  independently.
- Server launch wrappers and historical `.bak`/`.orig` copies containing
  machine-specific paths or stale implementations: not portable canonical
  source.
- Core B experiment results and future-direction material: outside this Core A
  release scope.
- Locked final-test IDs, labels, predictions, and metrics: never accessed or
  distributed by this task; `core_a_final_test_access=0`.

Historical/failed/retry evidence was used only to explain provenance or
engineering status. It was not silently promoted to canonical source or merged
into the result tables.
