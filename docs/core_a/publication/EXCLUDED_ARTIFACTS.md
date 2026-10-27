# Excluded artifacts and reasons

> Historical scope: this list describes the 2026-09-10 Core A publication
> snapshot. A later Core A V2 amended analysis used 602 same-source final labels
> after prior access was disclosed; see [the V2 analysis](../FINAL_V2_AMENDED_ANALYSIS.md).

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
- Locked final-test IDs, labels, predictions, and metrics: not accessed in the
  2026-09-10 publication snapshot (`core_a_final_test_access=0`). The later V2
  amended analysis's 602-label score is documented separately; row-level values
  remain excluded.

Historical/failed/retry evidence was used only to explain provenance or
engineering status. It was not silently promoted to canonical source or merged
into the result tables.
