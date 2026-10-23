# Results

Commit only small, curated final summaries and figures with provenance. Raw predictions, checkpoints, feature tensors, TensorBoard/W&B caches, and full training outputs remain on the GPU Server.


The V2 amended Core A result and Core B studies have separate aggregate tables under `core_a/` and `core_b/`. The Core A V2 result uses disclosed same-source final-label access. Core B V2 final scores use sealed predictions followed by target reads; raw payloads and per-example arrays remain server-only. See `docs/core_a/FINAL_V2_AMENDED_ANALYSIS.md`, `docs/core_b/`, and `docs/releases/server-sync-20261022T082418Z/`.
