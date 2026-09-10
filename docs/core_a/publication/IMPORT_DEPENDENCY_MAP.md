# Core A import and provenance map

## Project-authored closure

The public `src/ee6008/` package is the current project-authored closure:

```text
config
├── data.clips
├── data.inventory
├── data.selection
└── data.splits
baselines ──> torchvision
cache ──> torch
probe ──> metrics
full_attentive ──> metrics
full_attentive_data ──> config, data.selection, full_attentive
models ──> external V-JEPA source at runtime
lora ──> torch
top_block_finetune ──> torch
```

Core A runners import this package through `PYTHONPATH=src:scripts/core_a`.
The runner-to-module links were traced with Python AST imports before copying.
The external V-JEPA implementation and its license are intentionally not
vendored; `models.py` fails closed unless an explicitly configured local source
tree is supplied.

## Selection decisions

- Current `src/ee6008/*.py` and `src/ee6008/data/*.py` were selected as the
  canonical project-authored implementation.
- The copied public-safe runner files were compared with current source by
  SHA-256; matching files were reused without algorithm changes.
- Current LoRA/top-block modules and their focused tests were added because the
  earlier public export predated the Full-120 LoRA evidence.
- Server launch wrappers, stale backups, private run controllers, and scripts
  whose only purpose is to consume excluded arrays remain outside the public
  canonical tree.

The exact selected-file hashes and public destinations are in
`SOURCE_SELECTION.csv`/`.json` and `PUBLICATION_MANIFEST.csv`/`.json`.
