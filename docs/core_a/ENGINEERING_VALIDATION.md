# Core A engineering validation

Engineering validation is reported separately from model performance.

## Code and import closure

The project-authored package in `src/ee6008/` contains the configuration,
dataset inventory/selection/split, clip preprocessing, feature cache, model
loader seam, pooled and attentive heads, metrics, probe, LoRA, and top-block
forward helpers. The Core A scripts import those modules through the
`PYTHONPATH=src:scripts/core_a` layout. The V-JEPA implementation itself is an
external dependency; it is not vendored in this repository.

The public copy of `data/clips.py` keeps decord optional and falls back to the
recorded OpenCV decoder when the optional C++ extension cannot import. This is
a portability guard only; it does not alter the decoder choice when decord is
usable.

The LoRA implementation inserts adapters only into blocks 10 and 11 at `qkv`
and `proj`, freezes base parameters, and zero-initializes `lora_B`. The focused
tests exercise the zero-init equivalence, frozen boundary, top-block forward,
token-shape guard, early-stopping tie break, and final-test split guard.

## Checkpoint and merge seam

The read-only replay evidence reports:

- FP64 algebraic merge maximum absolute difference:
  `1.687538997430238e-14`.
- Best checkpoint U32/M32 maximum absolute difference:
  `4.053115844726562e-06`, with zero values outside tolerance.
- Last checkpoint U32/M32 maximum absolute difference:
  `6.794929504394531e-06`, with zero values outside tolerance.
- BF16 merged versus unmerged maximum absolute difference: `0.0625`, with
  four values outside tolerance for best and five for last.
- BF16 merged/unmerged argmax disagreement: zero in the recorded replay, but
  the merged export is still unsupported because value-level equivalence failed.

Therefore D-P-L is represented by `unmerged_base_plus_adapter` inference. The
checkpoint and prediction hashes are recorded as server-only artifact metadata;
the files themselves are not distributed.

## Writer, schema, and metric checks

The accepted source gate checked finite logits, unique episode IDs, labels,
predictions, argmax consistency, strict model/head load, and development-only
selection. The unified analysis independently recomputed the aggregate metrics
and recorded that D-P-L has a missing source `RUN_METRICS.json`. No placeholder
zero was substituted.

The source analysis QA passed its cross-file checks, including nine current
cells, row counts, bootstrap shape/range, Stage1 acceptance, no-new-training,
Stage2 non-launch, figure links, visual QA, final-test lock, and state
reconciliation. The public figure QA is rerun from this clone.

## Interpretation boundary

Passing an engineering check means that a seam or artifact relationship is
consistent. It does not make a single-seed development result a final result,
does not repair the BF16 merged-export failure, and does not establish causal
dynamics understanding.
