# Core A V2 amended analysis

**Status:** completed as an amended same-source analysis. The labels used for this score had already been accessed; this is not untouched blind confirmation.

The V2 correction used the corrected 120-class mapping, then rescored the existing six frozen/LoRA, dynamic/static prediction cells on the same 602-row primary final population. No model training, feature extraction, or model forward was run for this score. Score attempt 01 stopped before metrics; attempt 02 completed the six cells and 5,000 shared paired bootstrap draws. An independent CPU arithmetic/provenance check passed.

## Results

All values below are means across the same three fixed seeds. The primary estimand is `A-FULL602`; the class-set-matched alternative is not applicable because the model and final class sets are identical.

| Program | Top-1 accuracy | Top-5 accuracy | Macro-F1 over 120 true classes |
| --- | ---: | ---: | ---: |
| A_DYNAMIC | 0.950166 | 0.988372 | 0.861032 |
| A_STATIC | 0.931894 | 0.996124 | 0.878978 |
| Dynamic minus static | +0.018272 | -0.007752 | -0.017947 |

The paired grouped-bootstrap interval for the Top-1 difference was `[+0.002769, +0.034330]`; Top-5 was `[-0.015504, -0.001107]`; Macro-F1 was `[-0.027277, +0.022766]`. These are conditional intervals for the named same-source population and fixed checkpoints. The direction differs by metric; this does not support a broad dynamic-video improvement claim.

## Exposure and provenance

- Final-label exposure predates this correction. The two V2 score attempts read the same 602 primary labels; the first yielded no metrics and no bootstrap draws.
- The mapping correction was established from accepted training source plus train/development targets. Predictions, IDs, weights, and training seeds were not changed.
- Original experiment-time Git commit is `UNKNOWN`: the server source directory is not a Git checkout. This release records file SHA-256 values for the exported code instead.
- See [per-seed results](../../experiments/core_a/FINAL_V2_PER_SEED.csv), [aggregate results](../../experiments/core_a/FINAL_V2_SUMMARY.csv), and [paired contrasts](../../experiments/core_a/FINAL_V2_PAIRED_CONTRASTS.csv).

The original Core A Stage1 results remain the separate development/validation screen described in [CURRENT_CONCLUSIONS.md](CURRENT_CONCLUSIONS.md). This amended V2 score does not replace or rewrite those historical records.
