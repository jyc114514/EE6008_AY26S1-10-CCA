> Archived study summary for `CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z` (source snapshot `20261007`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Return to ChatGPT — Core B fusion/query mechanism probe

**Status: complete.** New experiment root: `reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z/`.

## Frozen development reference

The earlier D3-EQ, corrected-scheduler, learning-diagnosis, and tiny-fit evidence is recorded in `DEVELOPMENT_REFERENCE_FREEZE.md/.json`; old source roots and outcomes were not edited. The freeze retains the exposed-dev boundary, fixed-seed conditional intervals, unresolved numeric action-index semantics, historical extractor provenance gaps, and human review pending. It is not a final model choice or benchmark promotion.

## Fusion and query audit

`reports/FUSION_QUERY_PATH_AUDIT.zh-CN.md` and `forensics/FUSION_QUERY_PATH_AUDIT.json` trace normalization, state/video/anchor projections, time/spatial/modality encodings, memory, decoder, and residual-to-native output. No hard identity, formula, axis/broadcast, null/mask, or cache error was found. The intervention was isolated to the fixed future-query component. Manual attention reconstruction passed all 16 calls at combined atol/rtol 1e-5; it is not causal evidence.

Old raw query effective FP32 ranks are H30 4/30 and H60 5/60 under the recorded singular-value threshold; centered raw values remain 4/5. Rank after LayerNorm/learned query projection is sample-dependent. None of these raw ranks is a decoder output-capacity bound. The H30 canonical Hadamard table excludes DC row 0, uses rows 1–30, preserves canonical column order and matches the old per-step RMS. It adds zero trainable parameters; H60 was used only for math/interface checks and prefix validation.

## Four paired train-only fits

All four new cells completed 2,000 CPU FP32 updates and were compared to their exact old-query train-panel baseline. Same INIT/model parameter count (890,780); candidate adds zero trainable parameters. Fresh-process reload reproduced each train-panel prediction bitwise.

| Model | Panel | Position old → new | G_Q position | Safe-FD old → new | G_Q FD | TV old → new | Strong old → new |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS11-D | P1 | 0.013758 → 0.004161 | +0.009597 | 0.243883 → 0.031980 | +0.211903 | 0.955 → 0.998 | 0/1 → 1/1 |
| TS11-D | P8 | 0.028444 → 0.004672 | +0.023772 | 0.572828 → 0.039830 | +0.532998 | 0.915 → 0.999 | 0/8 → 8/8 |
| STATE-CTRL | P1 | 0.007698 → 0.003479 | +0.004219 | 0.064129 → 0.034595 | +0.029534 | 0.995 → 1.005 | 1/1 → 1/1 |
| STATE-CTRL | P8 | 0.013037 → 0.004478 | +0.008560 | 0.208386 → 0.040784 | +0.167602 | 0.943 → 1.005 | 6/8 → 8/8 |

Positive `G_Q` means same-panel old-query error minus orthogonal-query error. At 100-update resolution, first strict all-three gate crossings were: TS11-D P1 800→100; TS11-D P8 6/8 at 900–1700 (median 1400), 2/8 >2000 → 8/8 at 200–400 (median 200); STATE-CTRL P1 800→100; STATE-CTRL P8 7/8 at 1200–1700 (median 1400), 1/8 >2000 → 8/8 at 200–500 (median 200). First crossings are not sustained passages; old-query trajectories can later leave the band.

The descriptive fusion×query interaction `G_Q(TS11-D)-G_Q(STATE-CTRL)` is positive on both panels: P1 position +0.005378 / FD +0.182369; P8 position +0.015212 / FD +0.365396. This suggests a follow-up question, but the Hadamard contrast changes distance, spectrum, smoothness, and conditioning together. It does not isolate video causality, prove forecast/generalization improvement, establish seed stability, or select a benchmark winner. No CI or p-value is reported; P1 is nested in P8 and the targets were used in fitting.

## Cost, QA, and access boundary

Recorded accepted-run elapsed time: Stage0 0.6123 s + formal fits 226.4691 s = 227.0813 s. This sum excludes uninstrumented whole-session inspection and plotting time. Filesystem free fraction at the resource gate was 79.59%; the current experiment root is far below the 5 GiB output cap. CPU and disk details are in `protocol/CPU_COST_FINAL.json` and `FINAL_ACCEPTANCE.json`.

Dev targets opened: 0. Final/held-out access: 0. GPU query/use/lease: 0. D5 training: no. All 178 frozen protocol source hashes rechecked successfully; 77 preexisting registered source files remained unchanged. Agent visual inspection of all six final figures passed; human review remains pending and submission-ready is false.

## Next step

One proposal is recorded in `NEXT_FULL_DATA_QUERY_GEOMETRY_PROPOSAL.zh-CN.md/.json`; it is not executed or authorized. It proposes six full-train H30 candidate runs (two models × three fixed seeds) paired with accepted corrected-scheduler old-query runs. Because existing dev was exposed, the proposal requires a separately identified, unexposed source-group evaluation population before any validation claim; without one, do not claim generalization or launch under this proposal.

## Key records

- Frozen formal protocol: `protocol/PROTOCOL_LOCKED.json` (SHA-256 `f757711b…990a02866`)
- Formal acceptance: `protocol/B_FORMAL_ACCEPTANCE.json`
- Paired tables/report: `results/QUERY_GEOMETRY_RUN_SUMMARY.csv`, `results/QUERY_GEOMETRY_CURVE_COMPARISON.csv`, `reports/QUERY_GEOMETRY_TRAIN_FIT_REPORT.zh-CN.md`
- Six figures: `figures/`; caption and alt text: `reports/FIGURE_CAPTIONS_AND_ALT.zh-CN.md`
- Failures and corrections: `logs/FAILURE_REGISTER.json`
- Final source verification, acceptance, and checksums: `protocol/FINAL_SOURCE_INTEGRITY.json`, `FINAL_ACCEPTANCE.json`, `CHECKSUMS.sha256`

Human review remains pending.
