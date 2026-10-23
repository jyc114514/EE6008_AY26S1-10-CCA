> Archived study summary for `CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z` (source snapshot `20260923`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B results audit — handoff

**Status: complete. Required result audit, D4 V3 core, D5 decision, visual QA, and whole-root checksum are finalized.**

## Main findings

- **Comparability:** 124 registry rows carry 13 family labels; 22 is the discovered source-root count. No all-row leaderboard is valid. The 603/2,322 primary-dev population is separate from the 64-row supplementary set (53 overlap; 11 supplementary-only).
- **D1:** The train-defined middle stratum is now present as a post-hoc extension (1,090 windows / 492 exact-video groups). S-HIST vs F-TOK-D gains and high-minus-low effects vary strongly by seed/horizon; broad visual disambiguation is not established. Exact per-seed gain/CI is in 02_SCIENTIFIC_RESULTS.csv.
- **D2:** Corrected event registry has 594 group-specific truth events. Dense panel has 118 task-balanced episodes, 3,718 queries, and about 0.249611 hours of total exposure. Lead≥0.5s recall is low while some F-TOK-D false-alarm rates are high; this is not evidence of natural event prevalence.
- **D3:** 36/36 actual prediction cells and 216/216 source-point reconciliation pass. Spatial/temporal effects are seed-dependent; actual training weighting deviates from the locked per-example objective, so D3 findings are exploratory.
- **V3_STATE:** 54/54 accepted and kept separate; original Holm counts remain 6/24 primary MAE, 5/24 FD secondary, 4/8 interactions. Its checksum manifest covers 1,258/1,272 files, with 14 generated pyc omissions.
- **D4:** status D4_V3_DENSE_CORE_COMPLETE; 36/36 dense V3 bundles, 3,600 matched pairs per horizon and 118 exact source groups. The same-absolute-time video ΔU has no stable positive advantage; see the descriptive cross-seed CI and matched new-level/FD columns in d4/D4_CROSS_SEED_DESCRIPTIVE.csv.
- **D5:** decision **redesign**; no probabilistic model was trained. Historical hand calibration/screening gates failed and the router gate did not authorize fitting.

## Work state

- No new model training; final/held-out access 0; original sealed roots unchanged.
- Optional D3 dense extension: complete; V3_STATE_dense_54_of_54; V3_STATE_D4_dense_complete; D3 dense predictions=36/36; D3 rolling analysis=COMPLETE_D3_DENSE_ROLLING_EXTENSION. It is resumable using d3_extension/run_optional_dense.sh, whose extraction job uses --resume and the shared GPU watcher.
- CPU tests: PASS; dashboard visual QA: PASS; whole-root SHA-256: verified.

## Key files

- 02_SCIENTIFIC_RESULTS.zh-CN.md — readable result and interpretation.
- 02_SCIENTIFIC_RESULTS.csv — unified per-seed/metric/CI/support export.
- 02_CLAIM_LEDGER.csv — fact, interpretation, alternative explanation, evidence boundary.
- figures/BENCHMARK_DECISION_DASHBOARD.png — dashboard; visual QA PASS.
- d4/D4_PAIR_MANIFEST.csv, d4/D4_PRIMARY_RESULTS_BY_SEED.csv, d4/D4_PAIRED_EFFECTS.csv, d4/D4_BOOTSTRAP.csv, d4/D4_EVENT_AMBIGUITY_SLICES.csv, d4/figures/ — D4 outputs.
- d5/D5_GO_NO_GO.md — evidence gate and decision.
- qa/TEST_RESULTS.json and qa/VISUAL_QA.json — verification state.


D3 dense rolling update result: all 36 model/seed/horizon bundles were accepted on the same 3,718-query panel; 3,600 matched adjacent pairs per horizon were joined to the locked absolute target frames/timestamps. Per-seed and descriptive cross-seed results are in d4/d3_extension/D3_D4_PAIRED_EFFECTS.csv and D3_D4_CROSS_SEED_DESCRIPTIVE.csv. This remains development-only and inherits the D3 training-objective deviation.
