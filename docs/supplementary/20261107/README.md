# Supplementary experiments — 7 November 2026

These three follow-up studies extend the recognition and command-forecasting experiments in the EE6008 report dated 24 October 2026. They were completed on 7 November and are documented in Appendix D of the V6 report. Historical results retain their original experimental scope; the final populations had already been evaluated, so these are same-source supplementary analyses, not new blind tests.

The studies are also separate from the earlier [development-only Core B specialization waypoint](../../core_b/studies/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z.md). Its 78 fits must not be combined with the 18 new fits reported here. Original reference analyses are [Core A](../../core_a/FINAL_V2_AMENDED_ANALYSIS.md) and [Core B](../../core_b/CURRENT_RESULTS_AND_LIMITATIONS.md).

| Study | Work performed on 7 November | Reference |
|---|---|---|
| Core A: repeated fourth second | New video features and three newly fitted recognition heads | Retained original first-frame and dynamic models |
| Core B: predicted task condition | New inference and scoring with retained forecasting weights | The same weights with true-task conditioning |
| Core B: five task-specific experts | Fifteen newly fitted experts and three newly fitted shared controls | Shared controls trained in the same environment |

## 1. Core A with the fourth second repeated

Eight frames at nominal times 3.000–3.875 seconds are taken from the original 32-frame clip (`frames[24:32]`) and repeated in order four times. This produces 32 frames while retaining motion within the fourth second. The original static control repeats the first single frame 32 times; the original dynamic input covers the opening four seconds. Four repetitions describe input construction, not four training seeds.

The experiment retains 2,757 training, 617 development-validation and 602 final episodes across 120 classes, the frozen V-JEPA 2.1 encoder, 384-pixel preprocessing, attentive head, batch size 48, original optimization settings and development-based checkpoint selection. Seeds are `20260825`, `20260826` and `20260827`. All 120 canonical output columns participate in ranking. Training retains the 20-epoch maximum and original early stopping; the three fits finish after 17, 17 and 20 epochs, selecting epochs 12, 12 and 15 respectively.

Final results are mean ± sample standard deviation across three fitted seeds. Means are percentages; standard deviations are percentage points.

| Input | Top-1 | Top-5 | Macro-F1, fixed 120 classes |
|---|---:|---:|---:|
| Original first frame repeated | 93.189 ± 1.476 | 99.612 ± 0.096 | 87.898 ± 2.729 |
| Fourth second repeated | 93.300 ± 1.413 | 98.228 ± 0.583 | 86.101 ± 2.253 |
| Original dynamic clip | 95.017 ± 0.288 | 98.837 ± 0.439 | 86.103 ± 0.884 |

The repeated fourth second establishes no overall advantage. Top-1 differs from the first-frame control by +0.111 percentage points, with conditional paired 95% interval [−1.550, +1.717]. Top-5 is lower by 1.384 points [−2.270, −0.609]. Against dynamic input, Top-1 is lower by 1.717 points [−2.769, −0.720], in the same direction for all three fitted seeds. Nearly equal aggregate Macro-F1 against dynamic input masks different class outcomes; it does not establish equivalence.

This comparison changes both observation timing and motion content relative to the first-frame control. Historical and new runs also use different execution environments. It therefore does not isolate a causal effect of observation time.

## 2. Core B with predicted task labels

This experiment uses the **original dynamic Core A**, separately from Study 1. Its three fits are paired in order with forecasting seeds `202610131`, `202610132` and `202610133`. Forecasting weights remain fixed. New inference compares true-task and predicted-task conditioning across seven retained conditions: state and video-and-state predictors, each with and without numerical-context dropout; the separate state reference; and the two fixed equal-weight mixtures.

Routing uses unrestricted Top-1 classification over 120 canonical tasks. Predictions outside the five supported forecasting tasks cause abstention. Missing predictions, missing observation cutoffs and classifier observations extending beyond a forecasting anchor are rejected. There is no five-class restriction or true-label fallback. The final classifier frame precedes the earliest forecasting anchor.

All three classifiers correctly identify all 41 final episodes, covering 162/162 anchors. True-task and predicted-task prediction arrays are identical for all seven conditions and fitted seeds. All three metric differences are zero, with conditional paired intervals [0, 0]. For the retained video-and-state model, the final means remain Position MAE **0.178383**, Safe-FD L1 **0.865373**, and Seam Safe-FD L1 **0.841089**.

Development coverage is 185/185 anchors for two fits and 181/185 (97.84%) for the third: one episode receives an unsupported task prediction, rejecting four anchors. Differences on accepted rows are at floating-point scale. Accepted-row means and intervals on the common 181-anchor population refer to different populations and must remain separately identified.

This demonstrates successful task-input substitution on the evaluated subset. It does not test robustness to an incorrect accepted routing decision, or imply perfect recognition across all 120 tasks.

## 3. Five task-specific video-and-state experts

Each of five tasks receives three newly fitted experts: folding a towel (`119`), placing a dumpling (`94`), pushing a duck (`90`), handing back a lunch bag (`141`), and clicking a mouse (`230`). The 15 experts are compared with three newly fitted shared video-and-state models. The **fresh shared models are the primary control**, distinct from the retained historical video-and-state model in Study 2.

All 18 fits start from the original corresponding downstream initialization, rather than fine-tuning trained models. Architecture and five-way task interface are retained, with each expert's task token fixed to its own task and no numerical-context dropout. Training retains AdamW, initial learning rate 0.0005, batch size 64, 160 epochs and the fixed final-epoch checkpoint. The objective is position L1 plus 0.1 times future-internal difference L1. Experts filter the original epoch order to disjoint task subsets, retain equal episode weighting within each task and keep partial batches; their cosine schedules use their actual update counts.

The original training population contains 1,280 anchors from 220 episodes. Final scoring retains 162 anchors from 41 episodes and aggregates anchors to episodes, then gives the five tasks equal weight. True-task and original-dynamic-Core-A routing give identical final results; predicted routing retains the development abstention described in Study 2.

All errors below are standardized and lower is better. Values are mean ± sample standard deviation across three fitted seeds. Differences are calculated from unrounded values as **expert minus fresh shared**; negative differences favour experts. This sign is opposite to the reference-minus-candidate gain convention in Equation (9) of the main report.

| Metric | Fresh shared | Five-expert bank | Difference [conditional 95% interval] | Relative change |
|---|---:|---:|---:|---:|
| Position MAE | 0.178189 ± 0.002062 | 0.172382 ± 0.001254 | −0.005807 [−0.010654, −0.000916] | −3.26% |
| Safe-FD L1 | 0.866862 ± 0.003675 | 0.868914 ± 0.002245 | +0.002052 [−0.007067, +0.011751] | +0.24% |
| Seam Safe-FD L1 | 0.840517 ± 0.013394 | 0.953398 ± 0.012528 | +0.112882 [+0.069655, +0.164475] | +13.43% |

All three fitted seeds improve Position and worsen Seam. The Position gain is task-dependent: pushing a duck improves by 15.53%, while handing back a lunch bag worsens by 4.29%. Clicking a mouse has only three final episodes and an interval crossing zero. Per-task comparisons are exploratory, without multiplicity adjustment. These results do not support a universal advantage across the five tasks.

Position averages 30 future rows; Safe-FD evaluates 29 future-internal differences. Seam measures first-future-row error under the original Safe-FD scales because the common anchor cancels. Lower average Position error can therefore coexist with higher Seam error. The absence of a separate Seam training penalty is not established as the cause of this outcome. These are offline standardized command errors, not physical accuracy, physical safety or closed-loop robot performance measures.

### Comparison limits

- Each predictor has 891,548 parameters. The five-expert bank stores 4,457,740 parameters, although only one expert is active per prediction. It performs 3,680 optimizer updates per seed versus 3,200 for shared learning, a 15% increase. Both systems have 204,800 sample presentations per seed. Total parameter and update budgets are not matched.
- Both newly fitted systems train on the historical `Q0` feature cache and evaluate the full-precision `CURRENT_PROFILE_V2` features described in Section 2.7. Numerical equivalence between those feature profiles has not been established. This is a shared condition, not a resolved numerical bridge.
- Reported intervals use 5,000 paired episode/source-cluster resamples, within task for Core B, conditional on the retained fitted models. Training seeds are not resampled. The `±` values are sample standard deviations, not confidence intervals.
- Core A's fixed-class bootstrap assigns zero F1 to missing classes; an average of 9.51 classes is absent per resample. Absolute Macro-F1 percentile intervals are omitted, and paired Macro-F1 intervals retain this limitation.
- Predictions were saved and sealed before follow-up target scoring. This does not erase previous exposure to the final populations or turn the studies into unseen confirmatory tests.

## Provenance and reproduction

The numbers are aligned with V6 Appendix D. The latest retained verification completed at 2026-11-07 17:37 UTC (8 November in China). It matched 361 saved artifact hashes and six additional historical Core A prediction hashes against the server. It checked 21 trained checkpoints: three Core A heads, 15 task experts and three fresh shared controls. It also independently read 112 sealed Core B prediction files and recomputed 8,856 episode-score rows from predictions and original scoring targets, with zero maximum difference from the saved scores. Core A verification independently rescored nine final prediction archives and 54 saved validation-epoch outputs.

The public [verification summary](../../../experiments/supplementary/20261107/verification.json) is a selected-field projection of those retained receipts. The [aggregate provenance record](../../../experiments/supplementary/20261107/provenance.json) names source roles and SHA-256 values. It does not expose private locations or per-example data. Preparation of this repository addition on 12 November did not reconnect to the server, retrain models or repeat full inference.

The [result directory](../../../experiments/supplementary/20261107/) contains:

| Files | Meaning |
|---|---|
| `core_a_final_summary.csv`, `core_a_per_seed.csv`, `core_a_paired_differences.csv` | Recognition results and paired differences against historical controls |
| `core_b_predicted_task_*.csv` | Retained-model inference results, per-seed aggregates, coverage and common-cohort intervals |
| `core_b_experts_macro.csv`, `core_b_experts_per_seed.csv`, `core_b_experts_paired_per_seed.csv` | Primary fresh expert/shared comparison on the full original population |
| `core_b_experts_per_task.csv`, `core_b_experts_paired_per_task_seed.csv` | Exploratory effects for each of the five tasks |
| `core_b_experts_budget.csv` | Parameter, training exposure and optimizer-update budgets |
| `verification.json`, `provenance.json` | Sanitized audit scope and source hashes |

Blank sample-SD fields for the single persistence control mean not applicable, not zero. Development predicted-task summaries use seed-specific accepted populations, while paired intervals use the common accepted population. Do not pool these estimands. Relative expert changes use unrounded five-task macro means.

The [public configuration](../../../configs/supplementary/20261107_public.json) is a readable projection of recorded settings, not a replacement for the original locked run configuration. The [source export and reproduction guide](../../../scripts/supplementary/20261107/README.md) distinguishes the prepared path adapters from the actual historical execution source. It also lists the authorized legacy dependencies needed for a real rerun.

From the repository root, the following check uses only the public tables and standard Python library:

```bash
python scripts/supplementary/20261107/verify_public_results.py
```

This checks arithmetic, population/seed consistency and this addition's [SHA-256 manifest](SHA256SUMS). It does not verify private prediction arrays or replay training. Full reproduction requires the authorized dataset snapshot, split/class manifests, frozen encoder checkpoint, downstream initialization, accepted source configuration and feature caches. Summary tables alone do not supply these inputs.

Cui Can's contribution to these follow-up studies covers research-question design, execution coordination, comparisons and result verification. Attribution for the original architecture, implementation and benchmark remains as documented in the main report and repository.
