# Core B current results and limitations

## Current V2 recovery

The V2 CURRENT_PROFILE_V2 path completed both development and final scoring from previously fitted checkpoints; V2 added no training. Its original Q0 bridge remains `FAIL_UNRESOLVED`; the cause is `UNKNOWN` and is not attributed to a GPU number. The selected single-model development reference is V0; VD is a separate dropout comparison. The cached-Q0 V9 development profile is retained separately.

| Split | Episode support | Anchor support | V0 position MAE | Persistence position MAE |
| --- | ---: | ---: | ---: | ---: |
| Development | 47 | 185 | 0.158397 | 0.222262 |
| Final recovery | 41 | 162 | 0.178383 | 0.238057 |

`position MAE` and `safe-FD L1` are standardized scores, not physical robot accuracy. The score source lists 22 sealed prediction bundles per split. Final target values were read only after the final prediction seal passed. The 41 final episodes overlap the Core A indexed-video population; session independence was not established. This is a disclosed same-project final evaluation, not an independent confirmation.

## Other completed Core B evidence

- **B1/B2/B3:** B1 inference 7/7, true-pixel B2 6/6, corrected B3 8/8 logical cells. B3 used one training seed, reused two B2 controls, and did not pass its candidate gate. Persistence remains a required control; on the legacy 614-row cohort its Arm-14 standardized MAE was lower than B-STATE.
- **V3 multi-anchor:** 36/36 formal training cells and 36/36 primary prediction bundles across three seeds, six input cells, and H30/H60. The study uses 603 development episodes and 2,322 windows. It is distinct from the V3 state-information study and from B3.
- **V3 state×video:** 54/54 accepted predictions across three seeds. Full-state video gains were small and mixed; larger dynamic-video differences occurred in the anchor-only condition, which still retains the anchor action. No universal video gain is established.
- **Fixed-model correspondence and observation delay:** their reports show condition-specific development evidence. The correspondence analysis has eight Holm-adjusted primary contrasts but retains a normal-input Arm FD guardrail caveat. The observation-delay study has mixed Arm/Hand results and no established dynamic-over-static advantage.
- **JY-02:** Stage2 passed its locked multi-seed development gate. It remains a development candidate; action slices are numeric indices and do not have verified physical semantics.
- **D3 corrected scheduler:** 12/12 fits and train/dev predictions were accepted; 24/24 dense secondary predictions passed. Correcting per-example episode weighting and scheduler calls did not produce a general video benefit. H30 Full-28 gains differed by model, and one TS11-D seed remained nearly static on both train and dev.
- **V4–V9 and command/query families:** the dated summaries retain positive, negative, incomplete, and support-limited results. V9 context dropout did not improve the full-context single model; its cached-Q0 profile is not CURRENT_PROFILE_V2.
- **VideoMAE Core B handoff:** the saved-prediction scorer was accepted, while fresh checkpoint replay remained 0/12 and portable integration remained `needs_fix`.

See [dated study summaries](studies/), [the 2026-10-14 44-family inventory](../../experiments/core_b/conclusion_audit_20261014/EXPERIMENT_INVENTORY.csv), and [the claim ledger](../../experiments/core_b/conclusion_audit_20261014/CLAIM_LEDGER.csv). Those are historical snapshots; the V2 final access disclosures below are later.

## Exposure and completion status

The Core A V2 score is an amended same-source analysis after disclosed label access. Core B CURRENT_PROFILE_V2 final targets were materialized after the complete prediction seal. Both exposures are recorded in the [release summary](../releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md) and their source QA receipts. No new scoring, training, feature extraction, or final payload reads occurred for this publication task.

Scientific result validation, visual QA, human scientific/team review, video recording, and course submission are separate states. Human review and course submission remain pending in the closeout evidence.
