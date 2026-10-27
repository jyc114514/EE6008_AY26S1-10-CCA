# Server sync release summary — 2026-10-22

## Scope

This release packages the source code and corresponding evidence added after the previous GitHub publication at base commit `f43c07b6055e993986ee1fd6a952261ec6634b71`. The public export is based on a source snapshot at `2026-10-22T08:24:18Z`. The source project tree itself has no Git metadata, so experiment-time source commits that cannot be proven remain `UNKNOWN`; per-file source and export SHA-256 values are recorded in the export indexes.

The release includes 1,026 selected Python files, 49 dated Core B study summaries, 44 aggregate result tables, and 76 selected figures/plot-data assets. A metadata-only inventory of the non-Git source tree counted 17,741 regular files and 59,830,330,035 bytes by file stat; the source tree was not copied wholesale. The release adds the Core A V2 amended analysis, Core B V2 development/final score summaries, current configs, release-level result and file indexes, and refreshed status language. The private source-to-export maps remain in the server-sync audit folder and are not included in this public tree.

## Result status and access disclosure

- **Core A historical Full-120 Stage1:** the 9/9 artifact gate applies to the historical single-seed development/validation runs; it is not a scientific hypothesis pass and is not the current project-wide final-label-access status.
- **Core A V2:** completed as an amended analysis after a corrected class mapping. Six existing prediction cells were scored on 602 same-source final labels after prior label exposure was disclosed. No new training or model forward pass was run. This is not an untouched blind confirmation. Independent CPU arithmetic/provenance validation passed.
- **Core B B1/B2/B3 and later studies:** the formal B1, true-pixel B2, proxy diagnostic, and B3 factorial remain separate rows. V3 multi-anchor, state/video, correspondence, observation-delay, D3 scheduler, task-specialization, and later V-series summaries are separately indexed; their limits are preserved in their dated reports.
- **Core B CURRENT_PROFILE V2:** the development and final recovery score bundles passed the recorded content/numeric checks. Predictions were sealed before final targets were read. The 41 final episodes overlap the Core A indexed-video population; session independence is unknown. This is not an independent confirmation, and standardized metrics are not physical robot accuracy.
- The original Core B Q0 bridge remains `FAIL_UNRESOLVED`; cause is `UNKNOWN`. Human scientific/team review, video recording, and course submission remain pending.

## Public/server boundary

Raw videos and frames, extracted features, checkpoints/adapters, row-level prediction and target arrays, full execution logs, caches, and personal course materials remain server-only. `SERVER_ONLY_ARTIFACT_REGISTER.csv` states the boundary; `EXCLUDED_ARTIFACTS.csv` records deliberate exclusions. Aggregate tables and figures do not constitute a data or run-artifact backup.

Three legacy aggregate plot inputs had a `prediction_path` column with local absolute paths. That column was removed; all metric columns and rows were preserved. Thirty-two changed CSV files were standardized from CRLF to LF; parsed data rows and metric values stayed the same, and the affected figure-index hashes were refreshed. The credential-assignment pattern scan found no matches in the public checkout. No GPU command, training, inference, or benchmark was run for this export.

## Indexes

- [Experiment and result index](EXPERIMENT_RESULT_INDEX.csv)
- [Python code export index](CODE_EXPORT_INDEX.csv)
- [Dated study document index](STUDY_DOCUMENT_INDEX.csv)
- [Aggregate result export index](RESULT_EXPORT_INDEX.csv)
- [Figure and plot-data export index](FIGURE_EXPORT_INDEX.csv)
- [Server-only artifact register](SERVER_ONLY_ARTIFACT_REGISTER.csv)
- [Excluded artifact register](EXCLUDED_ARTIFACTS.csv)
- [Validation report](VALIDATION_REPORT.md)
- [Public file manifest](PUBLIC_FILE_MANIFEST.csv)
- [SHA-256 checksums](CHECKSUMS.sha256)

