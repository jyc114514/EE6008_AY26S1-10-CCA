# VideoMAE development evidence

Contributor: Xueshen Zhang. Only small aggregate records are included. No videos, labels, per-example predictions, feature arrays, model weights or training checkpoints are redistributed here. Original datasets and third-party weights retain their own access and license requirements.

## Included records

- `CPU_ZERO_TOLERANCE_REPLAY_20261013.json`: unchanged downloaded server replay summary for twelve models and train/dev predictions. Status is `FAIL_REPLAY_TOLERANCE`, with both tolerances zero. This is a historical CPU comparison against GPU predictions, not a current global acceptance failure.
- `CPU_ZERO_TOLERANCE_REPLAY_20261013.csv`: a tabular projection of that JSON for inspection.
- `DEVELOPMENT_REPORTED_SUMMARY.json`: transcription of reported development aggregate values. It is not the original server metrics export and does not contain fabricated per-seed forecasting results.
- `FILE_MANIFEST.json`: sizes and SHA256 hashes of this contribution's code, config, tests and evidence files, excluding the manifest itself.

## Scope

Core A used 2,757 train and 617 validation episodes, 120 classes, sixteen frames from the opening four seconds, frozen VideoMAE-small and a 384-dimensional pooled feature with a linear head. Seeds were 6008, 7 and 42. Validation Macro-F1 selected checkpoints. Reported three-seed means: Top-1 94.3274%, Macro-F1 88.2317%, Top-5 98.9735%. These are development-validation results, separate from the final V-JEPA recognition study.

Core B used two modes, two horizons and seeds 20261001/2/3, producing twelve runs. Population: 17,757 train anchors and 2,322 dev anchors; the dev reduction used 603 source groups. The H30 VideoMAE mean was lower than the V-JEPA reference, but the difference concentrated in seed 20261002. Neither visual model established an average gain over its corresponding state control. The body-part semantics of `action[0:14]` and `action[14:28]` are not asserted.

## Acceptance timeline

The included 13 October CPU replay completed all twelve inference/scoring jobs but failed exact equality. Its largest train prediction difference is 0.003096103668, dev difference 0.002043843269, and metric-field difference 0.000116482377. The last number is not specifically a Position MAE difference.

The newer team record, [VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z](../../docs/core_b/studies/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z.md), reports all 24 GPU prediction arrays exactly equal to historical GPU predictions. It also identifies the largest CPU metric difference as a `tv_ratio_window_p50` field and records an initially failed metadata validation and its repair. Overall handoff acceptance remains partial in that record, with provenance and erratum actions outstanding. No retraining, encoder extraction or final/heldout access occurred in that replay scope.

These development and acceptance records must not be combined with the different final five-task benchmark population or aggregation scheme.

## Restricted artifacts

The full handoff remains in the server archive identified as `server-only://project-artifacts/reports/benchmarks/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z`. Obtain access through the team owner. It contains model manifests, original and safe weights, per-run configs, train/dev predictions, metrics, caches and normalizers. No claim is made that those large artifacts are uploaded by this contribution.


## Public source paths and evidence scope

The public script copies now use configurable paths and contain no account-specific path defaults. They change only path configuration and validation; no training, scoring, video extraction, or replay was performed, and the saved result records above were not recomputed.

The contribution's `FILE_MANIFEST.json` identifies the current public export bytes. Its hashes do not establish which exact source bytes produced the historical predictions or server replay.

The historical CPU zero-tolerance replay remains `FAIL_REPLAY_TOLERANCE`. The separate 18 October GPU replay reported 24/24 prediction arrays matching; the overall handoff remained partial with provenance and erratum items outstanding. This path update does not change either record.

## 公开源码路径与证据范围

公开脚本副本已改为使用可配置路径，且不再包含账号专属的路径默认值。修改限于路径配置和验证；没有执行训练、评分、视频特征提取或重放，上述已保存结果没有重新计算。

本贡献的 `FILE_MANIFEST.json` 标识当前公开导出文件的字节。其哈希不证明历史预测或服务器重放使用的源码与当前公开副本完全一致。

历史 CPU zero-tolerance 重放仍记录为 `FAIL_REPLAY_TOLERANCE`。另有 10 月 18 日 GPU 重放记录称 24/24 预测数组一致，但整体 handoff 仍为 partial，且 provenance 与 erratum 项尚待处理。本次路径调整不会改变这两份历史记录。
