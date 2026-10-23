# Independent post-release acceptance

- Audit timestamp: 2026-10-23T07:18:50Z
- Repository: `jyc114514/EE6008_AY26S1-10-CCA`
- PR: [#2](https://github.com/jyc114514/EE6008_AY26S1-10-CCA/pull/2)
- Original upload head reviewed: `4988301e113719d7c7796235c32a89b10ceec51a`
- Base/main at review: `f43c07b6055e993986ee1fd6a952261ec6634b71`
- Independent copy method: GitHub tree/blob API, fetched by the pinned commit; no source-checkout files copied.
- Result: **NEEDS_FIXES**. This is a publication-readiness result, not a conclusion about the reported science.

## Independent public-copy integrity

The fetched commit tree contained 1,417 files (25,264,972 bytes), no symlinks, and no paths outside the repository. Every fetched blob matched the GitHub tree SHA, byte size, and mode.

In `docs/releases/server-sync-20261022T082418Z/`:

- `PUBLIC_FILE_MANIFEST.csv`: 1,236 entries; all paths, sizes and SHA-256 values matched.
- `CHECKSUMS.sha256`: 1,237 lines; all hashes matched. Its additional covered path is the manifest itself. The checksum file does not hash itself.
- Code index: 1,026 entries; document index (`STUDY_DOCUMENT_INDEX.csv`): 49; result index: 44; figure index: 76. Every indexed path, size and SHA-256 matched.
- These counts cover different scopes. The PR diff count is 1,238; it is not expected to equal any one export-index count.

The original manifest/checksum set remains scoped to the original upload head above. The versioned acceptance manifest and checksum added by this update cover only the files changed or added by this acceptance update.

## Excluded code and result-entry mapping

`EXCLUDED_CODE_IMPACT.csv` identifies all 18 excluded candidates using project-relative paths, with no server-absolute path. A static scan of public Python imports, dynamic-loader string arguments and subprocess literals found no direct references to the excluded module names. That is a static result; it cannot prove that runtime-constructed paths never refer to them.

The table distinguishes administrative/lock wrappers from study analyzers and feature extraction. It records material study-level replay gaps for the hand-calibrated analysis, P1/P2 R3D feature extraction, V7 single-head scorer and collapse temporal-variation replay. In particular, the public V7 ensemble scorer is a different study and is not a substitute for the excluded single-head scorer. These gaps are not hidden by the result summaries.

`ENTRYPOINT_RESULT_MAP.csv` has 32 rows: all 23 `EXPERIMENT_RESULT_INDEX.csv` run entries plus nine priority study/result rows for frozen V3, V4-V9 and cached-Q0. It points to concrete study scorers/analysers where the source tree supports the mapping; it uses `UNKNOWN` or `PARTIAL` where source-to-export identity is not proven. The nine historical Core A Full-120 Stage1 rows remain `UNKNOWN` at source-file level because the recorded runtime commit is not bound to an exact exported file snapshot. V4-V9 study notes and figures are not represented as public aggregate CSVs where those CSVs are absent.

## CPU checks and boundary

The existing project environment was Python 3.12.14, pytest 9.1.1, NumPy 2.5.2, scikit-learn 1.9.0 and PyArrow 25.0.1. The checks used `CUDA_VISIBLE_DEVICES=''`; no GPU device was accessed.

Core A command from the independent copy (with `CHECKOUT` set to its root and `PYTHON` to the project Python 3.12 interpreter):

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONPATH="$CHECKOUT/scripts/core_a/final_recovery_v2:$CHECKOUT/scripts/final_evaluation/v1_historical" "$PYTHON" -m pytest -q scripts/core_a/final_recovery_v2/test_core_a_v2_scorer.py
```

Observed result: **7 passed**.

Core B command:

```bash
env CUDA_VISIBLE_DEVICES='' EE6008_PROJECT_ROOT="$CHECKOUT" PYTHONPATH="$CHECKOUT/scripts/core_b/final_recovery_v2:$CHECKOUT/scripts/final_evaluation/v1_historical" "$PYTHON" -m pytest -q scripts/core_b/final_recovery_v2/test_core_b_v2_contracts.py -k 'not anchor_frame_schedule_stops_at_anchor'
```

On the unmodified upload copy this produced **5 passed, 1 failed, 1 deselected**. The failure is in the synthetic final-prefix test fixture: it placed `final_input_manifest_sha256` under `final_inputs`, while the public builder reads it from `inputs`; population metadata belongs under `final_inputs`. The PR corrects only that mock lock structure. A subsequent local run on the edited copy showed 6 passed and 1 deselected, but no fresh post-update independent run was available for this acceptance record. Therefore this result is not treated as a verified pass for the patched PR.

The deselected `test_anchor_frame_schedule_stops_at_anchor` reads the server-only DEV and FINAL current-profile input manifests. The other final-named tests use temporary synthetic parquet/CSV/NPZ fixtures. No real final labels, prediction payloads, raw data, checkpoints, training, inference or feature extraction were used.

## Remaining release gaps

- The excluded hand-calibrated analyzer, P1/P2 R3D extractor and V7 single-head scorer are study-critical paths without an equivalent proven public entrypoint or repaired copy.
- Exact source-file binding remains UNKNOWN for the nine historical Core A Full-120 Stage1 runs.
- A fresh independent rerun of the corrected Core B test fixture is pending.

These findings prevent `READY_FOR_MERGE`. No PR merge was performed, and no scientific limitation was edited or relaxed.
