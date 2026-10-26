# Independent post-release acceptance (2026-10-26)

Audit timestamp: 2026-10-26T03:32:21Z.

**Recommendation: NEEDS_FIXES.** The uploaded public copy passes the file and CPU contract checks below. The legacy B1 result path remains non-replayable from the public tree because three imported modules are unavailable and no exact public replacement was established.

## Reviewed remote snapshot

- Repository: jyc114514/EE6008_AY26S1-10-CCA, PR #2.
- Base: main at f43c07b6055e993986ee1fd6a952261ec6634b71.
- Original publication snapshot: 4988301e113719d7c7796235c32a89b10ceec51a.
- Head independently reviewed: d3c2dd144d498b9f5aa656f8a61f371fc28974de.
- The PR was open and unmerged when checked; no merge or main update was made.
- Retrieval: GitHub codeload ZIP pinned to the head SHA above. Archive SHA-256: 3554bdf656a24b37f384dbc11efaf3509f741b5f4ed6de7136eccd2e5deb7b84; compressed size 12,795,284 bytes; extracted public tree 1,562 files / 29,075,876 bytes; symlinks: 0. The archive was extracted into a new acceptance-only directory; the copy has no path or symlink into the original server work tree.

The first acceptance report for the original 4988301e snapshot remains preserved at POST_RELEASE_INDEPENDENT_ACCEPTANCE_20261023.md. It records the original 1,236-entry public manifest and its 1,237-line checksum file. Those counts describe the original snapshot, not the current PR diff or later repair payload.

## Independent manifest, checksum, and index checks

At the reviewed d3c2dd1 head:

- RELEASE_REPAIR_MANIFEST_20261024.csv: 149 entries.
- RELEASE_REPAIR_CHECKSUMS_20261024.sha256: 149 entries, identical path coverage.
- All 149 files matched manifest byte size, SHA-256, and Git blob SHA; every checksum matched.
- The full public archive contained 1,562 files. The PR API reported 1,383 changed files at the reviewed head. The 149 manifest entries and 149 checksum entries have narrower scopes and are not expected to equal either count.
- CODE_EXPORT_INDEX.csv: 1,042 rows; STUDY_DOCUMENT_INDEX.csv: 49; RESULT_EXPORT_INDEX.csv: 135; FIGURE_EXPORT_INDEX.csv: 105. All indexed paths exist and the recorded export sizes and SHA-256 values match after the correction described below.
- EXPERIMENT_RESULT_INDEX.csv and ENTRYPOINT_RESULT_MAP.csv each contain 38 rows. All 339 parsed public path references in the entrypoint map resolve.

One stale index value was found and corrected in this follow-up. The Core B contract test export row expected 21,326 bytes and SHA-256 7fa20f9e6bee557b0d6931c9963f37d7633ad2181af95656afe334d6fc4f6edf. The published file is 21,338 bytes with SHA-256 8a0ef95591a693fa3407303d7852655f873d65c5e7aa60079798c8c6026cb162. The public test file already contained the lock-fixture correction; this follow-up updates CODE_EXPORT_INDEX.csv to reflect its actual exported bytes and marks that disposition. Test/scoring logic was not edited in this follow-up.

POST_RELEASE_ACCEPTANCE_MANIFEST_20261026.csv and POST_RELEASE_ACCEPTANCE_CHECKSUMS_20261026.sha256 are the matching versioned sidecars for this follow-up. They cover the prior 149 repair-payload paths plus this acceptance report (150 entries). The two sidecar files do not hash themselves. The 20261024 repair sidecars remain historical: they verified the d3c2dd1 snapshot and do not validate the follow-up bytes.

## Excluded code and result mapping

EXCLUDED_CODE_IMPACT.csv contains all 18 candidates using relative paths or stable artifact IDs, without server-absolute paths. The table identifies 16 path/attachment-bound exclusions and two syntax/export-transformation cases:

- EXCL-CODE-003 through EXCL-CODE-016, plus EXCL-CODE-018: 15 clearly marked repaired public copies or corrected export transformations; original historical files remain unchanged.
- EXCL-CODE-001 and EXCL-CODE-002: one-off recovery/server-path wrappers with no current indexed result dependency.
- EXCL-CODE-017: a host lease/resource wrapper; no metric, statistical, plotting, or result-generation dependency was found.

The table records no public imports (18/18), no direct dynamic-load references found by the static public-Python scan (18/18), and no public subprocess references (18/18). The repaired hand-calibrated analyzer, R3D extractor, V7 single-head scorer, and collapse temporal-variation replay remain research entrypoints; their complete replays need study-specific server artifacts. R3D is a source-hash-bound repaired copy with explicit project/study/input/raw/weight arguments and opaque output artifact IDs. Its earlier CPU checks covered Python 3.10/3.12 parsing, import-time inspection, CLI help, and a synthetic artifact-ID contract; no video decode, weight load, extraction, or model forward was run.

The 38-row entrypoint map binds Core A V2 amended scoring/validation/figures; Core B CURRENT_PROFILE_V2 development and final paths/config/summary tables/figures; B1/B2/B3; corrected D3 scheduler; V3; task specialization; and later V-series summaries to concrete public paths where evidence supports that mapping. Core A V2 and Core B CURRENT_PROFILE_V2 map rows record prior content/numeric validation; this pass checked their table/figure references and export hashes without recomputing metrics. The existing public aggregate tables and plots were retained unchanged. No per-sample metrics or bootstrap intervals were recomputed.

Specific unresolved points remain visible:

- B1's public legacy entrypoints import checkpoint_utils.py, fresh_process_batch8_d_a_l_fixed.py, and src.models.attentive_pooler; that module closure is unavailable in the public tree. Similarly named code in other contexts was not treated as an equivalent without provenance. The B1 aggregate table remains historical development evidence; a standalone B1 rerun is not claimed.
- B2 true-pixel preparation/analysis has public entrypoints but needs server-only inputs, predictions, normalization, and receipts. The B2 proxy is explicitly diagnostic, not a substitute for true-pixel B2. B3 is mapped to its accepted 8/8 artifact gate; the single-seed candidate gate did not pass and the evidence remains development-only.
- D3 corrected-scheduler entrypoints, public configuration, aggregate tables, and figures are bound; the scheduler correction does not establish a general causal scheduler or video benefit.
- Nine historical Core A Full-120 Stage1 source-file bindings remain UNKNOWN. V9 cached-Q0 has result tables but its exact producing entrypoint/acquisition remains UNKNOWN. Other rows keep RUN_PROVENANCE_UNKNOWN where the saved evidence does not bind a run to an exact source snapshot.
- Major summary image paths and hashes resolve. Visual rendering of the current images was NOT_RUN in this continuation because the local image-view sandbox failed; existing map checks and table/export bindings were preserved. This record does not claim a fresh visual inspection.

## CPU-only acceptance

Preflight statically inspected the selected test files and imported local modules before execution. Import-time behavior only loads Python modules, defines contract/constants, and inserts local paths; no real result payload, target array, checkpoint, video, model weight, job, or download is accessed. The test functions that read experiment-like data either read the specifically documented server-only input CSVs or build their own temporary synthetic files.

Existing interpreter and packages: Python 3.12.14, pytest 9.1.1, NumPy 2.5.2, scikit-learn 1.9.0, PyArrow 25.0.1, PyTorch 2.12.1+cu130, and OpenCV 5.0.0.93. CUDA_VISIBLE_DEVICES was empty. The commands ran from the root of the independent codeload copy. For portability, PYTHON below denotes that already existing Python 3.12.14 interpreter; the test run did not install dependencies.

    env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts/core_a/final_recovery_v2:scripts/final_evaluation/v1_historical" "$PYTHON" -B -m pytest -q -p no:cacheprovider scripts/core_a/final_recovery_v2/test_core_a_v2_scorer.py

Result: exit 0; 7 passed.

    env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' EE6008_PROJECT_ROOT=. PYTHONPATH="scripts/core_b/final_recovery_v2:scripts/final_evaluation/v1_historical" "$PYTHON" -B -m pytest -q -p no:cacheprovider scripts/core_b/final_recovery_v2/test_core_b_v2_contracts.py -k 'not anchor_frame_schedule_stops_at_anchor'

Result: exit 0; 6 passed, 1 deselected. The three previously deselected final-named tests were individually reviewed and ran using only temporary synthetic fixtures: access-receipt gate, final-prefix causal-history reader, and final-target anchor+1-to-30 reader. The one test still deselected, test_anchor_frame_schedule_stops_at_anchor, opens server-only DEV and FINAL current-profile input CSV manifests absent from the public archive. It reads neither labels nor prediction arrays.

No GPU command/device access, training, inference, feature extraction, checkpoint loading, real final/heldout payload reading, raw-data reading, or new experiment was performed. Full study replay requiring server-only artifacts is NOT_RUN / REQUIRES_SERVER_ARTIFACT.

## Required disclosures and conclusion

The scientific disclosures remain unchanged: Core A prior label exposure and amended same-source analysis; Core B same-source overlap and session independence UNKNOWN; Q0 bridge FAIL_UNRESOLVED; standardized metrics are not physical robot accuracy.

**NEEDS_FIXES** remains the acceptance result because the prioritized B1 historical result path still lacks its exact public dependency closure. The package integrity checks and the safe Core A/Core B CPU contract tests pass. This status does not assert that all experiments must be rerun; it records the unresolved public B1 replay boundary without supplying an unverified replacement.
