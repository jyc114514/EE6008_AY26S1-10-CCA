# Release validation report — 2026-10-22

**Validation time:** 2026-10-22 08:51 UTC
**Base commit:** `f43c07b6055e993986ee1fd6a952261ec6634b71` (`main`)
**Local export branch:** `codex/sync-server-code-results-20261022T082418Z`
**Source snapshot:** `2026-10-22T08:24:18Z`
**Remote `main` recheck:** `f43c07b6055e993986ee1fd6a952261ec6634b71`; task branch did not exist before upload.

## Scope and provenance

The public source tree used for this release has no Git metadata. Experiment-time source commits that cannot be established from source evidence are recorded as `UNKNOWN`. The clone's `.git/index` and file SHA-1 values were used to compute the local change set because the execution environment has no `git` executable. The remote `main` ref was rechecked immediately before upload and remained at the base commit.

A metadata-only walk of the non-Git source tree counted 17,741 regular files with 59,830,330,035 bytes of file size. The tree was not copied wholesale. Per-file source and exported SHA-256 values are recorded in the sanitized code, document, result, and figure indexes. The full source-to-export maps remain in the private audit directory outside this clone.

The release export includes 1,026 selected Python files, 49 dated Core B study summaries, 44 aggregate result tables, and 76 selected figure/plot-data assets. Eighteen code candidates are excluded because of server/attachment path dependence or syntax failures; row-level data, checkpoints, raw media, and full logs remain server-only.

## Automated checks

- 1074 Python files parsed with `ast.parse`: **0 syntax errors**.
- 104 CSV files and 27 JSON files parsed: **0 parse errors**.
- 44 PNG and 17 SVG images verified structurally: **0 image/XML errors**.
- Visual inspection completed for `results/core_a/figures/CORE_A_FINAL_CONDITIONS_V2.png` and `results/core_b/figures/current_profile_v2/CORE_B_CURRENT_PROFILE_V2_DEV_FINAL_SUMMARY.png`.
- Local Markdown links: **129 checked, 0 broken**. Links to omitted source artifacts use `server-only://` identifiers.
- Result index: **23 rows, 0 missing local code/metrics/figure/provenance references**.
- Private export-map SHA-256 checks: code **1026/1026**, study documents **49/49**, aggregate tables **44/44**, figures/plot data **76/76** matched.
- Changed text-file whitespace and final-newline check: **0 trailing-whitespace lines, 0 missing final newlines**. This custom check replaces `git diff --check`, which could not run without the Git executable.
- Privacy scan of the public checkout: **0** local home-path, project-user, or host-label matches; **0** credential-assignment pattern matches; **0** forbidden payload files; **0** secret-named files.
- Three legacy aggregate plot inputs had their local `prediction_path` column removed; remaining rows and metric columns were preserved.
- Thirty-two changed CSV files were normalized from CRLF to LF. Parsed result rows and metric values matched their private pre-normalization backups; affected figure-index hash fields were refreshed.

## CPU tests

- `test_core_a_v2_scorer.py`: **7 passed**.
- `test_core_b_v2_contracts.py`: **4 passed, 3 deselected**. The deselected cases read per-episode input manifests or construct final-split parquet fixtures; those checks were outside this release verification scope.
- The system Python did not include pytest; tests were run in the existing project Python 3.12 environment. No environment was installed or changed.

## Access boundary

No GPU command, CUDA check, training, feature extraction, inference, or benchmark was run. No real final/heldout payload was read for this release. The Core A V2 and Core B V2 aggregates were exported from their existing audited evidence; their distinct final-label access disclosures and limits are documented in the result summaries.

The release currently contains 17 modified and 1,218 added files before this report, the manifest, and the checksum file. The final PR should contain 17 modified and 1,221 added files once those three release records are included. `PUBLIC_FILE_MANIFEST.csv` excludes itself and `CHECKSUMS.sha256`; the checksum file covers all changed release files except itself.
