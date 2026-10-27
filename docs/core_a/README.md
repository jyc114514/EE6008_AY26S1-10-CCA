# Core A evidence release

This directory documents the curated Core A release for partial-observation
task recognition on the G1 Humanoid Everyday line. The release is as-of
2026-09-10 and is based on the sealed `CORE_A_UNIFIED_ANALYSIS_20260909T052330Z`
evidence package plus earlier hashed Core A records.

The public scientific task is task-ID recognition from a four-second observed
prefix. It is not an action-anticipation result. This page records the original
2026-09-10 release: its model metrics were development/validation-only and
`core_a_final_test_access=0` for those runs. A later Core A V2 amended analysis
used 602 same-source final labels after prior label exposure was disclosed; see
[the V2 analysis](FINAL_V2_AMENDED_ANALYSIS.md). That analysis is not an untouched
blind confirmation.

Start here:

- [Current conclusions](CURRENT_CONCLUSIONS.md)
- [Experiment rationale and coverage](EXPERIMENTS_AND_RATIONALE.md)
- [Data and evaluation protocol](DATA_AND_EVALUATION_PROTOCOL.md)
- [Reproduction guide](REPRODUCTION_GUIDE.md)
- [Engineering validation](ENGINEERING_VALIDATION.md)
- [Limitations](LIMITATIONS.md)
- [Publication manifest and exclusions](publication/PUBLICATION_MANIFEST.md)

The source package is under `src/ee6008/`; portable launchers and audits are
under `scripts/core_a/`; small aggregate tables and figures are under
`results/core_a/`. Server-only artifact IDs in the tables are stable pointers,
not downloadable files.
