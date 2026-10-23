> Archived study summary for `JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z` (source snapshot `20260913`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# JY-02 候选冻结与 held-out 协议

Freeze root: `server-only://source-code/reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z`

Status: protocol preparation complete. Candidate: `JY02-F-R-D`. Fresh seeds: `20260921,20260922,20260923`. Stage2 source checksum: `284/284 OK`.

The package freezes all 12 Stage2 checkpoints by absolute path, size and SHA-256, plus code/config/normalizer/split/metric provenance. Large checkpoints and prediction/target arrays are not copied. The physical v3 final population exists as a label-free index, but clean blind availability is `unknown` because historical feature metadata contained label-bearing fields. Route A is recommended conditionally; Route B/C are recorded fallbacks.

This package did not read final targets/labels/predictions, did not run final inference, did not train or tune, and did not modify old artifacts. Read `EVALUATION_ROUTE_DECISION.md`, `CLAIM_HIERARCHY.md`, `PROFESSOR_DECISION_REQUIRED.zh-CN.md` and the pending execution Prompt before any future action.
