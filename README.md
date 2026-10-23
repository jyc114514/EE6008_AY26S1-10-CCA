# EE6008 AY26S1-10-CCA

[English](README.md) | [简体中文](README.zh-CN.md)

**Joint-Embedding Predictive Modelling for Humanoid Everyday Action Anticipation**

This is the public research repository for the NTU EE6008 AY26S1-10-CCA course project. It contains curated code and evidence for Core A and Core B, with experiment status, data exposure, provenance, and server-only artifact boundaries recorded separately.

## Objective

The working direction is to evaluate joint-embedding video representations for humanoid everyday action understanding and anticipation. The tentative pipeline is:

V-JEPA / V-JEPA 2 baseline -> selected public metadata and a verified dataset subset -> feature extraction or frozen representation -> current-action recognition baseline -> action anticipation / additional task -> small modification -> ablation and failure analysis.

The historical Core A Full-120 Stage1 line documented in [docs/core_a/](docs/core_a/) is development/validation evidence, not an action-anticipation result. A separate Core A V2 amended analysis scores existing predictions on disclosed same-source final labels; it is not an untouched blind confirmation.

## Repository versus GPU Server

GitHub contains source code, configs, metadata, notes, small final summaries, and reproducibility records. The GPU Server contains raw videos, checkpoints, extracted features, training outputs, logs, and temporary files. data_metadata is not data_raw; results is not outputs.

## Structure

| Path | Purpose |
| --- | --- |
| configs/ | Portable experiment configuration skeletons; paths use environment variables. |
| docs/ | Scope, collaboration rules, server layout, templates, and sanitized project materials. |
| data_metadata/ | Small manifests, label maps, and split definitions only. |
| docs/core_b/ | Dated Core B study summaries, result boundaries, and reproduction notes. |
| experiments/core_b/ | Public aggregate Core B result tables and study indexes. |
| results/core_b/ | Selected aggregate figures and plot data. |
| src/ | Project-authored Core A package and existing scaffold modules. |
| scripts/ | Portable Core A runners, audits, and figure QA entrypoints. |
| experiments/ | Core A catalog, run-level aggregates, effects, and acceptance summaries. |
| results/ | Small, curated Core A summaries, plot data, and figures; not raw training output. |
| tests/ | Focused Core A unit and contract tests. |

## Collaboration workflow

main is the stable branch. Use task branches such as feature/vjepa2-loader, experiment/g1-recognition-baseline, fix/dataset-path, or docs/weekly-update; do not create permanent personal branches.

    git pull origin main
    git switch -c feature/short-description
    git add -- confirmed/path
    git commit -m "type: short description"
    git push -u origin feature/short-description

Open a pull request when the change is ready. Small documentation fixes may be merged directly into main when the team agrees. Never commit raw data, credentials, checkpoints, or unreviewed large artifacts.

## Reproducibility

Each real run should record the dataset revision, split, Git commit, run ID, random seed, model/checkpoint identifier, restricted artifact ID, metrics, owner, date, and status in experiments/runs.csv. The run registry distinguishes development, validation, and final/heldout scoring. Historical access statements are scoped to their own runs.

## Contributors

This is a five-student course project owned by jyc114514. Additional collaborator invitations are pending reliable GitHub usernames from the team; no usernames are guessed in this repository.

## Project materials

The original local folders remain outside this public repository. Public copies here are deliberately curated. Copyright-sensitive paper PDFs, login-state files, private forms, raw dataset files, checkpoints, and large audit artifacts are represented by links or excluded notes instead of being redistributed.

## Status

- Repository scaffold: initialized; Core A evidence release added on this branch.
- Historical Core A Full-120 Stage1: 9/9 accepted source cells, development/validation only; D-P-L uses unmerged derived inference.
- Core A V2: six cells rescored on 602 same-source final labels after a disclosed mapping correction; no new training; not an independent blind confirmation.
- Core B: B1/B2/B3, V3, D3, query/ablation, and V2 CURRENT_PROFILE development/final results are indexed separately.
- Core B V2 final targets were read after the 22-bundle prediction seal; 41 final episodes overlap the Core A indexed-video population, and session independence is unknown.
- Human scientific/team review, video recording, and course submission remain pending in the closeout record.
- License: no new license is asserted by this evidence export; third-party/data licenses remain separately governed.

## Server sync release

The audited code/results export since the prior publication is indexed in [the release summary](docs/releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md), [the experiment result index](docs/releases/server-sync-20261022T082418Z/EXPERIMENT_RESULT_INDEX.csv), and [the Core B study line index](docs/core_b/EXPERIMENT_LINE_INDEX.md). The release distinguishes historical development-only access statements from later disclosed same-source final-label analyses.
