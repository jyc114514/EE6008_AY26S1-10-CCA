# EE6008 AY26S1-10-CCA

[English](README.md) | [简体中文](README.zh-CN.md)

**Joint-Embedding Predictive Modelling for Humanoid Everyday Action Anticipation**

This is the public research repository for the NTU EE6008 AY26S1-10-CCA five-student course project. This branch adds a curated, path-redacted Core A evidence release while retaining the original repository layout.

## Objective

The working direction is to evaluate joint-embedding video representations for humanoid everyday action understanding and anticipation. The tentative pipeline is:

V-JEPA / V-JEPA 2 baseline -> selected public metadata and a verified dataset subset -> feature extraction or frozen representation -> current-action recognition baseline -> action anticipation / additional task -> small modification -> ablation and failure analysis.

Core A is the verified development/validation task-recognition line documented in [docs/core_a/](docs/core_a/). It is not an action-anticipation result, and it reports no final-test metrics.

## Repository versus GPU Server

GitHub contains source code, configs, metadata, notes, small final summaries, and reproducibility records. The GPU Server contains raw videos, checkpoints, extracted features, training outputs, logs, and temporary files. data_metadata is not data_raw; results is not outputs.

## Structure

| Path | Purpose |
| --- | --- |
| configs/ | Portable experiment configuration skeletons; paths use environment variables. |
| docs/ | Scope, collaboration rules, server layout, templates, and sanitized project materials. |
| data_metadata/ | Small manifests, label maps, and split definitions only. |
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

Each real run should record the dataset revision, split, Git commit, run ID, random seed, model/checkpoint identifier, restricted artifact ID, metrics, owner, date, and status in experiments/runs.csv. The current branch records only sanitized development/validation evidence.

## Contributors

This is a five-student course project owned by jyc114514. Additional collaborator invitations are pending reliable GitHub usernames from the team; no usernames are guessed in this repository.

## Project materials

The original local folders remain outside this public repository. Public copies here are deliberately curated. Copyright-sensitive paper PDFs, login-state files, private forms, raw dataset files, checkpoints, and large audit artifacts are represented by links or excluded notes instead of being redistributed.

## Status

- Repository scaffold: initialized; Core A evidence release added on this branch.
- Core A task/data contract: G1 duplicate-safe v3, 120 primary classes, development/validation only.
- Full-120 Stage1: 9/9 acceptance in the source evidence package; D-P-L uses unmerged derived inference.
- Final-test access: 0; Stage2 was not launched.
- License: no new license is asserted by this evidence export; third-party/data licenses remain separately governed.
