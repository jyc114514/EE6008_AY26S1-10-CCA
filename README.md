# EE6008 AY26S1-10-CCA

**Joint-Embedding Predictive Modelling for Humanoid Everyday Action Anticipation**

This is the public research repository for the NTU EE6008 AY26S1-10-CCA five-student course project. It is currently an initial, reproducible project scaffold: V-JEPA/V-JEPA 2 adaptation, the exact downstream task, and the first dataset split remain subject to team and supervisor confirmation.

## Objective

The working direction is to evaluate joint-embedding video representations for humanoid everyday action understanding and anticipation. The tentative pipeline is:

V-JEPA / V-JEPA 2 baseline -> selected public metadata and a verified dataset subset -> feature extraction or frozen representation -> current-action recognition baseline -> action anticipation / additional task -> small modification -> ablation and failure analysis.

No baseline result, dataset split, or performance number is claimed by this initial commit.

## Repository versus GPU Server

GitHub contains source code, configs, metadata, notes, small final summaries, and reproducibility records. The GPU Server contains raw videos, checkpoints, extracted features, training outputs, logs, and temporary files. data_metadata is not data_raw; results is not outputs.

## Structure

| Path | Purpose |
| --- | --- |
| configs/ | Portable experiment configuration skeletons; paths use environment variables. |
| docs/ | Scope, collaboration rules, server layout, templates, and sanitized project materials. |
| data_metadata/ | Small manifests, label maps, and split definitions only. |
| src/ | Future dataset, model, feature, training, evaluation, and utility code. |
| scripts/ | Pipeline notes until commands are verified on the assigned server. |
| experiments/ | Run naming rules and the version-controlled experiment ledger. |
| results/ | Small, curated final summaries and figures; not raw training output. |
| tests/ | Future unit and smoke tests. |

## Collaboration workflow

main is the stable branch. Use task branches such as feature/vjepa2-loader, experiment/g1-recognition-baseline, fix/dataset-path, or docs/weekly-update; do not create permanent personal branches.

    git pull origin main
    git switch -c feature/short-description
    git add -- confirmed/path
    git commit -m "type: short description"
    git push -u origin feature/short-description

Open a pull request when the change is ready. Small documentation fixes may be merged directly into main when the team agrees. Never commit raw data, credentials, checkpoints, or unreviewed large artifacts.

## Reproducibility

Each real run should record the dataset revision, split, config path, Git commit, run ID, random seed, model/checkpoint identifier, server, output path, metrics, owner, date, and status in experiments/runs.csv. The ledger starts empty by design.

## Contributors

This is a five-student course project owned by jyc114514. Additional collaborator invitations are pending reliable GitHub usernames from the team; no usernames are guessed in this repository.

## Project materials

The original local folders remain outside this public repository. Public copies here are deliberately curated. Copyright-sensitive paper PDFs, login-state files, private forms, raw dataset files, checkpoints, and large audit artifacts are represented by links or excluded notes instead of being redistributed.

## Status

- Repository scaffold: initialized.
- Public-data and task choice: tentative / pending team confirmation.
- Baseline implementation: not yet claimed.
- Experiments and metrics: none recorded yet.
