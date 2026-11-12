# EE6008 AY26S1-10-CCA

[English](README.md) | [简体中文](README.zh-CN.md)

**Joint-Embedding Predictive Modelling for Humanoid Everyday Action Anticipation**

This public NTU EE6008 course-project repository contains research code, curated experiment evidence, and reproduction notes for two related lines of work: Core A video-based task recognition and Core B state/action trajectory prediction studies.

> **Snapshot scope:** The historical Core A/Core B summaries below describe the 2026-10-27 public snapshot and the linked dated release records. The separately dated 7 November supplementary studies are indexed below. Training, scoring and label-access statements apply to their named study.

## Research lines

### Core A: partial-observation task recognition

Core A studies G1 task-ID recognition from a four-second observed video prefix. The public package includes a V-JEPA 2.1 feature-extraction path, recognition baselines, probe and adaptation utilities, evaluation code, and curated evidence.

The historical Full-120 Stage1 screen passed its 9/9 artifact/run acceptance checks. Those runs used one seed and development/validation data; 9/9 is not a claim that every scientific hypothesis passed. The D-P-L result is marked `accepted_derived_unmerged`: it was recomputed from the unmerged base-plus-adapter prediction after the merged export path failed.

Core A V2 rescored six existing prediction cells on 602 same-source final labels after prior label exposure and a class-mapping correction had been disclosed. It added no training or model forward pass and is an amended analysis, not an untouched blind confirmation. The results are mixed across metrics and do not establish action anticipation, isolated motion understanding, or robot deployment performance.

Start with [current conclusions](docs/core_a/CURRENT_CONCLUSIONS.md), the [amended V2 analysis](docs/core_a/FINAL_V2_AMENDED_ANALYSIS.md), and the [Core A guide](docs/core_a/README.md).

### Core B: state/action trajectory prediction studies

Core B contains separate dated studies and aggregate results for multiple model, input, and temporal-prediction settings. Its historical CURRENT_PROFILE_V2 path scored development and final splits from previously fitted checkpoints; it added no training. The records show 22 prediction bundles sealed for each split; final targets were read only after the final prediction seal passed.

| Split | Episodes / anchors | V0 position MAE | Persistence position MAE |
| --- | ---: | ---: | ---: |
| Development | 47 / 185 | 0.158397 | 0.222262 |
| Final recovery | 41 / 162 | 0.178383 | 0.238057 |

These are standardized metrics, not physical robot accuracy. The 41 same-source final episodes overlap the Core A indexed-video population, and session independence was not established. This final evaluation is not an independent confirmation. The original Q0 bridge remains `FAIL_UNRESOLVED`; its cause is `UNKNOWN`. See [current results and limitations](docs/core_b/CURRENT_RESULTS_AND_LIMITATIONS.md), the [study index](docs/core_b/EXPERIMENT_LINE_INDEX.md), and the [Core B guide](docs/core_b/README.md).

## Supplementary studies completed 7 November 2026

The [three-study supplement](docs/supplementary/20261107/README.md) records a repeated fourth-second recognition clip, predicted-task conditioning with retained forecasting weights, and five newly trained task experts against fresh shared controls. It includes aggregate results, a public settings summary, source exports and retained verification evidence.

The repeated clip does not match the historical dynamic Top-1 result. Correct final task predictions preserve forecasts on the evaluated subset. Experts reduce final Position MAE by 3.26% but increase Seam error by 13.43% against the fresh shared control. The supplement documents previous final-set exposure, conditional uncertainty and unequal model/update budgets. These studies are separate from the earlier development-only task-specialization waypoint.

## Repository map

| Path | Contents |
| --- | --- |
| `src/ee6008/` | Current project-authored Core A package: model loading, configuration, baselines, probes, and utilities. |
| `src/models/`, `src/training/`, `src/evaluation/` | Older scaffold directories; their READMEs describe conventions, not the current implementation. |
| `scripts/core_a/`, `scripts/core_b/`, `scripts/final_evaluation/`, `scripts/audits/` | Core A/B runners, historical study entrypoints, evaluation and audit tools. Entry points have different data and environment requirements. |
| `configs/`, `data_metadata/` | Public configuration summaries and metadata; these are not raw datasets. |
| `docs/core_a/`, `docs/core_b/` | Study scope, conclusions, limitations, and reproduction boundaries. |
| `experiments/core_a/`, `experiments/core_b/` | Curated aggregate tables and experiment indexes. |
| `results/core_a/`, `results/core_b/` | Selected figures and plot inputs; not full training outputs. |
| `tests/` | Focused unit and contract tests. |

## Set up and public CPU checks

The project environment targets Python 3.11–3.12. Review [environment.yml](environment.yml) and [requirements.txt](requirements.txt); PyTorch and video dependencies may need a build compatible with the authorized runtime. The setup commands below are documented instructions; they were not run as part of this README update.

From the repository root in a Bash-compatible shell:

```bash
conda env create -f environment.yml
conda activate ee6008-cca
```

The public repository does not include raw video, feature tensors, real checkpoints, per-example prediction/target arrays, or execution logs. Full training or replay needs the corresponding authorized data, weights, runtime, and server artifacts.

The published CPU contract checks are:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="scripts/core_a/final_recovery_v2:scripts/final_evaluation/v1_historical" python3 -B -m pytest -q -p no:cacheprovider scripts/core_a/final_recovery_v2/test_core_a_v2_scorer.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' EE6008_PROJECT_ROOT=. PYTHONPATH="scripts/core_b/final_recovery_v2:scripts/final_evaluation/v1_historical" python3 -B -m pytest -q -p no:cacheprovider scripts/core_b/final_recovery_v2/test_core_b_v2_contracts.py -k 'not anchor_frame_schedule_stops_at_anchor'
```

The Core B schedule test is excluded because it reads server-only development/final input manifests. These checks cover public scorer and contract fixtures; they do not run training, full inference, video feature extraction, or historical experiment replay.

The B1 public dependency closure also has a synthetic CPU/import check:

```bash
B1_DIR=scripts/core_b/legacy_archives/code/b1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$B1_DIR:$B1_DIR/public_support/analysis_scripts" \
  python3 -B -m pytest -q -p no:cacheprovider "$B1_DIR/public_support/tests/test_b1_public_dependency_closure.py"
```

It exercises public imports, path guards, and synthetic CPU contracts, including a randomly initialized pooler and a synthetic checkpoint payload. It does not replay historical B1 results or run B1 training. Historical helper source identity remains partially verified; full B1 numerical replay was not run. The B1 runner's physical GPU 3 setting applies to that historical configuration, not to every project workflow. See the [B1 repair record](docs/releases/server-sync-20261022T082418Z/B1_PUBLIC_DEPENDENCY_REPAIR_20261026.md) and [entrypoint/result map](docs/releases/server-sync-20261022T082418Z/ENTRYPOINT_RESULT_MAP.csv).

Core A figures can be regenerated from the curated plot inputs with the commands in [scripts/core_a/README.md](scripts/core_a/README.md). A plot rebuild does not repeat an experiment.

## Collaboration and run records

This is a five-student course project. `main` is the stable branch; changes are developed on task branches and proposed through pull requests. Record each real run in `experiments/runs.csv`, including dataset revision and split, Git commit, run ID, seed, model/checkpoint, metrics, owner, date, and status. Keep development, validation, and final/heldout records distinct; access statements apply to their named run.

## Data, provenance, and licensing

Raw videos and frames, checkpoints, feature caches, full predictions and targets, and run logs remain in the authorized research environment. Public artifact IDs are references, not downloadable data. Read the dated [server-sync release summary](docs/releases/server-sync-20261022T082418Z/RELEASE_SUMMARY.md) and [experiment/result index](docs/releases/server-sync-20261022T082418Z/EXPERIMENT_RESULT_INDEX.csv) for export provenance and historical study boundaries. Earlier development-only access statements apply only to their named studies; later V2 final-label disclosures are documented separately.

The root [SHA256SUMS](SHA256SUMS) and [Core A publication manifest](docs/core_a/publication/PUBLICATION_MANIFEST.csv) preserve the 2026-09-10 Core A release snapshot. Their README hashes match those files at commit [f43c07b](https://github.com/jyc114514/EE6008_AY26S1-10-CCA/tree/f43c07b6055e993986ee1fd6a952261ec6634b71); they are not checksums for the current `main` tree.

No new blanket license is asserted by this evidence export. Project, third-party code, datasets, and model weights may have separate terms; a citation or source link does not grant redistribution rights.
