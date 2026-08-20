# GitHub and GPU Server layout

## Recommended server workspace

    /shared/ee6008/
    ├── repo/EE6008_AY26S1-10-CCA/
    ├── data_raw/HumanoidEveryday/{G1,H1}/
    ├── checkpoints/vjepa2/
    ├── features/{g1,h1}/vjepa2/
    ├── outputs/<member>/<run_id>/
    ├── logs/
    ├── tmp/
    └── envs/

<member> is an agreed attribution name, not a substitute for GitHub account names. The five members should not share one mutable outputs/current/ directory.

## Mapping

| GitHub repository | GPU Server |
| --- | --- |
| repo/ | Versioned clone of the Git repository. |
| data_metadata/ | Small manifests, split definitions, and label maps that describe data. |
| (not in GitHub) | data_raw/ contains the actual videos/images and is not committed. |
| results/ | Small final summaries/figures selected from experiments. |
| (not in GitHub) | outputs/ contains checkpoints, predictions, caches, and intermediate files. |

The repository must remain useful without access to private server storage: paths are supplied through environment variables and no local Windows or user-specific server path is hard-coded.
