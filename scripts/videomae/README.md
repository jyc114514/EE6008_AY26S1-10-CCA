# VideoMAE development experiments

Contributor: Xueshen Zhang. This directory contains public path-adapted copies of development scripts for the frozen VideoMAE recognition baseline, the H30/H60 forecasting matrix and the model handoff. It complements the team's existing V-JEPA release and does not replace the final recognition or five-task forecasting experiments.

## Entry points

| Stage | Scripts |
| --- | --- |
| Strict checkpoint loading and Core A features | `verify_videomae_strict_load.py`, `preflight_videomae_core_a_strict.py`, `prepare_videomae_core_a_features_job.py` |
| Core A classifier | `train_videomae_core_a_strict_probes.py` |
| Core B features and scales | `extract_videomae_core_b_cache.py`, `audit_videomae_core_b_cache.py`, `fit_videomae_core_b_normalizer.py` |
| Training gates and formal training | `run_videomae_core_b_training_stage0.py`, `prepare_videomae_formal_job.py`, `train_videomae_core_b_formal.py`, `queue_videomae_core_b_remaining.py` |
| Prediction audit and scoring | `audit_videomae_core_b_predictions.py`, `evaluate_videomae_core_b.py`, `batch_evaluate_videomae_core_b.py` |
| Development comparison | `compare_videomae_vjepa_core_b.py` |
| Handoff and safe-weight replay | `package_videomae_handoff.py`, `close_videomae_core_b_handoff.py` |

Other scripts retain historical smoke tests, static-input checks and diagnostic probes. Their presence does not mean every planned experiment produced accepted results. The earlier non-strict Core A scores are superseded by the strict rerun.

## Running and dependencies

Python 3.12 was used on the server. The historical CPU replay records torch 2.12.1+cu130 and numpy 2.5.2. Feature extraction additionally needs transformers, safetensors, OpenCV and Pillow; scoring needs pyarrow. These records describe the environment used, not a newly validated cross-platform installation lock.

```bash
python scripts/videomae/train_videomae_core_a_strict_probes.py --help
python scripts/videomae/close_videomae_core_b_handoff.py --help
python -m unittest discover -s tests/videomae -p 'test_*.py'
```

GPU runs must use the team's lease watcher and resource policy. Historical runs preserve study-specific code, dependency, cache, and run-directory assumptions. Their dependencies include the team's D3 predictor/evaluator code, input manifests, train-only target cache, feature cache, normalizers and pretrained encoder weights. Inspect each command's arguments before running. This code archive alone is not a self-contained data or model release.

The handoff `prepare` command defaults to restricted checkpoint loading. Its optional `--owner-authorized-pickle` flag is for an explicitly authorized owner converting trusted, hash-verified historical training checkpoints. Do not use it on untrusted files or to bypass an approval policy. Replay loads safetensors and only hashes the preserved original checkpoints.

## Evidence and later acceptance

Public lightweight data and provenance are in [results/videomae](../../results/videomae/README.md). Preserve the historical CPU zero-tolerance failure. The team's later [18 October acceptance summary](../../docs/core_b/studies/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z.md) records exact GPU replay, while retaining unresolved handoff scope items. These are separate checks.

The study code already under `scripts/core_b/studies/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z/` is an existing team archive. This directory adds the contributor's development workflow without modifying that separate archive or its integrity records. Path-portability edits to the public copies do not establish that they are byte-identical to the source files used by historical runs.


## Path configuration

The Core B helpers and job generators use `EE6008_VIDEOMAE_SCRATCH_NAME` as a single directory name. Set it to one ASCII name matching `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`; the resolved scratch directory is under `<project-root>/tmp/<name>`. The helper validates this boundary when the path is used. Newly prepared job JSON records the validated name in its environment, and the queue preserves it for child jobs.

The Task A command-line tools now require explicit paths:

| Script | Required path arguments |
| --- | --- |
| `audit_videomae_task_a.py` | `--manifest`, `--feature-dir`, `--result-dir`, `--archive-source` |
| `compare_videomae_static_dynamic.py` | `--run-dir` |
| `extract_videomae_features_task_a.py` | `--manifest`, `--g1-root`, `--output-dir`; also provide `--config` pointing to an available profile |
| `train_videomae_linear_probe_task_a.py` | `--feature-dir`, `--output-dir` |

The static-run shell wrapper needs `EE6008_PROJECT_ROOT`, `EE6008_VIDEOMAE_SCRATCH_NAME`, `VIDEOMAE_ENV_LAUNCHER`, `VIDEOMAE_MANIFEST`, `VIDEOMAE_G1_ROOT`, and `VIDEOMAE_CONFIG`, plus its existing GPU index argument. A supplied `HF_HOME` must be absolute; when omitted, the wrapper uses `<scratch>/hf_cache`. The project root, scratch, and run directories must already exist.

These path changes do not make the public clone a complete runtime. Authorized video data, manifests, model code, pretrained weights, and the matching Python environment are still required. Do not treat a public code export or its current hash manifest as proof of the exact source that generated historical predictions.

## 路径配置

Core B helper 和任务生成脚本使用 `EE6008_VIDEOMAE_SCRATCH_NAME` 作为单个目录名。它须是匹配 `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` 的 ASCII 名称；实际 scratch 位于 `<项目根目录>/tmp/<名称>` 下，并在使用路径时检查边界。新准备的 job JSON 会记录该值，队列会把它传给子任务。

Core A Task A 命令行工具现在要求显式提供路径参数：

| 脚本 | 必需路径参数 |
| --- | --- |
| `audit_videomae_task_a.py` | `--manifest`、`--feature-dir`、`--result-dir`、`--archive-source` |
| `compare_videomae_static_dynamic.py` | `--run-dir` |
| `extract_videomae_features_task_a.py` | `--manifest`、`--g1-root`、`--output-dir`；并通过 `--config` 指定可用配置 |
| `train_videomae_linear_probe_task_a.py` | `--feature-dir`、`--output-dir` |

静态运行 shell wrapper 需要设置 `EE6008_PROJECT_ROOT`、`EE6008_VIDEOMAE_SCRATCH_NAME`、`VIDEOMAE_ENV_LAUNCHER`、`VIDEOMAE_MANIFEST`、`VIDEOMAE_G1_ROOT` 和 `VIDEOMAE_CONFIG`，并提供原有的 GPU index 参数。若设置 `HF_HOME`，它必须是绝对路径；省略时 wrapper 使用 `<scratch>/hf_cache`。项目根目录、scratch 和 run 目录须预先存在。

这些路径调整没有让公开 clone 成为完整运行环境。仍需获授权的视频数据与 manifest、模型代码、预训练权重和匹配的 Python 环境。当前公开源码及其 hash manifest 不能证明它们与生成历史预测时的源码逐字节相同。
