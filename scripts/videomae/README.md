# VideoMAE development experiments

Contributor: Xueshen Zhang. This directory preserves the development scripts for the frozen VideoMAE recognition baseline, the H30/H60 forecasting matrix and the model handoff. It complements the team's existing V-JEPA release and does not replace the final recognition or five-task forecasting experiments.

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

GPU runs must use the team's lease watcher and resource policy. Historical runners intentionally pin hashes and original server layout. Their dependencies include the team's D3 predictor/evaluator code, input manifests, train-only target cache, feature cache, normalizers and pretrained encoder weights. Inspect each command's arguments before running. This code archive alone is not a self-contained data or model release.

The handoff `prepare` command defaults to restricted checkpoint loading. Its optional `--owner-authorized-pickle` flag is for an explicitly authorized owner converting trusted, hash-verified historical training checkpoints. Do not use it on untrusted files or to bypass an approval policy. Replay loads safetensors and only hashes the preserved original checkpoints.

## Evidence and later acceptance

Public lightweight data and provenance are in [results/videomae](../../results/videomae/README.md). Preserve the historical CPU zero-tolerance failure. The team's later [18 October acceptance summary](../../docs/core_b/studies/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z.md) records exact GPU replay, while retaining unresolved handoff scope items. These are separate checks.

The study code already under `scripts/core_b/studies/VIDEOMAE_CORE_B_HANDOFF_V2_20261013T065841Z/` is an existing team archive. This directory adds the contributor's full development workflow without modifying that archive or its integrity records.
