# Supplementary experiment source, 7 November 2026

These are sanitized exports of the project-authored follow-up scripts. They
cover the fourth-second repeated clip, predicted task conditioning, and five
task-specific forecasting experts with a newly fitted shared control. They build
on the original Core A and Core B implementations already documented in this
repository; they do not replace their authorship or original results.

The publication work changes path configuration, removes private dependency
fallbacks, and normalizes text to UTF-8/LF. The temporal selection, model and
training logic, canonical class mapping, rejection rules, prediction sealing,
metric calculations, bootstrap settings, and pinned source hashes are retained.
The export has new script hashes. Its hashes must not be substituted into old
run receipts or presented as the bytes executed in the historical experiment.
No training, feature extraction, or real-data evaluation was rerun to publish
this source.

## Script index

| Source ID | Public script | Role |
|---|---|---|
| SUPP-CODE-01 | `core_a/run_core_a_followup.py` | Extract, fit and predict the eight-frame fourth-second cycle repeated four times |
| SUPP-CODE-02 | `core_a/check_protocol.py` | Synthetic temporal-selection check; external-source checks only with an explicit source argument |
| SUPP-CODE-03 | `core_b/run_cascade.py` | Original predictors with true or unrestricted predicted task conditioning |
| SUPP-CODE-04 | `core_b/score_cascade.py` | Sealed cascade scoring, coverage and matched/common-cohort comparisons |
| SUPP-CODE-05 | `core_b/test_contracts.py` | Synthetic routing, causal cutoff, aggregation and bootstrap tests |
| SUPP-CODE-06 | `core_b_experts/prepare_protocol.py` | Freeze the expert-study protocol against authorized external inputs |
| SUPP-CODE-07 | `core_b_experts/train_experts.py` | Fit three shared controls and fifteen experts from the original initialization |
| SUPP-CODE-08 | `core_b_experts/evaluate_experts.py` | Freeze eighteen fits, seal both splits, then score and compare |
| SUPP-CODE-09 | `core_b_experts/test_train_experts.py` | Synthetic task filtering, episode weights, update counts and cache locks |
| SUPP-CODE-10 | `baseline_a/decode_original_predictions.py` | Decode original dynamic Core A outputs into canonical task IDs for routing |
| SUPP-CODE-11 | `analysis_a/compare_core_a.py` | Fixed-602-episode recognition comparison and conditional paired intervals |
| SUPP-CODE-12 | `analysis_a/test_compare_core_a.py` | Synthetic metric and paired-bootstrap checks against scikit-learn |

`SOURCE_EXPORT_HASHES.json` records the original and exported byte hashes under
these logical source IDs. The private path-to-source mapping is not published.
`test_export_configuration.py` is a new publication-only test of explicit path
configuration. It is not a historical experiment script.

## Dependencies and CPU checks

Install the repository's `requirements.txt` in a suitable environment. The
supplemental runtime dependencies are listed in `requirements-runtime.txt`;
the default synthetic checks need only `requirements-tests.txt` (NumPy and the
Python standard library). The optional scikit-learn comparison uses
`requirements-metric-reference.txt`. No scripts download
datasets, pretrained weights, or dependencies. A compatible CUDA/PyTorch
environment and the original external V-JEPA source/weights are required for
feature extraction and fitting. CPU-only checks do not validate that runtime.

From the repository root:

```bash
python scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107/core_a/check_protocol.py
python scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107/core_b/test_contracts.py
python scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107/core_b_experts/test_train_experts.py
python scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107/test_export_configuration.py
```

These commands use synthetic arrays and temporary directories. They require no
private inputs, network access, GPU, server account, or configured legacy paths.
Do not pass a source directory to `check_protocol.py` unless you intend to run
its additional real-asset checks.
The expert cache-lock test probes filesystem hardlink support and reports a
skip if the platform does not permit it. The production atomic-lock operation
is unchanged; the numerical expert tests still run.

After installing the optional metric-reference dependency, also run:

```bash
python scripts/supplementary/SUPPLEMENTARY_EXPERIMENTS_V1_20261107/analysis_a/test_compare_core_a.py
```

## Explicit external configuration

Formal execution requires an authorized, read-only mirror of the original
asset tree, provided through `--source-root`. The public repository is organized
for publication and is **not** that runtime tree. Two environment variables
replace the account-specific constants used in the historical scripts:

* `EE6008_LEGACY_SOURCE_PREFIX`: the exact canonical absolute prefix embedded in
  the external historical configuration/manifests. It is a string to strip when
  relocating those recorded paths into `--source-root`; it is not the current
  source directory and has no guessed default. Every formal runner requires it.
* `EE6008_LEGACY_REPLAY_REL`: the historical recovery output directory relative
  to `--source-root`. The cascade, its scorer, expert protocol preparation and
  expert evaluation require it. Absolute paths and `..` components are rejected.

Set these from your authorized asset inventory before launching a process.
For example, with deliberately nonfunctional placeholders:

```bash
export EE6008_LEGACY_SOURCE_PREFIX='/exact/prefix/from/authorized/manifests'
export EE6008_LEGACY_REPLAY_REL='outputs/<asset-owner>/<recovery-directory>'
```

The scripts do not discover accounts or infer these values. Configuration is
checked before production data access. Passing `--help` remains possible without
setting it. Keep generated models, features, prediction bundles and receipts
outside both the read-only source tree and the public repository.

## Required source modules and public counterparts

The following mapping locates the existing public code. The left column is
relative to the external legacy source root; the right is relative to this
repository. It is an index, not an instruction to bypass a hash check.

| External legacy module | Public counterpart |
|---|---|
| `repo/ee6008-humanoid-vjepa/src/ee6008/` | `src/ee6008/` |
| `repo/ee6008-humanoid-vjepa/scripts/run_batch48_ablation_matrix_v3.py` | `scripts/core_a/run_batch48_ablation_matrix_v3.py` |
| `reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/code/v2_model.py` | `scripts/core_b/studies/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z/code/v2_model.py` |
| Same study's `code/locked_source/D3_model.py` | Same public study's `code/locked_source/D3_model.py` |
| Same study's `code/v9_common.py` | Same public study's `code/v9_common.py` |
| Same study's `code/locked_source/d3_eq_contract.py` | Same public study's `code/locked_source/d3_eq_contract.py` |
| Same study's `code/v2_scheduler.py` | Same public study's `code/v2_scheduler.py` |
| Same study's `code/prepare_v2_data.py` | Same public study's `code/prepare_v2_data.py` |

Publication adaptations and line endings can change file hashes even where the
model logic is unchanged. The public counterparts are not asserted to match
the historical pinned byte hashes. Formal replay requires the exact authorized
source versions named by the preserved hash checks. This package does not relax
those checks or claim an unattended full-training reproduction from a clone.

Other required external assets include the frozen split and canonical class
maps, accepted configuration/model matrix, original video/checkpoint inputs for
Core A, and Core B initialization, train cache, sampler orders, normalizer,
metric scales, prefixes, targets and retained baseline models. These are not
embedded here. Source filenames and hashes remain specified by the runners.

## Execution sequence in an authorized environment

Use each script's `--help` for its required directories and arguments. For Core
A, `prepare`, `extract`, `smoke`, `train`, `predict` and `score` are separate
stages. Feature extraction must cover the needed splits; the three fitting
seeds are 20260825, 20260826 and 20260827. Four clip repetitions are not four
fits. `analysis_a/compare_core_a.py` then compares the new final predictions
with the two retained original conditions. Its `--preflight-only` path checks
prediction identities and hashes before scoring, and its output directory must
be new. The full video cache is large and is intentionally external.

For predicted conditioning, run `baseline_a/decode_original_predictions.py`
against the original dynamic classifiers. The routing CSV fields are
`episode_id,predicted_task_id,observation_end_frame`. Pass one file for each
original A seed through repeated `--core-a SEED=CSV` arguments to
`core_b/run_cascade.py`, separately for `dev` and `final_test`, then score each
sealed output using `score_cascade.py`. Do not replace these classifiers with
the fourth-second fits: they define separate interventions.

For experts, prepare the protocol, perform the bounded smoke checks, and fit
each of the three B seeds in shared mode and for each of the five supported
tasks. `evaluate_experts.py` expects models under
`<run-root>/models/{shared,task_<id>}/seed<seed>/` with manifests and completion
receipts. It expects the original dynamic A files under
`<run-root>/inputs/core_a_dynamic_<split>_seed<seed>.csv`, plus the historical
cascade root supplied with `--historical-root` (or the documented sibling
default). `preflight` can validate inputs while models are still pending;
`predict` requires the complete model matrix and seals both splits; `score`
then uses the original metrics on CPU. It does not select a checkpoint or tune
using final targets.

Historical comparisons retain their original population, feature and runtime
qualifications. Expert capacity and update budgets differ from the shared
control. Paired intervals condition on the fitted models; rejected routing
cases require coverage reporting. See the supplementary result documentation
for the scientific interpretation. SSH helpers, queue/watch scripts, credentials,
raw predictions, large assets and vendored dependencies are excluded here.
