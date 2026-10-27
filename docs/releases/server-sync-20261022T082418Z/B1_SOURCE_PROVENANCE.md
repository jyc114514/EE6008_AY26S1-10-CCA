# B1 dependency source provenance (2026-10-26)

## Scope

The public dependency closure is for the canonical B1 entrypoints already listed in `ENTRYPOINT_RESULT_MAP.csv`:

- `scripts/core_b/legacy_archives/code/b1/run_core_b_baseline.py`
- `scripts/core_b/legacy_archives/code/b1/accept_core_b_b1.py`

The associated historical result tables remain `experiments/core_b/b1_b2_b3/B1_RESULTS.csv` and `experiments/core_b/conclusion_audit_20261014/KEY_COMPARISONS.csv`. The existing development-only `accepted_inference_7_of_7` status is retained. This repair does not replay or recalculate those results.

## Project-owned checkpoint dependency

The package at `scripts/core_b/legacy_archives/code/b1/public_support/analysis_scripts/checkpoint_utils/` is an exact byte copy from source artifact `B1-SOURCE-RECOVERY-20260904`:

| Public path | SHA-256 | Source SHA-256 | Relationship |
|---|---|---|---|
| `__init__.py` | `bfbc397e6e18063bb469cfaf1cab08ce0fb838a559f838395cbb830efc1907a8` | same | Exact copy |
| `checkpoint_contract.py` | `b62f5b2a2152c19c4e593d92796306d33a9216e4e2728499ff13ccade09bef7b` | same | Exact copy |

The B1 runner calls `capture_rng_state`, `restore_rng_state`, `validate_checkpoint`, `load_checkpoint`, and `save_checkpoint`; the adapter calls `build_checkpoint_payload`; the acceptance entrypoint calls `load_checkpoint`. The package preserves schema version checks, physical batch 8 / accumulation 4 / effective batch 32, `final_test_count == 0`, CPU tensor validation, safe `torch.load(..., weights_only=True)`, and its existing errors. No checkpoint file was opened in this repair.

## Narrow adapter for the checkpoint-fix helper

The public `fresh_process_batch8_d_a_l_fixed.py` is a new, explicitly named B1 adapter under `public_support/analysis_scripts/`. It exposes only the two definitions called by the B1 runner:

- `autocast_context(device)`, called in B1 training/prediction loops;
- `prepare_checkpoint_payload(...)`, called by `checkpoint_payload(...)`.

Both function definitions were extracted verbatim from source artifact `B1-SOURCE-RECOVERY-20260904`, source SHA-256 `00b604181612359539fd55a6804e3a72130ae67feb03ce69fba1cc0011f2a754`. The adapter SHA-256 is recorded in `B1_DEPENDENCY_DELTA_MANIFEST_20261026.csv`. Its only added material is a provenance header and imports for the referenced names. The full historical helper remains unchanged in its archive; its unrelated training, inference, file IO, and device-monitoring code is not exposed through this B1 adapter.

`autocast_context` returns `nullcontext()` for CPU and CUDA autocast only for a CUDA device. `prepare_checkpoint_payload` forwards the B1 contract fields to the restored `checkpoint_utils.build_checkpoint_payload` without changing keys or defaults.

## Third-party attentive pooler

The B1 model imports the official `AttentivePooler` class from the V-JEPA2 subset in `public_support/vjepa2/`. Upstream commit, file hashes, recursive modules, required `timm` version, and MIT license are recorded in [`public_support/vjepa2/UPSTREAM_SOURCE.md`](../../../scripts/core_b/legacy_archives/code/b1/public_support/vjepa2/UPSTREAM_SOURCE.md). The public runner keeps B1’s existing constructor values (`num_queries=1`, `embed_dim=768`, `num_heads=16`, `depth=4`, `use_activation_checkpointing=False`). No mean-pooling, identity, substitute model, or weights were introduced.

The runner accepts `--source-root` only when it resolves to this public B1 V-JEPA2 subset. It checks for collisions in `sys.modules` before importing generic `src.*` names and verifies the resolved `__file__` paths for the pooler and all recursive project modules. The acceptance entrypoint derives its checkpoint package from its own public directory; it does not use the working directory or a private `PYTHONPATH` fallback.

## Historical identity limit

The pre-repair public B1 runner hash `0674dfb1e8590ce0bd49e9c776fd817ce340f993dcda8156cf8dd8bcf4cb84d3` exactly matches the corresponding `before` runner snapshot in `B1-SOURCE-RECOVERY-20260904`. This binds the public entrypoint to an archived source snapshot. The source snapshot also contains the project helper and checkpoint package hashes above, and the B1 recovery queue records public-equivalent `--fix-root` and `--source-root` roles.

The available locked B1 run configs do not record per-run helper hashes or those root values. Therefore the project source has been recovered and its public interfaces are CPU-tested, but exact helper file identity for each historical B1 training run is not proven. `B1_SOURCE_IDENTITY=PARTIAL`; this repair does not claim historical numerical replay.

## Import and workload boundary

Importing both B1 entrypoints and resolving the public dependencies was checked in a fresh process with `CUDA_VISIBLE_DEVICES` empty. The test forbids `torch.load`, `numpy.load`, CUDA availability/seed/synchronization/memory calls, and subprocess launch during import. `run_core_b_baseline.run()` and `accept_core_b_b1.preflight()` were not called. Their real normalized input archives, token maps and token tensors, run manifests, checkpoints, predictions, status receipts, and curves remain server-only artifacts identified in the dependency closure.
