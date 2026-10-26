# B1 public dependency repair (2026-10-26)

The B1 release gap at prior PR head `47a994c6154ff5a2a9baab9fb0027dbcc10be3e6` was that the public runner could not resolve its project checkpoint helpers, its checkpoint-fix helper, or the official attentive-pooler source closure. This follow-up publishes the required project files, a narrow adapter for the two helper functions B1 calls, the exact official V-JEPA2 source subset and license, and explicit path guards.

## CPU-only public check

From the repository root, set the public B1 paths and run:

```bash
B1_DIR=scripts/core_b/legacy_archives/code/b1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$B1_DIR:$B1_DIR/public_support/analysis_scripts" \
  python3 -B -m pytest -q -p no:cacheprovider "$B1_DIR/public_support/tests/test_b1_public_dependency_closure.py"
```

This covers public `__file__` resolution, import-time side-effect guards, the CPU autocast adapter, B1 checkpoint-payload integration with a synthetic model/state, strict rejection of altered schema/final-test fields, mocked `weights_only=True` CPU checkpoint dispatch using a temporary sentinel, wrong-root and generic-`src` collision rejection, and a small randomly initialized official AttentivePooler on synthetic CPU tensors. It does not validate B1 training, checkpoint replay, predictions, or historical metrics.

Existing project regression commands remain available in the root README: Core A V2 scorer and Core B V2 contracts. The one Core B schedule test remains deselected because it reads server-only DEV/FINAL input manifests. No B1 acceptance/preflight function is run by the CPU suite.

## Full B1 runtime boundary

The public project dependency closure is complete, but a real B1 run still requires authorized GPU3 execution and server-only artifacts: `CORE_B_FULL_INPUTS_1p0s.npz`, `CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json`, `CORE_B_TOKEN_PATHS_1p0s.json`, per-episode token `.pt` files, `CORE_B_FULL_DATA_MANIFEST.json`, feasibility metadata, and the corresponding run output/config/status/checkpoint/prediction artifacts for acceptance. The acceptance entrypoint reads prediction and checkpoint files only when its preflight/replay functions are called.

When an authorized research run is conducted separately, the public runner’s path arguments must resolve to the public copy:

- `--fix-root scripts/core_b/legacy_archives/code/b1/public_support`
- `--source-root scripts/core_b/legacy_archives/code/b1/public_support/vjepa2`

The B1 training entrypoint still needs `--root`, `--project-root`, `--feasibility-root`, `--run-root`, `--lane`, and `--physical-gpu 3`; these are path/config arguments, not supplied by this package. This repair did not invoke that entrypoint or access any of the artifacts above.

## Versioned integrity scope

This follow-up adds 17 files and modifies 8 files relative to starting head `47a994c6154ff5a2a9baab9fb0027dbcc10be3e6`. `B1_DEPENDENCY_DELTA_MANIFEST_20261026.csv` lists those 25 changed payload files; `B1_DEPENDENCY_DELTA_CHECKSUMS_20261026.sha256` covers the payload plus the delta manifest and does not hash itself. The original release manifest, earlier repair sidecars, and prior acceptance records retain their original snapshot scopes.

## Preserved claim boundary

The old B1 tables remain historical development evidence. Source is recovered and the public interfaces are tested, but exact per-run helper identity remains unproven because the locked run configs do not contain the imported source hashes or root values. Full B1 numerical replay is `NOT_RUN`. See `B1_SOURCE_PROVENANCE.md`, `B1_DEPENDENCY_CLOSURE.csv`, and `B1_CPU_VALIDATION.json` for evidence and limits.
