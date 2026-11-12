"""CPU-only normalized head/gradient check; no real future targets or training run."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    project, cache = args.project_root.resolve(), args.cache_root.resolve()
    code = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
    bench = project / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
    eq = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    norm = json.loads(args.normalizer.read_text())
    if (manifest["status"] != "PASS" or norm["status"] != "LOCKED"
            or norm["cache_manifest_sha256"] != digest(manifest_path)
            or digest(args.normalizer) != "622a2973cfc7d90c9584f1a124f08820091f5f1610a81082d613d71b58eed509"):
        raise ValueError("Cache/normalizer identity mismatch")
    if digest(cache / "ANCHOR_INDEX.npz") != manifest["anchor_index_sha256"]:
        raise ValueError("Index identity mismatch")
    with np.load(cache / "ANCHOR_INDEX.npz", allow_pickle=False) as index:
        selected = np.flatnonzero(index["split_code"] == 0)[:2]
    inputs = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    if digest(inputs) != manifest["input_sha256"]:
        raise ValueError("Input identity mismatch")
    with np.load(inputs, allow_pickle=False) as data:
        tensors = [torch.from_numpy(data[key][selected].astype(np.float32)) for key in
                   ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")]
    grids = []
    for row in selected:
        shard = next(s for s in manifest["shards"] if s["start_row"] <= row < s["end_row"])
        path = Path(shard["array"]).resolve()
        if path.parent != cache / "shards" or digest(path) != shard["sha256"]:
            raise ValueError("Selected shard identity mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        grids.append(values[int(row) - shard["start_row"]].copy())
    grids = np.stack(grids)
    mean, std = np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["safe_std"], dtype=np.float32)
    if mean.shape != (384,) or std.shape != (384,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalizer schema mismatch")
    grids = (grids - mean) / std
    if not np.isfinite(grids).all():
        raise ValueError("Nonfinite normalized features")
    dynamic, static = torch.from_numpy(grids[:, 0]), torch.from_numpy(grids[:, 1])
    sys.path.insert(0, str(code))
    model_module = importlib.import_module("d3_model")
    contract = importlib.import_module("d3_eq_contract")
    registry = json.loads((eq / "protocol/D3_EQ_SEED_REGISTRY.json").read_text())
    results = []
    for seed in (20261001, 20261002, 20261003):
        init_path = bench / f"d3/runs/INIT_S{seed}.pt"
        record = registry["seeds"][str(seed)]
        if digest(init_path) != record["init_state_file_sha256"]:
            raise ValueError("Historical init hash mismatch")
        initial = torch.load(init_path, map_location="cpu", weights_only=True)
        for horizon in (30, 60):
            for mode in ("TS11-D", "TS11-S"):
                model_module.VIDEO_DIM = 768
                model = model_module.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=mode)
                model.load_state_dict(initial, strict=True)
                if model_module.state_hash(model) != record["init_state_hash"]:
                    raise ValueError("Historical state hash mismatch")
                # Replace only the incompatible input projection after strict historical load.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model.video_projection = torch.nn.Linear(384, 128)
                model_module.VIDEO_DIM = 384
                for key, value in model.state_dict().items():
                    if not key.startswith("video_projection.") and not torch.equal(value, initial[key]):
                        raise ValueError(f"Common initialization changed: {key}")
                model.eval()
                state, mask, relative, anchor = tensors
                residual = model(state, mask, relative, dynamic, static, anchor)
                if residual.shape != (2, horizon, 28) or residual.detach().abs().max().item() != 0:
                    raise ValueError("Initial output mismatch")
                # Synthetic targets exercise the loss interface without loading real targets.
                target = anchor[:, None, :].expand(-1, horizon, -1) + 0.1
                loss = contract.per_sample_loss(residual, target, anchor, torch.ones(28), torch.ones(28)).mean()
                loss.backward()
                gradient = model.output.weight.grad
                if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum().item() == 0:
                    raise ValueError("Synthetic output gradient failure")
                results.append({"seed": seed, "horizon": horizon, "mode": mode,
                                "parameters": model_module.parameter_count(model), "status": "PASS"})
    print(json.dumps({"status": "PASS_NORMALIZED_HEAD_CPU", "cells_checked": len(results),
                      "results": results, "common_initialization_preserved": True,
                      "model_source_sha256": digest(code / "d3_model.py"),
                      "loss_source_sha256": digest(code / "d3_eq_contract.py"),
                      "optimizer_steps": 0, "gpu_used": False, "real_targets_loaded": False,
                      "final_test_access": 0, "formal_training_ready": False}, indent=2))


if __name__ == "__main__":
    main()
