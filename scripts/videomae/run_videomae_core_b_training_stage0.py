"""Four isolated train-only GPU Stage0 cells; never starts formal training."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys

from path_config import scratch_name, scratch_root


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
    require(physical in {"0", "1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == physical,
            "Shared GPU watcher required")
    project, cache, output = args.project_root.resolve(), args.cache_root.resolve(), args.output_root.resolve()
    require(output.is_relative_to((scratch_root(project)).resolve()), "Personal output required")
    require(not output.exists(), "Fresh Stage0 output required")
    import numpy as np
    import torch

    torch.set_num_threads(2)
    require(torch.cuda.is_available(), "CUDA unavailable")
    code = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
    hashes = {"d3_model.py": "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589",
              "d3_eq_contract.py": "7359c956bf405092dade2e49e72f236759099330357652469c610d422c770b1c",
              "corrected_scheduler.py": "493c54acfedb7a1471b5b63c6e584846f01df5bf425964efe9b4129c4ef5c0f4"}
    for name, expected in hashes.items():
        require(digest(code / name) == expected, f"Source changed: {name}")
    sys.path.insert(0, str(code))
    models = importlib.import_module("d3_model")
    contract = importlib.import_module("d3_eq_contract")
    scheduling = importlib.import_module("corrected_scheduler")
    eq = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
    inputs_path = v3 / "data/V3_MAIN_INPUTS.npz"
    require(digest(inputs_path) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input hash mismatch")
    with np.load(inputs_path, allow_pickle=False) as data:
        rows = np.flatnonzero(data["split_code"] == 0)
        ids, episodes = data["anchor_id"][rows], data["episode_index"][rows]
        x = {key: data[key][rows].astype(np.float32) for key in
             ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")}
    require(len(rows) == 17757 and len(np.unique(episodes)) == 2712, "Train population mismatch")
    target_path = eq / "d3/train_targets_trainonly.npz"
    require(digest(target_path) == "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1", "Target hash mismatch")
    with np.load(target_path, allow_pickle=False) as targets:
        require(np.array_equal(ids, targets["anchor_id"]) and np.array_equal(episodes, targets["episode_index"]), "Target join mismatch")
        y = targets["targets_native_60"].copy()
        require(y.shape == (17757, 60, 28), "Target shape mismatch")
        require(np.max(np.abs(x["anchor_command_native"] - targets["anchor_command_native"])) <= 1e-6, "Anchor join mismatch")
    target_norm_path = v3 / "data/V3_NORMALIZERS.json"
    require(digest(target_norm_path) == "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c", "Target normalizer mismatch")
    target_norm = json.loads(target_norm_path.read_text())
    sigma, fd = [torch.tensor(target_norm[key], dtype=torch.float32, device="cuda") for key in ("target_sigma", "safe_fd_scale")]
    require(bool(torch.isfinite(sigma).all() and (sigma > 0).all() and torch.isfinite(fd).all() and (fd > 0).all()), "Invalid target scales")
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    norm = json.loads(args.normalizer.read_text())
    require(manifest["status"] == "PASS" and manifest["logical_shape"] == [20079, 2, 16, 4, 384], "Full cache required")
    require(norm["cache_manifest_sha256"] == digest(manifest_path)
            and digest(args.normalizer) == "622a2973cfc7d90c9584f1a124f08820091f5f1610a81082d613d71b58eed509", "Video normalizer mismatch")
    require(digest(cache / "ANCHOR_INDEX.npz") == manifest["anchor_index_sha256"], "Index hash mismatch")
    with np.load(cache / "ANCHOR_INDEX.npz", allow_pickle=False) as index:
        require(np.array_equal(index["cache_index"], np.arange(20079)) and np.array_equal(index["anchor_id"][rows], ids), "Feature join mismatch")
    mean, std = np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["safe_std"], dtype=np.float32)
    require(mean.shape == std.shape == (384,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), "Video scales mismatch")
    verified = set()

    def read_grid(source_rows):
        pairs = []
        for row in source_rows:
            shard = next(s for s in manifest["shards"] if s["start_row"] <= row < s["end_row"])
            path = Path(shard["array"]).resolve()
            require(path.parent == cache / "shards", "Unexpected shard path")
            if path not in verified:
                require(digest(path) == shard["sha256"], "Shard hash mismatch")
                verified.add(path)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            pairs.append(values[int(row) - shard["start_row"]].copy())
        grid = (np.stack(pairs) - mean) / std
        require(bool(np.isfinite(grid).all()), "Nonfinite features")
        return torch.from_numpy(grid[:, 0]).cuda(), torch.from_numpy(grid[:, 1]).cuda()

    init_path = project / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/runs/INIT_S20261001.pt"
    require(digest(init_path) == "62058e33fdddcfa4ec5cbfa46702ea39f4fa004a8fbb11644026919db103b851", "Init file mismatch")
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    weights = contract.episode_equal_weights(episodes)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("TS11-D", "STATE-CTRL"):
        for horizon in (30, 60):
            seed = 20261001
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            models.VIDEO_DIM = 768
            model = models.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=mode)
            model.load_state_dict(initial, strict=True)
            require(models.state_hash(model) == "b4aeb7f8eb9a2c59ca2fbf552c5b6464b36d69abb227d701d963e3c17f0ccf5a", "Init state mismatch")
            if mode == "TS11-D":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model.video_projection = torch.nn.Linear(384, 128)
                models.VIDEO_DIM = 384
            for key, value in model.state_dict().items():
                if mode == "STATE-CTRL" or not key.startswith("video_projection."):
                    require(torch.equal(value, initial[key]), f"Common init changed: {key}")
            model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
            scheduler = scheduling.UpdateDrivenCosineAnnealingLR(optimizer, T_max=11120, eta_min=1e-5)
            scheduler.set_epoch(1)
            torch.cuda.reset_peak_memory_stats()
            order = contract.shuffled_epoch_order(len(rows), seed, 1)[:256]
            losses = []
            for step in range(4):
                positions = order[step * 64:(step + 1) * 64]
                state, mask, relative, anchor = [torch.as_tensor(x[key][positions], device="cuda") for key in
                    ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")]
                if mode == "STATE-CTRL":
                    dynamic = static = torch.zeros(64, 16, 4, 768, device="cuda")
                else:
                    dynamic, static = read_grid(rows[positions])
                target = torch.as_tensor(y[positions, :horizon], device="cuda")
                optimizer.zero_grad(set_to_none=True)
                pred = model(state, mask, relative, dynamic, static, anchor)
                sample = contract.per_sample_loss(pred, target, anchor, sigma, fd)
                loss = (sample * torch.as_tensor(weights[positions], dtype=torch.float32, device="cuda")).sum() / 64
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            require(scheduler.step_calls == 4 and scheduler.legacy_epoch_step_calls == 1, "Scheduler count mismatch")
            model.eval()
            with torch.no_grad():
                reference = model(state, mask, relative, dynamic, static, anchor).cpu()
            cell = output / f"{mode}_H{horizon}_S{seed}"
            cell.mkdir()
            checkpoint = cell / "checkpoint_stage0.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "seed": seed, "horizon": horizon, "mode": mode,
                        "optimizer_steps": 4, "final_test_access": 0}, checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            with torch.no_grad():
                error = (reference - model(state, mask, relative, dynamic, static, anchor).cpu()).abs().max().item()
            require(error == 0, "Checkpoint forward reload mismatch")
            report = {"status": "PASS", "mode": mode, "horizon": horizon, "seed": seed,
                      "optimizer_steps": 4, "scheduler_calls": scheduler.step_calls, "rows_seen": 256,
                      "losses": losses, "reload_max_abs_error": error, "parameters": models.parameter_count(model),
                      "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 1024**2),
                      "checkpoint_sha256": digest(checkpoint)}
            (cell / "STAGE0_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
            results.append(report)
            print(json.dumps(report), flush=True)
            scheduler.close()
            del model, optimizer, scheduler, payload, pred, loss, sample, reference
            torch.cuda.empty_cache()
    report = {"status": "PASS_TRAINING_STAGE0_4_OF_4", "results": results,
              "peak_reserved_mib": max(r["peak_reserved_mib"] for r in results),
              "source_hashes": hashes, "script_sha256": digest(Path(__file__)),
              "cache_manifest_sha256": digest(manifest_path), "normalizer_sha256": digest(args.normalizer),
              "target_cache_sha256": digest(target_path), "physical_gpu": physical,
              "torch": torch.__version__, "numpy": np.__version__,
              "gpu_used": True, "dev_targets_loaded": False, "final_test_access": 0,
              "formal_training_started": False}
    (output / "STAGE0_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
