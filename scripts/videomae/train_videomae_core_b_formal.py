"""Isolated corrected-scheduler training; exports target-blind predictions."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time


from path_config import scratch_root
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage0-report", type=Path, required=True)
    parser.add_argument("--mode", choices=("TS11-D", "STATE-CTRL"), required=True)
    parser.add_argument("--horizon", type=int, choices=(30, 60), required=True)
    parser.add_argument("--seed", type=int, choices=(20261001, 20261002, 20261003), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-check", action="store_true",
                        help="Two-update save/restore equivalence test only; no formal result")
    args = parser.parse_args()
    physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
    require(physical in {"0", "1", "2", "3"} and os.environ.get("CUDA_VISIBLE_DEVICES") == physical, "Shared watcher required")
    project, cache_run, output = args.project_root.resolve(), args.cache_run.resolve(), args.output_root.resolve()
    require(output.is_relative_to(scratch_root(project)), "Output is outside the configured scratch root")
    require(not output.exists() or args.resume, "Existing output requires --resume")
    require(not args.resume or output.is_dir(), "Resume directory missing")
    require(not (args.resume and args.resume_check), "Resume-check requires a fresh run")
    import numpy as np
    import torch

    torch.set_num_threads(2)
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    code = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
    expected = {"d3_model.py": "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589",
                "d3_eq_contract.py": "7359c956bf405092dade2e49e72f236759099330357652469c610d422c770b1c",
                "corrected_scheduler.py": "493c54acfedb7a1471b5b63c6e584846f01df5bf425964efe9b4129c4ef5c0f4"}
    for name, sha in expected.items():
        require(digest(code / name) == sha, f"Source mismatch: {name}")
    sys.path.insert(0, str(code))
    models, contract, sched = [importlib.import_module(name) for name in ("d3_model", "d3_eq_contract", "corrected_scheduler")]
    cache = cache_run / "cache"
    manifest_path = cache / "CACHE_MANIFEST.json"
    norm_path = cache_run / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json"
    manifest, norm = [json.loads(p.read_text()) for p in (manifest_path, norm_path)]
    stage0 = json.loads(args.stage0_report.read_text())
    require(stage0["status"] == "PASS_TRAINING_STAGE0_4_OF_4" and len(stage0["results"]) == 4, "Stage0 gate failed")
    require({(r["mode"], r["horizon"]) for r in stage0["results"]} == {(m, h) for m in ("TS11-D", "STATE-CTRL") for h in (30, 60)}, "Stage0 cells mismatch")
    require(all(r["status"] == "PASS" and r["optimizer_steps"] == r["scheduler_calls"] == 4 and r["reload_max_abs_error"] == 0 for r in stage0["results"]), "Stage0 checks failed")
    require(stage0["script_sha256"] == "d0c60684cbb79a03b78eddf5df514b044884a10b679f4d062726e1137ca036c9", "Unexpected Stage0 runner")
    require(stage0["source_hashes"] == expected and stage0["cache_manifest_sha256"] == digest(manifest_path), "Stage0 source mismatch")
    require(manifest["status"] == "PASS" and manifest["logical_shape"] == [20079, 2, 16, 4, 384], "Cache mismatch")
    require(digest(norm_path) == stage0["normalizer_sha256"] == "622a2973cfc7d90c9584f1a124f08820091f5f1610a81082d613d71b58eed509", "Video normalizer mismatch")
    require(norm["cache_manifest_sha256"] == digest(manifest_path), "Normalizer binding mismatch")
    v3 = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
    input_path = v3 / "data/V3_MAIN_INPUTS.npz"
    require(digest(input_path) == manifest["input_sha256"] == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input mismatch")
    with np.load(input_path, allow_pickle=False) as data:
        metadata = {k: data[k].copy() for k in ("anchor_id", "episode_index", "split_code")}
        x = {k: data[k].astype(np.float32) for k in ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")}
    require(set(metadata["split_code"].tolist()) == {0, 1}, "Unexpected split")
    train_rows, dev_rows = [np.flatnonzero(metadata["split_code"] == s) for s in (0, 1)]
    require(len(train_rows) == 17757 and len(dev_rows) == 2322, "Population mismatch")
    require(digest(cache / "ANCHOR_INDEX.npz") == manifest["anchor_index_sha256"], "Index mismatch")
    with np.load(cache / "ANCHOR_INDEX.npz", allow_pickle=False) as index:
        require(np.array_equal(index["cache_index"], np.arange(20079)), "Cache row order mismatch")
        for key in metadata:
            require(np.array_equal(index[key], metadata[key]), f"Index mismatch: {key}")
    eq = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    target_path = eq / "d3/train_targets_trainonly.npz"
    require(digest(target_path) == stage0["target_cache_sha256"] == "e1fd959d752c23f4e0112399d1fb443e92aa1aa112f442320600f33349f32ca1", "Train target mismatch")
    with np.load(target_path, allow_pickle=False) as data:
        require(np.array_equal(data["anchor_id"], metadata["anchor_id"][train_rows]) and np.array_equal(data["episode_index"], metadata["episode_index"][train_rows]), "Train target join mismatch")
        require(np.max(np.abs(data["anchor_command_native"] - x["anchor_command_native"][train_rows])) <= 1e-6, "Anchor mismatch")
        targets = data["targets_native_60"].copy()
    require(targets.shape == (17757, 60, 28) and np.isfinite(targets).all(), "Train targets invalid")
    target_norm_path = v3 / "data/V3_NORMALIZERS.json"
    require(digest(target_norm_path) == "343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c", "Target scales mismatch")
    target_norm = json.loads(target_norm_path.read_text())
    sigma, fd = [torch.tensor(target_norm[k], dtype=torch.float32, device="cuda") for k in ("target_sigma", "safe_fd_scale")]
    mean, std = [np.asarray(norm[k], dtype=np.float32) for k in ("mean", "safe_std")]
    require(mean.shape == std.shape == (384,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), "Invalid feature scales")
    require(bool(torch.isfinite(sigma).all() and (sigma > 0).all() and torch.isfinite(fd).all() and (fd > 0).all()), "Invalid target scales")
    for shard in manifest["shards"]:
        path = Path(shard["array"]).resolve()
        require(path.parent == cache / "shards" and digest(path) == shard["sha256"], "Shard integrity mismatch")
    registry = json.loads((eq / "protocol/D3_EQ_SEED_REGISTRY.json").read_text())["seeds"][str(args.seed)]
    init_path = project / f"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/runs/INIT_S{args.seed}.pt"
    require(digest(init_path) == registry["init_state_file_sha256"], "Historical init mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    models.VIDEO_DIM = 768
    model = models.D3TimeSpacePredictor(horizon=args.horizon, seed=args.seed, video_mode=args.mode)
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    model.load_state_dict(initial, strict=True)
    require(models.state_hash(model) == registry["init_state_hash"], "Initial state mismatch")
    if args.mode == "TS11-D":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            model.video_projection = torch.nn.Linear(384, 128)
        models.VIDEO_DIM = 384
    for key, value in model.state_dict().items():
        if args.mode == "STATE-CTRL" or not key.startswith("video_projection."):
            require(torch.equal(value, initial[key]), f"Common initialization changed: {key}")
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = sched.UpdateDrivenCosineAnnealingLR(optimizer, T_max=11120, eta_min=1e-5)
    weights = contract.episode_equal_weights(metadata["episode_index"][train_rows])
    identity = {"seed": args.seed, "mode": args.mode, "horizon": args.horizon,
                "resume_check": args.resume_check, "source_hashes": expected, "runner_sha256": digest(Path(__file__)),
                "cache_manifest_sha256": digest(manifest_path), "normalizer_sha256": digest(norm_path),
                "target_cache_sha256": digest(target_path), "stage0_report_sha256": digest(args.stage0_report),
                "target_normalizer_sha256": digest(target_norm_path), "init_sha256": digest(init_path),
                "epochs": 40, "batch": 64, "updates": 11120, "checkpoint_rule": "last40",
                "final_test_access": 0, "dev_targets_loaded": False}
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        require(json.loads((output / "CONFIG.json").read_text()) == identity, "Resume identity mismatch")
    else:
        atomic_json(output / "CONFIG.json", identity)
    atomic_json(output / "ENVIRONMENT.json", {"hostname": socket.gethostname(), "physical_gpu": physical,
                "torch": torch.__version__, "numpy": np.__version__, "argv": sys.argv,
                "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID")})

    def rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

    def payload(epoch):
        return {"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": rng(), "epoch": epoch}

    def save_checkpoint(path, epoch):
        temporary = path.with_suffix(".pt.part")
        torch.save(payload(epoch), temporary)
        os.replace(temporary, path)

    def restore(path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(saved["identity"] == identity, "Checkpoint identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["cpu"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        return saved["epoch"]

    def batch(source_rows):
        state, mask, relative, anchor = [torch.as_tensor(x[k][source_rows], device="cuda") for k in
                                       ("history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")]
        if args.mode == "STATE-CTRL":
            dynamic = static = torch.zeros(len(source_rows), 16, 4, 768, device="cuda")
        else:
            grids = []
            for row in source_rows:
                shard = next(s for s in manifest["shards"] if s["start_row"] <= row < s["end_row"])
                values = np.load(shard["array"], mmap_mode="r", allow_pickle=False)
                grids.append(values[int(row) - shard["start_row"]].copy())
            grids = (np.stack(grids) - mean) / std
            require(bool(np.isfinite(grids).all()), "Nonfinite normalized features")
            dynamic, static = torch.from_numpy(grids[:, 0]).cuda(), torch.from_numpy(grids[:, 1]).cuda()
        return state, mask, relative, dynamic, static, anchor

    def update(positions):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = batch(train_rows[positions])
        pred = model(*values)
        sample = contract.per_sample_loss(pred, torch.as_tensor(targets[positions, :args.horizon], device="cuda"), values[-1], sigma, fd)
        loss = (sample * torch.as_tensor(weights[positions], dtype=torch.float32, device="cuda")).sum() / len(positions)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.item()

    torch.cuda.reset_peak_memory_stats()
    checkpoint = output / "checkpoint_last.pt"
    if args.resume_check:
        scheduler.set_epoch(1)
        order = contract.shuffled_epoch_order(17757, args.seed, 1)
        update(order[:64])
        save_checkpoint(checkpoint, 0)
        loss_expected = update(order[64:128])
        expected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expected_lr = scheduler.get_last_lr()[0]
        restore(checkpoint)
        loss_actual = update(order[64:128])
        require(loss_actual == loss_expected and expected_lr == scheduler.get_last_lr()[0] and scheduler.step_calls == 2, "Resume loss/LR mismatch")
        require(all(torch.equal(v.cpu(), expected_state[k]) for k, v in model.state_dict().items()), "Resume parameter mismatch")
        atomic_json(output / "RESUME_CHECK.json", {"status": "PASS_EXACT_RESUME", "loss": loss_actual,
                    "peak_reserved_mib": (torch.cuda.max_memory_reserved() + 1024**2 - 1) // 1024**2,
                    "runner_sha256": identity["runner_sha256"], "formal_training_started": False, "final_test_access": 0})
        print((output / "RESUME_CHECK.json").read_text())
        return
    start_epoch = restore(checkpoint) + 1 if args.resume and checkpoint.exists() else 1
    require(scheduler.step_calls == (start_epoch - 1) * 278, "Resume update count mismatch")
    started = time.monotonic()
    for epoch in range(start_epoch, 41):
        scheduler.set_epoch(epoch)
        order = contract.shuffled_epoch_order(17757, args.seed, epoch)
        losses = [update(order[start:start + 64]) for start in range(0, len(order), 64)]
        scheduler.step()
        require(scheduler.step_calls == epoch * 278, "Optimizer/scheduler count mismatch")
        save_checkpoint(checkpoint, epoch)
        sched.write_lr_trace(output / "LR_TRACE.csv", scheduler.update_records)
        report = {"status": "running", "epoch": epoch, "updates": scheduler.step_calls,
                  "lr": scheduler.get_last_lr()[0], "mean_update_loss": float(np.mean(losses)),
                  "elapsed_seconds": time.monotonic() - started}
        atomic_json(output / "RUN_STATUS.json", report)
        print(json.dumps(report), flush=True)
    require(scheduler.step_calls == 11120 and abs(scheduler.get_last_lr()[0] - 1e-5) < 2e-11, "Final scheduler mismatch")
    model.eval()
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        predictions = np.empty((len(rows), args.horizon, 28), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                values = batch(rows[start:start + 64])
                predictions[start:start + 64] = (values[-1][:, None, :] + sigma[None, None, :] * model(*values)).cpu().numpy()
        require(bool(np.isfinite(predictions).all()), "Nonfinite predictions")
        path = output / f"predictions_{name}.npz"
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, anchor_id=metadata["anchor_id"][rows], episode_index=metadata["episode_index"][rows], predictions_native=predictions)
        os.replace(temporary, path)
    complete = {"status": "PASS_TRAIN_AND_EXPORT", "epochs": 40, "optimizer_steps": 11120,
                "checkpoint_sha256": digest(checkpoint), "train_prediction_sha256": digest(output / "predictions_train.npz"),
                "dev_prediction_sha256": digest(output / "predictions_dev.npz"),
                "peak_reserved_mib": (torch.cuda.max_memory_reserved() + 1024**2 - 1) // 1024**2,
                "metrics_evaluated": False, "dev_targets_loaded": False, "final_test_access": 0}
    atomic_json(output / "RUN_STATUS.json", complete)
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
