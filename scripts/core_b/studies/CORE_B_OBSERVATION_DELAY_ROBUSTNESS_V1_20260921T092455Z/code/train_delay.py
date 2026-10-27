#!/usr/bin/env python3
"""Train one locked clean/robust observation-delay cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from delay_contract import DELAYS, EXP, V3, delay_transform_rows, load_main, robust_delay_index
from delay_model import DelayAwarePredictor, parameter_count, state_hash


CELLS = ("CLEAN-N", "CLEAN-D", "CLEAN-S", "ROBUST-N", "ROBUST-D", "ROBUST-S")
EPOCHS = 40
ACTION_DIM = 28
HAND = 14


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def default_json(value: object) -> object:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return str(value)


def atomic_json(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=default_json) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle: torch.save(value, handle)
    os.replace(part, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class DelayDataset(Dataset):
    def __init__(self, data, normalizer, features, rows, video_mean, video_std, cell, seed, horizon):
        self.data = data; self.normalizer = normalizer; self.features = features; self.rows = rows.astype(np.int64)
        self.video_mean = video_mean.astype(np.float32); self.video_std = video_std.astype(np.float32)
        self.cell = cell; self.seed = seed; self.horizon = horizon; self.epoch = 1
        self.robust = cell.startswith("ROBUST"); self.video_level = cell.split("-")[1]
        self.selected = np.empty((len(data["history_native"]), len(DELAYS), 32), dtype=np.int16)
        self.ages = np.empty((len(data["history_native"]), len(DELAYS), 32), dtype=np.float32)
        # Precompute only indices/ages; native/video arrays remain source-backed.
        for delay_index in range(len(DELAYS)):
            _, age, _, selected = delay_transform_rows(data, normalizer, np.arange(len(data["history_native"]), dtype=np.int64), delay_index)
            self.selected[:, delay_index] = selected.astype(np.int16)
            self.ages[:, delay_index] = age

    def __len__(self): return len(self.rows)

    def delay_index(self, row: int) -> int:
        return robust_delay_index(self.seed, self.epoch, row) if self.robust else 0

    def __getitem__(self, index: int):
        row = int(self.rows[index]); delay_index = self.delay_index(row)
        native = self.data["history_native"][row].astype(np.float32, copy=True)
        native[:, :43] = native[self.selected[row, delay_index], :43]
        state_mean = np.asarray(self.normalizer["state_mean"], dtype=np.float32)
        state_std = np.asarray(self.normalizer["state_safe_std"], dtype=np.float32)
        state = ((native - state_mean[None, :]) / state_std[None, :]).astype(np.float32)
        video_index = 0 if self.video_level == "D" else 1
        video = np.asarray(self.features[row, video_index], dtype=np.float32)
        video = (video - self.video_mean) / self.video_std
        if self.video_level == "N": video = np.zeros_like(video)
        return (torch.from_numpy(state), torch.from_numpy(self.data["history_mask"][row]), torch.from_numpy(self.data["history_relative_delta_standardized"][row]), torch.from_numpy(self.ages[row, delay_index]), torch.from_numpy(video), torch.from_numpy(self.data["anchor_command_native"][row]), torch.from_numpy(self.data["targets_native"][row, :self.horizon]), torch.tensor(int(self.data["episode_index"][row]), dtype=torch.long), torch.tensor(delay_index, dtype=torch.long), torch.tensor(row, dtype=torch.long))


def load_data():
    data, normalizer = load_main()
    features = np.load(V3 / "cache/V3_TEMPORAL_FEATURES.npy", mmap_mode="r")
    video_normalizer = json.loads((V3 / "data/V3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8"))
    stats = {"target_sigma": torch.tensor(normalizer["target_sigma"], dtype=torch.float32), "safe_fd": torch.tensor(normalizer["safe_fd_scale"], dtype=torch.float32), "video_mean": np.asarray(video_normalizer["mean"], dtype=np.float32), "video_std": np.asarray(video_normalizer["safe_std"], dtype=np.float32)}
    return data, normalizer, features, stats


def batch_loss(model, batch, stats, device):
    state, mask, relative, age, video, anchor, target, episode, delay_index, row = [value.to(device) for value in batch]
    residual = model.forward_delay(state, mask, relative, age, video, anchor)
    sigma = stats["target_sigma"].to(device).view(1, 1, ACTION_DIM)
    prediction = anchor.unsqueeze(1) + residual * sigma
    fd_scale = stats["safe_fd"].to(device).view(1, 1, ACTION_DIM)
    pos_error = torch.abs(prediction - target) / sigma
    position = 0.5 * pos_error[:, :, :HAND].mean(dim=(1, 2)) + 0.5 * pos_error[:, :, HAND:].mean(dim=(1, 2))
    fd_error = torch.abs(torch.diff(prediction, dim=1) - torch.diff(target, dim=1)) / fd_scale
    fd = 0.5 * fd_error[:, :, :HAND].mean(dim=(1, 2)) + 0.5 * fd_error[:, :, HAND:].mean(dim=(1, 2))
    return position + 0.1 * fd, position, fd, prediction, episode, delay_index, row


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, steps):
    atomic_torch(path, {"model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "epoch": epoch, "optimizer_steps": steps, "seed": config["seed"], "rng_state_torch": torch.get_rng_state(), "rng_state_numpy": np.random.get_state(), "rng_state_python": random.getstate()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELLS, required=True); parser.add_argument("--horizon", type=int, choices=(60,), required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--stage0", action="store_true"); parser.add_argument("--max-optimizer-steps", type=int); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--accumulation", type=int, default=1); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(); run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()): raise FileExistsError(run_root)
    if args.stage0 and args.max_optimizer_steps != 4: raise ValueError("Stage0 requires 4 steps")
    if not args.stage0 and args.max_optimizer_steps is not None: raise ValueError("max steps only Stage0")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if device.type == "cuda": torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed); torch.set_num_threads(1); run_root.mkdir(parents=True)
    data, normalizer, features, stats = load_data(); train_rows = np.flatnonzero(data["split_code"] == 0)
    if args.stage0: train_rows = train_rows[: min(len(train_rows), args.batch_size * args.accumulation * args.max_optimizer_steps)]
    episode_counts = {int(e): int(np.sum(data["episode_index"][train_rows] == e)) for e in np.unique(data["episode_index"][train_rows])}; episode_count = len(episode_counts); total_windows = len(train_rows)
    dataset = DelayDataset(data, normalizer, features, train_rows, stats["video_mean"], stats["video_std"], args.cell, args.seed, args.horizon)
    model = DelayAwarePredictor(horizon=args.horizon, seed=args.seed).to(device); initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    steps_per_epoch = math.ceil(total_windows / args.batch_size / args.accumulation); expected = args.max_optimizer_steps if args.stage0 else EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(expected, 1), eta_min=1e-5)
    config = {"status": "LOCKED", "protocol_id": "CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1", "cell": args.cell, "seed": args.seed, "horizon_steps": args.horizon, "epochs": 0 if args.stage0 else EPOCHS, "batch_size": args.batch_size, "accumulation": args.accumulation, "effective_batch": args.batch_size * args.accumulation, "steps_per_epoch": steps_per_epoch, "expected_optimizer_steps": expected, "train_windows": total_windows, "train_episodes": episode_count, "parameter_count": parameter_count(model), "initial_model_hash": initial_hash, "v3_input_sha256": digest(V3 / "data/V3_MAIN_INPUTS.npz"), "v3_cache_sha256": digest(V3 / "cache/V3_TEMPORAL_FEATURES.npy"), "normalizer_sha256": digest(V3 / "data/V3_NORMALIZERS.json"), "video_normalizer_sha256": digest(V3 / "data/V3_VIDEO_NORMALIZER.json"), "delay_contract_sha256": digest(EXP / "DELAY_TRANSFORM_CONTRACT.json"), "schedule_contract_sha256": digest(EXP / "TRAIN_DELAY_SCHEDULE.json"), "model_code_sha256": digest(EXP / "code/delay_model.py"), "final_test_access": 0, "encoder_updated": False, "new_video_extraction_started": False}
    atomic_json(run_root / "RUN_CONFIG_LOCKED.json", config); atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "optimizer_steps": 0, "final_test_access": 0})
    fixture = next(iter(DataLoader(dataset, batch_size=min(args.batch_size, 8), shuffle=False, num_workers=0))); model.eval()
    with torch.inference_mode(): _, _, _, fixture_pred, _, _, _ = batch_loss(model, fixture, stats, device)
    persistence_error = float(torch.max(torch.abs(fixture_pred - fixture[5].to(device).unsqueeze(1))))
    if persistence_error > 1e-6: raise RuntimeError(f"zero residual not Persistence: {persistence_error}")
    steps = 0; curves = []
    try:
        for epoch in range(1, EPOCHS + 1 if not args.stage0 else 100000):
            if steps >= expected: break
            dataset.epoch = epoch; model.train(); generator = torch.Generator().manual_seed(args.seed + epoch); loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0); optimizer.zero_grad(set_to_none=True); micro = 0; sum_loss = sum_pos = sum_fd = sum_weight = 0.0; delay_counts = np.zeros(len(DELAYS), dtype=np.int64)
            for batch in loader:
                if steps >= expected: break
                total, pos, fd, _pred, episode, delay_index, _row = batch_loss(model, batch, stats, device)
                episode_np = episode.detach().cpu().numpy(); delay_np = delay_index.detach().cpu().numpy(); weights_np = np.asarray([total_windows / (episode_count * episode_counts[int(e)]) for e in episode_np], dtype=np.float32); weights = torch.from_numpy(weights_np).to(device); loss = (total * weights).sum() / weights.sum(); loss.backward(); micro += 1; delay_counts += np.bincount(delay_np, minlength=len(DELAYS)); sum_loss += float(np.sum(total.detach().cpu().numpy() * weights_np)); sum_pos += float(np.sum(pos.detach().cpu().numpy() * weights_np)); sum_fd += float(np.sum(fd.detach().cpu().numpy() * weights_np)); sum_weight += float(weights_np.sum())
                if micro == args.accumulation or steps + 1 >= expected:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); steps += 1; micro = 0
            curves.append({"epoch": epoch, "optimizer_steps": steps, "loss_weighted": sum_loss / max(sum_weight, 1e-12), "position_weighted": sum_pos / max(sum_weight, 1e-12), "fd_weighted": sum_fd / max(sum_weight, 1e-12), "learning_rate": scheduler.get_last_lr()[0], "delay_counts": delay_counts.tolist()})
            atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "last_completed_epoch": epoch, "optimizer_steps": steps, "final_test_access": 0})
            if not args.stage0 and epoch in (10, 20, 30): save_checkpoint(run_root / f"checkpoint_epoch{epoch}.pt", model, optimizer, scheduler, config, epoch, steps)
        if steps != expected: raise RuntimeError(f"optimizer steps {steps} != {expected}")
        checkpoint_name = "checkpoint_stage0.pt" if args.stage0 else "checkpoint_last.pt"; model.eval(); save_checkpoint(run_root / checkpoint_name, model, optimizer, scheduler, config, epoch, steps)
        with torch.inference_mode(): _, _, _, pred, _, _, _ = batch_loss(model, fixture, stats, device)
        reload = DelayAwarePredictor(horizon=args.horizon, seed=args.seed).to(device); payload = torch.load(run_root / checkpoint_name, map_location="cpu", weights_only=False); reload.load_state_dict(payload["model_state_dict"], strict=True); reload.eval()
        with torch.inference_mode(): _, _, _, replay, _, _, _ = batch_loss(reload, fixture, stats, device)
        reload_diff = float(torch.max(torch.abs(pred - replay)).detach().cpu())
        if reload_diff > 1e-6: raise RuntimeError(f"reload mismatch {reload_diff}")
        peak_allocated = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if device.type == "cuda" else 0; peak_reserved = int(torch.cuda.max_memory_reserved() // (1024 * 1024)) if device.type == "cuda" else 0
        atomic_json(run_root / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves, "final_test_access": 0}); atomic_json(run_root / ("STAGE0_REPORT.json" if args.stage0 else "RUN_COMPLETE.json"), {"status": "PASS", "run_id": run_root.name, "cell": args.cell, "seed": args.seed, "horizon": args.horizon, "epoch": epoch, "optimizer_steps": steps, "checkpoint": checkpoint_name, "parameter_count": parameter_count(reload), "initial_persistence_max_abs": persistence_error, "checkpoint_reload_prediction_max_abs": reload_diff, "peak_memory_allocated_mib": peak_allocated, "peak_memory_reserved_mib": peak_reserved, "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"), "final_test_access": 0, "encoder_updated": False, "new_video_extraction_started": False})
        atomic_json(run_root / "RUN_STATUS.json", {"status": "complete", "optimizer_steps": steps, "checkpoint": checkpoint_name, "final_test_access": 0}); print(json.dumps({"status": "PASS", "cell": args.cell, "seed": args.seed, "horizon": args.horizon, "optimizer_steps": steps, "peak_memory_reserved_mib": peak_reserved}, sort_keys=True)); return 0
    except Exception as exc:
        atomic_json(run_root / "RUN_STATUS.json", {"status": "failed", "optimizer_steps": steps, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0}); raise


if __name__ == "__main__": raise SystemExit(main())
