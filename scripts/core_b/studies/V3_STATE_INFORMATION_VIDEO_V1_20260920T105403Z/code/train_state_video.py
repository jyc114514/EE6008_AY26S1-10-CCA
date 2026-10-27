#!/usr/bin/env python3
"""Train one locked state-information x video development cell."""

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

from state_video_model import StateVideoPredictor, parameter_count, state_hash


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))
EPOCHS = 40
ACTION_DIM = 28
HAND = 14


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def atomic_json(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        torch.save(value, handle)
    os.replace(part, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StateVideoDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], features: np.ndarray, rows: np.ndarray, video_mean: torch.Tensor, video_std: torch.Tensor, cell: str, horizon: int):
        self.data = data
        self.features = features
        self.rows = rows.astype(np.int64)
        self.video_mean = video_mean.cpu().numpy()
        self.video_std = video_std.cpu().numpy()
        self.state_level, self.video_level = cell.split("-")
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        video_index = 0 if self.video_level == "D" else 1
        video = np.asarray(self.features[row, video_index], dtype=np.float32)
        video = (video - self.video_mean) / self.video_std
        if self.video_level == "N":
            video = np.zeros_like(video)
        return (
            torch.from_numpy(self.data["history_standardized"][row]),
            torch.from_numpy(self.data["history_mask"][row]),
            torch.from_numpy(self.data["history_relative_delta_standardized"][row]),
            torch.from_numpy(video),
            torch.from_numpy(self.data["anchor_command_native"][row]),
            torch.from_numpy(self.data["targets_native"][row, : self.horizon]),
            torch.tensor(int(self.data["episode_index"][row]), dtype=torch.long),
        )


def load_data() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, torch.Tensor]]:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in bundle.files}
    features = np.load(V3 / "cache/V3_TEMPORAL_FEATURES.npy", mmap_mode="r")
    normalizer = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    video_normalizer = json.loads((V3 / "data/V3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8"))
    stats = {
        "target_sigma": torch.tensor(normalizer["target_sigma"], dtype=torch.float32),
        "safe_fd": torch.tensor(normalizer["safe_fd_scale"], dtype=torch.float32),
        "video_mean": torch.tensor(video_normalizer["mean"], dtype=torch.float32),
        "video_std": torch.tensor(video_normalizer["safe_std"], dtype=torch.float32),
    }
    return data, features, stats


def batch_loss(model: StateVideoPredictor, batch, stats: dict[str, torch.Tensor], cell: str, device: torch.device):
    state, mask, relative_delta, video, anchor, target, episode = [value.to(device) for value in batch]
    state_level, video_level = cell.split("-")
    residual = model.forward_locked(state, mask, relative_delta, video, anchor, state_level=state_level, video_level=video_level)
    target_sigma = stats["target_sigma"].to(device).view(1, 1, ACTION_DIM)
    prediction = anchor.unsqueeze(1) + residual * target_sigma
    fd_scale = stats["safe_fd"].to(device).view(1, 1, ACTION_DIM)
    position_error = torch.abs(prediction - target) / target_sigma
    position = 0.5 * position_error[:, :, :HAND].mean(dim=(1, 2)) + 0.5 * position_error[:, :, HAND:].mean(dim=(1, 2))
    fd_error = torch.abs(torch.diff(prediction, dim=1) - torch.diff(target, dim=1)) / fd_scale
    fd = 0.5 * fd_error[:, :, :HAND].mean(dim=(1, 2)) + 0.5 * fd_error[:, :, HAND:].mean(dim=(1, 2))
    return position + 0.1 * fd, position, fd, prediction, episode


def save_checkpoint(path: Path, model, optimizer, scheduler, config: dict[str, object], epoch: int, steps: int) -> None:
    atomic_torch(path, {"model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "epoch": epoch, "optimizer_steps": steps, "seed": config["seed"], "rng_state_torch": torch.get_rng_state(), "rng_state_numpy": np.random.get_state(), "rng_state_python": random.getstate()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--horizon", type=int, choices=(30, 60), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(run_root)
    if args.stage0 and args.max_optimizer_steps != 4:
        raise ValueError("Stage0 requires exactly four optimizer steps")
    if not args.stage0 and args.max_optimizer_steps is not None:
        raise ValueError("max optimizer steps is only valid for Stage0")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    torch.set_num_threads(1)
    run_root.mkdir(parents=True)
    data, features, stats = load_data()
    train_rows = np.flatnonzero(data["split_code"] == 0)
    if args.stage0:
        train_rows = train_rows[: min(len(train_rows), args.batch_size * args.accumulation * args.max_optimizer_steps)]
    episode_counts = {int(episode): int(np.sum(data["episode_index"][train_rows] == episode)) for episode in np.unique(data["episode_index"][train_rows])}
    episode_count = len(episode_counts)
    total_windows = len(train_rows)
    dataset = StateVideoDataset(data, features, train_rows, stats["video_mean"], stats["video_std"], args.cell, args.horizon)
    model = StateVideoPredictor(horizon=args.horizon, seed=args.seed).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    steps_per_epoch = math.ceil(total_windows / args.batch_size / args.accumulation)
    expected_steps = args.max_optimizer_steps if args.stage0 else EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(expected_steps, 1), eta_min=1e-5)
    config = {
        "status": "LOCKED", "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_V1", "cell": args.cell,
        "state_level": args.cell.split("-")[0], "video_level": args.cell.split("-")[1], "horizon_steps": args.horizon,
        "seed": args.seed, "epochs": 0 if args.stage0 else EPOCHS, "batch_size": args.batch_size,
        "accumulation": args.accumulation, "effective_batch": args.batch_size * args.accumulation,
        "steps_per_epoch": steps_per_epoch, "expected_optimizer_steps": expected_steps, "train_windows": total_windows,
        "train_episodes": episode_count, "parameter_count": parameter_count(model), "initial_model_hash": initial_hash,
        "v3_input_sha256": digest(V3 / "data/V3_MAIN_INPUTS.npz"), "v3_cache_sha256": digest(V3 / "cache/V3_TEMPORAL_FEATURES.npy"),
        "v3_normalizer_sha256": digest(V3 / "data/V3_NORMALIZERS.json"), "v3_video_normalizer_sha256": digest(V3 / "data/V3_VIDEO_NORMALIZER.json"),
        "mask_contract_sha256": digest(EXP / "INPUT_MASK_CONTRACT.json"), "model_code_sha256": digest(EXP / "code/state_video_model.py"),
        "source_model_sha256": digest(EXP / "code/model_v3_source.py"), "final_test_access": 0,
        "encoder_updated": False, "new_encoder_forward_for_extraction": False,
    }
    atomic_json(run_root / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "optimizer_steps": 0, "final_test_access": 0})
    fixture = next(iter(DataLoader(dataset, batch_size=min(args.batch_size, 8), shuffle=False, num_workers=0)))
    model.eval()
    with torch.inference_mode():
        _, _, _, fixture_prediction, _ = batch_loss(model, fixture, stats, args.cell, device)
    persistence_error = float(torch.max(torch.abs(fixture_prediction - fixture[4].to(device).unsqueeze(1))))
    if persistence_error > 1e-6:
        raise RuntimeError(f"zero residual is not Persistence: {persistence_error}")
    steps = 0
    curves: list[dict[str, object]] = []
    try:
        for epoch in range(1, EPOCHS + 1 if not args.stage0 else 100000):
            if steps >= expected_steps:
                break
            model.train()
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            weighted_loss = weighted_position = weighted_fd = weight_sum = 0.0
            for batch in loader:
                if steps >= expected_steps:
                    break
                total, position, fd, _prediction, episode = batch_loss(model, batch, stats, args.cell, device)
                episode_np = episode.detach().cpu().numpy()
                weights_np = np.asarray([total_windows / (episode_count * episode_counts[int(value)]) for value in episode_np], dtype=np.float32)
                weights = torch.from_numpy(weights_np).to(device)
                loss = (total * weights).sum() / weights.sum()
                loss.backward()
                micro += 1
                weighted_loss += float(np.sum(total.detach().cpu().numpy() * weights_np))
                weighted_position += float(np.sum(position.detach().cpu().numpy() * weights_np))
                weighted_fd += float(np.sum(fd.detach().cpu().numpy() * weights_np))
                weight_sum += float(weights_np.sum())
                if micro == args.accumulation or steps + 1 >= expected_steps:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    steps += 1
                    micro = 0
            curves.append({"epoch": epoch, "optimizer_steps": steps, "loss_weighted": weighted_loss / max(weight_sum, 1e-12), "position_weighted": weighted_position / max(weight_sum, 1e-12), "fd_weighted": weighted_fd / max(weight_sum, 1e-12), "learning_rate": scheduler.get_last_lr()[0]})
            atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "last_completed_epoch": epoch, "optimizer_steps": steps, "final_test_access": 0})
            if not args.stage0 and epoch in (10, 20, 30):
                save_checkpoint(run_root / f"checkpoint_epoch{epoch}.pt", model, optimizer, scheduler, config, epoch, steps)
        if steps != expected_steps:
            raise RuntimeError(f"optimizer steps {steps} != {expected_steps}")
        checkpoint_name = "checkpoint_stage0.pt" if args.stage0 else "checkpoint_last.pt"
        model.eval()
        save_checkpoint(run_root / checkpoint_name, model, optimizer, scheduler, config, epoch, steps)
        with torch.inference_mode():
            _, _, _, prediction, _ = batch_loss(model, fixture, stats, args.cell, device)
        reloaded = StateVideoPredictor(horizon=args.horizon, seed=args.seed).to(device)
        payload = torch.load(run_root / checkpoint_name, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(payload["model_state_dict"], strict=True)
        reloaded.eval()
        with torch.inference_mode():
            _, _, _, reloaded_prediction, _ = batch_loss(reloaded, fixture, stats, args.cell, device)
        reload_diff = float(torch.max(torch.abs(prediction - reloaded_prediction)).detach().cpu())
        if reload_diff > 1e-6:
            raise RuntimeError(f"checkpoint reload mismatch: {reload_diff}")
        peak_allocated = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved() // (1024 * 1024)) if device.type == "cuda" else 0
        atomic_json(run_root / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves, "final_test_access": 0})
        atomic_json(run_root / ("STAGE0_REPORT.json" if args.stage0 else "RUN_COMPLETE.json"), {"status": "PASS", "run_id": run_root.name, "cell": args.cell, "horizon": args.horizon, "seed": args.seed, "epoch": epoch, "optimizer_steps": steps, "checkpoint": checkpoint_name, "parameter_count": parameter_count(reloaded), "initial_persistence_max_abs": persistence_error, "checkpoint_reload_prediction_max_abs": reload_diff, "peak_memory_allocated_mib": peak_allocated, "peak_memory_reserved_mib": peak_reserved, "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"), "final_test_access": 0, "encoder_updated": False})
        atomic_json(run_root / "RUN_STATUS.json", {"status": "complete", "optimizer_steps": steps, "checkpoint": checkpoint_name, "final_test_access": 0})
        print(json.dumps({"status": "PASS", "cell": args.cell, "horizon": args.horizon, "seed": args.seed, "optimizer_steps": steps, "peak_memory_reserved_mib": peak_reserved}, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(run_root / "RUN_STATUS.json", {"status": "failed", "optimizer_steps": steps, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
