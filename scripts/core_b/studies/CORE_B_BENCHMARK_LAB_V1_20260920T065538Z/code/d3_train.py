"""D3 controlled training runner with resumable 10-epoch checkpoints."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, sha256_file, utc_now
from d3_model import D3TimeSpacePredictor, parameter_count, state_hash


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


class GridStore:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        groups: dict[int, list[tuple[int, int]]] = {}
        for output_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            local = int(source_index) - int(shard["start_row"])
            groups.setdefault(shard_index, []).append((output_index, local))
        for shard_index, pairs in groups.items():
            shard = self.shards[shard_index]
            values = np.load(shard["array"], mmap_mode="r")
            for output_index, local in pairs:
                dynamic[output_index] = values[local, 0]
                static[output_index] = values[local, 1]
        return dynamic, static


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(value, part)
    os.replace(part, path)


def make_init(path: Path, horizon: int, seed: int, video_mode: str) -> dict[str, torch.Tensor]:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=video_mode)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    atomic_torch_save(path, state)
    return state


def loss_value(pred_residual: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor, sigma: torch.Tensor, safe_fd: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target_residual = (target - anchor[:, None, :]) / sigma[None, None, :]
    position_error = torch.abs(pred_residual - target_residual)
    position = 0.5 * (position_error[..., :14].mean() + position_error[..., 14:].mean())
    predicted_native = anchor[:, None, :] + sigma[None, None, :] * pred_residual
    target_fd = torch.diff(target, dim=1) / safe_fd[None, None, :]
    predicted_fd = torch.diff(predicted_native, dim=1) / safe_fd[None, None, :]
    fd_error = torch.abs(predicted_fd - target_fd)
    fd = 0.5 * (fd_error[..., :14].mean() + fd_error[..., 14:].mean())
    total = position + 0.1 * fd
    return total, {"loss": float(total.detach().cpu()), "position": float(position.detach().cpu()), "fd": float(fd.detach().cpu())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"])
    parser.add_argument("--horizon", type=int, required=True, choices=[30, 60])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D3 training requires approved CUDA")
    if args.batch_size not in {32, 64} or args.accumulation not in {1, 2}:
        raise ValueError("only locked batch64/accum1 or OOM fallback batch32/accum2 is allowed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    args.run_root.mkdir(parents=True, exist_ok=True)
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    normalizers = read_json(V3 / "data/V3_NORMALIZERS.json")
    video_norm = read_json(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json")
    cache_manifest = read_json(BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json")
    store = GridStore(cache_manifest)
    train_indices = np.flatnonzero(data["split_code"] == 0).astype(np.int64)
    if args.stage0:
        train_indices = train_indices[: min(64, len(train_indices))]
    episodes = data["episode_index"][train_indices]
    unique, counts = np.unique(episodes, return_counts=True)
    weight_by_episode = {int(episode): len(train_indices) / (len(unique) * int(count)) for episode, count in zip(unique, counts)}
    weights = np.asarray([weight_by_episode[int(episode)] for episode in episodes], dtype=np.float32)
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32), device=device)
    safe_fd = torch.tensor(np.asarray(normalizers["safe_fd_scale"], dtype=np.float32), device=device)
    mean = np.asarray(video_norm["mean"], dtype=np.float32)
    std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    init_path = BENCH_ROOT / f"d3/runs/INIT_S{args.seed}.pt"
    init_state = make_init(init_path, args.horizon, args.seed, args.config)
    model = D3TimeSpacePredictor(horizon=args.horizon, seed=args.seed, video_mode=args.config).to(device)
    model.load_state_dict(init_state, strict=True)
    init_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    epochs = 1 if args.stage0 else 40
    steps_per_epoch = int(np.ceil(len(train_indices) / args.batch_size))
    expected_steps = 4 if args.stage0 else epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(expected_steps, 1), eta_min=1e-5)
    optimizer_steps = 0
    last_report = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.arange(len(train_indices), dtype=np.int64)
        np.random.default_rng(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for batch_number, start in enumerate(range(0, len(order), args.batch_size)):
            selected = order[start : start + args.batch_size]
            source = train_indices[selected]
            dynamic, static = store.get(source)
            dynamic = ((dynamic - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            static = ((static - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            state = torch.tensor(data["history_standardized"][source], device=device)
            mask = torch.tensor(data["history_mask"][source], device=device)
            relative = torch.tensor(data["history_relative_delta_standardized"][source], device=device)
            anchor = torch.tensor(data["anchor_command_native"][source], device=device)
            target = torch.tensor(data["targets_native"][source, : args.horizon], device=device)
            pred = model(state, mask, relative, torch.tensor(dynamic, device=device), torch.tensor(static, device=device), anchor)
            loss, report = loss_value(pred, target, anchor, sigma, safe_fd)
            loss = loss * torch.tensor(weights[selected], device=device).mean()
            (loss / args.accumulation).backward()
            if (batch_number + 1) % args.accumulation == 0 or start + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            epoch_losses.append(report)
            if args.stage0 and optimizer_steps >= 4:
                break
        scheduler.step()
        last_report = {"epoch": epoch, "optimizer_steps": optimizer_steps, "loss": float(np.mean([item["loss"] for item in epoch_losses])), "position": float(np.mean([item["position"] for item in epoch_losses])), "fd": float(np.mean([item["fd"] for item in epoch_losses])), "lr": float(scheduler.get_last_lr()[0])}
        if not args.stage0 and epoch % 10 == 0:
            atomic_torch_save(args.run_root / f"checkpoint_epoch{epoch}.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "optimizer_steps": optimizer_steps, "seed": args.seed, "config": args.config, "horizon": args.horizon, "init_hash": init_hash, "final_test_access": 0})
    if args.stage0:
        atomic_torch_save(args.run_root / "checkpoint_stage0.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": 1, "optimizer_steps": optimizer_steps, "seed": args.seed, "config": args.config, "horizon": args.horizon, "init_hash": init_hash, "final_test_access": 0})
        atomic_write_json(args.run_root / "STAGE0_REPORT.json", {"status": "PASS", "config": args.config, "horizon": args.horizon, "seed": args.seed, "optimizer_steps": optimizer_steps, "init_hash": init_hash, "parameter_count": parameter_count(model), "final_test_access": 0})
    else:
        atomic_torch_save(args.run_root / "checkpoint_last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epochs, "optimizer_steps": optimizer_steps, "seed": args.seed, "config": args.config, "horizon": args.horizon, "init_hash": init_hash, "final_test_access": 0})
        atomic_write_json(args.run_root / "RUN_COMPLETE.json", {"status": "PASS", "config": args.config, "horizon": args.horizon, "seed": args.seed, "optimizer_steps": optimizer_steps, "epochs": epochs, "init_hash": init_hash, "parameter_count": parameter_count(model), "last_report": last_report, "encoder_updated": False, "final_test_access": 0})
    atomic_write_json(args.run_root / "RUN_STATUS.json", {"status": "complete", "config": args.config, "horizon": args.horizon, "seed": args.seed, "optimizer_steps": optimizer_steps, "last_report": last_report, "final_test_access": 0})
    print(json.dumps({"status": "PASS", "stage0": args.stage0, "config": args.config, "horizon": args.horizon, "seed": args.seed, "optimizer_steps": optimizer_steps}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
