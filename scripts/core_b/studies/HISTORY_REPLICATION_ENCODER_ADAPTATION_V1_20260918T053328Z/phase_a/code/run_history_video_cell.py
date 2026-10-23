#!/usr/bin/env python3
"""Train one matched History×Video cell or its real-GPU Stage0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import HistoryVideoTCNDCT, loss_components, native_prediction, parameter_count, set_all_seeds, state_hash


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
SOURCE_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
INPUT = SOURCE_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
PREPROCESS_STATS = SOURCE_ROOT / "PREPROCESS_STATS.json"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
SEED = 20260940
HISTORY_STEPS = 32
STATE_DIM = 71
VIDEO_DIM = 768
HAND_DIM = 14
TARGET_STEPS = 30
DEFAULT_BATCH = 64
DEFAULT_ACCUMULATION = 1
EPOCHS = 40
LEARNING_RATE = 1e-3
ETA_MIN = 1e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
CELLS = {
    "HI-S-LAST": ("LAST", "S"),
    "HI-S-HIST": ("HIST", "S"),
    "HI-F-LAST": ("LAST", "F"),
    "HI-F-HIST": ("HIST", "F"),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        torch.save(value, handle)
    os.replace(part, path)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data(input_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(input_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    required = {"episode_index", "split_code", "original_task_index", "targets_native", "targets_standardized", "persistence_native", "history_standardized", "history_mask", "history_relative_delta_standardized", "dynamic_video_feature"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"input keys missing: {missing}")
    if data["targets_native"].shape != (3355, TARGET_STEPS, 28) or data["history_standardized"].shape != (3355, HISTORY_STEPS, STATE_DIM):
        raise ValueError("input shape contract failed")
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise ValueError("train/dev count contract failed")
    with NORMALIZER.open(encoding="utf-8") as handle:
        target_norm = json.load(handle)["1.0s"]
    target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
    target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
    preprocess = json.loads(PREPROCESS_STATS.read_text(encoding="utf-8"))
    stats = {
        "target_mean_hand": target_mean[:, :HAND_DIM],
        "target_std_hand": target_std[:, :HAND_DIM],
        "residual_sigma": np.asarray(preprocess["residual_sigma"], dtype=np.float32),
        "fd_scale": np.asarray(preprocess["fd_scale"], dtype=np.float32),
        "lambda_fd": float(preprocess["lambda_fd"]),
    }
    if stats["residual_sigma"].shape != (HAND_DIM,) or stats["fd_scale"].shape != (HAND_DIM,) or np.any(stats["residual_sigma"] <= 0) or np.any(stats["fd_scale"] <= 0):
        raise ValueError("train-only loss scales are invalid")
    if not np.isfinite(data["history_standardized"]).all() or not np.isfinite(data["dynamic_video_feature"]).all():
        raise ValueError("input features are non-finite")
    return data, stats


def cell_inputs(data: dict[str, np.ndarray], cell: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell}")
    state_mode, video_mode = CELLS[cell]
    history = data["history_standardized"].astype(np.float32)
    if state_mode == "LAST":
        state = np.repeat(history[:, -1:, :], HISTORY_STEPS, axis=1)
    else:
        state = history
    mask = data["history_mask"].astype(np.float32)
    relative_delta = data["history_relative_delta_standardized"].astype(np.float32)
    if video_mode == "S":
        video = np.zeros_like(data["dynamic_video_feature"], dtype=np.float32)
    else:
        video = data["dynamic_video_feature"].astype(np.float32)
    anchor = data["persistence_native"][:, 0, :HAND_DIM].astype(np.float32)
    target_native = data["targets_native"][:, :, :HAND_DIM].astype(np.float32)
    target_std = data["targets_standardized"][:, :, :HAND_DIM].astype(np.float32)
    ids = data["episode_index"].astype(np.int64)
    tasks = data["original_task_index"].astype(np.int64)
    return state, mask, relative_delta, video, anchor, target_native, target_std, ids, tasks


def as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device, non_blocking=device.type == "cuda")


def make_loader(values: tuple[np.ndarray, ...], indices: np.ndarray, batch_size: int, seed: int, device: torch.device) -> DataLoader:
    tensors = tuple(torch.from_numpy(np.asarray(value[indices], dtype=np.float32)) for value in values)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")


def save_checkpoint(path: Path, model: HistoryVideoTCNDCT, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.CosineAnnealingLR, *, cell: str, epoch: int, optimizer_steps: int, config: dict[str, Any]) -> None:
    atomic_torch(path, {
        "cell": cell,
        "seed": SEED,
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_python": random.getstate(),
    })


def batch_loss(model: HistoryVideoTCNDCT, batch: tuple[torch.Tensor, ...], device: torch.device, stats: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state, mask, delta, video, anchor, target_native, target_std = (item.to(device, non_blocking=device.type == "cuda") for item in batch)
    scaled = model(state, mask, delta, video)
    prediction_native = native_prediction(scaled, anchor, as_tensor(stats["residual_sigma"], device))
    total, position, fd_scaled = loss_components(prediction_native, target_native, target_std, as_tensor(stats["target_mean_hand"], device), as_tensor(stats["target_std_hand"], device), as_tensor(stats["fd_scale"], device), torch.tensor(stats["lambda_fd"], dtype=torch.float32, device=device))
    return total, position, fd_scaled, prediction_native


def run(args: argparse.Namespace) -> int:
    global SEED
    SEED = int(args.seed)
    if args.cell not in CELLS:
        raise ValueError(f"unknown cell {args.cell}")
    if args.batch_size <= 0 or args.accumulation <= 0:
        raise ValueError("batch size and accumulation must be positive")
    if args.stage0 and args.max_optimizer_steps != 8:
        raise ValueError("Stage0 must use exactly 8 optimizer steps")
    if not args.stage0 and args.max_optimizer_steps is not None:
        raise ValueError("max optimizer steps is only valid for Stage0")
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing existing run root: {run_root}")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir()
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in child")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    set_all_seeds(SEED, use_cuda=args.device == "cuda")
    torch.set_num_threads(1)
    input_path = args.input_root.resolve()
    if input_path != INPUT.resolve():
        raise ValueError(f"input path is not the locked experiment input: {input_path}")
    data, stats = load_data(input_path)
    state, mask, relative_delta, video, anchor, target_native, target_std, ids, tasks = cell_inputs(data, args.cell)
    train_indices = np.flatnonzero(data["split_code"] == 0)
    if args.stage0:
        train_indices = train_indices[: min(args.smoke_rows, len(train_indices))]
    if len(train_indices) == 0:
        raise ValueError("no train rows")
    loader_values = (state, mask, relative_delta, video, anchor, target_native, target_std)
    steps_per_epoch = math.ceil(math.ceil(len(train_indices) / args.batch_size) / args.accumulation)
    expected_steps = args.max_optimizer_steps if args.stage0 else EPOCHS * steps_per_epoch
    model = HistoryVideoTCNDCT(seed=SEED)
    initial_model_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(expected_steps, 1), eta_min=ETA_MIN)
    config: dict[str, Any] = {
        "status": "LOCKED",
        "protocol_id": "HISTORY_VIDEO_INFORMATION_MATRIX_V1",
        "cell": args.cell,
        "state_input": CELLS[args.cell][0],
        "video_input": CELLS[args.cell][1],
        "seed": SEED,
        "device": str(device),
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "none"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "none"),
        "history_shape": [32, 71],
        "video_shape": [768],
        "output_shape": [30, 14],
        "parameter_count": parameter_count(model),
        "initial_model_hash": initial_model_hash,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "effective_batch": args.batch_size * args.accumulation,
        "steps_per_epoch": steps_per_epoch,
        "expected_optimizer_steps": expected_steps,
        "epochs": 0 if args.stage0 else EPOCHS,
        "stage0": args.stage0,
        "smoke_rows": len(train_indices) if args.stage0 else None,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "eta_min": ETA_MIN,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "checkpoint_selection": "last_epoch_only" if not args.stage0 else "stage0_reload_check",
        "input_sha256": digest(input_path),
        "preprocess_stats_sha256": digest(PREPROCESS_STATS),
        "normalizer_sha256": digest(NORMALIZER),
        "model_code_sha256": digest(ROOT / "code/model.py"),
        "train_code_sha256": digest(ROOT / "code/run_history_video_cell.py"),
        "dev_target_used_in_fit": False,
        "final_test_access": 0,
    }
    atomic_json(run_root / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": args.cell, "seed": SEED, "stage0": args.stage0, "optimizer_steps": 0, "final_test_access": 0})
    model = model.to(device)
    initial_count = min(len(train_indices), args.batch_size)
    initial_loader = make_loader(loader_values, train_indices[:initial_count], initial_count, SEED, device)
    initial_batch = next(iter(initial_loader))
    with torch.inference_mode():
        initial_total, _initial_position, _initial_fd, initial_prediction = batch_loss(model, initial_batch, device, stats)
    initial_anchor = initial_batch[4].to(device).unsqueeze(1)
    initial_persistence_max_abs = float(torch.max(torch.abs(initial_prediction - initial_anchor)).detach().cpu())
    if initial_persistence_max_abs > 1e-6:
        raise RuntimeError(f"zero-output persistence initialization failed: {initial_persistence_max_abs}")
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    epoch = 0
    try:
        while (args.stage0 and optimizer_steps < expected_steps) or (not args.stage0 and epoch < EPOCHS):
            epoch += 1
            loader = make_loader(loader_values, train_indices, args.batch_size, SEED + epoch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            cycle_samples = 0
            cycle_microbatches = 0
            epoch_loss_sum = 0.0
            epoch_position_sum = 0.0
            epoch_fd_sum = 0.0
            epoch_samples = 0
            for batch_index, batch in enumerate(loader, start=1):
                if optimizer_steps >= expected_steps:
                    break
                total, position, fd_scaled, _prediction = batch_loss(model, batch, device, stats)
                if not torch.isfinite(total) or not torch.isfinite(position) or not torch.isfinite(fd_scaled):
                    raise FloatingPointError(f"non-finite loss in {args.cell}")
                batch_size = int(batch[0].shape[0])
                (total * batch_size).backward()
                cycle_samples += batch_size
                cycle_microbatches += 1
                epoch_loss_sum += float(total.detach().cpu()) * batch_size
                epoch_position_sum += float(position.detach().cpu()) * batch_size
                epoch_fd_sum += float(fd_scaled.detach().cpu()) * batch_size
                epoch_samples += batch_size
                if cycle_microbatches == args.accumulation or batch_index == len(loader) or optimizer_steps + 1 >= expected_steps:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(cycle_samples)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"non-finite gradient in {args.cell}")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1
                    cycle_samples = 0
                    cycle_microbatches = 0
            curves.append({"epoch": epoch, "optimizer_steps": optimizer_steps, "loss": epoch_loss_sum / max(epoch_samples, 1), "position_loss": epoch_position_sum / max(epoch_samples, 1), "fd_scaled_loss": epoch_fd_sum / max(epoch_samples, 1), "learning_rate": scheduler.get_last_lr()[0]})
            atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": args.cell, "seed": SEED, "stage0": args.stage0, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0})
            if not args.stage0 and epoch % 10 == 0:
                save_checkpoint(run_root / f"checkpoint_epoch{epoch}.pt", model, optimizer, scheduler, cell=args.cell, epoch=epoch, optimizer_steps=optimizer_steps, config=config)
        if optimizer_steps != expected_steps:
            raise RuntimeError(f"optimizer steps {optimizer_steps} != expected {expected_steps}")
        checkpoint_name = "checkpoint_stage0.pt" if args.stage0 else "checkpoint_last.pt"
        save_checkpoint(run_root / checkpoint_name, model, optimizer, scheduler, cell=args.cell, epoch=epoch, optimizer_steps=optimizer_steps, config=config)
        write_rows(run_root / "EPOCH_CURVES.csv", curves)
        atomic_json(run_root / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves, "final_test_access": 0})
        model.eval()
        with torch.inference_mode():
            _final_total, _final_position, _final_fd, final_prediction = batch_loss(model, initial_batch, device, stats)
        reload_model = HistoryVideoTCNDCT(seed=SEED)
        payload = torch.load(run_root / checkpoint_name, map_location="cpu", weights_only=False)
        reload_model.load_state_dict(payload["model_state_dict"], strict=True)
        reload_model = reload_model.to(device)
        reload_model.eval()
        with torch.inference_mode():
            reload_total, _reload_position, _reload_fd, reload_prediction = batch_loss(reload_model, initial_batch, device, stats)
        reload_max_abs = float(torch.max(torch.abs(reload_prediction - final_prediction)).detach().cpu())
        if not np.isfinite(reload_max_abs) or reload_max_abs > 1e-6:
            raise RuntimeError(f"checkpoint reload prediction mismatch: {reload_max_abs}")
        peak_allocated = int(torch.cuda.max_memory_allocated() // (1024 * 1024)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved() // (1024 * 1024)) if device.type == "cuda" else 0
        run_summary = {
            "status": "PASS",
            "cell": args.cell,
            "seed": SEED,
            "stage0": args.stage0,
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "checkpoint": checkpoint_name,
            "checkpoint_selection": "last_epoch_only" if not args.stage0 else "stage0_reload_check",
            "initial_persistence_max_abs": initial_persistence_max_abs,
            "checkpoint_reload_prediction_max_abs": reload_max_abs,
            "parameter_count": parameter_count(reload_model),
            "model_hash_final": state_hash(reload_model),
            "peak_memory_allocated_mib": peak_allocated,
            "peak_memory_reserved_mib": peak_reserved,
            "prediction_shape_hand": [len(initial_batch[0]), TARGET_STEPS, HAND_DIM],
            "all_finite": True,
            "dev_inference_started": False,
            "final_test_access": 0,
        }
        if args.stage0:
            atomic_json(run_root / "STAGE0_REPORT.json", run_summary)
        else:
            atomic_json(run_root / "RUN_COMPLETE.json", run_summary)
        atomic_json(run_root / "RUN_STATUS.json", {"status": "complete", "cell": args.cell, "seed": SEED, "stage0": args.stage0, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "checkpoint": checkpoint_name, "peak_memory_allocated_mib": peak_allocated, "peak_memory_reserved_mib": peak_reserved, "final_test_access": 0})
        print(json.dumps(run_summary, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(run_root / "RUN_STATUS.json", {"status": "failed", "cell": args.cell, "seed": SEED, "stage0": args.stage0, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=sorted(CELLS))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--smoke-rows", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--accumulation", type=int, default=DEFAULT_ACCUMULATION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
