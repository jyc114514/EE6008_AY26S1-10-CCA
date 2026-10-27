#!/usr/bin/env python3
"""Run one locked HD-D{0,1}-L{0,1} cell or its real-GPU Stage0."""

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

from model import HandDecoder, SelectorHead, compose_hand_prediction, loss_components, set_all_seeds, state_hash, total_loss


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
TEMPORAL_ROOT = REPO / "reports/core_b/HAND_TEMPORAL_FIDELITY_V1_20260914T152845Z"
SEED = 20260933
FOLDS = 3
STEPS = 30
HAND = 14
CONTEXT = 1336
DECODER_HIDDEN = 160
BASE_BATCH = 8
BASE_ACCUMULATION = 4
BASE_EPOCHS = 20
SELECTOR_BATCH = 32
SELECTOR_EPOCHS = 40
SELECTOR_MIN_DELTA = 1e-5
SELECTOR_PATIENCE = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
CELLS = ("HD-D0-L0", "HD-D0-L1", "HD-D1-L0", "HD-D1-L1")
SPECS = {
    "HD-D0-L0": {"decoder": "D0", "loss": "L0"},
    "HD-D0-L1": {"decoder": "D0", "loss": "L1"},
    "HD-D1-L0": {"decoder": "D1", "loss": "L0"},
    "HD-D1-L1": {"decoder": "D1", "loss": "L1"},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(part, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        torch.save(payload, handle)
    os.replace(part, path)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fold_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761) + np.uint64(1013904223)) % np.uint64(FOLDS)).astype(np.int64)


def selector_calibration_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return (((ids * np.uint64(2246822519) + np.uint64(3266489917)) % np.uint64(10)) == 0)


def load_contract() -> dict[str, Any]:
    data = load_npz(INPUT)
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("canonical train/dev counts failed")
    if data["targets_standardized"].shape != (3355, 30, 28) or data["targets_native"].shape != (3355, 30, 28):
        raise RuntimeError("canonical target shape failed")
    state = data["state_action_history"].astype(np.float32)
    dynamic = data["dynamic_pooled"].astype(np.float32)
    context = np.concatenate((dynamic, state), axis=1)
    if context.shape != (3355, CONTEXT):
        raise RuntimeError("context shape failed")
    state_summary = state.mean(axis=1, keepdims=True).astype(np.float32)
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float32)
    if mean.shape != (30, 28) or scale.shape != (30, 28) or np.any(scale <= 0):
        raise RuntimeError("normalizer shape/value failed")
    locked = load_npz(LOCKED_ARM)
    dev_ids = data["episode_index"][dev_mask].astype(np.int64)
    dev_tasks = data["original_task_index"][dev_mask].astype(np.int64)
    if not np.array_equal(locked["episode_index"], dev_ids) or not np.array_equal(locked["original_task_index"], dev_tasks):
        raise RuntimeError("locked Arm IDs failed")
    if locked["predictions_standardized"].shape != (614, 30, 28) or locked["predictions_native"].shape != (614, 30, 28):
        raise RuntimeError("locked Arm shape failed")
    if not np.array_equal(locked["targets_standardized"], data["targets_standardized"][dev_mask]) or not np.array_equal(locked["persistence_standardized"], data["persistence_standardized"][dev_mask]):
        raise RuntimeError("locked Arm target/persistence pairing failed")
    loss_contract = json.loads((ROOT / "TEMPORAL_LOSS_CONTRACT.json").read_text(encoding="utf-8"))
    dct_contract = json.loads((ROOT / "DCT_CONTRACT.json").read_text(encoding="utf-8"))
    if loss_contract.get("lambda_fd", 0) <= 0 or dct_contract.get("K") != 16:
        raise RuntimeError("locked DCT/loss contract failed")
    return {
        "data": data,
        "train_mask": train_mask,
        "dev_mask": dev_mask,
        "context": context,
        "state_summary": state_summary,
        "mean": mean,
        "scale": scale,
        "safe_scale": np.asarray(loss_contract["safe_scale_j"], dtype=np.float32),
        "lambda_fd": float(loss_contract["lambda_fd"]),
        "dct_k": int(dct_contract["K"]),
        "locked": locked,
    }


def decoder_offset(decoder: str) -> int:
    return 0 if decoder == "D0" else 10000


def base_seed(decoder: str, fold: int | None = None, *, all_train: bool = False, stage0: bool = False) -> int:
    offset = decoder_offset(decoder)
    if stage0:
        return SEED + 7000 + offset
    if all_train:
        return SEED + 3000 + offset
    return SEED + offset + int(fold or 0) * 1000


def selector_seed(decoder: str, *, stage0: bool = False) -> int:
    return SEED + (8000 if stage0 else 5000) + decoder_offset(decoder)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def predict_base(model: HandDecoder, context: np.ndarray, persistence_std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(context), SELECTOR_BATCH):
            x = to_device(context[start : start + SELECTOR_BATCH], device)
            p = to_device(persistence_std[start : start + SELECTOR_BATCH], device)
            values.append((p + model(x)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def train_base(
    *,
    cell: str,
    decoder: str,
    loss_mode: str,
    context: np.ndarray,
    persistence_std: np.ndarray,
    target_std: np.ndarray,
    persistence_native: np.ndarray,
    target_native: np.ndarray,
    mean_hand: torch.Tensor,
    scale_hand: torch.Tensor,
    safe_scale: torch.Tensor,
    lambda_fd: torch.Tensor,
    run_dir: Path,
    seed: int,
    device: torch.device,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "final_test_access": 0, "optimizer_steps": 0})
    set_all_seeds(seed)
    torch.set_num_threads(1)
    tensors = tuple(to_device(value, torch.device("cpu")) for value in (context, persistence_std, target_std, persistence_native, target_native))
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=BASE_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")
    steps_per_epoch = math.ceil(len(loader) / BASE_ACCUMULATION)
    total_steps = BASE_EPOCHS * steps_per_epoch if max_optimizer_steps is None else max_optimizer_steps
    model = HandDecoder(decoder=decoder, seed=seed, dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    config = {
        "status": "LOCKED",
        "cell": cell,
        "role": run_dir.name,
        "seed": seed,
        "decoder": decoder,
        "loss_mode": loss_mode,
        "input_dim": CONTEXT,
        "hidden_dim": DECODER_HIDDEN,
        "output_shape": list(model.output_shape),
        "physical_batch": BASE_BATCH,
        "gradient_accumulation": BASE_ACCUMULATION,
        "epochs": BASE_EPOCHS,
        "optimizer_steps": total_steps,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "input_sha256": digest(INPUT),
        "normalizer_sha256": digest(NORMALIZER),
        "dct_K": 16,
        "lambda_fd": float(lambda_fd.detach().cpu()),
        "dev_target_used_in_fit": False,
        "final_test_access": 0,
        "encoder_or_lora_modified": False,
    }
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, BASE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_value = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(loader, start=1):
            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                break
            context_b, p_std_b, target_std_b, _p_native_b, target_native_b = (value.to(device, non_blocking=True) for value in batch)
            prediction_std = p_std_b + model(context_b)
            total, position, fd_scaled = total_loss(loss_mode, prediction_std, target_std_b, target_native_b, mean_hand, scale_hand, safe_scale, lambda_fd)
            if not torch.isfinite(total) or not torch.isfinite(position) or not torch.isfinite(fd_scaled):
                raise FloatingPointError(f"non-finite {cell} base loss")
            (total / BASE_ACCUMULATION).backward()
            total_loss_value += float(total.detach().cpu())
            batch_count += 1
            if batch_index % BASE_ACCUMULATION == 0 or batch_index == len(loader) or (max_optimizer_steps is not None and optimizer_steps + 1 >= max_optimizer_steps):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"non-finite {cell} base gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
        curves.append({"epoch": epoch, "loss": total_loss_value / max(batch_count, 1), "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0})
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break
    expected_steps = total_steps
    if optimizer_steps != expected_steps:
        raise RuntimeError(f"{cell}/{run_dir.name} optimizer steps {optimizer_steps} != {expected_steps}")
    payload = {
        "cell": cell,
        "role": run_dir.name,
        "seed": seed,
        "decoder": decoder,
        "loss_mode": loss_mode,
        "epoch": curves[-1]["epoch"] if curves else 0,
        "optimizer_steps": optimizer_steps,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "initial_state_hash": initial_hash,
    }
    atomic_torch(run_dir / "checkpoint_best.pt", payload)
    atomic_torch(run_dir / "checkpoint_last.pt", payload)
    reloaded = HandDecoder(decoder=decoder, seed=seed, dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
    reload_payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    reloaded.load_state_dict(reload_payload["model_state_dict"], strict=True)
    reference = predict_base(model, context[: min(len(context), SELECTOR_BATCH)], persistence_std[: min(len(context), SELECTOR_BATCH)], device)
    replay = predict_base(reloaded, context[: min(len(context), SELECTOR_BATCH)], persistence_std[: min(len(context), SELECTOR_BATCH)], device)
    replay_max = float(np.max(np.abs(reference - replay)))
    if replay_max > 1e-6:
        raise RuntimeError(f"{cell}/{run_dir.name} checkpoint next-step replay failed: {replay_max}")
    predictions = predict_base(reloaded, context, persistence_std, device)
    atomic_npz(run_dir / "TRAIN_PREDICTIONS.npz", predictions_hand_standardized=predictions, persistence_hand_standardized=persistence_std, episode_count=np.asarray([len(context)], dtype=np.int64))
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves})
    write_rows(run_dir / "EPOCH_CURVES.csv", curves)
    marker = {
        "status": "complete",
        "cell": cell,
        "role": run_dir.name,
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "prediction_shape": list(predictions.shape),
        "strict_checkpoint_reload": "PASS",
        "next_step_replay_max_abs": replay_max,
        "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt"),
        "prediction_sha256": digest(run_dir / "TRAIN_PREDICTIONS.npz"),
        "final_test_access": 0,
    }
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "complete", "cell": cell, "role": run_dir.name, "seed": seed, "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()})
    return {"marker": marker, "model": reloaded, "predictions": predictions}


def selector_predict(model: SelectorHead, context: np.ndarray, summary: np.ndarray, magnitude: np.ndarray, p_std: np.ndarray, neural_std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    q_values: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(context), SELECTOR_BATCH):
            x = to_device(context[start : start + SELECTOR_BATCH], device)
            s = to_device(summary[start : start + SELECTOR_BATCH], device)
            m = to_device(magnitude[start : start + SELECTOR_BATCH], device)
            p = to_device(p_std[start : start + SELECTOR_BATCH], device)
            n = to_device(neural_std[start : start + SELECTOR_BATCH], device)
            q = model.probability(x, s, m)
            predictions.append((p + q * (n - p)).cpu().numpy())
            q_values.append(q.cpu().numpy())
    return np.concatenate(q_values).astype(np.float32), np.concatenate(predictions).astype(np.float32)


def train_selector(
    *,
    cell: str,
    decoder: str,
    loss_mode: str,
    contract: dict[str, Any],
    oof: dict[str, np.ndarray],
    dev_base_std: np.ndarray,
    run_dir: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    data = contract["data"]
    train_mask = contract["train_mask"]
    dev_mask = contract["dev_mask"]
    context_train = contract["context"][train_mask]
    summary_train = contract["state_summary"][train_mask]
    p_train = oof["persistence_hand_standardized"].astype(np.float32)
    n_train = oof["oof_predictions_hand_standardized"].astype(np.float32)
    t_train = oof["targets_hand_standardized"].astype(np.float32)
    tn_train = oof["targets_hand_native"].astype(np.float32)
    p_native_train = oof["persistence_hand_native"].astype(np.float32)
    magnitude_train = np.abs(n_train - p_train).mean(axis=2, keepdims=True).astype(np.float32)
    ids = oof["episode_index"].astype(np.int64)
    calibration = selector_calibration_assignment(ids)
    fit_positions = np.flatnonzero(~calibration)
    calibration_positions = np.flatnonzero(calibration)
    if len(fit_positions) != 2477 or len(calibration_positions) != 264:
        raise RuntimeError("selector split contract failed")
    dataset = TensorDataset(*(to_device(value, torch.device("cpu")) for value in (context_train, summary_train, magnitude_train, p_train, n_train, t_train, tn_train, p_native_train)))
    fit_loader = DataLoader(torch.utils.data.Subset(dataset, fit_positions.tolist()), batch_size=SELECTOR_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0)
    calibration_loader = DataLoader(torch.utils.data.Subset(dataset, calibration_positions.tolist()), batch_size=SELECTOR_BATCH, shuffle=False, num_workers=0)
    mean_hand = to_device(contract["mean"][:, :HAND], device)
    scale_hand = to_device(contract["scale"][:, :HAND], device)
    safe_scale = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    set_all_seeds(seed)
    torch.set_num_threads(1)
    model = SelectorHead(seed=seed).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(fit_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SELECTOR_EPOCHS * steps_per_epoch)
    best_value = float("inf")
    best_epoch = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, SELECTOR_EPOCHS + 1):
        model.train()
        fit_total = 0.0
        fit_batches = 0
        for batch in fit_loader:
            context_b, summary_b, magnitude_b, p_b, n_b, t_b, tn_b, _pn_b = (value.to(device, non_blocking=True) for value in batch)
            q = model.probability(context_b, summary_b, magnitude_b)
            hand = compose_hand_prediction(p_b, n_b, q)
            total, position, fd_scaled = total_loss(loss_mode, hand, t_b, tn_b, mean_hand, scale_hand, safe_scale, lambda_fd)
            if not torch.isfinite(total) or not torch.isfinite(position) or not torch.isfinite(fd_scaled):
                raise FloatingPointError(f"non-finite {cell} selector loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite {cell} selector gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            fit_total += float(total.detach().cpu())
            fit_batches += 1
        calibration_values: list[float] = []
        with torch.inference_mode():
            for batch in calibration_loader:
                context_b, summary_b, magnitude_b, p_b, n_b, t_b, tn_b, _pn_b = (value.to(device, non_blocking=True) for value in batch)
                q = model.probability(context_b, summary_b, magnitude_b)
                hand = compose_hand_prediction(p_b, n_b, q)
                total, _position, _fd_scaled = total_loss(loss_mode, hand, t_b, tn_b, mean_hand, scale_hand, safe_scale, lambda_fd)
                calibration_values.append(float(total.cpu()))
        calibration_value = float(np.mean(calibration_values))
        if calibration_value < best_value - SELECTOR_MIN_DELTA:
            best_value = calibration_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        curves.append({"epoch": epoch, "fit_loss": fit_total / max(fit_batches, 1), "calibration_loss": calibration_value, "optimizer_steps": optimizer_steps, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "stage": "selector", "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "best_epoch": best_epoch, "final_test_access": 0})
        # Keep the fixed epoch budget and do not let dev influence stopping.
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"{cell} selector best state missing")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    atomic_torch(run_dir / "checkpoint_last.pt", {"cell": cell, "seed": seed, "model_state_dict": last_state, "optimizer_steps": optimizer_steps})
    atomic_torch(run_dir / "checkpoint_best.pt", {"cell": cell, "seed": seed, "model_state_dict": best_state, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "initial_state_hash": initial_hash})
    reloaded = SelectorHead(seed=seed).to(device)
    reload_payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    reloaded.load_state_dict(reload_payload["model_state_dict"], strict=True)
    dev_context = contract["context"][dev_mask]
    dev_summary = contract["state_summary"][dev_mask]
    p_dev = data["persistence_standardized"][dev_mask, :, :HAND].astype(np.float32)
    magnitude_dev = np.abs(dev_base_std - p_dev).mean(axis=2, keepdims=True).astype(np.float32)
    q_dev, hand_dev_std = selector_predict(reloaded, dev_context, dev_summary, magnitude_dev, p_dev, dev_base_std, device)
    locked_arm_std = contract["locked"]["predictions_standardized"][:, :, 14:28].astype(np.float32)
    locked_arm_native = contract["locked"]["predictions_native"][:, :, 14:28].astype(np.float32)
    hand_dev_native = hand_dev_std.astype(np.float64) * contract["scale"][:, :HAND][None, :, :] + contract["mean"][:, :HAND][None, :, :]
    full_dev_std = np.concatenate((hand_dev_std, locked_arm_std), axis=2)
    full_dev_native = np.concatenate((hand_dev_native.astype(np.float32), locked_arm_native), axis=2)
    dev_target_std = data["targets_standardized"][dev_mask].astype(np.float32)
    hand_mae = float(np.abs(hand_dev_std - dev_target_std[:, :, :HAND]).mean())
    full_mae = float(np.abs(full_dev_std - dev_target_std).mean())
    p_hand_mae = float(np.abs(p_dev - dev_target_std[:, :, :HAND]).mean())
    metrics = {
        "hand_mae": hand_mae,
        "full28_mae": full_mae,
        "persistence_hand_mae": p_hand_mae,
        "hand_skill_vs_persistence": (p_hand_mae - hand_mae) / p_hand_mae,
        "q_mean": float(q_dev.mean()),
        "q_collapse_fraction": float(np.mean((q_dev < 0.05) | (q_dev > 0.95))),
        "prediction_hand_shape": list(hand_dev_std.shape),
        "prediction_full_shape": list(full_dev_std.shape),
    }
    atomic_npz(
        run_dir / "PREDICTIONS_DEV.npz",
        hand_predictions_standardized=hand_dev_std,
        hand_predictions_native=hand_dev_native.astype(np.float32),
        hand_persistence_standardized=p_dev,
        hand_persistence_native=data["persistence_native"][dev_mask, :, :HAND].astype(np.float32),
        hand_base_predictions_standardized=dev_base_std,
        q=q_dev,
        full_predictions_standardized=full_dev_std,
        full_predictions_native=full_dev_native,
        locked_arm_standardized=locked_arm_std,
        locked_arm_native=locked_arm_native,
        episode_index=data["episode_index"][dev_mask].astype(np.int64),
        original_task_index=data["original_task_index"][dev_mask].astype(np.int64),
    )
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves})
    write_rows(run_dir / "EPOCH_CURVES.csv", curves)
    config = {
        "status": "LOCKED",
        "cell": cell,
        "stage": "selector",
        "seed": seed,
        "decoder": decoder,
        "loss_mode": loss_mode,
        "selector_input": "past FUSION context + past state summary + OOF/all-train base residual magnitude + learned time embedding",
        "input_dim": CONTEXT,
        "selector_output_shape": [STEPS, 1],
        "physical_batch": SELECTOR_BATCH,
        "max_epochs": SELECTOR_EPOCHS,
        "patience": SELECTOR_PATIENCE,
        "optimizer_steps": optimizer_steps,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "oof_path": str(run_dir.parent.parent.parent / "oof" / f"{cell}_OOF.npz"),
        "dct_K": 16,
        "lambda_fd": float(lambda_fd.detach().cpu()),
        "selector_fit_rows": int(len(fit_positions)),
        "selector_calibration_rows": int(len(calibration_positions)),
        "dev_target_used_in_fit": False,
        "future_motion_input": False,
        "arm_locked": True,
        "final_test_access": 0,
    }
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "model": "selector_only_hand_gate", "selector_parameters_only": True, "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()), "initial_state_hash": initial_hash, "best_state_hash": state_hash(reloaded), "arm_prediction_source_locked": True, "final_test_access": 0})
    atomic_json(run_dir / "RUN_METRICS.json", {"status": "complete", "cell": cell, "seed": seed, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "metrics": metrics, "initial_state_hash": initial_hash, "best_state_hash": state_hash(reloaded), "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "checkpoint_last_sha256": digest(run_dir / "checkpoint_last.pt"), "prediction_sha256": digest(run_dir / "PREDICTIONS_DEV.npz"), "dev_target_used_in_fit": False, "final_test_access": 0})
    marker = {"status": "complete", "cell": cell, "seed": seed, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "prediction_hand_shape": list(hand_dev_std.shape), "prediction_full_shape": list(full_dev_std.shape), "q_shape": list(q_dev.shape), "strict_checkpoint_reload": "PASS", "final_test_access": 0, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "PREDICTIONS_DEV.npz")}
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "complete", "cell": cell, "stage": "selector", "seed": seed, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()})
    return {"marker": marker, "metrics": metrics, "q": q_dev, "hand_std": hand_dev_std, "hand_native": hand_dev_native, "full_std": full_dev_std, "full_native": full_dev_native}


def run_stage0(cell: str, root: Path, device: torch.device) -> dict[str, object]:
    spec = SPECS[cell]
    contract = load_contract()
    data = contract["data"]
    train = contract["train_mask"]
    positions = np.flatnonzero(train)[:32]
    context = contract["context"][positions]
    p_std = data["persistence_standardized"][positions, :, :HAND].astype(np.float32)
    t_std = data["targets_standardized"][positions, :, :HAND].astype(np.float32)
    p_native = data["persistence_native"][positions, :, :HAND].astype(np.float32)
    t_native = data["targets_native"][positions, :, :HAND].astype(np.float32)
    mean_hand = to_device(contract["mean"][:, :HAND], device)
    scale_hand = to_device(contract["scale"][:, :HAND], device)
    safe_scale = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    run_dir = root / "runs" / "stage0" / cell
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = base_seed(spec["decoder"], stage0=True)
    set_all_seeds(seed)
    model = HandDecoder(decoder=spec["decoder"], seed=seed, dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
    before = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for step in range(4):
        batch = slice((step * BASE_BATCH) % 32, ((step * BASE_BATCH) % 32) + BASE_BATCH)
        prediction = p_std[batch]
        context_b = to_device(context[batch], device)
        p_b = to_device(p_std[batch], device)
        t_b = to_device(t_std[batch], device)
        tn_b = to_device(t_native[batch], device)
        prediction = p_b + model(context_b)
        total, position, fd_scaled = total_loss(spec["loss"], prediction, t_b, tn_b, mean_hand, scale_hand, safe_scale, lambda_fd)
        if not torch.isfinite(total) or not torch.isfinite(position) or not torch.isfinite(fd_scaled):
            raise FloatingPointError(f"Stage0 non-finite loss {cell}")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Stage0 non-finite gradient {cell}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    after = state_hash(model)
    with torch.inference_mode():
        stage_prediction = p_b + model(context_b)
    checkpoint = run_dir / "checkpoint_stage0.pt"
    atomic_torch(checkpoint, {"cell": cell, "seed": seed, "decoder": spec["decoder"], "loss_mode": spec["loss"], "optimizer_steps": 4, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})
    reload_model = HandDecoder(decoder=spec["decoder"], seed=seed, dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    with torch.inference_mode():
        replay = p_b + reload_model(context_b)
    replay_error = float(torch.max(torch.abs(stage_prediction - replay)).cpu())
    if replay_error > 1e-6:
        raise RuntimeError(f"Stage0 checkpoint replay failed {cell}: {replay_error}")
    # Re-evaluate the selector smoke on all 32 frozen train-only examples so
    # context, summary, q and proposal tensors share exactly the same batch.
    context_t = to_device(context, device)
    p_t = to_device(p_std, device)
    t_t = to_device(t_std, device)
    tn_t = to_device(t_native, device)
    with torch.inference_mode():
        neural_t = p_t + model(context_t)
    selector_seed_value = selector_seed(spec["decoder"], stage0=True)
    selector = SelectorHead(seed=selector_seed_value).to(device)
    selector_before = state_hash(selector)
    selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    neural = neural_t.detach()
    magnitude = torch.mean(torch.abs(neural - p_t), dim=2, keepdim=True)
    summary = to_device(contract["state_summary"][positions], device)
    for _ in range(4):
        q = selector.probability(context_t, summary, magnitude)
        hand = compose_hand_prediction(p_t, neural, q)
        total, position, fd_scaled = total_loss(spec["loss"], hand, t_t, tn_t, mean_hand, scale_hand, safe_scale, lambda_fd)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(selector.parameters(), GRADIENT_CLIP)
        if not torch.isfinite(total) or not torch.isfinite(norm):
            raise FloatingPointError(f"Stage0 selector non-finite {cell}")
        selector_optimizer.step()
        selector_optimizer.zero_grad(set_to_none=True)
    selector_after = state_hash(selector)
    with torch.inference_mode():
        q = selector.probability(context_t, summary, magnitude)
        hand = compose_hand_prediction(p_t, neural, q)
    passed = bool(tuple(stage_prediction.shape) == (BASE_BATCH, STEPS, HAND) and tuple(q.shape) == (len(positions), STEPS, 1) and torch.isfinite(stage_prediction).all() and torch.isfinite(q).all() and torch.isfinite(hand).all() and float(q.min()) >= 0.0 and float(q.max()) <= 1.0 and before != after and selector_before != selector_after and replay_error <= 1e-6)
    marker = {
        "status": "PASS" if passed else "FAIL",
        "cell": cell,
        "seed": SEED,
        "base_stage0_seed": seed,
        "selector_stage0_seed": selector_seed_value,
        "optimizer_steps_base": 4,
        "optimizer_steps_selector": 4,
        "decoder": spec["decoder"],
        "loss_mode": spec["loss"],
        "prediction_hand_shape": list(stage_prediction.shape),
        "q_shape": list(q.shape),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "finite_loss_gradient": True,
        "base_parameters_changed": before != after,
        "selector_parameters_changed": selector_before != selector_after,
        "strict_checkpoint_reload": "PASS",
        "next_step_replay_max_abs": replay_error,
        "physical_gpu": int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1")),
        "final_test_access": 0,
        "completed_utc": now(),
    }
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", marker)
    return marker


def run_full(cell: str, root: Path, device: torch.device) -> dict[str, object]:
    spec = SPECS[cell]
    contract = load_contract()
    data = contract["data"]
    train = contract["train_mask"]
    dev = contract["dev_mask"]
    context_train = contract["context"][train]
    p_std_train = data["persistence_standardized"][train, :, :HAND].astype(np.float32)
    t_std_train = data["targets_standardized"][train, :, :HAND].astype(np.float32)
    p_native_train = data["persistence_native"][train, :, :HAND].astype(np.float32)
    t_native_train = data["targets_native"][train, :, :HAND].astype(np.float32)
    train_ids = data["episode_index"][train].astype(np.int64)
    train_tasks = data["original_task_index"][train].astype(np.int64)
    folds = fold_assignment(train_ids)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2] or len(np.unique(train_ids)) != len(train_ids):
        raise RuntimeError("fold assignment contract failed")
    run_root = root / "runs" / cell
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": cell, "seed": SEED, "final_test_access": 0})
    mean_hand = to_device(contract["mean"][:, :HAND], device)
    scale_hand = to_device(contract["scale"][:, :HAND], device)
    safe_scale = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    oof_pred = np.full((len(train_ids), STEPS, HAND), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    fold_runs: list[dict[str, object]] = []
    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        if set(train_ids[fit].tolist()) & set(train_ids[heldout].tolist()):
            raise RuntimeError("OOF overlap")
        run = train_base(cell=cell, decoder=spec["decoder"], loss_mode=spec["loss"], context=context_train[fit], persistence_std=p_std_train[fit], target_std=t_std_train[fit], persistence_native=p_native_train[fit], target_native=t_native_train[fit], mean_hand=mean_hand, scale_hand=scale_hand, safe_scale=safe_scale, lambda_fd=lambda_fd, run_dir=run_root / f"base_fold_{fold}", seed=base_seed(spec["decoder"], fold), device=device)
        fold_model = HandDecoder(decoder=spec["decoder"], seed=base_seed(spec["decoder"], fold), dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
        fold_model.load_state_dict(torch.load(run_root / f"base_fold_{fold}/checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        oof_pred[heldout] = predict_base(fold_model, context_train[heldout], p_std_train[heldout], device)
        row = {"run": f"base_fold_{fold}", "fit_rows": int(len(fit)), "heldout_rows": int(len(heldout)), "overlap": 0, "prediction_shape": list(oof_pred[heldout].shape), "status": "PASS", "strict_reload": "PASS", "final_test_access": 0}
        fold_rows.append(row)
        fold_runs.append(run["marker"])
    if not np.isfinite(oof_pred).all():
        raise RuntimeError("OOF coverage incomplete")
    all_run = train_base(cell=cell, decoder=spec["decoder"], loss_mode=spec["loss"], context=context_train, persistence_std=p_std_train, target_std=t_std_train, persistence_native=p_native_train, target_native=t_native_train, mean_hand=mean_hand, scale_hand=scale_hand, safe_scale=safe_scale, lambda_fd=lambda_fd, run_dir=run_root / "base_all_train", seed=base_seed(spec["decoder"], all_train=True), device=device)
    all_model = HandDecoder(decoder=spec["decoder"], seed=base_seed(spec["decoder"], all_train=True), dct_k=16, hidden_dim=DECODER_HIDDEN).to(device)
    all_model.load_state_dict(torch.load(run_root / "base_all_train/checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    dev_base = predict_base(all_model, contract["context"][dev], data["persistence_standardized"][dev, :, :HAND].astype(np.float32), device)
    dev_ids = data["episode_index"][dev].astype(np.int64)
    dev_tasks = data["original_task_index"][dev].astype(np.int64)
    atomic_npz(run_root / "base_all_train/DEV_BASE_PREDICTIONS.npz", predictions_hand_standardized=dev_base, episode_index=dev_ids, original_task_index=dev_tasks)
    atomic_npz(root / "oof" / f"{cell}_OOF.npz", episode_index=train_ids, original_task_index=train_tasks, fold=folds, targets_hand_standardized=t_std_train, targets_hand_native=t_native_train, persistence_hand_standardized=p_std_train, persistence_hand_native=p_native_train, oof_predictions_hand_standardized=oof_pred)
    oof_summary = {"status": "PASS", "cell": cell, "coverage": int(np.isfinite(oof_pred).all(axis=(1, 2)).sum()), "expected_coverage": len(train_ids), "fold_counts": {str(fold): int((folds == fold).sum()) for fold in range(FOLDS)}, "fold_overlap": 0, "oof_prediction_shape": list(oof_pred.shape), "dev_base_prediction_shape": list(dev_base.shape), "final_test_access": 0}
    atomic_json(root / "oof" / f"{cell}_OOF_SUMMARY.json", oof_summary)
    selector = train_selector(cell=cell, decoder=spec["decoder"], loss_mode=spec["loss"], contract=contract, oof={"episode_index": train_ids, "original_task_index": train_tasks, "targets_hand_standardized": t_std_train, "targets_hand_native": t_native_train, "persistence_hand_standardized": p_std_train, "persistence_hand_native": p_native_train, "oof_predictions_hand_standardized": oof_pred}, dev_base_std=dev_base, run_dir=run_root / "selector", seed=selector_seed(spec["decoder"]), device=device)
    atomic_npz(root / "predictions" / f"{cell}_PREDICTIONS_DEV.npz", hand_predictions_standardized=selector["hand_std"], hand_predictions_native=selector["hand_native"].astype(np.float32), q=selector["q"], full_predictions_standardized=selector["full_std"], full_predictions_native=selector["full_native"], episode_index=dev_ids, original_task_index=dev_tasks, locked_arm_standardized=contract["locked"]["predictions_standardized"][:, :, 14:28].astype(np.float32), locked_arm_native=contract["locked"]["predictions_native"][:, :, 14:28].astype(np.float32))
    run_marker = {"status": "complete", "cell": cell, "seed": SEED, "decoder": spec["decoder"], "loss_mode": spec["loss"], "base_folds": "3/3", "all_train_base": "complete", "selector": "complete", "prediction_hand_shape": list(selector["hand_std"].shape), "prediction_full_shape": list(selector["full_std"].shape), "q_shape": list(selector["q"].shape), "strict_checkpoint_reload": "PASS", "oof_coverage": oof_summary["coverage"], "oof_overlap": 0, "final_test_access": 0, "completed_utc": now()}
    atomic_json(run_root / "RUN_COMPLETE.json", run_marker)
    atomic_json(run_root / "RUN_STATUS.json", run_marker)
    atomic_json(run_root / "OOF_ACCEPTANCE.json", {"status": "PASS", "rows": fold_rows + [{"run": "base_all_train", "fit_rows": 2741, "heldout_rows": 0, "overlap": 0, "prediction_shape_dev": list(dev_base.shape), "status": "PASS", "strict_reload": "PASS", "final_test_access": 0}], "all_rows_accepted": True, "final_test_access": 0})
    return {"marker": run_marker, "metrics": selector["metrics"], "oof": oof_summary, "fold_markers": fold_runs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--phase", choices=("stage0", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "stage0" and not torch.cuda.is_available():
        raise RuntimeError("real GPU Stage0 requested but CUDA is unavailable in this child")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA child runtime unavailable")
    if args.phase == "stage0":
        result = run_stage0(args.cell, args.root, device)
    else:
        result = run_full(args.cell, args.root, device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
