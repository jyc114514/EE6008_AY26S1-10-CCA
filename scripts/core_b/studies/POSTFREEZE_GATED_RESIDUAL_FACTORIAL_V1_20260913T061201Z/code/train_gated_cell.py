#!/usr/bin/env python3
"""Train one post-freeze JY-02 persistence-gated residual exploratory cell."""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import PersistenceGatedResidualModel, parameter_count, split_hidden_for_parameter_match


PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
JY02_ROOT = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
INPUT_PATH = JY02_ROOT / "inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
THRESHOLD_PATH = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_THRESHOLD_SOURCE.json"

PHYSICAL_BATCH = 8
ACCUMULATION = 4
EFFECTIVE_BATCH = 32
MAX_EPOCHS = 20
MIN_EPOCHS = 5
PATIENCE = 5
MIN_DELTA = 0.0001
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
STEPS = 30
DIMS = 28
SEED = 20260924

CELL_SPECS: dict[str, dict[str, Any]] = {
    "E-S-SH-U": {"input": "STATE", "head_coupling": "shared", "residual_gate": "ungated", "input_dim": 568},
    "E-S-SH-G": {"input": "STATE", "head_coupling": "shared", "residual_gate": "group_time_gated", "input_dim": 568},
    "E-S-SP-U": {"input": "STATE", "head_coupling": "split", "residual_gate": "ungated", "input_dim": 568},
    "E-S-SP-G": {"input": "STATE", "head_coupling": "split", "residual_gate": "group_time_gated", "input_dim": 568},
    "E-F-SH-U": {"input": "FUSION", "head_coupling": "shared", "residual_gate": "ungated", "input_dim": 1336},
    "E-F-SH-G": {"input": "FUSION", "head_coupling": "shared", "residual_gate": "group_time_gated", "input_dim": 1336},
    "E-F-SP-U": {"input": "FUSION", "head_coupling": "split", "residual_gate": "ungated", "input_dim": 1336},
    "E-F-SP-G": {"input": "FUSION", "head_coupling": "split", "residual_gate": "group_time_gated", "input_dim": 1336},
}
CELL_ORDER = tuple(CELL_SPECS)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_contract_data() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if not INPUT_PATH.is_file() or not NORMALIZER_PATH.is_file() or not THRESHOLD_PATH.is_file():
        raise FileNotFoundError("locked development input, normalizer or train-threshold file is missing")
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    split = data["split_code"]
    train_mask = split == 0
    dev_mask = split == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("JY02 train/dev count contract failed")
    if data["targets_standardized"].shape != (3355, 30, 28):
        raise RuntimeError("JY02 target shape contract failed")
    normalizer = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))["1.0s"]
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return data, train_mask, dev_mask, {"normalizer": normalizer, "thresholds": thresholds}


def build_model(spec: dict[str, Any], seed: int) -> tuple[nn.Module, int, int | None]:
    split_hidden = None
    hidden = 256
    if spec["head_coupling"] == "split":
        split_hidden, _ = split_hidden_for_parameter_match(int(spec["input_dim"]))
        hidden = split_hidden
    model = PersistenceGatedResidualModel(
        int(spec["input_dim"]),
        head_coupling=str(spec["head_coupling"]),
        residual_gate=str(spec["residual_gate"]),
        hidden_dim=hidden,
        seed=seed,
    )
    return model, hidden, split_hidden


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _metric_dict(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray) -> dict[str, float | bool]:
    error = pred - target
    native_error = pred_native - target_native
    arm_error = error[:, :, 14:28]
    hand_error = error[:, :, :14]
    native_arm_error = native_error[:, :, 14:28]
    target_arm_native = target_native[:, :, 14:28]
    pred_arm_native = pred_native[:, :, 14:28]
    target_arm_std = target[:, :, 14:28]
    pred_arm_std = pred[:, :, 14:28]
    pred_tv = np.abs(np.diff(pred_arm_native, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target_arm_native, axis=1)).sum(axis=1).mean(axis=1)
    valid_tv = target_tv > 1e-12
    tv_ratio = float(np.mean(pred_tv[valid_tv] / target_tv[valid_tv])) if np.any(valid_tv) else float("nan")
    return {
        "arm_full_mae_std": float(np.abs(arm_error).mean()),
        "hand_full_mae_std": float(np.abs(hand_error).mean()),
        "full28_full_mae_std": float(np.abs(error).mean()),
        "full28_full_rmse_std": float(np.sqrt(np.mean(error**2))),
        "arm_full_mae_native": float(np.abs(native_arm_error).mean()),
        "hand_full_mae_native": float(np.abs(native_error[:, :, :14]).mean()),
        "full28_full_mae_native": float(np.abs(native_error).mean()),
        "full28_full_rmse_native": float(np.sqrt(np.mean(native_error**2))),
        "arm_early_mae_std": float(np.abs(arm_error[:, :10]).mean()),
        "arm_middle_mae_std": float(np.abs(arm_error[:, 10:20]).mean()),
        "arm_late_mae_std": float(np.abs(arm_error[:, 20:]).mean()),
        "arm_endpoint_mae_std": float(np.abs(arm_error[:, -1]).mean()),
        "arm_first_difference_mae_std": float(np.abs(np.diff(pred_arm_std, axis=1) - np.diff(target_arm_std, axis=1)).mean()),
        "arm_first_difference_mae_native": float(np.abs(np.diff(pred_arm_native, axis=1) - np.diff(target_arm_native, axis=1)).mean()),
        "native_tv_ratio": tv_ratio,
        "native_tv_mean": float(pred_tv.mean()),
        "target_native_tv_mean": float(target_tv.mean()),
        "all_finite": bool(np.isfinite(pred).all() and np.isfinite(pred_native).all()),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    model.eval()
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    persistence_list: list[np.ndarray] = []
    gate_list: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, target, anchor, episode, task in loader:
            inputs = inputs.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            with _autocast(device):
                residual, gates = model(inputs)
            pred_list.append((anchor + residual.float()).cpu().numpy())
            target_list.append(target.numpy())
            persistence_list.append(anchor.cpu().numpy())
            gate_list.append(gates.float().cpu().numpy())
            ids.append(episode.numpy())
            tasks.append(task.numpy())
    pred = np.concatenate(pred_list).astype(np.float32)
    target = np.concatenate(target_list).astype(np.float32)
    persistence = np.concatenate(persistence_list).astype(np.float32)
    gates = np.concatenate(gate_list).astype(np.float32)
    episode_ids = np.concatenate(ids).astype(np.int64)
    task_ids = np.concatenate(tasks).astype(np.int64)
    pred_native = pred * std[None, :, :] + mean[None, :, :]
    target_native = target * std[None, :, :] + mean[None, :, :]
    persistence_native = persistence * std[None, :, :] + mean[None, :, :]
    return {
        "predictions_standardized": pred,
        "targets_standardized": target,
        "persistence_standardized": persistence,
        "predictions_native": pred_native.astype(np.float32),
        "targets_native": target_native.astype(np.float32),
        "persistence_native": persistence_native.astype(np.float32),
        "gate_values": gates,
        "episode_index": episode_ids,
        "original_task_index": task_ids,
        "metrics": _metric_dict(pred, target, persistence, pred_native, target_native, persistence_native),
    }


def _gate_summary(gates: np.ndarray) -> dict[str, float]:
    eps = 1e-7
    result: dict[str, float] = {}
    for group, index in (("arm", 0), ("hand", 1)):
        values = gates[:, :, index].reshape(-1)
        entropy = -(values * np.log(np.clip(values, eps, 1 - eps)) + (1 - values) * np.log(np.clip(1 - values, eps, 1 - eps)))
        result[f"gate_{group}_mean"] = float(values.mean())
        result[f"gate_{group}_p05"] = float(np.quantile(values, 0.05))
        result[f"gate_{group}_p25"] = float(np.quantile(values, 0.25))
        result[f"gate_{group}_p50"] = float(np.quantile(values, 0.50))
        result[f"gate_{group}_p75"] = float(np.quantile(values, 0.75))
        result[f"gate_{group}_p95"] = float(np.quantile(values, 0.95))
        result[f"gate_{group}_collapse_fraction"] = float(np.mean((values < 0.05) | (values > 0.95)))
        result[f"gate_{group}_entropy"] = float(entropy.mean())
    return result


def train_cell(cell_id: str, *, root: Path, seed: int = SEED, device: torch.device | None = None) -> dict[str, Any]:
    if cell_id not in CELL_SPECS:
        raise ValueError(cell_id)
    if seed <= 0:
        raise ValueError("seed must be positive")
    spec = CELL_SPECS[cell_id]
    run_dir = root / "runs" / cell_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": seed, "optimizer_steps": 0, "final_test_access": 0, "started_utc": now()})
    try:
        set_seed(seed)
        torch.set_num_threads(1)
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data, train_mask, dev_mask, contracts = load_contract_data()
        mean = np.asarray(contracts["normalizer"]["mean"], dtype=np.float32)
        std = np.asarray(contracts["normalizer"]["safe_std"], dtype=np.float32)
        state = data["state_action_history"].astype(np.float32)
        dynamic = data["dynamic_pooled"].astype(np.float32)
        contexts = state if spec["input"] == "STATE" else np.concatenate((dynamic, state), axis=1)
        targets = data["targets_standardized"].astype(np.float32)
        persistence = data["persistence_standardized"].astype(np.float32)
        residual_targets = targets - persistence
        ids = data["episode_index"].astype(np.int64)
        tasks = data["original_task_index"].astype(np.int64)
        train_dataset = TensorDataset(torch.from_numpy(contexts[train_mask]), torch.from_numpy(residual_targets[train_mask]), torch.from_numpy(persistence[train_mask]), torch.from_numpy(ids[train_mask]), torch.from_numpy(tasks[train_mask]))
        dev_dataset = TensorDataset(torch.from_numpy(contexts[dev_mask]), torch.from_numpy(targets[dev_mask]), torch.from_numpy(persistence[dev_mask]), torch.from_numpy(ids[dev_mask]), torch.from_numpy(tasks[dev_mask]))
        loader_seed = seed + 17
        train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(loader_seed), num_workers=0, pin_memory=device.type == "cuda")
        dev_loader = DataLoader(dev_dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        model, hidden_dim, split_hidden = build_model(spec, seed + CELL_ORDER.index(cell_id) * 1000)
        model = model.to(device)
        model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected_count = parameter_count(int(spec["input_dim"]), head_coupling=str(spec["head_coupling"]), residual_gate=str(spec["residual_gate"]), hidden_dim=hidden_dim)
        if model_parameter_count != expected_count:
            raise RuntimeError("parameter count calculation mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(math.ceil(len(train_dataset) / PHYSICAL_BATCH) / ACCUMULATION)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * steps_per_epoch)
        config = {
            "protocol_id": "core_b_postfreeze_gated_residual_factorial_v1_20260913",
            "cell_id": cell_id,
            "factors": spec,
            "seed": seed,
            "component_seeds": {"head": seed + CELL_ORDER.index(cell_id) * 1000 + 11, "gate": seed + CELL_ORDER.index(cell_id) * 1000 + 211, "loader": loader_seed},
            "input_path": str(INPUT_PATH),
            "input_sha256": digest(INPUT_PATH),
            "target_mode": "residual_to_persistence",
            "target_shape": [30, 28],
            "arm_slice": "14:28",
            "hand_slice": "0:14",
            "physical_batch": PHYSICAL_BATCH,
            "gradient_accumulation": ACCUMULATION,
            "nominal_effective_batch": EFFECTIVE_BATCH,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": 1.0,
            "scheduler": "CosineAnnealingLR",
            "steps_per_epoch": steps_per_epoch,
            "max_epochs": MAX_EPOCHS,
            "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "precision": "BF16 autocast; FP32 parameters",
            "split_hidden": split_hidden,
            "parameter_count": model_parameter_count,
            "encoder_or_lora_modified": False,
            "raw_video": False,
            "final_test_access": 0,
            "selection_metric": "Arm-14 full-horizon standardized MAE, then Full-28 MAE, then earlier epoch",
            "gate_contract": {"time_embedding_dim": 32, "groups": ["arm", "hand"], "range": [0, 1], "initial_value": 0.9, "future_target_input": False},
        }
        atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
        atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "model": "persistence_gated_residual_direct", "head_coupling": spec["head_coupling"], "residual_gate": spec["residual_gate"], "parameter_count": model_parameter_count, "encoder_frozen_cache_only": True, "trainable_encoder": False})
        best: dict[str, Any] = {"best_epoch": None, "best_arm": float("inf"), "best_full": float("inf"), "bad_epochs": 0, "stop_reason": None}
        best_payload: dict[str, Any] | None = None
        curves: list[dict[str, Any]] = []
        global_step = 0
        started = time.monotonic()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            pending = 0
            cycle_samples = 0
            total_loss = 0.0
            total_elements = 0
            for batch_index, (batch_context, batch_residual, batch_anchor, _episode, _task) in enumerate(train_loader, start=1):
                batch_context = batch_context.to(device, non_blocking=True)
                batch_residual = batch_residual.to(device, non_blocking=True)
                batch_anchor = batch_anchor.to(device, non_blocking=True)
                if pending == 0:
                    optimizer.zero_grad(set_to_none=True)
                    cycle_samples = 0
                with _autocast(device):
                    predicted_residual, _gates = model(batch_context)
                    predicted_absolute = batch_anchor + predicted_residual
                    target_absolute = batch_anchor + batch_residual
                    loss_sum = F.mse_loss(predicted_absolute.float(), target_absolute.float(), reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("non-finite exploratory loss")
                loss_sum.backward()
                total_loss += float(loss_sum.detach().cpu())
                total_elements += int(batch_residual.numel())
                cycle_samples += int(batch_residual.shape[0])
                pending += 1
                if pending == ACCUMULATION or batch_index == len(train_loader):
                    denominator = cycle_samples * STEPS * DIMS
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(denominator)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite exploratory gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    pending = 0
                del predicted_residual, predicted_absolute, target_absolute, loss_sum
            evaluation = evaluate(model, dev_loader, device, mean, std)
            arm = float(evaluation["metrics"]["arm_full_mae_std"])
            full = float(evaluation["metrics"]["full28_full_mae_std"])
            improved = arm < best["best_arm"] - MIN_DELTA or (abs(arm - best["best_arm"]) <= MIN_DELTA and full < best["best_full"] - MIN_DELTA)
            if improved:
                best.update({"best_epoch": epoch, "best_arm": arm, "best_full": full, "bad_epochs": 0})
                best_payload = {"cell_id": cell_id, "seed": seed, "epoch": epoch, "global_optimizer_step": global_step, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "early_stopping_state": dict(best)}
            else:
                best["bad_epochs"] += 1
            curves.append({"epoch": epoch, "train_loss": total_loss / total_elements, "validation_arm_mae": arm, "validation_full_mae": full, "validation_hand_mae": evaluation["metrics"]["hand_full_mae_std"], "global_optimizer_step": global_step, "best_epoch": best["best_epoch"], "bad_epochs": best["bad_epochs"]})
            atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": seed, "last_completed_epoch": epoch, "best_epoch": best["best_epoch"], "optimizer_steps": global_step, "final_test_access": 0, "updated_utc": now()})
            print(f"POSTFREEZE_PROGRESS cell={cell_id} epoch={epoch}/{MAX_EPOCHS} arm_mae={arm:.6f} full_mae={full:.6f}", flush=True)
            if epoch >= MIN_EPOCHS and best["bad_epochs"] >= PATIENCE:
                best["stop_reason"] = "patience_exhausted"
                break
        if best_payload is None:
            raise RuntimeError("no best exploratory payload")
        if best["stop_reason"] is None:
            best["stop_reason"] = "max_epochs_reached" if len(curves) == MAX_EPOCHS else "completed"
        last_payload = {"cell_id": cell_id, "seed": seed, "epoch": len(curves), "global_optimizer_step": global_step, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "early_stopping_state": dict(best)}
        atomic_torch(run_dir / "checkpoint_best.pt", best_payload)
        atomic_torch(run_dir / "checkpoint_last.pt", last_payload)
        best_loaded = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_loaded["model_state_dict"], strict=True)
        best_eval = evaluate(model, dev_loader, device, mean, std)
        last_loaded = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(last_loaded["model_state_dict"], strict=True)
        last_eval = evaluate(model, dev_loader, device, mean, std)
        for suffix, result in (("best", best_eval), ("last", last_eval)):
            atomic_npz(run_dir / f"predictions_{suffix}.npz", episode_index=result["episode_index"], original_task_index=result["original_task_index"], targets_standardized=result["targets_standardized"], targets_native=result["targets_native"], persistence_standardized=result["persistence_standardized"], persistence_native=result["persistence_native"], predictions_standardized=result["predictions_standardized"], predictions_native=result["predictions_native"], residual_predictions_standardized=(result["predictions_standardized"] - result["persistence_standardized"]), gate_values=result["gate_values"])
        threshold = contracts["thresholds"]
        target_arm = best_eval["targets_standardized"][:, :, 14:28]
        motion = np.abs(np.diff(target_arm, axis=1)).mean(axis=(1, 2))
        q25 = float(threshold["q25"])
        q75 = float(threshold["q75"])
        stratum = np.where(motion < q25, "low", np.where(motion >= q75, "high", "mid"))
        metrics = dict(best_eval["metrics"])
        persistence_error = best_eval["persistence_standardized"] - best_eval["targets_standardized"]
        metrics["arm_relative_skill_vs_persistence"] = float((np.abs(persistence_error[:, :, 14:28]).mean() - metrics["arm_full_mae_std"]) / np.abs(persistence_error[:, :, 14:28]).mean())
        metrics.update(_gate_summary(best_eval["gate_values"]))
        metrics.update({"gate_low_n": int(np.sum(stratum == "low")), "gate_mid_n": int(np.sum(stratum == "mid")), "gate_high_n": int(np.sum(stratum == "high")), "motion_q25_train": q25, "motion_q75_train": q75})
        atomic_json(run_dir / "GATE_SUMMARY.json", {"cell_id": cell_id, "seed": seed, "gated": spec["residual_gate"] != "ungated", "summary": _gate_summary(best_eval["gate_values"]), "time_mean": {"arm": best_eval["gate_values"][:, :, 0].mean(axis=0).tolist(), "hand": best_eval["gate_values"][:, :, 1].mean(axis=0).tolist()}, "motion": motion.tolist(), "stratum": stratum.tolist()})
        atomic_json(run_dir / "RUN_METRICS.json", {"status": "complete", "cell_id": cell_id, "seed": seed, "best_epoch": best["best_epoch"], "last_epoch": len(curves), "stop_reason": best["stop_reason"], "best": metrics, "last": last_eval["metrics"], "optimizer_steps": global_step, "train_count": 2741, "development_count": 614, "raw_video": False, "encoder_or_lora_modified": False, "final_test_access": 0, "elapsed_seconds": time.monotonic() - started, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "checkpoint_last_sha256": digest(run_dir / "checkpoint_last.pt"), "best_prediction_sha256": digest(run_dir / "predictions_best.npz"), "last_prediction_sha256": digest(run_dir / "predictions_last.npz"), "gate_summary": _gate_summary(best_eval["gate_values"])})
        with (run_dir / "epoch_curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
        atomic_json(run_dir / "epoch_curves.json", {"curves": curves})
        atomic_json(status_path, {"status": "complete", "cell_id": cell_id, "seed": seed, "best_epoch": best["best_epoch"], "last_epoch": len(curves), "optimizer_steps": global_step, "final_test_access": 0, "completed_utc": now(), "exit_code": 0})
        atomic_json(run_dir / "RUN_COMPLETE.json", {"status": "complete", "cell_id": cell_id, "seed": seed, "optimizer_steps": global_step, "final_test_access": 0, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz")})
        return {"cell_id": cell_id, "seed": seed, "status": "complete", "best_epoch": best["best_epoch"], "optimizer_steps": global_step, "best": metrics, "path": str(run_dir), "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz")}
    except Exception as exc:
        atomic_json(run_dir / "RUN_FAILED.json", {"status": "failed", "cell_id": cell_id, "seed": seed, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        atomic_json(status_path, {"status": "failed", "cell_id": cell_id, "seed": seed, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=CELL_ORDER, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = train_cell(args.cell, root=args.root, seed=args.seed, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
