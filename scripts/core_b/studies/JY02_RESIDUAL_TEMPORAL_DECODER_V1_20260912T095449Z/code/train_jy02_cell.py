#!/usr/bin/env python3
"""Train one JY-02 cached-feature cell under the locked contract."""

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

from models import DirectChunkDecoder, TemporalGRUDecoder


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
SEED = 20260903
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"

CELL_SPECS: dict[str, dict[str, Any]] = {
    "JY02-S-A-D": {"input": "STATE", "target": "absolute", "decoder": "direct_chunk", "input_dim": 568, "branch": "state"},
    "JY02-S-R-D": {"input": "STATE", "target": "residual_to_persistence", "decoder": "direct_chunk", "input_dim": 568, "branch": "state"},
    "JY02-S-A-T": {"input": "STATE", "target": "absolute", "decoder": "temporal_gru", "input_dim": 568, "branch": "state", "hidden_size": 256},
    "JY02-S-R-T": {"input": "STATE", "target": "residual_to_persistence", "decoder": "temporal_gru", "input_dim": 568, "branch": "state", "hidden_size": 256},
    "JY02-F-A-D": {"input": "FUSION", "target": "absolute", "decoder": "direct_chunk", "input_dim": 1336, "branch": "fusion"},
    "JY02-F-R-D": {"input": "FUSION", "target": "residual_to_persistence", "decoder": "direct_chunk", "input_dim": 1336, "branch": "fusion"},
    "JY02-F-A-T": {"input": "FUSION", "target": "absolute", "decoder": "temporal_gru", "input_dim": 1336, "branch": "fusion", "hidden_size": 256},
    "JY02-F-R-T": {"input": "FUSION", "target": "residual_to_persistence", "decoder": "temporal_gru", "input_dim": 1336, "branch": "fusion", "hidden_size": 256},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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


def build_model(spec: dict[str, Any], seed: int) -> nn.Module:
    if spec["decoder"] == "direct_chunk":
        return DirectChunkDecoder(int(spec["input_dim"]), seed=seed + 101)
    return TemporalGRUDecoder(int(spec["input_dim"]), int(spec["hidden_size"]), seed_base=seed + 201)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, residual: bool) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    persistence: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, target, anchor, episode, task in loader:
            inputs = inputs.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = model(inputs).float()
            absolute = anchor + raw if residual else raw
            predictions.append(absolute.cpu().numpy())
            targets.append(target.numpy())
            persistence.append(anchor.cpu().numpy())
            ids.append(episode.numpy())
            tasks.append(task.numpy())
    pred = np.concatenate(predictions, axis=0).astype(np.float32)
    target = np.concatenate(targets, axis=0).astype(np.float32)
    anchor = np.concatenate(persistence, axis=0).astype(np.float32)
    episode_ids = np.concatenate(ids, axis=0).astype(np.int64)
    task_ids = np.concatenate(tasks, axis=0).astype(np.int64)
    error = pred - target
    abs_error = np.abs(error)
    metrics = {
        "arm_mae": float(abs_error[:, :, 14:28].mean()),
        "hand_mae": float(abs_error[:, :, :14].mean()),
        "full_mae": float(abs_error.mean()),
        "full_rmse": float(np.sqrt(np.mean(error**2))),
        "arm_early_mae": float(abs_error[:, :10, 14:28].mean()),
        "arm_middle_mae": float(abs_error[:, 10:20, 14:28].mean()),
        "arm_late_mae": float(abs_error[:, 20:, 14:28].mean()),
        "arm_endpoint_mae": float(abs_error[:, -1, 14:28].mean()),
        "arm_first_difference_mae": float(np.abs(np.diff(pred[:, :, 14:28], axis=1) - np.diff(target[:, :, 14:28], axis=1)).mean()),
        "native_tv_ratio": float(np.mean(np.divide(np.abs(np.diff(pred[:, :, 14:28], axis=1)).sum(axis=1).mean(axis=1), np.abs(np.diff(target[:, :, 14:28], axis=1)).sum(axis=1).mean(axis=1), out=np.full(len(pred), np.nan), where=np.abs(np.diff(target[:, :, 14:28], axis=1)).sum(axis=1).mean(axis=1) > 1e-12))),
        "all_finite": bool(np.isfinite(pred).all()),
    }
    return {"predictions_standardized": pred, "targets_standardized": target, "persistence_standardized": anchor, "episode_index": episode_ids, "original_task_index": task_ids, "metrics": metrics}


def train_cell(cell_id: str, *, root: Path, seed: int = SEED, max_epochs: int = MAX_EPOCHS, device: torch.device | None = None, run_group: str = "stage1") -> dict[str, Any]:
    if cell_id not in CELL_SPECS:
        raise ValueError(cell_id)
    if seed <= 0:
        raise ValueError("seed must be positive")
    spec = CELL_SPECS[cell_id]
    run_dir = root / "runs" / run_group / cell_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": seed, "optimizer_steps": 0, "final_test_access": 0, "started_utc": now()})
    try:
        set_seed(seed)
        device = device or torch.device("cuda:0")
        data_path = root / "inputs/JY02_TARGETS.npz"
        with np.load(data_path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        split = data["split_code"]
        train_mask = split == 0
        dev_mask = split == 1
        if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
            raise RuntimeError("JY02 train/dev count contract failed")
        state = data["state_action_history"].astype(np.float32)
        dynamic = data["dynamic_pooled"].astype(np.float32)
        inputs = state if spec["input"] == "STATE" else np.concatenate([dynamic, state], axis=1)
        targets = data["targets_standardized"].astype(np.float32)
        anchor = data["persistence_standardized"].astype(np.float32)
        target_for_training = targets - anchor if spec["target"] == "residual_to_persistence" else targets
        ids = data["episode_index"].astype(np.int64)
        tasks = data["original_task_index"].astype(np.int64)
        train_dataset = TensorDataset(torch.from_numpy(inputs[train_mask]), torch.from_numpy(target_for_training[train_mask]), torch.from_numpy(anchor[train_mask]), torch.from_numpy(ids[train_mask]), torch.from_numpy(tasks[train_mask]))
        dev_dataset = TensorDataset(torch.from_numpy(inputs[dev_mask]), torch.from_numpy(targets[dev_mask]), torch.from_numpy(anchor[dev_mask]), torch.from_numpy(ids[dev_mask]), torch.from_numpy(tasks[dev_mask]))
        train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=True)
        dev_loader = DataLoader(dev_dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0, pin_memory=True)
        model = build_model(spec, seed).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected_parameter_count = 361544 if spec["input"] == "STATE" and spec["decoder"] == "direct_chunk" else 558152 if spec["input"] == "FUSION" and spec["decoder"] == "direct_chunk" else 402076 if spec["input"] == "STATE" else 598684
        if parameter_count != expected_parameter_count:
            raise RuntimeError(f"parameter count mismatch {parameter_count} != {expected_parameter_count}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(math.ceil(len(train_dataset) / PHYSICAL_BATCH) / ACCUMULATION)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * steps_per_epoch)
        config = {"protocol_id": "core_b_jy02_residual_temporal_decoder_v1_20260912", "cell_id": cell_id, **spec, "seed": seed, "component_seeds": {"direct": seed + 101, "temporal_context": seed + 202, "temporal_embedding": seed + 203, "temporal_gru": seed + 204, "temporal_output": seed + 205}, "input_path": str(data_path), "input_sha256": digest(data_path), "target_mode": spec["target"], "physical_batch": PHYSICAL_BATCH, "gradient_accumulation": ACCUMULATION, "nominal_effective_batch": EFFECTIVE_BATCH, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": 1.0, "scheduler": "CosineAnnealingLR", "steps_per_epoch": steps_per_epoch, "max_epochs": max_epochs, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "precision": "BF16 autocast; FP32 parameters", "encoder_or_lora_modified": False, "raw_video": False, "final_test_access": 0, "selection_metric": "arm14 full-horizon MAE then full28 MAE then earlier epoch", "parameter_count": parameter_count}
        atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
        atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "model": spec["decoder"], "input": spec["input"], "target": spec["target"], "parameter_count": parameter_count, "encoder_frozen_cache_only": True, "trainable_encoder": False})
        early: dict[str, Any] = {"best_epoch": None, "best_arm_mae": float("inf"), "best_full_mae": float("inf"), "bad_epochs": 0, "stop_reason": None}
        best_payload: dict[str, Any] | None = None
        curves: list[dict[str, Any]] = []
        global_step = 0
        started = time.monotonic()
        for epoch in range(1, max_epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True); pending = 0; cycle_samples = 0; total_loss = 0.0; total_elements = 0
            for batch_index, (batch_inputs, batch_target, batch_anchor, _episode, _task) in enumerate(train_loader, start=1):
                batch_inputs = batch_inputs.to(device, non_blocking=True); batch_target = batch_target.to(device, non_blocking=True); batch_anchor = batch_anchor.to(device, non_blocking=True)
                if pending == 0:
                    optimizer.zero_grad(set_to_none=True); cycle_samples = 0
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw = model(batch_inputs)
                    absolute = batch_anchor + raw if spec["target"] == "residual_to_persistence" else raw
                    loss_sum = F.mse_loss(absolute.float(), (batch_anchor + batch_target if spec["target"] == "residual_to_persistence" else batch_target).float(), reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("non-finite JY02 loss")
                loss_sum.backward(); total_loss += float(loss_sum.detach().cpu()); total_elements += int(batch_target.numel()); cycle_samples += int(batch_target.shape[0]); pending += 1
                if pending == ACCUMULATION or batch_index == len(train_loader):
                    denominator = cycle_samples * STEPS * DIMS
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(denominator)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite JY02 gradient")
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1; pending = 0
                del raw, absolute, loss_sum, batch_inputs, batch_target, batch_anchor
            evaluation = evaluate(model, dev_loader, device, spec["target"] == "residual_to_persistence")
            arm = float(evaluation["metrics"]["arm_mae"]); full = float(evaluation["metrics"]["full_mae"])
            improved = arm < early["best_arm_mae"] - MIN_DELTA or (abs(arm - early["best_arm_mae"]) <= MIN_DELTA and full < early["best_full_mae"] - MIN_DELTA)
            if improved:
                early.update({"best_epoch": epoch, "best_arm_mae": arm, "best_full_mae": full, "bad_epochs": 0})
                best_payload = {"cell_id": cell_id, "seed": seed, "epoch": epoch, "global_optimizer_step": global_step, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "early_stopping_state": dict(early)}
            else:
                early["bad_epochs"] += 1
            curves.append({"epoch": epoch, "train_loss": total_loss / total_elements, "validation_arm_mae": arm, "validation_full_mae": full, "validation_hand_mae": evaluation["metrics"]["hand_mae"], "global_optimizer_step": global_step, "best_epoch": early["best_epoch"], "bad_epochs": early["bad_epochs"]})
            atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": seed, "last_completed_epoch": epoch, "best_epoch": early["best_epoch"], "optimizer_steps": global_step, "final_test_access": 0, "updated_utc": now()})
            print(f"JY02_PROGRESS cell={cell_id} epoch={epoch}/{max_epochs} arm_mae={arm:.6f} full_mae={full:.6f}", flush=True)
            if epoch >= MIN_EPOCHS and early["bad_epochs"] >= PATIENCE:
                early["stop_reason"] = "patience_exhausted"; break
        if best_payload is None:
            raise RuntimeError("no best JY02 payload")
        if early["stop_reason"] is None:
            early["stop_reason"] = "max_epochs_reached" if len(curves) == max_epochs else "completed"
        last_payload = {"cell_id": cell_id, "seed": seed, "epoch": len(curves), "global_optimizer_step": global_step, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "early_stopping_state": dict(early)}
        atomic_torch(run_dir / "checkpoint_best.pt", best_payload); atomic_torch(run_dir / "checkpoint_last.pt", last_payload)
        best_loaded = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False); model.load_state_dict(best_loaded["model_state_dict"], strict=True); best_eval = evaluate(model, dev_loader, device, spec["target"] == "residual_to_persistence")
        last_loaded = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False); model.load_state_dict(last_loaded["model_state_dict"], strict=True); last_eval = evaluate(model, dev_loader, device, spec["target"] == "residual_to_persistence")
        norm = json.loads((REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text(encoding="utf-8"))["1.0s"]
        target_mean = np.asarray(norm["mean"], dtype=np.float32); target_std = np.asarray(norm["safe_std"], dtype=np.float32)
        for suffix, result in (("best", best_eval), ("last", last_eval)):
            prediction_native = result["predictions_standardized"] * target_std[None, :, :] + target_mean[None, :, :]
            target_native = result["targets_standardized"] * target_std[None, :, :] + target_mean[None, :, :]
            raw_prediction = result["predictions_standardized"] - result["persistence_standardized"] if spec["target"] == "residual_to_persistence" else result["predictions_standardized"]
            atomic_npz(run_dir / f"predictions_{suffix}.npz", episode_index=result["episode_index"], original_task_index=result["original_task_index"], targets_standardized=result["targets_standardized"], targets_native=target_native.astype(np.float32), persistence_standardized=result["persistence_standardized"], predictions_standardized=result["predictions_standardized"], predictions_native=prediction_native.astype(np.float32), residual_predictions_standardized=raw_prediction.astype(np.float32))
        atomic_json(run_dir / "RUN_METRICS.json", {"status": "complete", "cell_id": cell_id, "seed": seed, "best_epoch": early["best_epoch"], "last_epoch": len(curves), "stop_reason": early["stop_reason"], "best": best_eval["metrics"], "last": last_eval["metrics"], "optimizer_steps": global_step, "train_count": 2741, "development_count": 614, "raw_video": False, "encoder_or_lora_modified": False, "final_test_access": 0, "elapsed_seconds": time.monotonic() - started, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "checkpoint_last_sha256": digest(run_dir / "checkpoint_last.pt"), "best_prediction_sha256": digest(run_dir / "predictions_best.npz"), "last_prediction_sha256": digest(run_dir / "predictions_last.npz"), "residual_reconstruction_max_abs": float(np.max(np.abs(best_eval["predictions_standardized"] - best_eval["targets_standardized"]))) if spec["target"] == "absolute" else float(np.max(np.abs((best_eval["persistence_standardized"] + best_eval["predictions_standardized"] - best_eval["persistence_standardized"]) - best_eval["predictions_standardized"])))})
        with (run_dir / "epoch_curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
        atomic_json(run_dir / "epoch_curves.json", {"curves": curves})
        atomic_json(status_path, {"status": "complete", "cell_id": cell_id, "seed": seed, "best_epoch": early["best_epoch"], "last_epoch": len(curves), "optimizer_steps": global_step, "final_test_access": 0, "completed_utc": now(), "exit_code": 0})
        atomic_json(run_dir / "RUN_COMPLETE.json", {"status": "complete", "cell_id": cell_id, "seed": seed, "optimizer_steps": global_step, "final_test_access": 0, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz")})
        return {"cell_id": cell_id, "status": "complete", "best_epoch": early["best_epoch"], "optimizer_steps": global_step, "best": best_eval["metrics"], "last": last_eval["metrics"], "path": str(run_dir), "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz")}
    except Exception as exc:
        atomic_json(run_dir / "RUN_FAILED.json", {"status": "failed", "cell_id": cell_id, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        atomic_json(status_path, {"status": "failed", "cell_id": cell_id, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cell", choices=tuple(CELL_SPECS), required=True); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--seed", type=int, default=SEED); parser.add_argument("--run-group", default="stage1"); args = parser.parse_args()
    result = train_cell(args.cell, root=args.root, seed=args.seed, run_group=args.run_group)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
