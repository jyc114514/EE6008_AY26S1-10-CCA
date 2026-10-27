#!/usr/bin/env python3
"""Train one locked Hand event-gate exploratory cell."""

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

from models import build_model, base_parameter_count, split_hidden_for_parameter_match


PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
JY02_ROOT = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
INPUT_PATH = JY02_ROOT / "inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_PATH = ROOT / "MOTION_THRESHOLDS_LOCKED.json"

PHYSICAL_BATCH = 8
ACCUMULATION = 4
EFFECTIVE_BATCH = 32
MAX_EPOCHS = 20
OPTIMIZER_STEPS_EXPECTED = 1720
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 1.0
STEPS = 30
DIMS = 28
HAND = slice(0, 14)
ARM = slice(14, 28)
SEED = 20260925
BOOTSTRAP = 10_000
STATS_SEED = 20260927

CELL_SPECS: dict[str, dict[str, Any]] = {
    "H-S-U": {"input": "STATE", "input_dim": 568, "event_mode": "U"},
    "H-S-W": {"input": "STATE", "input_dim": 568, "event_mode": "W"},
    "H-S-T": {"input": "STATE", "input_dim": 568, "event_mode": "T"},
    "H-F-U": {"input": "FUSION", "input_dim": 1336, "event_mode": "U"},
    "H-F-W": {"input": "FUSION", "input_dim": 1336, "event_mode": "W"},
    "H-F-T": {"input": "FUSION", "input_dim": 1336, "event_mode": "T"},
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


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def group_step_motion(target: np.ndarray, persistence: np.ndarray, group: slice) -> np.ndarray:
    first = np.abs(target[:, 0, group] - persistence[:, 0, group]).mean(axis=1)
    transitions = np.abs(np.diff(target[:, :, group], axis=1)).mean(axis=2)
    return np.concatenate((first[:, None], transitions), axis=1)


def load_contract_data() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if not INPUT_PATH.is_file() or not NORMALIZER_PATH.is_file() or not THRESHOLD_PATH.is_file():
        raise FileNotFoundError("locked development input, normalizer or motion threshold registry is missing")
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("JY-02 train/dev count contract failed")
    if data["targets_standardized"].shape != (3355, STEPS, DIMS):
        raise RuntimeError("JY-02 target shape contract failed")
    normalizer = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))["1.0s"]
    threshold_registry = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    if threshold_registry.get("source_split") != "train" or threshold_registry.get("source_rows") != 2741:
        raise RuntimeError("motion threshold registry is not train-only")
    thresholds = {key: float(value) for key, value in threshold_registry["thresholds"].items()}
    train_target = data["targets_standardized"][train_mask].astype(np.float64)
    train_persistence = data["persistence_standardized"][train_mask].astype(np.float64)
    train_hand_motion = group_step_motion(train_target, train_persistence, HAND)
    pooled = np.concatenate((group_step_motion(train_target, train_persistence, ARM).reshape(-1), train_hand_motion.reshape(-1)))
    expected = {
        "COMMON_STEP_Q25": float(np.quantile(pooled, 0.25)),
        "COMMON_STEP_Q75": float(np.quantile(pooled, 0.75)),
        "HAND_STEP_Q75": float(np.quantile(train_hand_motion.reshape(-1), 0.75)),
        "HAND_EPISODE_MEAN_Q75": float(np.quantile(train_hand_motion.mean(axis=1), 0.75)),
    }
    for key, value in expected.items():
        if not math.isclose(thresholds[key], value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"locked threshold mismatch for {key}")
    return data, train_mask, dev_mask, {"normalizer": normalizer, "thresholds": thresholds}


def event_labels(target: np.ndarray, persistence: np.ndarray, thresholds: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = group_step_motion(target.astype(np.float64), persistence.astype(np.float64), HAND)
    step = (motion >= thresholds["HAND_STEP_Q75"]).astype(np.float32)
    window = (motion.mean(axis=1) >= thresholds["HAND_EPISODE_MEAN_Q75"]).astype(np.float32)
    return motion.astype(np.float32), step, window


def event_prevalence(step_labels: np.ndarray, window_labels: np.ndarray, mode: str) -> float:
    value = float(window_labels.mean() if mode == "W" else step_labels.mean())
    if mode == "U":
        value = float(window_labels.mean())
    if not 0.0 < value < 1.0:
        raise RuntimeError(f"{mode} event prevalence is degenerate: {value}")
    return value


def build_cell_model(cell_id: str, seed: int, train_prevalence: float) -> tuple[nn.Module, int, int, int]:
    spec = CELL_SPECS[cell_id]
    model, hidden_dim, parameter_count = build_model(int(spec["input_dim"]), str(spec["event_mode"]), seed, train_prevalence)
    base_count = base_parameter_count(int(spec["input_dim"]), hidden_dim)
    added = parameter_count - base_count
    if added < 0 or added > 0.05 * base_count + 1e-9:
        raise RuntimeError(f"event branch parameter budget failed for {cell_id}: added={added}, base={base_count}")
    return model, hidden_dim, parameter_count, added


def episode_metrics(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - target
    persistence_error = persistence - target
    native_error = pred_native - target_native
    result: dict[str, np.ndarray] = {
        "arm_mae": np.abs(error[:, :, ARM]).mean(axis=(1, 2)),
        "hand_mae": np.abs(error[:, :, HAND]).mean(axis=(1, 2)),
        "full_mae": np.abs(error).mean(axis=(1, 2)),
        "arm_early": np.abs(error[:, :10, ARM]).mean(axis=(1, 2)),
        "arm_middle": np.abs(error[:, 10:20, ARM]).mean(axis=(1, 2)),
        "arm_late": np.abs(error[:, 20:, ARM]).mean(axis=(1, 2)),
        "arm_endpoint": np.abs(error[:, -1, ARM]).mean(axis=1),
        "hand_early": np.abs(error[:, :10, HAND]).mean(axis=(1, 2)),
        "hand_middle": np.abs(error[:, 10:20, HAND]).mean(axis=(1, 2)),
        "hand_late": np.abs(error[:, 20:, HAND]).mean(axis=(1, 2)),
        "hand_endpoint": np.abs(error[:, -1, HAND]).mean(axis=1),
        "full_early": np.abs(error[:, :10]).mean(axis=(1, 2)),
        "full_middle": np.abs(error[:, 10:20]).mean(axis=(1, 2)),
        "full_late": np.abs(error[:, 20:]).mean(axis=(1, 2)),
        "full_endpoint": np.abs(error[:, -1]).mean(axis=1),
        "arm_first_difference": np.abs(np.diff(pred[:, :, ARM], axis=1) - np.diff(target[:, :, ARM], axis=1)).mean(axis=(1, 2)),
        "hand_first_difference": np.abs(np.diff(pred[:, :, HAND], axis=1) - np.diff(target[:, :, HAND], axis=1)).mean(axis=(1, 2)),
        "full_first_difference": np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)).mean(axis=(1, 2)),
        "persistence_arm_mae": np.abs(persistence_error[:, :, ARM]).mean(axis=(1, 2)),
        "persistence_hand_mae": np.abs(persistence_error[:, :, HAND]).mean(axis=(1, 2)),
        "persistence_full_mae": np.abs(persistence_error).mean(axis=(1, 2)),
    }
    for name, pred_group, target_group in (
        ("arm", pred_native[:, :, ARM], target_native[:, :, ARM]),
        ("hand", pred_native[:, :, HAND], target_native[:, :, HAND]),
        ("full", pred_native, target_native),
    ):
        pred_tv = np.abs(np.diff(pred_group, axis=1)).sum(axis=1).mean(axis=1)
        target_tv = np.abs(np.diff(target_group, axis=1)).sum(axis=1).mean(axis=1)
        ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 1e-12)
        result[f"{name}_native_tv_ratio"] = ratio
        result[f"{name}_native_tv_closeness"] = np.abs(ratio - 1.0)
    return result


def evaluate(model: nn.Module, contexts: np.ndarray, targets: np.ndarray, persistence: np.ndarray, ids: np.ndarray, tasks: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(contexts), PHYSICAL_BATCH):
            batch_context = torch.from_numpy(contexts[start:start + PHYSICAL_BATCH]).to(device, non_blocking=True)
            batch_anchor = torch.from_numpy(persistence[start:start + PHYSICAL_BATCH]).to(device, non_blocking=True)
            with autocast(device):
                residual, gate, event_probability = model(batch_context)
            predictions.append((batch_anchor + residual.float()).cpu().numpy())
            gates.append(gate.float().cpu().numpy())
            if event_probability is None:
                event_probabilities.append(gate[:, :, 1].float().cpu().numpy())
            elif event_probability.ndim == 1:
                event_probabilities.append(event_probability[:, None].expand(-1, STEPS).float().cpu().numpy())
            else:
                event_probabilities.append(event_probability.float().cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float32)
    gate_values = np.concatenate(gates).astype(np.float32)
    event_probability = np.concatenate(event_probabilities).astype(np.float32)
    target = targets.astype(np.float32)
    anchor = persistence.astype(np.float32)
    pred_native = pred * std[None, :, :] + mean[None, :, :]
    target_native = target * std[None, :, :] + mean[None, :, :]
    anchor_native = anchor * std[None, :, :] + mean[None, :, :]
    per = episode_metrics(pred, target, anchor, pred_native, target_native, anchor_native)
    return {
        "predictions_standardized": pred,
        "targets_standardized": target,
        "persistence_standardized": anchor,
        "predictions_native": pred_native.astype(np.float32),
        "targets_native": target_native.astype(np.float32),
        "persistence_native": anchor_native.astype(np.float32),
        "gate_values": gate_values,
        "event_probability": event_probability,
        "episode_index": ids.astype(np.int64),
        "original_task_index": tasks.astype(np.int64),
        "per": per,
    }


def aggregate_metrics(evaluation: dict[str, Any]) -> dict[str, float | bool]:
    per = evaluation["per"]
    result: dict[str, float | bool] = {}
    names = (
        "arm_mae", "hand_mae", "full_mae", "arm_early", "arm_middle", "arm_late", "arm_endpoint",
        "hand_early", "hand_middle", "hand_late", "hand_endpoint", "full_early", "full_middle", "full_late", "full_endpoint",
        "arm_first_difference", "hand_first_difference", "full_first_difference", "arm_native_tv_ratio", "hand_native_tv_ratio", "full_native_tv_ratio",
        "arm_native_tv_closeness", "hand_native_tv_closeness", "full_native_tv_closeness", "persistence_arm_mae", "persistence_hand_mae", "persistence_full_mae",
    )
    for name in names:
        result[name] = float(np.nanmean(per[name]))
    result["arm_skill_vs_persistence"] = float((result["persistence_arm_mae"] - result["arm_mae"]) / result["persistence_arm_mae"])
    result["hand_skill_vs_persistence"] = float((result["persistence_hand_mae"] - result["hand_mae"]) / result["persistence_hand_mae"])
    result["full_skill_vs_persistence"] = float((result["persistence_full_mae"] - result["full_mae"]) / result["persistence_full_mae"])
    result["all_finite"] = bool(np.isfinite(evaluation["predictions_standardized"]).all() and np.isfinite(evaluation["predictions_native"]).all() and np.isfinite(evaluation["gate_values"]).all() and np.isfinite(evaluation["event_probability"]).all())
    return result


def gate_summary(gates: np.ndarray, event_probability: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, index in (("arm", 0), ("hand", 1)):
        values = gates[:, :, index].reshape(-1)
        entropy = -(values * np.log(np.clip(values, 1e-7, 1 - 1e-7)) + (1 - values) * np.log(np.clip(1 - values, 1e-7, 1 - 1e-7)))
        result.update({
            f"gate_{name}_mean": float(values.mean()),
            f"gate_{name}_p05": float(np.quantile(values, 0.05)),
            f"gate_{name}_p25": float(np.quantile(values, 0.25)),
            f"gate_{name}_p50": float(np.quantile(values, 0.50)),
            f"gate_{name}_p75": float(np.quantile(values, 0.75)),
            f"gate_{name}_p95": float(np.quantile(values, 0.95)),
            f"gate_{name}_collapse_fraction": float(np.mean((values < 0.05) | (values > 0.95))),
            f"gate_{name}_entropy": float(entropy.mean()),
        })
    event_values = event_probability.reshape(-1)
    result.update({
        "event_q_mean": float(event_values.mean()),
        "event_q_p05": float(np.quantile(event_values, 0.05)),
        "event_q_p50": float(np.quantile(event_values, 0.50)),
        "event_q_p95": float(np.quantile(event_values, 0.95)),
        "event_q_collapse_fraction": float(np.mean((event_values < 0.05) | (event_values > 0.95))),
        "event_q_low_fraction": float(np.mean(event_values < 0.05)),
        "event_q_high_fraction": float(np.mean(event_values > 0.95)),
    })
    return result


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not len(labels):
        return float("nan")
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    cumulative = np.cumsum(ordered_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * ordered_labels) / positives)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def bootstrap_event(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise RuntimeError("event bootstrap pairing failure")
    point = average_precision(labels, scores)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP):
        indices = rng.integers(0, len(labels), size=len(labels))
        value = average_precision(labels[indices], scores[indices])
        if np.isfinite(value):
            estimates.append(value)
    return {"effect": float(point), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "replicates": BOOTSTRAP, "N_episode": int(len(labels))}


def event_diagnostics(evaluation: dict[str, Any], step_labels: np.ndarray, window_labels: np.ndarray, motion: np.ndarray, tasks: np.ndarray, thresholds: dict[str, float], rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    q = evaluation["event_probability"].astype(np.float64)
    mode = str(evaluation["event_mode"])
    if mode == "W":
        primary_scores = q[:, 0]
    elif mode == "T":
        primary_scores = q.max(axis=1)
    else:
        primary_scores = q.mean(axis=1)
    primary = bootstrap_event(window_labels.astype(np.int8), primary_scores, rng)
    event_summary: dict[str, Any] = {
        "event_mode": mode,
        "primary_definition": "W q_hand scalar; T max_t(q_hand[t]); U mean_t(existing_hand_gate)",
        "train_prevalence": float(evaluation["train_event_prevalence"]),
        "development_window_prevalence": float(window_labels.mean()),
        "development_step_prevalence": float(step_labels.mean()),
        "primary_auprc": primary,
        "primary_auprc_prevalence_baseline": float(window_labels.mean()),
        "primary_auroc": auroc(window_labels, primary_scores),
        "primary_brier": float(np.mean((primary_scores - window_labels) ** 2)),
        "descriptive_step_auprc": average_precision(step_labels, q),
        "descriptive_step_auroc": auroc(step_labels, q),
        "descriptive_step_brier": float(np.mean((q - step_labels) ** 2)),
    }
    calibration_rows: list[dict[str, object]] = []
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0 if index < 9 else 1.0
        mask = (primary_scores >= low) & (primary_scores <= high if index == 9 else primary_scores < high)
        calibration_rows.append({"event_mode": mode, "bin": index, "lower": low, "upper": high, "N": int(mask.sum()), "mean_score": float(primary_scores[mask].mean()) if mask.any() else float("nan"), "observed_window_prevalence": float(window_labels[mask].mean()) if mask.any() else float("nan")})
    common_low = thresholds["COMMON_STEP_Q25"]
    common_high = thresholds["COMMON_STEP_Q75"]
    strata = np.where(motion.mean(axis=1) < common_low, "low", np.where(motion.mean(axis=1) >= common_high, "high", "mid"))
    task_rows: list[dict[str, object]] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        task_rows.append({"event_mode": mode, "stratum_type": "task", "stratum": int(task), "N": int(mask.sum()), "mean_primary_score": float(primary_scores[mask].mean()), "window_prevalence": float(window_labels[mask].mean())})
    motion_rows: list[dict[str, object]] = []
    for stratum in ("low", "mid", "high"):
        mask = strata == stratum
        motion_rows.append({"event_mode": mode, "stratum_type": "motion", "stratum": stratum, "N": int(mask.sum()), "mean_primary_score": float(primary_scores[mask].mean()) if mask.any() else float("nan"), "window_prevalence": float(window_labels[mask].mean()) if mask.any() else float("nan"), "mean_step_score": float(q[mask].mean()) if mask.any() else float("nan")})
    time_rows = [{"event_mode": mode, "time_index": int(t), "mean_event_probability": float(q[:, t].mean()), "observed_step_prevalence": float(step_labels[:, t].mean())} for t in range(STEPS)]
    return event_summary, calibration_rows, task_rows + motion_rows, time_rows


def train_cell(cell_id: str, root: Path, seed: int = SEED, device: torch.device | None = None) -> dict[str, Any]:
    if cell_id not in CELL_SPECS:
        raise ValueError(cell_id)
    if seed != SEED:
        raise ValueError(f"this locked matrix requires seed {SEED}")
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
        thresholds = contracts["thresholds"]
        state = data["state_action_history"].astype(np.float32)
        dynamic = data["dynamic_pooled"].astype(np.float32)
        context_all = state if CELL_SPECS[cell_id]["input"] == "STATE" else np.concatenate((dynamic, state), axis=1)
        target_all = data["targets_standardized"].astype(np.float32)
        persistence_all = data["persistence_standardized"].astype(np.float32)
        ids_all = data["episode_index"].astype(np.int64)
        tasks_all = data["original_task_index"].astype(np.int64)
        _train_motion, train_step_labels, train_window_labels = event_labels(target_all[train_mask], persistence_all[train_mask], thresholds)
        dev_motion, dev_step_labels, dev_window_labels = event_labels(target_all[dev_mask], persistence_all[dev_mask], thresholds)
        mode = str(CELL_SPECS[cell_id]["event_mode"])
        train_prevalence = event_prevalence(train_step_labels, train_window_labels, mode)
        positive_weight = float((1.0 - train_prevalence) / train_prevalence) if mode in {"W", "T"} else 1.0
        train_dataset = TensorDataset(
            torch.from_numpy(context_all[train_mask]),
            torch.from_numpy(target_all[train_mask] - persistence_all[train_mask]),
            torch.from_numpy(persistence_all[train_mask]),
            torch.from_numpy(ids_all[train_mask]),
            torch.from_numpy(tasks_all[train_mask]),
            torch.from_numpy(train_step_labels),
            torch.from_numpy(train_window_labels),
        )
        train_loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")
        model_seed = seed + CELL_ORDER.index(cell_id) * 1000
        model, hidden_dim, parameter_count, added_parameters = build_cell_model(cell_id, model_seed, train_prevalence)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION)
        if steps_per_epoch * MAX_EPOCHS != OPTIMIZER_STEPS_EXPECTED:
            raise RuntimeError(f"optimizer step contract failed: {steps_per_epoch * MAX_EPOCHS}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * steps_per_epoch)
        config = {
            "protocol_id": "core_b_hand_motion_event_fallback_v1_20260913",
            "cell_id": cell_id,
            "input": CELL_SPECS[cell_id]["input"],
            "input_dim": CELL_SPECS[cell_id]["input_dim"],
            "head_coupling": "split_arm_hand",
            "event_mode": mode,
            "seed": seed,
            "component_seeds": {"model": model_seed, "loader": seed + 17},
            "input_path": str(INPUT_PATH),
            "input_sha256": digest(INPUT_PATH),
            "normalizer_path": str(NORMALIZER_PATH),
            "normalizer_sha256": digest(NORMALIZER_PATH),
            "threshold_path": str(THRESHOLD_PATH),
            "threshold_sha256": digest(THRESHOLD_PATH),
            "target_mode": "residual_to_persistence",
            "target_shape": [STEPS, DIMS],
            "hand_slice": "0:14",
            "arm_slice": "14:28",
            "physical_batch": PHYSICAL_BATCH,
            "gradient_accumulation": ACCUMULATION,
            "nominal_effective_batch": EFFECTIVE_BATCH,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": "CosineAnnealingLR",
            "steps_per_epoch": steps_per_epoch,
            "max_epochs": MAX_EPOCHS,
            "optimizer_steps_expected": OPTIMIZER_STEPS_EXPECTED,
            "precision": "BF16 autocast; FP32 parameters",
            "hidden_dim": hidden_dim,
            "parameter_count": parameter_count,
            "base_u_parameter_count": base_parameter_count(int(CELL_SPECS[cell_id]["input_dim"]), hidden_dim),
            "added_event_parameters": added_parameters,
            "added_event_parameter_fraction": added_parameters / base_parameter_count(int(CELL_SPECS[cell_id]["input_dim"]), hidden_dim),
            "event_lambda": 0.0 if mode == "U" else 0.1,
            "train_event_prevalence": train_prevalence,
            "event_positive_weight": positive_weight,
            "event_loss": "class-weighted BCE; detached context for event branch" if mode != "U" else "disabled",
            "event_label_input_to_model": False,
            "event_future_label_read_at_inference": False,
            "arm_gate": "existing unsupervised group-time Arm gate unchanged",
            "encoder_or_lora_modified": False,
            "raw_video": False,
            "final_test_access": 0,
            "selection_metric": "Arm-14 full-horizon standardized MAE, then Full-28 MAE, fixed 20 epochs",
        }
        atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
        atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "model": "split_residual_with_context_only_hand_event_gate", "event_mode": mode, "strict_output_shape": [STEPS, DIMS], "parameter_count": parameter_count, "added_event_parameters": added_parameters, "added_parameter_budget_pass": added_parameters <= 0.05 * config["base_u_parameter_count"], "encoder_frozen_cache_only": True, "event_loss_context_detached": mode != "U", "event_label_as_model_input": False, "final_test_access": 0})
        best: dict[str, Any] = {"best_epoch": None, "best_arm": float("inf"), "best_full": float("inf")}
        best_payload: dict[str, Any] | None = None
        curves: list[dict[str, Any]] = []
        global_step = 0
        started = time.monotonic()
        event_pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rec_total = 0.0
            event_total = 0.0
            batches = 0
            for batch_index, (_context, _residual, _anchor, _episode, _task, batch_step_labels, batch_window_labels) in enumerate(train_loader, start=1):
                context = _context.to(device, non_blocking=True)
                residual_target = _residual.to(device, non_blocking=True)
                anchor = _anchor.to(device, non_blocking=True)
                step_labels = batch_step_labels.to(device, non_blocking=True)
                window_labels = batch_window_labels.to(device, non_blocking=True)
                with autocast(device):
                    residual, _gates, _event_probability = model(context)
                    prediction = anchor + residual
                    target_absolute = anchor + residual_target
                    rec_loss = F.mse_loss(prediction.float(), target_absolute.float(), reduction="mean")
                    if mode == "W":
                        event_logits = model.event_logits(context.detach())
                        assert event_logits is not None
                        event_loss = F.binary_cross_entropy_with_logits(event_logits.float(), window_labels.float(), pos_weight=event_pos_weight)
                    elif mode == "T":
                        event_logits = model.event_logits(context.detach())
                        assert event_logits is not None
                        event_loss = F.binary_cross_entropy_with_logits(event_logits.float(), step_labels.float(), pos_weight=event_pos_weight)
                    else:
                        event_loss = torch.zeros((), device=device, dtype=rec_loss.dtype)
                    total_loss = rec_loss + (0.1 if mode != "U" else 0.0) * event_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("non-finite event exploratory loss")
                (total_loss / ACCUMULATION).backward()
                rec_total += float(rec_loss.detach().cpu())
                event_total += float(event_loss.detach().cpu())
                batches += 1
                if batch_index % ACCUMULATION == 0 or batch_index == len(train_loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite event exploratory gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                del prediction, target_absolute, residual, total_loss
            # Development target labels are not passed into the model; they are
            # only used after forward evaluation for event diagnostics.
            dev_eval = evaluate(model, context_all[dev_mask], target_all[dev_mask], persistence_all[dev_mask], ids_all[dev_mask], tasks_all[dev_mask], mean, std, device)
            dev_eval["event_mode"] = mode
            dev_eval["train_event_prevalence"] = train_prevalence
            agg = aggregate_metrics(dev_eval)
            arm = float(agg["arm_mae"])
            full = float(agg["full_mae"])
            improved = arm < best["best_arm"] - 0.0001 or (abs(arm - best["best_arm"]) <= 0.0001 and full < best["best_full"] - 0.0001)
            if improved:
                best.update({"best_epoch": epoch, "best_arm": arm, "best_full": full})
                best_payload = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            curves.append({"epoch": epoch, "train_reconstruction_mse": rec_total / batches, "train_event_bce": event_total / batches, "validation_arm_mae": arm, "validation_full_mae": full, "validation_hand_mae": agg["hand_mae"], "global_optimizer_step": global_step, "best_epoch": best["best_epoch"], "learning_rate": scheduler.get_last_lr()[0]})
            atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": seed, "last_completed_epoch": epoch, "best_epoch": best["best_epoch"], "optimizer_steps": global_step, "final_test_access": 0, "updated_utc": now()})
            print(f"EVENT_PROGRESS cell={cell_id} epoch={epoch}/{MAX_EPOCHS} arm_mae={arm:.6f} full_mae={full:.6f} event_bce={event_total / batches:.6f}", flush=True)
        if global_step != OPTIMIZER_STEPS_EXPECTED or best_payload is None:
            raise RuntimeError(f"fixed optimizer step or best checkpoint contract failed: steps={global_step}, best={best_payload is not None}")
        last_payload = {"cell_id": cell_id, "seed": seed, "epoch": MAX_EPOCHS, "global_optimizer_step": global_step, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}
        atomic_torch(run_dir / "checkpoint_last.pt", last_payload)
        best_checkpoint = {"cell_id": cell_id, "seed": seed, "epoch": best["best_epoch"], "global_optimizer_step": global_step, "model_state_dict": best_payload, "config": config}
        atomic_torch(run_dir / "checkpoint_best.pt", best_checkpoint)
        reloaded, _hidden, _params, _added = build_cell_model(cell_id, model_seed, train_prevalence)
        reloaded.load_state_dict(best_payload, strict=True)
        reloaded = reloaded.to(device)
        best_eval = evaluate(reloaded, context_all[dev_mask], target_all[dev_mask], persistence_all[dev_mask], ids_all[dev_mask], tasks_all[dev_mask], mean, std, device)
        best_eval["event_mode"] = mode
        best_eval["train_event_prevalence"] = train_prevalence
        agg = aggregate_metrics(best_eval)
        if not agg["all_finite"] or best_eval["predictions_standardized"].shape != (614, STEPS, DIMS):
            raise RuntimeError("best checkpoint prediction contract failed")
        event_summary, calibration_rows, support_rows, time_rows = event_diagnostics(best_eval, dev_step_labels, dev_window_labels, dev_motion, tasks_all[dev_mask], thresholds, np.random.default_rng(STATS_SEED + CELL_ORDER.index(cell_id)))
        summary = gate_summary(best_eval["gate_values"], best_eval["event_probability"])
        atomic_npz(run_dir / "predictions_best.npz", predictions_standardized=best_eval["predictions_standardized"], targets_standardized=best_eval["targets_standardized"], persistence_standardized=best_eval["persistence_standardized"], predictions_native=best_eval["predictions_native"], targets_native=best_eval["targets_native"], persistence_native=best_eval["persistence_native"], gate_values=best_eval["gate_values"], event_probability=best_eval["event_probability"], event_step_labels=dev_step_labels, event_window_labels=dev_window_labels, episode_index=best_eval["episode_index"], original_task_index=best_eval["original_task_index"])
        per_rows: list[dict[str, object]] = []
        for index, episode in enumerate(best_eval["episode_index"]):
            per_rows.append({"cell_id": cell_id, "episode_index": int(episode), "original_task_index": int(best_eval["original_task_index"][index]), "arm_mae": float(best_eval["per"]["arm_mae"][index]), "hand_mae": float(best_eval["per"]["hand_mae"][index]), "full_mae": float(best_eval["per"]["full_mae"][index]), "arm_first_difference": float(best_eval["per"]["arm_first_difference"][index]), "hand_first_difference": float(best_eval["per"]["hand_first_difference"][index]), "arm_native_tv_closeness": float(best_eval["per"]["arm_native_tv_closeness"][index]), "hand_native_tv_closeness": float(best_eval["per"]["hand_native_tv_closeness"][index]), "gate_arm": float(best_eval["gate_values"][index, :, 0].mean()), "gate_hand": float(best_eval["gate_values"][index, :, 1].mean()), "event_q_mean": float(best_eval["event_probability"][index].mean()), "event_q_max": float(best_eval["event_probability"][index].max()), "event_window_label": int(dev_window_labels[index]), "event_step_label_fraction": float(dev_step_labels[index].mean())})
        write_csv(run_dir / "PER_EPISODE.csv", per_rows)
        write_csv(run_dir / "EVENT_CALIBRATION.csv", calibration_rows)
        write_csv(run_dir / "EVENT_SUPPORT.csv", support_rows)
        write_csv(run_dir / "EVENT_TIME.csv", time_rows)
        write_csv(run_dir / "EPOCH_CURVES.csv", curves)
        atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "curves": curves})
        atomic_json(run_dir / "EVENT_DIAGNOSTICS.json", event_summary)
        metrics = {"status": "complete", "cell_id": cell_id, "seed": seed, "event_mode": mode, "best_epoch": best["best_epoch"], "last_epoch": MAX_EPOCHS, "optimizer_steps": global_step, "train_count": int(train_mask.sum()), "development_count": int(dev_mask.sum()), "best": agg, "event": event_summary, "gate": summary, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "checkpoint_last_sha256": digest(run_dir / "checkpoint_last.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz"), "final_test_access": 0, "encoder_or_lora_modified": False, "raw_video": False, "elapsed_seconds": time.monotonic() - started}
        atomic_json(run_dir / "RUN_METRICS.json", metrics)
        atomic_json(status_path, {"status": "complete", "cell_id": cell_id, "seed": seed, "best_epoch": best["best_epoch"], "optimizer_steps": global_step, "final_test_access": 0, "completed_utc": now(), "exit_code": 0})
        atomic_json(run_dir / "RUN_COMPLETE.json", {"status": "complete", "cell_id": cell_id, "seed": seed, "optimizer_steps": global_step, "prediction_shape": [614, STEPS, DIMS], "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_best_sha256": digest(run_dir / "predictions_best.npz"), "final_test_access": 0})
        return {"cell_id": cell_id, "seed": seed, "status": "complete", "best_epoch": best["best_epoch"], "optimizer_steps": global_step, "metrics": agg, "event": event_summary, "path": str(run_dir)}
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
    result = train_cell(args.cell, args.root, seed=args.seed, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
