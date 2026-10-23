#!/usr/bin/env python3
"""Run the locked SEL-U temporal-fidelity decomposition and counterfactuals."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
SEL_ROOT = REPO / "reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z"
POSTFREEZE_ROOT = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z"
HAND_EVENT_ROOT = REPO / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
BOOTSTRAP = 10_000
TV_FLOOR = 1e-6
CELL_ORDER = ("CF-D0-G0", "CF-D0-G1", "CF-D1-G0", "CF-D1-G1")
CORRECTED_CELLS = CELL_ORDER[1:]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n",
        encoding="utf-8",
    )


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
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def dct_basis(length: int = 30) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)[None, :]
    frequency = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (time + 0.5) * frequency) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def dct_project(residual: np.ndarray, basis: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.einsum("kt,etj->ekj", basis, residual)
    projected = np.einsum("tk,ekj->etj", basis[:k].T, coefficients[:, :k])
    return projected, coefficients


def smooth_gate(q: np.ndarray) -> np.ndarray:
    padded = np.pad(q, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return np.clip(0.25 * padded[:, :-2] + 0.50 * padded[:, 1:-1] + 0.25 * padded[:, 2:], 0.0, 1.0)


def signed_terms(persistence: np.ndarray, q: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return anchor, residual-motion and gate-switch signed terms."""

    return (
        np.diff(persistence, axis=1),
        q[:, 1:] * np.diff(residual, axis=1),
        np.diff(q, axis=1) * residual[:, :-1],
    )


def event_motion(target: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    first = np.abs(target[:, :1] - persistence[:, :1]).mean(axis=2)
    transitions = np.abs(np.diff(target, axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= 1e-12 or np.std(y[valid]) <= 1e-12:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    values = values[valid]
    if len(values) < 2:
        raise ValueError("paired bootstrap requires at least two finite episodes")
    estimates = np.empty(BOOTSTRAP, dtype=np.float64)
    offset = 0
    while offset < BOOTSTRAP:
        count = min(500, BOOTSTRAP - offset)
        index = rng.integers(0, len(values), size=(count, len(values)))
        estimates[offset : offset + count] = values[index].mean(axis=1)
        offset += count
    return {
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": BOOTSTRAP,
        "N_episode": int(len(values)),
    }


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator, *, subtract_from_one: bool = False) -> dict[str, float | int]:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64).reshape(-1)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    numerator = numerator[valid]
    denominator = denominator[valid]
    if len(numerator) < 2:
        raise ValueError("paired bootstrap ratio requires at least two valid episodes")
    estimates = np.empty(BOOTSTRAP, dtype=np.float64)
    offset = 0
    while offset < BOOTSTRAP:
        count = min(500, BOOTSTRAP - offset)
        index = rng.integers(0, len(numerator), size=(count, len(numerator)))
        ratios = numerator[index].mean(axis=1) / denominator[index].mean(axis=1)
        estimates[offset : offset + count] = 1.0 - ratios if subtract_from_one else ratios
        offset += count
    point = float(numerator.mean() / denominator.mean())
    if subtract_from_one:
        point = 1.0 - point
    return {
        "effect": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": BOOTSTRAP,
        "N_episode": int(len(numerator)),
    }


def tv_closeness(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_tv = np.abs(np.diff(prediction, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    valid = target_tv > TV_FLOOR
    ratio = np.divide(prediction_tv, target_tv, out=np.full_like(prediction_tv, np.nan), where=valid)
    return np.abs(ratio - 1.0), valid


def turning_points(prediction: np.ndarray) -> np.ndarray:
    delta = np.diff(prediction, axis=1).mean(axis=2)
    signs = np.sign(delta)
    return np.sum(signs[:, 1:] * signs[:, :-1] < 0, axis=1)


def high_frequency_ratio(prediction: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coefficients = np.einsum("kt,etj->ekj", basis, prediction)
    energy = (coefficients**2).sum(axis=2)
    total = energy.sum(axis=1)
    high = energy[:, 16:].sum(axis=1)
    return np.divide(high, total, out=np.zeros_like(high), where=total > 1e-12)


def lag_alignment(term: np.ndarray, target_delta: np.ndarray) -> dict[str, float]:
    term_series = term.mean(axis=2)
    target_series = target_delta.mean(axis=2)
    zero = safe_corr(term_series, target_series)
    lag_values: list[tuple[int, float]] = []
    for lag in range(-3, 4):
        if lag < 0:
            x, y = term_series[:, :lag], target_series[:, -lag:]
        elif lag > 0:
            x, y = term_series[:, lag:], target_series[:, :-lag]
        else:
            x, y = term_series, target_series
        lag_values.append((lag, safe_corr(x, y)))
    finite = [(lag, value) for lag, value in lag_values if value is not None and np.isfinite(value)]
    best_lag, best_value = max(finite, key=lambda item: abs(item[1])) if finite else (0, None)
    return {"zero_lag_correlation": zero, "best_abs_lag": int(best_lag), "best_abs_lag_correlation": None if best_value is None else float(best_value)}


def term_summary(term: np.ndarray, target_delta: np.ndarray) -> dict[str, object]:
    flat = term.reshape(-1)
    target_flat = target_delta.reshape(-1)
    valid = np.abs(target_flat) > 1e-12
    cosine_num = np.sum(term * target_delta, axis=2)
    cosine_den = np.linalg.norm(term, axis=2) * np.linalg.norm(target_delta, axis=2)
    cosine = np.divide(cosine_num, cosine_den, out=np.full_like(cosine_num, np.nan), where=cosine_den > 1e-12)
    sign = np.divide(
        (np.sign(term) == np.sign(target_delta)) * (np.abs(target_delta) > 1e-12) * (np.abs(term) > 1e-12),
        (np.abs(target_delta) > 1e-12) * (np.abs(term) > 1e-12),
        out=np.full_like(term, np.nan, dtype=np.float64),
        where=(np.abs(target_delta) > 1e-12) & (np.abs(term) > 1e-12),
    )
    cosine_mean = np.nanmean(cosine) if np.isfinite(cosine).any() else None
    sign_mean = np.nanmean(sign) if np.isfinite(sign).any() else None
    return {
        "mae": float(np.mean(np.abs(term))),
        "rmse": float(np.sqrt(np.mean(term**2))),
        "energy": float(np.sum(term**2)),
        "mean_abs": float(np.mean(np.abs(term))),
        "target_nonzero_fraction": float(valid.mean()),
        "cosine_mean": None if cosine_mean is None else float(cosine_mean),
        "sign_agreement_mean": None if sign_mean is None else float(sign_mean),
        **lag_alignment(term, target_delta),
    }


def task_joint_support(
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    tasks: np.ndarray,
) -> dict[str, object]:
    task_rows: list[dict[str, object]] = []
    positive_supported: list[float] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        gain = float(np.mean(baseline_error[mask] - candidate_error[mask]))
        supported = int(mask.sum()) >= 5
        task_rows.append({"task": int(task), "N_episode": int(mask.sum()), "supported": supported, "improvement": gain})
        if supported and gain > 0:
            positive_supported.append(gain)
    joint_gain = baseline_error.mean(axis=1).mean(axis=0) - candidate_error.mean(axis=1).mean(axis=0)
    positive_joint = joint_gain[joint_gain > 0]
    return {
        "task_rows": task_rows,
        "supported_positive_task_count": len(positive_supported),
        "max_positive_task_share": max(positive_supported) / sum(positive_supported) if positive_supported else 1.0,
        "joint_effects": joint_gain.tolist(),
        "positive_joint_count": int((joint_gain > 0).sum()),
        "max_positive_joint_share": float(positive_joint.max() / positive_joint.sum()) if len(positive_joint) else 1.0,
    }


def metric_bundle(
    hand_std: np.ndarray,
    hand_native: np.ndarray,
    target_std: np.ndarray,
    target_native: np.ndarray,
    arm_std: np.ndarray,
    arm_native: np.ndarray,
    target_full_std: np.ndarray,
    target_full_native: np.ndarray,
    high_motion: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    full_std = np.concatenate((hand_std, arm_std), axis=2)
    full_native = np.concatenate((hand_native, arm_native), axis=2)
    hand_error_std = np.abs(hand_std - target_std).mean(axis=(1, 2))
    hand_error_native = np.abs(hand_native - target_native).mean(axis=(1, 2))
    full_error_std = np.abs(full_std - target_full_std).mean(axis=(1, 2))
    full_error_native = np.abs(full_native - target_full_native).mean(axis=(1, 2))
    hand_fd_std = np.abs(np.diff(hand_std, axis=1) - np.diff(target_std, axis=1)).mean(axis=(1, 2))
    hand_fd_native = np.abs(np.diff(hand_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2))
    full_fd_std = np.abs(np.diff(full_std, axis=1) - np.diff(target_full_std, axis=1)).mean(axis=(1, 2))
    full_fd_native = np.abs(np.diff(full_native, axis=1) - np.diff(target_full_native, axis=1)).mean(axis=(1, 2))
    tv, tv_valid = tv_closeness(hand_native, target_native)
    late_std = np.abs(hand_std[:, 20:] - target_std[:, 20:]).mean(axis=(1, 2))
    endpoint_std = np.abs(hand_std[:, -1] - target_std[:, -1]).mean(axis=1)
    return {
        "hand_error_std": hand_error_std,
        "hand_error_native": hand_error_native,
        "full_error_std": full_error_std,
        "full_error_native": full_error_native,
        "hand_fd_std": hand_fd_std,
        "hand_fd_native": hand_fd_native,
        "full_fd_std": full_fd_std,
        "full_fd_native": full_fd_native,
        "tv": tv,
        "tv_valid": tv_valid,
        "late_std": late_std,
        "endpoint_std": endpoint_std,
        "high_motion_mask": high_motion,
        "turning_points": turning_points(hand_native),
    }


def scalar(value: object) -> float:
    return float(np.asarray(value).item())


def main() -> int:
    rng = np.random.default_rng(20260936)
    with np.load(INPUT, allow_pickle=False) as data:
        dev = data["split_code"] == 1
        target_std_full = data["targets_standardized"][dev].astype(np.float64)
        target_native_full = data["targets_native"][dev].astype(np.float64)
        persistence_std_full = data["persistence_standardized"][dev].astype(np.float64)
        persistence_native_full = data["persistence_native"][dev].astype(np.float64)
        episode_index = data["episode_index"][dev].astype(np.int64)
        tasks = data["original_task_index"][dev].astype(np.int64)
    with np.load(SEL_ROOT / "runs/SEL-U/PREDICTIONS_DEV.npz", allow_pickle=False) as sel, np.load(SEL_ROOT / "runs/base_all_train/DEV_BASE_PREDICTIONS.npz", allow_pickle=False) as base, np.load(POSTFREEZE_ROOT / "runs/E-F-SP-G/predictions_best.npz", allow_pickle=False) as arm_bundle:
        hand_y_std = sel["hand_predictions_standardized"].astype(np.float64)
        persistence_std = sel["hand_persistence_standardized"].astype(np.float64)
        hand_base_std = sel["hand_base_predictions_standardized"].astype(np.float64)
        q = sel["q"].astype(np.float64)
        base_std = base["predictions_hand_standardized"].astype(np.float64)
        arm_std = arm_bundle["predictions_standardized"][:, :, 14:28].astype(np.float64)
        arm_native = arm_bundle["predictions_native"][:, :, 14:28].astype(np.float64)
        locked_arm_std = sel["locked_arm_standardized"].astype(np.float64)
        sel_ids = sel["episode_index"].astype(np.int64)
        base_ids = base["episode_index"].astype(np.int64)
        arm_ids = arm_bundle["episode_index"].astype(np.int64)
    if target_std_full.shape != (614, 30, 28) or hand_y_std.shape != (614, 30, 14) or q.shape != (614, 30, 1):
        raise RuntimeError("canonical temporal-fidelity shapes failed")
    if not (np.array_equal(sel_ids, episode_index) and np.array_equal(base_ids, episode_index) and np.array_equal(arm_ids, episode_index)):
        raise RuntimeError("episode IDs are not aligned")
    if not np.array_equal(locked_arm_std.astype(np.float32), arm_std.astype(np.float32)):
        raise RuntimeError("locked Arm is not bitwise identical")
    if not np.array_equal(persistence_std.astype(np.float32), persistence_std_full[:, :, :14].astype(np.float32)):
        raise RuntimeError("persistence arrays are not aligned")
    if not np.isfinite(np.concatenate((target_std_full.reshape(-1), target_native_full.reshape(-1), hand_y_std.reshape(-1), base_std.reshape(-1), q.reshape(-1)))).all():
        raise RuntimeError("non-finite temporal-fidelity input")

    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    std_scale = np.asarray(normalizer["safe_std"], dtype=np.float64)
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    hand_scale = std_scale[:, :14]
    hand_mean = mean[:, :14]
    to_native = lambda values: values * hand_scale[None, :, :] + hand_mean[None, :, :]
    hand_y_native = to_native(hand_y_std)
    persistence_native = persistence_native_full[:, :, :14]
    hand_base_native = to_native(hand_base_std)
    persistence_from_std_native = to_native(persistence_std)
    target_std = target_std_full[:, :, :14]
    target_native = target_native_full[:, :, :14]
    target_full_std = target_std_full
    target_full_native = target_native_full
    r_std = hand_base_std - persistence_std
    r_native = hand_base_native - persistence_native
    q_smooth = smooth_gate(q)
    basis = dct_basis(30)
    dct_registry = json.loads((ROOT / "DCT_K_REGISTRY.json").read_text(encoding="utf-8"))
    dct_residual_native, dct_coefficients = dct_project(r_native, basis, int(dct_registry["selected_K"]))
    # The four cells are deterministic transforms of the accepted SEL-U arrays.
    predictions_std = {
        "CF-D0-G0": persistence_std + q * r_std,
        "CF-D0-G1": persistence_std + q_smooth * r_std,
        "CF-D1-G0": persistence_std + q * (dct_residual_native / hand_scale[None, :, :]),
        "CF-D1-G1": persistence_std + q_smooth * (dct_residual_native / hand_scale[None, :, :]),
    }
    # DCT is defined in native space; projected residuals are converted back
    # with the fixed time-indexed normalizer for standardized diagnostics.
    predictions_native = {
        "CF-D0-G0": persistence_native + q * r_native,
        "CF-D0-G1": persistence_native + q_smooth * r_native,
        "CF-D1-G0": persistence_native + q * dct_residual_native,
        "CF-D1-G1": persistence_native + q_smooth * dct_residual_native,
    }
    replay_std_error = np.abs(predictions_std["CF-D0-G0"] - hand_y_std)
    replay_native_error = np.abs(predictions_native["CF-D0-G0"] - hand_y_native)
    if float(replay_native_error.max()) > 1e-5 or float(replay_std_error.max()) > 1e-5:
        raise RuntimeError("CF-D0-G0 does not replay SEL-U")

    thresholds = json.loads((HAND_EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"]
    motion = event_motion(target_std, persistence_std)
    episode_motion = motion.mean(axis=1)
    high_motion = episode_motion >= float(thresholds["COMMON_STEP_Q75"])
    low_motion = episode_motion < float(thresholds["COMMON_STEP_Q25"])
    metric_input = {"PERSISTENCE": metric_bundle(persistence_std, persistence_native, target_std, target_native, arm_std, arm_native, target_full_std, target_full_native, high_motion)}
    for cell in CELL_ORDER:
        native = predictions_native[cell]
        metric_input[cell] = metric_bundle(predictions_std[cell], native, target_std, target_native, arm_std, arm_native, target_full_std, target_full_native, high_motion)

    persistence = metric_input["PERSISTENCE"]
    cell_metrics: dict[str, dict[str, object]] = {}
    support_metrics: dict[str, object] = {}
    for cell in ("PERSISTENCE",) + CELL_ORDER:
        current = metric_input[cell]
        candidate_std = persistence_std if cell == "PERSISTENCE" else predictions_std[cell]
        hand_gain = np.asarray(persistence["hand_error_std"]) - np.asarray(current["hand_error_std"])
        fd_harm = np.asarray(current["hand_fd_native"]) - np.asarray(persistence["hand_fd_native"])
        tv_valid = np.asarray(persistence["tv_valid"]) & np.asarray(current["tv_valid"])
        tv_effect_values = np.asarray(persistence["tv"])[tv_valid] - np.asarray(current["tv"])[tv_valid]
        level_effect = bootstrap_mean(hand_gain, rng)
        full_effect = bootstrap_mean(np.asarray(persistence["full_error_std"]) - np.asarray(current["full_error_std"]), rng)
        fd_effect = bootstrap_mean(np.asarray(persistence["hand_fd_native"]) - np.asarray(current["hand_fd_native"]), rng)
        tv_effect = bootstrap_mean(tv_effect_values, rng)
        high_mask = np.asarray(high_motion)
        high_skill = float((np.asarray(persistence["hand_error_std"])[high_mask].mean() - np.asarray(current["hand_error_std"])[high_mask].mean()) / np.asarray(persistence["hand_error_std"])[high_mask].mean())
        late_skill = float((np.asarray(persistence["late_std"]).mean() - np.asarray(current["late_std"]).mean()) / np.asarray(persistence["late_std"]).mean())
        endpoint_skill = float((np.asarray(persistence["endpoint_std"]).mean() - np.asarray(current["endpoint_std"]).mean()) / np.asarray(persistence["endpoint_std"]).mean())
        tv_relative = float((np.asarray(current["tv"])[tv_valid].mean() - np.asarray(persistence["tv"])[tv_valid].mean()) / np.asarray(persistence["tv"])[tv_valid].mean())
        support = task_joint_support(np.abs(persistence_std - target_std), np.abs(candidate_std - target_std), tasks)
        support_metrics[cell] = support
        cell_metrics[cell] = {
            "hand_mae_std": float(np.asarray(current["hand_error_std"]).mean()),
            "hand_mae_native": float(np.asarray(current["hand_error_native"]).mean()),
            "full28_mae_std": float(np.asarray(current["full_error_std"]).mean()),
            "full28_mae_native": float(np.asarray(current["full_error_native"]).mean()),
            "hand_first_difference_mae_std": float(np.asarray(current["hand_fd_std"]).mean()),
            "hand_first_difference_mae_native": float(np.asarray(current["hand_fd_native"]).mean()),
            "full28_first_difference_mae_std": float(np.asarray(current["full_fd_std"]).mean()),
            "full28_first_difference_mae_native": float(np.asarray(current["full_fd_native"]).mean()),
            "native_tv_closeness": float(np.asarray(current["tv"])[tv_valid].mean()),
            "native_tv_valid_episode_count": int(tv_valid.sum()),
            "late_mae_std": float(np.asarray(current["late_std"]).mean()),
            "endpoint_mae_std": float(np.asarray(current["endpoint_std"]).mean()),
            "high_motion_skill_vs_persistence": high_skill,
            "late_skill_vs_persistence": late_skill,
            "endpoint_skill_vs_persistence": endpoint_skill,
            "hand_level_gain_std": float(hand_gain.mean()),
            "fd_harm_native": float(fd_harm.mean()),
            "native_tv_relative_change": tv_relative,
            "hand_improvement_vs_persistence": level_effect,
            "full_improvement_vs_persistence_hybrid": full_effect,
            "first_difference_improvement_native": fd_effect,
            "native_tv_improvement": tv_effect,
            "q_mean": float(q.mean()) if cell == "PERSISTENCE" else float(q_smooth.mean()) if cell.endswith("G1") else float(q.mean()),
            "q_collapse_fraction": float(np.mean((q < 0.05) | (q > 0.95))) if cell.endswith("G0") or cell == "PERSISTENCE" else float(np.mean((q_smooth < 0.05) | (q_smooth > 0.95))),
        }

    write_json(ROOT / "analysis/TASK_JOINT_SUPPORT.json", support_metrics)
    write_rows(ROOT / "analysis/TASK_JOINT_SUPPORT.csv", [{"cell": cell, "supported_positive_task_count": support_metrics[cell]["supported_positive_task_count"], "max_positive_task_share": support_metrics[cell]["max_positive_task_share"], "positive_joint_count": support_metrics[cell]["positive_joint_count"], "max_positive_joint_share": support_metrics[cell]["max_positive_joint_share"]} for cell in CELL_ORDER])

    base_gain = float(cell_metrics["CF-D0-G0"]["hand_level_gain_std"])
    base_fd_harm_native = float(cell_metrics["CF-D0-G0"]["fd_harm_native"])
    if base_gain <= 0 or base_fd_harm_native <= 0:
        raise RuntimeError(f"Gate B denominator invalid: level_gain={base_gain}, fd_harm={base_fd_harm_native}")

    gate_rows: list[dict[str, object]] = []
    corrected_pass: dict[str, bool] = {}
    for cell in CORRECTED_CELLS:
        current = cell_metrics[cell]
        gain_values = np.asarray(persistence["hand_error_std"]) - np.asarray(metric_input[cell]["hand_error_std"])
        d0_gain_values = np.asarray(persistence["hand_error_std"]) - np.asarray(metric_input["CF-D0-G0"]["hand_error_std"])
        harm_values = np.asarray(metric_input[cell]["hand_fd_native"]) - np.asarray(persistence["hand_fd_native"])
        d0_harm_values = np.asarray(metric_input["CF-D0-G0"]["hand_fd_native"]) - np.asarray(persistence["hand_fd_native"])
        retention = bootstrap_ratio(gain_values, d0_gain_values, rng)
        harm_reduction = bootstrap_ratio(harm_values, d0_harm_values, rng, subtract_from_one=True)
        support = support_metrics[cell]
        checks = {
            "level_gain_retention_ge_80pct": retention["effect"] >= 0.80,
            "first_difference_harm_reduction_ge_50pct": harm_reduction["effect"] >= 0.50,
            "endpoint_not_over_1pct_reverse": current["endpoint_skill_vs_persistence"] >= -0.01,
            "high_motion_not_over_1pct_reverse": current["high_motion_skill_vs_persistence"] >= -0.01,
            "late_not_over_1pct_reverse": current["late_skill_vs_persistence"] >= -0.01,
            "native_tv_not_over_1pct_degraded": current["native_tv_relative_change"] <= 0.01,
            "task_support_not_single": support["supported_positive_task_count"] >= 5 and support["max_positive_task_share"] < 0.50,
            "joint_support_not_single": support["positive_joint_count"] >= 5 and support["max_positive_joint_share"] < 0.50,
            "schema_identity_bootstrap_qa": True,
        }
        passed = bool(all(checks.values()))
        corrected_pass[cell] = passed
        gate_rows.append({"cell": cell, "level_gain_retention": retention, "fd_harm_reduction": harm_reduction, "checks": checks, "gate_b_pass": passed})
        current["level_gain_retention"] = retention
        current["fd_harm_reduction"] = harm_reduction
        current["gate_checks"] = checks
        current["gate_b_pass"] = passed

    passed_cells = [cell for cell, passed in corrected_pass.items() if passed]
    passed_set = set(passed_cells)
    if passed_set == {"CF-D0-G1"}:
        route = "Q"
    elif passed_set == {"CF-D1-G0"}:
        route = "R"
    elif "CF-D1-G1" in passed_set and ("CF-D0-G1" in passed_set or "CF-D1-G0" in passed_set):
        route = "QR"
    elif passed_set == {"CF-D1-G1"}:
        route = "QR"
    elif not passed_set:
        route = "NONE"
    else:
        route = "NONE"

    delta_q = np.diff(q[:, :, 0], axis=1)
    sign_change = (np.sign(delta_q[:, 1:]) * np.sign(delta_q[:, :-1])) < 0
    extreme = (q[:, :, 0] < 0.05) | (q[:, :, 0] > 0.95)
    extreme_switch = ((q[:, 1:, 0] < 0.05) & (q[:, :-1, 0] > 0.95)) | ((q[:, 1:, 0] > 0.95) & (q[:, :-1, 0] < 0.05))
    target_delta_std = np.diff(target_std, axis=1)
    target_delta_native = np.diff(target_native, axis=1)
    std_anchor, std_residual_motion, std_gate_switch = signed_terms(persistence_std, q, r_std)
    native_anchor, native_residual_motion, native_gate_switch = signed_terms(persistence_native, q, r_native)
    term_std = {"anchor_normalizer": std_anchor, "residual_motion": std_residual_motion, "gate_switch": std_gate_switch}
    term_native = {"anchor": native_anchor, "residual_motion": native_residual_motion, "gate_switch": native_gate_switch}
    term_energy_native = {key: float(np.sum(value**2)) for key, value in term_native.items()}
    native_total_term_energy = sum(term_energy_native.values())
    fd_definition = {
        "standardized_replay_max_abs": float(replay_std_error.max()),
        "standardized_replay_mean_abs": float(replay_std_error.mean()),
        "native_replay_max_abs": float(replay_native_error.max()),
        "native_replay_mean_abs": float(replay_native_error.mean()),
        "standardized_identity_max_abs": float(np.max(np.abs(np.diff(hand_y_std, axis=1) - (term_std["anchor_normalizer"] + term_std["residual_motion"] + term_std["gate_switch"]))),),
        "native_identity_max_abs": float(np.max(np.abs(np.diff(hand_y_native, axis=1) - (term_native["anchor"] + term_native["residual_motion"] + term_native["gate_switch"]))),),
        "native_anchor_step_max_abs": float(np.abs(term_native["anchor"]).max()),
        "native_term_energy": term_energy_native,
        "native_term_energy_share": {key: value / native_total_term_energy for key, value in term_energy_native.items()},
        "standardized_terms": {key: term_summary(value, target_delta_std) for key, value in term_std.items()},
        "native_terms": {key: term_summary(value, target_delta_native) for key, value in term_native.items()},
    }
    q_diagnostics = {
        "q_mean": float(q.mean()),
        "q_quantiles": {str(level): float(np.quantile(q, level)) for level in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
        "q_extreme_fraction": float(extreme.mean()),
        "delta_q_mean": float(delta_q.mean()),
        "delta_q_abs_mean": float(np.abs(delta_q).mean()),
        "delta_q_abs_quantiles": {str(level): float(np.quantile(np.abs(delta_q), level)) for level in (0.5, 0.9, 0.95, 0.99)},
        "delta_q_sign_change_rate": float(sign_change.mean()),
        "extreme_to_extreme_switch_rate": float(extreme_switch.mean()),
        "high_motion_episode_count": int(high_motion.sum()),
        "low_motion_episode_count": int(low_motion.sum()),
        "episode_correlation_level_vs_fd_error": safe_corr(np.asarray(metric_input["CF-D0-G0"]["hand_error_std"]), np.asarray(metric_input["CF-D0-G0"]["hand_fd_std"])),
    }

    result_rows = []
    for cell in ("PERSISTENCE",) + CELL_ORDER:
        row_metrics = cell_metrics["PERSISTENCE"] if cell == "PERSISTENCE" else cell_metrics[cell]
        result_rows.append({"cell": cell, **row_metrics})
    write_rows(ROOT / "analysis/COUNTERFACTUAL_RESULTS.csv", result_rows)
    write_json(ROOT / "analysis/COUNTERFACTUAL_RESULTS.json", {"status": "PASS", "cells": cell_metrics, "gate_b_rows": gate_rows, "route": route})
    (ROOT / "analysis/COUNTERFACTUAL_RESULTS.md").write_text("# Counterfactual results\n\n" + "\n".join(f"- `{cell}`: Hand MAE `{cell_metrics[cell]['hand_mae_std']:.6f}`, native FD `{cell_metrics[cell]['hand_first_difference_mae_native']:.6f}`, Gate B `{cell_metrics[cell].get('gate_b_pass', 'baseline')}`" for cell in CELL_ORDER) + f"\n\nRoute Gate B: `{route}`.\n", encoding="utf-8")
    write_rows(ROOT / "analysis/COUNTERFACTUAL_PAIRED_EFFECTS.csv", [{"cell": cell, "hand_effect": cell_metrics[cell]["hand_improvement_vs_persistence"], "full_effect": cell_metrics[cell]["full_improvement_vs_persistence_hybrid"], "fd_effect_native": cell_metrics[cell]["first_difference_improvement_native"], "tv_effect": cell_metrics[cell]["native_tv_improvement"]} for cell in CELL_ORDER])
    write_json(ROOT / "analysis/COUNTERFACTUAL_PAIRED_EFFECTS.json", {cell: {key: cell_metrics[cell][key] for key in ("hand_improvement_vs_persistence", "full_improvement_vs_persistence_hybrid", "first_difference_improvement_native", "native_tv_improvement")} for cell in CELL_ORDER})
    write_rows(ROOT / "analysis/MAIN_RESULTS.csv", result_rows)
    write_json(ROOT / "analysis/MAIN_RESULTS.json", {"status": "PASS", "cells": cell_metrics, "gate_b": gate_rows})
    (ROOT / "analysis/MAIN_RESULTS.md").write_text("# Main results\n\nThe primary diagnostic space is native command-value space; standardized metrics are retained for comparability.\n", encoding="utf-8")
    write_rows(ROOT / "analysis/PAIRED_EFFECTS.csv", [{"cell": cell, "level_gain_std": cell_metrics[cell]["hand_level_gain_std"], "fd_harm_native": cell_metrics[cell]["fd_harm_native"], "tv_relative_change": cell_metrics[cell]["native_tv_relative_change"]} for cell in CELL_ORDER])
    write_json(ROOT / "analysis/PAIRED_EFFECTS.json", {cell: {"level_gain_std": cell_metrics[cell]["hand_level_gain_std"], "fd_harm_native": cell_metrics[cell]["fd_harm_native"], "tv_relative_change": cell_metrics[cell]["native_tv_relative_change"]} for cell in CELL_ORDER})
    factorial_rows = []
    for factor, low, high in (("gate", "CF-D0-G0", "CF-D0-G1"), ("decoder", "CF-D0-G0", "CF-D1-G0")):
        factorial_rows.append({"factor": factor, "level_gain_change_std": cell_metrics[high]["hand_level_gain_std"] - cell_metrics[low]["hand_level_gain_std"], "fd_harm_change_native": cell_metrics[high]["fd_harm_native"] - cell_metrics[low]["fd_harm_native"]})
    factorial_rows.append({"factor": "decoder_x_gate", "level_gain_change_std": cell_metrics["CF-D1-G1"]["hand_level_gain_std"] - cell_metrics["CF-D1-G0"]["hand_level_gain_std"] - cell_metrics["CF-D0-G1"]["hand_level_gain_std"] + cell_metrics["CF-D0-G0"]["hand_level_gain_std"], "fd_harm_change_native": cell_metrics["CF-D1-G1"]["fd_harm_native"] - cell_metrics["CF-D1-G0"]["fd_harm_native"] - cell_metrics["CF-D0-G1"]["fd_harm_native"] + cell_metrics["CF-D0-G0"]["fd_harm_native"]})
    write_rows(ROOT / "analysis/FACTORIAL_EFFECTS.csv", factorial_rows)
    write_json(ROOT / "analysis/FACTORIAL_EFFECTS.json", {"status": "PASS", "rows": factorial_rows})
    write_json(ROOT / "analysis/FD_SIGNED_DECOMPOSITION.json", {"status": "PASS", **fd_definition})
    write_rows(ROOT / "analysis/FD_SIGNED_DECOMPOSITION.csv", [{"space": space, "term": term, **summary} for space, terms in (("standardized", fd_definition["standardized_terms"]), ("native", fd_definition["native_terms"])) for term, summary in terms.items()])
    (ROOT / "analysis/FD_SIGNED_DECOMPOSITION.md").write_text("# Signed decomposition\n\n" + json.dumps(fd_definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(ROOT / "analysis/GATE_SWITCH_DIAGNOSTICS.json", q_diagnostics)
    write_rows(ROOT / "analysis/GATE_SWITCH_DIAGNOSTICS.csv", [{"metric": key, "value": value} for key, value in q_diagnostics.items() if not isinstance(value, dict)])
    write_rows(ROOT / "analysis/SELECTOR_DIAGNOSTICS.csv", [{"metric": key, "value": value} for key, value in q_diagnostics.items() if not isinstance(value, dict)])
    write_json(ROOT / "analysis/TRAJECTORY_METRICS.json", {"status": "PASS", "episode_correlation_level_vs_fd": q_diagnostics["episode_correlation_level_vs_fd_error"]})

    selected: list[tuple[int, str]] = []
    baseline_level_gain = np.asarray(persistence["hand_error_std"]) - np.asarray(metric_input["CF-D0-G0"]["hand_error_std"])
    baseline_fd_harm = np.asarray(metric_input["CF-D0-G0"]["hand_fd_native"]) - np.asarray(persistence["hand_fd_native"])
    for index in np.argsort(baseline_level_gain)[-3:]:
        selected.append((int(index), "best_level"))
    for index in np.argsort(baseline_level_gain)[:3]:
        selected.append((int(index), "worst_level"))
    for index in np.argsort(baseline_fd_harm)[-3:]:
        selected.append((int(index), "regression_fd"))
    for index in np.argsort(baseline_fd_harm)[:3]:
        selected.append((int(index), "best_fd"))
    unique_selected: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in selected:
        if item[0] not in seen:
            seen.add(item[0])
            unique_selected.append(item)
    trajectory_rows = []
    for index, role in unique_selected[:12]:
        trajectory_rows.append({"row_index": index, "episode_index": int(episode_index[index]), "task": int(tasks[index]), "role": role, "d0g0_hand_mae_std": float(metric_input["CF-D0-G0"]["hand_error_std"][index]), "d0g0_fd_native": float(metric_input["CF-D0-G0"]["hand_fd_native"][index]), "d0g0_turning_points": int(metric_input["CF-D0-G0"]["turning_points"][index])})
    write_rows(ROOT / "analysis/TRAJECTORY_METRICS.csv", trajectory_rows)

    np.savez_compressed(
        ROOT / "data/COUNTERFACTUAL_PREDICTIONS.npz",
        episode_index=episode_index,
        original_task_index=tasks,
        target_hand_std=target_std,
        target_hand_native=target_native,
        persistence_hand_std=persistence_std,
        persistence_hand_native=persistence_native,
        q=q,
        q_smooth=q_smooth,
        residual_hand_std=r_std,
        residual_hand_native=r_native,
        dct_residual_hand_native=dct_residual_native,
        gate_switch_native=term_native["gate_switch"],
        residual_motion_native=term_native["residual_motion"],
        **{cell.lower().replace("-", "_") + "_std": predictions_std[cell] for cell in CELL_ORDER},
        **{cell.lower().replace("-", "_") + "_native": predictions_native[cell] for cell in CELL_ORDER},
    )

    gate_result = {
        "status": "PASS",
        "gate_b": "pass" if route != "NONE" else "fail",
        "route": route,
        "corrected_cells_pass": corrected_pass,
        "cells": gate_rows,
        "denominators": {"d0_level_gain_std": base_gain, "d0_fd_harm_native": base_fd_harm_native},
        "training_started": False,
        "final_test_access": 0,
    }
    write_json(ROOT / "ROUTE_GATE_B.json", {"status": "EVALUATED", **gate_result})
    (ROOT / "ROUTE_GATE_B_DECISION.md").write_text("# Route Gate B decision\n\n" + f"Gate B: `{gate_result['gate_b']}`. Route: `{route}`.\n\n" + "\n".join(f"- `{row['cell']}`: `{row['gate_b_pass']}`; retention `{row['level_gain_retention']['effect']:.3f}`; FD-harm reduction `{row['fd_harm_reduction']['effect']:.3f}`" for row in gate_rows) + "\n", encoding="utf-8")
    write_json(ROOT / "ROUTE_GATE_B_DECISION.json", gate_result)

    figure_status = generate_figures(cell_metrics, support_metrics, fd_definition, q_diagnostics, gate_result, predictions_native, target_native, persistence_native, arm_std, locked_arm_std, q, trajectory_rows, basis)
    write_json(ROOT / "FIGURE_GENERATION.json", figure_status)
    write_json(ROOT / "analysis/RESULTS_SUMMARY.json", {"cell_metrics": cell_metrics, "gate_b": gate_result, "figures": figure_status})
    (ROOT / "analysis/RESULTS_SUMMARY.md").write_text("# Results summary\n\n" + f"Gate B: `{gate_result['gate_b']}`; route: `{route}`.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "gate_b": gate_result["gate_b"], "route": route, "corrected_cells_pass": corrected_pass, "figure_status": figure_status["status"], "native_replay_max_abs": fd_definition["native_replay_max_abs"]}, sort_keys=True))
    return 0


def save_figure(plt, figure, stem: str) -> None:
    figure.tight_layout()
    for extension in ("png", "svg", "pdf"):
        figure.savefig(ROOT / "figures" / f"{stem}.{extension}", dpi=160 if extension == "png" else None)
    plt.close(figure)


def generate_figures(cell_metrics, support_metrics, fd_definition, q_diagnostics, gate_result, predictions_native, target_native, persistence_native, arm_std, locked_arm_std, q, trajectory_rows, basis) -> dict[str, object]:
    try:
        import matplotlib.pyplot as plt
        names = ["Persistence"] + list(CELL_ORDER)
        labels = ["P"] + [cell.replace("CF-", "") for cell in CELL_ORDER]
        ROOT.joinpath("figures").mkdir(parents=True, exist_ok=True)
        figure_records: list[dict[str, object]] = []

        fig, ax = plt.subplots(figsize=(7, 5))
        x = [cell_metrics["PERSISTENCE"]["hand_mae_std"]] + [cell_metrics[cell]["hand_mae_std"] for cell in CELL_ORDER]
        fd = [cell_metrics["PERSISTENCE"]["hand_first_difference_mae_std"]] + [cell_metrics[cell]["hand_first_difference_mae_std"] for cell in CELL_ORDER]
        ax.scatter(x, fd, color=["#777777", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c"])
        label_offsets = {"P": (4, 4), "D0-G0": (4, 4), "D0-G1": (4, 22), "D1-G0": (4, 4), "D1-G1": (4, 22)}
        for xi, yi, label in zip(x, fd, labels):
            ax.annotate(label, (xi, yi), xytext=label_offsets[label], textcoords="offset points")
        ax.set_xlabel("Hand standardized MAE")
        ax.set_ylabel("Hand standardized first-difference MAE")
        ax.set_title("SEL-U level versus temporal fidelity")
        save_figure(plt, fig, "SEL_U_LEVEL_FD_TRADEOFF")
        figure_records.append({"file": "SEL_U_LEVEL_FD_TRADEOFF", "alt_text": "Scatter of level Hand MAE against first-difference MAE for Persistence and four counterfactual cells."})

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        q_values = np.diff(q[:, :, 0], axis=1).reshape(-1)
        axes[0].hist(q_values, bins=40, color="#984ea3")
        axes[0].set_title("Delta q distribution")
        axes[0].set_xlabel("Delta q")
        axes[1].bar(["extreme", "sign-change", "extreme-switch"], [q_diagnostics["q_extreme_fraction"], q_diagnostics["delta_q_sign_change_rate"], q_diagnostics["extreme_to_extreme_switch_rate"]], color="#984ea3")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Gate switching rates")
        save_figure(plt, fig, "DELTA_Q_DISTRIBUTION")
        figure_records.append({"file": "DELTA_Q_DISTRIBUTION", "alt_text": "Gate q change quantile summary and switching rates."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        terms = fd_definition["native_term_energy_share"]
        ax.bar(list(terms), list(terms.values()), color=["#999999", "#377eb8", "#e41a1c"])
        ax.set_ylabel("native signed-term energy share")
        ax.set_title("Signed decomposition term energy")
        ax.tick_params(axis="x", rotation=20)
        save_figure(plt, fig, "TERM_MAGNITUDE_COMPARISON")
        figure_records.append({"file": "TERM_MAGNITUDE_COMPARISON", "alt_text": "Energy shares of native anchor, residual-motion and gate-switch terms."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        term_names = ["anchor", "residual_motion", "gate_switch"]
        cosine = [fd_definition["native_terms"][name]["cosine_mean"] or 0.0 for name in term_names]
        sign = [fd_definition["native_terms"][name]["sign_agreement_mean"] or 0.0 for name in term_names]
        positions = np.arange(len(term_names))
        ax.bar(positions - 0.18, cosine, width=0.36, label="cosine")
        ax.bar(positions + 0.18, sign, width=0.36, label="sign agreement")
        ax.set_xticks(positions, term_names, rotation=20)
        ax.set_ylim(-1, 1)
        ax.legend(frameon=False)
        ax.set_title("Term alignment with ground-truth delta")
        save_figure(plt, fig, "TERM_GT_ALIGNMENT")
        figure_records.append({"file": "TERM_GT_ALIGNMENT", "alt_text": "Cosine and sign agreement of signed decomposition terms with target change."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        corrected = list(CORRECTED_CELLS)
        ax.scatter([cell_metrics[cell]["hand_level_gain_std"] / cell_metrics["CF-D0-G0"]["hand_level_gain_std"] for cell in corrected], [cell_metrics[cell]["fd_harm_reduction"]["effect"] for cell in corrected], color="#4daf4a")
        label_offsets = {"CF-D0-G1": (4, 4), "CF-D1-G0": (4, -18), "CF-D1-G1": (4, 18)}
        for cell in corrected:
            ax.annotate(cell.replace("CF-", ""), (cell_metrics[cell]["hand_level_gain_std"] / cell_metrics["CF-D0-G0"]["hand_level_gain_std"], cell_metrics[cell]["fd_harm_reduction"]["effect"]), xytext=label_offsets[cell], textcoords="offset points")
        ax.axvline(0.8, color="red", linestyle="--")
        ax.axhline(0.5, color="red", linestyle="--")
        ax.set_xlabel("level-gain retention")
        ax.set_ylabel("FD-harm reduction")
        ax.set_title("Gate B counterfactual trade-off")
        save_figure(plt, fig, "COUNTERFACTUAL_LEVEL_FD")
        figure_records.append({"file": "COUNTERFACTUAL_LEVEL_FD", "alt_text": "Counterfactual level-gain retention versus first-difference-harm reduction with Gate B thresholds."})

        fig, ax = plt.subplots(figsize=(7, 4.5))
        registry = json.loads((ROOT / "DCT_K_REGISTRY.json").read_text(encoding="utf-8"))
        ks = [int(key) for key in registry["median_cumulative_energy"]]
        cov = [registry["median_cumulative_energy"][str(key)] for key in ks]
        ax.plot(ks, cov, marker="o")
        ax.axhline(0.95, color="red", linestyle="--")
        ax.axvline(registry["selected_K"], color="#377eb8", linestyle=":")
        ax.set_ylim(0, 1)
        ax.set_xlabel("K")
        ax.set_ylabel("train median cumulative energy")
        ax.set_title("Train-only DCT K registry")
        save_figure(plt, fig, "DCT_TRAIN_ENERGY")
        figure_records.append({"file": "DCT_TRAIN_ENERGY", "alt_text": "Train-only cumulative DCT energy curve showing locked K."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        full_values = [cell_metrics["CF-D0-G0"]["full28_mae_std"]] + [cell_metrics[cell]["full28_mae_std"] for cell in corrected]
        ax.bar(["D0G0"] + [cell.replace("CF-", "") for cell in corrected], full_values, color="#4daf4a")
        ax.set_ylabel("Full-28 standardized MAE")
        ax.set_title("Counterfactual Full-28 MAE")
        ax.tick_params(axis="x", rotation=20)
        save_figure(plt, fig, "COUNTERFACTUAL_FULL_MAE")
        figure_records.append({"file": "COUNTERFACTUAL_FULL_MAE", "alt_text": "Full-28 MAE bars for the four deterministic counterfactual cells."})

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fd_values = [cell_metrics[cell]["hand_first_difference_mae_native"] for cell in CELL_ORDER]
        tv_values = [cell_metrics[cell]["native_tv_closeness"] for cell in CELL_ORDER]
        positions = np.arange(len(CELL_ORDER))
        ax.bar(positions - 0.18, fd_values, width=0.36, label="native FD MAE")
        ax.bar(positions + 0.18, tv_values, width=0.36, label="native TV closeness")
        ax.set_xticks(positions, [cell.replace("CF-", "") for cell in CELL_ORDER])
        ax.legend(frameon=False)
        ax.set_title("Native first-difference and TV diagnostics")
        save_figure(plt, fig, "FD_TV_FREQUENCY")
        figure_records.append({"file": "FD_TV_FREQUENCY", "alt_text": "Native first-difference and TV-closeness bars across counterfactual cells."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        positions = np.arange(len(corrected))
        width = 0.25
        ax.bar(positions - width, [cell_metrics[cell]["high_motion_skill_vs_persistence"] for cell in corrected], width=width, label="high motion")
        ax.bar(positions, [cell_metrics[cell]["late_skill_vs_persistence"] for cell in corrected], width=width, label="late")
        ax.bar(positions + width, [cell_metrics[cell]["endpoint_skill_vs_persistence"] for cell in corrected], width=width, label="endpoint")
        ax.set_xticks(positions, [cell.replace("CF-", "") for cell in corrected])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("relative skill components")
        ax.legend(frameon=False)
        ax.set_title("Horizon and motion guardrails")
        save_figure(plt, fig, "HORIZON_MOTION_ENDPOINT")
        figure_records.append({"file": "HORIZON_MOTION_ENDPOINT", "alt_text": "High-motion and late-horizon relative skill components for corrected cells."})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        positions = np.arange(len(corrected))
        width = 0.35
        ax.bar(positions - width / 2, [support_metrics[cell]["supported_positive_task_count"] for cell in corrected], width=width, label="positive tasks")
        ax.bar(positions + width / 2, [support_metrics[cell]["positive_joint_count"] for cell in corrected], width=width, alpha=0.65, label="positive joints")
        ax.set_xticks(positions, [cell.replace("CF-", "") for cell in corrected])
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        ax.set_title("Task/joint support")
        save_figure(plt, fig, "TASK_JOINT_SUPPORT")
        figure_records.append({"file": "TASK_JOINT_SUPPORT", "alt_text": "Positive supported task and joint counts for corrected counterfactuals."})

        representative = int(trajectory_rows[0]["episode_index"]) if trajectory_rows else 0
        index = int(trajectory_rows[0]["row_index"]) if trajectory_rows else 0
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for axis, cell in zip(axes.ravel(), CELL_ORDER):
            axis.plot(target_native[index, :, 0], label="GT", color="black")
            axis.plot(persistence_native[index, :, 0], label="P", color="#999999")
            axis.plot(predictions_native[cell][index, :, 0], label=cell.replace("CF-", ""))
            axis.set_title(cell)
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle("Representative Hand command trajectory (joint 0)")
        save_figure(plt, fig, "REPRESENTATIVE_TRAJECTORIES")
        figure_records.append({"file": "REPRESENTATIVE_TRAJECTORIES", "alt_text": "Representative ground-truth, persistence and four counterfactual Hand trajectories in command space."})

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
        plot_rows = []
        for role in ("worst_level", "best_level", "regression_fd"):
            match = next((row for row in trajectory_rows if row.get("role") == role), None)
            if match is not None:
                plot_rows.append(match)
        plot_rows = (plot_rows or trajectory_rows or [{"row_index": index, "episode_index": representative, "role": "fallback"}])[:3]
        for axis, row in zip(axes, plot_rows):
            idx = int(row.get("row_index", index))
            axis.plot(target_native[idx, :, 0], color="black", label="GT")
            axis.plot(predictions_native["CF-D0-G0"][idx, :, 0], color="#377eb8", label="D0G0")
            axis.set_title(str(row["role"]))
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle("Worst/best/regression trajectory examples")
        save_figure(plt, fig, "WORST_BEST_REGRESSION")
        figure_records.append({"file": "WORST_BEST_REGRESSION", "alt_text": "Selected trajectory examples associated with level or first-difference diagnostics."})

        fig, ax = plt.subplots(figsize=(6, 4))
        arm_diff = [0.0, 0.0, 0.0, 0.0]
        ax.bar(["D0G0", "D0G1", "D1G0", "D1G1"], arm_diff, color="#377eb8")
        ax.set_ylim(0, 1)
        ax.text(1.5, 0.5, "bitwise difference = 0", ha="center", va="center")
        ax.set_ylabel("max absolute Arm difference")
        ax.set_title("Arm non-interference contract")
        save_figure(plt, fig, "ARM_NON_INTERFERENCE")
        figure_records.append({"file": "ARM_NON_INTERFERENCE", "alt_text": "Arm bitwise non-interference diagnostic, with zero difference across cells."})

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        ax.text(0.02, 0.85, f"Gate B: {gate_result['gate_b']}", fontsize=16, weight="bold")
        ax.text(0.02, 0.70, f"Route: {gate_result['route']}", fontsize=13)
        ax.text(0.02, 0.55, "Only corrected cells passing all pre-locked thresholds may train.", fontsize=11)
        ax.text(0.02, 0.40, "Development-only; outcome-exposed; no final-test access.", fontsize=11)
        save_figure(plt, fig, "DECISION_SUMMARY")
        figure_records.append({"file": "DECISION_SUMMARY", "alt_text": "Decision summary showing Gate B status, route and development-only boundary."})

        write_rows(ROOT / "figures/FIGURE_DATA.csv", [{"figure": item["file"], "gate_b": gate_result["gate_b"], "route": gate_result["route"]} for item in figure_records])
        write_rows(ROOT / "captions/FIGURE_MANIFEST.csv", figure_records)
        write_json(ROOT / "captions/FIGURE_MANIFEST.json", {"status": "PASS", "figures": figure_records, "formats": ["png", "svg", "pdf"], "physical_units_claimed": False})
        return {"status": "PASS", "figure_count": len(figure_records), "formats": ["png", "svg", "pdf"], "files": [item["file"] for item in figure_records]}
    except Exception as exc:
        return {"status": f"FAIL:{type(exc).__name__}:{exc}", "figure_count": 0, "files": [], "traceback": traceback.format_exc()}


if __name__ == "__main__":
    raise SystemExit(main())
