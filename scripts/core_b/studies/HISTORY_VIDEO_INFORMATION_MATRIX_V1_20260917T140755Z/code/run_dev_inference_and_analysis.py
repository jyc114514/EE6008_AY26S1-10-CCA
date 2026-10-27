#!/usr/bin/env python3
"""Run fixed-last development inference and the locked History×Video analysis."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import HistoryVideoTCNDCT, native_prediction, state_hash


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
PREPROCESS_STATS = ROOT / "PREPROCESS_STATS.json"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
BOOTSTRAP = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/paired_episode_bootstrap_indices.npy"
MOTION_STRATA = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_STRATA_MANIFEST.csv"
REPRESENTATIVE = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/V2_TRAJECTORY_REPRESENTATIVE_EPISODES.csv"
SEED = 20260939
HAND = 14
ARM = 14
STEPS = 30
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_data() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    preprocess = json.loads(PREPROCESS_STATS.read_text(encoding="utf-8"))
    stats = {
        "target_mean": np.asarray(normalizer["mean"], dtype=np.float32),
        "target_std": np.asarray(normalizer["safe_std"], dtype=np.float32),
        "residual_sigma": np.asarray(preprocess["residual_sigma"], dtype=np.float32),
        "fd_scale": np.asarray(preprocess["fd_scale"], dtype=np.float32),
        "lambda_fd": float(preprocess["lambda_fd"]),
    }
    if data["targets_native"].shape != (3355, STEPS, 28) or data["history_standardized"].shape != (3355, 32, 71):
        raise ValueError("analysis input shape contract failed")
    if stats["target_mean"].shape != (STEPS, 28) or stats["target_std"].shape != (STEPS, 28):
        raise ValueError("target normalizer shape failed")
    return data, stats


def model_inputs(data: dict[str, np.ndarray], cell: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_mode, video_mode = ({"HI-S-LAST": ("LAST", "S"), "HI-S-HIST": ("HIST", "S"), "HI-F-LAST": ("LAST", "F"), "HI-F-HIST": ("HIST", "F")})[cell]
    history = data["history_standardized"].astype(np.float32)
    state = np.repeat(history[:, -1:, :], 32, axis=1) if state_mode == "LAST" else history
    video = np.zeros_like(data["dynamic_video_feature"], dtype=np.float32) if video_mode == "S" else data["dynamic_video_feature"].astype(np.float32)
    return state, data["history_mask"].astype(np.float32), data["history_relative_delta_standardized"].astype(np.float32), video, data["persistence_native"][:, 0, :HAND].astype(np.float32)


def infer_cell(data: dict[str, np.ndarray], stats: dict[str, Any], cell: str, checkpoint: Path, ids: np.ndarray, device: torch.device) -> tuple[np.ndarray, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("cell") != cell or payload.get("epoch") != 40 or payload.get("optimizer_steps") != 1720:
        raise ValueError(f"formal checkpoint contract failed for {cell}")
    model = HistoryVideoTCNDCT(seed=SEED)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    state, mask, delta, video, anchor = model_inputs(data, cell)
    dev_mask = data["split_code"] == 1
    state = state[dev_mask]
    mask = mask[dev_mask]
    delta = delta[dev_mask]
    video = video[dev_mask]
    anchor = anchor[dev_mask]
    predictions: list[np.ndarray] = []
    sigma = torch.from_numpy(stats["residual_sigma"]).to(device)
    with torch.inference_mode():
        for start in range(0, len(ids), 128):
            stop = min(start + 128, len(ids))
            scaled = model(torch.from_numpy(state[start:stop]).to(device), torch.from_numpy(mask[start:stop]).to(device), torch.from_numpy(delta[start:stop]).to(device), torch.from_numpy(video[start:stop]).to(device))
            predictions.append(native_prediction(scaled, torch.from_numpy(anchor[start:stop]).to(device), sigma).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0).astype(np.float32)
    if prediction.shape != (len(ids), STEPS, HAND) or not np.isfinite(prediction).all():
        raise ValueError(f"prediction shape/finite contract failed for {cell}")
    return prediction, state_hash(model)


def standardize(native: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    if native.shape[-1] == HAND:
        mean = stats["target_mean"][:, :HAND]
        scale = stats["target_std"][:, :HAND]
    elif native.shape[-1] == 28:
        mean = stats["target_mean"]
        scale = stats["target_std"]
    else:
        raise ValueError(f"unsupported native output dimension: {native.shape}")
    return ((native.astype(np.float32) - mean[None]) / scale[None]).astype(np.float32)


def slope_baseline(data: dict[str, np.ndarray], dev_positions: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    history = data["history_native"][dev_positions].astype(np.float64)
    selected_timestamps = data["history_selected_timestamps"][dev_positions].astype(np.float64)
    target_timestamps = data["target_timestamps"][dev_positions].astype(np.float64)
    prediction = np.empty((len(dev_positions), STEPS, 28), dtype=np.float32)
    fallback = 0
    for row in range(len(dev_positions)):
        times = selected_timestamps[row]
        commands = history[row, :, 43:71]
        t0 = float(times[-1])
        valid = np.flatnonzero(times >= t0 - 0.25 - 1e-9)
        unique: list[int] = []
        for index in valid.tolist():
            if not unique or abs(float(times[index]) - float(times[unique[-1]])) > 1e-9:
                unique.append(index)
        anchor = commands[-1]
        if len(unique) < 2 or float(times[unique[-1]] - times[unique[0]]) <= 1e-9:
            prediction[row] = np.repeat(anchor[None], STEPS, axis=0)
            fallback += 1
            continue
        slope = (commands[unique[-1]] - commands[unique[0]]) / (times[unique[-1]] - times[unique[0]])
        prediction[row] = (anchor[None] + slope[None] * (target_timestamps[row, :, None] - t0)).astype(np.float32)
    return prediction, {"window_seconds": 0.25, "fallback_episodes": fallback, "usable_episodes": int(len(dev_positions) - fallback), "definition": "no-fit linear extrapolation from first/last unique command samples in the last 0.25 seconds"}


def metric_arrays(pred_native: np.ndarray, pred_std: np.ndarray, target_native: np.ndarray, target_std: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    hand_native = pred_native[:, :, :HAND]
    target_hand_native = target_native[:, :, :HAND]
    hand_std = pred_std[:, :, :HAND]
    target_hand_std = target_std[:, :, :HAND]
    full_std_error = np.abs(pred_std - target_std)
    hand_std_error = np.abs(hand_std - target_hand_std)
    full_native_error = np.abs(pred_native - target_native)
    hand_native_error = np.abs(hand_native - target_hand_native)
    pred_delta_h = np.diff(hand_native, axis=1)
    target_delta_h = np.diff(target_hand_native, axis=1)
    pred_delta_std = np.diff(hand_std, axis=1)
    target_delta_std = np.diff(target_hand_std, axis=1)
    target_tv = np.abs(target_delta_h).sum(axis=1).mean(axis=1)
    pred_tv = np.abs(pred_delta_h).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 1e-12)
    target_sign = np.sign(target_delta_h)
    valid_sign = np.abs(target_delta_h) > 1e-12
    agreement = np.divide(((np.sign(pred_delta_h) == target_sign) & valid_sign).sum(axis=(1, 2)), valid_sign.sum(axis=(1, 2)), out=np.full(len(pred_native), np.nan), where=valid_sign.sum(axis=(1, 2)) > 0)
    return {
        "hand_mae_std": hand_std_error.mean(axis=(1, 2)),
        "hand_mae_native": hand_native_error.mean(axis=(1, 2)),
        "full_mae_std": full_std_error.mean(axis=(1, 2)),
        "full_mae_native": full_native_error.mean(axis=(1, 2)),
        "full_rmse_native": np.sqrt(np.mean((pred_native - target_native) ** 2, axis=(1, 2))),
        "hand_fd_native": np.abs(pred_delta_h - target_delta_h).mean(axis=(1, 2)),
        "hand_fd_std": np.abs(pred_delta_std - target_delta_std).mean(axis=(1, 2)),
        "full_fd_native": np.abs(np.diff(pred_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2)),
        "hand_anchor_to_first_native": np.abs((hand_native[:, 0] - anchor) - (target_hand_native[:, 0] - anchor)).mean(axis=1),
        "hand_tv": pred_tv,
        "target_hand_tv": target_tv,
        "hand_tv_ratio": ratio,
        "hand_tv_closeness": np.abs(ratio - 1.0),
        "hand_direction_agreement": agreement,
        "early_hand_mae_std": hand_std_error[:, :10].mean(axis=(1, 2)),
        "middle_hand_mae_std": hand_std_error[:, 10:20].mean(axis=(1, 2)),
        "late_hand_mae_std": hand_std_error[:, 20:30].mean(axis=(1, 2)),
        "endpoint_hand_mae_std": hand_std_error[:, 29].mean(axis=1),
        "early_hand_mae_native": hand_native_error[:, :10].mean(axis=(1, 2)),
        "middle_hand_mae_native": hand_native_error[:, 10:20].mean(axis=(1, 2)),
        "late_hand_mae_native": hand_native_error[:, 20:30].mean(axis=(1, 2)),
        "endpoint_hand_mae_native": hand_native_error[:, 29].mean(axis=1),
        "per_joint_hand_mae_std": hand_std_error.mean(axis=1),
        "per_time_hand_mae_std": hand_std_error.mean(axis=2),
        "per_joint_hand_fd_native": np.abs(pred_delta_h - target_delta_h).mean(axis=1),
    }


def bootstrap_stats(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    difference = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    estimates = difference[indices].mean(axis=1)
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = difference[np.isfinite(difference)]
    for start in range(0, len(indices), 500):
        size = min(500, len(indices) - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(valid)))
        exceed += int(np.sum(np.abs((signs * valid[None]).mean(axis=1)) >= abs(valid.mean())))
    return {"effect_control_minus_candidate": float(valid.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "win_rate_candidate_better": float(np.mean(valid > 0)), "p_signflip": float((1 + exceed) / (len(indices) + 1)), "N_episode": int(len(valid)), "replicates": int(len(indices))}


def ratio_bootstrap(candidate: np.ndarray, control: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(candidate, dtype=np.float64)[indices].mean(axis=1)
    denominator = np.asarray(control, dtype=np.float64)[indices].mean(axis=1)
    ratios = numerator / denominator - 1.0
    return {"relative_change_candidate_vs_control": float(np.mean(candidate) / np.mean(control) - 1.0), "ci_low": float(np.quantile(ratios, 0.025)), "ci_high": float(np.quantile(ratios, 0.975)), "replicates": int(len(indices))}


def holm_adjust(rows: list[dict[str, object]], p_key: str = "p_signflip") -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][p_key]))
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(rows[index][p_key]) * (len(rows) - rank)))
        adjusted[index] = running
    for index, value in enumerate(adjusted):
        rows[index]["holm_p"] = value


def main() -> int:
    analysis = ROOT / "analysis"
    data, stats = load_data()
    dev_positions = np.flatnonzero(data["split_code"] == 1)
    train_positions = np.flatnonzero(data["split_code"] == 0)
    dev_ids = data["episode_index"][dev_positions].astype(np.int64)
    dev_tasks = data["original_task_index"][dev_positions].astype(np.int64)
    if len(dev_positions) != 614 or len(train_positions) != 2741:
        raise ValueError("dev/train split contract failed")
    locked = {key: value.copy() for key, value in np.load(LOCKED_ARM, allow_pickle=False).items()}
    if not np.array_equal(locked["episode_index"], dev_ids) or locked["predictions_native"].shape != (614, STEPS, 28):
        raise ValueError("locked Arm dev seam failed")
    arm_native = locked["predictions_native"][:, :, 14:].astype(np.float32)
    arm_std = locked["predictions_standardized"][:, :, 14:].astype(np.float32)
    target_native = data["targets_native"][dev_positions].astype(np.float32)
    target_std = data["targets_standardized"][dev_positions].astype(np.float32)
    persistence_native = data["persistence_native"][dev_positions].astype(np.float32)
    persistence_std = data["persistence_standardized"][dev_positions].astype(np.float32)
    anchor = persistence_native[:, 0, :HAND]
    models_native: dict[str, np.ndarray] = {
        "PERSISTENCE": persistence_native,
        "FIXED_HYBRID": np.concatenate((persistence_native[:, :, :HAND], arm_native), axis=2),
    }
    models_std: dict[str, np.ndarray] = {
        "PERSISTENCE": persistence_std,
        "FIXED_HYBRID": np.concatenate((persistence_std[:, :, :HAND], arm_std), axis=2),
    }
    run_acceptance: list[dict[str, object]] = []
    for cell in CELLS:
        run_dir = ROOT / "runs" / cell / "formal"
        complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        if complete.get("status") != "PASS" or complete.get("epoch") != 40 or complete.get("optimizer_steps") != 1720:
            raise ValueError(f"formal run not accepted: {cell}")
        prediction_hand, final_hash = infer_cell(data, stats, cell, run_dir / "checkpoint_last.pt", dev_ids, torch.device("cpu"))
        prediction_std_hand = standardize(prediction_hand, stats)
        full_native = np.concatenate((prediction_hand, arm_native), axis=2)
        full_std = np.concatenate((prediction_std_hand, arm_std), axis=2)
        models_native[cell] = full_native
        models_std[cell] = full_std
        np.savez_compressed(run_dir / "predictions_last.npz", episode_index=dev_ids, original_task_index=dev_tasks, targets_native=target_native, targets_standardized=target_std, persistence_native=persistence_native, persistence_standardized=persistence_std, predictions_hand_native=prediction_hand, predictions_hand_standardized=prediction_std_hand, predictions_full_native=full_native, predictions_full_standardized=full_std, locked_arm_native=arm_native, locked_arm_standardized=arm_std)
        write_json(run_dir / "DEV_INFERENCE_ACCEPTANCE.json", {"status": "PASS", "cell": cell, "checkpoint": str(run_dir / "checkpoint_last.pt"), "checkpoint_sha256": digest(run_dir / "checkpoint_last.pt"), "model_hash": final_hash, "episode_index_shape": [614], "prediction_hand_shape": [614, STEPS, HAND], "prediction_full_shape": [614, STEPS, 28], "arm_bitwise_equal_locked": bool(np.array_equal(full_native[:, :, 14:], arm_native) and np.array_equal(full_std[:, :, 14:], arm_std)), "final_test_access": 0})
        run_acceptance.append({"cell": cell, "status": "PASS", "epoch": 40, "optimizer_steps": 1720, "prediction_hand_shape": "(614,30,14)", "prediction_full_shape": "(614,30,28)", "arm_bitwise_equal_locked": True, "final_test_access": 0})
    slope_native, slope_status = slope_baseline(data, dev_positions)
    slope_std = standardize(slope_native, stats)
    models_native["SLOPE_BASELINE"] = slope_native
    models_std["SLOPE_BASELINE"] = slope_std
    np.savez_compressed(analysis / "SLOPE_BASELINE_DEV.npz", episode_index=dev_ids, original_task_index=dev_tasks, predictions_native=slope_native, predictions_standardized=slope_std, target_timestamps=data["target_timestamps"][dev_positions], final_test_access=np.asarray(0, dtype=np.int8))
    write_json(analysis / "SLOPE_BASELINE_STATUS.json", {"status": "PASS", **slope_status, "final_test_access": 0})

    metric: dict[str, dict[str, np.ndarray]] = {}
    for name in models_native:
        metric[name] = metric_arrays(models_native[name], models_std[name], target_native, target_std, anchor)
    metric_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "SLOPE_BASELINE", "FIXED_HYBRID", *CELLS):
        item = metric[name]
        metric_rows.append({"model": name, "N_episode": 614, "hand_mae_std": float(item["hand_mae_std"].mean()), "hand_mae_native": float(item["hand_mae_native"].mean()), "full_mae_std": float(item["full_mae_std"].mean()), "full_mae_native": float(item["full_mae_native"].mean()), "full_rmse_native": float(item["full_rmse_native"].mean()), "hand_fd_native": float(item["hand_fd_native"].mean()), "hand_fd_std": float(item["hand_fd_std"].mean()), "full_fd_native": float(item["full_fd_native"].mean()), "hand_anchor_to_first_native": float(item["hand_anchor_to_first_native"].mean()), "hand_tv_mean": float(item["hand_tv"].mean()), "target_hand_tv_mean": float(item["target_hand_tv"].mean()), "hand_tv_ratio_mean_finite": float(np.nanmean(item["hand_tv_ratio"])), "hand_tv_closeness_mean_finite": float(np.nanmean(item["hand_tv_closeness"])), "hand_direction_agreement_mean": float(np.nanmean(item["hand_direction_agreement"])), "zero_target_tv_episodes": int(np.sum(~np.isfinite(item["hand_tv_ratio"]))), "early_hand_mae_std": float(item["early_hand_mae_std"].mean()), "middle_hand_mae_std": float(item["middle_hand_mae_std"].mean()), "late_hand_mae_std": float(item["late_hand_mae_std"].mean()), "endpoint_hand_mae_std": float(item["endpoint_hand_mae_std"].mean()), "slope_fallback_episodes": slope_status["fallback_episodes"] if name == "SLOPE_BASELINE" else 0, "final_test_access": 0})
    write_csv(analysis / "MAIN_RESULTS.csv", metric_rows)
    write_json(analysis / "MAIN_RESULTS.json", {"status": "PASS", "rows": metric_rows, "development_only": True, "full_definition": "new Hand prediction concatenated with locked E-F-SP-G Arm prediction", "native_fd_definition": "29 future-to-future differences", "final_test_access": 0})

    temporal_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "SLOPE_BASELINE", "FIXED_HYBRID", *CELLS):
        item = metric[name]
        for horizon, sl in (("early", slice(0, 10)), ("middle", slice(10, 20)), ("late", slice(20, 30)), ("endpoint", 29)):
            if horizon == "endpoint":
                temporal_rows.append({"model": name, "horizon": horizon, "hand_mae_std": float(item["endpoint_hand_mae_std"].mean()), "hand_mae_native": float(item["endpoint_hand_mae_native"].mean()), "N_episode": 614})
            else:
                temporal_rows.append({"model": name, "horizon": horizon, "hand_mae_std": float(np.abs(models_std[name][:, sl, :HAND] - target_std[:, sl, :HAND]).mean()), "hand_mae_native": float(np.abs(models_native[name][:, sl, :HAND] - target_native[:, sl, :HAND]).mean()), "N_episode": 614})
    write_csv(analysis / "TEMPORAL_METRICS.csv", temporal_rows)

    per_task_rows: list[dict[str, object]] = []
    per_joint_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "SLOPE_BASELINE", "FIXED_HYBRID", *CELLS):
        item = metric[name]
        for task in sorted(np.unique(dev_tasks).tolist()):
            mask = dev_tasks == task
            per_task_rows.append({"model": name, "task_index": int(task), "N_episode": int(mask.sum()), "hand_mae_std": float(item["hand_mae_std"][mask].mean()), "hand_fd_native": float(item["hand_fd_native"][mask].mean()), "full_mae_std": float(item["full_mae_std"][mask].mean())})
        for joint in range(HAND):
            per_joint_rows.append({"model": name, "hand_joint": joint, "hand_mae_std": float(item["per_joint_hand_mae_std"][:, joint].mean()), "hand_fd_native": float(item["per_joint_hand_fd_native"][:, joint].mean())})
    write_csv(analysis / "PER_TASK_METRICS.csv", per_task_rows)
    write_csv(analysis / "PER_JOINT_METRICS.csv", per_joint_rows)

    bootstrap = np.load(BOOTSTRAP, allow_pickle=False)
    if bootstrap.shape != (10000, 614):
        raise ValueError(f"paired bootstrap shape failed: {bootstrap.shape}")
    pair_rows: list[dict[str, object]] = []
    contrast_specs = [("S-HIST_vs_S-LAST", "HI-S-LAST", "HI-S-HIST"), ("F-HIST_vs_F-LAST", "HI-F-LAST", "HI-F-HIST"), ("F-HIST_vs_S-HIST", "HI-S-HIST", "HI-F-HIST"), ("F-LAST_vs_S-LAST", "HI-S-LAST", "HI-F-LAST")]
    for contrast, control, candidate in contrast_specs:
        for metric_name in ("hand_mae_std", "hand_fd_native"):
            row = {"family": "four_fixed_contrasts_co_primary", "contrast": contrast, "control": control, "candidate": candidate, "metric": metric_name, **bootstrap_stats(metric[control][metric_name], metric[candidate][metric_name], bootstrap, 20260990 + len(pair_rows))}
            pair_rows.append(row)
    holm_adjust(pair_rows)
    for candidate, control in (("HI-F-HIST", "PERSISTENCE"), ("HI-S-HIST", "PERSISTENCE"), ("HI-F-HIST", "FIXED_HYBRID"), ("HI-S-HIST", "FIXED_HYBRID"), ("SLOPE_BASELINE", "PERSISTENCE")):
        for metric_name in ("hand_mae_std", "hand_fd_native", "full_mae_std"):
            if metric_name not in metric[candidate] or metric_name not in metric[control]:
                continue
            pair_rows.append({"family": "baseline_or_composite_descriptive", "contrast": f"{candidate}_vs_{control}", "control": control, "candidate": candidate, "metric": metric_name, **bootstrap_stats(metric[control][metric_name], metric[candidate][metric_name], bootstrap, 20261030 + len(pair_rows))})
    write_csv(analysis / "PAIRED_EFFECTS.csv", pair_rows)
    write_json(analysis / "PAIRED_EFFECTS.json", {"status": "PASS", "rows": pair_rows, "bootstrap_replicates": 10000, "bootstrap_unit": "episode", "signflip_null": "paired episode exchangeability within development population", "holm_family": "four fixed contrasts x two co-primary metrics", "final_test_access": 0})

    factorial_specs = [
        ("history_main_effect_STATE", "HI-S-LAST", "HI-S-HIST"),
        ("history_main_effect_FUSION", "HI-F-LAST", "HI-F-HIST"),
        ("video_main_effect_LAST", "HI-S-LAST", "HI-F-LAST"),
        ("video_main_effect_HIST", "HI-S-HIST", "HI-F-HIST"),
    ]
    factorial_rows: list[dict[str, object]] = []
    for effect_name, control, candidate_name in factorial_specs:
        for metric_name in ("hand_mae_std", "hand_fd_native"):
            factorial_rows.append({"effect": effect_name, "metric": metric_name, "descriptive_only": True, "control": control, "candidate": candidate_name, **bootstrap_stats(metric[control][metric_name], metric[candidate_name][metric_name], bootstrap, 20261200 + len(factorial_rows))})
    for metric_name in ("hand_mae_std", "hand_fd_native"):
        interaction = (metric["HI-F-LAST"][metric_name] - metric["HI-F-HIST"][metric_name]) - (metric["HI-S-LAST"][metric_name] - metric["HI-S-HIST"][metric_name])
        factorial_rows.append({"effect": "history_video_interaction", "metric": metric_name, "descriptive_only": True, "control": "difference_of_history_effects", "candidate": "fusion_minus_state", **bootstrap_stats(interaction, np.zeros_like(interaction), bootstrap, 20261220 + len(factorial_rows))})
    write_csv(analysis / "HISTORY_VIDEO_FACTORIAL.csv", factorial_rows)

    ratio_rows: list[dict[str, object]] = []
    for name in ("SLOPE_BASELINE", "HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST"):
        ratio_rows.append({"candidate": name, "metric": "hand_fd_native", **ratio_bootstrap(metric[name]["hand_fd_native"], metric["PERSISTENCE"]["hand_fd_native"], bootstrap)})
    write_csv(analysis / "FD_MARGIN_RESULTS.csv", ratio_rows)

    strata_rows = {int(row["episode_id"]): row for row in csv.DictReader(MOTION_STRATA.open(encoding="utf-8", newline=""))}
    strata = np.asarray([strata_rows[int(episode)]["motion_stratum"] for episode in dev_ids], dtype=object)
    motion_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "SLOPE_BASELINE", "FIXED_HYBRID", *CELLS):
        for stratum in ("low", "mid", "high"):
            mask = strata == stratum
            motion_rows.append({"model": name, "motion_stratum": stratum, "N_episode": int(mask.sum()), "hand_mae_std": float(metric[name]["hand_mae_std"][mask].mean()), "hand_fd_native": float(metric[name]["hand_fd_native"][mask].mean())})
    write_csv(analysis / "MOTION_STRATA_RESULTS.csv", motion_rows)
    if REPRESENTATIVE.is_file():
        representative_rows = list(csv.DictReader(REPRESENTATIVE.open(encoding="utf-8", newline="")))
        write_csv(analysis / "REPRESENTATIVE_EPISODES.csv", representative_rows)

    candidate = "HI-F-HIST"
    hand_vs_p = bootstrap_stats(metric["PERSISTENCE"]["hand_mae_std"], metric[candidate]["hand_mae_std"], bootstrap, 20261100)
    full_vs_hybrid = bootstrap_stats(metric["FIXED_HYBRID"]["full_mae_std"], metric[candidate]["full_mae_std"], bootstrap, 20261101)
    fd_margin = ratio_bootstrap(metric[candidate]["hand_fd_native"], metric["PERSISTENCE"]["hand_fd_native"], bootstrap)
    candidate_gate_status = "CANDIDATE_GATE_PASS" if hand_vs_p["ci_low"] > 0 and full_vs_hybrid["ci_low"] > 0 and fd_margin["relative_change_candidate_vs_control"] <= 0.01 and fd_margin["ci_high"] <= 0.01 and all(row["status"] == "PASS" for row in run_acceptance) else "CANDIDATE_GATE_FAIL"
    candidate_decision = {
        "status": candidate_gate_status,
        "candidate_priority": ["HI-F-HIST", "HI-S-HIST", "HI-F-LAST", "HI-S-LAST"],
        "selected_candidate": candidate,
        "hand_improvement_vs_persistence": hand_vs_p,
        "full_improvement_vs_fixed_hybrid": full_vs_hybrid,
        "native_fd_relative_change_vs_persistence": fd_margin,
        "gates": {
            "hand_ci_lower_gt_zero": hand_vs_p["ci_low"] > 0,
            "full_ci_lower_gt_zero": full_vs_hybrid["ci_low"] > 0,
            "native_fd_point_change_le_1pct": fd_margin["relative_change_candidate_vs_control"] <= 0.01,
            "native_fd_ci_upper_le_1pct": fd_margin["ci_high"] <= 0.01,
            "all_four_formal_runs_accepted": all(row["status"] == "PASS" for row in run_acceptance),
        },
        "multi_seed_authorized": False,
        "final_test_access": 0,
    }
    write_json(analysis / "CANDIDATE_DECISION.json", candidate_decision)
    write_json(ROOT / "RUN_ACCEPTANCE.json", {"status": "PASS", "stage0": "4/4", "formal_matrix": "4/4", "rows": run_acceptance, "slope_baseline": slope_status, "final_test_access": 0})
    print(json.dumps({"status": "PASS", "formal_runs": len(run_acceptance), "dev_rows": len(dev_positions), "candidate": candidate, "hand_mae": float(metric[candidate]["hand_mae_std"].mean()), "hand_fd_native": float(metric[candidate]["hand_fd_native"].mean()), "fd_relative_change": fd_margin["relative_change_candidate_vs_control"], "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
