#!/usr/bin/env python3
"""Run the locked train-OOF constrained-alpha calibration for HD-D1-L1."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
EVENT_ROOT = REPO / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
BOOTSTRAP_PATH = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/paired_episode_bootstrap_indices.npy"
BOOTSTRAP = 10_000
STEPS = 30
HAND = 14


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sampled = values[indices]
    estimates = np.nanmean(sampled, axis=1)
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) != BOOTSTRAP:
        raise RuntimeError(f"bootstrap finite replicate count {len(estimates)} != {BOOTSTRAP}")
    valid = values[np.isfinite(values)]
    return {"effect": float(valid.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "median": float(np.median(valid)), "replicates": BOOTSTRAP, "N_episode": int(len(valid))}


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, BOOTSTRAP, 500):
        count = min(500, BOOTSTRAP - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.abs((signs * values[None, :]).mean(axis=1)) >= observed))
    return float((1 + exceed) / (BOOTSTRAP + 1))


def paired(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    improvement = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    result = bootstrap_mean(improvement, indices)
    result.update({"win_rate": float(np.mean(improvement[np.isfinite(improvement)] > 0)), "p_raw": signflip_p(improvement, seed)})
    return result


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if len(numerator) != indices.shape[1] or len(denominator) != indices.shape[1]:
        raise RuntimeError("ratio arrays are not aligned with episode bootstrap")
    num = np.nanmean(numerator[indices], axis=1)
    den = np.nanmean(denominator[indices], axis=1)
    values = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > 1e-12)
    values = values[np.isfinite(values)]
    if len(values) != BOOTSTRAP:
        raise RuntimeError("ratio bootstrap denominator failure")
    return {"effect": float(np.nanmean(numerator) / np.nanmean(denominator)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": BOOTSTRAP, "N_episode": int(np.isfinite(numerator).sum())}


def event_motion(target: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    first = np.abs(target[:, :1] - persistence[:, :1]).mean(axis=2)
    transitions = np.abs(np.diff(target, axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def metrics(hand_std: np.ndarray, full_std: np.ndarray, hand_native: np.ndarray, full_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> dict[str, np.ndarray]:
    target_hand_std = target_std[:, :, :HAND]
    target_hand_native = target_native[:, :, :HAND]
    hand_err = np.abs(hand_std - target_hand_std)
    full_err = np.abs(full_std - target_std)
    return {
        "hand_mae_std": hand_err.mean(axis=(1, 2)),
        "hand_fd_std": np.abs(np.diff(hand_std, axis=1) - np.diff(target_hand_std, axis=1)).mean(axis=(1, 2)),
        "hand_fd_native": np.abs(np.diff(hand_native, axis=1) - np.diff(target_hand_native, axis=1)).mean(axis=(1, 2)),
        "full_mae_std": full_err.mean(axis=(1, 2)),
        "full_fd_std": np.abs(np.diff(full_std, axis=1) - np.diff(target_std, axis=1)).mean(axis=(1, 2)),
        "tv_closeness_hand": tv_closeness(hand_native, target_hand_native),
        "early_mae_std": hand_err[:, :10].mean(axis=(1, 2)),
        "middle_mae_std": hand_err[:, 10:20].mean(axis=(1, 2)),
        "late_mae_std": hand_err[:, 20:].mean(axis=(1, 2)),
        "endpoint_mae_std": hand_err[:, -1].mean(axis=1),
    }


def tv_closeness(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_tv = np.abs(np.diff(pred, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 1e-6)
    return np.abs(ratio - 1.0)


def task_joint_support(improvement: np.ndarray, tasks: np.ndarray, joint_improvement: np.ndarray) -> dict[str, object]:
    task_values: list[float] = []
    task_rows: list[dict[str, object]] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        value = float(improvement[mask].mean())
        supported = int(mask.sum()) >= 5
        task_rows.append({"task": int(task), "N_episode": int(mask.sum()), "supported": supported, "improvement": value})
        if supported and value > 0:
            task_values.append(value)
    positive_joints = joint_improvement[joint_improvement > 0]
    return {"task_rows": task_rows, "supported_positive_task_count": len(task_values), "max_positive_task_share": float(max(task_values) / sum(task_values)) if task_values else 1.0, "positive_joint_count": int((joint_improvement > 0).sum()), "max_positive_joint_share": float(positive_joints.max() / positive_joints.sum()) if len(positive_joints) else 1.0}


def main() -> int:
    calibration_dir = ROOT / "calibration"
    data = load_npz(INPUT)
    train_mask = data["split_code"] == 0
    oof = load_npz(DCT_ROOT / "oof/HD-D1-L1_OOF.npz")
    if oof["oof_predictions_hand_standardized"].shape != (2741, 30, 14):
        raise RuntimeError("HD-D1-L1 OOF shape failed")
    train_ids = oof["episode_index"].astype(np.int64)
    if len(np.unique(train_ids)) != 2741 or not np.array_equal(train_ids, data["episode_index"][train_mask]):
        raise RuntimeError("train OOF IDs failed")
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)[:, :HAND]
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)[:, :HAND]
    p_train_std = oof["persistence_hand_standardized"].astype(np.float64)
    n_train_std = oof["oof_predictions_hand_standardized"].astype(np.float64)
    t_train_std = oof["targets_hand_standardized"].astype(np.float64)
    p_train_native = oof["persistence_hand_native"].astype(np.float64)
    t_train_native = oof["targets_hand_native"].astype(np.float64)
    n_train_native = n_train_std * scale[None, :, :] + mean[None, :, :]
    alphas = np.arange(1001, dtype=np.float64) / 1000.0
    p_level = float(np.abs(p_train_std - t_train_std).mean())
    p_fd = float(np.abs(np.diff(p_train_native, axis=1) - np.diff(t_train_native, axis=1)).mean())
    rows: list[dict[str, object]] = []
    level_values: list[float] = []
    fd_values: list[float] = []
    for alpha in alphas:
        pred_std = p_train_std + alpha * (n_train_std - p_train_std)
        pred_native = p_train_native + alpha * (n_train_native - p_train_native)
        level = float(np.abs(pred_std - t_train_std).mean())
        fd = float(np.abs(np.diff(pred_native, axis=1) - np.diff(t_train_native, axis=1)).mean())
        level_values.append(level)
        fd_values.append(fd)
    ref_gain = p_level - level_values[-1]
    if ref_gain <= 0:
        raise RuntimeError(f"alpha=1 train level gain is not positive: {ref_gain}")
    for index, alpha in enumerate(alphas):
        gain = p_level - level_values[index]
        retention = gain / ref_gain
        fd_relative = (fd_values[index] - p_fd) / p_fd
        rows.append({"alpha": float(alpha), "train_hand_mae_std": level_values[index], "train_native_fd_mae": fd_values[index], "train_fd_relative_change_vs_persistence": fd_relative, "train_level_gain": gain, "train_level_gain_retention_vs_alpha1": retention, "feasible": bool(fd_relative <= 0.0 and retention >= 0.80)})
    feasible = [row for row in rows if row["feasible"] and float(row["alpha"]) > 0]
    if not feasible:
        selected_alpha = 0.0
        cal_available = False
    else:
        selected_alpha = min(feasible, key=lambda row: (float(row["train_hand_mae_std"]), float(row["alpha"]))) ["alpha"]
        cal_available = True
    write_csv(calibration_dir / "ALPHA_FRONTIER.csv", rows)
    alpha_registry = {"status": "LOCKED", "source_cell": "HD-D1-L1", "selected_alpha": float(selected_alpha), "alpha_positive_feasible": cal_available, "selection_rule": "lowest train OOF Hand standardized MAE among native FD relative<=0 and level retention>=80%; ties choose smaller alpha", "grid": {"start": 0.0, "stop": 1.0, "step": 0.001, "count": 1001}, "train_rows": 2741, "train_persistence_hand_mae_std": p_level, "train_persistence_native_fd_mae": p_fd, "train_alpha1_level_gain": ref_gain, "input_sha256": digest(INPUT), "dct_oof_sha256": digest(DCT_ROOT / "oof/HD-D1-L1_OOF.npz"), "normalizer_sha256": digest(NORMALIZER), "dev_used_for_alpha": False, "final_test_access": 0}
    write_json(calibration_dir / "ALPHA_REGISTRY.json", alpha_registry)
    (calibration_dir / "ALPHA_REGISTRY_SHA256.txt").write_text(digest(calibration_dir / "ALPHA_REGISTRY.json") + "  ALPHA_REGISTRY.json\n", encoding="utf-8")
    write_json(calibration_dir / "ALPHA_LOCK_BARRIER.json", {"status": "PASS", "registry_written": True, "registry_sha256": digest(calibration_dir / "ALPHA_REGISTRY.json"), "dev_read_started_after_registry": True, "selected_alpha": float(selected_alpha)})

    # Only after the alpha registry and its hash are present do we read dev outputs.
    dev_mask = data["split_code"] == 1
    target_std = data["targets_standardized"][dev_mask].astype(np.float64)
    target_native = data["targets_native"][dev_mask].astype(np.float64)
    p_dev_std_full = data["persistence_standardized"][dev_mask].astype(np.float64)
    p_dev_native_full = data["persistence_native"][dev_mask].astype(np.float64)
    p_dev_std = p_dev_std_full[:, :, :HAND]
    p_dev_native = p_dev_native_full[:, :, :HAND]
    d1_bundle = load_npz(DCT_ROOT / "predictions/HD-D1-L1_PREDICTIONS_DEV.npz")
    d1_std = d1_bundle["hand_predictions_standardized"].astype(np.float64)
    d1_native = d1_bundle["hand_predictions_native"].astype(np.float64)
    locked_arm = load_npz(LOCKED_ARM)
    arm_std = locked_arm["predictions_standardized"][:, :, 14:28].astype(np.float64)
    arm_native = locked_arm["predictions_native"][:, :, 14:28].astype(np.float64)
    if not np.array_equal(d1_bundle["episode_index"], data["episode_index"][dev_mask]) or not np.array_equal(d1_bundle["original_task_index"], data["original_task_index"][dev_mask]):
        raise RuntimeError("D1-L1 dev IDs failed")
    cal_std = p_dev_std + float(selected_alpha) * (d1_std - p_dev_std)
    cal_native = p_dev_native + float(selected_alpha) * (d1_native - p_dev_native)
    cal_full_std = np.concatenate((cal_std, arm_std), axis=2)
    cal_full_native = np.concatenate((cal_native, arm_native), axis=2)
    persistence_full_std = np.concatenate((p_dev_std, arm_std), axis=2)
    persistence_full_native = np.concatenate((p_dev_native, arm_native), axis=2)
    alpha1_full_std = np.concatenate((d1_std, arm_std), axis=2)
    alpha1_full_native = np.concatenate((d1_native, arm_native), axis=2)
    metric_cal = metrics(cal_std, cal_full_std, cal_native, cal_full_native, target_std, target_native)
    metric_p = metrics(p_dev_std, persistence_full_std, p_dev_native, persistence_full_native, target_std, target_native)
    metric_alpha1 = metrics(d1_std, alpha1_full_std, d1_native, alpha1_full_native, target_std, target_native)
    bootstrap = np.load(BOOTSTRAP_PATH, allow_pickle=False)
    if bootstrap.shape != (BOOTSTRAP, 614):
        raise RuntimeError("paired bootstrap shape failed")
    pair_specs = [("hand_mae_std", "Hand standardized MAE"), ("hand_fd_native", "Hand native first-difference"), ("hand_fd_std", "Hand standardized first-difference"), ("full_mae_std", "Full-28 standardized MAE"), ("tv_closeness_hand", "Hand native TV closeness")]
    pair_rows: list[dict[str, object]] = []
    pair_lookup: dict[str, dict[str, object]] = {}
    for metric_name, label in pair_specs:
        stats = paired(metric_p[metric_name], metric_cal[metric_name], bootstrap, 20261000 + len(pair_rows))
        pair_lookup[metric_name] = stats
        pair_rows.append({"comparison": "CAL_vs_PERSISTENCE_HYBRID", "metric": metric_name, "metric_label": label, **stats, "control_mean": float(np.nanmean(metric_p[metric_name])), "candidate_mean": float(np.nanmean(metric_cal[metric_name]))})
    cal_vs_alpha = []
    for metric_name, label in pair_specs:
        stats = paired(metric_alpha1[metric_name], metric_cal[metric_name], bootstrap, 20261100 + len(cal_vs_alpha))
        cal_vs_alpha.append({"comparison": "CAL_vs_ALPHA1_D1L1", "metric": metric_name, "metric_label": label, **stats, "control_mean": float(np.nanmean(metric_alpha1[metric_name])), "candidate_mean": float(np.nanmean(metric_cal[metric_name]))})
    write_csv(calibration_dir / "CAL_PAIRED_EFFECTS.csv", pair_rows + cal_vs_alpha)
    task_index = data["original_task_index"][dev_mask].astype(np.int64)
    motion = event_motion(target_std[:, :, :HAND], p_dev_std).mean(axis=1)
    thresholds = json.loads((EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"]
    high = motion >= float(thresholds["COMMON_STEP_Q75"])
    joint_gain = np.abs(p_dev_std - target_std[:, :, :HAND]).mean(axis=1) - np.abs(cal_std - target_std[:, :, :HAND]).mean(axis=1)
    task_support = task_joint_support(metric_p["hand_mae_std"] - metric_cal["hand_mae_std"], task_index, joint_gain.mean(axis=0))
    retention = ratio_stats(metric_p["hand_mae_std"] - metric_cal["hand_mae_std"], metric_p["hand_mae_std"] - metric_alpha1["hand_mae_std"], bootstrap)
    fd_rel = ratio_stats(metric_cal["hand_fd_native"] - metric_p["hand_fd_native"], metric_p["hand_fd_native"], bootstrap)
    std_fd_rel = ratio_stats(metric_cal["hand_fd_std"] - metric_p["hand_fd_std"], metric_p["hand_fd_std"], bootstrap)
    tv_rel = ratio_stats(metric_cal["tv_closeness_hand"] - metric_p["tv_closeness_hand"], metric_p["tv_closeness_hand"], bootstrap)
    def skill(mask: np.ndarray, key: str) -> float:
        return float((np.mean(metric_p[key][mask]) - np.mean(metric_cal[key][mask])) / np.mean(metric_p[key][mask])) if mask.any() else float("nan")
    checks = {"alpha_positive": float(selected_alpha) > 0, "hand_ci_beats_persistence": float(pair_lookup["hand_mae_std"]["ci_low"]) > 0, "full_ci_beats_locked_hybrid": float(pair_lookup["full_mae_std"]["ci_low"]) > 0, "level_gain_retention_ge_80pct": float(retention["effect"]) >= 0.80, "native_fd_degradation_le_1pct": float(fd_rel["effect"]) <= 0.01, "standardized_fd_degradation_le_1pct": float(std_fd_rel["effect"]) <= 0.01, "native_tv_degradation_le_1pct": float(tv_rel["effect"]) <= 0.01, "high_motion_not_over_1pct_reverse": skill(high, "hand_mae_std") >= -0.01, "late_not_over_1pct_reverse": skill(np.ones(614, dtype=bool), "late_mae_std") >= -0.01, "endpoint_not_over_1pct_reverse": skill(np.ones(614, dtype=bool), "endpoint_mae_std") >= -0.01, "task_support_not_single": int(task_support["supported_positive_task_count"]) >= 5 and float(task_support["max_positive_task_share"]) < 0.50, "joint_support_not_single": int(task_support["positive_joint_count"]) >= 5 and float(task_support["max_positive_joint_share"]) < 0.50, "schema_oof_and_hash_qa": True}
    cal_result = {"status": "PASS", "selected_alpha": float(selected_alpha), "alpha_registry_sha256": digest(calibration_dir / "ALPHA_REGISTRY.json"), "hand_mae_std": float(np.mean(metric_cal["hand_mae_std"])), "hand_mae_native": float(np.mean(np.abs(cal_native - target_native[:, :, :HAND]))), "native_fd_mae": float(np.mean(metric_cal["hand_fd_native"])), "standardized_fd_mae": float(np.mean(metric_cal["hand_fd_std"])), "full28_mae_std": float(np.mean(metric_cal["full_mae_std"])), "hand_gain_vs_persistence": float((np.mean(metric_p["hand_mae_std"]) - np.mean(metric_cal["hand_mae_std"])) / np.mean(metric_p["hand_mae_std"])), "full_gain_vs_locked_hybrid": float((np.mean(metric_p["full_mae_std"]) - np.mean(metric_cal["full_mae_std"])) / np.mean(metric_p["full_mae_std"])), "native_fd_relative_change_vs_persistence": fd_rel, "standardized_fd_relative_change_vs_persistence": std_fd_rel, "native_tv_closeness_relative_change_vs_persistence": tv_rel, "level_gain_retention_vs_alpha1": retention, "high_motion_relative_skill": skill(high, "hand_mae_std"), "late_relative_skill": skill(np.ones(614, dtype=bool), "late_mae_std"), "endpoint_relative_skill": skill(np.ones(614, dtype=bool), "endpoint_mae_std"), "task_joint_support": task_support, "checks": checks, "cal_gate": bool(all(checks.values())), "N_episode": 614, "bootstrap_replicates": BOOTSTRAP, "dev_used_for_alpha": False, "final_test_access": 0}
    write_json(calibration_dir / "CAL_RESULTS.json", cal_result)
    write_csv(calibration_dir / "CAL_RESULTS.csv", [{"model": "PERSISTENCE_HYBRID", **{key: float(np.nanmean(value)) for key, value in metric_p.items() if isinstance(value, np.ndarray)}}, {"model": "CALIBRATED_D1L1", **{key: float(np.nanmean(value)) for key, value in metric_cal.items() if isinstance(value, np.ndarray)}}])
    (calibration_dir / "CAL_RESULTS.md").write_text("# Constrained alpha calibration\n\n" + f"Selected alpha=`{selected_alpha:.3f}`; CAL gate=`{'PASS' if cal_result['cal_gate'] else 'FAIL'}`. Native FD relative change versus Persistence is `{fd_rel['effect']:.6f}`.\n\n" + "\n".join(f"- `{key}`: `{value}`" for key, value in checks.items()) + "\n", encoding="utf-8")
    np.savez_compressed(ROOT / "calibration/CAL_PREDICTIONS_DEV.npz", episode_index=data["episode_index"][dev_mask], original_task_index=task_index, hand_predictions_standardized=cal_std.astype(np.float32), hand_predictions_native=cal_native.astype(np.float32), full_predictions_standardized=cal_full_std.astype(np.float32), full_predictions_native=cal_full_native.astype(np.float32), persistence_hand_standardized=p_dev_std.astype(np.float32), persistence_hand_native=p_dev_native.astype(np.float32), q=np.ones((614, 30, 1), dtype=np.float32) * float(selected_alpha))
    print(json.dumps({"status": "PASS", "selected_alpha": float(selected_alpha), "cal_gate": bool(cal_result["cal_gate"]), "hand_mae": cal_result["hand_mae_std"], "native_fd": cal_result["native_fd_mae"], "fd_relative": fd_rel["effect"], "registry_sha256": digest(calibration_dir / "ALPHA_REGISTRY.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
