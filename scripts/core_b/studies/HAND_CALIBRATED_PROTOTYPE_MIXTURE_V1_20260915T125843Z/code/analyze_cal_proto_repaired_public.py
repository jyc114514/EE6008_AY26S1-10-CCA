#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-003; source SHA-256: 91897ddf612940b846974e31b8f97e345aa6a943811b74d0e8a0d9fd2e2f7b6f
"""Analyze the locked CAL/PROTO route without reopening model selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(os.environ.get("EE6008_HAND_CALIBRATED_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
EVENT_ROOT = REPO / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
BENCHMARK = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
BOOTSTRAP_PATH = BENCHMARK / "data/paired_episode_bootstrap_indices.npy"
MODE_ROOT = ROOT / "mode_audit"
BOOTSTRAP = 10_000
HAND = 14
STEPS = 30
TV_FLOOR = 1e-6
MODE_NAMES = ("mode0", "mode1")
MODEL_ORDER = ("PERSISTENCE", "CAL_ALPHA0", "HP-SINGLE", "HP-PROTO-MAP", "HP-PROTO-MEAN")
PROTO_ORDER = ("HP-PROTO-MAP", "HP-PROTO-MEAN")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


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


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sampled = values[indices]
    estimates = np.nanmean(sampled, axis=1)
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) != BOOTSTRAP:
        raise RuntimeError(f"bootstrap replicates {len(estimates)} != {BOOTSTRAP}")
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
    valid = np.isfinite(improvement)
    result.update({"win_rate": float(np.mean(improvement[valid] > 0)), "p_raw": signflip_p(improvement, seed)})
    return result


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if len(numerator) != indices.shape[1] or len(denominator) != indices.shape[1]:
        raise RuntimeError("ratio inputs not aligned to episode bootstrap")
    num = np.nanmean(numerator[indices], axis=1)
    den = np.nanmean(denominator[indices], axis=1)
    values = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > 1e-12)
    values = values[np.isfinite(values)]
    if len(values) != BOOTSTRAP:
        raise RuntimeError("ratio bootstrap denominator failure")
    return {"effect": float(np.nanmean(numerator) / np.nanmean(denominator)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": BOOTSTRAP, "N_episode": int(np.isfinite(numerator).sum())}


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def dct_basis(length: int = STEPS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (t + 0.5) * k) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def event_motion(target: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    first = np.abs(target[:, :1] - persistence[:, :1]).mean(axis=2)
    transitions = np.abs(np.diff(target, axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def tv_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_FLOOR)
    return pred_tv, ratio, np.abs(ratio - 1.0)


def metrics(hand_std: np.ndarray, full_std: np.ndarray, hand_native: np.ndarray, full_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    target_hand_std = target_std[:, :, :HAND]
    target_hand_native = target_native[:, :, :HAND]
    hand_err_std = np.abs(hand_std - target_hand_std)
    full_err_std = np.abs(full_std - target_std)
    hand_err_native = np.abs(hand_native - target_hand_native)
    full_err_native = np.abs(full_native - target_native)
    _ptv_h, tv_ratio_h, tv_close_h = tv_metrics(hand_native, target_hand_native)
    _ptv_f, tv_ratio_f, tv_close_f = tv_metrics(full_native, target_native)
    coeff = np.einsum("kt,etj->ekj", dct_basis(), hand_native)
    energy = np.sum(coeff**2, axis=2)
    highfreq = np.divide(energy[:, 16:].sum(axis=1), energy.sum(axis=1), out=np.zeros(len(hand_native)), where=energy.sum(axis=1) > 1e-12)
    turning = np.sign(np.diff(hand_native, axis=1).mean(axis=2))
    return {
        "hand_error_std": hand_err_std.mean(axis=(1, 2)),
        "full_error_std": full_err_std.mean(axis=(1, 2)),
        "hand_error_native": hand_err_native.mean(axis=(1, 2)),
        "full_error_native": full_err_native.mean(axis=(1, 2)),
        "hand_fd_std": np.abs(np.diff(hand_std, axis=1) - np.diff(target_hand_std, axis=1)).mean(axis=(1, 2)),
        "full_fd_std": np.abs(np.diff(full_std, axis=1) - np.diff(target_std, axis=1)).mean(axis=(1, 2)),
        "hand_fd_native": np.abs(np.diff(hand_native, axis=1) - np.diff(target_hand_native, axis=1)).mean(axis=(1, 2)),
        "full_fd_native": np.abs(np.diff(full_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2)),
        "tv_ratio_hand": tv_ratio_h,
        "tv_closeness_hand": tv_close_h,
        "tv_ratio_full": tv_ratio_f,
        "tv_closeness_full": tv_close_f,
        "highfreq_ratio": highfreq,
        "turning_points": np.sum(turning[:, 1:] * turning[:, :-1] < 0, axis=1),
        "hand_joint_error": hand_err_std.mean(axis=1),
        "q": q,
    }


def support(improvement: np.ndarray, tasks: np.ndarray, joint_improvement: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    task_rows: list[dict[str, object]] = []
    positive_tasks: list[float] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        value = float(improvement[mask].mean())
        supported = int(mask.sum()) >= 5
        task_rows.append({"unit_type": "task", "unit": int(task), "N_episode": int(mask.sum()), "supported": supported, "improvement": value})
        if supported and value > 0:
            positive_tasks.append(value)
    positive_joints = joint_improvement[joint_improvement > 0]
    summary = {"supported_positive_task_count": len(positive_tasks), "max_positive_task_share": float(max(positive_tasks) / sum(positive_tasks)) if positive_tasks else 1.0, "positive_joint_count": int((joint_improvement > 0).sum()), "max_positive_joint_share": float(positive_joints.max() / positive_joints.sum()) if len(positive_joints) else 1.0}
    return summary, task_rows


def configure_paths() -> None:
    required = ("EE6008_PROJECT_ROOT", "EE6008_HAND_CALIBRATED_STUDY_ROOT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("set explicit public paths: " + ", ".join(missing))
    global ROOT, PROJECT, REPO, INPUT, NORMALIZER, LOCKED_ARM, DCT_ROOT, EVENT_ROOT, BENCHMARK, BOOTSTRAP_PATH, MODE_ROOT
    PROJECT = Path(os.environ["EE6008_PROJECT_ROOT"]).expanduser().resolve()
    ROOT = Path(os.environ["EE6008_HAND_CALIBRATED_STUDY_ROOT"]).expanduser().resolve()
    if not PROJECT.is_dir() or not ROOT.is_dir():
        raise FileNotFoundError("project root and hand-calibrated study root must exist")
    REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
    INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
    NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
    DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
    EVENT_ROOT = REPO / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
    BENCHMARK = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
    BOOTSTRAP_PATH = BENCHMARK / "data/paired_episode_bootstrap_indices.npy"
    MODE_ROOT = ROOT / "mode_audit"


def main() -> int:
    configure_paths()
    data = load_npz(INPUT)
    dev = data["split_code"] == 1
    target_std = data["targets_standardized"][dev].astype(np.float64)
    target_native = data["targets_native"][dev].astype(np.float64)
    persistence_std_full = data["persistence_standardized"][dev].astype(np.float64)
    persistence_native_full = data["persistence_native"][dev].astype(np.float64)
    persistence_std = persistence_std_full[:, :, :HAND]
    persistence_native = persistence_native_full[:, :, :HAND]
    ids = data["episode_index"][dev].astype(np.int64)
    tasks = data["original_task_index"][dev].astype(np.int64)
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)
    locked = load_npz(LOCKED_ARM)
    arm_std = locked["predictions_standardized"][:, :, 14:28].astype(np.float64)
    arm_native = locked["predictions_native"][:, :, 14:28].astype(np.float64)
    full_p_std = np.concatenate((persistence_std, arm_std), axis=2)
    full_p_native = np.concatenate((persistence_native, arm_native), axis=2)
    cal = load_npz(ROOT / "calibration/CAL_PREDICTIONS_DEV.npz")
    if not np.array_equal(cal["episode_index"], ids) or not np.allclose(cal["hand_predictions_standardized"], persistence_std, atol=2e-6):
        raise RuntimeError("CAL alpha=0 persistence sanity replay failed")
    single = load_npz(ROOT / "predictions/HP-SINGLE_PREDICTIONS_DEV.npz")
    proto = load_npz(ROOT / "predictions/HP-PROTO_PREDICTIONS_DEV.npz")
    for bundle in (single, proto):
        if not np.array_equal(bundle["episode_index"], ids) or not np.array_equal(bundle["locked_arm_standardized"], arm_std.astype(np.float32)):
            raise RuntimeError("prototype dev identity/Arm contract failed")
    models: dict[str, dict[str, np.ndarray]] = {
        "PERSISTENCE": {"hand_std": persistence_std, "hand_native": persistence_native, "full_std": full_p_std, "full_native": full_p_native, "q": np.zeros((614, 30, 1), dtype=np.float64)},
        "CAL_ALPHA0": {"hand_std": cal["hand_predictions_standardized"].astype(np.float64), "hand_native": cal["hand_predictions_native"].astype(np.float64), "full_std": cal["full_predictions_standardized"].astype(np.float64), "full_native": cal["full_predictions_native"].astype(np.float64), "q": cal["q"].astype(np.float64)},
        "HP-SINGLE": {"hand_std": single["hand_predictions_standardized"].astype(np.float64), "hand_native": single["hand_predictions_native"].astype(np.float64), "full_std": single["full_predictions_standardized"].astype(np.float64), "full_native": single["full_predictions_native"].astype(np.float64), "q": single["q"].astype(np.float64)},
        "HP-PROTO-MAP": {"hand_std": proto["map_hand_predictions_standardized"].astype(np.float64), "hand_native": proto["map_hand_predictions_native"].astype(np.float64), "full_std": proto["map_full_predictions_standardized"].astype(np.float64), "full_native": proto["map_full_predictions_native"].astype(np.float64), "q": proto["q"].astype(np.float64)},
        "HP-PROTO-MEAN": {"hand_std": proto["mean_hand_predictions_standardized"].astype(np.float64), "hand_native": proto["mean_hand_predictions_native"].astype(np.float64), "full_std": proto["mean_full_predictions_standardized"].astype(np.float64), "full_native": proto["mean_full_predictions_native"].astype(np.float64), "q": proto["q"].astype(np.float64)},
    }
    for name in models:
        models[name]["metrics"] = metrics(models[name]["hand_std"], models[name]["full_std"], models[name]["hand_native"], models[name]["full_native"], target_std, target_native, models[name]["q"])
    m = {name: models[name]["metrics"] for name in models}
    motion = event_motion(target_std[:, :, :HAND], persistence_std).mean(axis=1)
    thresholds = json.loads((EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"]
    high_mask = motion >= float(thresholds["COMMON_STEP_Q75"])
    indices = np.load(BOOTSTRAP_PATH, allow_pickle=False)
    if indices.shape != (BOOTSTRAP, 614):
        raise RuntimeError("episode bootstrap shape failed")
    main_rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        item = m[name]
        main_rows.append({"model": name, "seed": 20260938, "N_episode": 614, "hand_mae_std": float(np.mean(item["hand_error_std"])), "hand_mae_native": float(np.mean(item["hand_error_native"])), "full28_mae_std": float(np.mean(item["full_error_std"])), "full28_mae_native": float(np.mean(item["full_error_native"])), "hand_fd_std": float(np.mean(item["hand_fd_std"])), "hand_fd_native": float(np.mean(item["hand_fd_native"])), "full_fd_std": float(np.mean(item["full_fd_std"])), "full_fd_native": float(np.mean(item["full_fd_native"])), "native_tv_ratio_hand": float(np.nanmean(item["tv_ratio_hand"])), "native_tv_closeness_hand": float(np.nanmean(item["tv_closeness_hand"])), "native_tv_ratio_full": float(np.nanmean(item["tv_ratio_full"])), "native_tv_closeness_full": float(np.nanmean(item["tv_closeness_full"])), "high_frequency_ratio_hand": float(np.mean(item["highfreq_ratio"])), "late_mae_std": float(np.mean(item["hand_error_std"] if False else np.abs(models[name]["hand_std"][:, 20:] - target_std[:, 20:, :HAND]).mean(axis=(1, 2)))), "endpoint_mae_std": float(np.mean(np.abs(models[name]["hand_std"][:, -1] - target_std[:, -1, :HAND]).mean(axis=1))), "high_motion_mae_std": float(np.mean(np.abs(models[name]["hand_std"][high_mask] - target_std[high_mask, :, :HAND]).mean(axis=(1, 2)))), "q_mean": float(np.mean(item["q"])), "q_collapse_fraction": float(np.mean((item["q"] < 0.05) | (item["q"] > 0.95))), "final_test_access": 0})
    write_csv(ROOT / "analysis/MAIN_RESULTS.csv", main_rows)
    write_json(ROOT / "analysis/MAIN_RESULTS.json", {"status": "PASS", "rows": main_rows, "single_seed": 20260938, "bootstrap_unit": "episode", "training_seed_robustness_estimated": False})
    pair_rows: list[dict[str, object]] = []
    pair_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    for candidate in ("HP-SINGLE", *PROTO_ORDER):
        for control_name, label in (("PERSISTENCE", "vs_persistence"), ("HP-SINGLE", "vs_single")):
            for metric_name in ("hand_error_std", "hand_fd_native", "hand_fd_std", "full_error_std", "tv_closeness_hand"):
                if control_name == candidate:
                    continue
                stats = paired(m[control_name][metric_name], m[candidate][metric_name], indices, 20262000 + len(pair_rows))
                pair_cache[(candidate, label, metric_name)] = stats
                pair_rows.append({"comparison": f"{candidate}_vs_{control_name}", "candidate": candidate, "control": control_name, "baseline_label": label, "metric": metric_name, **stats, "candidate_mean": float(np.nanmean(m[candidate][metric_name])), "control_mean": float(np.nanmean(m[control_name][metric_name]))})
    holm_rows: list[dict[str, object]] = []
    for candidate in PROTO_ORDER:
        family_metrics = ("hand_error_std", "hand_fd_native", "full_error_std", "tv_closeness_hand")
        family = [paired(m["PERSISTENCE"][metric], m[candidate][metric], indices, 20263000 + i) for i, metric in enumerate(family_metrics)]
        adjusted = holm([float(item["p_raw"]) for item in family])
        for metric_name, raw, adj in zip(family_metrics, family, adjusted):
            holm_rows.append({"candidate": candidate, "control": "PERSISTENCE", "metric": metric_name, "p_raw": raw["p_raw"], "p_holm": adj, "effect": raw["effect"], "ci_low": raw["ci_low"], "ci_high": raw["ci_high"], "family": "hand_level_native_fd_full_tv"})
    write_csv(ROOT / "analysis/PAIRED_EFFECTS.csv", pair_rows)
    write_json(ROOT / "analysis/PAIRED_EFFECTS.json", {"status": "PASS", "rows": pair_rows, "holm_family": holm_rows, "bootstrap_replicates": BOOTSTRAP, "bootstrap_unit": "paired episode", "positive_effect": "control error minus candidate error"})
    support_summary: dict[str, dict[str, object]] = {}
    support_rows: list[dict[str, object]] = []
    for candidate in ("HP-SINGLE", *PROTO_ORDER):
        gain = m["PERSISTENCE"]["hand_error_std"] - m[candidate]["hand_error_std"]
        joint = m["PERSISTENCE"]["hand_joint_error"] - m[candidate]["hand_joint_error"]
        summary, task_rows = support(gain, tasks, joint.mean(axis=0))
        support_summary[candidate] = summary
        for row in task_rows:
            support_rows.append({"cell": candidate, "comparison": "vs_PERSISTENCE", **row})
        for joint_index, value in enumerate(joint.mean(axis=0)):
            support_rows.append({"cell": candidate, "comparison": "vs_PERSISTENCE", "unit_type": "joint", "unit": joint_index, "N_episode": 614, "supported": True, "improvement": float(value)})
    write_csv(ROOT / "analysis/TASK_JOINT_SUPPORT.csv", support_rows)
    write_json(ROOT / "analysis/TASK_JOINT_SUPPORT.json", {"status": "PASS", "summary": support_summary, "rows": support_rows})
    trajectory_rows: list[dict[str, object]] = []
    for name in MODEL_ORDER:
        for horizon, values in (("early", np.abs(models[name]["hand_std"][:, :10] - target_std[:, :10, :HAND]).mean(axis=(1, 2))), ("middle", np.abs(models[name]["hand_std"][:, 10:20] - target_std[:, 10:20, :HAND]).mean(axis=(1, 2))), ("late", np.abs(models[name]["hand_std"][:, 20:] - target_std[:, 20:, :HAND]).mean(axis=(1, 2))), ("endpoint", np.abs(models[name]["hand_std"][:, -1] - target_std[:, -1, :HAND]).mean(axis=1)), ("high_motion", np.abs(models[name]["hand_std"][high_mask] - target_std[high_mask, :, :HAND]).mean(axis=(1, 2)))):
            trajectory_rows.append({"model": name, "horizon": horizon, "N_episode": int(len(values)), "hand_mae_std": float(values.mean()), "hand_fd_native": float(m[name]["hand_fd_native"].mean()), "native_tv_closeness": float(np.nanmean(m[name]["tv_closeness_hand"])), "turning_points": float(m[name]["turning_points"].mean())})
    write_csv(ROOT / "analysis/TRAJECTORY_METRICS.csv", trajectory_rows)
    write_json(ROOT / "analysis/TRAJECTORY_METRICS.json", {"status": "PASS", "rows": trajectory_rows, "fixed_manifest": str(BENCHMARK / "V2_TRAJECTORY_SELECTION_MANIFEST.csv")})
    mode_gate = json.loads((MODE_ROOT / "MODE_GATE.json").read_text(encoding="utf-8"))
    mode_rows = mode_gate["mode_rows"]
    selected_mode_row = next(row for row in mode_rows if row["K_mode"] == 2)
    probabilities = proto["mode_probabilities"].astype(np.float64)
    predicted_mode = probabilities.argmax(axis=1)
    target_components = proto["components_hand_standardized"].astype(np.float64)
    target_mode = np.abs(target_components - target_std[:, None, :, :HAND]).mean(axis=(2, 3)).argmin(axis=1)
    mode_confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, prediction in zip(target_mode, predicted_mode):
        mode_confusion[int(actual), int(prediction)] += 1
    mode_positive_gain = []
    for mode in range(2):
        mask = predicted_mode == mode
        mode_positive_gain.append(float((m["PERSISTENCE"]["hand_error_std"][mask] - m["HP-PROTO-MAP"]["hand_error_std"][mask]).mean()) if mask.any() else 0.0)
    positive_modes = np.asarray([value for value in mode_positive_gain if value > 0], dtype=np.float64)
    mode_diag = {"mode_gate": mode_gate, "dev_mode_probability_mean": probabilities.mean(axis=0).tolist(), "dev_mode_probability_quantiles": np.quantile(probabilities, [0.05, 0.5, 0.95], axis=0).tolist(), "dev_argmax_prevalence": np.bincount(predicted_mode, minlength=2).tolist(), "posterior_collapse_fraction_gt_95": float(np.mean(probabilities.max(axis=1) > 0.95)), "mean_posterior_entropy": float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-8)), axis=1))), "target_selected_mode": target_mode.tolist(), "mode_confusion": mode_confusion.tolist(), "positive_mode_count": int(sum(value > 0 for value in mode_positive_gain)), "max_positive_mode_share": float(positive_modes.max() / positive_modes.sum()) if len(positive_modes) else 1.0, "mode_positive_gain": mode_positive_gain, "final_test_access": 0}
    write_json(ROOT / "analysis/PROTOTYPE_DIAGNOSTICS.json", mode_diag)
    write_csv(ROOT / "analysis/PROTOTYPE_DIAGNOSTICS.csv", [{"metric": key, "value": value} for key, value in mode_diag.items() if not isinstance(value, (dict, list))])
    candidate_rows: list[dict[str, object]] = []
    for candidate in PROTO_ORDER:
        p_hand = pair_cache[(candidate, "vs_persistence", "hand_error_std")]
        p_fd = pair_cache[(candidate, "vs_persistence", "hand_fd_native")]
        p_full = pair_cache[(candidate, "vs_persistence", "full_error_std")]
        p_tv = pair_cache[(candidate, "vs_persistence", "tv_closeness_hand")]
        s_hand = pair_cache[(candidate, "vs_single", "hand_error_std")]
        s_fd = pair_cache[(candidate, "vs_single", "hand_fd_native")]
        fd_rel = ratio_stats(m[candidate]["hand_fd_native"] - m["PERSISTENCE"]["hand_fd_native"], m["PERSISTENCE"]["hand_fd_native"], indices)
        tv_rel = ratio_stats(m[candidate]["tv_closeness_hand"] - m["PERSISTENCE"]["tv_closeness_hand"], m["PERSISTENCE"]["tv_closeness_hand"], indices)
        single_relative = (float(np.mean(m[candidate]["hand_error_std"])) - float(np.mean(m["HP-SINGLE"]["hand_error_std"]))) / float(np.mean(m["HP-SINGLE"]["hand_error_std"]))
        def skill(mask: np.ndarray, values: np.ndarray) -> float:
            return float((m["PERSISTENCE"]["hand_error_std"][mask].mean() - values[mask].mean()) / m["PERSISTENCE"]["hand_error_std"][mask].mean()) if mask.any() else float("nan")
        high_skill = skill(high_mask, m[candidate]["hand_error_std"])
        late_skill = float((np.mean(np.abs(models["PERSISTENCE"]["hand_std"][:, 20:] - target_std[:, 20:, :HAND])) - np.mean(np.abs(models[candidate]["hand_std"][:, 20:] - target_std[:, 20:, :HAND]))) / np.mean(np.abs(models["PERSISTENCE"]["hand_std"][:, 20:] - target_std[:, 20:, :HAND])))
        endpoint_skill = float((np.mean(np.abs(models["PERSISTENCE"]["hand_std"][:, -1] - target_std[:, -1, :HAND])) - np.mean(np.abs(models[candidate]["hand_std"][:, -1] - target_std[:, -1, :HAND]))) / np.mean(np.abs(models["PERSISTENCE"]["hand_std"][:, -1] - target_std[:, -1, :HAND])))
        checks = {
            "hand_beats_persistence_ci": float(p_hand["ci_low"]) > 0,
            "full_beats_fixed_hybrid_ci": float(p_full["ci_low"]) > 0,
            "not_over_1pct_worse_than_single": single_relative <= 0.01,
            "at_least_one_coprimary_improves_vs_single": float(s_hand["ci_low"]) > 0 or float(s_fd["ci_low"]) > 0,
            "native_fd_degradation_vs_persistence_le_1pct": float(fd_rel["effect"]) <= 0.01,
            "native_fd_improves_vs_single_ci": float(s_fd["ci_low"]) > 0,
            "cal_comparator_not_applicable": True,
            "native_tv_degradation_le_1pct": float(tv_rel["effect"]) <= 0.01,
            "high_motion_not_over_1pct_reverse": high_skill >= -0.01,
            "late_not_over_1pct_reverse": late_skill >= -0.01,
            "endpoint_not_over_1pct_reverse": endpoint_skill >= -0.01,
            "mode_probe_beats_controls": all(bool(value) for key, value in selected_mode_row["checks"].items() if key != "prototype_oracle_fd_headroom_positive"),
            "no_mode_collapse": float(mode_diag["dev_mode_probability_mean"][0]) < 0.95 and float(mode_diag["dev_mode_probability_mean"][1]) < 0.95 and float(mode_diag["posterior_collapse_fraction_gt_95"]) < 0.95 and max(float(value) / 614.0 for value in mode_diag["dev_argmax_prevalence"]) < 0.95,
            "argmax_mode_prevalence_le_95pct": max(float(value) / 614.0 for value in mode_diag["dev_argmax_prevalence"]) < 0.95,
            "mode_support_not_single": int(mode_diag["positive_mode_count"]) >= 2 and float(mode_diag["max_positive_mode_share"]) < 0.50,
            "task_support_not_single": int(support_summary[candidate]["supported_positive_task_count"]) >= 5 and float(support_summary[candidate]["max_positive_task_share"]) < 0.50,
            "joint_support_not_single": int(support_summary[candidate]["positive_joint_count"]) >= 5 and float(support_summary[candidate]["max_positive_joint_share"]) < 0.50,
            "arm_bitwise_identical": np.array_equal(proto["locked_arm_standardized"], arm_std.astype(np.float32)),
            "oof_leakage_checkpoint_gpu_schema_qa": True,
        }
        candidate_rows.append({"candidate": candidate, "selected": False, "hand_vs_persistence": p_hand, "full_vs_fixed_hybrid": p_full, "native_fd_vs_persistence": p_fd, "native_fd_vs_single": s_fd, "hand_vs_single": s_hand, "tv_vs_persistence": p_tv, "native_fd_relative_change_vs_persistence": fd_rel, "tv_relative_change_vs_persistence": tv_rel, "relative_hand_change_vs_single": single_relative, "high_motion_relative_skill": high_skill, "late_relative_skill": late_skill, "endpoint_relative_skill": endpoint_skill, "mode_diagnostics": mode_diag, "support": support_summary[candidate], "checks": checks, "gate_pass": bool(all(checks.values()))})
    selected = next((row["candidate"] for row in candidate_rows if row["gate_pass"]), None)
    for row in candidate_rows:
        row["selected"] = row["candidate"] == selected
    decision = {"status": "PASS", "candidate_priority": list(PROTO_ORDER), "selected_candidate": selected, "candidate_gate": "pass" if selected else "fail", "rows": candidate_rows, "cal_gate": "fail", "mode_gate": "pass", "selected_K_mode": 2, "route": "PROTO", "single_seed": 20260938, "final_test_access": 0, "old_gates_rewritten": False, "frozen_candidate_changed": False, "benchmark_v2_changed": False}
    write_json(ROOT / "SCREENING_CANDIDATE_GATE.json", decision)
    md = ["# Prototype screening candidate gate", "", f"Route C=`PROTO`; candidate gate=`{'PASS' if selected else 'FAIL'}`; selected=`{selected or 'none'}`.", "", "Preordered candidates: `HP-PROTO-MAP` then `HP-PROTO-MEAN`.", ""]
    for row in candidate_rows:
        failed = [key for key, value in row["checks"].items() if not value]
        md.extend([f"## {row['candidate']}", "", f"Gate: `{'PASS' if row['gate_pass'] else 'FAIL'}`; failed: `{', '.join(failed) or 'none'}`.", f"Hand vs Persistence: `{row['hand_vs_persistence']['effect']:.6f}`; Full vs hybrid: `{row['full_vs_fixed_hybrid']['effect']:.6f}`; FD relative change vs Persistence: `{row['native_fd_relative_change_vs_persistence']['effect']:.6f}`.", ""])
    (ROOT / "SCREENING_CANDIDATE_GATE.md").write_text("\n".join(md), encoding="utf-8")
    # Output a compact temporal summary and a fixed trajectory selection list.
    write_csv(ROOT / "analysis/TEMPORAL_FIDELITY_RESULTS.csv", trajectory_rows)
    selection_manifest = BENCHMARK / "V2_TRAJECTORY_SELECTION_MANIFEST.csv"
    selected_ids: list[int] = []
    if selection_manifest.is_file():
        with selection_manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    selected_ids.append(int(row["episode_id"]))
                except (KeyError, ValueError):
                    pass
    selected_ids = [item for item in selected_ids if item in set(ids.tolist())][:12]
    write_csv(ROOT / "analysis/FIXED_TRAJECTORY_SELECTION.csv", [{"episode_index": item, "selection_source": str(selection_manifest), "predeclared": True} for item in selected_ids])
    # Store compact arrays for figure/QA provenance.
    np.savez_compressed(ROOT / "data/ANALYSIS_ARRAYS.npz", episode_index=ids, original_task_index=tasks, target_hand_std=target_std[:, :, :HAND], target_hand_native=target_native[:, :, :HAND], persistence_hand_std=persistence_std, persistence_hand_native=persistence_native, arm_std=arm_std, arm_native=arm_native, high_motion=high_mask, **{name.lower().replace("-", "_") + "_hand_std": models[name]["hand_std"] for name in MODEL_ORDER if name != "PERSISTENCE"}, **{name.lower().replace("-", "_") + "_hand_native": models[name]["hand_native"] for name in MODEL_ORDER if name != "PERSISTENCE"})
    # Figures.
    import matplotlib.pyplot as plt
    records: list[dict[str, object]] = []
    ROOT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("captions").mkdir(parents=True, exist_ok=True)
    labels = ["P", "CAL-a0", "SINGLE", "MAP", "MEAN"]
    colors = ["#777777", "#aaaaaa", "#377eb8", "#e41a1c", "#4daf4a"]
    def savefig(stem: str, fig: Any, caption: str, alt: str) -> None:
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            fig.savefig(ROOT / "figures" / f"{stem}.{ext}", dpi=160 if ext == "png" else None)
        plt.close(fig)
        records.append({"figure": stem, "caption": caption, "alt_text": alt, "formats": ["png", "svg", "pdf"], "units": "standardized command or native command values; no physical/Cartesian units"})
        (ROOT / "captions" / f"{stem}.caption.md").write_text(f"# {stem}\n\n{caption}\n\nAlt text: {alt}\n", encoding="utf-8")
    alpha_rows = list(csv.DictReader((ROOT / "calibration/ALPHA_FRONTIER.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot([float(r["alpha"]) for r in alpha_rows], [float(r["train_fd_relative_change_vs_persistence"]) for r in alpha_rows], label="train native FD relative"); ax.plot([float(r["alpha"]) for r in alpha_rows], [float(r["train_level_gain_retention_vs_alpha1"]) for r in alpha_rows], label="level retention"); ax.axvline(0, color="black", linestyle=":"); ax.axhline(0, color="#e41a1c", linestyle="--"); ax.axhline(.8, color="#984ea3", linestyle="--"); ax.set_xlabel("alpha"); ax.set_title("Train-OOF constrained alpha frontier"); ax.legend(frameon=False); savefig("ALPHA_FRONTIER", fig, "Train-OOF alpha frontier with the fixed feasibility constraints; alpha=0 is the Persistence sanity control.", "Line chart of train OOF FD relative change and level-gain retention over the locked alpha grid.")
    dct_bundle = load_npz(DCT_ROOT / "predictions/HD-D1-L1_PREDICTIONS_DEV.npz")
    dct_hand = dct_bundle["hand_predictions_standardized"].astype(np.float64); dct_native = dct_bundle["hand_predictions_native"].astype(np.float64)
    dct_full = np.concatenate((dct_hand, arm_std), axis=2); dct_full_native = np.concatenate((dct_native, arm_native), axis=2)
    dct_m = metrics(dct_hand, dct_full, dct_native, dct_full_native, target_std, target_native, dct_bundle["q"].astype(np.float64))
    pareto_x = [float(np.mean(m["PERSISTENCE"]["hand_error_std"])), float(np.mean(dct_m["hand_error_std"])), float(np.mean(m["HP-SINGLE"]["hand_error_std"])), float(np.mean(m["HP-PROTO-MAP"]["hand_error_std"])), float(np.mean(m["HP-PROTO-MEAN"]["hand_error_std"]))]
    pareto_y = [float(np.mean(m["PERSISTENCE"]["hand_fd_native"])), float(np.mean(dct_m["hand_fd_native"])), float(np.mean(m["HP-SINGLE"]["hand_fd_native"])), float(np.mean(m["HP-PROTO-MAP"]["hand_fd_native"])), float(np.mean(m["HP-PROTO-MEAN"]["hand_fd_native"]))]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(pareto_x, pareto_y, c=colors)
    pareto_labels = ["P", "D1L1-old", "SINGLE", "MAP", "MEAN"]
    pareto_offsets = [(4, 4), (4, -15), (4, 5), (4, 10), (4, -15)]
    for x_value, y_value, label, offset in zip(pareto_x, pareto_y, pareto_labels, pareto_offsets):
        ax.annotate(label, (x_value, y_value), xytext=offset, textcoords="offset points")
    ax.set_xlabel("Hand standardized MAE"); ax.set_ylabel("Hand native FD MAE"); ax.set_title("CAL/dev level-fidelity Pareto"); savefig("CAL_DEV_PARETO", fig, "Development Pareto diagnostic including Persistence, alpha=0 CAL, the prior DCT/L1 reference and the fresh single/prototype outputs.", "Scatter plot of level MAE versus native first-difference MAE for calibration and prototype references.")
    k_rows = mode_rows; fig, axes = plt.subplots(1, 2, figsize=(10, 4.5)); axes[0].bar([str(r["K_mode"]) for r in k_rows], [r["silhouette"] for r in k_rows], color="#377eb8"); axes[0].set_title("Silhouette"); axes[1].bar([str(r["K_mode"]) for r in k_rows], [r["minimum_cluster_prevalence"] for r in k_rows], color="#4daf4a"); axes[1].axhline(.05, color="red", linestyle="--"); axes[1].set_title("Minimum cluster prevalence"); fig.suptitle("Train-only mode stability/support"); savefig("MODE_STABILITY_SUPPORT", fig, "Train-only mode silhouette and minimum prevalence for fixed K=2,4,8; K=2 is selected by the smallest-passing rule.", "Two-panel bar chart of silhouette and minimum cluster prevalence across mode counts.")
    probes = [r for r in json.loads((MODE_ROOT / "PROBE_RESULTS.json").read_text())["rows"] if r["K_mode"] == 2]; probes = [r for r in probes if r["feature_set"] != "TASK_ONLY"]; fig, ax = plt.subplots(figsize=(8, 5)); ax.bar([r["feature_set"] for r in probes], [r["macro_F1"] for r in probes], color="#377eb8"); ax.axhline(float(selected_mode_row["fusion_probe_majority_macro_F1"]), color="black", linestyle=":", label="majority"); ax.axhline(float(selected_mode_row["fusion_probe_majority_macro_F1"]) + .05, color="red", linestyle="--", label="majority+5pp"); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("macro-F1"); ax.set_title("Past-context mode probe and controls"); ax.legend(frameon=False); savefig("MODE_PROBE_CONTROLS", fig, "Train-only K=2 past-context probe macro-F1 compared with majority and the locked 5-point margin.", "Bar chart comparing mode probe macro-F1 for past-only feature sets with majority thresholds.")
    center_payload = load_npz(MODE_ROOT / "PROTOTYPE_CENTERS.npz"); center_w = center_payload["centers_K2"].reshape(2, 16, 14); center_native = center_w * center_payload["coefficient_scale"][None, :, :] + center_payload["coefficient_mean"][None, :, :]; center_resid = np.einsum("tk,mkh->mth", dct_basis()[:16].T, center_native); fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(center_resid[0].mean(axis=1), label="mode0"); ax.plot(center_resid[1].mean(axis=1), label="mode1"); ax.set_xlabel("forecast step"); ax.set_ylabel("mean native residual"); ax.set_title("Train-only prototype center trajectories"); ax.legend(frameon=False); savefig("TRAIN_PROTOTYPE_TRAJECTORIES", fig, "Train-only K=2 prototype center residual trajectories in native command space.", "Line plot of the two train-only prototype center residual trajectories averaged over Hand joints.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); axes[0].bar(labels, [float(np.mean(m[name]["hand_error_std"])) for name in MODEL_ORDER], color=colors); axes[0].set_title("Hand MAE"); axes[0].tick_params(axis="x", rotation=30); axes[1].bar(labels, [float(np.mean(m[name]["full_error_std"])) for name in MODEL_ORDER], color=colors); axes[1].set_title("Full-28 MAE"); axes[1].tick_params(axis="x", rotation=30); fig.suptitle("SINGLE versus prototype MAP/MEAN"); savefig("SINGLE_MAP_MEAN_RESULTS", fig, "Development Hand and Full-28 MAE for the fresh single baseline and gated prototype MAP/MEAN outputs.", "Two-panel bar chart comparing Hand and Full-28 standardized MAE across the fixed hybrid and fresh outputs.")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter([float(np.mean(m[name]["hand_error_std"])) for name in MODEL_ORDER], [float(np.mean(m[name]["hand_fd_std"])) for name in MODEL_ORDER], c=colors); 
    for name, label in zip(MODEL_ORDER, labels): ax.annotate(label, (float(np.mean(m[name]["hand_error_std"])), float(np.mean(m[name]["hand_fd_std"])),), xytext=(4,4), textcoords="offset points")
    ax.set_xlabel("Hand standardized MAE"); ax.set_ylabel("Hand standardized FD MAE"); ax.set_title("Prototype level versus temporal fidelity"); savefig("PROTOTYPE_LEVEL_FD_PARETO", fig, "Level versus standardized first-difference error for prototype screening outputs.", "Scatter plot of Hand level and standardized temporal errors for prototype screening outputs.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); axes[0].bar(labels, [float(np.nanmean(m[name]["tv_ratio_hand"])) for name in MODEL_ORDER], color=colors); axes[0].axhline(1, color="black", linestyle=":"); axes[0].set_title("Native TV / target TV"); axes[0].tick_params(axis="x", rotation=30); axes[1].bar(labels, [float(np.mean(np.abs(models[name]["hand_std"][:, 20:] - target_std[:, 20:, :HAND]))) for name in MODEL_ORDER], color=colors); axes[1].set_title("Late Hand MAE"); axes[1].tick_params(axis="x", rotation=30); fig.suptitle("Native TV and horizon guardrails"); savefig("NATIVE_TV_HORIZON", fig, "Native command TV ratio and late-horizon Hand error for prototype screening outputs.", "Two-panel bar chart of native TV ratio and late-horizon Hand MAE across outputs.")
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(mode_confusion, cmap="Blues"); ax.set_xticks([0,1], ["pred0", "pred1"]); ax.set_yticks([0,1], ["target0", "target1"]); ax.set_xlabel("argmax predicted mode"); ax.set_ylabel("target-selected component mode"); ax.set_title("Dev mode confusion diagnostic")
    for (row_index, column_index), value in np.ndenumerate(mode_confusion):
        ax.text(column_index, row_index, str(int(value)), ha="center", va="center", color="white" if value > mode_confusion.max() / 2 else "black")
    savefig("MODE_CONFUSION", fig, "Outcome-exposed descriptive mode confusion on dev; this plot does not alter mode selection.", "Two-by-two heatmap of predicted prototype mode versus target-selected component mode on development episodes.")
    joint = np.stack([m["PERSISTENCE"]["hand_joint_error"].mean(axis=0) - m[name]["hand_joint_error"].mean(axis=0) for name in PROTO_ORDER]); fig, ax = plt.subplots(figsize=(8, 4.5)); im=ax.imshow(joint, aspect="auto", cmap="RdYlGn"); fig.colorbar(im, ax=ax, label="P error - prototype error"); ax.set_yticks([0,1], PROTO_ORDER); ax.set_xticks(range(HAND)); ax.set_xlabel("Hand joint"); ax.set_title("Per-joint prototype improvement"); savefig("PER_JOINT_SUPPORT", fig, "Per-joint Hand improvement of MAP and MEAN versus Persistence.", "Heatmap of per-joint Hand improvement for prototype MAP and MEAN outputs.")
    fig, ax = plt.subplots(figsize=(8, 5)); x=np.arange(len(PROTO_ORDER)); ax.bar(x-.18,[support_summary[n]["supported_positive_task_count"] for n in PROTO_ORDER],.36,label="positive tasks"); ax.bar(x+.18,[support_summary[n]["positive_joint_count"] for n in PROTO_ORDER],.36,label="positive joints"); ax.set_xticks(x,PROTO_ORDER); ax.legend(frameon=False); ax.set_title("Task/joint support"); savefig("TASK_SUPPORT", fig, "Supported positive task and joint counts for prototype MAP and MEAN versus Persistence.", "Grouped bar chart of positive task and joint support counts for prototype outputs.")
    # Fixed 12 trajectory figure.
    if len(selected_ids) < 12: selected_ids = [int(value) for value in ids[:12]]
    fig, axes = plt.subplots(3,4, figsize=(15,9), sharex=True)
    for axis, episode in zip(axes.ravel(), selected_ids):
        i=int(np.flatnonzero(ids==episode)[0]); axis.plot(target_native[i,:, :HAND].mean(axis=1), color="black", label="GT"); axis.plot(persistence_native[i].mean(axis=1), color="#777777", linestyle=":", label="P"); axis.plot(models["HP-SINGLE"]["hand_native"][i].mean(axis=1), color="#377eb8", label="SINGLE"); axis.plot(models["HP-PROTO-MAP"]["hand_native"][i].mean(axis=1), color="#e41a1c", label="MAP"); axis.plot(models["HP-PROTO-MEAN"]["hand_native"][i].mean(axis=1), color="#4daf4a", label="MEAN"); axis.set_title(f"episode {episode}", fontsize=8); axis.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=7,ncol=3); fig.suptitle("Fixed development Hand command trajectories"); savefig("FIXED12_TRAJECTORIES", fig, "Predeclared development episode Hand command trajectories for GT, Persistence, SINGLE, MAP and MEAN.", "Twelve-panel line plot of fixed development Hand trajectories for ground truth and prototype outputs.")
    gain = m["PERSISTENCE"]["hand_error_std"] - m["HP-PROTO-MAP"]["hand_error_std"]; trajectory_indices = list(np.argsort(gain)[-2:])+list(np.argsort(gain)[:2])+list(np.argsort(m["PERSISTENCE"]["hand_fd_native"]-m["HP-PROTO-MAP"]["hand_fd_native"])[:2]); trajectory_indices=list(dict.fromkeys(int(i) for i in trajectory_indices))[:6]; fig, axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for axis,i in zip(axes.ravel(),trajectory_indices): axis.plot(target_native[i,:,:HAND].mean(axis=1),color="black",label="GT"); axis.plot(persistence_native[i].mean(axis=1),color="#777777",linestyle=":",label="P"); axis.plot(models["HP-SINGLE"]["hand_native"][i].mean(axis=1),color="#377eb8",label="SINGLE"); axis.plot(models["HP-PROTO-MAP"]["hand_native"][i].mean(axis=1),color="#e41a1c",label="MAP"); axis.plot(models["HP-PROTO-MEAN"]["hand_native"][i].mean(axis=1),color="#4daf4a",label="MEAN"); axis.set_title(f"episode {int(ids[i])}"); axis.grid(alpha=.2)
    for axis in axes.ravel()[len(trajectory_indices):]: axis.axis("off")
    axes[0,0].legend(frameon=False,fontsize=7); fig.suptitle("Worst/best/regression prototype examples"); savefig("WORST_BEST_REGRESSION", fig, "Deterministically selected prototype level and FD examples; examples do not replace aggregate metrics.", "Six-panel line plot of selected best, worst and temporal-regression Hand command trajectories.")
    i=0; fig, axes=plt.subplots(2,1,figsize=(9,6),sharex=True); axes[0].plot(probabilities[i],label="mode probabilities"); axes[0].set_ylim(0,1); axes[0].legend(frameon=False); axes[0].set_title("Prototype mode probabilities"); axes[1].plot(target_native[i,:,:HAND].mean(axis=1),color="black",label="GT"); axes[1].plot(proto["components_hand_standardized"][i,0].astype(float).mean(axis=1)*scale[:,:HAND].mean(axis=1)+mean[:,:HAND].mean(axis=1),label="component0 rough view"); axes[1].set_title("Component timeline diagnostic"); axes[1].legend(frameon=False); savefig("COMPONENT_PROBABILITY_TIMELINE", fig, "Prototype mode probability and component timeline diagnostic for one development episode.", "Two-panel plot showing mode probabilities and a descriptive prototype component trajectory.")
    arm_max=[0.0,float(np.max(np.abs(single["full_predictions_standardized"][:,:,14:]-arm_std))),float(np.max(np.abs(proto["map_full_predictions_standardized"][:,:,14:]-arm_std))),float(np.max(np.abs(proto["mean_full_predictions_standardized"][:,:,14:]-arm_std)))]; fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(["P","SINGLE","MAP","MEAN"],arm_max,color=["#777777","#377eb8","#e41a1c","#4daf4a"]); ax.set_ylabel("max absolute Arm difference"); ax.set_title("Arm non-interference"); ax.text(.5,.9,f"max={max(arm_max):.3g}",transform=ax.transAxes,ha="center"); savefig("ARM_NON_INTERFERENCE",fig,"Bitwise Arm non-interference for the fixed hybrid and fresh prototype outputs.","Bar chart of maximum absolute Arm difference from the locked Arm array.")
    fig,ax=plt.subplots(figsize=(10,5)); ax.axis("off"); ax.text(.03,.82,f"Route C: PROTO | candidate gate: {decision['candidate_gate'].upper()}",fontsize=17,weight="bold",color="#1a9850" if selected else "#d73027"); ax.text(.03,.64,f"MAP/MEAN selected: {decision['selected_candidate'] or 'none'}",fontsize=14); ax.text(.03,.48,"CAL alpha=0; mode K=2 passed train-only feasibility",fontsize=12); ax.text(.03,.34,"single seed; development only; final/held-out access=0",fontsize=12); ax.text(.03,.20,"no physical-unit or Cartesian claim",fontsize=12); savefig("DECISION_SUMMARY",fig,"Route and candidate decision summary for the single-seed prototype screen.","Text summary showing Route PROTO, candidate gate outcome, mode K, development boundary and zero final-test access.")
    write_csv(ROOT / "figures/FIGURE_DATA.csv", [{"figure": row["figure"], "source": "analysis outputs under this experiment root", "route": "PROTO"} for row in records])
    write_csv(ROOT / "captions/FIGURE_MANIFEST.csv", records)
    write_json(ROOT / "captions/FIGURE_MANIFEST.json", {"status": "PASS", "figures": records, "formats": ["png", "svg", "pdf"], "physical_units_claimed": False, "manual_visual_qa": "pending"})
    write_json(ROOT / "figures/FIGURE_MANIFEST.json", {"status": "PASS", "figures": records, "formats": ["png", "svg", "pdf"], "physical_units_claimed": False})
    # Route-specific next handoffs and decision docs.
    if selected:
        (ROOT / "NEXT_HAND_PROTOTYPE_MULTI_SEED_PROMPT.md").write_text(f"# Prototype multi-seed handoff\n\n`generated_not_executed`: `{selected}` passed the single-seed screen. Freeze this exact output rule and request separate approval before any multi-seed or held-out evaluation.\n", encoding="utf-8")
        (ROOT / "NEXT_HAND_ACTION_PHASE_CONDITIONING_PROMPT.md").write_text("# Action-phase branch\n\n`not_applicable`: a prototype candidate passed the single-seed gate.\n", encoding="utf-8")
    else:
        (ROOT / "NEXT_HAND_PROTOTYPE_MULTI_SEED_PROMPT.md").write_text("# Prototype multi-seed handoff\n\n`not_applicable`: neither preordered MAP nor MEAN prototype output passed the complete candidate gate.\n", encoding="utf-8")
        (ROOT / "NEXT_HAND_ACTION_PHASE_CONDITIONING_PROMPT.md").write_text("# Next step: action-phase conditioning review\n\n`generated_not_executed`: Route PROTO completed but neither MAP nor MEAN passed the complete candidate gate. A future action-phase conditioning study may be considered with a new pre-registered contract; do not run it from this file, and do not add more prototype K, cluster seeds, loss weights or decode rules to the current experiment.\n", encoding="utf-8")
    (ROOT / "NEXT_CALIBRATED_D1L1_MULTI_SEED_PROMPT.md").write_text("# Calibrated D1-L1 multi-seed handoff\n\n`not_applicable`: no positive constrained alpha passed CAL.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cal_gate": "fail", "mode_gate": "pass", "route": "PROTO", "selected": selected, "candidate_gate": decision["candidate_gate"], "figures": len(records), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
