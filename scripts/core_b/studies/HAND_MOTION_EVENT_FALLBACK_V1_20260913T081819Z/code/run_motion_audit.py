#!/usr/bin/env python3
"""Run the locked CPU-only Hand/Arm motion audit and derived hybrids."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
PRIOR_PREDICTION_PATH = ROOT.parent / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
PRIOR_RUN_METRICS_PATH = ROOT.parent / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/RUN_METRICS.json"
FREEZE_CHECKSUM_PATH = ROOT.parent / "JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256"
STAGE2_CHECKSUM_PATH = ROOT.parent / "JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z/qa/CHECKSUMS.sha256"

BOOTSTRAP = 10_000
AUDIT_SEED = 20260926
PLANNED_TRAINING_SEED = 20260925
TV_DENOMINATOR_FLOOR = 1e-6
STEPS = 30
DIMS = 28
HAND = slice(0, 14)
ARM = slice(14, 28)


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_or_raise(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite values in {name}")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise RuntimeError("bootstrap requires at least two finite episodes")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "effect": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": BOOTSTRAP,
        "N_episode": int(len(values)),
    }


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if len(numerator) != len(denominator) or len(numerator) < 2:
        raise RuntimeError("ratio bootstrap pairing failure")
    if np.any(denominator < 0) or not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise RuntimeError("invalid ratio bootstrap inputs")
    indices = rng.integers(0, len(numerator), size=(BOOTSTRAP, len(numerator)))
    denom_sum = denominator[indices].sum(axis=1)
    estimates = np.divide(numerator[indices].sum(axis=1), denom_sum, out=np.full(BOOTSTRAP, np.nan), where=denom_sum > 1e-12)
    estimates = estimates[np.isfinite(estimates)]
    point = float(numerator.sum() / denominator.sum()) if denominator.sum() > 1e-12 else float("nan")
    return {
        "effect": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": BOOTSTRAP,
        "N_episode": int(len(numerator)),
    }


def paired_effect(candidate: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    """Return baseline minus candidate; positive means candidate improves."""
    return bootstrap_mean(np.asarray(baseline) - np.asarray(candidate), rng)


def group_step_motion(target: np.ndarray, persistence: np.ndarray, group: slice) -> np.ndarray:
    first = np.abs(target[:, 0, group] - persistence[:, 0, group]).mean(axis=1)
    transitions = np.abs(np.diff(target[:, :, group], axis=1)).mean(axis=2)
    return np.concatenate((first[:, None], transitions), axis=1)


def group_tv_increments(target_native: np.ndarray, group: slice) -> np.ndarray:
    return np.abs(np.diff(target_native[:, :, group], axis=1)).mean(axis=2)


def episode_error(pred: np.ndarray, target: np.ndarray, group: slice) -> np.ndarray:
    return np.abs(pred[:, :, group] - target[:, :, group]).mean(axis=(1, 2))


def episode_horizon_error(pred: np.ndarray, target: np.ndarray, group: slice, horizon: slice) -> np.ndarray:
    return np.abs(pred[:, horizon, group] - target[:, horizon, group]).mean(axis=(1, 2))


def trajectory_metrics(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray) -> dict[str, float]:
    error = pred - target
    native_error = pred_native - target_native
    result: dict[str, float] = {
        "arm_mae_std": float(np.abs(error[:, :, ARM]).mean()),
        "hand_mae_std": float(np.abs(error[:, :, HAND]).mean()),
        "full28_mae_std": float(np.abs(error).mean()),
        "arm_mae_native": float(np.abs(native_error[:, :, ARM]).mean()),
        "hand_mae_native": float(np.abs(native_error[:, :, HAND]).mean()),
        "full28_mae_native": float(np.abs(native_error).mean()),
        "arm_early_mae_std": float(np.abs(error[:, :10, ARM]).mean()),
        "arm_middle_mae_std": float(np.abs(error[:, 10:20, ARM]).mean()),
        "arm_late_mae_std": float(np.abs(error[:, 20:, ARM]).mean()),
        "arm_endpoint_mae_std": float(np.abs(error[:, -1, ARM]).mean()),
        "hand_early_mae_std": float(np.abs(error[:, :10, HAND]).mean()),
        "hand_middle_mae_std": float(np.abs(error[:, 10:20, HAND]).mean()),
        "hand_late_mae_std": float(np.abs(error[:, 20:, HAND]).mean()),
        "hand_endpoint_mae_std": float(np.abs(error[:, -1, HAND]).mean()),
        "full28_early_mae_std": float(np.abs(error[:, :10]).mean()),
        "full28_middle_mae_std": float(np.abs(error[:, 10:20]).mean()),
        "full28_late_mae_std": float(np.abs(error[:, 20:]).mean()),
        "full28_endpoint_mae_std": float(np.abs(error[:, -1]).mean()),
        "arm_first_difference_mae_std": float(np.abs(np.diff(pred[:, :, ARM], axis=1) - np.diff(target[:, :, ARM], axis=1)).mean()),
        "hand_first_difference_mae_std": float(np.abs(np.diff(pred[:, :, HAND], axis=1) - np.diff(target[:, :, HAND], axis=1)).mean()),
        "full28_first_difference_mae_std": float(np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)).mean()),
    }
    for name, values_pred, values_target in (
        ("arm", pred_native[:, :, ARM], target_native[:, :, ARM]),
        ("hand", pred_native[:, :, HAND], target_native[:, :, HAND]),
        ("full28", pred_native, target_native),
    ):
        pred_tv = np.abs(np.diff(values_pred, axis=1)).sum(axis=1).mean(axis=1)
        target_tv = np.abs(np.diff(values_target, axis=1)).sum(axis=1).mean(axis=1)
        ratios = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_DENOMINATOR_FLOOR)
        result[f"{name}_native_tv_ratio"] = float(np.nanmean(ratios))
        result[f"{name}_native_tv_closeness"] = float(np.nanmean(np.abs(ratios - 1.0)))
        result[f"{name}_native_tv_valid_fraction"] = float(np.mean(target_tv > TV_DENOMINATOR_FLOOR))
    # Persistence is passed for a stable schema and for a direct sanity check.
    result["persistence_full28_mae_std"] = float(np.abs(persistence - target).mean())
    result["all_finite"] = float(np.isfinite(pred).all() and np.isfinite(pred_native).all())
    return result


def load_inputs() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if not INPUT_PATH.is_file() or not NORMALIZER_PATH.is_file() or not PRIOR_PREDICTION_PATH.is_file():
        raise FileNotFoundError("locked development input, normalizer or prior candidate prediction is missing")
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    split = data["split_code"]
    train_mask = split == 0
    dev_mask = split == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("JY-02 train/dev row contract failed")
    if data["targets_standardized"].shape != (3355, STEPS, DIMS):
        raise RuntimeError("JY-02 target shape contract failed")
    for key in ("targets_standardized", "persistence_standardized", "targets_native", "persistence_native"):
        finite_or_raise(key, data[key])
    normalizer = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))["1.0s"]
    with np.load(PRIOR_PREDICTION_PATH, allow_pickle=False) as archive:
        prior = {key: archive[key].copy() for key in archive.files}
    if prior["predictions_standardized"].shape != (614, STEPS, DIMS):
        raise RuntimeError("prior E-F-SP-G prediction shape contract failed")
    dev_ids = data["episode_index"][dev_mask]
    if not np.array_equal(prior["episode_index"], dev_ids):
        raise RuntimeError("prior candidate episode pairing failed")
    if not np.array_equal(prior["targets_standardized"], data["targets_standardized"][dev_mask]):
        raise RuntimeError("prior candidate target pairing failed")
    if not np.array_equal(prior["persistence_standardized"], data["persistence_standardized"][dev_mask]):
        raise RuntimeError("prior candidate persistence pairing failed")
    finite_or_raise("prior predictions", prior["predictions_standardized"])
    return data, train_mask, dev_mask, {"normalizer": normalizer, "prior": prior}


def threshold_registry(data: dict[str, np.ndarray], train_mask: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    target = data["targets_standardized"].astype(np.float64)
    persistence = data["persistence_standardized"].astype(np.float64)
    arm = group_step_motion(target[train_mask], persistence[train_mask], ARM)
    hand = group_step_motion(target[train_mask], persistence[train_mask], HAND)
    pooled = np.concatenate((arm.reshape(-1), hand.reshape(-1)))
    thresholds = {
        "COMMON_STEP_Q25": float(np.quantile(pooled, 0.25)),
        "COMMON_STEP_Q75": float(np.quantile(pooled, 0.75)),
        "HAND_STEP_Q75": float(np.quantile(hand.reshape(-1), 0.75)),
        "HAND_EPISODE_MEAN_Q75": float(np.quantile(hand.mean(axis=1), 0.75)),
    }
    train_motion = {"arm": arm, "hand": hand, "pooled": pooled}
    write_json(ROOT / "MOTION_THRESHOLDS_LOCKED.json", {
        "status": "PASS",
        "source_split": "train",
        "source_rows": int(train_mask.sum()),
        "definitions": {
            "event_compatible_step_motion": "t0=abs(target[0]-persistence[0]); t1..29=abs(target[t]-target[t-1])",
            "common_step_q25": "q25 of pooled Arm and Hand step values",
            "common_step_q75": "q75 of pooled Arm and Hand step values",
            "hand_step_q75": "q75 of Hand step values",
            "hand_episode_mean_q75": "q75 of Hand episode mean-step-motion",
        },
        "thresholds": thresholds,
        "input_sha256": digest(INPUT_PATH),
        "normalizer_sha256": digest(NORMALIZER_PATH),
        "locked_utc": now(),
    })
    return thresholds, train_motion


def hybrid_rows(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray, prior_name: str, rng: np.random.Generator) -> tuple[dict[str, float], list[dict[str, object]]]:
    point = trajectory_metrics(pred, target, persistence, pred_native, target_native, persistence_native)
    rows: list[dict[str, object]] = []
    candidates = {
        "PERSISTENCE": persistence,
        "E-F-SP-G": pred,
        "HYB-A_NN-H_P": None,
        "HYB-A_P-H_NN": None,
    }
    hyb_arm = persistence.copy()
    hyb_arm[:, :, ARM] = pred[:, :, ARM]
    hyb_hand = persistence.copy()
    hyb_hand[:, :, HAND] = pred[:, :, HAND]
    candidates["HYB-A_NN-H_P"] = hyb_arm
    candidates["HYB-A_P-H_NN"] = hyb_hand
    native_candidates = {
        "PERSISTENCE": persistence_native,
        "E-F-SP-G": pred_native,
        "HYB-A_NN-H_P": np.concatenate((persistence_native[:, :, HAND], pred_native[:, :, ARM]), axis=2),
        "HYB-A_P-H_NN": np.concatenate((pred_native[:, :, HAND], persistence_native[:, :, ARM]), axis=2),
    }
    for name, candidate in candidates.items():
        assert candidate is not None
        metrics = trajectory_metrics(candidate, target, persistence, native_candidates[name], target_native, persistence_native)
        arm_per = episode_error(candidate, target, ARM)
        hand_per = episode_error(candidate, target, HAND)
        full_per = episode_error(candidate, target, slice(0, 28))
        p_arm = episode_error(persistence, target, ARM)
        p_hand = episode_error(persistence, target, HAND)
        p_full = episode_error(persistence, target, slice(0, 28))
        metrics.update({
            "arm_skill_vs_persistence": float((p_arm.mean() - arm_per.mean()) / p_arm.mean()),
            "hand_skill_vs_persistence": float((p_hand.mean() - hand_per.mean()) / p_hand.mean()),
            "full28_skill_vs_persistence": float((p_full.mean() - full_per.mean()) / p_full.mean()),
            "is_derived_control": float(name.startswith("HYB-")),
        })
        rows.append({"model": name, **metrics})
    # Paired effects are kept separately so the sign convention is explicit.
    effects: list[dict[str, object]] = []
    metric_groups = {
        "Arm-14 MAE": (ARM, "arm"),
        "Hand-14 MAE": (HAND, "hand"),
        "Full-28 MAE": (slice(0, 28), "full28"),
    }
    per_model = {name: candidates[name] for name in candidates}
    for candidate_name in ("HYB-A_NN-H_P", "HYB-A_P-H_NN"):
        for baseline_name in ("PERSISTENCE", "E-F-SP-G"):
            for metric_name, (group, prefix) in metric_groups.items():
                candidate_values = episode_error(per_model[candidate_name], target, group)
                baseline_values = episode_error(per_model[baseline_name], target, group)
                effect = paired_effect(candidate_values, baseline_values, rng)
                effects.append({"candidate": candidate_name, "baseline": baseline_name, "metric": metric_name, **effect, "effect_definition": "baseline MAE minus candidate MAE; positive is improvement", "outcome_exposed_derived_control": True})
    return {row["model"]: row for row in rows}, effects


def main() -> int:
    rng = np.random.default_rng(AUDIT_SEED)
    data, train_mask, dev_mask, contracts = load_inputs()
    thresholds, train_motion = threshold_registry(data, train_mask)
    target = data["targets_standardized"][dev_mask].astype(np.float64)
    persistence = data["persistence_standardized"][dev_mask].astype(np.float64)
    target_native = data["targets_native"][dev_mask].astype(np.float64)
    persistence_native = data["persistence_native"][dev_mask].astype(np.float64)
    prior = contracts["prior"]
    candidate = prior["predictions_standardized"].astype(np.float64)
    candidate_native = prior["predictions_native"].astype(np.float64)
    finite_or_raise("development target", target)
    finite_or_raise("development persistence", persistence)
    finite_or_raise("development native target", target_native)
    finite_or_raise("development native persistence", persistence_native)
    finite_or_raise("E-F-SP-G native predictions", candidate_native)

    arm_motion = group_step_motion(target, persistence, ARM)
    hand_motion = group_step_motion(target, persistence, HAND)
    arm_motion_native = group_step_motion(target_native, persistence_native, ARM)
    hand_motion_native = group_step_motion(target_native, persistence_native, HAND)
    common_low = thresholds["COMMON_STEP_Q25"]
    common_high = thresholds["COMMON_STEP_Q75"]
    hand_event_threshold = thresholds["HAND_STEP_Q75"]
    hand_episode_event_threshold = thresholds["HAND_EPISODE_MEAN_Q75"]
    hand_event = hand_motion >= hand_event_threshold
    hand_episode_event = hand_motion.mean(axis=1) >= hand_episode_event_threshold
    arm_low_fraction = (arm_motion < common_low).mean(axis=1)
    hand_low_fraction = (hand_motion < common_low).mean(axis=1)
    arm_high_fraction = (arm_motion >= common_high).mean(axis=1)
    hand_high_fraction = (hand_motion >= common_high).mean(axis=1)
    arm_tv_inc = group_tv_increments(target_native, ARM)
    hand_tv_inc = group_tv_increments(target_native, HAND)
    arm_tv = arm_tv_inc.sum(axis=1)
    hand_tv = hand_tv_inc.sum(axis=1)
    hand_step_event_tv = (hand_tv_inc * hand_event[:, 1:]).sum(axis=1)
    hand_episode_event_tv = hand_tv * hand_episode_event
    arm_tv_nonzero = arm_tv > 1e-12
    hand_tv_nonzero = hand_tv > 1e-12

    # H1/H2 and the diagnostic candidate deficit are all episode-paired.
    motion_difference = bootstrap_mean(hand_motion.mean(axis=1) - arm_motion.mean(axis=1), rng)
    low_difference = bootstrap_mean(hand_low_fraction - arm_low_fraction, rng)
    native_motion_difference = bootstrap_mean(hand_motion_native.mean(axis=1) - arm_motion_native.mean(axis=1), rng)
    hand_step_tv_share = bootstrap_ratio(hand_step_event_tv, hand_tv, rng)
    hand_episode_tv_share = bootstrap_ratio(hand_episode_event_tv, hand_tv, rng)
    hand_candidate_step_error = np.abs(candidate[:, :, HAND] - target[:, :, HAND]).mean(axis=2)
    hand_persistence_step_error = np.abs(persistence[:, :, HAND] - target[:, :, HAND]).mean(axis=2)
    step_gap = hand_candidate_step_error - hand_persistence_step_error
    non_event_values = np.divide((step_gap * (~hand_event)).sum(axis=1), (~hand_event).sum(axis=1), out=np.full(len(step_gap), np.nan), where=(~hand_event).sum(axis=1) > 0)
    event_values = np.divide((step_gap * hand_event).sum(axis=1), hand_event.sum(axis=1), out=np.full(len(step_gap), np.nan), where=hand_event.sum(axis=1) > 0)
    c_values = non_event_values - event_values
    c_gap = bootstrap_mean(c_values, rng)
    c_non_event = bootstrap_mean(non_event_values, rng)
    c_event = bootstrap_mean(event_values, rng)
    c_pass = bool(c_non_event["effect"] > 0 and c_gap["ci_low"] > 0)
    a_pass = bool(motion_difference["ci_high"] < 0 and low_difference["ci_low"] > 0)
    b_pass = bool(hand_step_tv_share["effect"] >= 0.50 and hand_episode_tv_share["effect"] >= 0.50 and hand_step_tv_share["ci_low"] > 0.50 and hand_episode_tv_share["ci_low"] > 0.50)
    evidence_gate = bool(a_pass or b_pass or c_pass)

    episode_ids = data["episode_index"][dev_mask].astype(np.int64)
    task_ids = data["original_task_index"][dev_mask].astype(np.int64)
    prior_hand_error = np.abs(candidate[:, :, HAND] - target[:, :, HAND]).mean(axis=(1, 2))
    prior_arm_error = np.abs(candidate[:, :, ARM] - target[:, :, ARM]).mean(axis=(1, 2))
    persistence_hand_error = np.abs(persistence[:, :, HAND] - target[:, :, HAND]).mean(axis=(1, 2))
    persistence_arm_error = np.abs(persistence[:, :, ARM] - target[:, :, ARM]).mean(axis=(1, 2))
    episode_rows: list[dict[str, object]] = []
    for i in range(len(episode_ids)):
        episode_rows.append({
            "episode_index": int(episode_ids[i]),
            "original_task_index": int(task_ids[i]),
            "arm_mean_step_motion_std": float(arm_motion[i].mean()),
            "hand_mean_step_motion_std": float(hand_motion[i].mean()),
            "arm_mean_step_motion_native": float(arm_motion_native[i].mean()),
            "hand_mean_step_motion_native": float(hand_motion_native[i].mean()),
            "hand_minus_arm_motion_std": float(hand_motion[i].mean() - arm_motion[i].mean()),
            "arm_low_motion_step_fraction": float(arm_low_fraction[i]),
            "hand_low_motion_step_fraction": float(hand_low_fraction[i]),
            "arm_high_motion_step_fraction": float(arm_high_fraction[i]),
            "hand_high_motion_step_fraction": float(hand_high_fraction[i]),
            "hand_event_step_fraction": float(hand_event[i].mean()),
            "hand_episode_event": bool(hand_episode_event[i]),
            "arm_tv_native": float(arm_tv[i]),
            "hand_tv_native": float(hand_tv[i]),
            "hand_step_event_tv_native": float(hand_step_event_tv[i]),
            "hand_candidate_mae_std": float(prior_hand_error[i]),
            "hand_persistence_mae_std": float(persistence_hand_error[i]),
            "hand_candidate_minus_persistence_gap_std": float(prior_hand_error[i] - persistence_hand_error[i]),
            "hand_non_event_gap_std": float(non_event_values[i]),
            "hand_event_gap_std": float(event_values[i]),
            "arm_candidate_mae_std": float(prior_arm_error[i]),
            "arm_persistence_mae_std": float(persistence_arm_error[i]),
            "outcome_exposed_derived_control": True,
        })
    write_csv(ROOT / "MOTION_AUDIT_PER_EPISODE.csv", episode_rows)

    hybrid_summary, hybrid_effects = hybrid_rows(candidate, target, persistence, candidate_native, target_native, persistence_native, "E-F-SP-G", rng)
    write_csv(ROOT / "HYBRID_RESULTS.csv", list(hybrid_summary.values()))
    write_csv(ROOT / "HYBRID_EFFECTS.csv", hybrid_effects)
    write_csv(ROOT / "DERIVED_GROUPWISE_HYBRID_RESULTS.csv", list(hybrid_summary.values()))
    write_json(ROOT / "DERIVED_GROUPWISE_HYBRID_RESULTS.json", {
        "status": "PASS",
        "outcome_exposure": "outcome-exposed derived development control",
        "source_candidate": str(PRIOR_PREDICTION_PATH),
        "source_candidate_sha256": digest(PRIOR_PREDICTION_PATH),
        "rows": list(hybrid_summary.values()),
        "effects": hybrid_effects,
        "bootstrap_replicates": BOOTSTRAP,
        "episode_unit": True,
    })

    summary = {
        "status": "PASS",
        "study_id": "JY02_HAND_MOTION_EVENT_FALLBACK_V1",
        "phase": "A_A2",
        "completed_utc": now(),
        "audit_seed": AUDIT_SEED,
        "planned_training_seed": PLANNED_TRAINING_SEED,
        "development_count": int(len(episode_ids)),
        "train_count": int(train_mask.sum()),
        "episode_unit": True,
        "outcome_exposed": True,
        "thresholds": thresholds,
        "motion": {
            "arm_mean_step_motion_std": bootstrap_mean(arm_motion.mean(axis=1), rng),
            "hand_mean_step_motion_std": bootstrap_mean(hand_motion.mean(axis=1), rng),
            "hand_minus_arm_std": motion_difference,
            "hand_minus_arm_native": native_motion_difference,
            "arm_low_motion_step_fraction": bootstrap_mean(arm_low_fraction, rng),
            "hand_low_motion_step_fraction": bootstrap_mean(hand_low_fraction, rng),
            "hand_minus_arm_low_motion_step_fraction": low_difference,
            "arm_high_motion_step_fraction": bootstrap_mean(arm_high_fraction, rng),
            "hand_high_motion_step_fraction": bootstrap_mean(hand_high_fraction, rng),
        },
        "event_concentration": {
            "hand_step_event_prevalence": float(hand_event.mean()),
            "hand_episode_event_prevalence": float(hand_episode_event.mean()),
            "hand_step_event_tv_share": hand_step_tv_share,
            "hand_episode_event_tv_share": hand_episode_tv_share,
            "arm_tv_nonzero_fraction": float(arm_tv_nonzero.mean()),
            "hand_tv_nonzero_fraction": float(hand_tv_nonzero.mean()),
        },
        "fallback_deficit_diagnostic": {
            "candidate": "E-F-SP-G",
            "hand_non_event_candidate_minus_persistence_gap": c_non_event,
            "hand_event_candidate_minus_persistence_gap": c_event,
            "non_event_minus_event_gap": c_gap,
            "positive_non_event_gap": bool(c_non_event["effect"] > 0),
        },
        "hypotheses": {
            "H1_hand_motion_lower_than_arm": bool(motion_difference["ci_high"] < 0),
            "H2_hand_low_motion_prevalence_higher": bool(low_difference["ci_low"] > 0),
            "H3_hand_event_concentrated_tv": b_pass,
            "H4_persistence_deficit_concentrated_non_event": c_pass,
            "H5_no_support": not evidence_gate,
        },
        "evidence_gate": {
            "A_group_motion_separation": a_pass,
            "B_event_concentration": b_pass,
            "C_observed_fallback_deficit_concentration": c_pass,
            "pass": evidence_gate,
            "phase_c_allowed": evidence_gate,
        },
        "derived_hybrid": {
            "status": "complete",
            "hyb_a_nn_h_p_full28_mae_std": hybrid_summary["HYB-A_NN-H_P"]["full28_mae_std"],
            "hyb_a_nn_h_p_vs_persistence": hybrid_summary["HYB-A_NN-H_P"]["full28_skill_vs_persistence"],
            "hyb_a_p_h_nn_full28_mae_std": hybrid_summary["HYB-A_P-H_NN"]["full28_mae_std"],
            "hyb_a_p_h_nn_vs_persistence": hybrid_summary["HYB-A_P-H_NN"]["full28_skill_vs_persistence"],
        },
        "source": {
            "input": str(INPUT_PATH),
            "input_sha256": digest(INPUT_PATH),
            "normalizer": str(NORMALIZER_PATH),
            "normalizer_sha256": digest(NORMALIZER_PATH),
            "prior_prediction": str(PRIOR_PREDICTION_PATH),
            "prior_prediction_sha256": digest(PRIOR_PREDICTION_PATH),
            "prior_run_metrics": str(PRIOR_RUN_METRICS_PATH),
            "freeze_checksum_manifest": str(FREEZE_CHECKSUM_PATH),
            "stage2_checksum_manifest": str(STAGE2_CHECKSUM_PATH),
            "final_test_access": 0,
        },
    }
    write_json(ROOT / "MOTION_AUDIT_SUMMARY.json", summary)
    write_json(ROOT / "PHASE_A_RUN_COMPLETE.json", {
        "status": "complete",
        "phase": "A_A2",
        "evidence_gate": "pass" if evidence_gate else "fail",
        "phase_c_allowed": evidence_gate,
        "audit_rows": int(len(episode_ids)),
        "derived_hybrids": 2,
        "bootstrap_replicates": BOOTSTRAP,
        "final_test_access": 0,
        "completed_utc": now(),
    })

    decision_text = "PASS" if evidence_gate else "FAIL"
    (ROOT / "MOTION_AUDIT_DECISION.md").write_text(f"""# Motion audit decision

Phase A/A2 status: `complete`.

- Development episodes: `{len(episode_ids)}`; train threshold rows: `{int(train_mask.sum())}`.
- H1 Hand mean-step-motion below Arm: `{summary['hypotheses']['H1_hand_motion_lower_than_arm']}`.
- H2 Hand low-motion prevalence above Arm: `{summary['hypotheses']['H2_hand_low_motion_prevalence_higher']}`.
- H3 Hand event TV concentration: `{summary['hypotheses']['H3_hand_event_concentrated_tv']}`.
- H4 observed E-F-SP-G Hand deficit concentrated in non-event steps: `{summary['hypotheses']['H4_persistence_deficit_concentrated_non_event']}`.

Pre-registered evidence gate: `{decision_text}`. Phase C six-cell training is `{'allowed by Phase A gate' if evidence_gate else 'not allowed; stop after CPU-only audit'}`.

This audit and both hybrids are development-set, outcome-exposed evidence. The
zero-training `HYB-A_NN-H_P` remains the required Full-28 comparator for any
future learned event-gate claim. No frozen candidate, final protocol, or
final-test access changed (`0`).
""", encoding="utf-8")

    # Phase A figures are descriptive and explicitly marked development-only.
    figure_status: dict[str, object] = {"status": "PASS", "files": []}
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7.2, 4.6))
        plt.boxplot([arm_motion.mean(axis=1), hand_motion.mean(axis=1)], tick_labels=["Arm-14", "Hand-14"], showfliers=False)
        plt.ylabel("Mean step motion (standardized)")
        plt.title("JY-02 development motion audit (outcome-exposed)")
        plt.tight_layout()
        path = ROOT / "figures/MOTION_GROUP_DISTRIBUTIONS.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figure_status["files"].append(str(path))

        names = ["PERSISTENCE", "E-F-SP-G", "HYB-A_NN-H_P", "HYB-A_P-H_NN"]
        display_names = ["Persistence", "E-F-SP-G", "HYB-A_NN-H_P", "HYB-A_P-H_NN"]
        values = [hybrid_summary[name]["full28_mae_std"] for name in names]
        plt.figure(figsize=(8.0, 4.6))
        bars = plt.bar(display_names, values, color=["#8c8c8c", "#377eb8", "#4daf4a", "#e41a1c"])
        plt.ylabel("Full-28 MAE (standardized)")
        plt.title("Derived group-wise controls (development only)")
        plt.xticks(rotation=18, ha="right")
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        path = ROOT / "figures/DERIVED_HYBRID_FULL28_MAE.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figure_status["files"].append(str(path))

        plt.figure(figsize=(6.8, 4.6))
        labels = ["step-event TV share", "episode-event TV share"]
        values = [hand_step_tv_share["effect"], hand_episode_tv_share["effect"]]
        plt.bar(labels, values, color="#984ea3")
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="locked 0.50 gate")
        plt.ylim(0, 1)
        plt.ylabel("Share of Hand native TV")
        plt.title("Hand event concentration (development only)")
        plt.xticks(rotation=15, ha="right")
        plt.legend(frameon=False)
        plt.tight_layout()
        path = ROOT / "figures/HAND_EVENT_TV_CONCENTRATION.png"
        plt.savefig(path, dpi=160)
        plt.close()
        figure_status["files"].append(str(path))
    except Exception as exc:
        figure_status = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "files": []}
    write_json(ROOT / "FIGURE_GENERATION.json", figure_status)

    print(json.dumps({
        "status": "PASS",
        "phase": "A_A2",
        "evidence_gate": "pass" if evidence_gate else "fail",
        "phase_c_allowed": evidence_gate,
        "hand_minus_arm_motion_ci": [motion_difference["ci_low"], motion_difference["ci_high"]],
        "hand_minus_arm_low_fraction_ci": [low_difference["ci_low"], low_difference["ci_high"]],
        "hyb_a_nn_h_p_full28_mae_std": hybrid_summary["HYB-A_NN-H_P"]["full28_mae_std"],
        "hyb_a_nn_h_p_full28_skill": hybrid_summary["HYB-A_NN-H_P"]["full28_skill_vs_persistence"],
        "figure_status": figure_status["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
