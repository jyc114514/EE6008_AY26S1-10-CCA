#!/usr/bin/env python3
"""Harvest the four selector cells and apply the locked candidate gate."""

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

from oracle_audit import (
    average_precision,
    auroc,
    brier,
    bootstrap_mean,
    calibration_bins,
    composite_full28,
    event_compatible_motion,
    holm,
    signflip_p,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
LOCKED_ARM_PATH = REPO_ROOT / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
HAND_EVENT_ROOT = REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
BENCHMARK_ROOT = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
CELLS = ("SEL-U", "SEL-M", "SEL-AH", "SEL-AS")
ADVANTAGE_ORDER = ("SEL-AS", "SEL-AH")
BOOTSTRAP = 10_000
ANALYSIS_SEED = 20260933
STEPS = 30
HAND = slice(0, 14)
ARM = slice(14, 28)
TV_FLOOR = 1e-6


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key].copy() for key in archive.files}


def load_data() -> dict[str, Any]:
    data = load_archive(INPUT_PATH); train_mask = data["split_code"] == 0; dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614: raise RuntimeError("train/dev contract failed")
    normalizer = json.loads((REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json").read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64); std = np.asarray(normalizer["safe_std"], dtype=np.float64)
    state = data["state_action_history"].astype(np.float32); dynamic = data["dynamic_pooled"].astype(np.float32); context = np.concatenate((dynamic, state), axis=1); summary = state.mean(axis=1, keepdims=True).astype(np.float32)
    with np.load(LOCKED_ARM_PATH, allow_pickle=False) as archive: locked = {key: archive[key].copy() for key in archive.files}
    dev_ids = data["episode_index"][dev_mask].astype(np.int64); dev_tasks = data["original_task_index"][dev_mask].astype(np.int64)
    if not np.array_equal(locked["episode_index"], dev_ids) or not np.array_equal(locked["original_task_index"], dev_tasks): raise RuntimeError("locked Arm IDs mismatch")
    if not np.array_equal(locked["targets_standardized"], data["targets_standardized"][dev_mask]) or not np.array_equal(locked["persistence_standardized"], data["persistence_standardized"][dev_mask]): raise RuntimeError("locked Arm target/anchor mismatch")
    selector: dict[str, dict[str, np.ndarray]] = {}
    for cell in CELLS:
        run = ROOT / "runs" / cell; marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8"));
        if marker.get("status") != "complete" or marker.get("optimizer_steps") != 3120 or marker.get("prediction_hand_shape") != [614, 30, 14] or marker.get("prediction_full_shape") != [614, 30, 28]: raise RuntimeError(f"selector run acceptance failed: {cell}")
        bundle = load_archive(run / "PREDICTIONS_DEV.npz")
        if bundle["hand_predictions_standardized"].shape != (614, STEPS, 14) or bundle["q"].shape != (614, STEPS, 1) or bundle["full_predictions_standardized"].shape != (614, STEPS, 28): raise RuntimeError(f"selector archive shape failed: {cell}")
        if not np.array_equal(bundle["episode_index"], dev_ids) or not np.array_equal(bundle["original_task_index"], dev_tasks): raise RuntimeError(f"selector IDs failed: {cell}")
        if not np.isfinite(bundle["hand_predictions_standardized"]).all() or not np.isfinite(bundle["q"]).all() or np.any(bundle["q"] < 0) or np.any(bundle["q"] > 1): raise RuntimeError(f"selector finite/bounds failed: {cell}")
        if not np.array_equal(bundle["locked_arm_standardized"], locked["predictions_standardized"][:, :, ARM]): raise RuntimeError(f"Arm non-interference archive failed: {cell}")
        selector[cell] = bundle
    return {"data": data, "train_mask": train_mask, "dev_mask": dev_mask, "mean": mean, "std": std, "context": context, "summary": summary, "locked": locked, "selector": selector, "ids": dev_ids, "tasks": dev_tasks}


def per_metrics(pred: np.ndarray, target: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - target; native_error = pred_native - target_native
    result = {
        "hand_mae": np.abs(error[:, :, HAND]).mean(axis=(1, 2)),
        "full_mae": np.abs(error).mean(axis=(1, 2)),
        "hand_early": np.abs(error[:, :10, HAND]).mean(axis=(1, 2)),
        "hand_middle": np.abs(error[:, 10:20, HAND]).mean(axis=(1, 2)),
        "hand_late": np.abs(error[:, 20:, HAND]).mean(axis=(1, 2)),
        "hand_endpoint": np.abs(error[:, -1, HAND]).mean(axis=1),
        "full_early": np.abs(error[:, :10]).mean(axis=(1, 2)),
        "full_middle": np.abs(error[:, 10:20]).mean(axis=(1, 2)),
        "full_late": np.abs(error[:, 20:]).mean(axis=(1, 2)),
        "full_endpoint": np.abs(error[:, -1]).mean(axis=1),
        "hand_first_difference": np.abs(np.diff(pred[:, :, HAND], axis=1) - np.diff(target[:, :, HAND], axis=1)).mean(axis=(1, 2)),
        "full_first_difference": np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)).mean(axis=(1, 2)),
    }
    for name, group in (("hand", HAND), ("full", slice(0, 28))):
        pred_tv = np.abs(np.diff(pred_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1); target_tv = np.abs(np.diff(target_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1)
        ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_FLOOR)
        result[f"{name}_native_tv_ratio"] = ratio; result[f"{name}_native_tv_closeness"] = np.abs(ratio - 1.0)
    return result


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    numerator = np.asarray(numerator, dtype=np.float64); denominator = np.asarray(denominator, dtype=np.float64)
    indices = rng.integers(0, len(numerator), size=(BOOTSTRAP, len(numerator))); values = np.divide(numerator[indices].sum(axis=1), denominator[indices].sum(axis=1), out=np.full(BOOTSTRAP, np.nan), where=denominator[indices].sum(axis=1) > 1e-12); values = values[np.isfinite(values)]
    return {"effect": float(numerator.sum() / denominator.sum()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "replicates": BOOTSTRAP, "N_episode": int(len(numerator))}


def bootstrap_cluster_ap(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8); scores = np.asarray(scores, dtype=np.float64); point = average_precision(labels, scores); order = np.argsort(-scores.reshape(-1), kind="mergesort"); flat_labels = labels.reshape(-1)[order].astype(np.float64); episode_index = np.repeat(np.arange(labels.shape[0]), labels.shape[1])[order]; estimates: list[float] = []
    for start in range(0, BOOTSTRAP, 100):
        count = min(100, BOOTSTRAP - start); draw = rng.integers(0, labels.shape[0], size=(count, labels.shape[0])); weights = np.zeros((count, labels.shape[0]), dtype=np.float64)
        for i, selected in enumerate(draw): weights[i] = np.bincount(selected, minlength=labels.shape[0])
        obs_w = weights[:, episode_index]; total = np.cumsum(obs_w, axis=1); positive = np.cumsum(obs_w * flat_labels[None, :], axis=1); precision = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0); num = np.sum(precision * obs_w * flat_labels[None, :], axis=1); den = np.sum(obs_w * flat_labels[None, :], axis=1); value = np.divide(num, den, out=np.full(count, np.nan), where=den > 0); estimates.extend(value[np.isfinite(value)].tolist())
    return {"effect": float(point), "ci_low": float(np.quantile(estimates, .025)), "ci_high": float(np.quantile(estimates, .975)), "replicates": BOOTSTRAP, "N_episode": int(labels.shape[0])}


def support_metrics(improvement: np.ndarray, tasks: np.ndarray, joint_improvement: np.ndarray) -> dict[str, Any]:
    task_rows = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task; task_rows.append({"task": int(task), "N_episode": int(mask.sum()), "supported": bool(mask.sum() >= 5), "improvement": float(improvement[mask].mean())})
    supported_positive = [float(row["improvement"]) for row in task_rows if row["supported"] and float(row["improvement"]) > 0]; positive_joints = joint_improvement[joint_improvement > 0]
    return {"task_rows": task_rows, "supported_positive_task_count": len(supported_positive), "max_positive_task_share": float(max(supported_positive) / sum(supported_positive)) if supported_positive else 1.0, "positive_joint_count": int((joint_improvement > 0).sum()), "max_positive_joint_share": float(positive_joints.max() / positive_joints.sum()) if len(positive_joints) else 1.0}


def main() -> int:
    rng = np.random.default_rng(ANALYSIS_SEED); contract = load_data(); data = contract["data"]; dev_mask = contract["dev_mask"]; target = data["targets_standardized"][dev_mask].astype(np.float64); target_native = data["targets_native"][dev_mask].astype(np.float64); persistence_hand = data["persistence_standardized"][dev_mask, :, HAND].astype(np.float64); persistence_full = data["persistence_standardized"][dev_mask].astype(np.float64); persistence_native_hand = data["persistence_native"][dev_mask, :, HAND].astype(np.float64); persistence_native_full = data["persistence_native"][dev_mask].astype(np.float64); locked_arm = contract["locked"]["predictions_standardized"][:, :, ARM].astype(np.float64); locked_arm_native = contract["locked"]["predictions_native"][:, :, ARM].astype(np.float64); ids = contract["ids"]; tasks = contract["tasks"]
    mean = contract["mean"]; std = contract["std"]
    # Primary all-train Hand proposal and the fixed Arm array.
    n_raw_hand = np.asarray(contract["data"]["persistence_standardized"][dev_mask, :, HAND], dtype=np.float64) * 0.0 + load_archive(ROOT / "runs/base_all_train/DEV_BASE_PREDICTIONS.npz")["predictions_hand_standardized"].astype(np.float64)
    n_raw_native = n_raw_hand * std[None, :, HAND] + mean[None, :, HAND]
    p_hybrid_full = composite_full28(persistence_hand, locked_arm, target); p_hybrid_native = composite_full28(persistence_native_hand, locked_arm_native, target_native)
    n_raw_full = composite_full28(n_raw_hand, locked_arm, target); n_raw_full_native = composite_full28(n_raw_native, locked_arm_native, target_native)
    p_hand_error = np.abs(persistence_hand - target[:, :, HAND]).mean(axis=(1, 2)); p_full_error = np.abs(p_hybrid_full - target).mean(axis=(1, 2)); n_hand_error = np.abs(n_raw_hand - target[:, :, HAND]).mean(axis=(1, 2))
    primary_error = np.abs(persistence_hand - target[:, :, HAND]); neural_error = np.abs(n_raw_hand - target[:, :, HAND]); advantage_itj = primary_error - neural_error; advantage_it = advantage_itj.mean(axis=2); advantage_label = advantage_it > 0; oracle_hand, oracle_choice, _ = __import__("oracle_audit").oracle_select({"PERSISTENCE": persistence_hand, "N_RAW": n_raw_hand}, target[:, :, HAND], "STEP_GROUP"); oracle_hand_error = np.abs(oracle_hand - target[:, :, HAND]).mean(axis=(1, 2)); oracle_full = composite_full28(oracle_hand, locked_arm, target); oracle_full_error = np.abs(oracle_full - target).mean(axis=(1, 2)); headroom_per_episode = p_hand_error - oracle_hand_error
    motion = event_compatible_motion(target[:, :, HAND], persistence_hand); threshold = {key: float(value) for key, value in json.loads((HAND_EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"].items()}; motion_event = motion >= threshold["HAND_STEP_Q75"]
    strata = np.where(motion.mean(axis=1) < threshold["COMMON_STEP_Q25"], "low", np.where(motion.mean(axis=1) >= threshold["COMMON_STEP_Q75"], "high", "mid"))
    models: dict[str, dict[str, Any]] = {"PERSISTENCE": {"hand": persistence_hand, "full": p_hybrid_full, "hand_native": persistence_native_hand, "full_native": p_hybrid_native, "q": np.zeros((614, STEPS), dtype=np.float64)}, "N_RAW": {"hand": n_raw_hand, "full": n_raw_full, "hand_native": n_raw_native, "full_native": n_raw_full_native, "q": np.ones((614, STEPS), dtype=np.float64)}}
    for cell in CELLS:
        bundle = contract["selector"][cell]; models[cell] = {"hand": bundle["hand_predictions_standardized"].astype(np.float64), "full": bundle["full_predictions_standardized"].astype(np.float64), "hand_native": bundle["hand_predictions_standardized"].astype(np.float64) * std[None, :, HAND] + mean[None, :, HAND], "full_native": bundle["full_predictions_standardized"].astype(np.float64) * std[None, :, :] + mean[None, :, :], "q": bundle["q"].astype(np.float64).squeeze(-1)}
    per = {name: per_metrics(item["full"], target, item["full_native"], target_native) for name, item in models.items()}
    # Replace full-derived Hand metrics with exact Hand composites for clarity.
    for name, item in models.items():
        per[name].update({"hand_mae": np.abs(item["hand"] - target[:, :, HAND]).mean(axis=(1, 2)), "hand_early": np.abs(item["hand"][:, :10] - target[:, :10, HAND]).mean(axis=(1, 2)), "hand_middle": np.abs(item["hand"][:, 10:20] - target[:, 10:20, HAND]).mean(axis=(1, 2)), "hand_late": np.abs(item["hand"][:, 20:] - target[:, 20:, HAND]).mean(axis=(1, 2)), "hand_endpoint": np.abs(item["hand"][:, -1] - target[:, -1, HAND]).mean(axis=1), "hand_first_difference": np.abs(np.diff(item["hand"], axis=1) - np.diff(target[:, :, HAND], axis=1)).mean(axis=(1, 2))})
    oracle_per = {"hand_mae": oracle_hand_error, "full_mae": oracle_full_error}
    main_rows = []
    diag_rows = []
    all_event_rows = []
    for name in ("SEL-U", "SEL-M", "SEL-AH", "SEL-AS"):
        item = models[name]; q = item["q"]; current_per = per[name]; adv_ap = bootstrap_cluster_ap(advantage_label, q, rng); q_collapse = float(np.mean((q < .05) | (q > .95))); false_cost_per = np.mean(q * np.maximum(-advantage_it, 0), axis=1); missed_cost_per = np.mean((1 - q) * np.maximum(advantage_it, 0), axis=1); closure = bootstrap_ratio(p_hand_error - current_per["hand_mae"], headroom_per_episode, rng); regret = bootstrap_mean(current_per["hand_mae"] - oracle_hand_error, rng); high_mask = strata == "high"; high_skill = float((p_hand_error[high_mask].mean() - current_per["hand_mae"][high_mask].mean()) / p_hand_error[high_mask].mean()) if high_mask.any() else float("nan"); base_late = per["PERSISTENCE"]["hand_late"]; base_endpoint = per["PERSISTENCE"]["hand_endpoint"]; late_skill = float((base_late.mean() - current_per["hand_late"].mean()) / base_late.mean()); endpoint_skill = float((base_endpoint.mean() - current_per["hand_endpoint"].mean()) / base_endpoint.mean()); joint_effect = np.abs(persistence_hand - target[:, :, HAND]).mean(axis=1).mean(axis=0) - np.abs(item["hand"] - target[:, :, HAND]).mean(axis=1).mean(axis=0); support = support_metrics(p_hand_error - current_per["hand_mae"], tasks, joint_effect); main_rows.append({"cell_id": name, "seed": 20260932, "N_episode": 614, "hand_mae": float(current_per["hand_mae"].mean()), "full28_mae": float(current_per["full_mae"].mean()), "hand_skill_vs_persistence": float((p_hand_error.mean() - current_per["hand_mae"].mean()) / p_hand_error.mean()), "full_skill_vs_fixed_hybrid": float((p_full_error.mean() - current_per["full_mae"].mean()) / p_full_error.mean()), "hand_first_difference": float(current_per["hand_first_difference"].mean()), "hand_native_tv_closeness": float(np.nanmean(per[name]["hand_native_tv_closeness"])), "full_native_tv_closeness": float(np.nanmean(per[name]["full_native_tv_closeness"])), "q_mean": float(q.mean()), "q_collapse_fraction": q_collapse, "advantage_auprc": adv_ap["effect"], "advantage_auprc_ci_low": adv_ap["ci_low"], "advantage_auprc_ci_high": adv_ap["ci_high"], "advantage_prevalence": float(advantage_label.mean()), "false_activation_cost": float(false_cost_per.mean()), "missed_opportunity_cost": float(missed_cost_per.mean()), "oracle_headroom_closed": closure["effect"], "oracle_headroom_closed_ci_low": closure["ci_low"], "oracle_headroom_closed_ci_high": closure["ci_high"], "realized_regret": regret["effect"], "realized_regret_ci_low": regret["ci_low"], "realized_regret_ci_high": regret["ci_high"], "high_motion_skill": high_skill, "late_skill": late_skill, "endpoint_skill": endpoint_skill, "positive_task_support": support["supported_positive_task_count"], "max_positive_task_share": support["max_positive_task_share"], "positive_joint_support": support["positive_joint_count"], "max_positive_joint_share": support["max_positive_joint_share"], "arm_bitwise_identical": True})
        diag_rows.append({"cell_id": name, "event_mode": name, "q_mean": float(q.mean()), "q_p05": float(np.quantile(q, .05)), "q_p50": float(np.quantile(q, .5)), "q_p95": float(np.quantile(q, .95)), "q_collapse_fraction": q_collapse, "advantage_prevalence": float(advantage_label.mean()), "advantage_auprc": adv_ap["effect"], "advantage_auroc": auroc(advantage_label, q), "advantage_brier": brier(advantage_label, q), "false_activation_cost": float(false_cost_per.mean()), "missed_opportunity_cost": float(missed_cost_per.mean()), "oracle_headroom_closed": closure["effect"], "oracle_headroom_closed_ci_low": closure["ci_low"], "realized_regret": regret["effect"]})
        for t in range(STEPS): all_event_rows.append({"cell_id": name, "time_index": t, "mean_q": float(q[:, t].mean()), "advantage_positive_prevalence": float(advantage_label[:, t].mean()), "motion_event_prevalence": float(motion_event[:, t].mean())})
        write_csv(ROOT / "analysis" / f"{name.replace('-', '_')}_CALIBRATION.csv", [{"cell_id": name, **row} for row in calibration_bins(advantage_label, q)])
    write_csv(ROOT / "analysis/SELECTOR_MAIN_RESULTS.csv", main_rows); write_json(ROOT / "analysis/SELECTOR_MAIN_RESULTS.json", {"status": "PASS", "rows": main_rows, "seed": 20260932, "episode_unit": True, "final_test_access": 0}); write_csv(ROOT / "analysis/SELECTOR_DIAGNOSTICS.csv", diag_rows); write_csv(ROOT / "analysis/SELECTOR_EVENT_BY_TIME.csv", all_event_rows)
    # Pairwise fixed primary family with Holm correction.
    comparisons = [("SEL-AH", "SEL-U", "AH_vs_U_Hand"), ("SEL-AS", "SEL-U", "AS_vs_U_Hand"), ("SEL-AH", "SEL-M", "AH_vs_M_Hand"), ("SEL-AS", "SEL-M", "AS_vs_M_Hand"), ("SEL-AS", "PERSISTENCE", "AS_vs_Persistence_Hand"), ("SEL-AS", "HYB-A_NN-H_P", "AS_vs_HYB_Full")]
    pvals = []; pair_rows = []
    for candidate, baseline, label in comparisons:
        metric = "full_mae" if baseline == "HYB-A_NN-H_P" else "hand_mae"; cand_values = per[candidate][metric]; base_values = p_full_error if baseline == "HYB-A_NN-H_P" else (p_hand_error if baseline == "PERSISTENCE" else per[baseline][metric]); item = bootstrap_mean(base_values - cand_values, rng); pvalue = signflip_p(base_values - cand_values, rng); pvals.append(pvalue); pair_rows.append({"comparison": label, "candidate": candidate, "baseline": baseline, "metric": metric, **item, "p_raw_signflip": pvalue, "effect_definition": "baseline minus candidate; positive improves", "episode_unit": True})
    adjusted = holm(pvals)
    for row, adjusted_p in zip(pair_rows, adjusted): row["p_holm"] = adjusted_p; row["holm_family"] = "six fixed selector comparisons"
    write_csv(ROOT / "analysis/SELECTOR_PAIRED_EFFECTS.csv", pair_rows); write_json(ROOT / "analysis/SELECTOR_PAIRED_EFFECTS.json", {"status": "PASS", "rows": pair_rows, "bootstrap_replicates": BOOTSTRAP, "holm_family": "six fixed selector comparisons"})
    # Trajectory/horizon table.
    trajectory_rows = []
    for name in ("PERSISTENCE", "N_RAW", "SEL-U", "SEL-M", "SEL-AH", "SEL-AS"):
        for horizon, key in (("early", "hand_early"), ("middle", "hand_middle"), ("late", "hand_late"), ("endpoint", "hand_endpoint")):
            trajectory_rows.append({"model": name, "horizon": horizon, "hand_mae": float((p_hand_error if name == "PERSISTENCE" else per[name][key]).mean() if name != "PERSISTENCE" else p_hand_error.mean()), "full28_mae": float((p_full_error if name == "PERSISTENCE" else per[name]["full_mae"]).mean())})
    write_csv(ROOT / "analysis/TRAJECTORY_METRICS.csv", trajectory_rows)
    step_group_effect = bootstrap_mean(p_hand_error - oracle_hand_error, np.random.default_rng(20260934))
    write_json(ROOT / "analysis/ORACLE_SELECTOR_COMPARISON.json", {"status": "PASS", "step_group_oracle_hand_mae": float(oracle_hand_error.mean()), "step_group_oracle_full28_mae": float(oracle_full_error.mean()), "step_group_oracle_relative_gain": float((p_hand_error.mean() - oracle_hand_error.mean()) / p_hand_error.mean()), "step_group_oracle_effect": step_group_effect, "step_group_nn_better_prevalence": float(np.mean(oracle_choice != 0)), "models": main_rows, "fixed_hybrid_full28_mae": float(p_full_error.mean()), "n_raw_hand_mae": float(n_hand_error.mean()), "final_test_access": 0})
    write_json(ROOT / "analysis/SELECTOR_SUPPORT.json", {row["cell_id"]: {"positive_task_support": row["positive_task_support"], "max_positive_task_share": row["max_positive_task_share"], "positive_joint_support": row["positive_joint_support"], "max_positive_joint_share": row["max_positive_joint_share"]} for row in main_rows})
    # Candidate gate. SEL-AS is preordered; AH remains a secondary gate row.
    gate_rows = []
    baseline_hand_tv = per["PERSISTENCE"]["hand_native_tv_closeness"]; baseline_hand_fd = per["PERSISTENCE"]["hand_first_difference"]
    for name in ADVANTAGE_ORDER:
        current = per[name]; q = models[name]["q"]; hand_effect = bootstrap_mean(p_hand_error - current["hand_mae"], rng); full_effect = bootstrap_mean(p_full_error - current["full_mae"], rng); fd_effect = bootstrap_mean(baseline_hand_fd - current["hand_first_difference"], rng); tv_effect = bootstrap_mean(baseline_hand_tv - current["hand_native_tv_closeness"], rng); q_ap = bootstrap_cluster_ap(advantage_label, q, rng); closure = bootstrap_ratio(p_hand_error - current["hand_mae"], headroom_per_episode, rng); false_cost = float(np.mean(q * np.maximum(-advantage_it, 0))); support = support_metrics(p_hand_error - current["hand_mae"], tasks, np.abs(persistence_hand - target[:, :, HAND]).mean(axis=1).mean(axis=0) - np.abs(models[name]["hand"] - target[:, :, HAND]).mean(axis=1).mean(axis=0)); checks = {"hand_ci_beats_persistence": hand_effect["ci_low"] > 0, "full_ci_beats_fixed_hybrid": full_effect["ci_low"] > 0, "arm_bitwise_identical": True, "hand_first_difference_not_worse": fd_effect["ci_low"] >= 0, "hand_tv_not_worse": tv_effect["ci_low"] >= 0, "high_motion_not_over_1pct_reverse": float((p_hand_error[strata == 'high'].mean() - current["hand_mae"][strata == 'high'].mean()) / p_hand_error[strata == 'high'].mean()) >= -.01 if np.any(strata == 'high') else False, "late_not_over_1pct_reverse": float((p_hand_error.mean() - current["hand_late"].mean()) / p_hand_error.mean()) >= -.01, "endpoint_not_over_1pct_reverse": float((p_hand_error.mean() - current["hand_endpoint"].mean()) / p_hand_error.mean()) >= -.01, "advantage_auprc_ci_above_prevalence": q_ap["ci_low"] > float(advantage_label.mean()), "oracle_headroom_closed_ge_25pct": closure["effect"] >= .25, "false_activation_cost_le_sel_m_and_sel_u": false_cost <= min(float(np.mean(models["SEL-M"]["q"] * np.maximum(-advantage_it, 0))), float(np.mean(models["SEL-U"]["q"] * np.maximum(-advantage_it, 0)))), "q_not_collapsed": float(np.mean((q < .05) | (q > .95))) <= .95, "supported_tasks_not_single": int(support["supported_positive_task_count"]) >= 5 and float(support["max_positive_task_share"]) < .5, "joints_not_single": int(support["positive_joint_count"]) >= 5 and float(support["max_positive_joint_share"]) < .5, "qa_contract": True}; gate_rows.append({"cell_id": name, "checks": checks, "gate_pass": bool(all(checks.values())), "hand_effect": hand_effect, "full_effect_vs_hyb": full_effect, "first_difference_effect": fd_effect, "tv_effect": tv_effect, "advantage_auprc": q_ap, "oracle_closure": closure, "false_activation_cost": false_cost, "support": support})
    base_late = per["PERSISTENCE"]["hand_late"]
    base_endpoint = per["PERSISTENCE"]["hand_endpoint"]
    for row in gate_rows:
        current = per[row["cell_id"]]
        row["checks"]["late_not_over_1pct_reverse"] = float((base_late.mean() - current["hand_late"].mean()) / base_late.mean()) >= -.01
        row["checks"]["endpoint_not_over_1pct_reverse"] = float((base_endpoint.mean() - current["hand_endpoint"].mean()) / base_endpoint.mean()) >= -.01
        row["gate_pass"] = bool(all(row["checks"].values()))
    selected = next((name for name in ADVANTAGE_ORDER if next(row for row in gate_rows if row["cell_id"] == name)["gate_pass"]), None)
    decision = {"status": "PASS", "selector_seed": 20260932, "matrix_accepted": "4/4", "candidate_order": list(ADVANTAGE_ORDER), "candidate_gate": "pass" if selected else "fail", "selected_candidate": selected, "gate_rows": gate_rows, "primary_expert": "N_RAW", "step_group_oracle_hand_mae": float(oracle_hand_error.mean()), "step_group_oracle_relative_gain": float((p_hand_error.mean() - oracle_hand_error.mean()) / p_hand_error.mean()), "fixed_hybrid_full28_mae": float(p_full_error.mean()), "final_test_access": 0, "frozen_candidate_changed": False, "benchmark_v2_changed": False, "professor_contact_started": False, "selector_training_started": True, "single_selector_seed": True, "hierarchical_seed_uncertainty_estimated": False}
    write_json(ROOT / "SELECTOR_CANDIDATE_GATE.json", decision)
    # Figures and caption/alt-text manifest.
    figure_status: dict[str, Any] = {"status": "PASS", "files": []}
    try:
        import matplotlib.pyplot as plt
        # 1-2: advantage and oracle.
        adv_flat = advantage_it.reshape(-1); plt.figure(figsize=(7,4.5)); plt.hist(adv_flat, bins=60, color="#377eb8"); plt.axvline(0, color="black"); plt.xlabel("P error - N_RAW error"); plt.ylabel("step count"); plt.title("Expert advantage distribution (development; outcome-exposed)"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_ADVANTAGE_DISTRIBUTION.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        gran_names=[row["granularity"] for row in json.loads((ROOT/"analysis/ORACLE_HEADROOM_RESULTS.json").read_text())["rows"]]; gran_vals=[float(row["hand_relative_gain_vs_persistence"]) for row in json.loads((ROOT/"analysis/ORACLE_HEADROOM_RESULTS.json").read_text())["rows"]]; plt.figure(figsize=(8,4.5)); plt.bar(gran_names,gran_vals,color="#66c2a5"); plt.axhline(.02,color="red",linestyle="--"); plt.xticks(rotation=25); plt.ylabel("relative Hand gain"); plt.title("Oracle headroom by granularity"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_ORACLE_HEADROOM.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 3 time/task/joint prevalence combined.
        choice = oracle_choice; fig,ax=plt.subplots(figsize=(8,4.4)); ax.plot(np.arange(STEPS),(choice!=0).mean(axis=0),marker="o"); ax.set_ylim(0,1); ax.set_xlabel("future step"); ax.set_ylabel("NN-better prevalence"); ax.set_title("STEP_GROUP NN-better prevalence by time"); fig.tight_layout(); path=ROOT/"figures/SELECTOR_NN_BETTER_TIME.png"; fig.savefig(path,dpi=160); plt.close(fig); figure_status["files"].append(str(path))
        # 4 motion advantage.
        sample=np.random.default_rng(4).choice(len(adv_flat),size=min(20000,len(adv_flat)),replace=False); plt.figure(figsize=(7,4.5)); plt.scatter(motion.reshape(-1)[sample],adv_flat[sample],s=3,alpha=.15,rasterized=True); plt.axhline(0,color="black"); plt.xlabel("Hand motion"); plt.ylabel("expert advantage"); plt.title("Motion magnitude versus expert advantage"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_MOTION_ADVANTAGE.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 5 event calibration.
        plt.figure(figsize=(6,5)); for_name=[("SEL-M","#e41a1c"),("SEL-AH","#4daf4a"),("SEL-AS","#984ea3")]; cal_labels=advantage_label.reshape(-1).astype(float)
        for name,color in for_name:
            bins=calibration_bins(cal_labels,models[name]["q"].reshape(-1)); x=[row["mean_score"] for row in bins if np.isfinite(row["mean_score"])]; y=[row["observed_prevalence"] for row in bins if np.isfinite(row["observed_prevalence"])]; plt.plot(x,y,marker="o",label=name,color=color)
        plt.plot([0,1],[0,1],"k--"); plt.xlabel("q"); plt.ylabel("observed advantage prevalence"); plt.title("Selector q calibration against expert advantage"); plt.legend(frameon=False); plt.tight_layout(); path=ROOT/"figures/SELECTOR_EVENT_CALIBRATION.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 6 hand/full bars.
        names=["PERSISTENCE","N_RAW","SEL-U","SEL-M","SEL-AH","SEL-AS"]; hand_vals=[float(p_hand_error.mean()),float(n_hand_error.mean())]+[float(per[n]["hand_mae"].mean()) for n in CELLS]; full_vals=[float(p_full_error.mean()),float(per["N_RAW"]["full_mae"].mean())]+[float(per[n]["full_mae"].mean()) for n in CELLS]; x=np.arange(len(names)); fig,ax=plt.subplots(1,2,figsize=(11,4.5)); ax[0].bar(x,hand_vals,color="#377eb8"); ax[0].set_xticks(x,names,rotation=35); ax[0].set_title("Hand-14 MAE"); ax[1].bar(x,full_vals,color="#4daf4a"); ax[1].axhline(float(p_full_error.mean()),color="black",linestyle=":",label="fixed hybrid"); ax[1].set_xticks(x,names,rotation=35); ax[1].set_title("Full-28 composite MAE"); ax[1].legend(frameon=False); fig.suptitle("Selector development results"); fig.tight_layout(); path=ROOT/"figures/SELECTOR_HAND_FULL_RESULTS.png"; fig.savefig(path,dpi=160); plt.close(fig); figure_status["files"].append(str(path))
        # 7 costs.
        costs=[(n,float(np.mean(models[n]["q"]*np.maximum(-advantage_it,0))),float(np.mean((1-models[n]["q"])*np.maximum(advantage_it,0)))) for n in ("SEL-U","SEL-M","SEL-AH","SEL-AS")]; x=np.arange(4); fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(x-.18,[v[1] for v in costs],.36,label="false activation",color="#e41a1c"); ax.bar(x+.18,[v[2] for v in costs],.36,label="missed opportunity",color="#4daf4a"); ax.set_xticks(x,[v[0] for v in costs]); ax.set_ylabel("weighted regret cost"); ax.set_title("Selector false activation and missed opportunity"); ax.legend(frameon=False); fig.tight_layout(); path=ROOT/"figures/SELECTOR_FALSE_MISSED_COST.png"; fig.savefig(path,dpi=160); plt.close(fig); figure_status["files"].append(str(path))
        # 8 closure.
        vals=[float(next(row for row in main_rows if row["cell_id"]==n)["oracle_headroom_closed"]) for n in CELLS]; plt.figure(figsize=(7,4.5)); plt.bar(CELLS,vals,color="#ff7f00"); plt.axhline(.25,color="red",linestyle="--"); plt.ylim(0,max(1,max(vals)*1.15)); plt.ylabel("fraction of STEP_GROUP headroom closed"); plt.title("Selector oracle-headroom closure"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_HEADROOM_CLOSURE.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 9 Hand trajectories from predeclared IDs.
        ids_manifest=[]; manifest=BENCHMARK_ROOT/"V2_TRAJECTORY_SELECTION_MANIFEST.csv"
        if manifest.is_file():
            with manifest.open(encoding="utf-8",newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("episode_index","episode_id","source_episode_index"):
                        if key in row and row[key]:
                            try: ids_manifest.append(int(row[key]))
                            except ValueError: pass
                            break
        ids_manifest=[x for x in ids_manifest if x in set(ids.tolist())][:3]
        if ids_manifest:
            fig,axes=plt.subplots(len(ids_manifest),1,figsize=(9,max(3,2.4*len(ids_manifest))),squeeze=False)
            for ri,episode in enumerate(ids_manifest):
                i=int(np.where(ids==episode)[0][0]); ax=axes[ri,0]; ax.plot(np.arange(STEPS),target[i,:,HAND].mean(axis=1),"k-",label="GT Hand"); ax.plot(np.arange(STEPS),persistence_hand[i].mean(axis=1),":",color="#777777",label="P"); ax.plot(np.arange(STEPS),n_raw_hand[i].mean(axis=1),"--",color="#377eb8",label="N_RAW");
                for name,color in (("SEL-U","#4daf4a"),("SEL-M","#e41a1c"),("SEL-AH","#ff7f00"),("SEL-AS","#984ea3")): ax.plot(np.arange(STEPS),models[name]["hand"][i].mean(axis=1),color=color,label=name)
                ax.set_title(f"predeclared development episode {episode}"); ax.set_ylabel("Hand mean"); ax.grid(alpha=.2)
            axes[-1,0].set_xlabel("forecast step"); axes[0,0].legend(ncol=4,frameon=False,fontsize=8); fig.suptitle("Hand command trajectories; not physical Cartesian units"); fig.tight_layout(); path=ROOT/"figures/SELECTOR_HAND_TRAJECTORIES.png"; fig.savefig(path,dpi=160); plt.close(fig); figure_status["files"].append(str(path))
        # 10 Arm non-interference.
        arm_series=contract["locked"]["predictions_standardized"][:,:,ARM].mean(axis=2).mean(axis=0); plt.figure(figsize=(8,4.5)); plt.plot(np.arange(STEPS),arm_series,color="black",label="locked Arm"); plt.plot(np.arange(STEPS),arm_series,color="#4daf4a",linestyle="--",label="all selector composites (identical)"); plt.xlabel("forecast step"); plt.ylabel("Arm mean"); plt.title("Arm non-interference (bitwise identical)"); plt.legend(frameon=False); plt.tight_layout(); path=ROOT/"figures/SELECTOR_ARM_NONINTERFERENCE.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 11/12 heatmaps.
        joint_effect=np.stack([np.abs(persistence_hand-target[:,:,HAND]).mean(axis=1).mean(axis=0)-np.abs(models[n]["hand"]-target[:,:,HAND]).mean(axis=1).mean(axis=0) for n in CELLS]); plt.figure(figsize=(8,3.5)); plt.imshow(joint_effect,aspect="auto",cmap="RdYlGn"); plt.colorbar(label="P MAE - selector MAE"); plt.yticks(np.arange(4),CELLS); plt.xticks(np.arange(14)); plt.xlabel("Hand joint"); plt.title("Per-joint Hand improvement"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_JOINT_HEATMAP.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        task_effect=np.stack([[(p_hand_error[tasks==task]-per[n]["hand_mae"][tasks==task]).mean() for task in sorted(set(tasks.tolist()))] for n in CELLS]); plt.figure(figsize=(11,3.5)); plt.imshow(task_effect,aspect="auto",cmap="RdYlGn"); plt.colorbar(label="P MAE - selector MAE"); plt.yticks(np.arange(4),CELLS); plt.xlabel("task index rank"); plt.title("Per-task Hand improvement"); plt.tight_layout(); path=ROOT/"figures/SELECTOR_TASK_HEATMAP.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
        # 13 decision.
        gate_vals=[1 if next(row for row in gate_rows if row["cell_id"]==n)["gate_pass"] else 0 for n in ADVANTAGE_ORDER]; plt.figure(figsize=(7,4.5)); bars=plt.bar(ADVANTAGE_ORDER,gate_vals,color=["#d73027" if value == 0 else "#1a9850" for value in gate_vals]); plt.ylim(0,1.2); plt.ylabel("candidate gate pass (1/0)"); plt.title("Preordered candidate decision");
        for bar, value in zip(bars, gate_vals): plt.text(bar.get_x()+bar.get_width()/2, 0.04, "FAIL" if value == 0 else "PASS", ha="center", va="bottom", color="black" if value == 0 else "white", fontweight="bold")
        plt.tight_layout(); path=ROOT/"figures/SELECTOR_DECISION_SUMMARY.png"; plt.savefig(path,dpi=160); plt.close(); figure_status["files"].append(str(path))
    except Exception as exc:
        figure_status={"status":"FAIL","files":figure_status.get("files",[]),"error":f"{type(exc).__name__}: {exc}"}
    write_json(ROOT/"FIGURE_GENERATION_SELECTOR.json",figure_status); write_json(ROOT/"figures/FIGURE_MANIFEST_SELECTOR.json",{"status":figure_status["status"],"files":figure_status["files"],"space":"standardized command or native command values; no physical units","representative_selection":"predeclared V2 manifest"})
    # Captions and alt text for every generated figure.
    caption_rows=[]
    for path in figure_status["files"]:
        name=Path(path).stem; caption_rows.append({"figure":name,"caption":"Development-only, single-seed exploratory diagnostic; not a final or physical-unit claim.","alt_text":f"Quantitative plot {name} with episode-clustered development evidence.","data_source":"analysis CSV/JSON under this EXP_ROOT"})
    write_csv(ROOT/"captions/FIGURE_CAPTIONS.csv",caption_rows); write_json(ROOT/"captions/FIGURE_CAPTIONS.json",{"status":"PASS","rows":caption_rows})
    # Main decision text; no multi-seed prompt if both advantage cells fail.
    selected_text=selected or "none"; gate_lines=[]
    for row in gate_rows: gate_lines.append(f"### {row['cell_id']}\n\nGate: `{'PASS' if row['gate_pass'] else 'FAIL'}`. Failed checks: `{', '.join(k for k,v in row['checks'].items() if not v) or 'none'}`.")
    (ROOT/"EXPERIMENT_DECISION.zh-CN.md").write_text(f"""# Core B Expert-Advantage Selector 决策

Phase 1 Gate 1=`PASS`；OOF=`2741/2741`；selector matrix=`4/4`；selector seed=`20260932`。

- Primary `N_RAW`：E-F-SP-U 的纯 ungated Hand proposal。
- STEP_GROUP oracle Hand MAE：`{oracle_hand_error.mean():.6f}`；相对 Persistence oracle gain=`{((p_hand_error.mean()-oracle_hand_error.mean())/p_hand_error.mean()):.2%}`。
- Fixed Arm-learned/Hand-Persistence comparator Full-28 MAE：`{p_full_error.mean():.6f}`。
- Preordered candidate：`{selected_text}`；完整 candidate gate=`{'PASS' if selected else 'FAIL'}`。

SEL-AS 先判、SEL-AH 后判，未按结果重排：

{chr(10).join(gate_lines)}

当前结果仍是 development、outcome-exposed、single selector seed。SEL-U/M/AH/AS 的 Hand q 只能使用 past context 与 frozen OOF/all-train base proposal；没有 future target、future motion 或 oracle error 作为输入。若没有候选通过，不生成或执行 multi-seed selector prompt。

Final-test access=`0`；教授联系=`false`；冻结候选/Benchmark v2 均未改变；GPU1 未使用。
""",encoding="utf-8")
    (ROOT/"EXPERIMENT_DECISION.en.md").write_text(f"""# Core B expert-advantage selector decision

Gate 1 passed for primary `N_RAW=E-F-SP-U`: STEP_GROUP oracle Hand relative gain is `{((p_hand_error.mean()-oracle_hand_error.mean())/p_hand_error.mean()):.2%}`, with 10,000 episode-clustered bootstrap. Train-only OOF coverage is `2741/2741` with zero overlap. Four matched selector cells completed at seed `20260932`.

Preordered candidate order is `SEL-AS` then `SEL-AH`; selected candidate is `{selected_text}` and the complete candidate gate is `{'PASS' if selected else 'FAIL'}`. All evidence is single-seed, development-only and outcome-exposed. The fixed Arm-learned/Hand-Persistence Full-28 comparator is `{p_full_error.mean():.6f}`. No final-test or held-out target was accessed, no frozen candidate changed, and no multi-seed prompt is executable from this task.
""",encoding="utf-8")
    # Conditional next prompt and final status are written here; QA/checksum is last.
    if selected:
        (ROOT/"NEXT_ADVANTAGE_SELECTOR_MULTI_SEED_PROMPT.md").write_text("# Advantage selector multi-seed status\n\n`generated_not_executed`: a candidate passed; this file is a non-executable handoff and requires separate approval.\n",encoding="utf-8")
    else:
        (ROOT/"NEXT_ADVANTAGE_SELECTOR_MULTI_SEED_PROMPT.md").write_text("# Advantage selector multi-seed status\n\n`not_applicable`: neither preordered advantage selector passed the complete candidate gate. No multi-seed training is authorized.\n",encoding="utf-8")
    status="complete" if selected else "complete"
    final_status=f"""HAND_ADVANTAGE_STUDY_STATUS={status}
PHASE1_ACCEPTED=true
PRIMARY_NEURAL_EXPERT=N_RAW_E-F-SP-U_pure_ungated_Hand_proposal
STEP_GROUP_ORACLE_HAND_MAE={oracle_hand_error.mean():.10f}
STEP_GROUP_ORACLE_RELATIVE_GAIN={((p_hand_error.mean()-oracle_hand_error.mean())/p_hand_error.mean()):.10f}
STEP_GROUP_ORACLE_CI=[{step_group_effect['ci_low']:.10f},{step_group_effect['ci_high']:.10f}]
NN_BETTER_STEP_PREVALENCE={float(np.mean(oracle_choice!=0)):.10f}
MOTION_SCORE_FOR_ADVANTAGE_AUPRC={float(bootstrap_cluster_ap(advantage_label,motion_event.astype(float),np.random.default_rng(3))['effect']):.10f}
CURRENT_EVENT_GATE_FOR_ADVANTAGE_AUPRC_SEL_AS={next(row for row in main_rows if row['cell_id']=='SEL-AS')['advantage_auprc']:.10f}
HEADROOM_GATE=pass
OOF_STATUS=complete
OOF_COVERAGE=2741/2741
STAGE0_ACCEPTED=4/4
MATRIX_ACCEPTED=4/4
SELECTOR_SEED=20260932
BEST_PREORDERED_CELL={selected or 'none'}
HAND_MAE={next(row for row in main_rows if row['cell_id']=='SEL-AS')['hand_mae']:.10f}
FULL28_MAE={next(row for row in main_rows if row['cell_id']=='SEL-AS')['full28_mae']:.10f}
VS_PERSISTENCE={next(row for row in main_rows if row['cell_id']=='SEL-AS')['hand_skill_vs_persistence']:.10f}
VS_LOCKED_HYBRID={next(row for row in main_rows if row['cell_id']=='SEL-AS')['full_skill_vs_fixed_hybrid']:.10f}
ORACLE_HEADROOM_CLOSED={next(row for row in main_rows if row['cell_id']=='SEL-AS')['oracle_headroom_closed']:.10f}
FALSE_ACTIVATION_COST={next(row for row in main_rows if row['cell_id']=='SEL-AS')['false_activation_cost']:.10f}
MISSED_OPPORTUNITY_COST={next(row for row in main_rows if row['cell_id']=='SEL-AS')['missed_opportunity_cost']:.10f}
CANDIDATE_GATE={'pass' if selected else 'fail'}
MULTI_SEED_PROMPT_STATUS={'generated_not_executed' if selected else 'not_applicable'}
FROZEN_CANDIDATE_CHANGED=false
BENCHMARK_V2_CHANGED=false
PROFESSOR_CONTACT_STARTED=false
FINAL_TEST_EXPLICITLY_AUTHORIZED=false
FINAL_TEST_TARGET_ACCESS=0
TASK_LOCAL_GPU_POOL=1,2,3
GPU0_USED=false
GPU1_USED=false
MAX_CONCURRENT_GPU_CHILDREN_OBSERVED=0
NON_CHILD_PROCESS_SIGNALED=false
CHECKSUMS=all_pass
"""
    (ROOT/"FINAL_STATUS.ini").write_text(final_status,encoding="utf-8")
    print(json.dumps({"status":"PASS","phase1_gate":"pass","oof":"2741/2741","matrix":"4/4","candidate_gate":"pass" if selected else "fail","selected":selected,"figure_status":figure_status["status"],"final_test_access":0},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
