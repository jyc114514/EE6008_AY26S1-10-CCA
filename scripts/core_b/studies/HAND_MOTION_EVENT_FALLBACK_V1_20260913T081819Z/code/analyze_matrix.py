#!/usr/bin/env python3
"""Analyze the six-cell Hand event-gate matrix with episode-clustered bootstrap."""

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
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
PRIOR_ROOT = ROOT.parent / "POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z"
PRIOR_PREDICTION_PATH = PRIOR_ROOT / "runs/E-F-SP-G/predictions_best.npz"
MANIFEST_PATH = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/V2_TRAJECTORY_SELECTION_MANIFEST.csv"

CELL_ORDER = ("H-S-U", "H-S-W", "H-S-T", "H-F-U", "H-F-W", "H-F-T")
CELL_FACTORS = {
    "H-S-U": {"input": "STATE", "event_mode": "U"},
    "H-S-W": {"input": "STATE", "event_mode": "W"},
    "H-S-T": {"input": "STATE", "event_mode": "T"},
    "H-F-U": {"input": "FUSION", "event_mode": "U"},
    "H-F-W": {"input": "FUSION", "event_mode": "W"},
    "H-F-T": {"input": "FUSION", "event_mode": "T"},
}
BOOTSTRAP = 10_000
ANALYSIS_SEED = 20260928
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("invalid episode bootstrap vector")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    estimates = values[indices].mean(axis=1)
    return {"effect": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "replicates": BOOTSTRAP, "N_episode": int(len(values)), "bootstrap_method": "hierarchical_seed_episode; one training seed so seed resampling is degenerate"}


def effect(candidate: np.ndarray, baseline: np.ndarray, rng: np.random.Generator) -> dict[str, float | int]:
    """Baseline minus candidate; positive means candidate improves."""
    return bootstrap_mean(np.asarray(baseline) - np.asarray(candidate), rng)


def group_tv(pred_native: np.ndarray, target_native: np.ndarray, group: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_DENOMINATOR_FLOOR)
    return pred_tv, target_tv, np.abs(ratio - 1.0)


def per_metrics(pred: np.ndarray, target: np.ndarray, persistence: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - target
    persistence_error = persistence - target
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
    for name, group in (("arm", ARM), ("hand", HAND), ("full", slice(0, 28))):
        _pred_tv, _target_tv, closeness = group_tv(pred_native, target_native, group)
        result[f"{name}_native_tv_closeness"] = closeness
        result[f"{name}_native_tv_ratio"] = np.divide(_pred_tv, _target_tv, out=np.full_like(_pred_tv, np.nan), where=_target_tv > 1e-12)
    return result


def load_prediction(path: Path, *, require_event: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        bundle = {key: archive[key].copy() for key in archive.files}
    required = {"predictions_standardized", "targets_standardized", "persistence_standardized", "predictions_native", "targets_native", "persistence_native", "gate_values", "episode_index", "original_task_index"}
    if require_event:
        required.add("event_probability")
    if not required.issubset(bundle):
        raise RuntimeError(f"prediction archive missing keys: {sorted(required - set(bundle))}")
    if bundle["predictions_standardized"].shape != (614, STEPS, DIMS) or bundle["gate_values"].shape != (614, STEPS, 2) or (require_event and bundle["event_probability"].shape != (614, STEPS)):
        raise RuntimeError(f"prediction shape contract failed: {path}")
    event_finite = (not require_event) or np.isfinite(bundle["event_probability"]).all()
    if not np.isfinite(bundle["predictions_standardized"]).all() or not np.isfinite(bundle["predictions_native"]).all() or not np.isfinite(bundle["gate_values"]).all() or not event_finite:
        raise RuntimeError(f"non-finite prediction archive: {path}")
    return bundle


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    if len(labels) != len(scores) or positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * ordered) / positives)


def task_support(candidate: np.ndarray, baseline: np.ndarray, tasks: np.ndarray) -> dict[str, object]:
    values = baseline - candidate
    rows = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        rows.append({"task": int(task), "N": int(mask.sum()), "effect": float(values[mask].mean())})
    positive = [float(row["effect"]) for row in rows if float(row["effect"]) > 0]
    max_share = max(positive) / sum(positive) if positive and sum(positive) > 0 else 1.0
    return {"positive_tasks": len(positive), "total_tasks": len(rows), "max_positive_effect_share": float(max_share), "rows": rows}


def joint_support(candidate: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict[str, object]:
    candidate_error = np.abs(candidate - target)[:, :, ARM].mean(axis=1)
    baseline_error = np.abs(baseline - target)[:, :, ARM].mean(axis=1)
    values = baseline_error.mean(axis=0) - candidate_error.mean(axis=0)
    positive = values[values > 0]
    max_share = float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else 1.0
    return {"positive_joints": int((values > 0).sum()), "total_joints": int(len(values)), "max_positive_effect_share": max_share, "effect_by_joint": values.tolist()}


def relative_skill(candidate: np.ndarray, persistence: np.ndarray) -> float:
    return float((persistence.mean() - candidate.mean()) / persistence.mean())


def metric_row(cell: str, bundle: dict[str, np.ndarray], per: dict[str, np.ndarray], event_metrics: dict[str, Any], threshold: dict[str, float], tasks: np.ndarray) -> dict[str, object]:
    gate = bundle["gate_values"]
    q = bundle["event_probability"]
    gate_rows = {}
    for group, index in (("arm", 0), ("hand", 1)):
        values = gate[:, :, index].reshape(-1)
        gate_rows[f"gate_{group}_mean"] = float(values.mean())
        gate_rows[f"gate_{group}_collapse_fraction"] = float(np.mean((values < 0.05) | (values > 0.95)))
        gate_rows[f"gate_{group}_entropy"] = float((-(values * np.log(np.clip(values, 1e-7, 1 - 1e-7)) + (1 - values) * np.log(np.clip(1 - values, 1e-7, 1 - 1e-7)))).mean())
    q_values = q.reshape(-1)
    arm_motion = np.abs(np.concatenate((bundle["targets_standardized"][:, :1, ARM] - bundle["persistence_standardized"][:, :1, ARM], np.diff(bundle["targets_standardized"][:, :, ARM], axis=1)), axis=1)).mean(axis=2).mean(axis=1)
    high_mask = arm_motion >= threshold["COMMON_STEP_Q75"]
    high_skill = float((per["persistence_arm_mae"][high_mask].mean() - per["arm_mae"][high_mask].mean()) / per["persistence_arm_mae"][high_mask].mean()) if high_mask.any() else float("nan")
    return {
        "cell_id": cell,
        **CELL_FACTORS[cell],
        "seed": 20260925,
        "N_episode": 614,
        "arm_mae": float(per["arm_mae"].mean()),
        "hand_mae": float(per["hand_mae"].mean()),
        "full28_mae": float(per["full_mae"].mean()),
        "arm_early_mae": float(per["arm_early"].mean()),
        "arm_middle_mae": float(per["arm_middle"].mean()),
        "arm_late_mae": float(per["arm_late"].mean()),
        "arm_endpoint_mae": float(per["arm_endpoint"].mean()),
        "arm_first_difference": float(per["arm_first_difference"].mean()),
        "hand_first_difference": float(per["hand_first_difference"].mean()),
        "full_first_difference": float(per["full_first_difference"].mean()),
        "arm_native_tv_ratio": float(np.nanmean(per["arm_native_tv_ratio"])),
        "arm_native_tv_closeness": float(np.nanmean(per["arm_native_tv_closeness"])),
        "hand_native_tv_closeness": float(np.nanmean(per["hand_native_tv_closeness"])),
        "full_native_tv_closeness": float(np.nanmean(per["full_native_tv_closeness"])),
        "arm_skill_vs_persistence": relative_skill(per["arm_mae"], per["persistence_arm_mae"]),
        "hand_skill_vs_persistence": relative_skill(per["hand_mae"], per["persistence_hand_mae"]),
        "full28_skill_vs_persistence": relative_skill(per["full_mae"], per["persistence_full_mae"]),
        "arm_high_motion_skill": high_skill,
        "event_primary_auprc": float(event_metrics["primary_auprc"]["effect"]),
        "event_primary_auprc_ci_low": float(event_metrics["primary_auprc"]["ci_low"]),
        "event_primary_auprc_ci_high": float(event_metrics["primary_auprc"]["ci_high"]),
        "event_prevalence_baseline": float(event_metrics["primary_auprc_prevalence_baseline"]),
        "event_primary_auroc": float(event_metrics["primary_auroc"]),
        "event_primary_brier": float(event_metrics["primary_brier"]),
        "event_q_mean": float(q_values.mean()),
        "event_q_collapse_fraction": float(np.mean((q_values < 0.05) | (q_values > 0.95))),
        **gate_rows,
        "positive_task_support": int(task_support(per["persistence_arm_mae"], per["arm_mae"], tasks)["positive_tasks"]),
        "positive_joint_support": int(joint_support(bundle["predictions_standardized"], bundle["persistence_standardized"], bundle["targets_standardized"])["positive_joints"]),
    }


def main() -> int:
    rng = np.random.default_rng(ANALYSIS_SEED)
    threshold = {key: float(value) for key, value in json.loads((ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"].items()}
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        input_data = {key: archive[key].copy() for key in archive.files}
    dev_mask = input_data["split_code"] == 1
    if int(dev_mask.sum()) != 614:
        raise RuntimeError("development count contract failed")
    dev_ids = input_data["episode_index"][dev_mask].astype(np.int64)
    dev_tasks = input_data["original_task_index"][dev_mask].astype(np.int64)
    prior = load_prediction(PRIOR_PREDICTION_PATH, require_event=False)
    target = prior["targets_standardized"].astype(np.float64)
    persistence = prior["persistence_standardized"].astype(np.float64)
    target_native = prior["targets_native"].astype(np.float64)
    persistence_native = prior["persistence_native"].astype(np.float64)
    if not np.array_equal(prior["episode_index"], dev_ids):
        raise RuntimeError("prior/dev episode pairing failed")

    bundles: dict[str, dict[str, np.ndarray]] = {}
    per: dict[str, dict[str, np.ndarray]] = {}
    event_info: dict[str, dict[str, Any]] = {}
    main_rows: list[dict[str, object]] = []
    per_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    for cell in CELL_ORDER:
        run = ROOT / "runs" / cell
        marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("optimizer_steps") != 1720 or marker.get("prediction_shape") != [614, STEPS, DIMS]:
            raise RuntimeError(f"run acceptance marker failed for {cell}")
        bundle = load_prediction(run / "predictions_best.npz")
        if not np.array_equal(bundle["episode_index"], dev_ids) or not np.array_equal(bundle["original_task_index"], dev_tasks):
            raise RuntimeError(f"pairing failed for {cell}")
        if not np.array_equal(bundle["targets_standardized"], target) or not np.array_equal(bundle["persistence_standardized"], persistence):
            raise RuntimeError(f"target/persistence mismatch for {cell}")
        metrics = json.loads((run / "RUN_METRICS.json").read_text(encoding="utf-8"))
        saved_event = metrics["event"]
        primary_scores = bundle["event_probability"].mean(axis=1) if CELL_FACTORS[cell]["event_mode"] == "U" else (bundle["event_probability"][:, 0] if CELL_FACTORS[cell]["event_mode"] == "W" else bundle["event_probability"].max(axis=1))
        # Recompute the point AP independently; the stored interval was produced
        # with the locked 10,000 episode bootstrap inside the training runner.
        dev_step_motion = np.abs(np.concatenate((target[:, :1, HAND] - persistence[:, :1, HAND], np.diff(target[:, :, HAND], axis=1)), axis=1)).mean(axis=2)
        dev_step_labels = dev_step_motion >= threshold["HAND_STEP_Q75"]
        dev_window_labels = dev_step_motion.mean(axis=1) >= threshold["HAND_EPISODE_MEAN_Q75"]
        point_ap = average_precision(dev_window_labels, primary_scores)
        if not np.isclose(point_ap, float(saved_event["primary_auprc"]["effect"]), rtol=0, atol=2e-6):
            raise RuntimeError(f"event AP point mismatch for {cell}: {point_ap} vs {saved_event['primary_auprc']['effect']}")
        bundles[cell] = bundle
        per[cell] = per_metrics(bundle["predictions_standardized"].astype(np.float64), target, persistence, bundle["predictions_native"].astype(np.float64), target_native, persistence_native)
        event_info[cell] = saved_event
        main_rows.append(metric_row(cell, bundle, per[cell], saved_event, threshold, dev_tasks))
        q = bundle["event_probability"]
        for time_index in range(STEPS):
            event_rows.append({"cell_id": cell, "time_index": time_index, "mean_event_probability": float(q[:, time_index].mean()), "observed_step_prevalence": float(dev_step_labels[:, time_index].mean()), "event_mode": CELL_FACTORS[cell]["event_mode"]})
        for task in sorted(set(dev_tasks.tolist())):
            mask = dev_tasks == task
            event_rows.append({"cell_id": cell, "time_index": "task", "task": int(task), "mean_event_probability": float(q[mask].mean()), "observed_step_prevalence": float(dev_step_labels[mask].mean()), "event_mode": CELL_FACTORS[cell]["event_mode"]})
        support = task_support(per[cell]["arm_mae"], per[cell]["persistence_arm_mae"], dev_tasks)
        joint = joint_support(bundle["predictions_standardized"], persistence, target)
        support_rows.append({"cell_id": cell, "task_positive": support["positive_tasks"], "task_total": support["total_tasks"], "task_max_positive_share": support["max_positive_effect_share"], "joint_positive": joint["positive_joints"], "joint_total": joint["total_joints"], "joint_max_positive_share": joint["max_positive_effect_share"]})
        for index, episode in enumerate(dev_ids):
            per_rows.append({"cell_id": cell, "episode_index": int(episode), "original_task_index": int(dev_tasks[index]), "arm_mae": float(per[cell]["arm_mae"][index]), "hand_mae": float(per[cell]["hand_mae"][index]), "full28_mae": float(per[cell]["full_mae"][index]), "arm_first_difference": float(per[cell]["arm_first_difference"][index]), "full_first_difference": float(per[cell]["full_first_difference"][index]), "arm_native_tv_closeness": float(per[cell]["arm_native_tv_closeness"][index]), "hand_native_tv_closeness": float(per[cell]["hand_native_tv_closeness"][index]), "full_native_tv_closeness": float(per[cell]["full_native_tv_closeness"][index]), "gate_arm": float(bundle["gate_values"][index, :, 0].mean()), "gate_hand": float(bundle["gate_values"][index, :, 1].mean()), "event_q_mean": float(bundle["event_probability"][index].mean()), "event_q_max": float(bundle["event_probability"][index].max()), "event_window_label": int(dev_window_labels[index]), "event_step_label_fraction": float(dev_step_labels[index].mean())})

    # Persistence and both derived controls are reconstructed from the same
    # paired development arrays; no new fit or target access is involved.
    hybrid_arm = persistence.copy(); hybrid_arm[:, :, ARM] = prior["predictions_standardized"][:, :, ARM]
    hybrid_hand = persistence.copy(); hybrid_hand[:, :, HAND] = prior["predictions_standardized"][:, :, HAND]
    hybrid_native_arm = persistence_native.copy(); hybrid_native_arm[:, :, ARM] = prior["predictions_native"][:, :, ARM]
    hybrid_native_hand = persistence_native.copy(); hybrid_native_hand[:, :, HAND] = prior["predictions_native"][:, :, HAND]
    control_bundles = {"PERSISTENCE": (persistence, persistence_native), "HYB-A_NN-H_P": (hybrid_arm, hybrid_native_arm), "HYB-A_P-H_NN": (hybrid_hand, hybrid_native_hand), "E-F-SP-G": (prior["predictions_standardized"].astype(np.float64), prior["predictions_native"].astype(np.float64))}
    control_per = {name: per_metrics(pred, target, persistence, pred_native, target_native, persistence_native) for name, (pred, pred_native) in control_bundles.items()}
    hybrid_rows: list[dict[str, object]] = []
    hybrid_effect_rows: list[dict[str, object]] = []
    for name in control_bundles:
        item = {"model": name, "outcome_exposed_derived_control": name.startswith("HYB-"), "arm_mae": float(control_per[name]["arm_mae"].mean()), "hand_mae": float(control_per[name]["hand_mae"].mean()), "full28_mae": float(control_per[name]["full_mae"].mean()), "arm_first_difference": float(control_per[name]["arm_first_difference"].mean()), "full_first_difference": float(control_per[name]["full_first_difference"].mean()), "arm_native_tv_closeness": float(np.nanmean(control_per[name]["arm_native_tv_closeness"])), "full_native_tv_closeness": float(np.nanmean(control_per[name]["full_native_tv_closeness"])), "full28_skill_vs_persistence": relative_skill(control_per[name]["full_mae"], control_per[name]["persistence_full_mae"])}
        hybrid_rows.append(item)
    for candidate_name in ("HYB-A_NN-H_P", "HYB-A_P-H_NN"):
        for baseline_name in ("PERSISTENCE", "E-F-SP-G"):
            for metric_name, key in (("Arm-14 MAE", "arm_mae"), ("Hand-14 MAE", "hand_mae"), ("Full-28 MAE", "full_mae"), ("Arm first-difference", "arm_first_difference"), ("Full first-difference", "full_first_difference"), ("Arm TV closeness", "arm_native_tv_closeness")):
                item = effect(control_per[candidate_name][key], control_per[baseline_name][key], rng)
                hybrid_effect_rows.append({"candidate": candidate_name, "baseline": baseline_name, "metric": metric_name, **item, "outcome_exposed_derived_control": True})

    persistence_per = control_per["PERSISTENCE"]
    effects: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for cell in CELL_ORDER:
        mode = CELL_FACTORS[cell]["event_mode"]
        input_branch = CELL_FACTORS[cell]["input"]
        baseline_u = "H-S-U" if input_branch == "STATE" else "H-F-U"
        comparisons = {
            "Arm_vs_Persistence": (per[cell]["arm_mae"], persistence_per["arm_mae"]),
            "Hand_vs_Persistence": (per[cell]["hand_mae"], persistence_per["hand_mae"]),
            "Full28_vs_Persistence": (per[cell]["full_mae"], persistence_per["full_mae"]),
            "Arm_first_difference_vs_U": (per[cell]["arm_first_difference"], per[baseline_u]["arm_first_difference"]),
            "Arm_TV_closeness_vs_U": (per[cell]["arm_native_tv_closeness"], per[baseline_u]["arm_native_tv_closeness"]),
            "Arm_vs_same_input_U": (per[cell]["arm_mae"], per[baseline_u]["arm_mae"]),
            "Full28_vs_HYB-A_NN-H_P": (per[cell]["full_mae"], control_per["HYB-A_NN-H_P"]["full_mae"]),
        }
        fusion_same_mode = None
        if input_branch == "FUSION":
            state_cell = f"H-S-{mode}"
            fusion_same_mode = state_cell
            comparisons["Fusion_Arm_vs_same_mode_STATE"] = (per[cell]["arm_mae"], per[state_cell]["arm_mae"])
        row_effects: dict[str, dict[str, Any]] = {}
        for label, (cand_values, base_values) in comparisons.items():
            item = effect(cand_values, base_values, rng)
            row_effects[label] = item
            effects.append({"cell_id": cell, "comparison": label, **item, "effect_definition": "baseline MAE or closeness minus candidate; positive is improvement", "episode_unit": True})
        arm_persist = row_effects["Arm_vs_Persistence"]
        arm_u = row_effects["Arm_vs_same_input_U"]
        hand_persist = row_effects["Hand_vs_Persistence"]
        full_persist = row_effects["Full28_vs_Persistence"]
        fd_u = row_effects["Arm_first_difference_vs_U"]
        tv_u = row_effects["Arm_TV_closeness_vs_U"]
        hybrid_effect = row_effects["Full28_vs_HYB-A_NN-H_P"]
        q_summary = event_info[cell]
        q_collapse = float(np.mean((bundles[cell]["event_probability"] < 0.05) | (bundles[cell]["event_probability"] > 0.95)))
        high_motion = np.abs(np.concatenate((target[:, :1, ARM] - persistence[:, :1, ARM], np.diff(target[:, :, ARM], axis=1)), axis=1)).mean(axis=2).mean(axis=1) >= threshold["COMMON_STEP_Q75"]
        late_skill = float((persistence_per["arm_late"].mean() - per[cell]["arm_late"].mean()) / persistence_per["arm_late"].mean())
        endpoint_skill = float((persistence_per["arm_endpoint"].mean() - per[cell]["arm_endpoint"].mean()) / persistence_per["arm_endpoint"].mean())
        high_skill = float((persistence_per["arm_mae"][high_motion].mean() - per[cell]["arm_mae"][high_motion].mean()) / persistence_per["arm_mae"][high_motion].mean()) if high_motion.any() else float("nan")
        task = task_support(per[cell]["arm_mae"], persistence_per["arm_mae"], dev_tasks)
        joint = joint_support(bundles[cell]["predictions_standardized"], persistence, target)
        checks = {
            "arm_ci_beats_persistence_by_2pct": arm_persist["ci_low"] >= 0.02 * float(persistence_per["arm_mae"].mean()),
            "arm_not_worse_than_same_input_U": mode == "U" or arm_u["ci_low"] >= 0.0,
            "hand_ci_beats_persistence": hand_persist["ci_low"] > 0.0,
            "full28_ci_beats_persistence": full_persist["ci_low"] > 0.0,
            "arm_first_difference_improves_vs_U": mode == "U" or fd_u["ci_low"] > 0.0,
            "arm_tv_closeness_improves_vs_U": mode == "U" or tv_u["ci_low"] > 0.0,
            "high_motion_not_over_1pct_reverse": high_skill >= -0.01,
            "late_not_over_1pct_reverse": late_skill >= -0.01,
            "endpoint_not_over_1pct_reverse": endpoint_skill >= -0.01,
            "event_auprc_ci_above_prevalence": float(q_summary["primary_auprc"]["ci_low"]) > float(q_summary["primary_auprc_prevalence_baseline"]),
            "event_q_not_collapsed": q_collapse <= 0.95,
            "task_support_not_single": int(task["positive_tasks"]) >= 3 and float(task["max_positive_effect_share"]) <= 0.5,
            "joint_support_not_single": int(joint["positive_joints"]) >= 3 and float(joint["max_positive_effect_share"]) <= 0.5,
            "full28_beats_zero_training_hybrid": hybrid_effect["ci_low"] > 0.0,
            "fusion_beats_same_mode_state": fusion_same_mode is None or row_effects["Fusion_Arm_vs_same_mode_STATE"]["ci_low"] > 0.0,
        }
        candidate_gate = bool(mode in {"W", "T"} and all(checks.values()))
        gate_rows.append({"cell_id": cell, "same_input_U": baseline_u, "event_mode": mode, "fusion_state_counterpart": fusion_same_mode or "", "gate_pass": candidate_gate, "checks": checks, "arm_high_motion_skill": high_skill, "arm_late_skill": late_skill, "arm_endpoint_skill": endpoint_skill, "event_q_collapse_fraction": q_collapse, "task_support": task, "joint_support": joint, "effects": row_effects})

    passed = [row["cell_id"] for row in gate_rows if row["gate_pass"]]
    priority = ["H-F-T", "H-F-W", "H-S-T", "H-S-W"]
    selected = next((cell for cell in priority if cell in passed), None)
    decision = {"status": "PASS", "seed": 20260925, "single_training_seed": True, "N_seed": 1, "best_arm_cell": min(main_rows, key=lambda row: float(row["arm_mae"]))["cell_id"], "candidate_gate": "pass" if selected else "fail", "selected_candidate": selected, "candidate_priority": priority, "gate_rows": gate_rows, "matrix_accepted": "6/6", "bootstrap_replicates": BOOTSTRAP, "hierarchical_bootstrap_seed_uncertainty": "degenerate with one training seed; CI is episode-clustered only", "final_test_access": 0, "frozen_candidate_changed": False, "final_protocol_changed": False, "new_training_after_matrix": False}
    write_csv(ROOT / "MATRIX_RESULTS.csv", main_rows)
    write_json(ROOT / "MATRIX_RESULTS.json", {"status": "PASS", "rows": main_rows, "bootstrap_replicates": BOOTSTRAP, "episode_unit": True, "final_test_access": 0})
    write_csv(ROOT / "MATRIX_PER_EPISODE.csv", per_rows)
    write_csv(ROOT / "EVENT_DIAGNOSTICS_ALL.csv", event_rows)
    write_csv(ROOT / "SUPPORT_DIAGNOSTICS.csv", support_rows)
    write_json(ROOT / "EVENT_DIAGNOSTICS_ALL.json", {"status": "PASS", "rows": event_info, "prevalence_baseline_definition": "development window-event prevalence; labels from train-only thresholds", "episode_clustered": True})
    write_json(ROOT / "HIERARCHICAL_BOOTSTRAP.json", {"status": "PASS", "N_seed": 1, "N_episode": 614, "replicates": BOOTSTRAP, "seed_resampling": "one observed training seed is resampled; therefore no training-seed uncertainty is estimated", "effects": effects, "derived_hybrid_effects": hybrid_effect_rows})
    write_csv(ROOT / "MATRIX_EFFECTS.csv", effects)
    write_json(ROOT / "MATRIX_EFFECTS.json", {"status": "PASS", "rows": effects, "bootstrap_replicates": BOOTSTRAP, "effect_sign": "baseline minus candidate; positive improves"})
    write_json(ROOT / "CANDIDATE_GATE_EVALUATIONS.json", decision)
    write_json(ROOT / "HYBRID_MATRIX_SUMMARY.json", {"status": "PASS", "rows": hybrid_rows, "effects": hybrid_effect_rows, "source": str(PRIOR_PREDICTION_PATH), "source_sha256": digest(PRIOR_PREDICTION_PATH), "outcome_exposed": True, "final_test_access": 0})
    write_csv(ROOT / "HYBRID_MATRIX_SUMMARY.csv", hybrid_rows)

    # A compact, outcome-labelled report is easier to review than raw JSON.
    table = "\n".join(f"| {row['cell_id']} | {row['event_mode']} | {float(row['arm_mae']):.6f} | {float(row['hand_mae']):.6f} | {float(row['full28_mae']):.6f} | {float(row['event_primary_auprc']):.4f} | {'PASS' if row['cell_id'] in passed else 'FAIL'} |" for row in main_rows)
    (ROOT / "EXPERIMENT_DECISION.zh-CN.md").write_text(f"""# Hand motion-event fallback exploratory decision

状态：`single_seed_exploratory_complete`；Stage0=`6/6`；matrix=`6/6`；seed=`20260925`。

| Cell | Event | Arm MAE | Hand MAE | Full-28 MAE | primary AUPRC | candidate gate |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 结论

候选门禁：`{'PASS: ' + selected if selected else 'FAIL'}`。当前没有通过完整门禁的 W/T cell 时，不能生成或执行 multi-seed follow-up；本轮结果仅是单 seed、development、outcome-exposed 的机制诊断。

Phase A 的原始“Hand 更低运动”解释没有得到支持；它只通过 C 路径进入本矩阵。所有新 cell 都必须同时超过 Persistence、同输入 U、以及零训练 `HYB-A_NN-H_P` 才能宣称 learned event fallback 有增量价值。

Single-seed hierarchical bootstrap 的 seed 层是退化的：它没有估计 training-seed uncertainty。不能把本轮 AUPRC 或 MAE 解释成跨 seed 稳定性，也不能替代 held-out/final evaluation。

final-test access=`0`；冻结候选未改变；final protocol 未改变；没有 LoRA、encoder、raw-video 或后续自动训练。
""", encoding="utf-8")

    # Deterministic trajectory and metric figures; representative IDs come from
    # the predeclared V2 manifest, never selected using this matrix outcome.
    figure_status: dict[str, object] = {"status": "PASS", "files": []}
    try:
        import matplotlib.pyplot as plt

        labels = ["Persistence"] + list(CELL_ORDER)
        full_values = [float(control_per["PERSISTENCE"]["full_mae"].mean())] + [float(per[cell]["full_mae"].mean()) for cell in CELL_ORDER]
        arm_values = [float(control_per["PERSISTENCE"]["arm_mae"].mean())] + [float(per[cell]["arm_mae"].mean()) for cell in CELL_ORDER]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        axes[0].bar(labels, arm_values, color=["#8c8c8c"] + ["#377eb8" if c.startswith("H-S") else "#4daf4a" for c in CELL_ORDER])
        axes[0].set_title("Arm-14 development MAE")
        axes[0].set_ylabel("standardized MAE")
        axes[0].tick_params(axis="x", rotation=35)
        axes[1].bar(labels, full_values, color=["#8c8c8c"] + ["#377eb8" if c.startswith("H-S") else "#4daf4a" for c in CELL_ORDER])
        axes[1].axhline(float(control_per["HYB-A_NN-H_P"]["full_mae"].mean()), color="#e41a1c", linestyle="--", label="zero-training HYB-A_NN-H_P")
        axes[1].set_title("Full-28 development MAE")
        axes[1].set_ylabel("standardized MAE")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle("JY-02 Hand event-gate matrix (single seed; development only)")
        fig.tight_layout()
        path = ROOT / "figures/MATRIX_ARM_FULL_MAE.png"; fig.savefig(path, dpi=160); plt.close(fig); figure_status["files"].append(str(path))

        fig, ax = plt.subplots(figsize=(9, 4.8))
        for cell in CELL_ORDER:
            ax.plot(np.arange(STEPS), bundles[cell]["event_probability"].mean(axis=0), label=cell)
        ax.plot(np.arange(STEPS), dev_step_labels.mean(axis=0), color="black", linewidth=2, linestyle="--", label="observed Hand event prevalence")
        ax.set_xlabel("forecast step"); ax.set_ylabel("probability / prevalence"); ax.set_ylim(0, 1); ax.set_title("Hand event probability by horizon (development only)"); ax.legend(ncol=2, frameon=False, fontsize=8); fig.tight_layout()
        path = ROOT / "figures/EVENT_PROBABILITY_BY_TIME.png"; fig.savefig(path, dpi=160); plt.close(fig); figure_status["files"].append(str(path))

        manifest_ids: list[int] = []
        if MANIFEST_PATH.is_file():
            with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("episode_index", "episode_id", "source_episode_index"):
                        if key in row and row[key]:
                            try: manifest_ids.append(int(row[key]))
                            except ValueError: pass
                            break
        manifest_ids = [value for value in manifest_ids if value in set(dev_ids.tolist())][:3]
        if not manifest_ids:
            manifest_ids = [int(value) for value in dev_ids[:3]]
        indices = [int(np.where(dev_ids == value)[0][0]) for value in manifest_ids]
        fig, axes = plt.subplots(len(indices), 1, figsize=(9, max(3.0, 2.5 * len(indices))), squeeze=False)
        for row_index, (episode_index, data_index) in enumerate(zip(manifest_ids, indices)):
            ax = axes[row_index, 0]
            ax.plot(np.arange(STEPS), target[data_index, :, ARM].mean(axis=1), color="black", linewidth=2, label="target Arm mean")
            ax.plot(np.arange(STEPS), persistence[data_index, :, ARM].mean(axis=1), color="#8c8c8c", linestyle=":", label="Persistence")
            for cell, color in (("H-F-U", "#4daf4a"), ("H-F-W", "#377eb8"), ("H-F-T", "#984ea3")):
                ax.plot(np.arange(STEPS), bundles[cell]["predictions_standardized"][data_index, :, ARM].mean(axis=1), color=color, alpha=0.85, label=cell)
            ax.set_title(f"predeclared development episode {episode_index}"); ax.set_ylabel("Arm mean"); ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("forecast step"); axes[0, 0].legend(ncol=4, frameon=False, fontsize=8); fig.suptitle("Representative command trajectories; not physical Cartesian trajectories"); fig.tight_layout()
        path = ROOT / "figures/REPRESENTATIVE_ARM_TRAJECTORIES.png"; fig.savefig(path, dpi=160); plt.close(fig); figure_status["files"].append(str(path))
    except Exception as exc:
        figure_status = {"status": "FAIL", "files": [], "error": f"{type(exc).__name__}: {exc}"}
    write_json(ROOT / "FIGURE_GENERATION_MATRIX.json", figure_status)
    write_json(ROOT / "MATRIX_ANALYSIS_PROVENANCE.json", {"status": "PASS", "analysis_script": str(Path(__file__)), "analysis_seed": ANALYSIS_SEED, "bootstrap_replicates": BOOTSTRAP, "episode_unit": True, "hierarchical_seed_count": 1, "seed_uncertainty_estimated": False, "input_sha256": digest(INPUT_PATH), "normalizer_sha256": digest(NORMALIZER_PATH), "prior_prediction_sha256": digest(PRIOR_PREDICTION_PATH), "final_test_access": 0, "frozen_candidate_changed": False, "final_protocol_changed": False})
    print(json.dumps({"status": "PASS", "matrix_accepted": "6/6", "best_arm_cell": decision["best_arm_cell"], "candidate_gate": decision["candidate_gate"], "selected_candidate": selected, "figure_status": figure_status["status"], "hyb_a_nn_h_p_full28": float(control_per["HYB-A_NN-H_P"]["full_mae"].mean())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
