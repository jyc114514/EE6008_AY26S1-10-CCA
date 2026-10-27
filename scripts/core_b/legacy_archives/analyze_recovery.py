#!/usr/bin/env python3
"""CPU-only six-cell recovery analysis using the locked B definitions."""

from __future__ import annotations
import os

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
HISTORY_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
INPUT = HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
BOOTSTRAP = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/paired_episode_bootstrap_indices.npy"
MOTION_STRATA = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_STRATA_MANIFEST.csv"
SOURCE_ANALYSIS = HISTORY_ROOT / "code/run_dev_inference_and_analysis.py"
HAND = 14
STEPS = 30
BOOTSTRAP_N = 10000
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")
MODELS = ("PERSISTENCE", "FIXED_HYBRID", *CELLS)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def json_default(value: object) -> object:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)


def import_source() -> Any:
    code_dir = str(SOURCE_ANALYSIS.parent)
    if code_dir not in sys.path: sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("locked_core_b_analysis_recovery", SOURCE_ANALYSIS)
    if spec is None or spec.loader is None: raise RuntimeError("source analysis import failed")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def finite_ratio(candidate: np.ndarray, control: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(candidate, dtype=np.float64)[indices].mean(axis=1)
    denominator = np.asarray(control, dtype=np.float64)[indices].mean(axis=1)
    values = numerator / denominator - 1.0
    return {"relative_change_candidate_vs_control": float(np.mean(candidate) / np.mean(control) - 1.0), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": int(len(indices)), "N_episode": int(len(candidate))}


def subset_indices(mask: np.ndarray, seed: int) -> np.ndarray:
    n = int(mask.sum())
    if n <= 0: raise RuntimeError("empty guardrail stratum")
    return np.random.default_rng(seed).integers(0, n, size=(BOOTSTRAP_N, n), dtype=np.int64)


def effect(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    source = import_source()
    return source.bootstrap_stats(control, candidate, indices, seed)


def add_holm(rows: list[dict[str, object]]) -> None:
    source = import_source()
    source.holm_adjust(rows, p_key="p_signflip")


def load_motion(dev_ids: np.ndarray) -> np.ndarray:
    with MOTION_STRATA.open(encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    by_id = {int(row["episode_id"]): row["motion_stratum"] for row in values}
    result = np.asarray([by_id.get(int(value), "") for value in dev_ids], dtype=object)
    if np.any(result == ""): raise RuntimeError("motion strata do not cover dev IDs")
    return result


def load_predictions(cell: str) -> dict[str, np.ndarray]:
    with np.load(RECOVERY / f"predictions/{cell}/shard_000.npz", allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def metric_row(model: str, metric: dict[str, np.ndarray], motion: np.ndarray, tasks: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {"model": model, "N_episode": 614, "task_count": int(len(np.unique(tasks))), "motion_low_N": int(np.sum(motion == "low")), "motion_mid_N": int(np.sum(motion == "mid")), "motion_high_N": int(np.sum(motion == "high"))}
    keys = ("hand_mae_std", "hand_mae_native", "full_mae_std", "full_mae_native", "full_rmse_native", "hand_fd_native", "hand_fd_std", "full_fd_native", "hand_anchor_to_first_native", "hand_tv", "target_hand_tv", "hand_tv_ratio", "hand_tv_closeness", "hand_direction_agreement", "early_hand_mae_std", "middle_hand_mae_std", "late_hand_mae_std", "endpoint_hand_mae_std", "early_hand_mae_native", "middle_hand_mae_native", "late_hand_mae_native", "endpoint_hand_mae_native")
    for key in keys: row[key] = float(np.nanmean(metric[key]))
    row["native_fd_relative_vs_persistence"] = np.nan
    return row


def one_seed_resource_costs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    log_root = RECOVERY / "logs"
    for path in sorted(log_root.glob("watcher_B-RECOVERY-*.json")):
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        started = data.get("started_utc") or data.get("launch_utc")
        completed = data.get("completed_utc")
        hours = np.nan
        if started and completed:
            try:
                a = datetime.fromisoformat(str(started).replace("Z", "+00:00")); b = datetime.fromisoformat(str(completed).replace("Z", "+00:00")); hours = (b - a).total_seconds() / 3600.0
            except Exception: pass
        rows.append({"job_id": data.get("job_id", path.stem), "status": data.get("status"), "physical_gpu": data.get("physical_gpu", data.get("selected_gpu")), "wall_gpu_hours": hours, "kind": "inference" if "INFER" in path.name else "replay", "retry": "RETRY1" in path.name, "exit_code": data.get("exit_code"), "final_test_access": 0})
    return rows


def main() -> int:
    source = import_source()
    data, stats = source.load_data()
    dev = np.flatnonzero(data["split_code"] == 1)
    if len(dev) != 614: raise RuntimeError("dev row count failed")
    dev_ids = data["episode_index"][dev].astype(np.int64)
    tasks = data["original_task_index"][dev].astype(np.int64)
    target_native = data["targets_native"][dev].astype(np.float32)
    target_std = data["targets_standardized"][dev].astype(np.float32)
    persistence_native = data["persistence_native"][dev].astype(np.float32)
    persistence_std = data["persistence_standardized"][dev].astype(np.float32)
    anchor = persistence_native[:, 0, :HAND]
    with np.load(LOCKED_ARM, allow_pickle=False) as arm:
        if not np.array_equal(arm["episode_index"], dev_ids): raise RuntimeError("Arm/dev ID seam failed")
        arm_native = arm["predictions_native"][..., 14:].astype(np.float32)
        arm_std = arm["predictions_standardized"][..., 14:].astype(np.float32)
    motion = load_motion(dev_ids)
    bootstrap = np.load(BOOTSTRAP, allow_pickle=False)
    if bootstrap.shape != (BOOTSTRAP_N, 614): raise RuntimeError(f"bootstrap shape failed: {bootstrap.shape}")
    metrics: dict[str, dict[str, np.ndarray]] = {}
    persistence_full_native = persistence_native.copy(); persistence_full_std = persistence_std.copy()
    metrics["PERSISTENCE"] = source.metric_arrays(persistence_full_native, persistence_full_std, target_native, target_std, anchor)
    fixed_native = np.concatenate((persistence_native[:, :, :HAND], arm_native), axis=2)
    fixed_std = np.concatenate((persistence_std[:, :, :HAND], arm_std), axis=2)
    metrics["FIXED_HYBRID"] = source.metric_arrays(fixed_native, fixed_std, target_native, target_std, anchor)
    for cell in CELLS:
        pred = load_predictions(cell)
        if not np.array_equal(pred["episode_index"], dev_ids): raise RuntimeError(f"prediction ID seam failed for {cell}")
        metrics[cell] = source.metric_arrays(pred["predictions_full_native"], pred["predictions_full_standardized"], target_native, target_std, anchor)

    analysis = RECOVERY / "analysis"
    metric_rows = [metric_row(model, metrics[model], motion, tasks) for model in MODELS]
    persistence_fd = float(np.mean(metrics["PERSISTENCE"]["hand_fd_native"]))
    for row in metric_rows:
        if row["model"] != "PERSISTENCE": row["native_fd_relative_vs_persistence"] = float(row["hand_fd_native"] / persistence_fd - 1.0)
    write_csv(analysis / "SIX_CELL_RESULTS.csv", metric_rows)

    contrasts = [("D-LORA4_vs_D-FRZ", "FT-D-FRZ", "FT-D-LORA4"), ("S-LORA4_vs_S-FRZ", "FT-S-FRZ", "FT-S-LORA4"), ("D-DENSE4_vs_D-FRZ", "FT-D-FRZ", "FT-D-DENSE4"), ("S-DENSE4_vs_S-FRZ", "FT-S-FRZ", "FT-S-DENSE4"), ("D-DENSE4_vs_D-LORA4", "FT-D-LORA4", "FT-D-DENSE4"), ("D-FRZ_vs_S-FRZ", "FT-S-FRZ", "FT-D-FRZ"), ("D-LORA4_vs_S-LORA4", "FT-S-LORA4", "FT-D-LORA4"), ("D-DENSE4_vs_S-DENSE4", "FT-S-DENSE4", "FT-D-DENSE4")]
    paired: list[dict[str, object]] = []
    for name, control, candidate in contrasts:
        for metric_name in ("hand_mae_std", "hand_fd_native"):
            value = effect(metrics[control][metric_name], metrics[candidate][metric_name], bootstrap, 20269000 + len(paired))
            paired.append({"contrast": name, "control": control, "candidate": candidate, "metric": metric_name, **value, "final_test_access": 0})
    add_holm(paired)
    write_csv(analysis / "PAIRED_EFFECTS.csv", paired)

    def candidate_gate(cell: str, matched: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        details: list[dict[str, object]] = []
        hand_match = effect(metrics[matched]["hand_mae_std"], metrics[cell]["hand_mae_std"], bootstrap, 20269100)
        full_match = effect(metrics["FIXED_HYBRID"]["full_mae_std"], metrics[cell]["full_mae_std"], bootstrap, 20269101)
        fd_p = finite_ratio(metrics[cell]["hand_fd_native"], metrics["PERSISTENCE"]["hand_fd_native"], bootstrap)
        fd_f = finite_ratio(metrics[cell]["hand_fd_native"], metrics[matched]["hand_fd_native"], bootstrap)
        details.extend([
            {"cell": cell, "gate": "hand_vs_matched_frz_ci_low_gt_zero", "value": hand_match["effect_control_minus_candidate"], "ci_low": hand_match["ci_low"], "ci_high": hand_match["ci_high"], "threshold": 0.0, "pass": bool(hand_match["ci_low"] > 0)},
            {"cell": cell, "gate": "full_vs_fixed_hybrid_ci_low_gt_zero", "value": full_match["effect_control_minus_candidate"], "ci_low": full_match["ci_low"], "ci_high": full_match["ci_high"], "threshold": 0.0, "pass": bool(full_match["ci_low"] > 0)},
            {"cell": cell, "gate": "native_fd_vs_persistence_relative_le_1pct", "value": fd_p["relative_change_candidate_vs_control"], "ci_low": fd_p["ci_low"], "ci_high": fd_p["ci_high"], "threshold": 0.01, "pass": bool(fd_p["relative_change_candidate_vs_control"] <= 0.01 and fd_p["ci_high"] <= 0.01)},
            {"cell": cell, "gate": "native_fd_vs_matched_frz_ci_high_le_1pct", "value": fd_f["relative_change_candidate_vs_control"], "ci_low": fd_f["ci_low"], "ci_high": fd_f["ci_high"], "threshold": 0.01, "pass": bool(fd_f["ci_high"] <= 0.01)},
        ])
        guard_specs = [("standardized_fd_vs_persistence", "hand_fd_std", "PERSISTENCE", None), ("tv_closeness_vs_persistence", "hand_tv_closeness", "PERSISTENCE", None), ("horizon_early_hand_mae_vs_persistence", "early_hand_mae_std", "PERSISTENCE", None), ("horizon_middle_hand_mae_vs_persistence", "middle_hand_mae_std", "PERSISTENCE", None), ("horizon_late_hand_mae_vs_persistence", "late_hand_mae_std", "PERSISTENCE", None), ("horizon_endpoint_hand_mae_vs_persistence", "endpoint_hand_mae_std", "PERSISTENCE", None), ("motion_low_hand_mae_vs_persistence", "hand_mae_std", "PERSISTENCE", "low"), ("motion_mid_hand_mae_vs_persistence", "hand_mae_std", "PERSISTENCE", "mid"), ("motion_high_hand_mae_vs_persistence", "hand_mae_std", "PERSISTENCE", "high")]
        for index, (gate_name, key, control_name, stratum) in enumerate(guard_specs):
            mask = np.ones(614, dtype=bool) if stratum is None else motion == stratum
            finite_mask = mask & np.isfinite(metrics[cell][key]) & np.isfinite(metrics[control_name][key])
            candidate_values = metrics[cell][key][finite_mask]
            control_values = metrics[control_name][key][finite_mask]
            # The source TV definition intentionally yields NaN when target
            # TV is zero.  Preserve the source finite-episode convention and
            # create a within-stratum paired bootstrap for the reduced set.
            indices = bootstrap if bool(finite_mask.all()) else subset_indices(finite_mask, 20269200 + index)
            value = finite_ratio(candidate_values, control_values, indices)
            details.append({"cell": cell, "gate": gate_name, "value": value["relative_change_candidate_vs_control"], "ci_low": value["ci_low"], "ci_high": value["ci_high"], "threshold": 0.01, "N_episode": int(mask.sum()), "pass": bool(value["relative_change_candidate_vs_control"] <= 0.01 and value["ci_high"] <= 0.01)})
        support = bool(np.isfinite(metrics[cell]["per_joint_hand_mae_std"]).all() and np.isfinite(metrics[cell]["per_joint_hand_fd_native"]).all())
        details.append({"cell": cell, "gate": "support_all_614_episodes_14_joints_3_motion_strata", "value": int(support), "ci_low": np.nan, "ci_high": np.nan, "threshold": 1, "pass": support})
        decision = {"cell": cell, "matched_frz": matched, "all_gates_pass": bool(all(bool(row["pass"]) for row in details)), "hand_vs_matched_frz": hand_match, "full_vs_fixed_hybrid": full_match, "native_fd_vs_persistence": fd_p, "native_fd_vs_matched_frz": fd_f, "final_test_access": 0}
        return details, decision

    all_gate_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for cell, matched in (("FT-D-DENSE4", "FT-D-FRZ"), ("FT-D-LORA4", "FT-D-FRZ"), ("FT-S-DENSE4", "FT-S-FRZ"), ("FT-S-LORA4", "FT-S-FRZ")):
        detail, decision = candidate_gate(cell, matched); all_gate_rows.extend(detail); decisions.append(decision)
    write_csv(analysis / "LOW_MOTION_GUARDRAILS.csv", [row for row in all_gate_rows if "motion_" in str(row["gate"]) or "standardized_fd" in str(row["gate"]) or "tv_closeness" in str(row["gate"]) or "horizon_" in str(row["gate"])])
    write_csv(analysis / "CANDIDATE_GATE_DETAIL.csv", all_gate_rows)

    interaction_rows: list[dict[str, object]] = []
    for adaptation, dynamic, static in (("LORA4", "FT-D-LORA4", "FT-S-LORA4"), ("DENSE4", "FT-D-DENSE4", "FT-S-DENSE4")):
        for metric_name in ("hand_mae_std", "hand_fd_native"):
            values = (metrics["FT-D-FRZ"][metric_name] - metrics[dynamic][metric_name]) - (metrics["FT-S-FRZ"][metric_name] - metrics[static][metric_name])
            indices = bootstrap
            estimates = values[indices].mean(axis=1)
            rng = np.random.default_rng(20269300 + len(interaction_rows)); signs = rng.choice(np.asarray([-1.0, 1.0]), size=(BOOTSTRAP_N, len(values))); exceed = int(np.sum(np.abs((signs * values[None]).mean(axis=1)) >= abs(values.mean())))
            interaction_rows.append({"adaptation": adaptation, "metric": metric_name, "effect_dynamic_gain_minus_static_gain": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "p_signflip": float((1 + exceed) / (BOOTSTRAP_N + 1)), "definition": "(D-FRZ)-(D-adapt) minus ((S-FRZ)-(S-adapt)); positive means larger dynamic adaptation gain", "final_test_access": 0})
    write_csv(analysis / "VIDEO_ADAPTATION_INTERACTIONS.csv", interaction_rows)

    task_joint: list[dict[str, object]] = []
    for model in CELLS:
        error = metrics[model]["per_joint_hand_mae_std"]
        persistence_error = metrics["PERSISTENCE"]["per_joint_hand_mae_std"]
        for task in np.unique(tasks):
            mask = tasks == task
            task_joint.append({"model": model, "level": "task", "index": int(task), "N_episode": int(mask.sum()), "hand_mae_std": float(np.mean(metrics[model]["hand_mae_std"][mask])), "improvement_vs_persistence": float(np.mean(metrics["PERSISTENCE"]["hand_mae_std"][mask]) - np.mean(metrics[model]["hand_mae_std"][mask])), "final_test_access": 0})
        for joint in range(HAND):
            task_joint.append({"model": model, "level": "joint", "index": joint, "N_episode": 614, "hand_mae_std": float(np.mean(error[:, joint])), "improvement_vs_persistence": float(np.mean(persistence_error[:, joint]) - np.mean(error[:, joint])), "final_test_access": 0})
    write_csv(analysis / "PER_TASK_JOINT.csv", task_joint)

    resource_rows = one_seed_resource_costs()
    write_csv(analysis / "RESOURCE_COSTS.csv", resource_rows + [{"job_id": "old_B_training_from_prior_harvest", "status": "complete", "physical_gpu": "2,3", "wall_gpu_hours": 17.840, "kind": "prior_training", "retry": False, "exit_code": 0, "final_test_access": 0}])
    dynamic_decisions = [row for row in decisions if row["cell"] in ("FT-D-DENSE4", "FT-D-LORA4")]
    candidate_gate = "pass" if all(row["all_gates_pass"] for row in dynamic_decisions) else "fail"
    write_json(analysis / "CANDIDATE_GATE_DETAIL.json", {"status": candidate_gate.upper(), "candidate_priority": ["FT-D-DENSE4", "FT-D-LORA4"], "decisions": decisions, "dynamic_candidate_gate": candidate_gate, "single_seed_development_only": True, "final_test_access": 0})
    write_json(analysis / "ANALYSIS_METADATA.json", {"status": "PASS", "prediction_acceptance": "6/6", "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_unit": "paired episode", "seed_robustness": "not_assessed; one B seed", "hierarchical_bootstrap": "not_used", "dev_reuse": True, "metrics_source": str(SOURCE_ANALYSIS), "motion_strata_source": str(MOTION_STRATA), "final_test_access": 0})
    np.save(analysis / "EPISODE_BOOTSTRAP_INDICES_SOURCE.npy", bootstrap)
    print(json.dumps({"status": "PASS", "candidate_gate": candidate_gate, "models": len(metrics), "paired_rows": len(paired), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
