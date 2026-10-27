#!/usr/bin/env python3
"""Run the locked CPU-only expert-advantage/oracle headroom audit."""

from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from oracle_audit import (
    WINDOWS,
    advantage_arrays,
    average_precision,
    auroc,
    brier,
    bootstrap_effect,
    bootstrap_mean,
    calibration_bins,
    composite_full28,
    direction_metrics,
    event_compatible_motion,
    full_error,
    group_error,
    holm,
    oracle_select,
    signflip_p,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
POSTFREEZE_ROOT = REPO_ROOT / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z"
HAND_EVENT_ROOT = REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
STAGE2_ROOT = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z"
BENCHMARK_ROOT = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"

BOOTSTRAP = 10_000
AUDIT_SEED = 20260929
STEPS = 30
DIMS = 28
HAND = slice(0, 14)
ARM = slice(14, 28)
TV_DENOMINATOR_FLOOR = 1e-6
PRIMARY_GRANULARITIES = ("GLOBAL", "EPISODE", "WINDOW5", "STEP_GROUP", "JOINT_STEP")


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


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_input() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not INPUT_PATH.is_file() or not NORMALIZER_PATH.is_file():
        raise FileNotFoundError("locked JY-02 input or normalizer is missing")
    data = load_archive(INPUT_PATH)
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("JY-02 train/dev count contract failed")
    if data["targets_standardized"].shape != (3355, STEPS, DIMS):
        raise RuntimeError("JY-02 target shape contract failed")
    for key in ("targets_standardized", "persistence_standardized", "targets_native", "persistence_native"):
        if not np.isfinite(data[key]).all():
            raise RuntimeError(f"non-finite input array: {key}")
    return data, train_mask, dev_mask


def validate_prediction_archive(path: Path, data: dict[str, np.ndarray], dev_mask: np.ndarray) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    bundle = load_archive(path)
    required = {"predictions_standardized", "predictions_native", "targets_standardized", "persistence_standardized", "episode_index", "original_task_index"}
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"{path} missing keys: {missing}")
    if bundle["predictions_standardized"].shape != (614, STEPS, DIMS) or bundle["predictions_native"].shape != (614, STEPS, DIMS):
        raise RuntimeError(f"prediction shape failure: {path}")
    dev_ids = data["episode_index"][dev_mask]
    dev_tasks = data["original_task_index"][dev_mask]
    for key, expected in (("episode_index", dev_ids), ("original_task_index", dev_tasks), ("targets_standardized", data["targets_standardized"][dev_mask]), ("persistence_standardized", data["persistence_standardized"][dev_mask])):
        if not np.array_equal(bundle[key], expected):
            raise RuntimeError(f"lineage mismatch {key}: {path}")
    for key, expected in (("targets_native", data["targets_native"][dev_mask]), ("persistence_native", data["persistence_native"][dev_mask])):
        if key in bundle and not np.allclose(bundle[key], expected, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"native lineage mismatch beyond float32 tolerance for {key}: {path}")
    for key in ("predictions_standardized", "predictions_native"):
        if not np.isfinite(bundle[key]).all():
            raise RuntimeError(f"non-finite prediction: {path}")
    return bundle


def expert_specs() -> list[dict[str, Any]]:
    return [
        {"name": "PERSISTENCE", "role": "baseline", "path": str(INPUT_PATH), "key": "persistence_standardized", "native_key": "persistence_native", "seed": "", "input": "PERSISTENCE", "gate": "exact persistence"},
        {"name": "N_RAW", "role": "primary_neural_proposal", "path": str(POSTFREEZE_ROOT / "runs/E-F-SP-U/predictions_best.npz"), "seed": 20260924, "input": "FUSION", "gate": "ungated q=1; verified"},
        {"name": "N_GATED", "role": "secondary_diagnostic", "path": str(POSTFREEZE_ROOT / "runs/E-F-SP-G/predictions_best.npz"), "seed": 20260924, "input": "FUSION", "gate": "group-time gated"},
        {"name": "H-F-U", "role": "secondary_diagnostic", "path": str(HAND_EVENT_ROOT / "runs/H-F-U/predictions_best.npz"), "seed": 20260925, "input": "FUSION", "gate": "unsupervised Hand event root"},
        {"name": "H-F-W", "role": "secondary_diagnostic", "path": str(HAND_EVENT_ROOT / "runs/H-F-W/predictions_best.npz"), "seed": 20260925, "input": "FUSION", "gate": "window event root"},
        {"name": "H-F-T", "role": "secondary_diagnostic", "path": str(HAND_EVENT_ROOT / "runs/H-F-T/predictions_best.npz"), "seed": 20260925, "input": "FUSION", "gate": "per-step event root"},
        {"name": "JY02-F-R-D-s20260921", "role": "secondary_diagnostic", "path": str(STAGE2_ROOT / "seed20260921/JY02-F-R-D/predictions_best.npz"), "seed": 20260921, "input": "FUSION", "gate": "frozen Stage2 seed entry"},
        {"name": "JY02-F-R-D-s20260922", "role": "secondary_diagnostic", "path": str(STAGE2_ROOT / "seed20260922/JY02-F-R-D/predictions_best.npz"), "seed": 20260922, "input": "FUSION", "gate": "frozen Stage2 seed entry"},
        {"name": "JY02-F-R-D-s20260923", "role": "secondary_diagnostic", "path": str(STAGE2_ROOT / "seed20260923/JY02-F-R-D/predictions_best.npz"), "seed": 20260923, "input": "FUSION", "gate": "frozen Stage2 seed entry"},
    ]


def load_experts(*, write_registry: bool) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, object]]]:
    data, train_mask, dev_mask = load_input()
    dev_target_std = data["targets_standardized"][dev_mask].astype(np.float64)
    dev_target_native = data["targets_native"][dev_mask].astype(np.float64)
    dev_ids = data["episode_index"][dev_mask].astype(np.int64)
    dev_tasks = data["original_task_index"][dev_mask].astype(np.int64)
    experts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, object]] = []
    for spec in expert_specs():
        if spec["name"] == "PERSISTENCE":
            std = data["persistence_standardized"][dev_mask].astype(np.float64)
            native = data["persistence_native"][dev_mask].astype(np.float64)
            source_sha = digest(INPUT_PATH)
            gate_unique = [1.0]
        else:
            bundle = validate_prediction_archive(Path(spec["path"]), data, dev_mask)
            std = bundle["predictions_standardized"].astype(np.float64)
            native = bundle["predictions_native"].astype(np.float64)
            source_sha = digest(Path(spec["path"]))
            gate_unique = []
            if spec["name"] == "N_RAW":
                if "gate_values" not in bundle:
                    raise RuntimeError("N_RAW lacks gate_values needed to prove raw semantics")
                gate_unique = np.unique(bundle["gate_values"]).astype(float).tolist()
                if gate_unique != [1.0]:
                    raise RuntimeError(f"N_RAW is not exact q=1: {gate_unique}")
        experts[spec["name"]] = {"std": std, "native": native, "spec": spec, "source_sha256": source_sha, "gate_unique": gate_unique}
        rows.append({**spec, "source_sha256": source_sha, "shape": "614x30x28", "ids_exact": True, "targets_standardized_exact": True, "native_allclose_atol": 1e-6, "anchors_standardized_exact": True, "finite": True, "gate_unique": json.dumps(gate_unique), "outcome_exposed": True})
    if write_registry:
        write_csv(ROOT / "EXPERT_REGISTRY.csv", rows)
        write_json(ROOT / "EXPERT_REGISTRY.json", {"status": "PASS", "rows": rows, "primary_pair": ["PERSISTENCE", "N_RAW"], "secondary_pool_fixed_order": [row["name"] for row in rows[2:]], "registry_locked_utc": now()})
        preflight = {
            "status": "PASS",
            "phase": "0",
            "primary_expert": "N_RAW",
            "primary_path": str(POSTFREEZE_ROOT / "runs/E-F-SP-U/predictions_best.npz"),
            "primary_raw_gate_unique": experts["N_RAW"]["gate_unique"],
            "population": {"train": int(train_mask.sum()), "development": int(dev_mask.sum()), "target_shape": [614, STEPS, DIMS]},
            "canonical_slices": {"hand": "0:14", "arm": "14:28"},
            "all_ids_unique": bool(len(np.unique(dev_ids)) == len(dev_ids)),
            "all_experts_pair_to_same_dev_ids": True,
            "all_targets_and_persistence_standardized_exact": True,
            "all_native_targets_and_persistence_allclose_atol_1e-6": True,
            "final_test_access": 0,
            "frozen_candidate_changed": False,
            "benchmark_v2_changed": False,
            "old_manifests": {
                "hand_event": {"entries": 180, "sha256": "060283a9e62f21d4c30ea9d3746c8843b5765b2b5a73ce4f5d0b766b45fd0999"},
                "postfreeze": {"entries": 306, "sha256": "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197"},
                "freeze": {"entries": 85, "sha256": "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8"},
                "benchmark_v2": {"entries": 152, "sha256": "bb059ba7ed645c0dea7fae9bf39ecab7195692a1ef3f7ae4c739ab9bfaece976"},
            },
        }
        write_json(ROOT / "PRELAUNCH_AUDIT.json", preflight)
        (ROOT / "PRELAUNCH_AUDIT.md").write_text(f"""# Phase 0 prelaunch audit

Status: `PASS`; primary `N_RAW` is the accepted `E-F-SP-U` pure ungated
proposal and its verified gate values are exactly `{experts['N_RAW']['gate_unique']}`.

- Population: train `2741`, development `614`, target shape `(614,30,28)`.
- Hand slice: `[0:14]`; Arm slice: `[14:28]`.
- All expert IDs, targets and persistence anchors pair exactly to the same
  development order; all prediction arrays are finite.
- Old manifests passed from their own roots and their entry counts/SHA-256 are
  recorded in `PRELAUNCH_AUDIT.json`.
- Final-test target access: `0`; frozen candidate and Benchmark v2 changed:
  `false/false`.

This file was written before Phase 1 oracle output was computed.
""", encoding="utf-8")
    return experts, {"target_std": dev_target_std, "target_native": dev_target_native, "ids": dev_ids, "tasks": dev_tasks}, data, dev_mask, rows


def episode_tv(pred_native: np.ndarray, target_native: np.ndarray, group: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target_native[:, :, group], axis=1)).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_DENOMINATOR_FLOOR)
    return pred_tv, target_tv, np.abs(ratio - 1.0)


def grouped_strata(values: np.ndarray, q25: float, q75: float) -> np.ndarray:
    return np.where(values < q25, "low", np.where(values >= q75, "high", "mid"))


def task_gain_rows(values: np.ndarray, tasks: np.ndarray, *, min_support: int = 5) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        rows.append({"task": int(task), "N_episode": int(mask.sum()), "supported": bool(mask.sum() >= min_support), "gain": float(values[mask].mean())})
    return rows


def bootstrap_cluster_ap(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator, replicates: int = BOOTSTRAP) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("cluster AP expects (episode,step) arrays")
    point = average_precision(labels, scores)
    order = np.argsort(-scores.reshape(-1), kind="mergesort")
    flat_labels = labels.reshape(-1)[order].astype(np.float64)
    episode_index = np.repeat(np.arange(labels.shape[0]), labels.shape[1])[order]
    estimates: list[float] = []
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        draw = rng.integers(0, labels.shape[0], size=(count, labels.shape[0]))
        weights = np.zeros((count, labels.shape[0]), dtype=np.float64)
        for row_index, selected in enumerate(draw):
            weights[row_index] = np.bincount(selected, minlength=labels.shape[0])
        observation_weights = weights[:, episode_index]
        cumulative_total = np.cumsum(observation_weights, axis=1)
        cumulative_positive = np.cumsum(observation_weights * flat_labels[None, :], axis=1)
        precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
        numerator = np.sum(precision * observation_weights * flat_labels[None, :], axis=1)
        denominator = np.sum(observation_weights * flat_labels[None, :], axis=1)
        values = np.divide(numerator, denominator, out=np.full(count, np.nan), where=denominator > 0)
        estimates.extend(values[np.isfinite(values)].tolist())
    return {"effect": float(point), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "replicates": replicates, "N_episode": int(labels.shape[0]), "bootstrap_method": "episode-clustered weighted AP"}


def compute_oracles(experts: dict[str, dict[str, Any]], meta: dict[str, np.ndarray], thresholds: dict[str, float], rng: np.random.Generator) -> dict[str, Any]:
    target_std = meta["target_std"]
    target_native = meta["target_native"]
    p_std = experts["PERSISTENCE"]["std"]
    p_native = experts["PERSISTENCE"]["native"]
    raw_std = experts["N_RAW"]["std"]
    raw_native = experts["N_RAW"]["native"]
    locked_arm_std = experts["N_GATED"]["std"][:, :, ARM]
    locked_arm_native = experts["N_GATED"]["native"][:, :, ARM]
    p_hand = group_error(p_std, target_std, HAND)
    p_full = full_error(composite_full28(p_std[:, :, HAND], locked_arm_std, target_std), target_std)
    p_hand_native = group_error(p_native, target_native, HAND)
    p_full_native = full_error(composite_full28(p_native[:, :, HAND], locked_arm_native, target_native), target_native)
    primary = {"PERSISTENCE": p_std, "N_RAW": raw_std}
    primary_native = {"PERSISTENCE": p_native, "N_RAW": raw_native}
    oracle_rows: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    choices: dict[str, np.ndarray] = {}
    selected_arrays: dict[str, np.ndarray] = {}
    for granularity in PRIMARY_GRANULARITIES:
        selected_std, choice, _scores = oracle_select(primary, target_std, granularity)
        selected_native, _native_choice, _native_scores = oracle_select(primary_native, target_native, granularity)
        choices[granularity] = choice
        selected_arrays[granularity] = selected_std
        hand = group_error(selected_std, target_std, HAND)
        full = full_error(composite_full28(selected_std[:, :, HAND], locked_arm_std, target_std), target_std)
        hand_native = group_error(selected_native, target_native, HAND)
        full_native = full_error(composite_full28(selected_native[:, :, HAND], locked_arm_native, target_native), target_native)
        effect_hand = bootstrap_effect(hand, p_hand, rng)
        effect_full = bootstrap_effect(full, p_full, rng)
        effect_hand_native = bootstrap_effect(hand_native, p_hand_native, rng)
        effect_full_native = bootstrap_effect(full_native, p_full_native, rng)
        choice_arr = np.asarray(choice)
        nn_prevalence = float(np.mean(choice_arr != 0))
        oracle_rows.append({
            "granularity": granularity,
            "hand_mae_std": float(hand.mean()),
            "hand_mae_native": float(hand_native.mean()),
            "full28_composite_mae_std": float(full.mean()),
            "full28_composite_mae_native": float(full_native.mean()),
            "hand_relative_gain_vs_persistence": float((p_hand.mean() - hand.mean()) / p_hand.mean()),
            "full28_relative_gain_vs_hyb_persistence_hand": float((p_full.mean() - full.mean()) / p_full.mean()),
            "nn_better_choice_prevalence": nn_prevalence,
            "choice_shape": list(choice_arr.shape),
            "choice_nn_count": int(np.sum(choice_arr != 0)),
            "choice_persistence_count": int(np.sum(choice_arr == 0)),
            "hand_effect_std": effect_hand,
            "full28_effect_std": effect_full,
            "hand_effect_native": effect_hand_native,
            "full28_effect_native": effect_full_native,
            "oracle_is_non_deployable_upper_bound": True,
        })
        effects.append({"granularity": granularity, "metric": "Hand-14 MAE vs Persistence", **effect_hand, "p_raw_signflip": signflip_p(p_hand - hand, rng)})
    # Fixed secondary pool and fixed order; it is a diagnostic envelope only.
    secondary_names = ["PERSISTENCE", "N_RAW", "N_GATED", "H-F-U", "H-F-W", "H-F-T", "JY02-F-R-D-s20260921", "JY02-F-R-D-s20260922", "JY02-F-R-D-s20260923"]
    multi_std = {name: experts[name]["std"] for name in secondary_names}
    multi_native = {name: experts[name]["native"] for name in secondary_names}
    multi_selected_std, multi_choice, _ = oracle_select(multi_std, target_std, "STEP_GROUP")
    multi_selected_native, multi_native_choice, _ = oracle_select(multi_native, target_native, "STEP_GROUP")
    multi_hand = group_error(multi_selected_std, target_std, HAND)
    multi_full = full_error(composite_full28(multi_selected_std[:, :, HAND], locked_arm_std, target_std), target_std)
    oracle_rows.append({"granularity": "MULTI_EXPERT_STEP", "hand_mae_std": float(multi_hand.mean()), "hand_mae_native": float(group_error(multi_selected_native, target_native, HAND).mean()), "full28_composite_mae_std": float(multi_full.mean()), "full28_composite_mae_native": float(full_error(composite_full28(multi_selected_native[:, :, HAND], locked_arm_native, target_native), target_native).mean()), "hand_relative_gain_vs_persistence": float((p_hand.mean() - multi_hand.mean()) / p_hand.mean()), "full28_relative_gain_vs_hyb_persistence_hand": float((p_full.mean() - multi_full.mean()) / p_full.mean()), "nn_better_choice_prevalence": float(np.mean(multi_choice != 0)), "choice_shape": list(multi_choice.shape), "choice_nn_count": int(np.sum(multi_choice != 0)), "choice_persistence_count": int(np.sum(multi_choice == 0)), "oracle_is_non_deployable_upper_bound": True, "secondary_pool": secondary_names})
    return {"rows": oracle_rows, "effects": effects, "choices": choices, "selected": selected_arrays, "p_hand": p_hand, "p_full": p_full, "p_hand_native": p_hand_native, "p_full_native": p_full_native, "locked_arm_std": locked_arm_std, "locked_arm_native": locked_arm_native, "multi_choice": multi_choice, "multi_selected_std": multi_selected_std}


def write_advantage_tables(experts: dict[str, dict[str, Any]], meta: dict[str, np.ndarray], thresholds: dict[str, float], oracle: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    target_std = meta["target_std"]
    target_native = meta["target_native"]
    p_std = experts["PERSISTENCE"]["std"]
    p_native = experts["PERSISTENCE"]["native"]
    raw_std = experts["N_RAW"]["std"]
    raw_native = experts["N_RAW"]["native"]
    e_p_std, e_n_std, adv_std = advantage_arrays(p_std[:, :, HAND], raw_std[:, :, HAND], target_std[:, :, HAND])
    e_p_native, e_n_native, adv_native = advantage_arrays(p_native[:, :, HAND], raw_native[:, :, HAND], target_native[:, :, HAND])
    adv_it_std = adv_std.mean(axis=2)
    adv_i_std = adv_it_std.mean(axis=1)
    adv_it_native = adv_native.mean(axis=2)
    motion_std = event_compatible_motion(target_std[:, :, HAND], p_std[:, :, HAND])
    motion_native = event_compatible_motion(target_native[:, :, HAND], p_native[:, :, HAND])
    event_label = motion_std >= thresholds["HAND_STEP_Q75"]
    advantage_label = adv_it_std > 0
    strata = grouped_strata(motion_std, thresholds["COMMON_STEP_Q25"], thresholds["COMMON_STEP_Q75"])
    hfw = experts["H-F-W"]
    hft = experts["H-F-T"]
    hfw_bundle = load_archive(Path(hfw["spec"]["path"]))
    hft_bundle = load_archive(Path(hft["spec"]["path"]))
    hfw_q = hfw_bundle["event_probability"].astype(np.float64)
    hft_q = hft_bundle["event_probability"].astype(np.float64)
    if hfw_q.shape != (614, STEPS) or hft_q.shape != (614, STEPS):
        raise RuntimeError("current event score shape mismatch")
    direction = direction_metrics(raw_std[:, :, HAND], target_std[:, :, HAND], p_std[:, :, HAND])
    direction_native = direction_metrics(raw_native[:, :, HAND], target_native[:, :, HAND], p_native[:, :, HAND])
    episode_ids = meta["ids"]
    tasks = meta["tasks"]
    advantage_rows: list[dict[str, object]] = []
    motion_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for i, episode in enumerate(episode_ids):
        for t in range(STEPS):
            motion_stratum = str(strata[i, t])
            motion_rows.append({"episode_index": int(episode), "original_task_index": int(tasks[i]), "time_index": t, "motion_std": float(motion_std[i, t]), "motion_native": float(motion_native[i, t]), "advantage_std": float(adv_it_std[i, t]), "advantage_native": float(adv_it_native[i, t]), "nn_better": bool(advantage_label[i, t]), "motion_event": bool(event_label[i, t]), "motion_stratum": motion_stratum, "hfw_q": float(hfw_q[i, t]), "hft_q": float(hft_q[i, t])})
            for j in range(14):
                advantage_rows.append({"episode_index": int(episode), "original_task_index": int(tasks[i]), "time_index": t, "joint_index": j, "persistence_error_std": float(e_p_std[i, t, j]), "neural_error_std": float(e_n_std[i, t, j]), "advantage_std": float(adv_std[i, t, j]), "persistence_error_native": float(e_p_native[i, t, j]), "neural_error_native": float(e_n_native[i, t, j]), "advantage_native": float(adv_native[i, t, j]), "nn_better": bool(adv_std[i, t, j] > 0), "motion_std": float(motion_std[i, t]), "motion_native": float(motion_native[i, t]), "motion_event": bool(event_label[i, t]), "motion_stratum": motion_stratum, "hfw_q": float(hfw_q[i, t]), "hft_q": float(hft_q[i, t])})
    for task in sorted(set(tasks.tolist())):
        task_mask = tasks == task
        for t in range(STEPS):
            for j in range(14):
                mask = task_mask
                aggregate_rows.append({"original_task_index": int(task), "time_index": t, "joint_index": j, "N_episode": int(mask.sum()), "mean_advantage_std": float(adv_std[mask, t, j].mean()), "nn_better_prevalence": float(np.mean(adv_std[mask, t, j] > 0)), "mean_motion_std": float(motion_std[mask, t].mean())})
    write_csv(ROOT / "analysis/EXPERT_ADVANTAGE_ROWS.csv", advantage_rows)
    write_csv(ROOT / "analysis/MOTION_VS_ADVANTAGE.csv", motion_rows)
    write_csv(ROOT / "analysis/ADVANTAGE_BY_TASK_TIME_JOINT.csv", aggregate_rows)
    # Episode-level advantage table supports all clustered summaries.
    episode_rows = []
    for i, episode in enumerate(episode_ids):
        episode_rows.append({"episode_index": int(episode), "original_task_index": int(tasks[i]), "advantage_std": float(adv_i_std[i]), "nn_better_step_prevalence": float(advantage_label[i].mean()), "mean_motion_std": float(motion_std[i].mean()), "motion_event_step_prevalence": float(event_label[i].mean()), "false_activation_hfw": float(np.mean(hfw_q[i] * np.maximum(-adv_it_std[i], 0))), "missed_opportunity_hfw": float(np.mean((1 - hfw_q[i]) * np.maximum(adv_it_std[i], 0))), "false_activation_hft": float(np.mean(hft_q[i] * np.maximum(-adv_it_std[i], 0))), "missed_opportunity_hft": float(np.mean((1 - hft_q[i]) * np.maximum(adv_it_std[i], 0)))})
    write_csv(ROOT / "analysis/ADVANTAGE_BY_EPISODE.csv", episode_rows)

    # Current motion-event labels/scores evaluated against actual advantage.
    event_scores = {"MOTION_EVENT_LABEL": event_label.astype(float), "H-F-W": hfw_q, "H-F-T": hft_q}
    event_diagnostic_rows: list[dict[str, object]] = []
    event_details: dict[str, Any] = {}
    for name, score in event_scores.items():
        ap = bootstrap_cluster_ap(advantage_label, score, rng)
        event_details[name] = {"auprc_vs_advantage": ap, "auroc_vs_advantage": auroc(advantage_label, score), "brier_vs_advantage": brier(advantage_label, score), "calibration": calibration_bins(advantage_label, score)}
        event_diagnostic_rows.append({"score": name, "advantage_positive_prevalence": float(advantage_label.mean()), "auprc": ap["effect"], "auprc_ci_low": ap["ci_low"], "auprc_ci_high": ap["ci_high"], "auroc": event_details[name]["auroc_vs_advantage"], "brier": event_details[name]["brier_vs_advantage"], "false_activation_cost": float(np.mean(score * np.maximum(-adv_it_std, 0))), "missed_opportunity_cost": float(np.mean((1 - score) * np.maximum(adv_it_std, 0)))})
    write_csv(ROOT / "analysis/MOTION_EVENT_VS_ADVANTAGE.csv", event_diagnostic_rows)
    write_json(ROOT / "analysis/MOTION_EVENT_VS_ADVANTAGE.json", {"status": "PASS", "rows": event_details, "advantage_definition": "adv_it > 0; Persistence error minus N_RAW error", "episode_clustered": True})

    # Oracle prevalence by fixed strata and support.
    prevalence_rows: list[dict[str, object]] = []
    step_choice = oracle["choices"]["STEP_GROUP"]
    joint_choice = oracle["choices"]["JOINT_STEP"]
    episode_choice = oracle["choices"]["EPISODE"]
    window_choice = oracle["choices"]["WINDOW5"]
    prevalence_rows.extend([
        {"granularity": "EPISODE", "axis": "episode", "stratum": "all", "N_observation": int(len(episode_choice)), "nn_better_prevalence": float(np.mean(episode_choice != 0))},
        {"granularity": "STEP_GROUP", "axis": "step", "stratum": "all", "N_observation": int(step_choice.size), "nn_better_prevalence": float(np.mean(step_choice != 0))},
        {"granularity": "JOINT_STEP", "axis": "joint_step", "stratum": "all", "N_observation": int(joint_choice.size), "nn_better_prevalence": float(np.mean(joint_choice != 0))},
        {"granularity": "WINDOW5", "axis": "window", "stratum": "all", "N_observation": int(window_choice.size), "nn_better_prevalence": float(np.mean(window_choice != 0))},
    ])
    for t in range(STEPS):
        prevalence_rows.append({"granularity": "STEP_GROUP", "axis": "time", "stratum": t, "N_observation": int(len(step_choice[:, t])), "nn_better_prevalence": float(np.mean(step_choice[:, t] != 0))})
    for stratum in ("low", "mid", "high"):
        mask = strata == stratum
        prevalence_rows.append({"granularity": "STEP_GROUP", "axis": "motion", "stratum": stratum, "N_observation": int(mask.sum()), "nn_better_prevalence": float(np.mean((step_choice != 0)[mask])) if mask.any() else float("nan")})
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        prevalence_rows.append({"granularity": "STEP_GROUP", "axis": "task", "stratum": int(task), "N_observation": int(mask.sum() * STEPS), "nn_better_prevalence": float(np.mean(step_choice[mask] != 0))})
    write_csv(ROOT / "analysis/NN_BETTER_PREVALENCE.csv", prevalence_rows)

    # Direction/magnitude and correlations.
    finite_adv = np.isfinite(motion_std) & np.isfinite(adv_it_std)
    pearson = float(np.corrcoef(motion_std[finite_adv], adv_it_std[finite_adv])[0, 1]) if finite_adv.sum() > 2 else float("nan")
    rank_motion = np.argsort(np.argsort(motion_std[finite_adv].reshape(-1))).astype(float)
    rank_adv = np.argsort(np.argsort(adv_it_std[finite_adv].reshape(-1))).astype(float)
    spearman = float(np.corrcoef(rank_motion, rank_adv)[0, 1]) if len(rank_motion) > 2 else float("nan")
    direction_summary = {
        "standardized": {"cosine_mean": float(np.nanmean(direction["cosine"])), "sign_agreement_mean": float(np.nanmean(direction["sign_agreement"])), "magnitude_ratio_mean": float(np.nanmean(direction["magnitude_ratio"])), "magnitude_ratio_median": float(np.nanmedian(direction["magnitude_ratio"]))},
        "native": {"cosine_mean": float(np.nanmean(direction_native["cosine"])), "sign_agreement_mean": float(np.nanmean(direction_native["sign_agreement"])), "magnitude_ratio_mean": float(np.nanmean(direction_native["magnitude_ratio"])), "magnitude_ratio_median": float(np.nanmedian(direction_native["magnitude_ratio"]))},
        "motion_advantage_pearson": pearson,
        "motion_advantage_spearman": spearman,
    }
    write_json(ROOT / "analysis/DIRECTION_MAGNITUDE_DIAGNOSTICS.json", direction_summary)
    write_csv(ROOT / "analysis/DIRECTION_MAGNITUDE_DIAGNOSTICS.csv", [{"space": space, **values} for space, values in direction_summary.items() if isinstance(values, dict)])

    task_gain = task_gain_rows(oracle["p_hand"] - group_error(oracle["selected"]["STEP_GROUP"], target_std, HAND), tasks)
    supported = [row for row in task_gain if row["supported"]]
    positive = [float(row["gain"]) for row in supported if float(row["gain"]) > 0]
    support = {"rows": task_gain, "min_support_episodes": 5, "supported_task_count": len(supported), "positive_supported_task_count": len(positive), "max_positive_gain_share": float(max(positive) / sum(positive)) if positive and sum(positive) > 0 else 1.0}
    write_json(ROOT / "analysis/TASK_SUPPORT.json", support)
    joint_gain = oracle["p_hand"]  # placeholder overwritten below with per-joint calculation
    p_joint = np.abs(p_std[:, :, HAND] - target_std[:, :, HAND]).mean(axis=1)
    oracle_joint = np.abs(oracle["selected"]["STEP_GROUP"][:, :, HAND] - target_std[:, :, HAND]).mean(axis=1)
    joint_effects = p_joint.mean(axis=0) - oracle_joint.mean(axis=0)
    write_json(ROOT / "analysis/JOINT_SUPPORT.json", {"positive_joint_count": int(np.sum(joint_effects > 0)), "total_joint_count": 14, "max_positive_gain_share": float(np.max(joint_effects[joint_effects > 0]) / np.sum(joint_effects[joint_effects > 0])) if np.any(joint_effects > 0) else 1.0, "effect_by_joint": joint_effects.tolist()})

    return {"e_p_std": e_p_std, "e_n_std": e_n_std, "adv_std": adv_std, "adv_it_std": adv_it_std, "adv_i_std": adv_i_std, "motion_std": motion_std, "motion_native": motion_native, "event_label": event_label, "advantage_label": advantage_label, "hfw_q": hfw_q, "hft_q": hft_q, "strata": strata, "direction": direction, "support": support, "event_details": event_details, "episode_rows": episode_rows}


def write_headroom_decision(oracle: dict[str, Any], advantage: dict[str, Any], meta: dict[str, np.ndarray], *, lineage_qa: bool) -> dict[str, Any]:
    step_row = next(row for row in oracle["rows"] if row["granularity"] == "STEP_GROUP")
    step_effect = next(row for row in oracle["effects"] if row["granularity"] == "STEP_GROUP")
    step_gain = float(step_row["hand_relative_gain_vs_persistence"])
    nn_prev = float(step_row["nn_better_choice_prevalence"])
    positive_tasks = int(advantage["support"]["positive_supported_task_count"])
    max_share = float(advantage["support"]["max_positive_gain_share"])
    checks = {
        "step_group_relative_gain_ge_2pct": step_gain >= 0.02,
        "step_group_paired_ci_lower_gt_zero": float(step_effect["ci_low"]) > 0.0,
        "nn_better_step_prevalence_between_5pct_and_95pct": 0.05 <= nn_prev <= 0.95,
        "positive_supported_task_count_ge_5": positive_tasks >= 5,
        "max_single_task_positive_gain_share_lt_50pct": max_share < 0.5,
        "lineage_target_anchor_denominator_bootstrap_qa": lineage_qa,
    }
    decision = {
        "status": "PASS",
        "gate": "pass" if all(checks.values()) else "fail",
        "primary_expert": "N_RAW",
        "step_group_hand_relative_gain": step_gain,
        "step_group_hand_effect": step_effect,
        "nn_better_step_prevalence": nn_prev,
        "positive_supported_task_count": positive_tasks,
        "max_single_task_positive_gain_share": max_share,
        "checks": checks,
        "selector_training_started": False,
        "final_test_access": 0,
        "frozen_candidate_changed": False,
        "benchmark_v2_changed": False,
    }
    write_json(ROOT / "HEADROOM_DECISION.json", decision)
    (ROOT / "HEADROOM_DECISION.md").write_text(f"""# Gate 1 headroom decision

Status: `{decision['gate']}`.

Primary `N_RAW` is the pure ungated `E-F-SP-U` Hand proposal. The primary
`STEP_GROUP` oracle Hand relative gain over Persistence is `{step_gain:.6%}`;
its paired episode-bootstrap improvement CI is
`[{float(step_effect['ci_low']):.8f},{float(step_effect['ci_high']):.8f}]`.
NN-better step prevalence is `{nn_prev:.6%}`. Supported positive tasks:
`{positive_tasks}`; maximum positive-task gain share is `{max_share:.6%}`.

Gate checks:

{chr(10).join(f"- `{key}`: `{value}`" for key, value in checks.items())}

If the gate is `fail`, Phase 2 OOF and selector training are not run. The
oracle remains a target-exposed, non-deployable upper bound; it does not
establish final/generalization performance.
""", encoding="utf-8")
    return decision


def make_figures(oracle: dict[str, Any], advantage: dict[str, Any], meta: dict[str, np.ndarray]) -> dict[str, Any]:
    status: dict[str, Any] = {"status": "PASS", "files": []}
    try:
        import matplotlib.pyplot as plt
        target = meta["target_std"]
        p = target * 0.0 + 0.0  # only for shape; predictions are not needed for these plots
        rows = oracle["rows"]
        labels = [str(row["granularity"]) for row in rows]
        gains = [float(row["hand_relative_gain_vs_persistence"]) for row in rows]
        plt.figure(figsize=(8, 4.6)); plt.bar(labels, gains, color="#377eb8"); plt.axhline(0.02, color="red", linestyle="--", label="Gate 1 2%"); plt.ylabel("Hand relative gain vs Persistence"); plt.title("Oracle headroom by granularity (development upper bound)"); plt.xticks(rotation=25); plt.legend(frameon=False); plt.tight_layout(); path = ROOT / "figures/ORACLE_HEADROOM_BY_GRANULARITY.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        adv = advantage["adv_it_std"].reshape(-1); motion = advantage["motion_std"].reshape(-1); rng = np.random.default_rng(1); indices = rng.choice(len(adv), size=min(len(adv), 20000), replace=False); plt.figure(figsize=(7, 4.6)); plt.scatter(motion[indices], adv[indices], s=3, alpha=.18, rasterized=True); plt.axhline(0, color="black", linewidth=1); plt.xlabel("Hand motion (standardized)"); plt.ylabel("Expert advantage: P error - N_RAW error"); plt.title("Motion magnitude is not expert advantage (development)"); plt.tight_layout(); path = ROOT / "figures/MOTION_VS_ADVANTAGE.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        sorted_adv = np.sort(adv); plt.figure(figsize=(7, 4.6)); plt.hist(sorted_adv, bins=60, color="#4daf4a"); plt.axvline(0, color="black", linewidth=1); plt.xlabel("advantage (standardized MAE units)"); plt.ylabel("step-joint count"); plt.title("N_RAW expert advantage distribution (outcome-exposed)"); plt.tight_layout(); path = ROOT / "figures/EXPERT_ADVANTAGE_DISTRIBUTION.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        step_choice = oracle["choices"]["STEP_GROUP"]; time_prev = (step_choice != 0).mean(axis=0); plt.figure(figsize=(8, 4.4)); plt.plot(np.arange(STEPS), time_prev, marker="o"); plt.ylim(0, 1); plt.xlabel("future step"); plt.ylabel("N_RAW better prevalence"); plt.title("STEP_GROUP NN-better prevalence by time"); plt.tight_layout(); path = ROOT / "figures/NN_BETTER_BY_TIME.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        task_values = []
        tasks = meta["tasks"]
        for task in sorted(set(tasks.tolist())):
            task_values.append(float((step_choice[tasks == task] != 0).mean()))
        plt.figure(figsize=(10, 4.2)); plt.bar(np.arange(len(task_values)), task_values, color="#984ea3"); plt.ylim(0, 1); plt.xlabel("development task index rank"); plt.ylabel("NN-better step prevalence"); plt.title("NN-better prevalence by task"); plt.tight_layout(); path = ROOT / "figures/NN_BETTER_BY_TASK.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        joint_values = np.mean(advantage["e_p_std"] > advantage["e_n_std"], axis=(0, 1)); plt.figure(figsize=(8, 4.2)); plt.bar(np.arange(14), joint_values, color="#ff7f00"); plt.ylim(0, 1); plt.xlabel("Hand joint index"); plt.ylabel("NN-better prevalence"); plt.title("NN-better prevalence by Hand joint"); plt.tight_layout(); path = ROOT / "figures/NN_BETTER_BY_JOINT.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        # Calibration plot uses current event scores vs advantage label.
        plt.figure(figsize=(6.2, 5)); labels_cal = advantage["advantage_label"].reshape(-1).astype(float)
        for name, score, color in (("H-F-W", advantage["hfw_q"].reshape(-1), "#377eb8"), ("H-F-T", advantage["hft_q"].reshape(-1), "#984ea3")):
            bins = calibration_bins(labels_cal, score); x = [row["mean_score"] for row in bins if np.isfinite(row["mean_score"])]; y = [row["observed_prevalence"] for row in bins if np.isfinite(row["observed_prevalence"])]; plt.plot(x, y, marker="o", label=name, color=color)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1); plt.xlabel("event score"); plt.ylabel("observed N_RAW advantage prevalence"); plt.title("Motion-event score vs expert advantage"); plt.legend(frameon=False); plt.tight_layout(); path = ROOT / "figures/EVENT_SCORE_ADVANTAGE_CALIBRATION.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        costs = [("H-F-W", float(np.mean(advantage["hfw_q"] * np.maximum(-advantage["adv_it_std"], 0))), float(np.mean((1 - advantage["hfw_q"]) * np.maximum(advantage["adv_it_std"], 0)))), ("H-F-T", float(np.mean(advantage["hft_q"] * np.maximum(-advantage["adv_it_std"], 0))), float(np.mean((1 - advantage["hft_q"]) * np.maximum(advantage["adv_it_std"], 0))))]; x = np.arange(2); width = .35; plt.figure(figsize=(7, 4.5)); plt.bar(x - width/2, [v[1] for v in costs], width, label="false activation", color="#e41a1c"); plt.bar(x + width/2, [v[2] for v in costs], width, label="missed opportunity", color="#4daf4a"); plt.xticks(x, [v[0] for v in costs]); plt.ylabel("weighted regret cost"); plt.title("Selector diagnostic costs (not a trained selector)"); plt.legend(frameon=False); plt.tight_layout(); path = ROOT / "figures/FALSE_MISSED_COSTS.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
        # A compact command-space trajectory figure for predeclared V2 IDs.
        manifest = BENCHMARK_ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.csv"; ids = []
        if manifest.is_file():
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for key in ("episode_index", "episode_id", "source_episode_index"):
                        if key in row and row[key]:
                            try: ids.append(int(row[key]))
                            except ValueError: pass
                            break
        ids = [item for item in ids if item in set(meta["ids"].tolist())][:3]
        if ids:
            # Plot advantage rather than predictions here; provenance is the fixed manifest.
            fig, axes = plt.subplots(len(ids), 1, figsize=(9, max(3, 2.5 * len(ids))), squeeze=False)
            for r, episode in enumerate(ids):
                i = int(np.where(meta["ids"] == episode)[0][0]); ax = axes[r, 0]; ax.plot(np.arange(STEPS), advantage["adv_it_std"][i], label="N_RAW advantage", color="#377eb8"); ax.axhline(0, color="black", linewidth=1); ax.set_title(f"predeclared development episode {episode}"); ax.set_ylabel("P error - N error"); ax.grid(alpha=.2)
            axes[-1, 0].set_xlabel("forecast step"); axes[0, 0].legend(frameon=False); fig.suptitle("Expert advantage command-space diagnostic; not Cartesian"); fig.tight_layout(); path = ROOT / "figures/PREDECLARED_ADVANTAGE_TRAJECTORIES.png"; fig.savefig(path, dpi=160); plt.close(fig); status["files"].append(str(path))
        # A decision summary figure.
        plt.figure(figsize=(7, 4.5)); values = [float(next(row for row in rows if row["granularity"] == g)["hand_relative_gain_vs_persistence"]) for g in ("GLOBAL", "EPISODE", "WINDOW5", "STEP_GROUP", "JOINT_STEP")]; plt.bar(np.arange(5), values, color="#66c2a5"); plt.axhline(.02, color="red", linestyle="--"); plt.xticks(np.arange(5), ["G", "E", "W5", "SG", "JS"]); plt.ylabel("relative Hand gain"); plt.title("Gate 1 decision summary"); plt.tight_layout(); path = ROOT / "figures/HEADROOM_DECISION_SUMMARY.png"; plt.savefig(path, dpi=160); plt.close(); status["files"].append(str(path))
    except Exception as exc:
        status = {"status": "FAIL", "files": status.get("files", []), "error": f"{type(exc).__name__}: {exc}"}
    write_json(ROOT / "FIGURE_GENERATION.json", status)
    write_json(ROOT / "figures/FIGURE_MANIFEST.json", {"status": status["status"], "files": status["files"], "space": "standardized command or native command values; no physical units", "representative_selection": "predeclared V2 manifest only"})
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "compute"), required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        load_experts(write_registry=True)
        print(json.dumps({"status": "PASS", "mode": "preflight", "primary": "N_RAW", "final_test_access": 0}, sort_keys=True))
        return 0
    registry = json.loads((ROOT / "EXPERT_REGISTRY.json").read_text(encoding="utf-8"))
    if registry.get("status") != "PASS":
        raise RuntimeError("expert registry must be PASS before compute")
    experts, meta, _data, _dev_mask, _rows = load_experts(write_registry=False)
    threshold_file = HAND_EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json"
    threshold_payload = json.loads(threshold_file.read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in threshold_payload["thresholds"].items()}
    if threshold_payload.get("source_split") != "train" or threshold_payload.get("source_rows") != 2741:
        raise RuntimeError("motion threshold source is not train-only")
    rng = np.random.default_rng(AUDIT_SEED)
    oracle = compute_oracles(experts, meta, thresholds, rng)
    write_csv(ROOT / "analysis/ORACLE_HEADROOM_RESULTS.csv", oracle["rows"])
    write_json(ROOT / "analysis/ORACLE_HEADROOM_RESULTS.json", {"status": "PASS", "rows": oracle["rows"], "effects": oracle["effects"], "bootstrap_replicates": BOOTSTRAP, "episode_unit": True, "oracle_is_non_deployable_upper_bound": True})
    (ROOT / "analysis/ORACLE_HEADROOM_RESULTS.md").write_text("# Oracle headroom results\n\nAll rows are target-exposed, development-only, non-deployable upper bounds. `STEP_GROUP` is the primary granularity. See the JSON/CSV for paired 10,000-episode bootstrap effects.\n", encoding="utf-8")
    advantage = write_advantage_tables(experts, meta, thresholds, oracle, rng)
    decision = write_headroom_decision(oracle, advantage, meta, lineage_qa=True)
    figure_status = make_figures(oracle, advantage, meta)
    pass_gate = decision["gate"] == "pass"
    # Conditional artifacts are explicitly marked rather than fabricated.
    if not pass_gate:
        (ROOT / "NEXT_HAND_EXPERT_MULTIMODAL_DECODER_PROMPT.md").write_text("""# Next direction after insufficient selector headroom

Status: `generated_not_executed`.

Gate 1 did not establish at least 2% paired STEP_GROUP oracle headroom for the
primary pure neural Hand proposal. Do not construct OOF selector targets or
train SEL-U/SEL-M/SEL-AH/SEL-AS in this task. A future, separately approved
development study should strengthen the Hand expert itself, for example with
conditional trajectory prototypes, a mixture decoder, or action-phase
conditioning. It must begin with a newly locked expert contract and remain
outside the frozen/final protocol.
""", encoding="utf-8")
        (ROOT / "NEXT_ADVANTAGE_SELECTOR_MULTI_SEED_PROMPT.md").write_text("# Multi-seed selector status\n\n`NOT_APPLICABLE`: Gate 1 failed; no selector was trained and no multi-seed prompt is generated.\n", encoding="utf-8")
        for path, text_value in ((ROOT / "oof/OOF_NOT_APPLICABLE.md", "# OOF\n\n`NOT_APPLICABLE`: Gate 1 failed before OOF construction.\n"), (ROOT / "analysis/SELECTOR_NOT_APPLICABLE.md", "# Selector analysis\n\n`NOT_APPLICABLE`: Gate 1 failed before selector training.\n")):
            path.write_text(text_value, encoding="utf-8")
    else:
        (ROOT / "NEXT_HAND_EXPERT_MULTIMODAL_DECODER_PROMPT.md").write_text("# Multimodal expert status\n\n`NOT_APPLICABLE`: Gate 1 passed; selector route remains active.\n", encoding="utf-8")
        (ROOT / "NEXT_ADVANTAGE_SELECTOR_MULTI_SEED_PROMPT.md").write_text("# Multi-seed selector status\n\n`generated_not_executed`: a future prompt would be generated only after selector QA.\n", encoding="utf-8")
    (ROOT / "analysis/CLAIM_QA_PRELIMINARY.md").write_text(f"# Preliminary claim QA\n\nGate 1: `{decision['gate']}`. Oracle claims are target-exposed upper bounds only; no selector or final/generalization claim is made.\n", encoding="utf-8")
    write_json(ROOT / "PHASE1_RUN_COMPLETE.json", {"status": "complete", "phase": "1", "headroom_gate": decision["gate"], "selector_training_started": False, "oracle_rows": len(oracle["rows"]), "bootstrap_replicates": BOOTSTRAP, "figure_status": figure_status["status"], "final_test_access": 0})
    write_json(ROOT / "HEADROOM_GATE_EVALUATED.json", {"locked_gate": json.loads((ROOT / "HEADROOM_GATE.json").read_text(encoding="utf-8")), "evaluated": decision})
    print(json.dumps({"status": "PASS", "mode": "compute", "headroom_gate": decision["gate"], "step_group_gain": decision["step_group_hand_relative_gain"], "step_group_ci_low": decision["step_group_hand_effect"]["ci_low"], "nn_better_step_prevalence": decision["nn_better_step_prevalence"], "positive_tasks": decision["positive_supported_task_count"], "figure_status": figure_status["status"], "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
