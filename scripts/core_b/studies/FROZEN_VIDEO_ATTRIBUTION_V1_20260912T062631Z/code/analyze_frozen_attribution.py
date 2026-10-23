#!/usr/bin/env python3
"""CPU-only analysis for the completed Frozen attribution bundles."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
B23 = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
INPUT = B23 / "inputs/B2_INPUTS_0125.npz"
TARGET_NORM = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
V2 = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
BOOTSTRAP = V2 / "data/paired_episode_bootstrap_indices.npy"
STATE_BUNDLE = REPO / "reports/coordination/CORE_A_CORE_B_DUAL_GPU_20260903T064316Z/recovery_gpu0_contractfix_20260904T073316Z/core_b/b1_inference_20260904T073316Z/B-STATE/best/predictions.npz"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(pred_std: np.ndarray, pred_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> dict[str, np.ndarray]:
    err = np.abs(pred_std - target_std)
    native_err = np.abs(pred_native - target_native)
    arm = err[:, :, 14:28]
    hand = err[:, :, :14]
    full = err
    diff = np.abs(np.diff(pred_std, axis=1) - np.diff(target_std, axis=1))
    native_diff = np.abs(np.diff(pred_native, axis=1) - np.diff(target_native, axis=1))
    target_changes = np.diff(target_native, axis=1)[:, :, 14:28]
    pred_changes = np.diff(pred_native, axis=1)[:, :, 14:28]
    target_tv = np.abs(target_changes).sum(axis=1).mean(axis=1)
    pred_tv = np.abs(pred_changes).sum(axis=1).mean(axis=1)
    endpoint_target = np.abs(target_native[:, -1, 14:28] - target_native[:, 0, 14:28]).mean(axis=1)
    endpoint_pred = np.abs(pred_native[:, -1, 14:28] - pred_native[:, 0, 14:28]).mean(axis=1)
    direction_valid = (np.abs(target_changes) > 1e-8) | (np.abs(pred_changes) > 1e-8)
    direction_agreement = np.divide(((np.sign(target_changes) == np.sign(pred_changes)) & direction_valid).sum(axis=(1, 2)), direction_valid.sum(axis=(1, 2)), out=np.full(len(pred_std), np.nan), where=direction_valid.sum(axis=(1, 2)) > 0)
    dot = (target_changes * pred_changes).sum(axis=2)
    den = np.linalg.norm(target_changes, axis=2) * np.linalg.norm(pred_changes, axis=2)
    direction_cosine = np.divide(dot.sum(axis=1), (den > 1e-8).sum(axis=1), out=np.full(len(pred_std), np.nan), where=(den > 1e-8).sum(axis=1) > 0)
    return {
        "arm_full": arm.mean(axis=(1, 2)), "hand_full": hand.mean(axis=(1, 2)), "full28_full": full.mean(axis=(1, 2)),
        "arm_early": arm[:, :10].mean(axis=(1, 2)), "arm_middle": arm[:, 10:20].mean(axis=(1, 2)), "arm_late": arm[:, 20:].mean(axis=(1, 2)), "arm_endpoint": arm[:, 29].mean(axis=1),
        "hand_early": hand[:, :10].mean(axis=(1, 2)), "hand_middle": hand[:, 10:20].mean(axis=(1, 2)), "hand_late": hand[:, 20:].mean(axis=(1, 2)), "hand_endpoint": hand[:, 29].mean(axis=1),
        "full_first_difference": diff.mean(axis=(1, 2)), "arm_first_difference": diff[:, :, 14:28].mean(axis=(1, 2)), "native_tv_ratio": np.divide(pred_tv, target_tv, out=np.full(len(pred_std), np.nan), where=target_tv > 1e-12),
        "native_endpoint_change_error": np.abs(endpoint_pred - endpoint_target), "native_direction_agreement": direction_agreement, "native_direction_cosine": direction_cosine,
        "target_mean_step_motion_native": np.abs(target_changes).mean(axis=(1, 2)), "pred_mean_step_motion_native": np.abs(pred_changes).mean(axis=(1, 2)),
        "native_arm_mae": native_err[:, :, 14:28].mean(axis=(1, 2)), "native_full_mae": native_err.mean(axis=(1, 2)),
    }


def scalar(value: np.ndarray) -> float:
    return float(np.nanmean(value)) if np.isfinite(value).any() else float("nan")


def bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sample = values[indices].mean(axis=1)
    return float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def load_bundle(path: Path, expected_target_std: np.ndarray, expected_target_native: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["episode_index"].astype(np.int64)
        target_std = archive["targets_standardized"].astype(np.float32)
        target_native = archive["targets_native"].astype(np.float32)
        pred_std = archive["predictions_standardized"].astype(np.float32)
        pred_native = archive["predictions_native"].astype(np.float32)
    if pred_std.shape != (len(ids), 30, 28) or not np.isfinite(pred_std).all() or not np.isfinite(pred_native).all():
        raise RuntimeError(f"invalid bundle {path}")
    if np.max(np.abs(target_std - expected_target_std[{int(i): int(np.where(expected_ids == i)[0][0]) for i in expected_ids.tolist()}])) if False else False:
        raise RuntimeError("unreachable")
    return {"path": path, "ids": ids, "target_std": target_std, "target_native": target_native, "pred_std": pred_std, "pred_native": pred_native, "sha256": digest(path)}


def align_bundle(bundle: dict[str, object], ids: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> dict[str, object]:
    bundle_ids = np.asarray(bundle["ids"], dtype=np.int64)
    index = {int(value): position for position, value in enumerate(bundle_ids.tolist())}
    positions = [index[int(value)] for value in ids.tolist()]
    b_target_std = np.asarray(bundle["target_std"])[positions]
    b_target_native = np.asarray(bundle["target_native"])[positions]
    if np.max(np.abs(b_target_std - target_std)) > 0 or np.max(np.abs(b_target_native - target_native)) > 0:
        raise RuntimeError(f"target lineage mismatch for {bundle['path']}")
    return {**bundle, "ids": ids, "target_std": target_std, "target_native": target_native, "pred_std": np.asarray(bundle["pred_std"])[positions], "pred_native": np.asarray(bundle["pred_native"])[positions]}


def main() -> int:
    with np.load(INPUT, allow_pickle=False) as archive:
        dev = archive["split_code"] == 1
        train = archive["split_code"] == 0
        ids = archive["episode_index"][dev].astype(np.int64)
        tasks = archive["original_task_index"][dev].astype(np.int64)
        target_std = archive["targets_standardized"][dev].astype(np.float32)
        target_native = archive["targets"][dev].astype(np.float32)
        last_action = archive["last_observed_action"][dev].astype(np.float32)
        train_target_std = archive["targets_standardized"][train].astype(np.float32)
    if target_std.shape != (614, 30, 28):
        raise RuntimeError("analysis population contract failed")
    norm = json.loads(TARGET_NORM.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["safe_std"], dtype=np.float32)
    persistence_native = np.repeat(last_action[:, None, :], 30, axis=1)
    persistence_std = (persistence_native - mean[None, :, :]) / scale[None, :, :]
    p_metrics = metrics(persistence_std, persistence_native, target_std, target_native)
    train_motion = np.abs(np.diff(train_target_std[:, :, 14:28], axis=1)).mean(axis=(1, 2))
    dev_motion = np.abs(np.diff(target_std[:, :, 14:28], axis=1)).mean(axis=(1, 2))
    q25, q75 = float(np.quantile(train_motion, 0.25)), float(np.quantile(train_motion, 0.75))
    strata = np.where(dev_motion <= q25, "low", np.where(dev_motion <= q75, "mid", "high"))
    bundles: dict[str, dict[str, object]] = {}
    mapping = {
        "VF-P0": ROOT / "predictions/B3-V-D-F/VF-P0/predictions.npz", "VF-P1": ROOT / "predictions/B3-V-D-F/VF-P1/predictions.npz", "VF-P2": ROOT / "predictions/B3-V-D-F/VF-P2/predictions.npz", "VF-P3": ROOT / "predictions/B3-V-D-F/VF-P3/predictions.npz", "VF-P4": ROOT / "predictions/B3-V-D-F/VF-P4/predictions.npz",
        "FF-P0": ROOT / "predictions/B3-FUS-D-F/FF-P0/predictions.npz", "FF-P1": ROOT / "predictions/B3-FUS-D-F/FF-P1/predictions.npz", "FF-P2": ROOT / "predictions/B3-FUS-D-F/FF-P2/predictions.npz", "FF-P3": ROOT / "predictions/B3-FUS-D-F/FF-P3/predictions.npz", "FF-P4": ROOT / "predictions/B3-FUS-D-F/FF-P4/predictions.npz",
        "FF-R1": ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/FF-R1/predictions.npz", "FF-R2": ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/FF-R2/predictions.npz", "FF-R3": ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/FF-R3/predictions.npz",
    }
    dev_position = {int(value): position for position, value in enumerate(ids.tolist())}
    for cell, path in mapping.items():
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            bundle = {key: archive[key].copy() for key in archive.files}
        bundle_ids = bundle["episode_index"].astype(np.int64)
        if cell == "FF-R2":
            subset_positions = np.asarray([dev_position[int(value)] for value in bundle_ids.tolist()], dtype=np.int64)
            aligned_ids = bundle_ids
            aligned_target_std = target_std[subset_positions]
            aligned_target_native = target_native[subset_positions]
        else:
            aligned_ids = ids
            aligned_target_std = target_std
            aligned_target_native = target_native
        bundles[cell] = align_bundle({"path": path, "ids": bundle_ids, "target_std": bundle["targets_standardized"], "target_native": bundle["targets_native"], "pred_std": bundle["predictions_standardized"], "pred_native": bundle["predictions_native"], "sha256": digest(path)}, aligned_ids, aligned_target_std, aligned_target_native)
    # Add derived controls and a read-only B-STATE reference.
    bundles["PERSISTENCE"] = {"path": INPUT, "ids": ids, "target_std": target_std, "target_native": target_native, "pred_std": persistence_std, "pred_native": persistence_native, "sha256": digest(INPUT)}
    if STATE_BUNDLE.exists():
        with np.load(STATE_BUNDLE, allow_pickle=False) as archive:
            state_bundle = {key: archive[key].copy() for key in archive.files}
        bundles["B-STATE"] = align_bundle({"path": STATE_BUNDLE, "ids": state_bundle["episode_index"], "target_std": state_bundle["targets_standardized"], "target_native": state_bundle["targets_native"], "pred_std": state_bundle["predictions_standardized"], "pred_native": state_bundle["predictions_native"], "sha256": digest(STATE_BUNDLE)}, ids, target_std, target_native)
    metric_cache: dict[str, dict[str, np.ndarray]] = {}
    per_episode_rows: list[dict[str, object]] = []
    main_rows: list[dict[str, object]] = []
    for cell, bundle in bundles.items():
        metric_cache[cell] = metrics(bundle["pred_std"], bundle["pred_native"], bundle["target_std"], bundle["target_native"])
        row = {"cell": cell, "N": len(bundle["ids"]), "sha256": bundle["sha256"], "arm_full_mae": scalar(metric_cache[cell]["arm_full"]), "hand_full_mae": scalar(metric_cache[cell]["hand_full"]), "full28_full_mae": scalar(metric_cache[cell]["full28_full"]), "arm_early_mae": scalar(metric_cache[cell]["arm_early"]), "arm_middle_mae": scalar(metric_cache[cell]["arm_middle"]), "arm_late_mae": scalar(metric_cache[cell]["arm_late"]), "arm_endpoint_mae": scalar(metric_cache[cell]["arm_endpoint"]), "first_difference_mae": scalar(metric_cache[cell]["arm_first_difference"]), "native_tv_ratio": scalar(metric_cache[cell]["native_tv_ratio"]), "native_direction_agreement": scalar(metric_cache[cell]["native_direction_agreement"]), "native_direction_cosine": scalar(metric_cache[cell]["native_direction_cosine"])}
        main_rows.append(row)
        id_to_position = {int(value): position for position, value in enumerate(bundle["ids"].tolist())}
        for episode, task in zip(bundle["ids"].tolist(), tasks[[int(np.where(ids == episode)[0][0]) for episode in bundle["ids"].tolist()]].tolist()):
            position = id_to_position[int(episode)]
            per_episode_rows.append({"cell": cell, "episode_id": int(episode), "task_index": int(task), "motion_stratum": str(strata[int(np.where(ids == episode)[0][0])]), **{key: float(value[position]) if np.isfinite(value[position]) else "" for key, value in metric_cache[cell].items()}})
    write_csv(ROOT / "MAIN_RESULTS.csv", main_rows)
    write_json(ROOT / "MAIN_RESULTS.json", {"rows": main_rows, "status": "PASS", "N": 614, "high_motion_N": int(np.sum(strata == "high")), "q25": q25, "q75": q75})
    write_csv(ROOT / "PER_EPISODE.csv", per_episode_rows)
    time_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    for cell, bundle in bundles.items():
        err = np.abs(bundle["pred_std"] - bundle["target_std"])
        for time_index in range(30):
            time_rows.append({"cell": cell, "time_index": time_index, "horizon_step": time_index + 1, "arm_mae": float(err[:, time_index, 14:28].mean()), "full28_mae": float(err[:, time_index].mean())})
        for joint in range(14):
            joint_rows.append({"cell": cell, "arm_joint_index": joint, "arm_mae": float(err[:, :, 14 + joint].mean())})
    write_csv(ROOT / "PER_TIME.csv", time_rows)
    write_csv(ROOT / "PER_JOINT.csv", joint_rows)
    task_rows: list[dict[str, object]] = []
    for cell, bundle in bundles.items():
        metric_values = metric_cache[cell]["arm_full"]
        bundle_positions = {int(value): position for position, value in enumerate(bundle["ids"].tolist())}
        for task in sorted(set(tasks.tolist())):
            episode_positions = [bundle_positions[int(ids[index])] for index in range(len(ids)) if int(tasks[index]) == int(task) and int(ids[index]) in bundle_positions]
            if episode_positions:
                task_rows.append({"cell": cell, "task_index": int(task), "N": len(episode_positions), "arm_full_mae": float(np.mean(metric_values[episode_positions]))})
    write_csv(ROOT / "PER_TASK.csv", task_rows)
    horizons = {"full": slice(0, 30), "early": slice(0, 10), "middle": slice(10, 20), "late": slice(20, 30), "endpoint": 29}
    horizon_rows: list[dict[str, object]] = []
    for cell in [key for key in bundles if key.startswith(("VF-", "FF-"))]:
        m = metric_cache[cell]
        for stratum in ("all", "high"):
            mask = np.ones(len(bundles[cell]["ids"]), dtype=bool) if stratum == "all" else strata[[int(np.where(ids == value)[0][0]) for value in bundles[cell]["ids"].tolist()]] == "high"
            for horizon in horizons:
                key = "arm_" + horizon
                horizon_rows.append({"cell": cell, "stratum": stratum, "horizon": horizon, "N": int(mask.sum()), "arm_mae": scalar(m[key][mask])})
    write_csv(ROOT / "MOTION_HORIZON_RESULTS.csv", horizon_rows)
    fidelity_rows = [{"cell": cell, "N": len(bundle["ids"]), "first_difference_mae": scalar(metric_cache[cell]["arm_first_difference"]), "native_tv_ratio": scalar(metric_cache[cell]["native_tv_ratio"]), "native_endpoint_change_error": scalar(metric_cache[cell]["native_endpoint_change_error"]), "native_direction_agreement": scalar(metric_cache[cell]["native_direction_agreement"]), "native_direction_cosine": scalar(metric_cache[cell]["native_direction_cosine"])} for cell, bundle in bundles.items()]
    write_csv(ROOT / "TEMPORAL_FIDELITY.csv", fidelity_rows)
    full_boot = np.load(BOOTSTRAP, allow_pickle=False)
    if full_boot.shape != (10000, 614):
        raise RuntimeError(f"V2 bootstrap shape changed: {full_boot.shape}")
    subset_boot_rng = np.random.default_rng(20260910)
    subset_boot = subset_boot_rng.integers(0, 589, size=(10000, 589), dtype=np.int32)
    np.save(ROOT / "statistics/paired_bootstrap_indices_N614.npy", full_boot)
    np.save(ROOT / "statistics/paired_bootstrap_indices_N589.npy", subset_boot)
    baseline_for = {"VF": "VF-P0", "FF": "FF-P0"}
    effect_rows: list[dict[str, object]] = []
    for cell in [key for key in bundles if key.startswith(("VF-P", "FF-P", "FF-R")) and key not in {"VF-P0", "FF-P0"}]:
        base = baseline_for["VF" if cell.startswith("VF") else "FF"]
        bundle = bundles[cell]
        base_positions = {int(value): position for position, value in enumerate(bundles[base]["ids"].tolist())}
        common_ids = bundle["ids"]
        base_indices = np.asarray([base_positions[int(value)] for value in common_ids.tolist()], dtype=np.int64)
        base_metrics = {key: value[base_indices] for key, value in metric_cache[base].items()}
        candidate_metrics = metric_cache[cell]
        boot = subset_boot if len(common_ids) == 589 else full_boot
        for metric_name in ("arm_full", "arm_early", "arm_middle", "arm_late", "arm_endpoint", "hand_full", "full28_full", "arm_first_difference", "native_tv_ratio", "native_direction_agreement"):
            damage = candidate_metrics[metric_name] - base_metrics[metric_name]
            ci_low, ci_high = bootstrap_ci(damage, boot)
            effect_rows.append({"cell": cell, "baseline": base, "metric": metric_name, "N": len(common_ids), "baseline_mean": scalar(base_metrics[metric_name]), "candidate_mean": scalar(candidate_metrics[metric_name]), "damage": scalar(damage), "damage_ci_low": ci_low, "damage_ci_high": ci_high, "median_damage": float(np.nanmedian(damage)), "candidate_better_win_rate": float(np.mean(damage < 0)), "bootstrap_replicates": 10000, "bootstrap_unit": "episode"})
    write_csv(ROOT / "PAIRED_EFFECTS.csv", effect_rows)
    write_json(ROOT / "PAIRED_EFFECTS.json", {"rows": effect_rows, "effect_sign": "positive damage means intervention worsens vs chronological matched", "bootstrap_replicates": 10000, "status": "PASS"})
    write_json(ROOT / "statistics/BOOTSTRAP_INDEX_METADATA.json", {"N614": {"path": str(ROOT / "statistics/paired_bootstrap_indices_N614.npy"), "source": str(BOOTSTRAP), "sha256": digest(BOOTSTRAP)}, "N589": {"path": str(ROOT / "statistics/paired_bootstrap_indices_N589.npy"), "seed": 20260910, "shape": [10000, 589]}, "status": "PASS"})
    print(json.dumps({"status": "PASS", "main_cells": 13, "main_results": len(main_rows), "effect_rows": len(effect_rows), "high_motion_N": int(np.sum(strata == "high"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
