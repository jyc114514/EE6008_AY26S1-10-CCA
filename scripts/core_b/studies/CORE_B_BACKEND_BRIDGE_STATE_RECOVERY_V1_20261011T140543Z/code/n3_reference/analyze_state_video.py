#!/usr/bin/env python3
"""CPU-only harvest for the locked 54-cell state/video study."""

from __future__ import annotations
import os

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
SEEDS = (20260953, 20260954, 20260955)
HORIZONS = (30, 60)
STATES = ("FULL", "OBS", "ANCHOR")
VIDEOS = ("N", "D", "S")
CELLS = tuple(f"{state}-{video}" for state in STATES for video in VIDEOS)
GROUPS = ("arm_mae_std", "hand_mae_std")
BOOT = 10_000
HIER = 20_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def episode_reduce(values: np.ndarray, episode_ids: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    return np.asarray([np.nanmean(values[episode_ids == episode]) for episode in episodes], dtype=np.float64)


def metric_episode(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray, sigma: np.ndarray, episode_ids: np.ndarray, episodes: np.ndarray) -> dict[str, np.ndarray]:
    error = np.abs(pred - target)
    std_error = error / sigma[None, None, :]
    fd_error = np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1))
    horizon = pred.shape[1]
    late_start = max(horizon // 2, horizon - 10)
    tv_pred = np.abs(np.diff(pred, axis=1)).sum(axis=1).mean(axis=1)
    tv_target = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    tv_ratio = np.divide(tv_pred, tv_target, out=np.full(len(pred), np.nan), where=tv_target > 1e-12)
    values = {
        "arm_mae_std": std_error[:, :, :14].mean(axis=(1, 2)),
        "hand_mae_std": std_error[:, :, 14:].mean(axis=(1, 2)),
        "full_mae_std": std_error.mean(axis=(1, 2)),
        "arm_mae_native": error[:, :, :14].mean(axis=(1, 2)),
        "hand_mae_native": error[:, :, 14:].mean(axis=(1, 2)),
        "full_mae_native": error.mean(axis=(1, 2)),
        "arm_fd_native": fd_error[:, :, :14].mean(axis=(1, 2)),
        "hand_fd_native": fd_error[:, :, 14:].mean(axis=(1, 2)),
        "full_fd_native": fd_error.mean(axis=(1, 2)),
        "arm_fd_scaled": (fd_error[:, :, :14] / sigma[:14]).mean(axis=(1, 2)),
        "hand_fd_scaled": (fd_error[:, :, 14:] / sigma[14:]).mean(axis=(1, 2)),
        "full_fd_scaled": (fd_error / sigma).mean(axis=(1, 2)),
        "full_anchor_to_first": np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)).mean(axis=1),
        "full_endpoint_mae_native": error[:, -1].mean(axis=1),
        "full_late_mae_std": std_error[:, late_start:].mean(axis=(1, 2)),
        "full_tv": tv_pred,
        "target_full_tv": tv_target,
        "full_tv_ratio": tv_ratio,
        "full_tv_closeness": np.abs(tv_ratio - 1.0),
    }
    return {key: episode_reduce(value, episode_ids, episodes) for key, value in values.items()}


def hierarchical(diff_by_seed: np.ndarray, seed: int) -> dict[str, float]:
    diff = np.asarray(diff_by_seed, dtype=np.float64)
    n = diff.shape[1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(HIER, dtype=np.float64)
    for start in range(0, HIER, 200):
        count = min(200, HIER - start)
        chosen = rng.integers(0, 3, size=(count, 3))
        total = np.zeros(count, dtype=np.float64)
        for slot in range(3):
            indices = rng.integers(0, n, size=(count, n))
            total += diff[chosen[:, slot, None], indices].mean(axis=1)
        estimates[start : start + count] = total / 3.0
    p = 2 * min((1 + np.sum(estimates <= 0)) / (HIER + 1), (1 + np.sum(estimates >= 0)) / (HIER + 1))
    return {"effect": float(diff.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "p_two_sided": float(p), "replicates": HIER, "seed_level": 3, "episode_level": n}


def paired_bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOT, dtype=np.float64)
    n = len(values)
    for start in range(0, BOOT, 250):
        count = min(250, BOOT - start)
        estimates[start : start + count] = values[rng.integers(0, n, size=(count, n))].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOT, n))
    p = (1 + np.sum(np.abs((signs * values[None]).mean(axis=1)) >= abs(values.mean()))) / (BOOT + 1)
    return {"effect": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "p_signflip": float(p), "replicates": BOOT, "episode_level": n}


def aggregate_signflip(diff_by_seed: np.ndarray, seed: int) -> float:
    values = np.asarray(diff_by_seed, dtype=np.float64).mean(axis=0)
    rng = np.random.default_rng(seed)
    exceed = 0
    observed = abs(values.mean())
    for start in range(0, BOOT, 250):
        count = min(250, BOOT - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.abs((signs * values[None]).mean(axis=1)) >= observed))
    return float((1 + exceed) / (BOOT + 1))


def holm(rows: list[dict[str, object]], p_key: str = "p_two_sided") -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][p_key]))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(rows[index][p_key]) * (len(rows) - rank)))
        rows[index]["holm_rank"] = rank + 1
        rows[index]["holm_p"] = running


def accepted_predictions(data: dict[str, np.ndarray]) -> tuple[dict[tuple[int, int, str], np.ndarray], np.ndarray, dict[str, object]]:
    dev = np.flatnonzero(data["split_code"] == 1)
    predictions: dict[tuple[int, int, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    accepted = 0
    for seed in SEEDS:
        for horizon in HORIZONS:
            for cell in CELLS:
                run_id = f"SIV-{cell}-H{horizon}-S{seed}"
                run_dir = EXP / f"runs/{run_id}/formal"
                complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
                meta = json.loads((EXP / f"predictions/{run_id}/PREDICTION_EXPORT_COMPLETE.json").read_text(encoding="utf-8"))
                with np.load(EXP / f"predictions/{run_id}/primary.npz", allow_pickle=False) as bundle:
                    pred = bundle["predictions_native"].copy()
                    prediction_ids = bundle["anchor_id"].copy()
                ok = complete.get("status") == "PASS" and complete.get("epoch") == 40 and complete.get("optimizer_steps") == 11120 and complete.get("final_test_access") == 0 and meta.get("status") == "PASS" and meta.get("state_unchanged") is True and meta.get("final_test_access") == 0 and pred.shape == (len(dev), horizon, 28) and np.array_equal(prediction_ids, data["anchor_id"][dev]) and np.isfinite(pred).all()
                accepted += int(ok)
                rows.append({"run_id": run_id, "seed": seed, "horizon_steps": horizon, "cell": cell, "checkpoint_epoch": complete.get("epoch"), "optimizer_steps": complete.get("optimizer_steps"), "prediction_shape": str(list(pred.shape)), "state_unchanged": meta.get("state_unchanged"), "status": "PASS" if ok else "FAIL", "final_test_access": 0})
                if ok:
                    predictions[(seed, horizon, cell)] = pred.astype(np.float64)
    write_csv(EXP / "analysis/PREDICTION_ACCEPTANCE.csv", rows)
    summary = {"status": "PASS" if accepted == 54 else "FAIL", "accepted": f"{accepted}/54", "primary_dev_windows": len(dev), "primary_dev_episodes": len(np.unique(data["episode_index"][dev])), "final_test_access": 0}
    write_json(EXP / "analysis/PREDICTION_ACCEPTANCE.json", summary)
    if accepted != 54:
        raise RuntimeError(summary)
    return predictions, dev, summary


def main() -> int:
    data = load_npz(V3 / "data/V3_MAIN_INPUTS.npz")
    supplementary = load_npz(V3 / "data/V3_SUPPLEMENTARY_1S_INPUTS.npz")
    normalizer = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizer["target_sigma"], dtype=np.float64)
    dev = np.flatnonzero(data["split_code"] == 1)
    episodes = np.unique(data["episode_index"][dev])
    target = data["targets_native"][dev].astype(np.float64)
    anchor = data["anchor_command_native"][dev].astype(np.float64)
    predictions, dev, acceptance = accepted_predictions(data)
    metrics_by_key: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    baseline_by_horizon: dict[int, dict[str, np.ndarray]] = {}
    baseline_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        y = target[:, :horizon]
        persistence = np.repeat(anchor[:, None, :], horizon, axis=1)
        baseline = metric_episode(persistence, y, anchor, sigma, data["episode_index"][dev], episodes)
        baseline_by_horizon[horizon] = baseline
        baseline_rows.append({"horizon_steps": horizon, "N_episode": len(episodes), "N_window": len(dev), **{key: float(np.mean(value)) for key, value in baseline.items()}, "final_test_access": 0})
        for seed in SEEDS:
            for cell in CELLS:
                metrics_by_key[(seed, horizon, cell)] = metric_episode(predictions[(seed, horizon, cell)], y, anchor, sigma, data["episode_index"][dev], episodes)
    write_csv(EXP / "analysis/BASELINE_RESULTS.csv", baseline_rows)

    result_rows: list[dict[str, object]] = []
    metric_keys = list(next(iter(metrics_by_key.values())).keys())
    for seed in SEEDS:
        for horizon in HORIZONS:
            baseline = baseline_by_horizon[horizon]
            for cell in CELLS:
                value = metrics_by_key[(seed, horizon, cell)]
                result_rows.append({"seed": seed, "horizon_steps": horizon, "cell": cell, "state_level": cell.split("-")[0], "video_level": cell.split("-")[1], "N_episode": len(episodes), **{key: float(np.mean(values)) for key, values in value.items()}, "arm_skill_vs_persistence": float(1 - np.mean(value["arm_mae_std"]) / np.mean(baseline["arm_mae_std"])), "hand_skill_vs_persistence": float(1 - np.mean(value["hand_mae_std"]) / np.mean(baseline["hand_mae_std"])), "full_skill_vs_persistence": float(1 - np.mean(value["full_mae_std"]) / np.mean(baseline["full_mae_std"])), "final_test_access": 0})
    write_csv(EXP / "analysis/RESULTS_BY_SEED.csv", result_rows)

    aggregate_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for cell in CELLS:
            subset = [row for row in result_rows if row["horizon_steps"] == horizon and row["cell"] == cell]
            row: dict[str, object] = {"horizon_steps": horizon, "cell": cell, "state_level": cell.split("-")[0], "video_level": cell.split("-")[1], "seed_count": 3}
            for key in metric_keys + ["arm_skill_vs_persistence", "hand_skill_vs_persistence", "full_skill_vs_persistence"]:
                values = np.asarray([float(item[key]) for item in subset])
                row[key + "_mean"] = float(np.mean(values))
                row[key + "_sd"] = float(np.std(values, ddof=1))
            aggregate_rows.append(row)
    write_csv(EXP / "analysis/AGGREGATE_RESULTS.csv", aggregate_rows)

    primary_rows: list[dict[str, object]] = []
    secondary_fd_rows: list[dict[str, object]] = []
    per_seed_effect_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for state in STATES:
            for video_control in ("N", "S"):
                control = f"{state}-{video_control}"
                candidate = f"{state}-D"
                contrast = f"{candidate}_vs_{control}"
                for group in GROUPS:
                    diff = np.stack([metrics_by_key[(seed, horizon, control)][group] - metrics_by_key[(seed, horizon, candidate)][group] for seed in SEEDS])
                    family_row = {"family": "primary_mae", "contrast": contrast, "state_level": state, "control": control, "candidate": candidate, "horizon_steps": horizon, "group": group, **hierarchical(diff, 204000 + len(primary_rows)), "p_signflip_episode": aggregate_signflip(diff, 205000 + len(primary_rows)), "final_test_access": 0}
                    primary_rows.append(family_row)
                    for slot, seed in enumerate(SEEDS):
                        per_seed = paired_bootstrap(diff[slot], 206000 + len(per_seed_effect_rows))
                        per_seed_effect_rows.append({"family": "primary_mae", "contrast": contrast, "state_level": state, "control": control, "candidate": candidate, "horizon_steps": horizon, "group": group, "seed": seed, **per_seed, "final_test_access": 0})
                    fd_group = group.replace("mae_std", "fd_native")
                    fd_diff = np.stack([metrics_by_key[(seed, horizon, control)][fd_group] - metrics_by_key[(seed, horizon, candidate)][fd_group] for seed in SEEDS])
                    secondary_fd_rows.append({"family": "secondary_fd", "contrast": contrast, "state_level": state, "control": control, "candidate": candidate, "horizon_steps": horizon, "group": group.replace("mae_std", "fd_native"), **hierarchical(fd_diff, 207000 + len(secondary_fd_rows)), "p_signflip_episode": aggregate_signflip(fd_diff, 208000 + len(secondary_fd_rows)), "final_test_access": 0})
    holm(primary_rows)
    holm(secondary_fd_rows)
    write_csv(EXP / "analysis/PRIMARY_VIDEO_EFFECTS.csv", primary_rows)
    write_csv(EXP / "analysis/FD_RESULTS.csv", secondary_fd_rows)
    write_csv(EXP / "analysis/VIDEO_EFFECTS_BY_SEED.csv", per_seed_effect_rows)

    gain_rows: list[dict[str, object]] = []
    for row in primary_rows:
        if str(row["control"]).endswith("-N"):
            gain_rows.append({"state_level": row["state_level"], "horizon_steps": row["horizon_steps"], "group": row["group"], "D_vs_N_effect": row["effect"], "D_vs_N_ci_low": row["ci_low"], "D_vs_N_ci_high": row["ci_high"], "D_vs_N_p": row["p_two_sided"], "D_vs_N_holm_p": row["holm_p"], "final_test_access": 0})
        if str(row["control"]).endswith("-S"):
            target_row = next(item for item in gain_rows if item["state_level"] == row["state_level"] and item["horizon_steps"] == row["horizon_steps"] and item["group"] == row["group"])
            target_row.update({"D_vs_S_effect": row["effect"], "D_vs_S_ci_low": row["ci_low"], "D_vs_S_ci_high": row["ci_high"], "D_vs_S_p": row["p_two_sided"], "D_vs_S_holm_p": row["holm_p"]})
    write_csv(EXP / "analysis/VIDEO_GAINS_BY_STATE_LEVEL.csv", gain_rows)

    interaction_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for group in GROUPS:
            for low, high in (("FULL", "OBS"), ("OBS", "ANCHOR")):
                d_low = np.stack([metrics_by_key[(seed, horizon, f"{low}-N")][group] - metrics_by_key[(seed, horizon, f"{low}-D")][group] for seed in SEEDS])
                d_high = np.stack([metrics_by_key[(seed, horizon, f"{high}-N")][group] - metrics_by_key[(seed, horizon, f"{high}-D")][group] for seed in SEEDS])
                diff = d_high - d_low
                interaction_rows.append({"family": "information_interaction", "contrast": f"video_gain({high})_minus_video_gain({low})", "low_state": low, "high_state": high, "horizon_steps": horizon, "group": group, **hierarchical(diff, 209000 + len(interaction_rows)), "p_signflip_episode": aggregate_signflip(diff, 210000 + len(interaction_rows)), "final_test_access": 0})
    holm(interaction_rows)
    write_csv(EXP / "analysis/INFORMATION_INTERACTIONS.csv", interaction_rows)

    task_joint_rows: list[dict[str, object]] = []
    task_values = np.unique(data["original_task_index"][dev])
    for seed in SEEDS:
        for horizon in HORIZONS:
            y = target[:, :horizon]
            for cell in CELLS:
                pred = predictions[(seed, horizon, cell)]
                error = np.abs(pred - y) / sigma[None, None, :]
                for task in task_values:
                    mask = data["original_task_index"][dev] == task
                    task_joint_rows.append({"seed": seed, "horizon_steps": horizon, "cell": cell, "level": "task", "index": int(task), "N_window": int(mask.sum()), "N_episode": int(len(np.unique(data["episode_index"][dev][mask]))), "arm_mae_std": float(error[mask, :, :14].mean()), "hand_mae_std": float(error[mask, :, 14:].mean()), "full_mae_std": float(error[mask].mean()), "final_test_access": 0})
                for joint in range(28):
                    task_joint_rows.append({"seed": seed, "horizon_steps": horizon, "cell": cell, "level": "joint", "index": joint, "N_window": len(pred), "N_episode": len(episodes), "arm_mae_std": float(error[:, :, joint].mean()) if joint < 14 else "", "hand_mae_std": float(error[:, :, joint].mean()) if joint >= 14 else "", "full_mae_std": float(error[:, :, joint].mean()), "final_test_access": 0})
    write_csv(EXP / "analysis/TASK_JOINT_SUPPORT.csv", task_joint_rows)

    # Motion strata are defined only from primary-train target dynamics, matching V3.
    train = np.flatnonzero(data["split_code"] == 0)
    train_episode_values = []
    for episode in np.unique(data["episode_index"][train]):
        rows = train[data["episode_index"][train] == episode]
        train_episode_values.append(np.mean(np.abs(np.diff(data["targets_native"][rows], axis=1)), axis=(0, 1)))
    train_episode_values = np.asarray(train_episode_values)
    q25 = np.quantile(train_episode_values, 0.25, axis=0)
    q75 = np.quantile(train_episode_values, 0.75, axis=0)
    dev_motion = []
    for episode in episodes:
        rows = dev[data["episode_index"][dev] == episode]
        dev_motion.append(np.mean(np.abs(np.diff(data["targets_native"][rows], axis=1)), axis=(0, 1)))
    dev_motion = np.asarray(dev_motion)
    motion_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            y = target[:, :horizon]
            for cell in CELLS:
                pred = predictions[(seed, horizon, cell)]
                err = np.abs(pred - y) / sigma[None, None, :]
                for group, values, lower, upper in (("arm", dev_motion[:, :14].mean(axis=1), float(q25[:14].mean()), float(q75[:14].mean())), ("hand", dev_motion[:, 14:].mean(axis=1), float(q25[14:].mean()), float(q75[14:].mean()))):
                    for stratum in ("low", "mid", "high"):
                        mask = values <= lower if stratum == "low" else values >= upper if stratum == "high" else (values > lower) & (values < upper)
                        window_mask = np.isin(data["episode_index"][dev], episodes[mask])
                        motion_rows.append({"seed": seed, "horizon_steps": horizon, "cell": cell, "group": group, "stratum": stratum, "N_episode": int(mask.sum()), "N_window": int(window_mask.sum()), "mae_std": float(err[window_mask, :, :14 if group == 'arm' else 14:].mean()), "final_test_access": 0})
    write_csv(EXP / "analysis/MOTION_RESULTS.csv", motion_rows)

    primary_set = set(map(int, np.unique(data["episode_index"][dev])))
    supplementary_set = set(map(int, np.unique(supplementary["episode_index"])))
    union = sorted(primary_set | supplementary_set)
    union_rows = [{"episode_index": episode, "in_primary": int(episode in primary_set), "in_supplementary": int(episode in supplementary_set), "category": "intersection" if episode in primary_set and episode in supplementary_set else "supplementary_only" if episode in supplementary_set else "primary_only", "final_test_access": 0} for episode in union]
    write_csv(EXP / "analysis/SUPPLEMENTARY_ID_UNION.csv", union_rows)
    write_json(EXP / "analysis/SUPPLEMENTARY_ID_UNION.json", {"primary_episodes": len(primary_set), "supplementary_episodes": len(supplementary_set), "intersection": len(primary_set & supplementary_set), "supplementary_only": len(supplementary_set - primary_set), "union": len(union), "not_independent_667": True, "final_test_access": 0})

    supplementary_rows: list[dict[str, object]] = []
    supp_episode_order = np.unique(supplementary["episode_index"])
    supp_target = supplementary["targets_native"].astype(np.float64)
    supp_anchor = supplementary["anchor_command_native"].astype(np.float64)
    for seed in SEEDS:
        for cell in CELLS:
            run_id = f"SIV-{cell}-H30-S{seed}"
            with np.load(EXP / f"predictions/{run_id}/supplementary_1s.npz", allow_pickle=False) as bundle:
                pred = bundle["predictions_native"].astype(np.float64)
            value = metric_episode(pred, supp_target, supp_anchor, sigma, supplementary["episode_index"], supp_episode_order)
            supplementary_rows.append({"seed": seed, "horizon_steps": 30, "cell": cell, "N_episode": len(supp_episode_order), "intersection_episode_count": len(set(map(int, supp_episode_order)) & primary_set), "supplementary_only_episode_count": len(set(map(int, supp_episode_order)) - primary_set), "full_mae_std": float(np.mean(value["full_mae_std"])), "full_fd_native": float(np.mean(value["full_fd_native"])), "final_test_access": 0})
    write_csv(EXP / "analysis/SUPPLEMENTARY_RESULTS.csv", supplementary_rows)

    resource_rows: list[dict[str, object]] = []
    for kind, pattern in (("stage0", "watcher_SIV-*-STAGE0.json"), ("formal", "watcher_SIV-*-FORMAL.json"), ("inference", "watcher_SIV-*-INFER.json")):
        for path in sorted((EXP / "logs").glob(pattern)):
            value = json.loads(path.read_text(encoding="utf-8"))
            start = value.get("launch_utc") or value.get("started_utc")
            end = value.get("completed_utc")
            wall = float("nan")
            if start and end:
                wall = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600.0
            resource_rows.append({"job_id": value.get("job_id"), "kind": kind, "status": value.get("status"), "physical_gpu": value.get("physical_gpu"), "wall_gpu_hours": wall, "peak_contract_mib": value.get("measured_peak_mib"), "oom_requeues": value.get("oom_requeue_count", 0), "non_child_process_signaled": value.get("non_child_process_signaled"), "final_test_access": 0})
    write_csv(EXP / "analysis/RESOURCE_COSTS.csv", resource_rows)

    meta = {"status": "PASS", "primary_dev_episodes": len(episodes), "primary_dev_windows": len(dev), "supplementary_episodes": len(supplementary_set), "supplementary_union": len(union), "seeds": list(SEEDS), "primary_video_family_count": len(primary_rows), "secondary_fd_family_count": len(secondary_fd_rows), "interaction_family_count": len(interaction_rows), "bootstrap_per_seed": BOOT, "hierarchical_replicates": HIER, "final_test_access": 0}
    write_json(EXP / "analysis/ANALYSIS_METADATA.json", meta)
    print(json.dumps({"status": "PASS", "accepted": acceptance["accepted"], "primary_rows": len(primary_rows), "fd_rows": len(secondary_fd_rows), "interaction_rows": len(interaction_rows), "task_joint_rows": len(task_joint_rows), "supp_union": len(union), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
