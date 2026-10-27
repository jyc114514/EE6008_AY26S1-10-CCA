#!/usr/bin/env python3
"""Accept 18 checkpoints / 72 bundles and compute locked delay analyses."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from delay_contract import DELAYS, EXP, V3, load_main


SEEDS = (20260956, 20260957, 20260958)
CELLS = ("CLEAN-N", "CLEAN-D", "CLEAN-S", "ROBUST-N", "ROBUST-D", "ROBUST-S")
DELAY_NAMES = ("D0", "D025", "D050", "D100")
GROUPS = ("arm", "hand")
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


def episode_reduce(values: np.ndarray, episode_ids: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    return np.asarray([np.nanmean(values[episode_ids == episode]) for episode in episodes], dtype=np.float64)


def metrics(pred: np.ndarray, target: np.ndarray, sigma: np.ndarray, fd_scale: np.ndarray, episode_ids: np.ndarray, episodes: np.ndarray) -> dict[str, np.ndarray]:
    err = np.abs(pred - target)
    std_err = err / sigma[None, None, :]
    fd = np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1))
    fd_std = fd / fd_scale[None, None, :]
    horizon = pred.shape[1]
    late_start = max(horizon // 2, horizon - 10)
    pred_tv = np.abs(np.diff(pred, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    tv_ratio = np.divide(pred_tv, target_tv, out=np.full(len(pred), np.nan), where=target_tv > 1e-12)
    values = {
        "arm_mae_std": std_err[:, :, :14].mean(axis=(1, 2)),
        "hand_mae_std": std_err[:, :, 14:].mean(axis=(1, 2)),
        "full_mae_std": std_err.mean(axis=(1, 2)),
        "arm_mae_native": err[:, :, :14].mean(axis=(1, 2)),
        "hand_mae_native": err[:, :, 14:].mean(axis=(1, 2)),
        "full_mae_native": err.mean(axis=(1, 2)),
        "arm_fd_native": fd[:, :, :14].mean(axis=(1, 2)),
        "hand_fd_native": fd[:, :, 14:].mean(axis=(1, 2)),
        "full_fd_native": fd.mean(axis=(1, 2)),
        "arm_fd_scaled": fd_std[:, :, :14].mean(axis=(1, 2)),
        "hand_fd_scaled": fd_std[:, :, 14:].mean(axis=(1, 2)),
        "full_fd_scaled": fd_std.mean(axis=(1, 2)),
        "full_late_mae_std": std_err[:, late_start:].mean(axis=(1, 2)),
        "full_endpoint_mae_native": err[:, -1].mean(axis=1),
        "full_tv": pred_tv,
        "target_full_tv": target_tv,
        "full_tv_ratio": tv_ratio,
        "full_tv_closeness": np.abs(tv_ratio - 1.0),
    }
    return {key: episode_reduce(value, episode_ids, episodes) for key, value in values.items()}


def hier_stats(diff: np.ndarray, seed: int, replicates: int = HIER) -> dict[str, float]:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = diff.shape[1]
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 200):
        count = min(200, replicates - start)
        chosen_seeds = rng.integers(0, diff.shape[0], size=(count, diff.shape[0]))
        total = np.zeros(count, dtype=np.float64)
        for slot in range(diff.shape[0]):
            chosen_eps = rng.integers(0, n, size=(count, n))
            total += diff[chosen_seeds[:, slot, None], chosen_eps].mean(axis=1)
        estimates[start:start + count] = total / diff.shape[0]
    p = 2 * min((1 + np.sum(estimates <= 0)) / (replicates + 1), (1 + np.sum(estimates >= 0)) / (replicates + 1))
    return {"effect": float(diff.mean()), "ci_low": float(np.quantile(estimates, .025)), "ci_high": float(np.quantile(estimates, .975)), "p_two_sided": float(p), "replicates": replicates, "seed_level": int(diff.shape[0]), "episode_level": int(n)}


def hier_relative(candidate: np.ndarray, control: np.ndarray, seed: int) -> dict[str, float]:
    """Crossed seed/episode bootstrap of candidate/control mean ratio minus 1."""
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    n = candidate.shape[1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(HIER, dtype=np.float64)
    for start in range(0, HIER, 200):
        count = min(200, HIER - start)
        chosen = rng.integers(0, candidate.shape[0], size=(count, candidate.shape[0]))
        cand_total = np.zeros(count, dtype=np.float64)
        ctrl_total = np.zeros(count, dtype=np.float64)
        for slot in range(candidate.shape[0]):
            ix = rng.integers(0, n, size=(count, n))
            cand_total += candidate[chosen[:, slot, None], ix].mean(axis=1)
            ctrl_total += control[chosen[:, slot, None], ix].mean(axis=1)
        estimates[start:start + count] = np.divide(cand_total, ctrl_total, out=np.ones_like(cand_total), where=ctrl_total > 1e-12) - 1.0
    effect = float(candidate.mean() / control.mean() - 1.0)
    p = 2 * min((1 + np.sum(estimates <= 0)) / (HIER + 1), (1 + np.sum(estimates >= 0)) / (HIER + 1))
    return {"relative_change": effect, "relative_ci_low": float(np.quantile(estimates, .025)), "relative_ci_high": float(np.quantile(estimates, .975)), "relative_p_two_sided": float(p)}


def seed_boot(diff: np.ndarray, seed: int) -> dict[str, float]:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(diff)
    estimates = np.empty(BOOT, dtype=np.float64)
    for start in range(0, BOOT, 250):
        count = min(250, BOOT - start)
        estimates[start:start + count] = diff[rng.integers(0, n, size=(count, n))].mean(axis=1)
    signs = rng.choice(np.array([-1., 1.]), size=(BOOT, n))
    p = (1 + np.sum(np.abs((signs * diff[None]).mean(axis=1)) >= abs(diff.mean()))) / (BOOT + 1)
    return {"effect": float(diff.mean()), "ci_low": float(np.quantile(estimates, .025)), "ci_high": float(np.quantile(estimates, .975)), "p_signflip": float(p), "replicates": BOOT, "episode_level": n}


def signflip_episode(diff_by_seed: np.ndarray, seed: int) -> float:
    values = np.asarray(diff_by_seed, dtype=np.float64).mean(axis=0)
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    exceed = 0
    for start in range(0, BOOT, 250):
        count = min(250, BOOT - start)
        signs = rng.choice(np.array([-1., 1.]), size=(count, len(values)))
        exceed += int(np.sum(np.abs((signs * values[None]).mean(axis=1)) >= observed))
    return float((1 + exceed) / (BOOT + 1))


def apply_holm(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["p_two_sided"]))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(rows[index]["p_two_sided"]) * (len(rows) - rank)))
        rows[index]["holm_rank"] = rank + 1
        rows[index]["holm_p"] = running


def main() -> int:
    data, normalizer = load_main()
    dev = np.flatnonzero(data["split_code"] == 1)
    train = np.flatnonzero(data["split_code"] == 0)
    episodes = np.unique(data["episode_index"][dev])
    episode_ids = data["episode_index"][dev]
    target = data["targets_native"][dev, :60].astype(np.float64)
    anchor = data["anchor_command_native"][dev].astype(np.float64)
    sigma = np.asarray(normalizer["target_sigma"], dtype=np.float64)
    fd_scale = np.asarray(normalizer["safe_fd_scale"], dtype=np.float64)
    pred_by: dict[tuple[int, str, int], np.ndarray] = {}
    metric_by: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    acceptance_rows: list[dict[str, object]] = []
    age_rows: list[dict[str, object]] = []
    source_query = data["history_query_timestamps"][dev].astype(np.float64)
    for seed in SEEDS:
        for cell in CELLS:
            run_id = f"ODR-{cell}-H60-S{seed}"
            run_dir = EXP / f"runs/{run_id}/formal"
            complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
            config = json.loads((run_dir / "RUN_CONFIG_LOCKED.json").read_text(encoding="utf-8"))
            for di, delay in enumerate(DELAYS):
                pred_path = EXP / f"predictions/{run_id}/delay_{di}.npz"
                meta_path = EXP / f"predictions/{run_id}/delay_{di}.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                with np.load(pred_path, allow_pickle=False) as z:
                    bundle = {k: z[k].copy() for k in z.files}
                expected_cutoff = data["anchor_timestamp"][dev].astype(np.float64) - float(delay)
                selected_ts = bundle["observation_selected_timestamps"].astype(np.float64)
                age = bundle["observation_age_seconds"].astype(np.float64)
                allowed_time = np.minimum(source_query, expected_cutoff[:, None])
                ids_ok = np.array_equal(bundle["anchor_id"], data["anchor_id"][dev]) and np.array_equal(bundle["episode_index"], episode_ids)
                timeline_ok = bool(np.all(selected_ts <= allowed_time + 1e-6) and np.allclose(age, np.maximum(0.0, source_query - selected_ts), atol=1e-6))
                shape_ok = bundle["predictions_native"].shape == (len(dev), 60, 28) and np.isfinite(bundle["predictions_native"]).all()
                ck_ok = complete.get("status") == "PASS" and complete.get("epoch") == 40 and complete.get("optimizer_steps") == 11120 and complete.get("checkpoint_reload_prediction_max_abs") == 0.0 and complete.get("initial_persistence_max_abs") == 0.0
                ok = bool(meta.get("status") == "PASS" and meta.get("state_unchanged") is True and meta.get("final_test_access") == 0 and config.get("final_test_access") == 0 and ck_ok and ids_ok and timeline_ok and shape_ok)
                acceptance_rows.append({"run_id": run_id, "seed": seed, "cell": cell, "delay_index": di, "delay_seconds": float(delay), "shape": str(list(bundle["predictions_native"].shape)), "checkpoint_epoch": complete.get("epoch"), "optimizer_steps": complete.get("optimizer_steps"), "ids_ok": ids_ok, "timeline_ok": timeline_ok, "finite_shape_ok": bool(shape_ok), "checkpoint_ok": bool(ck_ok), "status": "PASS" if ok else "FAIL", "final_test_access": 0})
                if not ok:
                    raise RuntimeError(f"prediction acceptance failed: {run_id} delay={delay}")
                prediction = bundle["predictions_native"].astype(np.float64)
                pred_by[(seed, cell, di)] = prediction
                metric_by[(seed, cell, di)] = metrics(prediction, target, sigma, fd_scale, episode_ids, episodes)
                ep_age = np.asarray([age[episode_ids == episode].mean() for episode in episodes])
                age_rows.append({"seed": seed, "cell": cell, "delay_index": di, "delay_seconds": float(delay), "age_mean_seconds": float(np.mean(age)), "age_median_seconds": float(np.median(age)), "age_p95_seconds": float(np.quantile(age, .95)), "age_max_seconds": float(np.max(age)), "episode_equal_mean_age_seconds": float(ep_age.mean()), "observation_steps": int(age.size), "final_test_access": 0})
    write_csv(EXP / "analysis/PREDICTION_ACCEPTANCE.csv", acceptance_rows)
    write_json(EXP / "analysis/PREDICTION_ACCEPTANCE.json", {"status": "PASS", "accepted": "72/72", "primary_dev_episodes": len(episodes), "primary_dev_windows": len(dev), "final_test_access": 0})
    write_csv(EXP / "analysis/OBSERVATION_AGE_DISTRIBUTION.csv", age_rows)

    baseline_rows: list[dict[str, object]] = []
    baseline_windows = np.repeat(anchor[:, None, :], 60, axis=1)
    baseline_metrics = metrics(baseline_windows, target, sigma, fd_scale, episode_ids, episodes)
    for key, values in baseline_metrics.items():
        baseline_rows.append({"horizon_steps": 60, "metric": key, "episode_equal_value": float(values.mean()), "delay_invariant": True, "final_test_access": 0})
    write_csv(EXP / "analysis/BASELINE_RECONCILIATION.csv", baseline_rows)
    invariant = all(np.array_equal(baseline_windows, np.repeat(anchor[:, None, :], 60, axis=1)) for _ in DELAYS)
    write_json(EXP / "analysis/BASELINE_RECONCILIATION.json", {"status": "PASS" if invariant else "FAIL", "persistence_unchanged_across_delays": invariant, "arm_mae_std": float(baseline_metrics["arm_mae_std"].mean()), "hand_mae_std": float(baseline_metrics["hand_mae_std"].mean()), "full_mae_std": float(baseline_metrics["full_mae_std"].mean()), "full_fd_native": float(baseline_metrics["full_fd_native"].mean()), "final_test_access": 0})

    results_by_seed: list[dict[str, object]] = []
    for seed in SEEDS:
        for cell in CELLS:
            for di, delay in enumerate(DELAYS):
                met = metric_by[(seed, cell, di)]
                row: dict[str, object] = {"seed": seed, "cell": cell, "train_policy": "uniform_four_delays" if cell.startswith("ROBUST") else "delay_zero_only", "delay_index": di, "delay_seconds": float(delay), "N_episode": len(episodes)}
                row.update({key: float(value.mean()) for key, value in met.items()})
                row["arm_skill_vs_persistence"] = float(1 - row["arm_mae_std"] / baseline_metrics["arm_mae_std"].mean())
                row["hand_skill_vs_persistence"] = float(1 - row["hand_mae_std"] / baseline_metrics["hand_mae_std"].mean())
                row["full_skill_vs_persistence"] = float(1 - row["full_mae_std"] / baseline_metrics["full_mae_std"].mean())
                row["final_test_access"] = 0
                results_by_seed.append(row)
    write_csv(EXP / "analysis/RESULTS_BY_SEED_AND_DELAY.csv", results_by_seed)

    agg_rows: list[dict[str, object]] = []
    for cell in CELLS:
        for di, delay in enumerate(DELAYS):
            subset = [r for r in results_by_seed if r["cell"] == cell and r["delay_index"] == di]
            row = {"cell": cell, "delay_index": di, "delay_seconds": float(delay), "seed_count": 3}
            for metric_name in metric_by[(SEEDS[0], cell, di)]:
                values = np.asarray([float(r[metric_name]) for r in subset])
                row[metric_name + "_mean"] = float(values.mean())
                row[metric_name + "_sd"] = float(values.std(ddof=1))
            agg_rows.append(row)
    write_csv(EXP / "analysis/AGGREGATE_CURVES.csv", agg_rows)

    # Stress estimand: equal mean over the three prespecified nonzero delays,
    # then paired episode mean; the same episode remains the resampling unit.
    stress_metrics: dict[tuple[int, str, str], np.ndarray] = {}
    for seed in SEEDS:
        for cell in CELLS:
            for metric_name in ("arm_mae_std", "hand_mae_std", "arm_fd_native", "hand_fd_native", "full_mae_std", "full_fd_native"):
                stress_metrics[(seed, cell, metric_name)] = np.mean(np.stack([metric_by[(seed, cell, di)][metric_name] for di in (1, 2, 3)]), axis=0)
    contrasts = (("ROBUST-D_vs_ROBUST-N", "ROBUST-N", "ROBUST-D"), ("ROBUST-D_vs_ROBUST-S", "ROBUST-S", "ROBUST-D"), ("ROBUST-D_vs_CLEAN-D", "CLEAN-D", "ROBUST-D"))
    primary_rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    for contrast, control, candidate in contrasts:
        for group in ("arm_mae_std", "hand_mae_std"):
            diff = np.stack([stress_metrics[(seed, control, group)] - stress_metrics[(seed, candidate, group)] for seed in SEEDS])
            row = {"contrast": contrast, "control": control, "candidate": candidate, "group": group, "estimand": "episode mean of equal-weight delay 0.25/0.50/1.00 MAE", **hier_stats(diff, 300000 + len(primary_rows)), "p_signflip_episode": signflip_episode(diff, 310000 + len(primary_rows)), "final_test_access": 0}
            primary_rows.append(row)
            for i, seed in enumerate(SEEDS):
                per_seed_rows.append({"contrast": contrast, "control": control, "candidate": candidate, "group": group, "seed": seed, **seed_boot(diff[i], 320000 + len(per_seed_rows)), "final_test_access": 0})
    apply_holm(primary_rows)
    write_csv(EXP / "analysis/PRIMARY_ROBUSTNESS_EFFECTS.csv", primary_rows)
    write_csv(EXP / "analysis/PRIMARY_ROBUSTNESS_BY_SEED.csv", per_seed_rows)

    fd_rows: list[dict[str, object]] = []
    for contrast, control, candidate in contrasts:
        for group in ("arm_fd_native", "hand_fd_native"):
            diff = np.stack([stress_metrics[(seed, control, group)] - stress_metrics[(seed, candidate, group)] for seed in SEEDS])
            fd_rows.append({"contrast": contrast, "control": control, "candidate": candidate, "group": group, "estimand": "episode mean native FD over equal-weight delayed conditions", **hier_stats(diff, 330000 + len(fd_rows)), "p_signflip_episode": signflip_episode(diff, 340000 + len(fd_rows)), "final_test_access": 0})
    apply_holm(fd_rows)
    write_csv(EXP / "analysis/FD_EFFECTS.csv", fd_rows)

    guard_rows: list[dict[str, object]] = []
    for group in ("arm_mae_std", "hand_mae_std", "arm_fd_native", "hand_fd_native"):
        diff = np.stack([metric_by[(seed, "ROBUST-D", 0)][group] - metric_by[(seed, "CLEAN-D", 0)][group] for seed in SEEDS])
        candidate = np.stack([metric_by[(seed, "ROBUST-D", 0)][group] for seed in SEEDS])
        control = np.stack([metric_by[(seed, "CLEAN-D", 0)][group] for seed in SEEDS])
        relative = hier_relative(candidate, control, 355000 + len(guard_rows))
        guard_rows.append({"contrast": "ROBUST-D_minus_CLEAN-D_at_delay0", "group": group, "positive_means": "robust model larger error", **hier_stats(diff, 350000 + len(guard_rows)), **relative, "practical_margin_reference": 0.01, "ci_upper_within_plus_1pct": bool(relative["relative_ci_high"] <= 0.01), "final_test_access": 0})
    write_csv(EXP / "analysis/CLEAN_GUARDRAILS.csv", guard_rows)

    video_gain_rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    for group in ("arm_mae_std", "hand_mae_std"):
        per_delay = []
        for di, delay in enumerate(DELAYS):
            gain = np.stack([metric_by[(seed, "ROBUST-N", di)][group] - metric_by[(seed, "ROBUST-D", di)][group] for seed in SEEDS])
            per_delay.append(gain)
            video_gain_rows.append({"group": group, "delay_index": di, "delay_seconds": float(delay), **hier_stats(gain, 360000 + len(video_gain_rows)), "positive_means": "dynamic video lower MAE than no video", "final_test_access": 0})
        delayed_minus_clean = np.mean(np.stack(per_delay[1:], axis=0), axis=0) - per_delay[0]
        interaction_rows.append({"contrast": "mean_G_delayed_minus_G0", "group": group, "definition": "G(delta)=MAE(ROBUST-N,delta)-MAE(ROBUST-D,delta); equal mean over 0.25/0.50/1.00 minus G(0)", "positive_means": "video gain grows with delayed feedback", **hier_stats(delayed_minus_clean, 370000 + len(interaction_rows)), "p_signflip_episode": signflip_episode(delayed_minus_clean, 380000 + len(interaction_rows)), "final_test_access": 0})
    write_csv(EXP / "analysis/VIDEO_GAIN_BY_DELAY.csv", video_gain_rows)
    write_csv(EXP / "analysis/VIDEO_DELAY_INTERACTIONS.csv", interaction_rows)

    clean_rows: list[dict[str, object]] = []
    for control, candidate, contrast in (("CLEAN-N", "CLEAN-D", "CLEAN-D_vs_CLEAN-N"), ("CLEAN-S", "CLEAN-D", "CLEAN-D_vs_CLEAN-S")):
        for group in ("arm_mae_std", "hand_mae_std"):
            diff = np.stack([metric_by[(seed, control, 0)][group] - metric_by[(seed, candidate, 0)][group] for seed in SEEDS])
            clean_rows.append({"contrast": contrast, "control": control, "candidate": candidate, "group": group, "delay_seconds": 0.0, **hier_stats(diff, 390000 + len(clean_rows)), "final_test_access": 0})
    write_csv(EXP / "analysis/CLEAN_INPUT_EFFECTS.csv", clean_rows)

    # Task/joint support with episode-equal aggregation at each task and joint.
    support_rows: list[dict[str, object]] = []
    task_ids = np.unique(data["original_task_index"][dev])
    for seed in SEEDS:
        for cell in CELLS:
            for di, delay in enumerate(DELAYS):
                pred = pred_by[(seed, cell, di)]
                std_err = np.abs(pred - target) / sigma[None, None, :]
                for task in task_ids:
                    mask = data["original_task_index"][dev] == task
                    eps = np.unique(episode_ids[mask])
                    row = {"seed": seed, "cell": cell, "delay_index": di, "delay_seconds": float(delay), "level": "task", "index": int(task), "N_episode": len(eps), "N_window": int(mask.sum())}
                    for key, sl in (("arm_mae_std", slice(0, 14)), ("hand_mae_std", slice(14, 28)), ("full_mae_std", slice(0, 28))):
                        row[key] = float(np.mean([std_err[mask][episode_ids[mask] == ep, :, sl].mean() for ep in eps]))
                    support_rows.append(row)
                for joint in range(28):
                    by_episode = episode_reduce(std_err[:, :, joint].mean(axis=1), episode_ids, episodes)
                    support_rows.append({"seed": seed, "cell": cell, "delay_index": di, "delay_seconds": float(delay), "level": "joint", "index": joint, "N_episode": len(episodes), "N_window": len(dev), "arm_mae_std": float(by_episode.mean()) if joint < 14 else "", "hand_mae_std": float(by_episode.mean()) if joint >= 14 else "", "full_mae_std": float(by_episode.mean())})
    write_csv(EXP / "analysis/TASK_JOINT_SUPPORT.csv", support_rows)

    # Motion strata thresholds are derived from primary train episodes only.
    train_values = []
    for episode in np.unique(data["episode_index"][train]):
        rows = train[data["episode_index"][train] == episode]
        train_values.append(np.mean(np.abs(np.diff(data["targets_native"][rows], axis=1)), axis=(0, 1)))
    train_values = np.asarray(train_values); q25 = np.quantile(train_values, .25, axis=0); q75 = np.quantile(train_values, .75, axis=0)
    dev_values = []
    for episode in episodes:
        rows = dev[episode_ids == episode]
        dev_values.append(np.mean(np.abs(np.diff(data["targets_native"][rows], axis=1)), axis=(0, 1)))
    dev_values = np.asarray(dev_values); motion_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for cell in CELLS:
            for di, delay in enumerate(DELAYS):
                pred = pred_by[(seed, cell, di)]; std_err = np.abs(pred - target) / sigma[None, None, :]
                for group, dims in (("arm", slice(0, 14)), ("hand", slice(14, 28))):
                    activity = dev_values[:, dims].mean(axis=1); low_thr = float(q25[dims].mean()); high_thr = float(q75[dims].mean())
                    for stratum, mask in (("low", activity <= low_thr), ("mid", (activity > low_thr) & (activity < high_thr)), ("high", activity >= high_thr)):
                        selected_eps = episodes[mask]; window_mask = np.isin(episode_ids, selected_eps)
                        motion_rows.append({"seed": seed, "cell": cell, "delay_index": di, "delay_seconds": float(delay), "group": group, "stratum": stratum, "N_episode": int(mask.sum()), "N_window": int(window_mask.sum()), "mae_std": float(std_err[window_mask, :, dims].mean())})
    write_csv(EXP / "analysis/MOTION_STRATA_RESULTS.csv", motion_rows)

    # Delay schedule verification: robust N/D/S share per-window delay counts.
    schedule_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        curves = {}
        for cell in ("ROBUST-N", "ROBUST-D", "ROBUST-S", "CLEAN-N", "CLEAN-D", "CLEAN-S"):
            run_id = f"ODR-{cell}-H60-S{seed}"
            payload = json.loads((EXP / f"runs/{run_id}/formal/EPOCH_CURVES.json").read_text(encoding="utf-8")); curves[cell] = payload["rows"]
        for epoch_idx in range(40):
            robust_counts = [curves[cell][epoch_idx]["delay_counts"] for cell in ("ROBUST-N", "ROBUST-D", "ROBUST-S")]
            same = robust_counts[0] == robust_counts[1] == robust_counts[2]
            clean_ok = all(curves[cell][epoch_idx]["delay_counts"] == [17757, 0, 0, 0] for cell in ("CLEAN-N", "CLEAN-D", "CLEAN-S"))
            schedule_rows.append({"seed": seed, "epoch": epoch_idx + 1, "robust_counts": json.dumps(robust_counts[0]), "robust_N_D_S_same": same, "clean_only_zero_delay": clean_ok, "robust_total": sum(robust_counts[0]), "final_test_access": 0})
    write_csv(EXP / "analysis/TRAIN_DELAY_SCHEDULE_QA.csv", schedule_rows)

    # Actual host/GPU time and concurrency from watcher records.
    resource_rows: list[dict[str, object]] = []
    for kind, pattern in (("stage0", "watcher_ODR-*-STAGE0.json"), ("formal", "watcher_ODR-*-FORMAL.json"), ("inference", "watcher_ODR-*-INFER.json")):
        for path in sorted((EXP / "logs").glob(pattern)):
            item = json.loads(path.read_text(encoding="utf-8")); start = item.get("launch_utc") or item.get("started_utc"); end = item.get("completed_utc"); wall = float("nan")
            if start and end: wall = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600
            resource_rows.append({"job_id": item.get("job_id"), "kind": kind, "status": item.get("status"), "physical_gpu": item.get("physical_gpu"), "wall_gpu_hours": wall, "peak_contract_mib": item.get("measured_peak_mib"), "oom_requeues": item.get("oom_requeue_count", 0), "non_child_process_signaled": item.get("non_child_process_signaled"), "final_test_access": 0})
    write_csv(EXP / "analysis/RESOURCE_COSTS.csv", resource_rows)
    metadata = {"status": "PASS", "training_runs": 18, "prediction_bundles": 72, "accepted_predictions": 72, "primary_dev_episodes": len(episodes), "primary_dev_windows": len(dev), "bootstrap_per_seed": BOOT, "crossed_seed_episode_bootstrap": HIER, "primary_holm_items": len(primary_rows), "secondary_fd_holm_items": len(fd_rows), "final_test_access": 0}
    write_json(EXP / "analysis/ANALYSIS_METADATA.json", metadata)
    print(json.dumps({"status": "PASS", "predictions": 72, "primary_effects": len(primary_rows), "fd_effects": len(fd_rows), "interactions": len(interaction_rows), "task_joint_rows": len(support_rows), "schedule_rows": len(schedule_rows), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
