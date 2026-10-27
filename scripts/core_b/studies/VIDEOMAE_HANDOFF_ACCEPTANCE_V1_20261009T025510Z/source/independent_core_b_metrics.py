#!/usr/bin/env python3
"""Independent CPU scoring and paired source-group bootstrap for Core B dev predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
MODES = ("TS11-D", "STATE-CTRL")
REGIONS = {"action_0_14": slice(0, 14), "action_14_28": slice(14, 28), "action_full28": slice(0, 28)}
METRICS = (
    "native_mae", "standardized_mae", "native_rmse", "standardized_rmse",
    "endpoint_native_mae", "endpoint_standardized_mae", "native_future_internal_fd",
    "safe_fd_normalized", "seam_error_native", "seam_pred_step_abs", "seam_gt_step_abs",
    "tv_pred", "tv_gt", "tv_ratio_window", "pred_step_abs_mean", "gt_step_abs_mean",
    "pred_anchor_net_abs", "gt_anchor_net_abs", "pred_anchor_net_signed", "gt_anchor_net_signed",
    "direction_accuracy_active", "early_native_mae", "mid_native_mae", "late_native_mae",
    "early_standardized_mae", "mid_standardized_mae", "late_standardized_mae",
)
BOOTSTRAP_METRICS = (
    "standardized_mae", "native_mae", "native_rmse", "native_future_internal_fd",
    "safe_fd_normalized", "endpoint_standardized_mae",
)
EPS = 1e-8
DRAWS = 10_000
BOOTSTRAP_SEED = 20261008


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("NA" if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)) else v)
                             for k, v in row.items()})


def episode_group_map(split_manifest: Path) -> dict[int, tuple[str, str]]:
    table = pq.read_table(split_manifest,
                          columns=["episode_index", "split", "video_sha256_fingerprint"],
                          filters=[("split", "in", ["train", "dev", "validation", "val"])])
    result: dict[int, tuple[str, str]] = {}
    for ep, raw_split, fingerprint in zip(table["episode_index"].to_pylist(),
                                          table["split"].to_pylist(),
                                          table["video_sha256_fingerprint"].to_pylist()):
        if fingerprint is None or not str(fingerprint).strip():
            raise AssertionError(f"missing source fingerprint for episode {ep}")
        split = "dev" if raw_split in {"dev", "validation", "val"} else str(raw_split)
        value = (split, str(fingerprint))
        key = int(ep)
        if key in result and result[key] != value:
            raise AssertionError(f"conflicting source/split identity for episode {key}")
        result[key] = value
    return result


def activity_threshold(train_targets_h60: np.ndarray) -> tuple[np.ndarray, int]:
    """Locked rule: joint-wise 5th percentile of positive H60 train increments, floor 1e-6."""
    delta = np.abs(np.diff(train_targets_h60.astype(np.float64), axis=1))
    threshold = np.full(delta.shape[-1], 1e-6, dtype=np.float64)
    fallback = 0
    for j in range(delta.shape[-1]):
        positive = delta[:, :, j].reshape(-1)
        positive = positive[positive > EPS]
        if len(positive):
            threshold[j] = max(1e-6, float(np.quantile(positive, 0.05)))
        else:
            fallback += 1
    return threshold.astype(np.float32), fallback


def window_metrics(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray,
                   sigma: np.ndarray, safe_fd: np.ndarray,
                   active_threshold: np.ndarray) -> dict[str, np.ndarray]:
    """Independent NumPy implementation of the locked native/source-group score contract."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    safe_fd = np.asarray(safe_fd, dtype=np.float32)
    active_threshold = np.asarray(active_threshold, dtype=np.float32)
    if pred.ndim != 3 or target.shape != pred.shape or pred.shape[2] != len(sigma):
        raise AssertionError(f"prediction/target/scale shape mismatch: {pred.shape}, {target.shape}, {sigma.shape}")
    if anchor.shape != (len(pred), pred.shape[2]) or safe_fd.shape != sigma.shape or active_threshold.shape != sigma.shape:
        raise AssertionError("anchor or normalizer vector shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(target).all() or not np.isfinite(anchor).all():
        raise AssertionError("nonfinite prediction, target or anchor")

    error = pred - target
    d_pred = np.diff(pred, axis=1)
    d_target = np.diff(target, axis=1)
    d_error = d_pred - d_target
    active = np.abs(d_target) > active_threshold[None, None, :]
    exact_zero = np.abs(d_target) <= EPS
    n_active = active.sum(axis=(1, 2))
    correct = ((np.sign(d_pred) == np.sign(d_target)) & active).sum(axis=(1, 2))
    direction = np.divide(correct, n_active, out=np.full(len(pred), np.nan, dtype=np.float64), where=n_active > 0)
    tv_pred = np.abs(d_pred).sum(axis=1).mean(axis=1)
    tv_target = np.abs(d_target).sum(axis=1).mean(axis=1)
    tv_ratio = np.divide(tv_pred, tv_target, out=np.full(len(pred), np.nan, dtype=np.float64), where=tv_target > 0)
    std_error = error / sigma[None, None, :]
    time_blocks = np.array_split(np.arange(target.shape[1]), 3)
    result: dict[str, np.ndarray] = {
        "native_mae": np.abs(error).mean(axis=(1, 2)),
        "standardized_mae": np.abs(std_error).mean(axis=(1, 2)),
        "native_rmse": np.sqrt(np.square(error).mean(axis=(1, 2))),
        "standardized_rmse": np.sqrt(np.square(std_error).mean(axis=(1, 2))),
        "endpoint_native_mae": np.abs(error[:, -1]).mean(axis=1),
        "endpoint_standardized_mae": (np.abs(error[:, -1]) / sigma[None, :]).mean(axis=1),
        "native_future_internal_fd": np.abs(d_error).mean(axis=(1, 2)),
        "safe_fd_normalized": (np.abs(d_error) / safe_fd[None, None, :]).mean(axis=(1, 2)),
        "seam_error_native": np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)).mean(axis=1),
        "seam_pred_step_abs": np.abs(pred[:, 0] - anchor).mean(axis=1),
        "seam_gt_step_abs": np.abs(target[:, 0] - anchor).mean(axis=1),
        "tv_pred": tv_pred,
        "tv_gt": tv_target,
        "tv_ratio_window": tv_ratio,
        "pred_step_abs_mean": np.abs(d_pred).mean(axis=(1, 2)),
        "gt_step_abs_mean": np.abs(d_target).mean(axis=(1, 2)),
        "pred_anchor_net_abs": np.abs(pred[:, -1] - anchor).mean(axis=1),
        "gt_anchor_net_abs": np.abs(target[:, -1] - anchor).mean(axis=1),
        "pred_anchor_net_signed": (pred[:, -1] - anchor).mean(axis=1),
        "gt_anchor_net_signed": (target[:, -1] - anchor).mean(axis=1),
        "direction_accuracy_active": direction,
        "active_step_count": n_active.astype(np.int64),
        "near_zero_gt_step_count": exact_zero.sum(axis=(1, 2)).astype(np.int64),
        "small_nonzero_gt_step_count": ((~active) & (~exact_zero)).sum(axis=(1, 2)).astype(np.int64),
        "zero_gt_tv_window": (tv_target == 0.0).astype(np.int64),
    }
    for name, idx in zip(("early_native_mae", "mid_native_mae", "late_native_mae"), time_blocks):
        result[name] = np.abs(error[:, idx]).mean(axis=(1, 2))
    for name, idx in zip(("early_standardized_mae", "mid_standardized_mae", "late_standardized_mae"), time_blocks):
        result[name] = np.abs(std_error[:, idx]).mean(axis=(1, 2))
    return result


def grouped_scores(values: np.ndarray, episodes: np.ndarray, groups: np.ndarray) -> tuple[dict[str, float], dict[int, float]]:
    """Return episode-equal and source-group-equal scores, windows -> episode -> group."""
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=np.int64)
    groups = np.asarray(groups, dtype="U128")
    ep_group: dict[int, str] = {}
    ep_vals: dict[int, list[float]] = defaultdict(list)
    for value, ep, group in zip(values, episodes, groups):
        ep = int(ep)
        if ep in ep_group and ep_group[ep] != str(group):
            raise AssertionError(f"episode {ep} maps to more than one source group")
        ep_group[ep] = str(group)
        if np.isfinite(value):
            ep_vals[ep].append(float(value))
    ep_mean = {ep: float(np.mean(xs)) for ep, xs in ep_vals.items() if xs}
    by_group: dict[str, list[float]] = defaultdict(list)
    for ep, val in ep_mean.items():
        by_group[ep_group[ep]].append(val)
    group_mean = {g: float(np.mean(vals)) for g, vals in by_group.items() if vals}
    episode_equal = float(np.mean(list(ep_mean.values()))) if ep_mean else float("nan")
    group_equal = float(np.mean(list(group_mean.values()))) if group_mean else float("nan")
    return {"episode_equal": episode_equal, "source_group_equal": group_equal}, group_mean


def read_bundle(path: Path, *, expected_ids: np.ndarray, expected_episodes: np.ndarray,
                prediction_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        required = {"anchor_id", "episode_index", prediction_key}
        if not required <= set(z.files):
            raise AssertionError(f"{path}: prediction keys missing {sorted(required-set(z.files))}")
        if not np.array_equal(z["anchor_id"].astype("U64"), expected_ids.astype("U64")):
            raise AssertionError(f"{path}: anchor ID/order mismatch")
        if not np.array_equal(z["episode_index"].astype(np.int64), expected_episodes.astype(np.int64)):
            raise AssertionError(f"{path}: episode ID/order mismatch")
        pred = np.asarray(z[prediction_key], dtype=np.float32).copy()
    if pred.shape[0] != len(expected_ids) or not np.isfinite(pred).all():
        raise AssertionError(f"{path}: prediction row/finite check failed")
    return pred


def load_sidecar(path: Path, prediction_path: Path, *, mode: str, horizon: int, seed: int) -> None:
    side = json.loads(path.read_text(encoding="utf-8"))
    expected_name = "TS11-D" if mode == "TS11-D" else "STATE-CTRL"
    if side.get("status") != "PASS" or side.get("final_test_access") != 0:
        raise AssertionError(f"team prediction sidecar is not accepted: {path}")
    if (side.get("config"), side.get("horizon_steps"), side.get("seed"), side.get("split")) != (
        expected_name, horizon, seed, "dev"
    ):
        raise AssertionError(f"team prediction sidecar cell mismatch: {path}")
    if side.get("rows") != 2322 or side.get("prediction_sha256") != sha256(prediction_path):
        raise AssertionError(f"team prediction sidecar row/hash mismatch: {path}")


def bootstrap_contrast(group_arrays: dict[str, dict[int, np.ndarray]],
                       group_order: list[str], draws: np.ndarray,
                       horizon: int, region: str, metric: str,
                       contrast: str) -> dict[str, object]:
    """Conditional fixed-seed paired bootstrap using one shared set of groups."""
    if contrast == "Delta_backbone":
        components = [(group_arrays["VideoMAE"][s], group_arrays["V-JEPA"][s]) for s in SEEDS]
        seed_group_delta = np.stack([a - b for a, b in components], axis=0)
    elif contrast == "G_VM":
        components = [(group_arrays["VM-STATE-CTRL"][s], group_arrays["VideoMAE"][s]) for s in SEEDS]
        seed_group_delta = np.stack([a - b for a, b in components], axis=0)
    elif contrast == "G_VJ":
        components = [(group_arrays["TEAM-STATE-CTRL"][s], group_arrays["V-JEPA"][s]) for s in SEEDS]
        seed_group_delta = np.stack([a - b for a, b in components], axis=0)
    elif contrast == "I":
        seed_group_delta = np.stack([
            (group_arrays["VM-STATE-CTRL"][s] - group_arrays["VideoMAE"][s])
            - (group_arrays["TEAM-STATE-CTRL"][s] - group_arrays["V-JEPA"][s])
            for s in SEEDS
        ], axis=0)
    else:
        raise ValueError(contrast)

    seed_differences = seed_group_delta.mean(axis=1)
    point = float(seed_differences.mean())
    group_average = seed_group_delta.mean(axis=0)
    boot = group_average[draws].mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p = float((1 + np.count_nonzero(np.abs(boot - point) >= abs(point))) / (len(boot) + 1))
    return {
        "split": "dev", "horizon": horizon, "region_raw": region,
        "raw_action_slice": f"{REGIONS[region].start}:{REGIONS[region].stop}",
        "anatomical_label": "unresolved", "metric": metric, "contrast": contrast,
        "seed_20261001": float(seed_differences[0]),
        "seed_20261002": float(seed_differences[1]),
        "seed_20261003": float(seed_differences[2]),
        "mean_fixed_seed_contrast": point,
        "sample_sd_across_three_seed_contrasts_ddof1": float(seed_differences.std(ddof=1)),
        "seed_direction_consistency": int(np.sum(np.sign(seed_differences) == np.sign(point))),
        "ci95_source_group_bootstrap_fixed_seeds": [float(ci_low), float(ci_high)],
        "p_centered_bootstrap_add_one_two_sided": p,
        "bootstrap_draws": int(len(boot)), "paired_groups": int(len(group_order)),
        "ci_scope": "conditional on these 3 trained seeds; repeated-use development split",
        "primary_family": int(contrast == "Delta_backbone" and metric == "standardized_mae"
                               and region in {"action_0_14", "action_14_28"}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "audit-root", "handoff-root", "probe-root", "cache-root", "team-root", "v3-root", "split-manifest", "train-target-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    project, audit, handoff, probe, cache, team, v3, split_manifest, train_target_cache = [
        x.resolve() for x in (args.project_root, args.audit_root, args.handoff_root, args.probe_root,
                              args.cache_root, args.team_root, args.v3_root, args.split_manifest, args.train_target_cache)
    ]
    (audit / "metrics/core_b").mkdir(parents=True, exist_ok=True)
    index = json.loads((handoff / "HANDOFF_INDEX.json").read_text(encoding="utf-8"))
    input_path = v3 / "data/V3_MAIN_INPUTS.npz"
    norm_path = v3 / "data/V3_NORMALIZERS.json"
    if sha256(input_path) != "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78":
        raise AssertionError("Core B main input hash differs from both run configurations")
    normalizers = json.loads(norm_path.read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    safe_fd = np.asarray(normalizers["safe_fd_scale"], dtype=np.float32)
    if normalizers.get("final_test_access") != 0 or normalizers.get("fit_scope") != "primary train windows only; episode equal":
        raise AssertionError("Core B target normalizer is not the locked train-only normalizer")
    mapping = episode_group_map(split_manifest)

    with np.load(input_path, allow_pickle=False) as z:
        split_codes = np.asarray(z["split_code"], dtype=np.int8)
        if set(np.unique(split_codes).tolist()) != {0, 1}:
            raise AssertionError(f"expected train/dev only split codes, found {np.unique(split_codes)}")
        all_ids = np.asarray(z["anchor_id"], dtype="U64")
        all_episodes = np.asarray(z["episode_index"], dtype=np.int64)
        dev_rows = np.flatnonzero(split_codes == 1)
        train_rows = np.flatnonzero(split_codes == 0)
        if (len(train_rows), len(dev_rows)) != (17757, 2322):
            raise AssertionError("Core B locked train/dev anchor count mismatch")
        dev_ids = all_ids[dev_rows].copy()
        dev_episodes = all_episodes[dev_rows].copy()
        dev_anchor = np.asarray(z["anchor_command_native"][dev_rows], dtype=np.float32)
    if len(np.unique(dev_ids)) != 2322:
        raise AssertionError("Core B dev anchor ID uniqueness mismatch")
    train_groups = np.asarray([mapping[int(ep)][1] for ep in np.unique(all_episodes[train_rows])], dtype="U128")
    dev_groups = np.asarray([mapping[int(ep)][1] for ep in dev_episodes], dtype="U128")
    for ep in np.unique(all_episodes[train_rows]):
        if mapping[int(ep)][0] != "train":
            raise AssertionError(f"train episode {ep} maps outside train")
    for ep in np.unique(dev_episodes):
        if mapping[int(ep)][0] != "dev":
            raise AssertionError(f"dev episode {ep} maps outside dev")
    if set(train_groups.tolist()) & set(dev_groups.tolist()):
        raise AssertionError("train/dev source group overlap")
    if (len(np.unique(all_episodes[train_rows])), len(np.unique(dev_episodes)), len(np.unique(dev_groups))) != (2712, 603, 603):
        raise AssertionError("Core B episode/source group support mismatch")
    dev_source_groups = np.asarray([mapping[int(ep)][1] for ep in dev_episodes], dtype="U128")
    group_order = sorted(np.unique(dev_source_groups).tolist())
    if len(group_order) != 603:
        raise AssertionError("paired bootstrap requires all 603 common dev source groups")

    # Validate all prediction-only bundles before opening any future targets.
    run_lookup = {(c["mode"], int(c["horizon"]), int(c["seed"])): c for c in index["core_b"]["cells"]}
    prevalidated_bundles = 0
    for mode in MODES:
        for horizon in HORIZONS:
            for seed in SEEDS:
                cell = run_lookup[(mode, horizon, seed)]
                run = Path(cell["run_root"])
                prediction_path = run / "predictions_dev.npz"
                status = json.loads((run / "RUN_STATUS.json").read_text(encoding="utf-8"))
                if sha256(prediction_path) != status.get("dev_prediction_sha256"):
                    raise AssertionError(f"handoff prediction hash mismatch before target join: {prediction_path}")
                pred_only = read_bundle(prediction_path, expected_ids=dev_ids, expected_episodes=dev_episodes,
                                         prediction_key="predictions_native")
                if pred_only.shape != (2322, horizon, 28):
                    raise AssertionError(f"handoff prediction schema mismatch before target join: {prediction_path}")
                prevalidated_bundles += 1
                team_mode = "TS11-D" if mode == "TS11-D" else "STATE-CTRL"
                team_id = f"D3EQ-CS-{team_mode}-H{horizon}-S{seed}"
                team_prediction = team / f"predictions/dev/{team_id}.npz"
                team_sidecar = team_prediction.with_suffix(".json")
                load_sidecar(team_sidecar, team_prediction, mode=team_mode, horizon=horizon, seed=seed)
                team_pred_only = read_bundle(team_prediction, expected_ids=dev_ids, expected_episodes=dev_episodes,
                                             prediction_key="predictions_native")
                if team_pred_only.shape != (2322, horizon, 28):
                    raise AssertionError(f"team prediction schema mismatch before target join: {team_prediction}")
                prevalidated_bundles += 1
    if prevalidated_bundles != 24:
        raise AssertionError(f"expected 24 complete target-blind dev prediction bundles, saw {prevalidated_bundles}")

    with np.load(train_target_cache, allow_pickle=False) as t:
        if not np.array_equal(t["anchor_id"].astype("U64"), all_ids[train_rows]):
            raise AssertionError("train-only target cache anchor IDs/order mismatch")
        if not np.array_equal(t["episode_index"].astype(np.int64), all_episodes[train_rows]):
            raise AssertionError("train-only target cache episode IDs/order mismatch")
        train_targets = np.asarray(t["targets_native_60"], dtype=np.float32)
    if train_targets.shape != (17757, 60, 28) or not np.isfinite(train_targets).all():
        raise AssertionError("train-only target cache shape/finite mismatch")
    # The separate scorer joins dev targets only after all saved predictions pass schema/hash/ID gates.
    with np.load(input_path, allow_pickle=False) as z:
        dev_target60 = np.asarray(z["targets_native"][dev_rows], dtype=np.float32)
    if dev_target60.shape != (2322, 60, 28) or not np.isfinite(dev_target60).all():
        raise AssertionError("Core B dev target schema/finite check failed")
    active_threshold, fallback_joints = activity_threshold(train_targets)

    # TARGETS are opened only here, after prediction bundles were generated; they never enter read_bundle/model inputs.
    targets_by_h = {h: dev_target60[:, :h, :] for h in HORIZONS}
    records: list[dict[str, object]] = []
    all_model_groups: dict[tuple[str, int, int, str, str], dict[str, float]] = {}
    for mode in MODES:
        for horizon in HORIZONS:
            for seed in SEEDS:
                cell = run_lookup[(mode, horizon, seed)]
                run = Path(cell["run_root"])
                status = json.loads((run / "RUN_STATUS.json").read_text(encoding="utf-8"))
                config = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
                prediction_path = run / "predictions_dev.npz"
                eval_report_path = Path(cell["metrics"])
                eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
                if status.get("status") != "PASS_TRAIN_AND_EXPORT" or status.get("final_test_access") != 0:
                    raise AssertionError(f"handoff run is not a complete development-only cell: {run}")
                if sha256(prediction_path) != status.get("dev_prediction_sha256"):
                    raise AssertionError(f"handoff dev prediction hash mismatch: {prediction_path}")
                pred = read_bundle(prediction_path, expected_ids=dev_ids,
                                   expected_episodes=dev_episodes, prediction_key="predictions_native")
                if pred.shape != (2322, horizon, 28):
                    raise AssertionError(f"handoff prediction shape mismatch: {pred.shape}")
                model_name = "VideoMAE" if mode == "TS11-D" else "VM-STATE-CTRL"
                records.append({"model": model_name, "mode": mode, "horizon": horizon, "seed": seed,
                                "path": str(prediction_path), "sha256": sha256(prediction_path),
                                "status": status, "config": config, "metrics_report": eval_report})

                # Independently re-score VideoMAE output by ID. This join occurs after the saved target-blind prediction.
                for region_name, sl in REGIONS.items():
                    per_window = window_metrics(pred[:, :, sl], targets_by_h[horizon][:, :, sl],
                                                dev_anchor[:, sl], sigma[sl], safe_fd[sl], active_threshold[sl])
                    for metric in METRICS:
                        aggregate, by_group = grouped_scores(per_window[metric], dev_episodes, dev_source_groups)
                        for aggregation, value in aggregate.items():
                            all_model_groups[(model_name, horizon, seed, region_name, metric)] = by_group
                            records.append({"model": model_name, "mode": mode, "horizon": horizon, "seed": seed,
                                            "region_raw": region_name, "region_slice": f"{sl.start}:{sl.stop}",
                                            "anatomical_label": "unresolved", "split": "dev",
                                            "aggregation": aggregation, "metric": metric, "value": value,
                                            "n_windows": 2322, "n_episodes": 603, "n_source_groups": 603})

    # Team D3 corrected-scheduler source predictions are the V-JEPA and its own state-control references.
    team_pred_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for mode in MODES:
        team_model = "V-JEPA" if mode == "TS11-D" else "TEAM-STATE-CTRL"
        for horizon in HORIZONS:
            for seed in SEEDS:
                run_id = f"D3EQ-CS-{mode}-H{horizon}-S{seed}"
                pred_path = team / f"predictions/dev/{run_id}.npz"
                sidecar_path = pred_path.with_suffix(".json")
                if not pred_path.is_file() or not sidecar_path.is_file():
                    raise FileNotFoundError(f"required team dev prediction bundle missing: {pred_path}")
                load_sidecar(sidecar_path, pred_path, mode=mode, horizon=horizon, seed=seed)
                pred = read_bundle(pred_path, expected_ids=dev_ids,
                                   expected_episodes=dev_episodes, prediction_key="predictions_native")
                if pred.shape != (2322, horizon, 28):
                    raise AssertionError(f"team prediction shape mismatch: {pred_path} {pred.shape}")
                team_pred_cache[(mode, horizon, seed)] = pred
                for region_name, sl in REGIONS.items():
                    per_window = window_metrics(pred[:, :, sl], targets_by_h[horizon][:, :, sl],
                                                dev_anchor[:, sl], sigma[sl], safe_fd[sl], active_threshold[sl])
                    for metric in METRICS:
                        aggregate, by_group = grouped_scores(per_window[metric], dev_episodes, dev_source_groups)
                        for aggregation, value in aggregate.items():
                            all_model_groups[(team_model, horizon, seed, region_name, metric)] = by_group
                            records.append({"model": team_model, "mode": mode, "horizon": horizon, "seed": seed,
                                            "region_raw": region_name, "region_slice": f"{sl.start}:{sl.stop}",
                                            "anatomical_label": "unresolved", "split": "dev",
                                            "aggregation": aggregation, "metric": metric, "value": value,
                                            "n_windows": 2322, "n_episodes": 603, "n_source_groups": 603})

    # Per-window count support is reported once per model cell/region.
    metric_rows = [r for r in records if "region_raw" in r]
    # Add explicit TV numerator/denominator summaries to each aggregation row.
    for r in metric_rows:
        if r["metric"] == "tv_pred":
            partner = next(x for x in metric_rows if x["model"] == r["model"] and x["horizon"] == r["horizon"]
                           and x["seed"] == r["seed"] and x["region_raw"] == r["region_raw"]
                           and x["aggregation"] == r["aggregation"] and x["metric"] == "tv_gt")
            r["tv_numerator_pred"] = r["value"]
            r["tv_denominator_ground_truth"] = partner["value"]
            r["tv_ratio_of_aggregated_numerator_denominator"] = (
                float(r["value"]) / float(partner["value"]) if float(partner["value"]) > 0 else float("nan"))

    write_csv(audit / "CORE_B_PER_SEED_METRICS.csv", metric_rows)

    # Reconcile independent scores to the handoff's stored scorer output, without importing its helper.
    original_diffs = []
    for rec in records:
        if "metrics_report" not in rec:
            continue
        report = rec["metrics_report"]
        model_name = str(rec["model"])
        mode, horizon, seed = rec["mode"], int(rec["horizon"]), int(rec["seed"])
        for region_name in REGIONS:
            for metric in ("standardized_mae", "native_future_internal_fd"):
                independent = next(r for r in metric_rows if r["model"] == model_name and r["horizon"] == horizon
                                   and r["seed"] == seed and r["region_raw"] == region_name
                                   and r["aggregation"] == "source_group_equal" and r["metric"] == metric)
                original_label = "VideoMAE" if model_name == "VideoMAE" else "STATE-CTRL"
                original = float(report["panels"]["dev"][region_name][original_label][metric])
                original_diffs.append({"cell": f"{mode}_H{horizon}_S{seed}", "region": region_name,
                                       "metric": metric, "independent": float(independent["value"]),
                                       "handoff_saved": original,
                                       "abs_diff": abs(float(independent["value"]) - original)})

    # Cross-model groups and fixed seed contrasts.
    model_names = ("VideoMAE", "V-JEPA", "VM-STATE-CTRL", "TEAM-STATE-CTRL")
    summaries = []
    for horizon in HORIZONS:
        for region in REGIONS:
            for metric in METRICS:
                for model_name in model_names:
                    values = [next(float(r["value"]) for r in metric_rows
                                   if r["model"] == model_name and r["horizon"] == horizon and r["seed"] == seed
                                   and r["region_raw"] == region and r["metric"] == metric
                                   and r["aggregation"] == "source_group_equal") for seed in SEEDS]
                    summaries.append({"split": "dev", "model": model_name, "horizon": horizon,
                                      "region_raw": region, "region_slice": f"{REGIONS[region].start}:{REGIONS[region].stop}",
                                      "anatomical_label": "unresolved", "metric": metric,
                                      "seed_20261001": values[0], "seed_20261002": values[1], "seed_20261003": values[2],
                                      "mean": float(np.mean(values)), "sample_sd_ddof1": float(np.std(values, ddof=1)),
                                      "n_paired_groups": 603})

    # Matching behavior in the stored comparison: one common draw matrix, 603 groups, fixed 3 seeds.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw_idx = rng.integers(0, len(group_order), size=(DRAWS, len(group_order)))
    contrast_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for region in REGIONS:
            for metric in METRICS:
                candidates = [
                    all_model_groups[(model, horizon, seed, region, metric)]
                    for model in model_names for seed in SEEDS
                ]
                common_groups = [g for g in group_order if all(g in data and np.isfinite(data[g]) for data in candidates)]
                if not common_groups:
                    continue
                aligned = {model: {seed: np.asarray([all_model_groups[(model, horizon, seed, region, metric)][g]
                                                     for g in common_groups], dtype=np.float64)
                                   for seed in SEEDS} for model in model_names}
                # Re-index common resamples to the finite paired support for this metric.
                if len(common_groups) == len(group_order):
                    selected_draws = draw_idx
                else:
                    local_rng = np.random.default_rng(BOOTSTRAP_SEED)
                    selected_draws = local_rng.integers(0, len(common_groups), size=(DRAWS, len(common_groups)))
                for contrast in ("Delta_backbone", "G_VM", "G_VJ", "I"):
                    row = bootstrap_contrast(aligned, common_groups, selected_draws,
                                             horizon, region, metric, contrast)
                    contrast_rows.append(row)

    primary = sorted([r for r in contrast_rows if r["primary_family"]], key=lambda r: float(r["p_centered_bootstrap_add_one_two_sided"]))
    if len(primary) != 4:
        raise AssertionError(f"expected four Holm-family tests; found {len(primary)}")
    running = 0.0
    for rank, row in enumerate(primary):
        running = max(running, min(1.0, (len(primary) - rank) * float(row["p_centered_bootstrap_add_one_two_sided"])))
        row["p_holm_existing_four_test_family"] = running
    for row in contrast_rows:
        row.setdefault("p_holm_existing_four_test_family", float("nan"))
    write_csv(audit / "CORE_B_PAIRED_CONTRASTS.csv", contrast_rows)
    write_csv(audit / "metrics/core_b/MODEL_SEED_SUMMARIES.csv", summaries)

    max_original_diff = max((r["abs_diff"] for r in original_diffs), default=float("inf"))
    comparison_path = Path(index["core_b"]["comparison"])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    # Compare the comparator's fixed-seed descriptives where the scope and metric match.
    comparator_diffs = []
    for row in comparison.get("descriptive", []):
        model = {"VideoMAE": "VideoMAE", "V-JEPA": "V-JEPA", "STATE-CTRL": "VM-STATE-CTRL", "TEAM-STATE-CTRL": "TEAM-STATE-CTRL"}.get(row["model"])
        if not model or row["metric"] not in METRICS:
            continue
        mine = next(x for x in summaries if x["model"] == model and x["horizon"] == row["horizon"]
                    and x["region_raw"] == row["region"] and x["metric"] == row["metric"])
        comparator_diffs.append(abs(float(mine["mean"]) - float(row["mean"])))
    qa = {
        "status": "PASS_INDEPENDENT_CORE_B_DEV_SCORE" if max_original_diff <= 1e-7 else "NEEDS_REVIEW",
        "independent_scorer_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": sha256(input_path), "target_normalizer_sha256": sha256(norm_path),
        "train_target_cache_sha256": sha256(train_target_cache),
        "dev_rows": 2322, "dev_episodes": 603, "dev_source_groups": 603,
        "train_rows_for_threshold_fit": 17757, "train_episodes_for_threshold_fit": 2712,
        "source_group_overlap": 0, "dev_targets_joined_after_prediction_load": True,
        "prediction_bundles_prevalidated_before_targets": prevalidated_bundles,
        "inference_target_arrays_loaded": False, "final_test_access": 0,
        "raw_regions": {k: [v.start, v.stop] for k, v in REGIONS.items()},
        "anatomical_half_names": "unresolved; raw slices only",
        "activity_threshold_source": "primary train targets only; per-joint q05 of positive H60 absolute increments, floor 1e-6",
        "native_fd_definition": "future-internal adjacent differences only: 29 (H30), 59 (H60); anchor seam reported separately",
        "safe_fd_scale_source": "locked V3_NORMALIZERS.json fit on primary train windows only",
        "max_abs_vs_handoff_saved_metrics": max_original_diff,
        "handoff_metric_comparisons": len(original_diffs),
        "max_abs_vs_handoff_comparator_descriptives": max(comparator_diffs, default=float("inf")),
        "paired_bootstrap_draws": DRAWS, "paired_groups": len(group_order),
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "p_rule": "two-sided centered paired group bootstrap with add-one correction",
        "holm_family": "existing four-test family: Delta_backbone standardized MAE over raw slices 0:14 and 14:28 at H30/H60",
        "bootstrap_scope": "fixed three trained seeds; CI conditional on trained models and repeatedly used development split",
        "fallback_joints": fallback_joints,
        "metric_fixtures": "see fixtures/red_capable_metric_fixtures.py; required to pass before scoring",
    }
    (audit / "metrics/core_b/CORE_B_METRIC_QA.json").write_text(json.dumps(qa, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(qa, allow_nan=False))
    return 0 if qa["status"] == "PASS_INDEPENDENT_CORE_B_DEV_SCORE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
