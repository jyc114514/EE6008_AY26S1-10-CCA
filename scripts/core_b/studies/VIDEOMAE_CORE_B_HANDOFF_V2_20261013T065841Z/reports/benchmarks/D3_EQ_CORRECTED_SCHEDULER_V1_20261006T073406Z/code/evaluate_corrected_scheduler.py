#!/usr/bin/env python3
"""Post-inference train/dev motion-fidelity and baseline evaluation.

This is a separate process from train_only_inference.py. It is the first task
process allowed to open targets_native or the D3-EQ train target cache.
"""
from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
DIAG_ROOT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
OUT = TASK_ROOT
REGIONS = {
    "action_0_14": slice(0, 14),
    "action_14_28": slice(14, 28),
    "action_full28": slice(0, 28),
}
EPS = 1e-8


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("NA" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
                             for key, value in row.items()})


def episode_group_map() -> dict[int, tuple[str, str]]:
    table = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"])
    episodes = table["episode_index"].to_numpy(zero_copy_only=False)
    splits = table["split"].to_pylist()
    fingerprints = table["video_sha256_fingerprint"].to_pylist()
    mapping: dict[int, tuple[str, str]] = {}
    for episode, split, fingerprint in zip(episodes, splits, fingerprints):
        if fingerprint is None or not str(fingerprint).strip():
            raise AssertionError(f"missing source fingerprint for episode {episode}")
        key = int(episode)
        raw_split = str(split)
        normalized_split = "dev" if raw_split in {"validation", "val"} else raw_split
        value = (normalized_split, str(fingerprint))
        if key in mapping and mapping[key] != value:
            raise AssertionError(f"conflicting duplicate-safe split entries for episode {key}")
        mapping[key] = value
    return mapping


class GroupReducer:
    """Vectorized windows->episode->source-group equal aggregation."""
    def __init__(self, episodes: np.ndarray, groups: np.ndarray) -> None:
        self.episodes = episodes.astype(np.int64, copy=False)
        self.groups = groups.astype("U128", copy=False)
        self.unique_episodes, self.episode_codes = np.unique(self.episodes, return_inverse=True)
        first = np.full(len(self.unique_episodes), len(self.episodes), dtype=np.int64)
        np.minimum.at(first, self.episode_codes, np.arange(len(self.episodes), dtype=np.int64))
        group_per_episode = self.groups[first]
        self.unique_groups, self.episode_group_codes = np.unique(group_per_episode, return_inverse=True)
        self.row_group_codes = self.episode_group_codes[self.episode_codes]
        windows_per_episode = np.bincount(self.episode_codes, minlength=len(self.unique_episodes))
        episodes_per_group = np.bincount(self.episode_group_codes, minlength=len(self.unique_groups))
        row_group_episode_counts = episodes_per_group[self.episode_group_codes[self.episode_codes]]
        self.row_weights = (1.0 / windows_per_episode[self.episode_codes]
                            / row_group_episode_counts
                            / max(len(self.unique_groups), 1))

    def mean(self, values: np.ndarray, selected: np.ndarray | None = None,
             return_groups: bool = False) -> tuple[float, dict[str, float]]:
        if selected is None:
            selected = np.ones(len(values), dtype=bool)
        idx = np.flatnonzero(selected)
        if not len(idx):
            return float("nan"), {}
        selected_values = values[idx].astype(np.float64, copy=False)
        finite = np.isfinite(selected_values)
        if not np.any(finite):
            return float("nan"), {}
        selected_group_count = len(np.unique(self.row_group_codes[idx]))
        if not return_groups and np.all(finite):
            scaled_weights = self.row_weights[idx] * (len(self.unique_groups) / selected_group_count)
            return float(np.dot(selected_values, scaled_weights)), {}
        ep_codes = self.episode_codes[idx[finite]]
        ep_sums = np.bincount(ep_codes, weights=selected_values[finite], minlength=len(self.unique_episodes))
        ep_counts = np.bincount(ep_codes, minlength=len(self.unique_episodes))
        active_ep = ep_counts > 0
        ep_means = ep_sums[active_ep] / ep_counts[active_ep]
        group_codes = self.episode_group_codes[active_ep]
        group_sums = np.bincount(group_codes, weights=ep_means, minlength=len(self.unique_groups))
        group_counts = np.bincount(group_codes, minlength=len(self.unique_groups))
        active_groups = group_counts > 0
        group_means_arr = group_sums[active_groups] / group_counts[active_groups]
        group_means = ({str(g): float(v) for g, v in zip(self.unique_groups[active_groups], group_means_arr)}
                       if return_groups else {})
        return (float(np.mean(group_means_arr)) if len(group_means_arr) else float("nan"), group_means)

    def groups_all_zero(self, values: np.ndarray, selected: np.ndarray) -> int:
        idx = np.flatnonzero(selected)
        if not len(idx):
            return 0
        codes = self.row_group_codes[idx]
        positive = (values[idx] > 0).astype(np.int64)
        positive_counts = np.bincount(codes, weights=positive, minlength=len(self.unique_groups))
        present = np.bincount(codes, minlength=len(self.unique_groups)) > 0
        return int(np.sum(present & (positive_counts == 0)))


def nested_group_mean(values: np.ndarray, episodes: np.ndarray, groups: np.ndarray,
                      selected: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    return GroupReducer(episodes, groups).mean(values, selected, return_groups=True)


def group_scores(values: np.ndarray, episodes: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return nested_group_mean(values, episodes, groups)[1]


def threshold_labels(scores: dict[str, float], q25: float, q75: float) -> dict[str, str]:
    if not scores:
        return {}
    if not np.isfinite(q25) or not np.isfinite(q75) or q25 >= q75:
        return {group: "degenerate_tie" for group in scores}
    result = {}
    for group, score in scores.items():
        if score <= q25:
            result[group] = "low"
        elif score >= q75:
            result[group] = "high"
        else:
            result[group] = "mid"
    return result


def fit_motion_threshold(scores: dict[str, float]) -> tuple[float, float]:
    arr = np.asarray(list(scores.values()), dtype=np.float64)
    if not len(arr):
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def slope_predictions(history: np.ndarray, times: np.ndarray, anchor: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact locked V3 SLOPE construction; fallback is anchor persistence."""
    n = len(anchor)
    result = np.repeat(anchor[:, None, :], horizon, axis=1).astype(np.float32)
    fallback = np.zeros(n, dtype=bool)
    commands = history[:, :, 43:71].astype(np.float64, copy=False)
    for row in range(n):
        t = times[row].astype(np.float64, copy=False)
        t0 = float(t[-1])
        valid = np.flatnonzero(t >= t0 - 0.25 - 1e-9)
        unique = np.unique(t[valid])
        if len(unique) < 2:
            fallback[row] = True
            continue
        first_t, last_t = float(unique[0]), float(unique[-1])
        first_i = int(np.flatnonzero(t == first_t)[0])
        last_i = int(np.flatnonzero(t == last_t)[-1])
        dt = last_t - first_t
        if dt <= 1e-12:
            fallback[row] = True
            continue
        slope = (commands[row, last_i] - commands[row, first_i]) / dt
        future_times = t0 + np.arange(1, horizon + 1, dtype=np.float64) / 30.0
        result[row] = (anchor[row][None, :] + slope[None, :] * (future_times - t0)[:, None]).astype(np.float32)
    return result, fallback


def activity_threshold(train_targets_h60: np.ndarray) -> tuple[np.ndarray, int]:
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


def window_metrics(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray, sigma: np.ndarray,
                   safe_fd: np.ndarray, active_threshold: np.ndarray,
                   slope_fallback: np.ndarray | None = None) -> dict[str, np.ndarray]:
    pred = pred.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    anchor = anchor.astype(np.float32, copy=False)
    err = pred - target
    std_err = err / sigma[None, None, :]
    d_pred = np.diff(pred, axis=1)
    d_target = np.diff(target, axis=1)
    d_err = d_pred - d_target
    active = np.abs(d_target) > active_threshold[None, None, :]
    exact_zero = np.abs(d_target) <= EPS
    n_active = active.sum(axis=(1, 2))
    direction = (np.sign(d_pred) == np.sign(d_target)) & active
    direction_acc = np.divide(direction.sum(axis=(1, 2)), n_active,
                              out=np.full(len(pred), np.nan, dtype=np.float64), where=n_active > 0)
    h = target.shape[1]
    splits = np.array_split(np.arange(h), 3)
    tv_p = np.abs(d_pred).sum(axis=1).mean(axis=1)
    tv_y = np.abs(d_target).sum(axis=1).mean(axis=1)
    ratio = np.divide(tv_p, tv_y, out=np.full(len(pred), np.nan, dtype=np.float64), where=tv_y > 0.0)
    out = {
        "native_mae": np.abs(err).mean(axis=(1, 2)),
        "standardized_mae": np.abs(std_err).mean(axis=(1, 2)),
        "native_rmse": np.sqrt(np.square(err).mean(axis=(1, 2))),
        "standardized_rmse": np.sqrt(np.square(std_err).mean(axis=(1, 2))),
        "endpoint_native_mae": np.abs(err[:, -1]).mean(axis=1),
        "endpoint_standardized_mae": (np.abs(err[:, -1]) / sigma[None, :]).mean(axis=1),
        "native_future_internal_fd": np.abs(d_err).mean(axis=(1, 2)),
        "safe_fd_normalized": (np.abs(d_err) / safe_fd[None, None, :]).mean(axis=(1, 2)),
        "seam_error_native": np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)).mean(axis=1),
        "seam_pred_step_abs": np.abs(pred[:, 0] - anchor).mean(axis=1),
        "seam_gt_step_abs": np.abs(target[:, 0] - anchor).mean(axis=1),
        "tv_pred": tv_p,
        "tv_gt": tv_y,
        "tv_ratio_window": ratio,
        "pred_step_abs_mean": np.abs(d_pred).mean(axis=(1, 2)),
        "gt_step_abs_mean": np.abs(d_target).mean(axis=(1, 2)),
        "pred_anchor_net_abs": np.abs(pred[:, -1] - anchor).mean(axis=1),
        "gt_anchor_net_abs": np.abs(target[:, -1] - anchor).mean(axis=1),
        "pred_anchor_net_signed": (pred[:, -1] - anchor).mean(axis=1),
        "gt_anchor_net_signed": (target[:, -1] - anchor).mean(axis=1),
        "direction_accuracy_active": direction_acc,
        "active_step_count": n_active.astype(np.int64),
        "near_zero_gt_step_count": exact_zero.sum(axis=(1, 2)).astype(np.int64),
        "small_nonzero_gt_step_count": ((~active) & (~exact_zero)).sum(axis=(1, 2)).astype(np.int64),
        "zero_gt_tv_window": (tv_y == 0.0).astype(np.int64),
    }
    for name, indices in zip(("early_native_mae", "mid_native_mae", "late_native_mae"), splits):
        out[name] = np.abs(err[:, indices]).mean(axis=(1, 2))
    for name, indices in zip(("early_standardized_mae", "mid_standardized_mae", "late_standardized_mae"), splits):
        out[name] = np.abs(std_err[:, indices]).mean(axis=(1, 2))
    if slope_fallback is not None:
        out["slope_fallback_window"] = slope_fallback.astype(np.int64)
    return out


def summarize_window_metrics(metrics: dict[str, np.ndarray], reducer: GroupReducer,
                             selected: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, values in metrics.items():
        mean, group_values = reducer.mean(values.astype(np.float64), selected)
        if name == "tv_pred":
            result["tv_pred_mean"] = mean
        elif name == "tv_gt":
            result["tv_gt_mean"] = mean
        else:
            result[name] = mean
    pred = result.get("tv_pred_mean", float("nan"))
    gt = result.get("tv_gt_mean", float("nan"))
    result["tv_ratio_group_weighted_means"] = pred / gt if math.isfinite(gt) and gt > 0.0 else float("nan")
    ratios, group_ratios = reducer.mean(metrics["tv_ratio_window"].astype(np.float64), selected, return_groups=True)
    result["tv_ratio_group_mean_distribution_mean"] = ratios
    finite = list(group_ratios.values())
    if finite:
        result["tv_ratio_group_p25"] = float(np.quantile(finite, 0.25))
        result["tv_ratio_group_p50"] = float(np.quantile(finite, 0.50))
        result["tv_ratio_group_p75"] = float(np.quantile(finite, 0.75))
    else:
        result.update({"tv_ratio_group_p25": float("nan"), "tv_ratio_group_p50": float("nan"), "tv_ratio_group_p75": float("nan")})
    window_ratios = metrics["tv_ratio_window"][selected]
    finite_window_ratios = window_ratios[np.isfinite(window_ratios)]
    result["tv_ratio_window_valid_count"] = int(len(finite_window_ratios))
    if len(finite_window_ratios):
        result["tv_ratio_window_p25"] = float(np.quantile(finite_window_ratios, 0.25))
        result["tv_ratio_window_p50"] = float(np.quantile(finite_window_ratios, 0.50))
        result["tv_ratio_window_p75"] = float(np.quantile(finite_window_ratios, 0.75))
    else:
        result.update({"tv_ratio_window_p25": float("nan"), "tv_ratio_window_p50": float("nan"), "tv_ratio_window_p75": float("nan")})
    result["n_windows"] = int(selected.sum())
    result["n_episodes"] = int(len(np.unique(reducer.episodes[selected])))
    result["n_source_groups"] = int(len(np.unique(reducer.groups[selected])))
    for count_name in ("active_step_count", "near_zero_gt_step_count", "small_nonzero_gt_step_count"):
        result[f"{count_name}_total"] = int(np.sum(metrics[count_name][selected]))
    result["zero_gt_tv_windows"] = int(np.sum(metrics["zero_gt_tv_window"][selected]))
    result["zero_gt_tv_groups"] = reducer.groups_all_zero(metrics["tv_gt"], selected)
    return result


def reduce_to_expected_order(ids: np.ndarray, expected_ids: np.ndarray, arr: np.ndarray, label: str) -> np.ndarray:
    ids = ids.astype("U96")
    expected_ids = expected_ids.astype("U96")
    if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(expected_ids.tolist()):
        raise AssertionError(f"{label} IDs do not exactly match the locked split population")
    if np.array_equal(ids, expected_ids):
        return arr
    row = {value: i for i, value in enumerate(ids.tolist())}
    return arr[np.asarray([row[value] for value in expected_ids.tolist()], dtype=np.int64)]


def load_prediction(run_id: str, split: str, expected_ids: np.ndarray, expected_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    if split == "train":
        root = OUT / "predictions/train"
    else:
        root = OUT / "predictions/dev"
    p = root / f"{run_id}.npz"
    side = root / f"{run_id}.json"
    if not p.is_file() or not side.is_file():
        raise FileNotFoundError(f"missing prediction bundle for {run_id} split={split}")
    metadata = json.loads(side.read_text(encoding="utf-8"))
    expected_status = "PASS"
    if metadata.get("status") != expected_status or int(metadata.get("rows", -1)) != expected_rows:
        raise AssertionError(f"unaccepted/partial prediction bundle: {run_id} {split}")
    if metadata.get("prediction_sha256") != sha_file(p):
        raise AssertionError(f"prediction checksum mismatch: {run_id} {split}")
    with np.load(p, allow_pickle=False) as bundle:
        pred_ids = bundle["anchor_id"].astype("U96")
        prediction = bundle["predictions_native"].astype(np.float32, copy=True)
        if "split" in bundle.files and str(bundle["split"].item()) != split:
            raise AssertionError(f"split label mismatch for {run_id}")
    if prediction.shape[0] != expected_rows:
        raise AssertionError(f"unexpected prediction rows {run_id}: {prediction.shape}")
    return reduce_to_expected_order(pred_ids, expected_ids, prediction, run_id), metadata


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split_code = archive["split_code"].astype(np.int8, copy=True)
    if set(np.unique(split_code).tolist()) != {0, 1}:
        raise AssertionError("evaluation includes an unexpected split code")
    ids_all = archive["anchor_id"].astype("U96", copy=True)
    episode_all = archive["episode_index"].astype(np.int64, copy=True)
    train_idx = np.flatnonzero(split_code == 0)
    dev_idx = np.flatnonzero(split_code == 1)
    train_ids = ids_all[train_idx]
    dev_ids = ids_all[dev_idx]
    train_episodes = episode_all[train_idx]
    dev_episodes = episode_all[dev_idx]
    mapping = episode_group_map()
    all_episode_split = {ep: split for ep, (split, _) in mapping.items()}

    def groups_for(episodes: np.ndarray, wanted_split: str) -> np.ndarray:
        values = []
        for ep in episodes:
            split, fingerprint = mapping[int(ep)]
            if split != wanted_split:
                raise AssertionError(f"episode {ep} mapped to {split}, expected {wanted_split}")
            values.append(fingerprint)
        return np.asarray(values, dtype="U128")

    train_groups = groups_for(train_episodes, "train")
    dev_groups = groups_for(dev_episodes, "dev")
    if set(train_groups.tolist()) & set(dev_groups.tolist()):
        raise AssertionError("duplicate-safe train/dev source groups overlap")
    if len(train_ids) != 17757 or len(np.unique(train_episodes)) != 2712:
        raise AssertionError("primary train population mismatch")
    if len(dev_ids) != 2322 or len(np.unique(dev_episodes)) != 603:
        raise AssertionError("primary dev population mismatch")

    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    safe_fd = np.asarray(normalizers["safe_fd_scale"], dtype=np.float32)
    if sigma.shape != (28,) or safe_fd.shape != (28,) or np.any(sigma <= 0) or np.any(safe_fd <= 0):
        raise AssertionError("normalizer vectors invalid")
    anchors_all = archive["anchor_command_native"].astype(np.float32, copy=True)
    history_all = archive["history_native"].astype(np.float32, copy=True)
    times_all = archive["history_selected_timestamps"].astype(np.float32, copy=True)
    dev_targets = archive["targets_native"][dev_idx].astype(np.float32, copy=True)
    train_cache_path = EQ / "d3/train_targets_trainonly.npz"
    train_manifest = json.loads((EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if train_manifest.get("status") != "PASS" or train_manifest.get("scope") != "primary_train_only":
        raise AssertionError("train-only target cache manifest is not accepted")
    if sha_file(train_cache_path) != train_manifest.get("target_cache_sha256"):
        raise AssertionError("train target-cache hash mismatch")
    with np.load(train_cache_path, allow_pickle=False) as cache:
        cache_ids = cache["anchor_id"].astype("U96")
        cache_eps = cache["episode_index"].astype(np.int64)
        train_targets_h60 = cache["targets_native_60"].astype(np.float32, copy=True)
        cache_anchors = cache["anchor_command_native"].astype(np.float32, copy=True)
    if not np.array_equal(cache_ids, train_ids) or not np.array_equal(cache_eps, train_episodes):
        raise AssertionError("train target cache is not exactly aligned to locked train inputs")
    if np.max(np.abs(cache_anchors - anchors_all[train_idx])) > 1e-6:
        raise AssertionError("train target anchors do not match locked V3 input anchors")
    train_targets = {30: train_targets_h60[:, :30], 60: train_targets_h60}
    dev_targets_by_h = {30: dev_targets[:, :30], 60: dev_targets}
    train_anchors = anchors_all[train_idx]
    dev_anchors = anchors_all[dev_idx]
    active_threshold, threshold_fallback_joints = activity_threshold(train_targets_h60)

    thresholds: dict[int, tuple[float, float]] = {}
    train_group_motion: dict[int, dict[str, float]] = {}
    for horizon in (30, 60):
        motion = np.abs(np.diff(train_targets[horizon], axis=1)) / sigma[None, None, :]
        per_window = motion.mean(axis=(1, 2))
        scores = group_scores(per_window, train_episodes, train_groups)
        train_group_motion[horizon] = scores
        thresholds[horizon] = fit_motion_threshold(scores)
    common_threshold = thresholds[60]

    split_data = {
        "train": {"ids": train_ids, "episodes": train_episodes, "groups": train_groups,
                  "anchors": train_anchors, "targets": train_targets},
        "dev": {"ids": dev_ids, "episodes": dev_episodes, "groups": dev_groups,
                "anchors": dev_anchors, "targets": dev_targets_by_h},
    }
    for split_population in split_data.values():
        split_population["reducer"] = GroupReducer(split_population["episodes"], split_population["groups"])
    labels: dict[tuple[str, int], dict[str, str]] = {}
    support_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for split_name, data in split_data.items():
        for horizon in (30, 60):
            y = data["targets"][horizon]
            window_motion = (np.abs(np.diff(y, axis=1)) / sigma[None, None, :]).mean(axis=(1, 2))
            scores = group_scores(window_motion, data["episodes"], data["groups"])
            common_y = data["targets"][60]
            common_motion = (np.abs(np.diff(common_y, axis=1)) / sigma[None, None, :]).mean(axis=(1, 2))
            common_scores = group_scores(common_motion, data["episodes"], data["groups"])
            for scope, threshold, scope_scores, motion_horizon in (
                    (f"H{horizon}_train_quantiles", thresholds[horizon], scores, horizon),
                    ("common_H60_train_quantiles", common_threshold, common_scores, 60)):
                lab = threshold_labels(scope_scores, *threshold)
                labels[(split_name, horizon, scope)] = lab
                q25, q75 = threshold
                threshold_rows.append({"split": split_name, "evaluated_horizon": horizon,
                                       "threshold_scope": scope, "train_group_q25": q25,
                                       "train_group_q75": q75, "threshold_fit_split": "train",
                                       "motion_score_horizon": motion_horizon,
                                       "degenerate_tie": int(not q25 < q75),
                                       "groups_scored": len(scope_scores)})
                for stratum in sorted(set(lab.values())):
                    chosen = {g for g, tag in lab.items() if tag == stratum}
                    mask = np.asarray([str(g) in chosen for g in data["groups"]], dtype=bool)
                    support_rows.append({"split": split_name, "evaluated_horizon": horizon,
                                         "threshold_scope": scope, "stratum": stratum,
                                         "n_source_groups": len(chosen),
                                         "n_episodes": len(np.unique(data["episodes"][mask])),
                                         "n_windows": int(mask.sum()), "train_group_q25": q25,
                                         "train_group_q75": q75,
                                         "threshold_fit_split": "train",
                                         "final_test_access": 0})
    write_csv(OUT / "MOTION_THRESHOLD_AUDIT.csv", threshold_rows)
    write_csv(OUT / "MOTION_STRATA_SUPPORT.csv", support_rows)
    (OUT / "MOTION_ACTIVITY_THRESHOLD.json").write_text(json.dumps({
        "status": "PASS", "source": "primary-train target cache only",
        "direction_activity_threshold_rule": "per-joint 5th percentile of positive absolute H60 train increments; floor=1e-6 in source-native units",
        "direction_threshold_vector": active_threshold.tolist(),
        "joints_using_floor_no_positive_training_steps": threshold_fallback_joints,
        "motion_score": "per window mean(|future diff| / target_sigma) over time and action[0:28]",
        "group_reduction": "windows within episode equally; episode means within video SHA source group equally",
        "strata_thresholds": {str(h): {"q25": thresholds[h][0], "q75": thresholds[h][1]} for h in (30, 60)},
        "ties": "if Q25 >= Q75 all groups receive degenerate_tie; no forced quantile split",
        "final_test_access": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (TASK_ROOT / "protocol/MATRIX_12.csv").open(newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    if len(runs) != 12:
        raise AssertionError("locked corrected-scheduler matrix must have 12 rows")
    baseline_rows: list[dict[str, Any]] = []
    baseline_keys: set[tuple[str, int, str, str]] = set()
    metric_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    amplitude_rows: list[dict[str, Any]] = []
    prior_reproduction_path = DIAG_ROOT / "DEV_METRIC_REPRODUCTION_CHECK.csv"
    dev_reproduction_checks = list(csv.DictReader(prior_reproduction_path.open(newline="", encoding="utf-8")))
    prior_reproduction_max_diff = max(float(row["abs_diff"]) for row in dev_reproduction_checks)
    if len(dev_reproduction_checks) != 216 or prior_reproduction_max_diff > 1e-5:
        raise AssertionError("prior D3-EQ scorer reproduction audit is absent or failed")
    prediction_acceptance = []

    # These baselines depend only on split, horizon and locked inputs. Compute
    # each once, not once per model seed/configuration.
    baseline_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for split_name, data in split_data.items():
        anchor = data["anchors"]
        if split_name == "train":
            history, timestamps = history_all[train_idx], times_all[train_idx]
        else:
            history, timestamps = history_all[dev_idx], times_all[dev_idx]
        for baseline_horizon in (30, 60):
            target = data["targets"][baseline_horizon]
            persistence = np.repeat(anchor[:, None, :], baseline_horizon, axis=1).astype(np.float32)
            slope, fallback = slope_predictions(history, timestamps, anchor, baseline_horizon)
            regions = {}
            for region_name, region in REGIONS.items():
                regions[region_name] = {
                    "persistence": window_metrics(persistence[:, :, region], target[:, :, region], anchor[:, region],
                                                  sigma[region], safe_fd[region], active_threshold[region]),
                    "slope": window_metrics(slope[:, :, region], target[:, :, region], anchor[:, region],
                                            sigma[region], safe_fd[region], active_threshold[region], fallback),
                }
            baseline_cache[(split_name, baseline_horizon)] = {
                "regions": regions, "slope_fallback": fallback,
            }
            print(json.dumps({"status": "baseline_cache_ready", "split": split_name,
                              "horizon_steps": baseline_horizon, "rows": len(anchor),
                              "slope_fallback_rows": int(fallback.sum())}), flush=True)

    for run in runs:
        run_id = run["run_id"]
        config, horizon, seed = run["config"], int(run["horizon_steps"]), int(run["seed"])
        print(json.dumps({"status": "scoring_cell", "run_id": run_id}), flush=True)
        model_data: dict[str, dict[str, np.ndarray]] = {}
        for split_name, data in split_data.items():
            expected_ids = data["ids"]
            expected_rows = len(expected_ids)
            pred, side = load_prediction(run_id, split_name, expected_ids, expected_rows)
            if pred.shape != (expected_rows, horizon, 28) or not np.isfinite(pred).all():
                raise AssertionError(f"prediction shape/finite check failed: {run_id} {split_name}")
            if split_name == "train":
                if side.get("target_arrays_loaded") is not False or side.get("optimizer_steps") != 0:
                    raise AssertionError("train inference sidecar violates target-blind/no-optimizer contract")
            prediction_acceptance.append({"run_id": run_id, "split": split_name, "rows": expected_rows,
                                          "prediction_sha256": side["prediction_sha256"],
                                          "checkpoint_sha256": side["checkpoint_sha256"],
                                          "target_arrays_loaded_during_inference": side.get("target_arrays_loaded"),
                                          "status": side["status"]})
            model_data[split_name] = {"prediction": pred, "sidecar": side}

        for split_name, data in split_data.items():
            target = data["targets"][horizon]
            anchor = data["anchors"]
            eps = data["episodes"]
            groups = data["groups"]
            reducer = data["reducer"]
            base = baseline_cache[(split_name, horizon)]
            slope_fallback = base["slope_fallback"]
            pred = model_data[split_name]["prediction"]
            stratum_specs = [("natural", np.ones(len(eps), dtype=bool))]
            hscope = f"H{horizon}_train_quantiles"
            for stratum in ("low", "mid", "high", "degenerate_tie"):
                group_set = {g for g, tag in labels[(split_name, horizon, hscope)].items() if tag == stratum}
                if group_set:
                    stratum_specs.append((f"H{horizon}_{stratum}", np.asarray([g in group_set for g in groups])))
            common_set = {g for g, tag in labels[(split_name, horizon, "common_H60_train_quantiles")].items() if tag in {"low", "mid", "high", "degenerate_tie"}}
            if common_set:
                for stratum in ("low", "mid", "high", "degenerate_tie"):
                    group_set = {g for g, tag in labels[(split_name, horizon, "common_H60_train_quantiles")].items() if tag == stratum}
                    if group_set and not (horizon == 60 and stratum in ("low", "mid", "high", "degenerate_tie")):
                        stratum_specs.append((f"common_H60_{stratum}", np.asarray([g in group_set for g in groups])))
                    elif group_set and horizon == 60:
                        # At H60 the horizon-specific and common labels are identical; keep one row family.
                        pass

            for region_name, region in REGIONS.items():
                sliced = {
                    "model": window_metrics(pred[:, :, region], target[:, :, region], anchor[:, region],
                                             sigma[region], safe_fd[region], active_threshold[region]),
                    "persistence": base["regions"][region_name]["persistence"],
                    "slope": base["regions"][region_name]["slope"],
                }
                for stratum_name, select in stratum_specs:
                    model_summary = summarize_window_metrics(sliced["model"], reducer, select)
                    persistence_summary = summarize_window_metrics(sliced["persistence"], reducer, select)
                    slope_summary = summarize_window_metrics(sliced["slope"], reducer, select)
                    row = {"split": split_name, "run_id": run_id, "config": config, "seed": seed,
                           "horizon_steps": horizon, "region_raw": region_name,
                           "region_slice": f"{region.start}:{region.stop}", "stratum": stratum_name,
                           "n_windows": model_summary["n_windows"], "n_episodes": model_summary["n_episodes"],
                           "n_source_groups": model_summary["n_source_groups"],
                           "region_semantics": "raw action coordinates; Hand/Arm mapping unresolved",
                           "slope_fallback_windows": int(np.sum(slope_fallback[select])),
                           "slope_fallback_fraction": float(np.mean(slope_fallback[select])) if np.any(select) else float("nan"),
                           "activity_threshold_source": "H60 train-only per-joint q05 positive increments; exploratory"}
                    for name, value in model_summary.items():
                        row[f"model_{name}"] = value
                    for name, value in persistence_summary.items():
                        row[f"persistence_{name}"] = value
                    for name, value in slope_summary.items():
                        row[f"slope_{name}"] = value
                    for metric_name in ("native_mae", "standardized_mae", "native_future_internal_fd", "safe_fd_normalized"):
                        row[f"gain_vs_persistence_{metric_name}"] = persistence_summary.get(metric_name, float("nan")) - model_summary.get(metric_name, float("nan"))
                        row[f"gain_vs_slope_{metric_name}"] = slope_summary.get(metric_name, float("nan")) - model_summary.get(metric_name, float("nan"))
                    metric_rows.append(row)
                    # New checkpoint weights differ from old D3; do not label this as scorer reproduction.
                    baseline_key = (split_name, horizon, region_name, stratum_name)
                    if baseline_key not in baseline_keys:
                        baseline_keys.add(baseline_key)
                        baseline_row = {"split": split_name, "horizon_steps": horizon,
                                        "region_raw": region_name, "region_slice": f"{region.start}:{region.stop}",
                                        "stratum": stratum_name,
                                        "n_windows": model_summary["n_windows"], "n_episodes": model_summary["n_episodes"],
                                        "n_source_groups": model_summary["n_source_groups"],
                                        "slope_fallback_windows": int(np.sum(slope_fallback[select])),
                                        "slope_fallback_fraction": float(np.mean(slope_fallback[select])) if np.any(select) else float("nan")}
                        baseline_row.update({f"persistence_{k}": v for k, v in persistence_summary.items()})
                        baseline_row.update({f"slope_{k}": v for k, v in slope_summary.items()})
                        baseline_rows.append(baseline_row)
                    if split_name == "dev":
                        fidelity_rows.append(row.copy())
                    amplitude_rows.append({key: row[key] for key in (
                        "split", "run_id", "config", "seed", "horizon_steps", "region_raw", "region_slice", "stratum",
                        "n_windows", "n_episodes", "n_source_groups", "model_tv_pred_mean", "model_tv_gt_mean",
                        "model_tv_ratio_group_weighted_means", "model_tv_ratio_group_p25", "model_tv_ratio_group_p50",
                        "model_tv_ratio_group_p75", "model_tv_ratio_window_valid_count", "model_tv_ratio_window_p25",
                        "model_tv_ratio_window_p50", "model_tv_ratio_window_p75",
                        "model_zero_gt_tv_windows", "model_zero_gt_tv_groups",
                        "model_pred_step_abs_mean", "model_gt_step_abs_mean", "model_pred_anchor_net_abs",
                        "model_gt_anchor_net_abs", "model_pred_anchor_net_signed", "model_gt_anchor_net_signed",
                        "model_direction_accuracy_active", "model_active_step_count_total",
                        "model_near_zero_gt_step_count_total", "model_small_nonzero_gt_step_count_total",
                        "persistence_tv_pred_mean", "persistence_tv_gt_mean",
                        "slope_tv_pred_mean", "slope_tv_gt_mean", "gain_vs_persistence_standardized_mae",
                        "gain_vs_slope_standardized_mae")})


    write_csv(OUT / "TRAIN_DEV_METRICS.csv", metric_rows)
    write_csv(OUT / "PERSISTENCE_SLOPE_BASELINES.csv", baseline_rows)
    write_csv(OUT / "DEV_MOTION_FIDELITY_BY_SEED.csv", fidelity_rows)
    write_csv(OUT / "AMPLITUDE_TV_DIAGNOSTICS.csv", amplitude_rows)
    write_csv(OUT / "TRAIN_PREDICTION_BUNDLE_AUDIT.csv", prediction_acceptance)
    write_csv(OUT / "DEV_METRIC_REPRODUCTION_CHECK.csv", dev_reproduction_checks)
    max_diff = max((float(x["abs_diff"]) for x in dev_reproduction_checks), default=float("inf"))
    qa = {
        "status": "PASS" if len(prediction_acceptance) == 24 and max_diff <= 1e-5 else "NEEDS_REVIEW",
        "train_prediction_bundles": sum(1 for x in prediction_acceptance if x["split"] == "train"),
        "dev_prediction_bundles_inferred": sum(1 for x in prediction_acceptance if x["split"] == "dev"),
        "train_rows_per_bundle": len(train_ids), "dev_rows_per_bundle": len(dev_ids),
        "train_episodes": int(len(np.unique(train_episodes))), "dev_episodes": int(len(np.unique(dev_episodes))),
        "train_source_groups": int(len(np.unique(train_groups))), "dev_source_groups": int(len(np.unique(dev_groups))),
        "source_group_overlap": 0, "dev_primary_metric_reproduction_rows": len(dev_reproduction_checks),
        "dev_primary_max_abs_difference": max_diff,
        "reproduction_scope": "prior D3-EQ as-run weights only; corrected-scheduler scores are not claimed to reproduce old weights",
        "prior_reproduction_audit_sha256": sha_file(prior_reproduction_path),
        "normalizer_sha256": sha_file(V3 / "data/V3_NORMALIZERS.json"),
        "train_target_cache_sha256": sha_file(train_cache_path),
        "final_test_access": 0, "heldout_loaded": False,
        "train_predictions_were_generated_without_target_arrays": True,
        "gpu_inference_hours": 0,
    }
    (OUT / "EVALUATION_QA.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Full six-configuration objective sensitivity, relabeled by raw coordinates.
    objective = list(csv.DictReader((EQ / "d3/analysis/D3_EQ_OBJECTIVE_SENSITIVITY.csv").open(newline="", encoding="utf-8")))
    label_map = {"Hand": "action[0:14]", "Arm": "action[14:28]", "Full28": "action[0:28]"}
    for row in objective:
        row["body"] = label_map.get(row["body"], row["body"])
    expected = {(cfg, h, body, metric) for cfg in ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
                for h in (30, 60) for body in label_map.values()
                for metric in ("standardized_mae", "native_future_internal_fd")}
    seen = {(r["config"], int(r["horizon_steps"]), r["body"], r["metric"]) for r in objective}
    if len(objective) != 72 or seen != expected:
        raise AssertionError("objective sensitivity must cover all 6 configs x 2 horizons x 3 regions x 2 metrics")
    write_csv(OUT / "D3_EQ_OBJECTIVE_SENSITIVITY_ALL_CONFIGS.csv", objective)
    (OUT / "FINAL_TEST_ACCESS.json").write_text(json.dumps({"final_test_access": 0, "heldout_access": 0,
                                                               "heldout_loaded": False,
                                                               "split_codes_used": [0, 1]}, indent=2) + "\n")
    print(json.dumps(qa, sort_keys=True))
    return 0 if qa["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
