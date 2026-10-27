#!/usr/bin/env python3
"""Paired old-as-run vs corrected-scheduler effects on locked train/dev groups."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
DIAG = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
sys.path.insert(0, str(TASK / "code"))
from evaluate_corrected_scheduler import GroupReducer  # noqa: E402

SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS11-D", "STATE-CTRL")
HORIZONS = (30, 60)
REGIONS = {
    "action_0_14": slice(0, 14),
    "action_14_28": slice(14, 28),
    "action_full28": slice(0, 28),
}
PRIMARY_REGIONS = ("action_full28",)
PRIMARY_METRICS = ("standardized_mae", "native_future_internal_fd")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260924
EPS = 1e-8


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("NA" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
                for key, value in row.items()
            })


def read_source_groups() -> dict[int, tuple[str, str]]:
    table = pq.read_table(SPLIT_MANIFEST, columns=[
        "episode_index", "split", "video_sha256_fingerprint"
    ])
    mapping: dict[int, tuple[str, str]] = {}
    for episode, split, fingerprint in zip(
        table["episode_index"].to_numpy(zero_copy_only=False),
        table["split"].to_pylist(),
        table["video_sha256_fingerprint"].to_pylist(),
    ):
        normalized = "dev" if str(split) in {"validation", "val"} else str(split)
        if fingerprint is None or not str(fingerprint).strip():
            raise AssertionError(f"missing source fingerprint for episode {episode}")
        key = int(episode)
        value = (normalized, str(fingerprint))
        if key in mapping and mapping[key] != value:
            raise AssertionError(f"conflicting duplicate-safe episode mapping {key}")
        mapping[key] = value
    return mapping


def load_prediction(path: Path, sidecar_path: Path, *, expected_ids: np.ndarray,
                    expected_episodes: np.ndarray, config: str, seed: int,
                    horizon: int, split: str, checkpoint_sha: str,
                    study_protocol_sha: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"prediction bundle missing: {path}")
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if side.get("status") != "PASS" or int(side.get("rows", -1)) != len(expected_ids):
        raise AssertionError(f"prediction sidecar not accepted: {path.name}")
    if side.get("prediction_sha256") != sha_file(path) or side.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError(f"prediction/checkpoint hash mismatch: {path.name}")
    if side.get("config") != config or int(side.get("seed", -1)) != seed or int(side.get("horizon_steps", -1)) != horizon:
        raise AssertionError(f"prediction cell metadata mismatch: {path.name}")
    if side.get("split") not in (None, split):
        raise AssertionError(f"prediction split metadata mismatch: {path.name}")
    if study_protocol_sha is not None and side.get("study_protocol_sha256") != study_protocol_sha:
        raise AssertionError(f"prediction task protocol hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as bundle:
        pred_ids = bundle["anchor_id"].astype("U96", copy=True)
        pred_episodes = bundle["episode_index"].astype(np.int64, copy=True)
        predictions = bundle["predictions_native"].astype(np.float32, copy=True)
        config_value = str(bundle["config"].item())
        seed_value = int(bundle["seed"].item())
        horizon_value = int(bundle["horizon_steps"].item())
        split_value = str(bundle["split"].item()) if "split" in bundle.files else split
    if config_value != config or seed_value != seed or horizon_value != horizon or split_value != split:
        raise AssertionError(f"prediction NPZ identity mismatch: {path.name}")
    if len(set(pred_ids.tolist())) != len(pred_ids) or set(pred_ids.tolist()) != set(expected_ids.tolist()):
        raise AssertionError(f"prediction IDs differ from locked {split}: {path.name}")
    order = {value: index for index, value in enumerate(pred_ids.tolist())}
    positions = np.asarray([order[value] for value in expected_ids.tolist()], dtype=np.int64)
    predictions = predictions[positions]
    pred_episodes = pred_episodes[positions]
    if not np.array_equal(pred_episodes, expected_episodes):
        raise AssertionError(f"prediction episode order differs from locked {split}: {path.name}")
    if predictions.shape != (len(expected_ids), horizon, 28) or not np.isfinite(predictions).all():
        raise AssertionError(f"prediction array shape/finite check failed: {path.name}")
    return predictions, side


def per_window_metrics(prediction: np.ndarray, target: np.ndarray, anchor: np.ndarray,
                       sigma: np.ndarray, active_threshold: np.ndarray,
                       region: slice) -> dict[str, np.ndarray]:
    p = prediction[:, :, region].astype(np.float64, copy=False)
    y = target[:, :, region].astype(np.float64, copy=False)
    a = anchor[:, region].astype(np.float64, copy=False)
    s = sigma[region].astype(np.float64, copy=False)
    tau = active_threshold[region].astype(np.float64, copy=False)
    error = p - y
    d_pred = np.diff(p, axis=1)
    d_gt = np.diff(y, axis=1)
    abs_gt = np.abs(d_gt)
    active = abs_gt > tau[None, None, :]
    positive = d_gt > tau[None, None, :]
    negative = d_gt < -tau[None, None, :]
    pred_zero = np.abs(d_pred) <= EPS
    pred_sign = np.where(pred_zero, 0.0, np.sign(d_pred))
    gt_sign = np.sign(d_gt)
    active_count = active.sum(axis=(1, 2))
    correct = ((pred_sign == gt_sign) & active).sum(axis=(1, 2))
    positive_count = positive.sum(axis=(1, 2))
    negative_count = negative.sum(axis=(1, 2))
    positive_correct = ((pred_sign > 0) & positive).sum(axis=(1, 2))
    negative_correct = ((pred_sign < 0) & negative).sum(axis=(1, 2))
    return {
        "standardized_mae": (np.abs(error) / s[None, None, :]).mean(axis=(1, 2)),
        "native_mae": np.abs(error).mean(axis=(1, 2)),
        "native_future_internal_fd": np.abs(np.diff(error, axis=1)).mean(axis=(1, 2)),
        "tv_pred": np.abs(d_pred).sum(axis=1).mean(axis=1),
        "tv_gt": np.abs(d_gt).sum(axis=1).mean(axis=1),
        "pred_step_abs_mean": np.abs(d_pred).mean(axis=(1, 2)),
        "gt_step_abs_mean": np.abs(d_gt).mean(axis=(1, 2)),
        "pred_anchor_net_abs": np.abs(p[:, -1, :] - a).mean(axis=1),
        "gt_anchor_net_abs": np.abs(y[:, -1, :] - a).mean(axis=1),
        "direction_accuracy_active": np.divide(
            correct, active_count, out=np.full(len(p), np.nan, dtype=np.float64), where=active_count > 0
        ),
        "positive_direction_recall": np.divide(
            positive_correct, positive_count, out=np.full(len(p), np.nan, dtype=np.float64), where=positive_count > 0
        ),
        "negative_direction_recall": np.divide(
            negative_correct, negative_count, out=np.full(len(p), np.nan, dtype=np.float64), where=negative_count > 0
        ),
        "active_step_count": active_count.astype(np.int64),
        "positive_gt_step_count": positive_count.astype(np.int64),
        "negative_gt_step_count": negative_count.astype(np.int64),
        "nochange_prediction_count": (pred_zero & active).sum(axis=(1, 2)).astype(np.int64),
    }


def group_means(reducer: GroupReducer, group_order: np.ndarray, values: np.ndarray) -> np.ndarray:
    _, by_group = reducer.mean(np.asarray(values, dtype=np.float64), return_groups=True)
    return np.asarray([by_group.get(str(group), np.nan) for group in group_order], dtype=np.float64)


def bootstrap_indices(n_groups: int, horizon: int, split: str, stratum: str = "natural") -> np.ndarray:
    salt = 0 if split == "train" else 1_000_000
    salt += {"natural": 0, "low": 100, "mid": 200, "high": 300, "degenerate_tie": 400}.get(stratum, 900)
    rng = np.random.default_rng(BOOTSTRAP_SEED + horizon + salt)
    return rng.integers(0, n_groups, size=(BOOTSTRAPS, n_groups), dtype=np.int32)


def bootstrap_summary(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    finite = vector[np.isfinite(vector)]
    if len(finite) != len(vector) or not len(vector):
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "p_centered_add_one": float("nan")}
    draws = vector[indices].mean(axis=1)
    estimate = float(vector.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    centered = draws - estimate
    p = (np.count_nonzero(np.abs(centered) >= abs(estimate)) + 1) / (len(draws) + 1)
    return {"estimate": estimate, "ci95_low": float(low), "ci95_high": float(high),
            "p_centered_add_one": float(p)}


def holm_adjust(rows: list[dict[str, Any]], key: str = "p_centered_add_one") -> None:
    valid = [(index, float(row[key])) for index, row in enumerate(rows)
             if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalue))
        rows[index]["p_holm_family"] = running


def group_effect_summary(values: np.ndarray, groups: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    selected = values[valid]
    if len(selected) != len(values) or not len(values):
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
                "p_centered_add_one": float("nan"), "n_source_groups": int(len(selected))}
    result = bootstrap_summary(values, indices)
    result["n_source_groups"] = int(len(groups))
    return result


def main() -> int:
    protocol = json.loads((TASK / "PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or protocol.get("study_id") != "CORRECTED_SCHEDULER_MATCHED_12":
        raise AssertionError("corrected-scheduler protocol is not locked")
    study_sha = sha_file(TASK / "PROTOCOL_LOCKED.json")
    matrix = list(csv.DictReader((TASK / "protocol/MATRIX_12.csv").open(newline="", encoding="utf-8")))
    if len(matrix) != 12:
        raise AssertionError("matched protocol matrix must contain 12 cells")
    groups_map = read_source_groups()

    # Load only conditioning metadata here; dev targets are not opened until all 48 old/new bundles pass.
    input_path = V3 / "data/V3_MAIN_INPUTS.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        split_code = archive["split_code"].astype(np.int8, copy=True)
        anchor_ids = archive["anchor_id"].astype("U96", copy=True)
        episodes_all = archive["episode_index"].astype(np.int64, copy=True)
        anchors_all = archive["anchor_command_native"].astype(np.float32, copy=True)
        history_all = archive["history_native"].astype(np.float32, copy=True)
        times_all = archive["history_selected_timestamps"].astype(np.float32, copy=True)
    split_data: dict[str, dict[str, Any]] = {}
    for split_name, code, expected_rows, expected_eps in (
        ("train", 0, 17757, 2712), ("dev", 1, 2322, 603)
    ):
        idx = np.flatnonzero(split_code == code)
        ids = anchor_ids[idx]
        episodes = episodes_all[idx]
        if len(idx) != expected_rows or len(np.unique(episodes)) != expected_eps or len(set(ids.tolist())) != expected_rows:
            raise AssertionError(f"locked {split_name} anchor/episode counts changed")
        group_values = []
        for episode in episodes:
            mapped_split, fingerprint = groups_map[int(episode)]
            if mapped_split != split_name:
                raise AssertionError(f"episode {episode} source split mismatch")
            group_values.append(fingerprint)
        groups = np.asarray(group_values, dtype="U128")
        if len(set(groups.tolist())) != expected_eps:
            raise AssertionError(f"{split_name} source groups not unique one-to-one with episodes")
        split_data[split_name] = {
            "indices": idx, "ids": ids, "episodes": episodes, "groups": groups,
            "anchors": anchors_all[idx], "history": history_all[idx], "times": times_all[idx],
        }
    if set(split_data["train"]["groups"].tolist()) & set(split_data["dev"]["groups"].tolist()):
        raise AssertionError("train/dev source groups overlap")

    bundles: dict[tuple[str, str, int, int], np.ndarray] = {}
    bundle_audit: list[dict[str, Any]] = []
    for row in matrix:
        config, seed, horizon = row["config"], int(row["seed"]), int(row["horizon_steps"])
        new_id = row["run_id"]
        old_id = row["matched_old_run_id"]
        old_run = EQ / "d3/runs" / old_id
        old_complete = json.loads((old_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        new_run = TASK / "runs/formal" / new_id
        new_complete = json.loads((new_run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        if old_complete.get("status") != "PASS" or new_complete.get("status") != "PASS":
            raise AssertionError(f"training incomplete before scoring: {new_id}")
        for split_name in ("train", "dev"):
            expected = split_data[split_name]
            if split_name == "train":
                old_root = DIAG / "predictions/train"
            else:
                old_root = EQ / "d3/predictions"
            new_root = TASK / "predictions" / split_name
            old_pred, old_side = load_prediction(
                old_root / f"{old_id}.npz", old_root / f"{old_id}.json",
                expected_ids=expected["ids"], expected_episodes=expected["episodes"],
                config=config, seed=seed, horizon=horizon, split=split_name,
                checkpoint_sha=old_complete["checkpoint_last_sha256"],
            )
            new_pred, new_side = load_prediction(
                new_root / f"{new_id}.npz", new_root / f"{new_id}.json",
                expected_ids=expected["ids"], expected_episodes=expected["episodes"],
                config=config, seed=seed, horizon=horizon, split=split_name,
                checkpoint_sha=new_complete["checkpoint_last_sha256"], study_protocol_sha=study_sha,
            )
            bundles[(split_name, "old", config, seed, horizon)] = old_pred
            bundles[(split_name, "new", config, seed, horizon)] = new_pred
            bundle_audit.append({
                "split": split_name, "new_run_id": new_id, "matched_old_run_id": old_id,
                "config": config, "seed": seed, "horizon_steps": horizon,
                "rows": len(expected["ids"]),
                "old_checkpoint_sha256": old_complete["checkpoint_last_sha256"],
                "new_checkpoint_sha256": new_complete["checkpoint_last_sha256"],
                "old_prediction_sha256": old_side["prediction_sha256"],
                "new_prediction_sha256": new_side["prediction_sha256"],
                "targets_loaded_during_inference": False,
                "status": "PASS",
            })
    if len(bundle_audit) != 24 or len(bundles) != 48:
        raise AssertionError("old/new train+dev prediction coverage must be 48 bundles (24 matched split pairs)")

    # Only now may the evaluator read target values.
    train_cache_path = EQ / "d3/train_targets_trainonly.npz"
    with np.load(train_cache_path, allow_pickle=False) as cache:
        train_ids = cache["anchor_id"].astype("U96", copy=True)
        train_eps = cache["episode_index"].astype(np.int64, copy=True)
        train_targets_h60 = cache["targets_native_60"].astype(np.float32, copy=True)
    if not np.array_equal(train_ids, split_data["train"]["ids"]) or not np.array_equal(train_eps, split_data["train"]["episodes"]):
        raise AssertionError("train-only target cache order differs from locked train IDs")
    with np.load(input_path, allow_pickle=False) as archive:
        dev_targets_h60 = archive["targets_native"][split_data["dev"]["indices"]].astype(np.float32, copy=True)
    if train_targets_h60.shape != (17757, 60, 28) or dev_targets_h60.shape != (2322, 60, 28):
        raise AssertionError("post-inference target shape mismatch")
    normalizers = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    active_threshold = np.asarray(json.loads((DIAG / "MOTION_ACTIVITY_THRESHOLD.json").read_text(encoding="utf-8"))["direction_threshold_vector"], dtype=np.float32)
    if sigma.shape != (28,) or active_threshold.shape != (28,):
        raise AssertionError("target/activity normalizer vectors must be 28-dimensional")

    targets_by_split = {
        "train": train_targets_h60,
        "dev": dev_targets_h60,
    }
    group_orders = {
        split: np.sort(np.unique(data["groups"]))
        for split, data in split_data.items()
    }
    reducers = {
        split: GroupReducer(data["episodes"], data["groups"])
        for split, data in split_data.items()
    }

    # Reuse the previously locked train-frozen motion thresholds and verify them from train targets only.
    motion_thresholds = json.loads((DIAG / "MOTION_ACTIVITY_THRESHOLD.json").read_text(encoding="utf-8"))
    q_thresholds = {int(h): (float(v["q25"]), float(v["q75"]))
                     for h, v in motion_thresholds["strata_thresholds"].items()}
    motion_labels: dict[tuple[str, int], dict[str, str]] = {}
    support_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        train_motion = np.abs(np.diff(train_targets_h60[:, :horizon], axis=1) / sigma[None, None, :]).mean(axis=(1, 2))
        train_scores = group_means(reducers["train"], group_orders["train"], train_motion)
        q25, q75 = q_thresholds[horizon]
        recomputed_q25, recomputed_q75 = np.quantile(train_scores, [0.25, 0.75])
        if not np.allclose([recomputed_q25, recomputed_q75], [q25, q75], rtol=0.0, atol=1e-7):
            raise AssertionError(f"train-frozen motion quantiles differ for horizon {horizon}")
        for split, targets in targets_by_split.items():
            per_window = np.abs(np.diff(targets[:, :horizon], axis=1) / sigma[None, None, :]).mean(axis=(1, 2))
            group_scores_vec = group_means(reducers[split], group_orders[split], per_window)
            labels: dict[str, str] = {}
            for group, score in zip(group_orders[split], group_scores_vec):
                if not np.isfinite(score):
                    labels[str(group)] = "NA"
                elif q25 >= q75:
                    labels[str(group)] = "degenerate_tie"
                elif score <= q25:
                    labels[str(group)] = "low"
                elif score >= q75:
                    labels[str(group)] = "high"
                else:
                    labels[str(group)] = "mid"
            motion_labels[(split, horizon)] = labels
            for stratum in ("low", "mid", "high", "degenerate_tie"):
                chosen = {g for g, label in labels.items() if label == stratum}
                selected = np.asarray([str(g) in chosen for g in split_data[split]["groups"]])
                if chosen:
                    support_rows.append({
                        "split": split, "horizon_steps": horizon, "stratum": stratum,
                        "n_source_groups": len(chosen),
                        "n_episodes": len(np.unique(split_data[split]["episodes"][selected])),
                        "n_windows": int(selected.sum()), "threshold_fit_split": "train",
                        "q25": q25, "q75": q75,
                    })

    group_metric_values: dict[tuple[str, str, str, int, int, str, str], np.ndarray] = {}
    group_support: dict[tuple[str, str, str, int, int, str], dict[str, int]] = {}
    metric_names = (
        "standardized_mae", "native_mae", "native_future_internal_fd",
        "tv_pred", "tv_gt", "pred_step_abs_mean", "gt_step_abs_mean",
        "pred_anchor_net_abs", "gt_anchor_net_abs", "direction_accuracy_active",
        "positive_direction_recall", "negative_direction_recall",
    )
    model_metric_rows: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
    for split, data in split_data.items():
        episodes, groups, anchors = data["episodes"], data["groups"], data["anchors"]
        reducer = reducers[split]
        group_order = group_orders[split]
        for horizon in HORIZONS:
            truth = targets_by_split[split][:, :horizon]
            persistence = np.repeat(anchors[:, None, :], horizon, axis=1)
            for region_name, region in REGIONS.items():
                persistence_metrics = per_window_metrics(persistence, truth, anchors, sigma, active_threshold, region)
                persistence_grouped = {name: group_means(reducer, group_order, values)
                                       for name, values in persistence_metrics.items()}
                p_tv_pred = persistence_grouped["tv_pred"]
                p_tv_gt = persistence_grouped["tv_gt"]
                p_ratio = float(np.mean(p_tv_pred) / np.mean(p_tv_gt)) if np.mean(p_tv_gt) > 0 else float("nan")
                persistence_rows.append({
                    "split": split, "horizon_steps": horizon, "region_raw": region_name,
                    "region_slice": f"{region.start}:{region.stop}",
                    "n_source_groups": len(group_order),
                    "standardized_mae": float(np.nanmean(persistence_grouped["standardized_mae"])),
                    "native_future_internal_fd": float(np.nanmean(persistence_grouped["native_future_internal_fd"])),
                    "tv_pred_mean": float(np.nanmean(p_tv_pred)),
                    "tv_gt_mean": float(np.nanmean(p_tv_gt)),
                    "tv_ratio_group_weighted_means": p_ratio,
                    "direction_accuracy_active": float(np.nanmean(persistence_grouped["direction_accuracy_active"])),
                    "final_test_access": 0,
                })
                for row in matrix:
                    config, seed, new_id = row["config"], int(row["seed"]), row["run_id"]
                    old_id = row["matched_old_run_id"]
                    for variant, prediction in (
                        ("old_as_run", bundles[(split, "old", config, seed, horizon)]),
                        ("corrected_scheduler", bundles[(split, "new", config, seed, horizon)]),
                    ):
                        per_window = per_window_metrics(prediction, truth, anchors, sigma, active_threshold, region)
                        grouped = {name: group_means(reducer, group_order, values)
                                   for name, values in per_window.items()}
                        key_base = (split, variant, config, seed, horizon, region_name)
                        for metric in metric_names:
                            group_metric_values[key_base + (metric,)] = grouped[metric]
                        group_support[key_base] = {
                            "n_source_groups": len(group_order),
                            "active_steps": int(np.sum(per_window["active_step_count"])),
                            "positive_gt_steps": int(np.sum(per_window["positive_gt_step_count"])),
                            "negative_gt_steps": int(np.sum(per_window["negative_gt_step_count"])),
                            "near_zero_prediction_active_steps": int(np.sum(per_window["nochange_prediction_count"])),
                        }
                        tvp, tvg = grouped["tv_pred"], grouped["tv_gt"]
                        tv_ratio = float(np.mean(tvp) / np.mean(tvg)) if np.mean(tvg) > 0 else float("nan")
                        model_metric_rows.append({
                            "split": split, "variant": variant, "run_id": new_id if variant == "corrected_scheduler" else old_id,
                            "config": config, "seed": seed, "horizon_steps": horizon,
                            "region_raw": region_name, "region_slice": f"{region.start}:{region.stop}",
                            "n_source_groups": len(group_order),
                            "standardized_mae": float(np.nanmean(grouped["standardized_mae"])),
                            "native_mae": float(np.nanmean(grouped["native_mae"])),
                            "native_future_internal_fd": float(np.nanmean(grouped["native_future_internal_fd"])),
                            "tv_pred_mean": float(np.nanmean(tvp)),
                            "tv_gt_mean": float(np.nanmean(tvg)),
                            "tv_ratio_group_weighted_means": tv_ratio,
                            "pred_step_abs_mean": float(np.nanmean(grouped["pred_step_abs_mean"])),
                            "gt_step_abs_mean": float(np.nanmean(grouped["gt_step_abs_mean"])),
                            "pred_anchor_net_abs": float(np.nanmean(grouped["pred_anchor_net_abs"])),
                            "gt_anchor_net_abs": float(np.nanmean(grouped["gt_anchor_net_abs"])),
                            "direction_accuracy_active": float(np.nanmean(grouped["direction_accuracy_active"])),
                            **group_support[key_base],
                            "final_test_access": 0,
                        })

    effect_group_rows: list[dict[str, Any]] = []
    seed_effect_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    video_seed_rows: list[dict[str, Any]] = []
    video_summary_rows: list[dict[str, Any]] = []
    paired_regions = REGIONS
    all_splits = ("train", "dev")
    for split in all_splits:
        group_order = group_orders[split]
        for config in CONFIGS:
            for seed in SEEDS:
                for horizon in HORIZONS:
                    for region_name in paired_regions:
                        for metric in PRIMARY_METRICS:
                            old_values = group_metric_values[(split, "old_as_run", config, seed, horizon, region_name, metric)]
                            new_values = group_metric_values[(split, "corrected_scheduler", config, seed, horizon, region_name, metric)]
                            gain = old_values - new_values
                            if not np.isfinite(gain).all():
                                raise AssertionError(f"primary paired loss contains NA: {(split,config,seed,horizon,region_name,metric)}")
                            for group, old_val, new_val, effect in zip(group_order, old_values, new_values, gain):
                                if region_name == "action_full28":
                                    effect_group_rows.append({
                                        "split": split, "config": config, "seed": seed, "horizon_steps": horizon,
                                        "metric": metric, "source_group": str(group),
                                        "old_as_run_loss": old_val, "corrected_scheduler_loss": new_val,
                                        "schedule_gain_old_minus_new": effect,
                                    })
                            indices = bootstrap_indices(len(group_order), horizon, split)
                            result = bootstrap_summary(gain, indices)
                            seed_effect_rows.append({
                                "split": split, "effect_family": "schedule_gain", "config": config,
                                "seed": seed, "horizon_steps": horizon, "region_raw": region_name,
                                "metric": metric, **result,
                                "seed_point": float(np.mean(gain)),
                                "min_seed": None, "max_seed": None,
                                "bootstrap_replicates": BOOTSTRAPS,
                                "bootstrap_unit": "paired duplicate-safe video source group",
                                "train_seed_uncertainty_included": False,
                                "final_test_access": 0,
                            })
    # Aggregate over seeds within each source group first, then bootstrap groups.
    for split in all_splits:
        group_order = group_orders[split]
        for config in CONFIGS:
            for horizon in HORIZONS:
                indices = bootstrap_indices(len(group_order), horizon, split)
                for region_name in paired_regions:
                    for metric in PRIMARY_METRICS:
                        seed_gains = []
                        for seed in SEEDS:
                            old_values = group_metric_values[(split, "old_as_run", config, seed, horizon, region_name, metric)]
                            new_values = group_metric_values[(split, "corrected_scheduler", config, seed, horizon, region_name, metric)]
                            seed_gains.append(old_values - new_values)
                        group_avg = np.mean(np.stack(seed_gains), axis=0)
                        result = bootstrap_summary(group_avg, indices)
                        seed_points = [float(np.mean(values)) for values in seed_gains]
                        family = "primary" if split == "dev" and region_name == "action_full28" else "auxiliary"
                        summary_rows.append({
                            "split": split, "effect_family": "schedule_gain", "family": family,
                            "config": config, "horizon_steps": horizon, "region_raw": region_name,
                            "metric": metric, **result,
                            "seed_points": json.dumps({str(seed): point for seed, point in zip(SEEDS, seed_points)}, sort_keys=True),
                            "seed_mean": float(np.mean(seed_points)),
                            "seed_min": min(seed_points), "seed_max": max(seed_points),
                            "estimand": "mean three fixed-seed gains within group; equal-weight groups",
                            "source_groups": len(group_order),
                            "bootstrap_replicates": BOOTSTRAPS,
                            "final_test_access": 0,
                        })
    # Video-specific gain difference is prespecified secondary on Full28.
    for split in all_splits:
        group_order = group_orders[split]
        for horizon in HORIZONS:
            indices = bootstrap_indices(len(group_order), horizon, split)
            for region_name in paired_regions:
                for metric in PRIMARY_METRICS:
                    per_seed = []
                    for seed in SEEDS:
                        ts_old = group_metric_values[(split, "old_as_run", "TS11-D", seed, horizon, region_name, metric)]
                        ts_new = group_metric_values[(split, "corrected_scheduler", "TS11-D", seed, horizon, region_name, metric)]
                        st_old = group_metric_values[(split, "old_as_run", "STATE-CTRL", seed, horizon, region_name, metric)]
                        st_new = group_metric_values[(split, "corrected_scheduler", "STATE-CTRL", seed, horizon, region_name, metric)]
                        gain_ts, gain_state = ts_old - ts_new, st_old - st_new
                        effect = gain_ts - gain_state
                        per_seed.append(effect)
                        seed_summary = bootstrap_summary(effect, indices)
                        video_seed_rows.append({
                            "split": split, "horizon_steps": horizon, "region_raw": region_name,
                            "metric": metric, "seed": seed, **seed_summary,
                            "seed_point": float(np.mean(effect)),
                            "estimand": "gain_TS11-D minus gain_STATE-CTRL; positive means larger fusion schedule gain",
                            "source_groups": len(group_order), "bootstrap_replicates": BOOTSTRAPS,
                            "final_test_access": 0,
                        })
                    group_avg = np.mean(np.stack(per_seed), axis=0)
                    result = bootstrap_summary(group_avg, indices)
                    points = [float(np.mean(values)) for values in per_seed]
                    family = "secondary" if split == "dev" and region_name == "action_full28" else "auxiliary"
                    video_summary_rows.append({
                        "split": split, "family": family, "horizon_steps": horizon,
                        "region_raw": region_name, "metric": metric, **result,
                        "seed_points": json.dumps({str(seed): point for seed, point in zip(SEEDS, points)}, sort_keys=True),
                        "seed_mean": float(np.mean(points)), "seed_min": min(points), "seed_max": max(points),
                        "estimand": "mean three fixed-seed video-specific gain differences within group; equal-weight groups",
                        "source_groups": len(group_order), "bootstrap_replicates": BOOTSTRAPS,
                        "final_test_access": 0,
                    })

    strata_effect_rows: list[dict[str, Any]] = []
    for split in all_splits:
        group_order = group_orders[split]
        labels = motion_labels
        for config in CONFIGS:
            for seed in SEEDS:
                for horizon in HORIZONS:
                    strata_map = labels[(split, horizon)]
                    for stratum in ("low", "mid", "high", "degenerate_tie"):
                        selected_group_indices = np.asarray([i for i, group in enumerate(group_order)
                                                             if strata_map.get(str(group)) == stratum], dtype=np.int64)
                        for metric in PRIMARY_METRICS:
                            old_values = group_metric_values[(split, "old_as_run", config, seed, horizon, "action_full28", metric)]
                            new_values = group_metric_values[(split, "corrected_scheduler", config, seed, horizon, "action_full28", metric)]
                            values = (old_values - new_values)[selected_group_indices]
                            if len(values) >= 2 and np.isfinite(values).all():
                                result = bootstrap_summary(values, bootstrap_indices(len(values), horizon, split, stratum))
                            else:
                                result = {"estimate": float("nan"), "ci95_low": float("nan"),
                                          "ci95_high": float("nan"), "p_centered_add_one": float("nan")}
                            strata_effect_rows.append({
                                "split": split, "config": config, "seed": seed,
                                "horizon_steps": horizon, "stratum": stratum, "metric": metric,
                                "region_raw": "action_full28", **result,
                                "n_source_groups": int(len(selected_group_indices)),
                                "low_support": len(selected_group_indices) < 20,
                                "threshold_fit_split": "train",
                                "final_test_access": 0,
                            })

    dev_primary = [row for row in summary_rows if row.get("family") == "primary"]
    dev_secondary = [row for row in video_summary_rows if row.get("family") == "secondary"]
    holm_adjust(dev_primary)
    holm_adjust(dev_secondary)
    write_csv(TASK / "analysis/SCHEDULE_GAINS_BY_SEED.csv", seed_effect_rows)
    write_csv(TASK / "analysis/SCHEDULE_GAINS_SUMMARY.csv", summary_rows)
    write_csv(TASK / "analysis/VIDEO_SPECIFIC_GAIN_BY_SEED.csv", video_seed_rows)
    write_csv(TASK / "analysis/MOTION_STRATA_EFFECTS.csv", strata_effect_rows)
    write_csv(TASK / "analysis/VIDEO_SPECIFIC_GAIN_SUMMARY.csv", video_summary_rows)
    write_csv(TASK / "analysis/GROUP_PAIRED_EFFECTS.csv", effect_group_rows)
    write_csv(TASK / "analysis/MODEL_METRICS_BY_SEED.csv", model_metric_rows)
    write_csv(TASK / "analysis/PERSISTENCE_BASELINES_BY_SPLIT.csv", persistence_rows)
    write_csv(TASK / "analysis/MOTION_STRATA_SUPPORT_REUSED.csv", support_rows)

    primary_rows = [row for row in summary_rows if row.get("split") == "dev" and row.get("family") == "primary"]
    secondary_rows = [row for row in video_summary_rows if row.get("split") == "dev" and row.get("family") == "secondary"]
    qa = {
        "status": "PASS" if len(primary_rows) == 8 and len(secondary_rows) == 4 else "FAIL",
        "study_id": "CORRECTED_SCHEDULER_MATCHED_12",
        "new_cells": len(matrix),
        "old_new_prediction_split_pairs": len(bundle_audit),
        "prediction_bundle_audit_rows": len(bundle_audit),
        "train_source_groups": len(group_orders["train"]),
        "dev_source_groups": len(group_orders["dev"]),
        "train_dev_source_group_overlap": 0,
        "primary_effect_rows": len(primary_rows),
        "secondary_video_specific_rows": len(secondary_rows),
        "bootstrap_replicates": BOOTSTRAPS,
        "dev_targets_loaded_after_prediction_acceptance": True,
        "train_target_scope": "primary_train_only cache",
        "test_final_heldout_access": 0,
        "action_half_anatomy": "unresolved; numeric raw slices only",
        "submission_ready": False,
        "human_review_status": "pending",
    }
    (TASK / "analysis/ANALYSIS_QA.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(qa, sort_keys=True))
    if qa["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
