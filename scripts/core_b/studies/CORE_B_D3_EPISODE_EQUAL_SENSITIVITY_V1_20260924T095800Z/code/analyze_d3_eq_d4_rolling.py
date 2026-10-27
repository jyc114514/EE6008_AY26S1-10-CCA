"""CPU same-absolute-future-time D4 rolling analysis for D3-EQ dense predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = Path(json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))["raw_root"]).resolve(strict=True)
OLD_PRED_ROOT = RESULT_ROOT / "d3_extension/predictions"
NEW_PRED_ROOT = EQ_ROOT / "d3/dense_predictions"
PAIR_PATH = EQ_ROOT / "protocol/D3_EQ_D4_PAIR_MANIFEST_INPUT_ONLY.csv"
INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
OUT = EQ_ROOT / "d3/analysis"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
CONTRASTS = (
    ("TS11-D", "TS10-D", "TS11D_vs_TS10D"),
    ("TS11-D", "TS01-D", "TS11D_vs_TS01D"),
    ("TS11-D", "TS11-S", "TS11D_vs_TS11S"),
    ("TS11-D", "STATE-CTRL", "TS11D_vs_STATE"),
)
BODIES = {"Hand": slice(0, 14), "Arm": slice(14, 28), "Full28": slice(0, 28)}
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260926


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty D4 CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prediction_grid(root: Path, dense_ids: np.ndarray) -> dict[tuple[str, int, int], np.ndarray]:
    result = {}
    for sidecar in sorted(root.glob("*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        path = sidecar.with_suffix(".npz")
        if not path.is_file() or sha256_file(path) != record.get("prediction_sha256"):
            raise AssertionError(f"D4 dense prediction hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            ids = data["dense_id"].astype("U96")
            predictions = data["predictions_native"].astype(np.float32, copy=True)
            key = (str(data["config"].item()), int(data["seed"].item()), int(data["horizon_steps"]))
        if len(set(ids.tolist())) != 3718 or set(ids.tolist()) != set(dense_ids.tolist()):
            raise AssertionError(f"D4 dense ID join mismatch: {path.name}")
        if predictions.shape != (3718, key[2], 28) or not np.isfinite(predictions).all():
            raise AssertionError(f"D4 prediction shape/finite check failed: {path.name}")
        id_index = {str(value): index for index, value in enumerate(ids)}
        predictions = predictions[np.asarray([id_index[str(value)] for value in dense_ids], dtype=np.int64)]
        if key in result:
            raise AssertionError(f"duplicate D4 prediction slot: {key}")
        result[key] = predictions
    if len(result) != 36:
        raise AssertionError(f"D4 requires 36 old/new model slots; found {len(result)} in {root}")
    return result


def common_frames(old_frame: int, new_frame: int, horizon: int, episode_frames: int) -> np.ndarray:
    if new_frame <= old_frame or new_frame - old_frame not in (7, 8):
        raise AssertionError("D4 pair must be adjacent at the observed 7/8-frame interval")
    start = new_frame + 1
    end = min(old_frame + horizon, episode_frames - 1)
    if end < start:
        return np.empty(0, dtype=np.int64)
    frames = np.arange(start, end + 1, dtype=np.int64)
    if np.any(frames - old_frame - 1 >= horizon) or np.any(frames - new_frame - 1 >= horizon):
        raise AssertionError("same absolute target frame falls outside one forecast horizon")
    return frames


def pair_metrics(old: np.ndarray, new: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    if old.shape != new.shape or old.shape != truth.shape or old.ndim != 2 or old.shape[1] != 28 or not len(old):
        raise AssertionError("D4 paired old/new/GT support must share shape [time,28]")
    result = {}
    for body, sl in BODIES.items():
        o, n, y, s = old[:, sl], new[:, sl], truth[:, sl], sigma[sl]
        old_error = np.abs(o - y) / s[None, :]
        new_error = np.abs(n - y) / s[None, :]
        result[f"{body}|L_old"] = float(old_error.mean())
        result[f"{body}|L_new"] = float(new_error.mean())
        result[f"{body}|U"] = float(old_error.mean() - new_error.mean())
        result[f"{body}|R"] = float(np.mean(np.abs(o - n) / s[None, :]))
        result[f"{body}|native_MAE_old"] = float(np.mean(np.abs(o - y)))
        result[f"{body}|native_MAE_new"] = float(np.mean(np.abs(n - y)))
        result[f"{body}|endpoint_new"] = float(np.mean(np.abs(n[-1] - y[-1]) / s))
        if len(o) > 1:
            result[f"{body}|FD_old"] = float(np.mean(np.abs(np.diff(o, axis=0) - np.diff(y, axis=0))))
            result[f"{body}|FD_new"] = float(np.mean(np.abs(np.diff(n, axis=0) - np.diff(y, axis=0))))
        else:
            result[f"{body}|FD_old"] = float("nan")
            result[f"{body}|FD_new"] = float("nan")
    return result


def group_means(pair_values: np.ndarray, pair_groups: np.ndarray, ordered_groups: np.ndarray) -> np.ndarray:
    output = np.empty(len(ordered_groups), dtype=np.float64)
    for index, group in enumerate(ordered_groups):
        selected = pair_values[pair_groups == group]
        if not len(selected):
            raise AssertionError(f"no D4 pairs for source group {group}")
        output[index] = float(np.mean(selected))
    return output


def summarize(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AssertionError("D4 group metric vector must be finite")
    estimate = float(values.mean())
    samples = values[draws].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def main() -> int:
    lock = json.loads((EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_INPUT_ONLY" or lock.get("final_test_access") != 0:
        raise AssertionError("D3-EQ D4 rolling protocol is not locked")
    pairs = list(csv.DictReader(PAIR_PATH.open(newline="", encoding="utf-8")))
    panel = np.load(INPUTS, allow_pickle=False)
    dense_ids = panel["dense_id"].astype("U96", copy=True)
    episodes = panel["episode_index"].astype(np.int64, copy=True)
    anchor_frames = panel["anchor_frame"].astype(np.int64, copy=True)
    anchor_times = panel["anchor_timestamp"].astype(np.float64, copy=True)
    id_index = {str(value): i for i, value in enumerate(dense_ids)}
    if len(pairs) != 7200 or len(dense_ids) != 3718:
        raise AssertionError("D4 pair/query counts differ from locked 7,200/3,718")
    group_audit = json.loads((EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    if group_audit.get("status") != "PASS" or group_audit["dense_panel"]["source_fingerprint_groups"] != 118:
        raise AssertionError("D4 dense source group mapping is not accepted")
    split_table = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"])
    source_group = {
        int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}")
        for row in split_table.to_pylist() if row["split"] in {"train", "validation"}
    }
    pair_groups = {str(row["pair_id"]): str(row["source_group_id"]) for row in pairs}
    if any(source_group[int(row["episode_index"])] != row["source_group_id"] for row in pairs):
        raise AssertionError("D4 source group IDs differ from locked duplicate-safe split fingerprints")
    ordered_groups = np.asarray(sorted(set(pair_groups.values())), dtype="U96")
    if len(ordered_groups) != 118:
        raise AssertionError("D4 rolling panel must have 118 source groups")

    source_registry = json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    raw_root = Path(source_registry["raw_root"]).resolve(strict=True)
    anchor_manifest = [row for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8")) if row["split"] == "dev" and row["population"] == "primary"]
    meta_by_episode = {}
    for row in anchor_manifest:
        meta_by_episode.setdefault(int(row["episode_index"]), row)
    episode_data = {}
    for episode in sorted(set(episodes.tolist())):
        row = meta_by_episode[int(episode)]
        raw_path = (raw_root / row["parquet_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("D4 dev Parquet path escaped the locked raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        dt = np.diff(timestamps)
        if actions.ndim != 2 or actions.shape[1] != 28 or not np.isfinite(actions).all() or not np.isfinite(timestamps).all():
            raise AssertionError(f"D4 raw episode schema/finite check failed: {episode}")
        if not len(dt) or np.any(dt <= 0):
            raise AssertionError(f"D4 raw timestamps are not monotonic: {episode}")
        episode_data[int(episode)] = (timestamps, actions)

    old_preds = load_prediction_grid(OLD_PRED_ROOT, dense_ids)
    new_preds = load_prediction_grid(NEW_PRED_ROOT, dense_ids)
    expected_keys = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in HORIZONS}
    if set(old_preds) != expected_keys or set(new_preds) != expected_keys:
        raise AssertionError("D4 requires 36 accepted old and 36 accepted D3-EQ dense bundles")
    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float64)

    metric_arrays: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    pair_result_rows = []
    pair_by_h = {h: [row for row in pairs if int(row["horizon_steps"]) == h] for h in HORIZONS}
    for horizon in HORIZONS:
        base = pair_by_h[horizon]
        if len(base) != 3600:
            raise AssertionError(f"D4 H{horizon} pair support is not 3,600")
        for config in CONFIGS:
            for seed in SEEDS:
                pred = new_preds[(config, seed, horizon)]
                measured = {f"{body}|{metric}": np.empty(len(base), dtype=np.float64) for body in BODIES for metric in ("L_old", "L_new", "U", "R", "native_MAE_old", "native_MAE_new", "FD_old", "FD_new", "endpoint_new")}
                for pair_index, row in enumerate(base):
                    episode = int(row["episode_index"])
                    old_frame, new_frame = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
                    old_id, new_id = row["old_dense_id"], row["new_dense_id"]
                    old_idx, new_idx = id_index[old_id], id_index[new_id]
                    if int(episodes[old_idx]) != episode or int(episodes[new_idx]) != episode:
                        raise AssertionError("D4 dense IDs do not match the locked pair episode")
                    if int(anchor_frames[old_idx]) != old_frame or int(anchor_frames[new_idx]) != new_frame:
                        raise AssertionError("D4 anchor frames do not match the fixed dense input IDs")
                    if abs(float(anchor_times[old_idx]) - float(row["old_anchor_timestamp"])) > 1e-6 or abs(float(anchor_times[new_idx]) - float(row["new_anchor_timestamp"])) > 1e-6:
                        raise AssertionError("D4 anchor timestamp join failed")
                    timestamps, actions = episode_data[episode]
                    frames = common_frames(old_frame, new_frame, horizon, len(actions))
                    expected_frames = np.arange(int(row["common_start_frame"]), int(row["common_end_frame"]) + 1, dtype=np.int64)
                    if not np.array_equal(frames, expected_frames) or len(frames) != int(row["common_target_steps"]):
                        raise AssertionError(f"D4 same-absolute-time support mismatch: {row['pair_id']}")
                    if not np.isfinite(timestamps[frames]).all() or np.any(np.diff(timestamps[frames]) <= 0):
                        raise AssertionError("D4 target timestamps are invalid/nonmonotonic")
                    old_forecast = pred[old_idx, frames - old_frame - 1]
                    new_forecast = pred[new_idx, frames - new_frame - 1]
                    truth = actions[frames].astype(np.float32)
                    values = pair_metrics(old_forecast, new_forecast, truth, sigma)
                    for name, value in values.items():
                        measured[name][pair_index] = value
                        body, metric = name.split("|", 1)
                    if config == "TS11-D" or config == "STATE-CTRL":
                        pair_result_rows.extend({
                            "pair_id": row["pair_id"], "source_group_id": row["source_group_id"],
                            "episode_index": episode, "config": config, "seed": seed,
                            "horizon_steps": horizon, "body": body, "metric": metric, "value": value,
                            "old_anchor_frame": old_frame, "new_anchor_frame": new_frame,
                            "common_start_frame": int(frames[0]), "common_end_frame": int(frames[-1]),
                            "common_target_steps": len(frames), "absolute_time_join": "PASS",
                            "final_test_access": 0,
                        } for name, value in values.items())
                metric_arrays[(config, seed, horizon)] = measured

    rng_by_h = {h: np.random.default_rng(BOOTSTRAP_SEED + h).integers(0, len(ordered_groups), size=(BOOTSTRAPS, len(ordered_groups)), dtype=np.int32) for h in HORIZONS}
    group_vector_by_h = {h: np.asarray([row["source_group_id"] for row in pair_by_h[h]], dtype="U96") for h in HORIZONS}
    model_rows = []
    for (config, seed, horizon), metrics in sorted(metric_arrays.items()):
        groups = group_vector_by_h[horizon]
        for metric_name, values in sorted(metrics.items()):
            body, metric = metric_name.split("|", 1)
            group_values = group_means(values, groups, ordered_groups)
            estimate, low, high = summarize(group_values, rng_by_h[horizon])
            model_rows.append({
                "config": config, "seed": seed, "horizon_steps": horizon,
                "body": body, "metric": metric, "estimate": estimate,
                "ci95_low": low, "ci95_high": high,
                "pairs": len(values), "source_groups": len(ordered_groups),
                "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video_sha256_fingerprint source group",
                "final_test_access": 0,
            })
    write_csv(OUT / "D3_EQ_D4_ROLLING_RESULTS_BY_SEED.csv", model_rows)
    write_csv(OUT / "D3_EQ_D4_PAIR_METRICS_PRIMARY_CELLS.csv", pair_result_rows)

    effects = []
    for horizon in HORIZONS:
        base_groups = group_vector_by_h[horizon]
        bidx = rng_by_h[horizon]
        for candidate, control, label in CONTRASTS:
            for seed in SEEDS:
                candidate_metrics = metric_arrays[(candidate, seed, horizon)]
                control_metrics = metric_arrays[(control, seed, horizon)]
                for body in BODIES:
                    for metric in ("U", "L_new", "FD_new", "R", "endpoint_new"):
                        name = f"{body}|{metric}"
                        pair_delta = candidate_metrics[name] - control_metrics[name]
                        group_delta = group_means(pair_delta, base_groups, ordered_groups)
                        estimate, low, high = summarize(group_delta, bidx)
                        effects.append({
                            "contrast": label, "candidate": candidate, "control": control,
                            "seed": seed, "horizon_steps": horizon, "body": body,
                            "metric": f"delta_{metric}",
                            "estimand": "candidate-control on identical adjacent pairs and common absolute target frames; U positive favors candidate; loss/FD/R negative favors candidate",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "pairs": len(pair_delta), "source_groups": len(ordered_groups),
                            "bootstrap_replicates": BOOTSTRAPS,
                            "final_test_access": 0,
                        })
    write_csv(OUT / "D3_EQ_D4_ROLLING_PAIRED_EFFECTS.csv", effects)

    objective_rows = []
    for horizon in HORIZONS:
        bidx = rng_by_h[horizon]
        pair_groups = group_vector_by_h[horizon]
        for config in CONFIGS:
            for body in BODIES:
                for metric in ("U", "L_new", "FD_new", "R"):
                    per_seed = {}
                    for seed in SEEDS:
                        key = (config, seed, horizon)
                        new = metric_arrays[key][f"{body}|{metric}"]
                        # Compute the same D4 metric for OLD_D3_ACTUAL on the
                        # same pair/time support, not from its pooled summary.
                        old_pred = old_preds[key]
                        panel_index = id_index
                        rows = pair_by_h[horizon]
                        old_values = np.empty(len(rows), dtype=np.float64)
                        for i, row in enumerate(rows):
                            ep = int(row["episode_index"])
                            a, b = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
                            frames = common_frames(a, b, horizon, len(episode_data[ep][1]))
                            oi, ni = panel_index[row["old_dense_id"]], panel_index[row["new_dense_id"]]
                            ts, actions = episode_data[ep]
                            old_f = old_pred[oi, frames - a - 1]
                            new_f = old_pred[ni, frames - b - 1]
                            values = pair_metrics(old_f, new_f, actions[frames].astype(np.float32), sigma)
                            old_values[i] = values[f"{body}|{metric}"]
                        per_seed[seed] = new - old_values
                    pooled = group_means(np.mean(np.stack([per_seed[s] for s in SEEDS]), axis=0), pair_groups, ordered_groups)
                    estimate, low, high = summarize(pooled, bidx)
                    objective_rows.append({
                        "comparison": "D3_EQ_V1_minus_OLD_D3_ACTUAL",
                        "config": config, "horizon_steps": horizon, "body": body, "metric": metric,
                        "estimate_three_seed_mean": estimate,
                        "ci95_low_descriptive": low, "ci95_high_descriptive": high,
                        "seed_estimates": json.dumps({str(s): float(per_seed[s].mean()) for s in SEEDS}, sort_keys=True),
                        "seed_sd": float(np.std([per_seed[s].mean() for s in SEEDS], ddof=1)),
                        "pairs": len(pair_by_h[horizon]), "source_groups": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS,
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_D4_ROLLING_OBJECTIVE_SENSITIVITY.csv", objective_rows)
    summary = {
        "status": "PASS",
        "protocol_id": lock["protocol_id"],
        "horizons": {str(h): len(pair_by_h[h]) for h in HORIZONS},
        "model_cells": len(metric_arrays),
        "source_groups": len(ordered_groups),
        "query_count": len(dense_ids),
        "common_absolute_target_frame_join": "PASS",
        "primary_estimands": ["U=L_old-L_new", "L_new", "FD_new", "R=revision magnitude", "U_TS11-D - U_STATE-CTRL"],
        "event_false_alarm_secondary": "not computed in this rolling scorer; thresholds/labels were not refit",
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "video_sha256_fingerprint source group",
        "dev_targets_loaded_after_training": True,
        "final_test_access": 0,
    }
    atomic_json(OUT / "D3_EQ_D4_ROLLING_QA.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
