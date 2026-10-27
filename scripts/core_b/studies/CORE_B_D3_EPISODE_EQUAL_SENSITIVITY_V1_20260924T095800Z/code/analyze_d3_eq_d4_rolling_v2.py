"""Canonical D4 rolling evaluator for same absolute target times (D3-EQ)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import analyze_d3_eq_d4_rolling as helpers


EQ_ROOT = helpers.EQ_ROOT
PROJECT_ROOT = helpers.PROJECT_ROOT
V3_ROOT = helpers.V3_ROOT
RAW_ROOT = helpers.RAW_ROOT
SPLIT_MANIFEST = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
OUT = helpers.OUT
SEEDS = helpers.SEEDS
HORIZONS = helpers.HORIZONS
CONFIGS = helpers.CONFIGS
BODIES = helpers.BODIES
BOOTSTRAPS = helpers.BOOTSTRAPS
BOOTSTRAP_SEED = helpers.BOOTSTRAP_SEED
write_csv = helpers.write_csv
atomic_json = helpers.atomic_json


def main() -> int:
    lock = json.loads((EQ_ROOT / "protocol/D3_EQ_D4_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    pair_path = EQ_ROOT / "protocol/D3_EQ_D4_PAIR_MANIFEST_INPUT_ONLY.csv"
    pairs = list(csv.DictReader(pair_path.open(newline="", encoding="utf-8")))
    panel = np.load(helpers.INPUTS, allow_pickle=False)
    dense_ids = panel["dense_id"].astype("U96", copy=True)
    episodes = panel["episode_index"].astype(np.int64, copy=True)
    frames_all = panel["anchor_frame"].astype(np.int64, copy=True)
    times_all = panel["anchor_timestamp"].astype(np.float64, copy=True)
    dense_index = {str(value): i for i, value in enumerate(dense_ids)}
    if len(pairs) != 7200 or len(dense_ids) != 3718:
        raise AssertionError("D4 pair or query row count differs from the locked contract")
    audit = json.loads((EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit["dense_panel"]["episode_to_source_group_bijection"] is not True:
        raise AssertionError("dense source-group identity is not accepted")
    split_table = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"])
    source_groups = {
        int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}")
        for row in split_table.to_pylist() if row["split"] in {"train", "validation"}
    }
    for row in pairs:
        episode = int(row["episode_index"])
        if source_groups[episode] != row["source_group_id"]:
            raise AssertionError("D4 pair source-group key differs from the duplicate-safe split")
    ordered_groups = np.asarray(sorted(set(source_groups[int(row["episode_index"])] for row in pairs)), dtype="U96")
    if len(ordered_groups) != 118:
        raise AssertionError("D4 rolling panel must have 118 source fingerprint groups")

    source_registry = json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    raw_root = Path(source_registry["raw_root"]).resolve(strict=True)
    anchor_manifest = [row for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8")) if row["split"] == "dev" and row["population"] == "primary"]
    meta_by_episode = {}
    for row in anchor_manifest:
        meta_by_episode.setdefault(int(row["episode_index"]), row)
    episode_data = {}
    panel_episodes = sorted(set(int(ep) for ep in episodes))
    for episode in panel_episodes:
        raw_path = (raw_root / meta_by_episode[episode]["parquet_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("D4 source action path escaped the locked raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        ts = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        dt = np.diff(ts)
        if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all() or not np.isfinite(ts).all() or np.any(dt <= 0):
            raise AssertionError(f"D4 raw dev episode schema/timestamp check failed: {episode}")
        episode_data[episode] = (ts, action)

    old_predictions = helpers.load_prediction_grid(helpers.OLD_PRED_ROOT, dense_ids)
    new_predictions = helpers.load_prediction_grid(helpers.NEW_PRED_ROOT, dense_ids)
    expected = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in HORIZONS}
    if set(old_predictions) != expected or set(new_predictions) != expected:
        raise AssertionError("D4 old/new dense prediction grids must each cover all 36 cells")
    sigma = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))["target_sigma"], dtype=np.float64)
    pair_by_h = {h: [row for row in pairs if int(row["horizon_steps"]) == h] for h in HORIZONS}
    if {h: len(rows) for h, rows in pair_by_h.items()} != {30: 3600, 60: 3600}:
        raise AssertionError("D4 rolling pairs must contain 3,600 rows per horizon")

    d4_metrics: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    pair_rows = []
    for horizon in HORIZONS:
        base = pair_by_h[horizon]
        for config in CONFIGS:
            for seed in SEEDS:
                pred = new_predictions[(config, seed, horizon)]
                values = {f"{body}|{metric}": np.empty(len(base), dtype=np.float64) for body in BODIES for metric in ("L_old", "L_new", "U", "R", "native_MAE_old", "native_MAE_new", "FD_old", "FD_new", "endpoint_new")}
                for i, row in enumerate(base):
                    episode = int(row["episode_index"])
                    old_frame, new_frame = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
                    old_index, new_index = dense_index[row["old_dense_id"]], dense_index[row["new_dense_id"]]
                    if int(episodes[old_index]) != episode or int(episodes[new_index]) != episode:
                        raise AssertionError("D4 dense ID/episode join mismatch")
                    if int(frames_all[old_index]) != old_frame or int(frames_all[new_index]) != new_frame:
                        raise AssertionError("D4 dense ID/frame join mismatch")
                    if abs(float(times_all[old_index]) - float(row["old_anchor_timestamp"])) > 1e-6 or abs(float(times_all[new_index]) - float(row["new_anchor_timestamp"])) > 1e-6:
                        raise AssertionError("D4 anchor timestamp join mismatch")
                    timestamps, actions = episode_data[episode]
                    target_frames = helpers.common_frames(old_frame, new_frame, horizon, len(actions))
                    expected_frames = np.arange(int(row["common_start_frame"]), int(row["common_end_frame"]) + 1, dtype=np.int64)
                    if not np.array_equal(target_frames, expected_frames) or len(target_frames) != int(row["common_target_steps"]):
                        raise AssertionError(f"D4 common absolute frame support mismatch: {row['pair_id']}")
                    if not np.isfinite(timestamps[target_frames]).all() or np.any(np.diff(timestamps[target_frames]) <= 0):
                        raise AssertionError("D4 common future timestamps are invalid")
                    old_forecast = pred[old_index, target_frames - old_frame - 1]
                    new_forecast = pred[new_index, target_frames - new_frame - 1]
                    truth = actions[target_frames].astype(np.float32)
                    measured = helpers.pair_metrics(old_forecast, new_forecast, truth, sigma)
                    for metric_name, metric_value in measured.items():
                        values[metric_name][i] = metric_value
                    if config in ("TS11-D", "STATE-CTRL"):
                        for body in BODIES:
                            pair_rows.append({
                                "pair_id": row["pair_id"],
                                "source_group_id": row["source_group_id"],
                                "episode_index": episode,
                                "config": config,
                                "seed": seed,
                                "horizon_steps": horizon,
                                "body": body,
                                "old_anchor_frame": old_frame,
                                "new_anchor_frame": new_frame,
                                "common_start_frame": int(target_frames[0]),
                                "common_end_frame": int(target_frames[-1]),
                                "common_target_steps": len(target_frames),
                                **{name.split("|", 1)[1]: float(value) for name, value in measured.items() if name.startswith(f"{body}|")},
                                "absolute_time_join": "PASS",
                                "final_test_access": 0,
                            })
                d4_metrics[(config, seed, horizon)] = values

    rng_by_h = {
        h: np.random.default_rng(BOOTSTRAP_SEED + h).integers(0, len(ordered_groups), size=(BOOTSTRAPS, len(ordered_groups)), dtype=np.int32)
        for h in HORIZONS
    }
    pair_groups_by_h = {h: np.asarray([row["source_group_id"] for row in pair_by_h[h]], dtype="U96") for h in HORIZONS}
    model_rows = []
    for (config, seed, horizon), metrics in sorted(d4_metrics.items()):
        for name, pair_values in sorted(metrics.items()):
            body, metric = name.split("|", 1)
            group_values = helpers.group_means(pair_values, pair_groups_by_h[horizon], ordered_groups)
            estimate, low, high = helpers.summarize(group_values, rng_by_h[horizon])
            model_rows.append({
                "config": config, "seed": seed, "horizon_steps": horizon,
                "body": body, "metric": metric,
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "pair_count": len(pair_values), "source_group_count": len(ordered_groups),
                "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video_sha256_fingerprint source group",
                "final_test_access": 0,
            })
    write_csv(OUT / "D3_EQ_D4_ROLLING_RESULTS_BY_SEED.csv", model_rows)
    write_csv(OUT / "D3_EQ_D4_PAIR_METRICS_TS11D_STATE.csv", pair_rows)

    effect_rows = []
    for horizon in HORIZONS:
        groups = pair_groups_by_h[horizon]
        for candidate, control, contrast in helpers.CONTRASTS:
            for seed in SEEDS:
                for body in BODIES:
                    for metric in ("U", "L_new", "FD_new", "R", "endpoint_new"):
                        key = f"{body}|{metric}"
                        difference = d4_metrics[(candidate, seed, horizon)][key] - d4_metrics[(control, seed, horizon)][key]
                        grouped = helpers.group_means(difference, groups, ordered_groups)
                        estimate, low, high = helpers.summarize(grouped, rng_by_h[horizon])
                        effect_rows.append({
                            "contrast": contrast, "candidate": candidate, "control": control,
                            "seed": seed, "horizon_steps": horizon, "body": body,
                            "metric": f"delta_{metric}",
                            "estimand": "candidate-control on identical adjacent pairs/common absolute target frames; U positive favors candidate, loss/FD/revision negative favors candidate",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "pair_count": len(difference), "source_group_count": len(ordered_groups),
                            "bootstrap_replicates": BOOTSTRAPS, "final_test_access": 0,
                        })
    write_csv(OUT / "D3_EQ_D4_ROLLING_PAIRED_EFFECTS.csv", effect_rows)

    sensitivity_rows = []
    for horizon in HORIZONS:
        groups = pair_groups_by_h[horizon]
        for config in CONFIGS:
            for body in BODIES:
                for metric in ("U", "L_new", "FD_new", "R"):
                    seed_deltas = {}
                    for seed in SEEDS:
                        key = (config, seed, horizon)
                        old_pred = old_predictions[key]
                        pair_values = np.empty(len(pair_by_h[horizon]), dtype=np.float64)
                        for i, row in enumerate(pair_by_h[horizon]):
                            episode = int(row["episode_index"])
                            a, b = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
                            frames = helpers.common_frames(a, b, horizon, len(episode_data[episode][1]))
                            oi, ni = dense_index[row["old_dense_id"]], dense_index[row["new_dense_id"]]
                            timestamps, actions = episode_data[episode]
                            old_forecast = old_pred[oi, frames - a - 1]
                            new_forecast = old_pred[ni, frames - b - 1]
                            value = helpers.pair_metrics(old_forecast, new_forecast, actions[frames].astype(np.float32), sigma)
                            pair_values[i] = value[f"{body}|{metric}"]
                        new_values = d4_metrics[key][f"{body}|{metric}"]
                        seed_deltas[seed] = new_values - pair_values
                    pooled_pairs = np.mean(np.stack([seed_deltas[s] for s in SEEDS]), axis=0)
                    grouped = helpers.group_means(pooled_pairs, groups, ordered_groups)
                    estimate, low, high = helpers.summarize(grouped, rng_by_h[horizon])
                    sensitivity_rows.append({
                        "comparison": "D3_EQ_V1_minus_OLD_D3_ACTUAL",
                        "config": config, "horizon_steps": horizon, "body": body, "metric": metric,
                        "estimate_three_seed_mean": estimate, "ci95_low_descriptive": low, "ci95_high_descriptive": high,
                        "seed_estimates": json.dumps({str(s): float(seed_deltas[s].mean()) for s in SEEDS}, sort_keys=True),
                        "seed_sd": float(np.std([seed_deltas[s].mean() for s in SEEDS], ddof=1)),
                        "pair_count": len(pair_by_h[horizon]), "source_group_count": len(ordered_groups),
                        "bootstrap_replicates": BOOTSTRAPS, "final_test_access": 0,
                    })
    write_csv(OUT / "D3_EQ_D4_ROLLING_OBJECTIVE_SENSITIVITY.csv", sensitivity_rows)

    summary = {
        "status": "PASS",
        "protocol_id": lock["protocol_id"],
        "old_and_new_dense_grids": "36/36",
        "pairs_by_horizon": {str(h): len(pair_by_h[h]) for h in HORIZONS},
        "source_groups": len(ordered_groups),
        "query_count": len(dense_ids),
        "same_absolute_future_frames_verified": True,
        "fusion_vs_state_utility_reported": True,
        "absolute_new_time_error_and_FD_reported": True,
        "event_false_alarm_secondary": "not computed in this version; fixed thresholds were not refit",
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "video_sha256_fingerprint source group",
        "final_test_access": 0,
    }
    atomic_json(OUT / "D3_EQ_D4_ROLLING_QA.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
