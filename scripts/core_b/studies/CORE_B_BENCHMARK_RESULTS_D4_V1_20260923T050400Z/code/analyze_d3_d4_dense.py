"""CPU-only same-absolute-target D4 extension for accepted D3 dense predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import d4_eval as d4
from pathlibs import BENCH_ROOT, NEXT_ROOT, V3_ROOT

ROOT = NEXT_ROOT
EXT = ROOT / "d3_extension"
OUT = ROOT / "d4/d3_extension"
INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
PANEL = BENCH_ROOT / "d2/D2_DENSE_PANEL.csv"
PAIR_INPUT = ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"
PAIR_TIME = ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv"
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
CONTRASTS = (
    ("TS11-D", "TS10-D", "TS11_vs_TS10", "spatial_addition_given_temporal_preserved"),
    ("TS11-D", "TS01-D", "TS11_vs_TS01", "temporal_addition_given_spatial_preserved"),
    ("TS11-D", "TS11-S", "TS11D_vs_TS11S", "dynamic_vs_static_video"),
    ("TS11-D", "STATE-CTRL", "TS11_vs_STATE", "full_video_vs_state_only"),
)
BODIES = ("Hand", "Arm", "Full28")
PAIR_METRICS = (
    "L_old", "L_new", "U", "R", "native_MAE_old", "native_MAE_new",
    "FD_old", "FD_new", "endpoint_old", "endpoint_new",
    "coord_error_lower_share", "coord_error_higher_share", "coord_error_tie_share",
)
EFFECT_METRICS = ("U", "L_new", "FD_new", "R", "endpoint_new", "native_MAE_new")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260930
CROSS_BOOTSTRAPS = 10_000
CROSS_BOOTSTRAP_SEED = 20260931


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: safe_value(value) for key, value in row.items()} for row in rows)


def bootstrap_counts(n_groups: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_groups, size=(BOOTSTRAPS, n_groups), dtype=np.int32)
    counts = np.zeros((BOOTSTRAPS, n_groups), dtype=np.uint16)
    rows = np.broadcast_to(np.arange(BOOTSTRAPS, dtype=np.int32)[:, None], draw.shape)
    np.add.at(counts, (rows, draw), 1)
    return counts


def group_means(values: np.ndarray, pair_groups: np.ndarray, ordered_groups: list[str]) -> np.ndarray:
    output = np.full(len(ordered_groups), np.nan, dtype=np.float64)
    for index, group in enumerate(ordered_groups):
        local = pair_groups == group
        if local.any():
            output[index] = float(np.mean(values[local]))
    return output


def summarize_groups(values: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(values)
    if not valid.any():
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(values[valid]))
    numerator = counts @ np.nan_to_num(values, nan=0.0)
    denominator = counts @ valid.astype(np.uint16)
    reps = np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)
    reps = reps[np.isfinite(reps)]
    low, high = np.quantile(reps, [0.025, 0.975])
    return point, float(low), float(high)


def build_context():
    with np.load(INPUTS, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    dense_ids = data["dense_id"].astype("U96")
    n = len(dense_ids)
    if n != 3718 or len(set(dense_ids.tolist())) != n:
        raise ValueError("fixed dense input must contain 3718 unique IDs")
    panel = read_csv(PANEL)
    if len(panel) != n or any(row["final_test_access"] != "0" for row in panel):
        raise ValueError("D2 dense panel row count/final-test marker failed")
    panel_ids = [f"dense_e{int(r['episode_index']):06d}_a{int(r['anchor_frame']):06d}" for r in panel]
    if panel_ids != dense_ids.tolist():
        raise ValueError("D2 panel IDs do not exactly match dense input order")
    for key, dtype in (("episode_index", np.int64), ("anchor_frame", np.int64)):
        expected = np.asarray([int(r[key]) for r in panel], dtype=dtype)
        if not np.array_equal(data[key], expected):
            raise ValueError(f"dense input {key} does not exactly match panel")
    panel_times = np.asarray([float(r["anchor_timestamp"]) for r in panel], dtype=np.float64)
    if not np.allclose(data["anchor_timestamp"], panel_times, atol=1e-6, rtol=0):
        raise ValueError("dense input anchor times do not match panel")

    source_map = d4.load_source_groups()
    anchors = read_csv(V3_ROOT / "ANCHOR_MANIFEST.csv")
    meta_by_ep = {}
    for row in anchors:
        if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0":
            meta_by_ep.setdefault(int(row["episode_index"]), row)
    episode_ids = sorted(set(data["episode_index"].astype(int).tolist()))
    if len(episode_ids) != 118 or any(ep not in source_map or ep not in meta_by_ep for ep in episode_ids):
        raise ValueError("dense panel must map to 118 primary-dev episodes and source groups")
    episode_data = {ep: d4.load_episode(meta_by_ep[ep]) for ep in episode_ids}
    group_by_episode = {ep: source_map[ep] for ep in episode_ids}
    if len(set(group_by_episode.values())) != 118:
        raise ValueError("dense panel must contain 118 exact source groups")
    dense_group = np.asarray([group_by_episode[int(ep)] for ep in data["episode_index"]], dtype="U64")
    dense_index = {str(value): i for i, value in enumerate(dense_ids.tolist())}

    pair_rows = read_csv(PAIR_INPUT)
    support_rows = read_csv(PAIR_TIME)
    if len(pair_rows) != 7200 or len(support_rows) != 7200:
        raise ValueError("locked D4 pair/time support must contain 7200 rows")
    if any(row["final_test_access"] != "0" for row in pair_rows + support_rows):
        raise ValueError("D4 support contains nonzero final-test marker")
    support_by_id = {row["pair_id"]: row for row in support_rows}
    if len(support_by_id) != 7200:
        raise ValueError("D4 time-support pair IDs are not unique")
    pair_groups = {}
    pairs_by_h = {30: [], 60: []}
    for row in pair_rows:
        horizon = int(row["horizon_steps"])
        if horizon not in pairs_by_h:
            raise ValueError(f"unexpected horizon {horizon}")
        pair_id = row["pair_id"]
        support = support_by_id.get(pair_id)
        if support is None:
            raise ValueError(f"missing locked target support: {pair_id}")
        episode = int(row["episode_index"])
        if episode != int(support["episode_index"]) or episode not in episode_data:
            raise ValueError(f"pair/time-support episode mismatch: {pair_id}")
        old_id, new_id = row["old_dense_id"], row["new_dense_id"]
        if old_id not in dense_index or new_id not in dense_index:
            raise ValueError(f"unknown dense ID in pair manifest: {pair_id}")
        old_frame, new_frame = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
        timestamps, _ = episode_data[episode]
        frames = np.asarray(json.loads(support["common_target_frames"]), dtype=np.int64)
        recorded_times = np.asarray(json.loads(support["common_target_timestamps"]), dtype=np.float64)
        expected_frames = d4.valid_common_frames(old_frame, new_frame, horizon, len(timestamps))
        if len(frames) != int(row["common_target_steps"]) or not np.array_equal(frames, expected_frames):
            raise ValueError(f"common absolute frame support mismatch: {pair_id}")
        if not np.allclose(recorded_times, timestamps[frames], atol=1e-9, rtol=0):
            raise ValueError(f"common absolute timestamps mismatch: {pair_id}")
        if int(data["anchor_frame"][dense_index[old_id]]) != old_frame or int(data["anchor_frame"][dense_index[new_id]]) != new_frame:
            raise ValueError(f"pair anchors do not join dense input metadata: {pair_id}")
        pairs_by_h[horizon].append(row)
        pair_groups[pair_id] = group_by_episode[episode]
    if len(pairs_by_h[30]) != 3600 or len(pairs_by_h[60]) != 3600:
        raise ValueError("expected exactly 3600 adjacent pairs per horizon")
    return {
        "data": data, "dense_ids": dense_ids, "dense_index": dense_index,
        "episode_data": episode_data, "group_by_episode": group_by_episode,
        "pairs_by_h": pairs_by_h, "support_by_id": support_by_id, "pair_groups": pair_groups,
    }


def save_preflight(context):
    summary = {
        "status": "PASS",
        "protocol_id": "CORE_B_D3_D4_ROLLING_DENSE_EXTENSION_V1",
        "dense_ids": 3718, "episodes": 118, "source_groups": 118,
        "pairs_h30": len(context["pairs_by_h"][30]), "pairs_h60": len(context["pairs_by_h"][60]),
        "locked_common_absolute_frame_and_timestamp_join": "PASS",
        "final_test_access": 0,
    }
    path = EXT / "qa/D3_D4_DENSE_TARGET_JOIN_PREFLIGHT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def load_predictions(context):
    accepted = read_json(EXT / "D3_DENSE_PREDICTION_ACCEPTANCE.json")
    if accepted.get("status") != "PASS" or int(accepted.get("accepted_bundles", -1)) != 36:
        raise ValueError("D3 dense inference acceptance must be 36/36 PASS")
    data = context["data"]
    expected_ids = context["dense_ids"]
    expected = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in (30, 60)}
    predictions = {}
    markers = sorted((EXT / "predictions").glob("D3-*.json"))
    for marker_path in markers:
        record = read_json(marker_path)
        if record.get("status") != "PASS" or record.get("final_test_access") != 0:
            raise ValueError(f"D3 dense marker is not accepted: {marker_path.name}")
        path = Path(record["prediction_path"])
        if not path.is_relative_to(EXT) or not path.is_file() or sha256(path) != record["prediction_sha256"]:
            raise ValueError(f"D3 dense prediction path/hash failed: {marker_path.name}")
        with np.load(path, allow_pickle=False) as bundle:
            ids = bundle["dense_id"].astype("U96")
            episode = bundle["episode_index"].astype(np.int64)
            frame = bundle["anchor_frame"].astype(np.int64)
            task = bundle["original_task_index"].astype(np.int64)
            timestamp = bundle["anchor_timestamp"].astype(np.float64)
            value = bundle["predictions_native"]
            horizon = int(bundle["horizon_steps"])
            config = str(bundle["config"].item())
            seed = int(bundle["seed"].item())
        key = (config, seed, horizon)
        if key not in expected or key in predictions:
            raise ValueError(f"duplicate or unexpected D3 dense cell {key}")
        if not np.array_equal(ids, expected_ids) or value.shape != (3718, horizon, 28) or not np.isfinite(value).all():
            raise ValueError(f"dense ID/shape/finiteness failed: {path.name}")
        for field, observed in (
            ("episode_index", episode), ("anchor_frame", frame), ("original_task_index", task),
        ):
            if not np.array_equal(observed, data[field]):
                raise ValueError(f"D3 dense metadata join failed for {field}: {path.name}")
        if not np.allclose(timestamp, data["anchor_timestamp"], atol=1e-6, rtol=0):
            raise ValueError(f"D3 dense timestamp join failed: {path.name}")
        expected_hash = hashlib.sha256(expected_ids.tobytes()).hexdigest()
        if record.get("dense_ids_sha256") != expected_hash or int(record.get("rows", -1)) != 3718:
            raise ValueError(f"D3 dense sidecar panel identity failed: {marker_path.name}")
        predictions[key] = value.astype(np.float32, copy=False)
    if set(predictions) != expected:
        raise ValueError(f"D3 dense grid is incomplete: {len(predictions)}/36")
    return predictions


def metric_arrays(context, key, prediction):
    config, seed, horizon = key
    output = {f"{body}|{metric}": [] for body in BODIES for metric in PAIR_METRICS}
    dense_index = context["dense_index"]
    pair_rows = context["pairs_by_h"][horizon]
    for pair in pair_rows:
        pair_id = pair["pair_id"]
        support = context["support_by_id"][pair_id]
        episode = int(pair["episode_index"])
        timestamps, actions = context["episode_data"][episode]
        frames = np.asarray(json.loads(support["common_target_frames"]), dtype=np.int64)
        old_frame, new_frame = int(pair["old_anchor_frame"]), int(pair["new_anchor_frame"])
        old_step, new_step = frames - old_frame - 1, frames - new_frame - 1
        if np.any(old_step < 0) or np.any(new_step < 0) or np.any(old_step >= horizon) or np.any(new_step >= horizon):
            raise ValueError(f"common target frames outside D3 native horizon: {pair_id}")
        truth = actions[frames]
        old_pred = prediction[dense_index[pair["old_dense_id"]], old_step]
        new_pred = prediction[dense_index[pair["new_dense_id"]], new_step]
        result = d4.pair_metrics(old_pred, new_pred, truth, d4.SIGMA)
        for body in BODIES:
            for metric in PAIR_METRICS:
                output[f"{body}|{metric}"].append(result[f"{body}|{metric}"])
    return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    context = build_context()
    if args.preflight:
        save_preflight(context)
        return
    predictions = load_predictions(context)
    OUT.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    model_values = {}
    for (config, seed, horizon), prediction in sorted(predictions.items()):
        values = metric_arrays(context, (config, seed, horizon), prediction)
        model_values[(config, seed, horizon)] = values
        for pair_idx, pair in enumerate(context["pairs_by_h"][horizon]):
            pair_id = pair["pair_id"]
            episode = int(pair["episode_index"])
            group = context["pair_groups"][pair_id]
            support = context["support_by_id"][pair_id]
            for body in BODIES:
                row = {
                    "pair_id": pair_id, "episode_index": episode, "source_group_id": group,
                    "config": config, "seed": seed, "horizon_steps": horizon, "body": body,
                    "old_dense_id": pair["old_dense_id"], "new_dense_id": pair["new_dense_id"],
                    "old_anchor_frame": pair["old_anchor_frame"], "new_anchor_frame": pair["new_anchor_frame"],
                    "common_target_frames": support["common_target_frames"],
                    "common_target_timestamps": support["common_target_timestamps"],
                    "common_target_steps": pair["common_target_steps"],
                }
                for metric in PAIR_METRICS:
                    row[metric] = values[f"{body}|{metric}"][pair_idx]
                row["final_test_access"] = 0
                pair_rows.append(row)
    write_csv(OUT / "D3_D4_PAIR_METRICS.csv", pair_rows)

    groups_by_h = {
        h: sorted(set(context["pair_groups"][row["pair_id"]] for row in context["pairs_by_h"][h]))
        for h in (30, 60)
    }
    counts_by_h = {h: bootstrap_counts(len(groups_by_h[h]), BOOTSTRAP_SEED) for h in (30, 60)}
    group_idx_by_h = {
        h: np.asarray([groups_by_h[h].index(context["pair_groups"][r["pair_id"]]) for r in context["pairs_by_h"][h]], dtype=np.int32)
        for h in (30, 60)
    }
    model_rows = []
    model_group = {}
    for cell, metrics in sorted(model_values.items()):
        config, seed, horizon = cell
        for full_name, values in metrics.items():
            body, metric = full_name.split("|", 1)
            groups = group_means(values, np.asarray([context["pair_groups"][r["pair_id"]] for r in context["pairs_by_h"][horizon]], dtype="U64"), groups_by_h[horizon])
            estimate, low, high = summarize_groups(groups, counts_by_h[horizon])
            model_group[(config, seed, horizon, body, metric)] = groups
            model_rows.append({
                "model": config, "seed": seed, "horizon_steps": horizon, "body": body, "metric": metric,
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "pairs": 3600, "episodes": 118, "source_groups": 118,
                "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "exact video_sha256_fingerprint source group",
                "final_test_access": 0,
            })
    write_csv(OUT / "D3_D4_PRIMARY_RESULTS_BY_SEED.csv", model_rows)

    effects = []
    group_effects = {}
    for horizon in (30, 60):
        for seed in SEEDS:
            for candidate, control, name, interpretation in CONTRASTS:
                for body in BODIES:
                    for metric in EFFECT_METRICS:
                        diff = model_values[(candidate, seed, horizon)][f"{body}|{metric}"] - model_values[(control, seed, horizon)][f"{body}|{metric}"]
                        group = group_means(diff, np.asarray([context["pair_groups"][r["pair_id"]] for r in context["pairs_by_h"][horizon]], dtype="U64"), groups_by_h[horizon])
                        estimate, low, high = summarize_groups(group, counts_by_h[horizon])
                        name_metric = f"delta_{metric}"
                        group_effects[(name, seed, horizon, body, name_metric)] = group
                        effects.append({
                            "contrast": name, "interpretation": interpretation, "candidate": candidate, "control": control,
                            "seed": seed, "horizon_steps": horizon, "body": body, "metric": name_metric,
                            "estimand": "candidate minus control on identical pairs and common absolute target times",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "pairs": 3600, "episodes": 118, "source_groups": 118,
                            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                            "final_test_access": 0,
                        })
            for body in BODIES:
                for metric in ("U", "L_new", "FD_new"):
                    l00 = model_values[("TS00-D", seed, horizon)][f"{body}|{metric}"]
                    l10 = model_values[("TS10-D", seed, horizon)][f"{body}|{metric}"]
                    l01 = model_values[("TS01-D", seed, horizon)][f"{body}|{metric}"]
                    l11 = model_values[("TS11-D", seed, horizon)][f"{body}|{metric}"]
                    terms = (
                        ("factorial_interaction_TS11_TS10_TS01_TS00", (l11-l10)-(l01-l00)),
                        ("spatial_marginal_main_effect", ((l01-l00)+(l11-l10))/2.0),
                        ("temporal_marginal_main_effect", ((l10-l00)+(l11-l01))/2.0),
                    )
                    for name, values in terms:
                        group = group_means(values, np.asarray([context["pair_groups"][r["pair_id"]] for r in context["pairs_by_h"][horizon]], dtype="U64"), groups_by_h[horizon])
                        estimate, low, high = summarize_groups(group, counts_by_h[horizon])
                        effects.append({
                            "contrast": name, "interpretation": "paired source-group factorial contrast on rolling metric",
                            "candidate": "", "control": "", "seed": seed, "horizon_steps": horizon, "body": body,
                            "metric": f"{name}|delta_{metric}", "estimand": "paired factorial contrast on identical pairs",
                            "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "pairs": 3600, "episodes": 118, "source_groups": 118,
                            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                            "final_test_access": 0,
                        })
    write_csv(OUT / "D3_D4_PAIRED_EFFECTS.csv", effects)

    cross = []
    rng = np.random.default_rng(CROSS_BOOTSTRAP_SEED)
    for contrast in ("TS11_vs_TS10", "TS11_vs_TS01", "TS11D_vs_TS11S"):
        for horizon in (30, 60):
            for body in ("Arm", "Hand"):
                for metric in ("delta_U", "delta_L_new", "delta_FD_new"):
                    matrix = np.vstack([group_effects[(contrast, seed, horizon, body, metric)] for seed in SEEDS])
                    seed_points = matrix.mean(axis=1)
                    reps = np.empty(CROSS_BOOTSTRAPS, dtype=np.float64)
                    for start in range(0, CROSS_BOOTSTRAPS, 250):
                        stop = min(start+250, CROSS_BOOTSTRAPS)
                        seed_draw = rng.integers(0, len(SEEDS), size=(stop-start, len(SEEDS)))
                        group_draw = rng.integers(0, matrix.shape[1], size=(stop-start, matrix.shape[1]))
                        reps[start:stop] = matrix[seed_draw[:, :, None], group_draw[:, None, :]].mean(axis=(1, 2))
                    low, high = np.quantile(reps, [0.025, 0.975])
                    cross.append({
                        "contrast": contrast, "horizon_steps": horizon, "body": body, "metric": metric,
                        "estimate_equal_seed_group_mean": float(seed_points.mean()),
                        "seed_mean_sd_n3": float(seed_points.std(ddof=1)),
                        "ci95_descriptive_low": float(low), "ci95_descriptive_high": float(high),
                        "seed_20261001": float(seed_points[0]), "seed_20261002": float(seed_points[1]),
                        "seed_20261003": float(seed_points[2]), "pairs_per_seed": 3600,
                        "episodes": 118, "source_groups": 118,
                        "bootstrap_replicates": CROSS_BOOTSTRAPS, "bootstrap_seed": CROSS_BOOTSTRAP_SEED,
                        "interval_type": "post-hoc descriptive hierarchical bootstrap; no p-values",
                        "final_test_access": 0,
                    })
    write_csv(OUT / "D3_D4_CROSS_SEED_DESCRIPTIVE.csv", cross)

    summary = {
        "status": "COMPLETE_D3_DENSE_ROLLING_EXTENSION",
        "protocol_id": "CORE_B_D3_D4_ROLLING_DENSE_EXTENSION_V1",
        "accepted_d3_dense_bundles": 36, "expected_d3_dense_bundles": 36,
        "pair_metric_rows": len(pair_rows), "per_seed_results": len(model_rows),
        "paired_effect_rows": len(effects), "cross_seed_descriptive_rows": len(cross),
        "pairs_per_horizon": 3600, "episodes": 118, "source_groups": 118,
        "target_id_frame_timestamp_join": "PASS; D4 locked target support reused without changes",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "cross_seed_bootstrap_replicates": CROSS_BOOTSTRAPS,
        "cross_seed_bootstrap_seed": CROSS_BOOTSTRAP_SEED,
        "training": "none", "final_test_access": 0,
    }
    (OUT / "D3_D4_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    context = build_context()
    if args.preflight:
        save_preflight(context)
    else:
        main()
