"""Post-hoc D4 rolling-update contrasts for the separate V3_STATE dense study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_d3_d4_dense as common


ROOT = common.ROOT
OUT = ROOT / "d4/v3_state_dense"
PRED_ROOT = ROOT / "v3_state_dense"
STATE_ROOT = common.V3_ROOT.parent / "V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
STATES = ("FULL", "OBS", "ANCHOR")
VIDEOS = ("N", "D", "S")
SEEDS = (20260953, 20260954, 20260955)
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260932
CROSS_BOOTSTRAPS = 10_000
CROSS_BOOTSTRAP_SEED = 20260933
METRICS = ("U", "L_new", "FD_new")


def load_state_predictions(context):
    acceptance = common.read_json(PRED_ROOT / "V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    if acceptance.get("status") != "PASS" or int(acceptance.get("accepted_bundles", -1)) != 54:
        raise ValueError("V3_STATE dense prediction acceptance must be 54/54 PASS")
    expected_ids = context["dense_ids"]
    data = context["data"]
    expected = {(state, video, seed, horizon) for state in STATES for video in VIDEOS for seed in SEEDS for horizon in (30, 60)}
    predictions = {}
    for marker_path in sorted((PRED_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        record = common.read_json(marker_path)
        if record.get("status") != "PASS" or record.get("final_test_access") != 0:
            raise ValueError(f"V3_STATE dense sidecar failed: {marker_path.name}")
        path = Path(record["prediction_path"])
        if not path.is_relative_to(PRED_ROOT) or not path.is_file() or common.sha256(path) != record["prediction_sha256"]:
            raise ValueError(f"V3_STATE dense prediction hash/path failed: {marker_path.name}")
        with np.load(path, allow_pickle=False) as z:
            ids = z["dense_id"].astype("U96")
            ep = z["episode_index"].astype(np.int64)
            frame = z["anchor_frame"].astype(np.int64)
            times = z["anchor_timestamp"].astype(np.float64)
            task = z["original_task_index"].astype(np.int64)
            pred = z["predictions_native"]
            state, video = str(z["state_level"].item()), str(z["video_level"].item())
            seed, horizon = int(z["seed"].item()), int(z["horizon_steps"])
        key = (state, video, seed, horizon)
        if key not in expected or key in predictions:
            raise ValueError(f"duplicate or unexpected V3_STATE cell: {key}")
        if not np.array_equal(ids, expected_ids) or pred.shape != (3718, horizon, 28) or not np.isfinite(pred).all():
            raise ValueError(f"V3_STATE dense ID/shape/finiteness mismatch: {path.name}")
        for field, values in (("episode_index", ep), ("anchor_frame", frame), ("original_task_index", task)):
            if not np.array_equal(values, data[field]):
                raise ValueError(f"V3_STATE dense metadata mismatch for {field}: {path.name}")
        if not np.allclose(times, data["anchor_timestamp"], atol=1e-6, rtol=0):
            raise ValueError(f"V3_STATE dense anchor timestamp mismatch: {path.name}")
        if record.get("dense_ids_sha256") != hashlib.sha256(expected_ids.tobytes()).hexdigest():
            raise ValueError(f"V3_STATE dense sidecar ID hash mismatch: {marker_path.name}")
        predictions[key] = pred.astype(np.float32, copy=False)
    if set(predictions) != expected:
        raise ValueError(f"V3_STATE dense prediction grid incomplete: {len(predictions)}/54")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("supply --run")

    context = common.build_context()
    predictions = load_state_predictions(context)
    OUT.mkdir(parents=True, exist_ok=True)
    pair_groups = context["pair_groups"]
    rows = []
    cell_points = {}
    per_seed_effects = {}
    count_cache = {h: common.bootstrap_counts(118, BOOTSTRAP_SEED) for h in (30, 60)}
    group_orders = {h: sorted(set(pair_groups[r["pair_id"]] for r in context["pairs_by_h"][h])) for h in (30, 60)}

    # The V3_STATE matched controls are N and S within the same state mask.
    for state in STATES:
        for seed in SEEDS:
            for horizon in (30, 60):
                pred_d = predictions[(state, "D", seed, horizon)]
                pred_n = predictions[(state, "N", seed, horizon)]
                pred_s = predictions[(state, "S", seed, horizon)]
                pair_list = context["pairs_by_h"][horizon]
                local = {
                    (control, body, metric): []
                    for control in ("N", "S") for body in ("Arm", "Hand", "Full28") for metric in METRICS
                }
                for pair in pair_list:
                    pair_id = pair["pair_id"]
                    support = context["support_by_id"][pair_id]
                    episode = int(pair["episode_index"])
                    times, actions = context["episode_data"][episode]
                    frames = np.asarray(json.loads(support["common_target_frames"]), dtype=np.int64)
                    af, bf = int(pair["old_anchor_frame"]), int(pair["new_anchor_frame"])
                    oi, ni = frames-af-1, frames-bf-1
                    if np.any(oi < 0) or np.any(ni < 0) or np.any(oi >= horizon) or np.any(ni >= horizon):
                        raise ValueError(f"V3_STATE pair target outside native horizon: {pair_id}")
                    truth = actions[frames]
                    old_idx = context["dense_index"][pair["old_dense_id"]]
                    new_idx = context["dense_index"][pair["new_dense_id"]]
                    metrics_d = common.d4.pair_metrics(pred_d[old_idx, oi], pred_d[new_idx, ni], truth, common.d4.SIGMA)
                    metrics_n = common.d4.pair_metrics(pred_n[old_idx, oi], pred_n[new_idx, ni], truth, common.d4.SIGMA)
                    metrics_s = common.d4.pair_metrics(pred_s[old_idx, oi], pred_s[new_idx, ni], truth, common.d4.SIGMA)
                    for control, base in (("N", metrics_n), ("S", metrics_s)):
                        for body in ("Arm", "Hand", "Full28"):
                            for metric in METRICS:
                                local[(control, body, metric)].append(
                                    metrics_d[f"{body}|{metric}"] - base[f"{body}|{metric}"]
                                )

                groups = np.asarray([pair_groups[r["pair_id"]] for r in pair_list], dtype="U64")
                ordered = group_orders[horizon]
                for control in ("N", "S"):
                    for body in ("Arm", "Hand", "Full28"):
                        for metric in METRICS:
                            values = np.asarray(local[(control, body, metric)], dtype=np.float64)
                            grouped = common.group_means(values, groups, ordered)
                            estimate, low, high = common.summarize_groups(grouped, count_cache[horizon])
                            contrast = f"{state}_D_vs_{control}"
                            per_seed_effects[(contrast, seed, horizon, body, f"delta_{metric}")] = grouped
                            rows.append({
                                "contrast": contrast, "state_level": state, "candidate": f"{state}-D",
                                "control": f"{state}-{control}", "seed": seed, "horizon_steps": horizon,
                                "body": body, "metric": f"delta_{metric}", "estimate": estimate,
                                "ci95_low": low, "ci95_high": high, "pairs": 3600,
                                "episodes": 118, "source_groups": 118,
                                "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
                                "final_test_access": 0,
                            })
    common.write_csv(OUT / "D4_PAIRED_EFFECTS.csv", rows)

    rng = np.random.default_rng(CROSS_BOOTSTRAP_SEED)
    cross = []
    for state in STATES:
        for control in ("N", "S"):
            contrast = f"{state}_D_vs_{control}"
            for horizon in (30, 60):
                for body in ("Arm", "Hand"):
                    for metric in ("delta_U", "delta_L_new", "delta_FD_new"):
                        matrix = np.vstack([per_seed_effects[(contrast, seed, horizon, body, metric)] for seed in SEEDS])
                        seed_points = matrix.mean(axis=1)
                        reps = np.empty(CROSS_BOOTSTRAPS, dtype=np.float64)
                        for start in range(0, CROSS_BOOTSTRAPS, 250):
                            stop = min(start + 250, CROSS_BOOTSTRAPS)
                            seed_draw = rng.integers(0, len(SEEDS), size=(stop-start, len(SEEDS)))
                            group_draw = rng.integers(0, matrix.shape[1], size=(stop-start, matrix.shape[1]))
                            reps[start:stop] = matrix[seed_draw[:, :, None], group_draw[:, None, :]].mean(axis=(1, 2))
                        low, high = np.quantile(reps, [0.025, 0.975])
                        cross.append({
                            "contrast": contrast, "state_level": state, "horizon_steps": horizon,
                            "body": body, "metric": metric,
                            "estimate_equal_seed_group_mean": float(seed_points.mean()),
                            "seed_mean_sd_n3": float(seed_points.std(ddof=1)),
                            "ci95_descriptive_low": float(low), "ci95_descriptive_high": float(high),
                            "seed_20260953": float(seed_points[0]), "seed_20260954": float(seed_points[1]),
                            "seed_20260955": float(seed_points[2]), "pairs_per_seed": 3600,
                            "episodes": 118, "source_groups": 118,
                            "bootstrap_replicates": CROSS_BOOTSTRAPS,
                            "bootstrap_seed": CROSS_BOOTSTRAP_SEED,
                            "interval_type": "post-hoc descriptive hierarchical bootstrap; no p-values",
                            "final_test_access": 0,
                        })
    common.write_csv(OUT / "D4_CROSS_SEED_DESCRIPTIVE.csv", cross)
    summary = {
        "status": "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION",
        "protocol_id": "CORE_B_V3_STATE_D4_DENSE_EXTENSION_V1",
        "accepted_bundles": 54, "expected_bundles": 54,
        "paired_effect_rows": len(rows), "cross_seed_descriptive_rows": len(cross),
        "pairs_per_horizon": 3600, "episodes": 118, "source_groups": 118,
        "target_join": "PASS; fixed D4 absolute frames/timestamps reused",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "cross_seed_bootstrap_replicates": CROSS_BOOTSTRAPS,
        "cross_seed_bootstrap_seed": CROSS_BOOTSTRAP_SEED,
        "outcome_exposure": "development-only post-hoc diagnostic; no p-values or Holm claim",
        "final_test_access": 0,
    }
    (OUT / "D4_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
