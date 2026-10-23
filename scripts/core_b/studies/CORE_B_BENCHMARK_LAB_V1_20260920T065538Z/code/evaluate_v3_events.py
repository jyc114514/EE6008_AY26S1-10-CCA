"""Evaluate sealed V3 trajectories on the locked sparse command-event panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_csv, read_json, write_csv
from d2_event import detect_events, load_episode, match_events


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = REPO_ROOT.parent.parent / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"


def main() -> int:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    thresholds = read_json(BENCH_ROOT / "d2/D2_EVENT_THRESHOLDS.json")
    theta = np.stack((np.asarray(thresholds["theta_on"], dtype=float), np.asarray(thresholds["theta_off"], dtype=float)), axis=1)
    anchor_to_index = {str(value): index for index, value in enumerate(data["anchor_id"])}
    dev_rows = {str(data["anchor_id"][index]): int(index) for index in np.flatnonzero(data["split_code"] == 1)}
    anchor_manifest = {row["anchor_id"]: row for row in read_csv(V3 / "ANCHOR_MANIFEST.csv")}
    episode_cache: dict[int, tuple[np.ndarray, np.ndarray, list[dict[str, object]]]] = {}
    output = []
    for marker_path in sorted((V3 / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = read_json(marker_path)
        if marker.get("status") != "PASS":
            continue
        cell, seed, horizon = str(marker["cell"]), int(marker["seed"]), int(marker["horizon_steps"])
        with np.load(marker_path.parent / "primary.npz", allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64")
            predictions = archive["predictions_native"]
        totals = {"hits": 0, "false_alarms": 0, "misses": 0, "predicted": 0, "truth": 0, "timing": []}
        windows = 0
        for row_index, anchor_id in enumerate(ids):
            key = str(anchor_id)
            if key not in dev_rows:
                continue
            source = dev_rows[key]
            episode = int(data["episode_index"][source])
            if episode not in episode_cache:
                row = anchor_manifest[key]
                timestamp, action = load_episode(row)
                episode_cache[episode] = (timestamp, action, detect_events(timestamp, action, theta))
            timestamp, action, truth_all = episode_cache[episode]
            anchor = int(data["anchor_frame"][source])
            if anchor + horizon >= len(action):
                continue
            predicted_future = predictions[row_index, :horizon]
            combined_ts = np.concatenate((timestamp[: anchor + 1], timestamp[anchor + 1: anchor + horizon + 1]))
            combined_action = np.concatenate((action[: anchor + 1], predicted_future), axis=0)
            predicted_events = detect_events(combined_ts, combined_action, theta)
            predicted = [float(item["onset_timestamp"]) for item in predicted_events if float(item["onset_timestamp"]) > float(timestamp[anchor])]
            truth = [float(item["onset_timestamp"]) for item in truth_all if float(timestamp[anchor]) < float(item["onset_timestamp"]) <= float(timestamp[anchor]) + (timestamp[anchor + horizon] - timestamp[anchor]) and not bool(item["left_censored"])]
            hits, false_alarms, misses, timing = match_events(predicted, truth)
            totals["hits"] += hits; totals["false_alarms"] += false_alarms; totals["misses"] += misses; totals["predicted"] += len(predicted); totals["truth"] += len(truth); totals["timing"].extend(timing)
            windows += 1
        output.append({"model": cell, "seed": seed, "horizon_steps": horizon, "N_window": windows, "hits": totals["hits"], "false_alarms": totals["false_alarms"], "misses": totals["misses"], "predicted": totals["predicted"], "truth": totals["truth"], "precision": totals["hits"] / totals["predicted"] if totals["predicted"] else None, "recall": totals["hits"] / totals["truth"] if totals["truth"] else None, "mean_timing_error_seconds": float(np.mean(totals["timing"])) if totals["timing"] else None, "status": "sparse_learned_complete", "final_test_access": 0})
    write_csv(BENCH_ROOT / "d2/D2_SPARSE_LEARNED_RESULTS.csv", output, list(output[0].keys()) if output else ["model"])
    summary = {"status": "SPARSE_LEARNED_COMPLETE", "bundles": len(output), "dense_status": "awaiting_dense_inference", "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "d2/D2_LEARNED_SUMMARY.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
