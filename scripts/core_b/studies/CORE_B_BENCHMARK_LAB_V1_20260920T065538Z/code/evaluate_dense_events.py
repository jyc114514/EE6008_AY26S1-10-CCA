"""Evaluate D2 dense learned predictions with forecast-episode clustering."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_csv, read_json, write_csv
from d2_event import detect_events, load_episode, match_events


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    thresholds = read_json(BENCH_ROOT / "d2/D2_EVENT_THRESHOLDS.json")
    theta = np.stack((np.asarray(thresholds["theta_on"], dtype=float), np.asarray(thresholds["theta_off"], dtype=float)), axis=1)
    panel = read_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    anchor_manifest = {row["episode_index"]: row for row in read_csv(V3 / "ANCHOR_MANIFEST.csv")}
    episode_cache = {}
    output = []
    for path in sorted((BENCH_ROOT / "predictions_dense").glob("V3-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            episode = archive["episode_index"].astype(int); frames = archive["anchor_frame"].astype(int); predictions = archive["predictions_native"]; horizon = int(archive["horizon_steps"]); cell = str(archive["cell"].item()); seed = int(archive["seed"].item())
        by_episode = defaultdict(list)
        for index in range(len(episode)): by_episode[int(episode[index])].append(index)
        hits = false = misses = predicted_count = truth_count = 0; leads = []; opportunity = 0
        for ep, indices in by_episode.items():
            if ep not in episode_cache:
                ts, action = load_episode(anchor_manifest[str(ep)])
                episode_cache[ep] = (ts, action, detect_events(ts, action, theta))
            ts, action, truth_all = episode_cache[ep]
            alarms = []
            for index in sorted(indices, key=lambda value: frames[value]):
                anchor = int(frames[index]); pred = predictions[index]
                if anchor + horizon >= len(action): continue
                combined_ts = np.concatenate((ts[: anchor + 1], ts[anchor + 1: anchor + horizon + 1]))
                combined_action = np.concatenate((action[: anchor + 1], pred), axis=0)
                events = detect_events(combined_ts, combined_action, theta)
                for event in events:
                    onset = float(event["onset_timestamp"])
                    if onset > float(ts[anchor]): alarms.append({"query": float(ts[anchor]), "onset": onset})
                gt = [float(item["onset_timestamp"]) for item in truth_all if not bool(item["left_censored"])]
                opportunity += sum(float(ts[anchor]) < value <= float(ts[anchor]) + (ts[anchor + horizon] - ts[anchor]) for value in gt)
            clusters = []
            for alarm in sorted(alarms, key=lambda value: (value["onset"], value["query"])):
                if not clusters or alarm["onset"] - clusters[-1]["first_onset"] > 0.2:
                    clusters.append({"first_onset": alarm["onset"], "first_query": alarm["query"]})
            gt_all = sorted({float(item["onset_timestamp"]) for item in truth_all if not bool(item["left_censored"])})
            predicted_count += len(clusters); truth_count += len(gt_all)
            used = set()
            for cluster in clusters:
                candidates = [(abs(cluster["first_onset"] - value), idx) for idx, value in enumerate(gt_all) if idx not in used and cluster["first_query"] < value <= cluster["first_query"] + horizon / 30.0 and abs(cluster["first_onset"] - value) <= 0.2]
                if candidates:
                    _, idx = min(candidates); used.add(idx); hits += 1; leads.append(gt_all[idx] - cluster["first_query"])
                else:
                    false += 1
            misses += len(gt_all) - len(used)
        exposure = 0.0
        for ep in by_episode:
            rows = [row for row in panel if row["episode_index"] == str(ep)]
            if len(rows) > 1: exposure += float(rows[-1]["anchor_timestamp"]) - float(rows[0]["anchor_timestamp"])
        output.append({"model": cell, "seed": seed, "horizon_steps": horizon, "episodes": len(by_episode), "queries": len(episode), "full_opportunity_events": opportunity, "hits": hits, "false_alarms": false, "misses": misses, "predicted_forecast_episodes": predicted_count, "truth_events": truth_count, "recall": hits / truth_count if truth_count else None, "recall_lead_ge_0_25": sum(value >= 0.25 for value in leads) / truth_count if truth_count else None, "recall_lead_ge_0_5": sum(value >= 0.5 for value in leads) / truth_count if truth_count else None, "recall_lead_ge_1": sum(value >= 1.0 for value in leads) / truth_count if truth_count else None, "mean_lead_seconds": float(np.mean(leads)) if leads else None, "false_alarms_per_hour": false / exposure * 3600.0 if exposure > 0 else None, "exposure_seconds": exposure, "status": "dense_learned_complete", "final_test_access": 0})
    write_csv(BENCH_ROOT / "d2/D2_DENSE_LEARNED_RESULTS.csv", output, list(output[0].keys()) if output else ["model"])
    summary = {"status": "DENSE_LEARNED_COMPLETE", "rows": len(output), "models": len({(row["model"], row["seed"]) for row in output}), "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__": raise SystemExit(main())
