"""D2 command-change event registry and fixed sparse/dense CPU baseline panel."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, atomic_write_json, read_csv, read_json, utc_now, write_csv, sha256_json


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
ANCHORS = V3 / "ANCHOR_MANIFEST.csv"
RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SIGMA = np.asarray(read_json(V3 / "data/V3_NORMALIZERS.json")["target_sigma"], dtype=np.float64)


def load_episode(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = RAW_ROOT / Path(row["parquet_path"])
    table = pq.read_table(path, columns=["timestamp", "action"])
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all() or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"timeline/action contract failed: {path}")
    return timestamp, action


def weighted_quantile(per_episode: list[np.ndarray], quantile: float) -> float:
    values = []
    weights = []
    for array in per_episode:
        if len(array):
            values.append(np.asarray(array, dtype=np.float64).reshape(-1))
            weights.append(np.full(len(array), 1.0 / len(array) / len(per_episode), dtype=np.float64))
    flat = np.concatenate(values)
    weight = np.concatenate(weights)
    order = np.argsort(flat, kind="mergesort")
    cumulative = np.cumsum(weight[order])
    return float(flat[order][np.searchsorted(cumulative, quantile, side="left")])


def q_signal(action: np.ndarray) -> np.ndarray:
    delta = np.abs(np.diff(action, axis=0)) / SIGMA[None, :]
    return np.stack((delta[:, :14].mean(axis=1), delta[:, 14:].mean(axis=1)), axis=1)


def detect_events(timestamp: np.ndarray, action: np.ndarray, thresholds: np.ndarray) -> list[dict[str, object]]:
    signal = q_signal(action)
    events: list[dict[str, object]] = []
    for group, name in enumerate(("Hand", "Arm")):
        on, off = float(thresholds[group, 0]), float(thresholds[group, 1])
        quiet = 0
        high = 0
        armed = False
        for q_index, value in enumerate(signal[:, group], start=1):
            if value <= off:
                quiet += 1
                high = 0
                if quiet >= 6:
                    armed = True
            elif value > on:
                quiet = 0
                if armed:
                    high += 1
                    if high >= 3:
                        onset_index = q_index - 2
                        events.append({"group": name, "onset_frame": int(onset_index), "onset_timestamp": float(timestamp[onset_index]), "signal_at_onset": float(signal[onset_index - 1, group]), "left_censored": bool(onset_index < 6)})
                        armed = False
                        high = 0
                else:
                    high = 0
            else:
                high = 0
                quiet = 0
    return sorted(events, key=lambda item: (item["onset_timestamp"], item["group"]))


def state_at_anchor(timestamp: np.ndarray, action: np.ndarray, anchor: int, thresholds: np.ndarray) -> tuple[bool, bool]:
    events = detect_events(timestamp[: anchor + 1], action[: anchor + 1], thresholds)
    # Re-run state with the same contract and expose whether a complete quiet arm is known.
    signal = q_signal(action[: anchor + 1])
    status = []
    for group in range(2):
        quiet = 0
        high = 0
        armed = False
        known = False
        for value in signal[:, group]:
            if value <= thresholds[group, 1]:
                quiet += 1
                high = 0
                if quiet >= 6:
                    armed = True
                    known = True
            elif value > thresholds[group, 0]:
                quiet = 0
                high = high + 1 if armed else 0
                if high >= 3:
                    armed = False
                    high = 0
            else:
                quiet = 0
                high = 0
        status.append((armed, known))
    return bool(status[0][0] and status[1][0]), bool(status[0][1] and status[1][1])


def match_events(predicted: list[float], truth: list[float], tolerance: float = 0.2) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    hits = 0
    timing: list[float] = []
    for value in sorted(predicted):
        candidates = [(abs(value - truth[index]), index) for index in range(len(truth)) if index not in used and abs(value - truth[index]) <= tolerance]
        if not candidates:
            continue
        distance, index = min(candidates)
        used.add(index)
        hits += 1
        timing.append(value - truth[index])
    return hits, len(predicted) - hits, len(truth) - hits, timing


def slope_prediction(action: np.ndarray, anchor: int, horizon: int) -> np.ndarray:
    slope = action[anchor] - action[max(0, anchor - 1)]
    steps = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
    return action[anchor][None, :] + steps * slope[None, :]


def choose_dense_episodes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["original_task_index"]].append(row)
    chosen = []
    for task in sorted(by_task, key=lambda value: int(value)):
        candidates = list({row["episode_index"]: row for row in by_task[task]}.items())
        candidates.sort(key=lambda item: hashlib.sha256(f"{item[0]}|CORE_B_D2_DENSE_V1".encode()).hexdigest())
        if candidates:
            chosen.append(candidates[0][1])
    return chosen[:128]


def evaluate_window(timestamp: np.ndarray, action: np.ndarray, anchor: int, horizon: int, thresholds: np.ndarray, model: str) -> dict[str, object]:
    future = np.repeat(action[anchor][None, :], horizon, axis=0) if model == "PERSISTENCE" else slope_prediction(action, anchor, horizon)
    prefix_ts = timestamp[: anchor + 1]
    prefix_action = action[: anchor + 1]
    future_ts = timestamp[anchor + 1: anchor + horizon + 1]
    combined_ts = np.concatenate((prefix_ts, future_ts))
    combined_action = np.concatenate((prefix_action, future), axis=0)
    pred_events = detect_events(combined_ts, combined_action, thresholds)
    predicted = [float(item["onset_timestamp"]) for item in pred_events if float(item["onset_timestamp"]) > float(timestamp[anchor])]
    truth_all = detect_events(timestamp, action, thresholds)
    truth = [float(item["onset_timestamp"]) for item in truth_all if float(timestamp[anchor]) < float(item["onset_timestamp"]) <= float(timestamp[anchor]) + (future_ts[-1] - timestamp[anchor] if len(future_ts) else 0.0) and not bool(item["left_censored"])]
    hits, false, missed, timing = match_events(predicted, truth)
    return {"predicted_count": len(predicted), "truth_count": len(truth), "hits": hits, "false_alarms": false, "misses": missed, "timing_error_seconds": float(np.mean(timing)) if timing else None, "predicted_onsets": predicted, "truth_onsets": truth}


def main() -> int:
    rows = [row for row in read_csv(ANCHORS) if row["split"] in {"train", "dev"} and row["population"] == "primary" and row["final_test_access"] == "0"]
    by_episode: dict[str, dict[str, str]] = {}
    for row in rows:
        by_episode.setdefault(row["episode_index"], row)
    train_rows = [row for row in rows if row["split"] == "train"]
    train_values = [[], []]
    train_episode_count = 0
    for episode in sorted({row["episode_index"] for row in train_rows}, key=int):
        ts, action = load_episode(by_episode[episode])
        signal = q_signal(action)
        train_values[0].append(signal[:, 0])
        train_values[1].append(signal[:, 1])
        train_episode_count += 1
    thresholds = np.zeros((2, 2), dtype=np.float64)
    for group in range(2):
        theta_on = max(weighted_quantile(train_values[group], 0.75), 1e-4)
        theta_off = min(weighted_quantile(train_values[group], 0.25), 0.5 * theta_on)
        thresholds[group] = (theta_on, theta_off)
    threshold_payload = {"protocol_id": "CORE_B_D2_COMMAND_CHANGE_V1", "theta_on": thresholds[:, 0].tolist(), "theta_off": thresholds[:, 1].tolist(), "group_order": ["Hand", "Arm"], "train_episodes": train_episode_count, "signal": "mean absolute native command difference / V3 train-only target_sigma over group joints", "state_machine": "6 quiet steps arm; 3 high steps onset; q values between thresholds hold state; initial missing history is unknown", "final_test_access": 0}
    threshold_payload["protocol_hash"] = sha256_json(threshold_payload)
    atomic_write_json(BENCH_ROOT / "protocol/D2_PROTOCOL_LOCKED.json", threshold_payload)
    atomic_write_json(BENCH_ROOT / "d2/D2_EVENT_THRESHOLDS.json", threshold_payload)

    dev_rows = [row for row in rows if row["split"] == "dev"]
    registry = []
    sparse = []
    dense_panel = []
    dense_eval = []
    dense_episodes = choose_dense_episodes(dev_rows)
    dense_ids = {row["episode_index"] for row in dense_episodes}
    for episode in sorted({row["episode_index"] for row in dev_rows}, key=int):
        meta = by_episode[episode]
        ts, action = load_episode(meta)
        events = detect_events(ts, action, thresholds)
        for event in events:
            registry.append({"split": "dev", "episode_index": episode, "task": meta["original_task_index"], **event, "final_test_access": 0})
        anchors = sorted({int(row["anchor_frame"]) for row in dev_rows if row["episode_index"] == episode})
        for anchor in anchors:
            if anchor < 119 or anchor + 60 >= len(action):
                continue
            for horizon in (30, 60):
                for model in ("PERSISTENCE", "COMMAND_SLOPE"):
                    result = evaluate_window(ts, action, anchor, horizon, thresholds, model)
                    sparse.append({"episode_index": episode, "task": meta["original_task_index"], "anchor_frame": anchor, "anchor_timestamp": float(ts[anchor]), "horizon_steps": horizon, "model": model, "coverage": "canonical_dev_primary", "hits": result["hits"], "false_alarms": result["false_alarms"], "misses": result["misses"], "predicted_count": result["predicted_count"], "truth_count": result["truth_count"], "timing_error_seconds": result["timing_error_seconds"], "final_test_access": 0})
        if episode in dense_ids:
            candidates = [119 + int(np.floor(7.5 * j)) for j in range(40)]
            queries = [a for a in candidates if a + 60 < len(action)]
            for a in queries:
                dense_panel.append({"episode_index": episode, "task": meta["original_task_index"], "anchor_frame": a, "anchor_timestamp": float(ts[a]), "query_index": queries.index(a), "future_1s_end_timestamp": float(ts[a + 30]), "future_2s_end_timestamp": float(ts[a + 60]), "block_id": f"{episode}:dense0", "final_test_access": 0})
                for horizon in (30, 60):
                    for model in ("PERSISTENCE", "COMMAND_SLOPE"):
                        result = evaluate_window(ts, action, a, horizon, thresholds, model)
                        dense_eval.append({"episode_index": episode, "anchor_frame": a, "horizon_steps": horizon, "model": model, **result})
    write_csv(BENCH_ROOT / "d2/D2_EVENT_REGISTRY.csv", registry, list(registry[0].keys()) if registry else ["episode_index"])
    write_csv(BENCH_ROOT / "d2/D2_SPARSE_RESULTS.csv", sparse, list(sparse[0].keys()) if sparse else ["episode_index"])
    write_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv", dense_panel, list(dense_panel[0].keys()) if dense_panel else ["episode_index"])
    dense_rows = []
    for model in ("PERSISTENCE", "COMMAND_SLOPE"):
        for horizon in (30, 60):
            subset = [row for row in dense_eval if row["model"] == model and int(row["horizon_steps"]) == horizon]
            exposure = 0.0
            for episode in dense_ids:
                episode_rows = [row for row in dense_panel if row["episode_index"] == episode]
                if len(episode_rows) >= 2:
                    exposure += float(episode_rows[-1]["anchor_timestamp"]) - float(episode_rows[0]["anchor_timestamp"])
            dense_rows.append({"model": model, "horizon_steps": horizon, "episodes": len(dense_ids), "queries": len(subset), "hits": sum(int(row["hits"]) for row in subset), "false_alarms": sum(int(row["false_alarms"]) for row in subset), "misses": sum(int(row["misses"]) for row in subset), "exposure_seconds": exposure, "false_alarms_per_hour": (sum(int(row["false_alarms"]) for row in subset) / exposure * 3600.0) if exposure > 0 else None, "status": "CPU_BASELINE_ONLY", "final_test_access": 0})
    write_csv(BENCH_ROOT / "d2/D2_DENSE_BASELINE_RESULTS.csv", dense_rows, list(dense_rows[0].keys()) if dense_rows else ["model"])
    summary = {"status": "CPU_BASELINES_COMPLETE", "sparse_status": "baseline_only", "dense_status": "baseline_only_fixed_panel", "event_registry_dev_rows": len(registry), "dense_episodes": len(dense_ids), "dense_queries": len(dense_panel), "learned_model_status": "awaiting V3 sealed prediction bundles", "final_test_access": 0, "protocol_hash": threshold_payload["protocol_hash"]}
    atomic_write_json(BENCH_ROOT / "d2/D2_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
