"""Lock D2_V3 event definition using only canonical primary-train raw actions."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement"
SOURCE_REGISTRY = json.loads((V3 / "SOURCE_REGISTRY.json").read_text())
RAW_ROOT = Path(SOURCE_REGISTRY["raw_root"]).resolve()
SPLIT_PATH = Path(SOURCE_REGISTRY["split_manifest"])
SIGMA = np.asarray(json.loads((V3 / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
ON_FLOOR = 1e-4
PAST_STEPS = 8
QUIET_STEPS = 6
HIGH_STEPS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_ecdf_quantile(values: list[np.ndarray], weights: list[np.ndarray], quantile: float) -> float:
    flat_values = np.concatenate(values).astype(np.float64, copy=False)
    flat_weights = np.concatenate(weights).astype(np.float64, copy=False)
    if not len(flat_values) or len(flat_values) != len(flat_weights):
        raise ValueError("empty/misaligned train ECDF inputs")
    order = np.argsort(flat_values, kind="mergesort")
    cumulative = np.cumsum(flat_weights[order], dtype=np.float64)
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return float(flat_values[order[index]])


def detect_onsets(timestamp: np.ndarray, moving_signal: np.ndarray, theta_on: float, theta_off: float) -> list[dict[str, object]]:
    events = []
    quiet = 0
    high = 0
    armed = False
    for moving_index, value in enumerate(moving_signal):
        frame = PAST_STEPS + moving_index
        if value <= theta_off:
            quiet += 1
            high = 0
            if quiet >= QUIET_STEPS:
                armed = True
        elif value > theta_on:
            quiet = 0
            if armed:
                high += 1
                if high >= HIGH_STEPS:
                    onset_frame = frame - (HIGH_STEPS - 1)
                    events.append({
                        "onset_frame": int(onset_frame),
                        "onset_timestamp": float(timestamp[onset_frame]),
                        "confirmation_frame": int(frame),
                    })
                    armed = False
                    high = 0
        else:
            quiet = 0
            high = 0
    return events


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_meta: dict[int, dict[str, str]] = {}
    with (V3 / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train" and row["population"] == "primary" and row["final_test_access"] == "0":
                train_meta.setdefault(int(row["episode_index"]), row)
    train_ids = set(train_meta)

    # Predicate filters return train-only split metadata; no dev/test actions,
    # videos, targets, predictions, or outcome tables are read in this stage.
    split_table = pq.read_table(
        SPLIT_PATH,
        columns=["episode_index", "split", "split_assignment_key", "video_sha256_fingerprint", "v3_benchmark_inclusion"],
        filters=[("split", "=", "train"), ("v3_benchmark_inclusion", "=", "primary")],
    )
    split_rows = split_table.to_pylist()
    source_group_by_episode = {
        int(row["episode_index"]): str(row["video_sha256_fingerprint"])
        for row in split_rows
        if int(row["episode_index"]) in train_ids
    }
    missing_group_ids = sorted(train_ids - set(source_group_by_episode))
    if missing_group_ids:
        raise AssertionError(f"train source-group identity missing for {len(missing_group_ids)} episodes")
    group_episodes: dict[str, list[int]] = defaultdict(list)
    for episode in sorted(train_ids):
        group_episodes[source_group_by_episode[episode]].append(episode)

    episode_data: dict[int, dict[str, object]] = {}
    q_values: dict[str, dict[int, np.ndarray]] = {"Hand": {}, "Arm": {}}
    moving_values: dict[str, dict[int, np.ndarray]] = {"Hand": {}, "Arm": {}}
    all_dt_errors: list[np.ndarray] = []
    raw_root = RAW_ROOT.resolve(strict=True)
    episode_summary_rows: list[dict[str, object]] = []

    for episode, row in sorted(train_meta.items()):
        raw_path = (raw_root / row["parquet_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("train Parquet path escaped raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all() or not np.isfinite(timestamp).all():
            raise AssertionError(f"train raw action/timestamp schema failed for episode {episode}")
        dt = np.diff(timestamp)
        if not len(dt) or np.any(dt <= 0):
            raise AssertionError(f"train raw timestamps not strictly increasing for episode {episode}")
        dt_error = np.abs(dt - 1.0 / 30.0)
        all_dt_errors.append(dt_error)
        if float(dt_error.max()) > 1e-5:
            raise AssertionError(f"D2_V3 definition not applicable: raw source is not verified 30Hz (episode {episode})")
        q = np.abs(np.diff(action, axis=0)) / SIGMA[None, :]
        q_group = {"Hand": q[:, :14].mean(axis=1), "Arm": q[:, 14:].mean(axis=1)}
        m_group = {
            name: np.convolve(value, np.ones(PAST_STEPS, dtype=np.float64) / PAST_STEPS, mode="valid")
            for name, value in q_group.items()
        }
        for name in ("Hand", "Arm"):
            q_values[name][episode] = q_group[name]
            moving_values[name][episode] = m_group[name]
        episode_data[episode] = {
            "timestamp": timestamp,
            "action": action,
            "task": int(row["original_task_index"]),
            "group": source_group_by_episode[episode],
            "moving": m_group,
        }

    G = len(group_episodes)
    values_by_body: dict[str, list[np.ndarray]] = {"Hand": [], "Arm": []}
    weights_by_body: dict[str, list[np.ndarray]] = {"Hand": [], "Arm": []}
    for group, episodes in sorted(group_episodes.items()):
        for episode in episodes:
            for body in ("Hand", "Arm"):
                values = moving_values[body][episode]
                if not len(values):
                    continue
                weights = np.full(len(values), 1.0 / G / len(episodes) / len(values), dtype=np.float64)
                values_by_body[body].append(values)
                weights_by_body[body].append(weights)

    thresholds = {}
    train_events: dict[str, list[dict[str, object]]] = {"Hand": [], "Arm": []}
    support_rows = []
    for body in ("Hand", "Arm"):
        q75 = weighted_ecdf_quantile(values_by_body[body], weights_by_body[body], 0.75)
        theta_on = max(q75, ON_FLOOR)
        theta_off = 0.5 * theta_on
        floor_applied = q75 <= ON_FLOOR
        thresholds[body] = {"train_q75": q75, "theta_on": theta_on, "theta_off": theta_off, "floor_applied": floor_applied}
        for episode, data_row in episode_data.items():
            events = detect_onsets(data_row["timestamp"], data_row["moving"][body], theta_on, theta_off)
            for event in events:
                train_events[body].append({
                    "episode_index": episode,
                    "source_group_id": data_row["group"],
                    "task": data_row["task"],
                    **event,
                })

    # Independent-event support is deduplicated at source-group/body/frame.
    independent_events = {}
    for body, events in train_events.items():
        dedup = {(event["source_group_id"], event["onset_frame"]): event for event in events}
        independent_events[body] = list(dedup.values())
    train_episode_rows = []
    for episode, data_row in sorted(episode_data.items()):
        row = {
            "episode_index": episode,
            "source_group_id": data_row["group"],
            "task": data_row["task"],
            "raw_frames": len(data_row["timestamp"]),
            "valid_moving_timestamps": len(data_row["moving"]["Hand"]),
            "Hand_q_exact_zero_fraction": float(np.mean(q_values["Hand"][episode] == 0)),
            "Arm_q_exact_zero_fraction": float(np.mean(q_values["Arm"][episode] == 0)),
            "Hand_m_quiet_fraction": float(np.mean(data_row["moving"]["Hand"] <= thresholds["Hand"]["theta_off"])),
            "Arm_m_quiet_fraction": float(np.mean(data_row["moving"]["Arm"] <= thresholds["Arm"]["theta_off"])),
            "Hand_event_records": sum(event["episode_index"] == episode for event in train_events["Hand"]),
            "Arm_event_records": sum(event["episode_index"] == episode for event in train_events["Arm"]),
            "final_test_access": 0,
        }
        train_episode_rows.append(row)

    max_dt_error = max(float(values.max()) for values in all_dt_errors if len(values))
    definition_status = "definition_not_applicable" if any(thresholds[body]["floor_applied"] for body in thresholds) else "locked_train_thresholds_ready"
    protocol = {
        "protocol_id": "CORE_B_D2_EVENT_MEASUREMENT_SENSITIVITY_V3",
        "status": definition_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_exposure": "Train thresholds/support only; no dev labels, model errors, dense predictions, videos, or held-out data were read before lock.",
        "source_population": "V3 primary train raw command timelines; split filtered to train; duplicate-safe exact video_sha256_fingerprint source groups.",
        "source_group_count": G,
        "train_episode_count": len(episode_data),
        "train_timestamps_are_unique": True,
        "signal": "q_g(u)=mean_j(abs(command[u,j]-command[u-1,j])/sigma_y[j]); m_g(t)=mean of the latest eight legal raw q steps.",
        "smoothing_window_raw_steps": PAST_STEPS,
        "threshold_algorithm": "source group equal; episode equal within group; unique raw timestamps equal within episode; weighted empirical CDF; Q75 is smallest value with cumulative weight >=0.75; ties by value.",
        "state_machine": "at least 6 consecutive m<=theta_off steps arms; first of 3 consecutive m>theta_on steps is onset; fire disarms until another complete quiet run.",
        "thresholds": thresholds,
        "event_definition": "smoothed command-change onset, not physical velocity/contact; confirmation occurs two steps after the recorded first-high onset.",
        "timebase": {"expected_hz": 30, "max_abs_dt_error_seconds": max_dt_error, "acceptance_limit_seconds": 1e-5},
        "train_independent_events": {body: {
            "event_records": len(train_events[body]),
            "independent_source_group_events": len(independent_events[body]),
            "source_groups_with_event": len({event["source_group_id"] for event in independent_events[body]}),
            "tasks_with_event": len({event["task"] for event in independent_events[body]}),
        } for body in ("Hand", "Arm")},
        "raw_root": str(RAW_ROOT),
        "raw_paths_written_to_outputs": False,
        "normalizer_sha256": sha256_file(V3 / "data/V3_NORMALIZERS.json"),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "anchor_manifest_sha256": sha256_file(V3 / "ANCHOR_MANIFEST.csv"),
        "source_group_column": "video_sha256_fingerprint",
        "video_frames_accessed": False,
        "dev_accessed": False,
        "final_test_access": 0,
    }
    (OUT / "D2_V3_PROTOCOL_LOCKED.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    support_summary = {
        "status": definition_status,
        "train_source_groups": G,
        "train_episodes": len(episode_data),
        "unique_train_tasks": len({row["task"] for row in episode_data.values()}),
        "thresholds": thresholds,
        "independent_events": protocol["train_independent_events"],
        "Hand_support_status": "adequate_for_train_measurement" if protocol["train_independent_events"]["Hand"]["independent_source_group_events"] >= 30 and protocol["train_independent_events"]["Hand"]["tasks_with_event"] >= 10 else "low_support_train",
        "Arm_support_status": "adequate_for_train_measurement" if protocol["train_independent_events"]["Arm"]["independent_source_group_events"] >= 30 and protocol["train_independent_events"]["Arm"]["tasks_with_event"] >= 10 else "low_support_train",
        "dev_accessed": False,
        "final_test_access": 0,
    }
    (OUT / "D2_V3_TRAIN_SUPPORT_SUMMARY.json").write_text(json.dumps(support_summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "D2_V3_TRAIN_EPISODE_SUPPORT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(train_episode_rows[0]))
        writer.writeheader()
        writer.writerows(train_episode_rows)
    event_rows = []
    for body in ("Hand", "Arm"):
        for event in independent_events[body]:
            event_rows.append({"body": body, **event, "final_test_access": 0})
    with (OUT / "D2_V3_TRAIN_INDEPENDENT_EVENTS.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["body", "source_group_id", "episode_index", "task", "onset_frame", "onset_timestamp", "confirmation_frame", "final_test_access"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)
    print(json.dumps({
        "status": definition_status,
        "train_source_groups": G,
        "train_episodes": len(episode_data),
        "train_tasks": support_summary["unique_train_tasks"],
        "Hand_theta_on": thresholds["Hand"]["theta_on"],
        "Arm_theta_on": thresholds["Arm"]["theta_on"],
        "Hand_independent_events": protocol["train_independent_events"]["Hand"]["independent_source_group_events"],
        "Arm_independent_events": protocol["train_independent_events"]["Arm"]["independent_source_group_events"],
        "dev_accessed": False,
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
