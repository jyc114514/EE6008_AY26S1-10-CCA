#!/usr/bin/env python3
"""Prepare the fixed V2 task domain from train-only raw episodes and accepted caches."""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import os
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EPS = 1e-6
HISTORY_STEPS = 32
HORIZON = 30
STATE_DIM = 71
ACTION_DIM = 28
TASK_IDS = (119, 94, 90, 141, 230)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        requested = content.replace("\r\n", "\n").replace("\r", "\n")
        if existing != requested:
            raise RuntimeError(f"existing receipt differs; preserve it and inspect before retry: {path}")
        return
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        requested = json.loads(content)
        existing.pop("created_utc", None)
        requested.pop("created_utc", None)
        if existing != requested:
            raise FileExistsError(f"existing receipt differs; preserving it for inspection: {path}")
        return
    path.write_text(content, encoding="utf-8")


def read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["frame_index", "timestamp", "action", "observation.arm_joints", "observation.hand_joints", "observation.leg_joints"],
    )
    frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)

    def matrix(name: str, width: int) -> np.ndarray:
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if values.shape != (len(frame), width) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} shape/finite failure {values.shape}")
        return values

    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"{path}: frame_index is not zero-based contiguous")
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"{path}: timestamp is nonfinite or nonmonotonic")
    return (
        frame,
        timestamp,
        matrix("action", ACTION_DIM),
        matrix("observation.arm_joints", 14),
        matrix("observation.hand_joints", 14),
        matrix("observation.leg_joints", 15),
    )




class Q0ShardedGrid:
    """Read accepted Q0 16x4 frozen-feature shards by canonical row index."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        if manifest.get("status") != "PASS" or int(manifest.get("final_test_access", -1)) != 0:
            raise RuntimeError("accepted Q0 feature cache manifest is not PASS/final0")
        self.manifest = manifest
        self.shape = tuple(int(x) for x in manifest["logical_shape"])
        self.shards = sorted(manifest["shards"], key=lambda row: int(row["start_row"]))
        self.starts = [int(row["start_row"]) for row in self.shards]
        self.arrays: dict[int, np.ndarray] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        cursor = 0
        for shard in self.shards:
            start, end = int(shard["start_row"]), int(shard["end_row"])
            if start != cursor or end <= start:
                raise RuntimeError(f"Q0 feature shards have a gap/overlap at {start}:{end}")
            cursor = end
        if cursor != self.shape[0] or self.shape[1:] != (2, 16, 4, 768):
            raise RuntimeError(f"Q0 feature logical shape/coverage mismatch: {self.shape}, covered={cursor}")

    def row(self, index: int, anchor_id: str, episode_id: str) -> np.ndarray:
        shard_index = bisect.bisect_right(self.starts, int(index)) - 1
        if shard_index < 0:
            raise IndexError(f"Q0 feature row has no shard: {index}")
        shard = self.shards[shard_index]
        start, end = int(shard["start_row"]), int(shard["end_row"])
        if index >= end:
            raise IndexError(f"Q0 feature row exceeds shard: {index} not in {start}:{end}")
        if shard_index not in self.arrays:
            array_path = Path(shard["array"])
            metadata_path = Path(shard["meta"])
            if not array_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"accepted Q0 shard missing: {array_path}")
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (end-start, 2, 16, 4, 768)
            if values.shape != expected_shape or values.dtype != np.float32:
                raise RuntimeError(f"Q0 shard shape/dtype mismatch {array_path}: {values.shape}/{values.dtype}")
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if meta.get("status") != "PASS" or int(meta.get("final_test_access", -1)) != 0:
                raise RuntimeError(f"Q0 shard metadata not accepted: {metadata_path}")
            if int(meta.get("start_row", -1)) != start or int(meta.get("end_row", -1)) != end:
                raise RuntimeError(f"Q0 shard metadata range mismatch: {metadata_path}")
            self.arrays[shard_index] = values
            self.metadata[shard_index] = meta
        local = int(index) - start
        meta = self.metadata[shard_index]
        if str(meta["anchor_ids"][local]) != str(anchor_id) or int(meta["episode_index"][local]) != int(episode_id):
            raise RuntimeError(f"Q0 feature row identity mismatch at {index}: {anchor_id}/{episode_id}")
        return self.arrays[shard_index][local]


def main() -> None:
    started = time.time()
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    task_lock = json.loads((ROOT / "protocol/TASKS_V2.json").read_text(encoding="utf-8"))
    if task_lock.get("status") != "locked_before_v2_dev_support_read":
        raise RuntimeError("the fixed task lock must exist before reading per-task dev support")
    tasks = task_lock["tasks"]
    task_names = {int(row["task_id"]): row["original_task_name"] for row in tasks}
    ranks = {int(row["task_id"]): int(row["rank"]) - 1 for row in tasks}
    v3 = Path(resolved["v3_root"])
    raw_root = Path(resolved["raw_dataset_root"]).resolve()
    input_path = Path(resolved["main_inputs"])
    split_manifest = Path(resolved["split_manifest"])
    anchor_path = v3 / "ANCHOR_MANIFEST.csv"
    allow_path = v3 / "DATA_ALLOWLIST.csv"
    feature_manifest_path = Path(resolved["feature_cache"])

    allowed = [row for row in read_csv(allow_path) if int(row["original_task_index"]) in TASK_IDS]
    allowed_by_ep = {str(row["episode_index"]): row for row in allowed}
    if len(allowed_by_ep) != len(allowed):
        raise RuntimeError("selected-task allowlist contains duplicate episode_index values")
    if any(row.get("final_test_access") != "0" for row in allowed):
        raise RuntimeError("selected-task allowlist has a nonzero heldout access marker")

    anchors_all = [row for row in read_csv(anchor_path) if int(row["original_task_index"]) in TASK_IDS and row["population"] == "primary"]
    anchors_by_id = {str(row["anchor_id"]): row for row in anchors_all}
    if len(anchors_by_id) != len(anchors_all):
        raise RuntimeError("selected primary anchor manifest contains duplicate anchor IDs")
    train_anchors = [row for row in anchors_all if row["split"] == "train"]
    dev_anchors = [row for row in anchors_all if row["split"] == "dev"]

    # Split/task/episode/frame arrays are metadata. Future targets in V3_MAIN_INPUTS are never requested.
    with np.load(input_path, allow_pickle=False) as archive:
        split_code = archive["split_code"]
        anchor_ids = archive["anchor_id"].astype("U64", copy=True)
        episodes = archive["episode_index"].astype(np.int64, copy=True)
        task_ids = archive["original_task_index"].astype(np.int64, copy=True)
        anchor_frames = archive["anchor_frame"].astype(np.int64, copy=True)
        video_paths = archive["video_path"].astype("U512", copy=True)
        history_frame_indices = archive["history_selected_frame_indices"].astype(np.int64, copy=True)
        split_codes = sorted(int(x) for x in np.unique(split_code).tolist())
        if split_codes != [0, 1]:
            raise RuntimeError(f"V3 main archive role codes differ from accepted train/dev contract: {split_codes}")
        train_archive_indices = np.flatnonzero(split_code == 0).astype(np.int64)
        dev_archive_indices = np.flatnonzero(split_code == 1).astype(np.int64)
    input_row_by_id = {str(anchor_ids[i]): int(i) for i in range(len(anchor_ids))}
    if len(input_row_by_id) != len(anchor_ids):
        raise RuntimeError("V3 input anchor IDs are not unique")

    # Accepted parent identity evidence supplies the observable train source-group keys.
    parent_root = Path(resolved["parent_root"])
    parent_identity_rows = read_csv(parent_root / "inventory/CLUSTER_IDENTITY_EVIDENCE.csv")
    parent_identity = {
        str(row["episode_id"]): str(row["cluster_id"])
        for row in parent_identity_rows
        if int(row["task_id"]) in TASK_IDS
    }
    dataset_id = str(resolved["dataset_content_id"])

    train_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_episodes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    failed_episodes: list[dict[str, str]] = []
    raw_row_count = 0
    for anchor in train_anchors:
        anchor_id = str(anchor["anchor_id"])
        task_id = int(anchor["original_task_index"])
        episode_id = str(anchor["episode_index"])
        allow = allowed_by_ep.get(episode_id)
        if allow is None or allow["split"] != "train":
            raise RuntimeError(f"selected primary row missing canonical train role: {anchor_id}")
        if episode_id not in parent_identity:
            raise RuntimeError(f"selected train episode missing parent source identity evidence: {episode_id}")
        input_index = input_row_by_id.get(anchor_id)
        if input_index is None:
            raise RuntimeError(f"train anchor missing from accepted V3 input rows: {anchor_id}")
        if (int(episodes[input_index]), int(task_ids[input_index]), int(anchor_frames[input_index]), str(video_paths[input_index])) != (
            int(episode_id), task_id, int(anchor["anchor_frame"]), str(anchor["video_path"])
        ):
            raise RuntimeError(f"anchor identity/input row mismatch: {anchor_id}")
        frame_hist = history_frame_indices[input_index]
        a = int(anchor["anchor_frame"])
        if frame_hist.shape != (HISTORY_STEPS,) or int(frame_hist.max()) > a or int(frame_hist.min()) < a - 119:
            raise RuntimeError(f"history frame leakage/bounds failure: {anchor_id}")
        if not np.all(np.diff(frame_hist) >= 0):
            raise RuntimeError(f"history frame order failure: {anchor_id}")
        if int(anchor["target_1s_start_frame"]) != a + 1 or int(anchor["target_1s_end_frame"]) != a + HORIZON:
            raise RuntimeError(f"H30 target metadata boundary failure: {anchor_id}")

        rel = Path(str(anchor["parquet_path"]))
        parquet_path = (raw_root / rel).resolve()
        if not parquet_path.is_relative_to(raw_root) or not parquet_path.is_file():
            raise FileNotFoundError(f"selected train raw parquet missing/outside accepted root: {rel}")
        if episode_id not in raw_episodes:
            try:
                raw_episodes[episode_id] = read_episode(parquet_path)
            except Exception as exc:
                failed_episodes.append({"episode_id": episode_id, "task_id": str(task_id), "error": f"{type(exc).__name__}: {exc}"})
                continue
        frame, timestamp, action, arm, hand, leg = raw_episodes[episode_id]
        if a >= len(frame) or not np.array_equal(frame[a], np.asarray(a)):
            raise RuntimeError(f"anchor frame not present in source record: {anchor_id}")
        t0 = float(timestamp[a])
        query = t0 + np.arange(-31, 1, dtype=np.float64) * 0.125
        selected = np.searchsorted(timestamp, query, side="right") - 1
        if np.any(selected < a - 119) or np.any(selected > a) or np.any(timestamp[selected] > query + 1e-9):
            raise RuntimeError(f"history timestamp rule failed: {anchor_id}")
        if np.any(timestamp[a + 1:a + HORIZON + 1] <= t0):
            raise RuntimeError(f"future target timestamp rule failed: {anchor_id}")
        history_native = np.concatenate((arm[selected], hand[selected], leg[selected], action[selected]), axis=1).astype(np.float32)
        target_native = action[a + 1:a + HORIZON + 1].astype(np.float32)
        anchor_native = action[a].astype(np.float32)
        relative_delta = (query - t0).astype(np.float32)
        if history_native.shape != (HISTORY_STEPS, STATE_DIM) or target_native.shape != (HORIZON, ACTION_DIM):
            raise RuntimeError(f"selected train array shape mismatch: {anchor_id}")
        if not (np.isfinite(history_native).all() and np.isfinite(target_native).all() and np.isfinite(anchor_native).all()):
            raise RuntimeError(f"selected train raw data nonfinite: {anchor_id}")
        cache_index = input_index
        train_by_task[task_id].append({
            "anchor_id": anchor_id, "episode_id": episode_id, "task_id": task_id,
            "cluster_id": parent_identity[episode_id], "cache_index": cache_index, "anchor_frame": a,
            "parquet_path": str(rel), "video_path": str(anchor["video_path"]),
            "history_native": history_native, "history_mask": np.ones(HISTORY_STEPS, dtype=np.float32),
            "relative_delta": relative_delta, "anchor_native": anchor_native, "target_native_h30": target_native,
            "history_selected_frame_indices": selected.astype(np.int64),
        })
        raw_row_count += 1

    # Use the N2-accepted Q0 four-bin grid cache, not the V3 global spatial-mean cache.
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    feature_grid = Q0ShardedGrid(feature_manifest)
    feature_shape = feature_grid.shape
    feature_exclusions: list[dict[str, str]] = []
    for task_id, rows in train_by_task.items():
        good = []
        for row in rows:
            idx = int(row["cache_index"])
            if idx < 0 or idx >= feature_shape[0]:
                feature_exclusions.append({"anchor_id": row["anchor_id"], "task_id": str(task_id), "reason": "cache_index_out_of_range"})
                continue
            try:
                values = feature_grid.row(idx, row["anchor_id"], row["episode_id"])
            except Exception as exc:
                feature_exclusions.append({"anchor_id": row["anchor_id"], "task_id": str(task_id), "reason": f"q0_grid_row_identity_or_load_failure:{type(exc).__name__}"})
                continue
            if values.shape != (2, 16, 4, 768) or not np.isfinite(values).all():
                feature_exclusions.append({"anchor_id": row["anchor_id"], "task_id": str(task_id), "reason": "feature_shape_or_finite_failure"})
                continue
            good.append(row)
        train_by_task[task_id] = good

    # Collapse known exact source-key repeats into one group. Parent evidence is episode/source level;
    # session and alias links remain unverified and are not inferred from episode numbering.
    cluster_to_tasks: dict[str, set[int]] = defaultdict(set)
    for task_id, rows in train_by_task.items():
        for row in rows:
            cluster_to_tasks[str(row["cluster_id"])].add(int(task_id))
    cross_task_clusters = {key: sorted(value) for key, value in cluster_to_tasks.items() if len(value) > 1}
    if cross_task_clusters:
        raise RuntimeError(f"source clusters span selected task IDs; resolve a shared fold before training: {cross_task_clusters}")

    assignments: dict[str, str] = {}
    inner_rows: list[dict[str, Any]] = []
    task_split_counts: dict[int, dict[str, int]] = {}
    for task_id in TASK_IDS:
        rows = train_by_task.get(task_id, [])
        by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_cluster[str(row["cluster_id"])].append(row)
        keys = sorted(by_cluster)
        ordered = sorted(keys, key=lambda key: hashlib.sha256(f"v2split20261013|{key}".encode("utf-8")).hexdigest())
        tune_n = math.ceil(0.2 * len(ordered)) if len(ordered) >= 2 else (1 if len(ordered) == 1 else 0)
        tune_clusters = set(ordered[:tune_n])
        for cluster_id in ordered:
            role = "inner_tune" if cluster_id in tune_clusters else "inner_fit"
            assignments[cluster_id] = role
            group_rows = by_cluster[cluster_id]
            first = group_rows[0]
            inner_rows.append({
                "task_id": task_id, "task_name_raw": task_names[task_id], "cluster_id": cluster_id,
                "episode_ids": ";".join(sorted({r["episode_id"] for r in group_rows})),
                "anchors": len(group_rows), "assignment_hash": hashlib.sha256(f"v2split20261013|{cluster_id}".encode("utf-8")).hexdigest(),
                "role": role, "source_record_id": first["parquet_path"],
            })
        fit_eps = {r["episode_id"] for r in rows if assignments.get(str(r["cluster_id"])) == "inner_fit"}
        tune_eps = {r["episode_id"] for r in rows if assignments.get(str(r["cluster_id"])) == "inner_tune"}
        task_split_counts[task_id] = {
            "train_episodes": len({r["episode_id"] for r in rows}),
            "train_clusters": len(keys),
            "train_anchors": len(rows),
            "fit_episodes": len(fit_eps),
            "fit_clusters": sum(1 for k in keys if assignments[k] == "inner_fit"),
            "fit_anchors": sum(1 for r in rows if assignments.get(str(r["cluster_id"])) == "inner_fit"),
            "tune_episodes": len(tune_eps),
            "tune_clusters": sum(1 for k in keys if assignments[k] == "inner_tune"),
            "tune_anchors": sum(1 for r in rows if assignments.get(str(r["cluster_id"])) == "inner_tune"),
        }

    # Exposed-dev support is metadata only, now permitted after TASKS_V2 lock. No future labels are read.
    train_identity = [row for row in allowed if row["split"] == "train"]
    dev_identity = [row for row in allowed if row["split"] == "dev"]
    train_keys = {str(r["split_assignment_key"]) for r in train_identity}
    train_ep_to_paths = {
        str(r["episode_index"]): {str(a["parquet_path"]), str(a["video_path"])}
        for r in train_identity
        for a in train_anchors
        if str(a["episode_index"]) == str(r["episode_index"])
    }
    train_records = {x for values in train_ep_to_paths.values() for x in values}
    duplicate_exclusions: list[dict[str, Any]] = []
    valid_dev_eps: dict[int, set[str]] = defaultdict(set)
    valid_dev_clusters: dict[int, set[str]] = defaultdict(set)
    valid_dev_anchor_count: CounterLike = CounterLike()
    for row in dev_anchors:
        ep = str(row["episode_index"]); tid = int(row["original_task_index"])
        allow = allowed_by_ep.get(ep)
        if allow is None or allow["split"] != "dev":
            continue
        key = str(allow["split_assignment_key"])
        paths = {str(row["parquet_path"]), str(row["video_path"])}
        overlap_reason = []
        if key in train_keys: overlap_reason.append("split_assignment_key_crosses_train_dev")
        if paths & train_records: overlap_reason.append("same_source_path_crosses_train_dev")
        if overlap_reason:
            duplicate_exclusions.append({"task_id": tid, "episode_id": ep, "anchor_id": row["anchor_id"], "reason": ";".join(overlap_reason)})
            continue
        valid_dev_eps[tid].add(ep)
        valid_dev_clusters[tid].add(key)
        valid_dev_anchor_count[tid] = valid_dev_anchor_count.get(tid, 0) + 1

    task_rows = []
    dev_rows = []
    train_array_rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        train_rows = train_by_task.get(task_id, [])
        roles = [assignments[str(r["cluster_id"])] for r in train_rows]
        c = task_split_counts[task_id]
        task_rows.append({
            "rank": ranks[task_id], "task_id": task_id, "task_name_raw": task_names[task_id],
            **c,
            "dev_episodes_after_overlap_exclusion": len(valid_dev_eps[task_id]),
            "dev_clusters_after_overlap_exclusion": len(valid_dev_clusters[task_id]),
            "dev_h30_anchors_after_overlap_exclusion": valid_dev_anchor_count.get(task_id, 0),
            "dev_status": "available" if valid_dev_eps[task_id] else "NA",
        })
        train_array_rows.extend(train_rows)

    if len({r["anchor_id"] for r in train_array_rows}) != len(train_array_rows):
        raise RuntimeError("selected train anchor IDs are duplicated after group join")

    n = len(train_array_rows)
    if n:
        arrays = {
            "anchor_id": np.asarray([r["anchor_id"] for r in train_array_rows], dtype="U64"),
            "episode_id": np.asarray([int(r["episode_id"]) for r in train_array_rows], dtype=np.int64),
            "task_id": np.asarray([int(r["task_id"]) for r in train_array_rows], dtype=np.int64),
            "task_rank": np.asarray([ranks[int(r["task_id"])] for r in train_array_rows], dtype=np.int8),
            "source_cluster": np.asarray([r["cluster_id"] for r in train_array_rows], dtype="U128"),
            "cache_index": np.asarray([int(r["cache_index"]) for r in train_array_rows], dtype=np.int64),
            "anchor_frame": np.asarray([int(r["anchor_frame"]) for r in train_array_rows], dtype=np.int64),
            "history_native": np.stack([r["history_native"] for r in train_array_rows]).astype(np.float32),
            "history_mask": np.stack([r["history_mask"] for r in train_array_rows]).astype(np.float32),
            "relative_delta": np.stack([r["relative_delta"] for r in train_array_rows]).astype(np.float32),
            "anchor_native": np.stack([r["anchor_native"] for r in train_array_rows]).astype(np.float32),
            "target_native_h30": np.stack([r["target_native_h30"] for r in train_array_rows]).astype(np.float32),
            "history_selected_frame_indices": np.stack([r["history_selected_frame_indices"] for r in train_array_rows]).astype(np.int64),
            "inner_role": np.asarray([assignments[str(r["cluster_id"])] for r in train_array_rows], dtype="U16"),
        }
        for key, value in arrays.items():
            if key not in {"anchor_id", "source_cluster", "inner_role"} and not np.isfinite(value).all():
                raise RuntimeError(f"prepared train array is nonfinite: {key}")
        data_path = ROOT / "data/TRAIN_SELECTED_H30.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        if data_path.exists():
            # Preserve a successful data artifact from an earlier failed receipt-writing attempt.
            with np.load(data_path, allow_pickle=False) as existing:
                if set(existing.files) != set(arrays):
                    raise RuntimeError("existing train subset NPZ has a different schema; preserve it and inspect manually")
                for key, value in arrays.items():
                    if not np.array_equal(existing[key], value):
                        raise RuntimeError(f"existing train subset NPZ differs on retry: {key}; preserve both attempts")
        else:
            np.savez_compressed(data_path, **arrays)
        data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    else:
        data_sha = ""

    write_csv(ROOT / "protocol/INNER_SPLIT.csv", [
        "task_id", "task_name_raw", "cluster_id", "episode_ids", "anchors", "assignment_hash", "role", "source_record_id",
    ], inner_rows)
    write_json(ROOT / "protocol/INNER_SPLIT_LOCK.json", {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "LOCKED", "algorithm": "sort sha256('v2split20261013|' + canonical_source_key); first ceil(0.2*n_clusters) per task is inner_tune",
        "dataset_content_id": dataset_id, "source_key": "accepted split_assignment_key; parent evidence is episode/source level",
        "task_cluster_counts": {str(k): v for k, v in task_split_counts.items()},
        "all_train_tune_groups_disjoint": True, "cross_task_groups": cross_task_clusters,
        "session_or_trial_ids": "UNKNOWN", "collection_independence": "unverified_episode_source_only",
        "dev_outcomes_used": False,
    })
    write_csv(ROOT / "inventory/TRAIN_TASKS_V2.csv", [
        "rank", "task_id", "task_name_raw", "train_episodes", "train_clusters", "train_anchors",
        "fit_episodes", "fit_clusters", "fit_anchors", "tune_episodes", "tune_clusters", "tune_anchors",
        "dev_episodes_after_overlap_exclusion", "dev_clusters_after_overlap_exclusion", "dev_h30_anchors_after_overlap_exclusion", "dev_status",
    ], task_rows)
    write_csv(ROOT / "inventory/DEV_SUPPORT_V2.csv", [
        "rank", "task_id", "task_name_raw", "dev_episodes_after_overlap_exclusion", "dev_clusters_after_overlap_exclusion",
        "dev_h30_anchors_after_overlap_exclusion", "overlap_excluded_anchors", "dev_status",
    ], [{
        "rank": ranks[int(row["task_id"])]+1, "task_id": row["task_id"], "task_name_raw": row["task_name_raw"],
        "dev_episodes_after_overlap_exclusion": row["dev_episodes_after_overlap_exclusion"],
        "dev_clusters_after_overlap_exclusion": row["dev_clusters_after_overlap_exclusion"],
        "dev_h30_anchors_after_overlap_exclusion": row["dev_h30_anchors_after_overlap_exclusion"],
        "overlap_excluded_anchors": sum(1 for x in duplicate_exclusions if int(x["task_id"]) == int(row["task_id"])),
        "dev_status": row["dev_status"],
    } for row in task_rows])
    write_csv(ROOT / "inventory/DEV_DUPLICATE_EXCLUSIONS.csv", ["task_id", "episode_id", "anchor_id", "reason"], duplicate_exclusions)
    write_csv(ROOT / "inventory/FEATURE_EXCLUSIONS.csv", ["anchor_id", "task_id", "reason"], feature_exclusions)
    write_csv(ROOT / "inventory/RAW_EPISODE_READ_FAILURES.csv", ["episode_id", "task_id", "error"], failed_episodes)

    qa = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if n > 0 and not failed_episodes and not feature_exclusions else "PASS_WITH_RECORDED_EXCLUSIONS",
        "fixed_task_ids": list(TASK_IDS), "train_h30_rows_actual": n,
        "train_episode_counts": {str(k): v["train_episodes"] for k, v in task_split_counts.items()},
        "fit_tune_counts": {str(k): {kk: v[kk] for kk in ("fit_episodes", "fit_clusters", "fit_anchors", "tune_episodes", "tune_clusters", "tune_anchors")} for k, v in task_split_counts.items()},
        "feature_cache_shape": list(feature_shape), "feature_cache_manifest_sha256": hashlib.sha256(feature_manifest_path.read_bytes()).hexdigest(), "feature_cache_finite_selected_train_rows": n-len(feature_exclusions),
        "state_shape": [n, HISTORY_STEPS, STATE_DIM], "target_shape": [n, HORIZON, ACTION_DIM],
        "train_state_target_finite": n > 0, "h30_start_is_anchor_plus_one": True,
        "h30_last_is_anchor_plus_30": True, "history_frame_max_le_anchor": True,
        "future_timestamps_after_anchor": True, "raw_target_source": "selected train-role Parquet records only",
        "v3_mixed_archive_target_key_read": False, "dev_target_or_outcome_read": False,
        "dev_support_metadata_read_after_task_lock": True, "dev_rows_excluded_for_known_overlap": len(duplicate_exclusions),
        "failed_train_episodes": len(failed_episodes), "feature_excluded_anchors": len(feature_exclusions),
        "train_data_npz_sha256": data_sha, "final_heldout_payload_access": 0,
    }
    write_json(ROOT / "qa/INPUT_AND_CACHE_QA.json", qa)
    write_json(ROOT / "inventory/OUTCOME_EXPOSURE.json", {
        "schema_version": 1, "created_utc": qa["created_utc"],
        "parent_aggregate_metadata_exposure_preserved": True,
        "parent_exposure_record": str(parent_root / "inventory/OUTCOME_EXPOSURE.json"),
        "fixed_task_lock_sha256": hashlib.sha256((ROOT / "protocol/TASKS_V2.json").read_bytes()).hexdigest(),
        "dev_support_metadata_read_after_lock": True,
        "dev_task_ids_read": list(TASK_IDS), "dev_task_support_rows_written": len(task_rows),
        "dev_target_labels_read": False, "dev_predictions_or_errors_read": False,
        "train_target_access": "selected fixed-task train Parquet only; H30 labels enter future train loss/scorer, not forward",
        "v3_mixed_role_target_arrays_read": False,
        "new_final_aggregate_counts_read_or_printed": False,
        "final_heldout_payload_access": 0,
        "collection_independence": "unverified_episode_source_only",
        "cache_lineage": "accepted_cache_with_partial_historical_binding",
    })
    print(json.dumps({
        "status": qa["status"], "train_rows": n, "train_episode_counts": qa["train_episode_counts"],
        "fit_tune_counts": qa["fit_tune_counts"], "dev_support": {str(r["task_id"]): {"episodes": r["dev_episodes_after_overlap_exclusion"], "anchors": r["dev_h30_anchors_after_overlap_exclusion"]} for r in task_rows},
        "known_train_dev_overlap_excluded": len(duplicate_exclusions), "feature_exclusions": len(feature_exclusions),
        "raw_episode_read_failures": len(failed_episodes), "run_root": str(ROOT),
    }, ensure_ascii=False, indent=2))


class CounterLike(dict):
    def get(self, key: Any, default: int = 0) -> int:
        return super().get(key, default)



if __name__ == "__main__":
    main()
