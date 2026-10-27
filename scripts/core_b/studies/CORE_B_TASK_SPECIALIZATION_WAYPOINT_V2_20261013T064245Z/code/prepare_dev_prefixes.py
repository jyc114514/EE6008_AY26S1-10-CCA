#!/usr/bin/env python3
"""Prepare only legal dev prefixes; future action rows are not returned by the reader."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

from prepare_v2_data import Q0ShardedGrid, read_csv

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
HISTORY_STEPS = 32
STATE_DIM = 71
ACTION_DIM = 28


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_npz_idempotent(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise RuntimeError("existing dev-prefix NPZ schema differs; preserving existing artifact")
            for name, value in arrays.items():
                if not np.array_equal(old[name], value):
                    raise RuntimeError(f"existing dev-prefix NPZ differs for {name}; preserving existing artifact")
    else:
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.part.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_matrix(column, name: str, width: int) -> np.ndarray:
    value = np.asarray(column.to_pylist(), dtype=np.float64)
    if value.shape != (len(column), width) or not np.isfinite(value).all():
        raise ValueError(f"dev prefix column {name} shape/finite failure: {value.shape}")
    return value


def main() -> None:
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    tasks_lock = json.loads((ROOT / "protocol/TASKS_V2.json").read_text(encoding="utf-8"))
    if tasks_lock.get("status") != "locked_before_v2_dev_support_read":
        raise RuntimeError("fixed task lock must precede dev prefix extraction")
    v3 = Path(resolved["v3_root"])
    raw_root = Path(resolved["raw_dataset_root"]).resolve()
    allowed = [row for row in read_csv(v3 / "DATA_ALLOWLIST.csv") if int(row["original_task_index"]) in TASK_IDS]
    allowed_dev = {str(row["episode_index"]): row for row in allowed if row.get("split") == "dev"}
    anchors = [
        row for row in read_csv(v3 / "ANCHOR_MANIFEST.csv")
        if int(row["original_task_index"]) in TASK_IDS and row.get("population") == "primary" and row.get("split") == "dev"
    ]
    exclusions = {str(row["anchor_id"]) for row in read_csv(ROOT / "inventory/DEV_DUPLICATE_EXCLUSIONS.csv")}
    anchors = [row for row in anchors if str(row["anchor_id"]) not in exclusions]

    inputs_path = Path(resolved["main_inputs"])
    with np.load(inputs_path, allow_pickle=False) as archive:
        # Metadata-only keys; never request a target array from the mixed-role archive.
        split_code = archive["split_code"]
        anchor_ids = archive["anchor_id"].astype("U64", copy=True)
        episode_ids = archive["episode_index"].astype(np.int64, copy=True)
        task_ids = archive["original_task_index"].astype(np.int64, copy=True)
        anchor_frames = archive["anchor_frame"].astype(np.int64, copy=True)
        video_paths = archive["video_path"].astype("U512", copy=True)
        history_frames = archive["history_selected_frame_indices"].astype(np.int64, copy=True)
    if sorted(int(x) for x in np.unique(split_code)) != [0, 1]:
        raise RuntimeError("accepted input split-code contract changed")
    row_by_anchor = {str(anchor_ids[i]): i for i in range(len(anchor_ids))}
    if len(row_by_anchor) != len(anchor_ids):
        raise RuntimeError("mixed-role input anchor IDs are not unique")

    rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_rank = {int(row["task_id"]): int(row["rank"]) - 1 for row in tasks_lock["tasks"]}
    for row in anchors:
        anchor_id = str(row["anchor_id"])
        episode_id = str(row["episode_index"])
        task_id = int(row["original_task_index"])
        allowed_row = allowed_dev.get(episode_id)
        if allowed_row is None:
            raise RuntimeError(f"dev anchor lacks accepted dev-role allowlist identity: {anchor_id}")
        input_idx = row_by_anchor.get(anchor_id)
        if input_idx is None or int(split_code[input_idx]) != 1:
            raise RuntimeError(f"dev anchor missing accepted dev prefix metadata row: {anchor_id}")
        if (int(episode_ids[input_idx]), int(task_ids[input_idx]), int(anchor_frames[input_idx]), str(video_paths[input_idx])) != (
            int(episode_id), task_id, int(row["anchor_frame"]), str(row["video_path"])
        ):
            raise RuntimeError(f"dev anchor identity mismatch: {anchor_id}")
        hist = history_frames[input_idx]
        anchor_frame = int(row["anchor_frame"])
        if hist.shape != (HISTORY_STEPS,) or int(hist.min()) < anchor_frame - 119 or int(hist.max()) > anchor_frame:
            raise RuntimeError(f"dev prefix frame bound failure: {anchor_id}")
        parquet_path = (raw_root / Path(str(row["parquet_path"]))).resolve()
        if not parquet_path.is_relative_to(raw_root) or not parquet_path.is_file():
            raise FileNotFoundError(f"dev source parquet missing/outside accepted raw root: {row['parquet_path']}")
        rows_by_episode[episode_id].append({
            "anchor_id": anchor_id, "episode_id": episode_id, "task_id": task_id,
            "source_cluster": str(allowed_row["split_assignment_key"]),
            "cache_index": int(input_idx), "anchor_frame": anchor_frame,
            "parquet_path": str(row["parquet_path"]), "history_frames": hist.copy(),
        })

    resolved_cache = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    manifest_path = Path(resolved_cache["feature_cache"])
    grid = Q0ShardedGrid(json.loads(manifest_path.read_text(encoding="utf-8")))
    result: list[dict[str, Any]] = []
    task_episode_counts: dict[str, set[str]] = {str(t): set() for t in TASK_IDS}
    task_anchor_counts = {str(t): 0 for t in TASK_IDS}
    columns = [
        "frame_index", "timestamp", "action", "observation.arm_joints",
        "observation.hand_joints", "observation.leg_joints",
    ]
    for episode_id, episode_rows in sorted(rows_by_episode.items(), key=lambda item: int(item[0])):
        requested_frames: set[int] = set()
        for row in episode_rows:
            requested_frames.update(int(x) for x in row["history_frames"].tolist())
            requested_frames.add(int(row["anchor_frame"]))
        sorted_requested = sorted(requested_frames)
        parquet_path = (raw_root / episode_rows[0]["parquet_path"]).resolve()
        # Arrow returns only explicit prefix frame IDs; no future target frames are selected.
        table = ds.dataset(str(parquet_path), format="parquet").to_table(
            columns=columns,
            filter=ds.field("frame_index").isin(sorted_requested),
        )
        frame = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        timestamp = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
        if len(frame) != len(sorted_requested) or set(frame.tolist()) != set(sorted_requested):
            raise RuntimeError(f"prefix-only row selection mismatch for episode {episode_id}")
        if not np.isfinite(timestamp).all():
            raise RuntimeError(f"nonfinite dev prefix timestamps in episode {episode_id}")
        order = np.argsort(frame)
        frame = frame[order]; timestamp = timestamp[order]
        action = parse_matrix(table.take(order)["action"], "action", ACTION_DIM)
        arm = parse_matrix(table.take(order)["observation.arm_joints"], "observation.arm_joints", 14)
        hand = parse_matrix(table.take(order)["observation.hand_joints"], "observation.hand_joints", 14)
        leg = parse_matrix(table.take(order)["observation.leg_joints"], "observation.leg_joints", 15)
        offset = {int(value): i for i, value in enumerate(frame.tolist())}
        for row in episode_rows:
            a = int(row["anchor_frame"])
            hist_idx = np.asarray(row["history_frames"], dtype=np.int64)
            positions = np.asarray([offset[int(x)] for x in hist_idx], dtype=np.int64)
            anchor_pos = offset[a]
            t0 = float(timestamp[anchor_pos])
            query = t0 + np.arange(-31, 1, dtype=np.float64) * 0.125
            if np.any(timestamp[positions] > query + 1e-9) or np.any(hist_idx > a):
                raise RuntimeError(f"dev prefix timestamp causality failure: {row['anchor_id']}")
            history = np.concatenate((arm[positions], hand[positions], leg[positions], action[positions]), axis=1).astype(np.float32)
            anchor = action[anchor_pos].astype(np.float32)
            relative = (query - t0).astype(np.float32)
            if history.shape != (HISTORY_STEPS, STATE_DIM) or not np.isfinite(history).all() or not np.isfinite(anchor).all():
                raise RuntimeError(f"dev prefix state/anchor shape or finite failure: {row['anchor_id']}")
            feature = grid.row(row["cache_index"], row["anchor_id"], episode_id)
            if feature.shape != (2, 16, 4, 768) or not np.isfinite(feature).all():
                raise RuntimeError(f"dev Q0 feature row shape/finite failure: {row['anchor_id']}")
            result.append({
                **row, "history_native": history, "history_mask": np.ones(HISTORY_STEPS, dtype=np.float32),
                "relative_delta": relative, "anchor_native": anchor,
                "history_selected_frame_indices": hist_idx.astype(np.int64),
            })
            task_episode_counts[str(row["task_id"])].add(episode_id)
            task_anchor_counts[str(row["task_id"])] += 1

    result.sort(key=lambda row: (TASK_IDS.index(int(row["task_id"])), int(row["episode_id"]), int(row["anchor_frame"]), row["anchor_id"]))
    n = len(result)
    arrays = {
        "anchor_id": np.asarray([row["anchor_id"] for row in result], dtype="U64"),
        "episode_id": np.asarray([int(row["episode_id"]) for row in result], dtype=np.int64),
        "task_id": np.asarray([int(row["task_id"]) for row in result], dtype=np.int64),
        "task_rank": np.asarray([task_rank[int(row["task_id"])] for row in result], dtype=np.int8),
        "source_cluster": np.asarray([row["source_cluster"] for row in result], dtype="U256"),
        "cache_index": np.asarray([row["cache_index"] for row in result], dtype=np.int64),
        "anchor_frame": np.asarray([row["anchor_frame"] for row in result], dtype=np.int64),
        "parquet_path": np.asarray([row["parquet_path"] for row in result], dtype="U512"),
        "history_native": np.stack([row["history_native"] for row in result]).astype(np.float32),
        "history_mask": np.stack([row["history_mask"] for row in result]).astype(np.float32),
        "relative_delta": np.stack([row["relative_delta"] for row in result]).astype(np.float32),
        "anchor_native": np.stack([row["anchor_native"] for row in result]).astype(np.float32),
        "history_selected_frame_indices": np.stack([row["history_selected_frame_indices"] for row in result]).astype(np.int64),
    }
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key not in {"anchor_id", "source_cluster", "parquet_path"}):
        raise RuntimeError("dev prefix NPZ contains nonfinite numeric values")
    data_path = ROOT / "data/DEV_PREFIXES_NO_LABELS.npz"
    data_sha = write_npz_idempotent(data_path, arrays)
    qa = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "dev_prefix_rows": n,
        "task_anchor_counts": task_anchor_counts,
        "task_episode_counts": {key: len(value) for key, value in task_episode_counts.items()},
        "input_archive_keys_requested": ["split_code", "anchor_id", "episode_index", "original_task_index", "anchor_frame", "video_path", "history_selected_frame_indices"],
        "future_target_columns_requested_from_input_archive": [],
        "parquet_filter": "frame_index in union(history_selected_frame_indices, anchor_frame); maximum selected frame per window <= anchor",
        "future_target_frames_read_or_returned": 0,
        "q0_feature_shape": [2, 16, 4, 768],
        "feature_cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dev_duplicate_exclusions_applied_before_feature_or_outcome_reads": len(exclusions),
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "qa/DEV_PREFIX_QA.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
