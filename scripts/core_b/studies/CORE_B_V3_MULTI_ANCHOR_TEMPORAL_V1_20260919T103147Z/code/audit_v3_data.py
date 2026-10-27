#!/usr/bin/env python3
"""CPU-only V3 anchor, split, task-support and storage audit."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
HISTORY_INPUT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/inputs/HISTORY_VIDEO_INPUTS.npz"
SOURCE_FPS = 30.0
ANCHOR_START = 119
PRIMARY_TRAIN_CAP = 8
PRIMARY_DEV_CAP = 4
SUPP_DEV_CAP = 4


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def equal_spaced(values: list[int], cap: int) -> list[int]:
    if not values or cap <= 0: return []
    n = min(cap, len(values))
    if n == 1: return [values[0]]
    indices = [int(math.floor(j * (len(values) - 1) / (n - 1))) for j in range(n)]
    chosen = [values[index] for index in indices]
    if len(set(chosen)) != len(chosen): raise RuntimeError("equal-spaced anchor duplicate")
    return chosen


def read_timeline(path: Path, episode: int) -> tuple[int, np.ndarray]:
    table = pq.read_table(path, columns=["frame_index", "timestamp"])
    frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"episode {episode}: frame index is not contiguous")
    if not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"episode {episode}: timestamp is not finite/increasing")
    median_step = float(np.median(np.diff(timestamp))) if len(timestamp) > 1 else np.nan
    if abs(median_step - 1.0 / SOURCE_FPS) > 2e-4:
        raise ValueError(f"episode {episode}: median timestamp step {median_step} is not 30Hz")
    return len(frame), timestamp


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    with np.load(HISTORY_INPUT, allow_pickle=False) as archive:
        canonical_ids = archive["episode_index"].astype(np.int64)
        split_code = archive["split_code"].astype(np.int8)
        tasks_from_input = archive["original_task_index"].astype(np.int64)
    if len(canonical_ids) != 3355 or len(np.unique(canonical_ids)) != 3355 or int(np.sum(split_code == 0)) != 2741 or int(np.sum(split_code == 1)) != 614:
        raise RuntimeError("canonical allowlist contract failed")
    split = pd.read_parquet(SPLIT)
    primary = split[split["v3_benchmark_inclusion"].eq("primary") & split["split"].isin(["train", "validation"])].copy()
    by_id = {int(row.episode_index): row for row in primary.itertuples(index=False)}
    if not set(canonical_ids.tolist()).issubset(by_id): raise RuntimeError("canonical IDs absent from primary manifest")
    split_values = {int(row.episode_index): str(row.split) for row in primary.itertuples(index=False)}
    if any((split_code[i] == 0 and split_values[int(e)] != "train") or (split_code[i] == 1 and split_values[int(e)] != "validation") for i, e in enumerate(canonical_ids)):
        raise RuntimeError("canonical split inheritance failed")
    group_by_id = {int(row.episode_index): str(row.split_assignment_key) for row in primary.itertuples(index=False)}
    groups = {}
    for e in canonical_ids.tolist(): groups.setdefault(group_by_id[int(e)], set()).add(split_values[int(e)])
    cross_groups = sorted(group for group, values in groups.items() if len(values) > 1)
    if cross_groups: raise RuntimeError(f"duplicate group crosses split: {cross_groups[:5]}")

    anchors: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    task_stats: dict[tuple[str, int], dict[str, int]] = {}
    primary_by_episode: dict[int, list[int]] = {}
    supplementary_by_episode: dict[int, list[int]] = {}
    total_video_bytes = 0
    for index, episode_value in enumerate(canonical_ids.tolist()):
        episode = int(episode_value); row = by_id[episode]
        parquet_path = (RAW_ROOT / Path(str(row.parquet_path))).resolve()
        if not parquet_path.is_relative_to(RAW_ROOT.resolve()) or not parquet_path.is_file(): raise FileNotFoundError(parquet_path)
        n_frames, timestamps = read_timeline(parquet_path, episode)
        candidate_2s = list(range(ANCHOR_START, max(ANCHOR_START, n_frames - 60), 30))
        candidate_1s = list(range(ANCHOR_START, max(ANCHOR_START, n_frames - 30), 30))
        # The range stop above is exclusive: legal last anchor satisfies
        # a+60 <= n_frames-1 (or a+30 <= n_frames-1 for 1s).
        candidate_2s = [a for a in candidate_2s if a + 60 < n_frames]
        candidate_1s = [a for a in candidate_1s if a + 30 < n_frames]
        split_name = "train" if split_code[index] == 0 else "dev"
        primary_cap = PRIMARY_TRAIN_CAP if split_name == "train" else PRIMARY_DEV_CAP
        selected = equal_spaced(candidate_2s, primary_cap)
        primary_by_episode[episode] = selected
        if not selected:
            exclusions.append({"episode_index": episode, "split": split_name, "task": int(tasks_from_input[index]), "reason": "no_legal_2s_anchor", "frame_count": n_frames, "candidate_1s_count": len(candidate_1s), "candidate_2s_count": 0})
        for j, anchor_frame in enumerate(selected):
            anchor_id = f"e{episode:06d}_a{j:02d}_f{anchor_frame:06d}"
            anchors.append({"anchor_id": anchor_id, "episode_index": episode, "split": split_name, "population": "primary", "primary_index": j, "original_task_index": int(tasks_from_input[index]), "anchor_frame": anchor_frame, "anchor_timestamp": float(timestamps[anchor_frame]), "target_1s_start_frame": anchor_frame + 1, "target_1s_end_frame": anchor_frame + 30, "target_2s_start_frame": anchor_frame + 1, "target_2s_end_frame": anchor_frame + 60, "candidate_1s_count": len(candidate_1s), "candidate_2s_count": len(candidate_2s), "video_path": str(row.video_path), "parquet_path": str(row.parquet_path), "video_size_bytes": int(row.video_size_bytes), "final_test_access": 0})
        if split_name == "dev":
            remaining = [a for a in candidate_1s if a not in selected]
            need = max(0, SUPP_DEV_CAP - len(selected))
            supplement = equal_spaced(remaining, need)
            supplementary_by_episode[episode] = supplement
            for j, anchor_frame in enumerate(supplement, start=len(selected)):
                anchor_id = f"e{episode:06d}_a{j:02d}_f{anchor_frame:06d}"
                anchors.append({"anchor_id": anchor_id, "episode_index": episode, "split": split_name, "population": "supplementary_1s", "primary_index": j, "original_task_index": int(tasks_from_input[index]), "anchor_frame": anchor_frame, "anchor_timestamp": float(timestamps[anchor_frame]), "target_1s_start_frame": anchor_frame + 1, "target_1s_end_frame": anchor_frame + 30, "target_2s_start_frame": "NA", "target_2s_end_frame": "NA", "candidate_1s_count": len(candidate_1s), "candidate_2s_count": len(candidate_2s), "video_path": str(row.video_path), "parquet_path": str(row.parquet_path), "video_size_bytes": int(row.video_size_bytes), "final_test_access": 0})
        total_video_bytes += int(row.video_size_bytes)
        key = (split_name, int(tasks_from_input[index])); task_stats.setdefault(key, {"episodes": 0, "primary_anchors": 0, "supplementary_1s_anchors": 0, "no_2s": 0}); task_stats[key]["episodes"] += 1; task_stats[key]["primary_anchors"] += len(selected); task_stats[key]["supplementary_1s_anchors"] += len(supplementary_by_episode.get(episode, [])); task_stats[key]["no_2s"] += int(not selected)

    primary_train = [row for row in anchors if row["split"] == "train" and row["population"] == "primary"]
    primary_dev = [row for row in anchors if row["split"] == "dev" and row["population"] == "primary"]
    supp_dev = [row for row in anchors if row["split"] == "dev" and row["population"] == "supplementary_1s"]
    unique_primary_clips = len({(int(row["episode_index"]), int(row["anchor_frame"])) for row in primary_train + primary_dev})
    unique_all_clips = len({(int(row["episode_index"]), int(row["anchor_frame"])) for row in primary_train + primary_dev + supp_dev})
    token_elements = 16 * 768
    cache_bytes_per_clip = token_elements * 4 * 2  # D and S float32 temporal tokens; GLOBAL is derived.
    cache_estimate_bytes = unique_all_clips * cache_bytes_per_clip
    write_csv(EXP / "ANCHOR_MANIFEST.csv", anchors)
    write_csv(EXP / "ELIGIBILITY_EXCLUSIONS.csv", exclusions)
    write_csv(EXP / "DATA_ALLOWLIST.csv", [{"episode_index": int(e), "split": "train" if split_code[i] == 0 else "dev", "original_task_index": int(tasks_from_input[i]), "split_assignment_key": group_by_id[int(e)], "final_test_access": 0} for i, e in enumerate(canonical_ids.tolist())])
    task_rows = [{"split": split_name, "original_task_index": task, **values, "final_test_access": 0} for (split_name, task), values in sorted(task_stats.items())]
    write_csv(EXP / "TASK_COVERAGE.csv", task_rows)
    timeline_rows = []
    for row in anchors:
        ep = int(row["episode_index"]); r = by_id[ep]; parquet_path = (RAW_ROOT / Path(str(r.parquet_path))).resolve(); _, ts = read_timeline(parquet_path, ep); a = int(row["anchor_frame"]); timeline_rows.append({"anchor_id": row["anchor_id"], "episode_index": ep, "split": row["split"], "population": row["population"], "anchor_frame": a, "anchor_timestamp": float(ts[a]), "past_start_frame": a - 119, "past_end_frame": a, "past_start_timestamp": float(ts[a - 119]), "past_end_timestamp": float(ts[a]), "target_1s_start_timestamp": float(ts[a + 1]), "target_1s_end_timestamp": float(ts[a + 30]), "target_2s_start_timestamp": float(ts[a + 1]) if row["target_2s_start_frame"] != "NA" else "NA", "target_2s_end_timestamp": float(ts[a + 60]) if row["target_2s_end_frame"] != "NA" else "NA", "past_le_anchor": True, "target_1s_gt_anchor": bool(ts[a + 1] > ts[a]), "target_2s_gt_anchor": bool(row["target_2s_start_frame"] == "NA" or ts[a + 1] > ts[a]), "final_test_access": 0})
    write_csv(EXP / "OBSERVATION_TARGET_TIMELINE.csv", timeline_rows)
    summary = {"status": "PASS", "canonical_train_episodes": 2741, "canonical_dev_episodes": 614, "primary_train_episodes": len({int(r["episode_index"]) for r in primary_train}), "primary_train_windows": len(primary_train), "primary_dev_episodes": len({int(r["episode_index"]) for r in primary_dev}), "primary_dev_windows": len(primary_dev), "supplementary_1s_dev_episodes": len({int(r["episode_index"]) for r in supp_dev}), "supplementary_1s_dev_windows": len(supp_dev), "episodes_without_primary_2s": len(exclusions), "tasks_with_primary_train_support": len({int(r["original_task_index"]) for r in primary_train}), "tasks_with_primary_dev_support": len({int(r["original_task_index"]) for r in primary_dev}), "unique_primary_clips": unique_primary_clips, "unique_all_clips": unique_all_clips, "token_layout_assumed_from_accepted_cache": {"raw_tokens": [9216, 768], "temporal_tokens": [16, 768], "spatial_grid": [24, 24], "temporal_slots": 16}, "estimated_cache_bytes_D_and_S_float32": cache_estimate_bytes, "estimated_cache_gib_D_and_S_float32": cache_estimate_bytes / (1024**3), "source_video_bytes_unique_canonical": total_video_bytes, "source_video_gib": total_video_bytes / (1024**3), "final_test_access": 0}
    write_json(EXP / "DATA_AUDIT_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
