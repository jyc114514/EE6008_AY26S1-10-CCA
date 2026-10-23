#!/usr/bin/env python3
"""Build V3 primary/supplementary input arrays and train-only normalizers."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
ANCHORS = EXP / "ANCHOR_MANIFEST.csv"
EPS = 1e-6
HISTORY_STEPS = 32
STATE_DIM = 71
ACTION_DIM = 28


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["frame_index", "timestamp", "action", "observation.arm_joints", "observation.hand_joints", "observation.leg_joints"])
    frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    def matrix(name: str, width: int) -> np.ndarray:
        values = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if values.shape != (len(frame), width) or not np.isfinite(values).all(): raise ValueError(f"{path}: {name} shape/finite failure {values.shape}")
        return values
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)) or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0): raise ValueError(f"{path}: timeline failure")
    return frame, timestamp, matrix("action", 28), matrix("observation.arm_joints", 14), matrix("observation.hand_joints", 14), matrix("observation.leg_joints", 15)


def episode_equal_stats(values: np.ndarray, episode_ids: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = []; second = []
    for episode in np.unique(episode_ids[train_mask]):
        selected = values[train_mask & (episode_ids == episode)].reshape(-1, values.shape[-1]).astype(np.float64)
        means.append(selected.mean(axis=0)); second.append((selected * selected).mean(axis=0))
    mean = np.mean(np.asarray(means), axis=0)
    variance = np.maximum(np.mean(np.asarray(second), axis=0) - mean * mean, 0.0)
    return mean.astype(np.float32), np.where(np.sqrt(variance) <= EPS, 1.0, np.sqrt(variance)).astype(np.float32)


def episode_equal_absdiff(values: np.ndarray, episode_ids: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    means = []
    for episode in np.unique(episode_ids[train_mask]):
        selected = values[train_mask & (episode_ids == episode)]
        means.append(np.mean(np.abs(np.diff(selected, axis=1)), axis=(0, 1)))
    mean = np.mean(np.asarray(means), axis=0)
    return np.maximum(mean, EPS).astype(np.float32)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    with part.open("wb") as handle: np.savez_compressed(handle, **arrays)
    part.replace(path)


def main() -> int:
    rows = read_rows(ANCHORS)
    primary = [row for row in rows if row["population"] == "primary"]
    supplementary = [row for row in rows if row["population"] == "supplementary_1s"]
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    def build(selected_rows: list[dict[str, str]], horizon: int) -> dict[str, np.ndarray]:
        n = len(selected_rows)
        history_native = np.empty((n, HISTORY_STEPS, STATE_DIM), dtype=np.float32)
        history_query = np.empty((n, HISTORY_STEPS), dtype=np.float32)
        history_selected = np.empty((n, HISTORY_STEPS), dtype=np.float32)
        history_frames = np.empty((n, HISTORY_STEPS), dtype=np.int64)
        targets = np.empty((n, horizon, ACTION_DIM), dtype=np.float32)
        anchors_native = np.empty((n, ACTION_DIM), dtype=np.float32)
        episode_ids = np.empty(n, dtype=np.int64); tasks = np.empty(n, dtype=np.int64); split_codes = np.empty(n, dtype=np.int8); anchor_frames = np.empty(n, dtype=np.int64); anchor_ts = np.empty(n, dtype=np.float32)
        anchor_ids = np.empty(n, dtype="U64"); video_paths = np.empty(n, dtype="U160")
        for i, row in enumerate(selected_rows):
            episode = int(row["episode_index"])
            if episode not in cache:
                cache[episode] = read_episode((RAW_ROOT / Path(row["parquet_path"])).resolve())
            frame, timestamp, action, arm, hand, leg = cache[episode]
            a = int(row["anchor_frame"]); t0 = float(timestamp[a])
            queries = t0 + np.arange(-31, 1, dtype=np.float64) * 0.125
            selected = np.searchsorted(timestamp, queries, side="right") - 1
            if np.any(selected < a - 119) or np.any(selected > a) or np.any(timestamp[selected] > queries + 1e-9): raise RuntimeError(f"history query failure episode={episode} anchor={a}")
            if np.any(timestamp[a + 1:a + horizon + 1] <= t0): raise RuntimeError(f"future timestamp failure episode={episode} anchor={a}")
            history_native[i] = np.concatenate((arm[selected], hand[selected], leg[selected], action[selected]), axis=1).astype(np.float32)
            history_query[i] = queries.astype(np.float32); history_selected[i] = timestamp[selected].astype(np.float32); history_frames[i] = frame[selected]
            targets[i] = action[a + 1:a + horizon + 1].astype(np.float32); anchors_native[i] = action[a].astype(np.float32)
            episode_ids[i] = episode; tasks[i] = int(row["original_task_index"]); split_codes[i] = 0 if row["split"] == "train" else 1; anchor_frames[i] = a; anchor_ts[i] = t0; anchor_ids[i] = row["anchor_id"]; video_paths[i] = row["video_path"]
        return {"anchor_id": anchor_ids, "episode_index": episode_ids, "original_task_index": tasks, "split_code": split_codes, "anchor_frame": anchor_frames, "anchor_timestamp": anchor_ts, "video_path": video_paths, "history_native": history_native, "history_query_timestamps": history_query, "history_selected_timestamps": history_selected, "history_selected_frame_indices": history_frames, "targets_native": targets, "anchor_command_native": anchors_native}

    main_data = build(primary, 60); supp_data = build(supplementary, 30)
    train_mask = main_data["split_code"] == 0
    main_episode_ids = main_data["episode_index"]
    state_mean, state_std = episode_equal_stats(main_data["history_native"], np.repeat(main_episode_ids[:, None], HISTORY_STEPS, axis=1).reshape(-1), np.repeat(train_mask[:, None], HISTORY_STEPS, axis=1).reshape(-1)) if False else (None, None)
    # The helper above expects a last-axis matrix and episode mask per row;
    # flatten the window/time axes explicitly for transparent episode weighting.
    history_flat = main_data["history_native"].reshape(-1, STATE_DIM)
    history_episode_flat = np.repeat(main_episode_ids, HISTORY_STEPS)
    history_train_flat = np.repeat(train_mask, HISTORY_STEPS)
    state_mean, state_std = episode_equal_stats(history_flat, history_episode_flat, history_train_flat)
    relative = (main_data["history_query_timestamps"] - main_data["history_query_timestamps"][:, -1:,]).astype(np.float32)
    relative_flat = relative.reshape(-1, 1); relative_mean, relative_std = episode_equal_stats(relative_flat, history_episode_flat, history_train_flat)
    target_mean, target_std = episode_equal_stats(main_data["targets_native"].reshape(-1, ACTION_DIM), np.repeat(main_episode_ids, 60), np.repeat(train_mask, 60))
    target_sigma = target_std.copy()
    fd_scale = episode_equal_absdiff(main_data["targets_native"], main_episode_ids, train_mask)
    safe_fd = np.maximum(fd_scale, 0.001 * target_sigma).astype(np.float32)
    main_data["history_mask"] = np.ones(main_data["history_native"].shape[:2], dtype=np.float32)
    main_data["history_relative_delta_seconds"] = relative
    main_data["history_standardized"] = ((main_data["history_native"] - state_mean[None, None]) / state_std[None, None]).astype(np.float32)
    main_data["history_relative_delta_standardized"] = ((relative - relative_mean[0]) / relative_std[0]).astype(np.float32)
    supp_relative = (supp_data["history_query_timestamps"] - supp_data["history_query_timestamps"][:, -1:,]).astype(np.float32)
    supp_data["history_mask"] = np.ones(supp_data["history_native"].shape[:2], dtype=np.float32)
    supp_data["history_relative_delta_seconds"] = supp_relative
    supp_data["history_standardized"] = ((supp_data["history_native"] - state_mean[None, None]) / state_std[None, None]).astype(np.float32)
    supp_data["history_relative_delta_standardized"] = ((supp_relative - relative_mean[0]) / relative_std[0]).astype(np.float32)
    normalizer = {"status": "LOCKED", "fit_scope": "primary train windows only; episode equal", "train_episodes": int(len(np.unique(main_episode_ids[train_mask]))), "train_windows": int(train_mask.sum()), "state_mean": state_mean, "state_safe_std": state_std, "relative_delta_mean": float(relative_mean[0]), "relative_delta_safe_std": float(relative_std[0]), "target_mean": target_mean, "target_sigma": target_sigma, "fd_scale": fd_scale, "safe_fd_scale": safe_fd, "epsilon": EPS, "fit_ids_sha256": hashlib.sha256(np.unique(main_episode_ids[train_mask]).tobytes()).hexdigest(), "final_test_access": 0}
    save_npz(EXP / "data/V3_MAIN_INPUTS.npz", main_data); save_npz(EXP / "data/V3_SUPPLEMENTARY_1S_INPUTS.npz", supp_data)
    (EXP / "data/V3_NORMALIZERS.json").write_text(json.dumps(normalizer, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (x.item() if isinstance(x, np.generic) else str(x))) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "main_rows": len(primary), "supp_rows": len(supplementary), "train_rows": int(train_mask.sum()), "dev_rows": int((~train_mask).sum()), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
