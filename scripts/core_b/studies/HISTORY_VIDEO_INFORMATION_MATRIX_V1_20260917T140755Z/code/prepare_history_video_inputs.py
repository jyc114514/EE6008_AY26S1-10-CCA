#!/usr/bin/env python3
"""Build the train/dev-only 32-query-point history and fixed-video inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
CANONICAL_INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
CANONICAL_MANIFEST = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS_MANIFEST.json"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
SPLIT_MANIFEST = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
B2_ROOT = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
B2_INPUT = B2_ROOT / "inputs/B2_INPUTS_0125.npz"
B2_PREPARATION = B2_ROOT / "inputs/B2_INPUT_PREPARATION.json"
B2_FRAME_MAPS = B2_ROOT / "inputs/B2_FRAME_MAPS.json"
B2_SELECTION = B2_ROOT / "B2_VALID_SELECTION.json"
SOURCE_FPS = 30.0
PREFIX_FRAME_COUNT = 120
HORIZON_STEPS = 30
ACTION_DIM = 28
ARM_DIM = 14
HAND_DIM = 14
LEG_DIM = 15
STATE_DIM = ARM_DIM + HAND_DIM + LEG_DIM + ACTION_DIM
HISTORY_STEPS = 32
HISTORY_INTERVAL_SECONDS = 0.125
SEED = 20260939


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(part, path)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def list_matrix(table: Any, name: str, width: int) -> np.ndarray:
    values = table[name].to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"missing values in {name}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} shape {array.shape} != (*,{width})")
    return array


def read_episode(path: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["frame_index", "timestamp", "action", "observation.arm_joints", "observation.hand_joints", "observation.leg_joints"])
    frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"episode {episode_index}: frame_index is not contiguous")
    if len(frame) < PREFIX_FRAME_COUNT + HORIZON_STEPS:
        raise ValueError(f"episode {episode_index}: insufficient rows {len(frame)}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"episode {episode_index}: timestamps are not finite and increasing")
    action = list_matrix(table, "action", ACTION_DIM)
    arm = list_matrix(table, "observation.arm_joints", ARM_DIM)
    hand = list_matrix(table, "observation.hand_joints", HAND_DIM)
    leg = list_matrix(table, "observation.leg_joints", LEG_DIM)
    arrays = (frame, timestamps, action, arm, hand, leg)
    if not all(np.isfinite(item).all() for item in arrays[1:]):
        raise ValueError(f"episode {episode_index}: non-finite state/action values")
    return arrays


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_scale(values: np.ndarray, *, axis: tuple[int, ...], epsilon: np.ndarray | float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=axis)
    std = values.std(axis=axis)
    safe = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), safe.astype(np.float32)


def robust_nonzero_median(values: np.ndarray, epsilon: np.ndarray) -> tuple[np.ndarray, list[int]]:
    result = np.empty(values.shape[-1], dtype=np.float64)
    constant: list[int] = []
    for joint in range(values.shape[-1]):
        current = np.abs(values[..., joint]).reshape(-1)
        nonzero = current[current > 0]
        if len(nonzero) == 0:
            result[joint] = epsilon[joint]
            constant.append(joint)
        else:
            result[joint] = max(float(np.median(nonzero)), float(epsilon[joint]))
    return result.astype(np.float32), constant


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    data = {key: value.copy() for key, value in np.load(CANONICAL_INPUT, allow_pickle=False).items()}
    required = {"episode_index", "split_code", "original_task_index", "targets_native", "targets_standardized", "persistence_native", "persistence_standardized", "dynamic_pooled"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"canonical input keys missing: {missing}")
    n = len(data["episode_index"])
    if n != 3355 or data["targets_native"].shape != (n, HORIZON_STEPS, ACTION_DIM):
        raise ValueError(f"canonical target shape failed: {data['targets_native'].shape}")
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise ValueError("canonical train/dev count failed")
    ids = data["episode_index"].astype(np.int64)
    if len(np.unique(ids)) != n:
        raise ValueError("canonical IDs are not unique")

    b2_data = {key: value.copy() for key, value in np.load(B2_INPUT, allow_pickle=False).items()}
    if not np.array_equal(data["episode_index"], b2_data["episode_index"]) or not np.array_equal(data["split_code"], b2_data["split_code"]):
        raise ValueError("canonical/B2 ID or split order mismatch")
    dynamic_replay_max_abs = float(np.max(np.abs(data["dynamic_pooled"].astype(np.float64) - b2_data["dynamic_pooled"].astype(np.float64))))
    if dynamic_replay_max_abs != 0.0:
        raise ValueError(f"canonical fixed video feature differs from accepted B2 dense cache: {dynamic_replay_max_abs}")

    split = pd.read_parquet(SPLIT_MANIFEST, columns=["episode_index", "split", "v3_benchmark_inclusion", "original_task_index", "parquet_path", "video_path"])
    split = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")].copy()
    split_map = {int(row.episode_index): row for row in split.itertuples(index=False)}
    if not set(ids.tolist()).issubset(set(split_map)):
        raise ValueError("canonical IDs are not covered by the primary train/validation manifest")
    for row_index, episode in enumerate(ids.tolist()):
        source = split_map[int(episode)]
        expected_split = "train" if train_mask[row_index] else "validation"
        if str(source.split) != expected_split or int(source.original_task_index) != int(data["original_task_index"][row_index]):
            raise ValueError(f"canonical split/task mismatch for episode {episode}")

    history_native = np.empty((n, HISTORY_STEPS, STATE_DIM), dtype=np.float32)
    history_timestamps = np.empty((n, HISTORY_STEPS), dtype=np.float64)
    history_frame_indices = np.empty((n, HISTORY_STEPS), dtype=np.int64)
    query_timestamps = np.empty((n, HISTORY_STEPS), dtype=np.float64)
    target_timestamps = np.empty((n, HORIZON_STEPS), dtype=np.float64)
    timeline_rows: list[dict[str, object]] = []
    target_max_abs_error = 0.0
    anchor_max_abs_error = 0.0
    all_past_max_timestamp = -np.inf
    all_target_min_timestamp = np.inf
    query_offsets = np.arange(-(HISTORY_STEPS - 1), 1, dtype=np.float64) * HISTORY_INTERVAL_SECONDS
    for row_index, episode in enumerate(ids.tolist()):
        source = split_map[int(episode)]
        relative_parquet = Path(str(source.parquet_path))
        parquet_path = (RAW_ROOT / relative_parquet).resolve()
        if not parquet_path.is_relative_to(RAW_ROOT.resolve()) or not parquet_path.is_file():
            raise FileNotFoundError(f"episode {episode}: invalid parquet path {parquet_path}")
        frame, timestamps, action, arm, hand, leg = read_episode(parquet_path, int(episode))
        t0 = float(timestamps[PREFIX_FRAME_COUNT - 1])
        target_ts = timestamps[PREFIX_FRAME_COUNT : PREFIX_FRAME_COUNT + HORIZON_STEPS]
        if not np.all(target_ts > t0):
            raise ValueError(f"episode {episode}: target timestamp is not strictly after t0")
        queries = t0 + query_offsets
        selected = np.searchsorted(timestamps, queries, side="right") - 1
        if np.any(selected < 0) or np.any(selected >= PREFIX_FRAME_COUNT):
            raise ValueError(f"episode {episode}: history query falls outside legal prefix")
        selected_ts = timestamps[selected]
        if np.any(selected_ts > queries + 1e-9) or np.any(selected_ts > t0 + 1e-9):
            raise ValueError(f"episode {episode}: selected history row is after query/anchor")
        values = np.concatenate((arm[selected], hand[selected], leg[selected], action[selected]), axis=1).astype(np.float32)
        history_native[row_index] = values
        history_timestamps[row_index] = selected_ts
        history_frame_indices[row_index] = frame[selected]
        query_timestamps[row_index] = queries
        target_timestamps[row_index] = target_ts
        target_error = float(np.max(np.abs(data["targets_native"][row_index] - action[PREFIX_FRAME_COUNT : PREFIX_FRAME_COUNT + HORIZON_STEPS])))
        anchor_error = float(np.max(np.abs(data["persistence_native"][row_index, 0] - action[PREFIX_FRAME_COUNT - 1])))
        target_max_abs_error = max(target_max_abs_error, target_error)
        anchor_max_abs_error = max(anchor_max_abs_error, anchor_error)
        all_past_max_timestamp = max(all_past_max_timestamp, float(selected_ts.max()))
        all_target_min_timestamp = min(all_target_min_timestamp, float(target_ts.min()))
        timeline_rows.append({
            "episode_index": int(episode),
            "split": "train" if train_mask[row_index] else "dev",
            "original_task_index": int(data["original_task_index"][row_index]),
            "parquet_path": str(relative_parquet),
            "video_path": str(source.video_path),
            "anchor_frame_index": PREFIX_FRAME_COUNT - 1,
            "anchor_timestamp_seconds": t0,
            "target_start_frame_index": PREFIX_FRAME_COUNT,
            "target_start_timestamp_seconds": float(target_ts[0]),
            "history_query_start_seconds": float(queries[0]),
            "history_query_end_seconds": float(queries[-1]),
            "history_valid_count": int(np.sum(selected >= 0)),
            "history_frame_indices_json": json.dumps(frame[selected].tolist()),
            "history_timestamps_json": json.dumps(selected_ts.tolist()),
            "query_timestamps_json": json.dumps(queries.tolist()),
            "target_timestamps_json": json.dumps(target_ts.tolist()),
            "target_raw_max_abs_error": target_error,
            "anchor_raw_max_abs_error": anchor_error,
            "all_past_timestamps_le_t0": bool(np.all(selected_ts <= t0 + 1e-9)),
            "all_future_timestamps_gt_t0": bool(np.all(target_ts > t0)),
            "fixed_video_cache_prefix_verified": True,
            "final_test_access": 0,
        })

    if target_max_abs_error > 2e-6 or anchor_max_abs_error > 2e-6:
        raise ValueError(f"canonical/raw target seam failed: target={target_max_abs_error}, anchor={anchor_max_abs_error}")
    if not np.all(history_frame_indices < PREFIX_FRAME_COUNT):
        raise ValueError("history contains frame at/after target start")
    if not np.isfinite(history_native).all():
        raise ValueError("history contains non-finite values")

    # Train-only feature standardization. The binary mask is structural and is
    # intentionally kept in {0,1}; the relative-time channel is fitted below.
    state_mean, state_std = fit_scale(history_native[train_mask], axis=(0, 1))
    history_standardized = ((history_native - state_mean[None, None] ) / state_std[None, None]).astype(np.float32)
    mask = np.ones((n, HISTORY_STEPS), dtype=np.float32)
    relative_delta = (query_timestamps - query_timestamps[:, -1:,]).astype(np.float32)
    delta_mean, delta_std = fit_scale(relative_delta[train_mask, :, None], axis=(0, 1))
    relative_delta_standardized = ((relative_delta - delta_mean[0]) / delta_std[0]).astype(np.float32)

    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    target_mean = np.asarray(normalizer["mean"], dtype=np.float64)
    target_std = np.asarray(normalizer["safe_std"], dtype=np.float64)
    target_native = data["targets_native"].astype(np.float64)
    persistence_native = data["persistence_native"].astype(np.float64)
    target_hand = target_native[:, :, :HAND_DIM]
    persistence_hand = persistence_native[:, :, :HAND_DIM]
    train_hand_residual = target_hand[train_mask] - persistence_hand[train_mask]
    epsilon = 1e-6 * np.maximum(1.0, np.percentile(np.abs(target_hand[train_mask]), 95, axis=(0, 1)))
    sigma_residual, residual_constant_joints = robust_nonzero_median(train_hand_residual, epsilon)
    train_future_delta = np.diff(target_hand[train_mask], axis=1)
    fd_scale, fd_constant_joints = robust_nonzero_median(train_future_delta, epsilon)
    target_std_all = (target_native - target_mean[None, :, :]) / target_std[None, :, :]
    persistence_std_all = (persistence_native - target_mean[None, :, :]) / target_std[None, :, :]
    p_position_episode = np.abs(persistence_std_all[train_mask, :, :HAND_DIM] - target_std_all[train_mask, :, :HAND_DIM]).mean(axis=(1, 2))
    p_fd_episode = (np.abs(np.diff(persistence_hand[train_mask], axis=1) - np.diff(target_hand[train_mask], axis=1)) / fd_scale[None, None, :]).mean(axis=(1, 2))
    lambda_fd = float(np.mean(p_position_episode) / np.mean(p_fd_episode))
    if not np.isfinite(lambda_fd) or lambda_fd <= 0:
        raise ValueError(f"invalid train Persistence lambda_fd={lambda_fd}")

    arrays = {
        "episode_index": ids,
        "split_code": data["split_code"].astype(np.int8),
        "original_task_index": data["original_task_index"].astype(np.int64),
        "targets_native": target_native.astype(np.float32),
        "targets_standardized": target_std_all.astype(np.float32),
        "persistence_native": persistence_native.astype(np.float32),
        "persistence_standardized": persistence_std_all.astype(np.float32),
        "history_native": history_native,
        "history_standardized": history_standardized,
        "history_mask": mask,
        "history_query_timestamps": query_timestamps.astype(np.float32),
        "history_selected_timestamps": history_timestamps.astype(np.float32),
        "history_selected_frame_indices": history_frame_indices,
        "history_relative_delta_seconds": relative_delta.astype(np.float32),
        "history_relative_delta_standardized": relative_delta_standardized,
        "target_timestamps": target_timestamps.astype(np.float32),
        "dynamic_video_feature": data["dynamic_pooled"].astype(np.float32),
    }
    atomic_npz(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz", arrays)
    write_rows(ROOT / "DATA_TIMELINE_MANIFEST.csv", timeline_rows)
    atomic_json(ROOT / "DATA_TIMELINE_MANIFEST.json", {
        "status": "PASS",
        "rows": n,
        "train_count": int(train_mask.sum()),
        "dev_count": int(dev_mask.sum()),
        "history_shape": list(history_native.shape),
        "history_state_source_order": ["observation.arm_joints", "observation.hand_joints", "observation.leg_joints", "action"],
        "history_state_dimension": STATE_DIM,
        "query_count": HISTORY_STEPS,
        "query_interval_seconds": HISTORY_INTERVAL_SECONDS,
        "anchor_definition": "last legal prefix row before target start: frame 119",
        "target_start_definition": "frame 120",
        "maximum_selected_past_timestamp": all_past_max_timestamp,
        "minimum_target_timestamp": all_target_min_timestamp,
        "target_raw_max_abs_error": target_max_abs_error,
        "anchor_raw_max_abs_error": anchor_max_abs_error,
        "all_past_timestamps_le_t0": bool(np.all(history_timestamps <= query_timestamps[:, -1:,] + 1e-9)),
        "all_future_timestamps_gt_t0": bool(np.all(target_timestamps > query_timestamps[:, -1:,])),
        "final_test_access": 0,
    })
    fit_ids = ids[train_mask].tolist()
    atomic_json(ROOT / "PREPROCESS_FIT_IDS.json", {
        "status": "LOCKED",
        "fit_scope": "canonical train only",
        "fit_count": len(fit_ids),
        "fit_ids_sha256": digest_json(fit_ids),
        "fit_ids": fit_ids,
        "state_mean_sha256": hashlib.sha256(state_mean.tobytes()).hexdigest(),
        "state_safe_std_sha256": hashlib.sha256(state_std.tobytes()).hexdigest(),
        "relative_delta_mean": float(delta_mean[0]),
        "relative_delta_safe_std": float(delta_std[0]),
        "target_normalizer_path": str(NORMALIZER),
        "target_normalizer_sha256": digest(NORMALIZER),
        "residual_sigma_rule": "train nonzero absolute future Hand residual median per joint with predeclared epsilon",
        "residual_sigma": sigma_residual,
        "residual_constant_joints": residual_constant_joints,
        "fd_scale_rule": "train nonzero absolute future-to-future Hand delta median per joint with the same epsilon rule",
        "fd_scale": fd_scale,
        "fd_constant_joints": fd_constant_joints,
        "lambda_fd": lambda_fd,
        "persistence_train_position_mean": float(np.mean(p_position_episode)),
        "persistence_train_fd_scaled_mean": float(np.mean(p_fd_episode)),
        "final_test_access": 0,
    })
    atomic_json(ROOT / "SOURCE_INPUT_SEAMS.json", {
        "status": "PASS",
        "canonical_input": str(CANONICAL_INPUT),
        "canonical_input_sha256": digest(CANONICAL_INPUT),
        "canonical_manifest": str(CANONICAL_MANIFEST),
        "canonical_manifest_sha256": digest(CANONICAL_MANIFEST),
        "b2_input": str(B2_INPUT),
        "b2_input_sha256": digest(B2_INPUT),
        "b2_preparation": str(B2_PREPARATION),
        "b2_preparation_sha256": digest(B2_PREPARATION),
        "b2_frame_maps": str(B2_FRAME_MAPS),
        "b2_frame_maps_sha256": digest(B2_FRAME_MAPS),
        "b2_selection": str(B2_SELECTION),
        "b2_selection_sha256": digest(B2_SELECTION),
        "fixed_video_feature_key": "dynamic_pooled",
        "fixed_video_feature_shape": list(data["dynamic_pooled"].shape),
        "fixed_video_feature_exact_replay_max_abs": dynamic_replay_max_abs,
        "video_raw_io": False,
        "video_prefix_source": "accepted B2 dense 0.125s/32-frame frozen cache lineage",
        "state_source_fields": ["observation.arm_joints", "observation.hand_joints", "observation.leg_joints", "action"],
        "state_source_dimension": STATE_DIM,
        "raw_data_root": str(RAW_ROOT),
        "raw_data_read_only": True,
        "final_test_access": 0,
    })
    atomic_json(ROOT / "PREPROCESS_STATS.json", {
        "status": "LOCKED",
        "state_mean": state_mean,
        "state_safe_std": state_std,
        "relative_delta_mean": delta_mean,
        "relative_delta_safe_std": delta_std,
        "residual_sigma": sigma_residual,
        "fd_scale": fd_scale,
        "lambda_fd": lambda_fd,
        "target_mean_source": str(NORMALIZER),
        "target_mean_sha256": digest(NORMALIZER),
        "fit_ids_sha256": digest_json(fit_ids),
        "final_test_access": 0,
    })
    print(json.dumps({"status": "PASS", "rows": n, "train": int(train_mask.sum()), "dev": int(dev_mask.sum()), "history_shape": list(history_native.shape), "target_raw_max_abs_error": target_max_abs_error, "anchor_raw_max_abs_error": anchor_max_abs_error, "lambda_fd": lambda_fd, "output": str(ROOT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
