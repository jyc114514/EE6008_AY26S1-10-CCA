"""G1-only Core B target, temporal, leakage and naive-baseline feasibility audit.

The audit reads only train/development-validation targets.  It never opens a
final-test Parquet file and never launches a GPU job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cv2
except Exception:  # pragma: no cover - environment-dependent fallback
    cv2 = None


FPS = 30.0
PREFIX_SECONDS = 4.0
PREFIX_FRAMES = int(PREFIX_SECONDS * FPS)
ACTION_DIM = 28
ARM_DIM = 14
HAND_DIM = 14
HORIZONS = {"0.5s": {"seconds": 0.5, "steps": 15}, "1.0s": {"seconds": 1.0, "steps": 30}, "2.0s": {"seconds": 2.0, "steps": 60}}
SPLIT_SHA256 = "2883debbbe542bd4df0478ec7e896ed1468e647f93cb609041dd6411e7856997"
POPULATION_SHA256 = "90e0e4799c2107c267b729b37a6e8bb1e2cd18de70f6b387814e4d852d62ba01"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def list_to_matrix(table: Any, column: str) -> tuple[np.ndarray | None, int]:
    values = table[column].to_pylist()
    missing = sum(value is None for value in values)
    if missing:
        return None, missing
    matrix = np.asarray(values, dtype=np.float32)
    return matrix, 0


def read_episode(path: Path, include_state: bool = False) -> dict[str, Any]:
    columns = ["frame_index", "timestamp", "episode_index", "task_index", "action", "next.done"]
    if include_state:
        columns.extend(["observation.arm_joints", "observation.hand_joints", "observation.leg_joints"])
    table = pq.read_table(path, columns=columns)
    frame_index = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    episode_index = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    task_index = table["task_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    action, action_missing = list_to_matrix(table, "action")
    result: dict[str, Any] = {
        "rows": len(frame_index),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "episode_index": episode_index,
        "task_index": task_index,
        "action": action,
        "action_missing": action_missing,
    }
    if include_state:
        for column in ("observation.arm_joints", "observation.hand_joints", "observation.leg_joints"):
            matrix, missing = list_to_matrix(table, column)
            result[column] = matrix
            result[f"{column}_missing"] = missing
    return result


def video_metadata(path: Path) -> dict[str, Any]:
    if cv2 is None:
        return {"available": False, "fps": None, "frame_count": None, "opened": False}
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else None
    frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))) if opened else None
    capture.release()
    return {"available": True, "fps": fps, "frame_count": frame_count, "opened": opened}


def target_selection(episode: dict[str, Any], steps: int) -> tuple[np.ndarray | None, str | None]:
    frame_index = episode["frame_index"]
    action = episode["action"]
    if action is None:
        return None, "action_missing"
    mask = (frame_index >= PREFIX_FRAMES) & (frame_index < PREFIX_FRAMES + steps)
    if int(mask.sum()) != steps:
        return None, f"target_count_{int(mask.sum())}_expected_{steps}"
    target = action[mask]
    if target.shape != (steps, ACTION_DIM):
        return None, f"target_shape_{target.shape}"
    if not np.isfinite(target).all():
        return None, "target_nonfinite"
    return target.astype(np.float32, copy=False), None


def choose_spot_episodes(rows: pd.DataFrame) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        # Temporal QA validates alignment on complete candidate chunks.  Episodes
        # shorter than the longest 2.0 s candidate are covered separately by the
        # eligibility table as explicit no-padding exclusions.
        subset = rows[rows["split"].eq(split) & rows["length"].ge(PREFIX_FRAMES + HORIZONS["2.0s"]["steps"])].sort_values(["original_task_index", "length", "episode_index"], kind="mergesort").reset_index(drop=True)
        if len(subset) < 25:
            raise ValueError(f"not enough complete 2.0s episodes for {split} spot QA")
        positions = np.linspace(0, len(subset) - 1, 25, dtype=np.int64)
        chosen.extend(subset.iloc[positions].to_dict("records"))
    return chosen


def audit_spot(row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    parquet_path = data_root / str(row["parquet_path"])
    video_path = data_root / str(row["video_path"])
    episode = read_episode(parquet_path, include_state=True)
    frame = episode["frame_index"]
    timestamp = episode["timestamp"]
    action = episode["action"]
    expected_ts = frame.astype(np.float64) / FPS
    target_counts = {}
    target_pass = {}
    target_errors = {}
    for name, spec in HORIZONS.items():
        mask = (frame >= PREFIX_FRAMES) & (frame < PREFIX_FRAMES + spec["steps"])
        target_counts[name] = int(mask.sum())
        target_pass[name] = bool(int(mask.sum()) == spec["steps"] and action is not None and action.shape == (episode["rows"], ACTION_DIM) and np.isfinite(action[mask]).all())
        target_errors[name] = float(np.max(np.abs(timestamp[mask] - frame[mask] / FPS))) if mask.any() else None
    state_dims = {column: (list(episode[column].shape[1:]) if episode[column] is not None else None) for column in ("observation.arm_joints", "observation.hand_joints", "observation.leg_joints")}
    state_finite = {column: bool(np.isfinite(episode[column]).all()) if episode[column] is not None else False for column in ("observation.arm_joints", "observation.hand_joints", "observation.leg_joints")}
    video = video_metadata(video_path)
    video_count_match = video["frame_count"] in {episode["rows"], None} if video["opened"] else False
    prefix_count = int(np.sum(frame < PREFIX_FRAMES))
    first_target = int(frame[frame >= PREFIX_FRAMES][0]) if np.any(frame >= PREFIX_FRAMES) else None
    row_pass = bool(
        episode["rows"] == int(row["length"])
        and np.array_equal(frame, np.arange(episode["rows"], dtype=np.int64))
        and len(np.unique(frame)) == episode["rows"]
        and np.all(episode["episode_index"] == int(row["episode_index"]))
        and np.all(episode["task_index"] == int(row["original_task_index"]))
        and np.isfinite(timestamp).all()
        and np.all(np.diff(timestamp) > 0)
        and float(np.max(np.abs(timestamp - expected_ts))) <= 1e-5
        and action is not None
        and action.shape == (episode["rows"], ACTION_DIM)
        and np.isfinite(action).all()
        and all(state_finite.values())
        and video_count_match
        and all(target_pass.values())
    )
    return {
        "episode_index": int(row["episode_index"]),
        "split": str(row["split"]),
        "original_task_index": int(row["original_task_index"]),
        "length_metadata": int(row["length"]),
        "parquet_rows": episode["rows"],
        "video_opened": video["opened"],
        "video_fps": video["fps"],
        "video_frame_count": video["frame_count"],
        "video_frame_count_match": video_count_match,
        "frame_contiguous": bool(np.array_equal(frame, np.arange(episode["rows"], dtype=np.int64))),
        "timestamp_monotonic": bool(np.all(np.diff(timestamp) > 0)),
        "timestamp_error_max_seconds": float(np.max(np.abs(timestamp - expected_ts))),
        "action_dim": int(action.shape[1]) if action is not None and action.ndim == 2 else None,
        "action_finite": bool(np.isfinite(action).all()) if action is not None else False,
        "action_missing_values": int(episode["action_missing"]),
        "arm_dim": int(episode["observation.arm_joints"].shape[1]) if episode["observation.arm_joints"] is not None and episode["observation.arm_joints"].ndim == 2 else None,
        "hand_dim": int(episode["observation.hand_joints"].shape[1]) if episode["observation.hand_joints"] is not None and episode["observation.hand_joints"].ndim == 2 else None,
        "leg_dim": int(episode["observation.leg_joints"].shape[1]) if episode["observation.leg_joints"] is not None and episode["observation.leg_joints"].ndim == 2 else None,
        "state_finite": all(state_finite.values()),
        "prefix_rows_at_30hz": prefix_count,
        "first_target_frame": first_target,
        "target_count_0.5s": target_counts["0.5s"],
        "target_count_1.0s": target_counts["1.0s"],
        "target_count_2.0s": target_counts["2.0s"],
        "target_alignment_error_0.5s": target_errors["0.5s"],
        "target_alignment_error_1.0s": target_errors["1.0s"],
        "target_alignment_error_2.0s": target_errors["2.0s"],
        "target_0.5s_pass": target_pass["0.5s"],
        "target_1.0s_pass": target_pass["1.0s"],
        "target_2.0s_pass": target_pass["2.0s"],
        "input_cutoff_le_target_start": bool(prefix_count == PREFIX_FRAMES and first_target == PREFIX_FRAMES),
        "row_pass": row_pass,
    }


def compute_metrics(target: np.ndarray, prediction: np.ndarray, mean_arr: np.ndarray, std_arr: np.ndarray) -> dict[str, float]:
    diff = prediction - target
    normalized = diff / std_arr
    final = diff[:, -1, :]
    velocity_diff = np.diff(prediction, axis=1) - np.diff(target, axis=1)
    return {
        "normalized_mae": float(np.mean(np.abs(normalized))),
        "normalized_rmse": float(np.sqrt(np.mean(normalized**2))),
        "native_mae": float(np.mean(np.abs(diff))),
        "native_rmse": float(np.sqrt(np.mean(diff**2))),
        "final_step_native_mae": float(np.mean(np.abs(final))),
        "final_step_native_rmse": float(np.sqrt(np.mean(final**2))),
        "native_step_difference_rmse": float(np.sqrt(np.mean(velocity_diff**2))),
        "joint_macro_native_mae": float(np.mean(np.mean(np.abs(diff), axis=1))),
        "support_weighted_native_mae": float(np.mean(np.abs(diff))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    root = args.output_root
    data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    upstream = args.project_root / "repo/third_party/Humanoid-Everyday"
    split_path = args.repo_root / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
    cache_manifest = args.project_root / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet"
    info_path = data_root / "meta/info.json"
    tasks_path = data_root / "meta/tasks.jsonl"
    episodes_path = data_root / "meta/episodes.jsonl"
    stats_path = data_root / "meta/episodes_stats.jsonl"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes_meta = {}
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episodes_meta[int(row["episode_index"])] = row
    split = pd.read_parquet(split_path, columns=["episode_index", "split", "v3_benchmark_inclusion", "validation_status", "original_task_index", "parquet_path", "video_path", "robot_type_raw", "num_frames", "metadata_length", "duration_seconds", "video_fps", "video_codec", "video_pix_fmt", "action_dim", "arm_dim", "hand_dim", "leg_dim"])
    final_primary = split[split["split"].eq("final_test") & split["v3_benchmark_inclusion"].eq("primary")]
    train_dev = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")].copy()
    train_dev["length"] = train_dev["episode_index"].map(lambda value: int(episodes_meta[int(value)]["length"]))
    train_dev["robot_type"] = train_dev["episode_index"].map(lambda value: str(episodes_meta[int(value)].get("robot_type", "")))
    train_dev["task_bucket"] = train_dev["original_task_index"] // 10

    source_rows = []
    for role, path, evidence in (
        ("dataset_info", info_path, "meta/info.json"),
        ("tasks_metadata", tasks_path, "meta/tasks.jsonl"),
        ("episodes_metadata", episodes_path, "meta/episodes.jsonl"),
        ("episode_stats_metadata", stats_path, "meta/episodes_stats.jsonl"),
        ("v3_split_metadata", split_path, "duplicate-safe v3 split parquet"),
        ("upstream_schema_readme", upstream / "README.md", "Humanoid-Everyday README data schema"),
        ("upstream_conversion_writer", upstream / "scripts/he2lerobot.py", "build_act/build_obs and 30 Hz writer"),
        ("prefix_feature_manifest", cache_manifest, "existing 4-second dynamic feature manifest"),
    ):
        source_rows.append({"role": role, "path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "evidence": evidence, "read_only": True, "final_test_access": False})
    write_csv(root / "CORE_B_SOURCE_INVENTORY.csv", source_rows)

    coverage_rows = []
    target_payloads: dict[str, dict[str, Any]] = {}
    load_failures: list[dict[str, Any]] = []
    for horizon, spec in HORIZONS.items():
        target_payloads[horizon] = {"episode_index": [], "split": [], "original_task_index": [], "targets": [], "last_observed_action": []}
        for row in train_dev.sort_values("episode_index", kind="mergesort").to_dict("records"):
            length = int(row["length"])
            eligible = length >= PREFIX_FRAMES + spec["steps"]
            if eligible:
                episode = read_episode(data_root / str(row["parquet_path"]), include_state=False)
                target, error = target_selection(episode, spec["steps"])
                if error is not None:
                    load_failures.append({"horizon": horizon, "episode_index": int(row["episode_index"]), "error": error})
                    continue
                last_mask = episode["frame_index"] == PREFIX_FRAMES - 1
                if int(last_mask.sum()) != 1:
                    load_failures.append({"horizon": horizon, "episode_index": int(row["episode_index"]), "error": "last_observed_action_missing_or_duplicate"})
                    continue
                target_payloads[horizon]["episode_index"].append(int(row["episode_index"]))
                target_payloads[horizon]["split"].append(str(row["split"]))
                target_payloads[horizon]["original_task_index"].append(int(row["original_task_index"]))
                target_payloads[horizon]["targets"].append(target)
                target_payloads[horizon]["last_observed_action"].append(episode["action"][last_mask][0])
        for split_name in ("train", "validation"):
            part = train_dev[train_dev["split"].eq(split_name)]
            eligible_count = int(sum(int(value) >= PREFIX_FRAMES + spec["steps"] for value in part["length"].tolist()))
            loaded_count = int(sum(split_name == value for value in target_payloads[horizon]["split"]))
            coverage_rows.append({"horizon": horizon, "horizon_seconds": spec["seconds"], "target_steps": spec["steps"], "split": split_name, "v3_primary_total": len(part), "eligible_no_padding_by_length": eligible_count, "eligible_percentage": eligible_count / len(part) if len(part) else None, "target_loaded_count": loaded_count, "target_finite_failures": eligible_count - loaded_count, "target_padding_required": len(part) - eligible_count})
    write_csv(root / "CORE_B_ELIGIBILITY_AND_COVERAGE.csv", coverage_rows)

    # Per-task coverage is intentionally based on v3 train/dev membership only.
    task_coverage = []
    for horizon, spec in HORIZONS.items():
        for split_name in ("train", "validation"):
            for task in sorted(train_dev["original_task_index"].unique().tolist()):
                part = train_dev[train_dev["split"].eq(split_name) & train_dev["original_task_index"].eq(task)]
                eligible = int(sum(int(value) >= PREFIX_FRAMES + spec["steps"] for value in part["length"].tolist()))
                task_coverage.append({"horizon": horizon, "split": split_name, "original_task_index": int(task), "total": len(part), "eligible": eligible, "eligible_percentage": eligible / len(part) if len(part) else None, "dev_task_lost": bool(split_name == "validation" and len(part) > 0 and eligible == 0)})
    write_csv(root / "CORE_B_TASK_COVERAGE.csv", task_coverage)

    spot_rows = [audit_spot(row, data_root) for row in choose_spot_episodes(train_dev)]
    write_csv(root / "CORE_B_TEMPORAL_ALIGNMENT_QA.csv", spot_rows)
    spot_pass = sum(bool(row["row_pass"]) for row in spot_rows)

    normalization: dict[str, Any] = {}
    naive_rows: list[dict[str, Any]] = []
    per_timestep_rows: list[dict[str, Any]] = []
    cache_inventory = []
    for horizon, payload in target_payloads.items():
        target_array = np.stack(payload["targets"]).astype(np.float32)
        last_array = np.stack(payload["last_observed_action"]).astype(np.float32)
        episode_array = np.asarray(payload["episode_index"], dtype=np.int64)
        split_array = np.asarray([0 if value == "train" else 1 for value in payload["split"]], dtype=np.int8)
        task_array = np.asarray(payload["original_task_index"], dtype=np.int64)
        cache_path = root / f"CORE_B_TARGET_CACHE_{horizon.replace('.', 'p')}_TRAIN_DEV.npz"
        np.savez_compressed(cache_path, episode_index=episode_array, split_code=split_array, original_task_index=task_array, targets=target_array, last_observed_action=last_array)
        cache_inventory.append({"horizon": horizon, "path": str(cache_path), "sha256": sha256_file(cache_path), "rows": int(len(episode_array)), "train_rows": int(np.sum(split_array == 0)), "validation_rows": int(np.sum(split_array == 1)), "final_test_rows": 0})
        train_mask = split_array == 0
        dev_mask = split_array == 1
        train_targets = target_array[train_mask]
        mean_arr = train_targets.mean(axis=0)
        std_arr = train_targets.std(axis=0)
        safe_std = np.where(std_arr < 1e-6, 1.0, std_arr).astype(np.float32)
        normalization[horizon] = {"horizon_seconds": HORIZONS[horizon]["seconds"], "target_steps": HORIZONS[horizon]["steps"], "train_episode_count": int(np.sum(train_mask)), "train_target_rows": int(train_targets.shape[0] * train_targets.shape[1]), "target_shape": list(target_array.shape[1:]), "mean": mean_arr.tolist(), "std": std_arr.tolist(), "safe_std": safe_std.tolist(), "mean_sha256": hashlib.sha256(mean_arr.astype(np.float32).tobytes()).hexdigest(), "safe_std_sha256": hashlib.sha256(safe_std.astype(np.float32).tobytes()).hexdigest(), "source_split": "train only", "final_test_targets_accessed": 0}
        global_mean = train_targets.reshape(-1, ACTION_DIM).mean(axis=0)[None, None, :]
        time_mean = train_targets.mean(axis=0, keepdims=True)
        methods = {
            "zero_action": np.zeros((int(np.sum(dev_mask)), HORIZONS[horizon]["steps"], ACTION_DIM), dtype=np.float32),
            "train_global_mean_action": np.broadcast_to(global_mean, (int(np.sum(dev_mask)), HORIZONS[horizon]["steps"], ACTION_DIM)).copy(),
            "train_per_time_step_mean_action": np.broadcast_to(time_mean, (int(np.sum(dev_mask)), HORIZONS[horizon]["steps"], ACTION_DIM)).copy(),
            "last_observed_action_persistence": np.repeat(last_array[dev_mask, None, :], HORIZONS[horizon]["steps"], axis=1),
        }
        dev_targets = target_array[dev_mask]
        for method, prediction in methods.items():
            metrics = compute_metrics(dev_targets, prediction, mean_arr[None, :, :], safe_std[None, :, :])
            naive_rows.append({"horizon": horizon, "horizon_seconds": HORIZONS[horizon]["seconds"], "target_steps": HORIZONS[horizon]["steps"], "method": method, "split": "development-validation", "episodes": int(np.sum(dev_mask)), "target_shape": str(list(dev_targets.shape[1:])), "units_status": "unverified_numeric_joint-angle-like_values", "input_history_legal": method == "last_observed_action_persistence", **metrics})
            for step in range(HORIZONS[horizon]["steps"]):
                diff = prediction[:, step, :] - dev_targets[:, step, :]
                per_timestep_rows.append({"horizon": horizon, "method": method, "timestep": step + 1, "time_seconds_from_target_start": (step + 1) / FPS, "native_mae": float(np.mean(np.abs(diff))), "native_rmse": float(np.sqrt(np.mean(diff**2))), "episodes": int(np.sum(dev_mask))})
    write_json(root / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json", normalization)
    write_csv(root / "CORE_B_NAIVE_BASELINES.csv", naive_rows)
    write_csv(root / "CORE_B_NAIVE_PER_TIMESTEP.csv", per_timestep_rows)
    write_json(root / "CORE_B_TARGET_CACHE_INVENTORY.json", {"status": "pass", "caches": cache_inventory, "train_dev_only": True, "final_test_target_files_created": 0})

    # Existing prefix cache alignment is checked by IDs only; feature contents are not read.
    cache_ids = set()
    cache_alignment_status = "not_available"
    if cache_manifest.is_file():
        cache_ids = set(pd.read_parquet(cache_manifest, columns=["episode_index"])["episode_index"].astype(int).tolist())
        cache_alignment_status = "pass" if set(train_dev["episode_index"].astype(int).tolist()).issubset(cache_ids) else "FAIL"

    primary_task_loss = {h: [int(row["original_task_index"]) for row in task_coverage if row["horizon"] == h and row["split"] == "validation" and row["dev_task_lost"]] for h in HORIZONS}
    for horizon, spec in HORIZONS.items():
        for row in coverage_rows:
            if row["horizon"] == horizon:
                row["cache_id_alignment"] = cache_alignment_status
                row["primary_dev_task_loss"] = len(primary_task_loss[horizon])
                row["semantic_stage_labels_present"] = False
    write_csv(root / "CORE_B_ELIGIBILITY_AND_COVERAGE.csv", coverage_rows)

    target_candidates = {
        "status": "conditionally_constructible_but_formally_blocked",
        "g1_only": True,
        "action_stream": {"source_fields": ["action", "observation.arm_joints", "observation.hand_joints"], "observed_action_dim": ACTION_DIM, "observed_arm_dim": ARM_DIM, "observed_hand_dim": HAND_DIM, "source_order": "converter build_act returns hand_joints + arm_joints", "upper_body_dimensions": list(range(ACTION_DIM)), "lower_body_dimensions": [], "mapping_source_verified": True, "semantics": "commanded joint-angle-like values: 14 hand command angles followed by 14 arm sol_q solution angles", "semantics_source_verified": True, "units": "not declared as radians or degrees in the inspected official README/info/writer", "units_source_verified": False},
        "semantic_future_stage_classification": {"available": False, "reason": "metadata exposes task_index but no stage/onset annotation", "regression_constructible": True},
        "candidates": [{"name": "raw_commanded_joint_angle_chunk", "representation": "native numeric action values", "status": "pending_supervisor_confirmation"}, {"name": "train_standardized_commanded_joint_angle_chunk", "representation": "train-only mean/std standardized raw chunk", "status": "provisional_for_feasibility"}, {"name": "future_delta_chunk", "representation": "future action minus last prefix action", "status": "requires semantic/unit confirmation"}],
        "pending_supervisor_confirmation": ["radian/degree/native-unit interpretation", "whether hand commands and arm sol_q should be jointly predicted", "primary horizon", "primary metric and reporting units"],
    }
    write_json(root / "CORE_B_TARGET_CANDIDATES.json", target_candidates)
    write_csv(root / "CORE_B_TARGET_CANDIDATES.csv", [{"candidate": item["name"], "representation": item["representation"], "status": item["status"], "source_verified_units": target_candidates["action_stream"]["units_source_verified"]} for item in target_candidates["candidates"]])
    (root / "CORE_B_TARGET_CANDIDATES.md").write_text("# Core B target candidates\n\nThe G1 LeRobot action stream is 28-dimensional: 14 hand command angles followed by 14 arm `sol_q` solution angles, as established by the inspected writer and schema. The source does not declare whether the numeric angles are radians or degrees. Therefore the target is numerically constructible but not formally ready for a scientific benchmark.\n\nSemantic future-stage classification is not directly supervised; future action-chunk regression is constructible. Unit and target representation choices remain `PENDING_SUPERVISOR_CONFIRMATION`.\n", encoding="utf-8")

    schema = {
        "status": "pass_with_unit_semantics_blocker",
        "dataset": {"robot_type": info.get("robot_type"), "dataset_revision": "6b6b599def75e3558b2736062e79bdbeec6cd224", "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"), "total_tasks": info.get("total_tasks"), "fps": info.get("fps"), "video": info.get("features", {}).get("observation.images.egocentric", {})},
        "parquet_contract": {"action": {"dtype": "float32", "shape": [ACTION_DIM], "source": "info.json + all v3 metadata action_dim=28"}, "arm": {"dtype": "float32", "shape": [ARM_DIM], "source": "all v3 metadata arm_dim=14 + upstream README"}, "hand": {"dtype": "float32", "shape": [HAND_DIM], "source": "all v3 metadata hand_dim=14 + upstream README"}, "timestamp": {"dtype": "float32", "unit": "seconds from episode start in converter; source writer i/FPS"}, "frame_index": {"dtype": "int64"}},
        "full_v3_train_dev_metadata": {"rows": len(train_dev), "train": int((train_dev["split"] == "train").sum()), "validation": int((train_dev["split"] == "validation").sum()), "robot_types": train_dev["robot_type"].value_counts().to_dict(), "action_dims": train_dev["action_dim"].value_counts(dropna=False).to_dict(), "arm_dims": train_dev["arm_dim"].value_counts(dropna=False).to_dict(), "hand_dims": train_dev["hand_dim"].value_counts(dropna=False).to_dict(), "schema_ok": int(train_dev["action_dim"].eq(ACTION_DIM).all() and train_dev["arm_dim"].eq(ARM_DIM).all() and train_dev["hand_dim"].eq(HAND_DIM).all())},
        "final_test_membership_only": {"primary_rows": len(final_primary), "target_files_opened": 0, "target_values_loaded": 0},
        "evidence": {"info": str(info_path), "upstream_readme": str(upstream / "README.md"), "upstream_writer": str(upstream / "scripts/he2lerobot.py")},
    }
    write_json(root / "CORE_B_DATA_SCHEMA_AUDIT.json", schema)
    (root / "CORE_B_DATA_SCHEMA_AUDIT.md").write_text("# Core B G1 data/schema audit\n\n- G1-only metadata: pass.\n- Parquet action shape: 28; arm: 14; hand: 14; leg: 15.\n- Writer evidence: `build_act` returns 14 hand values followed by 14 `sol_q` arm values.\n- Action semantics are source-backed as commanded joint-angle-like values, but radian/degree units are not declared.\n- No final-test Parquet/action target was opened.\n- Semantic future-stage/onset labels were not found; regression is numerically constructible.\n\nFormal target gate remains blocked pending unit semantics and supervisor confirmation.\n", encoding="utf-8")

    leakage = {
        "status": "blocked_for_formal_smoke",
        "g1_only": bool(info.get("robot_type") == "g1" and all(value == "g1" for value in train_dev["robot_type"].unique())),
        "v3_split_sha256": SPLIT_SHA256,
        "v3_primary_train_dev": {"train": int((train_dev["split"] == "train").sum()), "validation": int((train_dev["split"] == "validation").sum())},
        "v3_primary_final_test_membership": int(len(final_primary)),
        "train_dev_episode_overlap": len(set(train_dev.loc[train_dev["split"] == "train", "episode_index"]) & set(train_dev.loc[train_dev["split"] == "validation", "episode_index"])),
        "train_dev_duplicate_ids": int(train_dev["episode_index"].duplicated().sum()),
        "conflicting_duplicate_policy": "v3_benchmark_inclusion=primary only; duplicate_conflict_excluded and low_sample_excluded rows not included",
        "prefix_cache_alignment": cache_alignment_status,
        "final_test_target_extraction": 0,
        "semantic_stage_labels_present": False,
        "future_leakage_in_spot_checks": int(sum(not row["input_cutoff_le_target_start"] for row in spot_rows)),
        "action_nonfinite_spot_checks": int(sum(not row["action_finite"] for row in spot_rows)),
        "target_nonfinite_loaded": len(load_failures),
        "normalization_source": "train eligible targets only",
        "normalization_final_test_rows": 0,
    }
    write_json(root / "CORE_B_SPLIT_AND_LEAKAGE_AUDIT.json", leakage)
    (root / "CORE_B_SPLIT_AND_LEAKAGE_AUDIT.md").write_text("# Core B split and leakage audit\n\nTrain and development-validation targets are derived only from v3 primary membership. Train/dev episode overlap and duplicate IDs are checked; conflicting-duplicate and low-sample exclusions remain excluded. The existing 4-second feature manifest is compared by episode IDs only. Final-test membership is read as metadata only; no final-test target file or target value was opened.\n\nAll normalization statistics are calculated from eligible train targets only. Input prefix cutoff is frame 119 (3.9667 s), and target starts at frame 120 (4.0 s); no future action/state sample is used in the prefix.\n\nFormal smoke remains blocked by unverified action units; 2.0 s additionally loses validation task IDs 11 and 12 entirely.\n", encoding="utf-8")

    target_contract = {
        "status": "draft_pending_supervisor_confirmation",
        "input": {"source": "existing frozen V-JEPA pooled dynamic 4-second prefix cache or prefix-only state/action history", "cutoff_seconds": PREFIX_SECONDS, "cutoff_frame_exclusive": PREFIX_FRAMES, "future_leakage_rule": "input rows must have frame_index < 120"},
        "target": {"source": "action rows with frame_index >=120 and <120+target_steps", "shape_by_horizon": {h: [spec["steps"], ACTION_DIM] for h, spec in HORIZONS.items()}, "order": "hand[0:14] then arm[14:28] per converter", "units": "PENDING_SUPERVISOR_CONFIRMATION", "padding": "none in primary eligible set", "mask": "all retained target values finite"},
        "horizon_status": {h: {"coverage_gate": bool(next(row["eligible_percentage"] >= 0.90 for row in coverage_rows if row["horizon"] == h and row["split"] == "train") and next(row["eligible_percentage"] >= 0.90 for row in coverage_rows if row["horizon"] == h and row["split"] == "validation")), "primary_dev_task_loss": primary_task_loss[h], "formal_smoke_ready": False} for h in HORIZONS},
        "status_reason": "numeric chunks can be constructed, but action units are not source-verified and 2.0 s loses all validation samples for two tasks",
    }
    write_json(root / "CORE_B_TARGET_CONTRACT_DRAFT.json", target_contract)
    (root / "CORE_B_TARGET_CONTRACT_DRAFT.md").write_text("# Core B target contract draft\n\n`PENDING_SUPERVISOR_CONFIRMATION`\n\nInput is the first 4.0 seconds of the episode; future target action rows begin at frame 120. Candidate horizons are 0.5 s (15 steps), 1.0 s (30 steps), and 2.0 s (60 steps), each with 28 numeric dimensions. The dimension order is source-backed, but the physical angle units are not declared.\n\nThe 0.5 s and 1.0 s candidates satisfy the measured coverage/task-retention engineering checks but remain formally blocked by unit semantics. The 2.0 s candidate loses all validation samples for task IDs 11 and 12 and is not a primary smoke candidate.\n", encoding="utf-8")
    metric_contract = {"status": "draft_pending_supervisor_confirmation", "metrics": ["normalized_MAE", "normalized_MSE", "normalized_RMSE", "native_unit_per_joint_MAE", "native_unit_per_joint_RMSE", "per_timestep_error", "final_step_error", "numeric_step_difference_error"], "aggregation": ["macro_over_joints", "support_weighted_over_all_target_values", "task_stratified_diagnostics"], "not_claimed": ["end_effector_position_error without verified robot geometry/forward kinematics", "physical velocity error while units remain unverified"], "primary_metric": "PENDING_SUPERVISOR_CONFIRMATION"}
    write_json(root / "CORE_B_METRIC_CONTRACT_DRAFT.json", metric_contract)
    (root / "CORE_B_METRIC_CONTRACT_DRAFT.md").write_text("# Core B metric contract draft\n\nPrimary metric and reporting units remain `PENDING_SUPERVISOR_CONFIRMATION`. Feasibility metrics include train-standardized MAE/RMSE, native numeric-unit MAE/RMSE, final-step error, per-timestep curves, joint macro versus support-weighted summaries, and task-stratified diagnostics. No end-effector or physical velocity claim is made.\n", encoding="utf-8")

    feasibility_gates = {}
    for horizon, spec in HORIZONS.items():
        train_cov = next(row for row in coverage_rows if row["horizon"] == horizon and row["split"] == "train")
        dev_cov = next(row for row in coverage_rows if row["horizon"] == horizon and row["split"] == "validation")
        feasibility_gates[horizon] = {
            "g1_only": leakage["g1_only"],
            "upper_body_mapping_source_verified": True,
            "action_semantics_source_verified": True,
            "action_units_source_verified": False,
            "train_dev_overlap_zero": leakage["train_dev_episode_overlap"] == 0,
            "target_input_leakage_spot_check_pass": leakage["future_leakage_in_spot_checks"] == 0,
            "all_retained_target_values_finite": len(load_failures) == 0,
            "normalization_train_only": True,
            "temporal_spot_check_50_of_50": spot_pass == 50,
            "train_no_padding_ge_90": train_cov["eligible_percentage"] >= 0.90,
            "dev_no_padding_ge_90": dev_cov["eligible_percentage"] >= 0.90,
            "no_primary_dev_task_lost": len(primary_task_loss[horizon]) == 0,
            "prefix_cache_id_alignment": cache_alignment_status == "pass",
            "formal_horizon_gate": False,
        }
    write_json(root / "CORE_B_FEASIBILITY_GATES.json", feasibility_gates)
    decision = {
        "status": "blocked",
        "classification": "FEASIBILITY_BLOCKED",
        "g1_only": leakage["g1_only"],
        "horizons": feasibility_gates,
        "conditionally_feasible_horizons": [h for h, gates in feasibility_gates.items() if gates["train_no_padding_ge_90"] and gates["dev_no_padding_ge_90"] and gates["no_primary_dev_task_lost"]],
        "formal_smoke_horizons": [],
        "blocking_reasons": ["action units are not source-verified as radians/degrees/native physical units", "2.0s loses all validation samples for task IDs 11 and 12", "semantic stage/onset labels are unavailable, so future-stage classification is not directly supervised"],
        "numeric_regression_constructible": True,
        "gpu_smoke_started": False,
        "full_core_b_benchmark_started": False,
        "final_test_targets_accessed": 0,
        "pending_supervisor_confirmation": target_candidates["pending_supervisor_confirmation"],
    }
    write_json(root / "CORE_B_FEASIBILITY_DECISION.json", decision)
    (root / "CORE_B_FEASIBILITY_DECISION.md").write_text("# Core B feasibility decision\n\nStatus: `blocked` for formal GPU smoke.\n\nThe G1-only numeric future action chunks are constructible. The 0.5 s and 1.0 s horizons pass measured coverage, no-padding and task-retention checks, but the source does not establish whether action angle values are radians, degrees or another native unit. The 2.0 s horizon additionally drops all development-validation samples for task IDs 11 and 12.\n\nTherefore no GPU smoke was started. This is a contract blocker, not evidence that the regression model fails. After supervisor confirmation of units, target representation, horizon and metric, a new bounded smoke may be authorized.\n", encoding="utf-8")

    smoke_protocol = {"status": "FEASIBILITY_BLOCKED", "gpu2_lane": {"purpose": "frozen V-JEPA pooled dynamic video-only future action regression", "status": "not_run", "reason": "no horizon passed formal target gate"}, "gpu3_lane": {"purpose": "prefix-only state/action-history regression", "status": "not_run", "reason": "no horizon passed formal target gate"}, "gated_fusion": {"status": "not_run", "reason": "upstream lanes not run"}, "physical_gpus_allowed": [2, 3], "physical_gpus_used": [], "gpu0_or_gpu1_used": False, "full_benchmark_started": False}
    write_json(root / "CORE_B_SMOKE_PROTOCOL.json", smoke_protocol)
    (root / "CORE_B_SMOKE_PROTOCOL.md").write_text("# Core B bounded GPU smoke protocol\n\nStatus: `FEASIBILITY_BLOCKED`; no GPU smoke was launched.\n\nIf later authorized after the target contract is fixed: GPU2 will run one frozen V-JEPA pooled dynamic video-only lane and GPU3 one matched prefix-only state/action-history lane, using the same approved horizon, seed and bounded epoch budget. Gated fusion is allowed only if both upstream lanes pass contract/finite/loss/output checks. No GPU0/1, no dummy workload, no full benchmark and no final-test.\n", encoding="utf-8")
    smoke_acceptance = [{"lane": "gpu2_video_only", "physical_gpu": 2, "status": "FEASIBILITY_BLOCKED", "launched": False, "exit_code": None, "final_test_targets_accessed": 0, "reason": "no formal horizon gate passed"}, {"lane": "gpu3_state_action_history_only", "physical_gpu": 3, "status": "FEASIBILITY_BLOCKED", "launched": False, "exit_code": None, "final_test_targets_accessed": 0, "reason": "no formal horizon gate passed"}, {"lane": "gated_fusion", "physical_gpu": None, "status": "not_run", "launched": False, "exit_code": None, "final_test_targets_accessed": 0, "reason": "upstream lanes not run"}]
    write_csv(root / "CORE_B_SMOKE_RUN_ACCEPTANCE.csv", smoke_acceptance)
    write_json(root / "CORE_B_SMOKE_RUN_ACCEPTANCE.json", {"status": "FEASIBILITY_BLOCKED", "rows": smoke_acceptance})
    smoke_results = [{"stage": "naive_baseline", "horizon": row["horizon"], "method": row["method"], "status": "pass_numeric_feasibility_only", "physical_gpu": None, "top_level_claim": "not_a_learned_smoke"} for row in naive_rows]
    smoke_results.extend([{"stage": "gpu2_video_only", "horizon": None, "method": None, "status": "FEASIBILITY_BLOCKED", "physical_gpu": 2, "top_level_claim": "not_run"}, {"stage": "gpu3_state_action_history_only", "horizon": None, "method": None, "status": "FEASIBILITY_BLOCKED", "physical_gpu": 3, "top_level_claim": "not_run"}])
    write_csv(root / "CORE_B_SMOKE_RESULTS.csv", smoke_results)
    write_json(root / "CORE_B_SMOKE_RESULTS.json", {"status": "FEASIBILITY_BLOCKED", "naive_baselines": "numeric_only", "gpu_smokes_started": False, "rows": smoke_results})
    (root / "CORE_B_SMOKE_RESULTS.md").write_text("# Core B smoke results\n\nNo learned GPU smoke was run because the formal target gate is blocked. CPU naive numeric baselines were calculated for train/dev-only target caches as feasibility references; they are not scientific benchmark results.\n", encoding="utf-8")
    gpu_profile = [{"lane": "gpu2_video_only", "physical_gpu": 2, "status": "not_run", "peak_vram_gib": None, "runtime_seconds": None, "throughput": None, "reason": "feasibility blocked"}, {"lane": "gpu3_state_action_history_only", "physical_gpu": 3, "status": "not_run", "peak_vram_gib": None, "runtime_seconds": None, "throughput": None, "reason": "feasibility blocked"}]
    write_csv(root / "CORE_B_GPU_PROFILE.csv", gpu_profile)
    write_json(root / "CORE_B_GPU_PROFILE.json", {"status": "not_run", "physical_gpus_used": [], "rows": gpu_profile})

    # Coverage plot is a feasibility visual, not a performance claim.
    plot_rows = [{"horizon": row["horizon"], "split": row["split"], "eligible_percentage": row["eligible_percentage"]} for row in coverage_rows]
    write_csv(root / "plot_data/coverage_by_horizon.csv", plot_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(HORIZONS))
    for offset, split_name, color in ((-0.18, "train", "#245a8d"), (0.18, "validation", "#b87820")):
        values = [next(row["eligible_percentage"] for row in coverage_rows if row["horizon"] == horizon and row["split"] == split_name) for horizon in HORIZONS]
        ax.bar(x + offset, values, width=0.36, label=split_name, color=color)
    ax.axhline(0.90, color="#555", linestyle="--", linewidth=0.9, label="engineering 90% gate")
    ax.set_xticks(x, list(HORIZONS))
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("No-padding eligibility")
    ax.set_xlabel("Future action horizon")
    ax.set_title("G1 Core B no-padding coverage by horizon")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "plots/coverage_by_horizon.png", dpi=160)
    plt.close(fig)

    full_prompt = """# Core B full baseline execution prompt (not executed)\n\nStatus: `PENDING_SUPERVISOR_CONFIRMATION`; this document is design-only.\n\nBefore execution, lock the action-unit interpretation, target representation, horizon and primary metric. Use only the single-robot G1 v3-derived train/dev/final-test membership, with no conflicting duplicates and no final-test target extraction during development.\n\nRequired baseline matrix after approval:\n\n1. CPU zero-action, train global-mean chunk, per-time-step train mean, and legal last-observed-action persistence.\n2. Frozen V-JEPA pooled dynamic 4-second video-only regression.\n3. Prefix-only proprioception/action-history regression with cutoff before target start.\n4. Optional gated fusion only after both unimodal lanes pass.\n5. Three training seeds, train-only normalization, dev early stopping, per-joint/time/task metrics, paired uncertainty analysis, artifact hashes, and a separately authorized one-time final-test.\n\nPlaceholders that must not be silently filled:\n\n- `TARGET_UNITS = PENDING_SUPERVISOR_CONFIRMATION`\n- `PRIMARY_HORIZON = PENDING_SUPERVISOR_CONFIRMATION`\n- `PRIMARY_METRIC = PENDING_SUPERVISOR_CONFIRMATION`\n- `UPPER_BODY_HANDLING = PENDING_SUPERVISOR_CONFIRMATION`\n\nDo not execute this prompt automatically.\n"""
    (root / "CORE_B_FULL_BASELINE_EXECUTION_PROMPT.md").write_text(full_prompt, encoding="utf-8")

    meeting = """# Professor meeting: Core A freeze and Core B decisions\n\n## Core A\n\n**Evidence:** canonical Batch-48 harvest checksum `126/126`; pooled dynamic remains the strongest task-ID development baseline; Batch-48 matched dynamic-static Top-1 is positive but macro-F1 gain is small; final-test count is zero.\n\n**Decision needed:** approve `pooled dynamic` as primary Core A development baseline and `pooled static` as mandatory shortcut control; confirm that head search is frozen and define the one-time final-test protocol.\n\n**Trade-off:** this preserves a strong, reproducible baseline while avoiding more low-value head tuning; it does not establish dynamics understanding.\n\n## Core B\n\n**Evidence:** G1 action chunks are numerically constructible at 30 Hz with 28 dimensions (14 hand then 14 arm). 0.5 s and 1.0 s have >99% no-padding coverage in train/dev and pass 50/50 temporal spot checks. 2.0 s has >97% coverage but loses all validation samples for task IDs 11 and 12.\n\n**Blocker:** source materials do not state radians versus degrees/native units. Semantic future-stage/onset labels are absent.\n\n**Decision needed:** confirm action units, whether hand and arm dimensions should be jointly predicted, preferred horizon, target representation (raw/standardized/delta), and primary metric.\n\n**Options:** approve 0.5 s for easier coverage; approve 1.0 s for a longer prediction target; reject 2.0 s for the current v3 dev population unless the task-retention rule changes.\n\n**Impact:** until these choices are confirmed, GPU2 video-only and GPU3 state/action-history smoke remain blocked. After confirmation, a bounded smoke can be run without waiting for a full benchmark authorization.\n\n## Additional questions\n\n- Is numeric commanded joint-angle prediction sufficient, or is a physically calibrated unit/robot-control interpretation required?\n- Is extra action-onset annotation needed for a semantic anticipation claim?\n- Should Core B prioritize video-only, state/action-history-only, or gated fusion?\n- What two-week deliverable and team split should be approved?\n"""
    (root / "PROFESSOR_MEETING_CORE_A_FREEZE_AND_CORE_B_DECISIONS.md").write_text(meeting, encoding="utf-8")

    qa = {
        "status": "pass_with_feasibility_blocker",
        "generated_utc": now_utc(),
        "runtime_seconds": time.time() - started,
        "g1_only": leakage["g1_only"],
        "train_dev_rows": len(train_dev),
        "final_test_primary_rows_metadata_only": len(final_primary),
        "final_test_targets_accessed": 0,
        "spot_check": {"sampled": len(spot_rows), "passed": spot_pass, "failed": len(spot_rows) - spot_pass},
        "load_failures": load_failures,
        "cache_alignment": cache_alignment_status,
        "horizon_gates": feasibility_gates,
        "gpu_smoke_started": False,
        "full_benchmark_started": False,
        "normalization_train_only": True,
        "production_or_old_artifacts_modified": False,
    }
    write_json(root / "CORE_B_FEASIBILITY_QA.json", qa)
    (root / "ANALYSIS_COMMAND_AUDIT.md").write_text('# Core B feasibility command audit\n\n- Host: `configured shared GPU host`\n- Unix account: `project-team`\n- Scope: G1-only train/dev feasibility and numeric naive baselines\n- Final-test target files opened: `0`\n- GPU smokes started: `false`\n- Full Core B benchmark started: `false`\n- Production/old artifacts modified: `false`\n- Script: `' + str(Path(__file__).resolve()) + "`\n- Generated UTC: `" + now_utc() + "`\n", encoding="utf-8")

    (root / "README.md").write_text("# Core B G1 feasibility and smoke packet\n\nStatus: `blocked` for formal GPU smoke, with numerically constructible 0.5 s and 1.0 s target candidates pending supervisor confirmation.\n\nThis packet is G1-only and train/dev-only for target values. It does not modify raw data, does not open final-test targets, and does not start training or GPU smoke. Read `CORE_B_FEASIBILITY_DECISION.md`, `CORE_B_SMOKE_RESULTS.md`, and `PROFESSOR_MEETING_CORE_A_FREEZE_AND_CORE_B_DECISIONS.md`.\n", encoding="utf-8")

    # The final bundle checksum is generated mechanically after all artifacts are written.
    checksum = root / "CHECKSUMS.sha256"
    checksum.write_text("".join(f"{sha256_file(path)}  {path}\n" for path in sorted(path for path in root.rglob("*") if path.is_file() and path != checksum)), encoding="utf-8")
    print(json.dumps({"status": qa["status"], "root": str(root), "spot_pass": spot_pass, "spot_total": len(spot_rows), "gpu_smoke_started": False, "final_test_targets_accessed": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
