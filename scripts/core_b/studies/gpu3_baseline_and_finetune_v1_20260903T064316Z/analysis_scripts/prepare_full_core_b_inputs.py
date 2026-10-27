"""Prepare the full eligible Core-B 1.0-second train/dev input contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

PREFIX_FRAMES = 120
HORIZON_STEPS = 30
ACTION_DIM = 28
ARM_DIM = 14
HAND_DIM = 14
LEG_DIM = 15
HISTORY_FRAME_INDICES = np.asarray([0, 17, 34, 51, 68, 85, 102, 119], dtype=np.int64)
STATE_DIM = ARM_DIM + HAND_DIM + LEG_DIM + ACTION_DIM
HORIZON_KEY = "1.0s"
TARGET_CACHE_NAME = "CORE_B_TARGET_CACHE_1p0s_TRAIN_DEV.npz"
DYNAMIC_FEATURE_ROOT = "features/g1/vjepa2/full-vjepa-b-opencv-20260826"
STATIC_FEATURE_ROOT = "features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def list_matrix(table: Any, name: str) -> np.ndarray:
    values = table[name].to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"missing values in {name}")
    return np.asarray(values, dtype=np.float32)


def feature_map(
    manifest_path: Path,
    episode_ids: set[int],
) -> tuple[dict[int, dict[str, object]], str]:
    manifest = pd.read_parquet(
        manifest_path,
        columns=[
            "episode_index",
            "feature_path",
            "feature_sha256",
            "pooled_shape",
            "token_shape",
        ],
    )
    selected = manifest[manifest["episode_index"].isin(episode_ids)].copy()
    if len(selected) != len(episode_ids) or selected["episode_index"].duplicated().any():
        raise ValueError(f"feature manifest coverage failed: {manifest_path}")
    result: dict[int, dict[str, object]] = {}
    for row in selected.itertuples(index=False):
        feature_path = Path(str(row.feature_path)).resolve()
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        pooled_shape = (
            json.loads(row.pooled_shape)
            if isinstance(row.pooled_shape, str)
            else list(row.pooled_shape)
        )
        token_shape = (
            json.loads(row.token_shape)
            if isinstance(row.token_shape, str)
            else list(row.token_shape)
        )
        if pooled_shape != [768] or token_shape != [9216, 768]:
            raise ValueError(f"feature shape contract failed: {row.episode_index}")
        result[int(row.episode_index)] = {
            "feature_path": str(feature_path),
            "feature_sha256": str(row.feature_sha256),
        }
    return result, sha256_file(manifest_path)


def load_pooled(path: Path, episode_index: int) -> np.ndarray:
    record = torch.load(path, map_location="cpu", weights_only=True)
    pooled = record.get("pooled") if isinstance(record, dict) else None
    if not torch.is_tensor(pooled) or tuple(pooled.shape) != (768,):
        raise ValueError(f"pooled feature shape failed: {episode_index}")
    pooled = pooled.float()
    if not torch.isfinite(pooled).all():
        raise ValueError(f"pooled feature is non-finite: {episode_index}")
    return pooled.numpy()


def state_history(data_root: Path, parquet_path: str, episode_index: int) -> np.ndarray:
    table = pq.read_table(
        data_root / parquet_path,
        columns=[
            "frame_index",
            "action",
            "observation.arm_joints",
            "observation.hand_joints",
            "observation.leg_joints",
        ],
    )
    frame = table["frame_index"].combine_chunks().to_numpy(
        zero_copy_only=False
    ).astype(np.int64)
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise ValueError(f"frame index contract failed: {episode_index}")
    if len(frame) <= int(HISTORY_FRAME_INDICES[-1]):
        raise ValueError(f"prefix history coverage failed: {episode_index}")
    action = list_matrix(table, "action")
    arm = list_matrix(table, "observation.arm_joints")
    hand = list_matrix(table, "observation.hand_joints")
    leg = list_matrix(table, "observation.leg_joints")
    if action.shape[1] != ACTION_DIM:
        raise ValueError(f"action dimension failed: {episode_index}")
    if arm.shape[1] != ARM_DIM or hand.shape[1] != HAND_DIM or leg.shape[1] != LEG_DIM:
        raise ValueError(f"state dimension failed: {episode_index}")
    history = np.concatenate(
        [
            arm[HISTORY_FRAME_INDICES],
            hand[HISTORY_FRAME_INDICES],
            leg[HISTORY_FRAME_INDICES],
            action[HISTORY_FRAME_INDICES],
        ],
        axis=1,
    )
    if history.shape != (len(HISTORY_FRAME_INDICES), STATE_DIM):
        raise ValueError(f"history shape failed: {episode_index}")
    if not np.isfinite(history).all():
        raise ValueError(f"history is non-finite: {episode_index}")
    return history.reshape(-1).astype(np.float32, copy=False)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "final_step_mae": float(np.abs(values[:, -1, :]).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    repo_root = args.repo_root.resolve()
    feasibility_root = args.feasibility_root.resolve()
    output_root = args.output_root.resolve()
    permitted_empty_dirs = {"figures", "figure_data", "figure_manifests"}
    existing = [
        path
        for path in output_root.iterdir()
        if path.name not in permitted_empty_dirs
        or (path.is_dir() and any(path.iterdir()))
    ] if output_root.exists() else []
    if existing:
        raise FileExistsError(f"refusing non-empty Core-B output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(exist_ok=True)
    (output_root / "figure_data").mkdir(exist_ok=True)
    (output_root / "figure_manifests").mkdir(exist_ok=True)
    data_root = project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    split_path = repo_root / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
    target_cache_path = feasibility_root / TARGET_CACHE_NAME
    target_norm_path = feasibility_root / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    dynamic_manifest_path = project_root / DYNAMIC_FEATURE_ROOT / "feature_manifest.parquet"
    static_manifest_path = project_root / STATIC_FEATURE_ROOT / "feature_manifest.parquet"
    split = pd.read_parquet(
        split_path,
        columns=[
            "episode_index",
            "split",
            "v3_benchmark_inclusion",
            "original_task_index",
            "parquet_path",
            "video_path",
            "video_sha256_fingerprint",
        ],
    )
    split = split[
        split["split"].isin(["train", "validation"])
        & split["v3_benchmark_inclusion"].eq("primary")
    ].copy()
    with np.load(target_cache_path, allow_pickle=False) as archive:
        target_data = {name: archive[name].copy() for name in archive.files}
    if target_data["targets"].shape[1:] != (HORIZON_STEPS, ACTION_DIM):
        raise ValueError("1.0-second target shape contract failed")
    if not np.all(np.isin(target_data["split_code"], [0, 1])):
        raise ValueError("target cache contains unexpected split code")
    if len(np.unique(target_data["episode_index"])) != len(target_data["episode_index"]):
        raise ValueError("target cache contains duplicate episode IDs")
    order = np.lexsort(
        (target_data["episode_index"], target_data["split_code"])
    )
    ids = target_data["episode_index"][order].astype(np.int64)
    split_code = target_data["split_code"][order].astype(np.int8)
    tasks = target_data["original_task_index"][order].astype(np.int64)
    targets = target_data["targets"][order].astype(np.float32)
    last_action = target_data["last_observed_action"][order].astype(np.float32)
    if not np.isfinite(targets).all() or not np.isfinite(last_action).all():
        raise ValueError("target cache contains non-finite values")
    train_mask = split_code == 0
    dev_mask = split_code == 1
    train_count = int(train_mask.sum())
    dev_count = int(dev_mask.sum())
    if train_count != 2741 or dev_count != 614:
        raise ValueError(f"unexpected eligible full population: {train_count}/{dev_count}")
    split_map = {
        int(row.episode_index): row
        for row in split.itertuples(index=False)
        if int(row.episode_index) in set(ids.tolist())
    }
    if len(split_map) != len(ids):
        raise ValueError("v3 split does not cover all eligible target IDs")
    ids_set = set(ids.tolist())
    dynamic_map, dynamic_hash = feature_map(dynamic_manifest_path, ids_set)
    static_map, static_hash = feature_map(static_manifest_path, ids_set)
    dynamic_pooled = np.asarray(
        [load_pooled(Path(dynamic_map[int(item)]["feature_path"]), int(item)) for item in ids],
        dtype=np.float32,
    )
    static_pooled = np.asarray(
        [load_pooled(Path(static_map[int(item)]["feature_path"]), int(item)) for item in ids],
        dtype=np.float32,
    )
    state_features = np.asarray(
        [
            state_history(
                data_root,
                str(split_map[int(item)].parquet_path),
                int(item),
            )
            for item in ids
        ],
        dtype=np.float32,
    )
    if state_features.shape != (len(ids), len(HISTORY_FRAME_INDICES) * STATE_DIM):
        raise ValueError("state/action history shape contract failed")
    input_arrays = {
        "dynamic_pooled": dynamic_pooled,
        "static_pooled": static_pooled,
        "state_action_history": state_features,
    }
    input_norm: dict[str, dict[str, object]] = {}
    normalized_inputs: dict[str, np.ndarray] = {}
    for name, values in input_arrays.items():
        mean = values[train_mask].mean(axis=0).astype(np.float32)
        std = values[train_mask].std(axis=0).astype(np.float32)
        safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        normalized = ((values - mean) / safe_std).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError(f"normalized input is non-finite: {name}")
        input_norm[name] = {
            "dimension": int(values.shape[1]),
            "mean": mean.tolist(),
            "safe_std": safe_std.tolist(),
            "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
            "safe_std_sha256": hashlib.sha256(safe_std.tobytes()).hexdigest(),
            "source": "eligible train targets/inputs only",
        }
        normalized_inputs[name] = normalized
    target_norm_all = json.loads(target_norm_path.read_text(encoding="utf-8"))
    target_norm = target_norm_all[HORIZON_KEY]
    target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
    target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
    if target_mean.shape != (HORIZON_STEPS, ACTION_DIM) or target_std.shape != (
        HORIZON_STEPS,
        ACTION_DIM,
    ):
        raise ValueError("target normalization shape failed")
    standardized_targets = ((targets - target_mean) / target_std).astype(np.float32)
    if not np.isfinite(standardized_targets).all():
        raise ValueError("standardized targets are non-finite")
    np.savez_compressed(
        output_root / "CORE_B_FULL_INPUTS_1p0s.npz",
        episode_index=ids,
        split_code=split_code,
        original_task_index=tasks,
        targets=targets,
        targets_standardized=standardized_targets,
        last_observed_action=last_action,
        dynamic_pooled=normalized_inputs["dynamic_pooled"],
        static_pooled=normalized_inputs["static_pooled"],
        state_action_history=normalized_inputs["state_action_history"],
    )
    token_paths = {
        "dynamic": {
            str(item): dynamic_map[int(item)]["feature_path"] for item in ids
        },
        "static": {
            str(item): static_map[int(item)]["feature_path"] for item in ids
        },
    }
    write_json(output_root / "CORE_B_TOKEN_PATHS_1p0s.json", token_paths)
    write_json(output_root / "CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json", input_norm)
    manifest_rows = []
    for index, item in enumerate(ids.tolist()):
        split_name = "train" if split_code[index] == 0 else "validation"
        source = split_map[int(item)]
        manifest_rows.append(
            {
                "episode_index": int(item),
                "split": split_name,
                "original_task_index": int(tasks[index]),
                "parquet_path": str(source.parquet_path),
                "video_path": str(source.video_path),
                "video_sha256_fingerprint": str(source.video_sha256_fingerprint),
                "dynamic_feature_path": dynamic_map[int(item)]["feature_path"],
                "static_feature_path": static_map[int(item)]["feature_path"],
                "target_horizon": HORIZON_KEY,
                "target_steps": HORIZON_STEPS,
                "target_dimensions": ACTION_DIM,
                "final_test_accessed": 0,
            }
        )
    with (output_root / "CORE_B_FULL_DATA_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_json(
        output_root / "CORE_B_FULL_DATA_MANIFEST.json",
        {
            "status": "pass",
            "train_count": train_count,
            "development_validation_count": dev_count,
            "total_eligible_count": len(ids),
            "episode_index_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "split_code_sha256": hashlib.sha256(split_code.tobytes()).hexdigest(),
            "target_cache": str(target_cache_path),
            "target_cache_sha256": sha256_file(target_cache_path),
            "target_normalization": str(target_norm_path),
            "target_normalization_sha256": sha256_file(target_norm_path),
            "dynamic_feature_manifest": str(dynamic_manifest_path),
            "dynamic_feature_manifest_sha256": dynamic_hash,
            "static_feature_manifest": str(static_manifest_path),
            "static_feature_manifest_sha256": static_hash,
            "prefix_seconds": 4.0,
            "prefix_frame_count": PREFIX_FRAMES,
            "primary_horizon_seconds": 1.0,
            "primary_horizon_steps": HORIZON_STEPS,
            "history_frame_indices": HISTORY_FRAME_INDICES.tolist(),
            "state_action_history_dimension": int(state_features.shape[1]),
            "target_dimension": ACTION_DIM,
            "arm_dimensions": ARM_DIM,
            "hand_dimensions": HAND_DIM,
            "target_order": "hand[0:14] then arm[14:28]",
            "physical_unit_status": "unresolved",
            "normalization_source": "train eligible rows only",
            "final_test_target_accessed": 0,
        },
    )
    dev_targets = targets[dev_mask]
    train_targets = targets[train_mask]
    global_mean = np.broadcast_to(
        train_targets.reshape(-1, ACTION_DIM).mean(axis=0),
        dev_targets.shape,
    ).copy()
    time_mean = np.broadcast_to(
        train_targets.mean(axis=0, keepdims=True), dev_targets.shape
    ).copy()
    persistence = np.repeat(last_action[dev_mask, None, :], HORIZON_STEPS, axis=1)
    naive_rows = []
    for method, prediction in (
        ("zero_action", np.zeros_like(dev_targets)),
        ("train_global_mean", global_mean),
        ("train_per_time_step_mean", time_mean),
        ("last_observed_action_persistence", persistence),
    ):
        error_native = prediction - dev_targets
        error_standardized = (prediction - target_mean) / target_std - (
            dev_targets - target_mean
        ) / target_std
        naive_rows.append(
            {
                "method": method,
                "horizon": HORIZON_KEY,
                "episodes": dev_count,
                "standardized_mae": float(np.abs(error_standardized).mean()),
                "standardized_rmse": float(np.sqrt(np.mean(error_standardized**2))),
                "native_command_value_mae": float(np.abs(error_native).mean()),
                "arm_standardized_mae": float(np.abs(error_standardized[:, :, 14:28]).mean()),
                "hand_standardized_mae": float(np.abs(error_standardized[:, :, :14]).mean()),
                "final_step_standardized_mae": float(
                    np.abs(error_standardized[:, -1, :]).mean()
                ),
                "final_test_accessed": 0,
            }
        )
    with (output_root / "CORE_B_NAIVE_BASELINES.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(naive_rows[0]))
        writer.writeheader()
        writer.writerows(naive_rows)
    write_json(
        output_root / "CORE_B_NAIVE_BASELINES.json",
        {"status": "pass", "rows": naive_rows, "final_test_accessed": 0},
    )
    target_contract = {
        "status": "locked_for_native_unit_engineering_run",
        "robot": "single_robot_G1",
        "prefix_seconds": 4.0,
        "prefix_frame_count": PREFIX_FRAMES,
        "primary_horizon": {"seconds": 1.0, "steps": HORIZON_STEPS},
        "target_shape": [HORIZON_STEPS, ACTION_DIM],
        "target_order": "hand[0:14] then arm[14:28]",
        "primary_trajectory_subset": "arm[14:28]",
        "secondary_trajectory_subset": "hand[0:14]",
        "physical_unit_status": "unresolved",
        "normalization": "train-only per-dimension mean/std",
        "padding": "none in eligible population",
        "final_test_target_accessed": 0,
    }
    write_json(output_root / "CORE_B_TARGET_AND_METRIC_CONTRACT.json", target_contract)
    (output_root / "CORE_B_TARGET_AND_METRIC_CONTRACT.md").write_text(
        "# Core B target and metric contract\n\n"
        "This GPU3 lane uses the single-robot G1 eligible train/dev population "
        "at the 1.0-second horizon. Targets are 30 future steps by 28 "
        "source-native command values, with hand[0:14] followed by arm[14:28]. "
        "Primary error is standardized MAE on arm dimensions; full-28 and hand "
        "errors are secondary. Native command-value error is reported without "
        "claiming radians or degrees. Normalization is train-only and final-test "
        "targets are not accessed.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "train_count": train_count,
                "dev_count": dev_count,
                "total": len(ids),
                "state_dimension": int(state_features.shape[1]),
                "final_test_accessed": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
