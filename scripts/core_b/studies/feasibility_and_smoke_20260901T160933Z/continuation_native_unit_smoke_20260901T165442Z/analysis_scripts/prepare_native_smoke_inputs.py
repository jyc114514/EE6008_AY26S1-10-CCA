"""Prepare compact, train/dev-only input caches for the bounded native-space smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HORIZONS = {"1p0s": 30, "0p5s": 15}
TRAIN_COUNT = 256
DEV_COUNT = 128
HISTORY_FRAME_INDICES = np.asarray([0, 17, 34, 51, 68, 85, 102, 119], dtype=np.int64)
STATE_DIM = 14 + 14 + 15 + 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def list_matrix(table, name: str) -> np.ndarray:
    values = table[name].to_pylist()
    if any(value is None for value in values):
        raise ValueError(f"missing values in {name}")
    return np.asarray(values, dtype=np.float32)


def choose_rows(cache: dict[str, np.ndarray], count: int, split_code: int) -> np.ndarray:
    mask = cache["split_code"] == split_code
    positions = np.flatnonzero(mask)
    order = np.lexsort((cache["episode_index"][positions], cache["original_task_index"][positions]))
    positions = positions[order]
    chosen = positions[np.linspace(0, len(positions) - 1, count, dtype=np.int64)]
    if len(np.unique(chosen)) != count:
        raise ValueError("deterministic smoke sample contains duplicate positions")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    feature_manifest_path = args.project_root / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet"
    split_path = args.repo_root / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
    target_cache_1 = args.feasibility_root / "CORE_B_TARGET_CACHE_1p0s_TRAIN_DEV.npz"
    target_cache_05 = args.feasibility_root / "CORE_B_TARGET_CACHE_0p5s_TRAIN_DEV.npz"
    norm_path = args.feasibility_root / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    split = pd.read_parquet(split_path, columns=["episode_index", "split", "v3_benchmark_inclusion", "original_task_index", "parquet_path"])
    split = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")].copy()
    cache_1 = {name: value for name, value in np.load(target_cache_1, allow_pickle=False).items()}
    cache_05 = {name: value for name, value in np.load(target_cache_05, allow_pickle=False).items()}
    train_positions = choose_rows(cache_1, TRAIN_COUNT, 0)
    dev_positions = choose_rows(cache_1, DEV_COUNT, 1)
    selected_positions = np.concatenate([train_positions, dev_positions])
    selected_ids = cache_1["episode_index"][selected_positions].astype(np.int64)
    selected_split = cache_1["split_code"][selected_positions].astype(np.int8)
    selected_tasks = cache_1["original_task_index"][selected_positions].astype(np.int64)
    if len(np.unique(selected_ids)) != TRAIN_COUNT + DEV_COUNT or np.any(selected_split[:TRAIN_COUNT] != 0) or np.any(selected_split[TRAIN_COUNT:] != 1):
        raise ValueError("smoke sample split/ID contract failed")
    target_positions = {}
    for horizon, cache in (("1p0s", cache_1), ("0p5s", cache_05)):
        lookup = {int(value): index for index, value in enumerate(cache["episode_index"].tolist())}
        target_positions[horizon] = np.asarray([lookup[int(value)] for value in selected_ids], dtype=np.int64)

    feature_manifest = pd.read_parquet(feature_manifest_path, columns=["episode_index", "feature_path", "feature_sha256", "pooled_shape"])
    feature_map = {int(row.episode_index): row for row in feature_manifest.itertuples(index=False) if int(row.episode_index) in set(selected_ids.tolist())}
    if len(feature_map) != len(selected_ids):
        raise ValueError("prefix feature manifest does not cover smoke IDs")
    video_features = []
    for episode_index in selected_ids.tolist():
        row = feature_map[int(episode_index)]
        record = torch.load(Path(row.feature_path), map_location="cpu", weights_only=True)
        pooled = record.get("pooled") if isinstance(record, dict) else None
        if pooled is None or tuple(pooled.shape) != (768,) or not torch.isfinite(pooled.float()).all():
            raise ValueError(f"video pooled feature contract failed for episode {episode_index}")
        video_features.append(pooled.float().numpy())
    video_features = np.asarray(video_features, dtype=np.float32)

    state_features = []
    for episode_index in selected_ids.tolist():
        row = split[split["episode_index"].eq(int(episode_index))].iloc[0]
        table = pq.read_table(data_root / str(row["parquet_path"]), columns=["frame_index", "action", "observation.arm_joints", "observation.hand_joints", "observation.leg_joints"])
        frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)) or len(frame) < 120:
            raise ValueError(f"state history frame contract failed for episode {episode_index}")
        action = list_matrix(table, "action")
        arm = list_matrix(table, "observation.arm_joints")
        hand = list_matrix(table, "observation.hand_joints")
        leg = list_matrix(table, "observation.leg_joints")
        if action.shape[1] != 28 or arm.shape[1] != 14 or hand.shape[1] != 14 or leg.shape[1] != 15:
            raise ValueError(f"state/action dimensions failed for episode {episode_index}")
        history = np.concatenate([arm[HISTORY_FRAME_INDICES], hand[HISTORY_FRAME_INDICES], leg[HISTORY_FRAME_INDICES], action[HISTORY_FRAME_INDICES]], axis=1)
        if history.shape != (len(HISTORY_FRAME_INDICES), STATE_DIM) or not np.isfinite(history).all():
            raise ValueError(f"state history finite/shape contract failed for episode {episode_index}")
        state_features.append(history.reshape(-1))
    state_features = np.asarray(state_features, dtype=np.float32)

    input_normalization = {}
    for name, values in (("video_pooled", video_features), ("state_action_history", state_features)):
        train_values = values[:TRAIN_COUNT]
        mean = train_values.mean(axis=0).astype(np.float32)
        std = train_values.std(axis=0).astype(np.float32)
        safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        input_normalization[name] = {"dimension": int(values.shape[1]), "mean": mean.tolist(), "safe_std": safe_std.tolist(), "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(), "safe_std_sha256": hashlib.sha256(safe_std.tobytes()).hexdigest(), "source": "smoke train subset only"}
    write_json(root / "NATIVE_SMOKE_INPUT_NORMALIZATION.json", input_normalization)

    for horizon in HORIZONS:
        cache = cache_1 if horizon == "1p0s" else cache_05
        positions = target_positions[horizon]
        targets = cache["targets"][positions].astype(np.float32)
        output = root / f"NATIVE_SMOKE_INPUTS_{horizon}.npz"
        np.savez_compressed(output, episode_index=selected_ids, split_code=selected_split, original_task_index=selected_tasks, targets=targets, video_pooled=video_features, state_action_history=state_features)

    sample_bytes = np.concatenate([selected_ids.view(np.uint8), selected_split.view(np.uint8), selected_tasks.view(np.uint8)])
    manifest = {"status": "pass", "train_count": TRAIN_COUNT, "dev_count": DEV_COUNT, "total_count": TRAIN_COUNT + DEV_COUNT, "sample_ids_sha256": hashlib.sha256(sample_bytes.tobytes()).hexdigest(), "episode_ids": selected_ids.tolist(), "split_code": selected_split.tolist(), "task_indices": selected_tasks.tolist(), "history_frame_indices": HISTORY_FRAME_INDICES.tolist(), "state_action_history_dimension": STATE_DIM * len(HISTORY_FRAME_INDICES), "video_pooled_dimension": 768, "target_dimensions": 28, "target_cache_source": {"1p0s": str(target_cache_1), "0p5s": str(target_cache_05)}, "target_cache_sha256": {"1p0s": sha256_file(target_cache_1), "0p5s": sha256_file(target_cache_05)}, "target_normalization_source": str(norm_path), "target_normalization_sha256": sha256_file(norm_path), "feature_manifest": str(feature_manifest_path), "feature_manifest_sha256": sha256_file(feature_manifest_path), "task_labels_used_as_model_input": False, "final_test_accessed": 0}
    write_json(root / "NATIVE_SMOKE_SAMPLE_MANIFEST.json", manifest)
    print(json.dumps({"status": "pass", "train_count": TRAIN_COUNT, "dev_count": DEV_COUNT, "feature_count": len(video_features), "final_test_accessed": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
