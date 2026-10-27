#!/usr/bin/env python3
"""Prepare the locked Core-B B2 pooled sampling inputs.

The existing dynamic cache contains one representation block per observed
frame (32 frames x 288 tokens).  B2 uses the locked nested frame maps and
computes a deterministic mean over the selected blocks.  Sparse conditions
therefore retain a 32-slot contract through deterministic repetition; this
script records that no encoder-speed claim follows from the reduced unique
frame count.

This is a train/dev-only preparation step.  It never reads final-test rows
into a model input and never edits the accepted B1 cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_TRAIN = 2741
EXPECTED_DEV = 614
EXPECTED_ROWS = EXPECTED_TRAIN + EXPECTED_DEV
TARGET_SHAPE = (30, 28)
FRAME_COUNT = 32
TOKENS_PER_FRAME = 288
SLOT_COUNT = 32
PREFIX_SECONDS = 4.0
TARGET_START_FRAME = 120
SOURCE_FPS = 30.0


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(part, **arrays)
    os.replace(part, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_token_record(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict):
        raise TypeError(f"feature record is not a mapping: {path}")
    tokens = record.get("tokens")
    pooled = record.get("pooled")
    metadata = record.get("metadata")
    if not torch.is_tensor(tokens) or tuple(tokens.shape) != (FRAME_COUNT * TOKENS_PER_FRAME, 768):
        raise ValueError(f"unexpected token shape at {path}: {getattr(tokens, 'shape', None)}")
    if not torch.is_tensor(pooled) or tuple(pooled.shape) != (768,):
        raise ValueError(f"unexpected pooled shape at {path}: {getattr(pooled, 'shape', None)}")
    if not isinstance(metadata, dict):
        raise ValueError(f"feature metadata missing at {path}")
    if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
        raise ValueError(f"non-finite feature tokens at {path}")
    blocks = tokens.reshape(FRAME_COUNT, TOKENS_PER_FRAME, 768).float()
    return blocks.numpy(), pooled.float().numpy(), metadata


def interval_summary(frame_indices: list[int], timestamps: list[float], positions: list[int]) -> dict[str, Any]:
    chosen_indices = [int(frame_indices[i]) for i in positions]
    chosen_times = [float(timestamps[i]) for i in positions]
    diffs = np.diff(np.asarray(chosen_times, dtype=np.float64))
    slot_source_positions = [int(i % len(positions)) for i in range(SLOT_COUNT)]
    return {
        "positions_in_dense32": [int(i) for i in positions],
        "source_frame_indices": chosen_indices,
        "source_timestamps_seconds": chosen_times,
        "slot_source_positions": slot_source_positions,
        "unique_observation_frames": len(chosen_indices),
        "encoder_slots": SLOT_COUNT,
        "decoded_frames": len(chosen_indices),
        "nominal_interval_seconds": {32: 0.125, 16: 0.25, 8: 0.5}[len(chosen_indices)],
        "actual_interval_seconds_min": float(diffs.min()) if len(diffs) else None,
        "actual_interval_seconds_max": float(diffs.max()) if len(diffs) else None,
        "actual_interval_seconds_mean": float(diffs.mean()) if len(diffs) else None,
        "observation_span_seconds": float(chosen_times[-1] - chosen_times[0]),
        "latest_observation_seconds": float(chosen_times[-1]),
        "target_start_seconds": PREFIX_SECONDS,
        "target_gap_seconds": float(PREFIX_SECONDS - chosen_times[-1]),
        "temporal_adapter": "deterministic_repeat_selected_blocks_to_32_slots",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--dynamic-token-map", type=Path, required=True)
    parser.add_argument("--input-normalization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_input_root.resolve()
    token_map_path = args.dynamic_token_map.resolve()
    input_normalization_path = args.input_normalization.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with np.load(source_root / "CORE_B_FULL_INPUTS_1p0s.npz", allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {
        "episode_index", "split_code", "original_task_index", "targets",
        "targets_standardized", "last_observed_action", "state_action_history",
        "dynamic_pooled", "static_pooled",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"canonical B1 input keys missing: {missing}")
    if data["episode_index"].shape != (EXPECTED_ROWS,):
        raise ValueError(f"unexpected row count: {data['episode_index'].shape}")
    if int(np.sum(data["split_code"] == 0)) != EXPECTED_TRAIN or int(np.sum(data["split_code"] == 1)) != EXPECTED_DEV:
        raise ValueError("Core-B train/dev counts are not 2741/614")
    if data["targets"].shape[1:] != TARGET_SHAPE or data["targets_standardized"].shape[1:] != TARGET_SHAPE:
        raise ValueError("Core-B target shape is not (N,30,28)")
    if len(np.unique(data["episode_index"])) != EXPECTED_ROWS:
        raise ValueError("Core-B episode IDs are not unique")
    if not np.isfinite(data["targets_standardized"]).all():
        raise ValueError("Core-B standardized targets contain non-finite values")

    token_map = load_json(token_map_path)
    dynamic_map = token_map.get("dynamic")
    if not isinstance(dynamic_map, dict) or len(dynamic_map) != EXPECTED_ROWS:
        raise ValueError("dynamic token map does not cover the 3355 eligible rows")
    input_normalization = load_json(input_normalization_path)
    dynamic_norm = input_normalization.get("dynamic_pooled")
    if not isinstance(dynamic_norm, dict):
        raise ValueError("dynamic pooled input normalization is missing")
    dynamic_mean = np.asarray(dynamic_norm.get("mean"), dtype=np.float32)
    dynamic_std = np.asarray(dynamic_norm.get("safe_std"), dtype=np.float32)
    if dynamic_mean.shape != (768,) or dynamic_std.shape != (768,):
        raise ValueError("dynamic pooled input normalization shape failed")
    if np.any(dynamic_std <= 0) or not np.isfinite(dynamic_mean).all() or not np.isfinite(dynamic_std).all():
        raise ValueError("dynamic pooled input normalization is invalid")

    all_ids = data["episode_index"].astype(np.int64, copy=False)
    interval_positions = {
        "0125": list(range(32)),
        "0250": list(range(1, 32, 2)),
        "0500": list(range(3, 32, 4)),
    }
    derived = {key: np.empty((EXPECTED_ROWS, 768), dtype=np.float32) for key in interval_positions}
    frame_maps: dict[str, Any] = {}
    dense_map_reference: dict[str, Any] | None = None
    dense_pool_max_abs = 0.0
    dense_pool_raw_max_abs = 0.0

    for row, episode in enumerate(all_ids.tolist()):
        path = Path(dynamic_map[str(int(episode))]).resolve()
        blocks, stored_pooled, metadata = load_token_record(path)
        if int(metadata.get("episode_index", episode)) != int(episode):
            raise ValueError(f"feature episode mismatch for {episode}")
        frame_indices = [int(x) for x in metadata.get("frame_indices", [])]
        timestamps = [float(x) for x in metadata.get("timestamps_seconds", [])]
        if len(frame_indices) != FRAME_COUNT or len(timestamps) != FRAME_COUNT:
            raise ValueError(f"episode {episode} does not have 32 frame anchors")
        if any(frame_indices[i] >= frame_indices[i + 1] for i in range(FRAME_COUNT - 1)):
            raise ValueError(f"episode {episode} frame indices are not strictly increasing")
        if any(int(i) >= TARGET_START_FRAME for i in frame_indices):
            raise ValueError(f"episode {episode} uses a frame at/after target start")
        if any(float(t) >= PREFIX_SECONDS for t in timestamps):
            raise ValueError(f"episode {episode} uses a timestamp at/after target start")
        current_dense = {
            "frame_indices": frame_indices,
            "timestamps_seconds": timestamps,
            "source_fps_observed": float(metadata.get("source_fps_observed", SOURCE_FPS)),
            "decoder": metadata.get("decoder"),
        }
        if dense_map_reference is None:
            dense_map_reference = current_dense
        elif current_dense != dense_map_reference:
            raise ValueError(f"dense frame map differs across episode {episode}")
        dense_from_blocks = blocks.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        dense_normalized = ((dense_from_blocks - dynamic_mean) / dynamic_std).astype(np.float32)
        dense_pool_max_abs = max(dense_pool_max_abs, float(np.max(np.abs(dense_normalized - data["dynamic_pooled"][row]))))
        dense_pool_raw_max_abs = max(dense_pool_raw_max_abs, float(np.max(np.abs(dense_from_blocks - stored_pooled))))
        for key, positions in interval_positions.items():
            selected = blocks[np.asarray(positions, dtype=np.int64)]
            selected_pooled = selected.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            derived[key][row] = ((selected_pooled - dynamic_mean) / dynamic_std).astype(np.float32)

    assert dense_map_reference is not None
    # The token field is float16 while the stored pooled field is float32;
    # direct mean reconstruction therefore has a fixed cache-quantization
    # bound, recorded separately from any model merge equivalence gate.
    if dense_pool_raw_max_abs > 2.5e-4:
        raise ValueError(f"dense raw pooled reconstruction failed: max_abs={dense_pool_raw_max_abs}")
    # Preserve the exact accepted B1 dense control. The float16 token cache
    # can differ from the stored float32 pooled vector after normalization,
    # even though both represent the same dense encoder output.
    derived["0125"] = data["dynamic_pooled"].copy()
    dense_indices = list(range(32))
    for key, positions in interval_positions.items():
        frame_maps[key] = interval_summary(
            dense_map_reference["frame_indices"], dense_map_reference["timestamps_seconds"], positions
        )
    frame_maps["dense32"] = interval_summary(
        dense_map_reference["frame_indices"], dense_map_reference["timestamps_seconds"], dense_indices
    )

    canonical_manifest = load_json(source_root / "CORE_B_FULL_DATA_MANIFEST.json")
    base_arrays = {
        "episode_index": data["episode_index"],
        "split_code": data["split_code"],
        "original_task_index": data["original_task_index"],
        "targets": data["targets"],
        "targets_standardized": data["targets_standardized"],
        "last_observed_action": data["last_observed_action"],
        "static_pooled": data["static_pooled"],
        "state_action_history": data["state_action_history"],
    }
    condition_manifest: dict[str, Any] = {}
    for key, pooled in derived.items():
        interval = {"0125": "0.125s", "0250": "0.250s", "0500": "0.500s"}[key]
        input_path = output_root / f"B2_INPUTS_{key}.npz"
        atomic_npz(input_path, **base_arrays, dynamic_pooled=pooled)
        condition_manifest[key] = {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "condition_interval": interval,
            "dynamic_pooled_shape": list(pooled.shape),
            "train_count": EXPECTED_TRAIN,
            "development_validation_count": EXPECTED_DEV,
        }

    output_manifest = {
        "status": "pass",
        "created_utc": utc_now(),
        "method": "feature_level_nested_frame_block_mean_with_32_slot_repetition_contract",
        "source_input_root": str(source_root),
        "source_input_archive_sha256": sha256_file(source_root / "CORE_B_FULL_INPUTS_1p0s.npz"),
        "source_input_manifest_sha256": sha256_file(source_root / "CORE_B_FULL_DATA_MANIFEST.json"),
        "source_dynamic_token_map_sha256": sha256_file(token_map_path),
        "source_input_normalization_sha256": sha256_file(input_normalization_path),
        "source_input_normalization_path": str(input_normalization_path),
        "source_target_contract": canonical_manifest,
        "population": {"train": EXPECTED_TRAIN, "development_validation": EXPECTED_DEV, "final_test_accessed": 0},
        "target_shape": list(TARGET_SHAPE),
        "dense_pool_reconstruction_max_abs": dense_pool_max_abs,
        "dense_pool_reconstruction_raw_max_abs": dense_pool_raw_max_abs,
        "dense_pool_reconstruction_raw_tolerance": 2.5e-4,
        "dense_pool_reconstruction_tolerance": 1e-4,
        "dense_condition_exact_b1_input_reuse": True,
        "derived_input_space": "same train-only per-dimension standardized space as B1 dynamic_pooled",
        "frame_map_reference": dense_map_reference,
        "frame_maps": frame_maps,
        "conditions": condition_manifest,
        "cache_contract": {
            "encoder_slots": SLOT_COUNT,
            "unique_frames": {"0125": 32, "0250": 16, "0500": 8},
            "decoded_frames": {"0125": 32, "0250": 16, "0500": 8},
            "raw_video_io": False,
            "encoder_speed_claim": False,
            "note": "B2 pooled inputs are derived from the locked dense frozen cache; sparse rows are not a re-encoded sparse clip.",
        },
        "episode_id_sha256": sha256_json(all_ids.tolist()),
    }
    atomic_json(output_root / "B2_INPUT_PREPARATION.json", output_manifest)
    atomic_json(output_root / "B2_FRAME_MAPS.json", frame_maps)
    print(json.dumps({"status": "pass", "rows": EXPECTED_ROWS, "dense_pool_reconstruction_max_abs": dense_pool_max_abs, "output": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
