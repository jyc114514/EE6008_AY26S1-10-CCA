#!/usr/bin/env python3
"""Prepare the corrected pixel-level B2 sparse feature inputs.

The previous B2 sparse proxy selected blocks from final contextual tokens.  A
formal corrected condition decodes the same dense 32-frame pixel sample,
selects nested pixel frames, repeats only those pixels to the fixed 32 encoder
slots, and runs the full V-JEPA encoder again before pooling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPECTED_ROWS = 3355
TRAIN_COUNT = 2741
DEV_COUNT = 614
FRAME_COUNT = 32
TARGET_SHAPE = (30, 28)
PIXEL_POSITIONS = {
    "0125": list(range(32)),
    "0250": list(range(1, 32, 2)),
    "0500": list(range(3, 32, 4)),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(part, **arrays)
    os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(value, part)
    os.replace(part, path)


def load_manifest(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {int(row["episode_index"]): row for row in rows}
    if len(result) != EXPECTED_ROWS:
        raise ValueError(f"B2 manifest coverage failed: {len(result)}")
    return result


def source_map(expected: list[int], observed: np.ndarray) -> dict[str, Any]:
    if observed.tolist() != expected:
        raise ValueError(f"dense pixel frame map mismatch: {observed.tolist()}")
    return {"frame_indices": [int(x) for x in observed.tolist()], "timestamps_seconds": (observed.astype(np.float64) / 30.0).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--canonical-input-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    repo_root = args.repo_root.resolve()
    canonical_root = args.canonical_input_root.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(args.manifest_csv.resolve())
    with np.load(canonical_root / "CORE_B_FULL_INPUTS_1p0s.npz", allow_pickle=False) as archive:
        base = {key: archive[key].copy() for key in archive.files}
    if base["episode_index"].shape != (EXPECTED_ROWS,) or base["targets"].shape[1:] != TARGET_SHAPE:
        raise ValueError("canonical Core-B population/target shape failed")
    if int(np.sum(base["split_code"] == 0)) != TRAIN_COUNT or int(np.sum(base["split_code"] == 1)) != DEV_COUNT:
        raise ValueError("canonical Core-B split count failed")
    dense_indices = [0, 4, 8, 11, 15, 19, 22, 26, 30, 34, 38, 41, 45, 49, 52, 56, 60, 64, 68, 71, 75, 79, 82, 86, 90, 94, 98, 101, 105, 109, 112, 116]
    pixel_maps = {
        key: {"dense_positions": positions, "unique_frames": len(positions), "encoder_slots": 32, "slot_source_positions": [int(i % len(positions)) for i in range(32)]}
        for key, positions in PIXEL_POSITIONS.items()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "PIXEL_SAMPLING_PROTOCOL.json", {"status":"locked", "created_utc":utc_now(), "dense_frame_indices":dense_indices, "conditions":pixel_maps, "method":"select_pixels_then_repeat_to_32_slots_then_encode_then_pool", "tubelet_size":2, "patch_size":16, "encoder_output_contract":"(T/2)*(H/16)*(W/16)=9216 final contextual tokens for 32 slots", "final_test_access":0})
    if args.dry_run:
        print(json.dumps({"status":"pass","rows":EXPECTED_ROWS,"conditions":pixel_maps,"dry_run":True},sort_keys=True)); return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu" and not args.allow_cpu:
        raise PermissionError("CPU pixel extraction requires --allow-cpu")
    config_path = repo_root / "configs/core_a_benchmark_v3_20260826.yaml"
    from ee6008.config import load_config
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.models import encode_clip, load_vjepa2_1_base
    config = load_config(config_path)
    model, model_info = load_vjepa2_1_base(source_root=config.vjepa_source_root, checkpoint=config.checkpoint, device=device, num_frames=32, resolution=384)
    condition_features = {key: np.empty((EXPECTED_ROWS, 768), dtype=np.float32) for key in ("0250", "0500")}
    cache_reports: list[dict[str, Any]] = []
    cache_root = output_root / "pixel_feature_cache"
    started = time.monotonic()
    for position, episode in enumerate(base["episode_index"].tolist(), 1):
        row = manifest[int(episode)]
        sample = decode_prefix(config.g1_root / row["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        source_map(dense_indices, sample.frame_indices)
        frames = sample.frames
        for key in ("0250", "0500"):
            positions = PIXEL_POSITIONS[key]
            selected = frames[np.asarray(positions, dtype=np.int64)]
            repeated = selected[np.asarray([i % len(positions) for i in range(32)], dtype=np.int64)]
            clip = preprocess_vjepa(repeated, resolution=384)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _tokens, pooled = encode_clip(model, clip.to(device, non_blocking=True))
            condition_features[key][position - 1] = pooled.float().cpu().numpy()
            entry = cache_root / key / "entries" / f"episode_{int(episode):06d}.pt"
            atomic_torch(entry, {"config_hash": hashlib.sha256(json.dumps({"condition":key,"method":"pixel_select_repeat_encode_pool","checkpoint_sha256":sha256_file(config.checkpoint)},sort_keys=True).encode()).hexdigest(), "episode_index":int(episode), "pooled":pooled.float().cpu(), "tokens":torch.empty((0,),dtype=torch.float16), "metadata": {"split":row["split"],"video_path":row["video_path"],"dense_frame_indices":dense_indices,"unique_frame_indices":[dense_indices[i] for i in positions],"unique_timestamps_seconds":[dense_indices[i]/30.0 for i in positions],"slot_source_positions":[int(i % len(positions)) for i in range(32)],"unique_observation_frames":len(positions),"encoder_slots":32,"pixel_level_sampling":True,"encoder_output":"final_contextual_tokens_then_mean_pool","tubelet_size":2,"patch_size":16,"decoder":sample.decoder}})
        if position == 1 or position % 100 == 0 or position == EXPECTED_ROWS:
            print(f"PIXEL_B2_PROGRESS {position}/{EXPECTED_ROWS}", flush=True)
    input_reports: dict[str, Any] = {}
    for key, features in condition_features.items():
        train_mask = base["split_code"] == 0
        mean = features[train_mask].mean(axis=0).astype(np.float32)
        std = features[train_mask].std(axis=0).astype(np.float32)
        safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        normalized = ((features - mean) / safe_std).astype(np.float32)
        input_name = f"TRUE_B2_INPUTS_{key}.npz"
        arrays = {name: base[name] for name in ("episode_index","split_code","original_task_index","targets","targets_standardized","last_observed_action","static_pooled","state_action_history")}
        arrays["dynamic_pooled"] = normalized
        atomic_npz(output_root / input_name, **arrays)
        atomic_json(output_root / f"TRUE_B2_INPUT_NORMALIZATION_{key}.json", {"dynamic_pooled":{"dimension":768,"mean":mean.tolist(),"safe_std":safe_std.tolist(),"fit_population":"train rows only","source":"pixel-level re-encoded feature cache"},"condition":key,"final_test_access":0})
        input_reports[key] = {"input_path":str(output_root/input_name),"input_sha256":sha256_file(output_root/input_name),"normalization_path":str(output_root/f"TRUE_B2_INPUT_NORMALIZATION_{key}.json"),"cache_root":str(cache_root/key),"train_count":TRAIN_COUNT,"dev_count":DEV_COUNT}
    atomic_json(output_root / "TRUE_B2_PIXEL_PREPARATION.json", {"status":"pass","created_utc":utc_now(),"method":"pixel_select_then_repeat_to_32_slots_then_encode_then_pool","rows":EXPECTED_ROWS,"conditions":input_reports,"model_info":model_info,"checkpoint":str(config.checkpoint),"checkpoint_sha256":sha256_file(config.checkpoint),"source_commit":config.source_commit,"elapsed_seconds":time.monotonic()-started,"final_test_access":0})
    print(json.dumps({"status":"pass","conditions":list(condition_features),"rows":EXPECTED_ROWS},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
