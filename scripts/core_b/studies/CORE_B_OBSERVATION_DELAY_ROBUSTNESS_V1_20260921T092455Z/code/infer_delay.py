#!/usr/bin/env python3
"""Fixed-last primary-dev inference for one run and one observation delay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from delay_contract import DELAYS, EXP, V3, delay_transform_rows, load_main
from delay_model import DelayAwarePredictor, state_hash


ACTION_DIM = 28
CELLS = ("CLEAN-N", "CLEAN-D", "CLEAN-S", "ROBUST-N", "ROBUST-D", "ROBUST-S")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): sha.update(block)
    return sha.hexdigest()


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); part = path.with_name(f".{path.name}.part")
    with part.open("wb") as handle: np.savez_compressed(handle, **values)
    part.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cell", choices=CELLS, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--delay-index", type=int, choices=(0, 1, 2, 3), required=True); parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(0); device = torch.device("cuda:0")
    data, normalizer = load_main(); features = np.load(V3 / "cache/V3_TEMPORAL_FEATURES.npy", mmap_mode="r"); video_normalizer = json.loads((V3 / "data/V3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8")); video_mean = np.asarray(video_normalizer["mean"], dtype=np.float32); video_std = np.asarray(video_normalizer["safe_std"], dtype=np.float32)
    target_sigma = torch.tensor(normalizer["target_sigma"], dtype=torch.float32, device=device)
    run_root = EXP / f"runs/ODR-{args.cell}-H60-S{args.seed}/formal"; checkpoint = run_root / "checkpoint_last.pt"; payload = torch.load(checkpoint, map_location="cpu", weights_only=False); model = DelayAwarePredictor(horizon=60, seed=args.seed).to(device); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval(); before = state_hash(model)
    dev = np.flatnonzero(data["split_code"] == 1); outputs = []; ages = []; selected_times = []; selected_history_indices = []
    video_level = args.cell.split("-")[1]
    for start in range(0, len(dev), 64):
        rows = dev[start : start + 64]
        state_np, age_np, selected_time_np, selected_index_np = delay_transform_rows(data, normalizer, rows, np.full(len(rows), args.delay_index, dtype=np.int64))
        state = torch.from_numpy(state_np).to(device); mask = torch.from_numpy(data["history_mask"][rows]).to(device); relative = torch.from_numpy(data["history_relative_delta_standardized"][rows]).to(device); age = torch.from_numpy(age_np).to(device); anchor = torch.from_numpy(data["anchor_command_native"][rows]).to(device)
        video_index = 0 if video_level == "D" else 1; video_np = np.asarray(features[rows, video_index], dtype=np.float32); video_np = (video_np - video_mean) / video_std
        if video_level == "N": video_np = np.zeros_like(video_np)
        video = torch.from_numpy(video_np).to(device)
        with torch.inference_mode(): residual = model.forward_delay(state, mask, relative, age, video, anchor); prediction = anchor.unsqueeze(1) + residual * target_sigma.view(1, 1, ACTION_DIM)
        outputs.append(prediction.cpu().numpy().astype(np.float32)); ages.append(age_np); selected_times.append(selected_time_np); selected_history_indices.append(selected_index_np)
    prediction = np.concatenate(outputs); age = np.concatenate(ages); selected_time = np.concatenate(selected_times); selected_index = np.concatenate(selected_history_indices); after = state_hash(model); bundle_id = f"{args.run_id}-D{args.delay_index}"; bundle_dir = EXP / f"predictions/{args.run_id}"; bundle_path = bundle_dir / f"delay_{args.delay_index}.npz"
    atomic_npz(bundle_path, anchor_id=data["anchor_id"][dev], episode_index=data["episode_index"][dev], original_task_index=data["original_task_index"][dev], anchor_frame=data["anchor_frame"][dev], anchor_timestamp=data["anchor_timestamp"][dev], requested_delay_seconds=np.asarray(DELAYS[args.delay_index], dtype=np.float32), observation_cutoff_timestamp=data["anchor_timestamp"][dev] - DELAYS[args.delay_index], observation_age_seconds=age.astype(np.float32), observation_selected_timestamps=selected_time.astype(np.float32), observation_selected_history_indices=selected_index.astype(np.int16), horizon_steps=np.asarray(60, dtype=np.int64), predictions_native=prediction)
    metadata = {"status": "PASS", "bundle_id": bundle_id, "run_id": args.run_id, "cell": args.cell, "seed": args.seed, "delay_index": args.delay_index, "delay_seconds": float(DELAYS[args.delay_index]), "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "shape": list(prediction.shape), "primary_windows": len(dev), "state_hash_before": before, "state_hash_after": after, "state_unchanged": before == after, "encoder_updated": False, "new_video_extraction_started": False, "final_test_access": 0, "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown")}
    (bundle_dir / f"delay_{args.delay_index}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(metadata, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
