#!/usr/bin/env python3
"""Fixed-last development inference for one state/video cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from state_video_model import StateVideoPredictor, state_hash


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
ACTION_DIM = 28
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **values)
    part.replace(path)


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key].copy() for key in bundle.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=CELLS)
    parser.add_argument("--horizon", type=int, required=True, choices=(30, 60))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal inference")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    main_data = load_bundle(V3 / "data/V3_MAIN_INPUTS.npz")
    supplementary_data = load_bundle(V3 / "data/V3_SUPPLEMENTARY_1S_INPUTS.npz")
    features = np.load(V3 / "cache/V3_TEMPORAL_FEATURES.npy", mmap_mode="r")
    normalizer = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    video_normalizer = json.loads((V3 / "data/V3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8"))
    target_sigma = torch.tensor(normalizer["target_sigma"], dtype=torch.float32, device=device)
    video_mean = torch.tensor(video_normalizer["mean"], dtype=torch.float32, device=device)
    video_std = torch.tensor(video_normalizer["safe_std"], dtype=torch.float32, device=device)
    checkpoint = EXP / f"runs/{args.run_id}/formal/checkpoint_last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = StateVideoPredictor(horizon=args.horizon, seed=args.seed).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    before = state_hash(model)
    state_level, video_level = args.cell.split("-")

    def run(data: dict[str, np.ndarray], indices: np.ndarray, cache_offset: int, bundle_name: str) -> tuple[int, ...]:
        outputs: list[np.ndarray] = []
        for start in range(0, len(indices), 64):
            rows = indices[start : start + 64]
            state = torch.from_numpy(data["history_standardized"][rows]).to(device)
            history_mask = torch.from_numpy(data["history_mask"][rows]).to(device)
            relative_delta = torch.from_numpy(data["history_relative_delta_standardized"][rows]).to(device)
            anchor = torch.from_numpy(data["anchor_command_native"][rows]).to(device)
            video_index = 0 if video_level == "D" else 1
            video_values = np.asarray(features[cache_offset + rows, video_index], dtype=np.float32)
            video = (torch.from_numpy(video_values).to(device) - video_mean) / video_std
            if video_level == "N":
                video = torch.zeros_like(video)
            with torch.inference_mode():
                residual = model.forward_locked(state, history_mask, relative_delta, video, anchor, state_level=state_level, video_level=video_level)
                prediction = anchor.unsqueeze(1) + residual * target_sigma.view(1, 1, ACTION_DIM)
            outputs.append(prediction.cpu().numpy().astype(np.float32))
        predictions = np.concatenate(outputs, axis=0)
        atomic_npz(EXP / f"predictions/{args.run_id}/{bundle_name}.npz", anchor_id=data["anchor_id"][indices], episode_index=data["episode_index"][indices], original_task_index=data["original_task_index"][indices], anchor_frame=data["anchor_frame"][indices], anchor_timestamp=data["anchor_timestamp"][indices], horizon_steps=np.asarray(args.horizon, dtype=np.int64), predictions_native=predictions)
        return predictions.shape

    primary_dev = np.flatnonzero(main_data["split_code"] == 1)
    primary_shape = run(main_data, primary_dev, 0, "primary")
    supplementary_shape = (0,)
    if args.horizon == 30:
        supplementary_shape = run(supplementary_data, np.arange(len(supplementary_data["episode_index"])), len(main_data["episode_index"]), "supplementary_1s")
    after = state_hash(model)
    metadata = {"status": "PASS", "run_id": args.run_id, "cell": args.cell, "state_level": state_level, "video_level": video_level, "horizon_steps": args.horizon, "seed": args.seed, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "primary_shape": list(primary_shape), "supplementary_shape": list(supplementary_shape), "state_hash_before": before, "state_hash_after": after, "state_unchanged": before == after, "encoder_updated": False, "new_encoder_forward_for_extraction": False, "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"), "final_test_access": 0}
    (EXP / f"predictions/{args.run_id}/PREDICTION_EXPORT_COMPLETE.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
