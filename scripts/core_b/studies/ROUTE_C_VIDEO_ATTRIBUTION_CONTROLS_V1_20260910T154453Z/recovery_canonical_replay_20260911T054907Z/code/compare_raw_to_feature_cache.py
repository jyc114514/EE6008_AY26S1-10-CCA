#!/usr/bin/env python3
"""Compare one raw replay path with the immutable canonical pooled feature."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("route_counterfactual_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diff_summary(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = candidate.astype(np.float32) - reference.astype(np.float32)
    return {
        "shape_candidate": list(candidate.shape),
        "shape_reference": list(reference.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "cosine": float(np.dot(candidate.reshape(-1), reference.reshape(-1)) / (np.linalg.norm(candidate) * np.linalg.norm(reference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--b2-input", type=Path, required=True)
    parser.add_argument("--route-runner", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--decoder", choices=("decord", "opencv"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("raw-to-feature comparison requires CUDA")
    runner = load_module(args.route_runner.resolve())
    sys.path.insert(0, str((args.project_root / "repo/ee6008-humanoid-vjepa/src").resolve()))
    sys.path.insert(0, str((args.project_root / "repo/third_party/vjepa2").resolve()))
    from ee6008.data import clips as clips_module
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.models import encode_clip, load_vjepa2_1_base

    data_manifest = pd.read_csv(args.data_manifest)
    data_row = data_manifest[data_manifest.episode_index == args.episode_id]
    if len(data_row) != 1:
        raise ValueError(f"expected one data manifest row for episode {args.episode_id}")
    data_row = data_row.iloc[0]
    feature_manifest = pd.read_parquet(args.feature_root / "feature_manifest.parquet")
    feature_row = feature_manifest[feature_manifest.episode_index == args.episode_id]
    if len(feature_row) != 1:
        raise ValueError(f"expected one feature manifest row for episode {args.episode_id}")
    feature_row = feature_row.iloc[0]
    feature_record_path = Path(str(feature_row.feature_path))
    feature_record = torch.load(feature_record_path, map_location="cpu", weights_only=True)
    metadata = feature_record["metadata"]

    with np.load(args.b2_input, allow_pickle=False) as archive:
        episode_ids = np.asarray(archive["episode_index"], dtype=np.int64)
        split_code = np.asarray(archive["split_code"])
        dynamic_pooled = np.asarray(archive["dynamic_pooled"], dtype=np.float32)
    input_indices = np.where((episode_ids == args.episode_id) & (split_code == 1))[0]
    if len(input_indices) != 1:
        raise ValueError("B2 input does not contain exactly one development row")
    input_index = int(input_indices[0])

    if args.decoder == "opencv":
        clips_module._DECORD_USABLE = False
    else:
        clips_module._DECORD_USABLE = True
    video_path = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d" / str(data_row.video_path)
    sample = decode_prefix(video_path, duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
    clip = preprocess_vjepa(sample.frames, resolution=384)
    device = torch.device("cuda")
    base_checkpoint = args.project_root / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
    model, model_info = load_vjepa2_1_base(
        source_root=args.project_root / "repo/third_party/vjepa2",
        checkpoint=base_checkpoint,
        device=device,
        num_frames=32,
        resolution=384,
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw_tokens, raw_pooled = encode_clip(model, clip.to(device, non_blocking=True))
    raw_tokens_np = raw_tokens.float().cpu().numpy()
    raw_pooled_np = raw_pooled.float().cpu().numpy()
    stored_tokens = feature_record["tokens"].float().numpy()
    stored_pooled = feature_record["pooled"].float().numpy()

    norm_path = args.project_root / "repo/ee6008-humanoid-vjepa/reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"
    norm = json.loads(norm_path.read_text(encoding="utf-8"))["dynamic_pooled"]
    normalized_raw = (raw_pooled_np - np.asarray(norm["mean"], dtype=np.float32)) / np.asarray(norm["safe_std"], dtype=np.float32)
    normalized_cached = dynamic_pooled[input_index]
    result = {
        "status": "diagnostic",
        "episode_id": args.episode_id,
        "decoder_requested": args.decoder,
        "decoder_observed": sample.decoder,
        "video_path": str(video_path),
        "video_sha256_fingerprint": str(data_row.video_sha256_fingerprint),
        "frame_indices": sample.frame_indices.tolist(),
        "timestamps_seconds": sample.timestamps_seconds.tolist(),
        "canonical_frame_indices": metadata["frame_indices"],
        "canonical_timestamps_seconds": metadata["timestamps_seconds"],
        "frame_indices_equal": sample.frame_indices.tolist() == metadata["frame_indices"],
        "timestamps_equal": sample.timestamps_seconds.tolist() == metadata["timestamps_seconds"],
        "clip_shape": list(clip.shape),
        "clip_sha256": sha256_bytes(clip.cpu().numpy().tobytes()),
        "raw_tokens_dtype": str(raw_tokens.dtype),
        "stored_tokens_dtype": str(feature_record["tokens"].dtype),
        "raw_pooled_dtype": str(raw_pooled.dtype),
        "stored_pooled_dtype": str(feature_record["pooled"].dtype),
        "seams": {
            "S1_frame_map": {"frame_indices_equal": sample.frame_indices.tolist() == metadata["frame_indices"], "timestamps_equal": sample.timestamps_seconds.tolist() == metadata["timestamps_seconds"]},
            "S2_decoder": {"requested": args.decoder, "observed": sample.decoder, "canonical": metadata["decoder"]},
            "S6_preprocessed_clip": {"shape": list(clip.shape), "sha256": sha256_bytes(clip.cpu().numpy().tobytes())},
            "S9_tokens_vs_cache": diff_summary(raw_tokens_np, stored_tokens),
            "S10_pooled_vs_cache": diff_summary(raw_pooled_np, stored_pooled),
            "S12_normalized_pooled_vs_B2_input": diff_summary(normalized_raw, normalized_cached),
        },
        "checkpoint_sha256": "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
        "encoder_precision": "CUDA BF16 autocast",
        "final_test_access": 0,
        "optimizer_steps": 0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
