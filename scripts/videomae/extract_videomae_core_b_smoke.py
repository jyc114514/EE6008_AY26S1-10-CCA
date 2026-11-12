"""Extract four target-blind Core B anchor fixtures using strict CPU VideoMAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path


CONFIG_SHA = "20b05a5026313946ec026dc0c7be4f1f641c47ab66ec7125d406403ce251165c"
WEIGHTS_SHA = "6af658f96396ed5f1207244a5a409ca4f6686f05f63b1be6b3609deee195eca7"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-frames", type=int, choices=(16, 32), default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allowed-gpus", type=int, nargs="+")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_root.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    if not output.is_relative_to(scratch) or output == scratch:
        raise ValueError("Output must be an isolated child of project/tmp/s126mdg33_02")
    if output.exists():
        raise FileExistsError("Use a fresh smoke output directory")

    import cv2
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

    torch.set_num_threads(2)
    device = torch.device("cpu")
    if args.device == "cuda":
        physical = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
        if not args.allowed_gpus or physical is None or int(physical) not in args.allowed_gpus:
            raise RuntimeError("CUDA Stage0 requires a watcher assignment within the task allowlist")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != physical:
            raise RuntimeError("Watcher physical/logical GPU mapping mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
    snapshots = project / "tmp/s126mdg33_02/hf_cache/hub/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots"
    config_dir = snapshots / "1bf2b82c809609f5df1ea7648aee577e2e486b63"
    weights_path = snapshots / "240e9734611173accbbf74cbdf4b641e4c431264/model.safetensors"
    if sha(config_dir / "config.json") != CONFIG_SHA or sha(weights_path) != WEIGHTS_SHA:
        raise AssertionError("Checkpoint identity differs from accepted strict CPU audit")
    config = VideoMAEConfig.from_pretrained(config_dir, local_files_only=True)
    config.num_frames = args.input_frames
    config._attn_implementation = "eager"
    model = VideoMAEForVideoClassification(config).cpu().eval()
    weights = load_file(str(weights_path), device="cpu")
    expected = model.state_dict()
    for layer in range(config.num_hidden_layers):
        prefix = f"videomae.encoder.layer.{layer}.attention.attention."
        if prefix + "q_bias" not in expected:
            weights[prefix + "query.bias"] = weights.pop(prefix + "q_bias")
            weights[prefix + "value.bias"] = weights.pop(prefix + "v_bias")
            weights[prefix + "key.bias"] = torch.zeros_like(weights[prefix + "query.bias"])
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.to(device)
    processor = VideoMAEImageProcessor.from_pretrained(config_dir, local_files_only=True)
    inputs_path = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    with np.load(inputs_path, allow_pickle=False) as archive:
        codes = archive["split_code"]
        if len(codes) != 20079 or set(np.unique(codes)) != {0, 1}:
            raise AssertionError("Unexpected primary train/dev population")
        selected = np.concatenate([np.flatnonzero(codes == split)[:2] for split in (0, 1)])
        if len(selected) != 4:
            raise AssertionError("Need two train and two dev fixtures")
        keys = ("anchor_id", "episode_index", "anchor_frame", "video_path", "split_code")
        data = {key: archive[key][selected].copy() for key in keys}
    raw_root = (project / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d").resolve()
    dynamic, static, frame_rows = [], [], []
    for row in range(4):
        anchor = int(data["anchor_frame"][row])
        if anchor < 119:
            raise AssertionError("Anchor does not support full history")
        indices = np.floor(np.linspace(anchor - 119, anchor, args.input_frames)).astype(np.int64)
        path = (raw_root / str(data["video_path"][row])).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        frames = {}
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot decode {path}")
            wanted = set(indices.tolist())
            for frame_index in range(anchor + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode failed at frame {frame_index}")
                if frame_index in wanted:
                    frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        rgb = [frames[int(index)] for index in indices]
        values = []
        for clip in (rgb, [rgb[-1]] * args.input_frames):
            pixels = processor([clip], return_tensors="pt")
            if tuple(pixels["pixel_values"].shape) != (1, args.input_frames, 3, 224, 224):
                raise AssertionError("Processor input shape mismatch")
            with torch.inference_mode():
                tokens = model.videomae(**{key: value.to(device) for key, value in pixels.items()}).last_hidden_state
            time_slots = args.input_frames // 2
            if tuple(tokens.shape) != (1, time_slots * 196, 384) or not torch.isfinite(tokens).all():
                raise AssertionError("Encoder output shape/finite mismatch")
            grid = tokens.reshape(1, time_slots, 14, 14, 384)
            quadrants = torch.stack([
                grid[:, :, r:r + 7, c:c + 7].mean(dim=(2, 3))
                for r in (0, 7) for c in (0, 7)
            ], dim=2)
            values.append(quadrants[0].cpu().numpy())
        dynamic.append(values[0]); static.append(values[1]); frame_rows.append(indices)
        print(json.dumps({"processed": row + 1, "total": 4}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "features_smoke.npz"
    np.savez_compressed(cache_path, cache_index=selected,
                        **{key: value for key, value in data.items() if key != "video_path"},
                        sampled_frame_indices=np.stack(frame_rows),
                        dynamic=np.stack(dynamic), static=np.stack(static))
    report = {
        "status": "PASS_CORE_B_FEATURE_STAGE0_GPU" if device.type == "cuda" else "PASS_CORE_B_FEATURE_SMOKE_CPU",
        "rows": 4, "train_rows": 2, "dev_rows": 2,
        "feature_shape_per_variant": [4, args.input_frames // 2, 4, 384],
        "input_frames": args.input_frames,
        "position_encoding_policy": "fixed sinusoidal positions rebuilt for effective frame count",
        "sampling": f"floor(linspace(anchor_frame-119,anchor_frame,{args.input_frames}))",
        "static_rule": "repeat_anchor_frame", "future_frames_read": False,
        "targets_loaded": False, "gpu_used": device.type == "cuda", "final_test_access": 0,
        "peak_allocated_mib": math.ceil(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0,
        "peak_reserved_mib": math.ceil(torch.cuda.max_memory_reserved() / 1024**2) if device.type == "cuda" else 0,
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU"),
        "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID"),
        "config_sha256": CONFIG_SHA, "weights_sha256": WEIGHTS_SHA,
        "input_sha256": sha(inputs_path),
        "processor_sha256": sha(config_dir / "preprocessor_config.json"),
        "script_sha256": sha(Path(__file__)),
        "feature_file": str(cache_path), "feature_sha256": sha(cache_path),
        "comparison_status": "interface/resource smoke only; formal benchmark pending",
    }
    (output / "SMOKE_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
