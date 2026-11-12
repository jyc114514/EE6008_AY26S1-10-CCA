from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.videomae_encoder import (  # noqa: E402
    VideoMAEEncoder,
    load_videomae_config,
    select_device,
)


DEFAULT_MANIFEST = (
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project/repo/"
    "ee6008-humanoid-vjepa/manifests/"
    "g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/"
    "core_a_protocol_v3_duplicate_safe/train_validation_label_manifest.parquet"
)
DEFAULT_G1_ROOT = (
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project/data_raw/"
    "HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
)
DEFAULT_OUTPUT_DIR = (
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project/tmp/"
    "s126mdg33_02/videomae_baseline_20260907/outputs/features"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen VideoMAE pooled features for Task A train/validation."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "videomae_profile.yaml"),
        help="Path to the VideoMAE profile YAML.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--g1-root", default=DEFAULT_G1_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        required=True,
        help="Only train/validation are allowed. Final-test is intentionally unsupported.",
    )
    parser.add_argument(
        "--variant",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Static repeats the frame at t=0 for all VideoMAE input frames.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional tiny-run cap for smoke testing.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model from the local Hugging Face cache only.",
    )
    return parser.parse_args()


def load_manifest(path: Path, split: str, max_episodes: int | None) -> Any:
    import pandas as pd

    df = pd.read_parquet(path)
    required = {"episode_index", "video_path", "split", "contiguous_label_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    df = df[df["split"] == split].copy()
    if "v3_benchmark_inclusion" in df.columns:
        df = df[df["v3_benchmark_inclusion"] == "primary"].copy()
    if "classification_inclusion" in df.columns:
        df = df[df["classification_inclusion"] == "primary"].copy()
    if "eligible_4s" in df.columns:
        df = df[df["eligible_4s"] == True].copy()  # noqa: E712

    df = df.sort_values("episode_index").reset_index(drop=True)
    if max_episodes is not None:
        df = df.head(max_episodes).copy()
    if df.empty:
        raise ValueError(f"No rows selected for split={split!r}")
    return df


def read_video_clip(
    video_path: Path,
    *,
    num_frames: int,
    sample_fps: float,
    window_seconds: float,
    variant: str = "dynamic",
) -> torch.Tensor:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required on the server to decode G1 videos.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata for {video_path}")

    timestamps = torch.arange(num_frames, dtype=torch.float64) / float(sample_fps)
    timestamps = torch.clamp(timestamps, max=max(0.0, float(window_seconds) - 1e-6))
    frame_indices = torch.clamp(
        torch.round(timestamps * source_fps).to(torch.int64),
        min=0,
        max=max(0, total_frames - 1),
    ).tolist()
    if variant == "static":
        frame_indices = frame_indices[:1]

    frames: list[torch.Tensor] = []
    last_rgb = None
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok:
            if last_rgb is None:
                capture.release()
                raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
            rgb = last_rgb
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last_rgb = rgb
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())

    capture.release()
    if variant == "static":
        frames = frames * num_frames
    return torch.stack(frames, dim=0).to(torch.uint8)


def flush_batch(
    *,
    encoder: VideoMAEEncoder,
    videos: list[torch.Tensor],
    pooled_chunks: list[torch.Tensor],
    token_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    batch = torch.stack(videos, dim=0)
    features = encoder(batch)
    pooled_chunks.append(features.pooled_features.detach().cpu().to(torch.float32))
    return tuple(features.token_features.shape[1:])


def main() -> int:
    args = parse_args()
    config = load_videomae_config(args.config)
    model_config = config["model"]
    device = select_device(force_cpu=args.cpu)

    manifest_path = Path(args.manifest)
    g1_root = Path(args.g1_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = VideoMAEEncoder(
        model_config["name"],
        device=device,
        freeze=bool(model_config.get("freeze", True)),
        local_files_only=args.local_files_only,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    expected_shape = encoder.expected_input_shape(batch_size=1)
    num_frames = int(expected_shape[1])

    df = load_manifest(manifest_path, args.split, args.max_episodes)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Model: {model_config['name']}")
    print(f"Split: {args.split}")
    print(f"Variant: {args.variant}")
    print(f"Episodes selected: {len(df)}")
    print(f"Input frames: {num_frames}")
    print(f"Sample FPS: {args.sample_fps}")
    print(f"Window seconds: {args.window_seconds}")

    pooled_chunks: list[torch.Tensor] = []
    batch_videos: list[torch.Tensor] = []
    episode_indices: list[int] = []
    labels: list[int] = []
    video_paths: list[str] = []
    token_shape: tuple[int, ...] | None = None
    start = time.perf_counter()

    for row_idx, row in df.iterrows():
        relative_video = Path(str(row["video_path"]))
        absolute_video = g1_root / relative_video
        clip = read_video_clip(
            absolute_video,
            num_frames=num_frames,
            sample_fps=float(args.sample_fps),
            window_seconds=float(args.window_seconds),
            variant=args.variant,
        )
        batch_videos.append(clip)
        episode_indices.append(int(row["episode_index"]))
        labels.append(int(row["contiguous_label_index"]))
        video_paths.append(str(relative_video))

        if len(batch_videos) == int(args.batch_size):
            token_shape = flush_batch(
                encoder=encoder,
                videos=batch_videos,
                pooled_chunks=pooled_chunks,
                token_shape=token_shape,
            )
            batch_videos.clear()
            print(f"processed {row_idx + 1}/{len(df)}", flush=True)

    if batch_videos:
        token_shape = flush_batch(
            encoder=encoder,
            videos=batch_videos,
            pooled_chunks=pooled_chunks,
            token_shape=token_shape,
        )

    pooled = torch.cat(pooled_chunks, dim=0)
    suffix = "_static" if args.variant == "static" else ""
    output_path = output_dir / f"videomae_{args.split}{suffix}_pooled.pt"
    metadata = {
        "model_name": model_config["name"],
        "split": args.split,
        "input_variant": args.variant,
        "static_frame_rule": "repeat_first_frame_at_t0" if args.variant == "static" else None,
        "manifest": str(manifest_path),
        "g1_root": str(g1_root),
        "num_examples": int(pooled.shape[0]),
        "feature_dim": int(pooled.shape[1]),
        "token_shape_per_example": list(token_shape) if token_shape is not None else None,
        "input_frames": num_frames,
        "sample_fps": float(args.sample_fps),
        "window_seconds": float(args.window_seconds),
        "encoder_frozen": bool(encoder.is_frozen),
        "elapsed_seconds": time.perf_counter() - start,
        "final_test_used": False,
    }
    torch.save(
        {
            "features": pooled,
            "labels": torch.tensor(labels, dtype=torch.long),
            "episode_indices": torch.tensor(episode_indices, dtype=torch.long),
            "video_paths": video_paths,
            "metadata": metadata,
        },
        output_path,
    )
    (output_dir / f"videomae_{args.split}{suffix}_pooled_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
