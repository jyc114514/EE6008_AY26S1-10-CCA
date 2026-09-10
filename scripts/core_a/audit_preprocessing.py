"""Audit OpenCV/ffmpeg decoding and re-encode ten validation-safe cache rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from ee6008.config import load_config
from ee6008.data import clips as clips_module
from ee6008.data.clips import preprocess_vjepa
from ee6008.models import encode_clip, load_vjepa2_1_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ffmpeg_rgb_frames(
    ffmpeg: str, video_path: Path, *, count: int, height: int, width: int
) -> np.ndarray:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}: {completed.stderr.decode(errors='replace')[:500]}"
        )
    expected = count * height * width * 3
    if len(completed.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg emitted {len(completed.stdout)} bytes; expected {expected}"
        )
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(
        count, height, width, 3
    )


def decode_opencv_for_audit(
    video_path: Path, *, duration_seconds: float, sample_fps: int, source_fps: float
):
    # Force the already-reviewed OpenCV path even if a future decord build
    # happens to initialize successfully. This does not mutate the raw input.
    clips_module._DECORD_USABLE = False
    sample = clips_module.decode_prefix(
        video_path,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        expected_source_fps=source_fps,
    )
    if sample.decoder != "opencv":
        raise AssertionError(f"expected OpenCV audit path, got {sample.decoder}")
    return sample


def array_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "shape_equal": True,
        "exact_frame_fraction": float(np.mean(np.all(left == right, axis=(1, 2, 3)))),
        "exact_pixel_fraction": float(np.mean(left == right)),
        "max_abs": int(difference.max(initial=0)),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, 0.99)),
    }


def tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape_equal": list(left.shape) == list(right.shape),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_finite": bool(torch.isfinite(left).all()),
        "right_finite": bool(torch.isfinite(right).all()),
    }
    if result["shape_equal"]:
        difference = (left.float() - right.float()).abs()
        result.update(
            {
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        left.float().reshape(1, -1), right.float().reshape(1, -1)
                    ).item()
                ),
            }
        )
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Core A preprocessing and decoder audit",
        "",
        f"Status: {report['status']}",
        f"Generated UTC: {report['generated_utc']}",
        f"Host/account: {report['hostname']} / {report['account']}",
        "",
        "## Contract",
        "",
            (
                "Ten train/validation episodes were selected by uniform rank coverage "
                "from the sorted eligible rows. OpenCV RGB frames were compared with "
                "ffmpeg rgb24 output at identical frame indices. The existing feature "
                "cache was read only; no raw file or cache entry was written."
            ),
        "",
        f"- selected episodes: {report['selected_episode_indices']}",
        f"- frame count: {report['frame_count']}",
        f"- sampled indices: {report['sampled_frame_indices']}",
        f"- sampled timestamps: {report['sampled_timestamps_seconds']}",
        "",
        "## Decoder comparison",
        "",
        "| episode | raw shape | exact frame fraction | max abs | mean abs | preprocessed max abs |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in report["episodes"]:
        raw = item["raw_decoder_comparison"]
        pre = item["preprocessed_comparison"]
        lines.append(
            f"| {item['episode_index']} | {raw.get('opencv_shape')} | "
            f"{raw.get('exact_frame_fraction', 'n/a')} | {raw.get('max_abs', 'n/a')} | "
            f"{raw.get('mean_abs', 'n/a')} | {pre.get('max_abs', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## V-JEPA cache comparison",
            "",
            (
                "Each row was decoded through the locked OpenCV path and passed "
                "through the local V-JEPA 2.1-B ema_encoder. The cache entry was "
                "loaded with weights_only=True."
            ),
            "",
            "| episode | pooled shape/finite | token shape/finite | new pooled vs cache cosine | new pooled max abs | pooling max abs |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for item in report["episodes"]:
        pooled = item["pooled_vs_cache"]
        tokens = item["tokens_vs_cache"]
        lines.append(
            f"| {item['episode_index']} | {pooled['left_shape']}/{pooled['left_finite']} "
            f"and {pooled['right_finite']} | {tokens['left_shape']}/{tokens['left_finite']} "
            f"and {tokens['right_finite']} | {pooled.get('cosine', 'n/a')} | "
            f"{pooled.get('max_abs', 'n/a')} | {item['new_pool_vs_new_tokens_mean']['max_abs']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- raw decoder exact-frame pass count: {report['raw_exact_frame_pass_count']}/10",
            f"- preprocessed tolerance pass count: {report['preprocessed_tolerance_pass_count']}/10",
            f"- feature shape/finite pass count: {report['feature_contract_pass_count']}/10",
            f"- feature comparison tolerance pass count: {report['feature_tolerance_pass_count']}/10",
            (
                "- Any nonzero compressed-video pixel difference is reported rather "
                "than silently rounded away. The feature tolerance is only a "
                "reproducibility comparison; it does not alter the historical cache."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--existing-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    rows = split_table[
        (split_table["split"].isin(("train", "validation")))
        & (split_table["validation_status"] == "valid")
        & split_table["eligible_4s"].astype(bool)
    ].sort_values("episode_index", kind="mergesort").reset_index(drop=True)
    if len(rows) < 10:
        raise RuntimeError("fewer than ten train/validation eligible rows")
    positions = np.linspace(0, len(rows) - 1, 10, dtype=int)
    selected = rows.iloc[positions].copy()
    existing_entry_root = args.existing_run / "vjepa2_1_b" / "entries"
    if not existing_entry_root.is_dir():
        raise FileNotFoundError(existing_entry_root)

    first_video = config.g1_root / str(selected.iloc[0].video_path)
    capture = cv2.VideoCapture(str(first_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {first_video}")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    frame_count = config.frame_count
    first_sample = decode_opencv_for_audit(
        first_video,
        duration_seconds=config.observation_seconds,
        sample_fps=config.sample_fps,
        source_fps=config.source_fps,
    )
    sampled_indices = first_sample.frame_indices.tolist()
    sampled_timestamps = first_sample.timestamps_seconds.tolist()
    ffmpeg_count = int(sampled_indices[-1]) + 1

    model, model_info = load_vjepa2_1_base(
        source_root=config.vjepa_source_root,
        checkpoint=config.checkpoint,
        device=device,
        num_frames=config.frame_count,
        resolution=config.input_resolution,
    )
    episodes: list[dict[str, Any]] = []
    raw_exact = 0
    preprocessed_pass = 0
    feature_contract_pass = 0
    feature_tolerance_pass = 0
    for row in selected.itertuples(index=False):
        episode_index = int(row.episode_index)
        video_path = config.g1_root / str(row.video_path)
        opencv_sample = decode_opencv_for_audit(
            video_path,
            duration_seconds=config.observation_seconds,
            sample_fps=config.sample_fps,
            source_fps=config.source_fps,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {video_path}")
        actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        ffmpeg_frames = ffmpeg_rgb_frames(
            str(args.ffmpeg),
            video_path,
            count=ffmpeg_count,
            height=actual_height,
            width=actual_width,
        )
        ffmpeg_sampled = ffmpeg_frames[np.asarray(opencv_sample.frame_indices)]
        raw_comparison = array_stats(opencv_sample.frames, ffmpeg_sampled)
        raw_comparison["opencv_shape"] = list(opencv_sample.frames.shape)
        raw_comparison["ffmpeg_shape"] = list(ffmpeg_sampled.shape)
        if raw_comparison.get("exact_frame_fraction") == 1.0:
            raw_exact += 1

        opencv_clip = preprocess_vjepa(opencv_sample.frames, config.input_resolution)
        ffmpeg_clip = preprocess_vjepa(ffmpeg_sampled, config.input_resolution)
        pre_comparison = tensor_stats(opencv_clip, ffmpeg_clip)
        if pre_comparison.get("max_abs", float("inf")) <= 1e-4:
            preprocessed_pass += 1

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            new_tokens, new_pooled = encode_clip(
                model, opencv_clip.to(device, non_blocking=True)
            )
        cache_path = existing_entry_root / f"episode_{episode_index:06d}.pt"
        record = torch.load(cache_path, map_location="cpu", weights_only=True)
        cached_tokens = record["tokens"].float()
        cached_pooled = record["pooled"].float()
        pooled_vs_cache = tensor_stats(new_pooled.cpu(), cached_pooled)
        tokens_vs_cache = tensor_stats(new_tokens.cpu(), cached_tokens)
        new_pool_vs_new_tokens = tensor_stats(
            new_pooled.cpu(), new_tokens.float().mean(dim=0).cpu()
        )
        cache_pool_vs_cache_tokens = tensor_stats(
            cached_pooled, cached_tokens.mean(dim=0)
        )
        contract_ok = (
            list(new_pooled.shape) == [768]
            and list(new_tokens.shape) == [9216, 768]
            and bool(torch.isfinite(new_pooled).all())
            and bool(torch.isfinite(new_tokens).all())
            and list(cached_pooled.shape) == [768]
            and list(cached_tokens.shape) == [9216, 768]
            and bool(torch.isfinite(cached_pooled).all())
            and bool(torch.isfinite(cached_tokens).all())
        )
        tolerance_ok = (
            pooled_vs_cache.get("cosine", 0.0) >= 0.999
            and pooled_vs_cache.get("max_abs", float("inf")) <= 0.05
            and tokens_vs_cache.get("cosine", 0.0) >= 0.999
            and tokens_vs_cache.get("mean_abs", float("inf")) <= 0.05
        )
        feature_contract_pass += int(contract_ok)
        feature_tolerance_pass += int(contract_ok and tolerance_ok)
        episodes.append(
            {
                "episode_index": episode_index,
                "split": str(row.split),
                "video_path": str(row.video_path),
                "raw_decoder_comparison": raw_comparison,
                "preprocessed_comparison": pre_comparison,
                "pooled_vs_cache": pooled_vs_cache,
                "tokens_vs_cache": tokens_vs_cache,
                "new_pool_vs_new_tokens_mean": new_pool_vs_new_tokens,
                "cache_pool_vs_cache_tokens_mean": cache_pool_vs_cache_tokens,
                "feature_contract_ok": contract_ok,
                "feature_tolerance_ok": tolerance_ok,
            }
        )

    report: dict[str, Any] = {
        "status": "pass"
        if raw_exact == 10
        and preprocessed_pass == 10
        and feature_contract_pass == 10
        and feature_tolerance_pass == 10
        else "review",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "device": str(device),
        "torch_version": torch.__version__,
        "dataset_revision": config.dataset_revision,
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "source_commit": config.source_commit,
        "existing_run": str(args.existing_run),
        "existing_run_manifest_sha256": sha256_file(
            args.existing_run / "feature_manifest.parquet"
        ),
        "selected_episode_indices": [int(value) for value in selected.episode_index],
        "selection_positions": [int(value) for value in positions],
        "frame_count": frame_count,
        "sampled_frame_indices": sampled_indices,
        "sampled_timestamps_seconds": sampled_timestamps,
        "ffmpeg_decode_frame_count": ffmpeg_count,
        "video_width": width,
        "video_height": height,
        "model_contract": model_info,
        "raw_exact_frame_pass_count": raw_exact,
        "preprocessed_tolerance_pass_count": preprocessed_pass,
        "feature_contract_pass_count": feature_contract_pass,
        "feature_tolerance_pass_count": feature_tolerance_pass,
        "episodes": episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "audit.md", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
