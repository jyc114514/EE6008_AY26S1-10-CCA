"""Deterministic timestamp-based RGB clip sampling and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from decord import VideoReader, cpu
    from decord._ffi.base import DECORDError
except (ImportError, OSError):
    # Decord is optional because some environments expose an incompatible
    # C++ runtime. The recorded project fallback is OpenCV.
    VideoReader = None
    cpu = None

    class DECORDError(Exception):
        """Placeholder exception when the optional decoder cannot import."""

    _DECORD_USABLE = False
else:
    _DECORD_USABLE = True


class ShortEpisodeError(ValueError):
    """Raised when an episode cannot provide the unpadded four-second prefix."""


@dataclass(frozen=True)
class ClipSample:
    frames: np.ndarray
    frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    source_fps: float
    decoder: str = "decord"


def timestamp_frame_indices(
    *,
    num_frames: int,
    source_fps: float,
    duration_seconds: float = 4.0,
    sample_fps: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if num_frames <= 0 or source_fps <= 0:
        raise ValueError("num_frames and source_fps must be positive")
    timestamps = (
        np.arange(round(duration_seconds * sample_fps), dtype=np.float64) / sample_fps
    )
    available_seconds = num_frames / source_fps
    if available_seconds + 1e-9 < duration_seconds:
        raise ShortEpisodeError(
            f"episode is {available_seconds:.6f}s, shorter than required {duration_seconds:.6f}s"
        )
    indices = np.rint(timestamps * source_fps).astype(np.int64)
    if indices[-1] >= num_frames or np.any(np.diff(indices) <= 0):
        raise ValueError(
            f"timestamp sampling produced invalid frame indices: {indices.tolist()}"
        )
    return indices, timestamps


def decode_prefix(
    video_path: Path,
    *,
    duration_seconds: float = 4.0,
    sample_fps: int = 8,
    expected_source_fps: float | None = None,
) -> ClipSample:
    global _DECORD_USABLE
    if not _DECORD_USABLE:
        return _decode_prefix_opencv(
            video_path,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            expected_source_fps=expected_source_fps,
        )
    try:
        reader = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    except (DECORDError, OSError):
        _DECORD_USABLE = False
        return _decode_prefix_opencv(
            video_path,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            expected_source_fps=expected_source_fps,
        )
    source_fps = float(reader.get_avg_fps())
    if expected_source_fps is not None and abs(source_fps - expected_source_fps) > 0.25:
        raise ValueError(f"unexpected source fps {source_fps} for {video_path}")
    indices, timestamps = timestamp_frame_indices(
        num_frames=len(reader),
        source_fps=source_fps,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
    )
    try:
        frames = reader.get_batch(indices.tolist()).asnumpy()
    except (DECORDError, OSError):
        _DECORD_USABLE = False
        return _decode_prefix_opencv(
            video_path,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            expected_source_fps=expected_source_fps,
        )
    if frames.shape[0] != len(indices) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected decoded frame shape {frames.shape}")
    return ClipSample(frames, indices, timestamps, source_fps, "decord")


def _decode_prefix_opencv(
    video_path: Path,
    *,
    duration_seconds: float,
    sample_fps: int,
    expected_source_fps: float | None,
) -> ClipSample:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        num_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected_source_fps is not None and abs(source_fps - expected_source_fps) > 0.25:
            raise ValueError(f"unexpected source fps {source_fps} for {video_path}")
        indices, timestamps = timestamp_frame_indices(
            num_frames=num_frames,
            source_fps=source_fps,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
        )
        wanted = {int(index) for index in indices}
        frames: list[np.ndarray] = []
        for frame_index in range(int(indices[-1]) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"OpenCV failed at frame {frame_index} for {video_path}"
                )
            if frame_index in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    stacked = np.stack(frames, axis=0)
    if stacked.shape[0] != len(indices) or stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise ValueError(f"unexpected OpenCV decoded frame shape {stacked.shape}")
    return ClipSample(stacked, indices, timestamps, source_fps, "opencv")


def preprocess_vjepa(frames: np.ndarray, resolution: int = 384) -> torch.Tensor:
    """Deterministic resize-shortest-side + center crop, returning C,T,H,W."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected T,H,W,3 frames, got {frames.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2)  # T,C,H,W
    _, _, height, width = tensor.shape
    scale = resolution / min(height, width)
    new_height = max(resolution, round(height * scale))
    new_width = max(resolution, round(width * scale))
    tensor = F.interpolate(
        tensor, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    top = (new_height - resolution) // 2
    left = (new_width - resolution) // 2
    tensor = tensor[:, :, top : top + resolution, left : left + resolution]
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3).contiguous()
