from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from model import V3Full28Predictor, parameter_count, state_hash

ALLOWED_PHYSICAL_GPUS = frozenset({2, 3})
V3_VIDEO_DIM = 768
R3D_VIDEO_DIM = 512
VIDEO_STEPS = 16
ACTION_DIM = 28
HALF_DIM = 14
WIDTH = 128
R3D_MEAN = (0.43216, 0.394666, 0.37645)
R3D_STD = (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_physical_gpu(value: str | None) -> int:
    if value is None or not value.strip().isdigit():
        raise RuntimeError("GPU_WATCHER_PHYSICAL_GPU must be set by the shared watcher")
    physical = int(value)
    if physical not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(f"physical GPU {physical} is outside the locked allowlist {sorted(ALLOWED_PHYSICAL_GPUS)}")
    return physical


def legal_frame_indices(anchor_frame: int) -> np.ndarray:
    if isinstance(anchor_frame, bool) or int(anchor_frame) < 119:
        raise ValueError("anchor must have a complete 120-frame prefix")
    values = np.floor(np.linspace(int(anchor_frame) - 119, int(anchor_frame), 32)).astype(np.int64)
    if values.shape != (32,) or values[0] < 0 or values[-1] != int(anchor_frame) or np.any(np.diff(values) < 0):
        raise ValueError("invalid V3 frame schedule")
    if np.any(values > int(anchor_frame)):
        raise ValueError("future frame entered input clip")
    return values


def common_pixels_r3d(frames: np.ndarray, resolution: int = 384) -> torch.Tensor:
    """V3 bilinear shortest-side resize and center crop, then R3D normalization; returns C,T,H,W."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"expected uint8 T,H,W,3 RGB frames; got {frames.shape} {frames.dtype}")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0).permute(0, 3, 1, 2)
    _, _, height, width = tensor.shape
    scale = resolution / min(height, width)
    new_height = max(resolution, round(height * scale))
    new_width = max(resolution, round(width * scale))
    tensor = F.interpolate(tensor, size=(new_height, new_width), mode="bilinear", align_corners=False)
    top = (new_height - resolution) // 2
    left = (new_width - resolution) // 2
    tensor = tensor[:, :, top:top + resolution, left:left + resolution]
    mean = torch.tensor(R3D_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(R3D_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3).contiguous()


def make_matched_alt_predictor(horizon: int, seed: int) -> tuple[V3Full28Predictor, dict[str, object]]:
    """Copy every same-shape V3 predictor tensor; isolate only the 512D visual projection RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        canonical = V3Full28Predictor(horizon=horizon, seed=seed, video_dim=V3_VIDEO_DIM)
        canonical_rng_after_init = torch.get_rng_state().clone()
        alternative = V3Full28Predictor(horizon=horizon, seed=seed, video_dim=R3D_VIDEO_DIM)
        canonical_state = canonical.state_dict()
        alternative_state = alternative.state_dict()
        copied = []
        for name, value in canonical_state.items():
            if name.startswith("video_projection."):
                continue
            if name not in alternative_state or alternative_state[name].shape != value.shape:
                raise RuntimeError(f"shared module mismatch at {name}")
            alternative_state[name] = value.detach().clone()
            copied.append(name)
        load_result = alternative.load_state_dict(alternative_state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"state-copy mismatch: missing={load_result.missing_keys}; unexpected={load_result.unexpected_keys}")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0xA1700000 + int(seed))
            alternative.video_projection.reset_parameters()
        shared_matches = all(torch.equal(canonical.state_dict()[k], alternative.state_dict()[k]) for k in copied)
        if not shared_matches:
            raise RuntimeError("shared initialization tensors differ after copy")
        shared_digest=hashlib.sha256()
        for name in sorted(copied):
            shared_digest.update(name.encode("utf-8"));shared_digest.update(canonical_state[name].detach().cpu().contiguous().numpy().tobytes())
        projection_digest=hashlib.sha256()
        for name,value in sorted(alternative.video_projection.state_dict().items()):
            projection_digest.update(name.encode("utf-8"));projection_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        report = {
            "seed": int(seed),
            "canonical_video_dim": V3_VIDEO_DIM,
            "alternative_video_dim": R3D_VIDEO_DIM,
            "shared_state_keys_copied": len(copied),
            "shared_initialization_verified": shared_matches,
            "shared_modules_sha256": shared_digest.hexdigest(),
            "canonical_full_initial_state_sha256": state_hash(canonical),
            "alternative_projection_sha256": projection_digest.hexdigest(),
            "alternative_full_initial_state_sha256": state_hash(alternative),
            "canonical_video_projection_parameters": int(canonical.video_projection.weight.numel() + canonical.video_projection.bias.numel()),
            "alternative_video_projection_parameters": int(alternative.video_projection.weight.numel() + alternative.video_projection.bias.numel()),
            "canonical_predictor_parameters": parameter_count(canonical),
            "alternative_predictor_parameters": parameter_count(alternative),
            "initial_rng_state_sha256": hashlib.sha256(canonical_rng_after_init.numpy().tobytes()).hexdigest(),
        }
    # Match the post-constructor CPU RNG state of the canonical V3 model; no CUDA generator is queried or changed.
    torch.set_rng_state(canonical_rng_after_init)
    return alternative, report


def episode_equal_moments(features: np.ndarray, episode_ids: np.ndarray, train_mask: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or len(features) != len(episode_ids) or len(features) != len(train_mask):
        raise ValueError("features/episode_ids/train_mask shape mismatch")
    means=[]
    seconds=[]
    for episode in np.unique(episode_ids[train_mask]):
        selected=features[train_mask & (episode_ids == episode)].astype(np.float64)
        if not len(selected):
            continue
        means.append(selected.mean(axis=0))
        seconds.append((selected * selected).mean(axis=0))
    if not means:
        raise ValueError("no train rows available for feature normalization")
    mean=np.asarray(means).mean(axis=0)
    variance=np.maximum(np.asarray(seconds).mean(axis=0)-mean*mean,0.0)
    std=np.sqrt(variance)
    safe_std=np.where(std <= epsilon,1.0,std)
    return mean.astype(np.float32),safe_std.astype(np.float32)
