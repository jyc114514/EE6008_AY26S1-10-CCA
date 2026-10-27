"""Pure pre-encoder pixel interventions for Frozen attribution."""

from __future__ import annotations

import hashlib

import numpy as np


def apply_pixel_intervention(frames: np.ndarray, condition: str, *, seed: int, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise ValueError(f"expected (32,H,W,3), got {frames.shape}")
    source = np.arange(32, dtype=np.int64)
    if condition == "P1":
        positions = np.zeros(32, dtype=np.int64)
    elif condition == "P2":
        positions = np.full(32, 31, dtype=np.int64)
    elif condition == "P3":
        positions = source[::-1]
    elif condition == "P4":
        positions = np.random.default_rng(seed + int(episode_id)).permutation(32).astype(np.int64)
        if np.array_equal(positions, source):
            positions = np.roll(positions, 1)
    else:
        raise ValueError(condition)
    if np.array_equal(positions, source):
        raise AssertionError("pixel intervention is identity")
    return frames[positions].copy(), positions


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
