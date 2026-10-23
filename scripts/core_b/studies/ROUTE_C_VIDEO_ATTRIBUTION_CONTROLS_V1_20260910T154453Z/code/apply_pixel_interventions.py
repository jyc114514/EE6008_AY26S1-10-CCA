#!/usr/bin/env python3
"""Pure, pre-encoder pixel temporal interventions for Route C."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

PIXEL_CONDITIONS = ("P0", "P1", "P2", "P3", "P4")


def _permutation(seed: int, episode_id: int, length: int) -> np.ndarray:
    permutation = np.random.default_rng(seed + int(episode_id)).permutation(length)
    if np.array_equal(permutation, np.arange(length)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def apply_pixel_intervention(frames: np.ndarray, condition: str, *, seed: int, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed frames and source-position map without timestamp sorting."""
    if frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise ValueError(f"expected 32,H,W,3 frames, got {frames.shape}")
    if condition not in PIXEL_CONDITIONS:
        raise ValueError(condition)
    source = np.arange(32, dtype=np.int64)
    if condition == "P0":
        positions = source
    elif condition == "P1":
        positions = np.zeros(32, dtype=np.int64)
    elif condition == "P2":
        positions = np.full(32, 31, dtype=np.int64)
    elif condition == "P3":
        positions = source[::-1]
    else:
        positions = _permutation(seed, episode_id, 32)
    if positions.min() < 0 or positions.max() >= 32:
        raise AssertionError("intervention reaches outside the observed prefix")
    transformed = frames[positions].copy()
    if condition == "P4" and np.array_equal(positions, source):
        raise AssertionError("seeded shuffle is unexpectedly identity")
    return transformed, positions


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def describe_intervention(original: np.ndarray, transformed: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    return {"original_shape": list(original.shape), "transformed_shape": list(transformed.shape), "position_map": positions.tolist(), "original_sha256": array_sha256(original), "transformed_sha256": array_sha256(transformed), "multiset_preserved": sorted(positions.tolist()) == list(range(32)) or bool(np.all((positions == 0) | (positions == 31))), "nonidentity": not np.array_equal(positions, np.arange(32))}
