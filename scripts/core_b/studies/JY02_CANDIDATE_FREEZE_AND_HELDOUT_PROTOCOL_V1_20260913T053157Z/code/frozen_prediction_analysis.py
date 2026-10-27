"""Separate analysis entry point that requires a frozen prediction bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_predictions(
    prediction_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_shape: tuple[int, int] = (30, 28),
) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions only when an adjacent manifest pins their hash/shape."""

    prediction_path = Path(prediction_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("frozen") is not True:
        raise ValueError("prediction manifest is not frozen")
    if manifest.get("prediction_sha256") != _sha256(prediction_path):
        raise ValueError("prediction hash does not match frozen manifest")
    with np.load(prediction_path, allow_pickle=False) as bundle:
        episode_index = np.asarray(bundle["episode_index"])
        predictions = np.asarray(bundle["predictions"])
    if tuple(predictions.shape[1:]) != expected_shape:
        raise ValueError(f"unexpected prediction shape: {predictions.shape}")
    if predictions.shape[0] != episode_index.shape[0]:
        raise ValueError("episode index and prediction row counts differ")
    return episode_index, predictions


def compute_mae_from_frozen_predictions(
    prediction_path: str | Path,
    manifest_path: str | Path,
    target_path: str | Path,
    *,
    allow_target_access: bool = False,
) -> float:
    """Compute a metric after the target-unlock phase, never during inference."""

    if not allow_target_access:
        raise PermissionError("target access requires the explicit post-inference analysis phase")
    episode_index, predictions = load_frozen_predictions(prediction_path, manifest_path)
    with np.load(target_path, allow_pickle=False) as bundle:
        target_episode_index = np.asarray(bundle["episode_index"])
        targets = np.asarray(bundle["targets"])
    if not np.array_equal(episode_index, target_episode_index):
        raise ValueError("prediction/target episode IDs are not paired")
    if targets.shape != predictions.shape:
        raise ValueError("prediction/target shapes differ")
    return float(np.mean(np.abs(predictions - targets)))
