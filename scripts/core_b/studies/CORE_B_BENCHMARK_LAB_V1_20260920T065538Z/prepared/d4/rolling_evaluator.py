"""D4 rolling-forecast evaluator skeleton; real effect tables are intentionally disabled."""

from __future__ import annotations

import numpy as np


def intersect_future_times(old_times: np.ndarray, new_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Return target timestamps present in both forecast supports."""

    old = set(np.asarray(old_times).tolist())
    new = set(np.asarray(new_times).tolist())
    return np.asarray([value for value in np.asarray(target_times).tolist() if value in old and value in new])


def revision_utility(old_error: np.ndarray, new_error: np.ndarray) -> np.ndarray:
    """Positive values mean the new forecast has lower error."""

    old_error = np.asarray(old_error, dtype=float)
    new_error = np.asarray(new_error, dtype=float)
    if old_error.shape != new_error.shape:
        raise ValueError("old and new error arrays must have the same shape")
    return old_error - new_error


def forecast_stability(old_forecast: np.ndarray, new_forecast: np.ndarray) -> float:
    """Raw forecast change, reported separately from utility."""

    old_forecast = np.asarray(old_forecast, dtype=float)
    new_forecast = np.asarray(new_forecast, dtype=float)
    if old_forecast.shape != new_forecast.shape:
        raise ValueError("forecast shapes must match")
    return float(np.mean(np.abs(new_forecast - old_forecast)))


def fixed_window_anchor_indices(frame_indices: np.ndarray, *, window_frames: int = 120) -> np.ndarray:
    """Make the 4-second observation rule explicit; do not grow history after updates."""

    frame_indices = np.asarray(frame_indices, dtype=int)
    if frame_indices.ndim != 1:
        raise ValueError("frame_indices must be one-dimensional")
    return frame_indices[frame_indices >= window_frames - 1]


def evaluate_placeholder(*_args, **_kwargs) -> None:
    raise RuntimeError("D4 real-input utility evaluation is disabled in this benchmark; status=prepared_not_executed")
