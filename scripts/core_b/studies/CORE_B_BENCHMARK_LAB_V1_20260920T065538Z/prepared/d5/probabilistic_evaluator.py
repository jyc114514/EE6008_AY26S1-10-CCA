"""D5 proper-score evaluator skeleton for future samples=(N,K,H,28)."""

from __future__ import annotations

import numpy as np


def _norm(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(values), axis=(-2, -1)))


def energy_score(samples: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Finite-sample Energy Score with an unbiased iid U-statistic term by default."""

    samples = np.asarray(samples, dtype=float)
    target = np.asarray(target, dtype=float)
    if samples.ndim != 4 or samples.shape[-1] != 28:
        raise ValueError("samples must have shape (N,K,H,28)")
    if target.shape != (samples.shape[0], samples.shape[2], 28):
        raise ValueError("target must have shape (N,H,28)")
    if weights is None:
        weights = np.full(samples.shape[1], 1.0 / samples.shape[1])
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (samples.shape[1],) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to 1")
    first = np.sum(weights[None, :] * _norm(samples - target[:, None, :, :]), axis=1)
    if samples.shape[1] < 2:
        return first
    pair = samples[:, :, None, :, :] - samples[:, None, :, :, :]
    pair_distance = _norm(pair)
    pair_weight = weights[None, :, None] * weights[None, None, :]
    second = 0.5 * np.sum(pair_weight * pair_distance, axis=(1, 2))
    return first - second


def weighted_finite_energy_score(samples: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return energy_score(samples, target, weights=weights)


def best_of_k(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.min(_norm(samples - target[:, None, :, :]), axis=1)


def path_coverage(samples: np.ndarray, target: np.ndarray, radius: float, *, simultaneous: bool = True) -> np.ndarray:
    distances = np.sqrt(np.sum(np.square(np.asarray(samples, dtype=float) - np.asarray(target, dtype=float)[:, None, :, :]), axis=-1))
    if simultaneous:
        return np.any(np.all(distances <= radius, axis=2), axis=1)
    return np.mean(np.any(distances <= radius, axis=1), axis=1)


def real_evaluation_disabled(*_args, **_kwargs) -> None:
    raise RuntimeError("D5 real distribution fitting/evaluation is disabled; status=prepared_not_executed")
