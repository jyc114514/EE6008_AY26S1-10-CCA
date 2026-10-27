"""Metric and aggregation functions for frozen EE6008 evaluation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


CORE_B_TASKS = (119, 94, 90, 141, 230)


def core_a_classification_metrics(logits: np.ndarray, labels: np.ndarray, class_count: int = 120) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != class_count or truth.shape != (len(values),):
        raise ValueError("Core A expects logits [N,120] and labels [N]")
    if len(truth) == 0 or not np.isfinite(values).all() or np.any((truth < 0) | (truth >= class_count)):
        raise ValueError("Core A labels/logits must be nonempty, finite, and within the fixed class map")
    order = np.argsort(-values, axis=1, kind="stable")
    top1 = float(np.mean(order[:, 0] == truth))
    top5 = float(np.mean(np.any(order[:, : min(5, class_count)] == truth[:, None], axis=1)))
    macro_f1 = float(f1_score(truth, order[:, 0], labels=np.arange(class_count), average="macro", zero_division=0))
    return {"top1_accuracy": top1, "top5_accuracy": top5, "macro_f1": macro_f1}


def core_b_per_anchor_metrics(
    prediction_native: np.ndarray,
    target_native: np.ndarray,
    anchor_native: np.ndarray,
    target_sigma: np.ndarray,
    safe_fd_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction_native, dtype=np.float64)
    target = np.asarray(target_native, dtype=np.float64)
    anchor = np.asarray(anchor_native, dtype=np.float64)
    sigma = np.asarray(target_sigma, dtype=np.float64)
    fd_scale = np.asarray(safe_fd_scale, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[1:] != (30, 28) or target.shape != prediction.shape:
        raise ValueError("Core B predictions and targets must both be [N,30,28]")
    if anchor.shape != (len(prediction), 28) or sigma.shape != (28,) or fd_scale.shape != (28,):
        raise ValueError("Core B anchors/normalizers have invalid shapes")
    if len(prediction) == 0 or not all(np.isfinite(x).all() for x in (prediction, target, anchor, sigma, fd_scale)):
        raise ValueError("Core B arrays must be nonempty and finite")
    if np.any(sigma <= 0) or np.any(fd_scale <= 0):
        raise ValueError("Core B normalizer scales must be positive")
    position_error = np.abs(prediction - target) / sigma[None, None, :]
    position = 0.5 * (
        position_error[..., :14].mean(axis=(1, 2))
        + position_error[..., 14:].mean(axis=(1, 2))
    )
    predicted_fd = np.diff(prediction, axis=1) / fd_scale[None, None, :]
    target_fd = np.diff(target, axis=1) / fd_scale[None, None, :]
    future_fd_error = np.abs(predicted_fd - target_fd)
    future_fd = 0.5 * (
        future_fd_error[..., :14].mean(axis=(1, 2))
        + future_fd_error[..., 14:].mean(axis=(1, 2))
    )
    seam_error = np.abs(
        (prediction[:, 0] - anchor) / fd_scale[None, :]
        - (target[:, 0] - anchor) / fd_scale[None, :]
    )
    seam = 0.5 * (seam_error[:, :14].mean(axis=1) + seam_error[:, 14:].mean(axis=1))
    return {"position_mae": position, "safe_fd_l1": future_fd, "seam_safe_fd_l1": seam}


def core_b_task_macro(
    per_anchor: Mapping[str, np.ndarray],
    episode_ids: Sequence[int],
    task_ids: Sequence[int],
    *,
    fixed_tasks: Sequence[int] = CORE_B_TASKS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = np.asarray(episode_ids, dtype=np.int64)
    tasks = np.asarray(task_ids, dtype=np.int64)
    if episodes.ndim != 1 or tasks.shape != episodes.shape or len(episodes) == 0:
        raise ValueError("episode_ids and task_ids must be nonempty aligned vectors")
    metrics = {name: np.asarray(values, dtype=np.float64) for name, values in per_anchor.items()}
    if set(metrics) != {"position_mae", "safe_fd_l1", "seam_safe_fd_l1"}:
        raise ValueError("unexpected Core B metric set")
    if any(value.shape != episodes.shape or not np.isfinite(value).all() for value in metrics.values()):
        raise ValueError("per-anchor metric arrays must align and be finite")
    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_metrics: dict[int, dict[str, float]] = {}
    for task in fixed_tasks:
        task_mask = tasks == int(task)
        task_episodes = np.unique(episodes[task_mask])
        per_episode: dict[int, dict[str, float]] = {}
        for episode in task_episodes.tolist():
            mask = task_mask & (episodes == int(episode))
            row = {name: float(value[mask].mean()) for name, value in metrics.items()}
            per_episode[int(episode)] = row
            episode_rows.append({"episode_id": int(episode), "task_id": int(task), "anchor_support": int(mask.sum()), **row})
        task_support_episodes = len(task_episodes)
        if task_support_episodes:
            summary = {
                name: float(np.mean([row[name] for row in per_episode.values()]))
                for name in metrics
            }
            task_metrics[int(task)] = summary
            task_rows.append({
                "task_id": int(task), "episode_support": task_support_episodes,
                "anchor_support": int(task_mask.sum()), **summary,
            })
        else:
            task_rows.append({"task_id": int(task), "episode_support": 0, "anchor_support": 0, **{name: None for name in metrics}})
    missing_tasks = [int(task) for task in fixed_tasks if int(task) not in task_metrics]
    macro: dict[str, Any] = {
        "fixed_tasks": list(map(int, fixed_tasks)),
        "observed_tasks": sorted(task_metrics),
        "episode_support": int(len(np.unique(episodes))),
        "anchor_support": int(len(episodes)),
        "status": "not_estimable" if missing_tasks else "estimable",
        "missing_tasks": missing_tasks,
    }
    if not missing_tasks:
        macro.update({name: float(np.mean([task_metrics[int(task)][name] for task in fixed_tasks])) for name in metrics})
    return macro, task_rows, episode_rows


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("interval values must be a nonempty finite vector")
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(array, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def sample_sd(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("sample SD requires at least two finite seed values")
    return float(array.std(ddof=1))
