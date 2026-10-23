#!/usr/bin/env python3
"""Pure feature-seam interventions and deterministic donor pairing."""

from __future__ import annotations

from typing import Any

import numpy as np


def _derange_group(indices: np.ndarray, seed: int) -> dict[int, int]:
    if len(indices) < 2:
        return {}
    rng = np.random.default_rng(seed + int(indices[0]))
    shuffled = indices[rng.permutation(len(indices))]
    if np.any(shuffled == indices):
        shuffled = np.roll(indices, 1)
    if np.any(shuffled == indices):
        raise AssertionError("same-task donor mapping has a fixed point")
    return {int(recipient): int(donor) for recipient, donor in zip(indices, shuffled)}


def same_task_donors(episode_ids: np.ndarray, task_ids: np.ndarray, *, seed: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for task in np.unique(task_ids):
        indices = np.flatnonzero(task_ids == task)
        mapping.update(_derange_group(episode_ids[indices], seed))
    return mapping


def cross_task_donors(episode_ids: np.ndarray, task_ids: np.ndarray, *, seed: int) -> dict[int, int]:
    if len(np.unique(task_ids)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    for _ in range(128):
        donors = episode_ids[rng.permutation(len(episode_ids))]
        if np.all(donors != episode_ids) and np.all(task_ids[np.searchsorted(episode_ids, donors)] != task_ids):
            return {int(recipient): int(donor) for recipient, donor in zip(episode_ids, donors)}
    # Deterministic fallback: sort by task/episode and rotate across task blocks.
    order = np.lexsort((episode_ids, task_ids))
    sorted_ids = episode_ids[order]
    sorted_tasks = task_ids[order]
    for shift in range(1, len(sorted_ids)):
        donors = np.roll(sorted_ids, shift)
        donor_task = sorted_tasks[np.roll(np.arange(len(sorted_tasks)), shift)]
        if np.all(donor_task != sorted_tasks):
            return {int(recipient): int(donor) for recipient, donor in zip(sorted_ids, donors)}
    raise ValueError("unable to construct cross-task donor derangement")


def apply_feature_intervention(recipient: np.ndarray, condition: str, *, donor: np.ndarray | None = None) -> np.ndarray:
    if recipient.ndim != 2 or recipient.shape[1] != 768:
        raise ValueError(f"expected pooled feature batch N,768, got {recipient.shape}")
    if condition == "R1":
        return np.zeros_like(recipient)
    if condition in {"R2", "R3"}:
        if donor is None or donor.shape != recipient.shape:
            raise ValueError("donor feature batch must match recipient batch")
        return donor.copy()
    if condition == "P0":
        return recipient.copy()
    raise ValueError(condition)


def pairing_summary(mapping: dict[int, int], episode_to_task: dict[int, int]) -> dict[str, Any]:
    pairs = [{"recipient_episode": recipient, "donor_episode": donor, "recipient_task": episode_to_task[recipient], "donor_task": episode_to_task[donor]} for recipient, donor in sorted(mapping.items())]
    return {"N": len(pairs), "fixed_points": sum(row["recipient_episode"] == row["donor_episode"] for row in pairs), "same_task_pairs": sum(row["recipient_task"] == row["donor_task"] for row in pairs), "pairs": pairs}
