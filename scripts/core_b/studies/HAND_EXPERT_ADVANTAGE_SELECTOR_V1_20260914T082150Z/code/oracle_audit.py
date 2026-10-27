"""Deterministic expert-advantage, oracle and clustered-bootstrap primitives."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


GRANULARITIES = ("GLOBAL", "EPISODE", "WINDOW5", "STEP_GROUP", "JOINT_STEP")
WINDOWS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30))


def validate_prediction_contract(prediction: np.ndarray, name: str, *, dims: int = 14) -> None:
    array = np.asarray(prediction)
    if array.ndim != 3 or array.shape[1:] != (30, dims):
        raise ValueError(f"{name} shape must be (N,30,{dims}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")


def validate_expert_mapping(experts: Mapping[str, np.ndarray], target: np.ndarray) -> None:
    if not experts or "PERSISTENCE" not in experts:
        raise ValueError("expert mapping must contain PERSISTENCE and at least one proposal")
    target = np.asarray(target)
    if target.ndim != 3:
        raise ValueError(f"target must be 3D, got {target.shape}")
    shape = target.shape
    for name, prediction in experts.items():
        if np.asarray(prediction).shape != shape:
            raise ValueError(f"expert {name} shape {np.asarray(prediction).shape} != target {shape}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"expert {name} is non-finite")
    if not np.isfinite(target).all():
        raise ValueError("target is non-finite")


def advantage_arrays(persistence: np.ndarray, neural: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Persistence error, neural error and Persistence-minus-neural advantage."""
    validate_expert_mapping({"PERSISTENCE": persistence, "N_RAW": neural}, np.asarray(target))
    persistence_error = np.abs(np.asarray(persistence) - np.asarray(target))
    neural_error = np.abs(np.asarray(neural) - np.asarray(target))
    return persistence_error, neural_error, persistence_error - neural_error


def _score_tensor(error_stack: np.ndarray, granularity: str) -> np.ndarray:
    # error_stack is (E,N,T,D). The returned tensor has the choice axes first
    # and expert last, preserving episode clustering.
    if granularity == "GLOBAL":
        return error_stack.mean(axis=(1, 2, 3))
    if granularity == "EPISODE":
        return error_stack.mean(axis=(2, 3)).transpose(1, 0)
    if granularity == "WINDOW5":
        values = [error_stack[:, :, start:end, :].mean(axis=(2, 3)).transpose(1, 0) for start, end in WINDOWS]
        return np.stack(values, axis=1)
    if granularity == "STEP_GROUP":
        return error_stack.mean(axis=3).transpose(1, 2, 0)
    if granularity == "JOINT_STEP":
        return error_stack.transpose(1, 2, 3, 0)
    raise ValueError(f"unknown granularity: {granularity}")


def _choice_from_scores(scores: np.ndarray, expert_names: tuple[str, ...]) -> np.ndarray:
    # PERSISTENCE is required to be first, so argmin's stable tie behavior is
    # the locked conservative tie rule.
    if expert_names[0] != "PERSISTENCE":
        raise ValueError("PERSISTENCE must be first for the tie rule")
    return np.argmin(scores, axis=-1)


def _materialize_choice(expert_stack: np.ndarray, choice: np.ndarray, granularity: str) -> np.ndarray:
    # expert_stack=(E,N,T,D)
    n = expert_stack.shape[1]
    result = np.empty((n, expert_stack.shape[2], expert_stack.shape[3]), dtype=expert_stack.dtype)
    if granularity == "GLOBAL":
        result[...] = expert_stack[int(choice[0])]
        return result
    if granularity == "EPISODE":
        for episode in range(n):
            result[episode] = expert_stack[int(choice[episode]), episode]
        return result
    if granularity == "WINDOW5":
        for episode in range(n):
            for window, (start, end) in enumerate(WINDOWS):
                result[episode, start:end] = expert_stack[int(choice[episode, window]), episode, start:end]
        return result
    if granularity == "STEP_GROUP":
        for episode in range(n):
            for step in range(expert_stack.shape[2]):
                result[episode, step] = expert_stack[int(choice[episode, step]), episode, step]
        return result
    if granularity == "JOINT_STEP":
        for episode in range(n):
            for step in range(expert_stack.shape[2]):
                for joint in range(expert_stack.shape[3]):
                    result[episode, step, joint] = expert_stack[int(choice[episode, step, joint]), episode, step, joint]
        return result
    raise ValueError(f"unknown granularity: {granularity}")


def oracle_select(experts: Mapping[str, np.ndarray], target: np.ndarray, granularity: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Select the lowest-error expert under one locked oracle granularity."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity: {granularity}")
    validate_expert_mapping(experts, np.asarray(target))
    names = tuple(experts)
    stack = np.stack([np.abs(np.asarray(experts[name]) - np.asarray(target)) for name in names], axis=0)
    scores_tensor = _score_tensor(stack, granularity)
    choice = _choice_from_scores(scores_tensor, names)
    if granularity == "GLOBAL":
        choice = np.full((np.asarray(target).shape[0],), int(choice), dtype=np.int64)
    selected = _materialize_choice(np.stack([np.asarray(experts[name]) for name in names], axis=0), choice, granularity)
    if granularity == "GLOBAL":
        score_dict = {name: np.asarray(scores_tensor[index]) for index, name in enumerate(names)}
    else:
        score_dict = {name: scores_tensor[..., index] for index, name in enumerate(names)}
    return selected, choice, score_dict


def composite_full28(hand: np.ndarray, arm: np.ndarray, template: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand)
    arm = np.asarray(arm)
    template = np.asarray(template)
    if hand.ndim != 3 or arm.ndim != 3 or template.ndim != 3 or hand.shape[:2] != arm.shape[:2] or template.shape[:2] != hand.shape[:2] or template.shape[2] != hand.shape[2] + arm.shape[2]:
        raise ValueError("Hand/Arm/template shapes are not canonical")
    return np.concatenate((hand, arm), axis=2)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, *, replicates: int = 10_000) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("bootstrap values must be finite and have at least two episodes")
    estimates = np.empty(replicates, dtype=np.float64)
    chunk = 500
    offset = 0
    while offset < replicates:
        count = min(chunk, replicates - offset)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[offset:offset + count] = values[indices].mean(axis=1)
        offset += count
    return {"effect": float(values.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "replicates": int(replicates), "N_episode": int(len(values))}


def bootstrap_effect(candidate_values: np.ndarray, baseline_values: np.ndarray, rng: np.random.Generator, *, replicates: int = 10_000) -> dict[str, float | int]:
    candidate_values = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline_values, dtype=np.float64).reshape(-1)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("paired bootstrap vectors must have equal shape")
    result = bootstrap_mean(baseline_values - candidate_values, rng, replicates=replicates)
    result["effect_definition"] = "baseline minus candidate; positive means candidate improves"
    return result


def signflip_p(values: np.ndarray, rng: np.random.Generator, *, replicates: int = 10_000) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = abs(float(values.mean()))
    extreme = 0
    processed = 0
    while processed < replicates:
        count = min(500, replicates - processed)
        signs = np.where(rng.random((count, len(values))) < 0.5, -1.0, 1.0)
        extreme += int(np.sum(np.abs((signs * values[None, :]).mean(axis=1)) >= observed))
        processed += count
    return float((extreme + 1) / (replicates + 1))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.zeros(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or int(labels.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * ordered) / ordered.sum())


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def calibration_bins(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    rows = []
    for index in range(bins):
        low = index / bins
        high = 1.0 if index == bins - 1 else (index + 1) / bins
        mask = (scores >= low) & (scores <= high if index == bins - 1 else scores < high)
        rows.append({"bin": index, "lower": low, "upper": high, "N": int(mask.sum()), "mean_score": float(scores[mask].mean()) if mask.any() else float("nan"), "observed_prevalence": float(labels[mask].mean()) if mask.any() else float("nan")})
    return rows


def event_compatible_motion(target: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    persistence = np.asarray(persistence)
    first = np.abs(target[:, :1] - persistence[:, :1]).mean(axis=2)
    transitions = np.abs(np.diff(target, axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def full_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(prediction) - np.asarray(target)).mean(axis=(1, 2))


def group_error(prediction: np.ndarray, target: np.ndarray, group: slice) -> np.ndarray:
    return np.abs(np.asarray(prediction)[:, :, group] - np.asarray(target)[:, :, group]).mean(axis=(1, 2))


def direction_metrics(proposal: np.ndarray, target: np.ndarray, persistence: np.ndarray, *, denominator_floor: float = 1e-6) -> dict[str, np.ndarray]:
    proposal_delta = np.asarray(proposal) - np.asarray(persistence)
    target_delta = np.asarray(target) - np.asarray(persistence)
    dot = np.sum(proposal_delta * target_delta, axis=2)
    proposal_norm = np.linalg.norm(proposal_delta, axis=2)
    target_norm = np.linalg.norm(target_delta, axis=2)
    cosine = np.divide(dot, proposal_norm * target_norm, out=np.full_like(dot, np.nan, dtype=np.float64), where=(proposal_norm * target_norm) > denominator_floor)
    sign_mask = (np.abs(proposal_delta) > denominator_floor) & (np.abs(target_delta) > denominator_floor)
    sign_agreement = np.divide((np.sign(proposal_delta) == np.sign(target_delta)) * sign_mask, sign_mask, out=np.full_like(proposal_delta, np.nan, dtype=np.float64), where=sign_mask)
    magnitude_ratio = np.divide(np.abs(proposal_delta), np.abs(target_delta), out=np.full_like(proposal_delta, np.nan, dtype=np.float64), where=np.abs(target_delta) > denominator_floor)
    return {"cosine": cosine, "sign_agreement": sign_agreement, "magnitude_ratio": magnitude_ratio}
