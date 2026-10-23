from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh


EPS = 1e-12


def episode_equal_weights(episode_ids: np.ndarray) -> np.ndarray:
    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or episodes.size == 0:
        raise ValueError("episode_ids must be a nonempty vector")
    _, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    weights = len(episodes) / (len(counts) * counts[inverse].astype(np.float64))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid episode-equal weights")
    if not np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError("episode-equal weights must average to one")
    return weights


def weighted_feature_stats(x: np.ndarray, weights: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("invalid feature matrix or weights")
    if np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("weights must be positive and finite")
    total = w.sum()
    mean = np.sum(x * w[:, None], axis=0) / total
    var = np.sum((x - mean) ** 2 * w[:, None], axis=0) / total
    raw_scale = np.sqrt(np.maximum(var, 0.0))
    constant = raw_scale <= eps
    scale = raw_scale.copy()
    scale[constant] = 1.0
    return mean, scale, constant


def standardize_features(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, constant: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = (x - mean) / scale
    if np.any(constant):
        z[:, constant] = 0.0
    return z


def episode_equal_target_scales(targets: np.ndarray, episode_ids: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    if targets.ndim != 3 or targets.shape[0] != len(episodes) or targets.shape[1] < 2:
        raise ValueError("targets must be N x H x D with H>=2")
    if not np.isfinite(targets).all():
        raise ValueError("nonfinite target")
    means, seconds, fd_means = [], [], []
    for ep in np.unique(episodes):
        rows = episodes == ep
        vals = targets[rows].reshape(-1, targets.shape[-1])
        means.append(vals.mean(axis=0))
        seconds.append(np.mean(vals * vals, axis=0))
        fd_means.append(np.abs(np.diff(targets[rows], axis=1)).mean(axis=(0, 1)))
    mean = np.mean(np.asarray(means), axis=0)
    second = np.mean(np.asarray(seconds), axis=0)
    sigma = np.sqrt(np.maximum(second - mean * mean, 0.0))
    sigma[sigma <= eps] = 1.0
    fd_scale = np.maximum(np.mean(np.asarray(fd_means), axis=0), eps)
    safe_fd = np.maximum(fd_scale, 0.001 * sigma)
    return sigma, fd_scale, safe_fd


def prepare_weighted_gram(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or w.shape != (len(x),):
        raise ValueError("X, Y, and weight dimensions disagree")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite fit data")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")
    total = float(w.sum())
    x_mean, x_scale, constant = weighted_feature_stats(x, w)
    z = standardize_features(x, x_mean, x_scale, constant)
    y_mean = np.sum(y * w[:, None], axis=0) / total
    yc = y - y_mean
    sqrtw = np.sqrt(w)
    zw = z * sqrtw[:, None]
    yw = yc * sqrtw[:, None]
    gram = (zw.T @ zw) / total
    rhs = (zw.T @ yw) / total
    gram = (gram + gram.T) * 0.5
    eigval, eigvec = eigh(gram, check_finite=False, driver="evd")
    eigval = np.maximum(eigval, 0.0)
    projected_rhs = eigvec.T @ rhs
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "constant_mask": constant,
        "y_mean": y_mean,
        "eigenvalues": eigval,
        "eigenvectors": eigvec,
        "projected_rhs": projected_rhs,
        "gram": gram,
        "rhs": rhs,
        "weight_sum": total,
        "rows": len(x),
    }


def solve_from_gram(prepared: dict[str, Any], lam: float) -> dict[str, Any]:
    if lam <= 0:
        raise ValueError("lambda must be positive")
    eigval = prepared["eigenvalues"]
    eigvec = prepared["eigenvectors"]
    projected_rhs = prepared["projected_rhs"]
    coef = eigvec @ (projected_rhs / (eigval[:, None] + lam))
    intercept = prepared["y_mean"]
    gram = prepared["gram"]
    rhs = prepared["rhs"]
    residual = (gram + lam * np.eye(len(eigval))) @ coef - rhs
    residual_ratio = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), EPS))
    condition = float((eigval[-1] + lam) / (eigval[0] + lam))
    return {
        "coef": coef,
        "intercept": intercept,
        "lambda": float(lam),
        "solver_residual_ratio": residual_ratio,
        "condition_number_regularized_gram": condition,
        "effective_alpha_sum_loss_convention": float(prepared["weight_sum"] * lam),
    }


def predict_residual(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    z = standardize_features(x, model["x_mean"], model["x_scale"], model["constant_mask"])
    return z @ model["coef"] + model["intercept"]


def predict_native(x: np.ndarray, anchor: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    residual = predict_residual(x, model)
    n = len(anchor)
    if residual.shape != (n, 30 * 28) or np.asarray(anchor).shape != (n, 28):
        raise ValueError("residual/anchor shape mismatch")
    return np.asarray(anchor, dtype=np.float64)[:, None, :] + residual.reshape(n, 30, 28)


def episode_equal_metric(values: np.ndarray, episode_ids: np.ndarray) -> tuple[float, dict[int, float]]:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    if values.shape[0] != len(episodes):
        raise ValueError("metric and episode IDs differ in length")
    out = {}
    for ep in np.unique(episodes):
        x = values[episodes == ep]
        x = x[np.isfinite(x)]
        if x.size:
            out[int(ep)] = float(x.mean())
    return (float(np.mean(list(out.values()))) if out else float("nan"), out)


def safe_ratio_skill(model_loss: float, persistence_loss: float) -> float:
    if not np.isfinite(model_loss) or not np.isfinite(persistence_loss) or persistence_loss == 0:
        return float("nan")
    return float(1.0 - model_loss / persistence_loss)


def horizon_window_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    sigma: np.ndarray,
    safe_fd: np.ndarray,
    active_threshold: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    safe_fd = np.asarray(safe_fd, dtype=np.float64)
    threshold = np.asarray(active_threshold, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (30, 28):
        raise ValueError("prediction/target shape must be N x 30 x 28")
    if anchor.shape != (len(pred), 28) or sigma.shape != (28,) or safe_fd.shape != (28,) or threshold.shape != (28,):
        raise ValueError("anchor/normalizer/threshold shape mismatch")
    if not 1 <= horizon <= 30:
        raise ValueError("horizon outside fixed H30 output")
    p = pred[:, :horizon]
    y = target[:, :horizon]
    e = p - y
    std_e = e / sigma[None, None, :]
    out: dict[str, np.ndarray] = {
        "endpoint_position_standardized_mae": np.abs(std_e[:, -1]).mean(axis=1),
        "prefix_position_standardized_mae": np.abs(std_e).mean(axis=(1, 2)),
        "endpoint_position_native_mae": np.abs(e[:, -1]).mean(axis=1),
        "prefix_position_native_mae": np.abs(e).mean(axis=(1, 2)),
        "seam_error_native": np.abs((p[:, 0] - anchor) - (y[:, 0] - anchor)).mean(axis=1),
        "seam_error_safe_normalized": (np.abs((p[:, 0] - anchor) - (y[:, 0] - anchor)) / safe_fd[None, :]).mean(axis=1),
        "seam_pred_step_abs_native": np.abs(p[:, 0] - anchor).mean(axis=1),
        "seam_gt_step_abs_native": np.abs(y[:, 0] - anchor).mean(axis=1),
        "future_internal_difference_count": np.full(len(pred), horizon - 1, dtype=np.int64),
    }
    if horizon == 1:
        for key in ("future_internal_fd_native_mae", "future_internal_fd_safe_normalized", "tv_pred", "tv_gt", "direction_accuracy_active", "positive_recall", "negative_recall", "balanced_direction_accuracy"):
            out[key] = np.full(len(pred), np.nan, dtype=np.float64)
        out["tv_ratio_window"] = np.full(len(pred), np.nan, dtype=np.float64)
        out["active_components"] = np.zeros(len(pred), dtype=np.int64)
        out["positive_components"] = np.zeros(len(pred), dtype=np.int64)
        out["negative_components"] = np.zeros(len(pred), dtype=np.int64)
        out["positive_correct"] = np.zeros(len(pred), dtype=np.int64)
        out["negative_correct"] = np.zeros(len(pred), dtype=np.int64)
        out["active_correct"] = np.zeros(len(pred), dtype=np.int64)
        out["zero_gt_tv_windows"] = np.zeros(len(pred), dtype=np.int64)
        return out
    dp = np.diff(p, axis=1)
    dy = np.diff(y, axis=1)
    de = dp - dy
    out["future_internal_fd_native_mae"] = np.abs(de).mean(axis=(1, 2))
    out["future_internal_fd_safe_normalized"] = (np.abs(de) / safe_fd[None, None, :]).mean(axis=(1, 2))
    tvp = np.abs(dp).sum(axis=(1, 2)) / (horizon - 1)
    tvy = np.abs(dy).sum(axis=(1, 2)) / (horizon - 1)
    out["tv_pred"] = tvp
    out["tv_gt"] = tvy
    out["tv_ratio_window"] = np.divide(tvp, tvy, out=np.full(len(pred), np.nan), where=tvy > 0)
    active = np.abs(dy) > threshold[None, None, :]
    positive = dy > threshold[None, None, :]
    negative = dy < -threshold[None, None, :]
    pred_sign = np.sign(dp)
    active_count = active.sum(axis=(1, 2))
    correct = (pred_sign == np.sign(dy)) & active
    pos_count = positive.sum(axis=(1, 2))
    neg_count = negative.sum(axis=(1, 2))
    pos_correct = ((pred_sign == 1) & positive).sum(axis=(1, 2))
    neg_correct = ((pred_sign == -1) & negative).sum(axis=(1, 2))
    out["active_components"] = active_count.astype(np.int64)
    out["positive_components"] = pos_count.astype(np.int64)
    out["negative_components"] = neg_count.astype(np.int64)
    out["active_correct"] = (correct.sum(axis=(1, 2))).astype(np.int64)
    out["positive_correct"] = pos_correct.astype(np.int64)
    out["negative_correct"] = neg_correct.astype(np.int64)
    out["direction_accuracy_active"] = np.divide(correct.sum(axis=(1, 2)), active_count, out=np.full(len(pred), np.nan), where=active_count > 0)
    pos_recall = np.divide(pos_correct, pos_count, out=np.full(len(pred), np.nan), where=pos_count > 0)
    neg_recall = np.divide(neg_correct, neg_count, out=np.full(len(pred), np.nan), where=neg_count > 0)
    out["positive_recall"] = pos_recall
    out["negative_recall"] = neg_recall
    out["balanced_direction_accuracy"] = np.where((pos_count > 0) & (neg_count > 0), 0.5 * (pos_recall + neg_recall), np.nan)
    out["zero_gt_tv_windows"] = (tvy == 0).astype(np.int64)
    return out


def episode_direction_metrics(metrics: dict[str, np.ndarray], episode_ids: np.ndarray) -> dict[int, dict[str, float]]:
    episodes = np.asarray(episode_ids)
    out: dict[int, dict[str, float]] = {}
    for ep in np.unique(episodes):
        rows = episodes == ep
        active = int(np.sum(metrics["active_components"][rows]))
        correct = int(np.sum(metrics["active_correct"][rows]))
        pos = int(np.sum(metrics["positive_components"][rows]))
        pos_correct = int(np.sum(metrics["positive_correct"][rows]))
        neg = int(np.sum(metrics["negative_components"][rows]))
        neg_correct = int(np.sum(metrics["negative_correct"][rows]))
        pr = float(pos_correct / pos) if pos else float("nan")
        nr = float(neg_correct / neg) if neg else float("nan")
        out[int(ep)] = {
            "direction_accuracy_active": float(correct / active) if active else float("nan"),
            "positive_recall": pr,
            "negative_recall": nr,
            "balanced_direction_accuracy": float(0.5 * (pr + nr)) if pos and neg else float("nan"),
            "active_components": active,
            "positive_components": pos,
            "negative_components": neg,
        }
    return out


def bootstrap_group_differences(
    group_a: dict[str, float],
    group_b: dict[str, float],
    group_ids: list[str],
    *,
    seed: int = 20261011,
    replicates: int = 10000,
) -> dict[str, float]:
    if set(group_a) != set(group_ids) or set(group_b) != set(group_ids):
        raise ValueError("paired bootstrap group keys differ")
    rng = np.random.default_rng(seed)
    n = len(group_ids)
    draws = rng.integers(0, n, size=(replicates, n))
    a = np.asarray([group_a[g] for g in group_ids], dtype=np.float64)
    b = np.asarray([group_b[g] for g in group_ids], dtype=np.float64)
    samples = (a[draws] - b[draws]).mean(axis=1)
    return {
        "mean_difference": float(np.mean(a - b)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_model_npz(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(
        path,
        x_mean=np.asarray(model["x_mean"], dtype=np.float64),
        x_scale=np.asarray(model["x_scale"], dtype=np.float64),
        constant_mask=np.asarray(model["constant_mask"], dtype=np.bool_),
        coef=np.asarray(model["coef"], dtype=np.float64),
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        selected_lambda=np.asarray([model["lambda"]], dtype=np.float64),
    )


def load_model_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "x_mean": z["x_mean"].copy(),
            "x_scale": z["x_scale"].copy(),
            "constant_mask": z["constant_mask"].copy(),
            "coef": z["coef"].copy(),
            "intercept": z["intercept"].copy(),
            "lambda": float(z["selected_lambda"][0]),
        }

