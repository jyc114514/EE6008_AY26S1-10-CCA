"""Metrics and state helpers for the full frozen attentive comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from ee6008.metrics import classification_metrics


def extended_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    calibration_bins: int = 15,
) -> dict[str, float | int]:
    """Return classification and fixed-bin calibration metrics.

    ECE uses ``calibration_bins`` equal-width confidence bins in [0, 1].
    Brier is the multiclass mean squared probability error summed over classes.
    """
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels have incompatible shapes")
    if logits.shape[1] != num_classes:
        raise ValueError("logit dimension does not match num_classes")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("logits contain NaN or Inf")

    logits = logits.float()
    labels = labels.long()
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    one_hot = F.one_hot(labels, num_classes=num_classes).to(probabilities.dtype)
    ece = torch.zeros((), dtype=probabilities.dtype)
    for bin_index in range(calibration_bins):
        lower = bin_index / calibration_bins
        upper = (bin_index + 1) / calibration_bins
        if bin_index == calibration_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if in_bin.any():
            weight = in_bin.float().mean()
            accuracy = correct[in_bin].float().mean()
            mean_confidence = confidence[in_bin].mean()
            ece = ece + weight * (accuracy - mean_confidence).abs()

    result: dict[str, float | int] = {
        **classification_metrics(logits, labels, num_classes),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item()),
        "ece": float(ece.item()),
        "mean_max_softmax_probability": float(confidence.mean().item()),
        "correct": int(correct.sum().item()),
        "total": int(labels.numel()),
    }
    return result


@dataclass
class EarlyStoppingState:
    """Serializable macro-F1/NLL early-stopping state."""

    best_epoch: int | None = None
    best_macro_f1: float = -math.inf
    best_nll: float = math.inf
    bad_epochs: int = 0
    stop_reason: str | None = None

    def update(
        self, *, epoch: int, macro_f1: float, nll: float, min_delta: float
    ) -> bool:
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.best_epoch is None:
            improved = True
        else:
            macro_difference = macro_f1 - self.best_macro_f1
            improved = macro_difference > min_delta or (
                abs(macro_difference) <= min_delta and nll < self.best_nll
            )
        if improved:
            self.best_epoch = epoch
            self.best_macro_f1 = float(macro_f1)
            self.best_nll = float(nll)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self, *, epoch: int, min_epochs: int, patience: int) -> bool:
        if min_epochs <= 0 or patience <= 0:
            raise ValueError("min_epochs and patience must be positive")
        return epoch >= min_epochs and self.bad_epochs >= patience

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EarlyStoppingState:
        return cls(
            best_epoch=(
                int(value["best_epoch"])
                if value.get("best_epoch") is not None
                else None
            ),
            best_macro_f1=float(value.get("best_macro_f1", -math.inf)),
            best_nll=float(value.get("best_nll", math.inf)),
            bad_epochs=int(value.get("bad_epochs", 0)),
            stop_reason=value.get("stop_reason"),
        )


def validate_token_shape(
    tokens: torch.Tensor,
    *,
    expected: tuple[int, int] = (9216, 768),
) -> None:
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected:
        actual = tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else None
        raise ValueError(f"expected token shape {expected}, got {actual}")
    if not torch.isfinite(tokens).all():
        raise FloatingPointError("tokens contain NaN or Inf")


def repeat_first_frame(clip: torch.Tensor) -> torch.Tensor:
    """Return a copy whose temporal frames all equal the first frame."""
    if clip.ndim != 4:
        raise ValueError("clip must have shape C,T,H,W")
    result = clip.clone()
    result[:, 1:] = result[:, :1]
    return result
