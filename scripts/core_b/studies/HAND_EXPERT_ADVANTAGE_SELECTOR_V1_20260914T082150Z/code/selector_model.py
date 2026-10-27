"""Matched past-context Hand expert selector head."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


STEPS = 30
TIME_DIM = 32
CONTEXT_DIM = 1336
FEATURE_DIM = CONTEXT_DIM + 1 + 1 + TIME_DIM
HIDDEN_DIM = 64


@contextmanager
def _seeded(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _reset_linear(linear: nn.Linear, seed: int) -> None:
    with _seeded(seed):
        linear.reset_parameters()


def selector_feature_matrix(context: np.ndarray, state_summary: np.ndarray, residual_magnitude: np.ndarray, time_features: np.ndarray | None = None) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    state_summary = np.asarray(state_summary, dtype=np.float32)
    residual_magnitude = np.asarray(residual_magnitude, dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != CONTEXT_DIM or state_summary.shape != (len(context), 1) or residual_magnitude.shape != (len(context), STEPS, 1):
        raise ValueError("selector feature inputs have an invalid shape")
    if time_features is None:
        time_features = np.zeros((len(context), STEPS, TIME_DIM), dtype=np.float32)
    time_features = np.asarray(time_features, dtype=np.float32)
    if time_features.shape != (len(context), STEPS, TIME_DIM):
        raise ValueError("time feature shape is invalid")
    context_steps = np.repeat(context[:, None, :], STEPS, axis=1)
    summary_steps = np.repeat(state_summary[:, None, :], STEPS, axis=1)
    return np.concatenate((context_steps, summary_steps, residual_magnitude, time_features), axis=2)


def compose_hand_prediction(persistence: np.ndarray, neural: np.ndarray, q: np.ndarray) -> np.ndarray:
    persistence = np.asarray(persistence)
    neural = np.asarray(neural)
    q = np.asarray(q)
    if persistence.shape != neural.shape or persistence.ndim != 3 or persistence.shape[1:] != (STEPS, 14) or q.shape != (persistence.shape[0], STEPS, 1):
        raise ValueError("persistence/neural/q shapes are invalid")
    if not np.isfinite(persistence).all() or not np.isfinite(neural).all() or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError("selector inputs must be finite and q must be in [0,1]")
    return persistence + q * (neural - persistence)


class SelectorHead(nn.Module):
    """A small selector-only head; base expert and Arm never enter this module."""

    def __init__(self, *, seed: int, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.time_embedding = nn.Parameter(torch.empty(STEPS, TIME_DIM))
        with _seeded(seed):
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        self.input = nn.Linear(FEATURE_DIM, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)
        _reset_linear(self.input, seed + 1)
        _reset_linear(self.output, seed + 2)

    def forward(self, context: torch.Tensor, state_summary: torch.Tensor, residual_magnitude: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != CONTEXT_DIM or state_summary.shape != (context.shape[0], 1) or residual_magnitude.shape != (context.shape[0], STEPS, 1):
            raise ValueError("selector model input shape is invalid")
        time_steps = self.time_embedding.unsqueeze(0).expand(context.shape[0], -1, -1)
        context_steps = context.unsqueeze(1).expand(-1, STEPS, -1)
        summary_steps = state_summary.unsqueeze(1).expand(-1, STEPS, -1)
        features = torch.cat((context_steps, summary_steps, residual_magnitude, time_steps), dim=-1)
        return self.output(self.activation(self.input(features)))

    def probability(self, context: torch.Tensor, state_summary: torch.Tensor, residual_magnitude: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(context, state_summary, residual_magnitude))


def state_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8")); hasher.update(value.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
