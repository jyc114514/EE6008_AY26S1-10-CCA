"""Matched Hand decoder, selector and locked dual-loss primitives."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STEPS = 30
HAND_DIMS = 14
CONTEXT_DIM = 1336
DECODER_HIDDEN_DIM = 160
SELECTOR_TIME_DIM = 32
SELECTOR_HIDDEN_DIM = 64
SELECTOR_FEATURE_DIM = CONTEXT_DIM + 1 + 1 + SELECTOR_TIME_DIM
PROTOTYPE_K = 2
PROTOTYPE_COEFFICIENTS = 16


@contextmanager
def seeded_torch(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def reset_linear(linear: nn.Linear, seed: int) -> None:
    with seeded_torch(seed):
        linear.reset_parameters()


def dct_basis(length: int = STEPS) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    frequency = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(torch.pi / length * (time + 0.5) * frequency) * torch.sqrt(torch.tensor(2.0 / length))
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


class HandDecoder(nn.Module):
    """D0 direct or D1 fixed-DCT residual decoder."""

    def __init__(self, *, decoder: str, seed: int, dct_k: int = 16, hidden_dim: int = DECODER_HIDDEN_DIM) -> None:
        super().__init__()
        if decoder not in {"D0", "D1"}:
            raise ValueError(f"unknown decoder {decoder}")
        if decoder == "D1" and not 0 < dct_k <= STEPS:
            raise ValueError("DCT K must be in 1..30")
        self.decoder = decoder
        self.dct_k = dct_k
        self.input = nn.Linear(CONTEXT_DIM, hidden_dim)
        self.activation = nn.GELU()
        output_dim = STEPS * HAND_DIMS if decoder == "D0" else dct_k * HAND_DIMS
        self.output = nn.Linear(hidden_dim, output_dim)
        reset_linear(self.input, seed)
        reset_linear(self.output, seed + 1)
        self.register_buffer("dct_basis", dct_basis(STEPS), persistent=True)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (STEPS, HAND_DIMS) if self.decoder == "D0" else (self.dct_k, HAND_DIMS)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != CONTEXT_DIM:
            raise ValueError(f"context must be (B,{CONTEXT_DIM})")
        raw = self.output(self.activation(self.input(context)))
        if self.decoder == "D0":
            return raw.reshape(-1, STEPS, HAND_DIMS)
        coefficients = raw.reshape(-1, self.dct_k, HAND_DIMS).float()
        # FP32 inverse DCT is explicit even when the caller uses autocast.
        return torch.einsum("tk,bkj->btj", self.dct_basis[: self.dct_k, :].T, coefficients)


class SelectorHead(nn.Module):
    """The matched reconstruction-trained raw-q Hand selector."""

    def __init__(self, *, seed: int, hidden_dim: int = SELECTOR_HIDDEN_DIM) -> None:
        super().__init__()
        self.time_embedding = nn.Parameter(torch.empty(STEPS, SELECTOR_TIME_DIM))
        with seeded_torch(seed):
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        self.input = nn.Linear(SELECTOR_FEATURE_DIM, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, 1)
        reset_linear(self.input, seed + 1)
        reset_linear(self.output, seed + 2)

    def forward(self, context: torch.Tensor, state_summary: torch.Tensor, residual_magnitude: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != CONTEXT_DIM or state_summary.shape != (len(context), 1) or residual_magnitude.shape != (len(context), STEPS, 1):
            raise ValueError("selector input shapes are invalid")
        time_steps = self.time_embedding.unsqueeze(0).expand(context.shape[0], -1, -1)
        context_steps = context.unsqueeze(1).expand(-1, STEPS, -1)
        summary_steps = state_summary.unsqueeze(1).expand(-1, STEPS, -1)
        features = torch.cat((context_steps, summary_steps, residual_magnitude, time_steps), dim=-1)
        return self.output(self.activation(self.input(features)))

    def probability(self, context: torch.Tensor, state_summary: torch.Tensor, residual_magnitude: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(context, state_summary, residual_magnitude))


class ConditionalPrototypeDecoder(nn.Module):
    """Past-context mode logits plus one shared DCT coefficient refinement."""

    def __init__(self, *, centers: torch.Tensor, seed: int, hidden_dim: int = DECODER_HIDDEN_DIM) -> None:
        super().__init__()
        if centers.ndim != 3 or centers.shape[1:] != (PROTOTYPE_COEFFICIENTS, HAND_DIMS) or centers.shape[0] < 2:
            raise ValueError("prototype centers must be (K,16,14)")
        self.mode_count = int(centers.shape[0])
        self.input = nn.Linear(CONTEXT_DIM, hidden_dim)
        self.activation = nn.GELU()
        self.mode_output = nn.Linear(hidden_dim, self.mode_count)
        self.refinement_output = nn.Linear(hidden_dim, PROTOTYPE_COEFFICIENTS * HAND_DIMS)
        self.register_buffer("centers", centers.detach().float().clone(), persistent=True)
        self.register_buffer("dct_basis", dct_basis(STEPS), persistent=True)
        reset_linear(self.input, seed)
        reset_linear(self.mode_output, seed + 1)
        reset_linear(self.refinement_output, seed + 2)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 2 or context.shape[1] != CONTEXT_DIM:
            raise ValueError(f"context must be (B,{CONTEXT_DIM})")
        hidden = self.activation(self.input(context))
        logits = self.mode_output(hidden)
        probabilities = torch.softmax(logits, dim=-1)
        delta = self.refinement_output(hidden).reshape(-1, PROTOTYPE_COEFFICIENTS, HAND_DIMS)
        coefficients = self.centers.unsqueeze(0) + delta.unsqueeze(1)
        residual = torch.einsum("tk,bmkh->bmth", self.dct_basis[:PROTOTYPE_COEFFICIENTS, :].T, coefficients)
        return probabilities, coefficients, residual


def prototype_components_standardized(
    persistence_native: torch.Tensor,
    coefficients: torch.Tensor,
    mean_hand: torch.Tensor,
    scale_hand: torch.Tensor,
) -> torch.Tensor:
    """Convert prototype native command trajectories to standardized space."""

    if persistence_native.ndim != 3 or persistence_native.shape[1:] != (STEPS, HAND_DIMS):
        raise ValueError("persistence_native must be (B,30,14)")
    if coefficients.ndim != 4 or coefficients.shape[2:] != (PROTOTYPE_COEFFICIENTS, HAND_DIMS):
        raise ValueError("coefficients must be (B,K,16,14)")
    if mean_hand.shape != (STEPS, HAND_DIMS) or scale_hand.shape != (STEPS, HAND_DIMS):
        raise ValueError("normalizer shape is invalid")
    basis = dct_basis(STEPS)[:PROTOTYPE_COEFFICIENTS, :].T.to(coefficients.device)
    residual = torch.einsum("tk,bmkh->bmth", basis, coefficients)
    native = persistence_native.unsqueeze(1) + residual
    return (native - mean_hand.unsqueeze(0).unsqueeze(0)) / scale_hand.unsqueeze(0).unsqueeze(0)


def compose_hand_prediction(persistence: torch.Tensor, neural: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if persistence.shape != neural.shape or persistence.ndim != 3 or persistence.shape[1:] != (STEPS, HAND_DIMS) or q.shape != (len(persistence), STEPS, 1):
        raise ValueError("persistence/neural/q shapes are invalid")
    return persistence + q * (neural - persistence)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_components(
    hand_prediction_std: torch.Tensor,
    target_std: torch.Tensor,
    target_native: torch.Tensor,
    mean_hand: torch.Tensor,
    scale_hand: torch.Tensor,
    safe_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical position MAE and train-scaled native FD MAE."""

    if hand_prediction_std.shape != target_std.shape or hand_prediction_std.shape[1:] != (STEPS, HAND_DIMS):
        raise ValueError("loss prediction/target shape is invalid")
    position = F.l1_loss(hand_prediction_std, target_std, reduction="mean")
    prediction_native = hand_prediction_std.float() * scale_hand + mean_hand
    delta_prediction = prediction_native[:, 1:] - prediction_native[:, :-1]
    delta_target = target_native[:, 1:] - target_native[:, :-1]
    fd_scaled = torch.mean(torch.abs(delta_prediction - delta_target) / safe_scale)
    return position, fd_scaled


def total_loss(
    loss_mode: str,
    hand_prediction_std: torch.Tensor,
    target_std: torch.Tensor,
    target_native: torch.Tensor,
    mean_hand: torch.Tensor,
    scale_hand: torch.Tensor,
    safe_scale: torch.Tensor,
    lambda_fd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position, fd_scaled = loss_components(hand_prediction_std, target_std, target_native, mean_hand, scale_hand, safe_scale)
    if loss_mode == "L0":
        return position, position, fd_scaled
    if loss_mode == "L1":
        return position + lambda_fd * fd_scaled, position, fd_scaled
    raise ValueError(f"unknown loss mode {loss_mode}")
