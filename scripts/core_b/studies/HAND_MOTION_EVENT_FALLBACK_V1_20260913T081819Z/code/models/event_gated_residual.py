"""Split residual decoder with a locked Arm gate and Hand event gate."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn


STEPS = 30
DIMS = 28
HAND_DIMS = 14
ARM_DIMS = 14
TIME_EMBEDDING_DIM = 32
GATE_GROUPS = 2  # Arm first, Hand second
EVENT_MODES = ("U", "W", "T")


@contextmanager
def _seeded(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _reset_linear(linear: nn.Linear, seed: int) -> None:
    with _seeded(seed):
        linear.reset_parameters()


class _MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, seed: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, output_dim)
        _reset_linear(self.input, seed)
        _reset_linear(self.output, seed + 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(context)))


class SplitResidualHead(nn.Module):
    """Independent Hand and Arm residual heads in canonical joint order."""

    def __init__(self, input_dim: int, hidden_dim: int, *, seed: int) -> None:
        super().__init__()
        self.hand = _MlpHead(input_dim, hidden_dim, STEPS * HAND_DIMS, seed=seed)
        self.arm = _MlpHead(input_dim, hidden_dim, STEPS * ARM_DIMS, seed=seed + 101)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hand = self.hand(context).reshape(-1, STEPS, HAND_DIMS)
        arm = self.arm(context).reshape(-1, STEPS, ARM_DIMS)
        return torch.cat((hand, arm), dim=-1)


class GroupTimePersistenceGate(nn.Module):
    """The existing two-group time gate; output order is Arm then Hand."""

    def __init__(self, input_dim: int, *, seed: int) -> None:
        super().__init__()
        self.time_embedding = nn.Parameter(torch.empty(STEPS, TIME_EMBEDDING_DIM))
        with _seeded(seed):
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        self.projection = nn.Linear(input_dim + TIME_EMBEDDING_DIM, GATE_GROUPS)
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, math.log(0.9 / 0.1))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        context_steps = context.unsqueeze(1).expand(-1, STEPS, -1)
        time_steps = self.time_embedding.unsqueeze(0).expand(batch, -1, -1)
        logits = self.projection(torch.cat((context_steps, time_steps), dim=-1))
        return torch.sigmoid(logits)


class HandEventGate(nn.Module):
    """Small context-only event gate for W (window) or T (per-step) labels."""

    def __init__(self, input_dim: int, mode: str, *, seed: int, train_prevalence: float) -> None:
        super().__init__()
        if mode not in {"W", "T"}:
            raise ValueError(f"event branch is only defined for W/T, got {mode}")
        self.mode = mode
        self.train_prevalence = float(train_prevalence)
        if not 0.0 < self.train_prevalence < 1.0:
            raise ValueError("event prevalence must be strictly between zero and one")
        if mode == "W":
            self.time_embedding = None
            self.projection = nn.Linear(input_dim, 1)
        else:
            self.time_embedding = nn.Parameter(torch.empty(STEPS, TIME_EMBEDDING_DIM))
            with _seeded(seed):
                nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
            self.projection = nn.Linear(input_dim + TIME_EMBEDDING_DIM, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, math.log(self.train_prevalence / (1.0 - self.train_prevalence)))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.mode == "W":
            return self.projection(context).squeeze(-1)
        batch = context.shape[0]
        context_steps = context.unsqueeze(1).expand(-1, STEPS, -1)
        time_steps = self.time_embedding.unsqueeze(0).expand(batch, -1, -1)
        return self.projection(torch.cat((context_steps, time_steps), dim=-1)).squeeze(-1)


class HandEventGatedResidualModel(nn.Module):
    """Residual prediction with an unchanged Arm gate and supervised Hand gate."""

    def __init__(self, input_dim: int, *, hidden_dim: int, event_mode: str, seed: int, train_event_prevalence: float) -> None:
        super().__init__()
        if event_mode not in EVENT_MODES:
            raise ValueError(f"unknown event mode: {event_mode}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.event_mode = event_mode
        self.head = SplitResidualHead(input_dim, hidden_dim, seed=seed + 11)
        self.base_gate = GroupTimePersistenceGate(input_dim, seed=seed + 211)
        self.event_gate = HandEventGate(input_dim, event_mode, seed=seed + 311, train_prevalence=train_event_prevalence) if event_mode in {"W", "T"} else None

    def raw_residual(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(context)

    def base_gate_values(self, context: torch.Tensor) -> torch.Tensor:
        return self.base_gate(context)

    def event_logits(self, context: torch.Tensor) -> torch.Tensor | None:
        return None if self.event_gate is None else self.event_gate(context)

    def gate_values_and_event(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base = self.base_gate_values(context)
        arm = base[:, :, 0:1]
        if self.event_mode == "U":
            hand = base[:, :, 1:2]
            event_probability = hand.squeeze(-1)
        else:
            logits = self.event_logits(context)
            assert logits is not None
            event_probability = torch.sigmoid(logits)
            hand = event_probability.unsqueeze(-1) if self.event_mode == "T" else event_probability[:, None, None].expand(-1, STEPS, -1)
        gates = torch.cat((arm, hand), dim=-1)
        return gates, event_probability

    @staticmethod
    def apply_group_gate(residual: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != DIMS or gates.shape[-1] != GATE_GROUPS:
            raise ValueError("canonical residual/gate dimensions are invalid")
        return torch.cat((residual[:, :, :HAND_DIMS] * gates[:, :, 1:2], residual[:, :, HAND_DIMS:] * gates[:, :, 0:1]), dim=-1)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        residual = self.raw_residual(context)
        gates, event_probability = self.gate_values_and_event(context)
        if gates.shape != (context.shape[0], STEPS, GATE_GROUPS):
            raise ValueError(f"gate shape {tuple(gates.shape)} is invalid")
        if not torch.isfinite(gates).all() or torch.any(gates < 0) or torch.any(gates > 1):
            raise ValueError("gate values must be finite and in [0,1]")
        return self.apply_group_gate(residual, gates), gates, event_probability


def split_hidden_for_parameter_match(input_dim: int, shared_hidden: int = 256) -> tuple[int, int]:
    """Select the pre-locked split hidden width closest to the shared head."""
    candidates = (128, 160, 192, 224, 256)
    shared = (input_dim * shared_hidden + shared_hidden) + (shared_hidden * STEPS * DIMS + STEPS * DIMS)
    scored = []
    for hidden in candidates:
        count = 2 * ((input_dim * hidden + hidden) + (hidden * STEPS * HAND_DIMS + STEPS * HAND_DIMS))
        scored.append((abs(count - shared), hidden, count))
    _, hidden, count = min(scored)
    return hidden, count


def base_parameter_count(input_dim: int, hidden_dim: int) -> int:
    model = HandEventGatedResidualModel(input_dim, hidden_dim=hidden_dim, event_mode="U", seed=1, train_event_prevalence=0.25)
    return sum(parameter.numel() for parameter in model.parameters())


def event_parameter_count(input_dim: int, hidden_dim: int, event_mode: str) -> int:
    model = HandEventGatedResidualModel(input_dim, hidden_dim=hidden_dim, event_mode=event_mode, seed=1, train_event_prevalence=0.25)
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(input_dim: int, event_mode: str, seed: int, train_event_prevalence: float) -> tuple[HandEventGatedResidualModel, int, int]:
    hidden_dim, _ = split_hidden_for_parameter_match(input_dim)
    model = HandEventGatedResidualModel(input_dim, hidden_dim=hidden_dim, event_mode=event_mode, seed=seed, train_event_prevalence=train_event_prevalence)
    return model, hidden_dim, sum(parameter.numel() for parameter in model.parameters())
