"""Persistence-gated residual heads for the post-freeze exploratory screen."""

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
GATE_GROUPS = 2  # arm first, hand second


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


class SharedResidualHead(nn.Module):
    """One direct head producing a canonical 30x28 residual."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, *, seed: int) -> None:
        super().__init__()
        self.mlp = _MlpHead(input_dim, hidden_dim, STEPS * DIMS, seed=seed)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.mlp(context).reshape(-1, STEPS, DIMS)


class SplitResidualHead(nn.Module):
    """Two independent heads, returned in Hand[0:14], Arm[14:28] order."""

    def __init__(self, input_dim: int, hidden_dim: int, *, seed: int) -> None:
        super().__init__()
        self.hand = _MlpHead(input_dim, hidden_dim, STEPS * HAND_DIMS, seed=seed)
        self.arm = _MlpHead(input_dim, hidden_dim, STEPS * ARM_DIMS, seed=seed + 101)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hand = self.hand(context).reshape(-1, STEPS, HAND_DIMS)
        arm = self.arm(context).reshape(-1, STEPS, ARM_DIMS)
        return torch.cat((hand, arm), dim=-1)


class GroupTimePersistenceGate(nn.Module):
    """Two bounded group-time gates; output order is Arm then Hand."""

    def __init__(self, input_dim: int, *, seed: int) -> None:
        super().__init__()
        self.time_embedding = nn.Parameter(torch.empty(STEPS, TIME_EMBEDDING_DIM))
        with _seeded(seed):
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        self.projection = nn.Linear(input_dim + TIME_EMBEDDING_DIM, GATE_GROUPS)
        # The output gate starts at sigmoid(logit(0.9)) and learns from context.
        nn.init.zeros_(self.projection.weight)
        nn.init.constant_(self.projection.bias, math.log(0.9 / 0.1))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        context_steps = context.unsqueeze(1).expand(-1, STEPS, -1)
        time_steps = self.time_embedding.unsqueeze(0).expand(batch, -1, -1)
        logits = self.projection(torch.cat((context_steps, time_steps), dim=-1))
        return torch.sigmoid(logits)


class PersistenceGatedResidualModel(nn.Module):
    """Direct residual model with optional Arm/Hand group-time gating."""

    def __init__(
        self,
        input_dim: int,
        *,
        head_coupling: str,
        residual_gate: str,
        hidden_dim: int,
        seed: int,
    ) -> None:
        super().__init__()
        if head_coupling not in {"shared", "split"}:
            raise ValueError(f"unknown head coupling: {head_coupling}")
        if residual_gate not in {"ungated", "group_time_gated"}:
            raise ValueError(f"unknown residual gate: {residual_gate}")
        self.head_coupling = head_coupling
        self.residual_gate = residual_gate
        if head_coupling == "shared":
            self.head = SharedResidualHead(input_dim, hidden_dim, seed=seed + 11)
        else:
            self.head = SplitResidualHead(input_dim, hidden_dim, seed=seed + 11)
        self.gate = GroupTimePersistenceGate(input_dim, seed=seed + 211) if residual_gate == "group_time_gated" else None

    def raw_residual(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(context)

    def gate_values(self, context: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return torch.ones((context.shape[0], STEPS, GATE_GROUPS), device=context.device, dtype=context.dtype)
        return self.gate(context)

    @staticmethod
    def apply_group_gate(residual: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != DIMS or gates.shape[-1] != GATE_GROUPS:
            raise ValueError("canonical residual/gate dimensions are invalid")
        arm_gate = gates[:, :, 0:1]
        hand_gate = gates[:, :, 1:2]
        return torch.cat((residual[:, :, :HAND_DIMS] * hand_gate, residual[:, :, HAND_DIMS:] * arm_gate), dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        *,
        gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.raw_residual(context)
        gates = self.gate_values(context) if gate_override is None else gate_override
        if gates.shape != (context.shape[0], STEPS, GATE_GROUPS):
            raise ValueError(f"gate shape {tuple(gates.shape)} is invalid")
        if not torch.isfinite(gates).all() or torch.any(gates < 0) or torch.any(gates > 1):
            raise ValueError("gate values must be finite and in [0,1]")
        return self.apply_group_gate(residual, gates), gates


def parameter_count(input_dim: int, *, head_coupling: str, residual_gate: str, hidden_dim: int) -> int:
    model = PersistenceGatedResidualModel(
        input_dim,
        head_coupling=head_coupling,
        residual_gate=residual_gate,
        hidden_dim=hidden_dim,
        seed=1,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def split_hidden_for_parameter_match(input_dim: int, shared_hidden: int = 256) -> tuple[int, int]:
    """Return (hidden, total params) closest to the shared ungated head.

    The candidate list is fixed before development performance is read.
    """

    candidates = (128, 160, 192, 224, 256)
    shared = parameter_count(input_dim, head_coupling="shared", residual_gate="ungated", hidden_dim=shared_hidden)
    scored = []
    for candidate in candidates:
        count = parameter_count(input_dim, head_coupling="split", residual_gate="ungated", hidden_dim=candidate)
        scored.append((abs(count - shared), candidate, count))
    _, hidden, count = min(scored)
    return hidden, count
