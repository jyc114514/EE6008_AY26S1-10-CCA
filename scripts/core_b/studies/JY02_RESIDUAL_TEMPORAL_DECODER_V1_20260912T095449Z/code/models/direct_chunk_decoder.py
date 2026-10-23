from __future__ import annotations

import torch
from torch import nn


class DirectChunkDecoder(nn.Module):
    """Canonical pooled direct head: one hidden GELU layer, 30x28 output."""

    def __init__(self, input_dim: int, *, seed: int | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 30 * 28))
        if seed is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                for module in self.net:
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context).reshape(-1, 30, 28)
