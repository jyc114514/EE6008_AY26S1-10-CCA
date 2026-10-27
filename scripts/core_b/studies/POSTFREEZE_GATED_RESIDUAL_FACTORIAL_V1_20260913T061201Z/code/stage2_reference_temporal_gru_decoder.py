from __future__ import annotations

import torch
from torch import nn


class TemporalGRUDecoder(nn.Module):
    """Non-teacher-forced temporal decoder with no predicted feedback."""

    def __init__(self, input_dim: int, hidden_size: int, *, seed_base: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.context_projection = nn.Linear(input_dim, hidden_size)
        self.time_embedding = nn.Parameter(torch.empty(30, 64))
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 28)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + 1)
            self.context_projection.reset_parameters()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + 2)
            nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + 3)
            self.gru.reset_parameters()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + 4)
            self.output.reset_parameters()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.context_projection(context)).unsqueeze(0)
        time_inputs = self.time_embedding.unsqueeze(0).expand(context.shape[0], -1, -1)
        decoded, _ = self.gru(time_inputs, hidden)
        return self.output(decoded)
