#!/usr/bin/env python3
"""V3 Full-28 temporal predictor with fixed-capacity state/video memory."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_DIM = 71
VIDEO_DIM = 768
ACTION_DIM = 28
HISTORY_STEPS = 32
VIDEO_STEPS = 16
WIDTH = 128


def sinusoidal(times: Tensor, dim: int) -> Tensor:
    if dim % 2: raise ValueError("sinusoidal dimension must be even")
    scales = torch.exp(torch.arange(0, dim, 2, device=times.device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    values = times.float().unsqueeze(-1) * scales
    return torch.cat((torch.sin(values), torch.cos(values)), dim=-1)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.dropout(F.relu(self.conv(F.pad(values, (self.left_padding, 0)))))


class V3Full28Predictor(nn.Module):
    """Same-capacity architecture for all six modality configurations."""

    def __init__(self, *, horizon: int, seed: int, video_dim: int = VIDEO_DIM) -> None:
        super().__init__()
        if horizon not in (30, 60): raise ValueError(horizon)
        self.horizon = horizon
        self.video_dim = int(video_dim)
        if self.video_dim <= 0: raise ValueError("video_dim must be positive")
        self.state_projection = nn.Linear(STATE_DIM + 2, WIDTH)
        self.tcn_blocks = nn.ModuleList([CausalResidualBlock(WIDTH, dilation) for dilation in (1, 2, 4, 8, 16)])
        self.video_projection = nn.Linear(self.video_dim, WIDTH)
        self.anchor_projection = nn.Linear(ACTION_DIM, WIDTH)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=WIDTH, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True, activation="relu")
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.output = nn.Linear(WIDTH, ACTION_DIM)
        self.register_buffer("state_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        self.register_buffer("video_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        self.register_buffer("anchor_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        self._reset(seed)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def _reset(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for module in (self.state_projection, self.video_projection, self.anchor_projection): module.reset_parameters()
            for block in self.tcn_blocks: block.conv.reset_parameters()
            for module in self.decoder.modules():
                if isinstance(module, nn.Linear): module.reset_parameters()
                elif isinstance(module, nn.MultiheadAttention):
                    module._reset_parameters()

    def forward(self, state: Tensor, history_mask: Tensor, relative_delta: Tensor, video_tokens: Tensor, anchor_command: Tensor, *, use_state: bool, use_video: bool) -> Tensor:
        if state.shape[1:] != (HISTORY_STEPS, STATE_DIM): raise ValueError(f"state shape {state.shape}")
        if video_tokens.shape[1:] != (VIDEO_STEPS, self.video_dim): raise ValueError(f"video shape {video_tokens.shape}")
        if anchor_command.shape[1:] != (ACTION_DIM,): raise ValueError(f"anchor shape {anchor_command.shape}")
        batch = state.shape[0]
        state_values = state if use_state else torch.zeros_like(state)
        state_meta = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
        state_tokens = self.state_projection(torch.cat((state_values, state_meta), dim=-1))
        state_tokens = state_tokens + sinusoidal(relative_delta * 8.0, WIDTH) + self.state_modality
        state_tokens = state_tokens.transpose(1, 2)
        for block in self.tcn_blocks: state_tokens = block(state_tokens)
        state_tokens = state_tokens.transpose(1, 2)
        video_values = video_tokens if use_video else torch.zeros_like(video_tokens)
        video_tokens_projected = self.video_projection(video_values) + sinusoidal(torch.linspace(-31.0, 0.0, VIDEO_STEPS, device=state.device).view(1, -1).expand(batch, -1), WIDTH) + self.video_modality
        anchor_token = self.anchor_projection(anchor_command).unsqueeze(1) + self.anchor_modality
        memory = torch.cat((state_tokens, video_tokens_projected, anchor_token), dim=1)
        query_times = torch.arange(1, self.horizon + 1, device=state.device, dtype=torch.float32) / 30.0
        queries = sinusoidal(query_times.view(1, -1).expand(batch, -1), WIDTH)
        decoded = self.decoder(queries, memory)
        return self.output(decoded)


def state_hash(model: nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())
