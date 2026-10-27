"""Locked D3 time x 2x2-space predictor with common 64-slot capacity."""

from __future__ import annotations

import math
import hashlib

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_DIM = 71
ACTION_DIM = 28
VIDEO_DIM = 768
TIME_SLOTS = 16
SPACE_BINS = 4
VIDEO_SLOTS = TIME_SLOTS * SPACE_BINS
WIDTH = 128


def sinusoidal(times: Tensor, dim: int) -> Tensor:
    if dim % 2:
        raise ValueError("sinusoidal dimension must be even")
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


def spatial_encoding(device: torch.device) -> Tensor:
    centers = torch.tensor([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]], device=device)
    row = sinusoidal(centers[:, 0], 64)
    col = sinusoidal(centers[:, 1], 64)
    return torch.cat((row, col), dim=-1).view(1, 1, 4, WIDTH).expand(1, TIME_SLOTS, 4, WIDTH).reshape(1, VIDEO_SLOTS, WIDTH)


class D3TimeSpacePredictor(nn.Module):
    def __init__(self, *, horizon: int, seed: int, video_mode: str) -> None:
        super().__init__()
        if horizon not in (30, 60):
            raise ValueError(horizon)
        if video_mode not in {"TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"}:
            raise ValueError(video_mode)
        self.horizon = horizon
        self.video_mode = video_mode
        self.state_projection = nn.Linear(STATE_DIM + 2, WIDTH)
        self.tcn_blocks = nn.ModuleList([CausalResidualBlock(WIDTH, dilation) for dilation in (1, 2, 4, 8, 16)])
        self.video_projection = nn.Linear(VIDEO_DIM, WIDTH)
        self.anchor_projection = nn.Linear(ACTION_DIM, WIDTH)
        decoder_layer = nn.TransformerDecoderLayer(d_model=WIDTH, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True, activation="relu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output = nn.Linear(WIDTH, ACTION_DIM)
        self.register_buffer("state_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        self.register_buffer("video_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        self.register_buffer("anchor_modality", torch.zeros(1, 1, WIDTH), persistent=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in (self.state_projection, self.video_projection, self.anchor_projection):
                module.reset_parameters()
            for block in self.tcn_blocks:
                block.conv.reset_parameters()
            for module in self.decoder.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
                elif isinstance(module, nn.MultiheadAttention):
                    module._reset_parameters()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _video_layout(self, dynamic: Tensor, static: Tensor) -> Tensor:
        if dynamic.shape[1:] != (TIME_SLOTS, SPACE_BINS, VIDEO_DIM) or static.shape != dynamic.shape:
            raise ValueError(f"D3 grid shape mismatch: dynamic={tuple(dynamic.shape)} static={tuple(static.shape)}")
        source = static if self.video_mode == "TS11-S" else dynamic
        if self.video_mode == "STATE-CTRL":
            return torch.zeros_like(source).reshape(source.shape[0], VIDEO_SLOTS, VIDEO_DIM)
        if self.video_mode == "TS00-D":
            values = source.mean(dim=(1, 2), keepdim=True).expand(-1, TIME_SLOTS, SPACE_BINS, -1)
        elif self.video_mode == "TS10-D":
            values = source.mean(dim=2, keepdim=True).expand(-1, -1, SPACE_BINS, -1)
        elif self.video_mode == "TS01-D":
            values = source.mean(dim=1, keepdim=True).expand(-1, TIME_SLOTS, -1, -1)
        else:
            values = source
        return values.reshape(values.shape[0], VIDEO_SLOTS, VIDEO_DIM)

    def forward(self, state: Tensor, history_mask: Tensor, relative_delta: Tensor, dynamic: Tensor, static: Tensor, anchor_command: Tensor) -> Tensor:
        if state.shape[1:] != (32, STATE_DIM) or history_mask.shape[1:] != (32,) or relative_delta.shape[1:] != (32,) or anchor_command.shape[1:] != (ACTION_DIM,):
            raise ValueError("D3 state/anchor shape mismatch")
        batch = state.shape[0]
        state_meta = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
        state_tokens = self.state_projection(torch.cat((state, state_meta), dim=-1))
        state_tokens = state_tokens + sinusoidal(relative_delta * 8.0, WIDTH) + self.state_modality
        state_tokens = state_tokens.transpose(1, 2)
        for block in self.tcn_blocks:
            state_tokens = block(state_tokens)
        state_tokens = state_tokens.transpose(1, 2)
        video_values = self._video_layout(dynamic, static)
        time_values = torch.linspace(-31.0, 0.0, TIME_SLOTS, device=state.device).view(1, TIME_SLOTS, 1).expand(batch, TIME_SLOTS, SPACE_BINS).reshape(batch, VIDEO_SLOTS)
        video_tokens = self.video_projection(video_values) + sinusoidal(time_values, WIDTH) + spatial_encoding(state.device) + self.video_modality
        anchor_token = self.anchor_projection(anchor_command).unsqueeze(1) + self.anchor_modality
        memory = torch.cat((state_tokens, video_tokens, anchor_token), dim=1)
        query_times = torch.arange(1, self.horizon + 1, device=state.device, dtype=torch.float32) / 30.0
        queries = sinusoidal(query_times.view(1, -1).expand(batch, -1), WIDTH)
        return self.output(self.decoder(queries, memory))


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())
