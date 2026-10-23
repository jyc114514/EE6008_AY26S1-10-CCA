#!/usr/bin/env python3
"""SIV Full predictor with the locked observation-age embedding."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from model_v3_source import V3Full28Predictor, parameter_count, sinusoidal, state_hash  # noqa: E402


WIDTH = 128
VIDEO_DIM = 768
VIDEO_STEPS = 16
ACTION_DIM = 28
HISTORY_STEPS = 32
STATE_DIM = 71


class DelayAwarePredictor(V3Full28Predictor):
    """Same SIV architecture plus one shared observation-age Linear(1,128)."""

    def __init__(self, *, horizon: int, seed: int) -> None:
        super().__init__(horizon=horizon, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7919)
            self.observation_age_projection = nn.Linear(1, WIDTH, bias=False)
            self.observation_age_projection.reset_parameters()

    def forward_delay(self, state: Tensor, history_mask: Tensor, relative_delta: Tensor, observation_age: Tensor, video_tokens: Tensor, anchor_command: Tensor) -> Tensor:
        if state.shape[1:] != (HISTORY_STEPS, STATE_DIM): raise ValueError(f"state shape {state.shape}")
        if observation_age.shape[1:] != (HISTORY_STEPS,): raise ValueError(f"age shape {observation_age.shape}")
        if video_tokens.shape[1:] != (VIDEO_STEPS, VIDEO_DIM): raise ValueError(f"video shape {video_tokens.shape}")
        if anchor_command.shape[1:] != (ACTION_DIM,): raise ValueError(f"anchor shape {anchor_command.shape}")
        batch = state.shape[0]
        state_meta = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
        state_tokens = self.state_projection(torch.cat((state, state_meta), dim=-1))
        state_tokens = state_tokens + sinusoidal(relative_delta * 8.0, WIDTH) + self.state_modality
        state_tokens = state_tokens + self.observation_age_projection(observation_age.unsqueeze(-1))
        state_tokens = state_tokens.transpose(1, 2)
        for block in self.tcn_blocks: state_tokens = block(state_tokens)
        state_tokens = state_tokens.transpose(1, 2)
        video_projected = self.video_projection(video_tokens) + sinusoidal(torch.linspace(-31.0, 0.0, VIDEO_STEPS, device=state.device).view(1, -1).expand(batch, -1), WIDTH) + self.video_modality
        anchor_token = self.anchor_projection(anchor_command).unsqueeze(1) + self.anchor_modality
        memory = torch.cat((state_tokens, video_projected, anchor_token), dim=1)
        query_times = torch.arange(1, self.horizon + 1, device=state.device, dtype=torch.float32) / 30.0
        queries = sinusoidal(query_times.view(1, -1).expand(batch, -1), WIDTH)
        decoded = self.decoder(queries, memory)
        return self.output(decoded)


__all__ = ["DelayAwarePredictor", "parameter_count", "state_hash"]
