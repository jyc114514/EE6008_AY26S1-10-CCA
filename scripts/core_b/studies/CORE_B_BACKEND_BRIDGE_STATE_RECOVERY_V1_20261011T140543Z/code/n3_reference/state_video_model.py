#!/usr/bin/env python3
"""V3 predictor wrapper with locked state-information and video masks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor


CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from model_v3_source import V3Full28Predictor, parameter_count, state_hash  # noqa: E402


STATE_LEVELS = ("FULL", "OBS", "ANCHOR")
VIDEO_LEVELS = ("N", "D", "S")
ACTION_START = 43
STATE_DIM = 71


class StateVideoPredictor(V3Full28Predictor):
    """Same V3 parameterization; only declared input values are masked."""

    def masked_inputs(
        self,
        state: Tensor,
        video_tokens: Tensor,
        *,
        state_level: str,
        video_level: str,
    ) -> tuple[Tensor, Tensor]:
        if state_level not in STATE_LEVELS:
            raise ValueError(state_level)
        if video_level not in VIDEO_LEVELS:
            raise ValueError(video_level)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"unexpected state width: {state.shape}")
        if state_level == "FULL":
            masked_state = state
        elif state_level == "OBS":
            masked_state = state.clone()
            masked_state[..., ACTION_START:] = 0.0
        else:  # ANCHOR: state/action history branch is entirely removed.
            masked_state = torch.zeros_like(state)
        if video_level == "N":
            masked_video = torch.zeros_like(video_tokens)
        else:
            masked_video = video_tokens
        return masked_state, masked_video

    def forward_locked(
        self,
        state: Tensor,
        history_mask: Tensor,
        relative_delta: Tensor,
        video_tokens: Tensor,
        anchor_command: Tensor,
        *,
        state_level: str,
        video_level: str,
    ) -> Tensor:
        masked_state, masked_video = self.masked_inputs(
            state, video_tokens, state_level=state_level, video_level=video_level
        )
        # The independent anchor token is always passed through unchanged.
        return super().forward(
            masked_state,
            history_mask,
            relative_delta,
            masked_video,
            anchor_command,
            use_state=True,
            use_video=True,
        )


__all__ = ["StateVideoPredictor", "parameter_count", "state_hash", "STATE_LEVELS", "VIDEO_LEVELS"]
