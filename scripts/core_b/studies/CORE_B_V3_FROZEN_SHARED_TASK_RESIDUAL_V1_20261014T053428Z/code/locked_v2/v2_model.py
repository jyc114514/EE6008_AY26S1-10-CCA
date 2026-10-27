"""V2 task-conditioned Q0 model and prefix-only waypoint variants."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOCKED_SOURCE = Path(__file__).resolve().parent / "locked_source"
if str(LOCKED_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCKED_SOURCE))

from D3_model import (  # noqa: E402
    ACTION_DIM,
    SPACE_BINS,
    TIME_SLOTS,
    VIDEO_DIM,
    VIDEO_SLOTS,
    WIDTH,
    D3TimeSpacePredictor,
    parameter_count,
    sinusoidal,
    spatial_encoding,
    state_hash,
)


TASK_COUNT = 5
WAYPOINT_DIM = ACTION_DIM


class V2Output(NamedTuple):
    action_residual: Tensor
    goal_prediction: Tensor | None


class V2TaskConditionedPredictor(D3TimeSpacePredictor):
    """Accepted Q0 backbone with a fixed task token and optional predicted waypoint."""

    def __init__(
        self,
        *,
        seed: int,
        video_mode: str,
        waypoint_mode: str | None = None,
    ) -> None:
        super().__init__(horizon=30, seed=seed, video_mode=video_mode)
        if waypoint_mode not in {None, "AUX_ONLY", "GOAL_COND"}:
            raise ValueError(f"invalid waypoint_mode={waypoint_mode}")
        self.waypoint_mode = waypoint_mode
        self.task_projection = nn.Linear(TASK_COUNT, WIDTH, bias=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 104729)
            self.task_projection.reset_parameters()
        if waypoint_mode is not None:
            self.goal_head = nn.Sequential(
                nn.Linear(WIDTH, WIDTH),
                nn.ReLU(),
                nn.Linear(WIDTH, WAYPOINT_DIM),
            )
            self.goal_projection = nn.Linear(WAYPOINT_DIM, WIDTH, bias=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 130363)
                for module in self.goal_head.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 130364)
                self.goal_projection.reset_parameters()
        else:
            self.goal_head = None
            self.goal_projection = None

    def _encode_prefix(self, prefix_inputs: dict[str, Tensor], task_id: Tensor) -> Tensor:
        state = prefix_inputs["state"]
        history_mask = prefix_inputs["history_mask"]
        relative_delta = prefix_inputs["relative_delta"]
        anchor_command = prefix_inputs["anchor_command"]
        if state.ndim != 3 or state.shape[1:] != (32, 71):
            raise ValueError(f"state must be [B,32,71], got {tuple(state.shape)}")
        batch = state.shape[0]
        if history_mask.shape != (batch, 32) or relative_delta.shape != (batch, 32):
            raise ValueError("history_mask and relative_delta must be [B,32]")
        if anchor_command.shape != (batch, ACTION_DIM):
            raise ValueError("anchor_command must be [B,28]")
        if task_id.shape != (batch,) or torch.any((task_id < 0) | (task_id >= TASK_COUNT)):
            raise ValueError("task_id must be a [B] vector with IDs in [0,4]")

        state_meta = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
        state_tokens = self.state_projection(torch.cat((state, state_meta), dim=-1))
        state_tokens = state_tokens + sinusoidal(relative_delta * 8.0, WIDTH) + self.state_modality
        state_tokens = state_tokens.transpose(1, 2)
        for block in self.tcn_blocks:
            state_tokens = block(state_tokens)
        state_tokens = state_tokens.transpose(1, 2)

        if self.video_mode == "STATE-CTRL":
            # State-only inference constructs the fixed masked input and never fetches RGB cache rows.
            dynamic = state.new_zeros((batch, TIME_SLOTS, SPACE_BINS, VIDEO_DIM))
            static = torch.zeros_like(dynamic)
        else:
            if "dynamic" not in prefix_inputs or "static" not in prefix_inputs:
                raise ValueError("fusion mode requires both frozen prefix feature tensors")
            dynamic = prefix_inputs["dynamic"]
            static = prefix_inputs["static"]
        video_values = self._video_layout(dynamic, static)
        expected_video = (batch, VIDEO_SLOTS, VIDEO_DIM)
        if tuple(video_values.shape) != expected_video:
            raise ValueError(f"Q0 video token shape mismatch: {tuple(video_values.shape)}")
        time_values = torch.linspace(-31.0, 0.0, TIME_SLOTS, device=state.device).view(1, TIME_SLOTS, 1)
        time_values = time_values.expand(batch, TIME_SLOTS, SPACE_BINS).reshape(batch, VIDEO_SLOTS)
        video_tokens = (
            self.video_projection(video_values)
            + sinusoidal(time_values, WIDTH)
            + spatial_encoding(state.device)
            + self.video_modality
        )
        anchor_token = self.anchor_projection(anchor_command).unsqueeze(1) + self.anchor_modality
        task_onehot = F.one_hot(task_id.long(), num_classes=TASK_COUNT).to(dtype=state.dtype)
        task_token = self.task_projection(task_onehot).unsqueeze(1)
        return torch.cat((state_tokens, video_tokens, anchor_token, task_token), dim=1)

    def _decode(
        self,
        memory: Tensor,
        *,
        goal_token_override: Tensor | None = None,
    ) -> Tensor:
        batch = memory.shape[0]
        if goal_token_override is not None:
            if goal_token_override.shape != (batch, 1, WIDTH):
                raise ValueError("goal token override must be [B,1,128]")
            memory = torch.cat((memory, goal_token_override), dim=1)
        query_times = torch.arange(1, 31, device=memory.device, dtype=torch.float32) / 30.0
        queries = sinusoidal(query_times.view(1, -1).expand(batch, -1), WIDTH)
        return self.output(self.decoder(queries, memory))

    def forward(self, prefix_inputs: dict[str, Tensor], task_id: Tensor) -> V2Output:
        """The public model API receives prefix inputs and task ID only; no target/label args."""
        memory = self._encode_prefix(prefix_inputs, task_id)
        if self.waypoint_mode is None:
            return V2Output(self._decode(memory), None)
        prefix_h = memory.mean(dim=1)
        goal_prediction = self.goal_head(prefix_h)
        if self.waypoint_mode == "GOAL_COND":
            goal_token = self.goal_projection(goal_prediction).unsqueeze(1)
        else:
            goal_token = memory.new_zeros((memory.shape[0], 1, WIDTH))
        return V2Output(self._decode(memory, goal_token_override=goal_token), goal_prediction)

    def qa_decode_with_goal_token(self, prefix_inputs: dict[str, Tensor], task_id: Tensor, token: Tensor) -> Tensor:
        """QA-only helper proving the context-memory injection changes action output."""
        memory = self._encode_prefix(prefix_inputs, task_id)
        return self._decode(memory, goal_token_override=token)


__all__ = [
    "ACTION_DIM", "TASK_COUNT", "V2Output", "V2TaskConditionedPredictor",
    "parameter_count", "state_hash", "WIDTH",
]
