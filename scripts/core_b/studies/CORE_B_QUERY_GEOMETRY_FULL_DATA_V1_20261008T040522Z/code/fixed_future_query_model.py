from __future__ import annotations

import torch
from torch import Tensor, nn

from d3_model import D3TimeSpacePredictor, sinusoidal, spatial_encoding


def forward_with_fixed_future_queries(
    base: D3TimeSpacePredictor,
    state: Tensor,
    history_mask: Tensor,
    relative_delta: Tensor,
    dynamic: Tensor,
    static: Tensor,
    anchor_command: Tensor,
    future_query_table: Tensor,
) -> Tensor:
    """Source-equivalent D3 forward with only the deterministic future query table replaced."""
    if future_query_table.shape != (base.horizon, 128):
        raise ValueError(f"candidate query shape mismatch: {tuple(future_query_table.shape)}")
    batch = state.shape[0]
    state_meta = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
    state_tokens = base.state_projection(torch.cat((state, state_meta), dim=-1))
    state_tokens = state_tokens + sinusoidal(relative_delta * 8.0, 128) + base.state_modality
    state_tokens = state_tokens.transpose(1, 2)
    for block in base.tcn_blocks:
        state_tokens = block(state_tokens)
    state_tokens = state_tokens.transpose(1, 2)

    video_values = base._video_layout(dynamic, static)
    time_values = torch.linspace(-31.0, 0.0, 16, device=state.device).view(1, 16, 1).expand(batch, 16, 4).reshape(batch, 64)
    video_tokens = base.video_projection(video_values) + sinusoidal(time_values, 128) + spatial_encoding(state.device) + base.video_modality
    anchor_token = base.anchor_projection(anchor_command).unsqueeze(1) + base.anchor_modality
    memory = torch.cat((state_tokens, video_tokens, anchor_token), dim=1)
    queries = future_query_table.to(device=state.device, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1)
    return base.output(base.decoder(queries, memory))


class FixedFutureQueryOverride(nn.Module):
    """Wrap a strictly initialized base model and hold the fixed query table as a buffer."""

    def __init__(self, base: D3TimeSpacePredictor, query_table: Tensor) -> None:
        super().__init__()
        if tuple(query_table.shape) != (base.horizon, 128):
            raise ValueError(f"query table {tuple(query_table.shape)} does not match H={base.horizon}")
        self.base = base
        self.register_buffer("future_query_table", query_table.detach().to(device="cpu", dtype=torch.float32).clone(), persistent=True)
        self.eval()

    def forward(self, state: Tensor, history_mask: Tensor, relative_delta: Tensor,
                dynamic: Tensor, static: Tensor, anchor_command: Tensor) -> Tensor:
        return forward_with_fixed_future_queries(
            self.base, state, history_mask, relative_delta, dynamic, static,
            anchor_command, self.future_query_table,
        )
