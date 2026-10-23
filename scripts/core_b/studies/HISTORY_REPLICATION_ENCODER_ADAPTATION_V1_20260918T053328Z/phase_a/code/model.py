"""Matched causal TCN history encoder with a fixed DCT-16 residual decoder."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HISTORY_STEPS = 32
STATE_DIM = 71
VIDEO_DIM = 768
HAND_DIM = 14
TARGET_STEPS = 30
DCT_K = 16
CHANNELS = 64
VIDEO_PROJECTION = 128
FUSION_HIDDEN = 128
DILATIONS = (1, 2, 4, 8, 16)


def dct_basis(length: int = TARGET_STEPS) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    frequency = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(torch.pi / length * (time + 0.5) * frequency) * torch.sqrt(torch.tensor(2.0 / length))
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


class CausalResidualBlock(nn.Module):
    """A left-padded, non-circular, dilated residual convolution block."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("causal block expects (B,C,T)")
        padded = F.pad(values, (self.left_padding, 0))
        update = self.conv(padded)
        update = F.relu(update)
        update = self.dropout(update)
        return values + update


class HistoryVideoTCNDCT(nn.Module):
    """Past-only TCN plus fixed-video projection and DCT residual output.

    Inputs are normalized state values `(B,32,71)`, a binary validity mask and
    standardized relative timestamps `(B,32)`, plus a fixed 768-D video
    feature. The output is a scaled `(B,30,14)` residual; callers convert it
    to native command values using the train-only residual scale.
    """

    def __init__(self, *, seed: int, state_input_dim: int = STATE_DIM, video_dim: int = VIDEO_DIM, dct_k: int = DCT_K) -> None:
        super().__init__()
        if dct_k != DCT_K:
            raise ValueError("this locked contract only supports DCT-K=16")
        self.state_input_dim = state_input_dim
        self.video_dim = video_dim
        self.dct_k = dct_k
        self.state_projection = nn.Linear(state_input_dim + 2, CHANNELS)
        self.tcn_blocks = nn.ModuleList([CausalResidualBlock(CHANNELS, dilation) for dilation in DILATIONS])
        self.video_projection = nn.Linear(video_dim, VIDEO_PROJECTION)
        self.fusion = nn.Sequential(nn.Linear(CHANNELS + VIDEO_PROJECTION, FUSION_HIDDEN), nn.ReLU(), nn.Dropout(0.1))
        self.output = nn.Linear(FUSION_HIDDEN, dct_k * HAND_DIM)
        self.register_buffer("dct_basis", dct_basis(TARGET_STEPS), persistent=True)
        self._reset(seed)
        # Zero output initialization makes the initial native prediction equal
        # to the persistence anchor for every cell and every input.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _reset(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in (self.state_projection, self.video_projection, self.fusion[0]):
                module.reset_parameters()
            for block in self.tcn_blocks:
                block.conv.reset_parameters()

    @property
    def output_shape(self) -> tuple[int, int]:
        return TARGET_STEPS, HAND_DIM

    def forward(self, state_values: torch.Tensor, history_mask: torch.Tensor, relative_delta: torch.Tensor, video_feature: torch.Tensor) -> torch.Tensor:
        if state_values.ndim != 3 or state_values.shape[1:] != (HISTORY_STEPS, self.state_input_dim):
            raise ValueError(f"state_values must be (B,{HISTORY_STEPS},{self.state_input_dim})")
        if history_mask.shape != state_values.shape[:2] or relative_delta.shape != state_values.shape[:2]:
            raise ValueError("history mask/time shapes do not match state values")
        if video_feature.ndim != 2 or video_feature.shape[1] != self.video_dim or video_feature.shape[0] != state_values.shape[0]:
            raise ValueError(f"video_feature must be (B,{self.video_dim})")
        metadata = torch.cat((history_mask.unsqueeze(-1), relative_delta.unsqueeze(-1)), dim=-1)
        encoded = self.state_projection(torch.cat((state_values, metadata), dim=-1)).transpose(1, 2)
        for block in self.tcn_blocks:
            encoded = block(encoded)
        valid_lengths = history_mask.gt(0).sum(dim=1).long().clamp(min=1, max=HISTORY_STEPS)
        last_index = (valid_lengths - 1).view(-1, 1, 1).expand(-1, CHANNELS, 1)
        state_context = encoded.gather(dim=2, index=last_index).squeeze(-1)
        video_context = F.relu(self.video_projection(video_feature))
        fused = self.fusion(torch.cat((state_context, video_context), dim=-1))
        coefficients = self.output(fused).reshape(-1, self.dct_k, HAND_DIM)
        return torch.einsum("tk,bkj->btj", self.dct_basis[: self.dct_k].T, coefficients)


def native_prediction(model_output_scaled: torch.Tensor, anchor_hand_native: torch.Tensor, residual_sigma: torch.Tensor) -> torch.Tensor:
    if model_output_scaled.ndim != 3 or model_output_scaled.shape[1:] != (TARGET_STEPS, HAND_DIM):
        raise ValueError("model output must be (B,30,14)")
    if anchor_hand_native.shape != (len(model_output_scaled), HAND_DIM):
        raise ValueError("anchor shape mismatch")
    if residual_sigma.shape != (HAND_DIM,):
        raise ValueError("residual sigma shape mismatch")
    return anchor_hand_native.unsqueeze(1) + model_output_scaled * residual_sigma.view(1, 1, HAND_DIM)


def loss_components(
    prediction_native: torch.Tensor,
    target_native: torch.Tensor,
    target_hand_standardized: torch.Tensor,
    target_mean_hand: torch.Tensor,
    target_std_hand: torch.Tensor,
    fd_scale: torch.Tensor,
    lambda_fd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction_native.shape != target_native.shape or prediction_native.shape[1:] != (TARGET_STEPS, HAND_DIM):
        raise ValueError("prediction/target native shape mismatch")
    if target_hand_standardized.shape != prediction_native.shape:
        raise ValueError("standardized target shape mismatch")
    prediction_standardized = (prediction_native - target_mean_hand) / target_std_hand
    position = F.l1_loss(prediction_standardized, target_hand_standardized, reduction="mean")
    prediction_delta = prediction_native[:, 1:] - prediction_native[:, :-1]
    target_delta = target_native[:, 1:] - target_native[:, :-1]
    fd_scaled = torch.mean(torch.abs(prediction_delta - target_delta) / fd_scale.view(1, 1, HAND_DIM))
    total = position + lambda_fd * fd_scaled
    return total, position, fd_scaled


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def set_all_seeds(seed: int, *, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())
