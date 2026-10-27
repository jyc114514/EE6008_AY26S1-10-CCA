"""CPU-testable D3-EQ objective, sampler and effective-batch arithmetic."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


def episode_equal_weights(episode_ids: np.ndarray) -> np.ndarray:
    """Return w_i=W/(E*n_e) using the complete locked training population."""
    episodes = np.asarray(episode_ids, dtype=np.int64)
    if episodes.ndim != 1 or not len(episodes):
        raise ValueError("episode_ids must be a non-empty 1D array")
    _, inverse, counts = np.unique(episodes, return_inverse=True, return_counts=True)
    weights = len(episodes) / (len(counts) * counts[inverse].astype(np.float64))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("episode weights must be positive and finite")
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("episode weights must have population mean one")
    return weights


def per_sample_loss(
    pred_residual: torch.Tensor,
    target_native: torch.Tensor,
    anchor_native: torch.Tensor,
    target_sigma: torch.Tensor,
    safe_fd_scale: torch.Tensor,
) -> torch.Tensor:
    """V3 position + 0.1 future-internal FD loss, reduced to one value per row."""
    if pred_residual.ndim != 3 or pred_residual.shape[-1] != 28:
        raise ValueError("pred_residual must have shape [batch,time,28]")
    if target_native.shape != pred_residual.shape:
        raise ValueError("target_native and pred_residual shapes differ")
    if anchor_native.shape != (pred_residual.shape[0], 28):
        raise ValueError("anchor_native must have shape [batch,28]")
    if target_sigma.shape != (28,) or safe_fd_scale.shape != (28,):
        raise ValueError("both normalizers must have shape [28]")
    target_residual = (target_native - anchor_native[:, None, :]) / target_sigma[None, None, :]
    pos_error = torch.abs(pred_residual - target_residual)
    pos = 0.5 * (
        pos_error[..., :14].mean(dim=(1, 2)) + pos_error[..., 14:].mean(dim=(1, 2))
    )
    pred_native = anchor_native[:, None, :] + target_sigma[None, None, :] * pred_residual
    target_fd = torch.diff(target_native, dim=1) / safe_fd_scale[None, None, :]
    pred_fd = torch.diff(pred_native, dim=1) / safe_fd_scale[None, None, :]
    fd_error = torch.abs(pred_fd - target_fd)
    fd = 0.5 * (
        fd_error[..., :14].mean(dim=(1, 2)) + fd_error[..., 14:].mean(dim=(1, 2))
    )
    return pos + 0.1 * fd


def weighted_batch_loss(sample_losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute sum_i(w_i*l_i)/B_actual; never normalize by sum(weights)."""
    if sample_losses.ndim != 1 or weights.shape != sample_losses.shape:
        raise ValueError("sample losses and weights must be same-shape vectors")
    if not sample_losses.numel():
        raise ValueError("empty batch is invalid")
    return torch.sum(sample_losses * weights) / sample_losses.numel()


def effective_update_groups(
    n_rows: int, *, microbatch_size: int = 64, accumulation_steps: int = 1
) -> Iterator[tuple[int, int, tuple[tuple[int, int], ...]]]:
    """Yield effective optimizer updates and constituent half-open microbatches."""
    if n_rows <= 0 or microbatch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("row count, microbatch size and accumulation must be positive")
    effective = microbatch_size * accumulation_steps
    for group_start in range(0, n_rows, effective):
        group_end = min(n_rows, group_start + effective)
        micros = tuple(
            (start, min(start + microbatch_size, group_end))
            for start in range(group_start, group_end, microbatch_size)
        )
        yield group_start, group_end, micros


def accumulated_group_loss(
    sample_losses: torch.Tensor,
    weights: torch.Tensor,
    group_start: int,
    group_end: int,
    microbatches: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Reconstruct one update's exact objective from its microbatch slices."""
    if sample_losses.ndim != 1 or weights.shape != sample_losses.shape:
        raise ValueError("sample losses and weights must be same-shape vectors")
    denominator = group_end - group_start
    if denominator <= 0:
        raise ValueError("empty effective update group")
    contributions = [
        torch.sum(sample_losses[start:end] * weights[start:end]) / denominator
        for start, end in microbatches
    ]
    return torch.stack(contributions).sum()


def shuffled_epoch_order(n_rows: int, seed: int, epoch: int) -> np.ndarray:
    if n_rows <= 0 or epoch < 1:
        raise ValueError("n_rows must be positive and epoch is one-based")
    order = np.arange(n_rows, dtype=np.int64)
    np.random.default_rng(seed + epoch).shuffle(order)
    return order


def historical_scheduler_spec(n_rows: int, effective_batch: int = 64) -> dict[str, int | str]:
    """Represent the actual historical scheduler call pattern, not its label."""
    steps_per_epoch = (n_rows + effective_batch - 1) // effective_batch
    return {
        "kind": "CosineAnnealingLR",
        "T_max": 40 * steps_per_epoch,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "scheduler_step_calls_per_epoch": 1,
        "scheduler_step_location": "once_after_each_epoch",
        "nominal_eta_min": 1e-5,
    }
