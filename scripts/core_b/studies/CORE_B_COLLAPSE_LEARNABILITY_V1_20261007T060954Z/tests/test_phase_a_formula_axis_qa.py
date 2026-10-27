import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCHED_CODE = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code'))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(SCHED_CODE))

from d3_eq_contract import episode_equal_weights  # noqa: E402
from collapse_diagnostic_core import loss_components  # noqa: E402


def test_persistence_against_linear_future_separates_29_internal_deltas_from_seam():
    anchor = torch.zeros((1, 28), dtype=torch.float32)
    sigma = torch.ones(28)
    safe_fd = torch.ones(28)
    times = torch.arange(1, 31, dtype=torch.float32).view(1, 30, 1)
    target = times.expand(1, 30, 28).clone()
    persistence_residual = torch.zeros((1, 30, 28), dtype=torch.float32)
    parts = loss_components(persistence_residual, target, anchor, sigma, safe_fd)
    assert parts["internal_difference_count"] == 29
    assert parts["fd"].item() == 1.0
    assert torch.equal(torch.diff(target, dim=1), torch.ones((1, 29, 28)))
    # The anchor-to-first-future difference is separate and is not part of L_delta.
    seam = target[:, :1] - anchor[:, None, :]
    assert torch.equal(seam, torch.ones((1, 1, 28)))


def test_step_future_counts_one_internal_jump_and_keeps_anchor_seam_separate():
    anchor = -torch.ones((1, 28), dtype=torch.float32)
    sigma = torch.ones(28)
    safe_fd = torch.ones(28)
    target = torch.zeros((1, 30, 28), dtype=torch.float32)
    target[:, 15:] = 1.0
    persistence_residual = torch.zeros_like(target)
    parts = loss_components(persistence_residual, target, anchor, sigma, safe_fd)
    deltas = torch.diff(target, dim=1)
    assert parts["internal_difference_count"] == 29
    assert torch.count_nonzero(deltas).item() == 28
    assert parts["fd"].item() == torch.tensor(1.0 / 29.0).item()
    seam = target[:, :1] - anchor[:, None, :]
    assert torch.equal(seam, torch.ones((1, 1, 28)))


def test_one_window_per_episode_panel_has_equal_weights():
    weights = episode_equal_weights(torch.tensor([10, 20, 30, 40]).numpy())
    assert (weights == 1.0).all()
