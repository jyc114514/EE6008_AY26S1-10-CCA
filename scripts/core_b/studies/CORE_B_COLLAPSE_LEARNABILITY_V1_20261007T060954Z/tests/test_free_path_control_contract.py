import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_free_path_control import IndependentFuturePath  # noqa: E402


def test_free_path_is_zero_initialized_and_has_one_parameter_per_output():
    model = IndependentFuturePath(n_samples=8, horizon=30, action_dim=28)
    assert tuple(model.residual.shape) == (8, 30, 28)
    assert torch.count_nonzero(model.residual).item() == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == 8 * 30 * 28


def test_free_path_outputs_are_independent_per_sample_without_shared_decoder():
    model = IndependentFuturePath(n_samples=8, horizon=30, action_dim=28)
    with torch.no_grad():
        model.residual[3, 12, 7] = 0.5
    dummy = torch.zeros((8, 1))
    prediction = model(dummy, dummy, dummy, dummy, dummy, dummy)
    assert prediction[3, 12, 7].item() == 0.5
    assert torch.count_nonzero(prediction[0]).item() == 0
    assert torch.count_nonzero(prediction[1]).item() == 0
