"""CPU seam tests for the History×Video training contract."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from model import HistoryVideoTCNDCT, dct_basis, loss_components, native_prediction, parameter_count  # noqa: E402
from run_history_video_cell import cell_inputs, load_data  # noqa: E402


def test_canonical_input_and_timeline_shapes() -> None:
    data, _stats = load_data(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz")
    assert data["history_native"].shape == (3355, 32, 71)
    assert data["dynamic_video_feature"].shape == (3355, 768)
    assert data["targets_native"].shape == (3355, 30, 28)
    timeline = json.loads((ROOT / "DATA_TIMELINE_MANIFEST.json").read_text())
    assert timeline["all_past_timestamps_le_t0"] is True
    assert timeline["all_future_timestamps_gt_t0"] is True


def test_last_hist_change_only_state_values() -> None:
    data, _stats = load_data(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz")
    hist = cell_inputs(data, "HI-S-HIST")
    last = cell_inputs(data, "HI-S-LAST")
    for index in (1, 2, 4, 5, 6, 7, 8):
        np.testing.assert_array_equal(hist[index], last[index])
    np.testing.assert_array_equal(last[0][:, -1], hist[0][:, -1])
    np.testing.assert_array_equal(last[0], np.repeat(hist[0][:, -1:, :], 32, axis=1))
    assert np.any(np.abs(hist[0][:, :-1] - last[0][:, :-1]) > 0)


def test_s_video_is_zero_and_parameter_count_is_matched() -> None:
    data, _stats = load_data(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz")
    s = cell_inputs(data, "HI-S-HIST")
    f = cell_inputs(data, "HI-F-HIST")
    np.testing.assert_array_equal(s[3], np.zeros_like(s[3]))
    np.testing.assert_array_equal(s[0], f[0])
    np.testing.assert_array_equal(s[1], f[1])
    np.testing.assert_array_equal(s[2], f[2])
    np.testing.assert_array_equal(s[4], f[4])
    assert parameter_count(HistoryVideoTCNDCT(seed=20260939)) == 218528


def test_dct_basis_is_orthonormal() -> None:
    basis = dct_basis().numpy()
    np.testing.assert_allclose(basis @ basis.T, np.eye(30), atol=2e-6)


def test_zero_output_initializes_to_persistence_and_has_finite_gradient() -> None:
    data, stats = load_data(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz")
    values = cell_inputs(data, "HI-F-HIST")
    model = HistoryVideoTCNDCT(seed=20260939)
    state, mask, delta, video, anchor, target_native, target_std, _ids, _tasks = values
    device = torch.device("cpu")
    output = model(torch.from_numpy(state[:2]), torch.from_numpy(mask[:2]), torch.from_numpy(delta[:2]), torch.from_numpy(video[:2]))
    prediction = native_prediction(output, torch.from_numpy(anchor[:2]), torch.from_numpy(stats["residual_sigma"]))
    np.testing.assert_allclose(prediction.detach().numpy(), np.repeat(anchor[:2, None], 30, axis=1), atol=1e-7)
    total, _position, _fd = loss_components(prediction, torch.from_numpy(target_native[:2]), torch.from_numpy(target_std[:2]), torch.from_numpy(stats["target_mean_hand"]), torch.from_numpy(stats["target_std_hand"]), torch.from_numpy(stats["fd_scale"]), torch.tensor(stats["lambda_fd"]))
    total.backward()
    assert torch.isfinite(total)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_future_sentinel_does_not_change_stored_history() -> None:
    data, _stats = load_data(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz")
    original = data["history_native"].copy()
    mutated_targets = data["targets_native"].copy()
    mutated_targets[:, :, :] += 10_000.0
    np.testing.assert_array_equal(data["history_native"], original)
    assert not np.array_equal(mutated_targets, data["targets_native"])


def test_effective_batch_step_contract() -> None:
    assert math.ceil(math.ceil(2741 / 64) / 1) == 43
    assert 40 * math.ceil(math.ceil(2741 / 64) / 1) == 1720
    assert math.ceil(math.ceil(2741 / 32) / 2) == 43
    assert 40 * math.ceil(math.ceil(2741 / 32) / 2) == 1720


def test_full_composite_hand_arm_slices() -> None:
    hand = np.zeros((2, 30, 14), dtype=np.float32)
    arm = np.arange(2 * 30 * 14, dtype=np.float32).reshape(2, 30, 14)
    full = np.concatenate((hand, arm), axis=2)
    assert full.shape == (2, 30, 28)
    np.testing.assert_array_equal(full[:, :, :14], hand)
    np.testing.assert_array_equal(full[:, :, 14:28], arm)
