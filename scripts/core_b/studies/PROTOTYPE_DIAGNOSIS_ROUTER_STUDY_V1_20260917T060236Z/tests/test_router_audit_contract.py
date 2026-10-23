"""Small CPU contract tests for the no-router audit path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from make_router_figures import normalise_probabilities  # noqa: E402
from run_router_audit import fold_assignment, metric_bundle, mode_metrics  # noqa: E402


def test_probability_rows_are_normalised_and_finite() -> None:
    values = normalise_probabilities(np.asarray([[2.0, 1.0], [0.0, 0.0], [1.0, 3.0]]))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    np.testing.assert_allclose(values[1], [0.5, 0.5])


def test_mode_metrics_accepts_non_unit_probabilities() -> None:
    result = mode_metrics(np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([[2.0, 0.0], [0.0, 3.0]]), 2)
    assert result["N"] == 2
    assert np.isfinite(result["nll"])
    assert np.isfinite(result["brier"])


def test_fold_assignment_is_deterministic_and_three_way() -> None:
    ids = np.arange(30, dtype=np.int64) * 17 + 4
    first = fold_assignment(ids)
    second = fold_assignment(ids)
    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)) == {0, 1, 2}


def test_metric_bundle_zero_error_when_prediction_equals_target() -> None:
    rng = np.random.default_rng(20260917)
    target_std = rng.normal(size=(3, 30, 28))
    target_native = rng.normal(size=(3, 30, 28))
    persistence = target_native.copy()
    metrics = metric_bundle(target_std[:, :, :14], target_std, target_native[:, :, :14], target_native, target_std, target_native, persistence)
    for key in ("hand_mae_std", "hand_mae_native", "full_mae_std", "hand_fd_native", "hand_fd_std", "full_fd_native"):
        np.testing.assert_allclose(metrics[key], 0.0)


def test_metric_bundle_native_fd_has_29_future_to_future_steps() -> None:
    target = np.zeros((1, 30, 28), dtype=np.float64)
    prediction = target.copy()
    prediction[:, 1:, 0] = 1.0
    metrics = metric_bundle(prediction[:, :, :14], prediction, prediction[:, :, :14], prediction, target, target, target)
    expected = 1.0 / (29.0 * 14.0)
    np.testing.assert_allclose(metrics["hand_fd_native"], expected)


def test_fixed_q_replay_formula_is_componentwise() -> None:
    persistence = np.asarray([[[1.0], [2.0], [3.0]]])
    components = np.asarray([[[[3.0], [4.0], [5.0]], [[-1.0], [0.0], [1.0]]]])
    q = np.asarray([[0.25, 0.5, 0.75]])
    replay = persistence[:, None] + q[:, None, :, None] * (components - persistence[:, None])
    np.testing.assert_allclose(replay[0, 0, :, 0], [1.5, 3.0, 4.5])
    np.testing.assert_allclose(replay[0, 1, :, 0], [0.5, 1.0, 1.5])
