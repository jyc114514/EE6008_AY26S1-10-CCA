from __future__ import annotations

import numpy as np
import pytest

from rolling_evaluator import evaluate_placeholder, fixed_window_anchor_indices, forecast_stability, intersect_future_times, revision_utility


def test_intersection_uses_absolute_times() -> None:
    assert np.array_equal(intersect_future_times(np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([1, 2, 3, 4])), np.array([2, 3]))


def test_revision_utility_direction_and_stability() -> None:
    assert np.array_equal(revision_utility(np.array([2.0, 1.0]), np.array([1.0, 1.5])), np.array([1.0, -0.5]))
    assert forecast_stability(np.zeros((2, 3)), np.ones((2, 3))) == 1.0


def test_fixed_window_does_not_use_early_history() -> None:
    assert np.array_equal(fixed_window_anchor_indices(np.array([0, 119, 120, 240])), np.array([119, 120, 240]))


def test_real_effect_table_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="prepared_not_executed"):
        evaluate_placeholder()
