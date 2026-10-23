from __future__ import annotations

import numpy as np
import pytest

from probabilistic_evaluator import best_of_k, energy_score, path_coverage, real_evaluation_disabled


def test_energy_score_zero_for_identical_single_support() -> None:
    samples = np.zeros((2, 3, 4, 28))
    target = np.zeros((2, 4, 28))
    assert np.allclose(energy_score(samples, target), 0.0)


def test_more_samples_does_not_change_identical_distribution_score() -> None:
    target = np.zeros((1, 2, 28))
    one = np.zeros((1, 1, 2, 28))
    many = np.zeros((1, 8, 2, 28))
    assert np.allclose(energy_score(one, target), energy_score(many, target))
    assert np.allclose(best_of_k(one, target), best_of_k(many, target))


def test_coverage_shapes_are_explicit() -> None:
    target = np.zeros((1, 2, 28))
    samples = np.zeros((1, 3, 2, 28))
    assert path_coverage(samples, target, 0.1, simultaneous=True).tolist() == [True]
    assert path_coverage(samples, target, 0.1, simultaneous=False).tolist() == [1.0]


def test_real_evaluation_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="prepared_not_executed"):
        real_evaluation_disabled()
