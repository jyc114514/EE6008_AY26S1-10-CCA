"""CPU toy tests for the locked temporal-fidelity transforms."""

from __future__ import annotations

import numpy as np

from temporal_fidelity_analysis import dct_basis, dct_project, signed_terms, smooth_gate


def test_signed_decomposition_is_exact_for_constant_anchor() -> None:
    rng = np.random.default_rng(7)
    persistence = np.repeat(rng.normal(size=(2, 1, 3)), 30, axis=1)
    residual = rng.normal(size=(2, 30, 3))
    q = rng.uniform(0, 1, size=(2, 30, 1))
    anchor, residual_term, gate_term = signed_terms(persistence, q, residual)
    prediction = persistence + q * residual
    assert np.max(np.abs(np.diff(prediction, axis=1) - (anchor + residual_term + gate_term))) < 1e-12
    assert np.max(np.abs(anchor)) == 0


def test_signed_decomposition_keeps_time_varying_anchor_explicit() -> None:
    persistence = np.arange(2 * 30 * 3, dtype=float).reshape(2, 30, 3) / 100
    residual = np.ones((2, 30, 3))
    q = np.full((2, 30, 1), 0.5)
    anchor, residual_term, gate_term = signed_terms(persistence, q, residual)
    prediction = persistence + q * residual
    assert np.max(np.abs(np.diff(prediction, axis=1) - (anchor + residual_term + gate_term))) < 1e-12
    assert np.max(np.abs(anchor)) > 0


def test_gate_smoothing_uses_replicate_edges_and_clip() -> None:
    q = np.array([[[0.0], [1.0], [0.0], [2.0]]])
    smoothed = smooth_gate(q)
    expected = np.array([[[0.25], [0.5], [0.75], [1.0]]])
    assert np.allclose(smoothed, expected)
    assert np.all((smoothed >= 0) & (smoothed <= 1))


def test_dct_round_trip_is_orthonormal() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(3, 30, 4))
    projected, _ = dct_project(values, dct_basis(30), 30)
    assert np.allclose(projected, values, atol=1e-12)


def test_dct_projection_uses_only_first_k_coefficients() -> None:
    basis = dct_basis(30)
    values = np.ones((1, 30, 1))
    projected, coefficients = dct_project(values, basis, 1)
    assert np.allclose(projected, values, atol=1e-12)
    assert np.allclose(coefficients[:, 1:], 0, atol=1e-12)


def test_transform_shapes_are_canonical() -> None:
    values = np.zeros((4, 30, 14))
    q = np.zeros((4, 30, 1))
    anchor, residual, gate = signed_terms(values, q, values)
    assert anchor.shape == residual.shape == gate.shape == (4, 29, 14)
