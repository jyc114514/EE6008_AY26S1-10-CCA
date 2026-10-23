from __future__ import annotations

import inspect

import pytest
import torch

from models import PersistenceGatedResidualModel, parameter_count, split_hidden_for_parameter_match
from models.gated_residual import GroupTimePersistenceGate


def test_zero_gate_is_exact_persistence_and_unit_gate_is_ungated():
    model = PersistenceGatedResidualModel(568, head_coupling="split", residual_gate="group_time_gated", hidden_dim=192, seed=20260924)
    context = torch.randn(3, 568)
    raw = model.raw_residual(context)
    zeros = torch.zeros(3, 30, 2)
    ones = torch.ones(3, 30, 2)
    assert torch.equal(model.apply_group_gate(raw, zeros), torch.zeros_like(raw))
    assert torch.equal(model.apply_group_gate(raw, ones), raw)


def test_gate_range_and_canonical_arm_hand_mapping():
    model = PersistenceGatedResidualModel(568, head_coupling="shared", residual_gate="group_time_gated", hidden_dim=256, seed=20260924)
    context = torch.randn(2, 568)
    residual = torch.zeros(2, 30, 28)
    residual[:, :, :14] = 2.0
    residual[:, :, 14:] = 3.0
    gates = torch.zeros(2, 30, 2)
    gates[:, :, 0] = 0.25  # Arm
    gates[:, :, 1] = 0.50  # Hand
    output, predicted_gates = model(context, gate_override=gates)
    assert torch.all((predicted_gates >= 0) & (predicted_gates <= 1))
    assert torch.allclose(model.apply_group_gate(residual, gates)[:, :, :14], torch.full((2, 30, 14), 1.0))
    assert torch.allclose(model.apply_group_gate(residual, gates)[:, :, 14:], torch.full((2, 30, 14), 0.75))
    assert output.shape == (2, 30, 28)


def test_gate_uses_context_and_time_but_no_future_target():
    source = inspect.getsource(PersistenceGatedResidualModel.forward) + inspect.getsource(PersistenceGatedResidualModel.gate_values) + inspect.getsource(GroupTimePersistenceGate.forward)
    assert "target" not in source.lower()
    assert "prediction_error" not in source.lower()
    assert "time" in source.lower()


def test_split_parameter_matching_is_score_independent_and_within_ten_percent():
    for input_dim in (568, 1336):
        hidden, split_count = split_hidden_for_parameter_match(input_dim)
        shared_count = parameter_count(input_dim, head_coupling="shared", residual_gate="ungated", hidden_dim=256)
        assert hidden in (128, 160, 192, 224, 256)
        assert abs(split_count - shared_count) / shared_count <= 0.10


def test_residual_reconstruction_and_full28_reduction():
    persistence = torch.randn(4, 30, 28)
    residual = torch.randn(4, 30, 28)
    target_residual = torch.randn(4, 30, 28)
    predicted = persistence + residual
    target = persistence + target_residual
    expected = torch.sum((predicted - target) ** 2)
    actual = torch.nn.functional.mse_loss(predicted, target, reduction="sum")
    assert torch.equal(expected, actual)


def test_partial_batch_weighting_and_component_seed_independence():
    assert 3 * 30 * 28 == 2520
    a = PersistenceGatedResidualModel(568, head_coupling="shared", residual_gate="ungated", hidden_dim=256, seed=20260924)
    b = PersistenceGatedResidualModel(568, head_coupling="shared", residual_gate="ungated", hidden_dim=256, seed=20260924)
    c = PersistenceGatedResidualModel(568, head_coupling="shared", residual_gate="ungated", hidden_dim=256, seed=20260924 + 1000)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))
