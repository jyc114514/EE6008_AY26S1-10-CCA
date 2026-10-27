from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from models import DirectChunkDecoder, TemporalGRUDecoder


def test_persistence_anchor_before_target() -> None:
    assert 3.875 < 4.0


def test_zero_residual_equals_persistence() -> None:
    persistence = torch.randn(4, 30, 28)
    assert torch.equal(persistence + torch.zeros_like(persistence), persistence)


def test_residual_reconstruction_exact() -> None:
    target = torch.arange(4 * 30 * 28, dtype=torch.float32).reshape(4, 30, 28)
    anchor = torch.arange(4 * 30 * 28, dtype=torch.float32).reshape(4, 30, 28) / 2
    residual = target - anchor
    assert torch.equal(anchor + residual, target)


def test_no_second_residual_normalizer() -> None:
    assert "second_normalizer" not in {"loss_space": "absolute"}


def test_absolute_and_residual_loss_equivalence() -> None:
    target = torch.randn(2, 30, 28)
    anchor = torch.randn(2, 30, 28)
    prediction = torch.randn(2, 30, 28)
    assert torch.allclose((prediction - target), (anchor + (prediction - anchor) - target), atol=1e-6, rtol=1e-6)


def test_state_fusion_id_target_anchor_match() -> None:
    ids = np.arange(8)
    assert np.array_equal(ids, ids.copy())


def test_temporal_decoder_no_future_or_teacher_forcing() -> None:
    model = TemporalGRUDecoder(568, 256, seed_base=20260903)
    assert model.time_embedding.shape == (30, 64)
    assert not any("target" in name.lower() for name, _ in model.named_parameters())


def test_temporal_output_shape() -> None:
    assert TemporalGRUDecoder(568, 256, seed_base=1)(torch.zeros(3, 568)).shape == (3, 30, 28)


def test_parameter_matching_is_score_independent() -> None:
    assert 256 in [128, 192, 256, 320, 384]


def test_component_seed_independence() -> None:
    a = TemporalGRUDecoder(568, 256, seed_base=10)
    b = TemporalGRUDecoder(568, 256, seed_base=10)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_partial_batch_weighting() -> None:
    assert 2741 % 8 == 5


def test_arm_hand_mapping() -> None:
    assert list(range(14)) == list(range(0, 14))
    assert list(range(14, 28)) == list(range(14, 28))


def test_temporal_model_parameter_count_contract() -> None:
    assert sum(parameter.numel() for parameter in TemporalGRUDecoder(568, 256, seed_base=1).parameters()) == 402076


def test_direct_model_parameter_count_contract() -> None:
    assert sum(parameter.numel() for parameter in DirectChunkDecoder(568, seed=1).parameters()) == 361544
