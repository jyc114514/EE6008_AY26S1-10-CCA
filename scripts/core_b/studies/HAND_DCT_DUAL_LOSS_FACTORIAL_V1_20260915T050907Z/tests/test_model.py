"""CPU contract tests for DCT decoder and temporal dual loss."""

from __future__ import annotations

import numpy as np
import torch

from model import HandDecoder, SelectorHead, compose_hand_prediction, dct_basis, loss_components, total_loss


def constants() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.zeros(30, 14)
    scale = torch.ones(30, 14)
    safe = torch.full((14,), 0.1)
    lam = torch.tensor(0.08478981481957804)
    return mean, scale, safe, lam


def test_decoder_output_shapes_and_dct_basis() -> None:
    context = torch.zeros(4, 1336)
    assert HandDecoder(decoder="D0", seed=1)(context).shape == (4, 30, 14)
    assert HandDecoder(decoder="D1", seed=1, dct_k=16)(context).shape == (4, 30, 14)
    basis = dct_basis(30)
    assert torch.allclose(basis @ basis.T, torch.eye(30), atol=1e-5)


def test_dct_full_round_trip() -> None:
    torch.manual_seed(3)
    values = torch.randn(2, 30, 14)
    basis = dct_basis(30)
    coefficients = torch.einsum("kt,btj->bkj", basis, values)
    reconstructed = torch.einsum("kt,bkj->btj", basis, coefficients)
    assert torch.allclose(values, reconstructed, atol=1e-5)


def test_l0_has_position_only_and_l1_has_finite_fd_gradient() -> None:
    torch.manual_seed(4)
    prediction = torch.randn(3, 30, 14, requires_grad=True)
    target_std = torch.randn(3, 30, 14)
    target_native = torch.randn(3, 30, 14)
    mean, scale, safe, lam = constants()
    l0, pos0, fd0 = total_loss("L0", prediction, target_std, target_native, mean, scale, safe, lam)
    l1, pos1, fd1 = total_loss("L1", prediction, target_std, target_native, mean, scale, safe, lam)
    assert torch.equal(l0, pos0)
    assert torch.isfinite(fd0) and torch.isfinite(fd1)
    assert torch.allclose(l1, pos1 + lam * fd1)
    l1.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_q_zero_and_one_are_persistence_and_neural() -> None:
    persistence = torch.zeros(2, 30, 14)
    neural = torch.ones(2, 30, 14)
    assert torch.equal(compose_hand_prediction(persistence, neural, torch.zeros(2, 30, 1)), persistence)
    assert torch.equal(compose_hand_prediction(persistence, neural, torch.ones(2, 30, 1)), neural)


def test_selector_gradient_is_finite_and_shape_is_locked() -> None:
    torch.manual_seed(5)
    model = SelectorHead(seed=5)
    context = torch.randn(4, 1336)
    summary = torch.randn(4, 1)
    magnitude = torch.rand(4, 30, 1)
    q = model.probability(context, summary, magnitude)
    assert q.shape == (4, 30, 1)
    q.mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_full_concat_preserves_arm_slice() -> None:
    hand = np.zeros((2, 30, 14), dtype=np.float32)
    arm = np.ones((2, 30, 14), dtype=np.float32)
    full = np.concatenate((hand, arm), axis=2)
    assert full.shape == (2, 30, 28)
    assert np.array_equal(full[:, :, 14:], arm)
