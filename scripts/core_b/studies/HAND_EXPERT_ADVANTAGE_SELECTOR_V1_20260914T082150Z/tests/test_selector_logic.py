import numpy as np
import torch

from selector_model import SelectorHead, compose_hand_prediction, selector_feature_matrix


def test_q_zero_and_one_recover_the_two_experts():
    persistence = np.zeros((2, 30, 14), dtype=np.float32)
    neural = np.ones_like(persistence)
    assert np.array_equal(compose_hand_prediction(persistence, neural, np.zeros((2, 30, 1), dtype=np.float32)), persistence)
    assert np.array_equal(compose_hand_prediction(persistence, neural, np.ones((2, 30, 1), dtype=np.float32)), neural)


def test_selector_features_are_context_and_base_only():
    context = np.zeros((3, 1336), dtype=np.float32)
    state_summary = np.ones((3, 1), dtype=np.float32)
    residual_magnitude = np.ones((3, 30, 1), dtype=np.float32)
    features = selector_feature_matrix(context, state_summary, residual_magnitude)
    assert features.shape == (3, 30, 1370)


def test_selector_only_gradients():
    model = SelectorHead(seed=7)
    context = torch.zeros((4, 1336))
    summary = torch.zeros((4, 1))
    magnitude = torch.zeros((4, 30, 1))
    logits = model(context, summary, magnitude)
    loss = logits.square().mean()
    loss.backward()
    assert logits.shape == (4, 30, 1)
    assert all(parameter.grad is not None for parameter in model.parameters())
