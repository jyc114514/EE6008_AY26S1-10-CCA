from __future__ import annotations

import pytest
import torch

from ee6008.full_attentive import (
    EarlyStoppingState,
    extended_classification_metrics,
    repeat_first_frame,
    validate_token_shape,
)


def test_extended_metrics_are_120_class_and_calibrated() -> None:
    logits = torch.zeros((4, 120), dtype=torch.float32)
    logits[0, 3] = 8.0
    logits[1, 17] = 8.0
    logits[2, 31] = 8.0
    logits[3, 49] = 8.0
    labels = torch.tensor([3, 17, 31, 49])
    metrics = extended_classification_metrics(logits, labels, 120)
    assert metrics["total"] == 4
    assert metrics["correct"] == 4
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert 0.0 <= metrics["brier"] <= 2.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["mean_max_softmax_probability"] <= 1.0


def test_token_pool_shape_guard_rejects_pooled_features() -> None:
    with pytest.raises(ValueError):
        validate_token_shape(torch.zeros((768,)))
    validate_token_shape(torch.zeros((9216, 768), dtype=torch.float16))


def test_repeated_first_frame_contract() -> None:
    clip = torch.arange(3 * 4 * 2 * 2, dtype=torch.float32).reshape(3, 4, 2, 2)
    repeated = repeat_first_frame(clip)
    assert torch.equal(repeated[:, 0:1].expand_as(repeated), repeated)
    assert not torch.equal(clip, repeated)


def test_early_stopping_uses_macro_f1_then_nll() -> None:
    state = EarlyStoppingState()
    assert state.update(epoch=1, macro_f1=0.70, nll=0.80, min_delta=0.001)
    # Within min_delta, lower NLL is the declared tie-break.
    assert state.update(epoch=2, macro_f1=0.7005, nll=0.60, min_delta=0.001)
    assert state.best_epoch == 2
    # Equal macro-F1 and worse NLL cannot replace an earlier best epoch.
    assert not state.update(epoch=3, macro_f1=0.7005, nll=0.70, min_delta=0.001)
    assert state.best_epoch == 2


def test_early_stopping_patience_has_no_off_by_one() -> None:
    state = EarlyStoppingState()
    state.update(epoch=1, macro_f1=1.0, nll=0.1, min_delta=0.001)
    for epoch in range(2, 6):
        assert not state.update(epoch=epoch, macro_f1=1.0, nll=0.2, min_delta=0.001)
        assert not state.should_stop(epoch=epoch, min_epochs=5, patience=5)
    assert not state.should_stop(epoch=5, min_epochs=5, patience=5)
    state.update(epoch=6, macro_f1=1.0, nll=0.2, min_delta=0.001)
    assert state.bad_epochs == 5
    assert state.should_stop(epoch=6, min_epochs=5, patience=5)


def test_early_stopping_state_round_trip_preserves_resume_state() -> None:
    state = EarlyStoppingState(
        best_epoch=4, best_macro_f1=0.91, best_nll=0.2, bad_epochs=3
    )
    restored = EarlyStoppingState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()
