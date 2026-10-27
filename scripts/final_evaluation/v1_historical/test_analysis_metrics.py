from __future__ import annotations

import numpy as np
import pytest

from analysis_metrics import core_a_classification_metrics, core_b_per_anchor_metrics, core_b_task_macro


def test_core_a_top1_and_fixed_class_macro_f1_match_hand_oracle() -> None:
    labels = np.asarray([0, 1, 2, 2], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 2], dtype=np.int64)
    logits = np.full((4, 3), -10.0)
    logits[np.arange(4), predictions] = 10.0
    result = core_a_classification_metrics(logits, labels, class_count=3)
    assert result["top1_accuracy"] == pytest.approx(0.75)
    assert result["top5_accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(7.0 / 9.0)


def test_core_b_equal_half_block_weight_and_three_fixed_metrics() -> None:
    target = np.zeros((1, 30, 28), dtype=np.float64)
    anchor = np.zeros((1, 28), dtype=np.float64)
    sigma = np.ones(28, dtype=np.float64)
    fd_scale = np.ones(28, dtype=np.float64)
    prediction = np.zeros_like(target)
    prediction[:, :, :14] = 1.0
    result = core_b_per_anchor_metrics(prediction, target, anchor, sigma, fd_scale)
    assert result["position_mae"].tolist() == pytest.approx([0.5])
    assert result["safe_fd_l1"].tolist() == pytest.approx([0.0])
    assert result["seam_safe_fd_l1"].tolist() == pytest.approx([0.5])


def test_core_b_future_difference_uses_29_adjacent_pairs() -> None:
    target = np.zeros((1, 30, 28), dtype=np.float64)
    anchor = np.zeros((1, 28), dtype=np.float64)
    prediction = np.zeros_like(target)
    prediction[0, :, :14] = np.arange(30, dtype=np.float64)[:, None]
    result = core_b_per_anchor_metrics(prediction, target, anchor, np.ones(28), np.ones(28))
    assert result["safe_fd_l1"].tolist() == pytest.approx([0.5])


def test_core_b_episode_then_task_macro_is_not_flat_anchor_mean() -> None:
    per_anchor = {
        "position_mae": np.asarray([0.0, 2.0, 10.0], dtype=np.float64),
        "safe_fd_l1": np.asarray([1.0, 3.0, 5.0], dtype=np.float64),
        "seam_safe_fd_l1": np.asarray([0.0, 2.0, 4.0], dtype=np.float64),
    }
    episodes = np.asarray([1, 1, 2], dtype=np.int64)
    tasks = np.asarray([119, 119, 119], dtype=np.int64)
    summary, task_rows, episode_rows = core_b_task_macro(per_anchor, episodes, tasks, fixed_tasks=(119,))
    assert summary["position_mae"] == pytest.approx(5.5)
    assert summary["safe_fd_l1"] == pytest.approx(3.5)
    assert summary["seam_safe_fd_l1"] == pytest.approx(2.5)
    assert task_rows[0]["episode_support"] == 2
    assert [row["anchor_support"] for row in episode_rows] == [2, 1]


def test_core_b_missing_fixed_task_is_not_estimable() -> None:
    per_anchor = {name: np.asarray([1.0]) for name in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")}
    summary, task_rows, _ = core_b_task_macro(per_anchor, [1], [119])
    assert summary["status"] == "not_estimable"
    assert summary["missing_tasks"] == [94, 90, 141, 230]
    assert len(task_rows) == 5


def test_core_b_rejects_nonfinite_or_wrong_shaped_targets() -> None:
    prediction = np.zeros((1, 30, 28), dtype=np.float64)
    anchor = np.zeros((1, 28), dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        target = prediction.copy(); target[0, 0, 0] = np.nan
        core_b_per_anchor_metrics(prediction, target, anchor, np.ones(28), np.ones(28))
    with pytest.raises(ValueError, match=r"\[N,30,28\]"):
        core_b_per_anchor_metrics(prediction[:, :29], prediction, anchor, np.ones(28), np.ones(28))
