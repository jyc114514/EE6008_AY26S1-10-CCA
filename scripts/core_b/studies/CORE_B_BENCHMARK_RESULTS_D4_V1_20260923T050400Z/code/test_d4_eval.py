"""Toy tests for same-absolute-time D4 rolling metrics and panel contracts."""

from __future__ import annotations

import numpy as np
import pytest

from d4_eval import bootstrap_counts, pair_metrics, summarize_groups, valid_common_frames, validate_query_panel


def test_seven_and_eight_frame_gaps_preserve_same_target_frames() -> None:
    seven = valid_common_frames(100, 107, 30, 200)
    eight = valid_common_frames(100, 108, 30, 200)
    assert seven.tolist() == list(range(108, 131))
    assert eight.tolist() == list(range(109, 131))
    assert len(seven) == 23 and len(eight) == 22


def test_empty_intersection_is_explicit() -> None:
    assert valid_common_frames(100, 107, 7, 200).size == 0
    assert valid_common_frames(100, 107, 30, 108).size == 0


def test_positive_negative_and_persistence_utility_directions() -> None:
    gt = np.zeros((4, 28), dtype=np.float64)
    bad = np.ones((4, 28), dtype=np.float64)
    good = np.zeros((4, 28), dtype=np.float64)
    gain = pair_metrics(bad, good, gt, np.ones(28))
    loss = pair_metrics(good, bad, gt, np.ones(28))
    same = pair_metrics(good, good, gt, np.ones(28))
    assert gain["Full28|U"] > 0
    assert loss["Full28|U"] < 0
    assert same["Full28|U"] == 0 and same["Full28|R"] == 0
    assert same["Full28|coord_error_tie_share"] == 1


def test_prediction_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError):
        pair_metrics(np.zeros((3, 28)), np.zeros((4, 28)), np.zeros((3, 28)), np.ones(28))


def test_query_panel_rejects_duplicate_query_and_accepts_7_8_gaps() -> None:
    rows = [
        {"episode_index": "1", "block_id": "1:dense0", "query_index": "0", "anchor_frame": "100"},
        {"episode_index": "1", "block_id": "1:dense0", "query_index": "1", "anchor_frame": "107"},
        {"episode_index": "1", "block_id": "1:dense0", "query_index": "2", "anchor_frame": "115"},
    ]
    validate_query_panel(rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate_query_panel(rows + [rows[1]])


def test_source_group_bootstrap_keeps_constant_group_mean() -> None:
    counts = bootstrap_counts(4)
    estimate, low, high = summarize_groups(np.full(4, 0.25), counts)
    assert estimate == pytest.approx(0.25)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)
