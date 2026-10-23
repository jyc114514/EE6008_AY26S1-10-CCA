from __future__ import annotations

import numpy as np

from apply_fusion_feature_interventions import donor_video_feature, zero_video_feature
from apply_pixel_interventions import apply_pixel_intervention


def test_frame_intervention_maps() -> None:
    frames = np.arange(32 * 2 * 2 * 3, dtype=np.uint8).reshape(32, 2, 2, 3)
    for condition in ("P1", "P2", "P3", "P4"):
        transformed, positions = apply_pixel_intervention(frames, condition, seed=20260910, episode_id=376)
        assert transformed.shape == frames.shape
        assert not np.array_equal(positions, np.arange(32))
        assert np.array_equal(transformed, frames[positions])


def test_runner_does_not_resort_frames() -> None:
    frames = np.repeat(np.arange(32, dtype=np.uint8)[:, None, None, None], 3, axis=3)
    transformed, positions = apply_pixel_intervention(frames, "P3", seed=1, episode_id=1)
    assert positions.tolist() == list(range(31, -1, -1))
    assert transformed[:, 0, 0, 0].tolist() == list(range(31, -1, -1))


def test_zero_changes_only_video_feature() -> None:
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    zero = zero_video_feature(features)
    assert np.array_equal(zero, np.zeros_like(features))


def test_donor_feature_does_not_change_shape() -> None:
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    donors = donor_video_feature(features, np.array([1, 2, 0]))
    assert donors.tolist() == features[[1, 2, 0]].tolist()
