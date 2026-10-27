"""Pure post-encoder video-feature interventions for Frozen fusion."""

from __future__ import annotations

import numpy as np


def zero_video_feature(normalized_features: np.ndarray) -> np.ndarray:
    return np.zeros_like(normalized_features, dtype=np.float32)


def donor_video_feature(normalized_features: np.ndarray, donor_positions: np.ndarray) -> np.ndarray:
    donor_positions = np.asarray(donor_positions, dtype=np.int64)
    if donor_positions.shape != (len(normalized_features),):
        raise ValueError("donor position shape mismatch")
    if np.any(donor_positions < 0) or np.any(donor_positions >= len(normalized_features)):
        raise ValueError("donor position outside recipient population")
    return np.asarray(normalized_features[donor_positions], dtype=np.float32).copy()
