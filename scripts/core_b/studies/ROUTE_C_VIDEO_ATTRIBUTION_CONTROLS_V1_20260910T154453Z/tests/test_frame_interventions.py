import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from apply_pixel_interventions import apply_pixel_intervention


def test_pixel_interventions_preserve_contract():
    frames = np.arange(32 * 2 * 2 * 3, dtype=np.uint8).reshape(32, 2, 2, 3)
    expected = {"P0": list(range(32)), "P1": [0] * 32, "P2": [31] * 32, "P3": list(range(31, -1, -1))}
    for condition, positions in expected.items():
        transformed, actual = apply_pixel_intervention(frames, condition, seed=20260910, episode_id=7)
        assert actual.tolist() == positions
        assert transformed.shape == frames.shape
    shuffled, positions = apply_pixel_intervention(frames, "P4", seed=20260910, episode_id=7)
    assert sorted(positions.tolist()) == list(range(32))
    assert not np.array_equal(positions, np.arange(32))
    assert sorted(shuffled.reshape(-1).tolist()) == sorted(frames.reshape(-1).tolist())
