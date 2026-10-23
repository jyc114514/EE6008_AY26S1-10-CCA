import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from compute_v2_metrics import motion_descriptors


def test_constant_motion_is_zero():
    array = np.zeros((2, 30, 28))
    values = motion_descriptors(array)
    assert np.allclose(values["mean_step_motion"], 0)
    assert np.allclose(values["total_variation"], 0)


def test_linear_motion_has_positive_change():
    array = np.zeros((1, 30, 28))
    array[0, :, 14:28] = np.arange(30)[:, None]
    values = motion_descriptors(array)
    assert values["mean_step_motion"][0] > 0
    assert values["endpoint_change"][0] > 0
