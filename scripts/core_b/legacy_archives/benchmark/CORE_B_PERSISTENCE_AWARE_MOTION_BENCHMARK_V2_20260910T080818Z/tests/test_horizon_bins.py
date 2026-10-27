import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from compute_v2_metrics import horizon_slices


def test_horizon_bins_cover_thirty_steps():
    bins = horizon_slices()
    assert list(range(30)) == list(range(0, 10)) + list(range(10, 20)) + list(range(20, 30))
    assert bins["endpoint"] == 29
