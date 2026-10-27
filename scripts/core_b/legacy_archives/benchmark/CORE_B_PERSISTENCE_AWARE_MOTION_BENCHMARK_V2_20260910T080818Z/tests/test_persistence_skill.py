import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from compute_v2_metrics import safe_relative


def test_positive_skill_convention():
    relative, reason = safe_relative(0.5, 1.0)
    assert relative == 0.5
    assert reason == ""


def test_zero_persistence_denominator_is_na():
    relative, reason = safe_relative(0.5, 0.0)
    assert relative == ""
    assert reason
