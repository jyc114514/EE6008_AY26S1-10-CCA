import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from compute_v2_metrics import lag_score


def test_delayed_prediction_has_negative_best_lag():
    target = np.repeat(np.arange(30, dtype=float)[None, :, None], 28, axis=2)
    prediction = np.zeros_like(target)
    prediction[:, 1:] = target[:, :-1]
    scores = {lag: lag_score(prediction, target, lag)[0] for lag in range(-5, 6)}
    assert min(scores, key=scores.get) == -1
