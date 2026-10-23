import numpy as np


def test_pairing_uses_same_episode_order():
    ids = np.array([4, 5, 9])
    other = np.array([9, 4, 5])
    lookup = {int(value): index for index, value in enumerate(other)}
    aligned = np.array([lookup[int(value)] for value in ids])
    assert aligned.tolist() == [1, 2, 0]
