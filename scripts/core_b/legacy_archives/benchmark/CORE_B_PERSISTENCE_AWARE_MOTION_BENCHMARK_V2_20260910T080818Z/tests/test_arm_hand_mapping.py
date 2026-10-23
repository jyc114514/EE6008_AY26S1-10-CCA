import numpy as np


def test_canonical_arm_hand_mapping():
    values = np.arange(28)
    assert values[0:14].tolist() == list(range(14))
    assert values[14:28].tolist() == list(range(14, 28))
