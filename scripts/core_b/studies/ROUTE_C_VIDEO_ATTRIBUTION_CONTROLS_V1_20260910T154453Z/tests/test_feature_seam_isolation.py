import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from apply_fusion_feature_interventions import apply_feature_intervention


def test_zero_feature_does_not_mutate_recipient():
    recipient = np.ones((3, 768), dtype=np.float32)
    before = recipient.copy()
    changed = apply_feature_intervention(recipient, "R1")
    assert np.array_equal(recipient, before)
    assert np.all(changed == 0)


def test_donor_feature_is_copied():
    recipient = np.ones((2, 768), dtype=np.float32)
    donor = np.full((2, 768), 2, dtype=np.float32)
    changed = apply_feature_intervention(recipient, "R2", donor=donor)
    assert np.array_equal(changed, donor)
    assert not np.shares_memory(changed, donor)
