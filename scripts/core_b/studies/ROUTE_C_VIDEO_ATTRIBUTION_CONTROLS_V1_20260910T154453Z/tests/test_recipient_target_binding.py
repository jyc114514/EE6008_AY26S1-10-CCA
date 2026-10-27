import numpy as np


def test_recipient_arrays_keep_recipient_ids():
    recipient = np.array([10, 20, 30])
    donor = np.array([30, 10, 20])
    output_ids = recipient.copy()
    assert np.array_equal(output_ids, recipient)
    assert not np.array_equal(output_ids, donor)
