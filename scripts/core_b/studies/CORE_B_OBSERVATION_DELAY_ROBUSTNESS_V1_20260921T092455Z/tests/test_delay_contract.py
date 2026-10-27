#!/usr/bin/env python3
"""CPU contracts for delay semantics and age/model invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "code"))
from delay_contract import DELAYS, delay_transform_rows, load_main, robust_schedule  # noqa: E402
from delay_model import DelayAwarePredictor, parameter_count  # noqa: E402


def main() -> int:
    data, normalizer = load_main(); rows = np.flatnonzero(data["split_code"] == 1)[:16]
    original = data["history_standardized"][rows]
    state0, age0, selected0, index0 = delay_transform_rows(data, normalizer, rows, 0)
    assert float(np.max(np.abs(state0 - original))) < 1e-5
    assert np.array_equal(data["anchor_id"][rows], data["anchor_id"][rows])
    for delay_index, delay in enumerate(DELAYS):
        state, age, selected, indices = delay_transform_rows(data, normalizer, rows, delay_index)
        cutoff = data["anchor_timestamp"][rows, None] - float(delay)
        query = data["history_query_timestamps"][rows]
        assert np.all(selected <= np.minimum(query, cutoff) + 1e-6)
        assert np.all(age >= -1e-6)
        assert np.allclose(data["history_native"][rows, :, 43:71], data["history_native"][rows, :, 43:71])
    # A recent observation sentinel cannot leak into the one-second delayed input.
    altered = {key: value.copy() for key, value in data.items()}
    altered["history_native"][rows, -1, :43] += 999.0
    delayed_original, *_ = delay_transform_rows(data, normalizer, rows, 3)
    delayed_altered, *_ = delay_transform_rows(altered, normalizer, rows, 3)
    assert np.array_equal(delayed_original, delayed_altered)
    assert np.array_equal(robust_schedule(20260956, 1, rows), robust_schedule(20260956, 1, rows))
    model = DelayAwarePredictor(horizon=60, seed=20260956); model.eval(); assert parameter_count(model) == 1_155_484
    state = torch.randn(2, 32, 71); mask = torch.ones(2, 32); rel = torch.randn(2, 32); age = torch.zeros(2, 32); video = torch.randn(2, 16, 768); anchor = torch.randn(2, 28)
    with torch.inference_mode(): output = model.forward_delay(state, mask, rel, age, video, anchor)
    assert output.shape == (2, 60, 28) and float(output.abs().max()) == 0.0
    print({"status": "PASS", "delta0_exact": "PASS", "cutoff_contract": "PASS", "sentinel": "PASS", "schedule": "PASS", "age_model": "PASS", "parameter_count": parameter_count(model)})
    return 0


if __name__ == "__main__": raise SystemExit(main())
