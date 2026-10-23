#!/usr/bin/env python3
"""Frozen observation-delay transform and deterministic robust schedule."""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SIV = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
EXP = REPO / "reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
DELAYS = np.asarray([0.0, 0.25, 0.50, 1.00], dtype=np.float32)
STATE_DIM = 71
OBS_DIM = 43
ACTION_DIM = 28
HISTORY_STEPS = 32


def load_main() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in bundle.files}
    normalizer = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    return data, normalizer


def delay_transform_rows(
    data: dict[str, np.ndarray],
    normalizer: dict[str, object],
    rows: np.ndarray,
    delay_indices: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized delayed state, age, selected timestamps and source indices.

    Only columns 0:43 are replaced. Columns 43:71 remain the original command
    history at each query. The selected observation timestamp is never allowed
    past min(query timestamp, anchor timestamp - requested delay).
    """
    rows = np.asarray(rows, dtype=np.int64)
    if np.isscalar(delay_indices):
        delay_indices = np.full(len(rows), int(delay_indices), dtype=np.int64)
    delay_indices = np.asarray(delay_indices, dtype=np.int64)
    if len(rows) != len(delay_indices) or np.any(delay_indices < 0) or np.any(delay_indices >= len(DELAYS)):
        raise ValueError("rows/delay_indices mismatch")
    native = data["history_native"][rows].astype(np.float32, copy=True)
    query = data["history_query_timestamps"][rows].astype(np.float64)
    available = data["history_selected_timestamps"][rows].astype(np.float64)
    anchor_time = data["anchor_timestamp"][rows].astype(np.float64)
    cutoff = anchor_time - DELAYS[delay_indices].astype(np.float64)
    limit = np.minimum(query, cutoff[:, None])
    selected_indices = np.empty((len(rows), HISTORY_STEPS), dtype=np.int64)
    for i in range(len(rows)):
        selected_indices[i] = np.searchsorted(available[i], limit[i], side="right") - 1
    selected_indices = np.clip(selected_indices, 0, HISTORY_STEPS - 1)
    batch_indices = np.arange(len(rows))[:, None]
    delayed = native.copy()
    delayed[:, :, :OBS_DIM] = native[batch_indices, selected_indices, :OBS_DIM]
    selected_times = available[batch_indices, selected_indices]
    age = np.maximum(0.0, query - selected_times).astype(np.float32)
    state_mean = np.asarray(normalizer["state_mean"], dtype=np.float32)
    state_std = np.asarray(normalizer["state_safe_std"], dtype=np.float32)
    standardized = ((delayed - state_mean[None, None, :]) / state_std[None, None, :]).astype(np.float32)
    return standardized, age, selected_times.astype(np.float32), selected_indices


def robust_delay_index(seed: int, epoch: int, row_id: int) -> int:
    payload = f"{seed}|{epoch}|{row_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % len(DELAYS)


def robust_schedule(seed: int, epoch: int, row_ids: np.ndarray) -> np.ndarray:
    return np.asarray([robust_delay_index(seed, epoch, int(row_id)) for row_id in row_ids], dtype=np.int64)
