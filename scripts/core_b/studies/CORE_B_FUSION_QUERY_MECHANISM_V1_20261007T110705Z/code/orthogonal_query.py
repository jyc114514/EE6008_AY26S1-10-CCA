from __future__ import annotations
import os

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED_CODE = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
if str(SCHED_CODE) not in sys.path:
    sys.path.insert(0, str(SCHED_CODE))
from d3_model import sinusoidal


def canonical_sylvester_hadamard(width: int = 128) -> Tensor:
    """Canonical Sylvester ordering H_1, H_2, ...; no row/column permutation."""
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    h = torch.ones((1, 1), dtype=torch.float32, device="cpu")
    while h.shape[0] < width:
        top = torch.cat((h, h), dim=1)
        bottom = torch.cat((h, -h), dim=1)
        h = torch.cat((top, bottom), dim=0)
    return h


def old_future_table(horizon: int, width: int = 128) -> Tensor:
    if horizon not in (30, 60):
        raise ValueError("the locked Core B horizons are 30 and 60")
    times = torch.arange(1, horizon + 1, dtype=torch.float32, device="cpu") / 30.0
    return sinusoidal(times, width).detach().to(dtype=torch.float32, device="cpu")


def hadamard_future_table(horizon: int, width: int = 128) -> tuple[Tensor, Tensor, Tensor]:
    """Use canonical Hadamard rows 1..H and match old per-step row RMS."""
    if horizon not in (30, 60):
        raise ValueError("the locked Core B horizons are 30 and 60")
    basis = canonical_sylvester_hadamard(width)
    old = old_future_table(horizon, width)
    row_rms = torch.sqrt(torch.mean(old * old, dim=1))
    rows = basis[1 : horizon + 1]
    if rows.shape != old.shape:
        raise AssertionError("selected Hadamard row table shape mismatch")
    candidate = rows * row_rms[:, None]
    return candidate.contiguous(), old.contiguous(), row_rms.contiguous()


def hash_float32_array(value: Tensor | np.ndarray) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy() if isinstance(value, Tensor) else value, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def table_contract(horizon: int) -> dict[str, object]:
    candidate, old, row_rms = hadamard_future_table(horizon)
    return {
        "horizon": horizon,
        "width": 128,
        "row_indices_zero_based": list(range(1, horizon + 1)),
        "column_order": "canonical Sylvester order; no permutation",
        "dc_row_zero_excluded": True,
        "physical_step_seconds": [float(x) for x in (torch.arange(1, horizon + 1, dtype=torch.float32) / 30.0).tolist()],
        "old_row_rms": [float(x) for x in row_rms.tolist()],
        "old_table_sha256": hash_float32_array(old),
        "candidate_table_sha256": hash_float32_array(candidate),
        "candidate_shape": list(candidate.shape),
        "candidate_dtype": str(candidate.dtype),
        "trainable_parameters_added": 0,
    }
