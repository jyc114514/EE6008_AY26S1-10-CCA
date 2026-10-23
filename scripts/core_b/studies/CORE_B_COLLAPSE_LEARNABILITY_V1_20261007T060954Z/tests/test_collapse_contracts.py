import os
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

OUT_ROOT = Path(__file__).resolve().parents[1]
SCHED_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'))
sys.path.insert(0, str(OUT_ROOT / "code"))
sys.path.insert(0, str(SCHED_ROOT / "code"))

from d3_eq_contract import per_sample_loss  # noqa: E402
from collapse_diagnostic_core import (  # noqa: E402
    choose_nested_high_panel,
    exact_input_signature,
    loss_components,
    reconstruct_native,
    select_stratified_groups,
    trajectory_decomposition,
)


def test_native_residual_round_trip_and_anchor_copy():
    anchor = torch.arange(56, dtype=torch.float32).reshape(2, 28)
    sigma = torch.linspace(0.2, 1.1, 28)
    residual = torch.zeros((2, 30, 28), dtype=torch.float32)
    native = reconstruct_native(anchor, residual, sigma)
    assert native.shape == (2, 30, 28)
    assert torch.equal(native, anchor[:, None, :].expand_as(native))
    recovered = (native - anchor[:, None, :]) / sigma[None, None, :]
    assert torch.equal(recovered, residual)


def test_loss_decomposition_matches_locked_full_loss():
    cpu_rng = torch.Generator(device="cpu").manual_seed(12)
    pred_residual = torch.randn(3, 30, 28, dtype=torch.float32, generator=cpu_rng) * 0.1
    anchor = torch.randn(3, 28, dtype=torch.float32, generator=cpu_rng)
    target = torch.randn(3, 30, 28, dtype=torch.float32, generator=cpu_rng)
    sigma = torch.linspace(0.2, 1.1, 28)
    safe_fd = torch.linspace(0.3, 1.3, 28)
    components = loss_components(pred_residual, target, anchor, sigma, safe_fd)
    locked = per_sample_loss(pred_residual, target, anchor, sigma, safe_fd)
    assert torch.allclose(components["position"] + components["weighted_fd"], locked, atol=1e-7, rtol=0)
    assert torch.allclose(components["full"], locked, atol=1e-7, rtol=0)


def test_ground_truth_injection_has_zero_position_and_future_internal_fd():
    anchor = torch.zeros((1, 28), dtype=torch.float32)
    sigma = torch.ones(28)
    safe_fd = torch.ones(28)
    target = torch.arange(30, dtype=torch.float32).view(1, 30, 1).expand(1, 30, 28)
    exact_residual = (target - anchor[:, None, :]) / sigma[None, None, :]
    components = loss_components(exact_residual, target, anchor, sigma, safe_fd)
    assert torch.allclose(components["position"], torch.zeros(1), atol=1e-7, rtol=0)
    assert torch.allclose(components["fd"], torch.zeros(1), atol=1e-7, rtol=0)
    # The horizon contains 30 positions and therefore exactly 29 internal future differences.
    assert components["internal_difference_count"] == 29


def test_time_constant_sample_offsets_are_not_anchor_copy():
    anchor = torch.zeros((2, 28), dtype=torch.float32)
    pred = torch.zeros((2, 30, 28), dtype=torch.float32)
    pred[0] += 0.25
    pred[1] += 1.25
    result = trajectory_decomposition(pred, anchor, tolerance=1e-7)
    assert result["anchor_copy"] is False
    assert result["time_constant_sample_fraction"] == 1.0
    assert result["between_sample_mean_offset_std"] == pytest.approx(0.5, abs=1e-7)
    assert result["within_sample_time_std_mean"] == pytest.approx(0.0, abs=1e-7)


def test_anchor_copy_is_distinguished_from_time_constant_offset():
    anchor = torch.randn((2, 28), dtype=torch.float32)
    pred = anchor[:, None, :].expand(2, 30, 28).clone()
    result = trajectory_decomposition(pred, anchor, tolerance=1e-7)
    assert result["anchor_copy"] is True
    assert result["anchor_copy_max_abs_offset"] == 0.0
    assert result["time_constant_sample_fraction"] == 1.0


def test_stratified_fixture_is_hash_deterministic_and_group_unique():
    records = []
    for stratum in ("low", "mid", "high"):
        for group_idx in range(6):
            for window_idx in range(2):
                records.append({
                    "anchor_id": f"{stratum}-{group_idx}-{window_idx}",
                    "source_group_id": f"{stratum}-group-{group_idx}",
                    "stratum": stratum,
                    "original_task_index": group_idx,
                })
    quotas = {"low": 2, "mid": 2, "high": 2}
    a = select_stratified_groups(records, quotas, protocol_id="test-v1")
    b = select_stratified_groups(records, quotas, protocol_id="test-v1")
    assert [x["anchor_id"] for x in a] == [x["anchor_id"] for x in b]
    assert len({x["source_group_id"] for x in a}) == 6
    assert {x["stratum"] for x in a} == set(quotas)
    assert {k: sum(x["stratum"] == k for x in a) for k in quotas} == quotas


def test_exact_input_signature_and_nested_collision_free_panel():
    records = []
    for idx in range(12):
        state = np.asarray([idx], dtype=np.float32)
        anchor = np.asarray([idx * 0.1], dtype=np.float32)
        ts11 = exact_input_signature({"state": state, "anchor": anchor, "video": np.asarray([idx + 1], dtype=np.float32)})
        state_sig = exact_input_signature({"state": state, "anchor": anchor, "video": np.zeros(1, dtype=np.float32)})
        records.append({"anchor_id": f"id-{idx}", "source_group_id": f"group-{idx}",
                        "original_task_index": idx % 3, "stratum": "high",
                        "signature_TS11-D": ts11, "signature_STATE-CTRL": state_sig,
                        "target_signature": hashlib.sha256(f"target-{idx}".encode()).hexdigest()})
    p1 = records[0]
    duplicate = {**records[0], "anchor_id": "duplicate-input", "source_group_id": "duplicate-group",
                 "target_signature": hashlib.sha256(b"different-target").hexdigest()}
    panel, rejected = choose_nested_high_panel(p1, [duplicate, *records[1:]], n_total=8, protocol_id="test-v1")
    assert len(panel) == 8 and panel[0]["anchor_id"] == p1["anchor_id"]
    assert len({x["source_group_id"] for x in panel}) == 8
    assert all(x["stratum"] == "high" for x in panel)
    assert any(x["anchor_id"] == "duplicate-input" and x["reason"] == "exact_input_collision" for x in rejected)
