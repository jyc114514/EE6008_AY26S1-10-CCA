from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
SCHED_CODE = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
for _p in (SCHED_CODE, LEARN / "code", ROOT / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from d3_model import D3TimeSpacePredictor, parameter_count, state_hash
from d3_eq_contract import per_sample_loss
from collapse_diagnostic_core import loss_components
from future_query_model import FixedFutureQueryOverride
from orthogonal_query import canonical_sylvester_hadamard, hadamard_future_table, hash_float32_array
import run_tiny_fit as old_runner


@pytest.fixture(scope="session")
def source_data():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("CPU-only contract tests require CUDA_VISIBLE_DEVICES empty")
    old_runner.set_cpu_runtime(1)
    panels, metadata, panel_lock = old_runner.load_panels()
    protocol = json.loads((LEARN / "protocol/B_PROTOCOL_LOCKED.json").read_text())
    init_state = old_runner.load_init(protocol)
    return panels, metadata, panel_lock, protocol, init_state


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_base(config: str, horizon: int, seed: int, init_state):
    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config)
    model.load_state_dict(init_state, strict=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    return model


def test_hadamard_is_canonical_dc_free_orthogonal_and_rms_matched():
    h1 = canonical_sylvester_hadamard(128)
    h2 = canonical_sylvester_hadamard(128)
    assert torch.equal(h1, h2)
    assert torch.equal(h1[0], torch.ones(128, dtype=torch.float32))
    for horizon in (30, 60):
        candidate, old, row_rms = hadamard_future_table(horizon)
        candidate2, _, _ = hadamard_future_table(horizon)
        assert torch.equal(candidate, candidate2)
        assert candidate.shape == (horizon, 128)
        assert torch.max(torch.abs(candidate.mean(dim=1))).item() < 1e-7
        normed = candidate / torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
        gram = normed @ normed.T
        assert torch.allclose(gram, torch.eye(horizon), atol=1e-6, rtol=1e-6)
        candidate_rms = torch.sqrt(torch.mean(candidate * candidate, dim=1))
        old_rms = torch.sqrt(torch.mean(old * old, dim=1))
        assert torch.allclose(candidate_rms, old_rms, atol=1e-7, rtol=1e-6)
        assert torch.allclose(candidate_rms, row_rms, atol=1e-7, rtol=1e-6)
        assert hash_float32_array(candidate) == hash_float32_array(candidate2)


def test_h30_is_h60_prefix_and_absolute_step_table_is_stable():
    q30, old30, _ = hadamard_future_table(30)
    q60, old60, _ = hadamard_future_table(60)
    assert torch.equal(q30, q60[:30])
    assert torch.equal(old30, old60[:30])
    times = torch.arange(1, 61, dtype=torch.float32) / 30.0
    assert times[0].item() == pytest.approx(1.0 / 30.0)
    assert times[29].item() == pytest.approx(1.0)
    assert times[59].item() == pytest.approx(2.0)


def test_only_future_query_branch_changes_and_init_stays_persistence(source_data):
    panels, metadata, panel_lock, protocol, init_state = source_data
    for config in ("TS11-D", "STATE-CTRL"):
        for horizon in (30, 60):
            base = make_base(config, horizon, int(protocol["seed"]), init_state)
            before_state = state_hash(base)
            before_parameters = parameter_hash(base)
            before_count = parameter_count(base)
            candidate, old, _ = hadamard_future_table(horizon)
            wrapped_old = FixedFutureQueryOverride(base, old)
            wrapped_candidate = FixedFutureQueryOverride(base, candidate)
            assert parameter_count(wrapped_candidate) == before_count
            assert parameter_hash(base) == before_parameters
            assert state_hash(base) == before_state
            assert sum(p.numel() for p in wrapped_candidate.parameters()) == before_count
            assert "future_query_table" in dict(wrapped_candidate.named_buffers())
            assert all("future_query_table" not in name for name, _ in wrapped_candidate.named_parameters())
            panel = panels["P1"]
            mask_before = panel["mask"].clone()
            with torch.no_grad():
                original = base(panel["state"], panel["mask"], panel["relative"], panel["dynamic"], panel["static"], panel["anchor"])
                old_override = wrapped_old(panel["state"], panel["mask"], panel["relative"], panel["dynamic"], panel["static"], panel["anchor"])
                candidate_init = wrapped_candidate(panel["state"], panel["mask"], panel["relative"], panel["dynamic"], panel["static"], panel["anchor"])
            assert torch.allclose(original, old_override, atol=1e-5, rtol=1e-5)
            assert torch.equal(candidate_init, torch.zeros_like(candidate_init))
            native = panel["anchor"][:, None, :] + panel["sigma"][None, None, :] * candidate_init
            persistence = panel["anchor"][:, None, :].expand_as(native)
            assert torch.equal(native, persistence)
            assert torch.equal(panel["mask"], mask_before)
            assert state_hash(base) == before_state


def test_locked_panel_ids_targets_and_exact_input_collisions_unchanged(source_data):
    panels, metadata, panel_lock, protocol, init_state = source_data
    rows = panel_lock["rows"]
    for panel in ("P1", "P8"):
        expected = sorted((x for x in rows if x["panel"] == panel), key=lambda x: int(x["panel_order"]))
        assert [x["anchor_id"] for x in expected] == metadata[panel]["anchor_id"].tolist()
        assert len({x["source_group_id"] for x in expected}) == len(expected)
        assert all(float(x["panel_objective_weight"]) == 1.0 for x in expected)
        for i, row in enumerate(expected):
            target = np.ascontiguousarray(panels[panel]["target"][i].numpy())
            assert hashlib.sha256(target.tobytes()).hexdigest() == row["target_signature"]
    assert metadata["P1"]["anchor_id"].tolist()[0] == metadata["P8"]["anchor_id"].tolist()[0]
    assert len(metadata["P8"]["anchor_id"]) == 8


def test_ground_truth_injection_is_zero_for_full_and_position_losses(source_data):
    panels, metadata, panel_lock, protocol, init_state = source_data
    for panel in ("P1", "P8"):
        t = panels[panel]
        gt_residual = (t["target"] - t["anchor"][:, None, :]) / t["sigma"][None, None, :]
        parts = loss_components(gt_residual, t["target"], t["anchor"], t["sigma"], t["safe_fd"])
        full = parts["full"].mean()
        position = parts["position"].mean()
        assert torch.allclose(full, torch.zeros_like(full), atol=1e-6, rtol=0)
        assert torch.allclose(position, torch.zeros_like(position), atol=1e-6, rtol=0)
        locked = per_sample_loss(gt_residual, t["target"], t["anchor"], t["sigma"], t["safe_fd"])
        assert torch.allclose(locked, parts["full"], atol=1e-6, rtol=0)


def test_query_buffer_survives_init_reload_and_fresh_process(source_data, tmp_path):
    panels, metadata, panel_lock, protocol, init_state = source_data
    config, panel, horizon, seed = "TS11-D", "P1", 30, int(protocol["seed"])
    base = make_base(config, horizon, seed, init_state)
    candidate, _, _ = hadamard_future_table(horizon)
    wrapper = FixedFutureQueryOverride(base, candidate)
    expected_hash = hash_float32_array(candidate)
    wrapper.base.load_state_dict(init_state, strict=True)
    assert hash_float32_array(wrapper.future_query_table) == expected_hash
    t = panels[panel]
    with torch.no_grad():
        residual = wrapper(t["state"], t["mask"], t["relative"], t["dynamic"], t["static"], t["anchor"])
        prediction = t["anchor"][:, None, :] + t["sigma"][None, None, :] * residual
    run_dir = tmp_path / "fresh_process_fixture"
    run_dir.mkdir()
    payload = {
        "run_id": "TEST-FRESH-QUERY-RELOAD", "config": config, "panel": panel,
        "horizon": horizon, "seed": seed, "optimizer_updates": 0,
        "wrapper_state": wrapper.state_dict(), "base_state_hash": state_hash(base),
        "parameter_count": parameter_count(base), "query_table_sha256": expected_hash,
    }
    torch.save(payload, run_dir / "checkpoint_stage0.pt")
    with (run_dir / "PREDICTIONS_STEP_0000.npz").open("wb") as f:
        np.savez_compressed(f, anchor_id=metadata[panel]["anchor_id"].astype("U96"),
                            prediction_native=prediction.numpy().astype(np.float32))
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES":"", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"})
    result = subprocess.run(
        [sys.executable, str(ROOT / "code/verify_query_checkpoint.py"), "--run-dir", str(run_dir)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fresh = json.loads((run_dir / "FRESH_PROCESS_RELOAD.json").read_text())
    assert fresh["status"] == "PASS"
    assert fresh["fresh_process"] is True
    assert fresh["prediction_bitwise_equal"] is True
    assert fresh["candidate_query_hash"] == expected_hash
