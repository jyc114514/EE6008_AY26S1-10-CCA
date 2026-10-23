from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
SCHED_CODE = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
if str(SCHED_CODE) not in sys.path:
    sys.path.insert(0, str(SCHED_CODE))
if str(LEARN / "code") not in sys.path:
    sys.path.insert(0, str(LEARN / "code"))
from d3_model import D3TimeSpacePredictor, parameter_count, state_hash
from future_query_model import FixedFutureQueryOverride
import run_tiny_fit as old_runner
from orthogonal_query import hash_float32_array


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(run_dir: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("fresh query checkpoint verifier requires CUDA_VISIBLE_DEVICES empty")
    old_runner.set_cpu_runtime(1)
    checkpoint_path = run_dir / "checkpoint_query_model.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "checkpoint_stage0.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = str(payload["config"])
    panel = str(payload["panel"])
    run_id = str(payload["run_id"])
    horizon = int(payload["horizon"])
    seed = int(payload["seed"])
    updates = int(payload["optimizer_updates"])
    if horizon != 30 or panel not in ("P1", "P8") or updates not in (0, 4, 2000):
        raise AssertionError(f"unexpected reload fixture scope: {run_id}")
    protocol = json.loads((LEARN / "protocol/B_PROTOCOL_LOCKED.json").read_text())
    init_path = PROJECT / f"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/runs/INIT_S{seed}.pt"
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    base = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config)
    base.load_state_dict(init_state, strict=True)
    if file_sha(init_path) != protocol["init_state_file_sha256"]:
        raise AssertionError("fresh process INIT file hash mismatch")
    candidate = payload["wrapper_state"]["future_query_table"]
    wrapper = FixedFutureQueryOverride(base, torch.zeros_like(candidate))
    wrapper.eval()
    missing = wrapper.load_state_dict(payload["wrapper_state"], strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise AssertionError("strict candidate wrapper state reload reported key differences")
    if state_hash(base) != payload["base_state_hash"]:
        raise AssertionError("fresh process base model state hash mismatch")
    if parameter_count(base) != int(payload["parameter_count"]):
        raise AssertionError("fresh process parameter count changed")
    if hash_float32_array(wrapper.future_query_table) != payload["query_table_sha256"]:
        raise AssertionError("candidate query buffer was overwritten or changed on reload")
    panels, metadata, _ = old_runner.load_panels()
    tensors = panels[panel]
    prediction_file = run_dir / ("PREDICTIONS_FINAL.npz" if updates == 2000 else f"PREDICTIONS_STEP_{updates:04d}.npz")
    with np.load(prediction_file, allow_pickle=False) as z:
        saved_ids = z["anchor_id"].astype("U96")
        saved_prediction = z["prediction_native"].astype(np.float32, copy=True)
    with torch.no_grad():
        residual = wrapper(tensors["state"], tensors["mask"], tensors["relative"],
                           tensors["dynamic"], tensors["static"], tensors["anchor"])
        native = tensors["anchor"][:, None, :] + tensors["sigma"][None, None, :] * residual
    current = native.cpu().numpy().astype(np.float32)
    id_equal = saved_ids.tolist() == metadata[panel]["anchor_id"].tolist()
    if not id_equal or not np.array_equal(saved_prediction, current):
        raise AssertionError("fresh process train-panel prediction differs from saved values")
    result = {
        "status": "PASS", "run_id": run_id, "config": config, "panel": panel,
        "horizon": horizon, "optimizer_updates": updates, "device": "cpu",
        "strict_state_dict_reload": True, "fresh_process": True,
        "candidate_query_hash": hash_float32_array(wrapper.future_query_table),
        "base_state_hash": state_hash(base), "parameter_count": parameter_count(base),
        "prediction_ids_equal": id_equal, "prediction_bitwise_equal": True,
        "prediction_sha256": file_sha(prediction_file), "dev_targets_opened": False,
        "final_or_heldout_access": 0,
    }
    (run_dir / "FRESH_PROCESS_RELOAD.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
