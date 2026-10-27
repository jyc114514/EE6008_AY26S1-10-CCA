"""CPU acceptance of 12 D3-EQ Stage0 cells, including fresh-process reload."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
CODE_ROOT = EQ_ROOT / "code"
PROTOCOL = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
TARGET_CACHE = EQ_ROOT / "d3/train_targets_trainonly.npz"
STAGE0_MANIFEST = EQ_ROOT / "configs/D3_EQ_STAGE0_JOBS.json"
GPU_HOUR_CAP = 48.0
PREDICTION_RESERVE_GPU_HOURS = 4.0


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def lease_hours(path: Path) -> float:
    if not path.is_file():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or int(data.get("exit_code", -1)) != 0:
        return 0.0
    start = datetime.fromisoformat(data["lease_acquired_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(data["lease_released_utc"].replace("Z", "+00:00"))
    return max((end - start).total_seconds() / 3600.0, 0.0)


def output_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.astype(np.float32)).tobytes()).hexdigest()


def reload_hash_fresh_process(checkpoint_path: Path, fixture_path: Path, config: str, horizon: int, seed: int) -> str:
    code = """
import hashlib,sys,numpy as np,torch
from d3_model import D3TimeSpacePredictor
torch.set_num_threads(1)
payload=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
model=D3TimeSpacePredictor(horizon=int(sys.argv[3]),seed=int(sys.argv[4]),video_mode=sys.argv[2])
model.load_state_dict(payload['model'],strict=True); model.eval()
with np.load(sys.argv[5],allow_pickle=False) as d:
    x=[torch.tensor(d[k]) for k in ('state','mask','relative','dynamic','static','anchor')]
with torch.inference_mode(): y=model(*x).cpu().numpy().astype(np.float32)
print(hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())
"""
    run = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint_path), config, str(horizon), str(seed), str(fixture_path)],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return run.stdout.strip()


def main() -> int:
    torch.set_num_threads(1)
    lock_hash = sha256_file(PROTOCOL)
    from d3_eq_train import load_train_bundle
    from d3_eq_contract import shuffled_epoch_order
    from d3_model import D3TimeSpacePredictor

    bundle = load_train_bundle()
    manifest = json.loads(STAGE0_MANIFEST.read_text(encoding="utf-8"))
    reports = []
    stage0_gpu_hours = 0.0
    expected_configs = {"TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"}
    cells = set()
    positions = int(shuffled_epoch_order(len(bundle["train_indices"]), 20261001, 1)[0])
    source_index = int(bundle["train_indices"][positions])
    dynamic, static = bundle["grid_store"].get(np.asarray([source_index], dtype=np.int64))
    dynamic = ((dynamic - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
    static = ((static - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
    inputs = bundle["inputs"]
    fixture_values = {
        "state": inputs["history_standardized"][positions:positions + 1].astype(np.float32),
        "mask": inputs["history_mask"][positions:positions + 1].astype(np.float32),
        "relative": inputs["history_relative_delta_standardized"][positions:positions + 1].astype(np.float32),
        "dynamic": dynamic,
        "static": static,
        "anchor": bundle["anchors"][positions:positions + 1].astype(np.float32),
    }

    with tempfile.TemporaryDirectory(prefix="d3eq-stage0-fixture-") as temporary:
        fixture_path = Path(temporary) / "train_fixture.npz"
        np.savez(fixture_path, **fixture_values)
        for item in manifest["jobs"]:
            job = json.loads(Path(item["job_path"]).read_text(encoding="utf-8"))
            root = Path(job["output_root"])
            report_path = root / "STAGE0_REPORT.json"
            checkpoint_path = root / "checkpoint_stage0.pt"
            if not report_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Stage0 output missing for {item['job_id']}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            key = (report.get("config"), int(report.get("horizon", -1)), int(report.get("seed", -1)))
            if key in cells or report.get("status") != "PASS" or int(report.get("optimizer_steps", -1)) != 4:
                raise AssertionError(f"Stage0 cell/status/step count invalid: {item['job_id']}")
            if (report.get("config"), int(report.get("horizon", -1)), int(report.get("seed", -1))) != (item["config"], int(item["horizon_steps"]), int(item["seed"])):
                raise AssertionError(f"Stage0 report/job identity mismatch: {item['job_id']}")
            if sha256_file(checkpoint_path) != report.get("checkpoint_sha256"):
                raise AssertionError(f"Stage0 checkpoint hash mismatch: {item['job_id']}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("protocol_hash") != lock_hash or int(payload.get("optimizer_steps", -1)) != 4 or payload.get("final_test_access") != 0:
                raise AssertionError(f"Stage0 checkpoint protocol/step/final gate mismatch: {item['job_id']}")
            if any(not torch.isfinite(value).all() for value in payload["model"].values() if torch.is_floating_point(value)):
                raise AssertionError(f"Stage0 checkpoint contains non-finite parameters: {item['job_id']}")
            if not np.isfinite(float(report_path.stat().st_size)):
                raise AssertionError("invalid Stage0 report file size")
            epoch_rows = list(csv.DictReader((root / "EPOCH_LOG.csv").open(newline="", encoding="utf-8")))
            if len(epoch_rows) != 1 or not np.isfinite(float(epoch_rows[0]["weighted_loss_sum_over_seen_rows"])):
                raise AssertionError(f"Stage0 loss report is missing/non-finite: {item['job_id']}")

            model = D3TimeSpacePredictor(horizon=int(item["horizon_steps"]), seed=int(item["seed"]), video_mode=str(item["config"]))
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            tensors = [torch.tensor(fixture_values[name]) for name in ("state", "mask", "relative", "dynamic", "static", "anchor")]
            with torch.inference_mode():
                expected = model(*tensors).cpu().numpy().astype(np.float32)
            parent_hash = output_sha(expected)
            child_hash = reload_hash_fresh_process(checkpoint_path, fixture_path, str(item["config"]), int(item["horizon_steps"]), int(item["seed"]))
            if parent_hash != child_hash:
                raise AssertionError(f"fresh-process checkpoint reload mismatch: {item['job_id']}")
            cell = {
                "job_id": item["job_id"], "config": item["config"], "horizon_steps": int(item["horizon_steps"]),
                "seed": int(item["seed"]), "optimizer_steps": 4,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "fresh_process_output_sha256": parent_hash,
                "peak_allocated_bytes": int(report["peak_allocated_bytes"]),
                "peak_reserved_bytes": int(report["peak_reserved_bytes"]),
                "loss_finite": True, "final_test_access": 0,
            }
            reports.append(cell)
            cells.add(key)
            stage0_gpu_hours += lease_hours(Path(item["status_path"]))

    if len(reports) != 12 or {config for config, horizon, seed in cells} != expected_configs or {horizon for config, horizon, seed in cells} != {30, 60} or {seed for config, horizon, seed in cells} != {20261001}:
        raise AssertionError("Stage0 acceptance does not cover the locked first-seed 12 cells")
    max_allocated = max(row["peak_allocated_bytes"] for row in reports)
    max_reserved = max(row["peak_reserved_bytes"] for row in reports)
    measured_peak_mib = math.ceil(max_reserved / (1024 * 1024))
    acceptance = {
        "status": "PASS",
        "accepted_cells": len(reports),
        "expected_cells": 12,
        "seed": 20261001,
        "steps_per_cell": 4,
        "fresh_process_reload_all_12": True,
        "max_peak_allocated_bytes": max_allocated,
        "max_peak_reserved_bytes": max_reserved,
        "formal_measured_peak_mib": measured_peak_mib,
        "formal_watcher_required_free_mib": measured_peak_mib + max(512, math.ceil(0.1 * measured_peak_mib)),
        "stage0_gpu_hours_actual": stage0_gpu_hours,
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "cells": reports,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json", acceptance)

    pre = json.loads((EQ_ROOT / "protocol/D3_EQ_RESOURCE_ESTIMATE_PRE_STAGE0.json").read_text(encoding="utf-8"))
    conservative_formal = float(pre["new_formal_36_conservative_gpu_hours"])
    projected = stage0_gpu_hours + conservative_formal + 4.0
    if projected > 48.0:
        raise RuntimeError("48 GPU-hour gate fails after actual Stage0 measurement")
    post = {
        "status": "PASS_RESOURCE_GATE" if projected <= 48.0 else "budget_insufficient",
        "measured_stage0_peak_mib": measured_peak_mib,
        "stage0_gpu_hours_actual": stage0_gpu_hours,
        "formal_36_conservative_gpu_hours": conservative_formal,
        "primary_and_dense_inference_reserved_gpu_hours": 4.0,
        "projected_gpu_hours": projected,
        "gpu_hour_ceiling": 48.0,
        "output_size_ceiling_gib": 50.0,
        "filesystem_free_fraction_min": 0.10,
        "filesystem_gate": "refresh through CPU-only resource snapshot before formal; refresh again before inference",
        "d3eq_train_cache_sha256": sha256_file(TARGET_CACHE),
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "protocol/D3_EQ_RESOURCE_ESTIMATE_POST_STAGE0.json", post)
    builder = EQ_ROOT / "code/build_d3_eq_jobs.py"
    build = subprocess.run([str(PROJECT_ROOT / "envs/jyc/bin/python"), str(builder), "--phase", "formal", "--peak-mib", str(measured_peak_mib)], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if build.returncode != 0:
        raise RuntimeError(f"formal job build failed after Stage0: {build.stdout}{build.stderr}")
    (EQ_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (EQ_ROOT / "logs/D3_EQ_FORMAL_QUEUE_BUILD.json").write_text(build.stdout + build.stderr, encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage0_cells": len(reports), "formal_measured_peak_mib": measured_peak_mib, "stage0_gpu_hours": stage0_gpu_hours, "projected_gpu_hours": projected, "fresh_process_reloads": 12, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
