#!/usr/bin/env python3
"""Create the 12 locked Phase-A fresh-seed formal watcher jobs."""

from __future__ import annotations
import os

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT / "envs/jyc/bin/python"
TRAINER = ROOT / "code/run_history_video_cell.py"
INPUT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/inputs/HISTORY_VIDEO_INPUTS.npz"
LEASE_DIR = ROOT / "locks/gpu_lease"
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
SEEDS = (20260940, 20260941, 20260942)


def main() -> int:
    stage0_paths = [ROOT / f"runs/{cell}/stage0_seed{SEEDS[0]}/STAGE0_REPORT.json" for cell in CELLS]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in stage0_paths]
    if any(report.get("status") != "PASS" or report.get("optimizer_steps") != 8 for report in reports):
        raise RuntimeError("all four Phase-A Stage0 reports must pass before formal job creation")
    peaks = [int(report.get("peak_memory_reserved_mib", 0)) for report in reports]
    if any(peak <= 0 for peak in peaks):
        raise RuntimeError("Stage0 did not report positive reserved memory")
    measured_peak = max(peaks)
    safety_margin = max(512, int((measured_peak * 0.10) + 0.999999))
    required_free = measured_peak + safety_margin
    jobs: list[dict[str, object]] = []
    job_dir = ROOT / "configs/jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for cell in CELLS:
            job_id = f"HRA-A-{cell}-FORMAL-S{seed}"
            run_root = ROOT / f"runs/{cell}/seed{seed}/formal"
            path = job_dir / f"{job_id}.json"
            if path.exists() or run_root.exists():
                raise FileExistsError(f"refusing existing formal artifact: {path} or {run_root}")
            job = {
                "job_id": job_id,
                "protocol_id": "HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_PHASE_A_FORMAL",
                "output_root": str(run_root),
                "log_path": str(ROOT / f"logs/child_{cell}_seed{seed}_formal.log"),
                "command": [
                    str(PYTHON), str(TRAINER), "--cell", cell,
                    "--run-root", str(run_root), "--input-root", str(INPUT),
                    "--seed", str(seed), "--batch-size", "64", "--accumulation", "1", "--device", "cuda",
                ],
                "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                "resource_estimate": {"measured_peak_mib": measured_peak, "estimate_type": "phase_a_stage0_measured_peak_reserved_mib"},
                "max_oom_requeues": 1,
                "final_test_access": 0,
            }
            path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            jobs.append(job)
    resource = {
        "status": "PASS",
        "stage0_reports": [str(path) for path in stage0_paths],
        "stage0_measured_peak_mib_by_cell": dict(zip(CELLS, peaks)),
        "formal_measured_peak_mib": measured_peak,
        "formal_safety_margin_mib": safety_margin,
        "formal_required_free_memory_mib": required_free,
        "formal_jobs_created": len(jobs),
        "final_test_access": 0,
    }
    (ROOT / "configs/A_RESOURCE_CONTRACT.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "LOCKED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "A",
        "cells": list(CELLS),
        "seeds": list(SEEDS),
        "run_count": len(jobs),
        "epochs": 40,
        "optimizer_steps": 1720,
        "checkpoint_policy": "last_epoch_only",
        "input_path": str(INPUT),
        "jobs": jobs,
        "lease_dir": str(LEASE_DIR),
        "visible_gpu_pool": "dynamic_all_visible",
        "max_concurrent_project_gpu_children": 2,
        "final_test_access": 0,
    }
    (ROOT / "configs/A_FORMAL_JOBS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_jobs": len(jobs), "measured_peak_mib": measured_peak, "required_free_memory_mib": required_free}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
