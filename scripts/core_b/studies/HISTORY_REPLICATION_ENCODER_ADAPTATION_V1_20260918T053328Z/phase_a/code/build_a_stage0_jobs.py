#!/usr/bin/env python3
"""Create the locked Phase-A GPU Stage0 queue without touching old roots."""

from __future__ import annotations
import os

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
WATCHER = REPO / "tools/gpu_lease_watcher.py"
PYTHON = PROJECT / "envs/jyc/bin/python"
TRAINER = ROOT / "code/run_history_video_cell.py"
INPUT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z/inputs/HISTORY_VIDEO_INPUTS.npz"
LEASE_DIR = ROOT / "locks/gpu_lease"
SEED = 20260940
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
PRE_REGISTERED_PEAK_MIB = 2048


def main() -> int:
    jobs: list[dict[str, object]] = []
    job_dir = ROOT / "configs/jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        job_id = f"HRA-A-{cell}-STAGE0-S{SEED}"
        run_root = ROOT / f"runs/{cell}/stage0_seed{SEED}"
        path = job_dir / f"{job_id}.json"
        if path.exists() or run_root.exists():
            raise FileExistsError(f"refusing existing Stage0 artifact: {path} or {run_root}")
        job = {
            "job_id": job_id,
            "protocol_id": "HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_PHASE_A_STAGE0",
            "output_root": str(run_root),
            "log_path": str(ROOT / f"logs/child_{cell}_stage0_seed{SEED}.log"),
            "command": [
                str(PYTHON), str(TRAINER), "--cell", cell,
                "--run-root", str(run_root), "--input-root", str(INPUT),
                "--seed", str(SEED), "--stage0", "--max-optimizer-steps", "8",
                "--smoke-rows", "128", "--batch-size", "64", "--accumulation", "1",
                "--device", "cuda",
            ],
            "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "resource_estimate": {"measured_peak_mib": PRE_REGISTERED_PEAK_MIB, "estimate_type": "phase_a_preregistered_stage0_upper_bound"},
            "max_oom_requeues": 1,
            "final_test_access": 0,
        }
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs.append(job)
    payload = {
        "status": "LOCKED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "A",
        "attempt": "STAGE0",
        "seed": SEED,
        "jobs": jobs,
        "lease_dir": str(LEASE_DIR),
        "visible_gpu_pool": "dynamic_all_visible",
        "max_concurrent_project_gpu_children": 2,
        "one_child_per_physical_gpu": True,
        "final_test_access": 0,
    }
    (ROOT / "configs/A_STAGE0_JOBS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resource = {
        "status": "PRE_REGISTERED",
        "stage0_peak_upper_bound_mib": PRE_REGISTERED_PEAK_MIB,
        "stage0_required_free_memory_mib": PRE_REGISTERED_PEAK_MIB + 512,
        "stage0_accepted": False,
        "formal_measured_peak_mib": None,
        "formal_required_free_memory_mib": None,
        "final_test_access": 0,
    }
    (ROOT / "configs/A_RESOURCE_CONTRACT.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "jobs": len(jobs), "seed": SEED, "peak_upper_bound_mib": PRE_REGISTERED_PEAK_MIB}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
