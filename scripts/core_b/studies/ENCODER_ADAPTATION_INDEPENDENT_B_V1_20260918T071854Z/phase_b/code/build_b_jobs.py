#!/usr/bin/env python3
"""Create bridge, Stage0 and post-Stage0 formal watcher jobs."""

from __future__ import annotations
import os

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT / "envs/jyc/bin/python"
RUNNER = ROOT / "code/run_b_encoder_adaptation.py"
LEASE_DIR = ROOT / "locks/gpu_lease"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")


def base_job(job_id: str, output_root: Path, log_path: Path, command: list[str], peak: int, protocol: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "protocol_id": protocol,
        "output_root": str(output_root),
        "log_path": str(log_path),
        "command": command,
        "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_LIBRARY_PATH": str(PROJECT / "envs/jyc/lib") + ":/usr/local/cuda-12.8/lib64"},
        "resource_estimate": {"measured_peak_mib": peak, "estimate_type": "independent_b_preregistered_or_stage0_measured"},
        "max_oom_requeues": 1,
        "final_test_access": 0,
    }


def main() -> int:
    job_dir = ROOT / "configs/jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    bridge_id = "B-BRIDGE-DYNAMIC"
    bridge_root = ROOT / "bridge"
    bridge_job = base_job(bridge_id, bridge_root, ROOT / "logs/child_bridge.log", [str(PYTHON), str(RUNNER), "--mode", "bridge", "--output", str(bridge_root / "RAW_CACHE_BRIDGE.json")], 10000, "CORE_B_ENCODER_ADAPTATION_INDEPENDENT_B_BRIDGE")
    (job_dir / f"{bridge_id}.json").write_text(json.dumps(bridge_job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage0_jobs = []
    for cell in CELLS:
        job_id = f"B-{cell}-STAGE0-S20260944"
        run_root = ROOT / f"runs/{cell}/stage0"
        job = base_job(job_id, run_root, ROOT / f"logs/child_{cell}_stage0.log", [str(PYTHON), str(RUNNER), "--mode", "train", "--cell", cell, "--run-root", str(run_root), "--stage0"], 20000, "CORE_B_ENCODER_ADAPTATION_INDEPENDENT_B_STAGE0")
        path = job_dir / f"{job_id}.json"
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage0_jobs.append(job)
    payload = {"status": "LOCKED", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bridge": bridge_job, "stage0": stage0_jobs, "peak_upper_bound_mib": 20000, "required_free_memory_mib": 22000, "max_project_gpu_children": 2, "visible_gpu_pool": "dynamic_all_visible", "final_test_access": 0}
    (ROOT / "configs/B_STAGE0_JOBS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "bridge_jobs": 1, "stage0_jobs": len(stage0_jobs), "required_free_memory_mib": 22000}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
