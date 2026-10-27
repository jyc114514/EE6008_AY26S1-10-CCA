"""Create watcher queue entries for D3 Stage0 and resumable extraction."""

from __future__ import annotations
import os

import json
from pathlib import Path

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json


PYTHON = os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python')
SCRIPT = str(BENCH_ROOT / "code/d3_extract_grid.py")
LEASE_DIR = REPO_ROOT / "reports/locks/gpu_lease"


def job(job_id: str, args: list[str], output_root: Path, log_path: Path) -> dict[str, object]:
    return {"job_id": job_id, "protocol_id": "CORE_B_D3_TIME_SPACE_FACTORIAL_V1", "output_root": str(output_root), "command": [PYTHON, SCRIPT, *args], "resource_estimate": {"estimate_type": "D3_STAGE0_PRE_REGISTERED_CONSERVATIVE", "measured_peak_mib": 12288}, "log_path": str(log_path), "max_oom_requeues": 1, "environment": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}, "final_test_access": 0}


def main() -> int:
    configs = {
        "stage0": job("D3-GRID-STAGE0", ["--stage0", "--stage0-rows", "16"], BENCH_ROOT / "d3/runs/D3-GRID-STAGE0", BENCH_ROOT / "logs/D3-GRID-STAGE0.log"),
        "full": job("D3-GRID-FULL", ["--resume"], BENCH_ROOT / "d3/runs/D3-GRID-FULL", BENCH_ROOT / "logs/D3-GRID-FULL.log"),
    }
    for name, value in configs.items():
        atomic_write_json(BENCH_ROOT / f"configs/D3_GRID_{name.upper()}_JOB.json", value)
    atomic_write_json(BENCH_ROOT / "configs/D3_SHARED_GPU_CONTRACT.json", {"lease_dir": str(LEASE_DIR), "max_children": 2, "one_child_per_gpu": True, "physical_gpu": "dynamic_all_visible", "gpu0_observed_busy": True, "gpu1_gpu2_gpu3_observed_memory_capacity_at_lock": "see host nvidia-smi evidence in run report", "no_external_signals": True, "final_test_access": 0})
    print(json.dumps({"stage0": str(BENCH_ROOT / "configs/D3_GRID_STAGE0_JOB.json"), "full": str(BENCH_ROOT / "configs/D3_GRID_FULL_JOB.json"), "lease_dir": str(LEASE_DIR)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
