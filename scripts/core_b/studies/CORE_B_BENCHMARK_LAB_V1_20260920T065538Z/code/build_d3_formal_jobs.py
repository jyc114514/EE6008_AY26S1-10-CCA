from __future__ import annotations
import os

import json

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json


PYTHON = os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python')
TRAIN = str(BENCH_ROOT / "code/d3_train.py")
CONFIGS = ["TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"]


def main() -> int:
    jobs = []
    for seed in (20261001, 20261002, 20261003):
        for horizon in (30, 60):
            for config in CONFIGS:
                job_id = f"D3-{config}-H{horizon}-S{seed}-FORMAL"
                root = BENCH_ROOT / "d3/runs" / job_id
                job = {"job_id": job_id, "protocol_id": "CORE_B_D3_TIME_SPACE_FACTORIAL_V1", "output_root": str(root), "command": [PYTHON, TRAIN, "--config", config, "--horizon", str(horizon), "--seed", str(seed), "--run-root", str(root), "--batch-size", "64", "--accumulation", "1"], "resource_estimate": {"estimate_type": "D3_STAGE0_MEASURED_OR_PRE_REGISTERED", "measured_peak_mib": 12288}, "log_path": str(BENCH_ROOT / "logs" / f"child_{job_id}.log"), "max_oom_requeues": 1, "environment": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}, "final_test_access": 0}
                path = BENCH_ROOT / "configs" / f"{job_id}.json"; atomic_write_json(path, job)
                jobs.append({"job_id": job_id, "job_path": str(path), "status_path": str(BENCH_ROOT / "logs" / f"watcher_{job_id}.json")})
    atomic_write_json(BENCH_ROOT / "configs/D3_FORMAL_TRAIN_JOBS.json", {"status": "LOCKED_AFTER_STAGE0", "jobs": jobs, "final_test_access": 0})
    print(json.dumps({"jobs": len(jobs)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
