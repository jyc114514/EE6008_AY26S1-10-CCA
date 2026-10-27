from __future__ import annotations
import os

import json

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json


def main() -> int:
    job = {"job_id": "D3-FORMAL-INFERENCE", "protocol_id": "CORE_B_D3_TIME_SPACE_FACTORIAL_V1", "output_root": str(BENCH_ROOT / "d3/predictions"), "command": [os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python'), str(BENCH_ROOT / "code/d3_infer.py")], "resource_estimate": {"estimate_type": "D3_FORMAL_INFERENCE", "measured_peak_mib": 8192}, "log_path": str(BENCH_ROOT / "logs/D3-FORMAL-INFERENCE.log"), "max_oom_requeues": 1, "environment": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}, "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "configs/D3_INFERENCE_JOB.json", job)
    print(json.dumps(job))
    return 0


if __name__ == "__main__": raise SystemExit(main())
