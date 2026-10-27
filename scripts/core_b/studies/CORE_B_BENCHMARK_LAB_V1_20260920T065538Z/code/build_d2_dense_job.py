from __future__ import annotations
import os

import json
from pathlib import Path

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json


def main() -> int:
    job = {"job_id": "D2-DENSE-V3-INFER", "protocol_id": "CORE_B_BENCHMARK_D2_COMMAND_CHANGE_V1", "output_root": str(BENCH_ROOT / "predictions_dense"), "command": [os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python'), str(BENCH_ROOT / "code/d2_dense_pipeline.py"), "--resume"], "resource_estimate": {"estimate_type": "D2_DENSE_FEATURE_AND_INFERENCE_STAGE0_MEASURED_FROM_V3", "measured_peak_mib": 8192}, "log_path": str(BENCH_ROOT / "logs/D2-DENSE-V3-INFER.log"), "max_oom_requeues": 1, "environment": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}, "final_test_access": 0}
    # argv items must be separate strings; keep the path with spaces as one item.
    job["command"][0] = os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python')
    atomic_write_json(BENCH_ROOT / "configs/D2_DENSE_JOB.json", job)
    atomic_write_json(BENCH_ROOT / "configs/D2_DENSE_GPU_CONTRACT.json", {"lease_dir": str(REPO_ROOT / "reports/locks/gpu_lease"), "measured_peak_mib": 8192, "final_test_access": 0})
    print(json.dumps(job))
    return 0


if __name__ == "__main__": raise SystemExit(main())
