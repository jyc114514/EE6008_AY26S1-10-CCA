#!/usr/bin/env python3
"""Build recovery-only watcher jobs; no CUDA is touched by this script."""

from __future__ import annotations
import os

import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
RUNNER = RECOVERY / "code/recovery_runner.py"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def job(job_id: str, output_root: Path, log_path: Path, argv: list[str]) -> dict[str, object]:
    return {
        "job_id": job_id,
        "protocol_id": "CORE_B_B_INFERENCE_RECOVERY_NO_TRAINING",
        "output_root": str(output_root),
        "command": [str(PYTHON), str(RUNNER), *argv],
        "resource_estimate": {"estimate_type": "conservative formal-forward_peak_reserved_mib", "measured_peak_mib": 2482},
        "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "log_path": str(log_path),
        "max_oom_requeues": 1,
        "final_test_access": 0,
    }


def main() -> int:
    replay_dir = RECOVERY / "data/jobs/replay"
    infer_dir = RECOVERY / "data/jobs/infer"
    replay_jobs: list[dict[str, object]] = []
    infer_jobs: list[dict[str, object]] = []
    for cell in CELLS:
        for path_name in ("a", "b") if cell in ("FT-D-FRZ", "FT-D-LORA4", "FT-D-DENSE4") else ():
            job_id = f"B-RECOVERY-REPLAY-{cell}-{path_name}"
            value = job(job_id, RECOVERY / f"replay/{cell}/{path_name}", RECOVERY / f"logs/child_{job_id}.log", ["--mode", "replay", "--cell", cell, "--path", path_name])
            write(replay_dir / f"{job_id}.json", value)
            replay_jobs.append(value)
        job_id = f"B-RECOVERY-INFER-{cell}"
        value = job(job_id, RECOVERY / f"predictions/{cell}", RECOVERY / f"logs/child_{job_id}.log", ["--mode", "infer", "--cell", cell])
        write(infer_dir / f"{job_id}.json", value)
        infer_jobs.append(value)
    write(RECOVERY / "data/REPLAY_JOBS.json", {"status": "LOCKED", "jobs": replay_jobs, "count": len(replay_jobs), "max_project_gpu_children": 2, "final_test_access": 0})
    write(RECOVERY / "data/INFERENCE_JOBS.json", {"status": "LOCKED_PENDING_REPLAY", "jobs": infer_jobs, "count": len(infer_jobs), "max_project_gpu_children": 2, "final_test_access": 0})
    print(json.dumps({"replay_jobs": len(replay_jobs), "inference_jobs": len(infer_jobs), "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
