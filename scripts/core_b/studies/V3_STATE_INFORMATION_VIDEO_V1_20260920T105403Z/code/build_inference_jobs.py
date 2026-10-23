#!/usr/bin/env python3
"""Create fixed-last inference jobs for every locked formal run."""

from __future__ import annotations
import os

import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
INFER = EXP / "code/infer_state_video.py"
SEEDS = (20260953, 20260954, 20260955)
HORIZONS = (30, 60)
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    jobs = []
    for seed in SEEDS:
        for cell in CELLS:
            for horizon in HORIZONS:
                run_id = f"SIV-{cell}-H{horizon}-S{seed}"
                job = {
                    "job_id": run_id + "-INFER",
                    "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_V1",
                    "output_root": str((EXP / f"predictions/{run_id}").resolve()),
                    "command": [str(PYTHON), str(INFER), "--cell", cell, "--horizon", str(horizon), "--seed", str(seed), "--run-id", run_id],
                    "resource_estimate": {"estimate_type": "stage0_measured_peak_plus_normal_variation", "measured_peak_mib": 1024},
                    "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    "log_path": str((EXP / f"logs/child_{run_id}_infer.log").resolve()),
                    "max_oom_requeues": 1,
                    "final_test_access": 0,
                    "encoder_updated": False,
                }
                jobs.append(job)
                write(EXP / f"configs/jobs/{run_id}-INFER.json", job)
    write(EXP / "configs/SIV_INFERENCE_JOBS.json", {"status": "LOCKED_PENDING_FORMAL", "jobs": jobs, "count": len(jobs), "final_test_access": 0})
    print(json.dumps({"status": "PASS", "inference_jobs": len(jobs), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
