#!/usr/bin/env python3
"""Create locked dynamic-watcher jobs for Stage0, formal training and 72 bundles."""

from __future__ import annotations
import os

import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
SEEDS = (20260956, 20260957, 20260958)
CELLS = ("CLEAN-N", "CLEAN-D", "CLEAN-S", "ROBUST-N", "ROBUST-D", "ROBUST-S")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def job(job_id, output, command, log):
    return {"job_id": job_id, "protocol_id": "CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1", "output_root": str(output.resolve()), "command": [str(PYTHON), *command], "resource_estimate": {"estimate_type": "conservative_pre_stage0_peak_mib", "measured_peak_mib": 4096}, "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, "log_path": str(log.resolve()), "max_oom_requeues": 1, "final_test_access": 0, "encoder_updated": False, "new_video_extraction_started": False}


def main() -> int:
    jobs_dir = EXP / "configs/jobs"; jobs_dir.mkdir(parents=True, exist_ok=True); stage0 = []; formal = []; inference = []
    for seed in SEEDS:
        for cell in CELLS:
            run_id = f"ODR-{cell}-H60-S{seed}"; formal_root = EXP / f"runs/{run_id}/formal"
            formal_job = job(run_id + "-FORMAL", formal_root, [str(EXP / "code/train_delay.py"), "--cell", cell, "--horizon", "60", "--seed", str(seed), "--run-root", str(formal_root), "--batch-size", "64", "--accumulation", "1"], EXP / f"logs/child_{run_id}_formal.log")
            formal.append(formal_job); write(jobs_dir / f"{run_id}-FORMAL.json", formal_job)
            if seed == SEEDS[0]:
                stage_root = EXP / f"runs/{run_id}/stage0"; stage_job = job(run_id + "-STAGE0", stage_root, [str(EXP / "code/train_delay.py"), "--cell", cell, "--horizon", "60", "--seed", str(seed), "--run-root", str(stage_root), "--stage0", "--max-optimizer-steps", "4", "--batch-size", "64", "--accumulation", "1"], EXP / f"logs/child_{run_id}_stage0.log")
                stage0.append(stage_job); write(jobs_dir / f"{run_id}-STAGE0.json", stage_job)
            for delay_index in range(4):
                bundle_job = job(run_id + f"-D{delay_index}-INFER", EXP / f"predictions/{run_id}", [str(EXP / "code/infer_delay.py"), "--cell", cell, "--seed", str(seed), "--delay-index", str(delay_index), "--run-id", run_id], EXP / f"logs/child_{run_id}_D{delay_index}_infer.log")
                inference.append(bundle_job); write(jobs_dir / f"{run_id}-D{delay_index}-INFER.json", bundle_job)
    write(EXP / "configs/ODR_STAGE0_JOBS.json", {"status": "LOCKED", "jobs": stage0, "count": len(stage0), "final_test_access": 0}); write(EXP / "configs/ODR_FORMAL_JOBS.json", {"status": "LOCKED_PENDING_STAGE0", "jobs": formal, "count": len(formal), "final_test_access": 0}); write(EXP / "configs/ODR_INFERENCE_JOBS.json", {"status": "LOCKED_PENDING_FORMAL", "jobs": inference, "count": len(inference), "final_test_access": 0})
    print(json.dumps({"status": "PASS", "stage0": len(stage0), "formal": len(formal), "inference": len(inference), "final_test_access": 0}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
