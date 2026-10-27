#!/usr/bin/env python3
"""Create the locked Stage0/formal watcher job manifests for 54 runs."""

from __future__ import annotations
import os

import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
TRAIN = EXP / "code/train_state_video.py"
SEEDS = (20260953, 20260954, 20260955)
HORIZONS = (30, 60)
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_job(job_id: str, output: Path, command: list[str], log: Path) -> dict[str, object]:
    return {
        "job_id": job_id,
        "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_V1",
        "output_root": str(output.resolve()),
        "command": [str(PYTHON), *command],
        "resource_estimate": {"estimate_type": "conservative_pre_stage0_peak_mib", "measured_peak_mib": 4096},
        "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "log_path": str(log.resolve()),
        "max_oom_requeues": 1,
        "final_test_access": 0,
        "encoder_updated": False,
    }


def main() -> int:
    job_dir = EXP / "configs/jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    stage0: list[dict[str, object]] = []
    formal: list[dict[str, object]] = []
    # This order interleaves state/video/horizon rather than binding one branch to a card.
    for seed in SEEDS:
        for cell in CELLS:
            for horizon in HORIZONS:
                run_id = f"SIV-{cell}-H{horizon}-S{seed}"
                formal_root = EXP / f"runs/{run_id}/formal"
                base = [str(TRAIN), "--cell", cell, "--horizon", str(horizon), "--seed", str(seed), "--run-root", str(formal_root), "--batch-size", "64", "--accumulation", "1"]
                stage_root = EXP / f"runs/{run_id}/stage0"
                stage_command = [str(TRAIN), "--cell", cell, "--horizon", str(horizon), "--seed", str(seed), "--run-root", str(stage_root), "--stage0", "--max-optimizer-steps", "4", "--batch-size", "64", "--accumulation", "1"]
                formal_job = make_job(run_id + "-FORMAL", formal_root, base, EXP / f"logs/child_{run_id}_formal.log")
                formal.append(formal_job)
                write(job_dir / f"{run_id}-FORMAL.json", formal_job)
                if seed == SEEDS[0]:
                    stage_job = make_job(run_id + "-STAGE0", stage_root, stage_command, EXP / f"logs/child_{run_id}_stage0.log")
                    stage0.append(stage_job)
                    write(job_dir / f"{run_id}-STAGE0.json", stage_job)
    write(EXP / "configs/SIV_STAGE0_JOBS.json", {"status": "LOCKED", "jobs": stage0, "count": len(stage0), "pilot_seed": SEEDS[0], "final_test_access": 0})
    write(EXP / "configs/SIV_FORMAL_JOBS.json", {"status": "LOCKED_PENDING_STAGE0", "jobs": formal, "count": len(formal), "final_test_access": 0})
    print(json.dumps({"status": "PASS", "stage0": len(stage0), "formal": len(formal), "interleaved": True, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
