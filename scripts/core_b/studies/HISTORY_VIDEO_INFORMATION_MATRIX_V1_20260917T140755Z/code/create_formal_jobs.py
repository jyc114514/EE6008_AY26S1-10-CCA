#!/usr/bin/env python3
"""Register measured Stage0 memory and create the four formal jobs."""

from __future__ import annotations
import os

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
INPUT = ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
TRAINER = ROOT / "code/run_history_video_cell.py"
LEASE_DIR = ROOT / "locks/gpu_lease"
SEED = 20260939


def main() -> int:
    reports: dict[str, dict[str, object]] = {}
    peaks: list[int] = []
    for cell in CELLS:
        report_path = ROOT / "runs" / cell / "stage0_retry1/STAGE0_REPORT.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("optimizer_steps") != 8:
            raise RuntimeError(f"Stage0 report is not accepted: {report_path}")
        peak = int(report.get("peak_memory_reserved_mib", 0))
        if peak <= 0:
            raise RuntimeError(f"Stage0 did not record a positive GPU peak: {report_path}")
        peaks.append(peak)
        reports[cell] = report
    measured_peak = max(peaks)
    jobs: list[dict[str, object]] = []
    for cell in CELLS:
        job_id = f"HVI-{cell}-FORMAL"
        run_root = ROOT / "runs" / cell / "formal"
        job = {
            "job_id": job_id,
            "protocol_id": "HISTORY_VIDEO_INFORMATION_MATRIX_V1_FORMAL",
            "output_root": str(run_root),
            "log_path": str(ROOT / "logs" / f"child_{cell}_formal.log"),
            "command": [str(PYTHON), str(TRAINER), "--cell", cell, "--run-root", str(run_root), "--input-root", str(INPUT), "--batch-size", "64", "--accumulation", "1", "--device", "cuda"],
            "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "resource_estimate": {"measured_peak_mib": measured_peak, "estimate_type": "stage0_measured_peak_reserved_mib"},
            "max_oom_requeues": 0,
            "final_test_access": 0,
        }
        path = ROOT / "configs/jobs" / f"{job_id}.json"
        if path.exists() or run_root.exists():
            raise FileExistsError(f"formal output already exists: {path} or {run_root}")
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs.append(job)
    resource = json.loads((ROOT / "RESOURCE_CONTRACT.json").read_text(encoding="utf-8"))
    resource.update({"stage0_accepted": "4/4", "stage0_attempt": "RETRY1", "formal_measured_peak_mib": measured_peak, "formal_safety_margin_mib": max(512, int(np_ceil(measured_peak * 0.10))), "formal_required_free_memory_mib": measured_peak + max(512, int(np_ceil(measured_peak * 0.10))), "formal_jobs_created": 4})
    (ROOT / "RESOURCE_CONTRACT.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "configs/RESOURCE_MEASUREMENTS.json").write_text(json.dumps({"status": "PASS", "stage0_reports": reports, "formal_measured_peak_mib": measured_peak, "formal_safety_margin_mib": max(512, int(np_ceil(measured_peak * 0.10))), "formal_required_free_memory_mib": measured_peak + max(512, int(np_ceil(measured_peak * 0.10))), "jobs": [job["job_id"] for job in jobs], "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "configs/FORMAL_JOBS.json").write_text(json.dumps({"status": "LOCKED", "jobs": jobs, "lease_dir": str(LEASE_DIR), "max_concurrent_project_gpu_children": 2, "formal_measured_peak_mib": measured_peak, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_jobs": len(jobs), "measured_peak_mib": measured_peak, "required_free_memory_mib": measured_peak + max(512, int(np_ceil(measured_peak * 0.10)))}, sort_keys=True))
    return 0


def np_ceil(value: float) -> int:
    return int(value) if int(value) == value else int(value) + 1


if __name__ == "__main__":
    raise SystemExit(main())
