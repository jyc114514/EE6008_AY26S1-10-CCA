#!/usr/bin/env python3
"""Register measured Stage0 peak and create six B formal jobs."""

from __future__ import annotations
import os

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = PROJECT / "envs/jyc/bin/python"
RUNNER = ROOT / "code/run_b_encoder_adaptation.py"
JOB_DIR = ROOT / "configs/jobs"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")


def main() -> int:
    reports = {}
    peaks = []
    for cell in CELLS:
        path = ROOT / f"runs/{cell}/stage0_retry1/STAGE0_REPORT.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("optimizer_steps") != 8:
            raise RuntimeError(f"Stage0 not accepted: {path}")
        report_path = str(path)
        reports[cell] = {"path": report_path, "peak_reserved_mib": report["peak_memory_reserved_mib"], "report": report}
        peaks.append(int(report["peak_memory_reserved_mib"]))
    measured = max(peaks)
    margin = max(512, math.ceil(measured * 0.10))
    required = measured + margin
    jobs = []
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        job_id = f"B-{cell}-FORMAL-S20260944"
        run_root = ROOT / f"runs/{cell}/formal"
        job = {
            "job_id": job_id,
            "protocol_id": "CORE_B_ENCODER_ADAPTATION_INDEPENDENT_B_FORMAL",
            "output_root": str(run_root),
            "log_path": str(ROOT / f"logs/child_{cell}_formal.log"),
            "command": [str(PYTHON), str(RUNNER), "--mode", "train", "--cell", cell, "--run-root", str(run_root)],
            "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_LIBRARY_PATH": str(PROJECT / "envs/jyc/lib") + ":/usr/local/cuda-12.8/lib64"},
            "resource_estimate": {"measured_peak_mib": measured, "estimate_type": "stage0_retry1_measured_peak_reserved_mib"},
            "max_oom_requeues": 1,
            "final_test_access": 0,
        }
        (JOB_DIR / f"{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs.append(job)
    payload = {"status": "LOCKED", "formal_seed": 20260944, "cells": list(CELLS), "run_count": 6, "epochs": 10, "optimizer_steps": 430, "measured_peak_by_cell": {cell: reports[cell]["peak_reserved_mib"] for cell in CELLS}, "measured_peak_mib": measured, "safety_margin_mib": margin, "required_free_memory_mib": required, "jobs": jobs, "final_test_access": 0}
    (ROOT / "configs/B_FORMAL_JOBS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "configs/B_RESOURCE_MEASUREMENTS.json").write_text(json.dumps({"status": "PASS", "stage0_attempt": "RETRY1", "reports": reports, "measured_peak_mib": measured, "safety_margin_mib": margin, "required_free_memory_mib": required, "formal_jobs": 6, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_jobs": 6, "measured_peak_mib": measured, "required_free_memory_mib": required}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
