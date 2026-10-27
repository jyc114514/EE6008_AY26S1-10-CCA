#!/usr/bin/env python3
"""Accept the 18 pilot Stage0 jobs and update the formal resource contract."""

from __future__ import annotations
import os

import json
import csv
import statistics
from datetime import datetime
from pathlib import Path


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z'))


def hours(start: str, end: str) -> float:
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (b - a).total_seconds() / 3600.0


def main() -> int:
    rows = []
    for status_path in sorted((EXP / "logs").glob("watcher_SIV-*-STAGE0.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        job_id = status["job_id"]
        report_path = EXP / f"runs/{job_id.removesuffix('-STAGE0')}/stage0/STAGE0_REPORT.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ok = status.get("status") == "complete" and status.get("exit_code") == 0 and report.get("status") == "PASS" and report.get("optimizer_steps") == 4 and report.get("checkpoint_reload_prediction_max_abs") == 0.0 and report.get("initial_persistence_max_abs") == 0.0 and status.get("non_child_process_signaled") is False and status.get("final_test_access", 0) == 0 and report.get("final_test_access", 0) == 0
        rows.append({
            "job_id": job_id,
            "cell": report.get("cell"),
            "horizon_steps": report.get("horizon"),
            "seed": report.get("seed"),
            "physical_gpu": status.get("physical_gpu"),
            "peak_memory_allocated_mib": report.get("peak_memory_allocated_mib"),
            "peak_memory_reserved_mib": report.get("peak_memory_reserved_mib"),
            "child_wall_hours": hours(status["launch_utc"], status["completed_utc"]),
            "status": "PASS" if ok else "FAIL",
            "final_test_access": 0,
        })
    if len(rows) != 18 or not all(row["status"] == "PASS" for row in rows):
        raise RuntimeError(f"Stage0 acceptance failed: {len(rows)} rows")
    max_peak = max(int(row["peak_memory_reserved_mib"]) for row in rows)
    # 1024 MiB is a conservative measured-peak envelope for the 204 MiB maximum
    # Stage0 reservation; the watcher adds its normal 512 MiB safety margin.
    formal_peak_contract = 1024
    formal_jobs_path = EXP / "configs/SIV_FORMAL_JOBS.json"
    formal_jobs = json.loads(formal_jobs_path.read_text(encoding="utf-8"))
    for job in formal_jobs["jobs"]:
        job["resource_estimate"] = {"estimate_type": "stage0_measured_peak_plus_normal_variation", "stage0_max_reserved_mib": max_peak, "measured_peak_mib": formal_peak_contract}
    formal_jobs["status"] = "LOCKED_READY_AFTER_STAGE0"
    formal_jobs["stage0_acceptance"] = "18/18"
    formal_jobs_path.write_text(json.dumps(formal_jobs, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage0_child_hours = sum(float(row["child_wall_hours"]) for row in rows)
    # A four-step pilot includes process/data startup, so scaling its whole wall
    # time by 2,780 would be invalid.  Calibrate startup-adjusted throughput with
    # the already completed, same-size V3 formal/stage0 jobs, then apply that
    # ratio to this new runner's measured Stage0 child time.
    v3_resource_path = EXP.parents[2] / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/analysis/RESOURCE_COSTS.csv"
    with v3_resource_path.open(encoding="utf-8", newline="") as handle:
        v3_rows = list(csv.DictReader(handle))
    v3_formal_hours = [float(row["wall_gpu_hours"]) for row in v3_rows if row["kind"] == "formal"]
    v3_stage0_hours = [float(row["wall_gpu_hours"]) for row in v3_rows if row["kind"] == "stage0"]
    v3_formal_per_run = sum(v3_formal_hours) / len(v3_formal_hours)
    v3_stage0_reference = statistics.median(v3_stage0_hours)
    startup_adjusted_ratio = v3_formal_per_run / v3_stage0_reference
    formal_gpu_hours = stage0_child_hours * startup_adjusted_ratio * (54 / 18)
    estimate = {
        "status": "PASS",
        "stage0": "18/18",
        "stage0_max_reserved_mib": max_peak,
        "formal_resource_contract_mib": formal_peak_contract,
        "watcher_required_free_memory_mib": formal_peak_contract + 512,
        "stage0_child_gpu_hours": stage0_child_hours,
        "v3_formal_per_run_gpu_hours_reference": v3_formal_per_run,
        "v3_stage0_median_child_gpu_hours_reference": v3_stage0_reference,
        "startup_adjusted_formal_to_stage0_ratio": startup_adjusted_ratio,
        "formal_gpu_hours_estimate_from_stage0_child_throughput": formal_gpu_hours,
        "formal_run_count": 54,
        "inference_and_analysis_reserve_gpu_hours": 1.0,
        "estimated_total_gpu_hours": formal_gpu_hours + 1.0,
        "budget_upper_bound_gpu_hours": 16,
        "within_budget": formal_gpu_hours + 1.0 <= 16,
        "actual_stage0_gpus": sorted({int(row["physical_gpu"]) for row in rows}),
        "dynamic_all_visible": True,
        "co_location_allowed": True,
        "max_project_gpu_children": 2,
        "final_test_access": 0,
    }
    with (EXP / "analysis/STAGE0_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP / "analysis/STAGE0_ACCEPTANCE.json").write_text(json.dumps({"status": "PASS", "accepted": "18/18", "rows": rows, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXP / "RESOURCE_ESTIMATE.json").write_text(json.dumps(estimate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(estimate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
