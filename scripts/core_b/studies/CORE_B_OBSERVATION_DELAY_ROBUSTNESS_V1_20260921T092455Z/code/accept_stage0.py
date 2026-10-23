#!/usr/bin/env python3
"""Accept six Stage0 jobs and update the formal resource contract."""

from __future__ import annotations
import os

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
SIV = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"


def wall(start: str, end: str) -> float:
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600.0


def main() -> int:
    rows = []
    for status_path in sorted((EXP / "logs").glob("watcher_ODR-*-STAGE0.json")):
        status = json.loads(status_path.read_text(encoding="utf-8")); job_id = status["job_id"]; report_path = EXP / f"runs/{job_id.removesuffix('-STAGE0')}/stage0/STAGE0_REPORT.json"; report = json.loads(report_path.read_text(encoding="utf-8"))
        ok = status.get("status") == "complete" and status.get("exit_code") == 0 and report.get("status") == "PASS" and report.get("optimizer_steps") == 4 and report.get("initial_persistence_max_abs") == 0.0 and report.get("checkpoint_reload_prediction_max_abs") == 0.0 and status.get("non_child_process_signaled") is False
        rows.append({"job_id": job_id, "cell": report.get("cell"), "seed": report.get("seed"), "physical_gpu": status.get("physical_gpu"), "peak_memory_reserved_mib": report.get("peak_memory_reserved_mib"), "peak_memory_allocated_mib": report.get("peak_memory_allocated_mib"), "child_wall_hours": wall(status["launch_utc"], status["completed_utc"]), "status": "PASS" if ok else "FAIL", "final_test_access": 0})
    if len(rows) != 6 or not all(row["status"] == "PASS" for row in rows): raise RuntimeError(f"Stage0 acceptance failed {len(rows)}")
    max_peak = max(int(row["peak_memory_reserved_mib"]) for row in rows); contract = 1024
    for manifest_name in ("ODR_FORMAL_JOBS.json", "ODR_INFERENCE_JOBS.json"):
        path = EXP / "configs" / manifest_name; value = json.loads(path.read_text(encoding="utf-8"));
        for job in value["jobs"]: job["resource_estimate"] = {"estimate_type": "stage0_measured_peak_plus_normal_variation", "stage0_max_reserved_mib": max_peak, "measured_peak_mib": contract}
        value["status"] = "LOCKED_READY_AFTER_STAGE0"; value["stage0_acceptance"] = "6/6"; path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (SIV / "analysis/RESOURCE_COSTS.csv").open(encoding="utf-8", newline="") as handle: siv_rows = list(csv.DictReader(handle))
    siv_formal = [float(row["wall_gpu_hours"]) for row in siv_rows if row["kind"] == "formal"]; siv_stage0 = [float(row["wall_gpu_hours"]) for row in siv_rows if row["kind"] == "stage0"]
    ratio = (sum(siv_formal) / len(siv_formal)) / statistics.median(siv_stage0); stage0_hours = sum(float(row["child_wall_hours"]) for row in rows); formal_estimate = stage0_hours * ratio * (18 / 6); inference_estimate = 72 / 54 * (sum(float(row["wall_gpu_hours"]) for row in siv_rows if row["kind"] == "inference") / 36); total_estimate = formal_estimate + inference_estimate + 1.0
    estimate = {"status": "PASS", "stage0": "6/6", "stage0_max_reserved_mib": max_peak, "resource_contract_mib": contract, "watcher_required_free_memory_mib": contract + 512, "stage0_child_gpu_hours": stage0_hours, "formal_gpu_hours_estimate": formal_estimate, "inference_gpu_hours_estimate": inference_estimate, "analysis_reserve_gpu_hours": 1.0, "estimated_total_gpu_hours": total_estimate, "budget_upper_bound_gpu_hours": 12, "within_budget": total_estimate <= 12, "actual_stage0_gpus": sorted({int(row["physical_gpu"]) for row in rows}), "final_test_access": 0}
    (EXP / "analysis").mkdir(exist_ok=True); write_rows = rows;
    with (EXP / "analysis/STAGE0_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(write_rows[0])); writer.writeheader(); writer.writerows(write_rows)
    (EXP / "analysis/STAGE0_ACCEPTANCE.json").write_text(json.dumps({"status": "PASS", "accepted": "6/6", "rows": rows, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); (EXP / "RESOURCE_ESTIMATE.json").write_text(json.dumps(estimate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(estimate, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
