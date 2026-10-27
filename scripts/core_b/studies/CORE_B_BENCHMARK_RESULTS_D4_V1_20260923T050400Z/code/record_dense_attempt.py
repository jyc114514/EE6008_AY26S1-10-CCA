"""Capture sanitized status and leased GPU-hours for a completed pipeline attempt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "d3_extension/status"
ATTEMPT_OUT = STATUS / "RESUMED_PIPELINE_ATTEMPT_001.json"


def main():
    mapping = (
        ("D3_DENSE_STAGE0", "D3_DENSE_STAGE0_WATCHER.json"),
        ("D3_DENSE_EXTRACT", "D3_DENSE_EXTRACT_WATCHER.json"),
        ("D3_DENSE_INFER", "D3_DENSE_INFER_WATCHER.json"),
        ("V3_STATE_DENSE_INFER", "V3_STATE_DENSE_INFER_WATCHER.json"),
    )
    stages = []
    total_seconds = 0.0
    for name, filename in mapping:
        raw = json.loads((STATUS / filename).read_text(encoding="utf-8"))
        start, end = raw.get("lease_acquired_utc"), raw.get("lease_released_utc")
        seconds = 0.0
        if start and end:
            seconds = (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
        total_seconds += seconds
        stages.append({
            "stage": name, "status": raw.get("status"), "exit_code": raw.get("exit_code"),
            "physical_gpu_index": raw.get("physical_gpu"), "logical_gpu": "cuda:0",
            "watcher_pid": raw.get("watcher_pid"), "child_pid": raw.get("child_pid"),
            "lease_acquired_utc": start, "lease_released_utc": end,
            "lease_seconds": round(seconds, 3),
            "lease_released": bool(raw.get("lease_released")),
            "project_child_slot_released": bool(raw.get("project_child_slot_released")),
            "non_child_process_signaled": bool(raw.get("non_child_process_signaled", False)),
            "oom_requeue_count": int(raw.get("oom_requeue_count", 0)),
        })
    record = {
        "attempt": "RESUMED_PIPELINE_ATTEMPT_001",
        "status": "D3_feature_cache_and_36_dense_predictions_pass; V3_STATE_dense_inference_failed_before_bundle_write",
        "tmux_session": "coreb_d3_siv_dense_resume_20260923T1528Z",
        "main_tmux_pane_pid": 1170249,
        "stages": stages,
        "gpu_hours_leased": round(total_seconds / 3600.0, 6),
        "failure_record": "d3_extension/invalid_runs/D3_SIV_MISSING_FEATURES_ATTEMPT_001/FAILURE_RECORD.json",
        "final_test_access": 0,
    }
    ATTEMPT_OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"attempt": record["attempt"], "gpu_hours_leased": record["gpu_hours_leased"], "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
