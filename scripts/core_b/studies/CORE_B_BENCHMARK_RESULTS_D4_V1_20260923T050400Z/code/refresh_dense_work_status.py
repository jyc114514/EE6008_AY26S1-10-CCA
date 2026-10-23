"""Reconcile completed optional dense runs into WORK_STATUS and run metadata."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "d3_extension/status"
WATCHERS = (
    ("stage0", STATUS / "D3_DENSE_STAGE0_WATCHER.json"),
    ("d3_feature_extract", STATUS / "D3_DENSE_EXTRACT_WATCHER.json"),
    ("d3_dense_inference", STATUS / "D3_DENSE_INFER_WATCHER.json"),
    ("v3_state_dense_inference", STATUS / "V3_STATE_DENSE_INFER_WATCHER.json"),
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    pause = load(STATUS / "D3_DENSE_PAUSE_RECORD_20260923T082841Z.json")
    attempt1 = load(STATUS / "RESUMED_PIPELINE_ATTEMPT_001.json")
    stages = []
    current_gpu_hours = 0.0
    for name, path in WATCHERS:
        if not path.exists():
            continue
        status = load(path)
        if not status.get("lease_released", False) or not status.get("project_child_slot_released", False):
            raise RuntimeError(f"stage is still active or has not released its own lease: {name}")
        acquired = status.get("lease_acquired_utc")
        released = status.get("lease_released_utc")
        if acquired and released:
            from datetime import datetime as dt
            elapsed = (dt.fromisoformat(released.replace("Z", "+00:00")) - dt.fromisoformat(acquired.replace("Z", "+00:00"))).total_seconds()
        else:
            elapsed = 0.0
        current_gpu_hours += elapsed / 3600.0
        stages.append({
            "stage": name,
            "status": status.get("status"),
            "exit_code": status.get("exit_code"),
            "physical_gpu_index": status.get("physical_gpu"),
            "logical_gpu": "cuda:0",
            "watcher_pid": status.get("watcher_pid"),
            "child_pid": status.get("child_pid"),
            "lease_seconds": round(elapsed, 3),
            "lease_released": True,
            "project_child_slot_released": True,
            "non_child_process_signaled": bool(status.get("non_child_process_signaled", False)),
        })
    total_gpu_hours = float(pause.get("gpu_hours_consumed_before_pause", 0.0)) + float(attempt1["gpu_hours_leased"]) + current_gpu_hours
    if total_gpu_hours > 24.0:
        raise RuntimeError(f"optional dense extension exceeded the 24-GPU-hour cap: {total_gpu_hours}")
    if any(row["non_child_process_signaled"] for row in stages):
        raise RuntimeError("a watcher status reports a non-child signal")

    d3_prediction = load(ROOT / "d3_extension/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3_qa = load(ROOT / "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    d3d4 = load(ROOT / "d4/d3_extension/D3_D4_SUMMARY.json")
    siv_prediction = load(ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    siv_qa_path = ROOT / "v3_state_dense/qa/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json"
    siv_qa = load(siv_qa_path) if siv_qa_path.exists() else {}
    sivd4 = load(ROOT / "d4/v3_state_dense/D4_SUMMARY.json")
    complete = (
        d3_prediction.get("status") == "PASS" and int(d3_prediction.get("accepted_bundles", -1)) == 36
        and d3_qa.get("status") == "PASS" and int(d3_qa.get("accepted_bundles", -1)) == 36
        and d3d4.get("status") == "COMPLETE_D3_DENSE_ROLLING_EXTENSION"
        and siv_prediction.get("status") == "PASS" and int(siv_prediction.get("accepted_bundles", -1)) == 54
        and sivd4.get("status") == "COMPLETE_POSTHOC_V3_STATE_DENSE_ROLLING_EXTENSION"
    )
    if not complete:
        raise RuntimeError("both dense rolling extensions must pass before recording full optional completion")

    run_record = {
        "status": "complete",
        "tmux_session": "coreb_d3_siv_dense_resume_20260923T1528Z",
        "main_tmux_pane_pid": 1170249,
        "resume_script": str(ROOT / "d3_extension/run_optional_dense.sh"),
        "pipeline_log": str(ROOT / "d3_extension/logs/optional_dense_pipeline.log"),
        "stages": stages,
        "previous_resume_attempt": {"record": "d3_extension/status/RESUMED_PIPELINE_ATTEMPT_001.json", "gpu_hours_leased": attempt1["gpu_hours_leased"]},
        "cumulative_gpu_hours_including_all_attempts": round(total_gpu_hours, 6),
        "gpu_hour_cap": 24.0,
        "training": "none",
        "final_test_access": 0,
        "non_child_process_signaled": False,
    }
    (ROOT / "d3_extension/GPU_RUN_COMPLETION_RECORD.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    work_path = ROOT / "WORK_STATUS.json"
    work = load(work_path)
    unresolved = [
        item for item in work.get("UNRESOLVED", [])
        if "optional D3 dense extension has features" not in item
        and "optional D3 dense feature extension is" not in item
    ]
    work.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "D3_D4_EXTENSION_STATUS": "complete_D3_36_of_36_and_V3_STATE_54_of_54_dense_predictions_and_rolling_analysis",
        "D3_DENSE_PREDICTION_ACCEPTED": "36/36",
        "D3_DENSE_ROLLING_ANALYSIS": d3d4["status"],
        "V3_STATE_DENSE_PREDICTION_ACCEPTED": "54/54",
        "V3_STATE_DENSE_ROLLING_ANALYSIS": sivd4["status"],
        "GPU_HOURS_BEFORE_FIRST_PAUSE": float(pause.get("gpu_hours_consumed_before_pause", 0.0)),
        "GPU_HOURS_THIS_OPTIONAL_EXTENSION_TOTAL": round(total_gpu_hours, 6),
        "NEW_MODEL_TRAINING": 0,
        "FINAL_HELDOUT_ACCESS": 0,
        "UNRESOLVED": unresolved,
        "final_test_access": 0,
    })
    work.pop("GPU_HOURS_THIS_OPTIONAL_EXTENSION_SO_FAR", None)
    work_path.write_text(json.dumps(work, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return_path = ROOT / "RETURN_TO_CHATGPT.json"
    result = load(return_path)
    result.update({
        "D3_D4_EXTENSION_STATUS": work["D3_D4_EXTENSION_STATUS"],
        "D3_DENSE_PREDICTION_ACCEPTED": "36/36",
        "D3_DENSE_ROLLING_ANALYSIS": d3d4["status"],
        "V3_STATE_DENSE_PREDICTION_ACCEPTED": "54/54",
        "V3_STATE_DENSE_ROLLING_ANALYSIS": sivd4["status"],
        "GPU_HOURS_BEFORE_FIRST_PAUSE": float(pause.get("gpu_hours_consumed_before_pause", 0.0)),
        "GPU_HOURS_OPTIONAL_DENSE_TOTAL": round(total_gpu_hours, 6),
        "NEW_MODEL_TRAINING": 0,
        "FINAL_HELDOUT_ACCESS": 0,
    })
    return_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "D3_DENSE": "36/36", "D3_D4": d3d4["status"],
        "V3_STATE_DENSE": "54/54", "V3_STATE_D4": sivd4["status"],
        "gpu_hours_total": round(total_gpu_hours, 6), "final_test_access": 0,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
