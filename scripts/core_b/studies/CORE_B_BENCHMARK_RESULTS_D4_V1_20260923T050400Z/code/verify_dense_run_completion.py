"""Assert the resumable inference-only GPU pipeline completed and released leases."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    stages = []
    for name in ("D3_DENSE_STAGE0", "D3_DENSE_EXTRACT", "D3_DENSE_INFER", "V3_STATE_DENSE_INFER"):
        path = ROOT / f"d3_extension/status/{name}_WATCHER.json"
        record = read(path)
        if record.get("status") != "complete" or record.get("exit_code") != 0:
            raise SystemExit(f"GPU watcher stage did not exit complete/0: {name}")
        if not record.get("lease_released") or not record.get("project_child_slot_released"):
            raise SystemExit(f"GPU watcher stage did not release its lease/slot: {name}")
        if record.get("non_child_process_signaled") is not False:
            raise SystemExit(f"watcher reports an external-process signal: {name}")
        stages.append({
            "stage": name, "status": record["status"], "exit_code": record["exit_code"],
            "physical_gpu_index": record.get("physical_gpu"),
            "watcher_pid": record.get("watcher_pid"), "child_pid": record.get("child_pid"),
            "lease_acquired_utc": record.get("lease_acquired_utc"),
            "lease_released_utc": record.get("lease_released_utc"),
            "lease_released": True, "project_child_slot_released": True,
            "non_child_process_signaled": False,
        })
    cache = read(ROOT / "d3_extension/cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json")
    d3 = read(ROOT / "d3_extension/qa/D3_DENSE_PREDICTION_ACCEPTANCE.json")
    state = read(ROOT / "v3_state_dense/V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json")
    if cache.get("status") != "PASS" or int(cache.get("rows_processed", -1)) != 3718:
        raise SystemExit("D3 dense feature cache is not complete 3718/3718")
    if d3.get("status") != "PASS" or int(d3.get("accepted_bundles", -1)) != 36:
        raise SystemExit("D3 dense prediction acceptance is not 36/36")
    if state.get("status") != "PASS" or int(state.get("accepted_bundles", -1)) != 54:
        raise SystemExit("V3_STATE dense prediction acceptance is not 54/54")
    part_files = []
    for rel in ("d3_extension/cache", "d3_extension/predictions", "v3_state_dense/predictions"):
        part_files.extend(str(path.relative_to(ROOT)) for path in (ROOT / rel).rglob("*.part"))
    if part_files:
        raise SystemExit(f"partial temporary files remain: {part_files[:8]}")
    pause = read(ROOT / "d3_extension/status/D3_DENSE_PAUSE_RECORD_20260923T082841Z.json")
    attempt1 = read(ROOT / "d3_extension/status/RESUMED_PIPELINE_ATTEMPT_001.json")
    total_hours = float(pause.get("gpu_hours_consumed_before_pause", 0.0)) + float(attempt1["gpu_hours_leased"])
    for stage in stages:
        start, end = stage.get("lease_acquired_utc"), stage.get("lease_released_utc")
        if start and end:
            from datetime import datetime as dt
            total_hours += (dt.fromisoformat(end.replace("Z", "+00:00")) - dt.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() / 3600.0
    if total_hours > 24.0:
        raise SystemExit(f"optional inference-only extension exceeded 24 GPU-hours: {total_hours}")
    summary = {
        "status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stages": stages, "d3_cache_rows": 3718, "d3_dense_bundles": 36,
        "v3_state_dense_bundles": 54, "gpu_hours_total_including_pause_and_first_resume_attempt": round(total_hours, 6),
        "no_partial_files": True, "training": "none", "final_test_access": 0,
    }
    path = ROOT / "d3_extension/qa/DENSE_RUN_COMPLETION_QA.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
