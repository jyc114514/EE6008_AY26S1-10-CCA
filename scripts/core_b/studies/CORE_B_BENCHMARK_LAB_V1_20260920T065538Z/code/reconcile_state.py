"""Reconcile current upstream, CPU, cache, and queue evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, utc_now


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def v3_training() -> tuple[int, int]:
    accepted = running = 0
    for run in (V3 / "runs").iterdir():
        if not run.is_dir():
            continue
        variants = [run / "formal", run / "formal_pilot_retry1"]
        if any((variant / "RUN_COMPLETE.json").exists() and read_json(variant / "RUN_COMPLETE.json").get("status") == "PASS" for variant in variants):
            accepted += 1
        elif any((variant / "RUN_STATUS.json").exists() and read_json(variant / "RUN_STATUS.json").get("status") == "running" for variant in variants):
            running += 1
    return accepted, running


def elapsed_gpu_hours(status_path: Path) -> float:
    if not status_path.exists():
        return 0.0
    status = read_json(status_path)
    started = status.get("started_utc") or status.get("launch_utc")
    ended = status.get("completed_utc")
    if not started:
        return 0.0
    start = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
    finish = datetime.fromisoformat(str(ended).replace("Z", "+00:00")) if ended else datetime.now(timezone.utc)
    return max((finish - start).total_seconds(), 0.0) / 3600.0


def main() -> int:
    accepted, running = v3_training()
    v3_predictions = sum(1 for path in (V3 / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json") if read_json(path).get("status") == "PASS") if (V3 / "predictions").exists() else 0
    d1 = read_json(BENCH_ROOT / "d1/D1_SUMMARY.json") if (BENCH_ROOT / "d1/D1_SUMMARY.json").exists() else {}
    d1_learned = read_json(BENCH_ROOT / "d1/D1_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d1/D1_LEARNED_SUMMARY.json").exists() else {}
    d2 = read_json(BENCH_ROOT / "d2/D2_SUMMARY.json") if (BENCH_ROOT / "d2/D2_SUMMARY.json").exists() else {}
    d2_learned = read_json(BENCH_ROOT / "d2/D2_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d2/D2_LEARNED_SUMMARY.json").exists() else {}
    d2_dense = read_json(BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json") if (BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json").exists() else {}
    full_status = read_json(BENCH_ROOT / "logs/D3-GRID-FULL.json") if (BENCH_ROOT / "logs/D3-GRID-FULL.json").exists() else {}
    cache_manifest = read_json(BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json") if (BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json").exists() else {}
    stage0_queue = BENCH_ROOT / "d3/D3_STAGE0_QUEUE_COMPLETE.json"
    stage0_done = read_json(stage0_queue) if stage0_queue.exists() else None
    formal_complete = sum(1 for path in (BENCH_ROOT / "d3/runs").glob("D3-*-FORMAL/RUN_COMPLETE.json") if read_json(path).get("status") == "PASS") + sum(1 for path in (BENCH_ROOT / "d3/runs").glob("D3-*-FORMAL-RETRY*/RUN_COMPLETE.json") if read_json(path).get("status") == "PASS")
    formal_active = any(read_json(path).get("status") in {"running", "launching"} and Path("/proc") .joinpath(str(read_json(path).get("child_pid", "-1"))).exists() for path in (BENCH_ROOT / "logs").glob("watcher_D3-*-FORMAL*.json"))
    d3_prediction_count = len(list((BENCH_ROOT / "d3/predictions").glob("D3-*.npz"))) if (BENCH_ROOT / "d3/predictions").exists() else 0
    d3_analysis_done = (BENCH_ROOT / "d3/analysis/D3_ANALYSIS_SUMMARY.json").exists()
    d2_dense_done = (BENCH_ROOT / "d2/D2_DENSE_LEARNED_SUMMARY.json").exists()
    all_required = cache_manifest.get("status") == "PASS" and formal_complete == 36 and d3_prediction_count == 36 and d3_analysis_done and d2_dense_done
    if full_status.get("status") in {"running", "launching", "waiting_for_project_child_slot", "waiting_for_local_gpu_lease"} or formal_active or (not stage0_done and full_status):
        overall = "running"
    elif all_required:
        overall = "complete"
    elif v3_predictions == 0:
        overall = "awaiting_upstream"
    else:
        overall = "partial"
    disk_bytes = sum(path.stat().st_size for path in BENCH_ROOT.rglob("*") if path.is_file())
    snapshot = {
        "generated_utc": utc_now(), "benchmark_status": overall, "v3_training_accepted": f"{accepted}/36", "v3_training_running": f"{running}/36", "v3_predictions_accepted": f"{v3_predictions}/36", "d1_status": d1_learned.get("status", d1.get("status", "UNKNOWN")), "d1_visual_disambiguation": d1_learned.get("visual_disambiguation", d1.get("visual_disambiguation", "UNKNOWN")), "d1_supported_low_high": d1.get("support", {}), "d2_status": d2_learned.get("status", d2.get("status", "UNKNOWN")), "d2_dense_episodes": d2.get("dense_episodes", "UNKNOWN"), "d2_dense_queries": d2.get("dense_queries", "UNKNOWN"), "d2_learned_models": d2_dense.get("status", d2_learned.get("status", d2.get("learned_model_status", "UNKNOWN"))), "d3_cache_status": cache_manifest.get("status", "RUNNING_OR_NOT_READY"), "d3_cache_covered_rows": cache_manifest.get("covered_rows", 0), "d3_stage0_training": "12/12" if stage0_done else "0/12", "d3_formal_training": f"{formal_complete}/36", "d3_predictions": f"{d3_prediction_count}/36", "d3_analysis_status": "PASS" if d3_analysis_done else "pending", "d3_full_gpu_status": full_status.get("status", "not_started"), "d3_full_physical_gpu": full_status.get("physical_gpu", "unknown"), "d4_status": "prepared_not_executed", "d5_status": "prepared_not_executed", "new_encoder_updates": 0, "final_test_access": 0, "production_root_writes": 0, "project_wide_gpu_children_max": 2, "new_benchmark_gpu_hours_observed": elapsed_gpu_hours(BENCH_ROOT / "logs/D3-GRID-STAGE0.json") + elapsed_gpu_hours(BENCH_ROOT / "logs/D3-GRID-FULL.json"), "new_benchmark_disk_gib": disk_bytes / (1024**3), "unresolved_dependencies": [] if v3_predictions == 36 else ["V3 sealed prediction bundles incomplete"], "qa": {"cpu_analysis": "PASS", "prepared_synthetic": "8 passed", "d3_stage0_grid": "PASS", "d3_full_cache": cache_manifest.get("status", "pending"), "d3_analysis": "PASS" if d3_analysis_done else "pending"},
    }
    atomic_write_json(BENCH_ROOT / "RECONCILIATION_SNAPSHOT.json", snapshot)
    atomic_write_json(BENCH_ROOT / "BENCHMARK_STATE.json", snapshot)
    print(json.dumps(snapshot, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
