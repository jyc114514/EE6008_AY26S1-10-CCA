"""Sequential D3-EQ queue runner; every CUDA child is launched by the shared watcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT_ROOT / "envs/jyc/bin/python"
WATCHER = REPO_ROOT / "tools/gpu_lease_watcher.py"
LEASE_DIR = REPO_ROOT / "reports/locks/gpu_lease"
PROTOCOL_ID = "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1"
GPU_HOUR_CAP = 48.0
GIB_CAP = 50.0
PREDICTION_RESERVE_GPU_HOURS = 4.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def append_ledger(record: dict[str, Any]) -> None:
    path = EQ_ROOT / "JOB_LEDGER.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def lease_hours(status_path: Path) -> float:
    if not status_path.is_file():
        return 0.0
    data = read_json(status_path)
    start = data.get("lease_acquired_utc")
    end = data.get("lease_released_utc")
    if not start or not end:
        return 0.0
    begin = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    finish = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    return max((finish - begin).total_seconds() / 3600.0, 0.0)


def run_one(item: dict[str, Any], *, max_wait_seconds: int) -> dict[str, Any]:
    job_path = Path(item["job_path"])
    status_path = Path(item["status_path"])
    log_path = Path(item["log_path"])
    job = read_json(job_path)
    command = [
        str(PYTHON), str(WATCHER),
        "--job-json", str(job_path),
        "--lease-dir", str(LEASE_DIR),
        "--status-path", str(status_path),
        "--poll-seconds", "15.0",
        "--max-wait-seconds", str(max_wait_seconds),
        "--max-oom-requeues", str(int(job.get("max_oom_requeues", 0))),
    ]
    started = utc_now()
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{started} command={json.dumps(command)}\n")
        handle.write(result.stdout)
        handle.write(result.stderr)
        handle.flush()
        os.fsync(handle.fileno())
    record = {
        "utc": utc_now(),
        "job_id": item["job_id"],
        "phase": item["phase"],
        "command": command,
        "exit_code": result.returncode,
        "watcher_status_path": str(status_path),
        "log_path": str(log_path),
        "lease_gpu_hours": lease_hours(status_path),
        "final_test_access": 0,
    }
    if status_path.is_file():
        state = read_json(status_path)
        record.update({
            "watcher_status": state.get("status"),
            "selected_gpu": state.get("selected_gpu"),
            "lease_released": state.get("lease_released"),
        })
    append_ledger(record)
    if result.returncode != 0:
        raise RuntimeError(f"watcher job failed: {item['job_id']} exit={result.returncode}; see {log_path}")
    return record


def report_for(item: dict[str, Any]) -> dict[str, Any] | None:
    root = Path(read_json(Path(item["job_path"]))["output_root"])
    marker = root / ("STAGE0_REPORT.json" if item["phase"] == "STAGE0" else "RUN_COMPLETE.json")
    if not marker.is_file():
        return None
    data = read_json(marker)
    return data if data.get("status") == "PASS" else None


def complete_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if report_for(item) is not None]


def load_jobs(phase: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest = EQ_ROOT / "configs" / f"D3_EQ_{phase.upper()}_JOBS.json"
    data = read_json(manifest)
    if data.get("protocol_id") != PROTOCOL_ID:
        raise AssertionError("job manifest protocol ID mismatch")
    return manifest, list(data["jobs"])


def accepted_stage0() -> dict[str, Any]:
    path = EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json"
    if not path.is_file():
        raise FileNotFoundError("12/12 Stage0 acceptance is required before formal training")
    data = read_json(path)
    if data.get("status") != "PASS" or data.get("accepted_cells") != 12:
        raise AssertionError("Stage0 acceptance is not 12/12 PASS")
    return data


def build_stage0_acceptance(items: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for item in items:
        result = report_for(item)
        if result is None:
            raise AssertionError(f"Stage0 report missing/not PASS: {item['job_id']}")
        if int(result.get("optimizer_steps", -1)) != 4:
            raise AssertionError(f"Stage0 must complete exactly 4 steps: {item['job_id']}")
        reports.append({
            "job_id": item["job_id"],
            "config": item["config"],
            "horizon_steps": item["horizon_steps"],
            "seed": item["seed"],
            "optimizer_steps": result["optimizer_steps"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
        })
    expected = {(config, horizon, 20261001) for config in ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL") for horizon in (30, 60)}
    got = {(row["config"], int(row["horizon_steps"]), int(row["seed"])) for row in reports}
    if got != expected or len(reports) != 12:
        raise AssertionError("Stage0 matrix coverage is not the locked first-seed 12 cells")
    max_reserved = max(int(row["peak_reserved_bytes"]) for row in reports)
    max_allocated = max(int(row["peak_allocated_bytes"]) for row in reports)
    measured_peak_mib = math.ceil(max_reserved / (1024 * 1024))
    watcher_gpu_hours = sum(lease_hours(Path(item["status_path"])) for item in items)
    result = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "accepted_cells": len(reports),
        "expected_cells": 12,
        "optimizer_steps_per_cell": 4,
        "max_peak_allocated_bytes": max_allocated,
        "max_peak_reserved_bytes": max_reserved,
        "formal_measured_peak_mib": measured_peak_mib,
        "formal_watcher_required_free_mib": measured_peak_mib + max(512, math.ceil(0.1 * measured_peak_mib)),
        "stage0_gpu_hours_actual": watcher_gpu_hours,
        "formal_seed": 20261001,
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "cells": reports,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json", result)
    pre = read_json(EQ_ROOT / "protocol/D3_EQ_RESOURCE_ESTIMATE_PRE_STAGE0.json")
    conservative_formal = float(pre["new_formal_36_conservative_gpu_hours"])
    reserve = watcher_gpu_hours + conservative_formal + PREDICTION_RESERVE_GPU_HOURS
    post = {
        "status": "PASS_RESOURCE_GATE" if reserve <= GPU_HOUR_CAP else "budget_insufficient",
        "measured_stage0_peak_mib": measured_peak_mib,
        "stage0_gpu_hours_actual": watcher_gpu_hours,
        "formal_36_conservative_gpu_hours": conservative_formal,
        "primary_and_dense_prediction_reserved_gpu_hours": PREDICTION_RESERVE_GPU_HOURS,
        "projected_reserved_total_gpu_hours": reserve,
        "gpu_hour_ceiling": GPU_HOUR_CAP,
        "gpu_hours_headroom": GPU_HOUR_CAP - reserve,
        "new_output_size_ceiling_gib": GIB_CAP,
        "minimum_filesystem_free_fraction": 0.10,
        "filesystem_gate": "must refresh live before formal; prior read-only snapshot is not a reservation",
        "old_output_baseline_bytes": pre["old_d3_run_checkpoint_bytes"] + pre["old_d3_sparse_prediction_bytes"],
        "train_target_cache_bytes": pre["new_train_target_cache_bytes"],
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "protocol/D3_EQ_RESOURCE_ESTIMATE_POST_STAGE0.json", post)
    if post["status"] != "PASS_RESOURCE_GATE":
        raise RuntimeError("resource budget gate failed; formal queue must not start")
    # Create formal jobs with the measured reserved-memory peak, only after
    # 12/12 Stage0 acceptance. This does not query a GPU.
    builder = EQ_ROOT / "code/build_d3_eq_jobs.py"
    command = [str(PYTHON), str(builder), "--phase", "formal", "--peak-mib", str(measured_peak_mib)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"formal queue generation failed: {result.stdout}{result.stderr}")
    (EQ_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (EQ_ROOT / "logs/D3_EQ_FORMAL_QUEUE_BUILD.json").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def run_phase(items: list[dict[str, Any]], *, max_wait_seconds: int) -> None:
    for item in items:
        if report_for(item) is not None:
            continue
        run_one(item, max_wait_seconds=max_wait_seconds)
        if report_for(item) is None:
            raise AssertionError(f"watcher exited successfully but artifact acceptance missing: {item['job_id']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage0", "seed1_formal", "remaining_formal", "all"), required=True)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        raise RuntimeError("GPU queue must run inside tmux for safe monitoring/recovery")
    if args.max_wait_seconds < 1:
        raise ValueError("max-wait-seconds must be positive")
    started = utc_now()
    atomic_json(EQ_ROOT / "logs/D3_EQ_QUEUE_STATUS.json", {"status": "running", "phase": args.phase, "started_utc": started, "final_test_access": 0})
    try:
        stage0_manifest, stage0_items = load_jobs("stage0")
        if args.phase in ("stage0", "all"):
            run_phase(stage0_items, max_wait_seconds=args.max_wait_seconds)
            if len(complete_items(stage0_items)) == 12:
                accepted_stage0()
            else:
                build_stage0_acceptance(stage0_items)
                accepted_stage0()
        if args.phase == "stage0":
            accepted_stage0()
            result = {"status": "stage0_complete", "stage0_cells": 12}
        else:
            stage0 = accepted_stage0()
            formal_manifest, formal_items = load_jobs("formal")
            if args.phase in ("seed1_formal", "all"):
                first_seed = [item for item in formal_items if int(item["seed"]) == 20261001]
                run_phase(first_seed, max_wait_seconds=args.max_wait_seconds)
            if args.phase in ("remaining_formal", "all"):
                first_seed = [item for item in formal_items if int(item["seed"]) == 20261001]
                if len(complete_items(first_seed)) != 12:
                    raise AssertionError("all first-seed formal runs must be accepted before the remaining 24")
                remainder = [item for item in formal_items if int(item["seed"]) != 20261001]
                run_phase(remainder, max_wait_seconds=args.max_wait_seconds)
            result = {
                "status": "formal_phase_complete_or_partial",
                "stage0_cells": stage0["accepted_cells"],
                "formal_complete_cells": len(complete_items(formal_items)),
            }
        result.update({"phase": args.phase, "completed_utc": utc_now(), "final_test_access": 0})
        atomic_json(EQ_ROOT / "logs/D3_EQ_QUEUE_STATUS.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed_or_interrupted",
            "phase": args.phase,
            "failed_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "final_test_access": 0,
        }
        atomic_json(EQ_ROOT / "logs/D3_EQ_QUEUE_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
