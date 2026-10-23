#!/usr/bin/env python3
"""Persistent, resumable Stage0 -> matched first seed -> remaining matrix queue."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
LEASE_DIR = PROJECT / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"
PROTOCOL_PATH = TASK / "PROTOCOL_LOCKED.json"
LAUNCH_GATE_PATH = TASK / "LAUNCH_GATE.json"
MATRIX_PATH = TASK / "protocol/MATRIX_12.csv"
STAGE0_MATRIX_PATH = TASK / "protocol/STAGE0_MATRIX_4.csv"
RESOURCE_PATH = TASK / "RESOURCE_ESTIMATE_PRE_STAGE0.json"
MAX_CHILDREN = 1
MAX_PROJECT_CHILDREN = 2
GPU_HOUR_CAP = 12.0
GPU_MEMORY_CAP_MIB = 20 * 1024
MAX_RETRIES = 1
MAX_CHILD_ATTEMPTS_PER_CELL = 3
WATCHER_MAX_WAIT_SECONDS = 1800
RESOURCE_WAIT_BACKOFF_SECONDS = 60
RESOURCE_WAIT_STATUSES = frozenset({
    "capacity_timeout",
    "project_child_slot_timeout",
    "local_gpu_lease_or_recheck_timeout",
})
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def effective_queue_concurrency(candidates: list[int] | tuple[int, ...]) -> int:
    active = {int(index) for index in candidates}
    if active != {2}:
        raise ValueError("this resumed run permits only physical GPU2")
    return min(MAX_PROJECT_CHILDREN, len(active))


def job_attempt_number(job_id: str) -> int:
    try:
        return int(str(job_id).rsplit("-ATTEMPT", 1)[1])
    except (IndexError, ValueError):
        return 0


def can_retry_failed_job(job_id: str) -> bool:
    return job_attempt_number(job_id) == 1


def child_attempt_count(run_id: str) -> int:
    ledger = TASK / "TASK_JOB_LEDGER.jsonl"
    if not ledger.exists():
        return 0
    count = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run_id") == run_id and row.get("attempt_kind") == "gpu_child_attempt":
            count += 1
    return count


def is_resource_wait_attempt(return_code: int, status: dict[str, Any]) -> bool:
    watcher_status = status.get("status")
    marked_high_util_wait = (
        watcher_status == "telemetry_error"
        and "D3_RESOURCE_WAIT_HIGH_UTIL" in str(status.get("error", ""))
    )
    return (
        int(return_code) == 125
        and (watcher_status in RESOURCE_WAIT_STATUSES or marked_high_util_wait)
        and not status.get("lease_acquired_utc")
        and not status.get("child_pid")
    )


def sha_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def attempt_gpu_hours(status: dict[str, Any]) -> float:
    start = status.get("lease_acquired_utc")
    end = status.get("lease_released_utc")
    if not start or not end or status.get("selected_gpu") not in {2, 3}:
        return 0.0
    return max(0.0, (parse_time(end) - parse_time(start)).total_seconds() / 3600.0)


def current_gpu_hours() -> float:
    ledger = TASK / "TASK_JOB_LEDGER.jsonl"
    if not ledger.exists():
        return 0.0
    total = 0.0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += float(json.loads(line).get("lease_gpu_hours", 0.0))
    return total


def accepted_stage0(row: dict[str, str]) -> bool:
    root = TASK / "runs/stage0" / f"{row['job_id']}"
    report_path = root / "STAGE0_REPORT.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seed_manifest = json.loads((TASK / "protocol/SEED_INIT_MANIFEST.json").read_text(encoding="utf-8"))["seeds"]
    expected_init = next(item["init_state_hash"] for item in seed_manifest if int(item["seed"]) == int(row["seed"]))
    return (
        report.get("status") == "PASS"
        and int(report.get("optimizer_steps", -1)) == 4
        and int(report.get("scheduler_step_calls", -1)) == 4
        and report.get("init_hash") == expected_init
    )


def accepted_formal(row: dict[str, str]) -> bool:
    root = TASK / "runs/formal" / row["run_id"]
    marker = root / "RUN_COMPLETE.json"
    checkpoint = root / "checkpoint_last.pt"
    if not marker.is_file() or not checkpoint.is_file():
        return False
    done = json.loads(marker.read_text(encoding="utf-8"))
    return (
        done.get("status") == "PASS"
        and int(done.get("epochs", -1)) == 40
        and int(done.get("optimizer_steps", -1)) == 11120
        and int(done.get("scheduler_step_calls", -1)) == 11120
        and done.get("study_protocol_hash") == sha_file(PROTOCOL_PATH)
        and done.get("checkpoint_last_sha256") == sha_file(checkpoint)
    )


def make_job(row: dict[str, Any], *, phase: str, measured_peak_mib: int, attempt: int) -> tuple[dict[str, Any], Path, Path]:
    stage0 = phase == "STAGE0"
    run_id = str(row.get("run_id") or str(row["job_id"]).removesuffix("-STAGE0"))
    cell_id = str(row["job_id"]) if stage0 else run_id
    job_id = f"{cell_id}-ATTEMPT{attempt}"
    run_root = TASK / ("runs/stage0" if stage0 else "runs/formal") / cell_id
    status_path = TASK / "logs" / f"watcher_{job_id}.json"
    job_path = TASK / "configs/jobs" / f"{job_id}.json"
    child_log = TASK / "logs" / f"child_{job_id}.log"
    command = [
        str(PYTHON), str(TASK / "code/physical_gpu_guard.py"),
        str(TASK / "code/d3_train_corrected.py"),
        "--config", str(row["config"]),
        "--horizon", str(row["horizon_steps"]),
        "--seed", str(row["seed"]),
        "--run-root", str(run_root),
        "--batch-size", "64",
        "--accumulation", "1",
    ]
    if stage0:
        command.append("--stage0")
    job = {
        "job_id": job_id,
        "protocol_id": "CORRECTED_SCHEDULER_MATCHED_12",
        "output_root": str(run_root.resolve()),
        "command": command,
        "resource_estimate": {"measured_peak_mib": int(measured_peak_mib)},
        "preferred_gpu": 2,
        "active_physical_gpu_candidates": [2],
        "own_queue_concurrency": 1,
        "max_oom_requeues": 0,
        "environment": {"D3_STUDY_PROTOCOL_SHA256": sha_file(PROTOCOL_PATH)},
        "log_path": str(child_log),
        "task_phase": phase,
        "matrix_run_id": cell_id,
        "final_test_access": 0,
    }
    job_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(job_path, job)
    return job, job_path, status_path


def record_attempt(job: dict[str, Any], status_path: Path, return_code: int) -> dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    gpu_hours = attempt_gpu_hours(status)
    resource_wait = is_resource_wait_attempt(return_code, status)
    record = {
        "utc": utc_now(),
        "phase": job["task_phase"],
        "run_id": job["matrix_run_id"],
        "job_id": job["job_id"],
        "watcher_return_code": return_code,
        "watcher_status": status.get("status"),
        "attempt_kind": "resource_wait" if resource_wait else "gpu_child_attempt",
        "gpu_lease_acquired": bool(status.get("lease_acquired_utc")),
        "child_started": bool(status.get("lease_acquired_utc")),
        "selected_gpu": status.get("selected_gpu"),
        "selected_gpu_uuid": (status.get("final_recheck") or {}).get("uuid"),
        "lease_gpu_hours": gpu_hours,
        "lease_released": status.get("lease_released"),
        "non_child_process_signaled": status.get("non_child_process_signaled", False),
        "final_test_access": 0,
        "status_path": str(status_path),
    }
    append_jsonl(TASK / "TASK_JOB_LEDGER.jsonl", record)
    return record


def launch_wave(rows: list[dict[str, Any]], *, phase: str, measured_peak_mib: int) -> list[dict[str, Any]]:
    if len(rows) > MAX_CHILDREN:
        raise AssertionError("submit at most the locked own-queue concurrency; use launch_resumable")
    pending: list[tuple[dict[str, Any], Path, Path, Any, Any]] = []
    completed: list[dict[str, Any]] = []
    for row in rows:
        if phase == "STAGE0" and accepted_stage0(row):
            completed.append({"run_id": row["job_id"], "status": "PASS_REUSED"})
            continue
        if phase == "FORMAL" and accepted_formal(row):
            completed.append({"run_id": row["run_id"], "status": "PASS_REUSED"})
            continue
        run_id = str(row["job_id"] if phase == "STAGE0" else row["run_id"])
        attempts_used = sum(
            1 for line in (TASK / "TASK_JOB_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("run_id") == run_id
        ) if (TASK / "TASK_JOB_LEDGER.jsonl").exists() else 0
        attempt = attempts_used + 1
        job, job_path, status_path = make_job(row, phase=phase, measured_peak_mib=measured_peak_mib, attempt=attempt)
        controller_log = (TASK / "logs" / f"controller_{job['job_id']}.log").open("a", encoding="utf-8")
        argv = [
            str(PYTHON), str(TASK / "code/gpu23_launcher.py"),
            "--job-json", str(job_path),
            "--lease-dir", str(LEASE_DIR),
            "--status-path", str(status_path),
            "--poll-seconds", "15",
            "--max-wait-seconds", str(WATCHER_MAX_WAIT_SECONDS),
        ]
        process = subprocess.Popen(argv, cwd=str(TASK), stdout=controller_log, stderr=subprocess.STDOUT, text=True)
        controller_log.close()
        pending.append((job, job_path, status_path, process, None))

    while pending:
        if STOP_REQUESTED:
            for job, _, _, process, _ in pending:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for job, _, status_path, process, _ in pending:
                return_code = process.wait()
                record_attempt(job, status_path, return_code)
            raise KeyboardInterrupt("queue stop requested; owned watcher children were signaled")
        next_pending = []
        for job, job_path, status_path, process, _ in pending:
            return_code = process.poll()
            if return_code is None:
                next_pending.append((job, job_path, status_path, process, None))
                continue
            if process.stdout is not None:
                process.stdout.close()
            attempt = record_attempt(job, status_path, return_code)
            if return_code != 0:
                result_status = "RESOURCE_WAIT" if attempt["attempt_kind"] == "resource_wait" else "FAILED"
                completed.append({**attempt, "status": result_status})
                continue
            completed.append({**attempt, "status": "PASS"})
        pending = next_pending
        atomic_json(TASK / "QUEUE_STATUS.json", {
            "status": "running",
            "phase": phase,
            "completed_jobs": completed,
            "pending_job_ids": [item[0]["job_id"] for item in pending],
            "actual_gpu_hours": current_gpu_hours(),
            "own_queue_concurrency": MAX_CHILDREN,
            "project_gpu_child_cap": MAX_PROJECT_CHILDREN,
            "active_physical_gpu_candidates": [2],
            "final_test_access": 0,
            "updated_utc": utc_now(),
        })
        if pending:
            time.sleep(2)
    return completed


def launch_resumable(rows: list[dict[str, Any]], *, phase: str,
                     measured_peak_mib: int) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    resource_wait_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["job_id"] if phase == "STAGE0" else row["run_id"])
        if phase == "STAGE0" and accepted_stage0(row):
            results[key] = {"run_id": key, "status": "PASS_REUSED"}
            continue
        if phase == "FORMAL" and accepted_formal(row):
            results[key] = {"run_id": key, "status": "PASS_REUSED"}
            continue
        attempts_used = child_attempt_count(key)
        if attempts_used >= MAX_CHILD_ATTEMPTS_PER_CELL:
            atomic_json(TASK / "QUEUE_STATUS.json", {
                "status": "technical_failure", "phase": phase, "run_id": key,
                "reason": "per-cell child-attempt ceiling reached",
                "child_attempts": attempts_used, "child_attempt_ceiling": MAX_CHILD_ATTEMPTS_PER_CELL,
                "actual_gpu_hours": current_gpu_hours(), "final_test_access": 0,
                "updated_utc": utc_now(),
            })
            raise RuntimeError(f"child-attempt ceiling reached for {key}: {attempts_used}")
        while True:
            if STOP_REQUESTED:
                atomic_json(TASK / "QUEUE_STATUS.json", {
                    "status": "paused", "phase": phase, "waiting_run_id": key,
                    "actual_gpu_hours": current_gpu_hours(), "final_test_access": 0,
                    "updated_utc": utc_now(),
                })
                raise KeyboardInterrupt("queue stop requested between owned watcher attempts")
            attempts = launch_wave([row], phase=phase, measured_peak_mib=measured_peak_mib)
            if len(attempts) != 1:
                raise AssertionError("single-card queue expected exactly one watcher result")
            attempt = attempts[0]
            if attempt.get("status") != "RESOURCE_WAIT":
                results[key] = attempt
                break
            resource_wait_counts[key] = resource_wait_counts.get(key, 0) + 1
            atomic_json(TASK / "QUEUE_STATUS.json", {
                "status": "resource_waiting",
                "phase": phase,
                "waiting_run_id": key,
                "resource_wait_attempts": resource_wait_counts[key],
                "last_watcher_status": attempt.get("watcher_status"),
                "last_job_id": attempt.get("job_id"),
                "active_physical_gpu_candidates": [2],
                "own_queue_concurrency": 1,
                "project_gpu_child_cap": MAX_PROJECT_CHILDREN,
                "actual_gpu_hours": current_gpu_hours(),
                "retry_backoff_seconds": RESOURCE_WAIT_BACKOFF_SECONDS,
                "final_test_access": 0,
                "updated_utc": utc_now(),
            })
            for _ in range(RESOURCE_WAIT_BACKOFF_SECONDS):
                if STOP_REQUESTED:
                    break
                time.sleep(1)
    return [results[str(row["job_id"] if phase == "STAGE0" else row["run_id"])] for row in rows]


def retry_failed(rows: list[dict[str, Any]], first_results: list[dict[str, Any]], *,
                 phase: str, measured_peak_mib: int) -> list[dict[str, Any]]:
    failed_results = [result for result in first_results if result.get("status") == "FAILED"]
    retryable_ids = {
        str(result["run_id"]) for result in failed_results
        if can_retry_failed_job(str(result.get("job_id", "")))
    }
    already_retried = [result for result in failed_results if not can_retry_failed_job(str(result.get("job_id", "")))]
    retry_rows = [
        row for row in rows
        if str(row["job_id"] if phase == "STAGE0" else row["run_id"]) in retryable_ids
    ]
    retry_results = launch_resumable(retry_rows, phase=phase, measured_peak_mib=measured_peak_mib) if retry_rows else []
    unresolved = already_retried + [row for row in retry_results if row.get("status") == "FAILED"]
    if unresolved:
        atomic_json(TASK / "QUEUE_STATUS.json", {
            "status": "technical_failure", "phase": phase,
            "results": first_results + retry_results,
            "unresolved_failures": unresolved,
            "max_automatic_retries": MAX_RETRIES,
            "final_test_access": 0, "updated_utc": utc_now(),
        })
        raise RuntimeError(f"bounded technical retry exhausted; matrix stops without changing recipe: {unresolved}")
    return first_results + retry_results


def handle_signal(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> int:
    global MAX_CHILDREN
    if os.environ.get("TMUX") is None:
        raise RuntimeError("long GPU queue must run inside its named persistent tmux session")
    session_name = os.environ.get("D3_TMUX_SESSION", "").strip()
    if not session_name:
        pane = os.environ.get("TMUX_PANE", "").strip()
        if pane:
            result = subprocess.run(["tmux", "display-message", "-p", "-t", pane, "#S"],
                                    capture_output=True, text=True, check=True)
            session_name = result.stdout.strip()
    if not session_name:
        raise RuntimeError("named tmux session could not be recorded; set D3_TMUX_SESSION")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or protocol.get("study_id") != "CORRECTED_SCHEDULER_MATCHED_12":
        raise AssertionError("task protocol is not locked")
    budget = protocol.get("resource_budget", {})
    if (protocol.get("final_test_access") != 0
            or budget.get("allowed_physical_gpu_ids") != [2, 3]
            or budget.get("active_launch_candidates") != [2]
            or budget.get("own_queue_concurrency") != 1):
        raise AssertionError("task scope, active GPU2 candidate, or single-card queue lock mismatch")
    resource = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))
    if resource.get("status") != "PASS_PRE_STAGE0":
        raise AssertionError("pre-Stage0 resource gate did not pass")

    with (STAGE0_MATRIX_PATH).open(newline="", encoding="utf-8") as handle:
        stage0_rows = list(csv.DictReader(handle))
    with MATRIX_PATH.open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    if len(stage0_rows) != 4 or len(matrix_rows) != 12:
        raise AssertionError("Stage0/formal matrix coverage mismatch")
    gate_host = json.loads((TASK / "RESOURCE_GATE_HOST.json").read_text(encoding="utf-8"))
    gpu_allow = json.loads((TASK / "protocol/GPU_ALLOWLIST_LOCKED.json").read_text(encoding="utf-8"))
    launch_gate = json.loads(LAUNCH_GATE_PATH.read_text(encoding="utf-8"))
    if gate_host.get("status") != "PASS" or gpu_allow.get("status") != "LOCKED":
        raise AssertionError("current host/GPU allowlist gate failed")
    candidates = [int(index) for index in gate_host.get("launch_candidates", [])]
    if (candidates != [2] or gpu_allow.get("physical_gpu_ids") != [2, 3]
            or gpu_allow.get("launch_candidates") != candidates):
        raise AssertionError("host gate and locked GPU2-only candidate set differ")
    MAX_CHILDREN = effective_queue_concurrency(candidates)
    if MAX_CHILDREN != 1 or gate_host.get("own_queue_concurrency") != 1:
        raise AssertionError("own queue concurrency must be exactly one")
    if (launch_gate.get("status") != "PASS"
            or launch_gate.get("protocol_sha256") != sha_file(PROTOCOL_PATH)
            or launch_gate.get("active_physical_gpu_candidates") != [2]
            or launch_gate.get("own_queue_concurrency") != 1
            or len(launch_gate.get("stage0_job_keys", [])) != 4
            or len(launch_gate.get("formal_job_keys", [])) != 12
            or launch_gate.get("gpu_queries_performed") != 0):
        raise AssertionError("CPU watcher dry-run launch gate is absent or mismatched")
    if launch_gate.get("source_hashes_sha256") != sha_file(TASK / "protocol/SOURCE_HASHES.json"):
        raise AssertionError("launch gate source manifest hash mismatch")
    source_hashes = json.loads((TASK / "protocol/SOURCE_HASHES.json").read_text(encoding="utf-8"))
    if source_hashes.get("task_code_sha256") != protocol.get("task_code_hashes"):
        raise AssertionError("protocol task-code hashes and source manifest differ")
    for relative, expected_hash in protocol.get("task_code_hashes", {}).items():
        if sha_file(TASK / relative) != expected_hash:
            raise AssertionError(f"locked task code changed before launch: {relative}")
    if (launch_gate.get("resource_gate_host_sha256") != sha_file(TASK / "RESOURCE_GATE_HOST.json")
            or launch_gate.get("gpu_allowlist_sha256") != sha_file(TASK / "protocol/GPU_ALLOWLIST_LOCKED.json")
            or launch_gate.get("matrix_sha256") != sha_file(MATRIX_PATH)
            or launch_gate.get("stage0_matrix_sha256") != sha_file(STAGE0_MATRIX_PATH)):
        raise AssertionError("launch gate resource/matrix/GPU lock hashes mismatch")
    host_age = (datetime.now(timezone.utc) - parse_time(str(gate_host["gpu_snapshot_observed_utc"]))).total_seconds()
    if host_age < 0 or host_age > 600:
        raise AssertionError("host GPU2/resource preflight is stale; refresh before launch")
    disk = shutil.disk_usage(PROJECT)
    if disk.free / disk.total < 0.10:
        raise AssertionError("current project filesystem free space fell below 10 percent")

    atomic_json(TASK / "QUEUE_STATUS.json", {
        "status": "running", "phase": "stage0", "started_utc": utc_now(),
        "session_name": session_name, "active_physical_gpu_candidates": [2],
        "own_queue_concurrency": MAX_CHILDREN,
        "project_gpu_child_cap": MAX_PROJECT_CHILDREN, "final_test_access": 0,
    })
    stage0_peak_mib = int(resource["measured_old_stage0_peak_reserved_mib"])
    stage0_results = retry_failed(
        stage0_rows, launch_resumable(stage0_rows, phase="STAGE0", measured_peak_mib=stage0_peak_mib),
        phase="STAGE0", measured_peak_mib=stage0_peak_mib,
    )
    if any(row.get("status") not in {"PASS", "PASS_REUSED"} for row in stage0_results) or not all(accepted_stage0(row) for row in stage0_rows):
        atomic_json(TASK / "QUEUE_STATUS.json", {"status": "technical_failure", "phase": "stage0", "results": stage0_results, "final_test_access": 0})
        return 2

    reports = [
        json.loads((TASK / "runs/stage0" / f"{row['job_id']}" / "STAGE0_REPORT.json").read_text(encoding="utf-8"))
        for row in stage0_rows
    ]
    measured_peak_mib = math.ceil(max(int(row["peak_reserved_bytes"]) for row in reports) / (1024 * 1024))
    if measured_peak_mib > GPU_MEMORY_CAP_MIB:
        atomic_json(TASK / "QUEUE_STATUS.json", {"status": "budget_insufficient", "reason": "measured peak exceeds 20GiB cap", "peak_mib": measured_peak_mib, "final_test_access": 0})
        return 3
    formal_required_free_mib = measured_peak_mib + max(512, math.ceil(0.10 * measured_peak_mib))
    post_stage0 = {
        "status": "PASS",
        "stage0_cells": "4/4",
        "measured_peak_reserved_mib": measured_peak_mib,
        "formal_watcher_required_free_mib": formal_required_free_mib,
        "gpu_memory_cap_mib": GPU_MEMORY_CAP_MIB,
        "final_test_access": 0,
    }
    atomic_json(TASK / "RESOURCE_GATE_POST_STAGE0.json", post_stage0)
    atomic_json(TASK / "STAGE0_ACCEPTANCE.json", {"status": "PASS", "cells": reports, "final_test_access": 0})

    first_seed = [row for row in matrix_rows if int(row["seed"]) == 20261001]
    rest = [row for row in matrix_rows if int(row["seed"]) != 20261001]
    first_seed_results = retry_failed(
        first_seed, launch_resumable(first_seed, phase="FORMAL", measured_peak_mib=measured_peak_mib),
        phase="FORMAL", measured_peak_mib=measured_peak_mib,
    )
    if any(row.get("status") not in {"PASS", "PASS_REUSED"} for row in first_seed_results) or not all(accepted_formal(row) for row in first_seed):
        atomic_json(TASK / "QUEUE_STATUS.json", {"status": "technical_failure", "phase": "first_seed_formal", "results": first_seed_results, "final_test_access": 0})
        return 2

    actual = current_gpu_hours()
    remaining_estimate = 2.0 * float(resource["old_other_8_formal_lease_gpu_hours"])
    projected = actual + remaining_estimate + 0.10
    if projected > GPU_HOUR_CAP:
        atomic_json(TASK / "RESOURCE_GATE_POST_FIRST_SEED.json", {
            "status": "budget_insufficient", "actual_gpu_hours": actual,
            "conservative_projected_total_gpu_hours": projected, "gpu_hour_cap": GPU_HOUR_CAP,
            "final_test_access": 0,
        })
        return 3
    atomic_json(TASK / "RESOURCE_GATE_POST_FIRST_SEED.json", {
        "status": "PASS", "actual_gpu_hours": actual,
        "conservative_projected_total_gpu_hours": projected, "gpu_hour_cap": GPU_HOUR_CAP,
        "final_test_access": 0,
    })

    # Preserve locked row order; submit each job only after its predecessor terminates.
    ordered_rest = sorted(rest, key=lambda row: (int(row["seed"]), int(row["horizon_steps"]), str(row["config"])))
    rest_results: list[dict[str, Any]] = []
    for offset in range(0, len(ordered_rest), MAX_CHILDREN):
        wave = ordered_rest[offset:offset + MAX_CHILDREN]
        wave_results = retry_failed(
            wave, launch_resumable(wave, phase="FORMAL", measured_peak_mib=measured_peak_mib),
            phase="FORMAL", measured_peak_mib=measured_peak_mib,
        )
        rest_results.extend(wave_results)
        if any(row.get("status") not in {"PASS", "PASS_REUSED"} for row in wave_results) or not all(accepted_formal(row) for row in wave):
            atomic_json(TASK / "QUEUE_STATUS.json", {"status": "technical_failure", "phase": "remaining_formal", "results": rest_results, "final_test_access": 0})
            return 2
        if current_gpu_hours() > GPU_HOUR_CAP:
            atomic_json(TASK / "QUEUE_STATUS.json", {"status": "budget_insufficient", "phase": "remaining_formal", "actual_gpu_hours": current_gpu_hours(), "gpu_hour_cap": GPU_HOUR_CAP, "final_test_access": 0})
            return 3

    actual = current_gpu_hours()
    if actual > GPU_HOUR_CAP:
        atomic_json(TASK / "QUEUE_STATUS.json", {"status": "budget_insufficient", "actual_gpu_hours": actual, "gpu_hour_cap": GPU_HOUR_CAP, "final_test_access": 0})
        return 3
    atomic_json(TASK / "RESOURCE_GATE_FINAL_TRAINING.json", {
        "status": "PASS", "actual_training_gpu_hours": actual,
        "gpu_hour_cap": GPU_HOUR_CAP, "final_test_access": 0,
    })
    atomic_json(TASK / "QUEUE_STATUS.json", {
        "status": "training_complete",
        "stage0_accepted": "4/4",
        "formal_training_accepted": "12/12",
        "session_name": session_name,
        "active_physical_gpu_candidates": [2],
        "own_queue_concurrency": 1,
        "actual_training_gpu_hours": actual,
        "completed_utc": utc_now(),
        "final_test_access": 0,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
