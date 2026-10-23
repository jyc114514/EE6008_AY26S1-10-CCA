#!/usr/bin/env python3
"""Run a V2 queue through the shared project lease watcher."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
PROJECT = Path(resolved["project_root"])
PYTHON = str(PROJECT / "envs/jyc/bin/python")
WATCHER = Path(resolved["shared_watcher"])
LEASE_DIR = ROOT / "coordination/locks/gpu_lease"
STATUS_DIR = ROOT / "coordination/watcher_status"
CONSOLE_DIR = STATUS_DIR / "console"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def watcher_cap() -> int:
    source = WATCHER.read_text(encoding="utf-8")
    match = re.search(r"^MAX_CONCURRENT_PROJECT_GPU_CHILDREN\s*=\s*(\d+)\s*$", source, re.MULTILINE)
    if not match:
        raise RuntimeError("unable to read the shared watcher enforced child cap")
    cap = int(match.group(1))
    if cap <= 0:
        raise RuntimeError("shared watcher reports a nonpositive child cap")
    return cap


def run_one(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    job_id = str(config["job_id"])
    output_root = Path(config["output_root"])
    command_script = Path(str(config.get("command", ["", ""])[1])).name if len(config.get("command", [])) > 1 else ""
    marker_name = str(config.get("completion_marker", "PREDICTION_COMPLETE.json" if command_script == "v2_inference.py" else "RUN_COMPLETE.json"))
    complete_path = output_root / marker_name
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") == "PASS":
            return {"job_id": job_id, "status": "already_complete", "returncode": 0}
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATUS_DIR / f"{job_id}.json"
    console_path = CONSOLE_DIR / f"{job_id}.log"
    command = [
        PYTHON, str(WATCHER), "--job-json", str(config_path),
        "--lease-dir", str(LEASE_DIR), "--status-path", str(status_path),
        "--poll-seconds", "10",
    ]
    with console_path.open("ab", buffering=0) as log:
        result = subprocess.run(command, cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT, check=False)
    complete = {}
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    watcher_status = {}
    if status_path.is_file():
        watcher_status = json.loads(status_path.read_text(encoding="utf-8"))
    status = "PASS" if result.returncode == 0 and complete.get("status") == "PASS" else "FAILED"
    return {
        "job_id": job_id, "status": status, "returncode": result.returncode,
        "run_complete": str(complete_path) if complete_path.exists() else None,
        "watcher_status": watcher_status.get("status"),
        "watcher_status_path": str(status_path),
        "console_log": str(console_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    queue_path = args.queue.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("schema_version") != 1 or not isinstance(queue.get("jobs"), list):
        raise RuntimeError("queue schema invalid")
    expected_hash = json.loads((ROOT / "inventory/SOURCE_ACCEPTANCE.json").read_text(encoding="utf-8"))["shared_watcher_sha256"]
    actual_hash = sha_file(WATCHER)
    if actual_hash != expected_hash:
        raise RuntimeError("shared watcher source hash differs from V2 source lock")
    configs = [Path(value).resolve() for value in queue["jobs"]]
    if len(set(configs)) != len(configs):
        raise RuntimeError("queue repeats watcher job configuration paths")
    cap = watcher_cap()
    started = utc_now()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cap) as pool:
        futures = {pool.submit(run_one, path): path for path in configs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"utc": utc_now(), **result}, sort_keys=True), flush=True)
    results.sort(key=lambda row: row["job_id"])
    summary = {
        "schema_version": 1,
        "phase": queue.get("phase"),
        "queue_path": str(queue_path),
        "started_utc": started,
        "completed_utc": utc_now(),
        "shared_watcher_sha256": actual_hash,
        "shared_watcher_max_children": cap,
        "scheduled_jobs": len(configs),
        "accepted_fits": sum(row["status"] in {"PASS", "already_complete"} for row in results),
        "failed_fits": sum(row["status"] == "FAILED" for row in results),
        "results": results,
    }
    out_path = ROOT / "coordination" / f"PHASE_STATUS_{str(queue.get('phase', 'unknown')).upper()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.part")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print(json.dumps({"status": "PASS" if not summary["failed_fits"] else "PARTIAL", **{k: summary[k] for k in ("phase", "scheduled_jobs", "accepted_fits", "failed_fits", "shared_watcher_max_children")}}, sort_keys=True), flush=True)
    return 0 if summary["failed_fits"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
