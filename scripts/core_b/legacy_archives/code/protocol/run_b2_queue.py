#!/usr/bin/env python3
"""Run remaining B2 watcher jobs sequentially on one permitted GPU child."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--watcher", type=Path, required=True)
    parser.add_argument("--start-after", default="B2-V-0125")
    args = parser.parse_args()
    exp = args.experiment_root.resolve()
    watcher = args.watcher.resolve()
    order = [
        "B2-V-0125", "B2-V-0250", "B2-V-0500",
        "B2-FUS-0125", "B2-FUS-0250", "B2-FUS-0500",
    ]
    start_index = order.index(args.start_after)
    jobs = [exp / "jobs" / f"{condition}_gpu1.json" for condition in order[start_index + 1:]]
    status_path = exp / "B2_QUEUE_STATUS.json"
    status: dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "physical_gpu": 1,
        "max_concurrent_children": 1,
        "jobs": [],
    }
    atomic_json(status_path, status)
    for job_path in jobs:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        condition = job_path.stem.split("_gpu1")[0]
        watcher_status = exp / "status" / f"{condition}_gpu1.json"
        command = [
            sys.executable,
            str(watcher),
            "--job-json", str(job_path),
            "--lease-dir", str(exp / "leases"),
            "--status-path", str(watcher_status),
            "--poll-seconds", "5",
            "--idle-samples", "2",
            "--allow-colocation",
            "--min-free-memory-mib", "14000",
        ]
        print(f"B2_QUEUE_START condition={condition} utc={utc_now()}", flush=True)
        returncode = subprocess.run(command, cwd=str(exp.parent.parent.parent.parent), check=False).returncode
        row = {"condition": condition, "job_path": str(job_path), "return_code": returncode, "completed_utc": utc_now()}
        status["jobs"].append(row)
        atomic_json(status_path, status)
        if returncode != 0:
            status.update({"status": "failed", "failed_condition": condition, "completed_utc": utc_now()})
            atomic_json(status_path, status)
            return returncode
    status.update({"status": "complete", "completed_utc": utc_now()})
    atomic_json(status_path, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
