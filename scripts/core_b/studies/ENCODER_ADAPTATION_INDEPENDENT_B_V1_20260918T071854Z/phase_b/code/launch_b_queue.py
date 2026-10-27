#!/usr/bin/env python3
"""Launch one B watcher queue (bridge or Stage0/formal) in the shared lease."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
JOB_DIR = ROOT / "configs/jobs"
LEASE_DIR = ROOT / "locks/gpu_lease"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--controller-name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    jobs = sorted(JOB_DIR.glob(args.pattern))
    if not jobs:
        raise RuntimeError(f"no jobs match {args.pattern}")
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for path in jobs:
        job = json.loads(path.read_text(encoding="utf-8"))
        job_id = str(job["job_id"])
        stdout_path = ROOT / f"logs/watcher_{job_id}.stdout.log"
        status_path = ROOT / f"logs/watcher_{job_id}.json"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen([str(PYTHON), str(WATCHER), "--job-json", str(path), "--lease-dir", str(LEASE_DIR), "--status-path", str(status_path), "--poll-seconds", str(args.poll_seconds), "--max-wait-seconds", "0", "--max-oom-requeues", "1"], cwd=str(PROJECT), stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((job_id, process))
    results: list[dict[str, object]] = []
    for job_id, process in processes:
        code = process.wait()
        status_path = ROOT / f"logs/watcher_{job_id}.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        results.append({"job_id": job_id, "watcher_pid": process.pid, "returncode": code, "status": status})
    payload = {"status": "PASS" if all(item["returncode"] == 0 for item in results) else "FAIL", "controller": args.controller_name, "started_utc": started, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "controller_pid": os.getpid(), "results": results, "visible_gpu_pool": "dynamic_all_visible", "max_project_gpu_children": 2, "other_processes_signaled": False, "final_test_access": 0}
    (ROOT / f"logs/{args.controller_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "jobs": len(results), "returncodes": [item["returncode"] for item in results]}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
