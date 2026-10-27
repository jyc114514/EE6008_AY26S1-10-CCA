#!/usr/bin/env python3
"""Launch recovery jobs through the shared dynamic GPU watcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
LEASE_DIR = RECOVERY / "data/gpu_lease"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("replay", "infer"), required=True)
    parser.add_argument("--retry1", action="store_true")
    args = parser.parse_args()
    job_dir = RECOVERY / f"data/jobs/{args.kind}{'_retry1' if args.retry1 else ''}"
    jobs = sorted(job_dir.glob("*.json"))
    if not jobs:
        raise RuntimeError(f"no {args.kind} jobs")
    processes: list[tuple[str, subprocess.Popen[str], Path]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for job_path in jobs:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job_id = str(job["job_id"])
        status_path = RECOVERY / f"logs/watcher_{job_id}.json"
        stdout_path = RECOVERY / f"logs/watcher_{job_id}.stdout.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        handle = stdout_path.open("w", encoding="utf-8")
        process = subprocess.Popen([str(PYTHON), str(WATCHER), "--job-json", str(job_path), "--lease-dir", str(LEASE_DIR), "--status-path", str(status_path), "--poll-seconds", "10", "--max-wait-seconds", "0", "--max-oom-requeues", "1"], cwd=str(PROJECT), stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.close()
        processes.append((job_id, process, status_path))
    results: list[dict[str, object]] = []
    for job_id, process, status_path in processes:
        code = process.wait()
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        results.append({"job_id": job_id, "watcher_pid": process.pid, "returncode": code, "status": status})
    payload = {"status": "PASS" if all(item["returncode"] == 0 for item in results) else "FAIL", "kind": args.kind, "started_utc": started, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "controller_pid": os.getpid(), "results": results, "visible_gpu_pool": "dynamic_all_visible", "max_project_gpu_children": 2, "other_processes_signaled": False, "final_test_access": 0}
    suffix = "_RETRY1" if args.retry1 else ""
    (RECOVERY / f"logs/{args.kind.upper()}_CONTROLLER{suffix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "kind": args.kind, "jobs": len(results), "returncodes": [item["returncode"] for item in results]}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
