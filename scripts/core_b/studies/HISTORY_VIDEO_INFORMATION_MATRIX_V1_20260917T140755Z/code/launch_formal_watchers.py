#!/usr/bin/env python3
"""Launch the four formal jobs through the shared dynamic watcher."""

from __future__ import annotations

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
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for cell in CELLS:
        job_id = f"HVI-{cell}-FORMAL"
        stdout_path = ROOT / "logs" / f"watcher_{job_id}.stdout.log"
        handle = stdout_path.open("w", encoding="utf-8")
        process = subprocess.Popen([str(PYTHON), str(WATCHER), "--job-json", str(JOB_DIR / f"{job_id}.json"), "--lease-dir", str(LEASE_DIR), "--status-path", str(ROOT / "logs" / f"watcher_{job_id}.json"), "--poll-seconds", "30", "--max-wait-seconds", "0", "--max-oom-requeues", "0"], cwd=str(ROOT.parent.parent.parent.parent), stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.close()
        processes.append((cell, process))
    results: list[dict[str, object]] = []
    for cell, process in processes:
        returncode = process.wait()
        status_path = ROOT / "logs" / f"watcher_HVI-{cell}-FORMAL.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        results.append({"cell": cell, "watcher_pid": process.pid, "returncode": returncode, "status": status})
    payload = {"status": "PASS" if all(item["returncode"] == 0 for item in results) else "FAIL", "started_utc": started, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "controller_pid": os.getpid(), "cells": results, "final_test_access": 0}
    (ROOT / "logs/FORMAL_CONTROLLER.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
