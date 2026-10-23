from __future__ import annotations
import os

import json
import copy
import subprocess
import time
from pathlib import Path

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, utc_now


PYTHON = os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python')
WATCHER = str(REPO_ROOT / "tools/gpu_lease_watcher.py")
LEASE_DIR = str(REPO_ROOT / "reports/locks/gpu_lease")


def main() -> int:
    log = BENCH_ROOT / "logs/D3-FORMAL-QUEUE.log"; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} queue_started\n")
        while not (BENCH_ROOT / "d3/D3_STAGE0_QUEUE_COMPLETE.json").exists():
            handle.write(f"{utc_now()} waiting_for_stage0\n"); handle.flush(); time.sleep(30)
        manifest = read_json(BENCH_ROOT / "configs/D3_FORMAL_TRAIN_JOBS.json")
        for item in manifest["jobs"]:
            status_path = Path(item["status_path"])
            if status_path.exists() and read_json(status_path).get("status") == "complete": continue
            effective = dict(item)
            if status_path.exists():
                original = read_json(Path(item["job_path"]))
                retry_id = f"{item['job_id']}-RETRY1"
                retry_root = BENCH_ROOT / "d3/runs" / retry_id
                retry_job = copy.deepcopy(original)
                retry_job["job_id"] = retry_id
                retry_job["output_root"] = str(retry_root)
                retry_job["log_path"] = str(BENCH_ROOT / "logs" / f"child_{retry_id}.log")
                retry_job["command"] = [retry_job["command"][0], retry_job["command"][1], "--config", retry_job["command"][3], "--horizon", retry_job["command"][5], "--seed", retry_job["command"][7], "--run-root", str(retry_root), "--batch-size", retry_job["command"][-3], "--accumulation", retry_job["command"][-1]]
                retry_path = BENCH_ROOT / "configs" / f"{retry_id}.json"
                atomic_write_json(retry_path, retry_job)
                effective = {"job_id": retry_id, "job_path": str(retry_path), "status_path": str(BENCH_ROOT / "logs" / f"watcher_{retry_id}.json")}
            command = [PYTHON, WATCHER, "--job-json", effective["job_path"], "--lease-dir", LEASE_DIR, "--status-path", effective["status_path"], "--poll-seconds", "15.0", "--max-wait-seconds", "0", "--max-oom-requeues", "1"]
            handle.write(f"{utc_now()} launch {effective['job_id']}\n"); handle.flush()
            result = subprocess.run(command, cwd=BENCH_ROOT, capture_output=True, text=True); handle.write(result.stdout + result.stderr); handle.flush()
            with (BENCH_ROOT / "JOB_LEDGER.jsonl").open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps({"utc": utc_now(), "job_id": effective["job_id"], "status": "complete" if result.returncode == 0 else "failed", "exit_code": result.returncode, "final_test_access": 0}) + "\n")
            if result.returncode != 0: return result.returncode
        atomic_write_json(BENCH_ROOT / "d3/D3_FORMAL_QUEUE_COMPLETE.json", {"status": "PASS", "jobs": len(manifest["jobs"]), "generated_utc": utc_now(), "final_test_access": 0})
        handle.write(f"{utc_now()} queue_complete\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
