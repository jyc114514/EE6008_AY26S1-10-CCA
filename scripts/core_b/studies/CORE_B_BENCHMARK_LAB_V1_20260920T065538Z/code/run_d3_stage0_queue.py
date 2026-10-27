"""Wait for cache acceptance, fit the normalizer, then run 12 D3 Stage0 jobs."""

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


def append_ledger(record: dict[str, object]) -> None:
    with (BENCH_ROOT / "JOB_LEDGER.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    log_path = BENCH_ROOT / "logs/D3-STAGE0-QUEUE.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{utc_now()} queue_started\n")
        while True:
            path = BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
            if path.exists() and read_json(path).get("status") == "PASS":
                break
            log.write(f"{utc_now()} waiting_for_full_cache\n")
            log.flush()
            time.sleep(30)
        normalizer = BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"
        if not normalizer.exists():
            result = subprocess.run([PYTHON, str(BENCH_ROOT / "code/fit_d3_normalizer.py")], cwd=BENCH_ROOT, capture_output=True, text=True)
            log.write(result.stdout + result.stderr)
            log.flush()
            if result.returncode != 0:
                append_ledger({"utc": utc_now(), "job_id": "D3-NORMALIZER", "status": "failed", "exit_code": result.returncode, "final_test_access": 0})
                return result.returncode
        manifest = read_json(BENCH_ROOT / "configs/D3_STAGE0_TRAIN_JOBS.json")
        for item in manifest["jobs"]:
            status_path = Path(item["status_path"])
            if status_path.exists() and read_json(status_path).get("status") == "complete":
                continue
            effective = dict(item)
            if status_path.exists() and read_json(status_path).get("status") == "failed":
                original = read_json(Path(item["job_path"]))
                retry_id = f"{item['job_id']}-RETRY1"
                retry_root = BENCH_ROOT / "d3/runs" / retry_id
                retry_job = copy.deepcopy(original)
                retry_job["job_id"] = retry_id
                retry_job["output_root"] = str(retry_root)
                retry_job["log_path"] = str(BENCH_ROOT / "logs" / f"child_{retry_id}.log")
                retry_job["command"] = [retry_job["command"][0], retry_job["command"][1], "--config", retry_job["command"][3], "--horizon", retry_job["command"][5], "--seed", retry_job["command"][7], "--run-root", str(retry_root), "--stage0", "--batch-size", retry_job["command"][-3], "--accumulation", retry_job["command"][-1]]
                retry_path = BENCH_ROOT / "configs" / f"{retry_id}.json"
                from common import atomic_write_json
                atomic_write_json(retry_path, retry_job)
                effective = {"job_id": retry_id, "job_path": str(retry_path), "status_path": str(BENCH_ROOT / "logs" / f"watcher_{retry_id}.json")}
            command = [PYTHON, WATCHER, "--job-json", effective["job_path"], "--lease-dir", LEASE_DIR, "--status-path", effective["status_path"], "--poll-seconds", "15.0", "--max-wait-seconds", "0", "--max-oom-requeues", "1"]
            log.write(f"{utc_now()} launch {effective['job_id']}\n")
            log.flush()
            result = subprocess.run(command, cwd=BENCH_ROOT, capture_output=True, text=True)
            log.write(result.stdout + result.stderr)
            log.flush()
            append_ledger({"utc": utc_now(), "job_id": effective["job_id"], "status": "complete" if result.returncode == 0 else "failed", "exit_code": result.returncode, "final_test_access": 0})
            if result.returncode != 0:
                return result.returncode
        atomic_write_json(BENCH_ROOT / "d3/D3_STAGE0_QUEUE_COMPLETE.json", {"status": "PASS", "jobs": len(manifest["jobs"]), "seed": manifest["seed"], "generated_utc": utc_now(), "final_test_access": 0})
        log.write(f"{utc_now()} queue_complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
