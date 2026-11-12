"""Prepare a two-update resume check; does not launch formal training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--stage0-report", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    project, root, runner = args.project_root.resolve(), args.job_root.resolve(), args.runner.resolve()
    if not root.is_relative_to((project / "tmp/s126mdg33_02").resolve()) or not runner.is_file():
        raise ValueError("Personal output and existing runner required")
    accepted = json.loads(args.stage0_report.read_text())
    if accepted.get("status") != "PASS_TRAINING_STAGE0_4_OF_4":
        raise ValueError("Accepted Stage0 required")
    root.mkdir(parents=True, exist_ok=False)
    job = {"job_id": "videomae-resume-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "protocol_id": "VIDEOMAE_CORRECTED_RESUME_CHECK_V1", "output_root": str(root),
           "command": [str(project / "envs/run_jyc.sh"), "python", str(runner),
                       "--project-root", str(project), "--cache-run", str(args.cache_run.resolve()),
                       "--stage0-report", str(args.stage0_report.resolve()), "--output-root", str(root / "check"),
                       "--mode", "TS11-D", "--horizon", "60", "--seed", "20261001", "--resume-check"],
           "resource_estimate": {"peak_memory_mib": 20480, "basis": "conservative new-runner check; not measured"},
           "max_oom_requeues": 0, "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
                    "--job-json", str(path), "--lease-dir",
                    str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_RESUME_CHECK_ONLY", "job_json": str(path), "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
