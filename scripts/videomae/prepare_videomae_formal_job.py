"""Prepare the first isolated 40-epoch VideoMAE job; dry-run only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--stage0-report", type=Path, required=True)
    parser.add_argument("--resume-check-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    project, root, runner = args.project_root.resolve(), args.job_root.resolve(), args.runner.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    if root == scratch or not root.is_relative_to(scratch):
        raise ValueError("New personal job directory required")
    check_root = args.resume_check_root.resolve()
    accepted_path = check_root / "check/RESUME_CHECK.json"
    accepted = json.loads(accepted_path.read_text())
    status = json.loads((check_root / "watcher_status.json").read_text())
    runner_hash = digest(runner)
    if (accepted.get("status") != "PASS_EXACT_RESUME" or accepted.get("runner_sha256") != runner_hash
            or accepted.get("formal_training_started") is not False or accepted.get("final_test_access") != 0
            or status.get("status") != "complete" or status.get("exit_code") != 0
            or not status.get("lease_released_utc")):
        raise ValueError("Same-runner completed resume check required")
    stage0 = json.loads(args.stage0_report.read_text())
    if stage0.get("status") != "PASS_TRAINING_STAGE0_4_OF_4":
        raise ValueError("Stage0 gate failed")
    cache_run = args.cache_run.resolve()
    if not (cache_run / "cache/CACHE_MANIFEST.json").is_file():
        raise FileNotFoundError("Full cache missing")
    root.mkdir(parents=True, exist_ok=False)
    # Use the existing conservative training reservation, not the tiny smoke peak.
    job = {"job_id": "videomae-formal-D-H60-S20261001-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "protocol_id": "VIDEOMAE_CORE_B_CORRECTED_FORMAL_V1", "output_root": str(root),
           "command": [str(project / "envs/run_jyc.sh"), "python", str(runner),
                       "--project-root", str(project), "--cache-run", str(cache_run),
                       "--stage0-report", str(args.stage0_report.resolve()), "--output-root", str(root / "run"),
                       "--mode", "TS11-D", "--horizon", "60", "--seed", "20261001"],
           "resource_estimate": {"peak_memory_mib": 20480,
                                 "basis": "conservative training reservation; smoke peak not formal peak"},
           "max_oom_requeues": 0, "runner_sha256": runner_hash,
           "resume_acceptance_sha256": digest(accepted_path),
           "gpu_authorization": "physical 0/1/2/3; one child; shared maximum two"}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
                    "--job-json", str(path), "--lease-dir",
                    str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_FORMAL_DRY_RUN_ONLY", "job_json": str(path),
                      "run_root": str(root / "run"), "epochs": 40, "updates": 11120,
                      "gpu_launched": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()
