"""Prepare a shared-watcher Stage0 job and validate it without launching CUDA."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from path_config import scratch_name, scratch_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cache-run", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    project, root, runner = args.project_root.resolve(), args.job_root.resolve(), args.runner.resolve()
    if not root.is_relative_to((scratch_root(project)).resolve()) or not runner.is_file():
        raise ValueError("Personal job root and existing runner required")
    cache_run = args.cache_run.resolve()
    normalizer = cache_run / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json"
    if not normalizer.is_file() or not (cache_run / "cache/CACHE_MANIFEST.json").is_file():
        raise FileNotFoundError("Accepted cache and normalizer required")
    root.mkdir(parents=True, exist_ok=False)
    job = {"job_id": "videomae-training-stage0-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "protocol_id": "VIDEOMAE_CORE_B_CORRECTED_TRAINING_STAGE0_V1",
           "output_root": str(root),
           "command": [str(project / "envs/run_jyc.sh"), "python", str(runner),
                       "--project-root", str(project), "--cache-root", str(cache_run / "cache"),
                       "--normalizer", str(normalizer), "--output-root", str(root / "stage0")],
           "resource_estimate": {"peak_memory_mib": 20480,
                                 "basis": "conservative preregistration for training; not cache extraction peak"},
           "max_oom_requeues": 0,
            "environment": {"EE6008_VIDEOMAE_SCRATCH_NAME": scratch_name()},
           "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
           "gpu_authorization": "physical 0/1/2/3; one child; shared maximum two"}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
                    "--job-json", str(path), "--lease-dir",
                    str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_DRY_RUN_ONLY", "job_json": str(path),
                      "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
