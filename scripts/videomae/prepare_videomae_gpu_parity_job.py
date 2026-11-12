"""Prepare a watcher-managed GPU short parity probe; no formal training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from evaluate_videomae_core_b import digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    project, runner, root = [p.resolve() for p in (args.project_root, args.runner, args.job_root)]
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(root.is_relative_to(scratch) and root != scratch and not root.exists() and runner.is_file(), "Fresh personal job and runner required")
    root.mkdir(parents=True, exist_ok=False)
    job = {"job_id": "videomae-gpu-parity-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "protocol_id": "VIDEOMAE_GPU_SHORT_PARITY_V1", "output_root": str(root),
           "command": [str(project / "envs/run_jyc.sh"), "python", str(runner),
                       "--project-root", str(project), "--output-root", str(root / "check"), "--device", "cuda"],
           "resource_estimate": {"peak_memory_mib": 20480, "basis": "conservative diagnostic bound, not measured"},
           "max_oom_requeues": 0, "runner_sha256": digest(runner)}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
                    "--job-json", str(path), "--lease-dir", str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_GPU_PARITY_DRY_RUN_ONLY", "job_root": str(root), "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
