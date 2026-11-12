"""Prepare and dry-run a Stage0 or full cache job; never launch GPU work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--phase", choices=("stage0", "full"), required=True)
    parser.add_argument("--stage0-manifest", type=Path)
    parser.add_argument("--job-root", type=Path)
    args = parser.parse_args()
    project, extractor = args.project_root.resolve(), args.extractor.resolve()
    if not extractor.is_file():
        raise FileNotFoundError(extractor)
    script_hash = hashlib.sha256(extractor.read_bytes()).hexdigest()
    resource = {"peak_memory_mib": 4096,
                "basis": "conservative Stage0 preregistration; prior identical forward reserved 1384 MiB"}
    if args.phase == "full":
        if args.stage0_manifest is None:
            raise ValueError("Full cache requires same-runner accepted Stage0 manifest")
        accepted = json.loads(args.stage0_manifest.read_text())
        if (accepted.get("status") != "PASS_STAGE0" or accepted.get("script_sha256") != script_hash
                or accepted.get("covered_rows") != 4 or accepted.get("frames") != 32
                or accepted.get("batch_size") != 1 or accepted.get("dtype") != "float32"):
            raise AssertionError("Stage0 acceptance/runner identity mismatch")
        peak = accepted.get("peak_reserved_mib")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
            raise AssertionError("Stage0 reserved peak is not a positive integer")
        resource = {"measured_peak_mib": peak, "basis": "same runner, FP32, 32 frames, batch1",
                    "stage0_manifest": str(args.stage0_manifest.resolve()),
                    "stage0_manifest_sha256": hashlib.sha256(args.stage0_manifest.read_bytes()).hexdigest()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = project / "tmp/s126mdg33_02" / f"videomae_cache_{args.phase}_{stamp}"
    if args.job_root is not None:
        root = args.job_root.resolve()
        scratch = (project / "tmp/s126mdg33_02").resolve()
        if root == scratch or not root.is_relative_to(scratch):
            raise ValueError("Job root must be a new directory inside personal project scratch")
    root.mkdir(parents=True, exist_ok=False)
    command = [str(project / "envs/run_jyc.sh"), "python", str(extractor),
               "--project-root", str(project), "--output-root", str(root / "cache")]
    if args.phase == "stage0":
        command.append("--stage0")
    job = {"job_id": f"videomae-cache-{args.phase}-{stamp}",
           "protocol_id": "VIDEOMAE_CORE_B_32FRAME_CACHE_V1",
           "output_root": str(root), "command": command,
           "resource_estimate": resource, "max_oom_requeues": 0,
           "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
           "gpu_authorization": "all visible physical GPUs 0,1,2,3; max two shared project children",
           "extractor_sha256": script_hash}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    watcher = project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
    subprocess.run([sys.executable, str(watcher), "--job-json", str(path),
                    "--lease-dir", str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_DRY_RUN_ONLY", "phase": args.phase,
                      "job_json": str(path), "run_root": str(root), "gpu_launched": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
