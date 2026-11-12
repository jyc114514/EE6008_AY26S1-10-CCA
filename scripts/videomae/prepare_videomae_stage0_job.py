"""Prepare an isolated four-anchor GPU Stage0 job without launching it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    extractor = args.extractor.resolve()
    if not extractor.is_file():
        raise FileNotFoundError(extractor)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = project / "tmp/s126mdg33_02" / f"videomae_stage0_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    job = {
        "job_id": f"videomae-core-b-stage0-{stamp}",
        "protocol_id": "VIDEOMAE_CORE_B_32FRAME_RESOURCE_STAGE0_V1",
        "output_root": str(root),
        "command": [str(project / "envs/run_jyc.sh"), "python", str(extractor),
                    "--project-root", str(project), "--output-root", str(root / "features"),
                    "--input-frames", "32", "--device", "cuda"],
        "resource_estimate": {
            "peak_memory_mib": 16384,
            "basis": "conservative pre-registered Stage0 estimate, NOT a measured peak",
            "formal_registration": "replace with same-runner measured reserved peak after Stage0",
        },
        "max_oom_requeues": 0,
        "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "extractor_sha256": hashlib.sha256(extractor.read_bytes()).hexdigest(),
        "gpu_authorization": "pending; add --allowed-gpus only after current task allocation is known",
    }
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    watcher = project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
    subprocess.run([sys.executable, str(watcher), "--job-json", str(path),
                    "--lease-dir", str(project / "reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_DRY_RUN_ONLY", "job_json": str(path),
                      "output_root": str(root), "gpu_launched": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
