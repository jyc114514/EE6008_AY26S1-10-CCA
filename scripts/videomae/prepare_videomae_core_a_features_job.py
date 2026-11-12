"""Prepare an isolated strict Core A extraction job through the shared watcher."""

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
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    project, root = args.project_root.resolve(), args.job_root.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(root.is_relative_to(scratch) and root != scratch and not root.exists(), "Fresh personal job required")
    report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    require(report["status"] == "PASS_CORE_A_STRICT_CPU_PREFLIGHT" and report["strict_load"]
            and report["final_test_access"] == 0, "Accepted strict preflight required")
    require(args.extractor.is_file(), "Extractor missing")
    root.mkdir(parents=True, exist_ok=False)
    job = {"job_id": "videomae-core-a-features-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "protocol_id": "VIDEOMAE_CORE_A_STRICT_FEATURES_V1", "output_root": str(root),
           "command": [str(project / "envs/run_jyc.sh"), "python", str(args.extractor.resolve()),
                       "--project-root", str(project), "--output-root", str(root / "features"),
                       "--phase", "full", "--device", "cuda", "--preflight-report", str(args.preflight_report.resolve())],
           "resource_estimate": {"peak_memory_mib": 20480, "basis": "conservative bound; not measured for Core A"},
           "max_oom_requeues": 0, "extractor_sha256": digest(args.extractor),
           "preflight_sha256": digest(args.preflight_report)}
    path = root / "job.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
                    "--job-json", str(path), "--lease-dir", str(project / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
                    "--status-path", str(root / "watcher_status.json"), "--dry-run"], check=True)
    print(json.dumps({"status": "PREPARED_CORE_A_FEATURES_DRY_RUN_ONLY", "job_root": str(root), "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
