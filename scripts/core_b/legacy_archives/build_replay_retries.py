#!/usr/bin/env python3
"""Create isolated retries after the recovery-local graph-retention defect."""

from __future__ import annotations
import os

import json
from pathlib import Path


RECOVERY = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z'))
FAILED_CELLS = ("FT-D-LORA4", "FT-D-DENSE4")


def main() -> int:
    out = RECOVERY / "data/jobs/replay_retry1"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cell in FAILED_CELLS:
        for path_name in ("a", "b"):
            old = json.loads((RECOVERY / f"data/jobs/replay/B-RECOVERY-REPLAY-{cell}-{path_name}.json").read_text(encoding="utf-8"))
            job_id = f"B-RECOVERY-REPLAY-{cell}-{path_name}-RETRY1"
            old["job_id"] = job_id
            old["output_root"] = str(RECOVERY / f"replay/{cell}/{path_name}_retry1")
            old["log_path"] = str(RECOVERY / f"logs/child_{job_id}.log")
            path = out / f"{job_id}.json"
            path.write_text(json.dumps(old, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            jobs.append(old)
    (RECOVERY / "data/REPLAY_RETRY1_JOBS.json").write_text(json.dumps({"status": "LOCKED", "jobs": jobs, "count": len(jobs), "reason": "prior recovery exporter retained autograd graph during forward; source OOM evidence preserved", "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "jobs": len(jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
