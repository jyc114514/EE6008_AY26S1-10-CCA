#!/usr/bin/env python3
"""Create retry1 Stage0 jobs while preserving the first technical failures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_DIR = ROOT / "configs/jobs"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")


def main() -> int:
    jobs = []
    for cell in CELLS:
        old_id = f"B-{cell}-STAGE0-S20260944"
        old = json.loads((JOB_DIR / f"{old_id}.json").read_text(encoding="utf-8"))
        new_id = f"B-{cell}-STAGE0-RETRY1-S20260944"
        new = dict(old)
        old_root = Path(str(old["output_root"]))
        new_root = old_root.parent / "stage0_retry1"
        new["job_id"] = new_id
        new["output_root"] = str(new_root)
        new["log_path"] = str(ROOT / f"logs/child_{cell}_stage0_retry1.log")
        command = list(old["command"])
        command[command.index("--run-root") + 1] = str(new_root)
        new["command"] = command
        path = JOB_DIR / f"{new_id}.json"
        path.write_text(json.dumps(new, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        jobs.append(new)
    (ROOT / "configs/B_STAGE0_RETRY1_JOBS.json").write_text(json.dumps({"status": "LOCKED", "attempt": "RETRY1", "jobs": jobs, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "jobs": len(jobs), "attempt": "RETRY1"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
