#!/usr/bin/env python3
"""Create preserved Stage0 retry-1 watcher jobs after a technical fix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
ATTEMPT = "RETRY1"


def main() -> int:
    output: list[dict[str, object]] = []
    for cell in CELLS:
        base_path = ROOT / "configs/jobs" / f"HVI-{cell}-STAGE0.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        job_id = f"HVI-{cell}-STAGE0-{ATTEMPT}"
        run_root = ROOT / "runs" / cell / "stage0_retry1"
        log_path = ROOT / "logs" / f"child_{cell}_stage0_retry1.log"
        command = list(base["command"])
        command[command.index("--run-root") + 1] = str(run_root)
        job = {
            **base,
            "job_id": job_id,
            "output_root": str(run_root),
            "log_path": str(log_path),
            "command": command,
            "protocol_id": "HISTORY_VIDEO_INFORMATION_MATRIX_V1_STAGE0_RETRY1",
            "max_oom_requeues": 0,
        }
        path = ROOT / "configs/jobs" / f"{job_id}.json"
        if path.exists() or run_root.exists():
            raise FileExistsError(f"retry output already exists: {path} or {run_root}")
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.append(job)
    (ROOT / "configs/STAGE0_RETRY1_JOBS.json").write_text(json.dumps({"status": "LOCKED", "attempt": ATTEMPT, "jobs": output, "final_test_access": 0}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "attempt": ATTEMPT, "jobs": len(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
