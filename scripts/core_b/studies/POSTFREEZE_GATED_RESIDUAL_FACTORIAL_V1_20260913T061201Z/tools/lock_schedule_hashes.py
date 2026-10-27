from __future__ import annotations
import os

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_TOOLS = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/tools'))
sys.path.insert(0, str(REPO_TOOLS))
from gpu_lease_watcher import command_hash  # noqa: E402


def main() -> None:
    schedule_path = ROOT / "RUN_SCHEDULE.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    for row in schedule["rows"]:
        job = json.loads(Path(row["job_path"]).read_text(encoding="utf-8"))
        row["command_hash"] = command_hash(job["command"])
        row.pop("command_hash_pending", None)
    schedule["status"] = "LOCKED_BEFORE_RESULTS"
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "RUN_SCHEDULE.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("wave,physical_gpu,cell_id,job_id,job_path,command_hash\n")
        for row in schedule["rows"]:
            handle.write(f"{row['wave']},{row['physical_gpu']},{row['cell_id']},{row['job_id']},{row['job_path']},{row['command_hash']}\n")
    print(json.dumps({"status": "PASS", "rows": len(schedule["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
