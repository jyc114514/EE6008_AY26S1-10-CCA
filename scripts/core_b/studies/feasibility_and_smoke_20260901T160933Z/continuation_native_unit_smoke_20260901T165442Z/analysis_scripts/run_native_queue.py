"""Run the locked bounded Core B smoke lanes serially on an authorized fallback GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LANES = [
    ("video_only", "1p0s"),
    ("state_action_history_only", "1p0s"),
    ("video_only", "0p5s"),
    ("state_action_history_only", "0p5s"),
    ("gated_fusion", "1p0s"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete_marker(run_dir: Path) -> dict[str, object] | None:
    marker = run_dir / "RUN_COMPLETE.json"
    if not marker.exists():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    script = root / "analysis_scripts" / "native_smoke.py"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue_path = root / "NATIVE_SMOKE_QUEUE_STATUS.json"
    state: dict[str, object] = {
        "status": "running",
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": str(args.physical_gpu),
        "lanes": [],
        "started_utc": utc_now(),
        "final_test_accessed": 0,
        "full_benchmark_started": False,
    }
    write_json(queue_path, state)
    for index, (lane, horizon) in enumerate(LANES, start=1):
        run_dir = root / "runs" / f"{lane}_{horizon}"
        if run_dir.exists() and any(run_dir.iterdir()):
            state["status"] = "blocked_existing_run_directory"
            state["blocked_at"] = {"lane": lane, "horizon": horizon}
            write_json(queue_path, state)
            return 2
        log_path = logs / f"{lane}_{horizon}.log"
        command = [sys.executable, str(script), "--root", str(root), "--physical-gpu", str(args.physical_gpu), "--lane", lane, "--horizon", horizon]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        entry = {"index": index, "lane": lane, "horizon": horizon, "exit_code": result.returncode, "log": str(log_path), "completed_utc": utc_now()}
        marker = complete_marker(run_dir)
        entry["completion_marker_status"] = marker.get("status") if marker else None
        state["lanes"].append(entry)
        write_json(queue_path, state)
        if result.returncode != 0 or marker is None or marker.get("status") != "complete" or marker.get("exit_code") != 0:
            state["status"] = "stopped_lane_failure"
            state["stopped_at"] = {"lane": lane, "horizon": horizon}
            write_json(queue_path, state)
            return result.returncode or 1
        if index == 2:
            # The primary gated-fusion lane is permitted only after both 1.0s
            # unimodal lanes have complete markers. Engineering pass includes
            # either signal-present or no-clear-signal; the latter is retained.
            pass
    state["status"] = "complete"
    state["completed_utc"] = utc_now()
    write_json(queue_path, state)
    print(json.dumps({"status": state["status"], "lanes": len(state["lanes"]), "physical_gpu": args.physical_gpu}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
