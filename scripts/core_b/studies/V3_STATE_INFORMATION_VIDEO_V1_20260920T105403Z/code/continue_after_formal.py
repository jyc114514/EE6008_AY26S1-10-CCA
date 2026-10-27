#!/usr/bin/env python3
"""Wait for all formal cells, then launch all fixed-last inference cells."""

from __future__ import annotations
import os

import json
import subprocess
import time
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
LAUNCH = EXP / "code/launch_queue.py"


def write_status(value: dict[str, object]) -> None:
    (EXP / "logs/SIV_AUTO_CONTINUATION.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    formal_paths = sorted((EXP / "logs").glob("watcher_SIV-*-FORMAL.json"))
    inference_manifest = EXP / "configs/SIV_INFERENCE_JOBS.json"
    if len(formal_paths) != 54 or not inference_manifest.is_file():
        raise RuntimeError("formal/inference manifests are incomplete")
    write_status({"status": "waiting_formal_54", "formal_count": len(formal_paths), "final_test_access": 0})
    while True:
        states = [json.loads(path.read_text(encoding="utf-8")) for path in formal_paths if path.is_file()]
        failures = [state for state in states if state.get("status") in {"failed", "oom_exhausted", "telemetry_error", "capacity_timeout", "local_gpu_lease_or_recheck_timeout"} or (state.get("status") == "complete" and state.get("exit_code") != 0)]
        complete = [state for state in states if state.get("status") == "complete" and state.get("exit_code") == 0]
        if failures:
            write_status({"status": "formal_failed_no_inference", "formal_complete": len(complete), "formal_failed": len(failures), "final_test_access": 0})
            return 1
        if len(complete) == 54:
            break
        time.sleep(30)
    manifest = json.loads(inference_manifest.read_text(encoding="utf-8"))
    manifest["status"] = "LOCKED_READY_AFTER_FORMAL"
    inference_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_status({"status": "launching_inference_54", "formal_complete": 54, "final_test_access": 0})
    result = subprocess.run([str(PYTHON), str(LAUNCH), "--pattern", "SIV-*-INFER.json", "--controller", "SIV_INFERENCE"], cwd=str(EXP), check=False)
    write_status({"status": "inference_complete" if result.returncode == 0 else "inference_failed", "formal_complete": 54, "inference_returncode": result.returncode, "final_test_access": 0})
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
