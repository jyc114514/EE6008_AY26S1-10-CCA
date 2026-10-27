#!/usr/bin/env python3
"""Wait for all formal cells, then launch all 72 fixed-delay bundles."""

from __future__ import annotations
import os

import json
import subprocess
import time
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1_20260921T092455Z"
PYTHON = PROJECT / "envs/jyc/bin/python"


def write(value):
    (EXP / "logs/ODR_AUTO_CONTINUATION.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    paths = sorted((EXP / "logs").glob("watcher_ODR-*-FORMAL.json")); write({"status": "waiting_formal_18", "formal_count": len(paths), "final_test_access": 0})
    while len(paths) < 18:
        time.sleep(10)
        paths = sorted((EXP / "logs").glob("watcher_ODR-*-FORMAL.json"))
        write({"status": "waiting_formal_18", "formal_count": len(paths), "final_test_access": 0})
    while True:
        states = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.is_file()]; failed = [state for state in states if state.get("status") in {"failed", "oom_exhausted", "telemetry_error", "capacity_timeout", "local_gpu_lease_or_recheck_timeout"} or (state.get("status") == "complete" and state.get("exit_code") != 0)]; complete = [state for state in states if state.get("status") == "complete" and state.get("exit_code") == 0]
        if failed: write({"status": "formal_failed_no_inference", "formal_complete": len(complete), "formal_failed": len(failed), "final_test_access": 0}); return 1
        if len(complete) == 18: break
        time.sleep(30)
    manifest_path = EXP / "configs/ODR_INFERENCE_JOBS.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["status"] = "LOCKED_READY_AFTER_FORMAL"; manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); write({"status": "launching_inference_72", "formal_complete": 18, "final_test_access": 0})
    result = subprocess.run([str(PYTHON), str(EXP / "code/launch_queue.py"), "--pattern", "ODR-*-D*-INFER.json", "--controller", "ODR_INFERENCE"], cwd=str(EXP), check=False); write({"status": "inference_complete" if result.returncode == 0 else "inference_failed", "formal_complete": 18, "inference_returncode": result.returncode, "final_test_access": 0}); return result.returncode


if __name__ == "__main__": raise SystemExit(main())
