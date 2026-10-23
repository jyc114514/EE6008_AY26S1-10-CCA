#!/usr/bin/env python3
"""Wait for B3 terminal and build a new formal 8/8 Core B analysis root."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ARCHIVE = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z"
EXP = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
CONTROLLER = EXP / "B3_CONTROLLER_STATUS.json"
SOURCE_CODE = ARCHIVE / "core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z/code"
DIRECTION_SOURCE = ARCHIVE / "core_b/future_directions/CORE_B_DIRECTION_PORTFOLIO_20260910T060958Z/generate_fixed_team_prompts.py"
PYTHON = PROJECT / "envs/jyc/bin/python"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(command: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"utc": now(), "command": command}), flush=True)
    result = subprocess.run(command, cwd=str(PROJECT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"returncode={result.returncode}: {command}")


def wait_for_terminal() -> dict:
    last = None
    while True:
        data = json.loads(CONTROLLER.read_text(encoding="utf-8"))
        signature = (data.get("status"), tuple(sorted((x.get("condition"), x.get("gpu")) for x in data.get("active_wave", []))), len(data.get("completed", [])))
        if signature != last:
            print(json.dumps({"utc": now(), "controller_status": data.get("status"), "active_wave": data.get("active_wave", []), "completed_count": len(data.get("completed", [])), "final_test_access": data.get("final_test_access", 0)}), flush=True)
            last = signature
        if data.get("status") in {"failed", "blocked"}:
            raise RuntimeError(f"B3 controller terminal failure: {data.get('status')}")
        if data.get("status") in {"complete", "complete_pending_analysis", "terminal"}:
            active = data.get("active_wave", [])
            if not active:
                return data
            if data.get("status") == "complete_pending_analysis":
                unresolved = []
                for item in active:
                    condition = item.get("condition", "")
                    paths = [EXP / "runs/corrected" / condition / "RUN_STATUS.json", EXP / "runs" / condition / "RUN_STATUS.json"]
                    status = next((json.loads(path.read_text(encoding="utf-8")) for path in paths if path.is_file()), {})
                    if status.get("status") != "complete" or status.get("exit_code") != 0:
                        unresolved.append(condition)
                if not unresolved:
                    return data
        time.sleep(60)


def main() -> int:
    controller = wait_for_terminal()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    formal = ARCHIVE / f"core_b/analysis/CORE_B_UNIFIED_ANALYSIS_{stamp}_FORMAL_B3_8OF8"
    direction = ARCHIVE / f"core_b/future_directions/CORE_B_DIRECTION_PORTFOLIO_{stamp}_FORMAL_B3_8OF8"
    for sub in ("data", "code", "figures", "figure_data", "conclusions", "evidence", "qa", "logs"):
        (formal / sub).mkdir(parents=True, exist_ok=False)
    (direction / "team_prompts").mkdir(parents=True, exist_ok=False)
    for path in SOURCE_CODE.glob("*.py"):
        shutil.copy2(path, formal / "code" / path.name)
    shutil.copy2(DIRECTION_SOURCE, direction / "generate_fixed_team_prompts.py")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""; env["CORE_B_ANALYSIS_OUT"] = str(formal)
    harvest = formal / "code/analyze_core_b_harvest.py"
    run([str(PYTHON), str(harvest)], env)
    run([str(PYTHON), str(formal / "code/make_core_b_figures.py")], env)
    run([str(PYTHON), str(formal / "code/qa_core_b_figures.py")], env)
    run([str(PYTHON), str(formal / "code/qa_core_b_analysis.py")], env)
    direction_env = env.copy(); direction_env["CORE_B_HARVEST_ROOT"] = str(formal); direction_env["CORE_B_DIRECTION_ROOT"] = str(direction)
    run([str(PYTHON), str(direction / "generate_fixed_team_prompts.py")], direction_env)
    (formal / "FORMAL_UPDATE_STATUS.json").write_text(json.dumps({"status": "formal_analysis_and_direction_design_complete_pending_manual_figure_review", "created_utc": now(), "controller": controller, "formal_root": formal, "direction_root": direction, "partial_snapshot_preserved": True, "new_gpu_jobs_started": False, "final_test_access": 0}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete_pending_manual_figure_review", "formal_root": str(formal), "direction_root": str(direction)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
