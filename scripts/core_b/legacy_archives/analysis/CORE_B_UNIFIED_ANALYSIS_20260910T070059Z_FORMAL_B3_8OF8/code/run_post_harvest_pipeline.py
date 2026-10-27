#!/usr/bin/env python3
"""Persistently wait for the existing B3 controller, then run CPU-only harvest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ANALYSIS = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z'))
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = PROJECT / "envs/jyc/bin/python"
CONTROLLER = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/B3_CONTROLLER_STATUS.json"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(script: Path) -> None:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps({"utc": now(), "run": str(script)}), flush=True)
    result = subprocess.run([str(PYTHON), str(script)], cwd=str(PROJECT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"pipeline step failed {script} returncode={result.returncode}")


def main() -> int:
    harvest = ANALYSIS / "code/analyze_core_b_harvest.py"
    figures = ANALYSIS / "code/make_core_b_figures.py"
    figure_qa = ANALYSIS / "code/qa_core_b_figures.py"
    analysis_qa = ANALYSIS / "code/qa_core_b_analysis.py"
    directions = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/future_directions/CORE_B_DIRECTION_PORTFOLIO_20260909T163929Z/generate_direction_portfolio.py"
    while True:
        data = json.loads(CONTROLLER.read_text(encoding="utf-8"))
        status = data.get("status")
        print(json.dumps({"utc": now(), "controller_status": status, "active_wave": data.get("active_wave", []), "completed": len(data.get("completed", []))}), flush=True)
        if status in {"failed", "blocked"}:
            raise RuntimeError(f"pre-existing B3 controller terminal failure: {status}")
        if status in {"complete", "complete_pending_analysis", "terminal"} and not data.get("active_wave"):
            break
        time.sleep(60)
    while True:
        try:
            run(harvest)
            break
        except Exception as exc:
            print(json.dumps({"utc": now(), "harvest_retry": str(exc)}), flush=True)
            time.sleep(60)
    run(figures)
    run(figure_qa)
    run(analysis_qa)
    run(directions)
    print(json.dumps({"utc": now(), "status": "complete", "harvest": str(ANALYSIS), "directions": str(directions.parent)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
