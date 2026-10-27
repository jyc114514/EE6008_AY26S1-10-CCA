#!/usr/bin/env python3
"""Persistently run the remaining dense B3 cells in two bounded waves."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z'))
WATCHER = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/coordination/CORE_A_CORE_B_POST_OUTAGE_20260907T055509Z/code/gpu_dynamic_lease_watcher.py'))
PYTHON = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python'))
CORRECTED_RAW = {"B3-V-D-L", "B3-V-S-L", "B3-FUS-D-L", "B3-FUS-S-L"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_status(value: dict[str, object]) -> None:
    path = EXP / "B3_CONTROLLER_STATUS.json"
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def launch(condition: str, gpu: int) -> subprocess.Popen[str]:
    suffix = "_corrected" if condition in CORRECTED_RAW else ""
    job = EXP / "jobs" / f"{condition}{suffix}_gpu{gpu}.json"
    status = EXP / "status" / f"{condition}_gpu{gpu}.json"
    command = [str(PYTHON),str(WATCHER),"--job-json",str(job),"--lease-dir",str(EXP/"leases"),"--status-path",str(status),"--poll-seconds","10","--idle-samples","2","--allow-colocation","--min-free-memory-mib","14000"]
    return subprocess.Popen(command, cwd=str(EXP), text=True)


def wait_wave(items: list[tuple[str,int]], status: dict[str,object]) -> bool:
    procs=[(condition,gpu,launch(condition,gpu)) for condition,gpu in items]
    status["active_wave"]=[{"condition":c,"gpu":g} for c,g,_ in procs]; status["updated_utc"]=now(); write_status(status)
    ok=True
    for condition,gpu,proc in procs:
        code=proc.wait(); status.setdefault("completed",[]).append({"condition":condition,"gpu":gpu,"return_code":code,"completed_utc":now()}); ok = ok and code == 0; status["updated_utc"]=now(); write_status(status)
    return ok


def main() -> int:
    status={"status":"running","started_utc":now(),"strategy":"dense_0.125s_32_pixel_frames","final_test_access":0}
    write_status(status)
    if not wait_wave([("B3-V-D-L",3),("B3-FUS-D-L",2)],status):
        status["status"]="failed_wave1"; write_status(status); return 1
    status["status"]="wave2_running"; write_status(status)
    if not wait_wave([("B3-V-S-L",3),("B3-FUS-S-F",2)],status):
        status["status"]="failed_wave2"; write_status(status); return 1
    status["status"]="wave3_running"; write_status(status)
    if not wait_wave([("B3-FUS-S-L",3)],status):
        status["status"]="failed_wave3"; write_status(status); return 1
    status["status"]="complete_pending_analysis"; status["completed_utc"]=now(); write_status(status); return 0


if __name__ == "__main__":
    raise SystemExit(main())
