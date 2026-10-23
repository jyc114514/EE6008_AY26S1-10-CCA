#!/usr/bin/env python3
"""Create V3 watcher jobs without touching CUDA."""

from __future__ import annotations
import os

import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
FEATURE = EXP / "code/extract_v3_features.py"
TRAIN = EXP / "code/train_v3.py"
CELLS = ("S-HIST", "F-POOL-D", "F-TOK-D", "F-TOK-S", "V-TOK-D-A", "V-TOK-S-A")
SEEDS = (20260950, 20260951, 20260952)
HORIZONS = (30, 60)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_job(job_id: str, output: Path, command: list[str], log: Path) -> dict[str, object]:
    return {"job_id": job_id, "protocol_id": "CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1", "output_root": str(output), "command": [str(PYTHON), *command], "resource_estimate": {"estimate_type": "pre_stage0_conservative_peak_mib", "measured_peak_mib": 8192}, "environment": {"PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}, "log_path": str(log), "max_oom_requeues": 1, "final_test_access": 0}


def main() -> int:
    jobs = EXP / "configs/jobs"; jobs.mkdir(parents=True, exist_ok=True)
    feature_job = make_job("V3-FEATURE-STAGE0", EXP/"cache/stage0", [str(FEATURE),"--stage0","--stage0-rows","16"], EXP/"logs/child_V3-FEATURE-STAGE0.log")
    write(jobs/"V3-FEATURE-STAGE0.json", feature_job)
    full_feature_job = make_job("V3-FEATURE-FULL", EXP/"cache", [str(FEATURE)], EXP/"logs/child_V3-FEATURE-FULL.log")
    write(jobs/"V3-FEATURE-FULL.json", full_feature_job)
    supplementary_resume_job = make_job("V3-FEATURE-SUPP-RESUME", EXP/"cache", [str(FEATURE),"--resume","--start-row","20079","--end-row","20143"], EXP/"logs/child_V3-FEATURE-SUPP-RESUME.log")
    write(jobs/"V3-FEATURE-SUPP-RESUME.json", supplementary_resume_job)
    stage0=[]; formal=[]
    for seed in SEEDS:
        for horizon in HORIZONS:
            for cell in CELLS:
                run_id=f"V3-{cell}-H{horizon}-S{seed}"; run_root=EXP/f"runs/{run_id}/formal"
                base=[str(TRAIN),"--cell",cell,"--horizon",str(horizon),"--seed",str(seed),"--run-root",str(run_root),"--batch-size","64","--accumulation","1"]
                st=make_job(run_id+"-STAGE0",EXP/f"runs/{run_id}/stage0",base[:1]+["--stage0","--max-optimizer-steps","4"]+base[1:],EXP/f"logs/child_{run_id}_stage0.log")
                fm=make_job(run_id+"-FORMAL",run_root,base,EXP/f"logs/child_{run_id}_formal.log")
                write(jobs/(run_id+"-STAGE0.json"),st); write(jobs/(run_id+"-FORMAL.json"),fm); stage0.append(st); formal.append(fm)
    write(EXP/"configs/V3_STAGE0_JOBS.json",{"status":"LOCKED","jobs":stage0,"count":len(stage0),"pilot_seed":SEEDS[0],"final_test_access":0})
    write(EXP/"configs/V3_FORMAL_JOBS.json",{"status":"LOCKED_PENDING_STAGE0","jobs":formal,"count":len(formal),"final_test_access":0})
    print(json.dumps({"status":"PASS","stage0":len(stage0),"formal":len(formal),"feature_jobs":3},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
