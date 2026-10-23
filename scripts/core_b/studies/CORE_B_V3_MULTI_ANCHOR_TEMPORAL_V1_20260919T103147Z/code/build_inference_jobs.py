#!/usr/bin/env python3
"""Build 36 fixed-last development inference watcher jobs."""

from __future__ import annotations
import os
import json
from pathlib import Path

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), '')); REPO=PROJECT/"repo/ee6008-humanoid-vjepa"; EXP=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"; PYTHON=PROJECT/"envs/jyc/bin/python"; INFER=EXP/"code/infer_v3.py"


def main()->int:
 jobs=[]
 for seed in (20260950,20260951,20260952):
  for horizon in (30,60):
   for cell in ("S-HIST","F-POOL-D","F-TOK-D","F-TOK-S","V-TOK-D-A","V-TOK-S-A"):
    run_id=f"V3-{cell}-H{horizon}-S{seed}"; value={"job_id":run_id+"-INFER","protocol_id":"CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_INFERENCE","output_root":str(EXP/f"predictions/{run_id}"),"command":[str(PYTHON),str(INFER),"--cell",cell,"--horizon",str(horizon),"--seed",str(seed),"--run-id",run_id],"resource_estimate":{"estimate_type":"pilot_formal_measured_peak_reserved_mib","measured_peak_mib":8192},"environment":{"PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},"log_path":str(EXP/f"logs/child_{run_id}_infer.log"),"max_oom_requeues":1,"final_test_access":0}; path=EXP/"configs/jobs"/(run_id+"-INFER.json"); path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); jobs.append(value)
 (EXP/"configs/V3_INFERENCE_JOBS.json").write_text(json.dumps({"status":"LOCKED","jobs":jobs,"count":len(jobs),"final_test_access":0},ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"status":"PASS","jobs":len(jobs)},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
