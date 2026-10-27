#!/usr/bin/env python3
"""Build isolated pilot Stage0 retry jobs after the batch-unpack bug."""

from __future__ import annotations
import os

import json
from pathlib import Path


EXP=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))


def main()->int:
    jobs=[]
    for path in sorted((EXP/"configs/jobs").glob("V3-*-S20260950-STAGE0.json")):
        old=json.loads(path.read_text()); run_root=EXP/f"runs/{path.stem.replace('-STAGE0','')}/stage0_retry1"; command=list(old["command"]); command[command.index("--run-root")+1]=str(run_root); value=dict(old); value["job_id"]+="-RETRY1"; value["output_root"]=str(run_root); value["command"]=command; value["log_path"]=str(EXP/f"logs/child_{value['job_id']}.log"); out=EXP/"configs/jobs"/(value["job_id"]+".json"); out.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); jobs.append(value)
    (EXP/"configs/V3_PILOT_STAGE0_RETRY1_JOBS.json").write_text(json.dumps({"status":"LOCKED","jobs":jobs,"count":len(jobs),"reason":"original Stage0 failed on raw batch unpack; retry uses isolated run roots and corrected train loop","final_test_access":0},ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","jobs":len(jobs)},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
