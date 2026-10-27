#!/usr/bin/env python3
"""Launch a V3 job subset through the shared dynamic GPU watcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO=PROJECT/"repo/ee6008-humanoid-vjepa"
EXP=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
PYTHON=PROJECT/"envs/jyc/bin/python"
WATCHER=REPO/"tools/gpu_lease_watcher.py"
LEASE=EXP/"locks/gpu_lease"


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--pattern",required=True); parser.add_argument("--controller",required=True); parser.add_argument("--poll-seconds",type=float,default=15.0); args=parser.parse_args()
    paths=sorted((EXP/"configs/jobs").glob(args.pattern))
    if not paths: raise RuntimeError(f"no jobs for {args.pattern}")
    processes=[]; start=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    for path in paths:
        job=json.loads(path.read_text()); job_id=job["job_id"]; status=EXP/f"logs/watcher_{job_id}.json"; stdout=EXP/f"logs/watcher_{job_id}.stdout.log"; stdout.parent.mkdir(parents=True,exist_ok=True); handle=stdout.open("w",encoding="utf-8")
        proc=subprocess.Popen([str(PYTHON),str(WATCHER),"--job-json",str(path),"--lease-dir",str(LEASE),"--status-path",str(status),"--poll-seconds",str(args.poll_seconds),"--max-wait-seconds","0","--max-oom-requeues","1"],cwd=str(PROJECT),stdout=handle,stderr=subprocess.STDOUT,text=True); handle.close(); processes.append((job_id,proc,status))
    results=[]
    for job_id,proc,status in processes:
        code=proc.wait(); value=json.loads(status.read_text()) if status.is_file() else {}; results.append({"job_id":job_id,"watcher_pid":proc.pid,"returncode":code,"status":value})
    payload={"status":"PASS" if all(r["returncode"]==0 for r in results) else "FAIL","controller":args.controller,"started_utc":start,"completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"controller_pid":os.getpid(),"results":results,"dynamic_all_visible":True,"max_project_gpu_children":2,"other_processes_signaled":False,"final_test_access":0}
    (EXP/f"logs/{args.controller}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"status":payload["status"],"jobs":len(results),"returncodes":[r["returncode"] for r in results]},sort_keys=True)); return 0 if payload["status"]=="PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
