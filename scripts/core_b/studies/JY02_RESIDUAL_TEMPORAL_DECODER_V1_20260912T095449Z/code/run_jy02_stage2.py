#!/usr/bin/env python3
"""Run the locked 4-config x 3-fresh-seed JY-02 confirmation matrix."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from train_jy02_cell import CELL_SPECS, atomic_json, train_cell


CONFIGS = ["JY02-S-A-D", "JY02-F-A-D", "JY02-S-R-D", "JY02-F-R-D"]
SEEDS = [20260921, 20260922, 20260923]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--run-group", required=True); args = parser.parse_args()
    run_root = args.root / "runs" / args.run_group; run_root.mkdir(parents=True, exist_ok=True)
    status = run_root / "STAGE2_STATUS.json"
    atomic_json(status, {"status": "running", "configs": CONFIGS, "seeds": SEEDS, "expected_runs": 12, "completed_runs": 0, "optimizer_steps": 0, "final_test_access": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    results=[]
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("Stage2 CUDA visibility contract failed")
        for seed in SEEDS:
            for cell in CONFIGS:
                result=train_cell(cell,root=args.root,seed=seed,device=torch.device("cuda:0"),run_group=args.run_group+"/seed"+str(seed))
                results.append(result)
                atomic_json(status,{"status":"running","configs":CONFIGS,"seeds":SEEDS,"expected_runs":12,"completed_runs":len(results),"last_cell":cell,"last_seed":seed,"optimizer_steps":sum(int(item["optimizer_steps"]) for item in results),"final_test_access":0})
        atomic_json(status,{"status":"complete","configs":CONFIGS,"seeds":SEEDS,"expected_runs":12,"completed_runs":len(results),"optimizer_steps":sum(int(item["optimizer_steps"]) for item in results),"final_test_access":0,"results":results})
        atomic_json(run_root/"STAGE2_RUN_COMPLETE.json",{"status":"complete","runs":len(results),"optimizer_steps":sum(int(item["optimizer_steps"]) for item in results),"final_test_access":0})
        print(json.dumps({"status":"complete","runs":len(results)},sort_keys=True,default=str)); return 0
    except Exception as exc:
        atomic_json(status,{"status":"failed","configs":CONFIGS,"seeds":SEEDS,"expected_runs":12,"completed_runs":len(results),"optimizer_steps":sum(int(item.get("optimizer_steps",0)) for item in results),"final_test_access":0,"error":f"{type(exc).__name__}: {exc}"})
        atomic_json(run_root/"STAGE2_RUN_FAILED.json",{"status":"failed","completed_runs":len(results),"error":f"{type(exc).__name__}: {exc}","final_test_access":0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
