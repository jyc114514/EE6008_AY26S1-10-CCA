#!/usr/bin/env python3
"""Sequentially run the locked eight-cell JY-02 Stage1 screen."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from train_jy02_cell import CELL_SPECS, SEED, atomic_json, train_cell


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--run-group", default="stage1"); args = parser.parse_args()
    run_root = args.root / "runs" / args.run_group
    run_root.mkdir(parents=True, exist_ok=True)
    status = run_root / "STAGE1_STATUS.json"
    atomic_json(status, {"status": "running", "stage1_cells": 8, "optimizer_steps": 0, "final_test_access": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    results = []
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Stage1 CUDA visibility contract failed")
        for cell_id in CELL_SPECS:
            result = train_cell(cell_id, root=args.root, seed=SEED, device=torch.device("cuda:0"), run_group=args.run_group)
            results.append(result)
            atomic_json(status, {"status": "running", "completed_cells": len(results), "stage1_cells": 8, "optimizer_steps": sum(int(item["optimizer_steps"]) for item in results), "final_test_access": 0, "last_cell": cell_id})
        atomic_json(status, {"status": "complete", "stage1_cells": 8, "completed_cells": len(results), "optimizer_steps": sum(int(item["optimizer_steps"]) for item in results), "final_test_access": 0, "results": results})
        atomic_json(run_root / "STAGE1_RUN_COMPLETE.json", {"status": "complete", "stage1_cells": 8, "completed_cells": len(results), "optimizer_steps": sum(int(item["optimizer_steps"]) for item in results), "final_test_access": 0})
        print(json.dumps({"status": "complete", "cells": len(results)}, sort_keys=True, default=str)); return 0
    except Exception as exc:
        atomic_json(status, {"status": "failed", "completed_cells": len(results), "stage1_cells": 8, "optimizer_steps": sum(int(item.get("optimizer_steps", 0)) for item in results), "final_test_access": 0, "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(run_root / "STAGE1_RUN_FAILED.json", {"status": "failed", "completed_cells": len(results), "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
