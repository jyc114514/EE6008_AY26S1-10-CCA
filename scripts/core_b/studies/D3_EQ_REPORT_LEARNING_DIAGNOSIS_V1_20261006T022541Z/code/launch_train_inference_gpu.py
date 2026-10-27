#!/usr/bin/env python3
"""Approved-only shared-watcher entrypoint for one GPU2/3 train-inference job."""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
ENV_PYTHON = PROJECT / "envs/jyc/bin/python"
sys.path.insert(0, str(OUT / "code"))
import gpu23_launcher as adapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    job = {
        "job_id": "D3EQ-TRAIN-ONLY-INFERENCE-36-20261006T022541Z",
        "protocol_id": "D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1",
        "output_root": str(OUT / "predictions/train"),
        "log_path": str(OUT / "logs/train_only_inference_gpu.log"),
        "command": [str(ENV_PYTHON), str(OUT / "code/train_only_inference.py"),
                    "--device", "cuda", "--resume", "--threads", "2", "--batch-size", "64"],
        "resource_estimate": {
            "measured_peak_mib": 84,
            "measurement_source": "accepted D3EQ-TS11-D-H60-S20261001 inference sidecar peak_allocated_bytes=87787008, rounded up; same model, batch=64; train states remain CPU-resident",
        },
        "environment": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "job": job,
                          "physical_gpu_allowlist": [2, 3], "training": False,
                          "optimizer_steps": 0, "final_test_access": 0}, indent=2))
        return 0
    watcher = adapter._load_watcher()
    code = adapter.run_registered_job_gpu23(
        job,
        lease_dir=OUT / "leases",
        status_path=OUT / "logs/train_only_inference_watcher.json",
        visible_snapshots_fn=lambda: watcher.snapshot_visible_gpus(nvidia_smi=str(adapter.FILTER_PATH)),
        snapshot_fn=lambda index: watcher.snapshot_gpu(index, nvidia_smi=str(adapter.FILTER_PATH)),
        poll_seconds=1.0,
        max_wait_seconds=30.0,
    )
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
