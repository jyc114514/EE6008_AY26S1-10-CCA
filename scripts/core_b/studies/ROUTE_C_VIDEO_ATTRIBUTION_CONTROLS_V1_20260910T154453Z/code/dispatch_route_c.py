#!/usr/bin/env python3
"""Sequential Route C dispatcher with dynamic GPU1/2/3 selection.

The dispatcher deliberately runs one raw-video child at a time because the
project's recorded raw-video I/O capacity is one.  The task-local watcher
still enforces the two-child and per-GPU leases, plus the idle gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
WATCHER = ROOT / "code/task_local_gpu_watcher.py"
RUNNER = ROOT / "code/run_counterfactual_inference.py"
PYTHON = PROJECT / "envs/jyc/bin/python"
REGISTRY = ROOT / "ROUTE_C_CHECKPOINT_REGISTRY.csv"
TIMELINE = ROOT / "ROUTE_C_RESOURCE_TIMELINE.csv"
LEASE_DIR = ROOT / "leases"
GPU_CANDIDATES = (1, 2, 3)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot(gpu: int) -> dict[str, object]:
    query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
    return {"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_used_mib": int(fields[3]), "memory_total_mib": int(fields[4]), "utilization_percent": int(fields[5]), "utc": now()}


def append_timeline(phase: str, status: str, snap: dict[str, object] | None, gpu: int | str = "", raw_token: str = "unacquired") -> None:
    new = not TIMELINE.exists() or TIMELINE.stat().st_size == 0
    with TIMELINE.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new: writer.writerow(["timestamp_utc", "phase", "status", "physical_gpu", "uuid", "utilization_percent", "memory_used_mib", "memory_total_mib", "raw_video_io_token"])
        writer.writerow([now(), phase, status, gpu, (snap or {}).get("uuid", ""), (snap or {}).get("utilization_percent", ""), (snap or {}).get("memory_used_mib", ""), (snap or {}).get("memory_total_mib", ""), raw_token])


def choose_gpu() -> tuple[int, dict[str, object]]:
    while True:
        for gpu in GPU_CANDIDATES:
            snap = snapshot(gpu)
            append_timeline("selection", "idle_candidate" if snap["utilization_percent"] <= 5 and snap["memory_used_mib"] <= 512 else "busy", snap, gpu)
            if snap["utilization_percent"] <= 5 and snap["memory_used_mib"] <= 512:
                return gpu, snap
        append_timeline("selection", "waiting_all_busy", None)
        time.sleep(30)


def job_for(model: str, gpu: int, row: dict[str, str]) -> tuple[dict[str, object], Path, Path]:
    branch = "fusion" if "FUS" in model else "video_only"
    adaptation = "lora" if model.endswith("-L") else "frozen"
    job_id = f"route_c_{model.lower().replace('-', '_')}_{gpu}_20260910"
    output = ROOT / "predictions" / model
    log_path = ROOT / "logs" / f"{model}_gpu{gpu}.log"
    status_path = ROOT / "logs" / f"{model}_watcher.json"
    command = [str(PYTHON), str(RUNNER), "--model", model, "--branch", branch, "--adaptation", adaptation, "--checkpoint-path", row["checkpoint_path"], "--canonical-prediction", row["canonical_prediction_path"], "--physical-gpu", str(gpu), "--output-root", str(output), "--batch-size", "2"]
    job = {"job_id": job_id, "physical_gpu": gpu, "protocol_id": "core_b_route_c_video_attribution_controls_v1_20260910", "output_root": str(output), "log_path": str(log_path), "command": command, "resource_estimate": {"raw_video_io": True, "gpu_memory_mib": 9000, "precision": "FP32 frozen / BF16 LoRA", "optimizer_steps": 0, "final_test_access": 0, "max_project_gpu_children": 2, "raw_video_io_capacity": 1}, "environment": {"PYTHONPATH": f"{REPO / 'src'}:{PROJECT / 'repo/third_party/vjepa2'}", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "ROUTE_C_PROTOCOL": "core_b_route_c_video_attribution_controls_v1_20260910"}}
    job_path = ROOT / "inputs" / f"job_{model}_gpu{gpu}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    return job, job_path, status_path


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--start-index", type=int, default=0); parser.add_argument("--poll-seconds", type=float, default=30.0); args = parser.parse_args()
    with REGISTRY.open(newline="") as handle: registry = {row["model"]: row for row in csv.DictReader(handle)}
    models = ["B3-FUS-D-F", "B3-FUS-D-L", "B3-V-D-F", "B3-V-D-L"]
    if not 0 <= args.start_index < len(models): raise ValueError("start-index outside model schedule")
    for model in models[args.start_index:]:
        gpu, selected = choose_gpu()
        job, job_path, status_path = job_for(model, gpu, registry[model])
        append_timeline("launch", "selected", selected, gpu, "acquire_after_idle")
        print(json.dumps({"event": "launching_watcher", "model": model, "gpu": gpu, "job": str(job_path)}, sort_keys=True), flush=True)
        result = subprocess.run([str(PYTHON), str(WATCHER), "--job-json", str(job_path), "--status-path", str(status_path), "--lease-dir", str(LEASE_DIR), "--cwd", str(REPO), "--poll-seconds", str(args.poll_seconds)], cwd=REPO)
        append_timeline("complete", "child_success" if result.returncode == 0 else "child_failure", selected, gpu, "released")
        print(json.dumps({"event": "child_finished", "model": model, "gpu": gpu, "returncode": result.returncode}, sort_keys=True), flush=True)
        if result.returncode != 0:
            (ROOT / "ROUTE_C_DISPATCH_STATUS.json").write_text(json.dumps({"status": "blocked_child_failure", "failed_model": model, "gpu": gpu, "returncode": result.returncode, "completed_utc": now()}, indent=2) + "\n")
            return result.returncode
    (ROOT / "ROUTE_C_DISPATCH_STATUS.json").write_text(json.dumps({"status": "all_four_model_lanes_complete", "models": models, "raw_video_children_concurrent": 1, "completed_utc": now()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
