#!/usr/bin/env python3
"""Persistent post-B3 controller for corrected pixel-level B2.

It waits for the already-running dense B3 child, then uses the task-local
watcher to serialize pixel extraction and the four new sparse pooled heads on
GPU3. It never signals or edits external processes.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
COORD = REPO / "reports/coordination/CORE_A_CORE_B_POST_OUTAGE_20260907T055509Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = COORD / "code/gpu_dynamic_lease_watcher.py"
GPU_UUID = "GPU-b9c07d60-9a04-0693-af83-6976ba6d4dfb"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def wait_for_b3(status_path: Path, status: dict[str, object]) -> None:
    while True:
        value = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        state = value.get("status")
        status["b3_last_status"] = state
        status["updated_utc"] = utc_now()
        atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
        if state in {"complete", "failed"}:
            status["b3_terminal_status"] = state
            return
        time.sleep(30)


def run_watcher(job_path: Path, status_path: Path) -> int:
    command = [
        str(PYTHON), str(WATCHER), "--job-json", str(job_path),
        "--lease-dir", str(EXP / "leases"), "--status-path", str(status_path),
        "--poll-seconds", "10", "--idle-samples", "2", "--allow-colocation",
        "--min-free-memory-mib", "14000",
    ]
    return subprocess.run(command, cwd=str(REPO), check=False).returncode


def make_job(job_id: str, command: list[str], output_root: Path, log_path: Path, *, raw: bool, gpu_mib: int, host_mib: int, protocol_id: str) -> dict[str, object]:
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "3",
        "DUAL_TRACK_ROLE": "core_b_true_b2",
        "LD_LIBRARY_PATH": str(PROJECT / "envs/jyc/lib") + ":/usr/local/cuda-12.8/lib64",
        "PYTHONPATH": ":".join([str(COORD / "code/locked_source/src"), str(COORD / "code/core_a"), str(COORD / "code/core_a/analysis_scripts"), str(REPO / "src")]),
    }
    return {"job_id":job_id,"physical_gpu":3,"gpu_uuid":GPU_UUID,"protocol_id":protocol_id,"output_root":str(output_root),"log_path":str(log_path),"command":command,"environment":env,"resource_estimate":{"physical_batch":8,"gradient_accumulation":4,"nominal_effective_batch":32,"precision":"BF16","raw_video_io":raw,"gpu_memory_mib":gpu_mib,"host_ram_reservation_mib":host_mib}}


def create_condition_root(name: str, input_archive: Path) -> Path:
    root = EXP / "inputs" / name
    root.mkdir(parents=True, exist_ok=True)
    links = {
        "CORE_B_FULL_INPUTS_1p0s.npz": input_archive,
        "CORE_B_FULL_DATA_MANIFEST.json": REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.json",
        "CORE_B_TOKEN_PATHS_1p0s.json": REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_TOKEN_PATHS_1p0s.json",
    }
    for name_value, target in links.items():
        link = root / name_value
        if not link.exists():
            link.symlink_to(target)
    return root


def main() -> int:
    status: dict[str, object] = {"status":"waiting_for_dense_b3","started_utc":utc_now(),"physical_gpu":3,"final_test_access":0}
    atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
    wait_for_b3(EXP / "runs/B3-V-D-L/RUN_STATUS.json", status)
    status["status"] = "pixel_extraction_queued"
    atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
    extract_job = EXP / "jobs/TRUE-B2-PIXEL-EXTRACT_gpu3.json"
    extract_output = EXP / "true_pixel_sampling"
    extract_command = [str(PYTHON), str(EXP / "protocol/prepare_true_b2_pixel.py"), "--project-root", str(PROJECT), "--repo-root", str(REPO), "--canonical-input-root", str(REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z"), "--manifest-csv", str(REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.csv"), "--output-root", str(extract_output), "--device", "cuda"]
    atomic_json(extract_job, make_job("core_b_true_b2_pixel_extract_20260908_gpu3", extract_command, extract_output, EXP / "logs/TRUE-B2-PIXEL-EXTRACT_gpu3.log", raw=True, gpu_mib=22000, host_mib=52000, protocol_id="core_b_b2_true_pixel_sampling_20260908"))
    extract_status = EXP / "status/TRUE-B2-PIXEL-EXTRACT_gpu3.json"
    result = run_watcher(extract_job, extract_status)
    status["pixel_extraction_return_code"] = result
    if result != 0:
        status["status"] = "blocked_pixel_extraction_failed"
        atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
        return result
    status["status"] = "sparse_heads_queued"
    atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
    jobs: list[Path] = []
    for interval in ("0250", "0500"):
        archive = extract_output / f"TRUE_B2_INPUTS_{interval}.npz"
        for branch, lane in (("V", "B-D-P-F"), ("FUS", "B-D-FUS-P-F")):
            condition = f"TRUE-B2-{branch}-{interval}"
            input_root = create_condition_root(condition, archive)
            run_root = EXP / "runs" / condition
            job_path = EXP / "jobs" / f"{condition}_gpu3.json"
            command = [str(PYTHON), str(EXP / "protocol/run_b2_pooled.py"), "--root", str(input_root), "--project-root", str(PROJECT), "--feasibility-root", str(REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z"), "--fix-root", str(COORD / "code/core_a"), "--source-root", str(REPO), "--run-root", str(run_root), "--lane", lane, "--condition-id", condition, "--protocol-id", "core_b_b2_true_pixel_sampling_20260908_batch8_accum4", "--physical-gpu", "3", "--seed", "20260903"]
            atomic_json(job_path, make_job(f"core_b_{condition.lower().replace('-', '_')}_20260908_gpu3", command, run_root, EXP / "logs" / f"{condition}_gpu3.log", raw=False, gpu_mib=4000, host_mib=4000, protocol_id="core_b_b2_true_pixel_sampling_20260908_batch8_accum4"))
            jobs.append(job_path)
    for job_path in jobs:
        condition = job_path.stem.replace("_gpu3", "")
        result = run_watcher(job_path, EXP / "status" / f"{condition}_gpu3.json")
        status.setdefault("sparse_head_results", []).append({"condition":condition,"return_code":result,"completed_utc":utc_now()})
        atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
        if result != 0:
            status["status"] = "blocked_sparse_head_failed"
            atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
            return result
    status["status"] = "true_b2_training_complete_pending_analysis"
    status["completed_utc"] = utc_now()
    atomic_json(EXP / "TRUE_B2_CONTROLLER_STATUS.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
