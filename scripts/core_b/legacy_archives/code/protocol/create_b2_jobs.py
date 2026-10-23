#!/usr/bin/env python3
"""Create immutable task-local B2 watcher job records."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = ROOT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
COORD = REPO / "reports/coordination/CORE_A_CORE_B_POST_OUTAGE_20260907T055509Z"
PYTHON = ROOT / "envs/jyc/bin/python"
GPU_UUID = "GPU-ceb12ae6-6ff4-4685-6309-544becf58fe6"
COMMON_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1",
    "DUAL_TRACK_ROLE": "core_b_b2",
    "LD_LIBRARY_PATH": str(ROOT / "envs/jyc/lib") + ":/usr/local/cuda-12.8/lib64",
    "PYTHONPATH": ":".join(
        [
            str(COORD / "code/locked_source/src"),
            str(COORD / "code/core_a"),
            str(COORD / "code/core_a/analysis_scripts"),
            str(REPO / "src"),
        ]
    ),
}


def write_once(path: Path, value: object) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def make_job(condition: str, lane: str, interval: str) -> dict[str, object]:
    input_root = EXP / "inputs" / condition
    run_root = EXP / "runs" / condition
    command = [
        str(PYTHON),
        str(EXP / "protocol/run_b2_pooled.py"),
        "--root", str(input_root),
        "--project-root", str(ROOT),
        "--feasibility-root", str(REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z"),
        "--fix-root", str(COORD / "code/core_a"),
        "--source-root", str(REPO),
        "--run-root", str(run_root),
        "--lane", lane,
        "--condition-id", condition,
        "--physical-gpu", "1",
        "--seed", "20260903",
    ]
    return {
        "job_id": f"core_b_{condition.lower().replace('-', '_')}_20260908_gpu1",
        "physical_gpu": 1,
        "gpu_uuid": GPU_UUID,
        "protocol_id": "core_b_b2_pooled_sampling_20260908_batch8_accum4",
        "output_root": str(run_root),
        "log_path": str(EXP / "logs" / f"{condition}_gpu1.log"),
        "command": command,
        "environment": COMMON_ENV,
        "resource_estimate": {
            "physical_batch": 8,
            "gradient_accumulation": 4,
            "nominal_effective_batch": 32,
            "precision": "BF16",
            "raw_video_io": False,
            "host_ram_reservation_mib": 4000,
            "gpu_memory_mib": 4000,
            "nominal_interval_seconds": float(interval),
        },
    }


def main() -> int:
    specs = [
        ("B2-V-0125", "B-D-P-F", "0.125"),
        ("B2-V-0250", "B-D-P-F", "0.250"),
        ("B2-V-0500", "B-D-P-F", "0.500"),
        ("B2-FUS-0125", "B-D-FUS-P-F", "0.125"),
        ("B2-FUS-0250", "B-D-FUS-P-F", "0.250"),
        ("B2-FUS-0500", "B-D-FUS-P-F", "0.500"),
    ]
    for condition, lane, interval in specs:
        write_once(EXP / "jobs" / f"{condition}_gpu1.json", make_job(condition, lane, interval))
    print(json.dumps({"status": "pass", "jobs": len(specs), "job_root": str(EXP / "jobs")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
