#!/usr/bin/env python3
"""Approved host-only aggregate GPU resource preflight; no process table is queried."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--snapshot-stdin", action="store_true",
                        help="consume the exact aggregate nvidia-smi CSV from an already-approved shell pipeline")
    args = parser.parse_args()
    audit = args.audit_root.resolve()
    query = "index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,pstate"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if args.snapshot_stdin:
        snapshot_text = sys.stdin.read()
    else:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        snapshot_text = result.stdout
    gpus = []
    for row in csv.reader(snapshot_text.splitlines()):
        fields = [x.strip() for x in row]
        if len(fields) != 9:
            raise RuntimeError(f"aggregate GPU resource row has unexpected field count: {len(fields)}")
        index, uuid, name, driver, used, total, util, temp, pstate = fields
        used_i, total_i = int(float(used)), int(float(total))
        gpus.append({
            "index": int(index), "uuid": uuid, "name": name, "driver_version": driver,
            "memory_used_mib": used_i, "memory_total_mib": total_i,
            "memory_free_mib": max(total_i - used_i, 0),
            "utilization_gpu_percent": int(float(util)),
            "temperature_c": None if temp in {"[Not Supported]", "N/A"} else int(float(temp)),
            "performance_state": pstate
        })
    if not gpus:
        raise RuntimeError("nvidia-smi returned no visible GPUs")
    pre_registered_peak_mib = 244
    safety_margin_mib = max(512, int((pre_registered_peak_mib * 0.10) + 0.999999))
    required_free_mib = pre_registered_peak_mib + safety_margin_mib
    schedulers = {name: bool(os.environ.get(name)) for name in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID")}
    if any(schedulers.values()):
        host_context = "scheduler_managed"
    elif os.isatty(0) or os.isatty(1):
        host_context = "interactive_or_login_role_unresolved"
    else:
        host_context = "noninteractive_without_scheduler_marker; login_vs_compute_role_unknown"
    receipt = {
        "status": "PASS_AGGREGATE_HOST_PREFLIGHT",
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "host_context": host_context,
        "scheduler_markers_present": schedulers,
        "tty": {"stdin": os.isatty(0), "stdout": os.isatty(1)},
        "command": command,
        "snapshot_source": "approved nvidia-smi pipeline stdin" if args.snapshot_stdin else "direct host nvidia-smi subprocess",
        "query_scope": "aggregate index/uuid/name/driver/memory/utilization/temperature/pstate only; no compute-process PID/account/command fields",
        "visible_gpu_count": len(gpus),
        "gpus": gpus,
        "stage0_admission_reference": {
            "pre_registered_peak_mib": pre_registered_peak_mib,
            "source": "maximum peak_reserved_mib=244 across the 12 verified historical epoch-40 model runs; training high-water is used only to admit the separate 64-row inference Stage0",
            "same_runner_measurement": False,
            "safety_margin_mib": safety_margin_mib,
            "required_free_memory_mib": required_free_mib,
            "capacity_candidates": [g["index"] for g in gpus if g["memory_free_mib"] >= required_free_mib]
        },
        "external_process_identity_recorded": False,
        "non_child_process_signaled": False,
        "new_training": 0,
        "optimizer_updates": 0,
        "new_encoder_extraction": 0,
        "final_test_access": 0
    }
    path = audit / "GPU_HOST_PREFLIGHT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".part")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps({"status": receipt["status"], "host_context": host_context,
                      "visible_gpu_count": len(gpus), "capacity_candidates": receipt["stage0_admission_reference"]["capacity_candidates"],
                      "required_free_memory_mib": required_free_mib}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

