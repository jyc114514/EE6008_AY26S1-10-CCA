"""Record an already approved, externally collected host/GPU aggregate preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
STAGE0_PEAK_MIB = 12288
STAGE0_REQUIRED_FREE_MIB = STAGE0_PEAK_MIB + max(512, 1230)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-user-approval", action="store_true")
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--node-role", required=True, choices=("login", "interactive_gpu", "approved_scheduler_gpu", "unknown"))
    parser.add_argument("--scheduler", required=True, choices=("none", "slurm", "other", "unknown"))
    parser.add_argument("--filesystem-capacity-bytes", type=int, required=True)
    parser.add_argument("--filesystem-free-bytes", type=int, required=True)
    parser.add_argument("--filesystem-user-available-bytes", type=int, required=True)
    parser.add_argument("--gpu-json", required=True, help="JSON array of aggregate index/name/memory-used/total/utilization rows; no process table")
    args = parser.parse_args()
    gpus = json.loads(args.gpu_json)
    if not isinstance(gpus, list) or any(not isinstance(row, dict) for row in gpus):
        raise ValueError("gpu-json must be a list of aggregate GPU dictionaries")
    required_fields = {"index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent"}
    for gpu in gpus:
        if set(gpu) < required_fields:
            raise ValueError("GPU aggregate row is missing a required field")
        if not isinstance(gpu["index"], int) or gpu["memory_total_mib"] <= 0:
            raise ValueError("invalid GPU index/memory")
    fraction = min(args.filesystem_free_bytes, args.filesystem_user_available_bytes) / args.filesystem_capacity_bytes
    capacity = [gpu for gpu in gpus if int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"]) >= STAGE0_REQUIRED_FREE_MIB]
    role_ok = args.node_role in {"interactive_gpu", "approved_scheduler_gpu"}
    gate_status = "PASS" if args.explicit_user_approval and role_ok and fraction >= 0.10 and len(capacity) > 0 else "BLOCKED"
    reasons = []
    if not args.explicit_user_approval:
        reasons.append("explicit_user_approval_missing")
    if not role_ok:
        reasons.append("node_role_not_approved_for_gpu_training")
    if fraction < 0.10:
        reasons.append("filesystem_free_below_10_percent")
    if not capacity:
        reasons.append("no_GPU_meets_Stage0_free_memory_contract")
    record = {
        "status": gate_status,
        "observed_utc": utc_now(),
        "explicit_user_approval": bool(args.explicit_user_approval),
        "approved_scope": "Core-B D3-EQ Stage0, formal 36-cell replay and locked inference only" if args.explicit_user_approval else "none",
        "node_name": args.node_name,
        "node_role": args.node_role,
        "scheduler": args.scheduler,
        "filesystem_capacity_bytes": args.filesystem_capacity_bytes,
        "filesystem_free_bytes": args.filesystem_free_bytes,
        "filesystem_user_available_bytes": args.filesystem_user_available_bytes,
        "filesystem_free_fraction": fraction,
        "filesystem_quota_note": "block availability is a point-in-time lower-bound check; it does not prove project quota",
        "stage0_pre_registered_peak_mib": STAGE0_PEAK_MIB,
        "stage0_required_free_mib": STAGE0_REQUIRED_FREE_MIB,
        "aggregate_gpu_inventory": sorted(gpus, key=lambda row: int(row["index"])),
        "candidate_gpu_indices_by_stage0_memory": [int(row["index"]) for row in capacity],
        "gpu_utilization_role": "ranking/observation only; not an idle gate",
        "compute_process_table_read": False,
        "other_users_signaled": False,
        "project_watcher_required": True,
        "project_gpu_children_max": 2,
        "project_children_per_physical_gpu_max": 1,
        "blocked_reasons": reasons,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_HOST_APPROVAL.json"
    atomic_json(out, record)
    print(json.dumps({"status": gate_status, "node_role": args.node_role, "filesystem_free_fraction": round(fraction, 4), "candidate_gpus": len(capacity), "required_free_mib": STAGE0_REQUIRED_FREE_MIB, "blocked_reasons": reasons, "final_test_access": 0}, sort_keys=True))
    return 0 if gate_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
