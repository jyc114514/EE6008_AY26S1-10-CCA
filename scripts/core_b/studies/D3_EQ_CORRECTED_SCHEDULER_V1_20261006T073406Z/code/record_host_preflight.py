"""Record a scoped host preflight from an already approved GPU2/3 aggregate query."""
from __future__ import annotations

import argparse
import json
import math
import os
import pwd
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
LEASE_DIR = PROJECT / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"
GPU_HOUR_CAP = 12.0
GPU_MEMORY_CAP_MIB = 20 * 1024
HIGH_ACTIVITY_THRESHOLD_PERCENT = 90


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def mib_available_memory() -> int:
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(pages * page_size // (1024 * 1024))


def parse_gpu(args: argparse.Namespace, index: int) -> dict[str, Any]:
    prefix = f"gpu{index}_"
    values = {
        "physical_index": index,
        "uuid": getattr(args, prefix + "uuid"),
        "name": getattr(args, prefix + "name"),
        "memory_used_mib": int(getattr(args, prefix + "used")),
        "memory_total_mib": int(getattr(args, prefix + "total")),
        "utilization_percent": int(getattr(args, prefix + "util")),
    }
    if not values["uuid"].startswith("GPU-") or values["memory_total_mib"] <= 0:
        raise ValueError(f"invalid aggregate GPU{index} row")
    values["memory_free_mib"] = max(values["memory_total_mib"] - values["memory_used_mib"], 0)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    for index in (2, 3):
        parser.add_argument(f"--gpu{index}-uuid", required=True)
        parser.add_argument(f"--gpu{index}-name", required=True)
        parser.add_argument(f"--gpu{index}-used", required=True, type=int)
        parser.add_argument(f"--gpu{index}-total", required=True, type=int)
        parser.add_argument(f"--gpu{index}-util", required=True, type=int)
    parser.add_argument("--launch-candidates", required=True,
                         help="comma-separated subset of task-authorized physical GPUs 2,3")
    parser.add_argument("--candidate-reason", required=True,
                         help="document why this run uses a strict subset; never widens authorization")
    parser.add_argument("--gpu-snapshot-observed-utc", required=True,
                         help="UTC timestamp bracketing the approved aggregate nvidia-smi sample")
    args = parser.parse_args()
    try:
        requested_candidates = [int(token.strip()) for token in args.launch_candidates.split(",") if token.strip()]
    except ValueError as exc:
        parser.error(f"invalid --launch-candidates: {exc}")
    if not requested_candidates or len(set(requested_candidates)) != len(requested_candidates):
        parser.error("--launch-candidates must be a nonempty unique set")
    if not set(requested_candidates) <= {2, 3}:
        parser.error("--launch-candidates may contain only physical GPUs 2 and 3")
    candidate_reason = args.candidate_reason.strip()
    if not candidate_reason:
        parser.error("--candidate-reason must be nonempty")
    try:
        gpu_snapshot_time = datetime.fromisoformat(args.gpu_snapshot_observed_utc.replace("Z", "+00:00"))
        if gpu_snapshot_time.tzinfo is None:
            raise ValueError("timestamp must include UTC offset")
    except ValueError as exc:
        parser.error(f"invalid --gpu-snapshot-observed-utc: {exc}")
    gpu_snapshot_age_seconds = (datetime.now(timezone.utc) - gpu_snapshot_time.astimezone(timezone.utc)).total_seconds()
    snapshot_fresh_for_launch = 0 <= gpu_snapshot_age_seconds <= 600

    gpu_rows = [parse_gpu(args, 2), parse_gpu(args, 3)]
    if len({row["uuid"] for row in gpu_rows}) != 2:
        raise AssertionError("GPU2/GPU3 UUIDs must be distinct")
    stage0_reports = []
    for config in ("TS11-D", "STATE-CTRL"):
        for horizon in (30, 60):
            report_path = EQ / "d3/stage0_runs" / f"D3EQ-{config}-H{horizon}-S20261001-STAGE0" / "STAGE0_REPORT.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "PASS" or int(report.get("optimizer_steps", -1)) != 4:
                raise AssertionError(f"historical matched Stage0 not accepted: {report_path}")
            stage0_reports.append(report)
    measured_peak_mib = math.ceil(max(int(row["peak_reserved_bytes"]) for row in stage0_reports) / (1024 * 1024))
    margin_mib = max(512, math.ceil(0.10 * measured_peak_mib))
    required_free_mib = measured_peak_mib + margin_mib
    capacity_candidates = [row["physical_index"] for row in gpu_rows if row["memory_free_mib"] >= required_free_mib]
    high_activity_candidates = [row["physical_index"] for row in gpu_rows
                                if row["utilization_percent"] >= HIGH_ACTIVITY_THRESHOLD_PERCENT]
    gpu_by_index = {row["physical_index"]: row for row in gpu_rows}
    launch_candidates = [
        index for index in requested_candidates
        if index in capacity_candidates
        and gpu_by_index[index]["utilization_percent"] < HIGH_ACTIVITY_THRESHOLD_PERCENT
    ]
    candidates_fully_available = len(launch_candidates) == len(requested_candidates)

    usage = shutil.disk_usage(PROJECT)
    free_fraction = usage.free / usage.total
    cpu_count = os.cpu_count()
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        user = str(os.getuid())
    hostname = socket.gethostname()
    lease_files = sorted(path.name for path in LEASE_DIR.glob("*.lease")) if LEASE_DIR.is_dir() else []
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    status = "PASS" if (hostname == os.environ.get("EE6008_HOST_LABEL", "shared-host") and free_fraction >= 0.10 and candidates_fully_available
                        and measured_peak_mib <= GPU_MEMORY_CAP_MIB and snapshot_fresh_for_launch) else "resource_waiting"

    allowlist = {
        "status": "LOCKED" if status == "PASS" else "NOT_LOCKED",
        "observed_utc": utc_now(),
        "gpu_snapshot_observed_utc": gpu_snapshot_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_age_seconds": gpu_snapshot_age_seconds,
        "snapshot_fresh_for_launch": snapshot_fresh_for_launch,
        "host": hostname,
        "physical_gpu_ids": [2, 3],
        "launch_candidates": launch_candidates if candidates_fully_available else [],
        "candidate_reason": candidate_reason,
        "high_activity_threshold_percent": HIGH_ACTIVITY_THRESHOLD_PERCENT,
        "high_activity_candidates": high_activity_candidates,
        "gpus": gpu_rows,
        "source": "approved nvidia-smi aggregate query with explicit -i 2,3",
        "process_table_queried": False,
        "final_test_access": 0,
    }
    gate = {
        "status": status,
        "observed_utc": utc_now(),
        "gpu_snapshot_observed_utc": gpu_snapshot_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gpu_snapshot_age_seconds": gpu_snapshot_age_seconds,
        "snapshot_fresh_for_launch": snapshot_fresh_for_launch,
        "host": hostname,
        "host_class": 'configured shared GPU host; prior accepted matched D3-EQ runs record direct CUDA training on this hostname; no Slurm job allocation currently observed; interactive-vs-login role remains an inference because no node-role file was found',
        "slurm_job_id": slurm_job_id,
        "unix_user": user,
        "cpu_logical_count": cpu_count,
        "host_memory_available_mib": mib_available_memory(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_free_fraction": free_fraction,
        "disk_free_fraction_min": 0.10,
        "user_quota_status": "not independently queried; filesystem free capacity recorded",
        "gpu_allowlist": [2, 3],
        "capacity_candidates": capacity_candidates,
        "high_activity_threshold_percent": HIGH_ACTIVITY_THRESHOLD_PERCENT,
        "high_activity_candidates": high_activity_candidates,
        "launch_candidates": launch_candidates if candidates_fully_available else [],
        "candidate_reason": candidate_reason,
        "own_queue_concurrency": len(launch_candidates) if candidates_fully_available else 0,
        "gpu_snapshots": gpu_rows,
        "historical_matched_stage0_peak_reserved_mib": measured_peak_mib,
        "watcher_memory_margin_mib": margin_mib,
        "watcher_required_free_mib": required_free_mib,
        "candidate_capacity_passed": candidates_fully_available,
        "gpu_hour_cap": GPU_HOUR_CAP,
        "gpu_memory_cap_mib": GPU_MEMORY_CAP_MIB,
        "project_gpu_lease_dir": str(LEASE_DIR),
        "project_lease_files_seen": lease_files,
        "project_lease_live_state": "shared watcher flock admission is authoritative; filenames may be stale metadata",
        "max_project_gpu_children": 2,
        "process_table_queried": False,
        "other_process_signaled": False,
        "final_test_access": 0,
    }
    atomic_json(TASK_ROOT / "protocol/GPU_ALLOWLIST_LOCKED.json", allowlist)
    atomic_json(TASK_ROOT / "RESOURCE_GATE_HOST.json", gate)
    print(json.dumps({
        "status": status,
        "host": hostname,
        "capacity_candidates": capacity_candidates,
        "high_activity_candidates": high_activity_candidates,
        "launch_candidates": launch_candidates if candidates_fully_available else [],
        "high_activity_threshold_percent": HIGH_ACTIVITY_THRESHOLD_PERCENT,
        "gpu_snapshot_observed_utc": gpu_snapshot_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_age_seconds": gpu_snapshot_age_seconds,
        "old_peak_reserved_mib": measured_peak_mib,
        "required_free_mib": required_free_mib,
        "disk_free_fraction": round(free_fraction, 6),
        "allowlist": [2, 3],
        "process_table_queried": False,
    }, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
