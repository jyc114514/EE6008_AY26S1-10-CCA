"""Run the shared project watcher through a strict task-local GPU2/3 filter."""
from __future__ import annotations
import os

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SHARED_WATCHER = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
ALLOWLIST_PATH = TASK_ROOT / "protocol/GPU_ALLOWLIST_LOCKED.json"
PROTOCOL_PATH = TASK_ROOT / "PROTOCOL_LOCKED.json"
sys.path.insert(0, str(TASK_ROOT / "code"))
from gpu23_filter import filter_snapshots, validate_allowlist, validate_live_identity  # noqa: E402


def load_watcher():
    spec = importlib.util.spec_from_file_location("shared_gpu_lease_watcher", SHARED_WATCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the shared project GPU watcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_allowed_uuids() -> dict[int, str]:
    data = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    if data.get("status") != "LOCKED":
        raise AssertionError("task GPU allowlist is not locked")
    authorized = {int(index) for index in data.get("physical_gpu_ids", [])}
    if authorized != {2, 3}:
        raise AssertionError("task authorization scope must remain exactly physical GPUs 2 and 3")
    recorded = {int(row["physical_index"]): str(row["uuid"]) for row in data["gpus"]}
    if set(recorded) != authorized:
        raise AssertionError("GPU identity records must cover the authorized scope")
    candidates = {int(index) for index in data.get("launch_candidates", [])}
    if candidates != {2}:
        raise AssertionError("this resumed run must have exactly physical GPU2 as its active candidate")
    return validate_allowlist({index: recorded[index] for index in candidates})


def query_snapshots(watcher: Any, indices: tuple[int, ...], nvidia_smi: str) -> list[Any]:
    if not indices or any(index not in (2, 3) for index in indices):
        raise ValueError("task watcher may query only physical GPU2 and GPU3")
    query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(
        [nvidia_smi, "-i", ",".join(map(str, indices)),
         f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return watcher._parse_nvidia_rows(result.stdout)


def build_callbacks(watcher: Any, allowed: dict[int, str], nvidia_smi: str):
    candidate_indices = tuple(sorted(allowed))

    def visible_snapshots() -> list[Any]:
        return filter_snapshots(query_snapshots(watcher, candidate_indices, nvidia_smi), allowed)

    def snapshot_one(index: int) -> Any:
        physical = int(index)
        if physical not in allowed:
            raise RuntimeError(f"shared watcher selected unauthorized physical GPU {physical}")
        rows = query_snapshots(watcher, (physical,), nvidia_smi)
        if len(rows) != 1:
            raise RuntimeError(f"physical GPU {physical} identity query returned an unexpected row count")
        validate_live_identity(rows[0].index, rows[0].uuid, allowed,
                               utilization_percent=rows[0].utilization_percent)
        valid = filter_snapshots(rows, allowed)
        if len(valid) != 1:
            raise RuntimeError(f"physical GPU {physical} no longer matches its locked active candidate")
        return valid[0]

    return visible_snapshots, snapshot_one


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--lease-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-wait-seconds", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or protocol.get("study_id") != "CORRECTED_SCHEDULER_MATCHED_12":
        raise AssertionError("task protocol identity/status is not locked")
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    output_root = Path(str(job.get("output_root", ""))).resolve()
    if not output_root.is_relative_to((TASK_ROOT / "runs").resolve()):
        raise ValueError("watcher job output_root must remain within this task's isolated runs/")
    if str(job.get("protocol_id")) != "CORRECTED_SCHEDULER_MATCHED_12":
        raise ValueError("watcher job protocol_id mismatch")
    job_env = dict(job.get("environment", {}))
    allowed = load_allowed_uuids()
    if not args.dry_run:
        job_env["D3_GPU_UUID_MAP"] = json.dumps({str(k): v for k, v in allowed.items()}, sort_keys=True)
        allowlist_doc = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
        name_map = {str(row["physical_index"]): row["name"] for row in allowlist_doc["gpus"]
                    if int(row["physical_index"]) in allowed}
        job_env["D3_GPU_NAME_MAP"] = json.dumps(name_map, sort_keys=True)
        job_env["D3_TASK_GPU_ALLOWLIST"] = ",".join(map(str, sorted(allowed)))
        job_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    job["environment"] = job_env

    watcher = load_watcher()
    watcher.validate_job(job)
    if args.dry_run:
        print(json.dumps({
            "status": "PASS_DRY_RUN",
            "job_id": job["job_id"],
            "protocol_id": job["protocol_id"],
            "output_root": str(output_root),
            "authorized_physical_gpu_ids": [2, 3],
            "active_physical_gpu_candidates": sorted(allowed),
            "gpu_query_performed": False,
        }, sort_keys=True))
        return 0

    visible, snapshot_one = build_callbacks(watcher, allowed, args.nvidia_smi)
    return watcher.run_registered_job(
        job,
        lease_dir=args.lease_dir.resolve(),
        status_path=args.status_path.resolve(),
        visible_snapshots_fn=visible,
        snapshot_fn=snapshot_one,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        nvidia_smi=args.nvidia_smi,
        max_oom_requeues=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
