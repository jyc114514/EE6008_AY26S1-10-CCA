#!/usr/bin/env python3
"""Build watcher jobs for fresh-process V2 checkpoint/model inference."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = str(PROJECT / "envs/jyc/bin/python")
resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
WATCHER = Path(resolved["shared_watcher"])
INFER_SCRIPT = ROOT / "code/v2_inference.py"
Q0_INFER_ESTIMATE = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/configs/D3_INFERENCE_JOB.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_idempotent(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing inference job differs; preserving it: {path}")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def build_one(training_watcher_path: Path, *, phase: str, dataset_kind: str,
              dataset_role: str | None, max_rows: int | None) -> tuple[str, Path]:
    training_outer = json.loads(training_watcher_path.read_text(encoding="utf-8"))
    fit_path = Path(training_outer["command"][-1])
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    model_dir = Path(fit["job_root"])
    model_name = "model_epoch160.safetensors" if dataset_kind == "train" else "model.safetensors"
    model_path = model_dir / model_name
    model_manifest_path = model_dir / "MODEL_MANIFEST.json"
    out_dir = ROOT / "predictions" / ("pilot_reload" if phase == "pilot_reload" else "dev") / str(fit["job_id"])
    job_id = f"v2_{phase}_{fit['job_id']}"
    inference_fit = {
        "schema_version": 1,
        "job_id": job_id,
        "phase": phase,
        "model_id": fit["model_id"],
        "training_job_id": fit["job_id"],
        "scope": fit["scope"],
        "task_id": fit["task_id"],
        "task_filter": 119 if phase == "pilot_reload" else None,
        "video_mode": fit["video_mode"],
        "waypoint_mode": fit.get("waypoint_mode"),
        "seed": int(fit["seed"]),
        "model_path": str(model_path),
        "model_manifest_path": str(model_manifest_path),
        "normalizer_path": str(fit["normalizer_path"]),
        "dataset_kind": dataset_kind,
        "dataset_role": dataset_role,
        "data_path": str(ROOT / ("data/TRAIN_SELECTED_H30.npz" if dataset_kind == "train" else "data/DEV_PREFIXES_NO_LABELS.npz")),
        "max_rows": max_rows,
        "batch_size": 32,
        "output_root": str(out_dir),
        "final_heldout_payload_access": 0,
    }
    job_json = ROOT / "configs/inference" / phase / f"{job_id}.fit.json"
    write_json_idempotent(job_json, inference_fit)
    infer_est = json.loads(Q0_INFER_ESTIMATE.read_text(encoding="utf-8"))
    peak = int(infer_est["resource_estimate"]["measured_peak_mib"])
    if peak != 8192:
        raise RuntimeError(f"accepted Q0 inference estimate changed: {peak}")
    watcher_config = {
        "job_id": job_id,
        "protocol_id": "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013",
        "output_root": str(out_dir),
        "command": [PYTHON, str(INFER_SCRIPT), "--job-json", str(job_json)],
        "resource_estimate": {
            "estimate_type": "V2_Q0_ACCEPTED_INFERENCE_PRE_REGISTERED",
            "measured_peak_mib": peak,
            "source_config": str(Q0_INFER_ESTIMATE),
            "source_config_sha256": digest(Q0_INFER_ESTIMATE),
            "phase": phase,
        },
        "log_path": str(ROOT / "logs/watcher_children" / f"{job_id}.log"),
        "max_oom_requeues": 1,
        "environment": {
            "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        },
        "final_test_access": 0,
    }
    watcher_json = ROOT / "configs/watcher" / phase / f"{job_id}.json"
    write_json_idempotent(watcher_json, watcher_config)
    return job_id, watcher_json


def queue_for(training_queue_path: Path, *, phase: str, dataset_kind: str,
              dataset_role: str | None = None, max_rows: int | None = None) -> list[str]:
    training_queue = json.loads(training_queue_path.read_text(encoding="utf-8"))
    result: list[str] = []
    for raw in training_queue["jobs"]:
        _, watcher = build_one(Path(raw), phase=phase, dataset_kind=dataset_kind,
                               dataset_role=dataset_role, max_rows=max_rows)
        result.append(str(watcher))
    return sorted(result)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot_reload", "formal_dev"), required=True)
    args = parser.parse_args()
    source_hash = digest(WATCHER)
    accepted_hash = json.loads((ROOT / "inventory/SOURCE_ACCEPTANCE.json").read_text())["shared_watcher_sha256"]
    if source_hash != accepted_hash:
        raise RuntimeError("shared watcher hash differs from V2 accepted source")
    if args.phase == "pilot_reload":
        jobs = queue_for(ROOT / "coordination/queues/SCREENING_PILOT.json",
                         phase="pilot_reload", dataset_kind="train", dataset_role="inner_tune", max_rows=1)
        if len(jobs) != 2:
            raise RuntimeError(f"pilot reload queue must contain two jobs, got {len(jobs)}")
    else:
        formal = queue_for(ROOT / "coordination/queues/FORMAL.json", phase="formal_dev", dataset_kind="dev")
        waypoint = queue_for(ROOT / "coordination/queues/WAYPOINT.json", phase="formal_dev", dataset_kind="dev")
        jobs = formal + waypoint
        if len(formal) != 36 or len(waypoint) != 6:
            raise RuntimeError("formal dev prediction queue size differs from locked 36+6")
    queue_path = ROOT / "coordination/queues" / f"{args.phase.upper()}.json"
    write_json_idempotent(queue_path, {
        "schema_version": 1, "phase": args.phase, "jobs": sorted(jobs),
        "inference_job_count": len(jobs), "labels_loaded": False,
        "final_heldout_payload_access": 0,
    })
    dry_rows = []
    lease_dir = ROOT / "coordination/locks/gpu_lease"
    for watcher_raw in sorted(jobs):
        watcher_path = Path(watcher_raw)
        watcher = json.loads(watcher_path.read_text(encoding="utf-8"))
        status_path = ROOT / "coordination/watcher_status" / f"dryrun_{watcher['job_id']}.json"
        result = subprocess.run(
            [PYTHON, str(WATCHER), "--job-json", str(watcher_path),
             "--lease-dir", str(lease_dir), "--status-path", str(status_path), "--dry-run"],
            cwd=str(PROJECT), text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"inference watcher dry-run failed for {watcher['job_id']}: {result.stderr}")
        item = json.loads(result.stdout)
        if item.get("status") != "valid":
            raise RuntimeError(f"inference watcher dry-run status invalid for {watcher['job_id']}")
        dry_rows.append({
            "job_id": watcher["job_id"], "required_free_memory_mib": item["required_free_memory_mib"],
            "measured_peak_mib": item["measured_peak_mib"], "watcher_child_cap": item["max_concurrent_project_gpu_children"],
            "visible_gpu_pool": item["visible_gpu_pool"], "dry_run_only": True,
        })
    dry_receipt = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS", "phase": args.phase, "jobs_checked": len(dry_rows),
        "gpu_device_queries": 0, "labels_loaded": False, "results": dry_rows,
        "final_heldout_payload_access": 0,
    }
    dry_path = ROOT / "qa" / ("INFERENCE_WATCHER_DRYRUN_PILOT.json" if args.phase == "pilot_reload" else "INFERENCE_WATCHER_DRYRUN_FORMAL_DEV.json")
    if dry_path.exists():
        existing = json.loads(dry_path.read_text(encoding="utf-8"))
        existing.pop("created_utc", None)
        comparable = dict(dry_receipt); comparable.pop("created_utc", None)
        if existing != comparable:
            raise RuntimeError(f"existing inference dry-run receipt differs: {dry_path}")
    else:
        dry_path.write_text(json.dumps(dry_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "phase": args.phase, "inference_job_count": len(jobs), "queue": str(queue_path), "watcher_dryrun": str(dry_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
