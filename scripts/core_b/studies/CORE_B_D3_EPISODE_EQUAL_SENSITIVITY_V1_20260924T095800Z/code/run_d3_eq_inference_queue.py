"""Sequential shared-watcher inference queue for all 36 D3-EQ cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT_ROOT / "envs/jyc/bin/python"
WATCHER = REPO_ROOT / "tools/gpu_lease_watcher.py"
LEASE_DIR = REPO_ROOT / "reports/locks/gpu_lease"
PROTOCOL_ID = "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def append_ledger(record: dict[str, Any]) -> None:
    path = EQ_ROOT / "INFERENCE_JOB_LEDGER.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def elapsed_lease_hours(path: Path) -> float:
    if not path.is_file():
        return 0.0
    item = json.loads(path.read_text(encoding="utf-8"))
    start, end = item.get("lease_acquired_utc"), item.get("lease_released_utc")
    if not start or not end:
        return 0.0
    a = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    b = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    return max((b - a).total_seconds() / 3600.0, 0.0)


def outputs_for(item: dict[str, Any], phase: str) -> tuple[Path, Path]:
    root = Path(item["output_root"])
    return root / f"{item['run_id']}.npz", root / f"{item['run_id']}.json"


def output_accepted(item: dict[str, Any], phase: str) -> dict[str, Any] | None:
    npz_path, sidecar_path = outputs_for(item, phase)
    if not npz_path.is_file() or not sidecar_path.is_file():
        return None
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or sha256_file(npz_path) != record.get("prediction_sha256"):
        return None
    return record


def run_one(item: dict[str, Any], *, phase: str, max_wait_seconds: int) -> None:
    if output_accepted(item, phase):
        return
    job = json.loads(Path(item["job_path"]).read_text(encoding="utf-8"))
    status_path = Path(item["status_path"])
    log_path = Path(item["log_path"])
    command = [
        str(PYTHON), str(WATCHER),
        "--job-json", item["job_path"],
        "--lease-dir", str(LEASE_DIR),
        "--status-path", str(status_path),
        "--poll-seconds", "15.0",
        "--max-wait-seconds", str(max_wait_seconds),
        "--max-oom-requeues", str(int(job.get("max_oom_requeues", 0))),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} command={json.dumps(command)}\n{result.stdout}{result.stderr}")
        handle.flush()
        os.fsync(handle.fileno())
    append_ledger({
        "utc": utc_now(), "phase": phase, "job_id": item["job_id"],
        "run_id": item["run_id"], "exit_code": result.returncode,
        "watcher_status_path": str(status_path), "log_path": str(log_path),
        "gpu_lease_hours": elapsed_lease_hours(status_path), "final_test_access": 0,
    })
    if result.returncode != 0:
        raise RuntimeError(f"watcher inference failed: {item['job_id']}; see {log_path}")
    if not output_accepted(item, phase):
        raise AssertionError(f"watcher exited successfully but output is not accepted: {item['run_id']}")


def build_acceptance(manifest: dict[str, Any], phase: str) -> dict[str, Any]:
    import csv
    if phase == "primary":
        with np.load(PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz", allow_pickle=False) as data:
            ids = data["anchor_id"][data["split_code"] == 1].astype("U64")
        expected_rows, id_key = 2322, "anchor_id"
    else:
        with np.load(PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz", allow_pickle=False) as data:
            ids = data["dense_id"].astype("U96")
        expected_rows, id_key = 3718, "dense_id"
    expected_id_hash = hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()
    bundles = []
    total_gpu_hours = 0.0
    for item in manifest["jobs"]:
        record = output_accepted(item, phase)
        if record is None:
            raise AssertionError(f"inference bundle missing/not accepted: {item['run_id']}")
        if int(record.get("rows", -1)) != expected_rows:
            raise AssertionError(f"inference row count mismatch: {item['run_id']}")
        if record.get("dev_ids_sha256", record.get("dense_ids_sha256")) != expected_id_hash:
            raise AssertionError(f"inference ID hash mismatch: {item['run_id']}")
        watcher_path = Path(item["status_path"])
        total_gpu_hours += elapsed_lease_hours(watcher_path)
        bundles.append({
            "run_id": item["run_id"],
            "config": item["config"],
            "seed": item["seed"],
            "horizon_steps": item["horizon_steps"],
            "prediction_sha256": record["prediction_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "rows": record["rows"],
            "id_hash": expected_id_hash,
        })
    if len(bundles) != 36:
        raise AssertionError(f"expected 36 {phase} prediction bundles, found {len(bundles)}")
    return {
        "status": "PASS",
        "phase": phase,
        "accepted_bundles": len(bundles),
        "expected_bundles": 36,
        "rows_per_bundle": expected_rows,
        "id_field": id_key,
        "id_sha256": expected_id_hash,
        "inference_gpu_hours_actual": total_gpu_hours,
        "bundles": bundles,
        "dev_targets_loaded_by_inference": False,
        "final_test_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("primary", "dense"), required=True)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        raise RuntimeError("long inference queue must run inside tmux")
    host_gate_path = EQ_ROOT / "qa/D3_EQ_HOST_APPROVAL.json"
    if not host_gate_path.is_file() or json.loads(host_gate_path.read_text(encoding="utf-8")).get("explicit_user_approval") is not True:
        raise PermissionError("explicit host approval gate is required")
    if args.phase == "primary":
        acceptance = EQ_ROOT / "qa/D3_EQ_TRAINING_ACCEPTANCE.json"
        gate = EQ_ROOT / "qa/D3_EQ_RESOURCE_GATE_POST_TRAINING.json"
    else:
        acceptance = EQ_ROOT / "qa/D3_EQ_PRIMARY_PREDICTION_ACCEPTANCE.json"
        gate = EQ_ROOT / "qa/D3_EQ_RESOURCE_GATE_PRE_DENSE.json"
        if json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8")).get("status") != "PASS":
            raise AssertionError("dense cache reuse audit is required before dense inference")
    if not acceptance.is_file() or json.loads(acceptance.read_text(encoding="utf-8")).get("status") != "PASS":
        raise PermissionError(f"prior-stage acceptance is missing for {args.phase} inference")
    if not gate.is_file() or json.loads(gate.read_text(encoding="utf-8")).get("status") != "PASS":
        raise PermissionError(f"resource gate is missing/not PASS for {args.phase} inference")
    manifest_path = EQ_ROOT / "configs" / f"D3_EQ_{args.phase.upper()}_INFERENCE_JOBS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("phase") != args.phase.upper() or len(manifest.get("jobs", [])) != 36:
        raise AssertionError("inference job manifest is not the locked 36-cell phase")
    atomic_json(EQ_ROOT / "logs/D3_EQ_{0}_INFERENCE_QUEUE_STATUS.json".format(args.phase.upper()), {"status": "running", "phase": args.phase, "started_utc": utc_now(), "final_test_access": 0})
    for item in manifest["jobs"]:
        run_one(item, phase=args.phase, max_wait_seconds=args.max_wait_seconds)
    result = build_acceptance(manifest, args.phase)
    acceptance_path = EQ_ROOT / "qa" / f"D3_EQ_{args.phase.upper()}_PREDICTION_ACCEPTANCE.json"
    atomic_json(acceptance_path, result)
    atomic_json(EQ_ROOT / "logs/D3_EQ_{0}_INFERENCE_QUEUE_STATUS.json".format(args.phase.upper()), {"status": "complete", "phase": args.phase, "completed_utc": utc_now(), "accepted_bundles": 36, "gpu_hours": result["inference_gpu_hours_actual"], "final_test_access": 0})
    print(json.dumps({"status": "PASS", "phase": args.phase, "accepted_bundles": 36, "rows_per_bundle": result["rows_per_bundle"], "gpu_hours": result["inference_gpu_hours_actual"], "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
