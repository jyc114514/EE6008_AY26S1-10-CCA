"""Build all-cell primary or dense inference watcher jobs after training acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT_ROOT / "envs/jyc/bin/python"
PRIMARY_INFER = EQ_ROOT / "code/d3_eq_infer.py"
DENSE_INFER = EQ_ROOT / "code/d3_eq_dense_infer_entry.py"
LEASE_DIR = REPO_ROOT / "reports/locks/gpu_lease"
PROTOCOL_ID = "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("primary", "dense"), required=True)
    parser.add_argument("--peak-mib", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.peak_mib < 1:
        raise ValueError("peak-mib must be positive")
    train_accept = EQ_ROOT / "qa/D3_EQ_TRAINING_ACCEPTANCE.json"
    if not train_accept.is_file() or json.loads(train_accept.read_text(encoding="utf-8")).get("status") != "PASS":
        raise FileNotFoundError("36/36 accepted D3-EQ training runs are required before inference queue generation")
    if args.phase == "dense":
        primary_accept = EQ_ROOT / "qa/D3_EQ_PRIMARY_PREDICTION_ACCEPTANCE.json"
        if not primary_accept.is_file() or json.loads(primary_accept.read_text(encoding="utf-8")).get("status") != "PASS":
            raise FileNotFoundError("36/36 primary predictions must be accepted before dense inference")
        dense_cache = json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
        if dense_cache.get("status") != "PASS":
            raise AssertionError("accepted dense cache is required")

    matrix_path = EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv"
    import csv
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    if len(matrix) != 36:
        raise AssertionError("locked D3-EQ matrix must contain 36 cells")
    infer_script = PRIMARY_INFER if args.phase == "primary" else DENSE_INFER
    output_root = EQ_ROOT / "d3" / ("predictions" if args.phase == "primary" else "dense_predictions")
    jobs = []
    for cell in matrix:
        run_id = str(cell["run_id"])
        job_id = f"{run_id}-{args.phase.upper()}-INFER"
        command = [str(PYTHON), str(infer_script), "--run-id", run_id, "--batch-size", "64", "--resume"]
        job_root = EQ_ROOT / "d3/inference_job_roots" / args.phase / run_id
        job_path = EQ_ROOT / "configs" / f"{args.phase}_infer" / f"{job_id}.json"
        status_path = EQ_ROOT / "logs" / f"watcher_{job_id}.json"
        log_path = EQ_ROOT / "logs" / f"child_{job_id}.log"
        job = {
            "job_id": job_id,
            "protocol_id": PROTOCOL_ID,
            "output_root": str(job_root),
            "command": command,
            "resource_estimate": {
                "measured_peak_mib": args.peak_mib,
                "estimate_type": "measured_from_D3_EQ_Stage0_max_reserved_peak",
                "basis": "Stage0 uses the same model and effective batch; prediction uses no backward graph; conservative admission reuses maximum Stage0 reserved memory",
            },
            "log_path": str(log_path),
            "max_oom_requeues": 0,
            "final_test_access": 0,
        }
        if job_path.exists() and not args.overwrite:
            raise FileExistsError(job_path)
        atomic_json(job_path, job)
        jobs.append({
            "job_id": job_id,
            "run_id": run_id,
            "config": cell["config"],
            "seed": int(cell["seed"]),
            "horizon_steps": int(cell["horizon_steps"]),
            "phase": args.phase,
            "job_path": str(job_path),
            "status_path": str(status_path),
            "log_path": str(log_path),
            "output_root": str(output_root),
            "lease_dir": str(LEASE_DIR),
        })
    manifest_path = EQ_ROOT / "configs" / f"D3_EQ_{args.phase.upper()}_INFERENCE_JOBS.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    atomic_json(manifest_path, {
        "protocol_id": PROTOCOL_ID,
        "phase": args.phase.upper(),
        "status": "LOCKED_NOT_STARTED",
        "job_count": len(jobs),
        "measured_peak_mib": args.peak_mib,
        "output_root": str(output_root),
        "max_parallel_children_from_queue": 1,
        "max_oom_requeues": 0,
        "jobs": jobs,
        "final_test_access": 0,
    })
    print(json.dumps({"status": "PASS", "phase": args.phase, "job_count": len(jobs), "peak_mib": args.peak_mib}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
