"""Generate watcher-contract D3-EQ Stage0 or formal job JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
PYTHON = PROJECT_ROOT / "envs/jyc/bin/python"
TRAINER = EQ_ROOT / "code/d3_train_eq.py"
LEASE_DIR = REPO_ROOT / "reports/locks/gpu_lease"
PROTOCOL_ID = "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1"
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    part.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_record(config: str, horizon: int, seed: int, *, stage0: bool, peak_mib: int) -> dict[str, Any]:
    phase = "STAGE0" if stage0 else "FORMAL"
    run_id = f"D3EQ-{config}-H{horizon}-S{seed}"
    job_id = f"{run_id}-{phase}"
    run_root = EQ_ROOT / "d3/stage0_runs" / job_id if stage0 else EQ_ROOT / "d3/runs" / run_id
    command = [
        str(PYTHON), str(TRAINER),
        "--config", config,
        "--horizon", str(horizon),
        "--seed", str(seed),
        "--run-root", str(run_root),
        "--batch-size", "64",
        "--accumulation", "1",
    ]
    if stage0:
        command.append("--stage0")
    job_path = EQ_ROOT / "configs" / phase.lower() / f"{job_id}.json"
    status_path = EQ_ROOT / "logs" / f"watcher_{job_id}.json"
    log_path = EQ_ROOT / "logs" / f"child_{job_id}.log"
    job = {
        "job_id": job_id,
        "protocol_id": PROTOCOL_ID,
        "output_root": str(run_root),
        "command": command,
        "resource_estimate": {
            "measured_peak_mib": int(peak_mib),
            "estimate_type": "historical_same_model_pre_registered_bound_for_stage0" if stage0 else "measured_from_d3eq_stage0",
            "basis": "old D3 Stage0 used the same 0.891M-parameter model at effective batch64; new Stage0 measures the actual peak before formal admission" if stage0 else "set from maximum allocated-memory report across the 12 accepted D3-EQ Stage0 jobs",
        },
        "log_path": str(log_path),
        "max_oom_requeues": 0,
        "final_test_access": 0,
    }
    return {
        "job_id": job_id,
        "run_id": run_id,
        "config": config,
        "horizon_steps": horizon,
        "seed": seed,
        "phase": phase,
        "job_path": str(job_path),
        "status_path": str(status_path),
        "log_path": str(log_path),
        "lease_dir": str(LEASE_DIR),
        "job": job,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage0", "formal"), required=True)
    parser.add_argument("--peak-mib", type=int, default=12288)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.peak_mib < 1:
        raise ValueError("peak-mib must be positive")
    if args.phase == "stage0" and args.peak_mib != 12288:
        raise ValueError("Stage0 admission bound is pre-registered at 12,288 MiB from historical same-model runs")
    if args.phase == "formal" and not (EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").is_file():
        raise FileNotFoundError("formal job queue requires accepted 12/12 Stage0 results")

    stage0 = args.phase == "stage0"
    seeds = (20261001,) if stage0 else SEEDS
    manifest = []
    for seed in seeds:
        for horizon in (30, 60):
            for config in CONFIGS:
                record = job_record(config, horizon, seed, stage0=stage0, peak_mib=args.peak_mib)
                path = Path(record["job_path"])
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f"job spec already exists: {path}")
                atomic_json(path, record["job"])
                manifest.append({key: record[key] for key in ("job_id", "run_id", "config", "horizon_steps", "seed", "phase", "job_path", "status_path", "log_path", "lease_dir")})

    manifest_path = EQ_ROOT / "configs" / f"D3_EQ_{args.phase.upper()}_JOBS.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"job manifest already exists: {manifest_path}")
    atomic_json(manifest_path, {
        "protocol_id": PROTOCOL_ID,
        "phase": args.phase.upper(),
        "status": "LOCKED_NOT_STARTED",
        "jobs": manifest,
        "job_count": len(manifest),
        "registered_peak_mib": args.peak_mib,
        "global_lease_dir": str(LEASE_DIR),
        "max_parallel_children_from_queue": 1,
        "max_oom_requeues": 0,
        "final_test_access": 0,
    })
    print(json.dumps({"status": "PASS", "phase": args.phase, "jobs": len(manifest), "peak_mib": args.peak_mib, "manifest_sha256": sha256_file(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
