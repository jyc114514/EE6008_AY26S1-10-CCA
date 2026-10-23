#!/usr/bin/env python3
"""Build immutable per-fit and shared-watcher job configs for V2."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON = str(PROJECT / "envs/jyc/bin/python")
WATCHER = Path(json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text())["shared_watcher"])
TRAIN_SCRIPT = ROOT / "code/v2_train_job.py"
SCREEN_SEED = 202610130
FORMAL_SEEDS = (202610131, 202610132, 202610133)
LR_CANDIDATES = (5e-4, 1e-3, 2e-3)
TASK_IDS = (119, 94, 90, 141, 230)
Q0_STAGE0_ESTIMATE = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/configs/D3-TS11-D-H30-S20261001-STAGE0.json"
Q0_INFER_ESTIMATE = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/configs/D3_INFERENCE_JOB.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_idempotent(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing_value = json.loads(path.read_text(encoding="utf-8"))
            requested_value = json.loads(content)
            existing_value.pop("created_utc", None)
            requested_value.pop("created_utc", None)
            identical = existing_value == requested_value
        except Exception:
            identical = path.read_text(encoding="utf-8") == content
        if not identical and path.name == "FORMAL_MATRIX.json":
            prior_contract = ROOT / "protocol/MODEL_AND_LOSS.pre-pilot-status-clarification-20261013T090000Z.bak"
            old_contract_hash = existing_value.get("model_and_loss_sha256")
            if prior_contract.is_file() and sha_file(prior_contract) == old_contract_hash:
                old_matrix = dict(existing_value)
                new_matrix = dict(requested_value)
                old_matrix.pop("model_and_loss_sha256", None)
                new_matrix.pop("model_and_loss_sha256", None)
                if old_matrix == new_matrix:
                    return
        if not identical:
            raise RuntimeError(f"existing job config differs; preserving it: {path}")
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def lr_slug(lr: float) -> str:
    return {5e-4: "lr0p0005", 1e-3: "lr0p001", 2e-3: "lr0p002"}[float(lr)]


def model_specs() -> list[dict[str, Any]]:
    specs = [
        {"model_id": "G-S", "scope": "shared", "task_id": None, "video_mode": "STATE-CTRL"},
        {"model_id": "G-V", "scope": "shared", "task_id": None, "video_mode": "TS11-D"},
    ]
    for task_id in TASK_IDS:
        specs.extend([
            {"model_id": f"S-S-{task_id}", "scope": "expert", "task_id": task_id, "video_mode": "STATE-CTRL"},
            {"model_id": f"S-V-{task_id}", "scope": "expert", "task_id": task_id, "video_mode": "TS11-D"},
        ])
    return specs


def watcher_job(fit: dict[str, Any], resource: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    job_id = str(fit["job_id"])
    out = Path(fit["job_root"])
    watcher_config_path = ROOT / "configs/watcher" / str(fit["phase"]) / f"{job_id}.json"
    fit_config_path = ROOT / "configs/fits" / str(fit["phase"]) / f"{job_id}.json"
    watcher_status_path = ROOT / "coordination/watcher_status" / f"{job_id}.json"
    log_path = ROOT / "logs/watcher_children" / f"{job_id}.log"
    watcher = {
        "job_id": job_id,
        "protocol_id": "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013",
        "output_root": str(out),
        "command": [PYTHON, str(TRAIN_SCRIPT), "--job-json", str(fit_config_path)],
        "resource_estimate": resource,
        "log_path": str(log_path),
        "max_oom_requeues": 1,
        "environment": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
        "final_test_access": 0,
    }
    return watcher, watcher_config_path, fit_config_path


def training_resource(phase: str) -> dict[str, Any]:
    source = Q0_STAGE0_ESTIMATE
    evidence = json.loads(source.read_text(encoding="utf-8"))
    peak = int(evidence["resource_estimate"]["measured_peak_mib"])
    if peak != 12288:
        raise RuntimeError(f"accepted Q0 Stage0 pre-registered estimate changed: {peak}")
    return {
        "estimate_type": "V2_Q0_ACCEPTED_STAGE0_PRE_REGISTERED_CONSERVATIVE",
        "measured_peak_mib": peak,
        "source_config": str(source),
        "source_config_sha256": sha_file(source),
        "source_protocol": "CORE_B_D3_TIME_SPACE_FACTORIAL_V1",
        "batch_contract": "effective_batch64; pre-registered OOM fallback microbatch32 accumulation2",
        "phase": phase,
    }


def build_one(phase: str, spec: dict[str, Any], *, seed: int, lr: float, epochs: int,
              waypoint_mode: str | None = None) -> tuple[str, Path, dict[str, Any]]:
    video = str(spec["video_mode"])
    suffix = f"{spec['model_id'].replace('-', '_')}_{lr_slug(lr)}_s{seed}"
    if waypoint_mode:
        suffix += f"_{waypoint_mode.lower()}"
    job_id = f"v2_{phase}_{suffix}"
    job_root = ROOT / "model_only" / phase / job_id
    init_scope = "shared" if spec["scope"] == "shared" else "expert"
    train_role = "inner_fit" if phase == "screening" else "outer_train"
    normalizer_path = ROOT / ("screening/NORMALIZER_SCREENING.json" if phase == "screening" else "normalizers/NORMALIZER_FORMAL.json")
    checkpoint_epochs = [40, 80, 160] if phase == "screening" else [int(epochs)]
    fit = {
        "schema_version": 1,
        "job_id": job_id,
        "phase": phase,
        "model_id": str(spec["model_id"]),
        "scope": str(spec["scope"]),
        "task_id": spec["task_id"],
        "task_ids": list(TASK_IDS) if spec["scope"] == "shared" else [int(spec["task_id"])],
        "video_mode": video,
        "waypoint_mode": waypoint_mode,
        "seed": int(seed),
        "lr": float(lr),
        "epochs": int(epochs),
        "checkpoint_epochs": checkpoint_epochs,
        "normalizer_path": str(normalizer_path),
        "train_role": train_role,
        "microbatch_size": 64,
        "accumulation_steps": 1,
        "oom_fallback": {"microbatch_size": 32, "accumulation_steps": 2, "effective_batch": 64},
        "job_root": str(job_root),
        "attempt_dir": str(job_root),
        "init_scope": init_scope,
        "final_heldout_payload_access": 0,
    }
    resource = training_resource(phase)
    watcher, watcher_path, fit_path = watcher_job(fit, resource)
    write_json_idempotent(fit_path, fit)
    write_json_idempotent(watcher_path, watcher)
    return job_id, watcher_path, fit


def build_screening() -> None:
    all_jobs: list[str] = []
    pilot_jobs: list[str] = []
    for spec in model_specs():
        for lr in LR_CANDIDATES:
            job_id, watcher_path, _ = build_one("screening", spec, seed=SCREEN_SEED, lr=lr, epochs=160)
            all_jobs.append(str(watcher_path))
            if spec["model_id"] in {"S-S-119", "S-V-119"} and float(lr) == 1e-3:
                pilot_jobs.append(str(watcher_path))
    if len(all_jobs) != 36 or len(pilot_jobs) != 2:
        raise RuntimeError(f"screening matrix count mismatch: {len(all_jobs)} total, {len(pilot_jobs)} pilots")
    all_jobs = sorted(all_jobs)
    pilot_jobs = sorted(pilot_jobs)
    pilot_set = set(pilot_jobs)
    remaining = [path for path in all_jobs if path not in pilot_set]
    write_json_idempotent(ROOT / "coordination/queues/SCREENING_ALL.json", {
        "schema_version": 1, "phase": "screening", "jobs": all_jobs, "fit_count": 36,
        "pilot_count": 2, "pilot_queue": str(ROOT / "coordination/queues/SCREENING_PILOT.json"),
        "remaining_count": 34,
    })
    write_json_idempotent(ROOT / "coordination/queues/SCREENING_PILOT.json", {
        "schema_version": 1, "phase": "screening_pilot", "jobs": pilot_jobs, "fit_count": 2,
        "logical_models": ["S-S-119", "S-V-119"], "lr": 1e-3, "seed": SCREEN_SEED,
    })
    write_json_idempotent(ROOT / "coordination/queues/SCREENING_REMAINING.json", {
        "schema_version": 1, "phase": "screening_remaining", "jobs": remaining, "fit_count": 34,
    })


def build_formal_and_waypoint() -> None:
    lock_path = ROOT / "protocol/LR_EPOCH_SELECTION.json"
    selection = json.loads(lock_path.read_text(encoding="utf-8"))
    if selection.get("status") not in {"LOCKED", "LOCKED_WITH_PARTIAL_SEARCH"} or len(selection.get("logical_model_choices", [])) != 12:
        raise RuntimeError("complete 12-model inner-tune selection lock required before formal jobs")
    formal_paths: list[str] = []
    waypoint_paths: list[str] = []
    choices = {str(row["model_id"]): row for row in selection["logical_model_choices"]}
    specs = model_specs()
    for spec in specs:
        row = choices[str(spec["model_id"])]
        for seed in FORMAL_SEEDS:
            _, watcher_path, _ = build_one(
                "formal", spec, seed=seed, lr=float(row["selected_lr"]), epochs=int(row["selected_epoch"])
            )
            formal_paths.append(str(watcher_path))
    sv119 = choices["S-V-119"]
    for mode in ("AUX_ONLY", "GOAL_COND"):
        spec = {"model_id": f"B-{mode}", "scope": "expert", "task_id": 119, "video_mode": "TS11-D"}
        for seed in FORMAL_SEEDS:
            _, watcher_path, _ = build_one(
                "waypoint", spec, seed=seed, lr=float(sv119["selected_lr"]),
                epochs=int(sv119["selected_epoch"]), waypoint_mode=mode
            )
            waypoint_paths.append(str(watcher_path))
    if len(formal_paths) != 36 or len(waypoint_paths) != 6:
        raise RuntimeError("formal/waypoint fit count mismatch")
    write_json_idempotent(ROOT / "coordination/queues/FORMAL.json", {
        "schema_version": 1, "phase": "formal", "jobs": sorted(formal_paths), "fit_count": 36,
        "models": 12, "seeds": list(FORMAL_SEEDS), "selection_lock_sha256": sha_file(lock_path),
    })
    write_json_idempotent(ROOT / "coordination/queues/WAYPOINT.json", {
        "schema_version": 1, "phase": "waypoint", "jobs": sorted(waypoint_paths), "fit_count": 6,
        "task_id": 119, "selected_sv_lr": sv119["selected_lr"], "selected_sv_epoch": sv119["selected_epoch"],
        "seeds": list(FORMAL_SEEDS),
    })
    contract_path = ROOT / "protocol/MODEL_AND_LOSS.json"
    formal_matrix = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_FORMAL_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_lock_path": str(lock_path),
        "selection_lock_sha256": sha_file(lock_path),
        "model_and_loss_sha256": sha_file(contract_path),
        "formal_seeds": list(FORMAL_SEEDS),
        "logical_models": [
            {
                "model_id": str(spec["model_id"]), "scope": str(spec["scope"]),
                "task_ids": list(TASK_IDS) if spec["scope"] == "shared" else [int(spec["task_id"])],
                "video_mode": str(spec["video_mode"]),
                "selected_lr": float(choices[str(spec["model_id"])]["selected_lr"]),
                "selected_epoch": int(choices[str(spec["model_id"])]["selected_epoch"]),
                "selected_inner_tune_j": float(choices[str(spec["model_id"])]["selected_j_tune"]),
                "fresh_init_each_seed": True,
            }
            for spec in specs
        ],
        "formal_fit_count": len(formal_paths),
        "waypoint_fit_count": len(waypoint_paths),
        "waypoint_task_id": 119,
        "waypoint_selected_lr": float(sv119["selected_lr"]),
        "waypoint_selected_epoch": int(sv119["selected_epoch"]),
        "dev_targets_used_to_create_matrix": False,
        "final_heldout_payload_access": 0,
    }
    write_json_idempotent(ROOT / "protocol/FORMAL_MATRIX.json", formal_matrix)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screening", "formal_waypoint"), required=True)
    args = parser.parse_args()
    if hashlib.sha256(WATCHER.read_bytes()).hexdigest() != json.loads((ROOT / "inventory/SOURCE_ACCEPTANCE.json").read_text())["shared_watcher_sha256"]:
        raise RuntimeError("shared watcher hash differs from V2 accepted source")
    if args.phase == "screening":
        build_screening()
    else:
        build_formal_and_waypoint()
    print(json.dumps({"status": "PASS", "phase": args.phase}, ensure_ascii=False))


if __name__ == "__main__":
    main()
