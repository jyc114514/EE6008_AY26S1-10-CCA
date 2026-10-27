from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from v7_common import (
    DEV_NPZ,
    NORMALIZER,
    PROJECT,
    REPO,
    ROOT,
    SEEDS,
    TASKS,
    TRAIN_NPZ,
    V2,
    V3,
    V4,
    V5,
    V6,
    atomic_json,
    load_train_prefix,
    ordered_receiver_hash,
    receiver_rows,
    sha_file,
    write_csv,
)

PYTHON = PROJECT / "envs/jyc/bin/python"
WATCHER = REPO / "tools/gpu_lease_watcher.py"
OWNER_STATE_PATH = ROOT / "coordination/OWNER_STATE.json"
RESOURCE_LEDGER_PATH = ROOT / "results/RESOURCE_LEDGER.csv"
STATUS_DIR = ROOT / "coordination/status"
LEASE_DIR = ROOT / "coordination/gpu_lease"
WATCHER_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}
LEDGER_FIELDS = [
    "job_id", "phase", "status", "physical_gpu", "logical_device",
    "cuda_visible_devices", "child_pid", "peak_allocated_mib",
    "peak_reserved_mib", "elapsed_seconds", "measured_peak_mib",
    "resource_basis", "watcher_return_code", "watcher_status_path",
    "watcher_log_path", "output_root", "completion_marker",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_update(**updates: Any) -> None:
    current = read_json(OWNER_STATE_PATH)
    current.update(updates)
    current["updated_utc"] = utc_now()
    atomic_json(OWNER_STATE_PATH, current)


def append_resource(row: dict[str, Any]) -> None:
    rows = []
    if RESOURCE_LEDGER_PATH.exists():
        with RESOURCE_LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({field: row.get(field, "") for field in LEDGER_FIELDS})
    write_csv(RESOURCE_LEDGER_PATH, LEDGER_FIELDS, rows)


def next_attempt(base: Path) -> tuple[int, Path]:
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        path = base / f"attempt_{attempt:02d}"
        if not path.exists():
            return attempt, path
        attempt += 1


def config_dir(name: str) -> Path:
    path = ROOT / "configs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def watcher_payload(
    *,
    job_id: str,
    protocol_id: str,
    output_root: Path,
    command: list[str],
    measured_peak_mib: int,
    basis: str,
    preferred_gpu: int | None,
    marker: str,
    log_path: Path,
    max_oom_requeues: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "protocol_id": protocol_id,
        "output_root": str(output_root),
        "command": command,
        "resource_estimate": {
            "measured_peak_mib": int(measured_peak_mib),
            "basis": basis,
        },
        "completion_marker": marker,
        "max_oom_requeues": int(max_oom_requeues),
        "log_path": str(log_path),
        "environment": WATCHER_ENV,
    }
    if preferred_gpu is not None:
        payload["preferred_gpu"] = int(preferred_gpu)
    return payload


def run_watcher_job(
    *,
    phase: str,
    job_id: str,
    job_config_path: Path,
    watch_config_path: Path,
    watcher_payload_value: dict[str, Any],
    marker_path: Path,
    measured_peak_mib: int,
    resource_basis: str,
    output_root: Path,
) -> dict[str, Any]:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    status_path = STATUS_DIR / f"{job_id}.json"
    log_path = Path(watcher_payload_value["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(watch_config_path, watcher_payload_value)
    command = [
        str(PYTHON), str(WATCHER),
        "--job-json", str(watch_config_path),
        "--lease-dir", str(LEASE_DIR),
        "--status-path", str(status_path),
    ]
    state_update(
        status="running",
        phase=phase,
        active_jobs=1,
        active_gpu_jobs=1,
        current_job_id=job_id,
        current_output_root=str(output_root),
        current_watcher_status_path=str(status_path),
        current_command=command,
    )
    print(f"[{utc_now()}] START {phase} {job_id}", flush=True)
    started = time.perf_counter()
    env = dict(os.environ)
    env.update(WATCHER_ENV)
    result = subprocess.run(command, cwd=str(PROJECT), env=env, check=False)
    elapsed = time.perf_counter() - started
    watcher_status: dict[str, Any] = {}
    if status_path.is_file():
        watcher_status = read_json(status_path)
    marker: dict[str, Any] = {}
    if marker_path.is_file():
        try:
            marker = read_json(marker_path)
        except Exception:
            marker = {}
    passed = result.returncode == 0 and marker.get("status") == "PASS"
    physical_gpu = marker.get("physical_gpu", watcher_status.get("physical_gpu", ""))
    row = {
        "job_id": job_id,
        "phase": phase,
        "status": "PASS" if passed else "FAILED",
        "physical_gpu": physical_gpu,
        "logical_device": marker.get("logical_device", marker.get("logical_device", "")),
        "cuda_visible_devices": marker.get("cuda_visible_devices", ""),
        "child_pid": watcher_status.get("child_pid", ""),
        "peak_allocated_mib": marker.get("peak_memory_allocated_mib", ""),
        "peak_reserved_mib": marker.get("peak_memory_reserved_mib", ""),
        "elapsed_seconds": marker.get("elapsed_seconds", elapsed),
        "measured_peak_mib": measured_peak_mib,
        "resource_basis": resource_basis,
        "watcher_return_code": result.returncode,
        "watcher_status_path": str(status_path),
        "watcher_log_path": str(log_path),
        "output_root": str(output_root),
        "completion_marker": str(marker_path) if marker_path.exists() else "",
    }
    append_resource(row)
    state_update(
        active_jobs=0,
        active_gpu_jobs=0,
        current_job_id=None,
        current_output_root=None,
        current_watcher_status_path=None,
        last_job_id=job_id,
        last_job_status=row["status"],
        last_physical_gpu=physical_gpu,
    )
    print(f"[{utc_now()}] END {phase} {job_id} status={row['status']} gpu={physical_gpu}", flush=True)
    return {"passed": passed, "row": row, "marker": marker, "watcher_status": watcher_status, "return_code": result.returncode}


def require_host_gate() -> dict[str, Any]:
    path = ROOT / "coordination/HOST_RESOURCE_GATE.json"
    if not path.is_file():
        raise RuntimeError("host GPU resource approval/gate record is missing")
    gate = read_json(path)
    if gate.get("status") != "PASS" or gate.get("command_kind") != "aggregate_nvidia_smi_query":
        raise RuntimeError("host GPU resource gate is not PASS or uses an unsupported inventory query")
    if not gate.get("approval_requested") or not gate.get("approval_granted"):
        raise RuntimeError("host GPU execution approval has not been recorded")
    if not gate.get("aggregate_inventory"):
        raise RuntimeError("host GPU aggregate inventory is empty")
    return gate


def require_current_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    owner = read_json(ROOT / "coordination/OWNER.lock")
    state = read_json(OWNER_STATE_PATH)
    if not owner.get("owner_lock_acquired") or owner.get("out_root") != str(ROOT):
        raise RuntimeError("V7 owner lock does not match this unique output root")
    if state.get("old_root_writes") != 0 or state.get("final_heldout_payload_access") != 0:
        raise RuntimeError("owner state has an invalid prior write/test-access counter")
    watcher_policy = read_json(V4 / "inventory/WATCHER_POLICY.json")
    if sha_file(WATCHER) != watcher_policy.get("source_sha256") or watcher_policy.get("max_project_children") != 2:
        raise RuntimeError("shared watcher differs from the accepted V4 two-child policy/hash")
    v4_state = read_json(V4 / "coordination/OWNER_STATE.json")
    late_v6 = read_json(V6.parent / "CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z/coordination/OWNER_STATE.json")
    if v4_state.get("status") != "complete" or int(v4_state.get("active_jobs", -1)) != 0 or not v4_state.get("all_gpu_leases_released"):
        raise RuntimeError("accepted V4 root still reports active work or unreleased leases")
    if late_v6.get("status") != "complete" or int(late_v6.get("active_gpu_jobs", -1)) != 0 or not late_v6.get("all_gpu_leases_released"):
        raise RuntimeError("separate controlled late-fusion V6 root still reports active work")
    return owner, state


def protocol_inputs() -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    source_map = read_json(ROOT / "protocol/SOURCE_MAP.json")
    pairing = read_json(ROOT / "protocol/INIT_SAMPLER_PAIRING.json")
    models = {(int(row["seed"]), str(row["role"])): row for row in source_map["models"]}
    pairing_by_seed = {int(row["seed"]): row for row in pairing["seeds"]}
    return source_map, pairing_by_seed, models


def save_fit_config(
    *, seed: int, program: str, pilot: bool, attempt: int, microbatch: int,
    accumulation: int, output_root: Path, job_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    source_map, pairing_by_seed, model_sources = protocol_inputs()
    pairing = pairing_by_seed[seed]
    target_lock = read_json(ROOT / "protocol/TARGET_LOCK.json")
    teacher_entries = read_json(ROOT / "protocol/TEACHER_BANK_SEAL.json")["entries"]
    teacher_v = next(row for row in teacher_entries if int(row.get("seed", -1)) == seed and row.get("teacher_role") == "V")
    teacher_t = next(row for row in teacher_entries if int(row.get("seed", -1)) == seed and row.get("teacher_role") == "T")
    target_entry = next(
        row for row in target_lock["target_entries"]
        if int(row["seed"]) == seed and row["program"] == program
    )
    init_path = Path(pairing["init_path"])
    init_source = model_sources[(seed, "V")]
    target_identity = hashlib.sha256(
        "\n".join(str(row["anchor_id"]) for row in receiver_rows(load_train_prefix())).encode("utf-8")
    ).hexdigest()
    order = np_load_first_order(pairing["sampler_orders_path"])
    sampler_epoch1_hash = hashlib.sha256(order.tobytes()).hexdigest()
    job_id = f"v7_{'pilot' if pilot else 'fit'}_{program}_s{seed}_a{attempt:02d}"
    job = {
        "schema_version": 1,
        "job_id": job_id,
        "program_id": program,
        "seed": seed,
        "pilot": pilot,
        "phase": "pilot" if pilot else "formal",
        "video_mode": "TS11-D",
        "waypoint_mode": None,
        "lr": 0.0005,
        "epochs": 1 if pilot else 160,
        "microbatch_size": microbatch,
        "accumulation_steps": accumulation,
        "effective_batch": 64,
        "init_path": str(init_path),
        "init_sha256": pairing["init_sha256"],
        "expected_init_state_hash": pairing["init_state_hash"],
        "init_manifest_path": str(V4 / "qa/COMMON_INIT_AND_RECIPE.json"),
        "init_manifest_sha256": sha_file(V4 / "qa/COMMON_INIT_AND_RECIPE.json"),
        "train_npz_sha256": source_map["data"]["train_npz_sha256"],
        "normalizer_sha256": source_map["data"]["normalizer_sha256"],
        "metric_lock_sha256": source_map["data"]["metric_scale_lock_sha256"],
        "teacher_bank_seal_sha256": sha_file(ROOT / "protocol/TEACHER_BANK_SEAL.json"),
        "target_lock_sha256": sha_file(ROOT / "protocol/TARGET_LOCK.json"),
        "training_target_path": target_entry["target_path"],
        "training_target_sha256": target_entry["target_sha256"],
        "teacher_V_path": teacher_v["prediction_path"],
        "teacher_V_sha256": teacher_v["prediction_sha256"],
        "teacher_T_path": teacher_t["prediction_path"],
        "teacher_T_sha256": teacher_t["prediction_sha256"],
        "sampler_orders_path": pairing["sampler_orders_path"],
        "sampler_orders_sha256": pairing["sampler_orders_sha256"],
        "sampler_epoch1_sha256": sampler_epoch1_hash,
        "ordered_receiver_ids_sha256": target_identity,
        "seed_program_order": pairing["program_order"],
        "job_root": str(job_root),
        "attempt_dir": str(output_root),
        "final_heldout_payload_access": 0,
    }
    job_config = config_dir("pilot" if pilot else "fits") / f"{job_id}.json"
    watcher_path = config_dir("watcher") / f"{job_id}.json"
    atomic_json(job_config, job)
    return job_config, watcher_path, job


def np_load_first_order(path: str) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r", allow_pickle=False)[0].copy()


def run_teacher_bank(source_map: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    train_data = load_train_prefix()
    ordered_hash = ordered_receiver_hash(receiver_rows(train_data))
    preferred: int | None = None
    v4_paths = read_json(V4 / "inventory/RESOLVED_PATHS.json")
    for seed in SEEDS:
        for role in ("S", "V"):
            source = next(row for row in source_map["models"] if int(row["seed"]) == seed and row["role"] == role)
            base = ROOT / "teacher_predictions_train" / f"seed{seed}" / role
            attempt, output_root = next_attempt(base)
            job_id = f"v7_teacher_{role}_s{seed}_a{attempt:02d}"
            job = {
                "schema_version": 1,
                "job_id": job_id,
                "seed": seed,
                "teacher_role": role,
                "video_mode": source["model_mode"],
                "model_path": source["model_path"],
                "model_manifest_path": source["model_manifest_path"],
                "model_sha256_expected": source["model_sha256"],
                "train_npz_sha256": source_map["data"]["train_npz_sha256"],
                "normalizer_sha256": source_map["data"]["normalizer_sha256"],
                "feature_cache_manifest_sha256": source_map["data"]["feature_cache_manifest_sha256"],
                "source_map_sha256": sha_file(ROOT / "protocol/SOURCE_MAP.json"),
                "ordered_receiver_ids_sha256": ordered_hash,
                "batch_size": 32,
                "output_root": str(output_root),
                "final_heldout_payload_access": 0,
            }
            job_config = config_dir("teacher") / f"{job_id}.json"
            watcher_path = config_dir("watcher") / f"{job_id}.json"
            atomic_json(job_config, job)
            basis = "accepted V4 inference measured <=100 MiB; preregistered 160 MiB for batch32"
            watch = watcher_payload(
                job_id=job_id,
                protocol_id="V7_FIXED_TRAIN_TEACHER_REPLAY_V1",
                output_root=output_root,
                command=[str(PYTHON), str(ROOT / "code/v7_teacher.py"), "--job-json", str(job_config)],
                measured_peak_mib=160,
                basis=basis,
                preferred_gpu=preferred,
                marker="PREDICTION_COMPLETE.json",
                log_path=ROOT / "logs/jobs" / f"{job_id}.log",
            )
            result = run_watcher_job(
                phase="teacher_train_replay",
                job_id=job_id,
                job_config_path=job_config,
                watch_config_path=watcher_path,
                watcher_payload_value=watch,
                marker_path=output_root / "PREDICTION_COMPLETE.json",
                measured_peak_mib=160,
                resource_basis=basis,
                output_root=output_root,
            )
            if not result["passed"]:
                raise RuntimeError(f"teacher train replay failed: {job_id}")
            receipt = result["marker"]
            preferred = int(receipt["physical_gpu"]) if str(receipt.get("physical_gpu", "")).isdigit() else preferred
            entries.append({
                "job_id": job_id,
                "seed": seed,
                "teacher_role": role,
                "status": "PASS",
                "prediction_path": receipt["prediction_path"],
                "prediction_sha256": receipt["prediction_sha256"],
                "prediction_rows_path": receipt["prediction_rows_path"],
                "prediction_rows_sha256": receipt["prediction_rows_sha256"],
                "completion_path": str(output_root / "PREDICTION_COMPLETE.json"),
                "completion_sha256": sha_file(output_root / "PREDICTION_COMPLETE.json"),
                "physical_gpu": receipt["physical_gpu"],
                "logical_device": receipt["logical_device"],
                "cuda_visible_devices": receipt["cuda_visible_devices"],
            })
    if len(entries) != 6:
        raise RuntimeError(f"expected six train teacher replays, got {len(entries)}")
    atomic_json(ROOT / "coordination/TEACHER_JOB_INDEX.json", {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "entries": entries,
        "accepted_source_prediction_bundles": 6,
        "old_roots_read_only": True,
        "final_heldout_payload_access": 0,
    })
    return entries


def run_cpu_target_builder() -> None:
    teacher_seal = ROOT / "protocol/TEACHER_BANK_SEAL.json"
    target_lock = ROOT / "protocol/TARGET_LOCK.json"
    if teacher_seal.exists() and target_lock.exists():
        if read_json(teacher_seal).get("status") == "PASS" and read_json(target_lock).get("status") == "LOCKED":
            print(f"[{utc_now()}] reusing sealed V7 train teacher bank and targets", flush=True)
            return
        raise RuntimeError("partial teacher/target seal exists; preserve and inspect")
    if teacher_seal.exists() or target_lock.exists():
        raise RuntimeError("one of teacher or target seals exists without the other")
    sys.path.insert(0, str(ROOT / "code"))
    from v7_build_targets import build
    build()


def run_pilot_once(
    seed: int,
    *,
    attempt: int,
    microbatch: int,
    accumulation: int,
    preferred_gpu: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    pairing = read_json(ROOT / "protocol/INIT_SAMPLER_PAIRING.json")
    seed_pair = next(row for row in pairing["seeds"] if int(row["seed"]) == seed)
    receipts: list[dict[str, Any]] = []
    failed: dict[str, Any] | None = None
    preferred = preferred_gpu
    for program in seed_pair["program_order"]:
        base = ROOT / "tmp/STAGE0" / f"seed{seed}" / program
        job_root = base
        output_root = base / f"attempt_{attempt:02d}"
        output_root.mkdir(parents=True, exist_ok=True)
        job_config, watcher_path, job = save_fit_config(
            seed=seed, program=program, pilot=True, attempt=attempt,
            microbatch=microbatch, accumulation=accumulation,
            output_root=output_root, job_root=job_root,
        )
        basis = "V4 accepted formal fits allocated 137.5/reserved 176 MiB; V3 max reserved 276 MiB; preregistered 384 MiB; Stage0 measures V7"
        watch = watcher_payload(
            job_id=job["job_id"],
            protocol_id="V7_STAGE0_FIXED_SOFT_TARGET_QA",
            output_root=output_root,
            command=[str(PYTHON), str(ROOT / "code/v7_train.py"), "--job-json", str(job_config)],
            measured_peak_mib=384,
            basis=basis,
            preferred_gpu=preferred,
            marker="PILOT_RECEIPT.json",
            log_path=ROOT / "logs/jobs" / f"{job['job_id']}.log",
        )
        result = run_watcher_job(
            phase="stage0_pilot",
            job_id=job["job_id"],
            job_config_path=job_config,
            watch_config_path=watcher_path,
            watcher_payload_value=watch,
            marker_path=output_root / "PILOT_RECEIPT.json",
            measured_peak_mib=384,
            resource_basis=basis,
            output_root=output_root,
        )
        if not result["passed"]:
            failed = {
                "program_id": program,
                "job_root": str(job_root),
                "output_root": str(output_root),
                "oom_fallback_path": str(job_root / "OOM_FALLBACK.json"),
                "status_path": str(STATUS_DIR / f"{job['job_id']}.json"),
                "return_code": result["return_code"],
            }
            if not (job_root / "OOM_FALLBACK.json").is_file():
                return receipts, failed, None
            return receipts, failed, {"physical_gpu": preferred}
        receipt = result["marker"]
        preferred = int(receipt["physical_gpu"]) if str(receipt.get("physical_gpu", "")).isdigit() else preferred
        receipts.append({"program_id": program, "seed": seed, "receipt": receipt, "path": str(output_root / "PILOT_RECEIPT.json")})
    return receipts, None, None


def run_stage0() -> dict[str, Any]:
    existing = ROOT / "qa/STAGE0.json"
    if existing.is_file() and read_json(existing).get("status") == "PASS":
        return read_json(existing)
    seed = 202610131
    first, failure, _ = run_pilot_once(seed, attempt=1, microbatch=64, accumulation=1, preferred_gpu=None)
    fallback_used = False
    attempts = first[:]
    if failure is not None:
        if not Path(failure["oom_fallback_path"]).is_file():
            raise RuntimeError(f"Stage0 non-OOM failure: {failure}")
        fallback_used = True
        retry, retry_failure, _ = run_pilot_once(seed, attempt=2, microbatch=32, accumulation=2, preferred_gpu=None)
        attempts.extend(retry)
        if retry_failure is not None:
            raise RuntimeError(f"Stage0 paired 32x2 retry failed: {retry_failure}")
    if len(attempts) < 3:
        raise RuntimeError(f"Stage0 did not produce three accepted pilot receipts: {len(attempts)}")
    selected_by_program: dict[str, dict[str, Any]] = {}
    for row in attempts:
        program = str(row["program_id"])
        if row["receipt"].get("status") == "PASS":
            selected_by_program[program] = row
    if set(selected_by_program) != {"GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"}:
        raise RuntimeError("Stage0 does not contain one accepted receipt for every program")
    receipts = [selected_by_program[program]["receipt"] for program in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")]
    init_hashes = {r["init_state_hash"] for r in receipts}
    rng_hashes = {r["cuda_rng_initial_sha256"] for r in receipts}
    order_hashes = {r["sampler_order_epoch1_sha256"] for r in receipts}
    if len(init_hashes) != 1 or len(rng_hashes) != 1 or len(order_hashes) != 1:
        raise RuntimeError("Stage0 target arms do not share INIT/sampler/initial dropout state")
    if any(int(r["optimizer_updates"]) != 2 or r.get("pilot_weights_discarded") is not True for r in receipts):
        raise RuntimeError("Stage0 receipt violates two-update/discarded-weight contract")
    cpu_qa = read_json(ROOT / "qa/CPU_PREFLIGHT.json")
    if cpu_qa.get("status") != "PASS":
        raise RuntimeError("CPU synthetic/schema QA did not pass before Stage0")
    selected_rows = [selected_by_program[p] for p in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")]
    devices = [str(row["receipt"]["physical_gpu"]) for row in selected_rows]
    peaks = [float(row["receipt"].get("peak_memory_reserved_mib", 0.0)) for row in selected_rows]
    stage = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "cpu_synthetic_qa_path": str(ROOT / "qa/CPU_PREFLIGHT.json"),
        "cpu_synthetic_qa_sha256": sha_file(ROOT / "qa/CPU_PREFLIGHT.json"),
        "seed": seed,
        "pilot_programs": ["GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"],
        "optimizer_updates_per_program": 2,
        "accepted_pilot_receipts": [
            {"program_id": row["program_id"], "path": row["path"], "sha256": sha_file(Path(row["path"]))}
            for row in selected_rows
        ],
        "discarded_pilot_weights": True,
        "failed_attempts_preserved": True,
        "oom_uniform_microbatch_fallback_used": fallback_used,
        "same_init_hash": next(iter(init_hashes)),
        "same_sampler_order_hash": next(iter(order_hashes)),
        "same_initial_cuda_rng_hash": next(iter(rng_hashes)),
        "physical_gpus_by_program": devices,
        "same_physical_gpu": len(set(devices)) == 1,
        "peak_memory_reserved_mib": max(peaks),
        "effective_batch": 64,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "qa/STAGE0.json", stage)
    return stage


def fresh_reload(model_path: Path, manifest_path: Path, receipt_path: Path, seed: int) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    job_path = receipt_path.with_name("fresh_reload_job.json")
    atomic_json(job_path, {
        "model_path": str(model_path),
        "seed": seed,
        "video_mode": "TS11-D",
        "expected_state_hash": manifest["final_state_hash"],
        "receipt_path": str(receipt_path),
    })
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "code/v7_reload_check.py"), "--job-json", str(job_path)],
        cwd=str(PROJECT), env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError(f"fresh CPU model reload QA failed for {model_path}: {result.stdout[-1500:]}")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS" or receipt.get("fresh_process") is not True:
        raise RuntimeError(f"fresh CPU model reload receipt not PASS: {model_path}")
    return receipt


def run_fit_once(
    seed: int,
    program: str,
    *,
    attempt: int,
    microbatch: int,
    accumulation: int,
    preferred_gpu: int | None,
    resource_peak_mib: int,
) -> dict[str, Any]:
    base = ROOT / "model_only" / program / f"seed{seed}"
    job_root = base
    output_root = base / f"attempt_{attempt:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    job_config, watcher_path, job = save_fit_config(
        seed=seed, program=program, pilot=False, attempt=attempt,
        microbatch=microbatch, accumulation=accumulation,
        output_root=output_root, job_root=job_root,
    )
    basis = (
        "V4 preregistered 384 MiB training estimate from accepted V2/V3 peak "
        f"and V7 Stage0 reserved peak {resource_peak_mib} MiB; watcher adds policy safety margin"
    )
    watch = watcher_payload(
        job_id=job["job_id"],
        protocol_id="V7_SINGLE_FUSION_HEAD_SOFT_TARGET_V1",
        output_root=output_root,
        command=[str(PYTHON), str(ROOT / "code/v7_train.py"), "--job-json", str(job_config)],
        measured_peak_mib=max(384, int(math.ceil(resource_peak_mib * 1.25))),
        basis=basis,
        preferred_gpu=preferred_gpu,
        marker="RUN_COMPLETE.json",
        log_path=ROOT / "logs/jobs" / f"{job['job_id']}.log",
        max_oom_requeues=0,
    )
    estimate = int(watch["resource_estimate"]["measured_peak_mib"])
    result = run_watcher_job(
        phase="formal_fit" if attempt == 1 else "formal_fit_paired_oom_retry",
        job_id=job["job_id"],
        job_config_path=job_config,
        watch_config_path=watcher_path,
        watcher_payload_value=watch,
        marker_path=output_root / "RUN_COMPLETE.json",
        measured_peak_mib=estimate,
        resource_basis=basis,
        output_root=output_root,
    )
    if not result["passed"]:
        return {
            "passed": False,
            "seed": seed,
            "program_id": program,
            "attempt": attempt,
            "output_root": str(output_root),
            "job_root": str(job_root),
            "oom_fallback_path": str(job_root / "OOM_FALLBACK.json"),
            "return_code": result["return_code"],
            "watcher_status": result["watcher_status"],
        }
    manifest_path = output_root / "MODEL_MANIFEST.json"
    complete_path = output_root / "RUN_COMPLETE.json"
    model_manifest = read_json(manifest_path)
    reload_receipt_path = output_root / "FRESH_RELOAD_QA.json"
    reload_receipt = fresh_reload(Path(model_manifest["model_file"]), manifest_path, reload_receipt_path, seed)
    return {
        "passed": True,
        "seed": seed,
        "program_id": program,
        "attempt": attempt,
        "output_root": str(output_root),
        "job_id": job["job_id"],
        "physical_gpu": model_manifest["physical_gpu"],
        "microbatch_size": model_manifest["microbatch_size"],
        "accumulation_steps": model_manifest["accumulation_steps"],
        "init_state_hash": model_manifest["init_state_hash"],
        "sampler_orders_sha256": model_manifest["sampler_orders_sha256"],
        "cuda_rng_initial_sha256": model_manifest["cuda_rng_initial_sha256"],
        "model_path": model_manifest["model_file"],
        "model_sha256": model_manifest["model_file_sha256"],
        "model_manifest_path": str(manifest_path),
        "model_manifest_sha256": sha_file(manifest_path),
        "completion_path": str(complete_path),
        "completion_sha256": sha_file(complete_path),
        "training_curves_path": str(output_root / "TRAINING_CURVES.csv"),
        "training_curves_sha256": sha_file(output_root / "TRAINING_CURVES.csv"),
        "fresh_reload_receipt_path": str(reload_receipt_path),
        "fresh_reload_receipt_sha256": sha_file(reload_receipt_path),
        "final_heldout_payload_access": 0,
    }


def run_formal_matrix(stage0: dict[str, Any]) -> dict[str, Any]:
    accepted: dict[tuple[str, int], dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    pairing = read_json(ROOT / "protocol/INIT_SAMPLER_PAIRING.json")
    stage_peaks = [float(row.get("peak_memory_reserved_mib", 0.0)) for row in stage0["accepted_pilot_receipts"]]
    stage_peak = max(stage_peaks, default=0.0)
    resource_peak = max(384, int(math.ceil(stage_peak * 1.25)))
    for seed in SEEDS:
        seed_pair = next(row for row in pairing["seeds"] if int(row["seed"]) == seed)
        program_order = list(seed_pair["program_order"])
        preferred: int | None = None
        seed_attempts: list[dict[str, Any]] = []
        fallback = False
        failure: dict[str, Any] | None = None
        for program in program_order:
            result = run_fit_once(
                seed, program, attempt=1, microbatch=64, accumulation=1,
                preferred_gpu=preferred, resource_peak_mib=resource_peak,
            )
            seed_attempts.append(result)
            attempts.append(result)
            if not result["passed"]:
                if Path(result["oom_fallback_path"]).is_file():
                    fallback = True
                    failure = result
                    break
                raise RuntimeError(f"formal fit failed for non-OOM reason: {result}")
            preferred = int(result["physical_gpu"]) if str(result["physical_gpu"]).isdigit() else preferred
        if fallback:
            # Preserve the attempt that OOMed and re-run all three paired arms
            # from the same INIT with the registered 32x2 effective batch.
            preferred = None
            for program in program_order:
                retry = run_fit_once(
                    seed, program, attempt=2, microbatch=32, accumulation=2,
                    preferred_gpu=preferred, resource_peak_mib=resource_peak,
                )
                attempts.append(retry)
                if not retry["passed"]:
                    raise RuntimeError(f"paired 32x2 OOM retry failed for {program}/{seed}: {retry}")
                preferred = int(retry["physical_gpu"]) if str(retry["physical_gpu"]).isdigit() else preferred
                accepted[(program, seed)] = retry
        else:
            if len(seed_attempts) != 3:
                raise RuntimeError(f"seed {seed} did not execute all three arms")
            for result in seed_attempts:
                accepted[(str(result["program_id"]), seed)] = result
        group = [accepted[(program, seed)] for program in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")]
        if len({row["init_state_hash"] for row in group}) != 1:
            raise RuntimeError(f"paired common INIT failed for seed {seed}")
        if len({row["sampler_orders_sha256"] for row in group}) != 1:
            raise RuntimeError(f"paired sampler order failed for seed {seed}")
        if len({row["cuda_rng_initial_sha256"] for row in group}) != 1:
            raise RuntimeError(f"paired initial CUDA RNG state failed for seed {seed}")
        if len({(row["microbatch_size"], row["accumulation_steps"]) for row in group}) != 1:
            raise RuntimeError(f"paired effective-batch realization differs within seed {seed}")

    if set(accepted) != {(program, seed) for program in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT") for seed in SEEDS}:
        raise RuntimeError("accepted formal matrix does not contain nine paired fits")
    entries = []
    for seed in SEEDS:
        seed_pair = next(row for row in pairing["seeds"] if int(row["seed"]) == seed)
        for program in seed_pair["program_order"]:
            entries.append(accepted[(program, seed)])
    index = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "accepted_fit_count": 9,
        "expected_fit_count": 9,
        "formal_epochs": 160,
        "optimizer_updates_per_fit": 3200,
        "total_accepted_optimizer_updates": 28800,
        "failed_attempt_count": sum(not row.get("passed") for row in attempts),
        "oom_fallback_seed_groups": sorted({int(row["seed"]) for row in attempts if row.get("attempt") == 2}),
        "attempts": attempts,
        "entries": entries,
        "last_checkpoint_only": True,
        "dev_selection_or_early_stopping": False,
        "teacher_parameters_updated": 0,
        "encoder_training": 0,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json", index)
    outcomes = []
    for row in entries:
        outcomes.append({
            "seed": row["seed"],
            "program_id": row["program_id"],
            "attempt": row["attempt"],
            "status": "PASS",
            "model_path": row["model_path"],
            "model_sha256": row["model_sha256"],
            "physical_gpu": row["physical_gpu"],
            "microbatch_size": row["microbatch_size"],
            "accumulation_steps": row["accumulation_steps"],
            "optimizer_steps": 3200,
            "fresh_process_reload": "PASS",
        })
    write_csv(ROOT / "protocol/MATRIX_9_ACCEPTED.csv", list(outcomes[0]), outcomes)
    return index


def run_dev_predictions(fit_index: dict[str, Any]) -> dict[str, Any]:
    entries = []
    preferred: int | None = None
    for fit in fit_index["entries"]:
        seed, program = int(fit["seed"]), str(fit["program_id"])
        base = ROOT / "predictions" / program / f"seed{seed}"
        attempt, output_root = next_attempt(base)
        job_id = f"v7_pred_{program}_s{seed}_a{attempt:02d}"
        model_manifest = read_json(Path(fit["model_manifest_path"]))
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "phase": "prediction",
            "program_id": program,
            "model_id": program,
            "condition": "MATCHED",
            "seed": seed,
            "task_ids": list(TASKS),
            "video_mode": "TS11-D",
            "model_path": fit["model_path"],
            "model_manifest_path": fit["model_manifest_path"],
            "model_sha256_expected": fit["model_sha256"],
            "data_path": str(DEV_NPZ),
            "train_data_path": str(TRAIN_NPZ),
            "normalizer_path": str(NORMALIZER),
            "eval_donor_maps_path": str(V4 / "protocol/EVAL_DONOR_MAPS.csv"),
            "resolved_paths_path": str(V4 / "inventory/RESOLVED_PATHS.json"),
            "output_root": str(output_root),
            "batch_size": 32,
            "expected_rows": 185,
            "final_heldout_payload_access": 0,
            "teachers_used_in_student_forward": False,
            "dev_target_values_read": 0,
        }
        job_config = config_dir("inference") / f"{job_id}.json"
        watcher_path = config_dir("watcher") / f"{job_id}.json"
        atomic_json(job_config, job)
        basis = "accepted V4 matched inference preregistered 160 MiB, batch32; same TS11-D head and cached Q0 prefix"
        watch = watcher_payload(
            job_id=job_id,
            protocol_id="V7_SINGLE_HEAD_TARGET_BLIND_DEV_PREDICTION",
            output_root=output_root,
            command=[str(PYTHON), str(V4 / "code/v4_infer.py"), "--job-json", str(job_config)],
            measured_peak_mib=160,
            basis=basis,
            preferred_gpu=preferred,
            marker="PREDICTION_COMPLETE.json",
            log_path=ROOT / "logs/jobs" / f"{job_id}.log",
        )
        result = run_watcher_job(
            phase="student_dev_prediction",
            job_id=job_id,
            job_config_path=job_config,
            watch_config_path=watcher_path,
            watcher_payload_value=watch,
            marker_path=output_root / "PREDICTION_COMPLETE.json",
            measured_peak_mib=160,
            resource_basis=basis,
            output_root=output_root,
        )
        if not result["passed"]:
            raise RuntimeError(f"student target-blind dev inference failed: {job_id}")
        receipt = result["marker"]
        preferred = int(receipt["physical_gpu"]) if str(receipt.get("physical_gpu", "")).isdigit() else preferred
        entries.append({
            "job_id": job_id,
            "program_id": program,
            "model_id": program,
            "seed": seed,
            "condition": "MATCHED",
            "model_path": fit["model_path"],
            "model_sha256": fit["model_sha256"],
            "model_manifest_path": fit["model_manifest_path"],
            "model_manifest_sha256": fit["model_manifest_sha256"],
            "prediction_path": receipt["prediction_path"],
            "prediction_sha256": receipt["prediction_sha256"],
            "prediction_rows_path": receipt["prediction_rows_path"],
            "prediction_rows_sha256": receipt["prediction_rows_sha256"],
            "completion_path": str(output_root / "PREDICTION_COMPLETE.json"),
            "completion_sha256": sha_file(output_root / "PREDICTION_COMPLETE.json"),
            "row_count": 185,
            "prediction_shape": [185, 30, 28],
            "physical_gpu": receipt["physical_gpu"],
            "dev_target_arrays_requested": False,
            "teachers_used_in_student_forward": False,
            "final_heldout_payload_access": 0,
        })
    if len(entries) != 9:
        raise RuntimeError(f"expected nine new student dev prediction bundles, got {len(entries)}")
    seal = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "study": "post_V6_single_fusion_head_supervised_soft_target_development",
        "planned_student_jobs": 9,
        "accepted_student_jobs": len(entries),
        "missing_student_jobs": 0,
        "nonempty_seal": True,
        "entries": entries,
        "all_ids_joined": True,
        "all_shapes_and_values_finite": True,
        "student_forward_is_single_head": True,
        "student_forward_uses_teacher_predictions": False,
        "student_forward_reads_dev_targets": False,
        "dev_target_values_read_before_seal": 0,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "qa/PREDICTION_SEAL.json", seal)
    return seal


def run_timing(preferred_gpu: int | None) -> dict[str, Any]:
    base = ROOT / "tmp/timing"
    attempt, output_root = next_attempt(base)
    job_id = f"v7_inference_timing_s{202610131}_a{attempt:02d}"
    job = {"schema_version": 1, "job_id": job_id, "output_root": str(output_root), "seed": 202610131}
    job_config = config_dir("timing") / f"{job_id}.json"
    watcher_path = config_dir("watcher") / f"{job_id}.json"
    atomic_json(job_config, job)
    basis = "accepted V4 matched inference preregistered 160 MiB; fixed batch32 cached-prefix fixture"
    watch = watcher_payload(
        job_id=job_id,
        protocol_id="V7_CACHED_INPUT_INFERENCE_PROFILE",
        output_root=output_root,
        command=[str(PYTHON), str(ROOT / "code/v7_timing.py"), "--job-json", str(job_config)],
        measured_peak_mib=160,
        basis=basis,
        preferred_gpu=preferred_gpu,
        marker="INFERENCE_TIMING.json",
        log_path=ROOT / "logs/jobs" / f"{job_id}.log",
    )
    result = run_watcher_job(
        phase="cached_inference_timing",
        job_id=job_id,
        job_config_path=job_config,
        watch_config_path=watcher_path,
        watcher_payload_value=watch,
        marker_path=output_root / "INFERENCE_TIMING.json",
        measured_peak_mib=160,
        resource_basis=basis,
        output_root=output_root,
    )
    if result["passed"]:
        receipt = result["marker"]
        rows = []
        for row in receipt["results"]:
            rows.append({
                "status": "PASS",
                "seed": receipt["seed"],
                "path": row["path"],
                "warmups": row["warmups"],
                "repeats": row["repeats"],
                "batch_size": row["batch_size"],
                "dtype": row["dtype"],
                "median_seconds": row["median_seconds"],
                "q1_seconds": row["q1_seconds"],
                "q3_seconds": row["q3_seconds"],
                "iqr_seconds": row["iqr_seconds"],
                "parameter_count": row["parameter_count"],
                "physical_gpu": receipt["physical_gpu"],
                "fixture": receipt["fixture"],
                "timing_scope": receipt["latency_scope"],
                "source_receipt_path": str(output_root / "INFERENCE_TIMING.json"),
            })
        write_csv(ROOT / "results/INFERENCE_PROFILE.csv", list(rows[0]), rows)
        return {"status": "PASS", "receipt_path": str(output_root / "INFERENCE_TIMING.json"), "receipt_sha256": sha_file(output_root / "INFERENCE_TIMING.json"), "rows": len(rows)}
    failed = {
        "status": "UNAVAILABLE",
        "job_id": job_id,
        "watcher_status_path": str(STATUS_DIR / f"{job_id}.json"),
        "reason": "timing child did not produce a PASS receipt",
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "qa/INFERENCE_TIMING_STATUS.json", failed)
    write_csv(ROOT / "results/INFERENCE_PROFILE.csv", ["status", "seed", "path", "reason"], [
        {"status": "UNAVAILABLE", "seed": 202610131, "path": "all", "reason": failed["reason"]},
    ])
    return failed


def main() -> int:
    launch_command = os.environ.get("V7_LAUNCH_COMMAND", "")
    session = os.environ.get("V7_TMUX_SESSION", "")
    log_path = os.environ.get("V7_LOG_PATH", "")
    if not session or not log_path or not launch_command:
        raise RuntimeError("V7 long GPU pipeline must run from its recorded tmux session")
    owner, initial_state = require_current_sources()
    gate = require_host_gate()
    prior_gpu = ROOT / "coordination/GPU_EXECUTION.json"
    resume_attempt = 0
    prior_failure_path = None
    if prior_gpu.exists():
        prior = read_json(prior_gpu)
        if prior.get("status") == "GPU_PHASE_COMPLETE":
            print(json.dumps({"status": "already_complete", "path": str(prior_gpu)}))
            return 0
        if prior.get("status") == "running":
            raise RuntimeError("existing GPU execution record is running; do not take over an active writer")
        if prior.get("status") == "partial":
            prior_state = read_json(OWNER_STATE_PATH)
            if int(prior_state.get("active_jobs", -1)) != 0 or int(prior_state.get("active_gpu_jobs", -1)) != 0:
                raise RuntimeError("partial root still reports active jobs; refusing writer takeover")
            if not prior_state.get("owner_lock_acquired"):
                raise RuntimeError("partial root owner lock is not held by this V7 writer")
            resume_attempt = int(prior.get("recovery_attempt", 0)) + 1
            prior_failure_path = ROOT / "coordination/GPU_PIPELINE_FAILURE.json"
            if not prior_failure_path.is_file():
                raise RuntimeError("partial execution lacks its preserved failure receipt")
            archived = ROOT / "coordination" / f"GPU_EXECUTION_ATTEMPT_{resume_attempt:02d}.json"
            if archived.exists():
                raise RuntimeError(f"recovery attempt record already exists: {archived}")
            atomic_json(archived, prior)
        else:
            raise RuntimeError(f"unexpected existing GPU execution state: {prior.get('status')}")
    started = utc_now()
    owner.update({"owner_pid": os.getpid(), "host": os.environ.get("EE6008_HOST_LABEL", "shared-host"), "owner_uid": os.getuid(), "owner_lock_acquired": True, "active_tmux_session": session, "updated_utc": started})
    atomic_json(ROOT / "coordination/OWNER.lock", owner)
    root_state = read_json(OWNER_STATE_PATH)
    root_state.update({
        "status": "running",
        "owner_pid": os.getpid(),
        "phase": "GPU_teacher_train_replay",
        "active_jobs": 0,
        "active_gpu_jobs": 0,
        "main_pid": os.getpid(),
        "tmux_session": session,
        "console_log_path": log_path,
        "launch_command": launch_command,
        "host_resource_gate_sha256": sha_file(ROOT / "coordination/HOST_RESOURCE_GATE.json"),
        "started_utc": started,
        "updated_utc": started,
    })
    atomic_json(OWNER_STATE_PATH, root_state)
    run_context_path = ROOT / "RUN_CONTEXT.json"
    run_context = read_json(run_context_path)
    run_context.update({
        "tmux_session": session, "main_pid": os.getpid(), "launch_command": launch_command,
        "console_log_path": log_path, "working_directory": str(PROJECT),
        "gpu_resource_gate_sha256": sha_file(ROOT / "coordination/HOST_RESOURCE_GATE.json"),
        "gpu_pipeline_started_utc": started, "gpu_queries_before_host_approval": 0,
    })
    atomic_json(run_context_path, run_context)
    atomic_json(ROOT / "coordination/GPU_EXECUTION.json", {
        "schema_version": 1,
        "status": "running",
        "recovery_attempt": resume_attempt,
        "previous_failure_path": str(prior_failure_path) if prior_failure_path else None,
        "started_utc": started,
        "host": os.environ.get("EE6008_HOST_LABEL", "shared-host"),
        "user": 'project-team',
        "python_bin": str(PYTHON),
        "working_directory": str(PROJECT),
        "repo_root": str(REPO),
        "branch_commit_dirty_state": "UNKNOWN; git unavailable and no formal repository source files are modified",
        "tmux_session": session,
        "main_pid": os.getpid(),
        "launch_command": launch_command,
        "console_log_path": log_path,
        "reattach_command": f"tmux attach -t {session}",
        "log_view_command": f"tail -f '{log_path}'",
        "safe_stop_method": f"tmux send-keys -t {session} C-c",
        "gpu_resource_gate": gate,
        "shared_watcher": str(WATCHER),
        "shared_watcher_sha256": sha_file(WATCHER),
        "maximum_project_children": 2,
        "current_pipeline_project_children": 1,
        "final_heldout_payload_access": 0,
    })
    try:
        source_map, _, _ = protocol_inputs()
        if not (ROOT / "qa/CPU_PREFLIGHT.json").is_file() or read_json(ROOT / "qa/CPU_PREFLIGHT.json").get("status") != "PASS":
            raise RuntimeError("CPU schema/synthetic preflight must pass before any GPU stage")
        teacher_index = ROOT / "coordination/TEACHER_JOB_INDEX.json"
        if teacher_index.exists():
            existing = read_json(teacher_index)
            if existing.get("status") != "PASS" or len(existing.get("entries", [])) != 6:
                raise RuntimeError("existing teacher job index is incomplete")
            teacher_entries = existing["entries"]
        else:
            teacher_entries = run_teacher_bank(source_map)
        state_update(phase="CPU_teacher_target_seal", active_jobs=0, active_gpu_jobs=0, teacher_bundles_accepted=6)
        run_cpu_target_builder()

        stage0 = run_stage0()
        fit_index = run_formal_matrix(stage0)
        fit_entries = fit_index["entries"]
        preferred = next((int(row["physical_gpu"]) for row in fit_entries if str(row.get("physical_gpu", "")).isdigit()), None)

        prediction_seal = run_dev_predictions(fit_index)
        timing = run_timing(preferred)
        all_gpu_rows = []
        if RESOURCE_LEDGER_PATH.exists():
            with RESOURCE_LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
                all_gpu_rows = list(csv.DictReader(handle))
        completed = utc_now()
        execution = read_json(ROOT / "coordination/GPU_EXECUTION.json")
        execution.update({
            "status": "GPU_PHASE_COMPLETE",
            "completed_utc": completed,
            "teacher_train_bundles_accepted": 6,
            "teacher_parameters_updated": 0,
            "outer_train_Y_read_after_teacher_bank_seal": True,
            "target_lock_sha256": sha_file(ROOT / "protocol/TARGET_LOCK.json"),
            "stage0_status": stage0["status"],
            "new_head_fits_accepted": int(fit_index["accepted_fit_count"]),
            "formal_optimizer_updates_accepted": int(fit_index["total_accepted_optimizer_updates"]),
            "student_dev_prediction_jobs_accepted": int(prediction_seal["accepted_student_jobs"]),
            "student_dev_task_views": 45,
            "prediction_seal_sha256": sha_file(ROOT / "qa/PREDICTION_SEAL.json"),
            "inference_timing_status": timing["status"],
            "resource_ledger_rows": len(all_gpu_rows),
            "peak_reserved_mib_training": max(
                [float(row["peak_reserved_mib"]) for row in all_gpu_rows if row.get("phase", "").startswith("formal_fit") and row.get("peak_reserved_mib") not in {"", None}] or [0.0]
            ),
            "old_root_writes": 0,
            "dev_target_values_read_by_GPU_pipeline": 0,
            "final_heldout_payload_access": 0,
        })
        atomic_json(ROOT / "coordination/GPU_EXECUTION.json", execution)
        state_update(
            status="running",
            phase="sealed_waiting_for_CPU_scoring",
            active_jobs=0,
            active_gpu_jobs=0,
            new_head_fits_accepted=9,
            student_dev_predictions_accepted=9,
            prediction_seal_status="PASS",
            completed_gpu_utc=completed,
        )
        print(f"[{completed}] V7 GPU matrix and nonempty target-blind prediction seal complete", flush=True)
        return 0
    except Exception as exc:
        failed_utc = utc_now()
        state_update(
            status="partial",
            phase="GPU_pipeline_failure",
            active_jobs=0,
            active_gpu_jobs=0,
            last_error=f"{type(exc).__name__}: {exc}",
            failed_utc=failed_utc,
            final_heldout_payload_access=0,
        )
        execution = read_json(ROOT / "coordination/GPU_EXECUTION.json")
        execution.update({
            "status": "partial",
            "failed_utc": failed_utc,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "final_heldout_payload_access": 0,
        })
        atomic_json(ROOT / "coordination/GPU_EXECUTION.json", execution)
        failure_base = ROOT / "coordination/GPU_PIPELINE_FAILURE.json"
        failure_path = failure_base
        if failure_path.exists():
            suffix = 2
            while True:
                failure_path = ROOT / "coordination" / f"GPU_PIPELINE_FAILURE_{suffix:02d}.json"
                if not failure_path.exists():
                    break
                suffix += 1
        atomic_json(failure_path, {
            "schema_version": 1,
            "status": "partial",
            "recovery_attempt": resume_attempt,
            "failed_utc": failed_utc,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "old_root_writes": 0,
            "final_heldout_payload_access": 0,
        })
        print(f"[{failed_utc}] V7 GPU pipeline partial: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
