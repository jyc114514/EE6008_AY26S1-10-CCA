#!/usr/bin/env python3
"""Build and verify the locked corrected-scheduler 12-cell protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
DIAG = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS11-D", "STATE-CTRL")
HORIZONS = (30, 60)
EXPECTED_TRAIN_ROWS = 17757
EXPECTED_TRAIN_EPISODES = 2712
EXPECTED_DEV_ROWS = 2322
EXPECTED_DEV_EPISODES = 603
EXPECTED_UPDATES = 11120
GPU_HOUR_CAP = 12.0
GPU_MEMORY_CAP_MIB = 20 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exact_source_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "bytes": path.stat().st_size,
    }


def verify_split_groups() -> dict[str, Any]:
    v3_path = V3 / "data/V3_MAIN_INPUTS.npz"
    with np.load(v3_path, allow_pickle=False) as archive:
        split = archive["split_code"].astype(np.int8, copy=True)
        episodes = archive["episode_index"].astype(np.int64, copy=True)
        ids = archive["anchor_id"].astype("U96", copy=True)
    if len(split) != 20079 or set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError("unexpected V3 train/dev split population")
    mapping_table = pq.read_table(
        SPLIT_MANIFEST,
        columns=["episode_index", "split", "video_sha256_fingerprint"],
    )
    mapping: dict[int, tuple[str, str]] = {}
    for episode, split_name, fingerprint in zip(
        mapping_table["episode_index"].to_numpy(zero_copy_only=False),
        mapping_table["split"].to_pylist(),
        mapping_table["video_sha256_fingerprint"].to_pylist(),
    ):
        normalized = "dev" if str(split_name) in {"validation", "val"} else str(split_name)
        if fingerprint is None or not str(fingerprint).strip():
            raise AssertionError(f"missing duplicate-safe source fingerprint for episode {episode}")
        key = int(episode)
        value = (normalized, str(fingerprint))
        if key in mapping and mapping[key] != value:
            raise AssertionError(f"conflicting duplicate-safe mapping for episode {key}")
        mapping[key] = value

    train_mask = split == 0
    dev_mask = split == 1
    train_episodes = np.unique(episodes[train_mask])
    dev_episodes = np.unique(episodes[dev_mask])
    if (int(train_mask.sum()), len(train_episodes), int(dev_mask.sum()), len(dev_episodes)) != (
        EXPECTED_TRAIN_ROWS, EXPECTED_TRAIN_EPISODES, EXPECTED_DEV_ROWS, EXPECTED_DEV_EPISODES
    ):
        raise AssertionError("train/dev row or episode counts differ from locked D3-EQ populations")
    train_groups = {mapping[int(ep)][1] for ep in train_episodes if mapping[int(ep)][0] == "train"}
    dev_groups = {mapping[int(ep)][1] for ep in dev_episodes if mapping[int(ep)][0] == "dev"}
    if len(train_groups) != EXPECTED_TRAIN_EPISODES or len(dev_groups) != EXPECTED_DEV_EPISODES:
        raise AssertionError("episode/source-group bijection differs from audited split")
    if train_groups & dev_groups:
        raise AssertionError("duplicate-safe train/dev source groups overlap")
    if len(set(ids[train_mask].tolist())) != EXPECTED_TRAIN_ROWS or len(set(ids[dev_mask].tolist())) != EXPECTED_DEV_ROWS:
        raise AssertionError("train/dev anchor IDs are not unique")
    return {
        "input_sha256": sha_file(v3_path),
        "anchor_id_array_sha256": hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(),
        "split_manifest_sha256": sha_file(SPLIT_MANIFEST),
        "train_rows": EXPECTED_TRAIN_ROWS,
        "train_episodes": EXPECTED_TRAIN_EPISODES,
        "train_source_groups": len(train_groups),
        "dev_rows": EXPECTED_DEV_ROWS,
        "dev_episodes": EXPECTED_DEV_EPISODES,
        "dev_source_groups": len(dev_groups),
        "source_group_overlap": 0,
        "episode_to_source_group_bijection": True,
    }


def validate_old_cells() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    protocol_path = EQ / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    matrix_path = EQ / "protocol/D3_EQ_MATRIX_36.csv"
    seed_registry_path = EQ / "protocol/D3_EQ_SEED_REGISTRY.json"
    protocol_sha = sha_file(protocol_path)
    seed_registry = json.loads(seed_registry_path.read_text(encoding="utf-8"))["seeds"]
    old_source_expectations = {
        "trainer_source_sha256": sha_file(EQ / "code/d3_train_eq.py"),
        "model_source_sha256": sha_file(EQ / "code/d3_model.py"),
        "target_cache_sha256": sha_file(EQ / "d3/train_targets_trainonly.npz"),
        "target_cache_manifest_sha256": sha_file(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"),
        "cache_manifest_sha256": sha_file(OLD / "d3/cache/D3_CACHE_MANIFEST.json"),
        "normalizer_sha256": sha_file(V3 / "data/V3_NORMALIZERS.json"),
        "video_normalizer_sha256": sha_file(OLD / "d3/cache/D3_VIDEO_NORMALIZER.json"),
    }
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        locked_rows = list(csv.DictReader(handle))
    old_matrix = {row["run_id"]: row for row in locked_rows}
    if len(locked_rows) != 36 or len(old_matrix) != 36:
        raise AssertionError("old D3-EQ source matrix must have exactly 36 unique cells")

    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        split = archive["split_code"].astype(np.int8, copy=True)
        all_ids = archive["anchor_id"].astype("U96", copy=True)
        all_eps = archive["episode_index"].astype(np.int64, copy=True)
    train_ids = all_ids[split == 0]
    dev_ids = all_ids[split == 1]
    train_eps = all_eps[split == 0]
    dev_eps = all_eps[split == 1]

    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seed_entries: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = seed_registry[str(seed)]
        init_path = Path(entry["init_state_path"])
        if sha_file(init_path) != entry["init_state_file_sha256"]:
            raise AssertionError(f"INIT file sha mismatch for seed {seed}")
        seed_entries[seed] = {
            "seed": seed,
            "init_state_path": str(init_path),
            "init_state_file_sha256": entry["init_state_file_sha256"],
            "init_state_hash": entry["init_state_hash"],
            "same_initial_state_across_configs_horizons": True,
        }
        source_records.append(exact_source_record(init_path, f"seed_init_{seed}"))

    for config in CONFIGS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                old_id = f"D3EQ-{config}-H{horizon}-S{seed}"
                old = old_matrix.get(old_id)
                if old is None:
                    raise AssertionError(f"matched old cell missing: {old_id}")
                if (old["config"], int(old["seed"]), int(old["horizon_steps"])) != (config, seed, horizon):
                    raise AssertionError(f"old matrix cell identity mismatch: {old_id}")
                run_root = EQ / "d3/runs" / old_id
                environment = json.loads((run_root / "RUN_ENVIRONMENT.json").read_text(encoding="utf-8"))
                complete = json.loads((run_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
                status = json.loads((run_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
                checkpoint = run_root / "checkpoint_last.pt"
                checkpoint_sha = sha_file(checkpoint)
                if complete.get("status") != "PASS" or status.get("status") != "PASS":
                    raise AssertionError(f"old run is not accepted: {old_id}")
                if int(complete.get("epochs", -1)) != 40 or int(complete.get("optimizer_steps", -1)) != EXPECTED_UPDATES:
                    raise AssertionError(f"old run training extent mismatch: {old_id}")
                if complete.get("checkpoint_last_sha256") != checkpoint_sha:
                    raise AssertionError(f"old checkpoint hash mismatch: {old_id}")
                if environment.get("protocol_hash") != protocol_sha or complete.get("protocol_hash") != protocol_sha:
                    raise AssertionError(f"old D3-EQ protocol hash mismatch: {old_id}")
                if environment.get("init_state_hash") != seed_entries[seed]["init_state_hash"]:
                    raise AssertionError(f"old INIT state mismatch: {old_id}")
                if (environment.get("effective_batch"), environment.get("physical_batch_size"),
                    environment.get("accumulation_steps")) != (64, 64, 1):
                    raise AssertionError(f"old effective batch mismatch: {old_id}")

                for key, expected_value in old_source_expectations.items():
                    if environment.get(key) != expected_value:
                        raise AssertionError(f"old as-run {key} mismatch: {old_id}")
                if environment.get("source_target_scope") != "primary_train_parquet_only; no dev target array loaded":
                    raise AssertionError(f"old training target scope mismatch: {old_id}")
                if complete.get("final_test_access") != 0 or environment.get("final_test_access") != 0:
                    raise AssertionError(f"old run final-test denial field mismatch: {old_id}")
                train_side_path = DIAG / "predictions/train" / f"{old_id}.json"
                train_pred_path = train_side_path.with_suffix(".npz")
                dev_side_path = EQ / "d3/predictions" / f"{old_id}.json"
                dev_pred_path = dev_side_path.with_suffix(".npz")
                train_side = json.loads(train_side_path.read_text(encoding="utf-8"))
                dev_side = json.loads(dev_side_path.read_text(encoding="utf-8"))
                if train_side.get("status") != "PASS" or dev_side.get("status") != "PASS":
                    raise AssertionError(f"old prediction sidecar not accepted: {old_id}")
                if (int(train_side.get("rows", -1)), int(dev_side.get("rows", -1))) != (
                    EXPECTED_TRAIN_ROWS, EXPECTED_DEV_ROWS
                ):
                    raise AssertionError(f"old prediction row count mismatch: {old_id}")
                if (sha_file(train_pred_path) != train_side.get("prediction_sha256")
                    or sha_file(dev_pred_path) != dev_side.get("prediction_sha256")):
                    raise AssertionError(f"old prediction checksum mismatch: {old_id}")
                if train_side.get("checkpoint_sha256") != checkpoint_sha or dev_side.get("checkpoint_sha256") != checkpoint_sha:
                    raise AssertionError(f"old predictions are not bound to checkpoint: {old_id}")
                if train_side.get("target_arrays_loaded") is not False:
                    raise AssertionError(f"old train inference was not target-blind: {old_id}")
                with np.load(train_pred_path, allow_pickle=False) as archive:
                    t_ids = archive["anchor_id"].astype("U96", copy=True)
                    t_eps = archive["episode_index"].astype(np.int64, copy=True)
                with np.load(dev_pred_path, allow_pickle=False) as archive:
                    d_ids = archive["anchor_id"].astype("U96", copy=True)
                    d_eps = archive["episode_index"].astype(np.int64, copy=True)
                if not np.array_equal(t_ids, train_ids) or not np.array_equal(t_eps, train_eps):
                    raise AssertionError(f"old train prediction IDs/order mismatch: {old_id}")
                if not np.array_equal(d_ids, dev_ids) or not np.array_equal(d_eps, dev_eps):
                    raise AssertionError(f"old dev prediction IDs/order mismatch: {old_id}")

                new_id = f"D3EQ-CS-{config}-H{horizon}-S{seed}"
                rows.append({
                    "run_id": new_id,
                    "config": config,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "matched_old_run_id": old_id,
                    "init_state_hash": seed_entries[seed]["init_state_hash"],
                    "expected_train_rows": EXPECTED_TRAIN_ROWS,
                    "expected_dev_rows": EXPECTED_DEV_ROWS,
                    "effective_batch": 64,
                    "physical_batch_size": 64,
                    "accumulation_steps": 1,
                    "epochs": 40,
                    "expected_optimizer_updates": EXPECTED_UPDATES,
                    "old_checkpoint_sha256": checkpoint_sha,
                    "old_train_prediction_sha256": train_side["prediction_sha256"],
                    "old_dev_prediction_sha256": dev_side["prediction_sha256"],
                    "old_trainer_source_sha256": environment["trainer_source_sha256"],
                    "old_model_source_sha256": environment["model_source_sha256"],
                    "old_protocol_sha256": protocol_sha,
                })
                for source_path, role in (
                    (run_root / "RUN_ENVIRONMENT.json", f"old_env_{old_id}"),
                    (run_root / "RUN_COMPLETE.json", f"old_complete_{old_id}"),
                    (run_root / "RUN_STATUS.json", f"old_status_{old_id}"),
                    (checkpoint, f"old_checkpoint_{old_id}"),
                    (train_side_path, f"old_train_sidecar_{old_id}"),
                    (train_pred_path, f"old_train_predictions_{old_id}"),
                    (dev_side_path, f"old_dev_sidecar_{old_id}"),
                    (dev_pred_path, f"old_dev_predictions_{old_id}"),
                ):
                    source_records.append(exact_source_record(source_path, role))

    if len(rows) != 12 or len({row["run_id"] for row in rows}) != 12:
        raise AssertionError("matched corrected-scheduler matrix coverage must be exactly 12")
    return rows, source_records, seed_entries


def build_resource_estimate(matched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ledger_path = EQ / "JOB_LEDGER.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    formal_hours = 0.0
    stage0_hours = 0.0
    matched_attempts = 0
    first_seed_formal_hours = 0.0
    other_seed_formal_hours = 0.0
    old_ids = {row["matched_old_run_id"] for row in matched_rows}
    for item in entries:
        job_id = str(item.get("job_id", ""))
        phase = str(item.get("phase", "")).upper()
        for old_id in old_ids:
            if job_id.startswith(old_id + "-FORMAL") and phase == "FORMAL":
                hours = float(item.get("lease_gpu_hours", 0.0))
                formal_hours += hours
                matched_attempts += 1
                if "S20261001" in old_id:
                    first_seed_formal_hours += hours
                else:
                    other_seed_formal_hours += hours
            if job_id.startswith(old_id + "-STAGE0") and phase == "STAGE0":
                stage0_hours += float(item.get("lease_gpu_hours", 0.0))
    if matched_attempts < 12 or formal_hours <= 0:
        raise AssertionError("matched old formal lease-hour evidence is incomplete")
    stage0_old_cells = [
        EQ / "d3/stage0_runs" / f"D3EQ-{config}-H{horizon}-S20261001-STAGE0" / "STAGE0_REPORT.json"
        for config in CONFIGS for horizon in HORIZONS
    ]
    stage0_reports = [json.loads(path.read_text(encoding="utf-8")) for path in stage0_old_cells]
    measured_peak_reserved = max(int(report["peak_reserved_bytes"]) for report in stage0_reports)
    measured_peak_mib = math.ceil(measured_peak_reserved / (1024 * 1024))
    projected_formal = formal_hours * 1.25
    conservative_formal = formal_hours * 2.0
    stage0_projected = max(stage0_hours, 0.01) * 1.5
    projected_total = projected_formal + stage0_projected + 0.10
    if projected_total > GPU_HOUR_CAP or measured_peak_mib > GPU_MEMORY_CAP_MIB:
        status = "budget_insufficient"
    else:
        status = "PASS_PRE_STAGE0"
    return {
        "status": status,
        "gpu_hour_cap": GPU_HOUR_CAP,
        "gpu_memory_cap_mib": GPU_MEMORY_CAP_MIB,
        "old_matched_12_formal_attempts": matched_attempts,
        "old_matched_12_formal_lease_gpu_hours": formal_hours,
        "old_first_seed_4_formal_lease_gpu_hours": first_seed_formal_hours,
        "old_other_8_formal_lease_gpu_hours": other_seed_formal_hours,
        "projected_new_formal_gpu_hours": projected_formal,
        "conservative_new_formal_gpu_hours": conservative_formal,
        "old_matched_stage0_4_lease_gpu_hours": stage0_hours,
        "projected_new_stage0_gpu_hours": stage0_projected,
        "projected_cpu_inference_gpu_hours": 0.0,
        "projected_total_including_optional_dense_reserve": projected_total,
        "measured_old_stage0_peak_reserved_mib": measured_peak_mib,
        "watcher_required_free_mib_initial": max(512, math.ceil(0.10 * measured_peak_mib)) + measured_peak_mib,
        "disk_free_fraction_min": 0.10,
        "final_test_access": 0,
    }


def build_preflight_job(row: dict[str, Any], *, phase: str, measured_peak_mib: int,
                        protocol_sha256: str) -> dict[str, Any]:
    stage0 = phase == "STAGE0"
    cell_id = str(row["job_id"]) if stage0 else str(row["run_id"])
    run_root = TASK / ("runs/stage0" if stage0 else "runs/formal") / cell_id
    command = [
        str(PROJECT / "envs/jyc/bin/python"), str(TASK / "code/physical_gpu_guard.py"),
        str(TASK / "code/d3_train_corrected.py"),
        "--config", str(row["config"]), "--horizon", str(row["horizon_steps"]),
        "--seed", str(row["seed"]), "--run-root", str(run_root),
        "--batch-size", "64", "--accumulation", "1",
    ]
    if stage0:
        command.append("--stage0")
    return {
        "job_id": f"{cell_id}-PREFLIGHT",
        "protocol_id": "CORRECTED_SCHEDULER_MATCHED_12",
        "output_root": str(run_root.resolve()),
        "command": command,
        "resource_estimate": {"measured_peak_mib": int(measured_peak_mib)},
        "preferred_gpu": 2,
        "max_oom_requeues": 0,
        "environment": {"D3_STUDY_PROTOCOL_SHA256": protocol_sha256},
        "log_path": str(TASK / "logs" / f"child_PREFLIGHT_{cell_id}.log"),
        "task_phase": phase,
        "matrix_run_id": cell_id,
        "active_physical_gpu_candidates": [2],
        "own_queue_concurrency": 1,
        "final_test_access": 0,
    }


def write_and_dryrun_job_specs(stage0_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]],
                               resource: dict[str, Any]) -> dict[str, Any]:
    protocol_path = TASK / "PROTOCOL_LOCKED.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = sha_file(protocol_path)
    if protocol.get("resource_budget", {}).get("active_launch_candidates") != [2]:
        raise AssertionError("launch plan requires the locked active candidate GPU2")
    for relative, expected in protocol["task_code_hashes"].items():
        if sha_file(TASK / relative) != expected:
            raise AssertionError(f"task code hash changed after protocol lock: {relative}")
    source_hashes_path = TASK / "protocol/SOURCE_HASHES.json"
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    if source_hashes.get("task_code_sha256") != protocol["task_code_hashes"]:
        raise AssertionError("source hash manifest and protocol code hashes differ")
    if sha_file(source_hashes_path) != protocol["source_hashes_sha256"]:
        raise AssertionError("source hash manifest differs from the protocol lock")

    plan_dir = TASK / "configs/jobs" / f"preflight_{protocol_sha256[:12]}"
    plan_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for phase, rows in (("STAGE0", stage0_rows), ("FORMAL", matrix_rows)):
        for row in rows:
            job = build_preflight_job(row, phase=phase,
                                      measured_peak_mib=int(resource["measured_old_stage0_peak_reserved_mib"]),
                                      protocol_sha256=protocol_sha256)
            path = plan_dir / f"{job['job_id']}.json"
            atomic_json(path, job)
            jobs.append((job, path))
    if len(jobs) != 16 or len({job["job_id"] for job, _ in jobs}) != 16:
        raise AssertionError("launch plan must contain 4 Stage0 and 12 formal unique job keys")

    outcomes = []
    for job, path in jobs:
        status_path = TASK / "logs" / f"dryrun_{job['job_id']}.json"
        result = subprocess.run(
            [str(PROJECT / "envs/jyc/bin/python"), str(TASK / "code/gpu23_launcher.py"),
             "--job-json", str(path), "--lease-dir",
             str(PROJECT / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"),
             "--status-path", str(status_path), "--dry-run"],
            cwd=str(TASK), capture_output=True, text=True, check=False,
        )
        try:
            output = json.loads(result.stdout.strip())
        except Exception as exc:
            raise RuntimeError(f"watcher dry-run returned invalid output for {job['job_id']}: {result.stdout!r}") from exc
        if (result.returncode != 0 or output.get("status") != "PASS_DRY_RUN"
                or output.get("gpu_query_performed") is not False
                or output.get("active_physical_gpu_candidates") != [2]
                or output.get("job_id") != job["job_id"]):
            raise AssertionError(f"watcher dry-run failed for {job['job_id']}: {output}")
        outcomes.append({"job_id": job["job_id"], "phase": job["task_phase"],
                         "job_spec_path": str(path), "job_spec_sha256": sha_file(path),
                         "status": output["status"], "gpu_query_performed": False})

    gate = {
        "status": "PASS",
        "created_utc": utc_now(),
        "protocol_sha256": protocol_sha256,
        "source_hashes_sha256": sha_file(source_hashes_path),
        "matrix_sha256": sha_file(TASK / "protocol/MATRIX_12.csv"),
        "stage0_matrix_sha256": sha_file(TASK / "protocol/STAGE0_MATRIX_4.csv"),
        "resource_gate_host_sha256": sha_file(TASK / "RESOURCE_GATE_HOST.json"),
        "gpu_allowlist_sha256": sha_file(TASK / "protocol/GPU_ALLOWLIST_LOCKED.json"),
        "authorized_physical_gpu_ids": [2, 3],
        "active_physical_gpu_candidates": [2],
        "own_queue_concurrency": 1,
        "stage0_job_keys": [job["job_id"] for job, _ in jobs if job["task_phase"] == "STAGE0"],
        "formal_job_keys": [job["job_id"] for job, _ in jobs if job["task_phase"] == "FORMAL"],
        "job_spec_count": len(outcomes),
        "watcher_dry_run_count": len(outcomes),
        "gpu_queries_performed": 0,
        "final_test_access": 0,
        "job_specs": outcomes,
    }
    atomic_json(TASK / "LAUNCH_GATE.json", gate)
    return gate


def main() -> int:
    run_timestamp = utc_now()
    protocol_path = EQ / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    matrix_path = EQ / "protocol/D3_EQ_MATRIX_36.csv"
    seed_registry_path = EQ / "protocol/D3_EQ_SEED_REGISTRY.json"
    protocol_sha = sha_file(protocol_path)
    if json.loads(protocol_path.read_text(encoding="utf-8")).get("status") != "LOCKED":
        raise AssertionError("source D3-EQ protocol is not locked")
    resource_gate_path = TASK / "RESOURCE_GATE_HOST.json"
    if not resource_gate_path.is_file():
        raise FileNotFoundError("fresh host GPU/CPU/disk resource gate is required before final protocol lock")
    resource_gate = json.loads(resource_gate_path.read_text(encoding="utf-8"))
    if resource_gate.get("status") != "PASS" or resource_gate.get("gpu_allowlist") != [2, 3]:
        raise AssertionError("host resource preflight has not accepted the authorized GPU2/3 pool")
    launch_candidates = [int(index) for index in resource_gate.get("launch_candidates", [])]
    if (launch_candidates != [2] or resource_gate.get("own_queue_concurrency") != 1
            or resource_gate.get("snapshot_fresh_for_launch") is not True):
        raise AssertionError("host resource gate must lock fresh physical GPU2-only, serial execution")
    snapshot_time = datetime.fromisoformat(
        str(resource_gate["gpu_snapshot_observed_utc"]).replace("Z", "+00:00")
    )
    snapshot_age = (datetime.now(timezone.utc) - snapshot_time.astimezone(timezone.utc)).total_seconds()
    if snapshot_age < 0 or snapshot_age > 600:
        raise AssertionError("GPU2 aggregate snapshot is stale at protocol-lock time")
    gpu_allow_path = TASK / "protocol/GPU_ALLOWLIST_LOCKED.json"
    gpu_allow = json.loads(gpu_allow_path.read_text(encoding="utf-8"))
    if (gpu_allow.get("status") != "LOCKED" or gpu_allow.get("physical_gpu_ids") != [2, 3]
            or gpu_allow.get("launch_candidates") != [2]):
        raise AssertionError("task GPU authorization or active-candidate lock is inconsistent")

    split_audit = verify_split_groups()
    rows, old_source_records, seed_entries = validate_old_cells()
    matched_columns = list(rows[0])
    matrix_path_new = TASK / "protocol/MATRIX_12.csv"
    write_csv(matrix_path_new, matched_columns, rows)
    matched_old_path = TASK / "protocol/MATCHED_OLD_CELLS.csv"
    write_csv(matched_old_path, matched_columns, rows)
    stage0_rows = []
    for config in CONFIGS:
        for horizon in HORIZONS:
            stage0_rows.append({
                "job_id": f"D3EQ-CS-{config}-H{horizon}-S20261001-STAGE0",
                "config": config,
                "seed": 20261001,
                "horizon_steps": horizon,
                "stage0_rows": 256,
                "expected_optimizer_updates": 4,
                "expected_scheduler_calls": 4,
                "run_root": str(TASK / "runs/stage0" / f"D3EQ-CS-{config}-H{horizon}-S20261001-STAGE0"),
            })
    write_csv(TASK / "protocol/STAGE0_MATRIX_4.csv", list(stage0_rows[0]), stage0_rows)

    trajectory_source_path = EQ / "protocol/D3_EQ_TRAJECTORY_SELECTION.json"
    trajectory_source = json.loads(trajectory_source_path.read_text(encoding="utf-8"))
    if trajectory_source.get("status") != "LOCKED_BEFORE_PREDICTION_OUTCOMES" or trajectory_source.get("prediction_outcomes_opened") is not False:
        raise AssertionError("fixed trajectory selection is not outcome-blind/locked")
    if len(trajectory_source.get("representatives", [])) < 1:
        raise AssertionError("no fixed trajectory IDs in source selection")
    trajectory_locked = {"status": "LOCKED", "source_path": str(trajectory_source_path),
                         "source_sha256": sha_file(trajectory_source_path), **trajectory_source}
    atomic_json(TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json", trajectory_locked)
    task_code_paths = (
        "code/d3_train_corrected.py",
        "code/corrected_scheduler.py",
        "code/d3_eq_contract.py",
        "code/d3_model.py",
        "code/gpu23_filter.py",
        "code/gpu23_launcher.py",
        "code/physical_gpu_guard.py",
        "code/record_host_preflight.py",
        "code/run_queue.py",
        "code/d3_corrected_infer.py",
        "code/evaluate_corrected_scheduler.py",
        "code/analyze_scheduler_effects.py",
        "code/test_corrected_scheduler.py",
        "code/test_gpu23_launcher.py",
        "code/prepare_protocol.py",
        "code/generate_figures.py",
        "figures/assets/publication.mplstyle",
        "figures/assets/color_palettes.py",
    )
    task_code_hashes = {relative: sha_file(TASK / relative) for relative in task_code_paths}
    source_records = list(old_source_records)
    source_records.extend([
        exact_source_record(protocol_path, "old_d3eq_locked_protocol"),
        exact_source_record(matrix_path, "old_d3eq_36_matrix"),
        exact_source_record(seed_registry_path, "old_d3eq_seed_registry"),
        exact_source_record(EQ / "code/d3_train_eq.py", "old_d3eq_trainer_source"),
        exact_source_record(EQ / "code/d3_model.py", "locked_model_source"),
        exact_source_record(EQ / "code/d3_eq_contract.py", "locked_episode_equal_loss_source"),
        exact_source_record(EQ / "d3/train_targets_trainonly.npz", "train_only_target_cache"),
        exact_source_record(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json", "train_target_cache_manifest"),
        exact_source_record(OLD / "d3/cache/D3_CACHE_MANIFEST.json", "accepted_spatial_dynamic_cache_manifest"),
        exact_source_record(OLD / "d3/cache/D3_VIDEO_NORMALIZER.json", "train_only_video_normalizer"),
        exact_source_record(V3 / "data/V3_MAIN_INPUTS.npz", "locked_v3_main_inputs_train_dev_only"),
        exact_source_record(V3 / "data/V3_NORMALIZERS.json", "locked_train_only_target_normalizers"),
        exact_source_record(SPLIT_MANIFEST, "duplicate_safe_source_group_manifest"),
        exact_source_record(DIAG / "MOTION_ACTIVITY_THRESHOLD.json", "train_frozen_activity_thresholds"),
        exact_source_record(DIAG / "MOTION_STRATA_SUPPORT.csv", "train_frozen_motion_strata_support"),
        exact_source_record(DIAG / "DEV_METRIC_REPRODUCTION_CHECK.csv", "prior_old_model_scorer_reproduction_audit"),
        exact_source_record(TASK / "protocol/ORIGINAL_TASK_PROMPT.txt", "user_task_prompt"),
        exact_source_record(matched_old_path, "matched_old_cells_table"),
        exact_source_record(trajectory_source_path, "outcome_blind_trajectory_selection_source"),
        exact_source_record(TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json", "locked_trajectory_selection"),
        exact_source_record(TASK / "RESOURCE_GATE_HOST.json", "current_host_resource_gate"),
        exact_source_record(gpu_allow_path, "task_gpu_allowlist"),
    ])
    source_records.extend(exact_source_record(TASK / relative, relative) for relative in task_code_paths)
    source_hashes_path = TASK / "protocol/SOURCE_HASHES.json"
    atomic_json(source_hashes_path, {
        "schema_version": "1.0",
        "created_utc": run_timestamp,
        "status": "PASS",
        "final_test_access": 0,
        "prohibited_sources_opened": [],
        "sources": source_records,
        "task_code_sha256": task_code_hashes,
    })

    seeds_path = TASK / "protocol/SEED_INIT_MANIFEST.json"
    atomic_json(seeds_path, {
        "status": "PASS",
        "init_from": "fresh original D3-EQ seed state; never old epoch40 checkpoint",
        "seeds": [seed_entries[seed] for seed in SEEDS],
        "all_configs_and_horizons_matched_by_seed": True,
        "final_test_access": 0,
    })
    acceptance = {
        "status": "PASS",
        "accepted_old_matched_cells": "12/12",
        "old_checkpoints": "epoch40; 11,120 optimizer updates; actual historical once-per-epoch scheduler",
        "old_train_predictions": "12/12 hash-verified and bound to matched checkpoint",
        "old_dev_predictions": "12/12 hash-verified and bound to matched checkpoint",
        "old_code_hashes": "trainer/model/cache/normalizer/protocol verified per cell",
        "input_target_scope": "train targets only during training/inference; dev target values restricted to separate evaluator after inference acceptance",
        "source_group_audit": split_audit,
        "fixed_trajectory_ids_locked_before_new_scores": [row["anchor_id"] for row in trajectory_source["representatives"]],
        "known_caveats": [
            "Action halves remain action[0:14] and action[14:28]; anatomy labels unresolved.",
            "TS11-S run-level extractor binding is not fixed by this TS11-D versus STATE-CTRL study.",
            "D5 remains no-training; final/held-out access is zero.",
        ],
        "submission_ready": False,
        "human_review_status": "pending",
    }
    atomic_json(TASK / "SOURCE_ACCEPTANCE.json", acceptance)
    atomic_text(TASK / "SOURCE_ACCEPTANCE.md", (
        "# SOURCE_ACCEPTANCE\n\n"
        "状态：PASS。匹配旧 D3-EQ 的 12 个单元全部通过 checkpoint、初始化、协议、训练行、batch、train prediction 和 dev prediction 哈希核验。"
        "新实验从原 seed INIT 重新训练；不会从旧 epoch-40 权重warm start。\n\n"
        f"训练支持：{EXPECTED_TRAIN_ROWS} anchors / {EXPECTED_TRAIN_EPISODES} episodes；"
        f"dev 支持：{EXPECTED_DEV_ROWS} anchors / {EXPECTED_DEV_EPISODES} source groups。\n\n"
        "已知限制：action只使用raw数字半区；TS11-S extractor运行级绑定不在本研究修复范围；D5不训练；final/held-out访问为0。"
    ))
    atomic_text(TASK / "OUTCOME_EXPOSURE.md", (
        "# Outcome exposure and claim boundary\n\n"
        "旧 D1/D2/D3-EQ与D4开发结果已在历史任务中暴露。本轮是同一development evidence上的matched optimizer sensitivity，不是独立确认。"
        "本轮只改变scheduler推进频率/时间尺度，不增加配置、seed、损失或模型。"
        "只在train/dev predictions全部验收后才允许独立evaluator访问dev target values。"
    ))

    comparisons = {
        "primary_family": [
            {"model": config, "horizon_steps": horizon, "metric": metric}
            for config in CONFIGS for horizon in HORIZONS
            for metric in ("standardized_mae_full28", "native_future_internal_fd_full28")
        ],
        "secondary_video_specific_family": [
            {"horizon_steps": horizon, "metric": metric}
            for horizon in HORIZONS
            for metric in ("standardized_mae_full28", "native_future_internal_fd_full28")
        ],
        "gain_sign": "old_as_run_loss - corrected_scheduler_loss; positive is improvement",
        "video_specific_gain": "gain_TS11-D - gain_STATE-CTRL; positive means larger fused-model schedule gain",
        "bootstrap": {
            "replicates": 10000,
            "unit": "paired duplicate-safe video source group",
            "resample": "shared resamples across old/new, config, horizon, metric; keep three seeds paired inside each group",
            "aggregate_estimand": "average three seed gains within group, then equal-weight source groups",
            "seed_level_reporting": "all three points, group-bootstrap percentile CI, min/max, descriptive mean",
        "trajectory_visuals": {
            "fixed_representatives": "protocol/TRAJECTORY_SELECTION_LOCKED.json; selected outcome-blind before prediction outcomes",
            "illustrative_failure": "after prediction acceptance, select the maximum old-as-run TS11-D/H60/S20261001 Full28 dev standardized-MAE anchor; deterministic anchor_id tie-break; corrected predictions excluded from selection",
            "inferential_role": "illustrative only; not used in primary or secondary estimates",
        },
            "training_seed_uncertainty": "not covered by group bootstrap; fixed three trained seeds",
            "bootstrap_seed": "20260924 + horizon_steps",
            "p_values": "centered bootstrap with add-one; Holm separately over the 8 primary and 4 secondary estimands",
        },
        "auxiliary_metrics": [
            "train/dev native and standardized MAE/RMSE",
            "native future-internal FD (29/59 differences), anchor-to-first, endpoint and early/mid/late error",
            "TVpred/TVGT as group-equal mean ratio, prediction amplitude and direction with frozen train thresholds",
            "Persistence and locked SLOPE baselines",
            "raw action[0:14] and action[14:28] coordinates; anatomical mapping unresolved",
            "train/dev motion strata, separately summarized and support-counted",
        ],
        "dev_target_access": "only after prediction bundle acceptance in the independent evaluator",
        "final_test_access": 0,
    }
    comparison_path = TASK / "protocol/COMPARISON_FAMILIES.json"
    atomic_json(comparison_path, comparisons)

    lr_path = TASK / "protocol/LR_TRACE_EXPECTED.csv"
    lr_rows = []
    eta_min = 1e-5
    lr0 = 1e-3
    total = EXPECTED_UPDATES
    for update_after in range(1, total + 1):
        update_before = update_after - 1
        lr_used = eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * update_before / total))
        lr_after = eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * update_after / total))
        lr_rows.append({
            "update_count_before": update_before,
            "update_count_after": update_after,
            "lr_used_for_update": f"{lr_used:.17g}",
            "lr_after_update": f"{lr_after:.17g}",
        })
    write_csv(lr_path, ["update_count_before", "update_count_after", "lr_used_for_update", "lr_after_update"], lr_rows)

    resource = build_resource_estimate(rows)
    atomic_json(TASK / "RESOURCE_ESTIMATE_PRE_STAGE0.json", resource)
    if resource["status"] != "PASS_PRE_STAGE0":
        raise AssertionError("pre-Stage0 resource estimate exceeds task budget")
    lock = {
        "trajectory_selection": "protocol/TRAJECTORY_SELECTION_LOCKED.json",
        "trajectory_selection_sha256": sha_file(TASK / "protocol/TRAJECTORY_SELECTION_LOCKED.json"),
        "schema_version": "1.0",
        "study_id": "CORRECTED_SCHEDULER_MATCHED_12",
        "status": "LOCKED",
        "locked_utc": utc_now(),
        "only_scientific_intervention": "CosineAnnealingLR step after each successful optimizer.step; T_max=11120; eta_min=1e-5",
        "source_protocol_id": "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1",
        "source_protocol_sha256": protocol_sha,
        "matrix": "protocol/MATRIX_12.csv",
        "matrix_sha256": sha_file(matrix_path_new),
        "resource_gate_host_sha256": sha_file(TASK / "RESOURCE_GATE_HOST.json"),
        "gpu_snapshot_observed_utc": resource_gate["gpu_snapshot_observed_utc"],
        "stage0_matrix": "protocol/STAGE0_MATRIX_4.csv",
        "source_acceptance_sha256": sha_file(TASK / "SOURCE_ACCEPTANCE.json"),
        "source_hashes": "protocol/SOURCE_HASHES.json",
        "matched_old_cells": "protocol/MATCHED_OLD_CELLS.csv",
        "matched_old_cells_sha256": sha_file(matched_old_path),
        "source_hashes_sha256": sha_file(source_hashes_path),
        "seed_init_manifest": "protocol/SEED_INIT_MANIFEST.json",
        "seed_init_manifest_sha256": sha_file(seeds_path),
        "comparison_families_sha256": sha_file(comparison_path),
        "expected_lr_trace_sha256": sha_file(lr_path),
        "gpu_allowlist": str(gpu_allow_path),
        "gpu_allowlist_sha256": sha_file(gpu_allow_path),
        "task_code_hashes": task_code_hashes,
        "train": {"rows": EXPECTED_TRAIN_ROWS, "episodes": EXPECTED_TRAIN_EPISODES, "batch": 64},
        "dev": {"rows": EXPECTED_DEV_ROWS, "source_groups": EXPECTED_DEV_EPISODES},
        "training": {"epochs": 40, "steps_per_epoch": 278, "updates_per_run": EXPECTED_UPDATES,
                     "optimizer": "AdamW lr=1e-3 betas=(0.9,0.999) eps=1e-8 wd=1e-4",
                     "gradient_clip_norm": 1.0, "dtype": "FP32"},
        "loss_contract": "D3-EQ samplewise episode weight W/(E*n_episode), weighted sum / actual batch size, Full28 position+0.1 future-internal FD",
        "resource_budget": {"gpu_hours": GPU_HOUR_CAP, "peak_memory_mib": GPU_MEMORY_CAP_MIB,
                            "max_project_children": 2, "allowed_physical_gpu_ids": [2, 3],
                            "active_launch_candidates": launch_candidates,
                            "own_queue_concurrency": 1,
                            "candidate_reason": resource_gate.get("candidate_reason", "")},
        "final_test_access": 0,
        "heldout_access": 0,
        "submission_ready": False,
        "human_review": {"status": "pending", "reviewer": "", "review_date": ""},
    }
    atomic_json(TASK / "PROTOCOL_LOCKED.json", lock)
    launch_gate = write_and_dryrun_job_specs(stage0_rows, rows, resource)
    print(json.dumps({
        "launch_gate": launch_gate["status"],
        "stage0_job_specs": len(launch_gate["stage0_job_keys"]),
        "formal_job_specs": len(launch_gate["formal_job_keys"]),
        "watcher_dry_runs": launch_gate["watcher_dry_run_count"],
        "status": "PASS_LOCKED",
        "locked_cells": len(rows),
        "old_formal_lease_gpu_hours": resource["old_matched_12_formal_lease_gpu_hours"],
        "projected_total_gpu_hours": resource["projected_total_including_optional_dense_reserve"],
        "max_old_stage0_peak_mib": resource["measured_old_stage0_peak_reserved_mib"],
        "protocol_sha256": sha_file(TASK / "PROTOCOL_LOCKED.json"),
        "matrix_sha256": sha_file(matrix_path_new),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
