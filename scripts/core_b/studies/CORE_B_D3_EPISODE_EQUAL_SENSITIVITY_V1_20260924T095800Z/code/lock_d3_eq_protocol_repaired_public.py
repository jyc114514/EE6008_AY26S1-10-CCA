# Repaired public copy of EXCL-CODE-016; source SHA-256: b87aa1da88f30bfdf5e46fd54f8eb8b6cde57089c4820117557cd192c13ee8f5
"""Generate the versioned D3-EQ 36-cell protocol after CPU gates pass."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
EQ_ROOT = Path(os.environ.get("EE6008_D3_EQ_STUDY_ROOT", PROJECT_ROOT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z")).expanduser().resolve()
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
ORIGINAL_TASK = Path(os.environ.get("EE6008_D3_EQ_ORIGINAL_TASK_ATTACHMENT", ".")).expanduser().resolve()
CURRENT_TASK = Path(os.environ.get("EE6008_D3_EQ_CURRENT_TASK_ATTACHMENT", ".")).expanduser().resolve()
ORIGINAL_TASK_SHA256 = "7ebece8c3ada6375c63744fdd35bbacc8e2fb447fef42e28bc7e866ed416c61e"
CURRENT_TASK_SHA256 = "a5b48c71e32d4f0257d24eda03577b25b1fa7d7760908d28ad092f6311976f09"
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def lease_seconds(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("exit_code", -999)) != 0 or data.get("status") != "complete":
        return 0.0
    start = datetime.fromisoformat(data["lease_acquired_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(data["lease_released_utc"].replace("Z", "+00:00"))
    return max((end - start).total_seconds(), 0.0)


def main() -> int:
    qa_cache = json.loads((EQ_ROOT / "qa/D3_EQ_CACHE_ID_AUDIT.json").read_text(encoding="utf-8"))
    qa_cpu = json.loads((EQ_ROOT / "qa/D3_EQ_CPU_TESTS.json").read_text(encoding="utf-8"))
    target_manifest = json.loads((EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if qa_cache.get("status") != "PASS" or qa_cache.get("dev_targets_loaded") is not False or qa_cache.get("final_test_access") != 0:
        raise AssertionError("cache/ID audit must pass before protocol lock")
    if qa_cpu.get("status") != "PASS" or qa_cpu.get("final_test_access") != 0:
        raise AssertionError("CPU D3-EQ regression tests must pass before protocol lock")
    if target_manifest.get("status") != "PASS" or target_manifest.get("train_windows") != 17757 or target_manifest.get("train_episodes") != 2712:
        raise AssertionError("train-only target cache must be complete before protocol lock")
    if sha256_file(ORIGINAL_TASK) != ORIGINAL_TASK_SHA256 or sha256_file(CURRENT_TASK) != CURRENT_TASK_SHA256:
        raise AssertionError("server-side task attachment hash no longer matches verified provenance")

    protocol_dir = EQ_ROOT / "protocol"
    if protocol_dir.exists() and any(protocol_dir.iterdir()):
        raise FileExistsError("protocol directory is not empty; refusing to replace a lock")
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (EQ_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (EQ_ROOT / "d3/analysis").mkdir(parents=True, exist_ok=True)

    source_code = EQ_ROOT / "code"
    source_map = {
        "original_task_server_attachment": ORIGINAL_TASK,
        "current_execution_authorization": CURRENT_TASK,
        "old_d3_trainer": BENCH_ROOT / "code/d3_train.py",
        "old_d3_trainer_copy": source_code / "d3_train_historical_original.py",
        "old_d3_model": BENCH_ROOT / "code/d3_model.py",
        "new_d3_model_copy": source_code / "d3_model.py",
        "new_d3eq_trainer": source_code / "d3_train_eq.py",
        "d3eq_loss_contract": source_code / "d3_eq_contract.py",
        "cpu_regression_tests": source_code / "test_d3_eq_contract.py",
        "cache_audit_code": source_code / "audit_d3_eq_cache.py",
        "train_target_builder": source_code / "build_train_target_cache.py",
        "train_target_sha256_csv": EQ_ROOT / "d3/TRAIN_PARQUET_SHA256.csv",
        "train_target_cache": EQ_ROOT / "d3/train_targets_trainonly.npz",
        "train_target_cache_manifest": EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json",
        "v3_inputs_npz": V3_ROOT / "data/V3_MAIN_INPUTS.npz",
        "v3_anchor_manifest": V3_ROOT / "ANCHOR_MANIFEST.csv",
        "v3_normalizers": V3_ROOT / "data/V3_NORMALIZERS.json",
        "old_d3_protocol": BENCH_ROOT / "protocol/D3_PROTOCOL_LOCKED.json",
        "old_v3_model_loss_contract": V3_ROOT / "MODEL_LOSS_CONTRACT.md",
        "d3_cache_manifest": BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json",
        "d3_video_normalizer": BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json",
        "cache_audit_result": EQ_ROOT / "qa/D3_EQ_CACHE_ID_AUDIT.json",
        "cpu_test_result": EQ_ROOT / "qa/D3_EQ_CPU_TESTS.json",
    }
    source_hashes = {
        key: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for key, path in source_map.items()
    }
    source_hash_path = protocol_dir / "D3_EQ_SOURCE_HASHES.json"
    atomic_json(source_hash_path, {"status": "LOCKED", "generated_utc": utc_now(), "sources": source_hashes, "final_test_access": 0})

    sys.path.insert(0, str(source_code))
    from d3_model import D3TimeSpacePredictor, state_hash

    seed_registry: dict[str, Any] = {}
    for seed in SEEDS:
        init_path = BENCH_ROOT / f"d3/runs/INIT_S{seed}.pt"
        state = torch.load(init_path, map_location="cpu", weights_only=True)
        model = D3TimeSpacePredictor(horizon=30, seed=seed, video_mode="TS00-D")
        model.load_state_dict(state, strict=True)
        seed_registry[str(seed)] = {
            "init_state_path": str(init_path),
            "init_state_file_sha256": sha256_file(init_path),
            "init_state_hash": state_hash(model),
            "same_initial_state_all_configs_and_horizons": True,
        }
    seed_path = protocol_dir / "D3_EQ_SEED_REGISTRY.json"
    atomic_json(seed_path, {
        "protocol_id": "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1",
        "status": "LOCKED",
        "seed_rule": "reuse the three matched historical D3 initial states; never warm-start trained weights",
        "seeds": seed_registry,
        "final_test_access": 0,
    })

    matrix_rows = []
    for seed in SEEDS:
        for horizon in (30, 60):
            for config in CONFIGS:
                matrix_rows.append({
                    "run_id": f"D3EQ-{config}-H{horizon}-S{seed}",
                    "config": config,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "horizon_seconds": horizon / 30.0,
                    "training_status": "LOCKED_NOT_STARTED",
                    "prediction_status": "LOCKED_NOT_STARTED",
                    "checkpoint_rule": "last40_epoch40",
                    "effective_batch": 64,
                    "final_test_access": 0,
                })
    matrix_path = protocol_dir / "D3_EQ_MATRIX_36.csv"
    columns = list(matrix_rows[0])
    with matrix_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(matrix_rows)

    old_formal_logs = BENCH_ROOT / "logs"
    formal_watcher = sorted(old_formal_logs.glob("watcher_D3-*-FORMAL*.json"))
    formal_gpu_seconds = sum(lease_seconds(path) for path in formal_watcher)
    formal_success = sum(lease_seconds(path) > 0 for path in formal_watcher)
    stage0_watcher = sorted(old_formal_logs.glob("watcher_D3-*-STAGE0*.json"))
    stage0_gpu_seconds = sum(lease_seconds(path) for path in stage0_watcher)
    stage0_success = sum(lease_seconds(path) > 0 for path in stage0_watcher)
    if formal_success != 36 or stage0_success != 12:
        raise AssertionError(f"historical watcher timing counts unexpected: formal={formal_success}, stage0={stage0_success}")
    old_completion_files = sorted((BENCH_ROOT / "d3/runs").glob("D3-*-FORMAL*/RUN_COMPLETE.json"))
    final_lrs = sorted({round(float(json.loads(path.read_text())["last_report"]["lr"]), 15) for path in old_completion_files})
    if len(old_completion_files) != 36 or final_lrs != [0.000999968393185]:
        raise AssertionError("historical 36-cell completion/LR evidence differs from the scheduler audit")

    estimated_stage0_gpu_hours = (stage0_gpu_seconds / 3600.0) * 4.0
    observed_formal_gpu_hours = formal_gpu_seconds / 3600.0
    resource_estimate = {
        "status": "PRE_STAGE0_ESTIMATE_LOCKED",
        "resource_basis": "historical watcher lease durations + old D3 artifact sizes; refresh host disk/GPU/node gates before Stage0",
        "historical_formal_successful_runs": formal_success,
        "historical_formal_gpu_hours": observed_formal_gpu_hours,
        "historical_formal_median_minutes": sorted([lease_seconds(path) / 60.0 for path in formal_watcher if lease_seconds(path) > 0])[formal_success // 2],
        "historical_stage0_successful_one_step_runs": stage0_success,
        "historical_stage0_gpu_hours": stage0_gpu_seconds / 3600.0,
        "new_stage0_scope": "12 first-seed cells x 4 optimizer steps",
        "new_stage0_gpu_hours_estimate": estimated_stage0_gpu_hours,
        "new_formal_36_best_estimate_gpu_hours": observed_formal_gpu_hours,
        "new_formal_36_conservative_factor": 2.0,
        "new_formal_36_conservative_gpu_hours": observed_formal_gpu_hours * 2.0,
        "primary_prediction_gpu_hours": "measure in Stage0/first accepted predictions; stop-gate before launch",
        "dense_prediction_gpu_hours": "optional; estimate from accepted primary inference, then use remaining budget only",
        "new_run_gpu_hour_ceiling": 48.0,
        "new_run_output_size_ceiling_gib": 50.0,
        "minimum_filesystem_free_fraction": 0.10,
        "old_d3_run_checkpoint_bytes": sum(p.stat().st_size for p in (BENCH_ROOT / "d3/runs").glob("**/*.pt")),
        "old_d3_sparse_prediction_bytes": sum(p.stat().st_size for p in (BENCH_ROOT / "d3/predictions").glob("*.npz")),
        "new_train_target_cache_bytes": (EQ_ROOT / "d3/train_targets_trainonly.npz").stat().st_size,
        "host_gpu_state": "not queried; explicit host approval and node-role check required",
        "final_test_access": 0,
    }
    resource_path = protocol_dir / "D3_EQ_RESOURCE_ESTIMATE_PRE_STAGE0.json"
    atomic_json(resource_path, resource_estimate)
    with (EQ_ROOT / "RESOURCE_ESTIMATE.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "gpu_hours_estimate", "basis", "status"])
        writer.writeheader()
        writer.writerows([
            {"phase": "Stage0_12x4_steps", "gpu_hours_estimate": estimated_stage0_gpu_hours, "basis": "4x prior 12 one-step Stage0 leases; will replace with measured", "status": "PRELIMINARY"},
            {"phase": "formal_36", "gpu_hours_estimate": observed_formal_gpu_hours, "basis": "36 successful historical D3 watcher GPU leases", "status": "BEST_ESTIMATE_PRE_STAGE0"},
            {"phase": "formal_36_conservative", "gpu_hours_estimate": observed_formal_gpu_hours * 2.0, "basis": "2x historical measured lease sum", "status": "CONSERVATIVE_ESTIMATE"},
            {"phase": "primary_predictions_36", "gpu_hours_estimate": "PENDING", "basis": "measure after accepted Stage0", "status": "NOT_STARTED"},
            {"phase": "dense_predictions_36", "gpu_hours_estimate": "OPTIONAL_PENDING", "basis": "only after primary accepted and budget recheck", "status": "NOT_STARTED"},
        ])

    lock = {
        "protocol_id": "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1",
        "protocol_version": "D3-EQ-V1",
        "status": "LOCKED",
        "locked_utc": utc_now(),
        "study_identity": "full 36-cell objective-sensitivity replay; development-only; not independent confirmation",
        "original_task_weight_requirement_verified": True,
        "old_d3_historical_lock_amended": False,
        "old_d3_classification": ["historical_actual_objective", "generated_lock_omission", "development_exploratory"],
        "old_d3_initial_task_nonconformance": "verified",
        "only_planned_scientific_intervention": "strict samplewise episode-equal weighted objective and mathematically equivalent gradient accumulation/tail handling",
        "train_windows": 17757,
        "train_episodes": 2712,
        "train_episode_window_count_min_max": [1, 8],
        "episode_weight": "w_i = W/(E*n_episode(i)); weights fit once on full primary-train rows and reused by every cell",
        "sample_objective": "l_i = 0.5*Arm14 standardized position L1 + 0.5*Hand14 standardized position L1 + 0.1*(0.5*Arm14 safe_d-normalized future-internal FD L1 + 0.5*Hand14 safe_d-normalized future-internal FD L1)",
        "effective_batch_objective": "sum_i(w_i*l_i)/B_actual; never divide by sum(weights)",
        "effective_batch": 64,
        "oom_fallback": "physical batch32 with accumulation2 only after measured OOM; each microbatch contribution divided by actual effective update-group size; partial 29-row tail divides by 29",
        "train_sampling": "each primary-train anchor exactly once per epoch; np.random.default_rng(seed+epoch).shuffle",
        "training": {
            "epochs": 40,
            "checkpoint_rule": "last40; save epoch10/20/30/40 checkpoints for resumability; final evaluation checkpoint is epoch40",
            "optimizer": "AdamW lr=1e-3 betas=(0.9,0.999) eps=1e-8 weight_decay=1e-4",
            "gradient_clip_norm": 1.0,
            "dtype": "FP32",
            "encoder_updated": False,
            "warm_start": False,
            "expected_optimizer_steps_per_epoch": 278,
            "expected_optimizer_steps_per_run": 11120,
            "scheduler_contract_vs_as_run": {
                "old_lock_text": "cosine to 1e-5",
                "historical_code": "CosineAnnealingLR(T_max=11120) called once after each epoch",
                "all_36_historical_epoch40_lr": final_lrs[0],
                "d3eq_choice": "preserve actual scheduler object, T_max and once-per-epoch step call to isolate objective change; do not claim eta_min was reached",
            },
            "resume_checkpoint_fields": ["model", "optimizer", "scheduler", "epoch", "sampler_rule", "Python RNG", "NumPy RNG", "Torch CPU RNG", "Torch CUDA RNG"],
        },
        "configs": list(CONFIGS),
        "horizons_steps": [30, 60],
        "seeds": [
            {"seed": int(seed), **seed_registry[str(seed)]}
            for seed in SEEDS
        ],
        "matrix_size": 36,
        "matrix_csv": "D3_EQ_MATRIX_36.csv",
        "matrix_sha256": sha256_file(matrix_path),
        "target_scope": "primary-train-only Parquet targets; V3 mixed NPZ target field is never loaded by training",
        "target_cache": {
            "path": "../d3/train_targets_trainonly.npz",
            "sha256": target_manifest["target_cache_sha256"],
            "train_id_hash": target_manifest["train_ids_sha256_ordered_newline"],
            "dev_targets_loaded": False,
            "normalizer_fit_ids_sha256": qa_cache["v3_train_normalizer_fit_episode_hash_matches"],
        },
        "feature_cache": {
            "rows": qa_cache["cache_rows"],
            "shards_checked": qa_cache["cache_shards_checked"],
            "shape": [20079, 2, 16, 4, 768],
            "dtype": "float32",
            "grid": qa_cache["cache_grid_layout"],
            "bridge_max_abs": qa_cache["cache_bridge_max_abs"],
            "id_order_check": "PASS",
            "final_test_access": 0,
        },
        "primary_dev": {"anchors": 2322, "source_episode_groups": 603, "task_categories_not_bootstrap_units": 118},
        "primary_metrics": ["Arm14 standardized MAE", "Hand14 standardized MAE"],
        "secondary_metrics": ["native future-internal FD", "Full28 standardized MAE", "endpoint", "TV", "task/joint diagnostics"],
        "primary_contrasts": ["TS11-D_vs_TS10-D", "TS11-D_vs_TS01-D", "TS11-D_vs_TS11-S", "TS11-D_vs_STATE-CTRL"],
        "factorial_interaction": "(L11-L10)-(L01-L00)",
        "evaluation": {
            "dev_ids": "same locked V3/D3 primary-dev IDs; ID join exact",
            "episode_equal_windows": True,
            "group_bootstrap_draws_per_seed": 10000,
            "primary_holm_family": "4 contrasts x 2 horizons x 2 body regions; pooled paired source-episode bootstrap; seed-specific intervals also reported",
            "seed_aggregate": "mean +/- SD across three seed estimates, descriptive; no window-level pseudoreplication",
            "objective_sensitivity": "old actual D3 and D3-EQ scored separately with common scoring code; matched same-seed/config/horizon/group contrasts",
            "final_test_access": 0,
        },
        "resource_limits": {
            "total_gpu_hours": 48.0,
            "new_output_bytes_gib": 50.0,
            "filesystem_free_fraction_min": 0.10,
            "project_gpu_children_max": 2,
            "project_children_per_physical_gpu_max": 1,
            "raw_video_producers_max": 1,
            "dynamic_visible_gpu_pool": True,
            "watcher_required": True,
            "tmux_required_for_long_runs": True,
        },
        "resource_estimate_path": "D3_EQ_RESOURCE_ESTIMATE_PRE_STAGE0.json",
        "resource_estimate_sha256": sha256_file(resource_path),
        "source_hashes_path": "D3_EQ_SOURCE_HASHES.json",
        "source_hashes_sha256": sha256_file(source_hash_path),
        "execution_gate": {
            "stage0": "12 first-seed cells x exactly four optimizer steps; train only; save/reload and capture CUDA peak",
            "formal": "start first-seed 12 only after Stage0 12/12 and resource gate; then remaining 24 without outcome-based selection",
            "host_gate": "explicit sandbox-out approval, determine node role, query GPU aggregate inventory and global leases before Stage0",
            "final_test_access": 0,
        },
        "final_test_access": 0,
    }
    lock["canonical_protocol_sha256"] = sha256_json(lock)
    lock_path = protocol_dir / "D3_EQ_PROTOCOL_LOCKED.json"
    atomic_json(lock_path, lock)

    atomic_json(EQ_ROOT / "STATUS.json", {
        "protocol_id": lock["protocol_id"],
        "updated_utc": utc_now(),
        "status": "locked_pre_stage0",
        "original_task_weight_requirement_verified": True,
        "old_d3_historical_lock_amended": False,
        "cache_id_shape_audit": "PASS",
        "train_only_target_cache": "PASS_17757_ROWS_2712_EPISODES",
        "cpu_contract_tests": "PASS_7_TESTS",
        "protocol_locked": True,
        "matrix": "36_CELLS_LOCKED",
        "gpu_state_query": "not_run_requires_explicit_host_approval",
        "host_node_role": "unverified",
        "gpu_training_started": False,
        "stage0": "0/12",
        "formal_training": "0/36",
        "primary_predictions": "0/36",
        "dense_predictions": "0/36_not_started",
        "final_heldout_access": 0,
        "next_gate": "obtain explicit host approval; determine node role; refresh filesystem and GPU lease/resource state before Stage0",
    })
    print(json.dumps({
        "status": "LOCKED",
        "matrix_cells": len(matrix_rows),
        "formal_gpu_hours_historical": round(observed_formal_gpu_hours, 3),
        "formal_gpu_hours_conservative": round(observed_formal_gpu_hours * 2.0, 3),
        "stage0_gpu_hours_estimate": round(estimated_stage0_gpu_hours, 3),
        "epoch40_lr_historical": final_lrs[0],
        "protocol_sha256": sha256_file(lock_path),
        "final_test_access": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
