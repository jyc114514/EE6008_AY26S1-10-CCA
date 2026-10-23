from __future__ import annotations
import os

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from v7_common import (
    DEV_NPZ,
    METRIC_LOCK,
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
    load_dev_prefix,
    load_train_prefix,
    receiver_rows,
    sha_file,
    write_csv,
    write_npy,
)

V2_SEAL = V2 / "protocol/PREDICTION_BATCH_SEAL.json"
V3_SEAL = V3 / "protocol/PREDICTION_BATCH_SEAL.json"
V5_SEAL = V5 / "qa/PREDICTION_SEAL.json"
V5_RECEIPT = V5 / "qa/FINAL_CHECKSUM_VERIFICATION.json"
V6_RECEIPT = V6 / "qa/FINAL_CHECKSUM_VERIFICATION.json"
V4_COMPAT = V4 / "qa/COMMON_INIT_AND_RECIPE.json"
V4_RECIPE = V4 / "protocol/MODEL_RECIPE_LOCK.json"
V4_INIT_COMPAT = V4 / "inventory/RECIPE_BASE_INIT_COMPATIBILITY.json"
V4_SOURCE_MAP = V4 / "inventory/SOURCE_MAP.json"
V2_MODELS = V2 / "model_only/formal"
TASK_ORDER = (119, 94, 90, 141, 230)
PROGRAM_ORDER = {
    202610131: ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"),
    202610132: ("SELF_SOFT", "MIX_SOFT", "GT_RETRAIN"),
    202610133: ("MIX_SOFT", "GT_RETRAIN", "SELF_SOFT"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha_file(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def load_accepted_model(seed: int, role: str) -> dict[str, Any]:
    if role == "S":
        directory = V2_MODELS / f"v2_formal_G_S_lr0p001_s{seed}"
        expected_mode = "STATE-CTRL"
    elif role == "V":
        directory = V2_MODELS / f"v2_formal_G_V_lr0p0005_s{seed}"
        expected_mode = "TS11-D"
    else:
        raise ValueError(role)
    manifest_path = directory / "MODEL_MANIFEST.json"
    complete_path = directory / "RUN_COMPLETE.json"
    manifest = read_json(manifest_path)
    complete = read_json(complete_path)
    if (
        manifest.get("status") != "PASS"
        or complete.get("status") != "PASS"
        or int(manifest.get("seed", -1)) != seed
        or int(manifest.get("epochs", -1)) != 160
        or int(manifest.get("optimizer_steps", -1)) != 3200
        or manifest.get("video_mode") != expected_mode
        or int(manifest.get("final_heldout_payload_access", -1)) != 0
        or complete.get("weights_sha256") != manifest.get("model_file_sha256")
    ):
        raise RuntimeError(f"accepted V2 {role} model failed fixed-source identity checks: {directory}")
    model_path = Path(manifest["model_file"])
    require_hash(model_path, manifest["model_file_sha256"])
    init_path = Path(manifest["init_path"])
    init_sha = require_hash(init_path)
    return {
        "seed": seed,
        "role": role,
        "model_role": "G-S" if role == "S" else "G-V",
        "model_mode": expected_mode,
        "model_path": str(model_path),
        "model_sha256": manifest["model_file_sha256"],
        "model_manifest_path": str(manifest_path),
        "model_manifest_sha256": sha_file(manifest_path),
        "run_complete_path": str(complete_path),
        "run_complete_sha256": sha_file(complete_path),
        "final_state_hash": manifest["final_state_hash"],
        "init_path": str(init_path),
        "init_sha256": init_sha,
        "init_state_hash": manifest["init_state_hash"],
        "source_prediction_seal": str(V2_SEAL),
        "source_prediction_seal_sha256": sha_file(V2_SEAL),
        "optimizer_steps": 3200,
        "epochs": 160,
        "train_role": "outer_train",
        "final_heldout_payload_access": 0,
    }


def accepted_prediction_entries() -> list[dict[str, Any]]:
    v2_seal = read_json(V2_SEAL)
    v3_seal = read_json(V3_SEAL)
    v5_seal = read_json(V5_SEAL)
    accepted_seal_states = {"PASS", "SEALED_BEFORE_TARGET_ACCESS"}
    if v2_seal.get("status") not in accepted_seal_states or int(v2_seal.get("final_heldout_payload_access", -1)) != 0:
        raise RuntimeError("accepted V2 prediction seal is not sealed/final0")
    if v3_seal.get("status") not in accepted_seal_states or int(v3_seal.get("final_heldout_payload_access", -1)) != 0:
        raise RuntimeError("accepted V3 prediction seal is not sealed/final0")
    if v5_seal.get("status") != "PASS" or int(v5_seal.get("final_heldout_payload_access", -1)) != 0:
        raise RuntimeError("accepted V5 prediction seal is not PASS/final0")
    entries: list[dict[str, Any]] = []
    for entry in v2_seal.get("prediction_entries", []):
        if entry.get("model_id") not in {"G-S", "G-V"}:
            continue
        prediction, rows = Path(entry["prediction_path"]), Path(entry["rows_path"])
        require_hash(prediction, entry["prediction_sha256"])
        require_hash(rows, entry["rows_sha256"])
        entries.append({
            "seed": int(entry["seed"]),
            "role": "S" if entry["model_id"] == "G-S" else "V",
            "source_root": "V2",
            "prediction_path": str(prediction),
            "prediction_sha256": entry["prediction_sha256"],
            "rows_path": str(rows),
            "rows_sha256": entry["rows_sha256"],
            "seal_path": str(V2_SEAL),
            "seal_sha256": sha_file(V2_SEAL),
            "completion_path": None,
            "completion_sha256": None,
        })
    for entry in v3_seal.get("new_prediction_entries", []):
        method = entry.get("method_id")
        if method not in {"BIAS_STATE", "BIAS_VIDEO"}:
            continue
        prediction, rows = Path(entry["prediction_path"]), Path(entry["prediction_rows_path"])
        require_hash(prediction, entry["prediction_sha256"])
        require_hash(rows, entry["prediction_rows_sha256"])
        entries.append({
            "seed": int(entry["seed"]),
            "role": method,
            "source_root": "V3",
            "prediction_path": str(prediction),
            "prediction_sha256": entry["prediction_sha256"],
            "rows_path": str(rows),
            "rows_sha256": entry["prediction_rows_sha256"],
            "seal_path": str(V3_SEAL),
            "seal_sha256": sha_file(V3_SEAL),
            "completion_path": entry.get("completion_marker"),
            "completion_sha256": entry.get("completion_sha256"),
        })
    for entry in v5_seal.get("entries", []):
        if "/predictions/MIX50/" not in str(entry.get("prediction_path", "")):
            continue
        prediction, rows = Path(entry["prediction_path"]), Path(entry["prediction_rows_path"])
        require_hash(prediction, entry["prediction_sha256"])
        require_hash(rows, entry["prediction_rows_sha256"])
        entries.append({
            "seed": int(entry["seed"]),
            "role": "T",
            "source_root": "V5",
            "prediction_path": str(prediction),
            "prediction_sha256": entry["prediction_sha256"],
            "rows_path": str(rows),
            "rows_sha256": entry["prediction_rows_sha256"],
            "seal_path": str(V5_SEAL),
            "seal_sha256": sha_file(V5_SEAL),
            "completion_path": None,
            "completion_sha256": entry.get("completion_sha256"),
        })
    expected_roles = {(seed, role) for seed in SEEDS for role in ("S", "V", "T", "BIAS_STATE", "BIAS_VIDEO")}
    observed = {(int(row["seed"]), str(row["role"])) for row in entries}
    if observed != expected_roles or len(entries) != 15:
        raise RuntimeError(f"evaluation reference source map incomplete: {sorted(observed)}")
    return entries


def verify_root_receipts() -> dict[str, Any]:
    v5 = read_json(V5_RECEIPT)
    v6 = read_json(V6_RECEIPT)
    if (
        v5.get("status") != "PASS"
        or int(v5.get("manifest_entry_count", -1)) != 255
        or int(v5.get("verified_entry_count", -1)) != 255
        or v5.get("manifest_sha256") != "5866688b48bef3ed940920affc15081dea052c51eca6eb86e7bcfd3a09c5acfa"
    ):
        raise RuntimeError("accepted V5 checksum receipt differs from the V7 source contract")
    if (
        v6.get("status") != "PASS"
        or int(v6.get("manifest_entry_count", -1)) != 98
        or int(v6.get("verified_entry_count", -1)) != 98
        or v6.get("manifest_sha256") != "be786c7a85ff86118734c16344ae5e06db52dee1792c52132a64301a1f1f9c9f"
    ):
        raise RuntimeError("accepted V6 checksum receipt differs from the V7 source contract")
    return {
        "V5": {"root": str(V5), "receipt_path": str(V5_RECEIPT), "receipt_sha256": sha_file(V5_RECEIPT),
               "manifest_entries": 255, "manifest_sha256": v5["manifest_sha256"], "status": "PASS"},
        "V6": {"root": str(V6), "receipt_path": str(V6_RECEIPT), "receipt_sha256": sha_file(V6_RECEIPT),
               "manifest_entries": 98, "manifest_sha256": v6["manifest_sha256"], "status": "PASS"},
    }


def write_pairing(data: dict[str, np.ndarray], model_sources: list[dict[str, Any]]) -> dict[str, Any]:
    code = str(V4 / "code")
    locked = str(V4 / "code/locked_source")
    if code not in sys.path:
        sys.path.insert(0, code)
    if locked not in sys.path:
        sys.path.insert(0, locked)
    from d3_eq_contract import shuffled_epoch_order

    manifest_by_seed: dict[int, dict[str, Any]] = {}
    source_by_seed = {seed: [r for r in model_sources if r["seed"] == seed] for seed in SEEDS}
    matrix_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        init_rows = [row for row in source_by_seed[seed]]
        init_hashes = {row["init_sha256"] for row in init_rows}
        init_state_hashes = {row["init_state_hash"] for row in init_rows}
        init_paths = {row["init_path"] for row in init_rows}
        if len(init_hashes) != 1 or len(init_state_hashes) != 1 or len(init_paths) != 1:
            raise RuntimeError(f"G-S/G-V accepted source models do not share the same INIT for seed {seed}")
        orders = np.stack([shuffled_epoch_order(1280, seed, epoch) for epoch in range(1, 161)])
        order_path = ROOT / "protocol" / f"SAMPLER_ORDERS_seed{seed}.npy"
        write_npy(order_path, orders)
        order_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in orders]
        init_source = init_rows[0]
        manifest_by_seed[seed] = {
            "seed": seed,
            "init_path": init_source["init_path"],
            "init_sha256": init_source["init_sha256"],
            "init_state_hash": init_source["init_state_hash"],
            "sampler_orders_path": str(order_path),
            "sampler_orders_sha256": sha_file(order_path),
            "sampler_orders_shape": list(orders.shape),
            "sampler_dtype": "int64",
            "sampler_epoch_order_hashes": order_hashes,
            "pilot_sample_order": "first 128 rows of epoch 1 order",
            "program_order": list(PROGRAM_ORDER[seed]),
            "programs_share_init": True,
            "programs_share_all_160_sample_orders": True,
            "programs_share_torch_numpy_python_seed": True,
            "dropout_flow_setup": "same model construction, INIT load, deterministic flags and initial RNG state per paired program",
            "target_is_the_only_protocol_difference": True,
        }
        for position, program in enumerate(PROGRAM_ORDER[seed], start=1):
            target_path = ROOT / "training_targets" / f"seed{seed}" / f"{program}.npy"
            matrix_rows.append({
                "matrix_order": len(matrix_rows) + 1,
                "seed": seed,
                "within_seed_order": position,
                "program": program,
                "target_path": str(target_path),
                "init_path": init_source["init_path"],
                "init_sha256": init_source["init_sha256"],
                "sampler_orders_path": str(order_path),
                "sampler_orders_sha256": sha_file(order_path),
                "video_mode": "TS11-D",
                "effective_batch": 64,
                "microbatch_size": 64,
                "accumulation_steps": 1,
                "epochs": 160,
                "expected_optimizer_steps": 3200,
                "checkpoint_rule": "last160",
                "dev_selection": False,
                "new_fit_slot": True,
            })
    pairing = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_FIT",
        "created_utc": utc_now(),
        "seeds": [manifest_by_seed[seed] for seed in SEEDS],
        "program_order_by_seed": {str(seed): list(PROGRAM_ORDER[seed]) for seed in SEEDS},
        "fit_count": 9,
        "pilot_fits": 3,
        "pilot_updates_per_fit": 2,
        "formal_updates_per_fit": 3200,
        "same_seed_three_arm_pairing": True,
        "target_role_is_the_only_condition_difference": True,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "protocol/INIT_SAMPLER_PAIRING.json", pairing)
    write_csv(ROOT / "protocol/MATRIX_9.csv", list(matrix_rows[0]), matrix_rows)
    return pairing


def main() -> int:
    if (ROOT / "protocol/SOURCE_MAP.json").exists():
        raise FileExistsError("protocol locks already exist; preserve and inspect before rerun")
    model_sources = [load_accepted_model(seed, role) for seed in SEEDS for role in ("S", "V")]
    root_receipts = verify_root_receipts()
    references = accepted_prediction_entries()
    v2_seal = read_json(V2 / "protocol/PREDICTION_BATCH_SEAL.json")
    v3_seal = read_json(V3 / "protocol/PREDICTION_BATCH_SEAL.json")
    v5_seal = read_json(V5 / "qa/PREDICTION_SEAL.json")
    v4_source = read_json(V4_SOURCE_MAP)
    train_data = load_train_prefix()
    dev_data = load_dev_prefix()
    train_rows = receiver_rows(train_data)
    dev_rows = receiver_rows(dev_data)
    if len(train_rows) != 1280 or len({row["episode_id"] for row in train_rows}) != 220:
        raise RuntimeError("accepted outer-train receiver support differs from 1280/220")
    if len(dev_rows) != 185 or len({row["episode_id"] for row in dev_rows}) != 47:
        raise RuntimeError("accepted exposed-dev receiver support differs from 185/47")
    dev_group_tasks: dict[str, set[int]] = {}
    for row in dev_rows:
        dev_group_tasks.setdefault(str(row["source_cluster"]), set()).add(int(row["task_id"]))
    if any(len(task_set) != 1 for task_set in dev_group_tasks.values()):
        raise RuntimeError("dev source group crosses task boundaries")
    tasks_support = {}
    for task in TASKS:
        train_mask = np.asarray(train_data["task_id"], dtype=np.int64) == task
        dev_mask = np.asarray(dev_data["task_id"], dtype=np.int64) == task
        tasks_support[str(task)] = {
            "outer_train_anchors": int(train_mask.sum()),
            "outer_train_episodes": len(np.unique(train_data["episode_id"][train_mask])),
            "exposed_dev_anchors": int(dev_mask.sum()),
            "exposed_dev_episodes": len(np.unique(dev_data["episode_id"][dev_mask])),
            "exposed_dev_source_groups": len({str(x) for x in dev_data["source_cluster"][dev_mask]}),
        }
    train_artifact = next((row for row in v4_source["source_artifacts"] if row.get("role") == "outer_train_with_train_targets"), None)
    if train_artifact is None or sha_file(TRAIN_NPZ) != train_artifact.get("sha256"):
        raise RuntimeError("V4 source map train archive digest mismatch")

    pairing = write_pairing(train_data, model_sources)
    model_recipe = read_json(V4_RECIPE)
    init_compat = read_json(V4_INIT_COMPAT)
    if model_recipe.get("status") != "LOCKED_BEFORE_V4_FITS":
        raise RuntimeError("accepted V4 recipe lock is not in the expected pre-fit locked state")
    if init_compat.get("status") != "PASS_REUSE_CONTRACT_SUPPORTED" or len(init_compat.get("fits", [])) != 3:
        raise RuntimeError("accepted V4 common INIT compatibility contract failed")
    v4_common_receipt = read_json(V4_COMPAT)
    if v4_common_receipt.get("status") != "PASS" or len(v4_common_receipt.get("fit_receipts", [])) != 6:
        raise RuntimeError("accepted V4 common INIT/recipe receipt failed")

    # Lock V7 recipes and analysis before any student fit or dev-target read.
    recipe_lock = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_STAGE0",
        "created_utc": utc_now(),
        "source_recipe_path": str(V4_RECIPE),
        "source_recipe_sha256": sha_file(V4_RECIPE),
        "source_init_compatibility_path": str(V4_INIT_COMPAT),
        "source_init_compatibility_sha256": sha_file(V4_INIT_COMPAT),
        "source_common_init_receipt_path": str(V4_COMPAT),
        "source_common_init_receipt_sha256": sha_file(V4_COMPAT),
        "recipe": model_recipe["recipe"],
        "student_video_mode": "TS11-D",
        "waypoint_mode": None,
        "loss_source": str(V4 / "code/locked_source/d3_eq_contract.py"),
        "loss_source_sha256": sha_file(V4 / "code/locked_source/d3_eq_contract.py"),
        "scheduler_source": str(V4 / "code/v2_scheduler.py"),
        "scheduler_source_sha256": sha_file(V4 / "code/v2_scheduler.py"),
        "sampler_source": str(V4 / "code/locked_source/d3_eq_contract.py"),
        "sampler_source_sha256": sha_file(V4 / "code/locked_source/d3_eq_contract.py"),
        "dropout_source": str(V4 / "code/locked_source/D3_model.py"),
        "dropout_source_sha256": sha_file(V4 / "code/locked_source/D3_model.py"),
        "external_encoder": "frozen accepted Q0 feature cache; no encoder training or extraction",
        "target_adaptation": "one complete native-coordinate trajectory Z passed to the original V2 per-sample objective",
        "objective": "position Full-28 L1 plus 0.1 future-internal safe-FD L1",
        "position_loss": "0.5*(hand 14-axis mean + arm 14-axis mean) over 30 future steps",
        "fd_loss": "0.5*(hand 14-axis mean + arm 14-axis mean) over 29 future-internal differences",
        "seam_training_loss": False,
        "task_episode_weights": "W/(5*E_task*n_episode_windows), FP64 bookkeeping",
        "weight_reduction": "sum(weight_i*loss_i)/B_actual; effective update accumulates microbatches against group_n",
        "teacher_forward_in_fit": False,
        "teacher_targets_stop_gradient": True,
        "dropout_probability_from_locked_model_source": 0.1,
        "checkpoint_rule": "last epoch 160, no dev/tune selection",
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "protocol/RECIPE_LOCK.json", recipe_lock)
    analysis_lock = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_STAGE0",
        "created_utc": utc_now(),
        "development_metrics": ["position_mae", "safe_fd_l1"],
        "seam_metric": "safe_fd_l1, reported separately",
        "primary_contrasts": ["MIX_VS_GT", "MIX_VS_SELF"],
        "secondary_contrasts": [
            "SELF_VS_GT", "MIX_VS_OLD_V", "MIX_VS_TEACHER",
            "MIX_VS_BIAS_STATE", "MIX_VS_BIAS_VIDEO",
        ],
        "gain_sign": "reference loss minus candidate loss; positive favors candidate",
        "task_aggregation": "anchors within source group, source groups within task, tasks equally",
        "seed_aggregation": "three paired source/student instances equally; report sample SD",
        "paired_source_group_bootstrap": {"draws": 5000, "seed": 2026101507, "ci": "two-sided percentile 95%", "multiplicity_adjustment": "none"},
        "bootstrap_scope": "conditional on three paired source/student instances and exposed development groups",
        "seed_block_loo": "drop all three student programs and corresponding S/V/T reference for one seed",
        "task_loo": "drop one task and re-average remaining task macros",
        "p_values": False,
        "equivalence_or_noninferiority_claim": False,
        "teacher_retention_ratio": {
            "formula": "(L(V_i)-L(student))/(L(V_i)-L(T_i))",
            "per_seed": True,
            "overall": "ratio of equally averaged three-seed losses, not mean of seed ratios",
            "epsilon": "1e-12*max(1,abs(L(V)),abs(L(T)))",
            "ratio_ci": False,
            "interpretation": "descriptive MAE point estimate only",
        },
        "teacher_excess_point_loss": "L(student)-L(T)<=0; no added margin",
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "protocol/ANALYSIS_LOCK.json", analysis_lock)
    pairing["analysis_lock_sha256"] = sha_file(ROOT / "protocol/ANALYSIS_LOCK.json")
    pairing["recipe_lock_sha256"] = sha_file(ROOT / "protocol/RECIPE_LOCK.json")
    atomic_json(ROOT / "protocol/INIT_SAMPLER_PAIRING.json", pairing)

    source_map = {
        "schema_version": 1,
        "status": "LOCKED",
        "created_utc": utc_now(),
        "study": "post_V6_single_fusion_head_supervised_soft_target_development",
        "root": str(ROOT),
        "repo_root": str(REPO),
        "git_state": "UNKNOWN; git executable is unavailable in this shell; no repository source file is modified",
        "host_observed": os.environ.get("EE6008_HOST_LABEL", "shared-host"),
        "unix_user_observed": 'project-team',
        "scheduler_environment_detected": False,
        "python_bin": str(PROJECT / "envs/jyc/bin/python"),
        "python_version": "3.12.14",
        "source_roots_read_only": [str(V2), str(V3), str(V4), str(V5), str(V6)],
        "old_root_writes": 0,
        "final_heldout_payload_access": 0,
        "data": {
            "train_npz": str(TRAIN_NPZ),
            "train_npz_sha256": sha_file(TRAIN_NPZ),
            "dev_prefix_npz": str(DEV_NPZ),
            "dev_prefix_npz_sha256": sha_file(DEV_NPZ),
            "dev_target_archive": str(V2 / "data/DEV_TARGETS_SCORER_ONLY.npz"),
            "dev_target_archive_opened": False,
            "normalizer": str(NORMALIZER),
            "normalizer_sha256": sha_file(NORMALIZER),
            "metric_scale_lock": str(METRIC_LOCK),
            "metric_scale_lock_sha256": sha_file(METRIC_LOCK),
            "feature_cache_manifest": read_json(V4 / "inventory/RESOLVED_PATHS.json")["feature_cache"],
            "feature_cache_manifest_sha256": read_json(V4 / "inventory/RESOLVED_PATHS.json")["feature_cache_manifest_sha256"],
            "corrected_split": str(V4 / "inventory/INNER_SPLIT.csv"),
            "corrected_split_sha256": sha_file(V4 / "inventory/INNER_SPLIT.csv"),
            "ordered_receiver_ids_source": "accepted ordered train archive row order; V7 source model outputs are joined by anchor/episode/task/anchor-frame/source-cluster",
            "split_rehashed": False,
        },
        "models": model_sources,
        "evaluation_reference_bundles": references,
        "accepted_parent_roots": root_receipts,
        "acceptance_evidence": {
            "V2_prediction_seal": {"path": str(V2_SEAL), "sha256": sha_file(V2_SEAL), "status": v2_seal["status"]},
            "V3_prediction_seal": {"path": str(V3_SEAL), "sha256": sha_file(V3_SEAL), "status": v3_seal["status"]},
            "V4_source_map": {"path": str(V4_SOURCE_MAP), "sha256": sha_file(V4_SOURCE_MAP), "fold_membership": v4_source["fold_membership"]},
            "V5_prediction_seal": {"path": str(V5_SEAL), "sha256": sha_file(V5_SEAL), "status": v5_seal["status"]},
            "V5_final_receipt": {"path": str(V5_RECEIPT), "sha256": sha_file(V5_RECEIPT), "status": "PASS", "entries": 255},
            "V6_final_receipt": {"path": str(V6_RECEIPT), "sha256": sha_file(V6_RECEIPT), "status": "PASS", "entries": 98},
            "V4_common_init_receipt": {"path": str(V4_COMPAT), "sha256": sha_file(V4_COMPAT), "status": v4_common_receipt["status"]},
            "V4_recipe_lock": {"path": str(V4_RECIPE), "sha256": sha_file(V4_RECIPE), "status": model_recipe["status"]},
        },
    }
    atomic_json(ROOT / "protocol/SOURCE_MAP.json", source_map)

    fold_audit = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "split_path": str(V4 / "inventory/INNER_SPLIT.csv"),
        "split_sha256": sha_file(V4 / "inventory/INNER_SPLIT.csv"),
        "split_rehashed": False,
        "outer_train": {"anchors": 1280, "episodes": 220, "source_groups": 220},
        "exposed_dev": {"anchors": 185, "episodes": 47, "source_groups": len(dev_group_tasks)},
        "task_support": tasks_support,
        "inner_tune_role": {
            "previously_used_by_teacher_fits_or_selection": True,
            "treated_as_fresh_validation": False,
            "formal_student_fit_includes_all_1280_outer_train_anchors": True,
            "used_for_checkpoint_selection": False,
        },
        "source_group_cross_task_validation": "PASS",
        "final_test_or_heldout_access": 0,
    }
    atomic_json(ROOT / "inventory/FOLD_ROLE_AUDIT.json", fold_audit)

    v6_macro = {}
    with (V6 / "results/FAMILY_EFFECTS.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_type") == "family_mean" and row.get("scope") == "macro" and row.get("task_id", "") == "" and row.get("family") in {"SS", "SV", "AA", "VV"}:
                if row.get("metric") == "seam_safe_fd_l1":
                    v6_macro[row["family"]] = float(row["mean_pair_loss"])
    expected_seam = {"SS": 0.7555581231779632, "SV": 0.765530337884103, "AA": 0.7673374477133187, "VV": 0.7824519395259427}
    if v6_macro != expected_seam:
        raise RuntimeError(f"V6 seam family table differs from the V7 context supplied: {v6_macro}")
    addendum = (
        "# V6 Context Addendum\n\n"
        "Append-only context for V7. The accepted V6 family table orders seam safe-FD as "
        "SS 0.755558 < SV 0.765530 < AA 0.767337 < VV 0.782452. "
        "V7 uses this table ordering wherever earlier V6 prose is ambiguous. "
        "This addendum does not edit V6 artifacts or recompute V6 losses.\n\n"
        f"Source: {V6 / 'results/FAMILY_EFFECTS.csv'}; V6 checksum receipt: "
        f"{V6_RECEIPT} ({sha_file(V6_RECEIPT)}).\n"
    )
    (ROOT / "inventory/V6_CONTEXT_ADDENDUM.md").write_text(addendum, encoding="utf-8")

    run_context = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "host": os.environ.get("EE6008_HOST_LABEL", "shared-host"),
        "user": 'project-team',
        "project_root": str(PROJECT),
        "repo_root": str(REPO),
        "branch_commit_dirty_state": "UNKNOWN; git executable unavailable and formal source repo state is not available in this shell",
        "python_bin": str(PROJECT / "envs/jyc/bin/python"),
        "python_version": "3.12.14",
        "scheduler_environment_detected": False,
        "gpu_queries_before_approval": 0,
        "gpu_workload_started": False,
        "old_root_writes": 0,
        "final_heldout_payload_access": 0,
    }
    atomic_json(ROOT / "RUN_CONTEXT.json", run_context)
    print(json.dumps({
        "status": "PASS",
        "model_sources": len(model_sources),
        "evaluation_reference_bundles": len(references),
        "matrix_rows": 9,
        "train_anchors": 1280,
        "dev_prefix_anchors": 185,
        "V5_manifest_entries": 255,
        "V6_manifest_entries": 98,
        "dev_target_values_read": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
