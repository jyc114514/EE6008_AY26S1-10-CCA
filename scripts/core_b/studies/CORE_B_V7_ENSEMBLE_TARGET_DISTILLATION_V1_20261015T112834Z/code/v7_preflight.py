#!/usr/bin/env python3
"""Verify frozen V2/V4/V6 inputs and write V7 protocol/source receipts."""
from __future__ import annotations
import os

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = PROJECT / "reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z"
V2 = PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
V4 = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
V5 = PROJECT / "reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
V6 = PROJECT / "reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
NORMALIZER_SHA = "22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d"
TRAIN_SHA = "28c446ce091e00e73c2d885f3a1552102d23fa5b832feecac497a8547e1aa0b3"
SPLIT_SHA = "ba31002f0a17b826f57cec93c7dfaa484d76b01a07fec5b9cda10de03fe041a8"
METRIC_SHA = "dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32"
CACHE_SHA = "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_file(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha(path)
    if expected and actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("MODEL_MANIFEST.json")):
        rows.append((path, read(path)))
    return rows


def choose_model(root: Path, role: str, seed: int) -> tuple[Path, dict[str, Any]]:
    if role == "A_COMMON":
        candidates = [(p, d) for p, d in manifests(V4 / "model_only/formal")
                      if d.get("program_id") == "A_STATE_COMMON" and int(d.get("seed", -1)) == seed]
    else:
        model_id = "G-S" if role == "S_OPT" else "G-V"
        candidates = [(p, d) for p, d in manifests(V2 / "model_only/formal")
                      if d.get("model_id") == model_id and int(d.get("seed", -1)) == seed]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one accepted {role} manifest for seed {seed}; found {len(candidates)}")
    return candidates[0]


def main() -> int:
    if not ROOT.is_dir() or not (ROOT / "coordination/OWNER.lock").is_file():
        raise RuntimeError("V7 root or exclusive owner lock is missing")
    v6_source = read(V6 / "inventory/SOURCE_MAP.json")
    v6_fold = read(V6 / "inventory/ACTUAL_FOLD_REUSE.json")
    v6_accept = read(V6 / "qa/FINAL_ACCEPTANCE.json")
    v6_exposure = read(V6 / "inventory/OUTCOME_EXPOSURE.json")
    if v6_fold.get("status") != "PASS" or v6_accept.get("status") != "complete":
        raise RuntimeError("V6 fold or terminal acceptance is not PASS/complete")
    if v6_source.get("final_heldout_payload_access") != 0 or v6_accept.get("final_heldout_payload_access") != 0:
        raise RuntimeError("V6 final/heldout access boundary is not zero")
    if sha(V6 / "CHECKSUMS.sha256") != "930df21ec2e95d4de74ac4ac6eb17c15bc2e00d785a0e7a4279855c1c41ce81b":
        raise RuntimeError("V6 checksum manifest identity differs from the attached V7 source lock")

    verified: list[dict[str, Any]] = []
    for row in v6_source["source_artifacts"]:
        path = Path(row["path"])
        actual = require_file(path, row["sha256"])
        verified.append({"path": str(path), "role": row["role"], "sha256": actual, "kind": "source_artifact"})
    for row in v6_source["source_prediction_bundles"]:
        prediction = Path(row["prediction_path"])
        rows_path = Path(row["prediction_rows_path"])
        seal_path = Path(row["seal_path"])
        verified.append({"path": str(prediction), "role": f"{row['role']}_seed{row['seed']}",
                         "sha256": require_file(prediction, row["prediction_sha256"]), "kind": "dev_prediction"})
        verified.append({"path": str(rows_path), "role": f"{row['role']}_rows_seed{row['seed']}",
                         "sha256": require_file(rows_path, row["prediction_rows_sha256"]), "kind": "dev_prediction_rows"})
        verified.append({"path": str(seal_path), "role": f"{row['role']}_seal_seed{row['seed']}",
                         "sha256": require_file(seal_path, row["seal_sha256"]), "kind": "prediction_seal"})

    resolved = read(V4 / "inventory/RESOLVED_PATHS.json")
    split_path = V2 / "protocol/INNER_SPLIT.csv"
    train_path = Path(resolved["train_npz"])
    dev_prefix_path = Path(resolved["dev_prefix_npz"])
    normalizer_path = Path(resolved["formal_normalizer_path"])
    metric_path = Path(resolved["metric_scale_lock_path"])
    cache_manifest_path = Path(resolved["feature_cache"])
    fixed_assets = {
        "split": (split_path, SPLIT_SHA),
        "train_selected_h30": (train_path, TRAIN_SHA),
        "dev_prefixes_no_labels": (dev_prefix_path, resolved["dev_prefix_sha256"]),
        "formal_normalizer": (normalizer_path, NORMALIZER_SHA),
        "metric_scale_lock": (metric_path, METRIC_SHA),
        "q0_feature_cache_manifest": (cache_manifest_path, CACHE_SHA),
    }
    asset_hashes = {}
    for name, (path, expected) in fixed_assets.items():
        asset_hashes[name] = {"path": str(path), "sha256": require_file(path, expected)}
    cache = read(cache_manifest_path)
    if cache.get("status") != "PASS" or cache.get("logical_shape") != [20079, 2, 16, 4, 768]:
        raise RuntimeError("accepted Q0 feature manifest has a status/shape mismatch")

    common_init = read(V4 / "qa/COMMON_INIT_AND_RECIPE.json")
    init_receipts = {int(r["seed"]): r for r in common_init["fit_receipts"] if r["program_id"] == "A_STATE_COMMON"}
    if len(init_receipts) != 3:
        raise RuntimeError("V4 common INIT receipts do not cover the three formal seeds")
    teacher_sources: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        init_path = V2 / f"model_only/INIT/seed_{seed}/shared_5task_base.safetensors"
        init_sha = require_file(init_path, init_receipts[seed]["init_sha256"])
        seed_roles = {}
        for role in ("S_OPT", "A_COMMON", "V_MATCHED"):
            manifest_path, manifest = choose_model(V2 if role != "A_COMMON" else V4, role, seed)
            if manifest.get("status") != "PASS" or int(manifest.get("epochs", 0)) != 160:
                raise RuntimeError(f"teacher {role} seed {seed} is not an accepted 160-epoch model")
            if manifest.get("normalizer_sha256") != NORMALIZER_SHA or manifest.get("training_input_sha256", manifest.get("train_selected_npz_sha256")) != TRAIN_SHA:
                raise RuntimeError(f"teacher {role} seed {seed} has an unexpected normalizer or train input")
            model_path = Path(manifest.get("model_file", manifest.get("model_path", "")))
            if not model_path.is_file():
                model_path = manifest_path.parent / "model.safetensors"
            model_sha = require_file(model_path, manifest.get("model_file_sha256", manifest.get("model_sha256")))
            model_init = Path(manifest.get("init_path", ""))
            if model_init.resolve() != init_path.resolve():
                raise RuntimeError(f"teacher {role} seed {seed} did not use the accepted fresh per-seed INIT")
            if manifest.get("init_state_hash") != init_receipts[seed]["init_state_hash"]:
                raise RuntimeError(f"teacher {role} seed {seed} INIT state hash differs")
            mode = str(manifest.get("video_mode", "STATE-CTRL" if role == "A_COMMON" else ""))
            expected_mode = "TS11-D" if role == "V_MATCHED" else "STATE-CTRL"
            if mode != expected_mode:
                raise RuntimeError(f"teacher {role} seed {seed} mode mismatch: {mode} != {expected_mode}")
            seed_roles[role] = {
                "seed": seed, "role": role, "model_path": str(model_path), "model_sha256": model_sha,
                "model_manifest_path": str(manifest_path), "model_manifest_sha256": sha(manifest_path),
                "model_state_hash": manifest.get("final_state_hash", manifest.get("model_state_hash")),
                "video_mode": mode, "epochs": 160, "lr": float(manifest.get("lr", manifest.get("learning_rate"))),
                "init_path": str(init_path), "init_sha256": init_sha,
                "init_state_hash": init_receipts[seed]["init_state_hash"],
                "normalizer_path": str(normalizer_path), "normalizer_sha256": NORMALIZER_SHA,
                "fit_role": "accepted_formal_model",
            }
            verified.append({"path": str(model_path), "role": f"{role}_seed{seed}_teacher_model",
                             "sha256": model_sha, "kind": "frozen_model"})
            verified.append({"path": str(manifest_path), "role": f"{role}_seed{seed}_model_manifest",
                             "sha256": sha(manifest_path), "kind": "model_manifest"})
        teacher_sources[str(seed)] = seed_roles

    recipe = read(V4 / "protocol/MODEL_RECIPE_LOCK.json")["recipe"]
    if (recipe["lr"] != 0.0005 or recipe["epochs"] != 160 or recipe["effective_batch"] != 64
            or recipe["microbatch_size"] != 64 or recipe["scheduler"] != "update-driven cosine; eta_min=1e-5; T_max=3200"):
        raise RuntimeError("V4 accepted common training recipe differs from the V7 lock")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    study_lock = {
        "schema_version": 1, "status": "LOCKED_BEFORE_V7_TRAINING", "created_utc": now,
        "study": "post_V6_ensemble_target_distillation_development",
        "tasks": list(TASKS), "formal_seeds": list(SEEDS),
        "fold_reuse_path": str(V6 / "inventory/ACTUAL_FOLD_REUSE.json"),
        "split_path": str(split_path), "split_sha256": SPLIT_SHA,
        "outer_train": {"episodes": 220, "anchors": 1280}, "dev": {"episodes": 47, "anchors": 185},
        "student_inputs": {"ST": "STATE-CTRL", "VD": "TS11-D"},
        "programs": ["ST_SELF", "ST_SS", "ST_SV", "VD_SELF", "VD_SS", "VD_SV"],
        "teachers": {"S": "G-S optimized STATE-CTRL, lr=0.001/160",
                     "A": "V4 A_STATE_COMMON STATE-CTRL, lr=0.0005/160",
                     "V": "G-V matched TS11-D, lr=0.0005/160",
                     "SS50": "0.5*S+0.5*A", "SV50": "0.5*S+0.5*V",
                     "SELF_ST": "A alias", "SELF_VD": "V alias"},
        "recipe": {"lr": 0.0005, "epochs": 160, "optimizer": "AdamW", "betas": [0.9, 0.999],
                   "eps": 1e-8, "weight_decay": 1e-4, "effective_batch": 64, "microbatch_size": 64,
                   "gradient_clip_norm": 1.0, "dtype": "float32", "amp": False,
                   "scheduler": "update-driven cosine", "eta_min": 1e-5, "T_max": 3200,
                   "checkpoint_epoch": 160, "sampler": "default_rng(seed+epoch).shuffle"},
        "normalizer_path": str(normalizer_path), "normalizer_sha256": NORMALIZER_SHA,
        "metric_scale_lock_path": str(metric_path), "metric_scale_lock_sha256": METRIC_SHA,
        "feature_cache_manifest_path": str(cache_manifest_path), "feature_cache_manifest_sha256": CACHE_SHA,
        "feature_cache_shape": [20079, 2, 16, 4, 768], "new_encoder_extraction": 0,
        "dev_labels_used_for_training_or_selection": False, "final_heldout_payload_access": 0,
        "teacher_train_predictions_are_oof": False, "teacher_parameter_updates": 0,
        "teacher_train_prediction_bundles_expected": 9, "formal_student_fits_expected": 18,
        "new_hard_only_baseline_fits_expected": 0, "new_screening_fits_expected": 0,
        "positive_gain_completion_gate": False, "minimum_support_completion_gate": False,
        "dev_beta_or_checkpoint_selection": False, "git_state": "UNKNOWN: no git executable/.git metadata in accepted V4 locator",
    }
    target_lock = {
        "schema_version": 1, "status": "LOCKED_BEFORE_V7_TRAINING", "created_utc": now,
        "beta": 0.5, "native_target_formula": "Z_train=(1-beta)*Y_train+beta*T_train",
        "objective": "position_L1(student,Z_train)+0.1*future_internal_safe_FD_L1(student,Z_train)",
        "teacher_combo_dtype": "float64", "teacher_prediction_space": "native numeric command values",
        "normalizer_refit_on_soft_target": False, "normalizer_source": "accepted outer-train real Y",
        "normalizer_sha256": NORMALIZER_SHA, "metric_scales_refit": False,
        "student_forward_inputs": ["accepted state/history", "task ID", "known command anchor", "matched prefix Q0 cache for VD"],
        "student_forward_receives_Y_or_T": False, "teacher_forward_receives_future_Y": False,
        "separate_L1_mixture": False, "classification_KL_or_temperature": False, "seam_loss_added": False,
        "teacher_training_membership_oof": False,
    }
    write(ROOT / "protocol/STUDY_LOCK.json", study_lock)
    write(ROOT / "protocol/TARGET_MIX_LOCK.json", target_lock)
    write(ROOT / "protocol/TEACHER_ROLE_MAP.json", {
        "schema_version": 1, "created_utc": now, "sources": teacher_sources,
        "compositions": {str(seed): {"SS50": ["S_OPT", "A_COMMON"], "SV50": ["S_OPT", "V_MATCHED"],
                                      "SELF_ST": "A_COMMON alias", "SELF_VD": "V_MATCHED alias"} for seed in SEEDS},
        "composition_weights": [0.5, 0.5], "native_float64": True,
        "outer_train_prediction_status": "not present in V6 accepted prediction manifest; nine source replays required",
        "oof": False, "train_prediction_use": "fixed teacher supervision only; no coefficient fitting",
    })
    write(ROOT / "inventory/FOLD_REUSE.json", v6_fold)
    write(ROOT / "inventory/OUTCOME_EXPOSURE.json", {
        "schema_version": 1, "created_utc": now, "prior_v5_v6": v6_exposure,
        "prior_primary_loss_evaluations_v5_plus_v6": 4440,
        "prior_target_archive_loads_v5_plus_v6": 6,
        "prior_v6_target_window_materializations": 370,
        "unique_dev_windows": 185, "legacy_counter_35523_unit": "unverified; not added to 4440",
        "v7_teacher_train_forward_rows": 0, "v7_train_target_windows_presented": 0,
        "v7_student_dev_logical_evaluations": 0, "v7_dev_target_archive_load_operations": 0,
        "v7_final_heldout_payload_access": 0,
    })
    source_map = {
        "schema_version": 1, "created_utc": now, "status": "PASS",
        "source_map_v6_path": str(V6 / "inventory/SOURCE_MAP.json"),
        "source_map_v6_sha256": sha(V6 / "inventory/SOURCE_MAP.json"),
        "v6_prediction_seal_sha256": read(V6 / "qa/PREDICTION_SEAL.json").get("seal_sha256"),
        "fixed_assets": asset_hashes, "verified_files": verified,
        "teacher_models": teacher_sources,
        "dev_reference_prediction_bundles": v6_source["source_prediction_bundles"],
        "source_roots_read_only": True, "new_extraction": 0, "final_heldout_payload_access": 0,
    }
    write(ROOT / "inventory/SOURCE_MAP.json", source_map)
    audit = {
        "schema_version": 1, "status": "PASS", "checked_utc": now,
        "v6_final_acceptance": "complete", "v6_checksum_manifest_sha256": sha(V6 / "CHECKSUMS.sha256"),
        "v6_source_artifacts_checked": len(v6_source["source_artifacts"]),
        "v6_prediction_bundle_payloads_checked": len(v6_source["source_prediction_bundles"]),
        "frozen_teacher_models_checked": sum(len(x) for x in teacher_sources.values()),
        "fixed_assets": asset_hashes, "v6_final_heldout_payload_access": 0,
        "dev_target_archive_content_opened": False, "old_root_writes": 0,
    }
    write(ROOT / "qa/SOURCE_AUDIT.json", audit)
    print(json.dumps({"status": "PASS", "v6_source_artifacts": audit["v6_source_artifacts_checked"],
                      "v6_dev_prediction_bundles": audit["v6_prediction_bundle_payloads_checked"],
                      "teacher_models": audit["frozen_teacher_models_checked"],
                      "train_anchors": 1280, "dev_anchors": 185, "dev_target_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
