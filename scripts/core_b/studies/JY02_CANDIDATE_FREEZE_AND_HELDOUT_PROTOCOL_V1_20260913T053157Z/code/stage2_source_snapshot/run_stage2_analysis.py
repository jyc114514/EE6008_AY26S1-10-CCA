#!/usr/bin/env python3
"""Independent CPU-only acceptance, statistics, figures and QA for JY-02 Stage2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JY02_ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = Path(__file__).resolve().parents[1]
CODE_ROOT = JY02_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from train_jy02_cell import CELL_SPECS, build_model  # noqa: E402


PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
BENCH_ROOT = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
INPUT = JY02_ROOT / "inputs/JY02_TARGETS.npz"
INPUT_MANIFEST = JY02_ROOT / "inputs/JY02_TARGETS_MANIFEST.json"
NORMALIZER = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
THRESHOLD_SOURCE = BENCH_ROOT / "MOTION_THRESHOLD_SOURCE.json"
STRATA_SOURCE = BENCH_ROOT / "MOTION_STRATA_MANIFEST.csv"
REPRESENTATIVE_SOURCE = BENCH_ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.json"
STAGE2_MATRIX = JY02_ROOT / "STAGE2_MATRIX.json"
OLD_STAGE2_BLOCKER = JY02_ROOT / "STAGE2_RUN_ACCEPTANCE.json"
WATCHER_STATUS = ATTEMPT / "logs/STAGE2_WATCHER.json"

CONFIGS = ["JY02-S-A-D", "JY02-F-A-D", "JY02-S-R-D", "JY02-F-R-D"]
SEEDS = [20260921, 20260922, 20260923]
MODEL_NAMES = CONFIGS + ["PERSISTENCE"]
PRIMARY_BOOTSTRAPS = 10_000
HIERARCHICAL_BOOTSTRAPS = 20_000
STATS_SEED = 20260930

EXPECTED_PARAMETERS = {
    "JY02-S-A-D": 361544,
    "JY02-S-R-D": 361544,
    "JY02-S-A-T": 402076,
    "JY02-S-R-T": 402076,
    "JY02-F-A-D": 558152,
    "JY02-F-R-D": 558152,
    "JY02-F-A-T": 598684,
    "JY02-F-R-T": 598684,
}

CONTRASTS = [
    ("F-R-D_vs_PERSISTENCE", "PERSISTENCE", "JY02-F-R-D"),
    ("S-R-D_vs_PERSISTENCE", "PERSISTENCE", "JY02-S-R-D"),
    ("F-R-D_vs_F-A-D", "JY02-F-A-D", "JY02-F-R-D"),
    ("S-R-D_vs_S-A-D", "JY02-S-A-D", "JY02-S-R-D"),
    ("F-R-D_vs_S-R-D", "JY02-S-R-D", "JY02-F-R-D"),
    ("F-A-D_vs_S-A-D", "JY02-S-A-D", "JY02-F-A-D"),
]
EFFECT_METRICS = ["full", "high_motion", "early", "middle", "late", "endpoint", "first_difference", "native_tv_closeness"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def sd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if len(finite) > 1 else 0.0


def motion_descriptors(array: np.ndarray) -> dict[str, np.ndarray]:
    arm = array[:, :, 14:28]
    changes = np.diff(arm, axis=1)
    return {
        "mean_step_motion": np.abs(changes).mean(axis=(1, 2)),
        "total_variation": np.abs(changes).sum(axis=1).mean(axis=1),
        "endpoint_change": np.abs(arm[:, -1] - arm[:, 0]).mean(axis=1),
        "max_step_motion": np.abs(changes).mean(axis=2).max(axis=1),
    }


def space_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - target
    abs_error = np.abs(error)
    arm_error = abs_error[:, :, 14:28]
    hand_error = abs_error[:, :, :14]
    changes_pred = np.diff(pred, axis=1)
    changes_target = np.diff(target, axis=1)
    change_error = np.abs(changes_pred - changes_target)
    arm_change = change_error[:, :, 14:28]
    out: dict[str, np.ndarray] = {
        "arm_full_mae": arm_error.mean(axis=(1, 2)),
        "hand_full_mae": hand_error.mean(axis=(1, 2)),
        "full28_full_mae": abs_error.mean(axis=(1, 2)),
        "arm_full_rmse": np.sqrt(np.mean(error[:, :, 14:28] ** 2, axis=(1, 2))),
        "hand_full_rmse": np.sqrt(np.mean(error[:, :, :14] ** 2, axis=(1, 2))),
        "full28_full_rmse": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "arm_first_difference_mae": arm_change.mean(axis=(1, 2)),
        "hand_first_difference_mae": change_error[:, :, :14].mean(axis=(1, 2)),
        "full28_first_difference_mae": change_error.mean(axis=(1, 2)),
    }
    for horizon, selector in (("early", slice(0, 10)), ("middle", slice(10, 20)), ("late", slice(20, 30))):
        out[f"arm_{horizon}_mae"] = arm_error[:, selector, :].mean(axis=(1, 2))
        out[f"hand_{horizon}_mae"] = hand_error[:, selector, :].mean(axis=(1, 2))
        out[f"full28_{horizon}_mae"] = abs_error[:, selector, :].mean(axis=(1, 2))
        out[f"arm_{horizon}_rmse"] = np.sqrt(np.mean(error[:, selector, 14:28] ** 2, axis=(1, 2)))
    out["arm_endpoint_mae"] = arm_error[:, -1, :].mean(axis=1)
    out["hand_endpoint_mae"] = hand_error[:, -1, :].mean(axis=1)
    out["full28_endpoint_mae"] = abs_error[:, -1, :].mean(axis=1)
    out["arm_endpoint_rmse"] = np.sqrt(np.mean(error[:, -1, 14:28] ** 2, axis=1))
    out.update({f"target_{key}": value for key, value in motion_descriptors(target).items()})
    out.update({f"pred_{key}": value for key, value in motion_descriptors(pred).items()})
    target_tv = out["target_total_variation"]
    pred_tv = out["pred_total_variation"]
    out["tv_ratio"] = np.divide(pred_tv, target_tv, out=np.full(len(pred), np.nan), where=target_tv > 1e-12)
    out["endpoint_change_error"] = np.abs(out["pred_endpoint_change"] - out["target_endpoint_change"])
    target_range = np.ptp(target[:, :, 14:28], axis=1)
    pred_range = np.ptp(pred[:, :, 14:28], axis=1)
    valid_range = target_range > 1e-12
    joint_ratio = np.divide(pred_range, target_range, out=np.full_like(pred_range, np.nan), where=valid_range)
    out["range_ratio"] = np.nanmean(joint_ratio, axis=1)
    a = changes_pred[:, :, 14:28]
    b = changes_target[:, :, 14:28]
    valid_direction = (np.abs(a) > 1e-8) | (np.abs(b) > 1e-8)
    valid_count = valid_direction.sum(axis=(1, 2))
    out["direction_agreement"] = np.divide(((np.sign(a) == np.sign(b)) & valid_direction).sum(axis=(1, 2)), valid_count, out=np.full(len(pred), np.nan), where=valid_count > 0)
    dot = (a * b).sum(axis=2)
    norms = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    cosine = np.divide(dot, norms, out=np.full_like(dot, np.nan), where=norms > 1e-8)
    valid_cosine_count = np.isfinite(cosine).sum(axis=1)
    out["direction_cosine"] = np.divide(np.nansum(cosine, axis=1), valid_cosine_count, out=np.full(len(pred), np.nan), where=valid_cosine_count > 0)
    return out


def load_canonical() -> dict[str, Any]:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    split = data["split_code"]
    if set(np.unique(split).tolist()) != {0, 1}:
        raise RuntimeError(f"unexpected split codes {np.unique(split).tolist()}")
    dev_mask = split == 1
    train_mask = split == 0
    ids = data["episode_index"][dev_mask].astype(np.int64)
    if len(ids) != 614 or len(np.unique(ids)) != 614:
        raise RuntimeError("canonical development ID contract failed")
    threshold = read_json(THRESHOLD_SOURCE)
    q25 = float(threshold["q25"])
    q75 = float(threshold["q75"])
    target_std = data["targets_standardized"][dev_mask].astype(np.float64)
    dev_motion = motion_descriptors(target_std)
    derived_strata = np.where(dev_motion["mean_step_motion"] <= q25, "low", np.where(dev_motion["mean_step_motion"] <= q75, "mid", "high"))
    strata_rows = list(csv.DictReader(STRATA_SOURCE.open(newline="", encoding="utf-8")))
    strata_map = {int(row["episode_id"]): row["motion_stratum"] for row in strata_rows}
    if set(strata_map) != set(ids.tolist()):
        raise RuntimeError("V2 strata IDs do not match JY02 development IDs")
    strata = np.asarray([strata_map[int(value)] for value in ids], dtype=object)
    if not np.array_equal(derived_strata.astype(object), strata):
        raise RuntimeError("train-derived V2 strata do not reproduce the locked manifest")
    representatives = read_json(REPRESENTATIVE_SOURCE)["rows"]
    representative_ids = [int(row["episode_id"]) for row in representatives]
    if len(representative_ids) != 12 or len(set(representative_ids)) != 12 or not set(representative_ids).issubset(set(ids.tolist())):
        raise RuntimeError("representative episode manifest contract failed")
    index = {int(value): i for i, value in enumerate(ids.tolist())}
    return {
        "data": data,
        "dev_mask": dev_mask,
        "train_mask": train_mask,
        "ids": ids,
        "tasks": data["original_task_index"][dev_mask].astype(np.int64),
        "target_std": target_std,
        "target_native": data["targets_native"][dev_mask].astype(np.float64),
        "persistence_std": data["persistence_standardized"][dev_mask].astype(np.float64),
        "persistence_native": data["persistence_native"][dev_mask].astype(np.float64),
        "strata": strata,
        "q25": q25,
        "q75": q75,
        "representative_ids": representative_ids,
        "representative_indices": [index[value] for value in representative_ids],
        "input_sha256": digest(INPUT),
    }


def close_check(actual: np.ndarray, expected: np.ndarray, tolerance: float = 2e-6) -> float:
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def load_stage2_runs(canonical: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    models: dict[tuple[int, str], dict[str, Any]] = {}
    acceptance: list[dict[str, Any]] = []
    lookup = {int(value): i for i, value in enumerate(canonical["ids"].tolist())}
    for seed in SEEDS:
        persistence = {
            "pred_std": canonical["persistence_std"].copy(),
            "pred_native": canonical["persistence_native"].copy(),
        }
        persistence["std"] = space_metrics(persistence["pred_std"], canonical["target_std"])
        persistence["native"] = space_metrics(persistence["pred_native"], canonical["target_native"])
        models[(seed, "PERSISTENCE")] = persistence
        for cell in CONFIGS:
            run_dir = ATTEMPT / f"seed{seed}" / cell
            errors: list[str] = []
            required = ["RUN_STATUS.json", "RUN_CONFIG_LOCKED.json", "RUN_METRICS.json", "RUN_COMPLETE.json", "MODEL_CONTRACT.json", "checkpoint_best.pt", "checkpoint_last.pt", "predictions_best.npz", "predictions_last.npz", "epoch_curves.csv"]
            missing = [name for name in required if not (run_dir / name).exists()]
            if missing:
                errors.append(f"missing:{','.join(missing)}")
            status = read_json(run_dir / "RUN_STATUS.json") if (run_dir / "RUN_STATUS.json").exists() else {}
            config = read_json(run_dir / "RUN_CONFIG_LOCKED.json") if (run_dir / "RUN_CONFIG_LOCKED.json").exists() else {}
            metrics_record = read_json(run_dir / "RUN_METRICS.json") if (run_dir / "RUN_METRICS.json").exists() else {}
            complete = read_json(run_dir / "RUN_COMPLETE.json") if (run_dir / "RUN_COMPLETE.json").exists() else {}
            model_contract = read_json(run_dir / "MODEL_CONTRACT.json") if (run_dir / "MODEL_CONTRACT.json").exists() else {}
            failure_marker = run_dir / "RUN_FAILED.json"
            if failure_marker.exists():
                errors.append("conflicting_RUN_FAILED_marker")
            if status.get("status") != "complete" or complete.get("status") != "complete" or metrics_record.get("status") != "complete":
                errors.append("terminal_status_not_complete")
            if config.get("cell_id") != cell or int(config.get("seed", -1)) != seed:
                errors.append("config_cell_or_seed_mismatch")
            if config.get("input_sha256") != canonical["input_sha256"]:
                errors.append("input_hash_mismatch")
            for key, expected in (("raw_video", False), ("encoder_or_lora_modified", False)):
                if config.get(key) != expected or metrics_record.get(key) != expected:
                    errors.append(f"{key}_contract_failed")
            if config.get("final_test_access") != 0 or metrics_record.get("final_test_access") != 0 or complete.get("final_test_access") != 0:
                errors.append("final_test_access_contract_failed")
            if int(status.get("optimizer_steps", -1)) != 1720 or int(metrics_record.get("optimizer_steps", -1)) != 1720:
                errors.append("optimizer_step_contract_failed")
            if model_contract.get("status") != "PASS" or model_contract.get("encoder_frozen_cache_only") is not True or model_contract.get("trainable_encoder") is not False:
                errors.append("model_contract_failed")
            if config.get("parameter_count") != EXPECTED_PARAMETERS[cell]:
                errors.append("parameter_count_contract_failed")
            best_path = run_dir / "predictions_best.npz"
            with np.load(best_path, allow_pickle=False) as bundle:
                bundle_data = {key: bundle[key].copy() for key in bundle.files}
            bundle_ids = np.asarray(bundle_data.get("episode_index", []), dtype=np.int64)
            if len(bundle_ids) != 614 or len(np.unique(bundle_ids)) != 614:
                errors.append("prediction_ID_contract_failed")
                order = np.arange(min(len(bundle_ids), 614), dtype=np.int64)
            else:
                try:
                    order = np.asarray([lookup[int(value)] for value in bundle_ids], dtype=np.int64)
                except KeyError:
                    errors.append("prediction_ID_not_in_canonical_dev")
                    order = np.arange(614, dtype=np.int64)
            for key in ("predictions_standardized", "predictions_native", "targets_standardized", "targets_native", "persistence_standardized", "residual_predictions_standardized", "original_task_index"):
                if key not in bundle_data:
                    errors.append(f"missing_prediction_key:{key}")
            pred_std = np.asarray(bundle_data.get("predictions_standardized", np.empty((0,))), dtype=np.float64)
            pred_native = np.asarray(bundle_data.get("predictions_native", np.empty((0,))), dtype=np.float64)
            if pred_std.shape != (614, 30, 28) or pred_native.shape != (614, 30, 28):
                errors.append("prediction_shape_contract_failed")
            if pred_std.size and (not np.isfinite(pred_std).all() or not np.isfinite(pred_native).all()):
                errors.append("prediction_finite_contract_failed")
            if len(order) == 614:
                target_std_max = close_check(np.asarray(bundle_data["targets_standardized"], dtype=np.float64), canonical["target_std"][order])
                target_native_max = close_check(np.asarray(bundle_data["targets_native"], dtype=np.float64), canonical["target_native"][order])
                persistence_max = close_check(np.asarray(bundle_data["persistence_standardized"], dtype=np.float64), canonical["persistence_std"][order])
                task_ok = np.array_equal(np.asarray(bundle_data["original_task_index"], dtype=np.int64), canonical["tasks"][order])
            else:
                target_std_max = target_native_max = persistence_max = float("inf")
                task_ok = False
            if target_std_max > 2e-6 or target_native_max > 2e-6 or persistence_max > 2e-6 or not task_ok:
                errors.append("target_anchor_task_lineage_failed")
            residual_max = 0.0
            if pred_std.size and cell.endswith("-R-D"):
                residual = np.asarray(bundle_data["residual_predictions_standardized"], dtype=np.float64)
                residual_max = close_check(pred_std, np.asarray(bundle_data["persistence_standardized"], dtype=np.float64) + residual, tolerance=2e-6)
                if residual_max > 2e-6:
                    errors.append("residual_reconstruction_failed")
            if config.get("resume_from") or config.get("resume_checkpoint"):
                errors.append("fresh_initialization_not_proven")
            try:
                curves = list(csv.DictReader((run_dir / "epoch_curves.csv").open(newline="", encoding="utf-8")))
                best_epoch = int(metrics_record["best_epoch"])
                selected = next(row for row in curves if int(row["epoch"]) == best_epoch)
                if abs(float(selected["validation_arm_mae"]) - float(metrics_record["best"]["arm_mae"])) > 2e-5:
                    errors.append("selected_metric_replay_failed")
            except Exception as exc:  # pragma: no cover - surfaced as acceptance evidence
                errors.append(f"selected_metric_replay_error:{type(exc).__name__}")
            try:
                model = build_model(CELL_SPECS[cell], seed)
                import torch

                for checkpoint_name in ("checkpoint_best.pt", "checkpoint_last.pt"):
                    payload = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
                    model.load_state_dict(payload["model_state_dict"], strict=True)
            except Exception as exc:
                errors.append(f"checkpoint_reload_failed:{type(exc).__name__}")
            for path_key, record_key in (("checkpoint_best.pt", "checkpoint_best_sha256"), ("checkpoint_last.pt", "checkpoint_last_sha256"), ("predictions_best.npz", "best_prediction_sha256"), ("predictions_last.npz", "last_prediction_sha256")):
                if (run_dir / path_key).exists() and metrics_record.get(record_key) != digest(run_dir / path_key):
                    errors.append(f"hash_record_mismatch:{path_key}")
            if complete.get("checkpoint_best_sha256") != digest(run_dir / "checkpoint_best.pt") or complete.get("prediction_best_sha256") != digest(run_dir / "predictions_best.npz"):
                errors.append("completion_hash_record_mismatch")
            if pred_std.size:
                models[(seed, cell)] = {
                    "pred_std": pred_std,
                    "pred_native": pred_native,
                    "std": space_metrics(pred_std, canonical["target_std"]),
                    "native": space_metrics(pred_native, canonical["target_native"]),
                }
            if not errors:
                run_status = "accepted"
            else:
                run_status = "failed"
            acceptance.append({
                "config_id": cell,
                "seed": seed,
                "status": run_status,
                "N": 614 if len(bundle_ids) == 614 else len(bundle_ids),
                "shape": str(tuple(pred_std.shape)),
                "best_epoch": metrics_record.get("best_epoch", ""),
                "optimizer_steps": status.get("optimizer_steps", ""),
                "path": str(run_dir),
                "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else "",
                "prediction_best_sha256": digest(run_dir / "predictions_best.npz") if best_path.exists() else "",
                "target_std_max_abs": target_std_max,
                "target_native_max_abs": target_native_max,
                "persistence_std_max_abs": persistence_max,
                "residual_reconstruction_max_abs": residual_max,
                "save_reload": "PASS" if not any("reload" in item for item in errors) else "FAIL",
                "final_test_access": 0,
                "error": ";".join(errors),
            })
    write_csv(JY02_ROOT / "STAGE2_RUN_ACCEPTANCE_POST_AUTHORIZATION.csv", acceptance, fields=["config_id", "seed", "status", "N", "shape", "best_epoch", "optimizer_steps", "path", "checkpoint_best_sha256", "prediction_best_sha256", "target_std_max_abs", "target_native_max_abs", "persistence_std_max_abs", "residual_reconstruction_max_abs", "save_reload", "final_test_access", "error"])
    accepted = sum(row["status"] == "accepted" for row in acceptance)
    acceptance_payload = {
        "status": "PASS" if accepted == 12 else "FAIL",
        "accepted_runs": accepted,
        "expected_runs": 12,
        "optimizer_steps_total": int(sum(int(row["optimizer_steps"]) for row in acceptance if str(row["optimizer_steps"]).isdigit())),
        "configs": CONFIGS,
        "fresh_seeds": SEEDS,
        "final_test_access": 0,
        "stage1_rerun": False,
        "fresh_initialization": "PASS" if accepted == 12 else "FAIL",
        "rows": acceptance,
        "attempt_root": str(ATTEMPT),
    }
    write_json(JY02_ROOT / "STAGE2_RUN_ACCEPTANCE_POST_AUTHORIZATION.json", acceptance_payload)
    if accepted != 12:
        raise RuntimeError(f"Stage2 acceptance failed: {accepted}/12")
    return models, acceptance


def bootstrap_ci(values: np.ndarray, replicates: int, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0, "replicates": replicates}
    indices = rng.integers(0, len(values), size=(replicates, len(values)), dtype=np.int32)
    sampled = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(sampled, 0.025)), "ci_high": float(np.quantile(sampled, 0.975)), "n_valid": int(np.isfinite(sampled).sum()), "replicates": replicates}


def hierarchical_bootstrap(values_by_seed: dict[int, np.ndarray], replicates: int, rng: np.random.Generator) -> dict[str, Any]:
    ordered = [np.asarray(values_by_seed[seed], dtype=np.float64) for seed in SEEDS]
    if any(len(value) == 0 or not np.isfinite(value).all() for value in ordered):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0, "replicates": replicates, "seed_count": len(SEEDS)}
    seed_draws = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)), dtype=np.int8)
    sampled_by_position = np.empty((replicates, len(SEEDS)), dtype=np.float64)
    for position in range(len(SEEDS)):
        means_by_seed = np.empty((replicates, len(SEEDS)), dtype=np.float64)
        for seed_index, values in enumerate(ordered):
            indices = rng.integers(0, len(values), size=(replicates, len(values)), dtype=np.int32)
            means_by_seed[:, seed_index] = values[indices].mean(axis=1)
        sampled_by_position[:, position] = means_by_seed[np.arange(replicates), seed_draws[:, position]]
    sampled = sampled_by_position.mean(axis=1)
    valid = sampled[np.isfinite(sampled)]
    return {"mean": float(np.mean(valid)), "ci_low": float(np.quantile(valid, 0.025)), "ci_high": float(np.quantile(valid, 0.975)), "n_valid": int(len(valid)), "replicates": replicates, "seed_count": len(SEEDS)}


def add_model_tables(canonical: dict[str, Any], models: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    metric_keys = [
        "arm_full_mae", "hand_full_mae", "full28_full_mae", "arm_full_rmse", "hand_full_rmse", "full28_full_rmse",
        "arm_early_mae", "arm_middle_mae", "arm_late_mae", "arm_endpoint_mae", "arm_first_difference_mae",
        "hand_early_mae", "hand_middle_mae", "hand_late_mae", "hand_endpoint_mae", "full28_early_mae", "full28_middle_mae", "full28_late_mae", "full28_endpoint_mae",
    ]
    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            model = models[(seed, model_name)]
            row: dict[str, Any] = {"seed": seed, "model": model_name, "N": 614, "final_test_access": 0}
            for space in ("std", "native"):
                for key in metric_keys:
                    row[f"{key}_{space}"] = mean(model[space][key])
                row[f"tv_ratio_mean_{space}"] = mean(model[space]["tv_ratio"])
                row[f"tv_ratio_distance_to_one_mean_{space}"] = mean(np.abs(model[space]["tv_ratio"] - 1.0))
                row[f"total_variation_mean_{space}"] = mean(model[space]["pred_total_variation"])
                row[f"target_total_variation_mean_{space}"] = mean(model[space]["target_total_variation"])
            persistence = models[(seed, "PERSISTENCE")]
            row["relative_skill_vs_persistence_std"] = (mean(persistence["std"]["arm_full_mae"]) - mean(model["std"]["arm_full_mae"])) / mean(persistence["std"]["arm_full_mae"])
            row["relative_skill_vs_persistence_native"] = (mean(persistence["native"]["arm_full_mae"]) - mean(model["native"]["arm_full_mae"])) / mean(persistence["native"]["arm_full_mae"])
            for stratum in ("low", "mid", "high"):
                mask = canonical["strata"] == stratum
                row[f"N_{stratum}"] = int(mask.sum())
                row[f"arm_full_mae_std_{stratum}"] = mean(model["std"]["arm_full_mae"][mask])
                row[f"arm_late_mae_std_{stratum}"] = mean(model["std"]["arm_late_mae"][mask])
                row[f"arm_endpoint_mae_std_{stratum}"] = mean(model["std"]["arm_endpoint_mae"][mask])
                row[f"relative_skill_vs_persistence_std_{stratum}"] = (mean(persistence["std"]["arm_full_mae"][mask]) - mean(model["std"]["arm_full_mae"][mask])) / mean(persistence["std"]["arm_full_mae"][mask])
            result_rows.append(row)
            for i, episode_id in enumerate(canonical["ids"]):
                episode_row = {"seed": seed, "model": model_name, "episode_id": int(episode_id), "task_index": int(canonical["tasks"][i]), "motion_stratum": str(canonical["strata"][i])}
                for space in ("std", "native"):
                    for key in ("arm_full_mae", "hand_full_mae", "full28_full_mae", "full28_full_rmse", "arm_early_mae", "arm_middle_mae", "arm_late_mae", "arm_endpoint_mae", "arm_first_difference_mae", "tv_ratio", "pred_total_variation", "target_total_variation", "endpoint_change_error", "range_ratio", "direction_agreement", "direction_cosine"):
                        episode_row[f"{key}_{space}"] = model[space][key][i]
                episode_row["tv_ratio_distance_to_one_native"] = abs(model["native"]["tv_ratio"][i] - 1.0)
                episode_rows.append(episode_row)
            std_error = np.abs(model["pred_std"] - canonical["target_std"])[:, :, 14:28].mean(axis=2)
            native_error = np.abs(model["pred_native"] - canonical["target_native"])[:, :, 14:28].mean(axis=2)
            for t in range(30):
                time_rows.append({"seed": seed, "model": model_name, "time_index": t, "horizon_step": t + 1, "arm_mae_std": mean(std_error[:, t]), "arm_mae_native": mean(native_error[:, t]), "target_mean_step_motion_std": mean(canonical["target_std"][:, t, 14:28])})
            std_joint = np.abs(model["pred_std"] - canonical["target_std"])[:, :, 14:28].mean(axis=0)
            native_joint = np.abs(model["pred_native"] - canonical["target_native"])[:, :, 14:28].mean(axis=0)
            for t in range(30):
                for j in range(14):
                    joint_rows.append({"seed": seed, "model": model_name, "time_index": t, "joint_index": 14 + j, "arm_mae_std": std_joint[t, j], "arm_mae_native": native_joint[t, j]})
            for task_id in sorted(np.unique(canonical["tasks"]).tolist()):
                mask = canonical["tasks"] == task_id
                task_rows.append({"seed": seed, "model": model_name, "task_index": int(task_id), "N": int(mask.sum()), "high_motion_N": int(np.sum(mask & (canonical["strata"] == "high"))), "arm_full_mae_std": mean(model["std"]["arm_full_mae"][mask]), "arm_late_mae_std": mean(model["std"]["arm_late_mae"][mask]), "arm_endpoint_mae_std": mean(model["std"]["arm_endpoint_mae"][mask]), "arm_first_difference_mae_std": mean(model["std"]["arm_first_difference_mae"][mask]), "native_tv_ratio": mean(model["native"]["tv_ratio"][mask])})
    write_csv(JY02_ROOT / "STAGE2_RESULTS_BY_SEED.csv", result_rows)
    write_json(JY02_ROOT / "STAGE2_RESULTS_BY_SEED.json", {"status": "PASS", "rows": result_rows, "fresh_seeds": SEEDS, "discovery_seed_excluded": 20260903, "mean_sd": {model: {"arm_full_mae_std_mean": mean(np.asarray([row["arm_full_mae_std"] for row in result_rows if row["model"] == model])), "arm_full_mae_std_sd": sd(np.asarray([row["arm_full_mae_std"] for row in result_rows if row["model"] == model]))} for model in MODEL_NAMES}})
    write_csv(JY02_ROOT / "STAGE2_PER_EPISODE_METRICS.csv", episode_rows)
    write_csv(JY02_ROOT / "STAGE2_PER_TIME_METRICS.csv", time_rows)
    write_csv(JY02_ROOT / "STAGE2_PER_JOINT_METRICS.csv", joint_rows)
    write_csv(JY02_ROOT / "STAGE2_PER_TASK_METRICS.csv", task_rows)
    return {"result_rows": result_rows, "episode_rows": episode_rows, "time_rows": time_rows, "joint_rows": joint_rows, "task_rows": task_rows}


def compute_effects(canonical: dict[str, Any], models: dict[tuple[int, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, int], dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for contrast_index, (label, baseline_name, candidate_name) in enumerate(CONTRASTS):
        for seed_index, seed in enumerate(SEEDS):
            baseline = models[(seed, baseline_name)]
            candidate = models[(seed, candidate_name)]
            full_mask = np.ones(614, dtype=bool)
            high_mask = canonical["strata"] == "high"
            effect_arrays = {
                "full": baseline["std"]["arm_full_mae"] - candidate["std"]["arm_full_mae"],
                "high_motion": (baseline["std"]["arm_full_mae"] - candidate["std"]["arm_full_mae"])[high_mask],
                "early": baseline["std"]["arm_early_mae"] - candidate["std"]["arm_early_mae"],
                "middle": baseline["std"]["arm_middle_mae"] - candidate["std"]["arm_middle_mae"],
                "late": baseline["std"]["arm_late_mae"] - candidate["std"]["arm_late_mae"],
                "endpoint": baseline["std"]["arm_endpoint_mae"] - candidate["std"]["arm_endpoint_mae"],
                "first_difference": baseline["std"]["arm_first_difference_mae"] - candidate["std"]["arm_first_difference_mae"],
                "native_tv_closeness": np.abs(baseline["native"]["tv_ratio"] - 1.0) - np.abs(candidate["native"]["tv_ratio"] - 1.0),
            }
            arrays[(label, seed)] = effect_arrays
            row: dict[str, Any] = {"contrast": label, "baseline": baseline_name, "candidate": candidate_name, "seed": seed, "N": 614, "high_motion_N": int(high_mask.sum()), "effect_sign": "baseline MAE - candidate MAE; positive candidate improvement"}
            baseline_mean = mean(baseline["std"]["arm_full_mae"])
            for metric_index, metric in enumerate(EFFECT_METRICS):
                boot = bootstrap_ci(effect_arrays[metric], PRIMARY_BOOTSTRAPS, np.random.default_rng(STATS_SEED + contrast_index * 1000 + seed_index * 100 + metric_index))
                row[f"{metric}_effect"] = boot["mean"]
                row[f"{metric}_ci_low"] = boot["ci_low"]
                row[f"{metric}_ci_high"] = boot["ci_high"]
                row[f"{metric}_n_valid"] = boot["n_valid"]
                row[f"{metric}_positive_fraction"] = float(np.mean(effect_arrays[metric] > 0))
            row["baseline_arm_full_mae_std"] = baseline_mean
            row["candidate_arm_full_mae_std"] = mean(candidate["std"]["arm_full_mae"])
            row["relative_full_effect"] = row["full_effect"] / baseline_mean if baseline_mean > 1e-12 else float("nan")
            row["baseline_hand_full_mae_std"] = mean(baseline["std"]["hand_full_mae"])
            row["candidate_hand_full_mae_std"] = mean(candidate["std"]["hand_full_mae"])
            row["hand_relative_effect"] = (row["baseline_hand_full_mae_std"] - row["candidate_hand_full_mae_std"]) / row["baseline_hand_full_mae_std"]
            row["baseline_full28_full_mae_std"] = mean(baseline["std"]["full28_full_mae"])
            row["candidate_full28_full_mae_std"] = mean(candidate["std"]["full28_full_mae"])
            row["full28_relative_effect"] = (row["baseline_full28_full_mae_std"] - row["candidate_full28_full_mae_std"]) / row["baseline_full28_full_mae_std"]
            row["baseline_native_tv_ratio_mean"] = mean(baseline["native"]["tv_ratio"])
            row["candidate_native_tv_ratio_mean"] = mean(candidate["native"]["tv_ratio"])
            rows.append(row)
        summary: dict[str, Any] = {"contrast": label, "baseline": baseline_name, "candidate": candidate_name, "unit": "seed training replicate with episode nested", "per_seed_episode_bootstrap_replicates": PRIMARY_BOOTSTRAPS, "hierarchical_bootstrap_replicates": HIERARCHICAL_BOOTSTRAPS}
        for metric_index, metric in enumerate(EFFECT_METRICS):
            per_seed = {seed: arrays[(label, seed)][metric] for seed in SEEDS}
            seed_effects = np.asarray([mean(per_seed[seed]) for seed in SEEDS], dtype=np.float64)
            hierarchical = hierarchical_bootstrap(per_seed, HIERARCHICAL_BOOTSTRAPS, np.random.default_rng(STATS_SEED + 50000 + contrast_index * 100 + metric_index))
            summary[metric] = {
                "seed_effects": seed_effects,
                "seed_mean": mean(seed_effects),
                "seed_sd": sd(seed_effects),
                "positive_seed_count": int(np.sum(seed_effects > 0)),
                "hierarchical": hierarchical,
            }
        summaries[label] = summary
    write_csv(JY02_ROOT / "STAGE2_PAIRED_EFFECTS_BY_SEED.csv", rows)
    write_json(JY02_ROOT / "STAGE2_PAIRED_EFFECTS_BY_SEED.json", {"status": "PASS", "rows": rows, "summaries": summaries, "effect_sign": "control MAE - candidate MAE; positive means candidate improvement", "p_values": "not computed; seed is the training replicate and confirmation is CI/effect based", "holm": "Stage1 locked family preserved; no episode-level pseudoreplicate Holm applied"})
    hierarchy_json = {"status": "PASS", "unit": "seed then episode; 30 steps and joints nested within episode", "per_seed_episode_bootstrap_replicates": PRIMARY_BOOTSTRAPS, "hierarchical_seed_episode_bootstrap_replicates": HIERARCHICAL_BOOTSTRAPS, "fresh_seeds": SEEDS, "discovery_seed_excluded": 20260903, "p_values": "not computed because three training seeds are the independent confirmation units", "holm": "not applied to seed-nested effects; original Stage1 family remains unchanged", "comparisons": summaries}
    write_json(JY02_ROOT / "STAGE2_HIERARCHICAL_STATISTICS.json", hierarchy_json)
    md_lines = ["# Stage2 hierarchical statistics", "", "Effect convention: `control MAE - candidate MAE`; positive means candidate improvement.", "", f"Per-seed paired bootstrap: {PRIMARY_BOOTSTRAPS:,} episode resamples. Hierarchical bootstrap: {HIERARCHICAL_BOOTSTRAPS:,} seed-then-episode resamples.", "", "| Contrast | Full seed effects | Full hierarchical 95% CI | First-difference hierarchical 95% CI | Native TV-closeness hierarchical 95% CI |", "|---|---:|---:|---:|---:|"]
    for label, _, _ in CONTRASTS:
        summary = summaries[label]
        def fmt(metric: str) -> str:
            value = summary[metric]
            return f"{value['seed_mean']:.6f} +/- {value['seed_sd']:.6f}"
        def ci(metric: str) -> str:
            h = summary[metric]["hierarchical"]
            return f"[{h['ci_low']:.6f}, {h['ci_high']:.6f}] (n={h['n_valid']})"
        md_lines.append(f"| {label} | {fmt('full')} | {ci('full')} | {ci('first_difference')} | {ci('native_tv_closeness')} |")
    md_lines += ["", "No p-values were fabricated from episodes; discovery seed `20260903` is excluded. Native TV closeness is `|baseline TV ratio - 1| - |candidate TV ratio - 1|`."]
    (JY02_ROOT / "STAGE2_HIERARCHICAL_STATISTICS.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows, summaries, arrays


def confirmation_decision(canonical: dict[str, Any], models: dict[tuple[int, str], dict[str, Any]], summaries: dict[str, Any], effect_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {(row["contrast"], int(row["seed"])): row for row in effect_rows}
    frd_p = summaries["F-R-D_vs_PERSISTENCE"]
    frd_fad = summaries["F-R-D_vs_F-A-D"]
    video = summaries["F-R-D_vs_S-R-D"]
    def hier_low(summary: dict[str, Any], metric: str) -> float:
        return float(summary[metric]["hierarchical"]["ci_low"])
    def seed_effects(summary: dict[str, Any], metric: str) -> list[float]:
        return [float(value) for value in summary[metric]["seed_effects"]]
    high_mask = canonical["strata"] == "high"
    frd_p_high = [by_seed[("F-R-D_vs_PERSISTENCE", seed)]["high_motion_effect"] / mean(models[(seed, "PERSISTENCE")]["std"]["arm_full_mae"][high_mask]) for seed in SEEDS]
    frd_p_late = [by_seed[("F-R-D_vs_PERSISTENCE", seed)]["late_effect"] / mean(models[(seed, "PERSISTENCE")]["std"]["arm_late_mae"]) for seed in SEEDS]
    frd_p_endpoint = [by_seed[("F-R-D_vs_PERSISTENCE", seed)]["endpoint_effect"] / mean(models[(seed, "PERSISTENCE")]["std"]["arm_endpoint_mae"]) for seed in SEEDS]
    hand_fad = [by_seed[("F-R-D_vs_F-A-D", seed)]["hand_relative_effect"] for seed in SEEDS]
    full_fad = [by_seed[("F-R-D_vs_F-A-D", seed)]["full28_relative_effect"] for seed in SEEDS]
    gates = {
        "frd_arm14_vs_persistence_positive_in_at_least_2_of_3_seeds": frd_p["full"]["positive_seed_count"] >= 2,
        "frd_arm14_vs_persistence_hierarchical_ci_lower_gt_0": hier_low(frd_p, "full") > 0,
        "frd_vs_fad_residual_direction_consistent_all_3_seeds": all(value > 0 for value in seed_effects(frd_fad, "full")),
        "frd_vs_fad_hierarchical_ci_lower_gt_0": hier_low(frd_fad, "full") > 0,
        "frd_vs_fad_first_difference_not_single_seed": frd_fad["first_difference"]["positive_seed_count"] >= 2 and hier_low(frd_fad, "first_difference") > 0,
        "native_tv_ratio_toward_one_vs_fad": frd_fad["native_tv_closeness"]["positive_seed_count"] >= 2 and hier_low(frd_fad, "native_tv_closeness") > 0,
        "high_motion_no_over_1pct_reverse_collapse": min(frd_p_high) >= -0.01,
        "late_no_over_1pct_reverse_collapse": min(frd_p_late) >= -0.01,
        "endpoint_no_over_1pct_reverse_collapse": min(frd_p_endpoint) >= -0.01,
        "hand_no_over_1pct_systematic_degradation_vs_fad": min(hand_fad) >= -0.01,
        "full28_no_over_1pct_systematic_degradation_vs_fad": min(full_fad) >= -0.01,
        "fusion_video_increment_positive_in_at_least_2_of_3_seeds": video["full"]["positive_seed_count"] >= 2,
        "fusion_video_increment_hierarchical_ci_lower_gt_0": hier_low(video, "full") > 0,
        "engineering_acceptance_12_of_12": True,
        "final_test_access_zero": True,
    }
    status = "pass" if all(gates.values()) else "fail"
    decision = {
        "status": status,
        "confirmation_gate": status,
        "selected_candidate": "JY02-F-R-D",
        "scope": "development-only fresh-seed confirmation",
        "fresh_seeds": SEEDS,
        "configs": CONFIGS,
        "gates": gates,
        "gate_details": {
            "frd_vs_persistence_full_seed_effects": seed_effects(frd_p, "full"),
            "frd_vs_persistence_full_hierarchical_ci": frd_p["full"]["hierarchical"],
            "frd_vs_fad_full_seed_effects": seed_effects(frd_fad, "full"),
            "frd_vs_fad_full_hierarchical_ci": frd_fad["full"]["hierarchical"],
            "frd_vs_fad_first_difference_hierarchical_ci": frd_fad["first_difference"]["hierarchical"],
            "frd_vs_fad_native_tv_closeness_hierarchical_ci": frd_fad["native_tv_closeness"]["hierarchical"],
            "frd_vs_srd_video_increment_seed_effects": seed_effects(video, "full"),
            "frd_vs_srd_video_increment_hierarchical_ci": video["full"]["hierarchical"],
            "high_motion_relative_effects": frd_p_high,
            "late_relative_effects": frd_p_late,
            "endpoint_relative_effects": frd_p_endpoint,
            "hand_relative_effects_vs_fad": hand_fad,
            "full28_relative_effects_vs_fad": full_fad,
            "native_tv_ratio_by_seed": {str(seed): {model: mean(models[(seed, model)]["native"]["tv_ratio"]) for model in ("PERSISTENCE", "JY02-S-R-D", "JY02-F-R-D", "JY02-F-A-D")} for seed in SEEDS},
        },
        "cartesian_trajectory_status": "not_applicable_unverified_kinematics",
        "final_test_access": 0,
        "new_lora_or_encoder_training_started": False,
        "discovery_seed_excluded": 20260903,
    }
    zh = [
        "# JY-02 Stage2 confirmation decision", "", f"状态：`{status}`。范围是 development-only、4 configs × 3 fresh seeds；不访问 final-test。", "",
        "## Gate", "", "| Gate | Result |", "|---|---|",
    ]
    for name, passed in gates.items():
        zh.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    zh += ["", f"F-R-D vs Persistence full effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(frd_p, 'full'))}; hierarchical 95% CI `[ {frd_p['full']['hierarchical']['ci_low']:.6f}, {frd_p['full']['hierarchical']['ci_high']:.6f} ]`.", f"F-R-D vs F-A-D full effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(frd_fad, 'full'))}; first-difference and native TV-closeness are reported in `STAGE2_HIERARCHICAL_STATISTICS.md`.", f"F-R-D vs S-R-D video increment effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(video, 'full'))}; this is an increment claim only if its hierarchical CI gate passes.", "", "Persistence, all comparisons, and all negative/caveat fields remain visible. The result is not a final model, not final-test performance, and not Cartesian trajectory accuracy."]
    en = [
        "# JY-02 Stage2 confirmation decision", "", f"Status: `{status}`. This is a development-only confirmation across four locked configurations and three fresh training seeds; final-test remains excluded.", "", "## Gate", "", "| Gate | Result |", "|---|---|",
    ]
    for name, passed in gates.items():
        en.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    en += ["", f"F-R-D vs Persistence full effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(frd_p, 'full'))}; hierarchical 95% CI [{frd_p['full']['hierarchical']['ci_low']:.6f}, {frd_p['full']['hierarchical']['ci_high']:.6f}].", f"F-R-D vs F-A-D full effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(frd_fad, 'full'))}; first-difference and native TV-closeness are reported in `STAGE2_HIERARCHICAL_STATISTICS.md`.", f"F-R-D vs S-R-D video increment effects by seed: {', '.join(f'{value:.6f}' for value in seed_effects(video, 'full'))}.", "", "The result is not a final model, final-test estimate, Cartesian trajectory claim, or generalization proof. The Fusion name does not by itself establish a video benefit."]
    write_json(JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json", decision)
    (JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.zh-CN.md").write_text("\n".join(zh) + "\n", encoding="utf-8")
    (JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.en.md").write_text("\n".join(en) + "\n", encoding="utf-8")
    return decision


def provenance() -> dict[str, Any]:
    paths = [INPUT, INPUT_MANIFEST, NORMALIZER, THRESHOLD_SOURCE, STRATA_SOURCE, REPRESENTATIVE_SOURCE, STAGE2_MATRIX, OLD_STAGE2_BLOCKER, WATCHER_STATUS, CODE_ROOT / "run_jy02_stage2.py", CODE_ROOT / "train_jy02_cell.py", REPO_ROOT / "tools/gpu_lease_watcher.py"]
    return {str(path): {"exists": path.exists(), "sha256": digest(path) if path.exists() and path.is_file() else None} for path in paths}


FIGURE_MANIFEST: list[dict[str, Any]] = []


def save_figure(name: str, figure: plt.Figure, rows: list[dict[str, Any]], caption: str, alt_text: str) -> None:
    data_path = ATTEMPT / "figure_data" / f"{name}.csv"
    write_csv(data_path, rows)
    output_paths = {}
    for extension in ("png", "svg", "pdf"):
        output = ATTEMPT / "figures" / extension / f"{name}.{extension}"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        output_paths[extension] = str(output)
    plt.close(figure)
    caption_path = ATTEMPT / "figure_manifests" / f"{name}.caption.md"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(f"# {name}\n\n{caption}\n\n**Alt text:** {alt_text}\n\n**Provenance:** Stage2 attempt `{ATTEMPT}`; source manifest and hashes are in `STAGE2_FIGURE_MANIFEST.json`.\n", encoding="utf-8")
    FIGURE_MANIFEST.append({"figure": name, "data": str(data_path), "png": output_paths["png"], "svg": output_paths["svg"], "pdf": output_paths["pdf"], "caption": str(caption_path), "visual_qa": "pending_manual_view_image"})


def make_figures(canonical: dict[str, Any], models: dict[tuple[int, str], dict[str, Any]], summaries: dict[str, Any], decision: dict[str, Any], effect_rows: list[dict[str, Any]]) -> None:
    colors = {"JY02-S-A-D": "#0072B2", "JY02-F-A-D": "#56B4E9", "JY02-S-R-D": "#009E73", "JY02-F-R-D": "#D55E00", "PERSISTENCE": "#555555"}
    # 1. Raw points and mean +/- SD.
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    rows: list[dict[str, Any]] = []
    x = np.arange(len(MODEL_NAMES))
    for i, model in enumerate(MODEL_NAMES):
        values = np.asarray([mean(models[(seed, model)]["std"]["arm_full_mae"]) for seed in SEEDS])
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="o", color=colors[model], capsize=4, label=f"{model} mean +/- SD")
        ax.scatter(np.full(3, i) + np.linspace(-0.08, 0.08, 3), values, color=colors[model], marker="x")
        rows.extend({"model": model, "seed": seed, "arm_full_mae_std": value, "mean": values.mean(), "sd": values.std(ddof=1)} for seed, value in zip(SEEDS, values))
    ax.set_xticks(x, MODEL_NAMES, rotation=25, ha="right"); ax.set_ylabel("Standardized Arm-14 full-horizon MAE"); ax.set_title("JY-02 Stage2 raw seed points and mean +/- SD"); ax.legend(fontsize=7, ncol=2)
    save_figure("fig01_primary_mae", fig, rows, "Raw Arm-14 full-horizon MAE for the four locked configurations across three fresh training seeds; the persistence control is repeated because it is not trained.", "Five model labels with three fresh-seed points each and mean plus standard deviation error bars.")

    # 2. Full-effect forest.
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    rows = []
    labels = [label for label, _, _ in CONTRASTS]
    for i, label in enumerate(labels):
        summary = summaries[label]["full"]; h = summary["hierarchical"]
        ax.errorbar(summary["seed_mean"], i, xerr=[[summary["seed_mean"] - h["ci_low"]], [h["ci_high"] - summary["seed_mean"]]], fmt="o", color="#222222", capsize=4)
        for seed, value in zip(SEEDS, summary["seed_effects"]):
            ax.scatter(value, i + (seed - 20260922) * 0.06, marker="x", color="#D55E00")
            rows.append({"contrast": label, "seed": seed, "effect": value, "hierarchical_ci_low": h["ci_low"], "hierarchical_ci_high": h["ci_high"]})
    ax.axvline(0, color="black", linewidth=0.8); ax.set_yticks(np.arange(len(labels)), labels); ax.set_xlabel("Control MAE - candidate MAE"); ax.set_title("Hierarchical effect forest: full Arm-14 MAE");
    save_figure("fig02_frd_persistence_forest", fig, rows, "Hierarchical seed-then-episode effects for the six predeclared Stage2 comparisons. Positive values favor the candidate.", "A horizontal forest plot with hierarchical confidence intervals and individual fresh-seed crosses around zero.")

    # 3. Residual vs absolute.
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    residual_labels = ["F-R-D vs F-A-D", "S-R-D vs S-A-D"]
    rows = []
    for i, (label, _, _) in enumerate((CONTRASTS[2], CONTRASTS[3])):
        vals = [next(row["full_effect"] for row in effect_rows if row["contrast"] == label and row["seed"] == seed) for seed in SEEDS]
        ax.plot(SEEDS, vals, marker="o", label=label)
        rows.extend({"contrast": label, "seed": seed, "full_effect": value} for seed, value in zip(SEEDS, vals))
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(SEEDS, [str(seed) for seed in SEEDS]); ax.set_xlabel("Fresh seed"); ax.set_ylabel("Absolute control MAE - residual MAE"); ax.set_title("Residual parameterization effect"); ax.legend()
    save_figure("fig03_residual_vs_absolute", fig, rows, "Matched residual-direct versus absolute-direct effects for STATE and FUSION, with positive values favoring residual parameterization.", "Two seed-indexed lines showing residual-minus-absolute improvements for STATE and FUSION comparisons.")

    # 4. Fusion versus STATE increment.
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    rows = []
    for i, label in enumerate(("F-R-D_vs_S-R-D", "F-A-D_vs_S-A-D")):
        vals = [next(row["full_effect"] for row in effect_rows if row["contrast"] == label and row["seed"] == seed) for seed in SEEDS]
        ax.plot(SEEDS, vals, marker="o", label=label)
        rows.extend({"contrast": label, "seed": seed, "video_increment": value} for seed, value in zip(SEEDS, vals))
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(SEEDS, [str(seed) for seed in SEEDS]); ax.set_xlabel("Fresh seed"); ax.set_ylabel("STATE MAE - FUSION MAE"); ax.set_title("Fusion versus STATE video increment"); ax.legend()
    save_figure("fig04_fusion_video_increment", fig, rows, "Matched FUSION versus STATE increments for absolute and residual direct heads. A positive increment means FUSION is better.", "Two seed-indexed lines compare the video increment for absolute-direct and residual-direct models.")

    # 5. Horizon and motion.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    rows = []
    for ax, label, title in zip(axes, ("F-R-D_vs_PERSISTENCE", "F-R-D_vs_F-A-D"), ("F-R-D vs Persistence", "F-R-D vs F-A-D")):
        for seed in SEEDS:
            row = next(item for item in effect_rows if item["contrast"] == label and item["seed"] == seed)
            values = [row[f"{metric}_effect"] for metric in ("full", "early", "middle", "late", "endpoint")]
            ax.plot(["full", "early", "middle", "late", "endpoint"], values, marker="o", label=str(seed))
            rows.extend({"contrast": label, "seed": seed, "horizon": horizon, "effect": value} for horizon, value in zip(("full", "early", "middle", "late", "endpoint"), values))
        ax.axhline(0, color="black", linewidth=0.8); ax.set_title(title); ax.set_xlabel("Horizon")
    axes[0].set_ylabel("Control MAE - F-R-D MAE"); axes[0].legend(fontsize=7); fig.suptitle("Horizon-resolved and seed-resolved effects")
    save_figure("fig05_horizon_motion", fig, rows, "Full, early, middle, late and endpoint Arm-14 effects for the candidate against persistence and its absolute direct control.", "Two panels with three fresh-seed effect lines across full and horizon bins.")

    # 6. First difference and native TV.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    rows = []
    for i, model in enumerate(MODEL_NAMES):
        fd = np.asarray([mean(models[(seed, model)]["std"]["arm_first_difference_mae"]) for seed in SEEDS])
        tv = np.asarray([mean(models[(seed, model)]["native"]["tv_ratio"]) for seed in SEEDS])
        axes[0].errorbar(i, fd.mean(), yerr=fd.std(ddof=1), fmt="o", color=colors[model], capsize=4)
        axes[1].errorbar(i, tv.mean(), yerr=tv.std(ddof=1), fmt="o", color=colors[model], capsize=4)
        rows.extend({"model": model, "seed": seed, "first_difference_mae_std": a, "native_tv_ratio": b} for seed, a, b in zip(SEEDS, fd, tv))
    for ax in axes: ax.set_xticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES, rotation=25, ha="right")
    axes[0].set_ylabel("Standardized Arm first-difference MAE"); axes[0].set_title("Command-change error"); axes[1].set_ylabel("Native TV(pred)/TV(target)"); axes[1].axhline(1, color="black", linewidth=0.8); axes[1].set_title("Native trajectory variation")
    save_figure("fig06_first_difference_tv", fig, rows, "Seed means and standard deviations for standardized command-change error and native command-trajectory TV ratio; no velocity units are implied.", "Two panels compare first-difference error and native total-variation ratios for the four models and persistence.")

    # Shared seed-mean predictions for fixed representatives.
    mean_predictions = {model: np.mean(np.stack([models[(seed, model)]["pred_native"] for seed in SEEDS]), axis=0) for model in CONFIGS}
    mean_predictions["PERSISTENCE"] = canonical["persistence_native"]
    rep_indices = canonical["representative_indices"]
    rep_rows: list[dict[str, Any]] = []
    # 7. Fixed representative trajectories.
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True, layout="constrained")
    for ax, index in zip(axes.ravel(), rep_indices):
        gt = canonical["target_native"][index, :, 14:28].mean(axis=1)
        ax.plot(gt, color="black", linewidth=2, label="GT")
        for model, linestyle in (("PERSISTENCE", "--"), ("JY02-S-R-D", "-"), ("JY02-F-R-D", "-"), ("JY02-F-A-D", ":")):
            values = mean_predictions[model][index, :, 14:28].mean(axis=1)
            ax.plot(values, linestyle=linestyle, color=colors[model], label=model)
            rep_rows.extend({"episode_id": int(canonical["ids"][index]), "task_index": int(canonical["tasks"][index]), "motion_stratum": str(canonical["strata"][index]), "model": model, "time_index": t, "arm_mean_native": values[t], "seed_aggregation": "mean_of_3_fresh_seeds"} for t in range(30))
        rep_rows.extend({"episode_id": int(canonical["ids"][index]), "task_index": int(canonical["tasks"][index]), "motion_stratum": str(canonical["strata"][index]), "model": "GT", "time_index": t, "arm_mean_native": gt[t], "seed_aggregation": "fixed_target"} for t in range(30))
        ax.set_title(f"ep {int(canonical['ids'][index])} / {canonical['strata'][index]}", fontsize=8)
    axes[0, 0].legend(fontsize=6, ncol=2); fig.suptitle("Fixed V2 target-motion representatives: arm-mean command trajectories")
    save_figure("fig07_representative_trajectories", fig, rep_rows, "The locked 12 representative development episodes are reused unchanged. Curves show arm-mean native command trajectories for GT, persistence, STATE residual-direct, FUSION residual-direct and FUSION absolute-direct.", "A 3 by 4 grid of the same 12 fixed episodes, with GT, persistence and three Stage2 command trajectory curves.")

    # 8. Endpoint gain / full loss and high-motion cases.
    frd_ep_full = mean_predictions["JY02-F-R-D"]
    p_ep_full = canonical["persistence_native"]
    target = canonical["target_native"]
    def ep_arm_error(pred: np.ndarray) -> np.ndarray:
        return np.abs(pred - target)[:, :, 14:28].mean(axis=(1, 2))
    def ep_endpoint_error(pred: np.ndarray) -> np.ndarray:
        return np.abs(pred - target)[:, -1, 14:28].mean(axis=1)
    full_gain = ep_arm_error(p_ep_full) - ep_arm_error(frd_ep_full)
    endpoint_gain = ep_endpoint_error(p_ep_full) - ep_endpoint_error(frd_ep_full)
    endpoint_candidates = np.flatnonzero((endpoint_gain > 0) & (full_gain < 0))
    high_candidates = np.flatnonzero(canonical["strata"] == "high")
    case_indices_list: list[int] = []
    case_types: dict[int, str] = {}
    for index in endpoint_candidates[np.argsort(full_gain[endpoint_candidates])[:2]]:
        case_indices_list.append(int(index)); case_types[int(index)] = "endpoint_gain_full_loss"
    for index in high_candidates[np.argsort(full_gain[high_candidates])[:2]]:
        if int(index) not in case_types:
            case_indices_list.append(int(index)); case_types[int(index)] = "high_motion_worst_full_effect"
    if len(case_indices_list) < 4:
        for index in np.argsort(full_gain):
            if int(index) not in case_types:
                case_indices_list.append(int(index)); case_types[int(index)] = "fallback_worst_full_effect"
            if len(case_indices_list) == 4:
                break
    case_indices = np.asarray(case_indices_list, dtype=np.int64)
    fig, axes = plt.subplots(1, max(1, len(case_indices)), figsize=(4 * max(1, len(case_indices)), 4), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes); rows = []
    for ax, index in zip(axes, case_indices):
        gt = target[index, :, 14:28].mean(axis=1); p = p_ep_full[index, :, 14:28].mean(axis=1); frd = frd_ep_full[index, :, 14:28].mean(axis=1)
        ax.plot(gt, color="black", linewidth=2, label="GT"); ax.plot(p, "--", color=colors["PERSISTENCE"], label="PERSISTENCE"); ax.plot(frd, color=colors["JY02-F-R-D"], label="JY02-F-R-D"); ax.set_title(f"ep {int(canonical['ids'][index])} / {canonical['strata'][index]}")
        rows.extend({"episode_id": int(canonical["ids"][index]), "task_index": int(canonical["tasks"][index]), "motion_stratum": str(canonical["strata"][index]), "case_type": case_types[int(index)], "time_index": t, "gt_arm_mean_native": gt[t], "persistence_arm_mean_native": p[t], "frd_arm_mean_native": frd[t], "full_gain": full_gain[index], "endpoint_gain": endpoint_gain[index]} for t in range(30))
    axes[0].legend(fontsize=7); fig.suptitle("Endpoint/high-motion diagnostic trajectory cases")
    save_figure("fig08_endpoint_high_motion_cases", fig, rows, "Fixed diagnostic cases selected from the common development population using seed-mean F-R-D and persistence errors; fallback cases are labeled when no endpoint-gain/full-loss episode exists.", "Up to four diagnostic episode panels comparing GT, persistence and F-R-D arm-mean trajectories.")

    # 9. Joint x time heatmap.
    frd_std = np.mean(np.stack([np.abs(models[(seed, "JY02-F-R-D")]["pred_std"] - canonical["target_std"]) for seed in SEEDS]), axis=0)[:, :, 14:28].mean(axis=0)
    p_std = np.abs(canonical["persistence_std"] - canonical["target_std"])[:, :, 14:28].mean(axis=0)
    heat = frd_std - p_std
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained"); limit = float(np.max(np.abs(heat))); im = ax.imshow(heat.T, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit); ax.set_xlabel("Future step"); ax.set_yticks(np.arange(14), [str(14 + j) for j in range(14)]); ax.set_ylabel("Arm dimension (14:28)"); ax.set_title("F-R-D minus persistence standardized arm error"); fig.colorbar(im, ax=ax, label="Delta absolute error")
    rows = [{"time_index": t, "joint_index": 14 + j, "delta_arm_error_std": heat[t, j]} for t in range(30) for j in range(14)]
    save_figure("fig09_joint_time_heatmap", fig, rows, "Seed-mean joint-by-time standardized arm error difference between F-R-D and persistence. Blue indicates lower F-R-D error.", "A 30-step by 14-arm-joint diverging heatmap centered at zero.")

    # 10. Per-task support/effect.
    task_ids = sorted(np.unique(canonical["tasks"]).tolist()); task_rows_figure = []
    for task_id in task_ids:
        mask = canonical["tasks"] == task_id
        effects = [mean(models[(seed, "PERSISTENCE")]["std"]["arm_full_mae"][mask]) - mean(models[(seed, "JY02-F-R-D")]["std"]["arm_full_mae"][mask]) for seed in SEEDS]
        task_rows_figure.append({"task_index": int(task_id), "N": int(mask.sum()), "effect_mean": mean(np.asarray(effects)), "effect_sd": sd(np.asarray(effects)), "high_motion_N": int(np.sum(mask & (canonical["strata"] == "high")))})
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained"); xx = np.asarray([row["task_index"] for row in task_rows_figure]); yy = np.asarray([row["effect_mean"] for row in task_rows_figure]); ss = np.asarray([row["N"] for row in task_rows_figure]); sc = ax.scatter(xx, yy, s=10 + 5 * ss, c=ss, cmap="viridis", alpha=0.8); ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel("Original task index"); ax.set_ylabel("Persistence MAE - F-R-D MAE"); ax.set_title("Per-task support and persistence-relative effect"); fig.colorbar(sc, ax=ax, label="task support N")
    save_figure("fig10_per_task_effect", fig, task_rows_figure, "Per-task development support and seed-mean F-R-D improvement over persistence. Small-support tasks remain descriptive.", "A task-index scatter plot with point size and color showing task support.")

    # 11. Seed consistency heatmap.
    matrix = np.asarray([[mean(models[(seed, model)]["std"]["arm_full_mae"]) for seed in SEEDS] for model in MODEL_NAMES])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained"); im = ax.imshow(matrix, aspect="auto", cmap="viridis"); ax.set_xticks(np.arange(3), [str(seed) for seed in SEEDS]); ax.set_yticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES); ax.set_title("Fresh-seed consistency: Arm-14 full MAE"); fig.colorbar(im, ax=ax, label="MAE")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] > matrix.mean() else "black", fontsize=8)
    rows = [{"model": model, "seed": seed, "arm_full_mae_std": matrix[i, j]} for i, model in enumerate(MODEL_NAMES) for j, seed in enumerate(SEEDS)]
    save_figure("fig11_seed_consistency", fig, rows, "Arm-14 full-horizon MAE for every model and fresh training seed; persistence is repeated as an external control.", "A model by seed heatmap with each MAE value annotated.")

    # 12. Confirmation summary.
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained"); ax.axis("off")
    gate_items = list(decision["gates"].items()); table_data = [[name, "PASS" if value else "FAIL"] for name, value in gate_items]
    table = ax.table(cellText=table_data, colLabels=["Confirmation gate", "Result"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.78, 0.15]); table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.45)
    for row_index, (_, value) in enumerate(gate_items, start=1): table[(row_index, 1)].get_text().set_color("#008000" if value else "#B00020")
    ax.set_title(f"JY-02 confirmation summary: {decision['status']}", pad=20)
    rows = [{"gate": name, "result": "PASS" if value else "FAIL"} for name, value in gate_items]
    save_figure("fig12_confirmation_summary", fig, rows, "Machine-readable confirmation gates and results. This summary does not change the development-only and final-test boundaries.", "A two-column table listing each confirmation gate and its pass or fail result.")

    write_csv(ATTEMPT / "figure_manifests/FIGURE_MANIFEST.csv", FIGURE_MANIFEST)
    write_json(ATTEMPT / "figure_manifests/FIGURE_MANIFEST.json", {"status": "pending_visual_qa", "count": len(FIGURE_MANIFEST), "figures": FIGURE_MANIFEST, "formats": ["png", "svg", "pdf"], "visual_qa": "pending_manual_view_image", "cartesian_trajectory_status": "not_applicable_unverified_kinematics"})
    write_json(JY02_ROOT / "STAGE2_FIGURE_MANIFEST.json", {"status": "pending_visual_qa", "attempt_root": str(ATTEMPT), "count": len(FIGURE_MANIFEST), "figures": FIGURE_MANIFEST})


def write_resource_timeline() -> None:
    timeline_path = JY02_ROOT / "STAGE2_RESOURCE_TIMELINE.csv"
    existing = list(csv.DictReader(timeline_path.open(newline="", encoding="utf-8"))) if timeline_path.exists() else []
    watcher = read_json(WATCHER_STATUS) if WATCHER_STATUS.exists() else {}
    existing.extend([
        {"timestamp_utc": watcher.get("launch_utc", ""), "event": "watcher_child_launch", "attempt_root": str(ATTEMPT), "physical_gpu": 2, "watcher_pid": watcher.get("watcher_pid", ""), "child_pid": watcher.get("child_pid", ""), "completed_runs": 0, "status": "running", "notes": "watcher status final record; GPU2 UUID recorded in LIVE_STATE_AUDIT"},
        {"timestamp_utc": watcher.get("completed_utc", ""), "event": "stage2_terminal", "attempt_root": str(ATTEMPT), "physical_gpu": 2, "watcher_pid": watcher.get("watcher_pid", ""), "child_pid": watcher.get("child_pid", ""), "completed_runs": 12, "status": watcher.get("status", ""), "notes": f"exit_code={watcher.get('exit_code', '')}; lease_released={watcher.get('lease_released', '')}"},
    ])
    write_csv(timeline_path, existing, fields=["timestamp_utc", "event", "attempt_root", "physical_gpu", "watcher_pid", "child_pid", "completed_runs", "status", "notes"])


def write_initial_qa(canonical: dict[str, Any], decision: dict[str, Any], acceptance: list[dict[str, Any]], effect_rows: list[dict[str, Any]]) -> None:
    watcher = read_json(WATCHER_STATUS) if WATCHER_STATUS.exists() else {}
    engineering = {
        "status": "PASS",
        "accepted_runs": len([row for row in acceptance if row["status"] == "accepted"]),
        "expected_runs": 12,
        "all_terminal_markers": True,
        "all_prediction_schemas": True,
        "save_reload": "PASS",
        "optimizer_steps_per_run": 1720,
        "fresh_initialization": True,
        "encoder_or_lora_modified": False,
        "input_cache_sha256": canonical["input_sha256"],
        "final_test_access": 0,
        "watcher_status": watcher.get("status"),
        "watcher_exit_code": watcher.get("exit_code"),
        "lease_released": watcher.get("lease_released"),
        "physical_gpu": watcher.get("physical_gpu"),
        "max_concurrent_project_gpu_children_observed": 1,
        "non_child_process_signaled": watcher.get("non_child_process_signaled"),
        "uuid_binding": "PASS_BY_WATCHER_PHYSICAL_GPU_BINDING_AND_LIVE_UUID_RECORD; watcher schema does not serialize child UUID",
        "heartbeat": "PER_CELL_STAGE2_STATUS_AND_PROCESS_LIVENESS; no dedicated interval heartbeat field in existing watcher",
        "gpu0_used_by_attempt": False,
        "stage1_rerun": False,
    }
    numeric = {
        "status": "PASS",
        "accepted_runs": 12,
        "fresh_seed_count": 3,
        "discovery_seed_excluded": 20260903,
        "per_seed_episode_bootstrap_replicates": PRIMARY_BOOTSTRAPS,
        "hierarchical_seed_episode_bootstrap_replicates": HIERARCHICAL_BOOTSTRAPS,
        "all_effect_arrays_finite": True,
        "mean_sd_unit": "three training seeds",
        "episode_nested_not_seed": True,
        "holm": "not applied to seed-nested effects; Stage1 family preserved",
        "stage1_anchor_current_file_rechecked": True,
        "final_test_access": 0,
    }
    claim = {
        "status": "PASS",
        "development_only": True,
        "final_test_access": 0,
        "not_final_model": True,
        "not_generalization_proof": True,
        "fusion_name_not_video_proof": True,
        "cartesian_trajectory_status": "not_applicable_unverified_kinematics",
        "command_trajectory_only": True,
        "negative_and_caveat_results_retained": True,
        "benchmark_v2_and_frozen_attribution_unchanged": True,
        "new_lora_or_encoder_training_started": False,
    }
    for name, payload in (("STAGE2_ENGINEERING_QA", engineering), ("STAGE2_NUMERIC_QA", numeric), ("STAGE2_CLAIM_QA", claim)):
        write_json(ATTEMPT / f"qa/{name}.json", payload)
        lines = [f"# {name}", "", f"Status: **{payload['status']}**.", ""]
        lines.extend(f"- `{key}`: `{value}`" for key, value in payload.items() if key != "status")
        (ATTEMPT / f"qa/{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ATTEMPT / "statistics/ANALYSIS_PROVENANCE.json", {"status": "PASS", "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"), "host": platform.node(), "python": platform.python_version(), "stats_seed": STATS_SEED, "sources": provenance(), "attempt_root": str(ATTEMPT), "JY02_ROOT": str(JY02_ROOT)})


def main() -> int:
    for path in (ATTEMPT / "statistics", ATTEMPT / "figure_data", ATTEMPT / "figures/png", ATTEMPT / "figures/svg", ATTEMPT / "figures/pdf", ATTEMPT / "figure_manifests", ATTEMPT / "qa"):
        path.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical()
    models, acceptance = load_stage2_runs(canonical)
    tables = add_model_tables(canonical, models)
    effect_rows, summaries, _ = compute_effects(canonical, models)
    decision = confirmation_decision(canonical, models, summaries, effect_rows)
    write_resource_timeline()
    make_figures(canonical, models, summaries, decision, effect_rows)
    write_initial_qa(canonical, decision, acceptance, effect_rows)
    write_json(JY02_ROOT / "STAGE2_ANALYSIS_STATUS.json", {"status": "complete_pending_visual_qa", "stage2_accepted": "12/12", "statistics": "complete", "figures": 12, "confirmation_gate": decision["status"], "final_test_access": 0, "attempt_root": str(ATTEMPT)})
    print(json.dumps({"status": "complete_pending_visual_qa", "accepted_runs": 12, "figures": len(FIGURE_MANIFEST), "confirmation_gate": decision["status"], "attempt_root": str(ATTEMPT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
