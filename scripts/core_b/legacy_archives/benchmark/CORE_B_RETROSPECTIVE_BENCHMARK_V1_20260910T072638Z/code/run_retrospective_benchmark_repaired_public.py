#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-018; source SHA-256: 781270b7dce85298aa7946b791444ef1617c23cc3695581455e5133f2ebf00d7
"""Independent CPU-only retrospective Core B benchmark.

The script reads accepted B1/B2/B3 prediction bundles and the locked train-only
normalizer.  It never imports torch, starts a model, accesses final-test data,
or writes to the source experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".").resolve()
ARCHIVE_CORE_B = Path(os.environ.get("EE6008_ARCHIVE_CORE_B", ".")).expanduser().resolve()
SOURCE = ARCHIVE_CORE_B / "analysis" / "CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"
INPUT_PATH = Path(os.environ.get("EE6008_RETROSPECTIVE_INPUT_PATH", ".")).expanduser().resolve()
NORMALIZER_PATH = Path(os.environ.get("EE6008_RETROSPECTIVE_NORMALIZER_PATH", ".")).expanduser().resolve()
BENCHMARK_ID = ROOT.name
SEED = 20260910
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 10_000
LAGS = tuple(range(-5, 6))
FIGURES: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n")


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar(row.get(key, "")) for key in fieldnames})


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def model_meta(row: dict[str, Any]) -> dict[str, Any]:
    cell = row["cell"]
    stage = row["stage"]
    representation = "unknown"
    if stage == "B1":
        representation = "attentive" if "-A-" in cell else "pooled" if "-P-" in cell else "state"
    interval = ""
    frames = ""
    if any(token in cell for token in ("0125",)):
        interval, frames = 0.125, 32
    elif "0250" in cell:
        interval, frames = 0.25, 16
    elif "0500" in cell:
        interval, frames = 0.5, 8
    if stage == "B3":
        interval, frames = 0.125, 32
        representation = "pooled"
    return {
        "stage": stage,
        "model": cell,
        "branch": row.get("branch", ""),
        "video_input": row.get("video_input", ""),
        "adaptation": row.get("adaptation", ""),
        "representation": representation,
        "interval_seconds": interval,
        "unique_observation_frames": frames,
        "encoder_slots": 32 if interval != "" else "",
        "source_kind": row.get("source_kind", ""),
        "acceptance_status": row.get("status", ""),
        "reused_from": row.get("reused_from", ""),
        "prediction_path": row.get("prediction_path", ""),
        "final_test_access": row.get("final_test_access", 0),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def align_indices(source_ids: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(source_ids.tolist())}
    if len(lookup) != len(source_ids):
        raise ValueError("duplicate episode IDs in prediction bundle")
    missing = [int(value) for value in requested_ids if int(value) not in lookup]
    if missing:
        raise ValueError(f"prediction bundle missing requested IDs: {missing[:5]}")
    return np.asarray([lookup[int(value)] for value in requested_ids], dtype=np.int64)


def load_prediction(row: dict[str, Any], common_ids: np.ndarray, canonical_target: np.ndarray) -> dict[str, Any]:
    path = Path(row["prediction_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = np.load(path, allow_pickle=False)
    required = {"episode_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native"}
    missing = required.difference(bundle.files)
    if missing:
        raise ValueError(f"{row['cell']} missing keys: {sorted(missing)}")
    ids = np.asarray(bundle["episode_index"], dtype=np.int64)
    indices = align_indices(ids, common_ids)
    pred_std = np.asarray(bundle["predictions_standardized"], dtype=np.float64)[indices]
    pred_native = np.asarray(bundle["predictions_native"], dtype=np.float64)[indices]
    target_std = np.asarray(bundle["targets_standardized"], dtype=np.float64)[indices]
    target_native = np.asarray(bundle["targets_native"], dtype=np.float64)[indices]
    if pred_std.shape != canonical_target.shape:
        raise ValueError(f"{row['cell']} shape {pred_std.shape} != {canonical_target.shape}")
    return {
        "pred_std": pred_std,
        "pred_native": pred_native,
        "target_std_source": target_std,
        "target_native_source": target_native,
        "path": path,
        "source_sha256": sha256(path),
    }


def metric_arrays(pred_std: np.ndarray, pred_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> dict[str, Any]:
    err_std = pred_std - target_std
    err_native = pred_native - target_native
    abs_std = np.abs(err_std)
    abs_native = np.abs(err_native)
    diff_std = np.diff(pred_std, axis=1) - np.diff(target_std, axis=1)
    diff_native = np.diff(pred_native, axis=1) - np.diff(target_native, axis=1)
    abs_diff_std = np.abs(diff_std)
    abs_diff_native = np.abs(diff_native)

    # Canonical project ordering is hand14=0:14 and arm14=14:28.
    def per_episode(abs_values: np.ndarray, dimensions: slice) -> np.ndarray:
        return abs_values[:, :, dimensions].mean(axis=(1, 2))

    def final_step(abs_values: np.ndarray, dimensions: slice) -> np.ndarray:
        return abs_values[:, -1, dimensions].mean(axis=1)

    return {
        "arm_mae_std": per_episode(abs_std, slice(14, 28)),
        "hand_mae_std": per_episode(abs_std, slice(0, 14)),
        "full28_mae_std": per_episode(abs_std, slice(0, 28)),
        "arm_final_std": final_step(abs_std, slice(14, 28)),
        "hand_final_std": final_step(abs_std, slice(0, 14)),
        "full28_final_std": final_step(abs_std, slice(0, 28)),
        "arm_first_diff_std": per_episode(abs_diff_std, slice(14, 28)),
        "hand_first_diff_std": per_episode(abs_diff_std, slice(0, 14)),
        "full28_first_diff_std": per_episode(abs_diff_std, slice(0, 28)),
        "full_rmse_std": np.sqrt(np.mean(err_std**2, axis=(1, 2))),
        "arm_mae_native": per_episode(abs_native, slice(14, 28)),
        "hand_mae_native": per_episode(abs_native, slice(0, 14)),
        "full28_mae_native": per_episode(abs_native, slice(0, 28)),
        "arm_final_native": final_step(abs_native, slice(14, 28)),
        "hand_final_native": final_step(abs_native, slice(0, 14)),
        "full28_final_native": final_step(abs_native, slice(0, 28)),
        "arm_first_diff_native": per_episode(abs_diff_native, slice(14, 28)),
        "hand_first_diff_native": per_episode(abs_diff_native, slice(0, 14)),
        "full28_first_diff_native": per_episode(abs_diff_native, slice(0, 28)),
        "full_rmse_native": np.sqrt(np.mean(err_native**2, axis=(1, 2))),
        "per_time_arm_std": abs_std[:, :, 14:].mean(axis=2),
        "per_time_hand_std": abs_std[:, :, :14].mean(axis=2),
        "per_time_full_std": abs_std.mean(axis=2),
        "per_joint_std": abs_std.mean(axis=1),
        "per_time_joint_std": abs_std,
        "per_time_arm_native": abs_native[:, :, 14:].mean(axis=2),
    }


def bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def sign_flip_pvalue(values: np.ndarray, signs: np.ndarray) -> float:
    observed = float(values.mean())
    centered = values - observed
    null = (signs * centered[None, :]).mean(axis=1)
    return float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def holm_adjust(rows: list[dict[str, Any]], family_key: str = "family", p_key: str = "p_value") -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get(p_key, "") != "":
            groups[str(row.get(family_key, "unassigned"))].append(index)
    for _, indexes in groups.items():
        ordered = sorted(indexes, key=lambda index: float(rows[index][p_key]))
        previous = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, max(previous, (count - rank) * float(rows[index][p_key])))
            rows[index]["holm_p_value"] = adjusted
            previous = adjusted


def save_figure(name: str, figure: plt.Figure, data_rows: list[dict[str, Any]], caption: str, alt_text: str) -> None:
    data_path = ROOT / "figure_data" / f"{name}.csv"
    write_csv(data_path, data_rows)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(ROOT / "figures" / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    figure.clear()
    plt.close(figure)
    (ROOT / "figures" / f"{name}.caption.md").write_text(f"# {name}\n\n{caption}\n\n**中文 alt text：** {alt_text}\n")
    FIGURES.append({
        "figure": name,
        "data": rel(data_path),
        "png": f"figures/{name}.png",
        "svg": f"figures/{name}.svg",
        "pdf": f"figures/{name}.pdf",
        "caption": f"figures/{name}.caption.md",
        "visual_qa": "pending_manual_view_image",
    })


def lag_metric(pred: np.ndarray, target: np.ndarray, lag: int) -> float:
    if lag > 0:
        left, right = pred[:, :-lag], target[:, lag:]
    elif lag < 0:
        left, right = pred[:, -lag:], target[:, :lag]
    else:
        left, right = pred, target
    return float(np.abs(left[:, :, 14:] - right[:, :, 14:]).mean())


def validate_lag_synthetic() -> dict[str, Any]:
    target = np.arange(30, dtype=np.float64)[None, :, None]
    target = np.repeat(target, 2, axis=0)
    pred = np.zeros_like(target)
    pred[:, 1:] = target[:, :-1]
    values = {str(lag): lag_metric(np.repeat(pred, 28, axis=2), np.repeat(target, 28, axis=2), lag) for lag in LAGS}
    best = min(values, key=values.get)
    return {"pass": best == "-1", "known_shift_lag": -1, "best_lag": int(best), "mae_by_lag": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-core-b", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--normalizer-path", required=True, type=Path)
    args = parser.parse_args()
    global ROOT, ARCHIVE_CORE_B, SOURCE, INPUT_PATH, NORMALIZER_PATH, BENCHMARK_ID
    ROOT = args.output_root.expanduser().resolve()
    ARCHIVE_CORE_B = args.archive_core_b.expanduser().resolve()
    SOURCE = ARCHIVE_CORE_B / "analysis" / "CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"
    INPUT_PATH = args.input_path.expanduser().resolve()
    NORMALIZER_PATH = args.normalizer_path.expanduser().resolve()
    if not ARCHIVE_CORE_B.is_dir() or not SOURCE.is_dir(): raise FileNotFoundError("archive Core B source is incomplete")
    if not INPUT_PATH.is_file() or not NORMALIZER_PATH.is_file(): raise FileNotFoundError("explicit retrospective input or normalizer is missing")
    ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_ID = ROOT.name
    for directory in ("data", "figures", "figure_data", "conclusions", "qa"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    inventory = load_json(SOURCE / "CORE_B_LOGICAL_CELL_INVENTORY.json")["rows"]
    accepted_rows = [row for row in inventory if row.get("stage") in {"B1", "B2", "B3"} and str(row.get("status", "")).startswith("accepted")]
    proxy_rows = [row for row in inventory if row.get("stage") == "B2_PROXY"]
    if len(accepted_rows) != 21 or len(proxy_rows) != 4:
        raise RuntimeError(f"unexpected formal/proxy inventory: {len(accepted_rows)}/{len(proxy_rows)}")

    canonical = np.load(INPUT_PATH, allow_pickle=False)
    dev_mask = canonical["split_code"] == 1
    train_mask = canonical["split_code"] == 0
    canonical_ids_all = np.asarray(canonical["episode_index"], dtype=np.int64)
    dev_ids = canonical_ids_all[dev_mask]
    if len(dev_ids) != 614 or len(np.unique(dev_ids)) != 614:
        raise RuntimeError("canonical development population is not 614 unique episodes")
    normalizer = load_json(NORMALIZER_PATH)["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    safe_std = np.asarray(normalizer["safe_std"], dtype=np.float64)
    canonical_target_std_all = np.asarray(canonical["targets_standardized"], dtype=np.float64)
    canonical_target_native_all = np.asarray(canonical["targets"], dtype=np.float64)
    canonical_expected_std_all = (canonical_target_native_all - mean[None, :, :]) / safe_std[None, :, :]
    normalizer_max_abs = float(np.max(np.abs(canonical_expected_std_all - canonical_target_std_all)))
    # The source targets are float32; this is a representation check, not a
    # performance gate.  Keep the observed discrepancy in the audit table.
    if normalizer_max_abs > 2e-6:
        raise RuntimeError(f"canonical train-only normalization mismatch: {normalizer_max_abs}")
    train_target_native = canonical_target_native_all[train_mask]
    global_native_vector = train_target_native.mean(axis=(0, 1))
    per_time_native = train_target_native.mean(axis=0)

    common_set = set(int(value) for value in dev_ids.tolist())
    loaded_sources: dict[str, dict[str, Any]] = {}
    for row in accepted_rows + proxy_rows:
        bundle = np.load(Path(row["prediction_path"]), allow_pickle=False)
        ids = np.asarray(bundle["episode_index"], dtype=np.int64)
        common_set.intersection_update(int(value) for value in ids.tolist())
        loaded_sources[row["cell"]] = {"bundle": bundle, "ids": ids, "row": row, "path": Path(row["prediction_path"])}
    common_ids = np.asarray(sorted(common_set), dtype=np.int64)
    if len(common_ids) != 614:
        raise RuntimeError(f"common accepted population is {len(common_ids)}, expected 614")
    canonical_indices = align_indices(dev_ids, common_ids)
    target_std = canonical_target_std_all[dev_mask][canonical_indices]
    target_native = canonical_target_native_all[dev_mask][canonical_indices]
    task_all = np.asarray(canonical["original_task_index"], dtype=np.int64)
    task = task_all[dev_mask][canonical_indices]
    last_action_all = np.asarray(canonical["last_observed_action"], dtype=np.float64)
    last_action = last_action_all[dev_mask][canonical_indices]

    models: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    model_rows = [model_meta(row) for row in accepted_rows]
    for row in accepted_rows:
        meta = model_meta(row)
        source = loaded_sources[row["cell"]]
        bundle = source["bundle"]
        indices = align_indices(source["ids"], common_ids)
        pred_std = np.asarray(bundle["predictions_standardized"], dtype=np.float64)[indices]
        pred_native = np.asarray(bundle["predictions_native"], dtype=np.float64)[indices]
        target_std_source = np.asarray(bundle["targets_standardized"], dtype=np.float64)[indices]
        target_native_source = np.asarray(bundle["targets_native"], dtype=np.float64)[indices]
        max_std = float(np.max(np.abs(target_std_source - target_std)))
        max_native = float(np.max(np.abs(target_native_source - target_native)))
        audit_rows.append({
            "model": row["cell"], "stage": row["stage"], "source_path": row["prediction_path"],
            "source_sha256": sha256(Path(row["prediction_path"])), "N": len(common_ids),
            "prediction_shape": str(tuple(pred_std.shape)), "target_shape": str(tuple(target_std.shape)),
            "target_std_max_abs_vs_canonical": max_std, "target_native_max_abs_vs_canonical": max_native,
            "finite_prediction": bool(np.isfinite(pred_std).all() and np.isfinite(pred_native).all()),
            "final_test_access": row.get("final_test_access", 0), "status": "PASS" if max(max_std, max_native) <= 1e-6 else "FAIL",
        })
        models[row["cell"]] = {**meta, "pred_std": pred_std, "pred_native": pred_native, "metrics": metric_arrays(pred_std, pred_native, target_std, target_native)}
        registry_rows.append({**meta, "registry_role": "formal_accepted", "included_in_main": True, "source_sha256": sha256(Path(row["prediction_path"])), "N": len(common_ids)})
    if any(row["status"] == "FAIL" for row in audit_rows):
        raise RuntimeError("at least one accepted source has target lineage mismatch")

    # Derived controls use only the locked development input and train-only target statistics.
    persistence_native = np.repeat(last_action[:, None, :], 30, axis=1)
    persistence_std = (persistence_native - mean[None, :, :]) / safe_std[None, :, :]
    global_native = np.repeat(global_native_vector[None, None, :], len(common_ids), axis=0)
    global_native = np.repeat(global_native, 30, axis=1)
    per_time_native_eval = np.repeat(per_time_native[None, :, :], len(common_ids), axis=0)
    controls = {
        "PERSISTENCE": (persistence_std, persistence_native, "last_observed_action repeated over horizon"),
        "GLOBAL_MEAN": ((global_native - mean[None, :, :]) / safe_std[None, :, :], global_native, "train-only global joint mean repeated over horizon"),
        "PER_TIME_MEAN": ((per_time_native_eval - mean[None, :, :]) / safe_std[None, :, :], per_time_native_eval, "train-only per-time joint mean"),
    }
    for name, (pred_std, pred_native, description) in controls.items():
        models[name] = {
            "stage": "CONTROL", "model": name, "branch": "control", "video_input": "none", "adaptation": "none",
            "representation": "derived", "interval_seconds": "", "unique_observation_frames": "", "encoder_slots": "",
            "source_kind": "derived_control", "acceptance_status": "derived_not_formal_cell", "reused_from": "",
            "prediction_path": str(INPUT_PATH), "final_test_access": 0, "description": description,
            "pred_std": pred_std, "pred_native": pred_native, "metrics": metric_arrays(pred_std, pred_native, target_std, target_native),
        }
        registry_rows.append({"stage": "CONTROL", "model": name, "branch": "control", "video_input": "none", "adaptation": "none", "representation": "derived", "interval_seconds": "", "unique_observation_frames": "", "encoder_slots": "", "source_kind": "derived_control", "acceptance_status": "not_formal_cell", "reused_from": "", "prediction_path": str(INPUT_PATH), "final_test_access": 0, "registry_role": "derived_control", "included_in_main": True, "source_sha256": sha256(INPUT_PATH), "N": len(common_ids), "description": description})
    for name in ("LINEAR_CONTROL", "RIDGE_CONTROL"):
        registry_rows.append({"stage": "CONTROL", "model": name, "branch": "control", "video_input": "none", "adaptation": "none", "representation": "unavailable", "interval_seconds": "", "unique_observation_frames": "", "encoder_slots": "", "source_kind": "no_locked_artifact", "acceptance_status": "not_available_no_locked_artifact", "reused_from": "", "prediction_path": "", "final_test_access": 0, "registry_role": "unavailable_control", "included_in_main": False, "source_sha256": "", "N": "", "description": "No accepted locked linear/ridge prediction artifact; no retraining permitted."})

    # Store proxy metrics separately; their contextual-token provenance never enters models/main results.
    proxy_metric_rows: list[dict[str, Any]] = []
    for row in proxy_rows:
        source = loaded_sources[row["cell"]]
        bundle = source["bundle"]
        indices = align_indices(source["ids"], common_ids)
        pred_std = np.asarray(bundle["predictions_standardized"], dtype=np.float64)[indices]
        pred_native = np.asarray(bundle["predictions_native"], dtype=np.float64)[indices]
        metrics = metric_arrays(pred_std, pred_native, target_std, target_native)
        proxy_metric_rows.append({"stage": "B2_PROXY", "model": row["cell"], "branch": row.get("branch", ""), "video_input": row.get("video_input", ""), "source_kind": row.get("source_kind", ""), "status": "excluded_from_formal_B2", "N": len(common_ids), "std_arm_mae": float(metrics["arm_mae_std"].mean()), "std_hand_mae": float(metrics["hand_mae_std"].mean()), "std_full28_mae": float(metrics["full28_mae_std"].mean()), "source_path": row["prediction_path"], "source_sha256": sha256(Path(row["prediction_path"]))})
        registry_rows.append({**model_meta(row), "registry_role": "proxy_excluded", "included_in_main": False, "source_sha256": sha256(Path(row["prediction_path"])), "N": len(common_ids), "description": "Contextual-token proxy after full encoder; not true-pixel sampling."})

    # Source artifact revalidation is deliberately limited to lightweight protocol, prediction and input artifacts.
    source_artifact_rows: list[dict[str, Any]] = []
    manifest_path = SOURCE / "SOURCE_MANIFEST.csv"
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            role = row["role"]
            should_hash = role in {"protocol_or_canonical_input", "prediction"} and path.exists()
            source_artifact_rows.append({
                **row, "exists_now": path.exists(),
                "current_sha256": sha256(path) if should_hash else "not_rehashed_checkpoint_or_missing",
                "hash_check": "PASS" if (not should_hash or sha256(path) == row["sha256"]) else "FAIL",
                "read_policy": "rehashed" if should_hash else "recorded_hash_only",
            })
    if any(row["hash_check"] == "FAIL" for row in source_artifact_rows):
        raise RuntimeError("source manifest prediction/input hash mismatch")

    # Common population and lineage tables.
    motion_score = np.abs(np.diff(target_native, axis=1)).mean(axis=(1, 2))
    order = np.argsort(motion_score, kind="mergesort")
    stratum_names = np.empty(len(common_ids), dtype=object)
    strata: dict[str, np.ndarray] = {}
    for name, indices in zip(("low", "mid", "high"), np.array_split(order, 3)):
        strata[name] = np.asarray(indices, dtype=np.int64)
        stratum_names[indices] = name
    population_rows = [{"episode_id": int(common_ids[i]), "task_index": int(task[i]), "motion_score_native": float(motion_score[i]), "motion_stratum": str(stratum_names[i]), "source_split": "development", "target_steps": 30, "target_dimensions": 28} for i in range(len(common_ids))]
    write_csv(ROOT / "data" / "COMMON_EVALUATION_POPULATION.csv", population_rows)
    write_json(ROOT / "data" / "COMMON_EVALUATION_POPULATION.json", {"N": len(common_ids), "episode_ids": common_ids.tolist(), "task_index": task.tolist(), "motion_strata": {key: common_ids[value].tolist() for key, value in strata.items()}, "final_test_access": 0})
    write_csv(ROOT / "data" / "TARGET_NORMALIZATION_AUDIT.csv", audit_rows)
    write_json(ROOT / "data" / "TARGET_NORMALIZATION_AUDIT.json", {"normalizer_path": str(NORMALIZER_PATH), "normalizer_sha256": sha256(NORMALIZER_PATH), "normalizer_horizon": normalizer["horizon_seconds"], "safe_std_min": float(safe_std.min()), "safe_std_max": float(safe_std.max()), "canonical_expected_vs_stored_max_abs": normalizer_max_abs, "rows": audit_rows, "status": "PASS"})
    write_csv(ROOT / "data" / "MODEL_SOURCE_REGISTRY.csv", registry_rows)
    write_csv(ROOT / "data" / "SOURCE_ARTIFACT_REGISTRY.csv", source_artifact_rows)
    write_json(ROOT / "data" / "SOURCE_ARTIFACT_REVALIDATION.json", {"source_manifest": str(manifest_path), "rows": len(source_artifact_rows), "rehashed_roles": ["protocol_or_canonical_input", "prediction"], "checkpoint_hash_policy": "recorded manifest hash only", "status": "PASS", "final_test_access": 0})

    acceptance_rows = []
    for stage, expected in (("B1", 7), ("B2", 6), ("B3", 8)):
        stage_rows = [row for row in accepted_rows if row["stage"] == stage]
        acceptance_rows.append({"stage": stage, "expected_cells": expected, "accepted_cells": len(stage_rows), "coverage": f"{len(stage_rows)}/{expected}", "status": "PASS" if len(stage_rows) == expected else "FAIL", "formal_or_auxiliary": "formal"})
    acceptance_rows.append({"stage": "B2_PROXY", "expected_cells": 4, "accepted_cells": 4, "coverage": "4/4", "status": "AUXILIARY_EXCLUDED", "formal_or_auxiliary": "proxy"})
    write_csv(ROOT / "ACCEPTANCE_AND_SCOPE.csv", acceptance_rows)
    write_json(ROOT / "ACCEPTANCE_AND_SCOPE.json", {"rows": acceptance_rows, "candidate_count": 0, "candidate_decision": "not_ready_b3_gate_not_passed", "b3_single_seed": True, "final_test_access": 0})
    (ROOT / "B2_PROXY_BOUNDARY.md").write_text("# B2 proxy boundary\n\nThe four `B2_PROXY` rows are contextual-token aggregation artifacts: sparse selection occurred after the full spatiotemporal encoder. They are retained for transparency, evaluated in a separate table, and excluded from true-pixel B2 selection, B3 strategy claims, and all formal pooled comparisons.\n")
    (ROOT / "SAMPLING_EFFICIENCY_BOUNDARY.md").write_text("# Sampling-efficiency boundary\n\nThe accepted records expose nominal intervals, unique observation frames, and 32 encoder slots. They do not provide comparable end-to-end wall-clock, peak-VRAM, or throughput measurements for all cells. This benchmark therefore reports frame/slot metadata but makes no runtime-speed or memory-efficiency claim.\n")

    eval_names = [row["cell"] for row in accepted_rows] + list(controls)
    main_result_rows: list[dict[str, Any]] = []
    model_ci: dict[str, tuple[float, float]] = {}
    bootstrap_rng = np.random.default_rng(SEED)
    bootstrap_indices = bootstrap_rng.integers(0, len(common_ids), size=(BOOTSTRAP_REPLICATES, len(common_ids)), dtype=np.int32)
    np.save(ROOT / "data" / "paired_episode_bootstrap_indices.npy", bootstrap_indices)
    for name in eval_names:
        model = models[name]
        metrics = model["metrics"]
        low, high = bootstrap_ci(metrics["arm_mae_std"], bootstrap_indices)
        model_ci[name] = (low, high)
        meta = {key: model.get(key, "") for key in ("stage", "branch", "video_input", "adaptation", "representation", "interval_seconds", "unique_observation_frames", "encoder_slots", "source_kind", "acceptance_status", "reused_from", "prediction_path", "final_test_access")}
        main_result_rows.append({"model": name, **meta, "N": len(common_ids), "std_arm_mae": float(metrics["arm_mae_std"].mean()), "std_arm_mae_ci_low": low, "std_arm_mae_ci_high": high, "std_hand_mae": float(metrics["hand_mae_std"].mean()), "std_full28_mae": float(metrics["full28_mae_std"].mean()), "std_arm_final_step_mae": float(metrics["arm_final_std"].mean()), "std_arm_first_diff_mae": float(metrics["arm_first_diff_std"].mean()), "std_rmse": float(metrics["full_rmse_std"].mean()), "native_arm_mae": float(metrics["arm_mae_native"].mean()), "native_hand_mae": float(metrics["hand_mae_native"].mean()), "native_full28_mae": float(metrics["full28_mae_native"].mean()), "native_arm_final_step_mae": float(metrics["arm_final_native"].mean()), "native_arm_first_diff_mae": float(metrics["arm_first_diff_native"].mean()), "native_rmse": float(metrics["full_rmse_native"].mean()), "final_test_access": 0})
    write_csv(ROOT / "CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv", main_result_rows)
    write_json(ROOT / "CORE_B_RETROSPECTIVE_MAIN_RESULTS.json", {"rows": main_result_rows, "primary": "std_arm_mae", "lower_is_better": True, "formal_models": [row["cell"] for row in accepted_rows], "derived_controls": list(controls), "final_test_access": 0})
    write_csv(ROOT / "CORE_B_PROXY_RESULTS_SEPARATE.csv", proxy_metric_rows)

    # Episode, time, joint, task and motion-stratified tables.
    per_episode_rows: list[dict[str, Any]] = []
    per_time_rows: list[dict[str, Any]] = []
    per_joint_time_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    metric_keys = ("arm_mae_std", "hand_mae_std", "full28_mae_std", "arm_final_std", "arm_first_diff_std", "full_rmse_std", "arm_mae_native", "hand_mae_native", "full28_mae_native")
    for name in eval_names:
        metrics = models[name]["metrics"]
        for i, episode_id in enumerate(common_ids):
            per_episode_rows.append({"model": name, "episode_id": int(episode_id), "task_index": int(task[i]), "motion_score_native": float(motion_score[i]), "motion_stratum": str(stratum_names[i]), **{key: float(metrics[key][i]) for key in metric_keys}})
        for time_index in range(30):
            per_time_rows.append({"model": name, "time_index": time_index, "arm_mae_std": float(metrics["per_time_arm_std"][:, time_index].mean()), "hand_mae_std": float(metrics["per_time_hand_std"][:, time_index].mean()), "full28_mae_std": float(metrics["per_time_full_std"][:, time_index].mean()), "arm_mae_native": float(metrics["per_time_arm_native"][:, time_index].mean())})
        mean_joint_time = metrics["per_time_joint_std"].mean(axis=0)
        for time_index in range(30):
            for joint_index in range(28):
                per_joint_time_rows.append({"model": name, "time_index": time_index, "joint_index": joint_index, "mae_std": float(mean_joint_time[time_index, joint_index])})
        for task_id in sorted(np.unique(task).tolist()):
            mask = task == task_id
            per_task_rows.append({"model": name, "task_index": int(task_id), "N": int(mask.sum()), "arm_mae_std": float(metrics["arm_mae_std"][mask].mean()), "full28_mae_std": float(metrics["full28_mae_std"][mask].mean()), "arm_first_diff_std": float(metrics["arm_first_diff_std"][mask].mean())})
        for stratum, indices in strata.items():
            motion_rows.append({"model": name, "motion_stratum": stratum, "N": len(indices), "motion_score_mean": float(motion_score[indices].mean()), "arm_mae_std": float(metrics["arm_mae_std"][indices].mean()), "full28_mae_std": float(metrics["full28_mae_std"][indices].mean()), "arm_first_diff_std": float(metrics["arm_first_diff_std"][indices].mean())})
    write_csv(ROOT / "CORE_B_PER_EPISODE_METRICS.csv", per_episode_rows)
    write_csv(ROOT / "CORE_B_PER_TIMESTEP_METRICS.csv", per_time_rows)
    write_csv(ROOT / "CORE_B_PER_JOINT_TIME_METRICS.csv", per_joint_time_rows)
    write_csv(ROOT / "CORE_B_PER_TASK_METRICS.csv", per_task_rows)
    write_csv(ROOT / "CORE_B_MOTION_STRATIFIED_METRICS.csv", motion_rows)

    # Paired formal effect families.
    comparison_specs = [
        ("B1_video_state_fusion", "PERSISTENCE", "B-STATE", "state_vs_persistence"),
        ("B1_video_state_fusion", "B-STATE", "B-D-P-F", "dynamic_video_vs_state"),
        ("B1_video_state_fusion", "B-STATE", "B-S-P-F", "static_video_vs_state"),
        ("B1_video_state_fusion", "B-STATE", "B-D-FUS-P-F", "dynamic_fusion_vs_state"),
        ("B1_video_state_fusion", "B-STATE", "B-S-FUS-P-F", "static_fusion_vs_state"),
        ("B1_video_state_fusion", "B-D-P-F", "B-D-A-F", "attentive_vs_pooled_dynamic"),
        ("B2_sampling", "B2-V-0125", "B2-V-0250", "video_dense_vs_0250"),
        ("B2_sampling", "B2-V-0125", "B2-V-0500", "video_dense_vs_0500"),
        ("B2_sampling", "B2-FUS-0125", "B2-FUS-0250", "fusion_dense_vs_0250"),
        ("B2_sampling", "B2-FUS-0125", "B2-FUS-0500", "fusion_dense_vs_0500"),
        ("B2_sampling", "B2-V-0125", "B2-FUS-0125", "fusion_vs_video_dense"),
        ("B3_lora_factorial", "B3-V-D-F", "B3-V-D-L", "video_dynamic_lora_gain"),
        ("B3_lora_factorial", "B3-V-S-F", "B3-V-S-L", "video_static_lora_gain"),
        ("B3_lora_factorial", "B3-FUS-D-F", "B3-FUS-D-L", "fusion_dynamic_lora_gain"),
        ("B3_lora_factorial", "B3-FUS-S-F", "B3-FUS-S-L", "fusion_static_lora_gain"),
        ("B3_lora_factorial", "B3-V-S-F", "B3-V-D-F", "video_dynamic_vs_static_frozen"),
        ("B3_lora_factorial", "B3-V-S-L", "B3-V-D-L", "video_dynamic_vs_static_lora"),
        ("B3_lora_factorial", "B3-FUS-S-F", "B3-FUS-D-F", "fusion_dynamic_vs_static_frozen"),
        ("B3_lora_factorial", "B3-FUS-S-L", "B3-FUS-D-L", "fusion_dynamic_vs_static_lora"),
        ("B3_lora_factorial", "B-STATE", "B3-FUS-D-L", "fusion_dynamic_lora_vs_state"),
    ]
    signs_rng = np.random.default_rng(SEED + 1)
    signs = signs_rng.choice(np.asarray([-1.0, 1.0]), size=(PERMUTATION_REPLICATES, len(common_ids))).astype(np.float32)
    np.save(ROOT / "data" / "paired_signflip_signs.npy", signs)
    effect_rows: list[dict[str, Any]] = []
    for family, baseline, comparator, label in comparison_specs:
        if baseline not in models or comparator not in models:
            continue
        difference = models[baseline]["metrics"]["arm_mae_std"] - models[comparator]["metrics"]["arm_mae_std"]
        low, high = bootstrap_ci(difference, bootstrap_indices)
        effect_rows.append({"family": family, "comparison": label, "baseline": baseline, "comparator": comparator, "metric": "std_arm_mae", "N": len(difference), "effect_baseline_minus_comparator": float(difference.mean()), "ci_low": low, "ci_high": high, "p_value": sign_flip_pvalue(difference, signs), "holm_p_value": "", "interpretation": "positive means comparator has lower error"})
    holm_adjust(effect_rows)
    write_csv(ROOT / "CORE_B_PAIRED_EFFECTS.csv", effect_rows)
    write_json(ROOT / "CORE_B_PAIRED_EFFECTS.json", {"rows": effect_rows, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "permutation_replicates": PERMUTATION_REPLICATES, "seed": SEED, "unit": "episode", "final_test_access": 0})

    rank_rows = []
    ranked_names = sorted(eval_names, key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    for rank, name in enumerate(ranked_names, start=1):
        rank_rows.append({"rank": rank, "model": name, "stage": models[name].get("stage", ""), "formal_cell": name in [row["cell"] for row in accepted_rows], "std_arm_mae": float(models[name]["metrics"]["arm_mae_std"].mean()), "ci_low": model_ci[name][0], "ci_high": model_ci[name][1]})
    write_csv(ROOT / "CORE_B_RANKS.csv", rank_rows)

    # Fixed representative episodes and diagnostic sets.
    selection_rng = np.random.default_rng(SEED)
    selected_indices: list[int] = []
    selection_rows: list[dict[str, Any]] = []
    for stratum in ("low", "mid", "high"):
        candidates = strata[stratum].copy()
        selection_rng.shuffle(candidates)
        chosen = np.sort(candidates[:4])
        selected_indices.extend(chosen.tolist())
        for index in chosen:
            selection_rows.append({"episode_id": int(common_ids[index]), "task_index": int(task[index]), "motion_score_native": float(motion_score[index]), "motion_stratum": stratum, "selection_seed": SEED, "selection_rule": "four random-within-tertile episodes; target-only"})
    selected_indices = sorted(selected_indices)
    write_csv(ROOT / "data" / "REPRESENTATIVE_EPISODES.csv", selection_rows)
    selected_set = np.asarray(selected_indices, dtype=np.int64)
    chosen_model = "B3-FUS-D-L"
    chosen_errors = models[chosen_model]["metrics"]["arm_mae_std"]
    persistence_errors = models["PERSISTENCE"]["metrics"]["arm_mae_std"]
    worst = np.argsort(chosen_errors)[-12:][::-1]
    best = np.argsort(chosen_errors)[:12]
    degradation = chosen_errors - persistence_errors
    largest_degradation = np.argsort(degradation)[-12:][::-1]
    write_json(ROOT / "data" / "DIAGNOSTIC_EPISODES.json", {"fixed_model": chosen_model, "worst": common_ids[worst].tolist(), "best": common_ids[best].tolist(), "largest_degradation_vs_persistence": common_ids[largest_degradation].tolist(), "selection_note": "diagnostic lists are not confirmatory model-selection samples"})
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_models = ["PERSISTENCE", "B-STATE", "B2-FUS-0125", "B3-FUS-D-F", "B3-FUS-D-L"]
    for index in selected_indices:
        for name in trajectory_models:
            model = models[name]
            for time_index in range(30):
                for arm_offset in range(14):
                    joint_index = 14 + arm_offset
                    trajectory_rows.append({"episode_id": int(common_ids[index]), "task_index": int(task[index]), "motion_stratum": str(stratum_names[index]), "model": name, "time_index": time_index, "joint_index": joint_index, "target_native": float(target_native[index, time_index, joint_index]), "prediction_native": float(model["pred_native"][index, time_index, joint_index]), "target_std": float(target_std[index, time_index, joint_index]), "prediction_std": float(model["pred_std"][index, time_index, joint_index])})
    write_csv(ROOT / "data" / "TRAJECTORY_REPRESENTATIVE_EPISODES.csv", trajectory_rows)
    write_json(ROOT / "data" / "TRAJECTORY_SELECTION_MANIFEST.json", {"N": 12, "episodes_per_stratum": 4, "seed": SEED, "strata": ["low", "mid", "high"], "models": trajectory_models, "joint_subset": "arm14", "same_manifest_all_models": True})

    # Rule-based failure taxonomy: descriptive flags, not manually adjudicated mechanisms.
    failure_rows: list[dict[str, Any]] = []
    for name in eval_names:
        model = models[name]
        metrics = model["metrics"]
        range_native = np.ptp(model["pred_native"], axis=1).mean(axis=1)
        q95_abs = float(np.quantile(metrics["arm_mae_std"], 0.95))
        q95_fd = float(np.quantile(metrics["arm_first_diff_std"], 0.95))
        q95_final = float(np.quantile(metrics["arm_final_std"], 0.95))
        for i, episode_id in enumerate(common_ids):
            if not np.isfinite(model["pred_std"][i]).all():
                label = "nonfinite_prediction"
            elif range_native[i] < 1e-5:
                label = "near_constant_prediction"
            elif metrics["arm_mae_std"][i] >= q95_abs:
                label = "high_absolute_error_top5pct"
            elif metrics["arm_first_diff_std"][i] >= q95_fd:
                label = "high_temporal_difference_error_top5pct"
            elif metrics["arm_final_std"][i] >= q95_final:
                label = "high_final_step_error_top5pct"
            else:
                label = "no_rule_trigger"
            failure_rows.append({"model": name, "episode_id": int(episode_id), "task_index": int(task[i]), "motion_stratum": str(stratum_names[i]), "rule_based_label": label, "manual_label": "not_reviewed", "arm_mae_std": float(metrics["arm_mae_std"][i]), "arm_first_diff_std": float(metrics["arm_first_diff_std"][i]), "arm_final_std": float(metrics["arm_final_std"][i]), "prediction_native_range_mean": float(range_native[i])})
    write_csv(ROOT / "CORE_B_FAILURE_TAXONOMY.csv", failure_rows)
    (ROOT / "FAILURE_TAXONOMY_RULES.md").write_text("# Failure taxonomy rules\n\nLabels are deterministic screening flags, not human mechanism diagnoses. Non-finite predictions and near-constant outputs are checked first; otherwise top-5% thresholds are computed separately per model for absolute arm MAE, first-difference MAE, and final-step MAE. `manual_label=not_reviewed` is intentional.\n")

    # Temporal alignment diagnostics, with synthetic validation before real scans.
    synthetic = validate_lag_synthetic()
    if not synthetic["pass"]:
        raise RuntimeError(f"synthetic lag validation failed: {synthetic}")
    lag_rows: list[dict[str, Any]] = []
    lag_curves: dict[str, dict[int, float]] = {}
    for name in eval_names:
        curve = {lag: lag_metric(models[name]["pred_std"], target_std, lag) for lag in LAGS}
        lag_curves[name] = curve
        best_lag = min(curve, key=curve.get)
        for lag, value in curve.items():
            lag_rows.append({"model": name, "lag": lag, "arm_mae_std": value, "is_zero_lag": lag == 0, "best_lag_for_model": int(best_lag), "diagnostic_status": "exploratory"})
    write_csv(ROOT / "CORE_B_TEMPORAL_ALIGNMENT_DIAGNOSTICS.csv", lag_rows)
    write_json(ROOT / "CORE_B_TEMPORAL_ALIGNMENT_DIAGNOSTICS.json", {"lags": list(LAGS), "zero_lag_primary": True, "synthetic_validation": synthetic, "rows": lag_rows, "interpretation": "exploratory integer-lag sensitivity only"})
    (ROOT / "TEMPORAL_ALIGNMENT_BOUNDARY.md").write_text("# Temporal alignment boundary\n\nZero-lag error is the primary result. The -5..+5 integer-lag scan is exploratory and was preceded by a synthetic shift validation. A best lag does not establish causal timing, correct a data error, or replace the locked zero-lag endpoint.\n")

    # Validity scorecard is explicit about what the artifacts can and cannot establish.
    scorecard = [
        {"dimension": "schema_and_lineage", "status": "PASS", "evidence": "all formal source bundles aligned to a common 614-episode ID set; shapes (614,30,28)"},
        {"dimension": "normalization", "status": "PASS", "evidence": "train-only 1.0s mean/safe_std reproduces stored standardized targets"},
        {"dimension": "formal_stage_coverage", "status": "PASS", "evidence": "B1 7/7, true-pixel B2 6/6, corrected B3 8/8"},
        {"dimension": "proxy_separation", "status": "PASS", "evidence": "four contextual-token proxy cells excluded from formal B2"},
        {"dimension": "discrimination", "status": "PASS", "evidence": "paired model/control comparisons computed on episode-level errors"},
        {"dimension": "temporal_sensitivity", "status": "WEAK", "evidence": "zero-lag and exploratory lag diagnostics exist; causal timing is not established"},
        {"dimension": "physical_construct_validity", "status": "UNKNOWN", "evidence": "command-value units/URDF/FK not verified by this benchmark"},
        {"dimension": "runtime_efficiency", "status": "UNKNOWN", "evidence": "comparable timing and peak-VRAM measurements are absent"},
        {"dimension": "cross_seed_stability", "status": "UNKNOWN", "evidence": "formal B3 is a single training seed"},
        {"dimension": "external_validity", "status": "UNKNOWN", "evidence": "development split only; final-test remains locked"},
        {"dimension": "outcome_independence", "status": "WEAK", "evidence": "retrospective analysis with known formal gate outcome; exposure disclosed"},
        {"dimension": "final_test_lock", "status": "PASS", "evidence": "no final-test arrays or jobs accessed; final_test_access=0"},
    ]
    write_csv(ROOT / "VALIDITY_SCORECARD.csv", scorecard)
    write_json(ROOT / "VALIDITY_SCORECARD.json", {"rows": scorecard, "status": "COMPLETE_WITH_CAVEATS"})

    # Figures use only tables/arrays generated in this run.
    formal_names = [row["cell"] for row in accepted_rows]
    color_map = {"PERSISTENCE": "#555555", "B-STATE": "#0072B2", "B2-FUS-0125": "#CC79A7", "B3-FUS-D-L": "#D55E00", "B3-FUS-D-F": "#009E73"}
    # 1. Main forest.
    ordered = sorted(formal_names, key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    y = np.arange(len(ordered))
    values = [float(models[name]["metrics"]["arm_mae_std"].mean()) for name in ordered]
    lows = [model_ci[name][0] for name in ordered]
    highs = [model_ci[name][1] for name in ordered]
    ax.errorbar(values, y, xerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)], fmt="o", color="#0072B2", capsize=3)
    ax.set_yticks(y, ordered); ax.invert_yaxis(); ax.set_xlabel("Standardized arm-14 MAE (lower is better)"); ax.set_title("Formal Core B models: primary endpoint")
    save_figure("fig01_main_results_forest", fig, [{"model": name, "std_arm_mae": values[i], "ci_low": lows[i], "ci_high": highs[i]} for i, name in enumerate(ordered)], "Formal accepted B1/B2/B3 models with episode-bootstrap 95% intervals.", "正式模型的标准化 arm-14 MAE 森林图，点为均值，横线为 episode bootstrap 95% 区间。")

    # 2. Error over horizon.
    horizon_names = ["PERSISTENCE", "B-STATE", "B2-FUS-0125", "B3-FUS-D-F", "B3-FUS-D-L"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    data_rows = []
    for name in horizon_names:
        vals = models[name]["metrics"]["per_time_arm_std"].mean(axis=0)
        ax.plot(np.arange(1, 31), vals, label=name, color=color_map.get(name), linewidth=2)
        data_rows.extend({"model": name, "time_index": t, "arm_mae_std": float(vals[t])} for t in range(30))
    ax.set_xlabel("Future time index"); ax.set_ylabel("Standardized arm MAE"); ax.set_title("Error over the 30-step horizon"); ax.legend(ncol=2, fontsize=8)
    save_figure("fig02_error_over_horizon", fig, data_rows, "Predeclared persistence/state/B2/B3 horizon comparison.", "五个预先指定模型在 30 个未来时间步上的标准化 arm MAE 曲线。")

    # 3. B2 sampling.
    b2_names = ["B2-V-0125", "B2-V-0250", "B2-V-0500", "B2-FUS-0125", "B2-FUS-0250", "B2-FUS-0500"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    data_rows = []
    x = np.arange(3); width = 0.35
    for offset, branch in zip((-width / 2, width / 2), ("video_only", "fusion")):
        vals = []
        for interval in (0.125, 0.25, 0.5):
            name = next(item for item in b2_names if item.startswith("B2-V-" if branch == "video_only" else "B2-FUS-") and models[item]["interval_seconds"] == interval)
            value = float(models[name]["metrics"]["arm_mae_std"].mean()); vals.append(value); data_rows.append({"branch": branch, "interval_seconds": interval, "frames": models[name]["unique_observation_frames"], "model": name, "std_arm_mae": value})
        ax.bar(x + offset, vals, width, label=branch)
    ax.set_xticks(x, ["0.125 / 32", "0.25 / 16", "0.5 / 8"]); ax.set_ylabel("Standardized arm-14 MAE"); ax.set_xlabel("Nominal interval / unique pixel frames"); ax.set_title("True-pixel B2 sampling comparison"); ax.legend()
    save_figure("fig03_b2_sampling", fig, data_rows, "True-pixel B2 accuracy by nominal sampling interval; dense controls are retained explicitly.", "真实像素 B2 在三种采样间隔和 video/fusion 两种输入下的 arm MAE 柱状图；不表示运行速度。")

    # 4. B3 factorial.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="constrained")
    data_rows = []
    for ax, branch, prefix in zip(axes, ("video_only", "fusion"), ("B3-V", "B3-FUS")):
        names = [f"{prefix}-S-F", f"{prefix}-D-F", f"{prefix}-S-L", f"{prefix}-D-L"]
        vals = [float(models[name]["metrics"]["arm_mae_std"].mean()) for name in names]
        ax.bar(["S-F", "D-F", "S-L", "D-L"], vals, color=["#56B4E9", "#0072B2", "#E69F00", "#D55E00"])
        ax.set_title(branch); ax.set_ylabel("Std arm MAE"); ax.tick_params(axis="x", rotation=35)
        data_rows.extend({"branch": branch, "model": name, "condition": label, "std_arm_mae": value} for name, label, value in zip(names, ("static frozen", "dynamic frozen", "static LoRA", "dynamic LoRA"), vals))
    fig.suptitle("Formal B3 factorial comparison")
    save_figure("fig04_b3_factorial", fig, data_rows, "B3 dynamic/static × frozen/LoRA results for video-only and fusion branches.", "B3 video-only 与 fusion 分支中 static/dynamic 和 frozen/LoRA 四个条件的标准化 arm MAE。")

    # 5. Motion vs improvement.
    gain = models["PERSISTENCE"]["metrics"]["arm_mae_std"] - models["B3-FUS-D-L"]["metrics"]["arm_mae_std"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(motion_score, gain, s=12, alpha=0.45, color="#D55E00"); ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel("Ground-truth motion score (native command values)"); ax.set_ylabel("Persistence error − B3-FUS-D-L error"); ax.set_title("Motion and model improvement")
    save_figure("fig05_motion_vs_improvement", fig, [{"episode_id": int(common_ids[i]), "motion_score_native": float(motion_score[i]), "gain_vs_persistence": float(gain[i])} for i in range(len(common_ids))], "Per-episode association between target motion and the predeclared B3 fusion dynamic LoRA improvement over persistence.", "每个 episode 的目标运动分数与 B3 fusion dynamic LoRA 相对 persistence 的误差改善散点图；不作因果解释。")

    # 6. Joint/time heatmap.
    diff_heat = models["B3-FUS-D-L"]["metrics"]["per_time_joint_std"].mean(axis=0) - models["PERSISTENCE"]["metrics"]["per_time_joint_std"].mean(axis=0)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    im = ax.imshow(diff_heat.T, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(diff_heat)), vmax=np.max(np.abs(diff_heat)))
    ax.set_xlabel("Future time index"); ax.set_ylabel("Joint index"); ax.set_title("B3-FUS-D-L minus persistence: joint × time MAE")
    fig.colorbar(im, ax=ax, label="Δ standardized absolute error")
    save_figure("fig06_joint_time_heatmap", fig, [{"time_index": t, "joint_index": j, "delta_b3_minus_persistence": float(diff_heat[t, j])} for t in range(30) for j in range(28)], "Joint-by-time error difference for the fixed B3 fusion dynamic LoRA comparison against persistence.", "B3 fusion dynamic LoRA 减去 persistence 的 28 个关节与 30 个时间步误差热图，蓝色表示 B3 误差更低。")

    # 7. Fixed trajectory examples.
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True, layout="constrained")
    data_rows = []
    for ax, index in zip(axes.ravel(), selected_indices):
        target_arm = target_native[index, :, 14:].mean(axis=1)
        ax.plot(target_arm, color="black", linewidth=2, label="target")
        for name in trajectory_models:
            pred_arm = models[name]["pred_native"][index, :, 14:].mean(axis=1)
            ax.plot(pred_arm, linewidth=1.0, label=name, color=color_map.get(name))
            data_rows.extend({"episode_id": int(common_ids[index]), "motion_stratum": str(stratum_names[index]), "time_index": t, "model": name, "arm_mean_native": float(pred_arm[t])} for t in range(30))
        data_rows.extend({"episode_id": int(common_ids[index]), "motion_stratum": str(stratum_names[index]), "time_index": t, "model": "TARGET", "arm_mean_native": float(target_arm[t])} for t in range(30))
        ax.set_title(f"ep {common_ids[index]} / {stratum_names[index]}", fontsize=8)
    axes[0, 0].legend(fontsize=6, ncol=2); fig.suptitle("Fixed 12-episode command-trajectory examples (arm mean)")
    save_figure("fig07_trajectory_representatives", fig, data_rows, "The same 12 target-motion-stratified episodes are shown for every predeclared model; traces are arm-14 means in native command values.", "按低中高目标运动 tertile 固定抽取的 12 个 episode，展示真实轨迹和五个预先指定模型的 arm-14 平均 command trajectory。")

    # 8. Distribution.
    distribution_names = ["PERSISTENCE", "B-STATE", "B2-FUS-0125", "B3-FUS-D-F", "B3-FUS-D-L"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    arrays = [models[name]["metrics"]["arm_mae_std"] for name in distribution_names]
    ax.boxplot(arrays, tick_labels=distribution_names, showfliers=False); ax.set_ylabel("Per-episode standardized arm MAE"); ax.set_title("Episode-level error distributions"); ax.tick_params(axis="x", rotation=25)
    save_figure("fig08_error_distributions", fig, [{"model": name, "episode_id": int(common_ids[i]), "arm_mae_std": float(models[name]["metrics"]["arm_mae_std"][i])} for name in distribution_names for i in range(len(common_ids))], "Per-episode primary errors for fixed baseline and B2/B3 comparisons; box whiskers exclude plotted outliers but all episodes remain in statistics.", "五个固定比较模型的 episode-level 标准化 arm MAE 分布箱线图。")

    # 9. Lag scan.
    lag_names = ["PERSISTENCE", "B-STATE", "B3-FUS-D-L"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    data_rows = []
    for name in lag_names:
        vals = [lag_curves[name][lag] for lag in LAGS]
        ax.plot(LAGS, vals, marker="o", label=name, color=color_map.get(name))
        data_rows.extend({"model": name, "lag": lag, "std_arm_mae": float(lag_curves[name][lag])} for lag in LAGS)
    ax.axvline(0, color="black", linewidth=0.8); ax.set_xlabel("Integer lag (exploratory; zero is primary)"); ax.set_ylabel("Standardized arm MAE"); ax.set_title("Exploratory temporal alignment sensitivity"); ax.legend()
    save_figure("fig09_temporal_alignment", fig, data_rows, "Integer lag scan after synthetic validation; zero-lag values remain the primary endpoint.", "三个固定模型在 -5 到 +5 整数时间滞后下的 arm MAE；0 lag 是主分析，其余为探索性。")

    # 10. Task error heatmap relative to persistence.
    task_ids = sorted(np.unique(task).tolist())
    task_plot_names = ["B-STATE", "B2-FUS-0125", "B3-V-D-L", "B3-FUS-D-F", "B3-FUS-D-L"]
    task_lookup = {(row["model"], int(row["task_index"])): float(row["arm_mae_std"]) for row in per_task_rows}
    task_delta = np.asarray([[task_lookup[(name, task_id)] - task_lookup[("PERSISTENCE", task_id)] for task_id in task_ids] for name in task_plot_names])
    vmax = float(np.max(np.abs(task_delta)))
    fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
    im = ax.imshow(task_delta, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(task_plot_names)), task_plot_names); ax.set_xlabel("Original task index (sorted)"); ax.set_title("Per-task arm MAE relative to persistence"); fig.colorbar(im, ax=ax, label="Δ standardized arm MAE")
    save_figure("fig10_task_error_heatmap", fig, [{"model": name, "task_index": task_id, "delta_vs_persistence": float(task_delta[i, j])} for i, name in enumerate(task_plot_names) for j, task_id in enumerate(task_ids)], "Per-task primary error differences for fixed model comparisons; task indices are dataset indices, not semantic labels.", "固定模型相对 persistence 的逐任务 arm MAE 差异热图；任务编号为数据集索引，不代表已审计的物理语义。")

    write_csv(ROOT / "FIGURE_MANIFEST.csv", FIGURES)
    write_json(ROOT / "FIGURE_MANIFEST.json", {"figures": FIGURES, "visual_qa_status": "pending_manual_view_image", "count": len(FIGURES)})

    # Bilingual conclusions are generated from the independent primary table.
    best_formal = min(formal_names, key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    best_frozen = min([name for name in formal_names if models[name].get("adaptation") == "frozen"], key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    best_lora = min([name for name in formal_names if models[name].get("adaptation") == "lora"], key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    dynamic_formal = [name for name in formal_names if models[name].get("video_input") == "dynamic"]
    best_dynamic = min(dynamic_formal, key=lambda name: float(models[name]["metrics"]["arm_mae_std"].mean()))
    persistence_value = float(models["PERSISTENCE"]["metrics"]["arm_mae_std"].mean())
    conclusion_en = f"""# Core B Retrospective Benchmark v1 — conclusions

This is a CPU-only retrospective analysis of accepted development artifacts, not a preregistered or blinded study. The common population is {len(common_ids)} episodes with target shape (30, 28); final-test access is zero.

Descriptively, the lowest primary standardized arm-14 MAE among formal cells is **{best_formal}**; the lowest frozen formal cell is **{best_frozen}**, the lowest LoRA formal cell is **{best_lora}**, and the lowest dynamic formal cell is **{best_dynamic}**. The persistence reference has primary MAE {persistence_value:.6f}. These rankings are descriptive and do not establish mechanism or cross-seed stability.

True-pixel B2 remains a six-cell accepted matrix. The selected dense strategy and its fallback are reported as protocol context; the four contextual-token proxy cells are separate and excluded from the formal B2 conclusion. B3 is accepted as an 8/8 single-seed matrix, but the formal candidate gate has zero candidates, so this benchmark does not authorize multi-seed confirmation or final-test access.

Episode bootstrap, per-time/per-joint/per-task, motion-stratified, fixed-trajectory, and exploratory lag diagnostics are included. Physical command units, runtime efficiency, external validity, and cross-seed stability remain unresolved.
"""
    conclusion_zh = f"""# Core B 回顾性 Benchmark v1——结论

这是基于已验收开发集产物的 CPU-only 回顾性分析，不是 preregistered 或 blinded 研究。共同评估集为 {len(common_ids)} 个 episode，目标形状为 (30, 28)，final-test 访问为 0。

按主指标标准化 arm-14 MAE 的描述性排序，正式单元中最低的是 **{best_formal}**；frozen 正式单元最低的是 **{best_frozen}**，LoRA 正式单元最低的是 **{best_lora}**，dynamic 正式单元最低的是 **{best_dynamic}**。persistence 参考值为 {persistence_value:.6f}。这些只是描述性排序，不能证明机制或跨 seed 稳定性。

真实像素 B2 仍是 6 格完整矩阵；dense fallback 与策略选择作为协议背景单独报告，4 个 contextual-token proxy 不进入正式 B2 结论。B3 已形成 8/8 单 seed 矩阵，但 candidate gate 为 0，因此本 benchmark 不授权 multi-seed confirmation 或 final-test。

报告包含 episode bootstrap、逐时间/逐关节/逐任务、运动分层、固定轨迹样本和探索性 lag 诊断。command 的物理单位、运行效率、外部有效性和跨 seed 稳定性仍未解决。
"""
    (ROOT / "conclusions" / "CORE_B_RETROSPECTIVE_CONCLUSIONS.en.md").write_text(conclusion_en)
    (ROOT / "conclusions" / "CORE_B_RETROSPECTIVE_CONCLUSIONS.zh-CN.md").write_text(conclusion_zh)
    (ROOT / "conclusions" / "PROFESSOR_UPDATE.zh-CN.md").write_text(f"""# Core B 回顾性 Benchmark v1——教授更新\n\n- 范围：只读取已验收 B1 7/7、真实像素 B2 6/6、corrected B3 8/8；没有新训练、推理或 final-test。\n- 统计：共同开发集 {len(common_ids)} episodes，主指标为 episode-level standardized arm-14 MAE；10,000 次 paired bootstrap，逐时间/关节/任务和运动分层。\n- 结果边界：最佳正式单元的描述性标签为 `{best_formal}`，但 B3 candidate gate 仍为 0，且这是单 seed。\n- 代理边界：旧 contextual-token B2 proxy 单独归档，不混入真实像素 B2。\n- 未决：command units/URDF/FK、运行效率、跨 seed 和 final-test。\n""")
    (ROOT / "conclusions" / "TEAM_BRIEF.en.md").write_text("""# Team brief\n\nCore B Retrospective Benchmark v1 is a read-only, CPU-only evidence package. Use the formal B1/B2/B3 tables for current claims, keep proxy B2 separate, and do not describe the single-seed B3 result as a general or final model. Any future work must start from the validity scorecard and explicit gates.\n""")
    (ROOT / "README.en.md").write_text(f"""# Core B Retrospective Benchmark v1\n\nBenchmark root: `{BENCHMARK_ID}`. This is a CPU-only retrospective analysis of accepted B1/B2/B3 development predictions. It uses a common {len(common_ids)}-episode intersection, train-only normalization, episode-level primary MAE, paired 10,000-replicate bootstrap, fixed 12-episode trajectory examples, and explicit proxy/outcome-exposure boundaries.\n\nStatus at computation end: `analysis_complete_pending_visual_qa`. See `BENCHMARK_PROTOCOL_LOCKED.md`, `OUTCOME_EXPOSURE_STATEMENT.md`, `CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv`, `CORE_B_PAIRED_EFFECTS.csv`, `VALIDITY_SCORECARD.csv`, and `FIGURE_MANIFEST.md/json`.\n""")
    (ROOT / "README.zh-CN.md").write_text(f"""# Core B 回顾性 Benchmark v1\n\n目录 `{BENCHMARK_ID}` 是 CPU-only 的回顾性证据包，只读取已验收的 B1/B2/B3 开发集 prediction。共同评估集为 {len(common_ids)} 个 episode；使用 train-only normalization、episode-level 主指标、10,000 次 paired bootstrap、固定 12 个轨迹 episode，并显式隔离 proxy 和 outcome exposure。\n\n当前计算状态：`analysis_complete_pending_visual_qa`。优先查看 `BENCHMARK_PROTOCOL_LOCKED.md`、`OUTCOME_EXPOSURE_STATEMENT.md`、`CORE_B_RETROSPECTIVE_MAIN_RESULTS.csv`、`CORE_B_PAIRED_EFFECTS.csv`、`VALIDITY_SCORECARD.csv` 和 `FIGURE_MANIFEST.json`。\n""")
    write_json(ROOT / "BENCHMARK_STATUS.json", {"status": "analysis_complete_pending_visual_qa", "benchmark_id": BENCHMARK_ID, "created_utc": "2026-09-10T07:26:38Z", "completed_utc": utc_now(), "formal_coverage": {"B1": "7/7", "B2_true_pixel": "6/6", "B3": "8/8"}, "common_population_N": len(common_ids), "target_shape": [len(common_ids), 30, 28], "bootstrap_replicates": BOOTSTRAP_REPLICATES, "final_test_access": 0, "gpu_jobs_started_by_analysis": False, "b3_candidate_count": 0, "figure_count": len(FIGURES), "visual_qa": "pending_manual_view_image"})
    print(json.dumps({"root": str(ROOT), "formal_models": len(formal_names), "common_N": len(common_ids), "figures": len(FIGURES), "status": "analysis_complete_pending_visual_qa"}))


if __name__ == "__main__":
    main()
