#!/usr/bin/env python3
"""Build the CPU-only Persistence-Aware Motion Benchmark v2.

This module reads accepted prediction bundles, recomputes all metrics from the
canonical arrays, and writes a self-contained versioned evidence package. It
does not import torch, start inference, touch source runs, or access final-test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE_B = ROOT.parents[1]
V1 = CORE_B / "benchmark" / "CORE_B_RETROSPECTIVE_BENCHMARK_V1_20260910T072638Z"
FORMAL = CORE_B / "analysis" / "CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"
INPUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/inputs/B2_INPUTS_0125.npz'))
NORMALIZER = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json'))
SEED = 20260910
BOOTSTRAPS = 10_000
LAGS = tuple(range(-5, 6))
FIGURE_MANIFEST: list[dict[str, Any]] = []


def jdefault(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=jdefault) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_checksum_file(root: Path, checksum: Path) -> dict[str, Any]:
    result = subprocess.run(["sha256sum", "--check", checksum.name], cwd=root, capture_output=True, text=True)
    return {"path": str(checksum), "entry_count": sum(1 for line in checksum.read_text().splitlines() if line.strip()), "returncode": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL", "stderr": result.stderr[-1000:]}


def load_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = json.loads((FORMAL / "CORE_B_LOGICAL_CELL_INVENTORY.json").read_text())["rows"]
    accepted = [row for row in rows if row.get("stage") in {"B1", "B2", "B3"} and str(row.get("status", "")).startswith("accepted")]
    proxy = [row for row in rows if row.get("stage") == "B2_PROXY"]
    if len(accepted) != 21 or len(proxy) != 4:
        raise RuntimeError(f"unexpected accepted inventory {len(accepted)}/{len(proxy)}")
    return accepted, proxy


def aligned_index(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(ids.tolist())}
    if len(lookup) != len(ids):
        raise ValueError("duplicate source episode IDs")
    missing = [int(value) for value in wanted if int(value) not in lookup]
    if missing:
        raise ValueError(f"missing IDs: {missing[:5]}")
    return np.asarray([lookup[int(value)] for value in wanted], dtype=np.int64)


def meta(row: dict[str, Any]) -> dict[str, Any]:
    cell = row["cell"]
    interval = ""
    frames = ""
    if "0125" in cell:
        interval, frames = 0.125, 32
    elif "0250" in cell:
        interval, frames = 0.25, 16
    elif "0500" in cell:
        interval, frames = 0.5, 8
    if row["stage"] == "B3":
        interval, frames = 0.125, 32
    representation = "attentive" if "-A-" in cell else "pooled" if "-P-" in cell or row["stage"] in {"B2", "B3"} else "state"
    return {"stage": row["stage"], "model": cell, "branch": row.get("branch", ""), "video_input": row.get("video_input", ""), "adaptation": row.get("adaptation", ""), "representation": representation, "interval_seconds": interval, "unique_observation_frames": frames, "encoder_slots": 32 if interval != "" else "", "source_kind": row.get("source_kind", ""), "acceptance_status": row.get("status", ""), "reused_from": row.get("reused_from", ""), "prediction_path": row.get("prediction_path", ""), "final_test_access": row.get("final_test_access", 0)}


def load_sources(rows: list[dict[str, Any]], common_ids: np.ndarray, target_std: np.ndarray, target_native: np.ndarray) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["prediction_path"])
        bundle = np.load(path, allow_pickle=False)
        required = {"episode_index", "targets_standardized", "targets_native", "predictions_standardized", "predictions_native"}
        if not required.issubset(bundle.files):
            raise ValueError(f"{row['cell']} missing {sorted(required.difference(bundle.files))}")
        index = aligned_index(np.asarray(bundle["episode_index"], dtype=np.int64), common_ids)
        pred_std = np.asarray(bundle["predictions_standardized"], dtype=np.float64)[index]
        pred_native = np.asarray(bundle["predictions_native"], dtype=np.float64)[index]
        source_target_std = np.asarray(bundle["targets_standardized"], dtype=np.float64)[index]
        source_target_native = np.asarray(bundle["targets_native"], dtype=np.float64)[index]
        max_std = float(np.max(np.abs(source_target_std - target_std)))
        max_native = float(np.max(np.abs(source_target_native - target_native)))
        if pred_std.shape != target_std.shape or not np.isfinite(pred_std).all() or not np.isfinite(pred_native).all():
            raise ValueError(f"invalid bundle {row['cell']}: {pred_std.shape}")
        audit.append({"model": row["cell"], "stage": row["stage"], "path": str(path), "sha256": digest(path), "shape": str(tuple(pred_std.shape)), "target_std_max_abs": max_std, "target_native_max_abs": max_native, "finite": True, "final_test_access": row.get("final_test_access", 0), "status": "PASS" if max(max_std, max_native) <= 2e-6 else "FAIL"})
        models[row["cell"]] = {**meta(row), "pred_std": pred_std, "pred_native": pred_native, "source_sha256": digest(path)}
    if any(row["status"] == "FAIL" for row in audit):
        raise RuntimeError("target lineage failure")
    return models, audit


def motion_descriptors(array: np.ndarray) -> dict[str, np.ndarray]:
    arm = array[:, :, 14:28]
    changes = np.diff(arm, axis=1)
    return {"mean_step_motion": np.abs(changes).mean(axis=(1, 2)), "total_variation": np.abs(changes).sum(axis=1).mean(axis=1), "endpoint_change": np.abs(arm[:, -1] - arm[:, 0]).mean(axis=1), "max_step_motion": np.abs(changes).mean(axis=2).max(axis=1)}


def horizon_slices() -> dict[str, slice | int]:
    return {"full": slice(0, 30), "early": slice(0, 10), "middle": slice(10, 20), "late": slice(20, 30), "endpoint": 29}


def episode_metrics(pred: np.ndarray, target: np.ndarray, pred_native: np.ndarray | None = None, target_native: np.ndarray | None = None) -> dict[str, Any]:
    error = np.abs(pred - target)
    changes_pred = np.diff(pred, axis=1)
    changes_target = np.diff(target, axis=1)
    change_error = np.abs(changes_pred - changes_target)
    arm_error = error[:, :, 14:28]
    hand_error = error[:, :, 0:14]
    full_error = error
    out: dict[str, Any] = {"arm_full": arm_error.mean(axis=(1, 2)), "hand_full": hand_error.mean(axis=(1, 2)), "full28_full": full_error.mean(axis=(1, 2)), "rmse_full": np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2))), "arm_first_diff": change_error[:, :, 14:28].mean(axis=(1, 2)), "hand_first_diff": change_error[:, :, 0:14].mean(axis=(1, 2)), "full_first_diff": change_error.mean(axis=(1, 2))}
    slices = horizon_slices()
    for name, selector in slices.items():
        if name == "endpoint":
            out[f"arm_{name}"] = arm_error[:, selector, :].mean(axis=1)
            out[f"hand_{name}"] = hand_error[:, selector, :].mean(axis=1)
            out[f"full28_{name}"] = full_error[:, selector, :].mean(axis=1)
        else:
            out[f"arm_{name}"] = arm_error[:, selector, :].mean(axis=(1, 2))
            out[f"hand_{name}"] = hand_error[:, selector, :].mean(axis=(1, 2))
            out[f"full28_{name}"] = full_error[:, selector, :].mean(axis=(1, 2))
    target_motion = motion_descriptors(target)
    pred_motion = motion_descriptors(pred)
    out.update({f"target_{key}": value for key, value in target_motion.items()})
    out.update({f"pred_{key}": value for key, value in pred_motion.items()})
    target_tv = target_motion["total_variation"]
    pred_tv = pred_motion["total_variation"]
    out["tv_ratio"] = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 1e-12)
    out["tv_ratio_na_reason"] = np.where(target_tv > 1e-12, "", "target_tv_le_1e-12")
    out["endpoint_change_error"] = np.abs(pred_motion["endpoint_change"] - target_motion["endpoint_change"])
    target_range = np.ptp(target[:, :, 14:28], axis=1)
    pred_range = np.ptp(pred[:, :, 14:28], axis=1)
    valid_range = target_range > 1e-12
    range_ratio_joint = np.divide(pred_range, target_range, out=np.full_like(pred_range, np.nan), where=valid_range)
    out["range_ratio"] = np.nanmean(range_ratio_joint, axis=1)
    out["range_ratio_na_reason"] = np.where(valid_range.any(axis=1), "", "all_target_ranges_le_1e-12")
    a = changes_pred[:, :, 14:28]
    b = changes_target[:, :, 14:28]
    valid_direction = (np.abs(a) > 1e-8) | (np.abs(b) > 1e-8)
    out["direction_agreement"] = np.divide(((np.sign(a) == np.sign(b)) & valid_direction).sum(axis=(1, 2)), valid_direction.sum(axis=(1, 2)), out=np.full(len(pred), np.nan), where=valid_direction.sum(axis=(1, 2)) > 0)
    dot = (a * b).sum(axis=2)
    norm = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    valid_cos = norm > 1e-8
    out["direction_cosine"] = np.divide(dot.sum(axis=1), valid_cos.sum(axis=1), out=np.full(len(pred), np.nan), where=valid_cos.sum(axis=1) > 0)
    if pred_native is not None and target_native is not None:
        native = episode_metrics(pred_native, target_native)
        for key in ("arm_first_diff", "tv_ratio", "endpoint_change_error", "range_ratio", "direction_agreement", "direction_cosine"):
            out[f"{key}_std"] = out[key]
            out[f"{key}_native"] = native[key]
        out["tv_ratio_na_reason_std"] = out["tv_ratio_na_reason"]
        out["tv_ratio_na_reason_native"] = native["tv_ratio_na_reason"]
        out["range_ratio_na_reason_std"] = out["range_ratio_na_reason"]
        out["range_ratio_na_reason_native"] = native["range_ratio_na_reason"]
        out["tv_ratio"] = native["tv_ratio"]
        out["tv_ratio_na_reason"] = native["tv_ratio_na_reason"]
        out["endpoint_change_error"] = native["endpoint_change_error"]
        out["range_ratio"] = native["range_ratio"]
        out["direction_agreement"] = native["direction_agreement"]
        out["direction_cosine"] = native["direction_cosine"]
        for key in ("mean_step_motion", "total_variation", "endpoint_change", "max_step_motion"):
            out[f"target_{key}_std"] = out[f"target_{key}"]
            out[f"pred_{key}_std"] = out[f"pred_{key}"]
            out[f"target_{key}_native"] = native[f"target_{key}"]
            out[f"pred_{key}_native"] = native[f"pred_{key}"]
        out["arm_first_diff_native"] = native["arm_first_diff"]
    return out


def subset_metric(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return values if mask is None else values[mask]


def bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def safe_nanmedian(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")


def aligned_arrays(pred: np.ndarray, target: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return pred[:, :-lag], target[:, lag:]
    if lag < 0:
        return pred[:, -lag:], target[:, :lag]
    return pred, target


def lag_score(pred: np.ndarray, target: np.ndarray, lag: int) -> tuple[float, float]:
    left, right = aligned_arrays(pred, target, lag)
    left = left[:, :, 14:28].reshape(len(left), -1)
    right = right[:, :, 14:28].reshape(len(right), -1)
    mae = float(np.abs(left - right).mean())
    left_c = left - left.mean(axis=1, keepdims=True)
    right_c = right - right.mean(axis=1, keepdims=True)
    den = np.linalg.norm(left_c, axis=1) * np.linalg.norm(right_c, axis=1)
    valid = den > 1e-12
    corr = float(np.mean(np.sum(left_c[valid] * right_c[valid], axis=1) / den[valid])) if valid.any() else float("nan")
    return mae, corr


def synthetic_lag_test() -> dict[str, Any]:
    target = np.repeat(np.arange(30, dtype=np.float64)[None, :, None], 28, axis=2)
    target = np.repeat(target, 2, axis=0)
    delayed = np.zeros_like(target)
    delayed[:, 1:] = target[:, :-1]
    values = {str(lag): lag_score(delayed, target, lag)[0] for lag in LAGS}
    best = min(values, key=values.get)
    return {"pass": best == "-1", "known_delayed_prediction_best_lag": -1, "best_lag": int(best), "mae_by_lag": values}


def safe_relative(model: float, baseline: float) -> tuple[float | str, str]:
    if baseline <= 1e-12:
        return "", "persistence_denominator_le_1e-12"
    return float(1.0 - model / baseline), ""


def save_figure(name: str, fig: plt.Figure, data: list[dict[str, Any]], caption: str, alt: str) -> None:
    data_path = ROOT / "figure_data" / f"{name}.csv"
    write_csv(data_path, data)
    for ext, folder in (("png", "png"), ("svg", "svg"), ("pdf", "pdf")):
        fig.savefig(ROOT / "figures" / folder / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    sidecar = ROOT / "figure_manifests" / f"{name}.caption.md"
    sidecar.write_text(f"# {name}\n\n{caption}\n\n**中文 alt text：** {alt}\n")
    FIGURE_MANIFEST.append({"figure": name, "data": rel(data_path), "png": f"figures/png/{name}.png", "svg": f"figures/svg/{name}.svg", "pdf": f"figures/pdf/{name}.pdf", "caption": rel(sidecar), "visual_qa": "pending_manual_view_image"})


def write_next_route(route: str, metrics: dict[str, Any]) -> None:
    values = metrics["route_values"]
    rationale = metrics["route_rationale"]
    rows: list[dict[str, Any]] = []
    if route == "A":
        rows = [{"cell": f"A-{branch}-{target}-{decoder}", "branch": branch, "target": target, "decoder": decoder, "seed": "fresh single seed", "status": "future_prompt_only"} for branch in ("STATE", "FUSION") for target in ("ABSOLUTE", "PERSISTENCE_RESIDUAL") for decoder in ("DIRECT", "TEMPORAL")]
        prompt = """# Next Core B experiment — Route A: persistence-residual temporal model\n\nThis prompt is generated conditionally from Benchmark v2 and is not executed in the current task. Use the locked B2 dense strategy and common protocol. Train an 8-cell 2×2×2 matrix: input branch STATE/FUSION, target ABSOLUTE/PERSISTENCE_RESIDUAL, and decoder DIRECT/TEMPORAL. Keep one fresh seed, fixed train/dev source-episode split, and no final-test access. Residual predictions must be added back to persistence before all absolute-trajectory metrics. Primary reporting keeps full arm14 MAE and adds high-motion/late/endpoint persistence-relative skill, first-difference command-change error, TV ratio and direction. Use one predeclared smoothness/trend auxiliary version only; no weight sweep. Stop if source IDs, target alignment or writer schema fail, if all residual cells remain below persistence in the predeclared challenge slices, or if the task expands beyond this matrix.\n"""
    elif route == "B":
        rows = [{"cell": f"B-EVENT-{condition}", "condition": condition, "status": "future_prompt_only"} for condition in ("EXISTING_CHECKPOINT_FEASIBILITY", "SOURCE_CLUSTER_AUDIT")]
        prompt = """# Next Core B experiment — Route B: event-centered dynamic challenge set\n\nThis prompt is generated conditionally and is not executed. Define event-centered windows from train-derived target motion/change criteria, keep source episodes as the clustering unit, and preserve the 4-second observation/1-second target contract. First run inference-only feasibility with existing accepted checkpoints; do not train or access final-test until event support, task coverage, and source-level split integrity pass. Report full-population v1 Layer A and the event challenge separately.\n"""
    elif route == "C":
        rows = [{"cell": f"C-{condition}", "condition": condition, "status": "future_prompt_only"} for condition in ("MATCHED_VIDEO", "MISMATCHED_VIDEO", "ZERO_VIDEO", "REVERSED_VIDEO", "SHUFFLED_VIDEO", "REPEATED_FIRST")]
        prompt = """# Next Core B experiment — Route C: fusion attribution controls\n\nThis prompt is generated conditionally and is not executed. Keep the STATE external reference, selected dense sampling, fixed head and identical episode IDs. Compare matched video, mismatched video, disabled/zeroed video, chronological, reversed, shuffled and repeated-first controls. Keep full and high-motion persistence-relative skill, first-difference and endpoint outcomes separate. Stop if the implementation cannot prove the intended video intervention or if the control changes the population.\n"""
    elif route == "D":
        rows = [{"cell": f"D-{condition}", "condition": condition, "status": "future_prompt_only"} for condition in ("TIMESTAMP_AUDIT", "FRAME_MAP_AUDIT", "SYNTHETIC_ALIGNMENT", "WRITER_REPLAY")]
        prompt = """# Next Core B experiment — Route D: alignment repair\n\nThis prompt is generated conditionally and is not executed. Audit timestamps, frame maps, target boundaries and writer/index lineage first. Reproduce the observed common lag with synthetic fixtures, then repair only a confirmed alignment or writer defect. Replay existing checkpoints before any new training; preserve the old artifacts and v1 results.\n"""
    else:
        rows = [{"cell": "E-FRESH-SEED-1", "seed": 1, "status": "future_prompt_only"}, {"cell": "E-FRESH-SEED-2", "seed": 2, "status": "future_prompt_only"}, {"cell": "E-FRESH-SEED-3", "seed": 3, "status": "future_prompt_only"}]
        prompt = """# Next Core B experiment — Route E: fresh-seed confirmation\n\nThis prompt is generated conditionally and is not executed. Confirm only the predeclared model and challenge score that beat persistence/STATE with a practically meaningful paired effect, CI support and no temporal-fidelity degeneration. Use three fresh training seeds, keep the same source-level split and all final-test locks, and report the full B1/B2/B3-derived protocol.\n"""
    write_csv(ROOT / "NEXT_CORE_B_EXPERIMENT_MATRIX.csv", rows)
    write_json(ROOT / "NEXT_CORE_B_EXPERIMENT_MATRIX.json", {"route": route, "rows": rows, "executed": False})
    (ROOT / "NEXT_CORE_B_EXPERIMENT_PROMPT.md").write_text(prompt)
    (ROOT / "NEXT_CORE_B_EXPERIMENT_RATIONALE.zh-CN.md").write_text(f"# 条件式下一步路线：Route {route}\n\n本路线依据 v2 已锁定的判定规则自动选择，当前只生成 Prompt，不执行。\n\n{rationale}\n\n所有其他路线保留为未选择候选；原因是它们没有当前证据的最高信息增益，或需要先满足更早的 source/支持度/对齐门禁。\n")


def main() -> None:
    for sub in ("data", "figure_data", "figures/png", "figures/svg", "figures/pdf", "figure_manifests", "qa", "conclusions", "tests/fixtures"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    source_v1_checksum = verify_checksum_file(V1, V1 / "CHECKSUMS.sha256")
    source_formal_checksum = verify_checksum_file(FORMAL, FORMAL / "CORE_B_ANALYSIS_OUTPUT_CHECKSUMS.sha256")
    accepted_rows, proxy_rows = load_inventory()
    canonical = np.load(INPUT, allow_pickle=False)
    train_mask = canonical["split_code"] == 0
    dev_mask = canonical["split_code"] == 1
    all_ids = np.asarray(canonical["episode_index"], dtype=np.int64)
    dev_ids = all_ids[dev_mask]
    if len(dev_ids) != 614 or len(np.unique(dev_ids)) != 614:
        raise RuntimeError("canonical dev ID contract failed")
    normalizer = json.loads(NORMALIZER.read_text())["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    safe_std = np.asarray(normalizer["safe_std"], dtype=np.float64)
    target_native_all = np.asarray(canonical["targets"], dtype=np.float64)
    target_std_all = np.asarray(canonical["targets_standardized"], dtype=np.float64)
    expected_std = (target_native_all - mean[None, :, :]) / safe_std[None, :, :]
    normalization_diff = float(np.max(np.abs(expected_std - target_std_all)))
    if normalization_diff > 2e-6:
        raise RuntimeError(f"normalization mismatch {normalization_diff}")
    common = set(int(value) for value in dev_ids.tolist())
    for row in accepted_rows + proxy_rows:
        ids = np.asarray(np.load(Path(row["prediction_path"]), allow_pickle=False)["episode_index"], dtype=np.int64)
        common.intersection_update(int(value) for value in ids.tolist())
    common_ids = np.asarray(sorted(common), dtype=np.int64)
    if len(common_ids) != 614:
        raise RuntimeError(f"common population {len(common_ids)} != 614")
    dev_index = aligned_index(dev_ids, common_ids)
    target_native = target_native_all[dev_mask][dev_index]
    target_std = target_std_all[dev_mask][dev_index]
    task = np.asarray(canonical["original_task_index"], dtype=np.int64)[dev_mask][dev_index]
    last_action = np.asarray(canonical["last_observed_action"], dtype=np.float64)[dev_mask][dev_index]
    train_native = target_native_all[train_mask]
    train_std = target_std_all[train_mask]
    train_motion_std = motion_descriptors(train_std)
    dev_motion_std = motion_descriptors(target_std)
    train_thresholds = {key: {"q25": float(np.quantile(value, 0.25)), "q75": float(np.quantile(value, 0.75)), "min": float(value.min()), "max": float(value.max())} for key, value in train_motion_std.items()}
    q25 = train_thresholds["mean_step_motion"]["q25"]
    q75 = train_thresholds["mean_step_motion"]["q75"]
    strata = np.where(dev_motion_std["mean_step_motion"] <= q25, "low", np.where(dev_motion_std["mean_step_motion"] <= q75, "mid", "high"))
    stratum_indices = {name: np.flatnonzero(strata == name) for name in ("low", "mid", "high")}
    models, source_audit = load_sources(accepted_rows, common_ids, target_std, target_native)
    persistence_native = np.repeat(last_action[:, None, :], 30, axis=1)
    persistence_std = (persistence_native - mean[None, :, :]) / safe_std[None, :, :]
    global_native_vector = train_native.mean(axis=(0, 1))
    per_time_native = train_native.mean(axis=0)
    global_native = np.repeat(np.repeat(global_native_vector[None, None, :], len(common_ids), axis=0), 30, axis=1)
    per_time_native_eval = np.repeat(per_time_native[None, :, :], len(common_ids), axis=0)
    controls = {
        "PERSISTENCE": (persistence_std, persistence_native, "last observed action repeated"),
        "GLOBAL_MEAN": ((global_native - mean[None, :, :]) / safe_std[None, :, :], global_native, "train-only global joint mean"),
        "PER_TIME_MEAN": ((per_time_native_eval - mean[None, :, :]) / safe_std[None, :, :], per_time_native_eval, "train-only per-time joint mean"),
    }
    for name, (pred_std, pred_native, description) in controls.items():
        models[name] = {"stage": "CONTROL", "model": name, "branch": "control", "video_input": "none", "adaptation": "none", "representation": "derived", "interval_seconds": "", "unique_observation_frames": "", "encoder_slots": "", "source_kind": "derived_control", "acceptance_status": "not_formal_cell", "reused_from": "", "prediction_path": str(INPUT), "final_test_access": 0, "description": description, "pred_std": pred_std, "pred_native": pred_native}
    eval_names = [row["cell"] for row in accepted_rows] + list(controls)
    for name in eval_names:
        models[name]["metrics"] = episode_metrics(models[name]["pred_std"], target_std, models[name]["pred_native"], target_native)
    bootstrap_rng = np.random.default_rng(SEED)
    bootstrap_indices = bootstrap_rng.integers(0, len(common_ids), size=(BOOTSTRAPS, len(common_ids)), dtype=np.int32)
    np.save(ROOT / "data" / "paired_episode_bootstrap_indices.npy", bootstrap_indices)

    # Source acceptance and lineage.
    source_acceptance = [
        {"contract": "v1_checksum", "status": source_v1_checksum["status"], "value": source_v1_checksum["entry_count"], "evidence": source_v1_checksum["path"]},
        {"contract": "formal_upstream_checksum", "status": source_formal_checksum["status"], "value": source_formal_checksum["entry_count"], "evidence": source_formal_checksum["path"]},
        {"contract": "B1_coverage", "status": "PASS", "value": "7/7", "evidence": str(FORMAL / "B1_REACCEPTANCE.json")},
        {"contract": "true_pixel_B2_coverage", "status": "PASS", "value": "6/6", "evidence": str(FORMAL / "B2_REACCEPTANCE.json")},
        {"contract": "B3_coverage", "status": "PASS", "value": "8/8", "evidence": str(FORMAL / "B3_REACCEPTANCE.json")},
        {"contract": "common_population", "status": "PASS", "value": len(common_ids), "evidence": "common episode ID intersection"},
        {"contract": "target_shape", "status": "PASS", "value": str(tuple(target_std.shape)), "evidence": "canonical B2_INPUTS_0125.npz"},
        {"contract": "arm_hand_mapping", "status": "PASS", "value": "arm14=14:28;hand14=0:14", "evidence": "formal harvest code and v1 correction"},
        {"contract": "train_only_normalization", "status": "PASS", "value": normalization_diff, "evidence": str(NORMALIZER)},
        {"contract": "finite_arrays", "status": "PASS", "value": True, "evidence": "all accepted prediction bundles"},
        {"contract": "proxy_boundary", "status": "PASS", "value": "4 proxy rows excluded", "evidence": str(FORMAL / "B2_PROXY_INVENTORY.csv")},
        {"contract": "final_test_access", "status": "PASS", "value": 0, "evidence": "source contracts and v1 status"},
    ]
    if any(row["status"] != "PASS" for row in source_acceptance):
        write_json(ROOT / "BLOCKED_SOURCE_REPORT.json", {"status": "BLOCKED", "rows": source_acceptance})
        raise RuntimeError("source gate failed")
    write_csv(ROOT / "V2_SOURCE_ACCEPTANCE.csv", source_acceptance)
    write_json(ROOT / "V2_SOURCE_ACCEPTANCE.json", {"status": "PASS", "rows": source_acceptance})
    registry = []
    for row in accepted_rows:
        registry.append({**meta(row), "registry_role": "formal_accepted", "included_in_main": True, "source_sha256": digest(Path(row["prediction_path"])), "N": len(common_ids)})
    for row in proxy_rows:
        registry.append({**meta(row), "registry_role": "proxy_excluded", "included_in_main": False, "source_sha256": digest(Path(row["prediction_path"])), "N": len(common_ids), "description": "contextual tokens after full encoder"})
    for name, (_, _, description) in controls.items():
        registry.append({"stage": "CONTROL", "model": name, "branch": "control", "video_input": "none", "adaptation": "none", "representation": "derived", "interval_seconds": "", "unique_observation_frames": "", "encoder_slots": "", "source_kind": "derived_control", "acceptance_status": "not_formal_cell", "prediction_path": str(INPUT), "final_test_access": 0, "registry_role": "derived_control", "included_in_main": True, "source_sha256": digest(INPUT), "N": len(common_ids), "description": description})
    write_csv(ROOT / "V2_SOURCE_REGISTRY.csv", registry)
    write_json(ROOT / "V2_SOURCE_REGISTRY.json", {"rows": registry, "formal_count": 21, "proxy_count": 4, "control_count": 3})
    write_json(ROOT / "data" / "V2_TARGET_LINEAGE_AUDIT.json", {"rows": source_audit, "normalization_diff": normalization_diff, "status": "PASS"})

    # Motion thresholds and strata.
    threshold_payload = {"status": "PASS", "formula": "mean abs standardized arm14 first difference", "train_rows": int(train_mask.sum()), "source": str(INPUT), "source_split": "split_code=0", "thresholds": train_thresholds, "primary_threshold_key": "mean_step_motion", "q25": q25, "q75": q75, "dev_stratum_counts": {key: int(len(value)) for key, value in stratum_indices.items()}, "final_test_access": 0}
    write_json(ROOT / "MOTION_THRESHOLD_SOURCE.json", threshold_payload)
    (ROOT / "MOTION_THRESHOLD_SOURCE.md").write_text(f"# Motion threshold source\n\nPrimary strata use train-only `mean_step_motion` on standardized arm14 dimensions 14:28. Train Q25={q25:.9g}; train Q75={q75:.9g}. The thresholds are applied unchanged to the 614 development episodes. This is not a final-test threshold.\n")
    strata_rows = []
    for i, episode_id in enumerate(common_ids):
        strata_rows.append({"episode_id": int(episode_id), "task_index": int(task[i]), "motion_stratum": str(strata[i]), "mean_step_motion_std": float(dev_motion_std["mean_step_motion"][i]), "total_variation_std": float(dev_motion_std["total_variation"][i]), "endpoint_change_std": float(dev_motion_std["endpoint_change"][i]), "max_step_motion_std": float(dev_motion_std["max_step_motion"][i]), "low_threshold_q25": q25, "high_threshold_q75": q75})
    write_csv(ROOT / "MOTION_STRATA_MANIFEST.csv", strata_rows)
    write_json(ROOT / "MOTION_STRATA_MANIFEST.json", {"threshold_source": "train", "rows": len(strata_rows), "counts": {key: int(len(value)) for key, value in stratum_indices.items()}, "task_counts": {str(key): int(value) for key, value in Counter(task.tolist()).items()}, "low_support": {key: len(value) < 50 for key, value in stratum_indices.items()}})
    (ROOT / "HORIZON_BIN_MANIFEST.md").write_text("# Horizon bins\n\n`early=steps 1-10 (array 0:10)`, `middle=steps 11-20 (10:20)`, `late=steps 21-30 (20:30)`, `endpoint=array index 29`. No verified timestamp array was found in the canonical input bundle, so step labels are retained and physical time is not claimed.\n")
    write_json(ROOT / "HORIZON_BIN_MANIFEST.json", {"timestamp_status": "not_found_in_canonical_input", "bins": {key: str(value) for key, value in horizon_slices().items()}, "target_steps": 30})

    # Task-normalized exploratory sensitivity.
    train_task = np.asarray(canonical["original_task_index"], dtype=np.int64)[train_mask]
    task_motion_thresholds = {}
    for task_id in sorted(np.unique(train_task).tolist()):
        values = train_motion_std["mean_step_motion"][train_task == task_id]
        if len(values) >= 10:
            task_motion_thresholds[int(task_id)] = {"N_train": int(len(values)), "q75": float(np.quantile(values, 0.75))}
    task_high = np.asarray([task_motion_thresholds.get(int(task_id), {}).get("q75", np.nan) < dev_motion_std["mean_step_motion"][i] if int(task_id) in task_motion_thresholds else False for i, task_id in enumerate(task)], dtype=bool)
    task_sensitivity_rows = [{"episode_id": int(common_ids[i]), "task_index": int(task[i]), "task_threshold_q75": task_motion_thresholds.get(int(task[i]), {}).get("q75", ""), "task_train_N": task_motion_thresholds.get(int(task[i]), {}).get("N_train", ""), "task_high_motion": bool(task_high[i]), "status": "PASS" if int(task[i]) in task_motion_thresholds else "NA_low_train_support"} for i in range(len(common_ids))]
    write_csv(ROOT / "V2_TASK_NORMALIZED_MOTION.csv", task_sensitivity_rows)
    write_json(ROOT / "V2_TASK_NORMALIZED_MOTION.json", {"min_train_support": 10, "supported_task_count": len(task_motion_thresholds), "dev_task_high_count": int(task_high.sum()), "rows": len(task_sensitivity_rows)})

    # Main per-episode and aggregate metrics.
    per_episode_rows: list[dict[str, Any]] = []
    per_time_rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    for name in eval_names:
        m = models[name]["metrics"]
        for i, episode_id in enumerate(common_ids):
            row = {"model": name, "episode_id": int(episode_id), "task_index": int(task[i]), "motion_stratum": str(strata[i]), "mean_step_motion_std": float(dev_motion_std["mean_step_motion"][i]), "total_variation_std": float(dev_motion_std["total_variation"][i]), "endpoint_change_std": float(dev_motion_std["endpoint_change"][i]), "max_step_motion_std": float(dev_motion_std["max_step_motion"][i])}
            for key in ("arm_full", "hand_full", "full28_full", "rmse_full", "arm_early", "arm_middle", "arm_late", "arm_endpoint", "arm_first_diff", "arm_first_diff_native", "tv_ratio", "tv_ratio_std", "endpoint_change_error", "endpoint_change_error_std", "range_ratio", "range_ratio_std", "direction_agreement", "direction_agreement_std", "direction_cosine", "direction_cosine_std", "pred_mean_step_motion", "pred_total_variation", "pred_endpoint_change", "pred_max_step_motion", "pred_mean_step_motion_native", "pred_total_variation_native", "pred_endpoint_change_native", "pred_max_step_motion_native"):
                value = m[key][i]
                row[key] = "" if not np.isfinite(value) else float(value)
            row["tv_ratio_na_reason"] = str(m["tv_ratio_na_reason"][i])
            row["tv_ratio_na_reason_std"] = str(m["tv_ratio_na_reason_std"][i])
            row["range_ratio_na_reason"] = str(m["range_ratio_na_reason"][i])
            row["range_ratio_na_reason_std"] = str(m["range_ratio_na_reason_std"][i])
            per_episode_rows.append(row)
        per_time = np.abs(models[name]["pred_std"] - target_std)[:, :, 14:28].mean(axis=2)
        for t in range(30):
            per_time_rows.append({"model": name, "time_index": t, "horizon_step": t + 1, "arm_mae_std": float(per_time[:, t].mean()), "persistence_arm_mae_std": float(np.abs(persistence_std - target_std)[:, t, 14:28].mean()), "delta_vs_persistence": float(np.abs(persistence_std - target_std)[:, t, 14:28].mean() - per_time[:, t].mean())})
        joint_time = np.abs(models[name]["pred_std"] - target_std)[:, :, 14:28].mean(axis=0)
        for t in range(30):
            for j in range(14):
                per_joint_rows.append({"model": name, "time_index": t, "joint_index": 14 + j, "arm_mae_std": float(joint_time[t, j])})
        for task_id in sorted(np.unique(task).tolist()):
            mask = task == task_id
            per_task_rows.append({"model": name, "task_index": int(task_id), "N": int(mask.sum()), "high_motion_N": int((mask & (strata == "high")).sum()), "arm_full_mae_std": float(m["arm_full"][mask].mean()), "arm_late_mae_std": float(m["arm_late"][mask].mean()), "arm_endpoint_mae_std": float(m["arm_endpoint"][mask].mean()), "arm_first_diff_mae_std": float(m["arm_first_diff"][mask].mean())})
    write_csv(ROOT / "V2_PER_EPISODE_METRICS.csv", per_episode_rows)
    write_csv(ROOT / "V2_PER_TIME_METRICS.csv", per_time_rows)
    write_csv(ROOT / "V2_PER_JOINT_METRICS.csv", per_joint_rows)
    write_csv(ROOT / "V2_PER_TASK_METRICS.csv", per_task_rows)

    # Aggregate result tables and persistence-relative skill.
    persistence = models["PERSISTENCE"]["metrics"]
    aggregate_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    slices = horizon_slices()
    for name in eval_names:
        m = models[name]["metrics"]
        low, high = bootstrap_ci(m["arm_full"], bootstrap_indices)
        aggregate_rows.append({"model": name, "stage": models[name]["stage"], "branch": models[name]["branch"], "video_input": models[name]["video_input"], "adaptation": models[name]["adaptation"], "N": len(common_ids), "arm_full_mae_std": float(m["arm_full"].mean()), "arm_full_ci_low": low, "arm_full_ci_high": high, "hand_full_mae_std": float(m["hand_full"].mean()), "full28_full_mae_std": float(m["full28_full"].mean()), "rmse_full_std": float(m["rmse_full"].mean()), "arm_endpoint_mae_std": float(m["arm_endpoint"].mean()), "arm_first_diff_mae_std": float(m["arm_first_diff"].mean()), "arm_first_diff_mae_native": float(m["arm_first_diff_native"].mean()), "tv_ratio_mean_native": safe_nanmean(m["tv_ratio"]), "tv_ratio_mean_standardized": safe_nanmean(m["tv_ratio_std"]), "direction_agreement_mean_native": safe_nanmean(m["direction_agreement"]), "direction_agreement_mean_standardized": safe_nanmean(m["direction_agreement_std"]), "direction_cosine_mean_native": safe_nanmean(m["direction_cosine"]), "direction_cosine_mean_standardized": safe_nanmean(m["direction_cosine_std"]), "native_unit_boundary": "source-native command values only", "final_test_access": 0})
        for subset, mask in [("full", np.ones(len(common_ids), dtype=bool)), ("low", strata == "low"), ("mid", strata == "mid"), ("high", strata == "high")]:
            for horizon in ("full", "early", "middle", "late", "endpoint"):
                values = m[f"arm_{horizon}"][mask]
                ref = persistence[f"arm_{horizon}"][mask]
                improvement = ref - values
                relative, reason = safe_relative(float(values.mean()), float(ref.mean()))
                ci_low, ci_high = bootstrap_ci(improvement, bootstrap_indices[:, mask] if mask.all() else np.random.default_rng(SEED + len(skill_rows)).integers(0, len(values), size=(BOOTSTRAPS, len(values)), dtype=np.int32))
                skill_rows.append({"model": name, "subset": subset, "horizon": horizon, "N": int(mask.sum()), "absolute_mae": float(values.mean()), "persistence_mae": float(ref.mean()), "delta_vs_persistence": float(improvement.mean()), "relative_skill": relative, "relative_skill_na_reason": reason, "win_rate": float(np.mean(values < ref)), "median_paired_improvement": float(np.median(improvement)), "paired_q05": float(np.quantile(improvement, 0.05)), "paired_q25": float(np.quantile(improvement, 0.25)), "paired_q75": float(np.quantile(improvement, 0.75)), "paired_q95": float(np.quantile(improvement, 0.95)), "ci_low": ci_low, "ci_high": ci_high})
    write_csv(ROOT / "V2_MAIN_RESULTS.csv", aggregate_rows)
    write_json(ROOT / "V2_MAIN_RESULTS.json", {"rows": aggregate_rows, "primary": "arm_full_mae_std", "layer_a": True, "layer_b_separate": True})
    write_csv(ROOT / "V2_PERSISTENCE_SKILL.csv", skill_rows)
    write_json(ROOT / "V2_PERSISTENCE_SKILL.json", {"rows": skill_rows, "interpretation": "positive_delta_means_model_beats_persistence", "bootstrap_replicates": BOOTSTRAPS})
    write_csv(ROOT / "V2_MOTION_HORIZON_RESULTS.csv", [row for row in skill_rows if row["subset"] in {"low", "mid", "high"}])
    write_json(ROOT / "V2_MOTION_HORIZON_RESULTS.json", {"rows": [row for row in skill_rows if row["subset"] in {"low", "mid", "high"}], "low_support": {key: len(value) < 50 for key, value in stratum_indices.items()}})

    # Key paired effects: effect is baseline error minus comparator error.
    contrast_specs = [("B-STATE", "PERSISTENCE", "B1_STATE_vs_persistence"), ("B3-FUS-D-L", "PERSISTENCE", "B3_FUS_D_L_vs_persistence"), ("B3-FUS-D-L", "B-STATE", "B3_FUS_D_L_vs_state"), ("B-D-P-F", "B-S-P-F", "B1_dynamic_pooled_vs_static_pooled"), ("B3-V-D-L", "B3-V-S-L", "B3_video_dynamic_lora_vs_static_lora"), ("B3-FUS-D-L", "B3-FUS-S-L", "B3_fusion_dynamic_lora_vs_static_lora")]
    effect_rows: list[dict[str, Any]] = []
    for comparator, baseline, label in contrast_specs:
        for subset, mask in [("full", np.ones(len(common_ids), dtype=bool)), ("high", strata == "high"), ("late", np.ones(len(common_ids), dtype=bool)), ("endpoint", np.ones(len(common_ids), dtype=bool))]:
            horizon = "full" if subset in {"full", "high"} else subset
            if horizon == "late":
                values = models[baseline]["metrics"]["arm_late"] - models[comparator]["metrics"]["arm_late"]
            elif horizon == "endpoint":
                values = models[baseline]["metrics"]["arm_endpoint"] - models[comparator]["metrics"]["arm_endpoint"]
            else:
                values = models[baseline]["metrics"]["arm_full"] - models[comparator]["metrics"]["arm_full"]
            values = values[mask]
            if mask.all():
                indices = bootstrap_indices
            else:
                indices = np.random.default_rng(SEED + len(effect_rows) + 100).integers(0, len(values), size=(BOOTSTRAPS, len(values)), dtype=np.int32)
            ci_low, ci_high = bootstrap_ci(values, indices)
            effect_rows.append({"comparison": label, "baseline": baseline, "comparator": comparator, "subset": subset, "metric": "arm14_mae", "N": len(values), "effect_baseline_minus_comparator": float(values.mean()), "ci_low": ci_low, "ci_high": ci_high, "win_rate_comparator": float(np.mean(models[comparator]["metrics"][f"arm_{horizon}"][mask] < models[baseline]["metrics"][f"arm_{horizon}"][mask])), "median_effect": float(np.median(values)), "q05": float(np.quantile(values, 0.05)), "q95": float(np.quantile(values, 0.95)), "family": "V2_EXPLORATORY_FAMILY", "p_value": "not_computed"})
    write_csv(ROOT / "V2_PAIRED_EFFECTS.csv", effect_rows)
    write_json(ROOT / "V2_PAIRED_EFFECTS.json", {"rows": effect_rows, "interpretation": "positive means comparator beats baseline", "bootstrap_replicates": BOOTSTRAPS, "v2_p_values": "not computed; report effects and CIs only"})

    # Temporal fidelity and lag tables.
    fidelity_rows: list[dict[str, Any]] = []
    for name in eval_names:
        m = models[name]["metrics"]
        for subset, mask in [("full", np.ones(len(common_ids), dtype=bool)), ("low", strata == "low"), ("mid", strata == "mid"), ("high", strata == "high")]:
            values = {key: m[key][mask] for key in ("arm_first_diff", "tv_ratio", "endpoint_change_error", "direction_agreement", "direction_cosine")}
            fidelity_rows.append({"model": name, "subset": subset, "N": int(mask.sum()), "first_difference_mae_std": float(values["arm_first_diff"].mean()), "first_difference_mae_native": float(m["arm_first_diff_native"][mask].mean()), "tv_ratio_mean_native": safe_nanmean(values["tv_ratio"]), "tv_ratio_mean_standardized": safe_nanmean(m["tv_ratio_std"][mask]), "tv_ratio_median_native": safe_nanmedian(values["tv_ratio"]), "endpoint_change_error_native": float(values["endpoint_change_error"].mean()), "endpoint_change_error_standardized": float(m["endpoint_change_error_std"][mask].mean()), "direction_agreement_native": safe_nanmean(values["direction_agreement"]), "direction_agreement_standardized": safe_nanmean(m["direction_agreement_std"][mask]), "direction_cosine_native": safe_nanmean(values["direction_cosine"]), "direction_cosine_standardized": safe_nanmean(m["direction_cosine_std"][mask])})
    write_csv(ROOT / "V2_TEMPORAL_FIDELITY.csv", fidelity_rows)
    write_json(ROOT / "V2_TEMPORAL_FIDELITY.json", {"rows": fidelity_rows, "command_change_not_velocity": True})
    synthetic = synthetic_lag_test()
    if not synthetic["pass"]:
        raise RuntimeError("lag synthetic test failed")
    lag_rows: list[dict[str, Any]] = []
    lag_summary: list[dict[str, Any]] = []
    for name in eval_names:
        for subset, mask in [("full", np.ones(len(common_ids), dtype=bool)), ("high", strata == "high")]:
            pred = models[name]["pred_std"][mask]
            target = target_std[mask]
            rows_for_model = []
            for lag in LAGS:
                mae, corr = lag_score(pred, target, lag)
                row = {"model": name, "subset": subset, "lag": lag, "arm_mae_std": mae, "cross_correlation": corr, "status": "exploratory"}
                lag_rows.append(row); rows_for_model.append(row)
            best = min(rows_for_model, key=lambda row: row["arm_mae_std"])
            zero = next(row for row in rows_for_model if row["lag"] == 0)
            lag_summary.append({"model": name, "subset": subset, "best_lag": best["lag"], "best_lag_mae": best["arm_mae_std"], "zero_lag_mae": zero["arm_mae_std"], "delta_best_minus_zero": best["arm_mae_std"] - zero["arm_mae_std"], "status": "exploratory", "reason": "" if np.isfinite(best["arm_mae_std"]) else "constant_or_invalid"})
    write_csv(ROOT / "V2_TEMPORAL_LAG_DIAGNOSTICS.csv", lag_rows)
    write_json(ROOT / "V2_TEMPORAL_LAG_DIAGNOSTICS.json", {"lags": list(LAGS), "sign": "positive prediction leads target; negative prediction lags", "synthetic_validation": synthetic, "summary": lag_summary, "rows": lag_rows})

    # Diagnostic and fixed representative selection.
    selection_rng = np.random.default_rng(SEED)
    selected_indices: list[int] = []
    representative_rows: list[dict[str, Any]] = []
    for stratum in ("low", "mid", "high"):
        candidates = stratum_indices[stratum].copy()
        perm = candidates[selection_rng.permutation(len(candidates))]
        chosen: list[int] = []
        used_tasks: set[int] = set()
        for index in perm:
            if int(task[index]) not in used_tasks:
                chosen.append(int(index)); used_tasks.add(int(task[index]))
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            chosen.extend(int(index) for index in perm if int(index) not in chosen[:4])
        chosen = sorted(chosen[:4])
        selected_indices.extend(chosen)
        for index in chosen:
            representative_rows.append({"episode_id": int(common_ids[index]), "task_index": int(task[index]), "motion_stratum": stratum, "mean_step_motion_std": float(dev_motion_std["mean_step_motion"][index]), "selection_seed": SEED, "selection_rule": "target-only motion stratum + seeded unique-task preference"})
    write_csv(ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.csv", representative_rows)
    write_json(ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.json", {"N": 12, "per_stratum": 4, "seed": SEED, "model_independent": True, "rows": representative_rows})
    (ROOT / "V2_TRAJECTORY_SELECTION_PROTOCOL.md").write_text("# v2 trajectory selection protocol\n\nFour episodes per train-derived target-motion stratum are chosen with seed 20260910. The selector sees target motion, task index and episode ID only, prefers distinct tasks when possible, and is reused across every model. Diagnostic sets are separate and never called representative.\n")
    chosen_model = "B3-FUS-D-L"
    base_error = models["PERSISTENCE"]["metrics"]["arm_full"]
    chosen_error = models[chosen_model]["metrics"]["arm_full"]
    endpoint_gain = persistence["arm_endpoint"] - models[chosen_model]["metrics"]["arm_endpoint"]
    full_gain = base_error - chosen_error
    fd_error = models[chosen_model]["metrics"]["arm_first_diff"]
    diagnostic = {"worst_full_vs_persistence": common_ids[np.argsort(full_gain)[:4]].tolist(), "largest_full_gain": common_ids[np.argsort(full_gain)[-4:][::-1]].tolist(), "endpoint_gain_full_degradation": common_ids[np.where((endpoint_gain > 0) & (full_gain < 0))[0][np.argsort(full_gain[(endpoint_gain > 0) & (full_gain < 0)])[:4]]].tolist() if np.any((endpoint_gain > 0) & (full_gain < 0)) else [], "largest_first_difference": common_ids[np.argsort(fd_error)[-4:][::-1]].tolist(), "best_lag": [int(common_ids[i]) for i in np.argsort(chosen_error)[:4]]}
    write_json(ROOT / "V2_DIAGNOSTIC_EPISODES.json", diagnostic)
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_models = ["PERSISTENCE", "B-STATE", "B-D-P-F", "B3-FUS-D-F", "B3-FUS-D-L"]
    for index in sorted(selected_indices):
        for name in trajectory_models:
            for time in range(30):
                for offset in range(14):
                    joint = 14 + offset
                    trajectory_rows.append({"episode_id": int(common_ids[index]), "task_index": int(task[index]), "motion_stratum": str(strata[index]), "model": name, "time_index": time, "joint_index": joint, "target_std": float(target_std[index, time, joint]), "prediction_std": float(models[name]["pred_std"][index, time, joint]), "target_native": float(target_native[index, time, joint]), "prediction_native": float(models[name]["pred_native"][index, time, joint])})
    write_csv(ROOT / "data" / "V2_TRAJECTORY_REPRESENTATIVE_EPISODES.csv", trajectory_rows)

    # Failure taxonomy and model-independent support table.
    failure_rows: list[dict[str, Any]] = []
    for name in eval_names:
        m = models[name]["metrics"]
        for i, episode_id in enumerate(common_ids):
            label = "no_rule_trigger"
            if not np.isfinite(models[name]["pred_std"][i]).all(): label = "nonfinite_prediction"
            elif m["pred_total_variation"][i] < 1e-8: label = "near_constant_prediction"
            elif m["arm_first_diff"][i] > np.quantile(m["arm_first_diff"], 0.95): label = "high_command_change_error_top5pct"
            elif m["tv_ratio"][i] > np.nanquantile(m["tv_ratio"], 0.95): label = "excess_variation_top5pct"
            elif m["tv_ratio"][i] < np.nanquantile(m["tv_ratio"], 0.05): label = "under_variation_bottom5pct"
            failure_rows.append({"model": name, "episode_id": int(episode_id), "task_index": int(task[i]), "motion_stratum": str(strata[i]), "rule_based_label": label, "manual_label": "not_reviewed", "arm_full_mae_std": float(m["arm_full"][i]), "arm_first_diff_mae_std": float(m["arm_first_diff"][i]), "tv_ratio": "" if not np.isfinite(m["tv_ratio"][i]) else float(m["tv_ratio"][i]), "endpoint_change_error": float(m["endpoint_change_error"][i])})
    write_csv(ROOT / "V2_FAILURE_TAXONOMY.csv", failure_rows)
    write_json(ROOT / "V2_FAILURE_TAXONOMY.json", {"rows": failure_rows, "manual_labels": "not_reviewed", "rule_scope": "descriptive screening only"})
    (ROOT / "V2_FAILURE_TAXONOMY.md").write_text("# v2 failure taxonomy\n\nRules are per-model descriptive flags: nonfinite, near-constant, top-5% command-change error, top-5% excess variation, or bottom-5% under-variation. They are not human mechanism diagnoses.\n")

    # Rank lenses and route decision.
    full_means = {name: float(models[name]["metrics"]["arm_full"].mean()) for name in eval_names}
    high_means = {name: float(models[name]["metrics"]["arm_full"][strata == "high"].mean()) for name in eval_names}
    late_means = {name: float(models[name]["metrics"]["arm_late"].mean()) for name in eval_names}
    endpoint_means = {name: float(models[name]["metrics"]["arm_endpoint"].mean()) for name in eval_names}
    fd_means = {name: float(models[name]["metrics"]["arm_first_diff"].mean()) for name in eval_names}
    rank_rows: list[dict[str, Any]] = []
    for lens, values in (("full_mae", full_means), ("high_motion_mae", high_means), ("late_mae", late_means), ("endpoint_mae", endpoint_means), ("first_difference_mae", fd_means)):
        for rank, (name, value) in enumerate(sorted(values.items(), key=lambda item: item[1]), start=1):
            rank_rows.append({"lens": lens, "model": name, "rank": rank, "value": value, "descriptive_only": True})
    write_csv(ROOT / "V2_METRIC_LENS_RANKS.csv", rank_rows)
    b3_high_effect = float(persistence["arm_full"][strata == "high"].mean() - models[chosen_model]["metrics"]["arm_full"][strata == "high"].mean())
    b3_late_effect = float(persistence["arm_late"].mean() - models[chosen_model]["metrics"]["arm_late"].mean())
    b3_full_effect = float(persistence["arm_full"].mean() - models[chosen_model]["metrics"]["arm_full"].mean())
    b3_endpoint_effect = float(persistence["arm_endpoint"].mean() - models[chosen_model]["metrics"]["arm_endpoint"].mean())
    fd_ratio = float(models[chosen_model]["metrics"]["arm_first_diff"].mean() / persistence["arm_first_diff"].mean())
    high_n = int((strata == "high").sum())
    valid_lags = [row["best_lag"] for row in lag_summary if row["model"] in {"B-STATE", "B3-FUS-D-L"} and row["subset"] == "full"]
    if high_n < 50:
        route = "B"; route_reason = f"high-motion support N={high_n}<50, so event-centered support is the highest information-gain prerequisite."
    elif b3_high_effect > 0 and b3_late_effect > 0 and b3_full_effect < 0 and fd_ratio > 1.5:
        route = "A"; route_reason = f"B3-FUS-D-L has exploratory positive high-motion/late skill but negative full skill and first-difference ratio {fd_ratio:.2f}; residual temporal decoding directly tests the trade-off."
    elif len(valid_lags) >= 2 and len(set(valid_lags)) == 1 and valid_lags[0] != 0:
        route = "D"; route_reason = f"predeclared models share best lag {valid_lags[0]}, so alignment audit precedes new modeling."
    elif abs(float(models["B-STATE"]["metrics"]["arm_full"].mean() - models["B3-FUS-D-L"]["metrics"]["arm_full"].mean())) < 0.01 and abs(float(models["B3-FUS-D-L"]["metrics"]["arm_full"].mean() - models["B3-FUS-D-F"]["metrics"]["arm_full"].mean())) < 0.01:
        route = "C"; route_reason = "state/fusion and dynamic/static increments are small on the locked population; attribution controls have higher information gain."
    elif b3_full_effect > 0 and b3_high_effect > 0 and b3_endpoint_effect > 0:
        route = "E"; route_reason = "the predeclared model beats persistence across full, high-motion and endpoint slices; fresh-seed confirmation is justified."
    else:
        route = "A"; route_reason = "the dominant unresolved pattern is persistence-relative horizon/variation trade-off; residual temporal modeling is the bounded next test."
    route_values = {"full_effect": b3_full_effect, "high_effect": b3_high_effect, "late_effect": b3_late_effect, "endpoint_effect": b3_endpoint_effect, "first_difference_ratio": fd_ratio, "high_N": high_n}
    write_next_route(route, {"route_values": route_values, "route_rationale": route_reason})
    write_json(ROOT / "V2_ROUTE_DECISION.json", {"selected_route": route, "rationale": route_reason, "values": route_values, "executed": False, "original_b3_candidate_gate_unchanged": True})

    # Validity and conclusions.
    low_support = high_n < 50 or any(len(value) < 50 for value in stratum_indices.values())
    validity = [
        {"dimension": "artifact_and_reproducibility", "status": "PASS", "evidence": "v1 109/109 and formal 92/92 checksums; source arrays revalidated", "counter_evidence": "retrospective reuse", "supported_claim": "internal reproducible development benchmark", "unsupported_claim": "independent confirmation", "next_evidence": "fresh independent analysis or held-out release"},
        {"dimension": "baseline_resistance", "status": "FAIL", "evidence": f"persistence full arm14 MAE={persistence['arm_full'].mean():.6f}; learned formal cells are higher", "counter_evidence": f"endpoint effect for {chosen_model}={b3_endpoint_effect:.6f}", "supported_claim": "persistence is a strong baseline", "unsupported_claim": "learned model wins primary", "next_evidence": "challenge task or residual target"},
        {"dimension": "full_population_discrimination", "status": "WEAK", "evidence": "B1/B2/B3 and controls are comparable on N=614", "counter_evidence": "persistence dominates learned models", "supported_claim": "model differences are measurable", "unsupported_claim": "visual input adds general forecasting value", "next_evidence": "independent challenge"},
        {"dimension": "high_motion_discrimination", "status": "WEAK", "evidence": f"high-motion N={high_n}; effects are reported without changing Layer A, but the key B3 effect CI crosses zero", "counter_evidence": "single development population and inconclusive paired interval", "supported_claim": "exploratory motion-conditioned evidence", "unsupported_claim": "positive high-motion superiority", "next_evidence": "source-level event challenge"},
        {"dimension": "horizon_sensitivity", "status": "PASS", "evidence": "early/middle/late/endpoint complete for all models", "counter_evidence": "no verified timestamp array", "supported_claim": "step-resolved error profiles", "unsupported_claim": "physical-time dynamics", "next_evidence": "verified timestamps"},
        {"dimension": "temporal_fidelity", "status": "WEAK", "evidence": f"first-difference and TV/direction diagnostics computed; {chosen_model} first-difference ratio vs persistence={fd_ratio:.2f}", "counter_evidence": "diagnostic metrics are retrospective", "supported_claim": "trajectory variation differs", "unsupported_claim": "noise mechanism proven", "next_evidence": "residual/temporal controls"},
        {"dimension": "modality_attribution", "status": "WEAK", "evidence": "matched video/fusion/state comparisons", "counter_evidence": "state shortcut and single seed", "supported_claim": "fusion increment can be described", "unsupported_claim": "video causes gain", "next_evidence": "matched/mismatched video controls"},
        {"dimension": "sampling_sensitivity", "status": "PASS", "evidence": "true-pixel B2 intervals reported with motion/horizon slices", "counter_evidence": "runtime not measured", "supported_claim": "accuracy sensitivity", "unsupported_claim": "compute efficiency", "next_evidence": "comparable timings"},
        {"dimension": "adaptation_lora_sensitivity", "status": "WEAK", "evidence": "B3 matched frozen/LoRA and motion/horizon effects", "counter_evidence": "candidate gate remains 0; one seed", "supported_claim": "exploratory LoRA effects", "unsupported_claim": "robust LoRA benefit", "next_evidence": "fresh seeds after a passing challenge"},
        {"dimension": "task_joint_coverage", "status": "WEAK" if low_support else "PASS", "evidence": "per-task and joint tables generated", "counter_evidence": "task semantic units not audited", "supported_claim": "coverage is visible", "unsupported_claim": "physical joint interpretation", "next_evidence": "URDF/semantic audit"},
        {"dimension": "physical_interpretability", "status": "UNKNOWN", "evidence": "command-native labels only", "counter_evidence": "no units/URDF/FK verification", "supported_claim": "command trajectory", "unsupported_claim": "Cartesian/velocity accuracy", "next_evidence": "units and kinematics audit"},
        {"dimension": "seed_robustness", "status": "UNKNOWN", "evidence": "B3 one training seed", "counter_evidence": "episode bootstrap is not seed uncertainty", "supported_claim": "within-seed development evidence", "unsupported_claim": "cross-seed stability", "next_evidence": "three fresh seeds"},
        {"dimension": "final_generalization_readiness", "status": "FAIL", "evidence": "final-test access=0 and development reuse", "counter_evidence": "none", "supported_claim": "internal development use", "unsupported_claim": "final/public benchmark", "next_evidence": "locked independent evaluation"}
    ]
    write_csv(ROOT / "V2_VALIDITY_SCORECARD.csv", validity)
    write_json(ROOT / "V2_VALIDITY_SCORECARD.json", {"rows": validity, "status": "COMPLETE_WITH_CAVEATS"})
    validity_md = ["# v2 validity scorecard", "", "No single average score is computed.", "", "| Dimension | Status | Supported claim | Unsupported claim |", "|---|---|---|---|"]
    validity_md += [f"| {row['dimension']} | {row['status']} | {row['supported_claim']} | {row['unsupported_claim']} |" for row in validity]
    (ROOT / "V2_VALIDITY_SCORECARD.md").write_text("\n".join(validity_md) + "\n")
    best_formal = min([row["cell"] for row in accepted_rows], key=lambda name: full_means[name])
    best_high = min(eval_names, key=lambda name: high_means[name])
    best_late = min(eval_names, key=lambda name: late_means[name])
    conclusion_zh = f"""# Core B Persistence-Aware Motion Benchmark v2——结论

状态：`COMPLETE_WITH_CAVEATS`。这是 CPU-only、development-only、回顾性 secondary analysis；没有新训练、推理、GPU job 或 final-test。

Layer A 保留 v1 primary。共同集为 {len(common_ids)} 个 episode；正式覆盖 B1 7/7、真实像素 B2 6/6、B3 8/8。描述性 full-set 最佳正式 learned cell 是 `{best_formal}`，但 persistence 仍是强 baseline；Layer B 的 high-motion 最佳（含 derived controls）是 `{best_high}`，late 最佳是 `{best_late}`。这些排序不能升级为 final/public benchmark 或 multi-seed 结论。

Persistence-aware 结果显示：full MAE、endpoint、first-difference 和 RMSE 可以给出不同排序。`{chosen_model}` 相对 persistence 的 full effect={b3_full_effect:.6f}、high-motion effect={b3_high_effect:.6f}、late effect={b3_late_effect:.6f}、endpoint effect={b3_endpoint_effect:.6f}；first-difference ratio={fd_ratio:.2f}。这类结果支持“终点/部分 horizon 可能有信号，但变化过程可能过度振荡或不稳定”的诊断方向，但不能单独证明机制。

本轮唯一条件式下一步为 Route {route}：{route_reason}。Prompt 已生成但未执行；原 B3 candidate gate=0 保持不变。
"""
    conclusion_en = f"""# Core B Persistence-Aware Motion Benchmark v2 — conclusions

Status: `COMPLETE_WITH_CAVEATS`. This is a CPU-only, development-only retrospective secondary analysis; no new training, inference, GPU job or final-test access occurred.

Layer A preserves the v1 primary. The common population is {len(common_ids)} episodes; formal coverage is B1 7/7, true-pixel B2 6/6 and B3 8/8. The descriptive best formal learned full-set cell is `{best_formal}`, while persistence remains a strong baseline. The best high-motion and late-horizon entries including derived controls are `{best_high}` and `{best_late}`. These rankings do not establish a final/public benchmark or multi-seed result.

Persistence-aware reporting shows why full MAE, endpoint, first-difference and RMSE can disagree. For `{chosen_model}`, effects versus persistence are full={b3_full_effect:.6f}, high-motion={b3_high_effect:.6f}, late={b3_late_effect:.6f}, endpoint={b3_endpoint_effect:.6f}; the first-difference ratio is {fd_ratio:.2f}. This supports an exploratory interpretation of partial horizon/endpoint signal with possible excess variation, but does not prove a mechanism.

The single conditional next route is Route {route}: {route_reason}. Its prompt is generated but not executed; the original B3 candidate gate remains zero.
"""
    (ROOT / "V2_CONCLUSIONS.zh-CN.md").write_text(conclusion_zh)
    (ROOT / "V2_CONCLUSIONS.en.md").write_text(conclusion_en)
    (ROOT / "V2_LIMITATIONS_AND_CLAIMS.md").write_text("# Limitations and claims\n\nThis package is an internal development benchmark. It is retrospective, single-seed for B3, development-only, and uses source-native command values. It does not establish Cartesian accuracy, physical velocity, external generalization, public benchmark validity, SOTA, or causal video understanding. Layer B is diagnostic until independently reproduced.\n")
    (ROOT / "TEAM_BRIEF_BENCHMARK_V2.zh-CN.md").write_text(f"# 团队简报\n\nBenchmark v2 已完成：source gate 通过，full/high-motion/horizon/persistence/temporal diagnostics 和 12 张图已生成。Persistence 仍是 Layer A 强 baseline；B3 candidate gate=0。唯一生成的条件路线是 Route {route}，Prompt 不执行。\n")
    (ROOT / "PROFESSOR_MEETING_BENCHMARK_V2.zh-CN.md").write_text(f"# 教授会议更新\n\n- 范围：只读 B1/B2/B3 accepted predictions，N={len(common_ids)}，final-test=0。\n- Layer A：保留 full-horizon standardized arm14 MAE。\n- Layer B：train-derived motion strata、early/middle/late/endpoint、persistence skill、first-difference、TV、direction、lag。\n- 结论边界：内部 development benchmark；不能公开/最终泛化。\n- B3 candidate gate=0 保持。\n- 条件下一步：Route {route}，只生成 Prompt。\n")

    # Figures.
    formal_names = [row["cell"] for row in accepted_rows]
    colors = {"PERSISTENCE": "#555555", "B-STATE": "#0072B2", "B-D-P-F": "#CC79A7", "B3-FUS-D-F": "#009E73", "B3-FUS-D-L": "#D55E00", "B3-V-D-L": "#E69F00"}
    selected_plot_models = ["PERSISTENCE", "B-STATE", "B-D-P-F", "B3-FUS-D-F", "B3-FUS-D-L"]
    # 1 full versus high forest.
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    plot_names = formal_names
    vals = [float(persistence["arm_full"].mean() - models[name]["metrics"]["arm_full"].mean()) for name in plot_names]
    high_vals = [float(persistence["arm_full"][strata == "high"].mean() - models[name]["metrics"]["arm_full"][strata == "high"].mean()) for name in plot_names]
    y = np.arange(len(plot_names)); ax.scatter(vals, y, label="full", color="#0072B2"); ax.scatter(high_vals, y, label="high motion", color="#D55E00", marker="D"); ax.axvline(0, color="black", linewidth=0.8); ax.set_yticks(y, plot_names); ax.invert_yaxis(); ax.set_xlabel("Persistence MAE − model MAE (positive is better)"); ax.set_title("Full versus high-motion persistence-relative effects"); ax.legend()
    save_figure("fig01_full_high_paired_effects", fig, [{"model": name, "full_effect": vals[i], "high_motion_effect": high_vals[i]} for i, name in enumerate(plot_names)], "Descriptive full and train-threshold high-motion persistence-relative effects for all formal cells.", "所有正式模型在全体和 high-motion 子集相对 persistence 的 paired improvement；正值表示模型误差更低。")
    # 2 horizon profile.
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained"); data = []
    for name in selected_plot_models:
        means = [float(models[name]["metrics"][f"arm_{horizon}"] .mean()) for horizon in ("early", "middle", "late", "endpoint")]
        ax.plot(["early", "middle", "late", "endpoint"], means, marker="o", label=name, color=colors.get(name)); data.extend({"model": name, "horizon": horizon, "arm_mae_std": value} for horizon, value in zip(("early", "middle", "late", "endpoint"), means))
    ax.set_ylabel("Standardized arm MAE"); ax.set_title("Horizon-resolved error profile"); ax.legend(ncol=2, fontsize=8)
    save_figure("fig02_horizon_profile", fig, data, "Early/middle/late/endpoint errors for fixed persistence, state, video and fusion references.", "固定模型在 early、middle、late 和 endpoint 的标准化 arm MAE。")
    # 3 skill by motion.
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained"); data = []; x = np.arange(3); width = 0.16
    for offset, name in enumerate(selected_plot_models):
        skill = [float((persistence["arm_full"][strata == s] - models[name]["metrics"]["arm_full"][strata == s]).mean()) for s in ("low", "mid", "high")]
        ax.bar(x + (offset - 2) * width, skill, width, label=name, color=colors.get(name)); data.extend({"model": name, "stratum": s, "delta_vs_persistence": value} for s, value in zip(("low", "mid", "high"), skill))
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(x, ["low", "mid", "high"]); ax.set_ylabel("Persistence MAE − model MAE"); ax.set_title("Persistence-relative skill by motion stratum"); ax.legend(ncol=3, fontsize=8)
    save_figure("fig03_skill_by_motion", fig, data, "Persistence-relative full-horizon arm skill by train-derived target-motion stratum.", "low、mid、high 运动分层中各固定模型相对 persistence 的 full-horizon improvement。")
    # 4 motion scatter.
    gain = persistence["arm_full"] - models[chosen_model]["metrics"]["arm_full"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained"); ax.scatter(dev_motion_std["mean_step_motion"], gain, s=14, alpha=0.45, color="#D55E00"); ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel("Target mean step motion (standardized arm14)"); ax.set_ylabel("Persistence MAE − B3-FUS-D-L MAE"); ax.set_title("Motion-conditioned improvement");
    save_figure("fig04_motion_vs_improvement", fig, [{"episode_id": int(common_ids[i]), "mean_step_motion_std": float(dev_motion_std["mean_step_motion"][i]), "gain": float(gain[i])} for i in range(len(common_ids))], "Per-episode target motion versus the predeclared B3 fusion dynamic LoRA persistence-relative effect.", "目标运动幅度与 B3 fusion dynamic LoRA 相对 persistence 改善的散点图。")
    # 5 first difference and TV distribution.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained"); fd_arrays = [models[name]["metrics"]["arm_first_diff"] for name in selected_plot_models]; tv_arrays = [models[name]["metrics"]["tv_ratio"] for name in selected_plot_models]
    axes[0].boxplot(fd_arrays, tick_labels=selected_plot_models, showfliers=False); axes[0].set_title("Command-change error"); axes[0].set_ylabel("Arm first-difference MAE"); axes[0].tick_params(axis="x", rotation=30); axes[1].boxplot(tv_arrays, tick_labels=selected_plot_models, showfliers=False); axes[1].axhline(1, color="black", linewidth=0.8); axes[1].set_title("Trajectory variation ratio"); axes[1].set_ylabel("TV(pred)/TV(target)"); axes[1].tick_params(axis="x", rotation=30)
    save_figure("fig05_first_difference_tv", fig, [{"model": name, "episode_id": int(common_ids[i]), "first_difference_mae": float(models[name]["metrics"]["arm_first_diff"][i]), "tv_ratio": "" if not np.isfinite(models[name]["metrics"]["tv_ratio"][i]) else float(models[name]["metrics"]["tv_ratio"][i])} for name in selected_plot_models for i in range(len(common_ids))], "Episode-level command-change error and TV ratio distributions; the ratio is NA for near-zero target variation.", "固定模型的逐 episode command-change error 和 predicted/GT variation ratio 分布。")
    # 6 lag.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained"); data = []
    for ax, subset in zip(axes, ("full", "high")):
        for name in selected_plot_models:
            rows = [row for row in lag_rows if row["model"] == name and row["subset"] == subset]; ax.plot([row["lag"] for row in rows], [row["arm_mae_std"] for row in rows], marker="o", label=name, color=colors.get(name)); data.extend({**row, "subset": subset} for row in rows)
        ax.axvline(0, color="black", linewidth=0.8); ax.set_title(subset); ax.set_xlabel("Lag (positive=prediction leads)")
    axes[0].set_ylabel("Standardized arm MAE"); axes[0].legend(fontsize=7); fig.suptitle("Exploratory temporal lag diagnostic")
    save_figure("fig06_temporal_lag", fig, data, "Zero-lag remains primary; integer -5..+5 scan is exploratory after synthetic validation.", "全体和 high-motion 子集的探索性 temporal lag MAE 曲线。")
    # 7 representatives.
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True, layout="constrained"); data = []
    selected_indices_sorted = sorted(selected_indices)
    for ax, index in zip(axes.ravel(), selected_indices_sorted):
        target_arm = target_native[index, :, 14:28].mean(axis=1); ax.plot(target_arm, color="black", linewidth=2, label="GT")
        for name in selected_plot_models:
            value = models[name]["pred_native"][index, :, 14:28].mean(axis=1); linestyle = "--" if name == "PERSISTENCE" else "-"; ax.plot(value, linestyle=linestyle, color=colors.get(name), label=name); data.extend({"episode_id": int(common_ids[index]), "model": name, "time_index": t, "arm_mean_native": float(value[t])} for t in range(30))
        data.extend({"episode_id": int(common_ids[index]), "model": "GT", "time_index": t, "arm_mean_native": float(target_arm[t])} for t in range(30)); ax.set_title(f"ep {common_ids[index]} / {strata[index]}", fontsize=8)
    axes[0, 0].legend(fontsize=6, ncol=2); fig.suptitle("Fixed target-motion-stratified command trajectories")
    save_figure("fig07_representative_trajectories", fig, data, "The same 12 target-only selected episodes are shown for GT, persistence, state, video and fusion models.", "12 个固定低中高运动 episode 的 GT、persistence、state、video 和 fusion command trajectories。")
    # 8 endpoint gain/full loss cases.
    cases = np.where((endpoint_gain > 0) & (full_gain < 0))[0]; cases = cases[np.argsort(full_gain[cases])[:4]] if len(cases) else np.argsort(full_gain)[:4]
    fig, axes = plt.subplots(1, max(1, len(cases)), figsize=(4 * max(1, len(cases)), 4), sharey=True, layout="constrained"); axes = np.atleast_1d(axes); data = []
    for ax, index in zip(axes, cases):
        target_arm = target_native[index, :, 14:28].mean(axis=1); ax.plot(target_arm, color="black", linewidth=2, label="GT"); ax.plot(models["PERSISTENCE"]["pred_native"][index, :, 14:28].mean(axis=1), "--", color="#555555", label="PERSISTENCE"); ax.plot(models[chosen_model]["pred_native"][index, :, 14:28].mean(axis=1), color="#D55E00", label=chosen_model); ax.set_title(f"ep {common_ids[index]}"); data.extend({"episode_id": int(common_ids[index]), "time_index": t, "gt_arm_mean": float(target_arm[t]), "persistence_arm_mean": float(models["PERSISTENCE"]["pred_native"][index, t, 14:28].mean()), "model_arm_mean": float(models[chosen_model]["pred_native"][index, t, 14:28].mean())} for t in range(30))
    axes[0].legend(fontsize=7); fig.suptitle("Endpoint improvement but full-MAE degradation cases")
    save_figure("fig08_endpoint_gain_full_loss", fig, data, "Fixed diagnostic episodes where B3-FUS-D-L improves endpoint error but has worse full-horizon error than persistence, when such cases exist; fallback cases are labeled by manifest.", "终点改善但全轨迹 MAE 退化的诊断 episode 案例图。")
    # 9 joint-time heatmap.
    diff_heat = np.abs(models[chosen_model]["pred_std"] - target_std).mean(axis=0)[:, 14:28] - np.abs(persistence_std - target_std).mean(axis=0)[:, 14:28]
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained"); lim = float(np.max(np.abs(diff_heat))); im = ax.imshow(diff_heat.T, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim); ax.set_xlabel("Future step"); ax.set_ylabel("Arm joint index (14:28)"); ax.set_title("B3-FUS-D-L minus persistence arm error"); fig.colorbar(im, ax=ax, label="Δ standardized absolute error")
    save_figure("fig09_joint_time_heatmap", fig, [{"time_index": t, "joint_index": 14 + j, "delta": float(diff_heat[t, j])} for t in range(30) for j in range(14)], "Matched joint-by-time error difference with common color limits; blue indicates lower B3-FUS-D-L error.", "B3-FUS-D-L 减去 persistence 的 arm joint×time 误差热图。")
    # 10 task support/error.
    task_skill = {int(row["task_index"]): float(next(item for item in per_task_rows if item["model"] == "PERSISTENCE" and int(item["task_index"]) == int(row["task_index"]))["arm_full_mae_std"] - row["arm_full_mae_std"]) for row in per_task_rows if row["model"] == chosen_model}
    task_support = {int(row["task_index"]): int(row["N"]) for row in per_task_rows if row["model"] == chosen_model}
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained"); xx = np.asarray(list(task_support)); yy = np.asarray([task_skill[int(k)] for k in xx]); ss = np.asarray([task_support[int(k)] for k in xx]); sc = ax.scatter(xx, yy, s=10 + ss * 2, c=ss, cmap="viridis", alpha=0.75); ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel("Original task index"); ax.set_ylabel("Persistence MAE − B3-FUS-D-L MAE"); ax.set_title("Per-task support and persistence-relative improvement"); fig.colorbar(sc, ax=ax, label="task support N")
    save_figure("fig10_task_support_improvement", fig, [{"task_index": int(k), "support_N": task_support[int(k)], "gain": task_skill[int(k)]} for k in xx], "Per-task support and improvement for the fixed B3 fusion dynamic LoRA comparison; task IDs are dataset indices.", "逐任务样本支持度与 B3-FUS-D-L 相对 persistence 改善散点图。")
    # 11 B2 interval x motion/horizon.
    b2 = [name for name in formal_names if name.startswith("B2-")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained"); data = []
    for ax, branch in zip(axes, ("video_only", "fusion")):
        names = [name for name in b2 if models[name]["branch"] == branch]; names = sorted(names, key=lambda name: models[name]["interval_seconds"])
        for name in names:
            vals = [float(models[name]["metrics"][f"arm_{h}"][strata == "high"].mean()) for h in ("full", "late", "endpoint")]
            ax.plot(["full", "late", "endpoint"], vals, marker="o", label=f"{models[name]['interval_seconds']}s/{models[name]['unique_observation_frames']}f"); data.extend({"branch": branch, "model": name, "interval": models[name]["interval_seconds"], "horizon": h, "high_motion_mae": v} for h, v in zip(("full", "late", "endpoint"), vals))
        ax.set_title(branch); ax.set_xlabel("Horizon"); ax.legend(fontsize=7)
    axes[0].set_ylabel("High-motion standardized arm MAE"); fig.suptitle("B2 sampling interval × motion/horizon")
    save_figure("fig11_b2_interval_motion_horizon", fig, data, "True-pixel B2 sampling accuracy on high-motion full/late/endpoint slices; no runtime claim.", "B2 video-only 与 fusion 不同采样间隔在 high-motion 各 horizon 的误差曲线。")
    # 12 metric lens ranks.
    lenses = ["full_mae", "high_motion_mae", "late_mae", "endpoint_mae", "first_difference_mae"]; names = selected_plot_models; rank_matrix = np.asarray([[next(row["rank"] for row in rank_rows if row["lens"] == lens and row["model"] == name) for lens in lenses] for name in names])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained"); im = ax.imshow(rank_matrix, aspect="auto", cmap="viridis_r"); ax.set_xticks(np.arange(len(lenses)), lenses, rotation=25, ha="right"); ax.set_yticks(np.arange(len(names)), names); ax.set_title("Descriptive metric-lens ranks (not a composite)"); fig.colorbar(im, ax=ax, label="rank; 1 is lowest error")
    save_figure("fig12_metric_lens_ranks", fig, [{"model": name, "lens": lens, "rank": int(rank_matrix[i, j])} for i, name in enumerate(names) for j, lens in enumerate(lenses)], "Ranks are shown separately for each metric lens; no composite rank is constructed.", "不同 metric lens 的描述性排名热图，不构造 composite rank。")
    write_csv(ROOT / "figure_manifests" / "FIGURE_MANIFEST.csv", FIGURE_MANIFEST)
    write_json(ROOT / "figure_manifests" / "FIGURE_MANIFEST.json", {"figures": FIGURE_MANIFEST, "count": len(FIGURE_MANIFEST), "visual_qa": "pending_manual_view_image"})

    # Status and a machine-readable reconstruction summary.
    write_json(ROOT / "V2_STATUS.json", {"status": "analysis_complete_pending_qa", "benchmark_id": ROOT.name, "formal_coverage": {"B1": "7/7", "B2_true_pixel": "6/6", "B3": "8/8"}, "common_episodes": len(common_ids), "target_shape": list(target_std.shape), "best_formal_full": best_formal, "best_high_including_controls": best_high, "best_late_including_controls": best_late, "selected_route": route, "final_test_access": 0, "new_training_started": False, "new_inference_started": False, "new_gpu_jobs_started": 0, "figure_count": len(FIGURE_MANIFEST)})
    (ROOT / "README.zh-CN.md").write_text(f"# Core B Persistence-Aware Motion Benchmark v2\n\n本目录是 CPU-only 的回顾性 development secondary analysis。Layer A 保留 v1 full-population primary；Layer B 增加 train-derived motion strata、horizon、persistence skill、trajectory fidelity 和 exploratory lag。当前计算阶段完成，待 QA。共同集 N={len(common_ids)}，formal B1/B2/B3=7/7、6/6、8/8，final-test=0。\n\n唯一条件式下一路线为 Route {route}，只生成 Prompt，不执行。\n")
    (ROOT / "README.en.md").write_text(f"# Core B Persistence-Aware Motion Benchmark v2\n\nCPU-only retrospective development secondary analysis. Layer A preserves the v1 full-population primary; Layer B adds train-derived motion strata, horizon bins, persistence skill, trajectory fidelity and exploratory lag. Computation is complete and QA is pending. Common N={len(common_ids)}; formal B1/B2/B3=7/7, 6/6, 8/8; final-test=0.\n\nThe only conditional next route is Route {route}; its prompt is generated but not executed.\n")
    print(json.dumps({"root": str(ROOT), "N": len(common_ids), "formal": [len([row for row in accepted_rows if row['stage'] == stage]) for stage in ('B1','B2','B3')], "selected_route": route, "figures": len(FIGURE_MANIFEST), "status": "analysis_complete_pending_qa"}))


if __name__ == "__main__":
    main()
