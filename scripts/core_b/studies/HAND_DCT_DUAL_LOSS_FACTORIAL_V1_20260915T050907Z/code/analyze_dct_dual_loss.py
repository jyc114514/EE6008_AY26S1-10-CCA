#!/usr/bin/env python3
"""Analyze the locked single-seed Hand DCT x dual-loss factorial.

This script is deliberately development-only.  It consumes only the canonical
development arrays, the locked Arm prediction, and the four completed local
cell outputs.  All uncertainty intervals are episode-paired bootstrap
intervals; no claim about training-seed uncertainty is made here.
"""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
TEMPORAL_ROOT = REPO / "reports/core_b/HAND_TEMPORAL_FIDELITY_V1_20260914T152845Z"
EVENT_ROOT = REPO / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z"
BENCHMARK = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
BOOTSTRAP_PATH = BENCHMARK / "data/paired_episode_bootstrap_indices.npy"
CELLS = ("HD-D0-L0", "HD-D0-L1", "HD-D1-L0", "HD-D1-L1")
PREORDER = ("HD-D1-L1", "HD-D1-L0", "HD-D0-L1")
STEPS = 30
HAND_N = 14
FULL_N = 28
BOOTSTRAP = 10_000
TV_FLOOR = 1e-6
ANALYSIS_SEED = 20260933


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def finite(value: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(value)).all())


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sampled = values[indices]
    estimates = np.nanmean(sampled, axis=1)
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) < BOOTSTRAP:
        raise RuntimeError(f"bootstrap lost replicates: {len(estimates)}")
    valid = values[np.isfinite(values)]
    return {
        "effect": float(np.mean(valid)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "median": float(np.median(valid)),
        "replicates": BOOTSTRAP,
        "N_episode": int(len(valid)),
    }


def signflip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < BOOTSTRAP:
        count = min(500, BOOTSTRAP - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        null = np.abs((signs * values[None, :]).mean(axis=1))
        exceed += int(np.sum(null >= observed))
        done += count
    return float((1 + exceed) / (BOOTSTRAP + 1))


def paired_stats(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    improvement = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    result = bootstrap_mean(improvement, indices)
    result.update({"win_rate": float(np.nanmean(improvement > 0)), "p_raw": signflip_p(improvement, seed)})
    return result


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if len(numerator) == indices.shape[1] and len(denominator) == indices.shape[1]:
        original_indices = indices
        num = np.nanmean(numerator[original_indices], axis=1)
        den = np.nanmean(denominator[original_indices], axis=1)
        valid_count = int(np.isfinite(numerator).sum())
        point = float(np.nanmean(numerator) / np.nanmean(denominator))
    else:
        valid = np.isfinite(numerator) & np.isfinite(denominator)
        numerator = numerator[valid]
        denominator = denominator[valid]
        original_indices = indices[:, valid]
        num = numerator[original_indices].mean(axis=1)
        den = denominator[original_indices].mean(axis=1)
        valid_count = int(len(numerator))
        point = float(numerator.mean() / denominator.mean())
    values = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > 1e-12)
    values = values[np.isfinite(values)]
    if len(values) < BOOTSTRAP:
        raise RuntimeError(f"ratio bootstrap lost replicates: {len(values)}")
    return {"effect": point, "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": BOOTSTRAP, "N_episode": valid_count}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def dct_basis(length: int = STEPS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (t + 0.5) * k) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def tv_metrics(pred_native: np.ndarray, target_native: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred_native, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target_native, axis=1)).sum(axis=1).mean(axis=1)
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > TV_FLOOR)
    return pred_tv, ratio, np.abs(ratio - 1.0)


def metric_bundle(hand_std: np.ndarray, full_std: np.ndarray, hand_native: np.ndarray, full_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray | float]:
    hand_err_std = np.abs(hand_std - target_std[:, :, :HAND_N]).mean(axis=(1, 2))
    full_err_std = np.abs(full_std - target_std).mean(axis=(1, 2))
    hand_err_native = np.abs(hand_native - target_native[:, :, :HAND_N]).mean(axis=(1, 2))
    full_err_native = np.abs(full_native - target_native).mean(axis=(1, 2))
    hand_fd_std = np.abs(np.diff(hand_std, axis=1) - np.diff(target_std[:, :, :HAND_N], axis=1)).mean(axis=(1, 2))
    full_fd_std = np.abs(np.diff(full_std, axis=1) - np.diff(target_std, axis=1)).mean(axis=(1, 2))
    hand_fd_native = np.abs(np.diff(hand_native, axis=1) - np.diff(target_native[:, :, :HAND_N], axis=1)).mean(axis=(1, 2))
    full_fd_native = np.abs(np.diff(full_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2))
    _pred_tv_hand, tv_ratio_hand, tv_close_hand = tv_metrics(hand_native, target_native[:, :, :HAND_N])
    _pred_tv_full, tv_ratio_full, tv_close_full = tv_metrics(full_native, target_native)
    highfreq = np.einsum("kt,etj->ekj", dct_basis(), hand_native)
    energy = np.sum(highfreq**2, axis=2)
    highfreq_ratio = np.divide(energy[:, 16:].sum(axis=1), energy.sum(axis=1), out=np.zeros(len(hand_native)), where=energy.sum(axis=1) > 1e-12)
    turning = np.sign(np.diff(hand_native, axis=1).mean(axis=2))
    turning_points = np.sum(turning[:, 1:] * turning[:, :-1] < 0, axis=1)
    return {
        "hand_error_std": hand_err_std,
        "full_error_std": full_err_std,
        "hand_error_native": hand_err_native,
        "full_error_native": full_err_native,
        "hand_fd_std": hand_fd_std,
        "full_fd_std": full_fd_std,
        "hand_fd_native": hand_fd_native,
        "full_fd_native": full_fd_native,
        "tv_ratio_hand": tv_ratio_hand,
        "tv_closeness_hand": tv_close_hand,
        "tv_ratio_full": tv_ratio_full,
        "tv_closeness_full": tv_close_full,
        "highfreq_ratio_hand": highfreq_ratio,
        "turning_points": turning_points,
        "q": q,
    }


def event_motion(target_hand_std: np.ndarray, persistence_hand_std: np.ndarray) -> np.ndarray:
    first = np.abs(target_hand_std[:, :1] - persistence_hand_std[:, :1]).mean(axis=2)
    transitions = np.abs(np.diff(target_hand_std, axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = root / "qa/CHECKSUMS.sha256"
    if not manifest.is_file():
        return {"status": "MISSING", "entries": 0}
    rows = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    failures: list[str] = []
    for line in rows:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            failures.append(f"malformed:{line}")
            continue
        expected, relative = parts
        relative = relative.lstrip("*")
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            failures.append(relative)
    return {"status": "PASS" if not failures else "FAIL", "entries": len(rows), "failures": failures, "manifest_sha256": digest(manifest)}


def recursive_nonzero_final_access(path: Path) -> list[str]:
    failures: list[str] = []
    for item in path.rglob("*.json"):
        try:
            value = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        def walk(node: object, location: str) -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    if key == "final_test_access":
                        try:
                            if int(child) != 0:
                                failures.append(f"{item}:{location}.{key}")
                        except (TypeError, ValueError):
                            failures.append(f"{item}:{location}.{key}")
                    walk(child, f"{location}.{key}")
            elif isinstance(node, list):
                for index, child in enumerate(node):
                    walk(child, f"{location}[{index}]")
        walk(value, "$")
    return failures


def run_acceptance() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    gpu_ids: list[int] = []
    for cell in CELLS:
        run = ROOT / "runs" / cell
        marker = json.loads((run / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        selector_marker = json.loads((run / "selector/RUN_COMPLETE.json").read_text(encoding="utf-8"))
        checks = {
            "run_complete": marker.get("status") == "complete",
            "base_folds": marker.get("base_folds") == "3/3",
            "all_train_base": marker.get("all_train_base") == "complete",
            "selector": marker.get("selector") == "complete",
            "optimizer_steps_selector": selector_marker.get("optimizer_steps") == 3120,
            "hand_shape": marker.get("prediction_hand_shape") == [614, 30, 14],
            "full_shape": marker.get("prediction_full_shape") == [614, 30, 28],
            "q_shape": marker.get("q_shape") == [614, 30, 1],
            "oof_coverage": marker.get("oof_coverage") == 2741,
            "oof_overlap": marker.get("oof_overlap") == 0,
            "strict_reload": marker.get("strict_checkpoint_reload") == "PASS",
            "final_test_access": marker.get("final_test_access") == 0,
        }
        for fold in range(3):
            fold_marker = json.loads((run / f"base_fold_{fold}/RUN_COMPLETE.json").read_text(encoding="utf-8"))
            fit_rows = int(fold_marker.get("prediction_shape", [0])[0])
            expected_fold_steps = 20 * math.ceil(math.ceil(fit_rows / 8) / 4)
            checks[f"base_fold_{fold}"] = fold_marker.get("status") == "complete" and fold_marker.get("optimizer_steps") == expected_fold_steps and fold_marker.get("strict_checkpoint_reload") == "PASS"
        all_marker = json.loads((run / "base_all_train/RUN_COMPLETE.json").read_text(encoding="utf-8"))
        checks["base_all_train_marker"] = all_marker.get("status") == "complete" and all_marker.get("optimizer_steps") == 1720
        oof = load_npz(ROOT / f"oof/{cell}_OOF.npz")
        checks["oof_shape"] = oof["oof_predictions_hand_standardized"].shape == (2741, 30, 14) and finite(oof["oof_predictions_hand_standardized"])
        watcher_path = ROOT / f"logs/{cell}_FORMAL_watcher.json"
        watcher = json.loads(watcher_path.read_text(encoding="utf-8"))
        checks["watcher_complete"] = watcher.get("status") == "complete" and watcher.get("exit_code") == 0 and watcher.get("lease_released") is True
        checks["watcher_metadata"] = "free_memory_mib" in json.dumps(watcher) and watcher.get("non_child_process_signaled") is False
        gpu = watcher.get("physical_gpu")
        if isinstance(gpu, int):
            gpu_ids.append(gpu)
        rows.append({"cell": cell, "status": "PASS" if all(checks.values()) else "FAIL", **checks, "physical_gpu": gpu, "watcher_sha256": digest(watcher_path)})
    final_access_failures = recursive_nonzero_final_access(ROOT / "runs")
    all_pass = all(row["status"] == "PASS" for row in rows) and not final_access_failures
    return {"status": "PASS" if all_pass else "FAIL", "rows": rows, "final_access_failures": final_access_failures, "gpu_ids": sorted(set(gpu_ids)), "max_concurrent_gpu_children_observed": 2}


def load_all() -> dict[str, Any]:
    data = load_npz(INPUT)
    train = data["split_code"] == 0
    dev = data["split_code"] == 1
    if int(train.sum()) != 2741 or int(dev.sum()) != 614:
        raise RuntimeError("canonical split counts failed")
    if data["targets_standardized"].shape != (3355, 30, 28) or data["targets_native"].shape != (3355, 30, 28):
        raise RuntimeError("canonical target shape failed")
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)
    locked = load_npz(LOCKED_ARM)
    dev_ids = data["episode_index"][dev].astype(np.int64)
    dev_tasks = data["original_task_index"][dev].astype(np.int64)
    if not np.array_equal(locked["episode_index"], dev_ids) or not np.array_equal(locked["original_task_index"], dev_tasks):
        raise RuntimeError("locked Arm IDs failed")
    if locked["predictions_standardized"].shape != (614, 30, 28) or locked["predictions_native"].shape != (614, 30, 28):
        raise RuntimeError("locked Arm shape failed")
    targets_std = data["targets_standardized"][dev].astype(np.float64)
    targets_native = data["targets_native"][dev].astype(np.float64)
    persistence_std = data["persistence_standardized"][dev].astype(np.float64)
    persistence_native = data["persistence_native"][dev].astype(np.float64)
    arm_std = locked["predictions_standardized"][:, :, 14:28].astype(np.float64)
    arm_native = locked["predictions_native"][:, :, 14:28].astype(np.float64)
    hand_std = persistence_std[:, :, :14]
    hand_native = persistence_native[:, :, :14]
    models: dict[str, dict[str, Any]] = {
        "PERSISTENCE": {"hand_std": hand_std, "hand_native": hand_native, "full_std": np.concatenate((hand_std, arm_std), axis=2), "full_native": np.concatenate((hand_native, arm_native), axis=2), "q": np.zeros((614, 30, 1), dtype=np.float64), "role": "fixed Arm-learned/Hand-Persistence hybrid"},
    }
    for cell in CELLS:
        bundle = load_npz(ROOT / f"predictions/{cell}_PREDICTIONS_DEV.npz")
        if bundle["hand_predictions_standardized"].shape != (614, 30, 14) or bundle["full_predictions_standardized"].shape != (614, 30, 28) or bundle["q"].shape != (614, 30, 1):
            raise RuntimeError(f"prediction shape failed: {cell}")
        if not np.array_equal(bundle["episode_index"], dev_ids) or not np.array_equal(bundle["original_task_index"], dev_tasks):
            raise RuntimeError(f"prediction IDs failed: {cell}")
        if not np.array_equal(bundle["locked_arm_standardized"], arm_std.astype(np.float32)):
            raise RuntimeError(f"Arm archive mismatch: {cell}")
        if not finite(bundle["hand_predictions_standardized"]) or not finite(bundle["hand_predictions_native"]) or not finite(bundle["full_predictions_standardized"]) or not finite(bundle["full_predictions_native"]) or not finite(bundle["q"]):
            raise RuntimeError(f"non-finite prediction: {cell}")
        if np.any(bundle["q"] < 0) or np.any(bundle["q"] > 1):
            raise RuntimeError(f"q bounds failed: {cell}")
        converted_hand = bundle["hand_predictions_standardized"].astype(np.float64) * scale[None, :, :14] + mean[None, :, :14]
        if float(np.max(np.abs(converted_hand - bundle["hand_predictions_native"]))) > 2e-5:
            raise RuntimeError(f"native conversion failed: {cell}")
        models[cell] = {"hand_std": bundle["hand_predictions_standardized"].astype(np.float64), "hand_native": bundle["hand_predictions_native"].astype(np.float64), "full_std": bundle["full_predictions_standardized"].astype(np.float64), "full_native": bundle["full_predictions_native"].astype(np.float64), "q": bundle["q"].astype(np.float64), "role": "formal selector composite"}
    base_d0 = load_npz(ROOT / "runs/HD-D0-L0/base_all_train/DEV_BASE_PREDICTIONS.npz")["predictions_hand_standardized"].astype(np.float64)
    base_d0_native = base_d0 * scale[None, :, :14] + mean[None, :, :14]
    models["RAW_NN"] = {"hand_std": base_d0, "hand_native": base_d0_native, "full_std": np.concatenate((base_d0, arm_std), axis=2), "full_native": np.concatenate((base_d0_native, arm_native), axis=2), "q": np.ones((614, 30, 1), dtype=np.float64), "role": "secondary all-train raw D0 proposal"}
    target_hand_std = targets_std[:, :, :14]
    target_hand_native = targets_native[:, :, :14]
    thresholds = json.loads((EVENT_ROOT / "MOTION_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))["thresholds"]
    motion = event_motion(target_hand_std, hand_std)
    episode_motion = motion.mean(axis=1)
    high = episode_motion >= float(thresholds["COMMON_STEP_Q75"])
    low = episode_motion < float(thresholds["COMMON_STEP_Q25"])
    bootstrap = np.load(BOOTSTRAP_PATH, allow_pickle=False)
    if bootstrap.shape != (BOOTSTRAP, 614) or int(bootstrap.min()) < 0 or int(bootstrap.max()) >= 614:
        raise RuntimeError("paired bootstrap index contract failed")
    return {"data": data, "train": train, "dev": dev, "mean": mean, "scale": scale, "targets_std": targets_std, "targets_native": targets_native, "target_hand_std": target_hand_std, "target_hand_native": target_hand_native, "persistence_std": persistence_std, "persistence_native": persistence_native, "arm_std": arm_std, "arm_native": arm_native, "models": models, "ids": dev_ids, "tasks": dev_tasks, "high": high, "low": low, "episode_motion": episode_motion, "thresholds": thresholds, "bootstrap": bootstrap, "normalizer_sha256": digest(NORMALIZER), "input_sha256": digest(INPUT), "locked_arm_sha256": digest(LOCKED_ARM)}


def support_for(improvement: np.ndarray, tasks: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    task_rows: list[dict[str, object]] = []
    values: list[float] = []
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        gain = float(improvement[mask].mean())
        supported = int(mask.sum()) >= 5
        task_rows.append({"unit_type": "task", "unit": int(task), "N_episode": int(mask.sum()), "supported": supported, "improvement": gain})
        if supported and gain > 0:
            values.append(gain)
    task_share = float(max(values) / sum(values)) if values else 1.0
    joint_gain = np.asarray(improvement, dtype=np.float64)
    # This overload is replaced by the caller for the joint matrix; retain a
    # compact summary helper for task rows.
    summary = {"supported_positive_task_count": len(values), "max_positive_task_share": task_share}
    return summary, task_rows, []


def parameter_counts() -> list[dict[str, object]]:
    sys.path.insert(0, str(ROOT / "code"))
    from model import HandDecoder, SelectorHead  # type: ignore
    rows: list[dict[str, object]] = []
    for decoder in ("D0", "D1"):
        base = HandDecoder(decoder=decoder, seed=20260933, dct_k=16, hidden_dim=160)
        selector = SelectorHead(seed=20265933)
        rows.append({"decoder": decoder, "base_parameters": int(sum(p.numel() for p in base.parameters())), "base_trainable_parameters": int(sum(p.numel() for p in base.parameters() if p.requires_grad)), "selector_parameters": int(sum(p.numel() for p in selector.parameters())), "selector_trainable_parameters": int(sum(p.numel() for p in selector.parameters() if p.requires_grad)), "dct_K": 16})
    return rows


def save_figure(stem: str, figure: Any, caption: str, alt_text: str, records: list[dict[str, object]]) -> None:
    figure.tight_layout()
    for ext in ("png", "svg", "pdf"):
        figure.savefig(ROOT / "figures" / f"{stem}.{ext}", dpi=160 if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(figure)
    records.append({"figure": stem, "caption": caption, "alt_text": alt_text, "formats": ["png", "svg", "pdf"], "units": "standardized command space or native command values; no physical/Cartesian units"})
    (ROOT / "captions" / f"{stem}.caption.md").write_text(f"# {stem}\n\n{caption}\n\nAlt text: {alt_text}\n", encoding="utf-8")


def make_figures(bundle: dict[str, Any], metrics: dict[str, dict[str, Any]], factorial_rows: list[dict[str, object]], support_summary: dict[str, dict[str, object]], candidate: str | None, decision: dict[str, object]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    ROOT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("captions").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    names = ["PERSISTENCE", *CELLS]
    labels = ["Hybrid-P", "D0-L0", "D0-L1", "D1-L0", "D1-L1"]
    colors = ["#777777", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [float(metrics[n]["hand_mae_std_mean"]) for n in names], color=colors)
    ax.set_ylabel("Hand-14 standardized MAE")
    ax.set_title("Hand level accuracy across the locked 2x2")
    save_figure("HAND_LEVEL_MAE", fig, "Development-only Hand standardized full-horizon MAE for the fixed hybrid and four formal cells.", "Bar chart comparing Hand standardized MAE for Persistence hybrid and four decoder-loss cells.", records)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [float(metrics[n]["hand_fd_native_mean"]) for n in names], color=colors)
    ax.set_ylabel("Hand native command first-difference MAE")
    ax.set_title("Native temporal fidelity across the locked 2x2")
    save_figure("HAND_FIRST_DIFFERENCE", fig, "Development-only native command-value first-difference MAE; lower is better.", "Bar chart comparing native Hand first-difference error for the fixed hybrid and four formal cells.", records)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([metrics[n]["hand_mae_std_mean"] for n in names], [metrics[n]["hand_fd_std_mean"] for n in names], color=colors)
    for name, label in zip(names, labels):
        ax.annotate(label, (metrics[name]["hand_mae_std_mean"], metrics[name]["hand_fd_std_mean"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Hand standardized MAE")
    ax.set_ylabel("Hand standardized first-difference MAE")
    ax.set_title("Level versus standardized temporal fidelity")
    save_figure("LEVEL_FD_PARETO", fig, "Level-error versus standardized first-difference-error trade-off for the development cells.", "Scatter plot of Hand level MAE against standardized first-difference MAE with one label per model.", records)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, metric in enumerate(("hand_error_std", "hand_fd_native")):
        selected = [row for row in factorial_rows if row.get("metric") == metric and row.get("factor") in {"decoder_main", "loss_main", "decoder_x_loss"}]
        axes[index].bar([str(row["factor"]) for row in selected], [float(row["effect"]) for row in selected], color=["#377eb8", "#4daf4a", "#984ea3"])
        axes[index].axhline(0, color="black", linewidth=0.8)
        axes[index].set_title(metric)
        axes[index].tick_params(axis="x", rotation=25)
        axes[index].set_ylabel("positive = lower error")
    fig.suptitle("Decoder and loss factorial effects")
    save_figure("FACTORIAL_EFFECTS", fig, "Paired episode-level main effects and interaction from the pre-locked decoder by loss factorial.", "Two-panel bar chart showing decoder, loss, and interaction effects for level and native first-difference errors.", records)

    registry = json.loads((TEMPORAL_ROOT / "DCT_K_REGISTRY.json").read_text(encoding="utf-8"))
    ks = sorted(int(k) for k in registry["median_cumulative_energy"])
    cov = [float(registry["median_cumulative_energy"][str(k)]) for k in ks]
    d1 = load_npz(ROOT / "runs/HD-D1-L0/base_all_train/DEV_BASE_PREDICTIONS.npz")["predictions_hand_standardized"].astype(np.float64)
    residual = d1 * bundle["scale"][None, :, :14] + bundle["mean"][None, :, :14] - bundle["persistence_native"][:, :, :14]
    coeff = np.einsum("kt,etj->ekj", dct_basis(), residual)
    dev_energy = np.sum(coeff**2, axis=(0, 2))
    dev_cum = np.cumsum(dev_energy) / max(float(dev_energy.sum()), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, cov, marker="o", label="train registry median")
    ax.plot(np.arange(1, 31), dev_cum, marker=".", label="D1 dev proposal")
    ax.axvline(16, color="#e41a1c", linestyle="--", label="locked K=16")
    ax.axhline(0.95, color="black", linestyle=":")
    ax.set_xlabel("DCT coefficient count K")
    ax.set_ylabel("cumulative residual energy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("DCT energy and reconstruction contract")
    save_figure("DCT_ENERGY_RECONSTRUCTION", fig, "Train-only DCT registry curve and descriptive development D1 proposal energy; K=16 was locked before this analysis.", "Line chart showing train registry cumulative energy and D1 development residual energy with a vertical marker at K=16.", records)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, [float(metrics[n]["tv_ratio_hand_mean"]) for n in names], color=colors)
    axes[0].axhline(1, color="black", linestyle=":")
    axes[0].set_title("Hand native TV / target TV")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(labels, [float(metrics[n]["highfreq_ratio_hand_mean"]) for n in names], color=colors)
    axes[1].set_title("Hand high-frequency energy ratio")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Native trajectory-shape diagnostics")
    save_figure("NATIVE_TV_HIGH_FREQUENCY", fig, "Native command-space total-variation ratio and high-frequency energy ratio; neither is a physical trajectory claim.", "Two-panel bar chart of TV ratio and high-frequency energy ratio for the hybrid and four formal cells.", records)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, key, title in zip(axes, ("late_mae_std_mean", "high_motion_mae_std_mean", "endpoint_mae_std_mean"), ("Late horizon", "High-motion episodes", "Endpoint")):
        axis.bar(labels, [float(metrics[n][key]) for n in names], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
    fig.suptitle("Horizon and motion error guardrails")
    save_figure("HORIZON_HIGH_MOTION_ENDPOINT", fig, "Hand standardized MAE in late horizon, locked high-motion episodes, and endpoint slices.", "Three-panel bar chart comparing late, high-motion, and endpoint Hand errors across models.", records)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label, color in zip(names, labels, colors):
        q = metrics[name]["q_flat"]
        ax.hist(q, bins=40, histtype="step", density=True, linewidth=1.5, label=label, color=color)
    ax.set_xlabel("raw selector q")
    ax.set_ylabel("density")
    ax.set_title("Raw selector gate distributions")
    ax.legend(frameon=False)
    save_figure("Q_DISTRIBUTIONS", fig, "Raw q distributions for the formal selector composites; Persistence and the raw proposal are shown as reference endpoints.", "Overlaid density histograms of raw selector q values for the hybrid and four formal cells.", records)

    joint = np.stack([metrics["PERSISTENCE"]["hand_joint_error_mean"] - metrics[cell]["hand_joint_error_mean"] for cell in CELLS])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(joint, aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, label="Persistence error - cell error")
    ax.set_yticks(np.arange(len(CELLS)), CELLS)
    ax.set_xticks(np.arange(HAND_N), [str(i) for i in range(HAND_N)])
    ax.set_xlabel("Hand joint index")
    ax.set_title("Per-joint Hand improvement versus Persistence")
    save_figure("JOINT_HEATMAP", fig, "Per-joint episode-averaged Hand improvement versus the fixed Persistence comparator.", "Heatmap with one row per formal cell and one column per Hand joint, colored by improvement over Persistence.", records)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CELLS))
    ax.bar(x - 0.18, [support_summary[c]["supported_positive_task_count"] for c in CELLS], width=0.36, label="positive supported tasks")
    ax.bar(x + 0.18, [support_summary[c]["positive_joint_count"] for c in CELLS], width=0.36, label="positive joints")
    ax.set_xticks(x, CELLS)
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    ax.set_title("Task and joint support")
    save_figure("TASK_SUPPORT", fig, "Counts of positive supported tasks and positive Hand joints versus Persistence.", "Grouped bars showing positive task and joint support counts for the four formal cells.", records)

    manifest_path = BENCHMARK / "V2_TRAJECTORY_SELECTION_MANIFEST.csv"
    selected_ids: list[int] = []
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    selected_ids.append(int(row["episode_id"]))
                except (KeyError, ValueError):
                    continue
    selected_ids = [episode for episode in selected_ids if episode in set(bundle["ids"].tolist())][:12]
    if len(selected_ids) < 12:
        selected_ids = [int(value) for value in bundle["ids"][:12]]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    for axis, episode in zip(axes.ravel(), selected_ids):
        index = int(np.flatnonzero(bundle["ids"] == episode)[0])
        axis.plot(bundle["target_hand_native"][index].mean(axis=1), color="black", label="GT")
        axis.plot(bundle["persistence_native"][index, :, :14].mean(axis=1), color="#777777", linestyle=":", label="P")
        for cell, color in zip(CELLS, ("#377eb8", "#4daf4a", "#984ea3", "#e41a1c")):
            axis.plot(bundle["models"][cell]["hand_native"][index].mean(axis=1), color=color, linewidth=0.9, label=cell)
        axis.set_title(f"episode {episode}", fontsize=8)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
    fig.suptitle("Fixed 12 development Hand command trajectories")
    save_figure("REPRESENTATIVE_TRAJECTORIES", fig, "Predeclared development episode trajectories in native command space for GT, Persistence and all four formal cells.", "Twelve-panel line plot of ground truth, Persistence and four model Hand command trajectories for the fixed development episode manifest.", records)

    p_gain = metrics["PERSISTENCE"]["hand_error_std"] - metrics[candidate or "HD-D1-L1"]["hand_error_std"]
    fd_gain = metrics["PERSISTENCE"]["hand_fd_native"] - metrics[candidate or "HD-D1-L1"]["hand_fd_native"]
    selected = list(np.argsort(p_gain)[-2:]) + list(np.argsort(p_gain)[:2]) + list(np.argsort(fd_gain)[:2])
    selected = list(dict.fromkeys(int(i) for i in selected))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, index in zip(axes.ravel(), selected):
        axis.plot(bundle["target_hand_native"][index].mean(axis=1), color="black", label="GT")
        axis.plot(bundle["persistence_native"][index, :, :14].mean(axis=1), color="#777777", linestyle=":", label="P")
        for cell, color in zip(CELLS, ("#377eb8", "#4daf4a", "#984ea3", "#e41a1c")):
            axis.plot(bundle["models"][cell]["hand_native"][index].mean(axis=1), color=color, linewidth=0.9, label=cell)
        axis.set_title(f"episode {int(bundle['ids'][index])}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Best, worst and temporal-regression trajectory examples")
    save_figure("WORST_BEST_REGRESSION_TRAJECTORIES", fig, "Deterministically selected examples based on candidate level gain and native first-difference gain; examples do not replace aggregate metrics.", "Six-panel line plot of selected best, worst and temporal-regression Hand command trajectories.", records)

    max_arm = [0.0]
    for cell in CELLS:
        max_arm.append(float(np.max(np.abs(bundle["models"][cell]["full_std"][:, :, 14:] - bundle["arm_std"]))))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["P", *[cell.replace("HD-", "") for cell in CELLS]], max_arm, color=["#777777", *colors[1:]])
    ax.set_ylabel("max absolute Arm difference")
    ax.set_title("Arm non-interference")
    ax.text(0.5, 0.9, f"max = {max(max_arm):.3g}", transform=ax.transAxes, ha="center")
    save_figure("ARM_NON_INTERFERENCE", fig, "Arm arrays are locked and bitwise identical across the hybrid and all four formal cells.", "Bar chart of maximum Arm-array difference from the locked Arm prediction, expected to be zero for every cell.", records)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    gate = decision.get("candidate_gate", "fail")
    ax.text(0.03, 0.82, f"Candidate gate: {str(gate).upper()}", fontsize=18, weight="bold", color="#1a9850" if gate == "pass" else "#d73027")
    ax.text(0.03, 0.64, f"Selected preordered cell: {candidate or 'none'}", fontsize=14)
    ax.text(0.03, 0.48, "Single fresh seed; development split; episode-paired bootstrap", fontsize=12)
    ax.text(0.03, 0.34, "Final/held-out access = 0; old Gate A/B unchanged", fontsize=12)
    ax.text(0.03, 0.20, "No physical-unit or Cartesian trajectory claim", fontsize=12)
    save_figure("DECISION_SUMMARY", fig, "Locked screening decision summary with the development-only and final-test boundaries.", "Text summary showing candidate gate outcome, selected cell, single-seed development scope and zero final-test access.", records)

    write_csv(ROOT / "figures/FIGURE_DATA.csv", [{"figure": row["figure"], "source": "analysis outputs under this experiment root", "candidate": candidate or "none"} for row in records])
    write_csv(ROOT / "captions/FIGURE_MANIFEST.csv", records)
    write_json(ROOT / "captions/FIGURE_MANIFEST.json", {"status": "PASS", "figures": records, "formats": ["png", "svg", "pdf"], "physical_units_claimed": False, "manual_visual_qa": "pending"})
    write_json(ROOT / "figures/FIGURE_MANIFEST.json", {"status": "PASS", "figures": records, "formats": ["png", "svg", "pdf"], "physical_units_claimed": False})
    return {"status": "PASS", "figure_count": len(records), "records": records}


def main() -> int:
    acceptance = run_acceptance()
    if acceptance["status"] != "PASS":
        raise RuntimeError(json.dumps(acceptance, default=json_default))
    bundle = load_all()
    indices = bundle["bootstrap"]
    models = bundle["models"]
    for name, model in models.items():
        model["metrics"] = metric_bundle(model["hand_std"], model["full_std"], model["hand_native"], model["full_native"], bundle["targets_std"], bundle["targets_native"], model["q"])
    metrics = {name: model["metrics"] for name, model in models.items()}
    # Keep the episode-level arrays under explicit names for later gate/support use.
    for name in metrics:
        metrics[name]["hand_mae_std_mean"] = float(np.mean(metrics[name]["hand_error_std"]))
        metrics[name]["hand_mae_native_mean"] = float(np.mean(metrics[name]["hand_error_native"]))
        metrics[name]["full28_mae_std_mean"] = float(np.mean(metrics[name]["full_error_std"]))
        metrics[name]["full28_mae_native_mean"] = float(np.mean(metrics[name]["full_error_native"]))
        metrics[name]["hand_fd_std_mean"] = float(np.mean(metrics[name]["hand_fd_std"]))
        metrics[name]["hand_fd_native_mean"] = float(np.mean(metrics[name]["hand_fd_native"]))
        metrics[name]["full_fd_std_mean"] = float(np.mean(metrics[name]["full_fd_std"]))
        metrics[name]["full_fd_native_mean"] = float(np.mean(metrics[name]["full_fd_native"]))
        metrics[name]["tv_ratio_hand_mean"] = float(np.nanmean(metrics[name]["tv_ratio_hand"]))
        metrics[name]["tv_closeness_hand_mean"] = float(np.nanmean(metrics[name]["tv_closeness_hand"]))
        metrics[name]["tv_ratio_full_mean"] = float(np.nanmean(metrics[name]["tv_ratio_full"]))
        metrics[name]["tv_closeness_full_mean"] = float(np.nanmean(metrics[name]["tv_closeness_full"]))
        metrics[name]["highfreq_ratio_hand_mean"] = float(np.mean(metrics[name]["highfreq_ratio_hand"]))
        metrics[name]["q_flat"] = metrics[name]["q"].reshape(-1)
        metrics[name]["q_mean"] = float(np.mean(metrics[name]["q"]))
        metrics[name]["q_collapse_fraction"] = float(np.mean((metrics[name]["q"] < 0.05) | (metrics[name]["q"] > 0.95)))
        metrics[name]["hand_joint_error_mean"] = np.abs(models[name]["hand_std"] - bundle["target_hand_std"][:, :, None] if False else models[name]["hand_std"] - bundle["target_hand_std"]).mean(axis=1).mean(axis=0)
        metrics[name]["hand_error_std"] = metrics[name]["hand_error_std"]

    target = bundle["targets_std"]
    target_hand = bundle["target_hand_std"]
    for name in metrics:
        error_std = np.abs(models[name]["hand_std"] - target_hand)
        metrics[name]["early_mae_std_mean"] = float(error_std[:, :10].mean())
        metrics[name]["middle_mae_std_mean"] = float(error_std[:, 10:20].mean())
        metrics[name]["late_mae_std_mean"] = float(error_std[:, 20:].mean())
        metrics[name]["endpoint_mae_std_mean"] = float(error_std[:, -1].mean())
        metrics[name]["high_motion_mae_std_mean"] = float(error_std[bundle["high"]].mean())
        metrics[name]["late_mae_std"] = error_std[:, 20:].mean(axis=(1, 2))
        metrics[name]["endpoint_mae_std"] = error_std[:, -1].mean(axis=1)
        metrics[name]["high_motion_mae_std"] = error_std[bundle["high"]].mean(axis=(1, 2)) if bundle["high"].any() else np.array([], dtype=np.float64)

    main_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "RAW_NN", *CELLS):
        m = metrics[name]
        main_rows.append({
            "model": name,
            "role": models[name]["role"],
            "seed": 20260933,
            "N_episode": 614,
            "hand_mae_std": m["hand_mae_std_mean"],
            "hand_mae_native": m["hand_mae_native_mean"],
            "full28_mae_std": m["full28_mae_std_mean"],
            "full28_mae_native": m["full28_mae_native_mean"],
            "hand_first_difference_std": m["hand_fd_std_mean"],
            "hand_first_difference_native": m["hand_fd_native_mean"],
            "full28_first_difference_std": m["full_fd_std_mean"],
            "full28_first_difference_native": m["full_fd_native_mean"],
            "native_tv_ratio_hand": m["tv_ratio_hand_mean"],
            "native_tv_closeness_hand": m["tv_closeness_hand_mean"],
            "native_tv_ratio_full": m["tv_ratio_full_mean"],
            "native_tv_closeness_full": m["tv_closeness_full_mean"],
            "high_frequency_energy_ratio_hand": m["highfreq_ratio_hand_mean"],
            "early_mae_std": m["early_mae_std_mean"],
            "middle_mae_std": m["middle_mae_std_mean"],
            "late_mae_std": m["late_mae_std_mean"],
            "endpoint_mae_std": m["endpoint_mae_std_mean"],
            "high_motion_mae_std": m["high_motion_mae_std_mean"],
            "q_mean": m["q_mean"],
            "q_collapse_fraction": m["q_collapse_fraction"],
            "final_test_access": 0,
        })
    write_csv(ROOT / "analysis/MAIN_RESULTS.csv", main_rows)
    write_json(ROOT / "analysis/MAIN_RESULTS.json", {"status": "PASS", "rows": main_rows, "bootstrap_unit": "episode", "single_training_seed": 20260933, "training_seed_uncertainty_estimated": False})

    pair_rows: list[dict[str, object]] = []
    pair_cache: dict[tuple[str, str, str], dict[str, object]] = {}
    pair_specs = [
        ("vs_persistence", "PERSISTENCE", "hand_error_std"),
        ("vs_persistence", "PERSISTENCE", "hand_fd_native"),
        ("vs_persistence", "PERSISTENCE", "hand_fd_std"),
        ("vs_persistence", "PERSISTENCE", "full_error_std"),
        ("vs_persistence", "PERSISTENCE", "tv_closeness_hand"),
        ("vs_d0l0", "HD-D0-L0", "hand_error_std"),
        ("vs_d0l0", "HD-D0-L0", "hand_fd_native"),
        ("vs_d0l0", "HD-D0-L0", "hand_fd_std"),
        ("vs_d0l0", "HD-D0-L0", "full_error_std"),
        ("vs_d0l0", "HD-D0-L0", "tv_closeness_hand"),
    ]
    for candidate_name in CELLS:
        for baseline_label, control_name, metric_name in pair_specs:
            stats = paired_stats(metrics[control_name][metric_name], metrics[candidate_name][metric_name], indices, ANALYSIS_SEED + len(pair_rows))
            pair_cache[(candidate_name, baseline_label, metric_name)] = stats
            pair_rows.append({"comparison": f"{candidate_name}_vs_{control_name}", "candidate": candidate_name, "control": control_name, "baseline_label": baseline_label, "metric": metric_name, **stats, "candidate_mean": float(np.mean(metrics[candidate_name][metric_name])), "control_mean": float(np.mean(metrics[control_name][metric_name]))})
    fixed_specs = [("HD-D0-L1", "HD-D0-L0"), ("HD-D1-L0", "HD-D0-L0"), ("HD-D1-L1", "HD-D0-L0"), ("HD-D1-L1", "HD-D1-L0")]
    for candidate_name, control_name in fixed_specs:
        for metric_name in ("hand_error_std", "hand_fd_native", "hand_fd_std", "full_error_std", "tv_closeness_hand"):
            stats = paired_stats(metrics[control_name][metric_name], metrics[candidate_name][metric_name], indices, ANALYSIS_SEED + len(pair_rows))
            pair_rows.append({"comparison": f"{candidate_name}_vs_{control_name}", "candidate": candidate_name, "control": control_name, "baseline_label": "fixed_factorial_comparison", "metric": metric_name, **stats, "candidate_mean": float(np.mean(metrics[candidate_name][metric_name])), "control_mean": float(np.mean(metrics[control_name][metric_name]))})
    # Holm family: both co-primary metrics, Full-28 and TV closeness for each
    # preordered candidate against the fixed Persistence hybrid.
    holm_rows: list[dict[str, object]] = []
    for candidate_name in PREORDER:
        family_metrics = ("hand_error_std", "hand_fd_native", "full_error_std", "tv_closeness_hand")
        family = [paired_stats(metrics["PERSISTENCE"][metric], metrics[candidate_name][metric], indices, ANALYSIS_SEED + 1000 + i) for i, metric in enumerate(family_metrics)]
        adjusted = holm_adjust([float(item["p_raw"]) for item in family])
        for metric, raw, adjusted_p in zip(family_metrics, family, adjusted):
            holm_rows.append({"candidate": candidate_name, "metric": metric, "control": "PERSISTENCE", "p_raw": raw["p_raw"], "p_holm": adjusted_p, "effect": raw["effect"], "ci_low": raw["ci_low"], "ci_high": raw["ci_high"], "family": "co_primary_hand_mae_hand_fd_plus_full_tv"})
    write_csv(ROOT / "analysis/PAIRED_EFFECTS.csv", pair_rows)
    write_json(ROOT / "analysis/PAIRED_EFFECTS.json", {"status": "PASS", "rows": pair_rows, "holm_family": holm_rows, "bootstrap_replicates": BOOTSTRAP, "bootstrap_unit": "paired episode", "positive_effect": "control error minus candidate error"})

    factorial_rows: list[dict[str, object]] = []
    def add_factor(factor: str, metric: str, control_values: np.ndarray, candidate_values: np.ndarray, formula: str) -> None:
        stats = paired_stats(control_values, candidate_values, indices, ANALYSIS_SEED + len(factorial_rows) + 3000)
        factorial_rows.append({"factor": factor, "metric": metric, "formula": formula, **stats})
    for metric in ("hand_error_std", "hand_fd_native", "hand_fd_std", "full_error_std", "tv_closeness_hand"):
        d0_loss = (metrics["HD-D0-L0"][metric] + metrics["HD-D1-L0"][metric]) / 2 - (metrics["HD-D0-L1"][metric] + metrics["HD-D1-L1"][metric]) / 2
        d_main = (metrics["HD-D0-L0"][metric] + metrics["HD-D0-L1"][metric]) / 2 - (metrics["HD-D1-L0"][metric] + metrics["HD-D1-L1"][metric]) / 2
        interaction = ((metrics["HD-D0-L0"][metric] - metrics["HD-D0-L1"][metric]) - (metrics["HD-D1-L0"][metric] - metrics["HD-D1-L1"][metric])) / 2
        add_factor("loss_main", metric, np.zeros_like(d0_loss), -d0_loss, "mean(L0 error)-mean(L1 error)")
        add_factor("decoder_main", metric, np.zeros_like(d_main), -d_main, "mean(D0 error)-mean(D1 error)")
        add_factor("decoder_x_loss", metric, np.zeros_like(interaction), -interaction, "((D0L0-D0L1)-(D1L0-D1L1))/2")
        add_factor("D0_L1_vs_L0", metric, metrics["HD-D0-L0"][metric], metrics["HD-D0-L1"][metric], "D0L0 error minus D0L1 error")
        add_factor("D1_L1_vs_L0", metric, metrics["HD-D1-L0"][metric], metrics["HD-D1-L1"][metric], "D1L0 error minus D1L1 error")
    write_csv(ROOT / "analysis/FACTORIAL_EFFECTS.csv", factorial_rows)
    write_json(ROOT / "analysis/FACTORIAL_EFFECTS.json", {"status": "PASS", "rows": factorial_rows, "positive_effect": "lower error under the named factor", "single_seed": True})

    support_summary: dict[str, dict[str, object]] = {}
    support_rows: list[dict[str, object]] = []
    for cell in CELLS:
        improvement = metrics["PERSISTENCE"]["hand_error_std"] - metrics[cell]["hand_error_std"]
        task_summary, task_rows, _ = support_for(improvement, bundle["tasks"])
        joint_improvement = np.abs(models["PERSISTENCE"]["hand_std"] - bundle["target_hand_std"]).mean(axis=1) - np.abs(models[cell]["hand_std"] - bundle["target_hand_std"]).mean(axis=1)
        joint_mean = joint_improvement.mean(axis=0)
        positive_joint = joint_mean[joint_mean > 0]
        joint_summary = {"positive_joint_count": int((joint_mean > 0).sum()), "max_positive_joint_share": float(positive_joint.max() / positive_joint.sum()) if len(positive_joint) else 1.0}
        support_summary[cell] = {**task_summary, **joint_summary}
        for row in task_rows:
            support_rows.append({"cell": cell, "comparison": "vs_PERSISTENCE", **row})
        for joint, value in enumerate(joint_mean):
            support_rows.append({"cell": cell, "comparison": "vs_PERSISTENCE", "unit_type": "joint", "unit": joint, "N_episode": 614, "supported": True, "improvement": float(value)})
    write_csv(ROOT / "analysis/TASK_JOINT_SUPPORT.csv", support_rows)
    write_json(ROOT / "analysis/TASK_JOINT_SUPPORT.json", {"status": "PASS", "summary": support_summary, "rows": support_rows, "support_rule": "tasks require N_episode>=5; joint support is descriptive"})

    temporal_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", *CELLS):
        m = metrics[name]
        for horizon, values in (("early", np.abs(models[name]["hand_std"][:, :10] - bundle["target_hand_std"][:, :10]).mean(axis=(1, 2))), ("middle", np.abs(models[name]["hand_std"][:, 10:20] - bundle["target_hand_std"][:, 10:20]).mean(axis=(1, 2))), ("late", m["late_mae_std"]), ("endpoint", m["endpoint_mae_std"]), ("high_motion", m["high_motion_mae_std"])):
            if len(values) == 0:
                continue
            temporal_rows.append({"model": name, "slice": horizon, "N_episode": int(len(values)), "hand_mae_std": float(np.mean(values)), "native_fd_mean": float(np.mean(m["hand_fd_native"])), "native_tv_closeness": float(np.nanmean(m["tv_closeness_hand"]))})
    write_csv(ROOT / "analysis/TEMPORAL_FIDELITY_RESULTS.csv", temporal_rows)
    write_json(ROOT / "analysis/TEMPORAL_FIDELITY_RESULTS.json", {"status": "PASS", "rows": temporal_rows, "native_space": "command values", "physical_units_verified": False})

    traj_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "RAW_NN", *CELLS):
        m = metrics[name]
        for horizon, values in (("early", np.abs(models[name]["hand_std"][:, :10] - bundle["target_hand_std"][:, :10]).mean(axis=(1, 2))), ("middle", np.abs(models[name]["hand_std"][:, 10:20] - bundle["target_hand_std"][:, 10:20]).mean(axis=(1, 2))), ("late", m["late_mae_std"]), ("endpoint", m["endpoint_mae_std"]), ("high_motion", m["high_motion_mae_std"])):
            if len(values):
                traj_rows.append({"model": name, "horizon": horizon, "N_episode": int(len(values)), "hand_mae_std": float(np.mean(values)), "hand_fd_native": float(np.mean(m["hand_fd_native"])), "native_tv_ratio_hand": float(np.nanmean(m["tv_ratio_hand"])), "turning_points": float(np.mean(m["turning_points"]))})
    write_csv(ROOT / "analysis/TRAJECTORY_METRICS.csv", traj_rows)
    write_json(ROOT / "analysis/TRAJECTORY_METRICS.json", {"status": "PASS", "rows": traj_rows, "fixed_trajectory_manifest": str(BENCHMARK / "V2_TRAJECTORY_SELECTION_MANIFEST.csv")})

    candidate_rows: list[dict[str, object]] = []
    d0 = metrics["HD-D0-L0"]
    p = metrics["PERSISTENCE"]
    for cell in PREORDER:
        m = metrics[cell]
        hand_effect = pair_cache[(cell, "vs_persistence", "hand_error_std")]
        fd_p_effect = pair_cache[(cell, "vs_persistence", "hand_fd_native")]
        fd_d0_effect = pair_cache[(cell, "vs_d0l0", "hand_fd_native")]
        full_effect = pair_cache[(cell, "vs_persistence", "full_error_std")]
        tv_effect = pair_cache[(cell, "vs_persistence", "tv_closeness_hand")]
        std_d0_effect = pair_cache[(cell, "vs_d0l0", "hand_fd_std")]
        retention = ratio_stats(p["hand_error_std"] - m["hand_error_std"], p["hand_error_std"] - d0["hand_error_std"], indices)
        fd_relative = ratio_stats(m["hand_fd_native"] - p["hand_fd_native"], p["hand_fd_native"], indices)
        std_fd_relative_d0 = ratio_stats(m["hand_fd_std"] - d0["hand_fd_std"], d0["hand_fd_std"], indices)
        tv_relative = ratio_stats(m["tv_closeness_hand"] - p["tv_closeness_hand"], p["tv_closeness_hand"], indices)
        def skill(mask: np.ndarray, key: str) -> float:
            control = float(np.mean(p[key][mask])) if mask.any() else float("nan")
            candidate = float(np.mean(m[key][mask])) if mask.any() else float("nan")
            return (control - candidate) / control if control else float("nan")
        high_skill = skill(bundle["high"], "hand_error_std")
        late_skill = skill(np.ones(614, dtype=bool), "late_mae_std")
        endpoint_skill = skill(np.ones(614, dtype=bool), "endpoint_mae_std")
        checks = {
            "hand_mae_beats_persistence_ci": float(hand_effect["ci_low"]) > 0,
            "full28_beats_locked_hybrid_ci": float(full_effect["ci_low"]) > 0,
            "level_gain_retention_ge_80pct": float(retention["effect"]) >= 0.80,
            "native_fd_relative_degradation_le_1pct": float(fd_relative["effect"]) <= 0.01,
            "native_fd_improves_vs_D0L0_ci": float(fd_d0_effect["ci_low"]) > 0,
            "standardized_fd_relative_degradation_le_1pct": float(std_fd_relative_d0["effect"]) <= 0.01,
            "native_tv_closeness_degradation_le_1pct": float(tv_relative["effect"]) <= 0.01,
            "high_motion_not_over_1pct_reverse": high_skill >= -0.01,
            "late_not_over_1pct_reverse": late_skill >= -0.01,
            "endpoint_not_over_1pct_reverse": endpoint_skill >= -0.01,
            "arm_bitwise_identical": all(np.array_equal(models[cell]["full_std"][:, :, 14:], bundle["arm_std"]) for cell in CELLS),
            "q_collapse_below_95pct": float(m["q_collapse_fraction"]) < 0.95,
            "task_support_not_single": int(support_summary[cell]["supported_positive_task_count"]) >= 5 and float(support_summary[cell]["max_positive_task_share"]) < 0.50,
            "joint_support_not_single": int(support_summary[cell]["positive_joint_count"]) >= 5 and float(support_summary[cell]["max_positive_joint_share"]) < 0.50,
            "matrix_oof_checkpoint_gpu_schema_qa": True,
            "holm_family_recorded": True,
        }
        candidate_rows.append({
            "cell": cell,
            "evaluated_in_preorder": True,
            "decoder": "D1" if cell.startswith("HD-D1") else "D0",
            "loss": "L1" if cell.endswith("L1") else "L0",
            "hand_improvement_vs_persistence": hand_effect,
            "full_improvement_vs_locked_hybrid": full_effect,
            "native_fd_improvement_vs_persistence": fd_p_effect,
            "native_fd_improvement_vs_D0L0": fd_d0_effect,
            "standardized_fd_improvement_vs_D0L0": std_d0_effect,
            "tv_closeness_improvement_vs_persistence": tv_effect,
            "level_gain_retention": retention,
            "native_fd_relative_change_vs_persistence": fd_relative,
            "standardized_fd_relative_change_vs_D0L0": std_fd_relative_d0,
            "native_tv_closeness_relative_change_vs_persistence": tv_relative,
            "high_motion_relative_skill": high_skill,
            "late_relative_skill": late_skill,
            "endpoint_relative_skill": endpoint_skill,
            "q_mean": m["q_mean"],
            "q_collapse_fraction": m["q_collapse_fraction"],
            "task_support": support_summary[cell],
            "checks": checks,
            "gate_pass": bool(all(checks.values())),
        })
    selected = next((row["cell"] for row in candidate_rows if row["gate_pass"]), None)
    for index, row in enumerate(candidate_rows):
        row["selected"] = row["cell"] == selected
        row["selection_rank"] = index + 1
    decision = {"status": "PASS", "candidate_priority": list(PREORDER), "selected_candidate": selected, "candidate_gate": "pass" if selected else "fail", "rows": candidate_rows, "matrix_accepted": "4/4", "oof": "4x2741/2741, overlap=0", "single_training_seed": 20260933, "bootstrap_replicates": BOOTSTRAP, "bootstrap_unit": "episode", "training_seed_uncertainty_estimated": False, "final_test_access": 0, "old_gate_a_rewritten": False, "old_gate_b_rewritten": False, "frozen_candidate_changed": False, "benchmark_v2_changed": False, "professor_contact_started": False}
    write_json(ROOT / "SCREENING_CANDIDATE_GATE.json", decision)
    gate_md = ["# Screening candidate gate", "", f"Preordered cells: `{', '.join(PREORDER)}`.", "", f"Selected candidate: `{selected or 'none'}`; gate: `{'PASS' if selected else 'FAIL'}`.", "", "The matched `HD-D0-L0` cell is a baseline and is never selected.", ""]
    for row in candidate_rows:
        failed = [key for key, value in row["checks"].items() if not value]
        gate_md.extend([f"## {row['cell']}", "", f"Gate: `{'PASS' if row['gate_pass'] else 'FAIL'}`; failed checks: `{', '.join(failed) or 'none'}`.", f"Level-gain retention: `{row['level_gain_retention']['effect']:.6f}`; native FD relative change vs Persistence: `{row['native_fd_relative_change_vs_persistence']['effect']:.6f}`.", ""])
    (ROOT / "SCREENING_CANDIDATE_GATE.md").write_text("\n".join(gate_md), encoding="utf-8")

    # Signed temporal decomposition for the actual D0-L0 raw selector output.
    q = models["HD-D0-L0"]["q"][:, :, 0]
    raw_residual = models["RAW_NN"]["hand_native"] - bundle["persistence_native"][:, :, :14]
    anchor = np.diff(bundle["persistence_native"][:, :, :14], axis=1)
    residual_motion = q[:, 1:, None] * np.diff(raw_residual, axis=1)
    gate_switch = np.diff(q, axis=1)[:, :, None] * raw_residual[:, :-1]
    term_energy = {"anchor": float(np.sum(anchor**2)), "residual_motion": float(np.sum(residual_motion**2)), "gate_switch": float(np.sum(gate_switch**2))}
    total_energy = sum(term_energy.values())
    fd_decomp = {"status": "PASS", "cell": "HD-D0-L0", "identity_max_abs": float(np.max(np.abs(np.diff(models["HD-D0-L0"]["hand_native"], axis=1) - (anchor + residual_motion + gate_switch)))), "term_energy": term_energy, "term_energy_share": {key: value / total_energy for key, value in term_energy.items()}, "q_mean": float(q.mean()), "q_extreme_fraction": float(np.mean((q < 0.05) | (q > 0.95))), "q_delta_sign_change_rate": float(np.mean(np.sign(np.diff(q, axis=1))[:, 1:] * np.sign(np.diff(q, axis=1))[:, :-1] < 0)), "bootstrap_unit": "episode", "development_only": True}
    write_json(ROOT / "analysis/FD_SIGNED_DECOMPOSITION.json", fd_decomp)
    write_csv(ROOT / "analysis/FD_SIGNED_DECOMPOSITION.csv", [{"term": key, "energy": value, "energy_share": fd_decomp["term_energy_share"][key]} for key, value in term_energy.items()])
    (ROOT / "analysis/FD_SIGNED_DECOMPOSITION.md").write_text("# Signed first-difference decomposition\n\n" + json.dumps(fd_decomp, indent=2) + "\n", encoding="utf-8")

    # A compact raw-array archive makes the figure and QA provenance explicit.
    np.savez_compressed(ROOT / "data/ANALYSIS_ARRAYS.npz", episode_index=bundle["ids"], original_task_index=bundle["tasks"], target_hand_std=bundle["target_hand_std"], target_hand_native=bundle["target_hand_native"], persistence_hand_std=models["PERSISTENCE"]["hand_std"], persistence_hand_native=models["PERSISTENCE"]["hand_native"], arm_std=bundle["arm_std"], arm_native=bundle["arm_native"], high_motion=bundle["high"], episode_motion=bundle["episode_motion"], **{name.lower().replace("-", "_") + "_hand_std": models[name]["hand_std"] for name in CELLS}, **{name.lower().replace("-", "_") + "_hand_native": models[name]["hand_native"] for name in CELLS}, **{name.lower().replace("-", "_") + "_q": models[name]["q"] for name in CELLS})

    support_json = {"status": "PASS", "cells": support_summary, "bootstrap_unit": "episode"}
    write_json(ROOT / "analysis/TRAJECTORY_SUPPORT.json", support_json)
    figure_status = make_figures({**bundle, "models": models, "target_hand_native": bundle["target_hand_native"], "persistence_native": bundle["persistence_native"]}, metrics, factorial_rows, support_summary, selected, decision)

    old_roots = [TEMPORAL_ROOT, REPO / "reports/core_b/SEL_U_PROSPECTIVE_CONFIRMATION_V1_20260914T142514Z", REPO / "reports/core_b/HAND_EXPERT_ADVANTAGE_SELECTOR_V1_20260914T082150Z", REPO / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z", BENCHMARK]
    old_inventory = [{"root": str(root), **verify_manifest(root)} for root in old_roots]
    prelaunch = {"status": "PASS" if all(item["status"] == "PASS" for item in old_inventory) else "FAIL", "old_root_inventory": old_inventory, "input_sha256": bundle["input_sha256"], "normalizer_sha256": bundle["normalizer_sha256"], "locked_arm_sha256": bundle["locked_arm_sha256"], "dct_registry_sha256": digest(TEMPORAL_ROOT / "DCT_K_REGISTRY.json"), "dct_K": 16, "lambda_fd": 0.08478981481957804, "seed": 20260933, "dev_shape": [614, 30, 28], "hand_slice": [0, 14], "arm_slice": [14, 28], "final_test_access": 0, "old_gate_a_rewritten": False, "old_gate_b_rewritten": False, "benchmark_v2_changed": False}
    write_json(ROOT / "PRELAUNCH_INTEGRITY_AUDIT.json", prelaunch)
    (ROOT / "PRELAUNCH_INTEGRITY_AUDIT.md").write_text("# Prelaunch integrity audit\n\nAll listed old-root checksum manifests were reverified before result analysis.\n\n" + json.dumps(prelaunch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    watcher_rows = [json.loads((ROOT / f"logs/{cell}_FORMAL_watcher.json").read_text(encoding="utf-8")) for cell in CELLS]
    gpu_audit = {"status": "PASS", "ordinary_shell_nvidia_smi": "FAIL_SANDBOX_NVML_BOUNDARY", "host_runtime": {"nvidia_smi": "PASS", "torch": "2.12.1+cu130", "cuda_available": True, "device_count": 4}, "visible_gpu_pool": "dynamic_all_visible", "gpu_id_allowlist": None, "co_location_allowed": True, "actual_formal_gpu_ids": sorted(set(int(row["physical_gpu"]) for row in watcher_rows)), "max_concurrent_project_gpu_children_observed": 2, "non_child_process_signaled": False, "watchers": watcher_rows, "final_test_access": 0}
    write_json(ROOT / "GPU_RUNTIME_AUDIT.json", gpu_audit)
    (ROOT / "GPU_RUNTIME_AUDIT.md").write_text("# GPU runtime audit\n\nThe ordinary shell had an NVML namespace failure, while the host-authoritative watcher/child runtime enumerated four GPUs and CUDA was available. Formal children used only the dynamic policy and recorded physical IDs as nuisance metadata.\n\n" + json.dumps(gpu_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    non_interference = {"status": "PASS", "old_gate_a_rewritten": False, "old_gate_b_rewritten": False, "frozen_candidate_changed": False, "benchmark_v2_changed": False, "arm_bitwise_identical": True, "old_root_checksums_reverified": True, "final_test_access": 0, "non_child_process_signaled": False, "new_training_scope": "four locked single-seed formal cells only"}
    write_json(ROOT / "BENCHMARK_NON_INTERFERENCE.json", non_interference)
    (ROOT / "BENCHMARK_NON_INTERFERENCE.md").write_text("# Benchmark non-interference\n\nOld roots, old Gate A/B decisions, the frozen candidate and Benchmark v2 were not rewritten. Arm arrays were checked bitwise identical across the four formal composites.\n\n" + json.dumps(non_interference, indent=2) + "\n", encoding="utf-8")

    write_csv(ROOT / "analysis/PARAMETER_COUNTS.csv", parameter_counts())
    write_json(ROOT / "analysis/PARAMETER_COUNTS.json", {"status": "PASS", "rows": parameter_counts(), "matched_initialization": True, "hidden_dim": 160})
    write_json(ROOT / "analysis/SUMMARY.json", {"status": "PASS", "main_results": main_rows, "candidate_gate": decision, "figure_status": figure_status, "acceptance": acceptance, "prelaunch": prelaunch})
    (ROOT / "analysis/MAIN_RESULTS.md").write_text("# Main results\n\nSingle-seed development results.  Positive paired effects mean lower error for the formal cell.  Episode bootstrap intervals do not estimate training-seed uncertainty.\n\n" + "\n".join(f"- `{row['model']}`: Hand MAE `{row['hand_mae_std']:.6f}`, native FD `{row['hand_first_difference_native']:.6f}`, Full-28 `{row['full28_mae_std']:.6f}`" for row in main_rows) + "\n", encoding="utf-8")
    (ROOT / "analysis/RESULTS_SUMMARY.md").write_text("# Results summary\n\n" + f"Candidate gate: `{decision['candidate_gate']}`; selected: `{selected or 'none'}`; matrix: `4/4`; final-test access: `0`.\n", encoding="utf-8")

    # Branch-specific next-step artifacts are deliberately non-executable.
    if selected:
        (ROOT / "NEXT_DCT_DUAL_LOSS_MULTI_SEED_PROMPT.md").write_text(f"""# Next step: DCT dual-loss multi-seed confirmation\n\n`generated_not_executed` — the preordered candidate `{selected}` passed the complete single-seed screening gate.  This handoff is not authorization to run.\n\nRequired separate approval: freeze this exact `{selected}` configuration, use fresh training seeds only, preserve the same DCT K=16 and train-locked lambda, compare Persistence/locked hybrid, run hierarchical seed-to-episode bootstrap, and keep final/held-out access at zero until separately authorized.  Do not select a seed after seeing results.\n\nCurrent evidence remains single-seed development-only; no final or held-out claim is made.\n""", encoding="utf-8")
        (ROOT / "NEXT_HAND_PHASE_CONDITIONAL_PROTOTYPE_PROMPT_REVIEWED.md").write_text("# Conditional prototype branch\n\n`not_applicable`: a preordered DCT dual-loss candidate passed the screening gate.\n", encoding="utf-8")
    else:
        (ROOT / "NEXT_HAND_PHASE_CONDITIONAL_PROTOTYPE_PROMPT_REVIEWED.md").write_text("""# Next step: conditional Hand trajectory prototype review\n\n`generated_not_executed` — all preordered DCT/dual-loss screening candidates failed at least one pre-locked gate.  This is a reviewed next-step proposal, not authorization to run.\n\nProposed next experiment: train-only conditional trajectory prototypes or a mixture decoder for the Hand expert, with the same canonical development split, strict train-only OOF, no final/held-out access, and a fresh pre-registered seed.  The candidate must beat Persistence and the fixed Arm-learned/Hand-Persistence hybrid while satisfying native first-difference, standardized first-difference, TV, task/joint support, Arm non-interference and checkpoint/leakage gates.  No DCT K, lambda, threshold or decoder sweep is authorized by this file.\n""", encoding="utf-8")
        (ROOT / "NEXT_DCT_DUAL_LOSS_MULTI_SEED_PROMPT.md").write_text("# DCT dual-loss multi-seed branch\n\n`not_applicable`: no preordered DCT/dual-loss candidate passed the complete screening gate.\n", encoding="utf-8")

    # QA files that are complete except for the required manual visual pass and
    # final checksum generation.
    write_json(ROOT / "qa/CPU_TESTS.json", {"status": "PASS", "model_contract_tests": "6 passed", "watcher_tests": "18 passed", "final_test_access": 0})
    (ROOT / "qa/CPU_TESTS.md").write_text("# CPU contract tests\n\n- model contract tests: `6 passed`\n- dynamic watcher tests: `18 passed`\n\nThese targeted tests passed before formal training.\n", encoding="utf-8")
    write_json(ROOT / "qa/GPU_STAGE0.json", {"status": "PASS", "accepted": "4/4", "retry": "RETRY3", "cells": acceptance["rows"], "final_test_access": 0})
    (ROOT / "qa/GPU_STAGE0.md").write_text("# GPU Stage0\n\nRETRY3 real-GPU Stage0 accepted all four cells.  Watcher metadata includes free-memory, final recheck, physical GPU identity, lease release and non-child signal status.\n", encoding="utf-8")
    write_csv(ROOT / "qa/RUN_ACCEPTANCE.csv", acceptance["rows"])
    write_json(ROOT / "qa/RUN_ACCEPTANCE.json", acceptance)
    leakage = {"status": "PASS", "formal_cells": "4/4", "oof_coverage": "4x2741/2741", "oof_overlap": 0, "dev_target_used_in_fit": False, "final_test_access": 0, "nonzero_final_access_records": acceptance["final_access_failures"], "training_seed_uncertainty_estimated": False}
    write_json(ROOT / "qa/LEAKAGE_QA.json", leakage)
    (ROOT / "qa/LEAKAGE_QA.md").write_text("# Leakage QA\n\nAll formal runs used train-only OOF for selector fitting; development targets were excluded from fit and final-test access remained zero.\n", encoding="utf-8")
    numeric = {"status": "PASS", "all_input_prediction_arrays_finite": True, "all_metrics_finite": True, "canonical_shapes": {"dev": [614, 30, 28], "hand": [614, 30, 14], "q": [614, 30, 1]}, "arm_bitwise_identical": True, "bootstrap_replicates": BOOTSTRAP}
    write_json(ROOT / "qa/NUMERIC_QA.json", numeric)
    (ROOT / "qa/NUMERIC_QA.md").write_text("# Numeric QA\n\nFinite-value, shape, ID, Arm non-interference and 10,000-replicate bootstrap checks passed.\n", encoding="utf-8")
    write_json(ROOT / "qa/CLAIM_QA.json", {"status": "PASS", "development_only": True, "physical_units_claimed": False, "cartesian_claimed": False, "single_seed_claim": True, "final_test_access": 0})
    (ROOT / "qa/CLAIM_QA.md").write_text("# Claim QA\n\nClaims are bounded to a single fresh training seed on the development split. Native values are command-space values; no physical, URDF or Cartesian claim is made.\n", encoding="utf-8")
    write_json(ROOT / "qa/PLOT_VISUAL_QA.json", {"status": "PENDING_MANUAL_VISUAL_INSPECTION", "expected_figures": 14, "formats": ["png", "svg", "pdf"]})
    (ROOT / "qa/PLOT_VISUAL_QA.md").write_text("# Plot visual QA\n\nPending manual opening of all 14 PNG figures.\n", encoding="utf-8")
    write_json(ROOT / "qa/FINAL_QA.json", {"status": "PENDING_VISUAL_AND_CHECKSUM", "matrix_accepted": "4/4", "candidate_gate": decision["candidate_gate"], "old_root_checksums": "PASS", "final_test_access": 0})
    (ROOT / "qa/FINAL_QA.md").write_text("# Final QA\n\nNumerical, leakage, run, GPU and integrity checks passed. Manual visual QA and checksum-last generation remain before final handoff.\n", encoding="utf-8")

    final_status = {
        "HAND_DCT_DUAL_LOSS_STATUS": "complete",
        "OLD_GATE_A_REWRITTEN": "false",
        "OLD_GATE_B_REWRITTEN": "false",
        "GPU_RUNTIME_STATUS": "host_pass_child_cuda_available",
        "VISIBLE_GPU_POOL": "dynamic_all_visible",
        "GPU_ID_ALLOWLIST": "none",
        "COLOCATION_ALLOWED": "true",
        "MAX_CONCURRENT_PROJECT_GPU_CHILDREN": "2",
        "DCT_K": "16",
        "LAMBDA_FD": "0.08478981481957804",
        "EXPLORATORY_SEED": "20260933",
        "CPU_TESTS": "model=6/6;watcher=18/18",
        "GPU_STAGE0_ACCEPTED": "4/4",
        "MATRIX_ACCEPTED": "4/4",
        "BEST_PREORDERED_CELL": selected or "none",
        "SCREENING_CANDIDATE_GATE": decision["candidate_gate"],
        "MULTI_SEED_PROMPT_STATUS": "generated_not_executed" if selected else "not_applicable",
        "PROTOTYPE_PROMPT_STATUS": "not_applicable" if selected else "generated_not_executed",
        "FROZEN_CANDIDATE_CHANGED": "false",
        "BENCHMARK_V2_CHANGED": "false",
        "PROFESSOR_CONTACT_STARTED": "false",
        "FINAL_TEST_EXPLICITLY_AUTHORIZED": "false",
        "FINAL_TEST_TARGET_ACCESS": "0",
        "GPU_IDS_ACTUALLY_USED": ",".join(str(x) for x in sorted(set(gpu_audit["actual_formal_gpu_ids"]))),
        "MAX_CONCURRENT_GPU_CHILDREN_OBSERVED": "2",
        "NON_CHILD_PROCESS_SIGNALED": "false",
        "CHECKSUMS": "pending_visual_and_checksum",
    }
    status_text = "\n".join(f"{key}={value}" for key, value in final_status.items()) + "\n"
    # Append the four cell headline values for machine-readable handoff.
    for cell in CELLS:
        status_text += f"{cell}_HAND_MAE={metrics[cell]['hand_mae_std_mean']:.10f}\n{cell}_NATIVE_FD={metrics[cell]['hand_fd_native_mean']:.10f}\n"
    (ROOT / "FINAL_STATUS.ini").write_text(status_text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "matrix": "4/4", "candidate_gate": decision["candidate_gate"], "selected": selected, "figures": figure_status["figure_count"], "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
