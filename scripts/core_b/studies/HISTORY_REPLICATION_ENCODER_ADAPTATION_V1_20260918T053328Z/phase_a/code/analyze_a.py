#!/usr/bin/env python3
"""Replay the 12 accepted Phase-A checkpoints and evaluate the locked A gate.

The source study's fixed CPU inference implementation is imported read-only
after its model/data hashes are checked.  This wrapper owns the fresh-seed
join, inference artifacts, seed-level summaries, crossed seed×episode
resampling, guardrails, and the conditional B decision.
"""

from __future__ import annotations
import os

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
ROOT = REPO / "reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z"
SOURCE_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
SOURCE_ANALYSIS = SOURCE_ROOT / "code/run_dev_inference_and_analysis.py"
SOURCE_MODEL = SOURCE_ROOT / "code/model.py"
LOCAL_MODEL = ROOT / "phase_a/code/model.py"
INPUT = SOURCE_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
BOOTSTRAP_SOURCE = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/paired_episode_bootstrap_indices.npy"
MOTION_STRATA = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_STRATA_MANIFEST.csv"
SEEDS = (20260940, 20260941, 20260942)
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
CANDIDATE = "HI-F-HIST"
HAND = 14
STEPS = 30
HIER_REPLICATES = 20000
EPISODE_REPLICATES = 10000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def import_source_analysis() -> Any:
    code_dir = str(SOURCE_ANALYSIS.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("locked_source_hvi_analysis", SOURCE_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import locked source analysis")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_motion_strata(dev_ids: np.ndarray) -> np.ndarray:
    rows = list(csv.DictReader(MOTION_STRATA.open(encoding="utf-8", newline="")))
    by_id = {int(row["episode_id"]): str(row["motion_stratum"]) for row in rows}
    values = np.asarray([by_id.get(int(episode), "") for episode in dev_ids], dtype=object)
    if np.any(values == ""):
        raise RuntimeError("motion strata do not cover all dev IDs")
    return values


def episode_bootstrap(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    difference = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    if difference.shape != (indices.shape[1],) or not np.isfinite(difference).all():
        raise ValueError(f"episode bootstrap shape/finite failed: {difference.shape}")
    estimates = difference[indices].mean(axis=1)
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = difference
    for start in range(0, len(indices), 500):
        size = min(500, len(indices) - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(valid)))
        exceed += int(np.sum(np.abs((signs * valid[None]).mean(axis=1)) >= abs(valid.mean())))
    return {
        "effect_control_minus_candidate": float(valid.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "win_rate_candidate_better": float(np.mean(valid > 0)),
        "p_signflip": float((1 + exceed) / (len(indices) + 1)),
        "N_episode": int(len(valid)),
        "replicates": int(len(indices)),
    }


def hierarchical_effect_from_diff(diff_by_seed: np.ndarray, *, seed: int) -> dict[str, object]:
    """Crossed seed×episode bootstrap for a control-minus-candidate effect."""
    diff = np.asarray(diff_by_seed, dtype=np.float64)
    if diff.ndim != 2 or diff.shape[0] != 3 or not np.isfinite(diff).all():
        raise ValueError(f"hierarchical diff shape/finite failed: {diff.shape}")
    n_episodes = diff.shape[1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(HIER_REPLICATES, dtype=np.float64)
    for start in range(0, HIER_REPLICATES, 256):
        stop = min(start + 256, HIER_REPLICATES)
        size = stop - start
        chosen_seeds = rng.integers(0, 3, size=(size, 3))
        total = np.zeros(size, dtype=np.float64)
        for slot in range(3):
            episode_indices = rng.integers(0, n_episodes, size=(size, n_episodes))
            total += diff[chosen_seeds[:, slot][:, None], episode_indices].mean(axis=1)
        estimates[start:stop] = total / 3.0
    lower = (int(np.sum(estimates <= 0)) + 1) / (HIER_REPLICATES + 1)
    upper = (int(np.sum(estimates >= 0)) + 1) / (HIER_REPLICATES + 1)
    return {
        "effect_control_minus_candidate": float(diff.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min(lower, upper))),
        "replicates": HIER_REPLICATES,
        "seed_level": 3,
        "episode_level": n_episodes,
        "resampling": "three seeds with replacement, then 614 episodes with replacement within each selected seed slot",
    }, estimates


def hierarchical_ratio(candidate_by_seed: np.ndarray, control_by_seed: np.ndarray, *, seed: int) -> dict[str, object]:
    candidate = np.asarray(candidate_by_seed, dtype=np.float64)
    control = np.asarray(control_by_seed, dtype=np.float64)
    if candidate.ndim != 2 or control.ndim != 2 or candidate.shape[0] != 3 or control.shape != candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(control).all():
        raise ValueError("hierarchical ratio input shape/finite failed")
    n_episodes = candidate.shape[1]
    rng = np.random.default_rng(seed)
    estimates = np.empty(HIER_REPLICATES, dtype=np.float64)
    for start in range(0, HIER_REPLICATES, 256):
        stop = min(start + 256, HIER_REPLICATES)
        size = stop - start
        chosen_seeds = rng.integers(0, 3, size=(size, 3))
        numerator = np.zeros(size, dtype=np.float64)
        denominator = np.zeros(size, dtype=np.float64)
        for slot in range(3):
            episode_indices = rng.integers(0, n_episodes, size=(size, n_episodes))
            numerator += candidate[chosen_seeds[:, slot][:, None], episode_indices].mean(axis=1)
            denominator += control[chosen_seeds[:, slot][:, None], episode_indices].mean(axis=1)
        estimates = estimates.astype(np.float64, copy=False)
        estimates[start:stop] = (numerator / denominator) / 3.0 * 3.0 - 1.0
    return {
        "relative_change_candidate_vs_control": float(candidate.mean() / control.mean() - 1.0),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": HIER_REPLICATES,
        "seed_level": 3,
        "episode_level": n_episodes,
        "resampling": "three seeds with replacement, then 614 episodes with replacement within each selected seed slot",
    }, estimates


def bootstrap_joint_rows(control_by_seed: np.ndarray, candidate_by_seed: np.ndarray, *, seed: int) -> tuple[dict[str, object], np.ndarray]:
    return hierarchical_effect_from_diff(np.asarray(control_by_seed) - np.asarray(candidate_by_seed), seed=seed)


def holm(rows: list[dict[str, object]], key: str = "p_two_sided") -> None:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i][key]))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(rows[index][key]) * (len(rows) - rank)))
        rows[index]["holm_p"] = running


def mean_metric(metric: dict[str, np.ndarray], key: str) -> float:
    return float(np.nanmean(metric[key]))


def main() -> int:
    analysis_dir = ROOT / "phase_a/analysis"
    qa_dir = ROOT / "phase_a/qa"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    source = import_source_analysis()
    if digest(SOURCE_MODEL) != digest(LOCAL_MODEL):
        raise RuntimeError("local Phase-A model differs from locked source model")
    data, stats = source.load_data()
    train_positions = np.flatnonzero(data["split_code"] == 0)
    dev_positions = np.flatnonzero(data["split_code"] == 1)
    if len(train_positions) != 2741 or len(dev_positions) != 614:
        raise RuntimeError("train/dev population failed during analysis")
    dev_ids = data["episode_index"][dev_positions].astype(np.int64)
    dev_tasks = data["original_task_index"][dev_positions].astype(np.int64)
    locked = {key: value.copy() for key, value in np.load(LOCKED_ARM, allow_pickle=False).items()}
    if not np.array_equal(locked["episode_index"], dev_ids) or locked["predictions_native"].shape != (614, STEPS, 28):
        raise RuntimeError("locked Arm dev ID/shape seam failed")
    arm_native = locked["predictions_native"][:, :, 14:].astype(np.float32)
    arm_std = locked["predictions_standardized"][:, :, 14:].astype(np.float32)
    target_native = data["targets_native"][dev_positions].astype(np.float32)
    target_std = data["targets_standardized"][dev_positions].astype(np.float32)
    persistence_native = data["persistence_native"][dev_positions].astype(np.float32)
    persistence_std = data["persistence_standardized"][dev_positions].astype(np.float32)
    anchor = persistence_native[:, 0, :HAND]
    motion_strata = load_motion_strata(dev_ids)
    bootstrap = np.load(BOOTSTRAP_SOURCE, allow_pickle=False)
    if bootstrap.shape != (EPISODE_REPLICATES, 614) or bootstrap.min() < 0 or bootstrap.max() >= 614:
        raise RuntimeError(f"source episode bootstrap shape failed: {bootstrap.shape}")
    np.save(analysis_dir / "EPISODE_BOOTSTRAP_INDICES_SOURCE.npy", bootstrap)

    metrics_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    prediction_records: list[dict[str, object]] = []
    for seed in SEEDS:
        per_seed: dict[str, dict[str, np.ndarray]] = {}
        persistence_full_native = persistence_native.copy()
        persistence_full_std = persistence_std.copy()
        per_seed["PERSISTENCE"] = source.metric_arrays(persistence_full_native, persistence_full_std, target_native, target_std, anchor)
        fixed_native = np.concatenate((persistence_native[:, :, :HAND], arm_native), axis=2)
        fixed_std = np.concatenate((persistence_std[:, :, :HAND], arm_std), axis=2)
        per_seed["FIXED_HYBRID"] = source.metric_arrays(fixed_native, fixed_std, target_native, target_std, anchor)
        for cell in CELLS:
            run_dir = ROOT / f"phase_a/runs/{cell}/seed{seed}/formal"
            complete_path = run_dir / "RUN_COMPLETE.json"
            checkpoint = run_dir / "checkpoint_last.pt"
            if not complete_path.is_file() or not checkpoint.is_file():
                raise RuntimeError(f"missing formal artifact for {cell} seed {seed}")
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if complete.get("status") != "PASS" or complete.get("epoch") != 40 or complete.get("optimizer_steps") != 1720:
                raise RuntimeError(f"formal completion contract failed for {cell} seed {seed}")
            if payload.get("seed") != seed or payload.get("cell") != cell or payload.get("epoch") != 40 or payload.get("optimizer_steps") != 1720:
                raise RuntimeError(f"checkpoint identity contract failed for {cell} seed {seed}")
            if payload.get("config", {}).get("model_code_sha256") != digest(LOCAL_MODEL):
                raise RuntimeError(f"model code hash mismatch for {cell} seed {seed}")
            prediction_hand, final_hash = source.infer_cell(data, stats, cell, checkpoint, dev_ids, torch.device("cpu"))
            prediction_std_hand = source.standardize(prediction_hand, stats)
            full_native = np.concatenate((prediction_hand, arm_native), axis=2)
            full_std = np.concatenate((prediction_std_hand, arm_std), axis=2)
            if not np.array_equal(full_native[:, :, 14:], arm_native) or not np.array_equal(full_std[:, :, 14:], arm_std):
                raise RuntimeError(f"Arm bitwise seam failed for {cell} seed {seed}")
            per_seed[cell] = source.metric_arrays(full_native, full_std, target_native, target_std, anchor)
            np.savez_compressed(
                run_dir / "predictions_last.npz",
                episode_index=dev_ids,
                original_task_index=dev_tasks,
                targets_native=target_native,
                targets_standardized=target_std,
                persistence_native=persistence_native,
                persistence_standardized=persistence_std,
                predictions_hand_native=prediction_hand,
                predictions_hand_standardized=prediction_std_hand,
                predictions_full_native=full_native,
                predictions_full_standardized=full_std,
                locked_arm_native=arm_native,
                locked_arm_standardized=arm_std,
                final_test_access=np.asarray(0, dtype=np.int8),
            )
            write_json(run_dir / "DEV_INFERENCE_ACCEPTANCE.json", {
                "status": "PASS",
                "cell": cell,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "model_hash": final_hash,
                "episode_index_shape": [614],
                "prediction_hand_shape": [614, STEPS, HAND],
                "prediction_full_shape": [614, STEPS, 28],
                "dev_mask_row_count": int(len(dev_positions)),
                "dev_ids_sha256": hashlib.sha256(dev_ids.tobytes()).hexdigest(),
                "arm_bitwise_equal_locked": True,
                "dtype": "float32",
                "device": "cpu",
                "final_test_access": 0,
            })
            prediction_records.append({"seed": seed, "cell": cell, "status": "PASS", "checkpoint": str(checkpoint), "prediction_shape": "(614,30,14)", "full_shape": "(614,30,28)", "arm_bitwise_equal_locked": True, "final_test_access": 0})
        metrics_by_seed[seed] = per_seed

    # Per-seed metric table and primary summaries.
    metric_rows: list[dict[str, object]] = []
    metric_names = ("PERSISTENCE", "FIXED_HYBRID", *CELLS)
    for seed in SEEDS:
        for model_name in metric_names:
            metric = metrics_by_seed[seed][model_name]
            row: dict[str, object] = {"seed": seed, "model": model_name, "N_episode": 614}
            for key in ("hand_mae_std", "hand_mae_native", "full_mae_std", "full_mae_native", "hand_fd_native", "hand_fd_std", "full_fd_native", "hand_tv_closeness", "early_hand_mae_std", "middle_hand_mae_std", "late_hand_mae_std", "endpoint_hand_mae_std", "hand_direction_agreement"):
                row[key] = mean_metric(metric, key)
            row["motion_low_N"] = int(np.sum(motion_strata == "low"))
            row["motion_mid_N"] = int(np.sum(motion_strata == "mid"))
            row["motion_high_N"] = int(np.sum(motion_strata == "high"))
            row["final_test_access"] = 0
            metric_rows.append(row)
    write_csv(analysis_dir / "A_PER_SEED_METRICS.csv", metric_rows)

    # Fixed 4×2 co-primary family, with per-seed episode bootstrap and a
    # crossed seed×episode aggregate.  No 1842-row pooling is performed.
    contrasts = [
        ("S-HIST_vs_S-LAST", "HI-S-LAST", "HI-S-HIST"),
        ("F-HIST_vs_F-LAST", "HI-F-LAST", "HI-F-HIST"),
        ("F-HIST_vs_S-HIST", "HI-S-HIST", "HI-F-HIST"),
        ("F-LAST_vs_S-LAST", "HI-S-LAST", "HI-F-LAST"),
    ]
    contrast_rows: list[dict[str, object]] = []
    for contrast_name, control_name, candidate_name in contrasts:
        for metric_key in ("hand_mae_std", "hand_fd_native"):
            diff_by_seed = np.stack([metrics_by_seed[seed][control_name][metric_key] - metrics_by_seed[seed][candidate_name][metric_key] for seed in SEEDS])
            aggregate, samples = hierarchical_effect_from_diff(diff_by_seed, seed=20261100 + len(contrast_rows))
            row: dict[str, object] = {"contrast": contrast_name, "control": control_name, "candidate": candidate_name, "metric": metric_key, **aggregate, "per_seed_point": [float(values.mean()) for values in diff_by_seed]}
            for index, seed in enumerate(SEEDS):
                row[f"seed{seed}_ci_low"] = episode_bootstrap(metrics_by_seed[seed][control_name][metric_key], metrics_by_seed[seed][candidate_name][metric_key], bootstrap, 20261200 + index)["ci_low"]
                row[f"seed{seed}_ci_high"] = episode_bootstrap(metrics_by_seed[seed][control_name][metric_key], metrics_by_seed[seed][candidate_name][metric_key], bootstrap, 20261200 + index)["ci_high"]
            contrast_rows.append(row)
    primary_rows = [row for row in contrast_rows]
    holm(primary_rows)
    write_csv(analysis_dir / "A_FIXED_CONTRASTS.csv", primary_rows)

    # Candidate effects and the inherited guardrail family.
    candidate_diff_hand = np.stack([metrics_by_seed[seed]["PERSISTENCE"]["hand_mae_std"] - metrics_by_seed[seed][CANDIDATE]["hand_mae_std"] for seed in SEEDS])
    candidate_diff_full = np.stack([metrics_by_seed[seed]["FIXED_HYBRID"]["full_mae_std"] - metrics_by_seed[seed][CANDIDATE]["full_mae_std"] for seed in SEEDS])
    candidate_fd = np.stack([metrics_by_seed[seed][CANDIDATE]["hand_fd_native"] for seed in SEEDS])
    persistence_fd = np.stack([metrics_by_seed[seed]["PERSISTENCE"]["hand_fd_native"] for seed in SEEDS])
    hand_aggregate, hand_samples = hierarchical_effect_from_diff(candidate_diff_hand, seed=20261301)
    full_aggregate, full_samples = hierarchical_effect_from_diff(candidate_diff_full, seed=20261302)
    fd_aggregate, fd_samples = hierarchical_ratio(candidate_fd, persistence_fd, seed=20261303)
    per_seed_candidate: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS):
        hand = episode_bootstrap(metrics_by_seed[seed]["PERSISTENCE"]["hand_mae_std"], metrics_by_seed[seed][CANDIDATE]["hand_mae_std"], bootstrap, 20261400 + index)
        full = episode_bootstrap(metrics_by_seed[seed]["FIXED_HYBRID"]["full_mae_std"], metrics_by_seed[seed][CANDIDATE]["full_mae_std"], bootstrap, 20261500 + index)
        fd = source.ratio_bootstrap(metrics_by_seed[seed][CANDIDATE]["hand_fd_native"], metrics_by_seed[seed]["PERSISTENCE"]["hand_fd_native"], bootstrap)
        per_seed_candidate.append({"seed": seed, "hand": hand, "full": full, "native_fd": fd, "hand_point": float(candidate_diff_hand[index].mean()), "full_point": float(candidate_diff_full[index].mean()), "native_fd_point": float(fd["relative_change_candidate_vs_control"]), "final_test_access": 0})

    guardrail_rows: list[dict[str, object]] = []
    def add_ratio_guardrail(name: str, key: str, control_name: str = "PERSISTENCE") -> None:
        cand = np.stack([metrics_by_seed[seed][CANDIDATE][key] for seed in SEEDS])
        ctrl = np.stack([metrics_by_seed[seed][control_name][key] for seed in SEEDS])
        finite = np.isfinite(cand).all(axis=0) & np.isfinite(ctrl).all(axis=0)
        if not finite.any():
            raise RuntimeError(f"no finite episodes for guardrail {name}")
        cand = cand[:, finite]
        ctrl = ctrl[:, finite]
        aggregate, _samples = hierarchical_ratio(cand, ctrl, seed=20261600 + len(guardrail_rows))
        point_by_seed = [float(cand[i].mean() / ctrl[i].mean() - 1.0) for i in range(3)]
        guardrail_rows.append({"guardrail": name, "candidate": CANDIDATE, "control": control_name, "N_episode": int(finite.sum()), "relative_change_point_by_seed": point_by_seed, **aggregate, "pass": bool(max(point_by_seed) <= 0.01 and aggregate["ci_high"] <= 0.01), "final_test_access": 0})

    add_ratio_guardrail("standardized_fd_vs_persistence", "hand_fd_std")
    add_ratio_guardrail("tv_closeness_vs_persistence", "hand_tv_closeness")
    for key, label in (("early_hand_mae_std", "early"), ("middle_hand_mae_std", "middle"), ("late_hand_mae_std", "late"), ("endpoint_hand_mae_std", "endpoint")):
        add_ratio_guardrail(f"horizon_{label}_hand_mae_vs_persistence", key)
    for stratum in ("low", "mid", "high"):
        mask = motion_strata == stratum
        cand = np.stack([metrics_by_seed[seed][CANDIDATE]["hand_mae_std"][mask] for seed in SEEDS])
        ctrl = np.stack([metrics_by_seed[seed]["PERSISTENCE"]["hand_mae_std"][mask] for seed in SEEDS])
        aggregate, _samples = hierarchical_ratio(cand, ctrl, seed=20261700 + len(guardrail_rows))
        point_by_seed = [float(cand[i].mean() / ctrl[i].mean() - 1.0) for i in range(3)]
        guardrail_rows.append({"guardrail": f"motion_{stratum}_hand_mae_vs_persistence", "candidate": CANDIDATE, "control": "PERSISTENCE", "N_episode": int(mask.sum()), "relative_change_point_by_seed": point_by_seed, **aggregate, "pass": bool(max(point_by_seed) <= 0.01 and aggregate["ci_high"] <= 0.01), "final_test_access": 0})

    candidate_metric = metrics_by_seed[SEEDS[0]][CANDIDATE]
    finite_support = all(np.isfinite(metrics_by_seed[seed][CANDIDATE]["per_joint_hand_mae_std"]).all() and np.isfinite(metrics_by_seed[seed][CANDIDATE]["per_joint_hand_fd_native"]).all() for seed in SEEDS)
    support_row = {"guardrail": "support_all_shared_episodes_joints_strata", "candidate": CANDIDATE, "control": "PERSISTENCE", "N_episode": 614, "joint_count": HAND, "motion_strata": {label: int(np.sum(motion_strata == label)) for label in ("low", "mid", "high")}, "finite": finite_support, "pass": bool(finite_support and len(np.unique(dev_tasks)) > 0 and all(np.sum(motion_strata == label) > 0 for label in ("low", "mid", "high"))), "final_test_access": 0}
    guardrail_rows.append(support_row)
    write_csv(analysis_dir / "A_GUARDRAILS.csv", guardrail_rows)

    # History/video interaction on the same crossed bootstrap basis.
    interactions: list[dict[str, object]] = []
    for metric_key in ("hand_mae_std", "hand_fd_native"):
        diff_by_seed = np.stack([
            (metrics_by_seed[seed]["HI-F-LAST"][metric_key] - metrics_by_seed[seed]["HI-F-HIST"][metric_key])
            - (metrics_by_seed[seed]["HI-S-LAST"][metric_key] - metrics_by_seed[seed]["HI-S-HIST"][metric_key])
            for seed in SEEDS
        ])
        aggregate, _samples = hierarchical_effect_from_diff(diff_by_seed, seed=20261800 + len(interactions))
        interactions.append({"effect": "video_conditional_history_gain", "metric": metric_key, **aggregate, "per_seed_point": [float(row.mean()) for row in diff_by_seed], "definition": "(F-LAST-F-HIST)-(S-LAST-S-HIST); positive means stronger history gain with video", "final_test_access": 0})
    write_json(analysis_dir / "A_HISTORY_VIDEO_INTERACTION.json", {"status": "PASS", "rows": interactions, "interpretation": "interaction is reported separately and is not inferred from discordant simple-effect significance", "final_test_access": 0})

    candidate_decision = {
        "status": "A_CANDIDATE_GATE_PASS" if (
            all(item["hand_point"] > 0 for item in per_seed_candidate)
            and hand_aggregate["ci_low"] > 0
            and all(item["full_point"] > 0 for item in per_seed_candidate)
            and full_aggregate["ci_low"] > 0
            and all(item["native_fd_point"] <= 0.01 for item in per_seed_candidate)
            and fd_aggregate["ci_high"] <= 0.01
            and all(bool(row.get("pass")) for row in guardrail_rows)
            and len(prediction_records) == 12
        ) else "A_CANDIDATE_GATE_FAIL",
        "candidate": CANDIDATE,
        "fresh_seeds": list(SEEDS),
        "discovery_seed_excluded": 20260939,
        "formal_acceptance": {"runs": len(prediction_records), "all_pass": len(prediction_records) == 12},
        "hand_vs_persistence": {"aggregate": hand_aggregate, "per_seed": per_seed_candidate},
        "full_vs_fixed_hybrid": {"aggregate": full_aggregate, "per_seed": per_seed_candidate},
        "native_fd_vs_persistence": {"aggregate": fd_aggregate, "per_seed": per_seed_candidate},
        "guardrails": guardrail_rows,
        "fixed_contrast_count": len(primary_rows),
        "hierarchical_replicates": HIER_REPLICATES,
        "episode_replicates_per_seed": EPISODE_REPLICATES,
        "input_sha256": digest(INPUT),
        "locked_arm_sha256": digest(LOCKED_ARM),
        "local_model_sha256": digest(LOCAL_MODEL),
        "source_model_sha256": digest(SOURCE_MODEL),
        "final_test_access": 0,
    }
    write_json(analysis_dir / "A_CANDIDATE_DECISION.json", candidate_decision)
    write_json(analysis_dir / "A_INFERENCE_QA.json", {
        "status": "PASS",
        "runs": prediction_records,
        "run_count": len(prediction_records),
        "dev_rows": 614,
        "dev_ids_sha256": hashlib.sha256(dev_ids.tobytes()).hexdigest(),
        "target_shape": [614, STEPS, 28],
        "arm_bitwise_equal_all_runs": True,
        "checkpoint_reload_verified_during_training": True,
        "fixed_last_checkpoint": True,
        "dtype": "float32",
        "device": "cpu",
        "source_input_sha256": digest(INPUT),
        "final_test_access": 0,
    })
    write_json(analysis_dir / "A_AGGREGATE_EFFECTS.json", {
        "status": "PASS",
        "candidate": CANDIDATE,
        "hand_vs_persistence": hand_aggregate,
        "full_vs_fixed_hybrid": full_aggregate,
        "native_fd_vs_persistence": fd_aggregate,
        "fixed_contrasts": primary_rows,
        "history_video_interaction": interactions,
        "resampling": "crossed seed×episode; do not pool 1842 seed-episode records as independent episodes",
        "final_test_access": 0,
    })
    np.savez_compressed(analysis_dir / "A_HIERARCHICAL_BOOTSTRAP_EFFECTS.npz", hand=hand_samples, full=full_samples, fd_ratio_minus_one=fd_samples, final_test_access=np.asarray(0, dtype=np.int8))
    write_json(ROOT / "phase_a/qa/A_GATE_STATUS.json", {
        "status": candidate_decision["status"],
        "candidate": CANDIDATE,
        "formal_runs": len(prediction_records),
        "b_triggered": candidate_decision["status"] == "A_CANDIDATE_GATE_PASS",
        "final_test_access": 0,
    })
    md_lines = [
        "# Phase-A fresh-seed decision",
        "",
        f"- Status: `{candidate_decision['status']}`",
        f"- Candidate: `{CANDIDATE}`",
        f"- Fresh seeds: {', '.join(map(str, SEEDS))}; discovery seed 20260939 excluded",
        f"- Formal accepted runs: {len(prediction_records)}/12",
        f"- Hand aggregate effect (Persistence - candidate): {hand_aggregate['effect_control_minus_candidate']:.8f}; 95% CI [{hand_aggregate['ci_low']:.8f}, {hand_aggregate['ci_high']:.8f}]",
        f"- Full aggregate effect (fixed hybrid - candidate): {full_aggregate['effect_control_minus_candidate']:.8f}; 95% CI [{full_aggregate['ci_low']:.8f}, {full_aggregate['ci_high']:.8f}]",
        f"- Native FD relative change: {100*fd_aggregate['relative_change_candidate_vs_control']:.4f}%; 95% CI [{100*fd_aggregate['ci_low']:.4f}%, {100*fd_aggregate['ci_high']:.4f}%]",
        f"- Crossed hierarchical resamples: {HIER_REPLICATES}; per-seed episode resamples: {EPISODE_REPLICATES}",
        "- All metrics use the shared 614 development IDs; no seed-episode pooling as independent episodes.",
        "- Final-test access: 0",
    ]
    (analysis_dir / "A_CANDIDATE_DECISION.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": candidate_decision["status"], "formal_runs": len(prediction_records), "hand_ci_low": hand_aggregate["ci_low"], "full_ci_low": full_aggregate["ci_low"], "fd_ci_high": fd_aggregate["ci_high"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
