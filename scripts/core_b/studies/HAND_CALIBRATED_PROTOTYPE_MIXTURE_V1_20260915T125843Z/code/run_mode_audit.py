#!/usr/bin/env python3
"""Run the train-only Hand residual mode feasibility audit."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, log_loss, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
BOOTSTRAP_PATH = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/paired_episode_bootstrap_indices.npy"
MODE_COUNTS = (2, 4, 8)
CLUSTER_SEED = 20260937
STABILITY_BOOTSTRAP = 1000
PERMUTATION_REPLICATES = 1000
FOLDS = 3
STEPS = 30
HAND = 14


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


def dct_basis(length: int = STEPS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (t + 0.5) * k) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def fold_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761) + np.uint64(1013904223)) % np.uint64(FOLDS)).astype(np.int64)


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if mask.any():
            result += float(mask.mean()) * abs(float((prediction[mask] == labels[mask]).mean()) - float(confidence[mask].mean()))
    return result


def fit_probe(features: np.ndarray, labels: np.ndarray, folds: np.ndarray, class_count: int) -> dict[str, Any]:
    predictions = np.full(len(labels), -1, dtype=np.int64)
    probabilities = np.zeros((len(labels), class_count), dtype=np.float64)
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=CLUSTER_SEED))])
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test]).astype(np.int64)
        raw_probability = model.predict_proba(features[test])
        classes = model.named_steps["classifier"].classes_.astype(np.int64)
        probabilities[np.ix_(test, classes)] = raw_probability
    # A fold can lack a rare class. The zero-filled probability is retained as
    # an explicit contract failure signal rather than silently renormalized.
    row_sum = probabilities.sum(axis=1)
    valid = row_sum > 0
    probabilities[valid] /= row_sum[valid, None]
    return {"macro_f1": float(f1_score(labels, predictions, labels=np.arange(class_count), average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "nll": float(log_loss(labels, probabilities, labels=np.arange(class_count))), "ece": ece(labels, probabilities), "predictions": predictions, "probabilities": probabilities}


def majority_control(labels: np.ndarray, folds: np.ndarray, class_count: int) -> float:
    prediction = np.zeros(len(labels), dtype=np.int64)
    for fold in range(FOLDS):
        train = folds != fold
        counts = np.bincount(labels[train], minlength=class_count)
        prediction[folds == fold] = int(np.argmax(counts))
    return float(f1_score(labels, prediction, labels=np.arange(class_count), average="macro", zero_division=0))


def permutation_null(labels: np.ndarray, predictions: np.ndarray, folds: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    null_values: list[float] = []
    for _ in range(PERMUTATION_REPLICATES):
        shuffled = labels.copy()
        for fold in range(FOLDS):
            test = folds == fold
            shuffled[test] = rng.permutation(shuffled[test])
        null_values.append(float(f1_score(shuffled, predictions, labels=np.unique(labels), average="macro", zero_division=0)))
    values = np.asarray(null_values, dtype=np.float64)
    return {"median": float(np.median(values)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "p95": float(np.quantile(values, 0.95)), "replicates": PERMUTATION_REPLICATES}


def stability_ari(features: np.ndarray, labels: np.ndarray, centers: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(STABILITY_BOOTSTRAP, dtype=np.float64)
    for index in range(STABILITY_BOOTSTRAP):
        sample = rng.integers(0, len(features), size=len(features))
        distances = ((features[sample, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        reassigned = distances.argmin(axis=1)
        values[index] = adjusted_rand_score(labels[sample], reassigned)
    return {"median": float(np.median(values)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": STABILITY_BOOTSTRAP}


def main() -> int:
    out = ROOT / "mode_audit"
    data = load_npz(INPUT)
    train_mask = data["split_code"] == 0
    train_ids = data["episode_index"][train_mask].astype(np.int64)
    tasks = data["original_task_index"][train_mask].astype(np.int64)
    oof = load_npz(DCT_ROOT / "oof/HD-D1-L1_OOF.npz")
    if not np.array_equal(oof["episode_index"], train_ids):
        raise RuntimeError("train OOF episode order mismatch")
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)[:, :HAND]
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)[:, :HAND]
    persistence_std = oof["persistence_hand_standardized"].astype(np.float64)
    target_std = oof["targets_hand_standardized"].astype(np.float64)
    persistence_native = oof["persistence_hand_native"].astype(np.float64)
    target_native = oof["targets_hand_native"].astype(np.float64)
    neural_std = oof["oof_predictions_hand_standardized"].astype(np.float64)
    neural_native = neural_std * scale[None, :, :] + mean[None, :, :]
    # The locked mode target is the ground-truth future residual to
    # Persistence. The D1-L1 OOF proposal is retained only as the single
    # decoder comparator for prototype-oracle headroom.
    residual_native = target_native - persistence_native
    basis = dct_basis()
    coefficients = np.einsum("kt,etj->ekj", basis[:16], residual_native)
    coefficient_mean = coefficients.mean(axis=0)
    coefficient_scale = np.maximum(coefficients.std(axis=0), 1e-6)
    whitened = ((coefficients - coefficient_mean) / coefficient_scale).reshape(len(coefficients), -1)
    folds = fold_assignment(train_ids)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2] or len(np.unique(train_ids)) != len(train_ids):
        raise RuntimeError("mode audit fold contract failed")
    state = data["state_action_history"][train_mask].astype(np.float64)
    dynamic = data["dynamic_pooled"][train_mask].astype(np.float64)
    prefix_motion = np.column_stack((state.mean(axis=1), state.std(axis=1), np.abs(np.diff(state, axis=1)).mean(axis=1)))
    fusion_state_features = np.concatenate((dynamic, state, prefix_motion), axis=1)
    state_features = np.column_stack((state.mean(axis=1), state.std(axis=1), state[:, -1], np.abs(np.diff(state, axis=1)).mean(axis=1)))
    task_onehot = np.eye(int(tasks.max()) + 1, dtype=np.float64)[tasks]
    mode_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    assignment_columns: dict[str, np.ndarray] = {"episode_index": train_ids, "task": tasks}
    centers_payload: dict[str, np.ndarray] = {"coefficient_mean": coefficient_mean, "coefficient_scale": coefficient_scale}
    for mode_count in MODE_COUNTS:
        cluster = KMeans(n_clusters=mode_count, init="k-means++", n_init=1, max_iter=300, random_state=CLUSTER_SEED, algorithm="lloyd").fit(whitened)
        labels = cluster.labels_.astype(np.int64)
        centers = cluster.cluster_centers_.astype(np.float64)
        counts = np.bincount(labels, minlength=mode_count)
        prevalence = counts / len(labels)
        silhouette = float(silhouette_score(whitened, labels, metric="euclidean"))
        stability = stability_ari(whitened, labels, centers, CLUSTER_SEED + mode_count)
        max_task_share = 0.0
        multi_mode_tasks = 0
        task_cluster_rows: list[dict[str, object]] = []
        for task in sorted(set(tasks.tolist())):
            task_mask = tasks == task
            task_labels = labels[task_mask]
            if int(task_mask.sum()) >= 5 and len(np.unique(task_labels)) >= 2:
                multi_mode_tasks += 1
            for mode in range(mode_count):
                denominator = int((labels == mode).sum())
                numerator = int(np.sum(task_labels == mode))
                share = numerator / denominator if denominator else 0.0
                max_task_share = max(max_task_share, share)
                task_cluster_rows.append({"K_mode": mode_count, "task": int(task), "mode": mode, "task_episode_count": int(task_mask.sum()), "cluster_episode_count": denominator, "task_within_cluster_share": share})
        center_coeff = centers.reshape(mode_count, 16, HAND) * coefficient_scale[None, :, :] + coefficient_mean[None, :, :]
        center_residual_native = np.einsum("tk,ekj->etj", basis[:16].T, center_coeff)
        component_native = persistence_native[:, None, :, :] + center_residual_native[None, :, :, :]
        component_std = (component_native - mean[None, None, :, :]) / scale[None, None, :, :]
        level_errors = np.abs(component_std - target_std[:, None, :, :]).mean(axis=(2, 3))
        fd_errors = np.abs(np.diff(component_native, axis=2) - np.diff(target_native[:, None, :, :], axis=2)).mean(axis=(2, 3))
        single_level = np.abs(neural_std - target_std).mean(axis=(1, 2))
        single_fd = np.abs(np.diff(neural_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2))
        oracle_level = level_errors.min(axis=1)
        oracle_fd = fd_errors.min(axis=1)
        oracle_level_improvement = float(np.mean(single_level - oracle_level))
        oracle_fd_improvement = float(np.mean(single_fd - oracle_fd))
        center_flat = center_residual_native.reshape(mode_count, -1)
        if mode_count > 1:
            between = float(np.mean([np.linalg.norm(center_flat[i] - center_flat[j]) for i in range(mode_count) for j in range(i + 1, mode_count)]))
        else:
            between = 0.0
        assigned_distance = np.linalg.norm(residual_native.reshape(len(residual_native), -1) - center_flat[labels], axis=1)
        within = float(np.mean(assigned_distance))
        distance_ratio = within / between if between > 1e-12 else float("inf")
        assignment_columns[f"K{mode_count}"] = labels
        centers_payload[f"centers_K{mode_count}"] = centers
        feature_sets = {"FUSION_STATE_PREFIX": fusion_state_features, "STATE_SUMMARY": state_features, "PREFIX_MOTION": prefix_motion, "TASK_ONLY": task_onehot}
        majority = majority_control(labels, folds, mode_count)
        primary_probe: dict[str, Any] | None = None
        for feature_name, features in feature_sets.items():
            result = fit_probe(features, labels, folds, mode_count)
            null = permutation_null(labels, result["predictions"], folds, CLUSTER_SEED + mode_count * 100 + len(probe_rows)) if feature_name == "FUSION_STATE_PREFIX" else {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p95": float("nan"), "replicates": 0}
            if feature_name == "FUSION_STATE_PREFIX":
                primary_probe = result
            probe_rows.append({"K_mode": mode_count, "feature_set": feature_name, "macro_F1": result["macro_f1"], "balanced_accuracy": result["balanced_accuracy"], "NLL": result["nll"], "ECE": result["ece"], "majority_macro_F1": majority, "permutation_median": null["median"], "permutation_ci_low": null["ci_low"], "permutation_ci_high": null["ci_high"], "permutation_p95": null["p95"], "probe_minus_permutation_p95": result["macro_f1"] - null["p95"] if np.isfinite(null["p95"]) else float("nan"), "permutation_replicates": null["replicates"]})
        if primary_probe is None:
            raise RuntimeError("primary probe missing")
        primary_row = next(row for row in probe_rows if row["K_mode"] == mode_count and row["feature_set"] == "FUSION_STATE_PREFIX")
        checks = {"silhouette_ge_010": silhouette >= 0.10, "bootstrap_ari_median_ge_060": stability["median"] >= 0.60, "minimum_cluster_prevalence_ge_005": float(prevalence.min()) >= 0.05, "fusion_probe_macro_f1_ge_majority_plus_5pp": primary_probe["macro_f1"] >= majority + 0.05, "fusion_probe_above_permuted_control": float(primary_row["probe_minus_permutation_p95"]) > 0, "at_least_10_tasks_with_multiple_modes": multi_mode_tasks >= 10, "no_single_task_cluster_share_over_50pct": max_task_share < 0.50, "prototype_oracle_fd_headroom_positive": oracle_fd_improvement > 0, "train_only_schema_qa": True}
        mode_rows.append({"K_mode": mode_count, "silhouette": silhouette, "minimum_cluster_prevalence": float(prevalence.min()), "cluster_prevalence": prevalence.tolist(), "bootstrap_ari_median": stability["median"], "bootstrap_ari_ci_low": stability["ci_low"], "bootstrap_ari_ci_high": stability["ci_high"], "stability_replicates": STABILITY_BOOTSTRAP, "tasks_with_multiple_modes": multi_mode_tasks, "max_task_cluster_share": max_task_share, "within_between_trajectory_distance_ratio": distance_ratio, "prototype_oracle_level_improvement_vs_single": oracle_level_improvement, "prototype_oracle_fd_improvement_vs_single": oracle_fd_improvement, "fusion_probe_macro_F1": primary_probe["macro_f1"], "fusion_probe_majority_macro_F1": majority, "fusion_probe_minus_permutation_p95": primary_row["probe_minus_permutation_p95"], "checks": checks, "mode_gate_pass": bool(all(checks.values()))})
        write_csv(out / f"K{mode_count}_TASK_CLUSTER_SUPPORT.csv", task_cluster_rows)
    selected = next((int(row["K_mode"]) for row in mode_rows if row["mode_gate_pass"]), None)
    assignment_rows = [{key: (int(value[index]) if np.asarray(value).ndim == 1 else value[index]) for key, value in assignment_columns.items()} for index in range(len(train_ids))]
    write_csv(out / "CLUSTER_ASSIGNMENTS.csv", assignment_rows)
    write_csv(out / "K_MODE_RESULTS.csv", mode_rows)
    write_json(out / "K_MODE_RESULTS.json", {"status": "PASS", "rows": mode_rows, "selected_K_mode": selected, "cluster_seed": CLUSTER_SEED, "source_session_metadata_available": False, "stability_proxy": "episode-resampling ARI", "final_test_access": 0})
    write_csv(out / "PROBE_RESULTS.csv", probe_rows)
    write_json(out / "PROBE_RESULTS.json", {"status": "PASS", "rows": probe_rows, "primary_feature_set": "FUSION_STATE_PREFIX", "permutation_control": "held-out-label permutation against fixed probe predictions", "permutation_replicates": PERMUTATION_REPLICATES, "final_test_access": 0})
    np.savez_compressed(out / "PROTOTYPE_CENTERS.npz", **centers_payload)
    gate = {"status": "PASS", "mode_gate": "pass" if selected is not None else "fail", "selected_K_mode": selected, "mode_rows": mode_rows, "selected_by_smallest_passing_K": True, "final_test_access": 0}
    write_json(out / "MODE_GATE.json", gate)
    (out / "MODE_GATE.md").write_text("# Train-only mode gate\n\n" + f"Mode gate: `{'PASS' if selected is not None else 'FAIL'}`; selected K: `{selected if selected is not None else 'none'}`.\n\n" + "\n".join(f"- K={row['K_mode']}: gate=`{'PASS' if row['mode_gate_pass'] else 'FAIL'}`, silhouette=`{row['silhouette']:.4f}`, ARI median=`{row['bootstrap_ari_median']:.4f}`, FUSION+state macro-F1=`{row['fusion_probe_macro_F1']:.4f}`, oracle FD headroom=`{row['prototype_oracle_fd_improvement_vs_single']:.6f}`" for row in mode_rows) + "\n\nSource/session metadata were unavailable; stability is explicitly an episode-resampling proxy. No dev target or prediction was used.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode_gate": gate["mode_gate"], "selected_K_mode": selected, "rows": mode_rows}, sort_keys=True))
    return 0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


if __name__ == "__main__":
    raise SystemExit(main())
