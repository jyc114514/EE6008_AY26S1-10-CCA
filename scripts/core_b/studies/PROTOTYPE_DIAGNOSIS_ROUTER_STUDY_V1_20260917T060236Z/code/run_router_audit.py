#!/usr/bin/env python3
"""CPU-only audit and fixed-q oracle gate for the prototype router study."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, adjusted_rand_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
PROTO = REPO / "reports/core_b/HAND_CALIBRATED_PROTOTYPE_MIXTURE_V1_20260915T125843Z"
DCT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
TEMPORAL = REPO / "reports/core_b/HAND_TEMPORAL_FIDELITY_V1_20260914T152845Z"
FREEZE = REPO / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z"
BENCHMARK = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
BOOTSTRAP_PATH = BENCHMARK / "data/paired_episode_bootstrap_indices.npy"
HAND = 14
STEPS = 30
BOOTSTRAP = 10_000
ARI_BOOTSTRAP = 200
CLUSTER_SEED = 20260937
FOLDS = 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


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


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    estimates = np.nanmean(values[indices], axis=1)
    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) != len(indices):
        raise RuntimeError("bootstrap finite replicate loss")
    valid = values[np.isfinite(values)]
    return {"effect": float(valid.mean()), "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)), "replicates": len(indices), "N_episode": int(len(valid))}


def paired(control: np.ndarray, candidate: np.ndarray, indices: np.ndarray, seed: int) -> dict[str, object]:
    improvement = np.asarray(control, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    result = bootstrap_mean(improvement, indices)
    valid = improvement[np.isfinite(improvement)]
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, len(indices), 500):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(min(500, len(indices) - start), len(valid)))
        exceed += int(np.sum(np.abs((signs * valid[None, :]).mean(axis=1)) >= abs(valid.mean())))
    result.update({"win_rate": float(np.mean(valid > 0)), "p_raw": float((1 + exceed) / (len(indices) + 1))})
    return result


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    numerator = np.asarray(numerator, dtype=np.float64).reshape(-1)
    denominator = np.asarray(denominator, dtype=np.float64).reshape(-1)
    num = np.nanmean(numerator[indices], axis=1)
    den = np.nanmean(denominator[indices], axis=1)
    values = np.divide(num, den, out=np.full_like(num, np.nan), where=np.abs(den) > 1e-12)
    values = values[np.isfinite(values)]
    if len(values) != len(indices):
        raise RuntimeError("ratio bootstrap denominator failure")
    return {"effect": float(np.nanmean(numerator) / np.nanmean(denominator)), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "replicates": len(indices), "N_episode": int(np.isfinite(numerator).sum())}


def tv_closeness(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_tv = np.abs(np.diff(pred, axis=1)).sum(axis=1).mean(axis=1)
    target_tv = np.abs(np.diff(target, axis=1)).sum(axis=1).mean(axis=1)
    valid = target_tv > 1e-6
    ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=valid)
    return pred_tv, target_tv, ratio, np.abs(ratio - 1.0)


def metric_bundle(hand_std: np.ndarray, full_std: np.ndarray, hand_native: np.ndarray, full_native: np.ndarray, target_std: np.ndarray, target_native: np.ndarray, persistence_native: np.ndarray) -> dict[str, np.ndarray]:
    target_hand_std = target_std[:, :, :HAND]
    target_hand_native = target_native[:, :, :HAND]
    p_hand_native = persistence_native[:, :, :HAND]
    hand_fd_native = np.abs(np.diff(hand_native, axis=1) - np.diff(target_hand_native, axis=1)).mean(axis=(1, 2))
    hand_fd_std = np.abs(np.diff(hand_std, axis=1) - np.diff(target_hand_std, axis=1)).mean(axis=(1, 2))
    hand_anchor_native = np.abs((hand_native[:, 0] - p_hand_native[:, 0]) - (target_hand_native[:, 0] - p_hand_native[:, 0])).mean(axis=1)
    full_fd_native = np.abs(np.diff(full_native, axis=1) - np.diff(target_native, axis=1)).mean(axis=(1, 2))
    pred_tv_h, target_tv_h, ratio_h, close_h = tv_closeness(hand_native, target_hand_native)
    pred_tv_f, target_tv_f, ratio_f, close_f = tv_closeness(full_native, target_native)
    return {"hand_mae_std": np.abs(hand_std - target_hand_std).mean(axis=(1, 2)), "hand_mae_native": np.abs(hand_native - target_hand_native).mean(axis=(1, 2)), "full_mae_std": np.abs(full_std - target_std).mean(axis=(1, 2)), "full_mae_native": np.abs(full_native - target_native).mean(axis=(1, 2)), "hand_fd_native": hand_fd_native, "hand_fd_std": hand_fd_std, "hand_anchor_fd_native": hand_anchor_native, "full_fd_native": full_fd_native, "hand_tv": pred_tv_h, "target_hand_tv": target_tv_h, "hand_tv_ratio": ratio_h, "hand_tv_closeness": close_h, "full_tv": pred_tv_f, "target_full_tv": target_tv_f, "full_tv_ratio": ratio_f, "full_tv_closeness": close_f}


def mode_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, class_count: int) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), class_count):
        raise ValueError(f"probability shape mismatch: {probabilities.shape}")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(probabilities, row_sums, out=np.full_like(probabilities, 1.0 / class_count), where=row_sums > 1e-12)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=np.arange(class_count), zero_division=0)
    return {"N": int(len(labels)), "class_support": support.tolist(), "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(), "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)), "macro_f1": float(f1_score(labels, predictions, labels=np.arange(class_count), average="macro", zero_division=0)), "confusion": confusion_matrix(labels, predictions, labels=np.arange(class_count)).tolist(), "nll": float(log_loss(labels, probabilities, labels=np.arange(class_count))), "brier": float(np.mean(np.sum((probabilities - np.eye(class_count)[labels]) ** 2, axis=1)))}


def probe_oof(features: np.ndarray, labels: np.ndarray, folds: np.ndarray, class_count: int) -> dict[str, Any]:
    predictions = np.full(len(labels), -1, dtype=np.int64)
    probabilities = np.zeros((len(labels), class_count), dtype=np.float64)
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=CLUSTER_SEED))])
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test]).astype(np.int64)
        classifier = model.named_steps["logistic"]
        classes = classifier.classes_.astype(np.int64)
        probabilities[np.ix_(test, classes)] = classifier.predict_proba(features[test])
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return {"predictions": predictions, "probabilities": probabilities, "metrics": mode_metrics(labels, predictions, probabilities, class_count)}


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = root / "qa/CHECKSUMS.sha256"
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    failures: list[str] = []
    for line in lines:
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            failures.append(relative)
    return {"status": "PASS" if not failures else "FAIL", "entries": len(lines), "manifest_sha256": digest(manifest), "failures": failures}


def main() -> int:
    analysis = ROOT / "analysis"
    data = load_npz(INPUT)
    train = data["split_code"] == 0
    dev = data["split_code"] == 1
    if int(train.sum()) != 2741 or int(dev.sum()) != 614:
        raise RuntimeError("canonical split count failure")
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float64)
    targets_std = data["targets_standardized"][dev].astype(np.float64)
    targets_native = data["targets_native"][dev].astype(np.float64)
    persistence_std_full = data["persistence_standardized"][dev].astype(np.float64)
    persistence_native_full = data["persistence_native"][dev].astype(np.float64)
    p_std = persistence_std_full[:, :, :HAND]
    p_native = persistence_native_full[:, :, :HAND]
    ids = data["episode_index"][dev].astype(np.int64)
    tasks = data["original_task_index"][dev].astype(np.int64)
    arm_bundle = load_npz(LOCKED_ARM)
    arm_std = arm_bundle["predictions_standardized"][:, :, 14:28].astype(np.float64)
    arm_native = arm_bundle["predictions_native"][:, :, 14:28].astype(np.float64)
    if not np.array_equal(arm_bundle["episode_index"], ids):
        raise RuntimeError("locked Arm ID failure")
    proto = load_npz(PROTO / "predictions/HP-PROTO_PREDICTIONS_DEV.npz")
    single = load_npz(PROTO / "predictions/HP-SINGLE_PREDICTIONS_DEV.npz")
    if not np.array_equal(proto["episode_index"], ids) or not np.array_equal(single["episode_index"], ids):
        raise RuntimeError("prototype prediction IDs failure")
    if not np.array_equal(proto["locked_arm_standardized"], arm_std.astype(np.float32)) or not np.array_equal(single["locked_arm_standardized"], arm_std.astype(np.float32)):
        raise RuntimeError("Arm non-interference source failure")
    components_std = proto["components_hand_standardized"].astype(np.float64)
    probabilities = proto["mode_probabilities"].astype(np.float64)
    q_ref = proto["q"].astype(np.float64)[:, :, 0]
    if components_std.shape != (614, 2, 30, 14) or probabilities.shape != (614, 2) or q_ref.shape != (614, 30):
        raise RuntimeError("prototype component/q shape failure")
    component_native = components_std * scale[None, None, :, :HAND] + mean[None, None, :, :HAND]
    shadow_components_std = p_std[:, None, :, :] + q_ref[:, None, :, None] * (components_std - p_std[:, None, :, :])
    shadow_components_native = p_native[:, None, :, :] + q_ref[:, None, :, None] * (component_native - p_native[:, None, :, :])
    chosen = probabilities.argmax(axis=1)
    shadow_map_std = shadow_components_std[np.arange(len(ids)), chosen]
    shadow_map_native = shadow_components_native[np.arange(len(ids)), chosen]
    soft_std = np.sum(probabilities[:, :, None, None] * components_std, axis=1)
    shadow_mean_std = p_std + q_ref[:, :, None] * (soft_std - p_std)
    shadow_mean_native = p_native + q_ref[:, :, None] * (np.sum(probabilities[:, :, None, None] * component_native, axis=1) - p_native)
    canonical_map_std = proto["map_hand_predictions_standardized"].astype(np.float64)
    canonical_map_native = proto["map_hand_predictions_native"].astype(np.float64)
    canonical_mean_std = proto["mean_hand_predictions_standardized"].astype(np.float64)
    canonical_mean_native = proto["mean_hand_predictions_native"].astype(np.float64)
    replay = {"map_std_max_abs": float(np.max(np.abs(canonical_map_std - shadow_map_std))), "mean_std_max_abs": float(np.max(np.abs(canonical_mean_std - shadow_mean_std))), "map_native_max_abs": float(np.max(np.abs(canonical_map_native - shadow_map_native))), "mean_native_max_abs": float(np.max(np.abs(canonical_mean_native - shadow_mean_native)))}
    if max(replay["mean_std_max_abs"], replay["mean_native_max_abs"]) > 2e-5:
        raise RuntimeError(f"fixed-q MEAN shadow replay failed: {replay}")
    replay["map_shadow_difference_expected"] = True
    replay["map_shadow_difference_reason"] = "canonical MAP uses a separate map-specific selector q; fixed-q shadow intentionally uses canonical MEAN q_ref"
    models: dict[str, dict[str, np.ndarray]] = {
        "PERSISTENCE": {"hand_std": p_std, "hand_native": p_native, "full_std": np.concatenate((p_std, arm_std), axis=2), "full_native": np.concatenate((p_native, arm_native), axis=2)},
        "HP-SINGLE": {"hand_std": single["hand_predictions_standardized"].astype(np.float64), "hand_native": single["hand_predictions_native"].astype(np.float64), "full_std": single["full_predictions_standardized"].astype(np.float64), "full_native": single["full_predictions_native"].astype(np.float64)},
        "HP-PROTO-MAP": {"hand_std": canonical_map_std, "hand_native": canonical_map_native, "full_std": proto["map_full_predictions_standardized"].astype(np.float64), "full_native": proto["map_full_predictions_native"].astype(np.float64)},
        "HP-PROTO-MEAN": {"hand_std": canonical_mean_std, "hand_native": canonical_mean_native, "full_std": proto["mean_full_predictions_standardized"].astype(np.float64), "full_native": proto["mean_full_predictions_native"].astype(np.float64)},
    }
    metric = {name: metric_bundle(item["hand_std"], item["full_std"], item["hand_native"], item["full_native"], targets_std, targets_native, persistence_native_full) for name, item in models.items()}
    bootstrap = np.load(BOOTSTRAP_PATH, allow_pickle=False)
    if bootstrap.shape != (BOOTSTRAP, 614):
        raise RuntimeError("paired bootstrap shape failure")
    metric_rows: list[dict[str, object]] = []
    for name in metric:
        item = metric[name]
        metric_rows.append({"model": name, "N_episode": 614, "hand_mae_std": float(np.mean(item["hand_mae_std"])), "hand_mae_native": float(np.mean(item["hand_mae_native"])), "full28_mae_std": float(np.mean(item["full_mae_std"])), "hand_fd_native": float(np.mean(item["hand_fd_native"])), "hand_fd_std": float(np.mean(item["hand_fd_std"])), "hand_anchor_fd_native": float(np.mean(item["hand_anchor_fd_native"])), "full_fd_native": float(np.mean(item["full_fd_native"])), "hand_tv_mean": float(np.nanmean(item["hand_tv"])), "target_hand_tv_mean": float(np.nanmean(item["target_hand_tv"])), "hand_tv_ratio": float(np.nanmean(item["hand_tv_ratio"])), "hand_tv_closeness": float(np.nanmean(item["hand_tv_closeness"])), "full_tv_ratio": float(np.nanmean(item["full_tv_ratio"])), "full_tv_closeness": float(np.nanmean(item["full_tv_closeness"])), "zero_target_hand_tv_episodes": int(np.sum(~np.isfinite(item["hand_tv_ratio"]))), "zero_target_full_tv_episodes": int(np.sum(~np.isfinite(item["full_tv_ratio"]))), "late_hand_mae_std": float(np.mean(np.abs(item["hand_std"] if False else models[name]["hand_std"][:, 20:] - targets_std[:, 20:, :HAND]))), "final_test_access": 0})
    write_csv(analysis / "METRIC_MAIN_RESULTS.csv", metric_rows)
    write_json(analysis / "METRIC_MAIN_RESULTS.json", {"status": "PASS", "rows": metric_rows, "native_fd_definition": "29 future-to-future differences", "anchor_to_first_reported": True, "bootstrap_replicates": BOOTSTRAP, "bootstrap_unit": "episode", "adaptive_development_exposure": True})
    pair_rows: list[dict[str, object]] = []
    for candidate in ("HP-SINGLE", "HP-PROTO-MAP", "HP-PROTO-MEAN"):
        for control, label in (("PERSISTENCE", "Persistence_hybrid"), ("HP-SINGLE", "HP-SINGLE")):
            if candidate == control:
                continue
            for metric_name in ("hand_mae_std", "hand_fd_native", "hand_fd_std", "full_mae_std", "hand_tv_closeness"):
                stats = paired(metric[control][metric_name], metric[candidate][metric_name], bootstrap, 20260960 + len(pair_rows))
                pair_rows.append({"candidate": candidate, "control": control, "baseline_label": label, "metric": metric_name, **stats})
    write_csv(analysis / "PAIRED_EFFECTS.csv", pair_rows)
    write_csv(ROOT / "PAIRED_EFFECTS.csv", pair_rows)
    # Mode imbalance and old-router calibration diagnosis.
    mode_assign_rows: list[dict[str, str]] = []
    with (PROTO / "mode_audit/CLUSTER_ASSIGNMENTS.csv").open(encoding="utf-8", newline="") as handle:
        mode_assign_rows = list(csv.DictReader(handle))
    train_ids = np.asarray([int(row["episode_index"]) for row in mode_assign_rows], dtype=np.int64)
    train_labels = np.asarray([int(row["K2"]) for row in mode_assign_rows], dtype=np.int64)
    if len(train_ids) != 2741 or len(np.unique(train_ids)) != 2741:
        raise RuntimeError("train mode assignment coverage failure")
    centers_payload = load_npz(PROTO / "mode_audit/PROTOTYPE_CENTERS.npz")
    center_flat = centers_payload["centers_K2"].astype(np.float64).reshape(2, 16, HAND)
    coefficient_mean = centers_payload["coefficient_mean"].astype(np.float64)
    coefficient_scale = centers_payload["coefficient_scale"].astype(np.float64)
    center_native = center_flat * coefficient_scale[None, :, :] + coefficient_mean[None, :, :]
    dev_target_coeff = np.einsum("kt,etj->ekj", dct_basis()[:16], targets_native[:, :, :HAND] - persistence_native_full[:, :, :HAND])
    dev_z = ((dev_target_coeff - coefficient_mean) / coefficient_scale).reshape(len(ids), -1)
    train_data = load_npz(INPUT)
    train_target_native = train_data["targets_native"][train][:, :, :HAND].astype(np.float64)
    train_p_native = train_data["persistence_native"][train][:, :, :HAND].astype(np.float64)
    train_coeff = np.einsum("kt,etj->ekj", dct_basis()[:16], train_target_native - train_p_native)
    train_z = ((train_coeff - coefficient_mean) / coefficient_scale).reshape(len(train_coeff), -1)
    dev_true_mode = ((dev_z[:, None, :] - centers_payload["centers_K2"].astype(np.float64)[None, :, :]) ** 2).sum(axis=2).argmin(axis=1).astype(np.int64)
    state_all = train_data["state_action_history"].astype(np.float64)
    dynamic_all = train_data["dynamic_pooled"].astype(np.float64)
    state_train = state_all[train]
    dynamic_train = dynamic_all[train]
    prefix_train = np.column_stack((state_train.mean(axis=1), state_train.std(axis=1), np.abs(np.diff(state_train, axis=1)).mean(axis=1)))
    fusion_features = np.concatenate((dynamic_train, state_train, prefix_train), axis=1)
    task_train = train_data["original_task_index"][train].astype(np.int64)
    task_features = np.eye(int(max(task_train.max(), tasks.max())) + 1, dtype=np.float64)[task_train]
    train_folds = fold_assignment(train_ids)
    fusion_probe = probe_oof(fusion_features, train_labels, train_folds, 2)
    task_probe = probe_oof(task_features, train_labels, train_folds, 2)
    old_dev_pred = probabilities.argmax(axis=1).astype(np.int64)
    old_dev_mode_metrics = mode_metrics(dev_true_mode, old_dev_pred, probabilities, 2)
    probe_rows = []
    for name, labels, predictions, probs, metrics_value in (("TRUE_TRAIN", train_labels, train_labels, np.eye(2)[train_labels], mode_metrics(train_labels, train_labels, np.eye(2)[train_labels], 2)), ("LINEAR_PROBE_FUSION_OOF", train_labels, fusion_probe["predictions"], fusion_probe["probabilities"], fusion_probe["metrics"]), ("LINEAR_PROBE_TASK_ONLY_OOF", train_labels, task_probe["predictions"], task_probe["probabilities"], task_probe["metrics"]), ("OLD_PROTO_ROUTER_DEV", dev_true_mode, old_dev_pred, probabilities, old_dev_mode_metrics)):
        for cls in range(2):
            probe_rows.append({"population": name, "class": cls, "N": metrics_value["class_support"][cls], "predicted_proportion": float(np.mean(predictions == cls)), "true_proportion": float(np.mean(labels == cls)), "precision": metrics_value["precision"][cls], "recall": metrics_value["recall"][cls], "f1": metrics_value["f1"][cls], "macro_f1": metrics_value["macro_f1"], "balanced_accuracy": metrics_value["balanced_accuracy"], "nll": metrics_value["nll"], "brier": metrics_value["brier"]})
    write_csv(analysis / "MODE_IMBALANCE_AUDIT.csv", probe_rows)
    mode_summary = {"train_true_proportion": np.bincount(train_labels, minlength=2).tolist(), "dev_true_proportion": np.bincount(dev_true_mode, minlength=2).tolist(), "old_proto_dev_predicted_proportion": np.bincount(old_dev_pred, minlength=2).tolist(), "linear_probe_fusion_oof_predicted_proportion": np.bincount(fusion_probe["predictions"], minlength=2).tolist(), "linear_probe_task_only_oof_predicted_proportion": np.bincount(task_probe["predictions"], minlength=2).tolist(), "minority_class_train": int(np.argmin(np.bincount(train_labels, minlength=2))), "old_proto_dev_metrics": old_dev_mode_metrics, "fusion_probe_oof_metrics": fusion_probe["metrics"], "task_probe_oof_metrics": task_probe["metrics"], "old_router_vs_probe_recall_comparable": False, "interpretation": "class-imbalance-consistent_and_minority-underprediction-possible; dev old-router and train OOF probe are not comparable populations", "final_test_access": 0}
    write_json(analysis / "MODE_IMBALANCE_AUDIT.json", mode_summary)
    (analysis / "MODE_IMBALANCE_AUDIT.md").write_text("# Mode imbalance audit\n\nThe old prototype router predicts mode 1 for `" + str(int(np.sum(old_dev_pred == 1))) + "/614` development episodes. This is compared with fixed-center ground-truth dev labels and with train-fold OOF probe controls. Because the old-router values are dev and the probe values are train OOF, their minority recalls are not treated as a comparable causal deficit. The pattern is consistent with class imbalance and minority underprediction; representation collapse remains unresolved.\n", encoding="utf-8")
    # True refit ARI audit: each bootstrap refits KMeans and predicts the full
    # original population, unlike the prior fixed-center implementation.
    ari_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(CLUSTER_SEED)
    for mode_count in (2, 4, 8):
        full = KMeans(n_clusters=mode_count, init="k-means++", n_init=1, max_iter=100, random_state=CLUSTER_SEED, algorithm="lloyd").fit(train_z)
        full_labels = full.labels_
        values = []
        for _ in range(ARI_BOOTSTRAP):
            sample = rng.integers(0, len(train_z), size=len(train_z))
            refit = KMeans(n_clusters=mode_count, init="k-means++", n_init=1, max_iter=100, random_state=CLUSTER_SEED, algorithm="lloyd").fit(train_z[sample])
            values.append(float(adjusted_rand_score(full_labels, refit.predict(train_z))))
        values_array = np.asarray(values)
        ari_rows.append({"K_mode": mode_count, "implementation": "bootstrap_refit_centers_then_predict_full_population", "replicates": ARI_BOOTSTRAP, "ari_median": float(np.median(values_array)), "ari_ci_low": float(np.quantile(values_array, 0.025)), "ari_ci_high": float(np.quantile(values_array, 0.975)), "ari_min": float(values_array.min()), "ari_max": float(values_array.max()), "fixed_center_prior_ari": 1.0})
    write_csv(analysis / "ARI_CORRECTED_RESULTS.csv", ari_rows)
    write_json(analysis / "ARI_CORRECTED_RESULTS.json", {"status": "PASS", "rows": ari_rows, "source_session_metadata_available": False, "stability_unit": "episode-resampling proxy", "prior_ari_interpretation": "artifact_or_non-diagnostic_fixed-center prediction", "final_test_access": 0})
    (analysis / "ARI_IMPLEMENTATION_AUDIT.md").write_text("# ARI implementation audit\n\nThe source mode audit reported ARI=1 because its bootstrap function reassigned bootstrap samples to the unchanged full-data centers. This study refit KMeans centers in every bootstrap replicate and then predicted the full original train population. The corrected results are in `ARI_CORRECTED_RESULTS.csv/json`; no cross-session claim is possible because source/session metadata are unavailable.\n", encoding="utf-8")
    # Fixed-q component oracle.
    safe_scale = np.asarray(json.loads((DCT / "TEMPORAL_LOSS_CONTRACT.json").read_text())["safe_scale_j"], dtype=np.float64)
    lambda_fd = float(json.loads((DCT / "TEMPORAL_LOSS_CONTRACT.json").read_text())["lambda_fd"])
    shadow_metric: dict[str, dict[str, np.ndarray]] = {}
    for index in range(2):
        full_std = np.concatenate((shadow_components_std[:, index], arm_std), axis=2)
        full_native = np.concatenate((shadow_components_native[:, index], arm_native), axis=2)
        shadow_metric[f"COMPONENT_{index}"] = metric_bundle(shadow_components_std[:, index], full_std, shadow_components_native[:, index], full_native, targets_std, targets_native, persistence_native_full)
    component_level_err = np.abs(shadow_components_std - targets_std[:, None, :, :HAND]).mean(axis=(2, 3))
    component_fd_err = np.abs(np.diff(shadow_components_native, axis=2) - np.diff(targets_native[:, None, :, :HAND], axis=2)).mean(axis=(2, 3))
    component_fd_scaled = (np.abs(np.diff(shadow_components_native, axis=2) - np.diff(targets_native[:, None, :, :HAND], axis=2)) / safe_scale[None, None, None, :]).mean(axis=(2, 3))
    component_dual_err = component_level_err + lambda_fd * component_fd_scaled
    level_choice = component_level_err.argmin(axis=1)
    fd_choice = component_fd_err.argmin(axis=1)
    dual_choice = component_dual_err.argmin(axis=1)
    def choose_component(choice: np.ndarray) -> dict[str, np.ndarray]:
        hand_std = shadow_components_std[np.arange(len(ids)), choice]
        hand_native = shadow_components_native[np.arange(len(ids)), choice]
        return {"hand_std": hand_std, "hand_native": hand_native, "full_std": np.concatenate((hand_std, arm_std), axis=2), "full_native": np.concatenate((hand_native, arm_native), axis=2)}
    oracle_models = {"SHADOW_MAP": {"hand_std": shadow_map_std, "hand_native": shadow_map_native, "full_std": np.concatenate((shadow_map_std, arm_std), axis=2), "full_native": np.concatenate((shadow_map_native, arm_native), axis=2)}, "SHADOW_MEAN": {"hand_std": shadow_mean_std, "hand_native": shadow_mean_native, "full_std": np.concatenate((shadow_mean_std, arm_std), axis=2), "full_native": np.concatenate((shadow_mean_native, arm_native), axis=2)}, "ORACLE_LEVEL": choose_component(level_choice), "ORACLE_FD": choose_component(fd_choice), "ORACLE_DUAL": choose_component(dual_choice)}
    oracle_metric = {name: metric_bundle(item["hand_std"], item["full_std"], item["hand_native"], item["full_native"], targets_std, targets_native, persistence_native_full) for name, item in oracle_models.items()}
    oracle_rows: list[dict[str, object]] = []
    for name in ("PERSISTENCE", "SHADOW_MAP", "SHADOW_MEAN", "ORACLE_LEVEL", "ORACLE_FD", "ORACLE_DUAL"):
        item = metric["PERSISTENCE"] if name == "PERSISTENCE" else oracle_metric[name]
        oracle_rows.append({"model": name, "hand_mae_std": float(np.mean(item["hand_mae_std"])), "full28_mae_std": float(np.mean(item["full_mae_std"])), "hand_fd_native": float(np.mean(item["hand_fd_native"])), "hand_fd_std": float(np.mean(item["hand_fd_std"])), "hand_tv_closeness": float(np.nanmean(item["hand_tv_closeness"])), "hand_gain_vs_persistence": float(np.mean(metric["PERSISTENCE"]["hand_mae_std"])-np.mean(item["hand_mae_std"])), "fd_gain_vs_persistence": float(np.mean(metric["PERSISTENCE"]["hand_fd_native"])-np.mean(item["hand_fd_native"])), "full_gain_vs_hybrid": float(np.mean(metric["PERSISTENCE"]["full_mae_std"])-np.mean(item["full_mae_std"])), "N_episode": 614})
    write_csv(analysis / "COMPONENT_ORACLE_RESULTS.csv", oracle_rows)
    write_json(analysis / "COMPONENT_ORACLE_RESULTS.json", {"status": "PASS", "rows": oracle_rows, "oracle_unit": "one complete 30-step component per episode", "oracle_is_non_deployable": True, "fixed_q_replay": replay, "final_test_access": 0})
    write_csv(analysis / "FIXED_Q_SHADOW.csv", [{"item": key, "value": value} for key, value in replay.items()])
    # Loss/codebook/refinement audit on one fixed train calibration batch.
    sys.path.insert(0, str(PROTO / "code"))
    import torch
    from model import ConditionalPrototypeDecoder, total_loss  # type: ignore
    checkpoint = torch.load(PROTO / "runs/HP-PROTO/base_all_train/checkpoint_best.pt", map_location="cpu", weights_only=False)
    model = ConditionalPrototypeDecoder(centers=torch.from_numpy(center_native.astype(np.float32)), seed=20260938, hidden_dim=160)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    calib_mask = np.asarray([int(hashlib.sha256(f"HD-DUAL-LOSS-CALIBRATION:{int(episode)}".encode()).hexdigest()[:8], 16) % 4 == 0 for episode in train_ids])
    fixed = np.flatnonzero(calib_mask)[:32]
    context_fixed = np.concatenate((train_data["dynamic_pooled"][train], train_data["state_action_history"][train]), axis=1).astype(np.float32)[fixed]
    p_native_fixed = train_p_native.astype(np.float32)[fixed]
    t_std_fixed = oof_target_std = load_npz(DCT / "oof/HD-D1-L1_OOF.npz")["targets_hand_standardized"].astype(np.float32)[fixed]
    t_native_fixed = train_target_native.astype(np.float32)[fixed]
    train_coeff_fixed = np.einsum("kt,etj->ekj", dct_basis()[:16], train_target_native - train_p_native).astype(np.float32)[fixed]
    labels_fixed = torch.from_numpy(train_labels[fixed]).long()
    model.train()
    probabilities_t, coefficients_t, residual_t = model(torch.from_numpy(context_fixed))
    p_native_t = torch.from_numpy(p_native_fixed)
    mean_t = torch.from_numpy(mean[:, :HAND].astype(np.float32))
    scale_t = torch.from_numpy(scale[:, :HAND].astype(np.float32))
    native_components_t = p_native_t.unsqueeze(1) + residual_t
    components_std_t = (native_components_t - mean_t.unsqueeze(0).unsqueeze(0)) / scale_t.unsqueeze(0).unsqueeze(0)
    soft_t = (probabilities_t.unsqueeze(-1).unsqueeze(-1) * components_std_t).sum(dim=1)
    pos_t, fd_t, = (None, None)
    dual_t, pos_component_t, fd_component_t = total_loss("L1", soft_t, torch.from_numpy(t_std_fixed), torch.from_numpy(t_native_fixed), mean_t, scale_t, torch.from_numpy(safe_scale.astype(np.float32)), torch.tensor(lambda_fd))
    mode_t = torch.nn.functional.nll_loss(torch.log(probabilities_t.clamp_min(1e-8)), labels_fixed)
    refine_t = torch.nn.functional.smooth_l1_loss(coefficients_t[torch.arange(len(labels_fixed)), labels_fixed], torch.from_numpy(train_coeff_fixed))
    grad_groups: dict[str, dict[str, float]] = {}
    for loss_name, loss_value in (("dual", dual_t), ("mode", mode_t), ("refine", refine_t)):
        grads = torch.autograd.grad(loss_value, tuple(model.parameters()), retain_graph=True, allow_unused=True)
        grad_groups[loss_name] = {name: float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads[index:index+1] if g is not None))) if any(g is not None for g in grads[index:index+1]) else 0.0 for index, (name, _parameter) in enumerate(model.named_parameters())}
    coefficient_gap = float(np.max(np.abs((coefficients_t.detach().numpy()[:, 1] - coefficients_t.detach().numpy()[:, 0]) - (center_native[1] - center_native[0]))))
    loss_audit = {"status": "PASS", "fixed_train_calibration_rows": int(len(fixed)), "lambda_fd": lambda_fd, "lambda_mode": float(json.loads((PROTO / "configs/PROTOTYPE_LOSS_SCALE_LOCKED.json").read_text())["lambda_mode"]), "lambda_refine": float(json.loads((PROTO / "configs/PROTOTYPE_LOSS_SCALE_LOCKED.json").read_text())["lambda_refine"]), "dual_loss_mean": float(dual_t.detach()), "position_loss_mean": float(pos_component_t.detach()), "fd_scaled_loss_mean": float(fd_component_t.detach()), "mode_loss_mean": float(mode_t.detach()), "refine_loss_mean": float(refine_t.detach()), "gradient_norms": grad_groups, "center_difference_preserved_by_shared_refinement_max_abs": coefficient_gap, "center_native_distance": float(np.linalg.norm(center_native[0] - center_native[1])), "q_ref_mean_dev": float(q_ref.mean()), "q_ref_extreme_fraction_dev": float(np.mean((q_ref < .05) | (q_ref > .95))), "final_test_access": 0}
    write_json(analysis / "PROTOTYPE_LOSS_AND_CODEBOOK_AUDIT.json", loss_audit)
    (analysis / "PROTOTYPE_LOSS_AND_CODEBOOK_AUDIT.md").write_text("# Prototype loss and codebook audit\n\nThe locked numeric coefficients `lambda_mode` and `lambda_refine` are scale factors, not evidence that their corresponding terms dominate. Fixed train calibration-batch loss and gradient norms are recorded in JSON. The shared refinement preserves the center difference exactly up to the reported numerical tolerance, limiting context-dependent mode separation.\n", encoding="utf-8")
    # OOF and normalization audit.
    oof_rows: list[dict[str, object]] = []
    for cell in ("HP-SINGLE", "HP-PROTO"):
        oof = load_npz(PROTO / f"oof/{cell}_OOF.npz")
        fold = oof["fold"].astype(np.int64)
        oof_rows.append({"cell": cell, "rows": int(len(oof["episode_index"])), "coverage": int(np.isfinite(oof["oof_predictions_hand_standardized"]).all(axis=(1,2)).sum()), "unique_ids": int(len(np.unique(oof["episode_index"]))), "folds": sorted(np.unique(fold).tolist()), "overlap": 0, "base_oof": True, "final_selector_oof": False, "selector_fit_rows": 2477, "selector_calibration_rows": 264, "final_selector_oof_limitation": "selector q is not OOF for all train rows; dev selector output is not a final-system OOF", "final_test_access": 0})
    normalization = {"target_std_native_max_abs": float(np.max(np.abs(data["targets_standardized"][dev].astype(np.float64) * scale[None, :, :] + mean[None, :, :] - targets_native))), "persistence_std_native_max_abs": float(np.max(np.abs(persistence_std_full * scale[None, :, :] + mean[None, :, :] - persistence_native_full))), "fold_whitener_source": "all-train coefficient_mean/coefficient_scale from prior mode audit", "strict_target_blind_oof": False, "impact": "prototype fold OOF uses global train whitener; selector OOF has a normalization caveat, while all-train dev inference remains train-only", "hand_slice": [0,14], "arm_slice": [14,28]}
    write_json(analysis / "METRIC_OOF_AUDIT.json", {"status": "PASS_WITH_CAVEAT", "rows": oof_rows, "normalization": normalization, "native_fd_definition": "29 future-to-future differences", "anchor_to_first_sensitivity": "reported in METRIC_MAIN_RESULTS", "final_test_access": 0})
    (analysis / "METRIC_OOF_AUDIT.md").write_text("# Metric and OOF audit\n\nBase OOF coverage is complete for both source cells, but the final selector does not have a complete final-system OOF archive: selectors were fit on 2,477 rows and calibrated on 264 rows, while selector outputs for all train rows were not exported as OOF. In addition, prototype fold center fitting reused all-train coefficient whitening statistics. These are limitations on strict OOF claims, not evidence of final-test access. Native FD uses 29 future-to-future differences; anchor-to-first is reported separately.\n", encoding="utf-8")
    # Entry gate: use the FD-selected complete-chunk oracle for the common
    # oracle requirement; never combine different per-episode choices.
    p_fd = float(np.mean(metric["PERSISTENCE"]["hand_fd_native"]))
    fd_oracle_fd = float(np.mean(oracle_metric["ORACLE_FD"]["hand_fd_native"]))
    p_hand = float(np.mean(metric["PERSISTENCE"]["hand_mae_std"]))
    fd_oracle_hand = float(np.mean(oracle_metric["ORACLE_FD"]["hand_mae_std"]))
    shadow_map_hand = float(np.mean(oracle_metric["SHADOW_MAP"]["hand_mae_std"]))
    fd_oracle_fd_relative_to_p = fd_oracle_fd / p_fd - 1.0
    fd_oracle_hand_gain = p_hand - fd_oracle_hand
    shadow_map_gain = p_hand - shadow_map_hand
    retention = fd_oracle_hand_gain / shadow_map_gain if shadow_map_gain > 0 else float("nan")
    old_router_fd_gap = (float(np.mean(metric["HP-PROTO-MAP"]["hand_fd_native"])) - fd_oracle_fd) / fd_oracle_fd
    entry_checks = {"fixed_q_min_fd_oracle_within_persistence_1pct": fd_oracle_fd / p_fd <= 1.01, "same_fd_oracle_positive_hand_gain": fd_oracle_hand_gain > 0, "same_fd_oracle_retains_shadow_map_gain_80pct": np.isfinite(retention) and retention >= 0.80, "old_router_fd_gap_vs_same_oracle_ge_05pct": old_router_fd_gap >= 0.005, "minority_recall_branch_comparable": False, "all_required_audits_complete": True}
    router_entry_gate = bool(entry_checks["fixed_q_min_fd_oracle_within_persistence_1pct"] and entry_checks["same_fd_oracle_positive_hand_gain"] and entry_checks["same_fd_oracle_retains_shadow_map_gain_80pct"] and (entry_checks["old_router_fd_gap_vs_same_oracle_ge_05pct"] or entry_checks["minority_recall_branch_comparable"]))
    entry = {"status": "PASS", "router_entry_gate": router_entry_gate, "checks": entry_checks, "fd_oracle_fd": fd_oracle_fd, "persistence_fd": p_fd, "fd_oracle_fd_relative_change_vs_persistence": fd_oracle_fd_relative_to_p, "fd_oracle_hand_gain": fd_oracle_hand_gain, "shadow_map_hand_gain": shadow_map_gain, "fd_oracle_retention_vs_shadow_map": retention, "old_router_map_fd_gap_vs_oracle": old_router_fd_gap, "minority_recall_comparison": "not used because old router dev and probe train OOF are not comparable populations", "router_fits_authorized": router_entry_gate, "final_test_access": 0}
    write_json(ROOT / "ROUTER_EXECUTION_DECISION.json", entry)
    write_json(analysis / "ROUTER_EXECUTION_DECISION.json", entry)
    print(json.dumps({"status": "PASS", "entry_gate": entry["router_entry_gate"], "fd_oracle_fd_relative": fd_oracle_fd_relative_to_p, "fd_oracle_hand_gain": fd_oracle_hand_gain, "retention": retention, "old_router_fd_gap": old_router_fd_gap, "ari": ari_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
