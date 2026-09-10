"""Independent acceptance audit and harvest for the batch-32 rescue study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ee6008.config import load_config
from ee6008.full_attentive_data import load_v3_population

SEEDS = (20260825, 20260826, 20260827)
NUM_CLASSES = 120
VALIDATION_COUNT = 617
TRAIN_COUNT = 2757
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260829
METRICS = (
    "top1",
    "top5",
    "mean_class_recall_at_1",
    "mean_class_recall_at_5",
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "mean_max_softmax_probability",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def numpy_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if labels.shape != (len(labels),) or logits.shape != (len(labels), NUM_CLASSES):
        raise ValueError(f"invalid label/logit shape: {labels.shape}, {logits.shape}")
    if not np.isfinite(logits).all():
        raise FloatingPointError("non-finite logits")
    predictions = np.argmax(logits, axis=1)
    ranking = np.argsort(-logits, axis=1, kind="stable")
    top5 = ranking[:, :5]
    correct = predictions == labels
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted).astype(np.float32)
    normalizer = np.sum(exp_logits, axis=1, keepdims=True, dtype=np.float32)
    probabilities = (exp_logits / normalizer).astype(np.float32)
    log_probabilities = shifted - np.log(normalizer).astype(np.float32)
    confidence = np.max(probabilities, axis=1)
    recalls: list[float] = []
    recalls5: list[float] = []
    f1s: list[float] = []
    per_class: list[dict[str, Any]] = []
    for class_index in range(NUM_CLASSES):
        class_mask = labels == class_index
        support = int(class_mask.sum())
        if support == 0:
            continue
        true_positive = int(np.sum(class_mask & (predictions == class_index)))
        false_positive = int(np.sum(~class_mask & (predictions == class_index)))
        recall = true_positive / support
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recall5 = float(np.mean(np.any(top5[class_mask] == class_index, axis=1)))
        recalls.append(recall)
        recalls5.append(recall5)
        f1s.append(f1)
        per_class.append(
            {
                "class_index": class_index,
                "support": support,
                "recall": recall,
                "recall_at_5": recall5,
                "f1": f1,
            }
        )
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        in_bin = (
            (confidence >= lower) & (confidence <= upper)
            if bin_index == 14
            else (confidence >= lower) & (confidence < upper)
        )
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(
                float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin]))
            )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return {
        "top1": float(np.mean(correct)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "mean_class_recall_at_1": float(np.mean(recalls)),
        "mean_class_recall_at_5": float(np.mean(recalls5)),
        "macro_f1": float(np.mean(f1s)),
        "nll": float(-np.mean(log_probabilities[np.arange(len(labels)), labels])),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "mean_max_softmax_probability": float(np.mean(confidence)),
        "correct": int(np.sum(correct)),
        "total": len(labels),
        "predictions": predictions,
        "top5_indices": top5,
        "confidence": confidence,
        "correct_array": correct,
        "per_class": per_class,
    }


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as records:
        required = {"episode_index", "labels", "logits", "predictions"}
        if not required.issubset(records.files):
            raise ValueError(f"prediction keys incomplete: {path}")
        episodes = records["episode_index"].astype(np.int64)
        labels = records["labels"].astype(np.int64)
        logits = records["logits"].astype(np.float32)
        stored_predictions = records["predictions"].astype(np.int64)
    if len(episodes) != VALIDATION_COUNT or len(np.unique(episodes)) != VALIDATION_COUNT:
        raise ValueError(f"validation episode IDs are not unique/complete: {path}")
    if not np.array_equal(stored_predictions, np.argmax(logits, axis=1)):
        raise ValueError(f"stored predictions disagree with logits: {path}")
    return {
        "path": path,
        "episode_index": episodes,
        "labels": labels,
        "logits": logits,
        "metrics": numpy_metrics(labels, logits),
    }


def compare_metric_dict(actual: dict[str, Any], recorded: dict[str, Any]) -> dict[str, float]:
    return {
        name: abs(float(actual[name]) - float(recorded[name]))
        for name in METRICS
        if name in recorded
    }


def parse_curves(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "best_so_far":
                row[key] = value.strip().lower() == "true"
            elif key == "host_load_1m":
                row[key] = None if value in {"", "None"} else float(value)
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
        rows.append(row)
    return rows


def replay_early_stopping(curves: list[dict[str, Any]]) -> dict[str, Any]:
    best_epoch = None
    best_macro = -math.inf
    best_nll = math.inf
    bad_epochs = 0
    stop_epoch = None
    replay_rows = []
    for row in curves:
        epoch = int(row["epoch"])
        macro = float(row["validation_macro_f1"])
        nll = float(row["validation_nll"])
        if best_epoch is None:
            improved = True
        else:
            difference = macro - best_macro
            improved = difference > 0.001 or (
                abs(difference) <= 0.001 and nll < best_nll
            )
        if improved:
            best_epoch = epoch
            best_macro = macro
            best_nll = nll
            bad_epochs = 0
        else:
            bad_epochs += 1
        should_stop = epoch >= 5 and bad_epochs >= 5
        replay_rows.append(
            {
                "epoch": epoch,
                "expected_best_so_far": improved,
                "recorded_best_so_far": bool(row["best_so_far"]),
                "expected_bad_epochs": bad_epochs,
                "recorded_bad_epochs": int(row["bad_epochs"]),
                "should_stop": should_stop,
            }
        )
        if should_stop:
            stop_epoch = epoch
            break
    if not curves or best_epoch is None:
        raise ValueError("empty epoch curve")
    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch or int(curves[-1]["epoch"]),
        "stop_reason": "patience_reached" if stop_epoch is not None else "max_epochs_reached",
        "bad_epochs_at_stop": replay_rows[-1]["expected_bad_epochs"],
        "rows": replay_rows,
        "state_matches_recorded": all(
            item["expected_best_so_far"] == item["recorded_best_so_far"]
            and item["expected_bad_epochs"] == item["recorded_bad_epochs"]
            for item in replay_rows
        ),
    }


def parse_checkpoint(path: Path, *, kind: str, seed: int, epoch: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    if checkpoint.get("kind") != kind or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint kind/seed mismatch: {path}")
    if int(checkpoint.get("epoch", -1)) != epoch:
        raise ValueError(f"checkpoint epoch mismatch: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint model state missing: {path}")
    if any(str(key).startswith("encoder") for key in state):
        raise ValueError(f"checkpoint contains encoder state: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": kind,
        "epoch": epoch,
        "model_state_keys": len(state),
        "model_parameter_count": sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)),
    }


def validate_run(
    root: Path,
    *,
    variant: str,
    expected_batch: int,
    expected_config_sha256: str,
    expected_feature_manifest_sha256: str,
    validation_episode_index: np.ndarray,
    validation_labels: np.ndarray,
    expected_optimizer_steps: int | None,
) -> dict[str, Any]:
    complete_path = root / "RUN_COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = read_json(complete_path)
    if complete.get("status") != "pass" or set(complete.get("completed_seeds", [])) != set(SEEDS):
        raise ValueError(f"incomplete run marker: {complete_path}")
    policy = complete.get("policy", {})
    if int(policy.get("batch_size", -1)) != expected_batch:
        raise ValueError(f"batch size mismatch in {complete_path}")
    if expected_batch == 32 and int(policy.get("gradient_accumulation_steps", -1)) != 1:
        raise ValueError("batch32 run does not prove accumulation=1")
    if complete.get("config_sha256") != expected_config_sha256:
        raise ValueError(f"config hash mismatch: {complete_path}")
    if complete.get("feature_manifest_sha256") != expected_feature_manifest_sha256:
        raise ValueError(f"feature manifest hash mismatch: {complete_path}")
    final_test_selected = complete.get("final_test_selected", 0)
    if final_test_selected not in {0} or complete.get("final_test_access") != "none":
        raise ValueError(f"final-test guard failed: {complete_path}")
    infos: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"
        summary = read_json(seed_dir / "seed_complete.json")
        if summary.get("status") != "pass" or int(summary.get("seed", -1)) != seed:
            raise ValueError(f"seed completion marker invalid: {seed_dir}")
        curves = parse_curves(seed_dir / "epoch_curves.csv")
        epochs = [int(row["epoch"]) for row in curves]
        expected_epochs = list(range(1, int(summary["epochs_completed"]) + 1))
        if epochs != expected_epochs:
            raise ValueError(f"epoch rows are not continuous: {seed_dir}")
        epoch_paths = sorted(seed_dir.glob("predictions_epoch_*.npz"))
        epoch_numbers = [int(path.stem.rsplit("_", 1)[1]) for path in epoch_paths]
        if epoch_numbers != expected_epochs:
            raise ValueError(f"epoch prediction rows are not continuous: {seed_dir}")
        if expected_batch == 32:
            if any(int(row["optimizer_steps"]) != expected_optimizer_steps for row in curves):
                raise ValueError(f"physical batch32 optimizer steps mismatch: {seed_dir}")
            if any(int(row["samples_exposed"]) != TRAIN_COUNT for row in curves):
                raise ValueError(f"sample exposure mismatch: {seed_dir}")
        replay = replay_early_stopping(curves)
        best = load_prediction(seed_dir / "predictions_best.npz")
        last = load_prediction(seed_dir / "predictions_last.npz")
        for prediction in (best, last):
            if not np.array_equal(prediction["episode_index"], validation_episode_index):
                raise ValueError(f"validation episode order mismatch: {prediction['path']}")
            if not np.array_equal(prediction["labels"], validation_labels):
                raise ValueError(f"validation labels mismatch: {prediction['path']}")
        best_diffs = compare_metric_dict(best["metrics"], summary["best_metrics"])
        last_diffs = compare_metric_dict(last["metrics"], summary["last_metrics"])
        if not best_diffs or not last_diffs or max(best_diffs.values()) > 2e-5 or max(last_diffs.values()) > 2e-5:
            raise ValueError(f"independent metric mismatch: {seed_dir}")
        if replay["best_epoch"] != int(summary["best_epoch"]):
            raise ValueError(f"best epoch replay mismatch: {seed_dir}")
        if replay["stop_epoch"] != int(summary["epochs_completed"]):
            raise ValueError(f"stop epoch replay mismatch: {seed_dir}")
        if replay["stop_reason"] != summary["stop_reason"] or not replay["state_matches_recorded"]:
            raise ValueError(f"early stopping replay mismatch: {seed_dir}")
        checkpoints = {
            "best": parse_checkpoint(
                seed_dir / "checkpoint_best.pt",
                kind="best",
                seed=seed,
                epoch=int(summary["best_epoch"]),
            ),
            "last": parse_checkpoint(
                seed_dir / "checkpoint_last.pt",
                kind="last",
                seed=seed,
                epoch=int(summary["epochs_completed"]),
            ),
        }
        if summary.get("checkpoint_best_sha256") != checkpoints["best"]["sha256"] or summary.get("checkpoint_last_sha256") != checkpoints["last"]["sha256"]:
            raise ValueError(f"checkpoint hash marker mismatch: {seed_dir}")
        epoch_metric_max = 0.0
        for epoch, path in zip(expected_epochs, epoch_paths):
            prediction = load_prediction(path)
            if not np.array_equal(prediction["episode_index"], validation_episode_index) or not np.array_equal(prediction["labels"], validation_labels):
                raise ValueError(f"epoch prediction alignment mismatch: {path}")
            curve = curves[epoch - 1]
            recorded = {
                "top1": curve["validation_top1"],
                "top5": curve["validation_top5"],
                "mean_class_recall_at_1": curve["validation_mean_class_recall_at_1"],
                "mean_class_recall_at_5": curve["validation_mean_class_recall_at_5"],
                "macro_f1": curve["validation_macro_f1"],
                "nll": curve["validation_nll"],
                "brier": curve["validation_brier"],
                "ece": curve["validation_ece"],
                "mean_max_softmax_probability": curve["validation_mean_max_softmax_probability"],
            }
            epoch_metric_max = max(epoch_metric_max, max(compare_metric_dict(prediction["metrics"], recorded).values()))
        if epoch_metric_max > 2e-5:
            raise ValueError(f"epoch curve metric mismatch: {seed_dir}")
        infos[seed] = {
            "summary": summary,
            "curves": curves,
            "best": best,
            "last": last,
            "replay": replay,
            "checkpoints": checkpoints,
            "best_metric_max_abs_diff": max(best_diffs.values()),
            "last_metric_max_abs_diff": max(last_diffs.values()),
        }
    return {
        "root": root,
        "variant": variant,
        "complete": complete,
        "seed_infos": infos,
        "status": "valid_early_stopped" if any(info["summary"]["stop_reason"] == "patience_reached" for info in infos.values()) else "valid_complete",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "sample_std": float(np.std([float(row[name]) for row in rows], ddof=1)),
        }
        for name in METRICS
    }


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * lower)
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "total": len(a),
    }


def bootstrap_pair(
    a: dict[int, dict[str, Any]],
    b: dict[int, dict[str, Any]],
    kind: str,
    *,
    rng_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    top1 = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    macro = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    seedwise: dict[str, dict[str, np.ndarray]] = {
        str(seed): {
            "top1": np.empty(BOOTSTRAP_REPETITIONS),
            "macro_f1": np.empty(BOOTSTRAP_REPETITIONS),
        }
        for seed in SEEDS
    }
    for repetition in range(BOOTSTRAP_REPETITIONS):
        top_deltas = []
        f1_deltas = []
        for seed in SEEDS:
            a_pred = a[seed][kind]
            b_pred = b[seed][kind]
            labels = a_pred["labels"]
            sampled_indices: list[int] = []
            for class_index in np.unique(labels):
                candidates = np.flatnonzero(labels == class_index)
                sampled_indices.extend(rng.choice(candidates, len(candidates), replace=True).tolist())
            sampled = np.asarray(sampled_indices, dtype=np.int64)
            sampled_labels = labels[sampled]
            a_predictions = a_pred["metrics"]["predictions"][sampled]
            b_predictions = b_pred["metrics"]["predictions"][sampled]
            a_top = float(np.mean(a_predictions == sampled_labels))
            b_top = float(np.mean(b_predictions == sampled_labels))
            a_f1 = macro_f1_from_predictions(sampled_labels, a_predictions)
            b_f1 = macro_f1_from_predictions(sampled_labels, b_predictions)
            seedwise[str(seed)]["top1"][repetition] = a_top - b_top
            seedwise[str(seed)]["macro_f1"][repetition] = a_f1 - b_f1
            top_deltas.append(a_top - b_top)
            f1_deltas.append(a_f1 - b_f1)
        top1[repetition] = np.mean(top_deltas)
        macro[repetition] = np.mean(f1_deltas)

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "q50": float(np.quantile(values, 0.5)),
        }

    return {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "rng_seed": rng_seed,
        "unit": "class-stratified episode bootstrap within each seed, then mean across three seeds",
        "seed_aggregated": {"top1": summary(top1), "macro_f1": summary(macro)},
        "seedwise": {
            seed: {metric: summary(values) for metric, values in metrics.items()}
            for seed, metrics in seedwise.items()
        },
    }


def macro_f1_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> float:
    f1s = []
    for class_index in range(NUM_CLASSES):
        mask = labels == class_index
        support = int(mask.sum())
        if support == 0:
            continue
        tp = int(np.sum(mask & (predictions == class_index)))
        fp = int(np.sum(~mask & (predictions == class_index)))
        recall = tp / support
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def pair_rows(
    a: dict[int, dict[str, Any]],
    b: dict[int, dict[str, Any]],
    *,
    pair_name: str,
    checkpoint_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_rows = []
    transition_rows = []
    class_rows = []
    kind_name = "best" if checkpoint_kind in {"best", "selected_best"} else "last"
    for seed in SEEDS:
        ap = a[seed][kind_name]
        bp = b[seed][kind_name]
        if not np.array_equal(ap["episode_index"], bp["episode_index"]) or not np.array_equal(ap["labels"], bp["labels"]):
            raise ValueError(f"episode alignment failed for {pair_name}, seed {seed}")
        metric_rows.append(
            {
                "pair": pair_name,
                "checkpoint_kind": checkpoint_kind,
                "seed": seed,
                "a_epoch": int(a[seed]["summary"]["best_epoch"] if kind_name == "best" else a[seed]["summary"]["epochs_completed"]),
                "b_epoch": int(b[seed]["summary"]["best_epoch"] if kind_name == "best" else b[seed]["summary"]["epochs_completed"]),
                **{f"{name}_delta": float(ap["metrics"][name] - bp["metrics"][name]) for name in METRICS},
            }
        )
        transition = exact_mcnemar(ap["metrics"]["correct_array"], bp["metrics"]["correct_array"])
        transition_rows.append({"pair": pair_name, "checkpoint_kind": checkpoint_kind, "seed": seed, **transition})
        a_classes = {item["class_index"]: item for item in ap["metrics"]["per_class"]}
        b_classes = {item["class_index"]: item for item in bp["metrics"]["per_class"]}
        for class_index in sorted(a_classes):
            ac = a_classes[class_index]
            bc = b_classes[class_index]
            class_rows.append(
                {
                    "pair": pair_name,
                    "checkpoint_kind": checkpoint_kind,
                    "seed": seed,
                    "class_index": class_index,
                    "support": ac["support"],
                    "a_recall": ac["recall"],
                    "b_recall": bc["recall"],
                    "delta_recall": ac["recall"] - bc["recall"],
                    "a_recall_at_5": ac["recall_at_5"],
                    "b_recall_at_5": bc["recall_at_5"],
                    "delta_recall_at_5": ac["recall_at_5"] - bc["recall_at_5"],
                    "a_f1": ac["f1"],
                    "b_f1": bc["f1"],
                    "delta_f1": ac["f1"] - bc["f1"],
                    "support_le_2": ac["support"] <= 2,
                }
            )
    bootstrap = bootstrap_pair(
        a,
        b,
        kind_name,
        rng_seed=BOOTSTRAP_SEED + (0 if pair_name == "batch32_dynamic_minus_static" else 1),
    )
    return metric_rows, transition_rows, class_rows, bootstrap


def read_old_aggregate(path: Path) -> dict[str, Any]:
    matrix = read_json(path / "matrix.json")
    per_seed = []
    for item in matrix.get("per_seed", []):
        prediction_path = Path(item["prediction_path"])
        with np.load(prediction_path, allow_pickle=False) as records:
            labels = records["labels"].astype(np.int64)
            logits = records["logits"].astype(np.float32)
        computed = numpy_metrics(labels, logits)
        per_seed.append({"seed": int(item["seed"]), **{name: computed[name] for name in METRICS}})
    if len(per_seed) != 3:
        raise ValueError(f"old aggregate does not contain three seeds: {path}")
    return {
        "method": matrix.get("method"),
        "source": str(path),
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
        "episode_level_pairing": False,
        "pairing_status": "paired_significance_not_available",
    }


def resource_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    snapshots: list[dict[str, Any]] = []
    current_utc = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RESOURCE_SNAPSHOT utc="):
            current_utc = line.split("utc=", 1)[1].split(" ", 1)[0]
        if line.startswith("gpu="):
            fields = [item.strip() for item in line.split("telemetry=", 1)[-1].split(",")]
            if len(fields) >= 5:
                try:
                    gpu = int(line.split("gpu=", 1)[1].split(" ", 1)[0])
                    snapshots.append(
                        {
                            "utc": current_utc,
                            "gpu": gpu,
                            "utilization_percent": int(fields[2]),
                            "memory_used_mib": int(fields[3]),
                            "memory_total_mib": int(fields[5] if len(fields) > 5 else fields[4]),
                            "compute_pid_count": int(line.split("compute_pid_count=", 1)[1].split()[0]),
                        }
                    )
                except (IndexError, ValueError):
                    continue
    by_gpu = {}
    for gpu in (2, 3):
        values = [row for row in snapshots if row["gpu"] == gpu]
        by_gpu[str(gpu)] = {
            "snapshot_count": len(values),
            "utilization_min": min((row["utilization_percent"] for row in values), default=None),
            "utilization_max": max((row["utilization_percent"] for row in values), default=None),
            "memory_used_min_mib": min((row["memory_used_mib"] for row in values), default=None),
            "memory_used_max_mib": max((row["memory_used_mib"] for row in values), default=None),
            "compute_pid_counts": sorted({row["compute_pid_count"] for row in values}),
        }
    return {"status": "pass", "path": str(path), "snapshot_count": len(snapshots), "by_gpu": by_gpu}


def copy_snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dynamic-run", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    parser.add_argument("--batch4-dynamic", type=Path, required=True)
    parser.add_argument("--batch4-static", type=Path, required=True)
    parser.add_argument("--dynamic-config", type=Path, required=True)
    parser.add_argument("--static-config", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--smoke-dynamic", type=Path, required=True)
    parser.add_argument("--smoke-static", type=Path, required=True)
    parser.add_argument("--nonparametric", type=Path, required=True)
    parser.add_argument("--nonparametric-second", type=Path, required=True)
    parser.add_argument("--resource-log", type=Path, required=True)
    parser.add_argument("--idle-gate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"harvest output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    population = load_v3_population(load_config(args.dynamic_config.resolve()), full=True)
    validation = population.rows[population.rows["split"].eq("validation")].sort_values("episode_index", kind="mergesort")
    validation_episode_index = validation["episode_index"].to_numpy(dtype=np.int64)
    validation_labels = validation["local_label_index"].to_numpy(dtype=np.int64)
    if len(validation) != VALIDATION_COUNT:
        raise ValueError("validation population is not 617 rows")
    dynamic_config_hash = sha256_file(args.dynamic_config.resolve())
    static_config_hash = sha256_file(args.static_config.resolve())
    dynamic_manifest_hash = str(read_json(args.input_preflight)["dynamic"]["feature_manifest_sha256"])
    static_manifest_hash = str(read_json(args.input_preflight)["static"]["feature_manifest_sha256"])
    dynamic = validate_run(
        args.dynamic_run.resolve(),
        variant="dynamic",
        expected_batch=32,
        expected_config_sha256=dynamic_config_hash,
        expected_feature_manifest_sha256=dynamic_manifest_hash,
        validation_episode_index=validation_episode_index,
        validation_labels=validation_labels,
        expected_optimizer_steps=math.ceil(TRAIN_COUNT / 32),
    )
    static = validate_run(
        args.static_run.resolve(),
        variant="static_repeated_first_frame",
        expected_batch=32,
        expected_config_sha256=static_config_hash,
        expected_feature_manifest_sha256=static_manifest_hash,
        validation_episode_index=validation_episode_index,
        validation_labels=validation_labels,
        expected_optimizer_steps=math.ceil(TRAIN_COUNT / 32),
    )
    batch4_dynamic = validate_run(
        args.batch4_dynamic.resolve(),
        variant="batch4_dynamic_control",
        expected_batch=4,
        expected_config_sha256=sha256_file(args.repo_root / "configs/frozen/core_a_full_attentive_dynamic_v3_20260828.yaml"),
        expected_feature_manifest_sha256=dynamic_manifest_hash,
        validation_episode_index=validation_episode_index,
        validation_labels=validation_labels,
        expected_optimizer_steps=None,
    )
    batch4_static = validate_run(
        args.batch4_static.resolve(),
        variant="batch4_static_control",
        expected_batch=4,
        expected_config_sha256=sha256_file(args.repo_root / "configs/frozen/core_a_full_attentive_static_v3_20260828.yaml"),
        expected_feature_manifest_sha256=static_manifest_hash,
        validation_episode_index=validation_episode_index,
        validation_labels=validation_labels,
        expected_optimizer_steps=None,
    )
    input_preflight = read_json(args.input_preflight)
    smoke_dynamic = read_json(args.smoke_dynamic)
    smoke_static = read_json(args.smoke_static)
    if input_preflight.get("status") != "pass" or smoke_dynamic.get("status") != "pass" or smoke_static.get("status") != "pass":
        raise ValueError("input preflight or smoke gate is not pass")
    nonparametric = read_json(args.nonparametric / "NONPARAMETRIC_BASELINES.json")
    nonparametric_second = read_json(args.nonparametric_second / "NONPARAMETRIC_BASELINES.json")
    deterministic = {}
    for key, record in nonparametric["methods"].items():
        other = nonparametric_second["methods"].get(key)
        if other is None:
            raise ValueError(f"non-parametric method missing in second run: {key}")
        deterministic[key] = {
            "prediction_content_hash_equal": record["prediction_content_sha256"] == other["prediction_content_sha256"],
            "metrics_equal": record["metrics"] == other["metrics"],
        }
    if not all(item["prediction_content_hash_equal"] and item["metrics_equal"] for item in deterministic.values()):
        raise ValueError("non-parametric determinism check failed")

    batch32_dynamic_infos = dynamic["seed_infos"]
    batch32_static_infos = static["seed_infos"]
    batch4_dynamic_infos = batch4_dynamic["seed_infos"]
    batch4_static_infos = batch4_static["seed_infos"]
    dynamic_static_rows, dynamic_static_transitions, dynamic_static_classes, dynamic_static_bootstrap = pair_rows(
        batch32_dynamic_infos,
        batch32_static_infos,
        pair_name="batch32_dynamic_minus_static",
        checkpoint_kind="selected_best",
    )
    dynamic_static_last_rows, dynamic_static_last_transitions, dynamic_static_last_classes, _ = pair_rows(
        batch32_dynamic_infos,
        batch32_static_infos,
        pair_name="batch32_dynamic_minus_static",
        checkpoint_kind="last",
    )
    batch32_b4_rows = []
    batch32_b4_transitions = []
    batch32_b4_classes = []
    bootstrap_results: dict[str, Any] = {
        "batch32_dynamic_minus_static": dynamic_static_bootstrap,
    }
    for name, new_run, old_run in (
        ("batch32_dynamic_minus_batch4_dynamic", batch32_dynamic_infos, batch4_dynamic_infos),
        ("batch32_static_minus_batch4_static", batch32_static_infos, batch4_static_infos),
    ):
        rows, transitions, classes, bootstrap = pair_rows(
            new_run,
            old_run,
            pair_name=name,
            checkpoint_kind="selected_best",
        )
        batch32_b4_rows.extend(rows)
        batch32_b4_transitions.extend(transitions)
        batch32_b4_classes.extend(classes)
        bootstrap_results[name] = bootstrap
    all_transition_rows = dynamic_static_transitions + dynamic_static_last_transitions + batch32_b4_transitions
    all_class_rows = dynamic_static_classes + dynamic_static_last_classes + batch32_b4_classes
    write_csv(output / "BATCH32_DYNAMIC_STATIC_COMPARISON.csv", dynamic_static_rows + dynamic_static_last_rows)
    write_csv(output / "BATCH32_VS_BATCH4_COMPARISON.csv", batch32_b4_rows)
    write_csv(output / "PAIRED_CORRECTNESS_TRANSITIONS.csv", all_transition_rows)
    write_csv(output / "PER_CLASS_DELTAS.csv", all_class_rows)
    write_json(output / "BOOTSTRAP_COMPARISONS.json", bootstrap_results)

    def selected_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
        return [info["best"]["metrics"] for info in (run["seed_infos"][seed] for seed in SEEDS)]

    def last_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
        return [info["last"]["metrics"] for info in (run["seed_infos"][seed] for seed in SEEDS)]

    per_seed_selected = []
    per_seed_last = []
    calibration_rows = []
    early_stop_replay: dict[str, Any] = {}
    for run_name, run in (
        ("batch32_dynamic", dynamic),
        ("batch32_static", static),
        ("batch4_dynamic", batch4_dynamic),
        ("batch4_static", batch4_static),
    ):
        early_stop_replay[run_name] = {}
        for seed in SEEDS:
            info = run["seed_infos"][seed]
            for checkpoint_kind, target, prediction_key in (
                ("selected_best", per_seed_selected, "best"),
                ("last", per_seed_last, "last"),
            ):
                prediction = info[prediction_key]
                target.append(
                    {
                        "run": run_name,
                        "seed": seed,
                        "checkpoint_kind": checkpoint_kind,
                        "epoch": int(info["summary"]["best_epoch"] if prediction_key == "best" else info["summary"]["epochs_completed"]),
                        **{name: prediction["metrics"][name] for name in METRICS},
                        "correct": prediction["metrics"]["correct"],
                        "total": prediction["metrics"]["total"],
                    }
                )
                calibration_rows.append(
                    {
                        "run": run_name,
                        "seed": seed,
                        "checkpoint_kind": checkpoint_kind,
                        "epoch": int(info["summary"]["best_epoch"] if prediction_key == "best" else info["summary"]["epochs_completed"]),
                        "nll": prediction["metrics"]["nll"],
                        "brier": prediction["metrics"]["brier"],
                        "ece": prediction["metrics"]["ece"],
                        "mean_max_softmax_probability": prediction["metrics"]["mean_max_softmax_probability"],
                    }
                )
            early_stop_replay[run_name][str(seed)] = info["replay"]
    write_csv(output / "PER_SEED_BATCH32_SELECTED_METRICS.csv", [row for row in per_seed_selected if row["run"].startswith("batch32")])
    write_csv(output / "PER_SEED_BATCH32_LAST_METRICS.csv", [row for row in per_seed_last if row["run"].startswith("batch32")])
    write_csv(output / "CALIBRATION_COMPARISON.csv", calibration_rows)
    write_json(output / "EARLY_STOP_REPLAY.json", early_stop_replay)

    pooled_dynamic = read_old_aggregate(args.repo_root / "reports/benchmarks/CORE_A_BENCHMARK_MATRIX_20260826_V3/vjepa2_1_b_full_existing_cache")
    pooled_static = read_old_aggregate(args.repo_root / "reports/benchmarks/CORE_A_BENCHMARK_MATRIX_20260826_V3/static_first_frame_vjepa2_1_b")
    pooled_gap = {
        "batch32_dynamic_attentive_minus_pooled_dynamic": aggregate(selected_rows(dynamic))["top1"]["mean"] - pooled_dynamic["aggregate"]["top1"]["mean"],
        "batch32_static_attentive_minus_pooled_static": aggregate(selected_rows(static))["top1"]["mean"] - pooled_static["aggregate"]["top1"]["mean"],
        "pooled_dynamic_minus_pooled_static": pooled_dynamic["aggregate"]["top1"]["mean"] - pooled_static["aggregate"]["top1"]["mean"],
    }

    throughput_rows = []
    for run_name, run, batch in (
        ("batch32_dynamic", dynamic, 32),
        ("batch32_static", static, 32),
        ("batch4_dynamic", batch4_dynamic, 4),
        ("batch4_static", batch4_static, 4),
    ):
        for seed in SEEDS:
            info = run["seed_infos"][seed]
            curves = info["curves"]
            total_time = float(info["summary"]["elapsed_seconds"])
            total_steps = int(info["summary"].get("total_optimizer_steps", sum(math.ceil(TRAIN_COUNT / batch) for _ in curves)))
            total_exposure = int(info["summary"].get("total_samples_exposed", TRAIN_COUNT * len(curves)))
            per_epoch_throughput = []
            for row in curves:
                if "samples_per_second" in row:
                    per_epoch_throughput.append(float(row["samples_per_second"]))
                else:
                    per_epoch_throughput.append(3374 / float(row["epoch_wall_time_seconds"]))
            throughput_rows.append(
                {
                    "run": run_name,
                    "seed": seed,
                    "batch_size": batch,
                    "epochs": len(curves),
                    "optimizer_steps_per_epoch": math.ceil(TRAIN_COUNT / batch),
                    "total_optimizer_steps": total_steps,
                    "total_sample_exposures": total_exposure,
                    "elapsed_seconds": total_time,
                    "gpu_hours": total_time / 3600,
                    "mean_epoch_samples_per_second": float(np.mean(per_epoch_throughput)),
                    "peak_vram_allocated_bytes": info["summary"].get("peak_vram_allocated_bytes", max(int(row.get("peak_vram_allocated_bytes", 0)) for row in curves)),
                    "peak_vram_reserved_bytes": info["summary"].get("peak_vram_reserved_bytes", max(int(row.get("peak_vram_reserved_bytes", 0)) for row in curves)),
                }
            )
    write_csv(output / "RESOURCE_AND_THROUGHPUT_SUMMARY.csv", throughput_rows + [
        {
            "run": "memory_smoke_dynamic",
            "seed": "",
            "batch_size": 32,
            "epochs": "",
            "optimizer_steps_per_epoch": "",
            "total_optimizer_steps": smoke_dynamic["optimizer_steps"],
            "total_sample_exposures": smoke_dynamic["sample_exposures"],
            "elapsed_seconds": smoke_dynamic["elapsed_seconds"],
            "gpu_hours": float(smoke_dynamic["elapsed_seconds"]) / 3600,
            "mean_epoch_samples_per_second": smoke_dynamic["samples_per_second"],
            "peak_vram_allocated_bytes": smoke_dynamic["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": smoke_dynamic["peak_vram_reserved_bytes"],
        },
        {
            "run": "memory_smoke_static",
            "seed": "",
            "batch_size": 32,
            "epochs": "",
            "optimizer_steps_per_epoch": "",
            "total_optimizer_steps": smoke_static["optimizer_steps"],
            "total_sample_exposures": smoke_static["sample_exposures"],
            "elapsed_seconds": smoke_static["elapsed_seconds"],
            "gpu_hours": float(smoke_static["elapsed_seconds"]) / 3600,
            "mean_epoch_samples_per_second": smoke_static["samples_per_second"],
            "peak_vram_allocated_bytes": smoke_static["peak_vram_allocated_bytes"],
            "peak_vram_reserved_bytes": smoke_static["peak_vram_reserved_bytes"],
        },
    ])

    selected_dynamic = aggregate(selected_rows(dynamic))
    selected_static = aggregate(selected_rows(static))
    selected_b4_dynamic = aggregate(selected_rows(batch4_dynamic))
    selected_b4_static = aggregate(selected_rows(batch4_static))
    d32_b4 = {name: selected_dynamic[name]["mean"] - selected_b4_dynamic[name]["mean"] for name in METRICS}
    s32_b4 = {name: selected_static[name]["mean"] - selected_b4_static[name]["mean"] for name in METRICS}
    ds = {name: selected_dynamic[name]["mean"] - selected_static[name]["mean"] for name in METRICS}
    dynamic_macro_seed = [row["macro_f1_delta"] for row in dynamic_static_rows if row["checkpoint_kind"] == "selected_best"]
    d32_b4_macro_seed = [row["macro_f1_delta"] for row in batch32_b4_rows if row["pair"] == "batch32_dynamic_minus_batch4_dynamic"]
    d32_b4_boot = bootstrap_results["batch32_dynamic_minus_batch4_dynamic"]["seed_aggregated"]["macro_f1"]["ci95"]
    low_support_rows = [row for row in all_class_rows if row["pair"] == "batch32_dynamic_minus_batch4_dynamic" and row["checkpoint_kind"] == "selected_best"]
    low_support_mean = float(
        np.mean([row["delta_f1"] for row in low_support_rows if row["support_le_2"]])
        if any(row["support_le_2"] for row in low_support_rows)
        else 0.0
    )
    regular_support_mean = float(
        np.mean([row["delta_f1"] for row in low_support_rows if not row["support_le_2"]])
        if any(not row["support_le_2"] for row in low_support_rows)
        else 0.0
    )
    strong_rescue = bool(d32_b4["macro_f1"] >= 0.010 and d32_b4["top1"] >= 0 and sum(value >= 0 for value in d32_b4_macro_seed) >= 2 and d32_b4_boot[0] > 0 and abs(low_support_mean) <= abs(regular_support_mean) + 0.02)
    modest_rescue = bool(not strong_rescue and (0.005 <= d32_b4["macro_f1"] < 0.010 or d32_b4_boot[0] <= 0 <= d32_b4_boot[1] or sum(value >= 0 for value in d32_b4_macro_seed) < 2))
    dynamic_boot = dynamic_static_bootstrap["seed_aggregated"]["top1"]["ci95"]
    meaningful_dynamic = bool(ds["top1"] >= 0.010 and ds["macro_f1"] >= 0.005 and sum(value > 0 for value in dynamic_macro_seed) >= 2 and dynamic_boot[0] > 0)
    if meaningful_dynamic:
        dynamic_signal = "meaningful_dynamic_advantage"
    elif ds["top1"] < 0 or ds["macro_f1"] < 0 or ds["top1"] < 0.010:
        dynamic_signal = "static_shortcut_still_dominant"
    else:
        dynamic_signal = "modest_or_uncertain_dynamic_signal"
    sum_batch32_time = sum(row["elapsed_seconds"] for row in throughput_rows if row["run"].startswith("batch32"))
    sum_batch4_time = sum(row["elapsed_seconds"] for row in throughput_rows if row["run"].startswith("batch4"))
    mean_batch32_speed = float(np.mean([row["mean_epoch_samples_per_second"] for row in throughput_rows if row["run"].startswith("batch32")]))
    mean_batch4_speed = float(np.mean([row["mean_epoch_samples_per_second"] for row in throughput_rows if row["run"].startswith("batch4")]))
    speedup = mean_batch32_speed / mean_batch4_speed if mean_batch4_speed else 0.0
    if speedup >= 1.5 and min(d32_b4["top1"], s32_b4["top1"]) >= -0.002 and min(d32_b4["macro_f1"], s32_b4["macro_f1"]) >= -0.002:
        efficiency_decision = "batch32_efficiency_success"
    elif speedup >= 1.5 and (d32_b4["top1"] < -0.002 or d32_b4["macro_f1"] < -0.002):
        efficiency_decision = "throughput_gain_with_metric_regression"
    else:
        efficiency_decision = "no_clear_efficiency_success"
    rescue_decision = "strong_batch32_rescue" if strong_rescue else "modest_or_uncertain_batch32_effect" if modest_rescue else "no_batch32_rescue"
    next_step = "paired top-4 LoRA feasibility study (await explicit authorization)" if strong_rescue else "Core B prefix-conditioned action anticipation / future action-chunk prediction (await explicit authorization)"
    if dynamic_signal == "meaningful_dynamic_advantage":
        conclusion = "batch32_rescues_head_and_reveals_dynamic_advantage"
    elif rescue_decision == "strong_batch32_rescue":
        conclusion = "batch32_rescues_head_optimization_but_not_dynamic_signal"
    elif efficiency_decision == "throughput_gain_with_metric_regression":
        conclusion = "batch32_improves_efficiency_only"
    else:
        conclusion = "batch32_does_not_explain_attentive_underperformance"

    locked_paths = {
        "dynamic_batch32_config": args.dynamic_config.resolve(),
        "static_batch32_config": args.static_config.resolve(),
        "dynamic_batch4_config": args.repo_root / "configs/frozen/core_a_full_attentive_dynamic_v3_20260828.yaml",
        "static_batch4_config": args.repo_root / "configs/frozen/core_a_full_attentive_static_v3_20260828.yaml",
        "v3_split": args.repo_root / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet",
        "dynamic_feature_manifest": args.project_root / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet",
        "static_feature_manifest": args.project_root / "features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet",
        "batch32_protocol": args.repo_root / "docs/protocols/CORE_A_BATCH32_RESCUE_PROTOCOL_V3_20260829.md",
        "batch32_runner": args.repo_root / "scripts/run_batch32_rescue_v3.py",
        "batch32_input_verifier": args.repo_root / "scripts/verify_batch32_inputs_v3.py",
        "nonparametric_runner": args.repo_root / "scripts/run_nonparametric_baselines_v3.py",
    }
    locked_hashes = {name: {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size_bytes": path.stat().st_size if path.is_file() else 0} for name, path in locked_paths.items()}
    write_json(output / "LOCKED_INPUT_HASHES.json", locked_hashes)
    nonparametric_bundle = nonparametric.copy()
    nonparametric_bundle["determinism_check"] = deterministic
    write_json(output / "NONPARAMETRIC_BASELINES.json", nonparametric_bundle)
    copy_snapshot(args.nonparametric / "NONPARAMETRIC_BASELINES.csv", output / "NONPARAMETRIC_BASELINES.csv")
    copy_snapshot(args.nonparametric / "NONPARAMETRIC_METHOD_CONTRACT.md", output / "NONPARAMETRIC_METHOD_CONTRACT.md")
    snapshot_names = (
        "scripts/run_batch32_rescue_v3.py",
        "scripts/verify_batch32_inputs_v3.py",
        "scripts/run_nonparametric_baselines_v3.py",
        "configs/frozen/core_a_full_attentive_dynamic_v3_batch32_20260829.yaml",
        "configs/frozen/core_a_full_attentive_static_v3_batch32_20260829.yaml",
    )
    for relative in snapshot_names:
        copy_snapshot(args.repo_root / relative, output / "snapshots" / Path(relative).name)

    run_acceptance = {
        "status": "pass",
        "engineering_pass": True,
        "dynamic": {"status": dynamic["status"], "seeds": {str(seed): {"best_epoch": dynamic["seed_infos"][seed]["summary"]["best_epoch"], "last_epoch": dynamic["seed_infos"][seed]["summary"]["epochs_completed"], "stop_reason": dynamic["seed_infos"][seed]["summary"]["stop_reason"], "best_metric_max_abs_diff": dynamic["seed_infos"][seed]["best_metric_max_abs_diff"], "last_metric_max_abs_diff": dynamic["seed_infos"][seed]["last_metric_max_abs_diff"], "early_stopping_replay": dynamic["seed_infos"][seed]["replay"]["state_matches_recorded"]} for seed in SEEDS}},
        "static": {"status": static["status"], "seeds": {str(seed): {"best_epoch": static["seed_infos"][seed]["summary"]["best_epoch"], "last_epoch": static["seed_infos"][seed]["summary"]["epochs_completed"], "stop_reason": static["seed_infos"][seed]["summary"]["stop_reason"], "best_metric_max_abs_diff": static["seed_infos"][seed]["best_metric_max_abs_diff"], "last_metric_max_abs_diff": static["seed_infos"][seed]["last_metric_max_abs_diff"], "early_stopping_replay": static["seed_infos"][seed]["replay"]["state_matches_recorded"]} for seed in SEEDS}},
        "memory_smoke": {"dynamic": smoke_dynamic, "static": smoke_static},
        "final_test_selected": 0,
        "final_test_access": "none",
        "no_backbone_trainable_parameters": True,
        "no_gradient_accumulation": True,
        "nonparametric_determinism": deterministic,
    }
    write_json(output / "RUN_ACCEPTANCE.json", run_acceptance)

    per_class_support = {int(row["class_index"]): int(row["support"]) for row in low_support_rows}
    report_json = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": "project-owner",
        "scope": "batch32 rescue; development-validation only",
        "conclusion": conclusion,
        "batch32_rescue_decision": rescue_decision,
        "dynamic_signal_decision": dynamic_signal,
        "efficiency_decision": efficiency_decision,
        "batch32_dynamic_selected": selected_dynamic,
        "batch32_static_selected": selected_static,
        "batch4_dynamic_selected": selected_b4_dynamic,
        "batch4_static_selected": selected_b4_static,
        "batch32_dynamic_minus_static": ds,
        "batch32_dynamic_minus_batch4_dynamic": d32_b4,
        "batch32_static_minus_batch4_static": s32_b4,
        "pooled_reference_gaps": pooled_gap,
        "pooled_references": {"dynamic": pooled_dynamic, "static": pooled_static},
        "bootstrap": bootstrap_results,
        "mcnemar_transitions": all_transition_rows,
        "correctness_transition_selected_best": {"batch32_dynamic_minus_static": [row for row in dynamic_static_transitions if row["checkpoint_kind"] == "selected_best"]},
        "overfit": {
            "batch32_dynamic_last_minus_selected": {name: selected_dynamic[name]["mean"] - aggregate(last_rows(dynamic))[name]["mean"] for name in METRICS},
            "batch32_static_last_minus_selected": {name: selected_static[name]["mean"] - aggregate(last_rows(static))[name]["mean"] for name in METRICS},
            "batch4_dynamic_last_minus_selected": {name: selected_b4_dynamic[name]["mean"] - aggregate(last_rows(batch4_dynamic))[name]["mean"] for name in METRICS},
            "batch4_static_last_minus_selected": {name: selected_b4_static[name]["mean"] - aggregate(last_rows(batch4_static))[name]["mean"] for name in METRICS},
            "interpretation": "calibration_or_confidence_overfit is reported separately from accuracy; train loss alone is not used as evidence",
        },
        "throughput": {
            "batch4_optimizer_steps_per_epoch": math.ceil(TRAIN_COUNT / 4),
            "batch32_optimizer_steps_per_epoch": math.ceil(TRAIN_COUNT / 32),
            "batch4_max_20_epoch_steps": math.ceil(TRAIN_COUNT / 4) * 20,
            "batch32_max_20_epoch_steps": math.ceil(TRAIN_COUNT / 32) * 20,
            "batch4_max_20_epoch_sample_exposures": TRAIN_COUNT * 20,
            "batch32_max_20_epoch_sample_exposures": TRAIN_COUNT * 20,
            "batch32_to_batch4_epoch_speed_ratio": speedup,
            "batch32_total_elapsed_seconds": sum_batch32_time,
            "batch4_total_elapsed_seconds": sum_batch4_time,
        },
        "support_audit": {"class_supports": per_class_support, "low_support_mean_delta_f1": low_support_mean, "regular_support_mean_delta_f1": regular_support_mean},
        "resource": resource_summary(args.resource_log),
        "idle_gate_log": str(args.idle_gate_log.resolve()),
        "final_test_selected": 0,
        "final_test_access": "none",
        "session_scene_operator_metadata": "not available; no session-independent claim",
    }
    write_json(output / "HARVEST_REPORT_20260829.json", report_json)

    next_step_text = f"""# Next-step decision

Decision: **{rescue_decision}**; dynamic-signal category: **{dynamic_signal}**; efficiency category: **{efficiency_decision}**.

Batch-32 dynamic minus batch-4 dynamic macro-F1 is `{d32_b4['macro_f1']:.6f}` and Top-1 is `{d32_b4['top1']:.6f}`. Batch-32 dynamic minus matched static macro-F1 is `{ds['macro_f1']:.6f}` and Top-1 is `{ds['top1']:.6f}`. The supporting 10,000-repetition class-stratified bootstrap, exact McNemar results, transition counts and low-support audit are saved beside this file.

Recommended next direction: **{next_step}**. This is a recommendation only; no next-stage run was started by this harvest.

Still locked: final-test, LoRA/adapter, Temporal-Residual, Core B, batch-size fallback, LR/loss tuning, new data/weight download, and backbone unfreezing.
"""
    (output / "NEXT_STEP_DECISION.md").write_text(next_step_text, encoding="utf-8")
    provenance = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": "project-owner",
        "repo_root": str(args.repo_root.resolve()),
        "project_root": str(args.project_root.resolve()),
        "dynamic_run": str(args.dynamic_run.resolve()),
        "static_run": str(args.static_run.resolve()),
        "batch4_dynamic_run": str(args.batch4_dynamic.resolve()),
        "batch4_static_run": str(args.batch4_static.resolve()),
        "input_preflight": str(args.input_preflight.resolve()),
        "smokes": {"dynamic": str(args.smoke_dynamic.resolve()), "static": str(args.smoke_static.resolve())},
        "nonparametric": str(args.nonparametric.resolve()),
        "resource_log": str(args.resource_log.resolve()),
        "idle_gate_log": str(args.idle_gate_log.resolve()),
        "locked_input_hashes": locked_hashes,
        "population": {"train": TRAIN_COUNT, "development_validation": VALIDATION_COUNT, "classes": NUM_CLASSES, "final_test_selected": 0},
        "final_test_access": "none",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_unit": "class-stratified episode bootstrap within seed, mean across seeds",
    }
    write_json(output / "PROVENANCE.json", provenance)
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(output)}" for path in checksum_paths) + "\n", encoding="utf-8")

    markdown = f"""# EE6008 Core A Batch-32 rescue harvest

## Conclusion

- Engineering acceptance: **pass**; dynamic and matched static each have 3/3 validated seeds.
- Batch-32 rescue: **{rescue_decision}**.
- Dynamic signal: **{dynamic_signal}**.
- Efficiency: **{efficiency_decision}**; mean epoch-throughput ratio batch32/batch4 = `{speedup:.3f}x`.
- Final-test selected/evaluated: **0**; no final-test labels or metrics were accessed.

## Selected-best development-validation metrics

| variant | Top-1 | Top-5 | mean recall@1 | mean recall@5 | macro-F1 | NLL | Brier | ECE | mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch32 dynamic | {selected_dynamic['top1']['mean']:.6f} +/- {selected_dynamic['top1']['sample_std']:.6f} | {selected_dynamic['top5']['mean']:.6f} +/- {selected_dynamic['top5']['sample_std']:.6f} | {selected_dynamic['mean_class_recall_at_1']['mean']:.6f} +/- {selected_dynamic['mean_class_recall_at_1']['sample_std']:.6f} | {selected_dynamic['mean_class_recall_at_5']['mean']:.6f} +/- {selected_dynamic['mean_class_recall_at_5']['sample_std']:.6f} | {selected_dynamic['macro_f1']['mean']:.6f} +/- {selected_dynamic['macro_f1']['sample_std']:.6f} | {selected_dynamic['nll']['mean']:.6f} +/- {selected_dynamic['nll']['sample_std']:.6f} | {selected_dynamic['brier']['mean']:.6f} +/- {selected_dynamic['brier']['sample_std']:.6f} | {selected_dynamic['ece']['mean']:.6f} +/- {selected_dynamic['ece']['sample_std']:.6f} | {selected_dynamic['mean_max_softmax_probability']['mean']:.6f} |
| batch32 static | {selected_static['top1']['mean']:.6f} +/- {selected_static['top1']['sample_std']:.6f} | {selected_static['top5']['mean']:.6f} +/- {selected_static['top5']['sample_std']:.6f} | {selected_static['mean_class_recall_at_1']['mean']:.6f} +/- {selected_static['mean_class_recall_at_1']['sample_std']:.6f} | {selected_static['mean_class_recall_at_5']['mean']:.6f} +/- {selected_static['mean_class_recall_at_5']['sample_std']:.6f} | {selected_static['macro_f1']['mean']:.6f} +/- {selected_static['macro_f1']['sample_std']:.6f} | {selected_static['nll']['mean']:.6f} +/- {selected_static['nll']['sample_std']:.6f} | {selected_static['brier']['mean']:.6f} +/- {selected_static['brier']['sample_std']:.6f} | {selected_static['ece']['mean']:.6f} +/- {selected_static['ece']['sample_std']:.6f} | {selected_static['mean_max_softmax_probability']['mean']:.6f} |

Selected epochs: batch32 dynamic `{[dynamic['seed_infos'][seed]['summary']['best_epoch'] for seed in SEEDS]}`; batch32 static `{[static['seed_infos'][seed]['summary']['best_epoch'] for seed in SEEDS]}`. Batch-4 controls and last-checkpoint values are in the CSV/JSON evidence.

## Pairwise deltas

- Batch32 dynamic - static: Top-1 `{ds['top1']:.6f}`, macro-F1 `{ds['macro_f1']:.6f}`.
- Batch32 dynamic - batch4 dynamic: Top-1 `{d32_b4['top1']:.6f}`, macro-F1 `{d32_b4['macro_f1']:.6f}`.
- Batch32 static - batch4 static: Top-1 `{s32_b4['top1']:.6f}`, macro-F1 `{s32_b4['macro_f1']:.6f}`.
- Batch32 dynamic - pooled dynamic Top-1: `{pooled_gap['batch32_dynamic_attentive_minus_pooled_dynamic']:.6f}`.
- Batch32 static - pooled static Top-1: `{pooled_gap['batch32_static_attentive_minus_pooled_static']:.6f}`.

The non-parametric frozen-feature baselines are in `NONPARAMETRIC_BASELINES.json/csv`; their paired pooled significance is unavailable because historical pooled prediction files lack episode IDs. The non-parametric runs were independently deterministic.

## Statistical and calibration QA

The 10,000-repetition class-stratified paired bootstrap is stored in `BOOTSTRAP_COMPARISONS.json`; exact per-seed McNemar and correctness transitions are in `PAIRED_CORRECTNESS_TRANSITIONS.csv`. Three shared 617-row copies are not treated as 1,851 independent episodes. Support-1/2 class extremes are retained in `PER_CLASS_DELTAS.csv` and are not used as standalone conclusions.

Selected-best vs last is separated into accuracy, macro-class and confidence/calibration changes in `HARVEST_REPORT_20260829.json` and `CALIBRATION_COMPARISON.csv`; near-zero train loss alone is not called accuracy overfit.

## Compute accounting

Batch4 has `690` optimizer steps/epoch and batch32 has `87`; at a 20-epoch maximum these are `13,800` and `1,740`. Maximum train sample exposures remain `55,140` per seed. Actual epochs, steps, exposures, wall time, GPU-hours and peak VRAM are in `RESOURCE_AND_THROUGHPUT_SUMMARY.csv`.

## Limitations and stop

The task remains development-validation task-ID classification with no trusted session/scene/operator metadata. High accuracy does not establish humanoid dynamics understanding, and repeated-first-frame is a diagnostic control rather than a perfect causal intervention. The harvest recommends **{next_step}** but does not execute it. No LoRA, Core B, Temporal-Residual, final-test, fallback batch, tuning, download, or backbone unfreezing was started.

Evidence paths: `RUN_ACCEPTANCE.json`, `LOCKED_INPUT_HASHES.json`, `EARLY_STOP_REPLAY.json`, `PROVENANCE.json`, and `SHA256SUMS`.
"""
    (output / "HARVEST_REPORT_20260829.md").write_text(markdown, encoding="utf-8")
    # Refresh checksums after the Markdown report is created.
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("\n".join(f"{sha256_file(path)}  {path.relative_to(output)}" for path in checksum_paths) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "batch32_rescue_decision": rescue_decision, "dynamic_signal_decision": dynamic_signal, "efficiency_decision": efficiency_decision, "final_test_selected": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
