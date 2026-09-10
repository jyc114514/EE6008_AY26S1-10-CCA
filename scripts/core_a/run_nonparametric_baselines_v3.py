"""Run frozen-feature, non-parametric Core A diagnostics.

This script deliberately has no trainable parameters, optimizer, backward
pass, or final-test code path.  It reads only the v3 train and
development-validation population and uses the already materialized pooled
features from the dynamic and repeated-first-frame token caches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ee6008.config import load_config
from ee6008.full_attentive_data import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_POOLED_SHAPE,
    EXPECTED_TOKEN_SHAPE,
    load_v3_population,
)

EXPECTED_TRAIN = 2757
EXPECTED_VALIDATION = 617
POOLING_COSINE_MIN = 0.99999
POOLING_MAX_ABS = 1e-3
SAMPLE_COUNT = 10
METHODS = (
    "cosine_knn1",
    "cosine_knn5",
    "cosine_knn20",
    "cosine_nearest_centroid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(named_arrays: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in named_arrays:
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def parse_shape(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return [int(item) for item in value]


def safe_feature_manifest(feature_run: Path) -> tuple[pd.DataFrame, str]:
    """Read feature paths and shapes without reading label columns."""
    manifest_path = feature_run / "feature_manifest.parquet"
    columns = [
        "episode_index",
        "feature_path",
        "feature_sha256",
        "pooled_shape",
        "token_shape",
    ]
    manifest = pd.read_parquet(manifest_path, columns=columns)
    if manifest["episode_index"].duplicated().any():
        raise ValueError(f"duplicate episode IDs in feature manifest: {manifest_path}")
    for column, expected in (
        ("pooled_shape", list(EXPECTED_POOLED_SHAPE)),
        ("token_shape", list(EXPECTED_TOKEN_SHAPE)),
    ):
        parsed = manifest[column].map(parse_shape)
        if not parsed.map(lambda item, expected=expected: item == expected).all():
            raise ValueError(f"unexpected {column} in {manifest_path}")
    manifest["resolved_feature_path"] = manifest["feature_path"].map(
        lambda value: (
            str(value)
            if Path(str(value)).is_absolute()
            else str(feature_run / str(value))
        )
    )
    if not manifest["resolved_feature_path"].map(
        lambda value: Path(value).is_file()
    ).all():
        raise FileNotFoundError(f"feature file is missing under {feature_run}")
    return manifest, sha256_file(manifest_path)


def join_v3_rows(
    population_rows: pd.DataFrame, feature_run: Path
) -> tuple[pd.DataFrame, str]:
    manifest, manifest_hash = safe_feature_manifest(feature_run)
    joined = population_rows.merge(
        manifest[["episode_index", "feature_path", "feature_sha256", "pooled_shape", "token_shape", "resolved_feature_path"]],
        on="episode_index",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"feature manifest does not cover v3 rows: {feature_run}")
    joined = joined.drop(columns=["_merge"])
    if len(joined) != len(population_rows):
        raise ValueError("feature join changed v3 population grain")
    return joined.sort_values("episode_index", kind="mergesort").reset_index(drop=True), manifest_hash


def source_metadata(feature_run: Path, manifest_hash: str) -> dict[str, Any]:
    keys = ("model_contract.json", "provenance.json", "resolved_config.json", "source_commits.json")
    metadata: dict[str, Any] = {
        "feature_run": str(feature_run.resolve()),
        "feature_manifest": str((feature_run / "feature_manifest.parquet").resolve()),
        "feature_manifest_sha256": manifest_hash,
    }
    for name in keys:
        path = feature_run / name
        if path.is_file():
            metadata[name] = read_json(path)
            metadata[f"{name}_sha256"] = sha256_file(path)
    return metadata


def sample_indices(length: int) -> list[int]:
    return sorted(set(np.linspace(0, length - 1, SAMPLE_COUNT, dtype=np.int64).tolist()))


def load_pooled(
    rows: pd.DataFrame,
    *,
    feature_run: Path,
    manifest_hash: str,
    old_static_pooled_run: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_positions = sample_indices(len(rows))
    sample_ids = {int(rows.iloc[position].episode_index) for position in sample_positions}
    pooled_values: list[np.ndarray] = []
    sample_audit: list[dict[str, Any]] = []
    for position, row in enumerate(rows.itertuples(index=False)):
        path = Path(row.resolved_feature_path)
        record = torch.load(path, map_location="cpu", weights_only=True)
        pooled = record.get("pooled")
        tokens = record.get("tokens")
        if not isinstance(pooled, torch.Tensor) or tuple(pooled.shape) != EXPECTED_POOLED_SHAPE:
            raise ValueError(f"pooled feature shape mismatch: {path}")
        if pooled.requires_grad or not torch.isfinite(pooled).all():
            raise ValueError(f"pooled feature is not finite/detached: {path}")
        pooled_values.append(pooled.detach().float().numpy())
        if int(row.episode_index) in sample_ids:
            if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
                raise ValueError(f"token feature shape mismatch: {path}")
            if not torch.isfinite(tokens).all():
                raise ValueError(f"token feature is not finite: {path}")
            recomputed = tokens.float().mean(dim=0)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    recomputed[None], pooled.float()[None], dim=1
                ).item()
            )
            max_abs = float((recomputed - pooled.float()).abs().max().item())
            expected_hash = str(row.feature_sha256)
            observed_hash = sha256_file(path)
            if expected_hash != observed_hash:
                raise ValueError(f"feature hash mismatch: {path}")
            sample_audit.append(
                {
                    "episode_index": int(row.episode_index),
                    "feature_sha256_matches_manifest": True,
                    "token_mean_vs_stored_pooled_cosine": cosine,
                    "token_mean_vs_stored_pooled_max_abs": max_abs,
                    "token_shape": list(tokens.shape),
                    "pooled_shape": list(pooled.shape),
                }
            )
    features = np.asarray(pooled_values, dtype=np.float32)
    if features.shape != (len(rows), EXPECTED_POOLED_SHAPE[0]):
        raise ValueError(f"unexpected pooled matrix shape: {features.shape}")
    if not np.isfinite(features).all():
        raise FloatingPointError("pooled matrix contains NaN or Inf")
    sample_audit.sort(key=lambda item: item["episode_index"])
    pooling_audit: dict[str, Any] = {
        "sample_count": len(sample_audit),
        "sample_episode_indices": [item["episode_index"] for item in sample_audit],
        "token_mean_vs_stored_pooled": sample_audit,
        "token_mean_vs_stored_max_abs": max(
            item["token_mean_vs_stored_pooled_max_abs"] for item in sample_audit
        ),
        "token_mean_vs_stored_min_cosine": min(
            item["token_mean_vs_stored_pooled_cosine"] for item in sample_audit
        ),
        "thresholds": {
            "max_abs_at_most": POOLING_MAX_ABS,
            "cosine_at_least": POOLING_COSINE_MIN,
        },
        "contract_pass": bool(
            len(sample_audit) == SAMPLE_COUNT
            and max(item["token_mean_vs_stored_pooled_max_abs"] for item in sample_audit)
            <= POOLING_MAX_ABS
            and min(item["token_mean_vs_stored_pooled_cosine"] for item in sample_audit)
            >= POOLING_COSINE_MIN
        ),
        "pooled_source_is_existing_record": True,
        "reencoded_encoder": False,
    }
    if old_static_pooled_run is not None:
        old_manifest_path = old_static_pooled_run / "feature_manifest.parquet"
        old_manifest = pd.read_parquet(
            old_manifest_path, columns=["episode_index", "feature_path", "pooled_shape"]
        )
        if old_manifest["episode_index"].duplicated().any():
            raise ValueError("old pooled reference has duplicate episode IDs")
        old_manifest = old_manifest.set_index("episode_index")
        compare_rows: list[dict[str, Any]] = []
        for item in sample_audit:
            episode = item["episode_index"]
            if episode not in old_manifest.index:
                raise ValueError(f"old pooled reference missing episode {episode}")
            old_path = Path(str(old_manifest.loc[episode, "feature_path"]))
            if not old_path.is_absolute():
                old_path = old_static_pooled_run / old_path
            old_record = torch.load(old_path, map_location="cpu", weights_only=True)
            old_pooled = old_record.get("pooled")
            if not isinstance(old_pooled, torch.Tensor) or tuple(old_pooled.shape) != EXPECTED_POOLED_SHAPE:
                raise ValueError(f"old pooled reference shape mismatch: {old_path}")
            new_pooled = features[rows["episode_index"].eq(episode).to_numpy().nonzero()[0][0]]
            difference = np.abs(new_pooled - old_pooled.float().numpy())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    torch.from_numpy(new_pooled)[None], old_pooled.float()[None], dim=1
                ).item()
            )
            compare_rows.append(
                {
                    "episode_index": episode,
                    "stored_token_cache_vs_existing_pooled_max_abs": float(difference.max()),
                    "stored_token_cache_vs_existing_pooled_cosine": cosine,
                }
            )
        pooling_audit["existing_pooled_reference"] = {
            "run": str(old_static_pooled_run.resolve()),
            "manifest_sha256": sha256_file(old_manifest_path),
            "sample_count": len(compare_rows),
            "samples": compare_rows,
            "max_abs": max(item["stored_token_cache_vs_existing_pooled_max_abs"] for item in compare_rows),
            "min_cosine": min(item["stored_token_cache_vs_existing_pooled_cosine"] for item in compare_rows),
            "contract_pass": bool(
                len(compare_rows) == SAMPLE_COUNT
                and max(item["stored_token_cache_vs_existing_pooled_max_abs"] for item in compare_rows)
                <= POOLING_MAX_ABS
                and min(item["stored_token_cache_vs_existing_pooled_cosine"] for item in compare_rows)
                >= POOLING_COSINE_MIN
            ),
        }
    return features, pooling_audit


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True).astype(np.float32)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("feature contains a non-positive or non-finite norm")
    return (features / norms).astype(np.float32, copy=False)


def rank_classes(vote_counts: np.ndarray, similarity_sums: np.ndarray) -> np.ndarray:
    class_ids = np.arange(vote_counts.shape[1], dtype=np.int64)
    ranked = np.empty_like(vote_counts, dtype=np.int64)
    for row_index in range(vote_counts.shape[0]):
        ranked[row_index] = np.lexsort(
            (class_ids, -similarity_sums[row_index], -vote_counts[row_index])
        )
    return ranked


def knn_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0 or k > len(train_features):
        raise ValueError(f"unsupported k={k}")
    similarities = (validation_features @ train_features.T).astype(np.float32)
    neighbor_order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    neighbor_labels = train_labels[neighbor_order]
    neighbor_scores = np.take_along_axis(similarities, neighbor_order, axis=1)
    vote_counts = np.zeros(
        (len(validation_features), EXPECTED_CLASS_COUNT), dtype=np.int32
    )
    similarity_sums = np.zeros_like(vote_counts, dtype=np.float32)
    query_indices = np.repeat(np.arange(len(validation_features)), k)
    np.add.at(vote_counts, (query_indices, neighbor_labels.reshape(-1)), 1)
    np.add.at(similarity_sums, (query_indices, neighbor_labels.reshape(-1)), neighbor_scores.reshape(-1))
    ranked = rank_classes(vote_counts, similarity_sums)
    return ranked[:, 0], ranked[:, :5], vote_counts, similarity_sums


def centroid_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.zeros((EXPECTED_CLASS_COUNT, train_features.shape[1]), dtype=np.float32)
    for class_index in range(EXPECTED_CLASS_COUNT):
        class_rows = train_features[train_labels == class_index]
        if len(class_rows) == 0:
            raise ValueError(f"train class {class_index} has no rows")
        centroids[class_index] = class_rows.mean(axis=0)
    centroids = l2_normalize(centroids)
    scores = (validation_features @ centroids.T).astype(np.float32)
    class_ids = np.arange(EXPECTED_CLASS_COUNT, dtype=np.int64)
    ranked = np.empty_like(scores, dtype=np.int64)
    for row_index in range(len(scores)):
        ranked[row_index] = np.lexsort((class_ids, -scores[row_index]))
    return ranked[:, 0], ranked[:, :5], scores


def metrics(labels: np.ndarray, predictions: np.ndarray, top5: np.ndarray) -> dict[str, Any]:
    supports = np.bincount(labels, minlength=EXPECTED_CLASS_COUNT)
    per_class: list[dict[str, Any]] = []
    recalls = []
    recalls5 = []
    f1s = []
    for class_index in range(EXPECTED_CLASS_COUNT):
        mask = labels == class_index
        support = int(supports[class_index])
        if support == 0:
            continue
        true_positive = int(np.sum(mask & (predictions == class_index)))
        false_positive = int(np.sum(~mask & (predictions == class_index)))
        recall = true_positive / support
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recall5 = float(np.mean(np.any(top5[mask] == class_index, axis=1)))
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
    return {
        "top1": float(np.mean(predictions == labels)),
        "top5": float(np.mean(np.any(top5 == labels[:, None], axis=1))),
        "mean_class_recall_at_1": float(np.mean(recalls)),
        "mean_class_recall_at_5": float(np.mean(recalls5)),
        "macro_f1": float(np.mean(f1s)),
        "correct": int(np.sum(predictions == labels)),
        "total": len(labels),
        "per_class": per_class,
    }


def unit_tests() -> dict[str, bool]:
    tie_features = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    tie_labels = np.asarray([1, 0], dtype=np.int64)
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    prediction, _top5, counts, sums = knn_predictions(
        tie_features,
        tie_labels,
        query,
        k=2,
    )
    if prediction[0] != 0 or counts[0, 0] != counts[0, 1] or sums[0, 0] != sums[0, 1]:
        raise AssertionError("k-NN tie-break is not dense-label deterministic")
    centroid_train = np.zeros(
        (EXPECTED_CLASS_COUNT * 2, EXPECTED_CLASS_COUNT), dtype=np.float32
    )
    centroid_labels = np.repeat(np.arange(EXPECTED_CLASS_COUNT), 2)
    for class_index in range(EXPECTED_CLASS_COUNT):
        centroid_train[2 * class_index : 2 * class_index + 2, class_index] = 1.0
    centroid_prediction, _, _ = centroid_predictions(
        centroid_train,
        centroid_labels,
        np.eye(EXPECTED_CLASS_COUNT, dtype=np.float32)[1:2],
    )
    if centroid_prediction[0] != 1:
        raise AssertionError("nearest-centroid unit test failed")
    return {"knn_tie_break": True, "nearest_centroid": True}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def method_contract() -> str:
    return """# Frozen-feature non-parametric inference contract

This artifact is not V-JEPA zero-shot classification. V-JEPA has no text-
aligned or native 120-class task-ID head here. The methods are diagnostics
over already materialized, frozen V-JEPA pooled features.

- Population: v3 primary train/development-validation only (2757/617), 120
  dense classes. Final-test is not selected or read.
- Feature: existing pooled `(768,)` records from the dynamic token cache and
  the repeated-first-frame token cache. No encoder was re-run.
- Preprocessing: every feature is independently L2-normalized; no PCA,
  learned normalization, or validation-derived choice is used.
- `cosine_knn1`, `cosine_knn5`, `cosine_knn20`: validation queries compare
  only with the training memory bank. Class ranking is vote count descending,
  cumulative cosine descending, then dense label ID ascending. Neighbor ties
  use stable train-row order.
- `cosine_nearest_centroid`: each class centroid is the mean of normalized
  train features, followed by L2 normalization. Classes rank by cosine
  descending, then dense label ID ascending.
- Top-1, Top-5, mean class recall@1/@5, and macro-F1 are defined. NLL, Brier,
  ECE, and confidence are `not_applicable`: these methods do not produce a
  calibrated class-probability score.
- There are no trainable parameters, optimizer steps, checkpoints, or
  backward calls.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dynamic-feature-run", type=Path, required=True)
    parser.add_argument("--static-feature-run", type=Path, required=True)
    parser.add_argument("--static-pooled-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started = time.monotonic()
    config = load_config(args.config.resolve())
    population = load_v3_population(config, full=True)
    rows = population.rows.copy()
    if len(rows) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise ValueError("unexpected v3 population size")
    if rows["episode_index"].duplicated().any():
        raise ValueError("v3 episode IDs are not unique")
    if set(rows["local_label_index"].unique()) != set(range(EXPECTED_CLASS_COUNT)):
        raise ValueError("v3 dense label map is not exactly 0..119")
    train_mask = rows["split"].eq("train").to_numpy()
    validation_mask = rows["split"].eq("validation").to_numpy()
    labels = rows["local_label_index"].to_numpy(dtype=np.int64)
    train_labels = labels[train_mask]
    validation_labels = labels[validation_mask]
    dynamic_rows, dynamic_manifest_hash = join_v3_rows(rows, args.dynamic_feature_run.resolve())
    static_rows, static_manifest_hash = join_v3_rows(rows, args.static_feature_run.resolve())
    if not np.array_equal(dynamic_rows["episode_index"], static_rows["episode_index"]):
        raise ValueError("dynamic/static episode IDs are not aligned")
    if not np.array_equal(dynamic_rows["local_label_index"], static_rows["local_label_index"]):
        raise ValueError("dynamic/static labels are not aligned")
    dynamic_features, dynamic_pooling = load_pooled(
        dynamic_rows,
        feature_run=args.dynamic_feature_run.resolve(),
        manifest_hash=dynamic_manifest_hash,
    )
    static_features, static_pooling = load_pooled(
        static_rows,
        feature_run=args.static_feature_run.resolve(),
        manifest_hash=static_manifest_hash,
        old_static_pooled_run=args.static_pooled_reference.resolve(),
    )
    if dynamic_features.shape != static_features.shape:
        raise ValueError("dynamic/static pooled matrix shapes differ")
    dynamic_norm = l2_normalize(dynamic_features)
    static_norm = l2_normalize(static_features)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    unit_test_results = unit_tests()
    method_rows: list[dict[str, Any]] = []
    result_records: dict[str, Any] = {}
    for variant, features in (("dynamic", dynamic_norm), ("static", static_norm)):
        train_features = features[train_indices]
        validation_features = features[validation_indices]
        for method in METHODS:
            if method.startswith("cosine_knn"):
                k = int(method.removeprefix("cosine_knn"))
                predictions, top5, vote_counts, similarity_sums = knn_predictions(
                    train_features,
                    train_labels,
                    validation_features,
                    k=k,
                )
                score_arrays = {
                    "vote_counts": vote_counts,
                    "similarity_sums": similarity_sums,
                }
            else:
                predictions, top5, centroid_scores = centroid_predictions(
                    train_features,
                    train_labels,
                    validation_features,
                )
                score_arrays = {"centroid_scores": centroid_scores}
            report_metrics = metrics(validation_labels, predictions, top5)
            prediction_path = output_dir / f"predictions_{variant}_{method}.npz"
            arrays = {
                "episode_index": rows.iloc[validation_indices]["episode_index"].to_numpy(dtype=np.int64),
                "labels": validation_labels,
                "predictions": predictions.astype(np.int64),
                "top5_indices": top5.astype(np.int64),
                **score_arrays,
            }
            atomic_npz(prediction_path, **arrays)
            content_hash = sha256_arrays(list(arrays.items()))
            record = {
                "variant": variant,
                "method": method,
                "prediction_path": str(prediction_path.resolve()),
                "prediction_file_sha256": sha256_file(prediction_path),
                "prediction_content_sha256": content_hash,
                "metrics": {key: value for key, value in report_metrics.items() if key != "per_class"},
                "per_class": report_metrics["per_class"],
                "nll": None,
                "brier": None,
                "ece": None,
                "mean_max_softmax_probability": None,
                "calibration_status": "not_applicable",
                "train_count": EXPECTED_TRAIN,
                "validation_count": EXPECTED_VALIDATION,
                "class_count": EXPECTED_CLASS_COUNT,
                "trainable_parameter_count": 0,
                "optimizer_created": False,
                "backward_called": False,
            }
            result_records[f"{variant}:{method}"] = record
            method_rows.append(
                {
                    "variant": variant,
                    "method": method,
                    **{key: value for key, value in report_metrics.items() if key != "per_class"},
                    "nll": "not_applicable",
                    "brier": "not_applicable",
                    "ece": "not_applicable",
                    "mean_max_softmax_probability": "not_applicable",
                    "prediction_content_sha256": content_hash,
                }
            )
    population_summary = {
        "train_count": EXPECTED_TRAIN,
        "development_validation_count": EXPECTED_VALIDATION,
        "population_rows": len(rows),
        "class_count": EXPECTED_CLASS_COUNT,
        "dense_label_values": list(range(EXPECTED_CLASS_COUNT)),
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "validation_episode_ids_unique": True,
        "final_test_selected": 0,
        "final_test_access": "none",
    }
    pooling_pass = bool(
        dynamic_pooling["contract_pass"]
        and static_pooling["contract_pass"]
        and static_pooling.get("existing_pooled_reference", {}).get("contract_pass", False)
    )
    summary = {
        "status": "pass" if pooling_pass else "blocked_pooling_contract",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "scope": "frozen-feature non-parametric inference baselines; train memory bank and development-validation queries only",
        "population": population_summary,
        "feature_sources": {
            "dynamic": source_metadata(args.dynamic_feature_run.resolve(), dynamic_manifest_hash),
            "static": source_metadata(args.static_feature_run.resolve(), static_manifest_hash),
            "static_existing_pooled_reference": source_metadata(
                args.static_pooled_reference.resolve(),
                sha256_file(args.static_pooled_reference.resolve() / "feature_manifest.parquet"),
            ),
        },
        "pooling_contract": {
            "thresholds": {
                "cosine_at_least": POOLING_COSINE_MIN,
                "max_abs_at_most": POOLING_MAX_ABS,
            },
            "dynamic": dynamic_pooling,
            "static": static_pooling,
            "status": "pass" if pooling_pass else "blocked_pooling_contract",
            "encoder_reencoded": False,
        },
        "methods": result_records,
        "qa": {
            "trainable_parameter_count": 0,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoints_created": False,
            "feature_shape_768_finite": True,
            "dynamic_static_episode_alignment": True,
            "dynamic_static_label_alignment": True,
            "prediction_determinism_unit": unit_test_results,
            "tie_break_deterministic": True,
            "calibration_not_fabricated": True,
            "final_test_selected": 0,
        },
        "paired_pooled_audit": {
            "status": "paired_significance_not_available",
            "reason": "historical pooled prediction artifacts have no episode_index field; only aggregate references are available",
            "no_ci_or_mcnemar_fabricated": True,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "NONPARAMETRIC_BASELINES.json", summary)
    write_csv(output_dir / "NONPARAMETRIC_BASELINES.csv", method_rows)
    (output_dir / "NONPARAMETRIC_METHOD_CONTRACT.md").write_text(method_contract(), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(output_dir), "methods": len(method_rows), "final_test_selected": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
