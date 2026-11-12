from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROJECT = Path(
    "/usr1/home/s126mdg33_02/EE6008 AY26S1-10-CCA Project"
)
RUN = PROJECT / "tmp/s126mdg33_02/videomae_baseline_20260907"
ARCHIVE_SOURCE = (
    PROJECT
    / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_a/"
    "analysis/CORE_A_UNIFIED_ANALYSIS_20260909T052330Z/SOURCE_MANIFEST.csv"
)
MANIFEST = (
    PROJECT
    / "repo/ee6008-humanoid-vjepa/manifests/"
    "g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/"
    "core_a_protocol_v3_duplicate_safe/train_validation_label_manifest.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only validation audit of the VideoMAE linear probe."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=RUN / "outputs/features")
    parser.add_argument("--result-dir", type=Path, default=RUN / "outputs/linear_probe")
    parser.add_argument("--seeds", type=int, nargs="+", default=[6008, 7, 42])
    parser.add_argument("--archive-source", type=Path, default=ARCHIVE_SOURCE)
    parser.add_argument("--reference-cell", default="D-P-F")
    return parser.parse_args()


def digest_json(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def primary_rows(manifest: Path, split: str):
    import pandas as pd

    frame = pd.read_parquet(manifest)
    required = {"episode_index", "split", "contiguous_label_index"}
    require(required <= set(frame.columns), "Manifest columns are missing")
    frame = frame[frame["split"] == split]
    for column in ("v3_benchmark_inclusion", "classification_inclusion"):
        require(column in frame.columns, f"Manifest lacks {column}")
        frame = frame[frame[column] == "primary"]
    require("eligible_4s" in frame.columns, "Manifest lacks eligible_4s")
    frame = frame[frame["eligible_4s"] == True]  # noqa: E712
    frame = frame.sort_values("episode_index").reset_index(drop=True)
    require(frame["episode_index"].is_unique, f"Duplicate {split} episode_index")
    return frame


def load_feature(path: Path, torch):
    data = torch.load(path, map_location="cpu", weights_only=True)
    require(
        {"features", "labels", "episode_indices", "metadata"} <= set(data),
        f"Feature keys missing: {path}",
    )
    return data


def validate_feature(data, rows, split: str) -> dict:
    ids = [int(value) for value in data["episode_indices"].tolist()]
    labels = [int(value) for value in data["labels"].tolist()]
    expected_ids = rows["episode_index"].astype(int).tolist()
    expected_labels = rows["contiguous_label_index"].astype(int).tolist()
    require(ids == expected_ids, f"{split}: feature episode IDs differ from manifest")
    require(labels == expected_labels, f"{split}: feature labels differ from manifest")
    require(len(ids) == len(set(ids)), f"{split}: duplicate feature episode IDs")
    require(data["features"].shape[0] == len(ids), f"{split}: feature count differs")
    require(data["metadata"].get("split") == split, f"{split}: metadata split differs")
    require(data["metadata"].get("final_test_used") is False, "Final-test flag is not false")
    return {"examples": len(ids), "ids_match_manifest": True, "labels_match_manifest": True}


def read_predictions(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = {
            "episode_index", "raw_label", "compact_label",
            "compact_prediction", "raw_prediction",
        }
        require(columns <= set(reader.fieldnames or []), f"Prediction columns missing: {path}")
        return [
            {column: int(row[column]) for column in columns}
            for row in reader
        ]


def macro_f1(pred, labels, classes: int, torch) -> float:
    scores = []
    for label in range(classes):
        tp = ((pred == label) & (labels == label)).sum().item()
        fp = ((pred == label) & (labels != label)).sum().item()
        fn = ((pred != label) & (labels == label)).sum().item()
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / classes


def calculate_metrics(logits, labels, torch) -> dict[str, float]:
    import torch.nn.functional as functional

    classes = logits.shape[1]
    pred = logits.argmax(dim=1)
    probs = logits.softmax(dim=1)
    confidence = probs.max(dim=1).values
    correct = pred == labels
    # The archive uses 15 equal-width confidence bins and weights by episode count.
    bins = torch.clamp((confidence * 15).long(), max=14)
    ece = 0.0
    for index in range(15):
        mask = bins == index
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    one_hot = functional.one_hot(labels, num_classes=classes).to(probs.dtype)
    return {
        "top1": correct.float().mean().item(),
        "top5": (logits.topk(5, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item(),
        "macro_f1": macro_f1(pred, labels, classes, torch),
        "nll": functional.cross_entropy(logits, labels).item(),
        "brier": ((probs - one_hot) ** 2).sum(dim=1).mean().item(),
        "ece_15": ece,
    }


def audit_archive_reference(source: Path, cell: str, validation, mapping, torch) -> dict:
    import numpy as np

    with source.open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row["role"] == "prediction_best" and row["cell"] == cell
        ]
    require(len(matches) == 1, f"Expected one archive prediction for {cell}")
    row = matches[0]
    path = Path(row["path"])
    require(path.stat().st_size < 10_000_000, "Archive prediction is unexpectedly large")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"],
        "Archive prediction SHA-256 differs from source manifest",
    )
    with np.load(path, allow_pickle=False) as data:
        require(
            {"episode_index", "labels", "logits", "predictions"} <= set(data.files),
            "Archive prediction fields are missing",
        )
        reference_ids = data["episode_index"].astype("int64").tolist()
        reference_labels = data["labels"].astype("int64").tolist()
        reference_logits = data["logits"]
        reference_predictions = data["predictions"].astype("int64").tolist()
    require(len(reference_ids) == 617, "Archive reference does not have 617 episodes")
    require(len(reference_ids) == len(set(reference_ids)), "Archive reference has duplicate IDs")
    require(reference_logits.shape == (617, 120), "Archive logits shape differs")
    require(np.isfinite(reference_logits).all(), "Archive logits are non-finite")
    require(
        reference_predictions == reference_logits.argmax(axis=1).tolist(),
        "Archive predictions differ from argmax",
    )
    videomae_ids = [int(value) for value in validation["episode_indices"].tolist()]
    videomae_raw = [int(value) for value in validation["labels"].tolist()]
    require(set(reference_ids) == set(videomae_ids), "Archive and VideoMAE episode sets differ")
    reference_by_id = dict(zip(reference_ids, reference_labels))
    aligned_reference = [reference_by_id[episode_id] for episode_id in videomae_ids]
    videomae_compact = [mapping[label] for label in videomae_raw]
    compact_mismatches = sum(
        left != right for left, right in zip(aligned_reference, videomae_compact)
    )
    raw_mismatches = sum(
        left != right for left, right in zip(aligned_reference, videomae_raw)
    )
    return {
        "cell": cell,
        "source_manifest": str(source),
        "reference_prediction": str(path),
        "sha256_matches_source_manifest": True,
        "episode_sets_match": True,
        "episode_order_matches": reference_ids == videomae_ids,
        "reference_label_range": [min(reference_labels), max(reference_labels)],
        "compact_label_mismatches": compact_mismatches,
        "raw_label_mismatches": raw_mismatches,
        "class_order_matches_compact": compact_mismatches == 0,
    }


def audit_seed(seed: int, result_dir: Path, validation, raw_to_compact, torch) -> dict:
    result = json.loads((result_dir / f"metrics_seed{seed}.json").read_text(encoding="utf-8"))
    logits = torch.load(
        result_dir / f"validation_logits_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    rows = read_predictions(result_dir / f"validation_predictions_seed{seed}.csv")
    ids = [int(value) for value in validation["episode_indices"].tolist()]
    raw_labels = [int(value) for value in validation["labels"].tolist()]
    metadata = result["metadata"]
    mapping = {int(key): int(value) for key, value in metadata["raw_to_compact_label"].items()}
    require(mapping == raw_to_compact, f"Seed {seed}: label mapping differs")
    require(metadata.get("final_test_used") is False, f"Seed {seed}: final-test flag")
    require(metadata.get("num_classes") == len(mapping), f"Seed {seed}: class count")
    require(logits.shape == (len(ids), len(mapping)), f"Seed {seed}: logits shape")
    require(torch.isfinite(logits).all().item(), f"Seed {seed}: non-finite logits")
    require(len(rows) == len(ids), f"Seed {seed}: prediction count")
    compact = torch.tensor([mapping[label] for label in raw_labels], dtype=torch.long)
    inverse = {value: key for key, value in mapping.items()}
    require(len(inverse) == len(mapping), f"Seed {seed}: non-bijective mapping")
    predictions = logits.argmax(dim=1).tolist()
    for index, row in enumerate(rows):
        expected = (
            ids[index], raw_labels[index], compact[index].item(),
            predictions[index], inverse[predictions[index]],
        )
        actual = tuple(row[key] for key in (
            "episode_index", "raw_label", "compact_label",
            "compact_prediction", "raw_prediction",
        ))
        require(actual == expected, f"Seed {seed}: prediction mismatch at row {index}")
    metrics = calculate_metrics(logits, compact, torch)
    for name in ("top1", "macro_f1", "nll"):
        require(
            math.isclose(metrics[name], result["metrics"][name], abs_tol=1e-6),
            f"Seed {seed}: stored {name} differs from recomputation",
        )
    return {
        "seed": seed,
        "episodes": len(ids),
        "prediction_rows_match": True,
        "stored_metrics_match": True,
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    require(
        args.manifest.name == "train_validation_label_manifest.parquet",
        "Only the train/validation label manifest is allowed",
    )
    import torch

    train_rows = primary_rows(args.manifest, "train")
    validation_rows = primary_rows(args.manifest, "validation")
    train = load_feature(args.feature_dir / "videomae_train_pooled.pt", torch)
    validation = load_feature(args.feature_dir / "videomae_validation_pooled.pt", torch)
    train_check = validate_feature(train, train_rows, "train")
    validation_check = validate_feature(validation, validation_rows, "validation")
    train_labels = set(int(value) for value in train["labels"].tolist())
    validation_labels = set(int(value) for value in validation["labels"].tolist())
    raw_to_compact = {
        label: index for index, label in enumerate(sorted(train_labels | validation_labels))
    }
    require(len(raw_to_compact) == 120, "Expected 120 primary labels")
    require(set(raw_to_compact.values()) == set(range(120)), "Compact label gap")
    archive_alignment = audit_archive_reference(
        args.archive_source,
        args.reference_cell,
        validation,
        raw_to_compact,
        torch,
    )
    report = {
        "status": (
            "pass" if archive_alignment["class_order_matches_compact"]
            else "needs_label_mapping_review"
        ),
        "manifest": str(args.manifest),
        "train": train_check,
        "validation": validation_check,
        "classes": len(raw_to_compact),
        "label_mapping_sha256": digest_json(raw_to_compact),
        "validation_id_label_sha256": digest_json([
            [int(episode_id), int(label)]
            for episode_id, label in zip(
                validation["episode_indices"].tolist(),
                validation["labels"].tolist(),
            )
        ]),
        "seeds": [
            audit_seed(seed, args.result_dir, validation, raw_to_compact, torch)
            for seed in args.seeds
        ],
        "final_test_access": 0,
        "archive_prediction_alignment": archive_alignment,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
