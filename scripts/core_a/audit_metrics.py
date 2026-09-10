"""Independently reproduce the train/validation linear-probe metrics."""

from __future__ import annotations

import argparse
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

from ee6008.data.selection import benchmark_primary_mask
from ee6008.metrics import classification_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_metrics(
    logits: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, Any]:
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("incompatible logits and labels")
    order = np.argsort(-logits, axis=1, kind="stable")
    top1_predictions = order[:, 0]
    top5_predictions = order[:, : min(5, logits.shape[1])]
    supports = np.bincount(labels, minlength=num_classes)
    recall1: list[float] = []
    recall5: list[float] = []
    f1s: list[float] = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(labels.tolist(), top1_predictions.tolist()):
        confusion[truth, prediction] += 1
    for class_index in range(num_classes):
        support = int(supports[class_index])
        if support == 0:
            continue
        truth_mask = labels == class_index
        recall1.append(float(np.mean(top1_predictions[truth_mask] == class_index)))
        recall5.append(
            float(np.mean(np.any(top5_predictions[truth_mask] == class_index, axis=1)))
        )
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "top1": float(np.mean(top1_predictions == labels)),
        "top5": float(np.mean(np.any(top5_predictions == labels[:, None], axis=1))),
        "mean_class_recall_at_1": float(np.mean(recall1)) if recall1 else 0.0,
        "mean_class_recall_at_5": float(np.mean(recall5)) if recall5 else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "class_support_min": int(supports[supports > 0].min()),
        "class_support_max": int(supports.max()),
        "supported_class_count": int(np.count_nonzero(supports)),
        "per_class_recall_at_5": recall5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    config_path = Path(args.config)
    raw_config = config_path.read_text(encoding="utf-8")
    # The audit intentionally reads the locked manifest and only selects
    # train/validation rows before any feature is loaded.
    import yaml

    config = yaml.safe_load(raw_config)
    manifest_dir = Path(config["manifest_dir"])
    split_table = pd.read_parquet(manifest_dir / "splits.parquet")
    task_ids = sorted(
        int(value)
        for value in split_table.loc[
            benchmark_primary_mask(split_table),
            "original_task_index",
        ].unique()
    )
    local_labels = {task_id: index for index, task_id in enumerate(task_ids)}
    rows = split_table[
        benchmark_primary_mask(split_table)
        & split_table["split"].isin(("train", "validation"))
    ].copy()
    feature_manifest = pd.read_parquet(args.feature_run / "feature_manifest.parquet")
    rows = rows.merge(
        feature_manifest[["episode_index", "feature_path"]],
        on="episode_index",
        how="inner",
        validate="one_to_one",
    ).sort_values("episode_index", kind="mergesort")
    if set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("audit did not obtain exactly train and validation rows")

    def load(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        labels = []
        for row in frame.itertuples(index=False):
            record = torch.load(row.feature_path, map_location="cpu", weights_only=True)
            features.append(record["pooled"].float())
            labels.append(local_labels[int(row.original_task_index)])
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    train_features, train_labels = load(rows[rows["split"] == "train"])
    validation_features, validation_labels = load(rows[rows["split"] == "validation"])
    torch.manual_seed(args.seed)
    model = torch.nn.Linear(train_features.shape[1], len(task_ids))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    for _ in range(args.epochs):
        logits = model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.inference_mode():
        validation_logits = model(validation_features).cpu()
    labels_np = validation_labels.numpy()
    logits_np = validation_logits.numpy()
    independent = independent_metrics(logits_np, labels_np, len(task_ids))
    project_metrics = classification_metrics(
        validation_logits, validation_labels, len(task_ids)
    )
    prediction_path = args.output_dir / "validation_predictions.npz"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        prediction_path,
        labels=labels_np,
        logits=logits_np,
        predictions=logits_np.argmax(axis=1),
    )
    report = {
        "status": "pass"
        if all(
            abs(float(independent[key]) - float(project_metrics[key])) <= 1e-7
            for key in (
                "top1",
                "top5",
                "mean_class_recall_at_1",
                "mean_class_recall_at_5",
                "macro_f1",
            )
        )
        else "review",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "feature_run": str(args.feature_run),
        "feature_manifest_sha256": sha256_file(
            args.feature_run / "feature_manifest.parquet"
        ),
        "split_path": str(manifest_dir / "splits.parquet"),
        "split_sha256": sha256_file(manifest_dir / "splits.parquet"),
        "task_ids": task_ids,
        "num_classes": len(task_ids),
        "train_count": len(train_labels),
        "validation_count": len(validation_labels),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": 0.05,
        "weight_decay": 1e-4,
        "class_weighting": "none",
        "early_stopping": False,
        "final_train_loss": final_loss,
        "final_test_access": "none",
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "independent_metrics": independent,
        "project_metrics": project_metrics,
        "top5_equals_mean_class_recall_at_5": abs(
            independent["top5"] - independent["mean_class_recall_at_5"]
        )
        <= 1e-12,
        "project_mean_class_recall_at_5_is_alias_of_top5": abs(
            project_metrics["mean_class_recall_at_5"] - project_metrics["top5"]
        )
        <= 1e-12,
        "interpretation": (
            "The independent per-class top-5 recall is computed separately. "
            "Equality with scalar top-5 is a measured result only if the "
            "reported values match; the historical project implementation "
            "assigned the scalar top-5 value directly, and the current "
            "implementation computes the per-class mean independently."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Independent metrics audit",
        "",
        f"Status: {report['status']}",
        f"Train/validation rows: {report['train_count']}/{report['validation_count']}",
        f"Classes: {report['num_classes']}",
        f"Seed/epochs: {report['seed']}/{report['epochs']}",
        "",
        "| metric | independent reference | project implementation | absolute difference |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "top1",
        "top5",
        "mean_class_recall_at_1",
        "mean_class_recall_at_5",
        "macro_f1",
    ):
        independent_value = float(report["independent_metrics"][key])
        project_value = float(report["project_metrics"][key])
        lines.append(
            f"| {key} | {independent_value:.12f} | {project_value:.12f} | "
            f"{abs(independent_value - project_value):.12f} |"
        )
    lines.extend(
        [
            "",
            (
                "Independent top-5 equals independent mean class recall@5: "
                f"{report['top5_equals_mean_class_recall_at_5']}."
            ),
            "",
            (
                "The saved prediction artifact contains validation labels and logits "
                "only. No final-test row was selected or loaded."
            ),
        ]
    )
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
