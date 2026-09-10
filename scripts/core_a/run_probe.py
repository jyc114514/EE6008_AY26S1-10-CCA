"""Run a pooled-feature probe using train/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ee6008.config import load_config
from ee6008.data.selection import benchmark_primary_mask
from ee6008.probe import train_linear_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--task-set", choices=("smoke", "pilot", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    config = load_config(args.config)
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    if args.task_set == "all":
        task_ids = sorted(
            int(value)
            for value in split_table.loc[
                benchmark_primary_mask(split_table),
                "original_task_index",
            ].unique()
        )
    else:
        if args.selection is None:
            raise ValueError("--selection is required for smoke/pilot task sets")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        task_ids = [int(value) for value in selection[f"{args.task_set}_tasks"]]
    feature_manifest = pd.read_parquet(args.feature_run / "feature_manifest.parquet")
    rows = split_table[split_table["original_task_index"].isin(task_ids)].merge(
        feature_manifest[["episode_index", "feature_path"]],
        on="episode_index",
        how="inner",
    )
    rows = rows[benchmark_primary_mask(rows)]
    rows = rows[rows["split"].isin(("train", "validation"))]
    if rows.empty or set(rows["split"]) - {"train", "validation"}:
        raise RuntimeError("probe requires train/validation features only")
    local_labels = {task_id: index for index, task_id in enumerate(task_ids)}
    train_rows = rows[rows["split"] == "train"]
    validation_rows = rows[rows["split"] == "validation"]

    def load_features(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        labels = []
        for row in frame.itertuples(index=False):
            record = torch.load(row.feature_path, map_location="cpu", weights_only=True)
            features.append(record["pooled"].float())
            labels.append(local_labels[int(row.original_task_index)])
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    train_features, train_labels = load_features(train_rows)
    validation_features, validation_labels = load_features(validation_rows)
    result = train_linear_probe(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        num_classes=len(task_ids),
        epochs=args.epochs,
    )
    output = {
        "task_set": args.task_set,
        "task_ids": task_ids,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "smoke_holdout": "validation",
        "final_test_access": "none",
        "metrics": result.metrics,
        "train_loss": result.train_loss,
        "epochs": result.epochs,
        "feature_run": str(args.feature_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
