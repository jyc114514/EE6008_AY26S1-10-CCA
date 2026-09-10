"""Run a fixed train/validation baseline matrix with three probe seeds."""

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
from torchvision.models.video import R3D_18_Weights

from ee6008.config import load_config
from ee6008.data.selection import benchmark_primary_mask
from ee6008.metrics import classification_metrics

STATE_COLUMNS = (
    "observation.arm_joints",
    "observation.hand_joints",
    "observation.leg_joints",
)
DEFAULT_SEEDS = (20260825, 20260826, 20260827)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(config) -> tuple[pd.DataFrame, list[int], dict[int, int]]:
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    task_ids = sorted(
        int(value)
        for value in split_table.loc[
            benchmark_primary_mask(split_table),
            "original_task_index",
        ].unique()
    )
    local_labels = {task: index for index, task in enumerate(task_ids)}
    rows = split_table[
        benchmark_primary_mask(split_table)
        & split_table["split"].isin(("train", "validation"))
    ].sort_values("episode_index", kind="mergesort")
    if set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("baseline matrix requires train and validation rows")
    return rows, task_ids, local_labels


def load_pooled(rows: pd.DataFrame, feature_run: Path) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = pd.read_parquet(feature_run / "feature_manifest.parquet")
    merged = rows.merge(
        manifest[["episode_index", "feature_path"]],
        on="episode_index",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(rows):
        raise RuntimeError("pooled feature manifest does not cover all rows")
    features = []
    for row in merged.itertuples(index=False):
        record = torch.load(row.feature_path, map_location="cpu", weights_only=True)
        features.append(record["pooled"].float())
    return torch.stack(features), merged["split"].eq("train").to_numpy()


def load_state(
    rows: pd.DataFrame, g1_root: Path, local_labels: dict[int, int]
) -> tuple[torch.Tensor, np.ndarray]:
    features = []
    for position, row in enumerate(rows.itertuples(index=False), 1):
        path = g1_root / str(row.parquet_path)
        table = pd.read_parquet(path, columns=["frame_index", *STATE_COLUMNS])
        table = table.sort_values("frame_index", kind="mergesort")
        first = table.iloc[0]
        vectors = [np.asarray(first[column], dtype=np.float32) for column in STATE_COLUMNS]
        features.append(np.concatenate(vectors))
        if position % 500 == 0 or position == len(rows):
            print(f"STATE_BASELINE_PROGRESS {position}/{len(rows)}", flush=True)
    return torch.from_numpy(np.stack(features)), rows["split"].eq("train").to_numpy()


def normalize_train_only(
    features: torch.Tensor, train_mask: np.ndarray
) -> tuple[torch.Tensor, dict[str, Any]]:
    train = features[torch.from_numpy(train_mask)]
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False)
    safe_std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized = (features - mean) / safe_std
    return normalized, {
        "dimension": int(features.shape[1]),
        "train_mean_sha256": hashlib.sha256(mean.numpy().tobytes()).hexdigest(),
        "train_std_sha256": hashlib.sha256(safe_std.numpy().tobytes()).hexdigest(),
        "zero_variance_dimensions": int((std < 1e-6).sum().item()),
    }


def labels_for(rows: pd.DataFrame, local_labels: dict[int, int]) -> torch.Tensor:
    return torch.tensor(
        [local_labels[int(value)] for value in rows["original_task_index"]],
        dtype=torch.long,
    )


def learned_logits(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
) -> tuple[torch.Tensor, float]:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    final_loss = 0.0
    for _ in range(epochs):
        logits = model(train_features)
        loss = loss_fn(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.inference_mode():
        return model(validation_features), final_loss


def reference_logits(
    method: str,
    train_labels: torch.Tensor,
    validation_count: int,
    num_classes: int,
    seed: int,
) -> torch.Tensor:
    if method == "chance":
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            (validation_count, num_classes), generator=generator, dtype=torch.float32
        )
    if method == "majority":
        counts = torch.bincount(train_labels, minlength=num_classes)
        majority = int(torch.argmax(counts).item())
        logits = torch.full((validation_count, num_classes), -1.0)
        logits[:, majority] = 1.0
        return logits
    raise ValueError(f"reference logits not defined for {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "chance",
            "majority",
            "state_only",
            "vjepa2_1_b",
            "static_first_frame_vjepa2_1_b",
            "r3d18_kinetics400_v1",
        ),
        required=True,
    )
    parser.add_argument("--feature-run", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    config = load_config(args.config)
    rows, task_ids, local_labels = load_rows(config)
    train_mask = rows["split"].eq("train").to_numpy(copy=True)
    validation_mask = ~train_mask
    labels = labels_for(rows, local_labels)
    train_labels = labels[torch.from_numpy(train_mask)]
    validation_labels = labels[torch.from_numpy(validation_mask)]
    source_info: dict[str, Any] = {}

    if args.method in {
        "vjepa2_1_b",
        "static_first_frame_vjepa2_1_b",
        "r3d18_kinetics400_v1",
    }:
        if args.feature_run is None:
            raise ValueError("--feature-run is required for a feature baseline")
        features, feature_train_mask = load_pooled(rows, args.feature_run)
        if not np.array_equal(feature_train_mask, train_mask):
            raise AssertionError("feature row ordering differs from split row ordering")
        source_info = {
            "feature_run": str(args.feature_run),
            "feature_manifest_sha256": sha256_file(
                args.feature_run / "feature_manifest.parquet"
            ),
            "feature_shape": list(features.shape),
            "feature_mode": (
                "repeated_first_frame"
                if args.method == "static_first_frame_vjepa2_1_b"
                else "normal"
            ),
        }
        if args.method == "r3d18_kinetics400_v1":
            weight_url = R3D_18_Weights.KINETICS400_V1.url
            weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weight_url).name
            if not weight_path.is_file():
                raise FileNotFoundError(f"official R3D weight is not cached: {weight_path}")
            source_info.update(
                {
                    "weights_enum": "R3D_18_Weights.KINETICS400_V1",
                    "weights_url": weight_url,
                    "weights_path": str(weight_path),
                    "weights_sha256": sha256_file(weight_path),
                }
            )
    elif args.method == "state_only":
        features, state_train_mask = load_state(rows, config.g1_root, local_labels)
        if not np.array_equal(state_train_mask, train_mask):
            raise AssertionError("state row ordering differs from split row ordering")
        features, normalization = normalize_train_only(features, train_mask)
        source_info = {
            "state_columns": list(STATE_COLUMNS),
            "state_feature_train_mask_matches": True,
            **normalization,
        }
    else:
        features = None

    if features is not None:
        train_features = features[torch.from_numpy(train_mask)]
        validation_features = features[torch.from_numpy(validation_mask)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        final_loss: float | None = None
        if args.method in {"chance", "majority"}:
            logits = reference_logits(
                args.method,
                train_labels,
                len(validation_labels),
                len(task_ids),
                seed,
            )
        else:
            logits, final_loss = learned_logits(
                train_features,
                train_labels,
                validation_features,
                num_classes=len(task_ids),
                seed=seed,
                epochs=args.epochs,
            )
        metrics = classification_metrics(logits, validation_labels, len(task_ids))
        prediction_path = args.output_dir / f"predictions_seed_{seed}.npz"
        np.savez_compressed(
            prediction_path,
            labels=validation_labels.numpy(),
            logits=logits.detach().cpu().numpy(),
            predictions=logits.argmax(dim=1).detach().cpu().numpy(),
        )
        per_seed.append(
            {
                "seed": seed,
                "metrics": metrics,
                "final_train_loss": final_loss,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
    metric_names = (
        "top1",
        "top5",
        "mean_class_recall_at_1",
        "mean_class_recall_at_5",
        "macro_f1",
    )
    aggregate = {
        name: {
            "mean": float(np.mean([item["metrics"][name] for item in per_seed])),
            "std": float(np.std([item["metrics"][name] for item in per_seed], ddof=1))
            if len(per_seed) > 1
            else 0.0,
        }
        for name in metric_names
    }
    report = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "method": args.method,
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "split_path": str(config.manifest_dir / "splits.parquet"),
        "split_sha256": sha256_file(config.manifest_dir / "splits.parquet"),
        "task_count": len(task_ids),
        "train_count": int(train_mask.sum()),
        "validation_count": int(validation_mask.sum()),
        "seeds": args.seeds,
        "epochs": args.epochs if args.method not in {"chance", "majority"} else None,
        "learning_rate": 0.05 if args.method not in {"chance", "majority"} else None,
        "weight_decay": 1e-4 if args.method not in {"chance", "majority"} else None,
        "class_weighting": "none",
        "early_stopping": False,
        "source": source_info,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "chance_theoretical_top1": 1.0 / len(task_ids)
        if args.method == "chance"
        else None,
        "final_test_access": "none",
    }
    (args.output_dir / "matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Core A baseline matrix",
        "",
        f"Method: {args.method}",
        f"Status: {report['status']}",
        f"Train/validation: {report['train_count']}/{report['validation_count']}",
        f"Seeds: {report['seeds']}",
        "",
        "| metric | mean | sample std |",
        "|---|---:|---:|",
    ]
    for name in metric_names:
        lines.append(
            f"| {name} | {aggregate[name]['mean']:.12f} | {aggregate[name]['std']:.12f} |"
        )
    lines.extend(
        [
            "",
            f"Source: {source_info}",
            "",
            (
                "All predictions and metrics use train/validation rows only. No "
                "final-test labels or metrics were accessed."
            ),
        ]
    )
    (args.output_dir / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
