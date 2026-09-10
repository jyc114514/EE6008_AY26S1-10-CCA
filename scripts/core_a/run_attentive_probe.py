"""Run the official-style V-JEPA attentive frozen probe on train/validation only.

The probe follows the local V-JEPA source implementation: a four-block
``AttentiveClassifier`` with sixteen attention heads consumes last-layer patch
tokens.  It intentionally never selects or scores ``final_test`` rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ee6008.config import load_config
from ee6008.data.selection import benchmark_primary_mask
from ee6008.metrics import classification_metrics

DEFAULT_SEEDS = (20260825, 20260826, 20260827)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: pd.DataFrame, local_labels: dict[int, int], feature_root: Path):
        self.paths: list[Path] = []
        self.labels: list[int] = []
        for row in rows.itertuples(index=False):
            path = Path(row.feature_path)
            if not path.is_absolute():
                path = feature_root / path
            self.paths.append(path)
            self.labels.append(local_labels[int(row.original_task_index)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        tokens = record.get("tokens")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] == 0:
            raise RuntimeError(f"invalid token record: {self.paths[index]}")
        return tokens.float(), torch.tensor(self.labels[index], dtype=torch.long)


def select_rows(
    config: Any,
    feature_run: Path,
    task_set: str,
    selection_path: Path | None,
) -> tuple[pd.DataFrame, list[int], dict[int, int]]:
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    primary = split_table[benchmark_primary_mask(split_table)]
    if task_set == "all":
        task_ids = sorted(int(value) for value in primary["original_task_index"].unique())
    else:
        if selection_path is None:
            raise ValueError("--selection is required for smoke/pilot task sets")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        task_ids = [int(value) for value in selection[f"{task_set}_tasks"]]
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise ValueError("task selection must contain unique non-empty task ids")
    local_labels = {task_id: index for index, task_id in enumerate(task_ids)}

    feature_manifest = pd.read_parquet(feature_run / "feature_manifest.parquet")
    required = {"episode_index", "feature_path"}
    missing = required - set(feature_manifest.columns)
    if missing:
        raise RuntimeError(f"feature manifest missing columns: {sorted(missing)}")
    rows = primary[primary["original_task_index"].isin(task_ids)].merge(
        feature_manifest[["episode_index", "feature_path"]],
        on="episode_index",
        how="inner",
        validate="one_to_one",
    )
    rows = rows[rows["split"].isin(("train", "validation"))].sort_values(
        "episode_index", kind="mergesort"
    )
    if rows.empty or set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("attentive probe requires non-empty train and validation rows")
    if rows["episode_index"].duplicated().any():
        raise RuntimeError("duplicate episode rows in attentive-probe input")
    if rows["split"].eq("final_test").any():
        raise RuntimeError("final_test row reached attentive probe")
    return rows, task_ids, local_labels


def make_loader(
    rows: pd.DataFrame,
    local_labels: dict[int, int],
    feature_run: Path,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TokenDataset(rows, local_labels, feature_run)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )


def run_seed(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    local_labels: dict[int, int],
    feature_run: Path,
    *,
    device: torch.device,
    seed: int,
    num_classes: int,
    embed_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    use_bfloat16: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_loader = make_loader(
        train_rows,
        local_labels,
        feature_run,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_rows,
        local_labels,
        feature_run,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    source_root = feature_run.parents[3] / "repo" / "third_party" / "vjepa2"
    if not (source_root / "src/models/attentive_pooler.py").is_file():
        source_root = Path(os.environ["EE6008_VJEPA_SOURCE_ROOT"])
    sys.path.insert(0, str(source_root))
    from src.models.attentive_pooler import AttentiveClassifier

    model = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=16,
        depth=4,
        num_classes=num_classes,
        use_activation_checkpointing=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    autocast_enabled = use_bfloat16 and device.type == "cuda"
    last_loss = math.nan
    for epoch in range(epochs):
        model.train()
        for step, (tokens, labels) in enumerate(train_loader, 1):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                logits = model(tokens)
                loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            if step == 1 or step == len(train_loader) or step % 50 == 0:
                print(
                    f"ATTENTIVE_PROGRESS seed={seed} epoch={epoch + 1}/{epochs} "
                    f"step={step}/{len(train_loader)} loss={last_loss:.6f}",
                    flush=True,
                )
        scheduler.step()

    model.eval()
    validation_logits: list[torch.Tensor] = []
    validation_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for tokens, labels in validation_loader:
            tokens = tokens.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                logits = model(tokens)
            validation_logits.append(logits.float().cpu())
            validation_labels.append(labels)
    logits_np = torch.cat(validation_logits).numpy()
    labels_np = torch.cat(validation_labels).numpy()
    metrics = classification_metrics(
        torch.from_numpy(logits_np), torch.from_numpy(labels_np), num_classes
    )
    return {
        "seed": seed,
        "metrics": metrics,
        "final_train_loss": last_loss,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "train_batches_per_epoch": len(train_loader),
        "validation_batches": len(validation_loader),
    }, logits_np, labels_np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--task-set", choices=("smoke", "pilot", "all"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--no-bfloat16", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")

    config = load_config(args.config)
    feature_run = args.feature_run.resolve()
    rows, task_ids, local_labels = select_rows(
        config, feature_run, args.task_set, args.selection
    )
    train_rows = rows[rows["split"].eq("train")]
    validation_rows = rows[rows["split"].eq("validation")]
    first_path = Path(train_rows.iloc[0].feature_path)
    if not first_path.is_absolute():
        first_path = feature_run / first_path
    first_record = torch.load(first_path, map_location="cpu", weights_only=True)
    tokens = first_record.get("tokens")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] == 0:
        raise RuntimeError("feature run does not contain non-empty token tensors")
    token_shape = list(tokens.shape)
    embed_dim = int(tokens.shape[1])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")

    per_seed: list[dict[str, Any]] = []
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        result, logits, labels = run_seed(
            train_rows,
            validation_rows,
            local_labels,
            feature_run,
            device=device,
            seed=seed,
            num_classes=len(task_ids),
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_bfloat16=not args.no_bfloat16,
        )
        prediction_path = args.output / f"predictions_seed_{seed}.npz"
        np.savez_compressed(prediction_path, labels=labels, logits=logits)
        result["prediction_path"] = str(prediction_path)
        result["prediction_sha256"] = sha256_file(prediction_path)
        per_seed.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    source_root = config.vjepa_source_root
    report = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "task_set": args.task_set,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "feature_run": str(feature_run),
        "feature_manifest_sha256": sha256_file(feature_run / "feature_manifest.parquet"),
        "token_shape": token_shape,
        "probe": {
            "implementation": "third_party/vjepa2/src/models/attentive_pooler.py",
            "source_commit": config.source_commit,
            "source_file_sha256": sha256_file(source_root / "src/models/attentive_pooler.py"),
            "depth": 4,
            "num_heads": 16,
            "last_layer_patch_tokens_only": True,
            "num_encoder_layers_requested": 1,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "bfloat16_autocast": not args.no_bfloat16 and device.type == "cuda",
        "device": str(device),
        "seeds": args.seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "final_test_access": "none",
        "label_scope": "train and validation only; no final-test labels or metrics accessed",
    }
    (args.output / "attentive_probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Core A attentive frozen probe",
        "",
        f"Task set: {args.task_set}; train/validation: {len(train_rows)}/{len(validation_rows)}",
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
            (
                "Official-style local implementation: last-layer patch tokens, "
                "AttentiveClassifier depth=4, num_heads=16."
            ),
            "",
            "No final-test labels or metrics were accessed.",
        ]
    )
    (args.output / "attentive_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
