"""Run the v3 full frozen attentive dynamic/static comparison.

The runner is deliberately validation-only: it selects v3 primary train and
development-validation rows and never loads final-test labels.  It writes
per-epoch validation predictions, calibration metrics, resumable checkpoints,
and an atomic completion marker for each seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ee6008.config import CoreAConfig, expand_path, load_config
from ee6008.full_attentive import (
    EarlyStoppingState,
    extended_classification_metrics,
    validate_token_shape,
)
from ee6008.full_attentive_data import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_TOKEN_SHAPE,
    Population,
    attach_feature_manifest,
    load_v3_population,
    sha256_file,
)

DEFAULT_SEEDS = (20260825, 20260826, 20260827)
EXPECTED_MAX_EPOCHS = 20
EXPECTED_MIN_EPOCHS = 5
EXPECTED_PATIENCE = 5
EXPECTED_MIN_DELTA = 0.001
EXPECTED_BATCH_SIZE = 4
EXPECTED_LEARNING_RATE = 0.001
EXPECTED_WEIGHT_DECAY = 1e-4
EXPECTED_NUM_HEADS = 16
EXPECTED_PROBE_DEPTH = 4
EXPECTED_CALIBRATION_BINS = 15


@dataclass(frozen=True)
class Experiment:
    core: CoreAConfig
    config_path: Path
    config_sha256: str
    raw: dict[str, Any]
    variant: str
    feature_run: Path
    output_root: Path
    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_heads: int
    probe_depth: int
    calibration_bins: int
    num_workers: int
    preload_tokens: bool
    use_bfloat16: bool
    seeds: tuple[int, ...]

    @property
    def policy(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "selection_metric": "validation_macro_f1",
            "tie_break_1": "lower_validation_nll",
            "tie_break_2": "earlier_epoch",
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "probe_depth": self.probe_depth,
            "num_heads": self.num_heads,
            "calibration_bins": self.calibration_bins,
        }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )


def atomic_write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_experiment(path: Path) -> Experiment:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"experiment config must be a mapping: {path}")
    core = load_config(path)
    variant = str(raw.get("variant", ""))
    if variant not in {"dynamic", "static_repeated_first_frame"}:
        raise ValueError(f"unsupported variant: {variant}")
    if core.allow_final_test_label_metrics:
        raise ValueError(
            "full attentive runner requires final-test label metrics disabled"
        )
    feature_run = expand_path(raw["feature_run"]).resolve()
    output_root = expand_path(raw["output_root"]).resolve()
    seeds = tuple(int(seed) for seed in raw.get("seeds", DEFAULT_SEEDS))
    experiment = Experiment(
        core=core,
        config_path=path,
        config_sha256=sha256_file(path),
        raw=raw,
        variant=variant,
        feature_run=feature_run,
        output_root=output_root,
        max_epochs=int(raw.get("max_epochs", EXPECTED_MAX_EPOCHS)),
        min_epochs=int(raw.get("min_epochs", EXPECTED_MIN_EPOCHS)),
        patience=int(raw.get("patience", EXPECTED_PATIENCE)),
        min_delta=float(raw.get("min_delta", EXPECTED_MIN_DELTA)),
        batch_size=int(raw.get("batch_size", EXPECTED_BATCH_SIZE)),
        learning_rate=float(raw.get("learning_rate", EXPECTED_LEARNING_RATE)),
        weight_decay=float(raw.get("weight_decay", EXPECTED_WEIGHT_DECAY)),
        num_heads=int(raw.get("num_heads", EXPECTED_NUM_HEADS)),
        probe_depth=int(raw.get("probe_depth", EXPECTED_PROBE_DEPTH)),
        calibration_bins=int(raw.get("calibration_bins", EXPECTED_CALIBRATION_BINS)),
        num_workers=int(raw.get("num_workers", 0)),
        preload_tokens=bool(raw.get("preload_tokens", True)),
        use_bfloat16=bool(raw.get("use_bfloat16", True)),
        seeds=seeds,
    )
    if experiment.max_epochs != EXPECTED_MAX_EPOCHS:
        raise ValueError("full comparison requires max_epochs=20")
    if experiment.min_epochs != EXPECTED_MIN_EPOCHS:
        raise ValueError("full comparison requires min_epochs=5")
    if experiment.patience != EXPECTED_PATIENCE:
        raise ValueError("full comparison requires patience=5")
    if experiment.min_delta != EXPECTED_MIN_DELTA:
        raise ValueError("full comparison requires min_delta=0.001")
    if experiment.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError("full comparison requires batch_size=4")
    if experiment.learning_rate != EXPECTED_LEARNING_RATE:
        raise ValueError("full comparison must preserve attentive pilot learning rate")
    if experiment.weight_decay != EXPECTED_WEIGHT_DECAY:
        raise ValueError("full comparison must preserve attentive pilot weight decay")
    if (
        experiment.num_heads != EXPECTED_NUM_HEADS
        or experiment.probe_depth != EXPECTED_PROBE_DEPTH
    ):
        raise ValueError("full comparison requires the official depth-4, 16-head probe")
    if experiment.calibration_bins != EXPECTED_CALIBRATION_BINS:
        raise ValueError("full comparison requires 15 calibration bins")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not feature_run.is_dir():
        raise FileNotFoundError(f"feature run is absent: {feature_run}")
    return experiment


def select_rows(
    population: Population,
    feature_rows: pd.DataFrame,
    *,
    task_ids: tuple[int, ...] | None = None,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
) -> pd.DataFrame:
    rows = feature_rows.copy()
    if task_ids is not None:
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be non-empty and unique")
        rows = rows[rows["original_task_index"].isin(task_ids)]
    if max_train_rows is not None:
        if max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")
        train = rows[rows["split"].eq("train")].head(max_train_rows)
    else:
        train = rows[rows["split"].eq("train")]
    if max_validation_rows is not None:
        if max_validation_rows <= 0:
            raise ValueError("max_validation_rows must be positive")
        validation = rows[rows["split"].eq("validation")].head(max_validation_rows)
    else:
        validation = rows[rows["split"].eq("validation")]
    selected = pd.concat((train, validation), ignore_index=True)
    if selected.empty or set(selected["split"].unique()) != {"train", "validation"}:
        raise ValueError("selected population must contain train and validation rows")
    if selected["episode_index"].duplicated().any():
        raise ValueError("selected population contains duplicate episode IDs")
    if selected["split"].eq("final_test").any():
        raise ValueError("final-test row reached the runner")
    del population
    return selected.sort_values("episode_index", kind="mergesort").reset_index(
        drop=True
    )


class TokenDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self, rows: pd.DataFrame, *, preload: bool, shared_cache: dict[str, Tensor]
    ):
        self.rows = rows.reset_index(drop=True)
        self.shared_cache = shared_cache
        self.preload = preload
        if preload:
            for path in self.rows["resolved_feature_path"]:
                self._load(path)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path_value: str) -> Tensor:
        path = str(path_value)
        if path not in self.shared_cache:
            record = torch.load(path, map_location="cpu", weights_only=True)
            tokens = record.get("tokens")
            validate_token_shape(tokens, expected=EXPECTED_TOKEN_SHAPE)
            self.shared_cache[path] = tokens
        return self.shared_cache[path]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.rows.iloc[index]
        tokens = self._load(row.resolved_feature_path).float()
        return (
            tokens,
            torch.tensor(int(row.local_label_index), dtype=torch.long),
            torch.tensor(int(row.episode_index), dtype=torch.long),
        )


def make_loader(
    rows: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    preload: bool,
    shared_cache: dict[str, Tensor],
    pin_memory: bool,
) -> DataLoader:
    dataset = TokenDataset(rows, preload=preload, shared_cache=shared_cache)
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": 0 if preload else num_workers,
        "pin_memory": pin_memory,
    }
    if kwargs["num_workers"]:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def build_head(experiment: Experiment, device: torch.device) -> nn.Module:
    source_root = experiment.core.vjepa_source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from src.models.attentive_pooler import AttentiveClassifier

    head = AttentiveClassifier(
        embed_dim=EXPECTED_TOKEN_SHAPE[1],
        num_heads=experiment.num_heads,
        depth=experiment.probe_depth,
        num_classes=EXPECTED_CLASS_COUNT,
        use_activation_checkpointing=True,
    ).to(device)
    return head


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_bfloat16: bool,
    loss_fn: nn.Module,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    top5_correct = 0
    for tokens, labels, _ in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_bfloat16):
            logits = model(tokens)
            loss = loss_fn(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is NaN or Inf")
        loss.backward()
        optimizer.step()
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total += batch_size
        predictions = logits.argmax(dim=1)
        correct += int(predictions.eq(labels).sum().item())
        top5_correct += int(
            logits.topk(min(5, logits.shape[1]), dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .sum()
            .item()
        )
    if total == 0:
        raise ValueError("empty training loader")
    return {
        "loss": total_loss / total,
        "top1": correct / total,
        "top5": top5_correct / total,
        "total": total,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, Any]:
    model.eval()
    logits_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    episode_list: list[Tensor] = []
    with torch.inference_mode():
        for tokens, labels, episodes in loader:
            tokens = tokens.to(device, non_blocking=True)
            with autocast_context(device, use_bfloat16):
                logits = model(tokens)
            logits = logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("validation logits contain NaN or Inf")
            logits_list.append(logits)
            labels_list.append(labels.cpu())
            episode_list.append(episodes.cpu())
    if not logits_list:
        raise ValueError("empty validation loader")
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    episodes = torch.cat(episode_list)
    metrics = extended_classification_metrics(
        logits,
        labels,
        EXPECTED_CLASS_COUNT,
        calibration_bins=EXPECTED_CALIBRATION_BINS,
    )
    return {
        "metrics": metrics,
        "logits": logits.numpy(),
        "labels": labels.numpy(),
        "episode_index": episodes.numpy(),
        "predictions": logits.argmax(dim=1).numpy(),
    }


def cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def cpu_optimizer_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_optimizer_state(item) for item in value)
    return value


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_state: EarlyStoppingState,
    *,
    experiment: Experiment,
    feature_manifest_sha256: str,
    population_sha256: str,
    seed: int,
    epoch: int,
    best_metrics: dict[str, Any] | None,
    include_optimizer: bool,
    kind: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "seed": seed,
        "epoch": epoch,
        "model_state_dict": cpu_state_dict(model),
        "scheduler_state_dict": scheduler.state_dict(),
        "early_stopping": early_state.to_dict(),
        "best_metrics": best_metrics,
        "experiment_config_sha256": experiment.config_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "population_sha256": population_sha256,
        "policy": experiment.policy,
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = cpu_optimizer_state(optimizer.state_dict())
    return payload


def restore_current(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    experiment: Experiment,
    feature_manifest_sha256: str,
    population_sha256: str,
    seed: int,
) -> tuple[int, EarlyStoppingState, dict[str, Any] | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "early_stopping",
        "experiment_config_sha256",
        "feature_manifest_sha256",
        "population_sha256",
    }
    if not required.issubset(payload):
        raise ValueError(f"resume checkpoint is missing state: {path}")
    if (
        payload["seed"] != seed
        or payload["experiment_config_sha256"] != experiment.config_sha256
    ):
        raise ValueError("resume checkpoint seed/config hash mismatch")
    if (
        payload["feature_manifest_sha256"] != feature_manifest_sha256
        or payload["population_sha256"] != population_sha256
    ):
        raise ValueError("resume checkpoint input hash mismatch")
    if payload.get("policy") != experiment.policy:
        raise ValueError("resume checkpoint early-stopping policy mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return (
        int(payload["epoch"]) + 1,
        EarlyStoppingState.from_dict(payload["early_stopping"]),
        payload.get("best_metrics"),
    )


CURVE_FIELDS = [
    "seed",
    "epoch",
    "train_loss",
    "train_top1",
    "train_top5",
    "validation_nll",
    "validation_top1",
    "validation_top5",
    "validation_mean_class_recall_at_1",
    "validation_mean_class_recall_at_5",
    "validation_macro_f1",
    "validation_brier",
    "validation_ece",
    "validation_mean_max_softmax_probability",
    "validation_correct",
    "validation_total",
    "learning_rate",
    "epoch_wall_time_seconds",
    "peak_vram_allocated_bytes",
    "peak_vram_reserved_bytes",
    "best_so_far",
    "bad_epochs",
    "host_load_1m",
]


def host_load_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return None


def write_epoch_prediction(seed_dir: Path, epoch: int, result: dict[str, Any]) -> Path:
    path = seed_dir / f"predictions_epoch_{epoch:03d}.npz"
    atomic_write_npz(
        path,
        episode_index=result["episode_index"],
        labels=result["labels"],
        logits=result["logits"],
        predictions=result["predictions"],
    )
    return path


def run_seed(
    experiment: Experiment,
    selected: pd.DataFrame,
    *,
    device: torch.device,
    seed: int,
    feature_manifest_sha256: str,
    population_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    seed_dir = experiment.output_root / f"seed_{seed}"
    seed_complete_path = seed_dir / "seed_complete.json"
    seed_contract = {
        "seed": seed,
        "config_sha256": experiment.config_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "population_sha256": population_sha256,
        "policy": experiment.policy,
    }
    if seed_complete_path.exists():
        completed = json.loads(seed_complete_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") != "pass"
            or completed.get("seed_contract") != seed_contract
        ):
            raise ValueError(f"completed seed has incompatible contract: {seed_dir}")
        return completed
    if seed_dir.exists() and any(seed_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"partial seed exists; use --resume only with a matching state: {seed_dir}"
        )
    seed_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(seed_dir / "seed_contract.json", seed_contract)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_head(experiment, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=experiment.max_epochs
    )
    loss_fn = nn.CrossEntropyLoss()
    shared_cache: dict[str, Tensor] = {}
    train_rows = selected[selected["split"].eq("train")]
    validation_rows = selected[selected["split"].eq("validation")]
    train_loader = make_loader(
        train_rows,
        batch_size=experiment.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=experiment.num_workers,
        preload=experiment.preload_tokens,
        shared_cache=shared_cache,
        pin_memory=device.type == "cuda",
    )
    validation_loader = make_loader(
        validation_rows,
        batch_size=experiment.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=experiment.num_workers,
        preload=experiment.preload_tokens,
        shared_cache=shared_cache,
        pin_memory=device.type == "cuda",
    )
    early_state = EarlyStoppingState()
    best_metrics: dict[str, Any] | None = None
    start_epoch = 1
    current_path = seed_dir / "checkpoint_current.pt"
    if resume and current_path.exists():
        start_epoch, early_state, best_metrics = restore_current(
            current_path,
            model,
            optimizer,
            scheduler,
            experiment=experiment,
            feature_manifest_sha256=feature_manifest_sha256,
            population_sha256=population_sha256,
            seed=seed,
        )
    elif resume and any(seed_dir.iterdir()):
        raise FileNotFoundError(
            f"resume requested but checkpoint_current.pt is absent: {seed_dir}"
        )

    curves: list[dict[str, Any]] = []
    curves_path = seed_dir / "epoch_curves.csv"
    if start_epoch > 1 and curves_path.exists():
        with curves_path.open(newline="", encoding="utf-8") as handle:
            curves = list(csv.DictReader(handle))
    last_validation: dict[str, Any] | None = None
    started = time.monotonic()
    stop_reason = "max_epochs_reached"
    for epoch in range(start_epoch, experiment.max_epochs + 1):
        epoch_started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            use_bfloat16=experiment.use_bfloat16,
            loss_fn=loss_fn,
        )
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            use_bfloat16=experiment.use_bfloat16,
        )
        last_validation = validation
        metrics = validation["metrics"]
        improved = early_state.update(
            epoch=epoch,
            macro_f1=float(metrics["macro_f1"]),
            nll=float(metrics["nll"]),
            min_delta=experiment.min_delta,
        )
        if improved:
            best_metrics = {key: value for key, value in metrics.items()}
            best_metrics["epoch"] = epoch
            best_prediction_path = seed_dir / "predictions_best.npz"
            atomic_write_npz(
                best_prediction_path,
                episode_index=validation["episode_index"],
                labels=validation["labels"],
                logits=validation["logits"],
                predictions=validation["predictions"],
            )
            atomic_write_torch(
                seed_dir / "checkpoint_best.pt",
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    early_state,
                    experiment=experiment,
                    feature_manifest_sha256=feature_manifest_sha256,
                    population_sha256=population_sha256,
                    seed=seed,
                    epoch=epoch,
                    best_metrics=best_metrics,
                    include_optimizer=False,
                    kind="best",
                ),
            )
        scheduler.step()
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        )
        curve = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_top1": train_result["top1"],
            "train_top5": train_result["top5"],
            "validation_nll": metrics["nll"],
            "validation_top1": metrics["top1"],
            "validation_top5": metrics["top5"],
            "validation_mean_class_recall_at_1": metrics["mean_class_recall_at_1"],
            "validation_mean_class_recall_at_5": metrics["mean_class_recall_at_5"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_brier": metrics["brier"],
            "validation_ece": metrics["ece"],
            "validation_mean_max_softmax_probability": metrics[
                "mean_max_softmax_probability"
            ],
            "validation_correct": metrics["correct"],
            "validation_total": metrics["total"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_wall_time_seconds": time.monotonic() - epoch_started,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "best_so_far": improved,
            "bad_epochs": early_state.bad_epochs,
            "host_load_1m": host_load_1m(),
        }
        curves.append(curve)
        write_epoch_prediction(seed_dir, epoch, validation)
        atomic_write_csv(curves_path, curves, CURVE_FIELDS)
        atomic_write_torch(
            current_path,
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                early_state,
                experiment=experiment,
                feature_manifest_sha256=feature_manifest_sha256,
                population_sha256=population_sha256,
                seed=seed,
                epoch=epoch,
                best_metrics=best_metrics,
                include_optimizer=True,
                kind="current",
            ),
        )
        print(
            f"FULL_ATTENTIVE_PROGRESS variant={experiment.variant} seed={seed} "
            f"epoch={epoch}/{experiment.max_epochs} train_loss={float(train_result['loss']):.6f} "
            f"val_macro_f1={float(metrics['macro_f1']):.6f} val_top1={float(metrics['top1']):.6f} "
            f"vram_allocated={peak_allocated}",
            flush=True,
        )
        if (
            epoch >= experiment.min_epochs
            and early_state.bad_epochs >= experiment.patience
        ):
            stop_reason = "patience_reached"
            early_state.stop_reason = stop_reason
            break

    if last_validation is None:
        raise RuntimeError("run did not execute an epoch")
    last_epoch = int(curves[-1]["epoch"])
    atomic_write_npz(
        seed_dir / "predictions_last.npz",
        episode_index=last_validation["episode_index"],
        labels=last_validation["labels"],
        logits=last_validation["logits"],
        predictions=last_validation["predictions"],
    )
    atomic_write_torch(
        seed_dir / "checkpoint_last.pt",
        checkpoint_payload(
            model,
            optimizer,
            scheduler,
            early_state,
            experiment=experiment,
            feature_manifest_sha256=feature_manifest_sha256,
            population_sha256=population_sha256,
            seed=seed,
            epoch=last_epoch,
            best_metrics=best_metrics,
            include_optimizer=False,
            kind="last",
        ),
    )
    if early_state.stop_reason is None:
        early_state.stop_reason = stop_reason
    completed = {
        "status": "pass",
        "variant": experiment.variant,
        "seed": seed,
        "epochs_completed": last_epoch,
        "best_epoch": early_state.best_epoch,
        "stop_reason": early_state.stop_reason,
        "best_metrics": best_metrics,
        "last_metrics": last_validation["metrics"],
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "seed_contract": seed_contract,
        "prediction_best_sha256": sha256_file(seed_dir / "predictions_best.npz"),
        "prediction_last_sha256": sha256_file(seed_dir / "predictions_last.npz"),
        "checkpoint_best_sha256": sha256_file(seed_dir / "checkpoint_best.pt"),
        "checkpoint_last_sha256": sha256_file(seed_dir / "checkpoint_last.pt"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_allocated_bytes": max(
            int(row["peak_vram_allocated_bytes"]) for row in curves
        ),
        "peak_vram_reserved_bytes": max(
            int(row["peak_vram_reserved_bytes"]) for row in curves
        ),
        "final_test_access": "none",
    }
    atomic_write_json(seed_complete_path, completed)
    return completed


def preflight(experiment: Experiment) -> dict[str, Any]:
    population = load_v3_population(experiment.core, full=True)
    feature_rows, feature_manifest_sha256 = attach_feature_manifest(
        population, experiment.feature_run
    )
    selected = select_rows(population, feature_rows)
    feature_shapes = {
        "token_shape": sorted(
            {str(value) for value in selected["token_shape"].unique()}
        ),
        "pooled_shape": sorted(
            {str(value) for value in selected["pooled_shape"].unique()}
        ),
    }
    return {
        "status": "pass",
        "variant": experiment.variant,
        "config_path": str(experiment.config_path),
        "config_sha256": experiment.config_sha256,
        "feature_run": str(experiment.feature_run),
        "feature_manifest_sha256": feature_manifest_sha256,
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "train_count": int((selected["split"] == "train").sum()),
        "validation_count": int((selected["split"] == "validation").sum()),
        "class_count": EXPECTED_CLASS_COUNT,
        "task_ids": list(population.task_ids),
        "feature_shapes": feature_shapes,
        "all_feature_files_exist": True,
        "final_test_selected": False,
        "policy": experiment.policy,
    }


def run_experiment(
    experiment: Experiment,
    *,
    device: torch.device,
    output: Path | None,
    task_ids: tuple[int, ...] | None,
    max_train_rows: int | None,
    max_validation_rows: int | None,
    seeds: tuple[int, ...] | None,
    resume: bool,
) -> dict[str, Any]:
    population = load_v3_population(experiment.core, full=True)
    feature_rows, feature_manifest_sha256 = attach_feature_manifest(
        population, experiment.feature_run
    )
    selected = select_rows(
        population,
        feature_rows,
        task_ids=task_ids,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
    )
    output_root = output.resolve() if output is not None else experiment.output_root
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root != experiment.output_root:
        experiment = Experiment(**{**experiment.__dict__, "output_root": output_root})
    selected_seeds = seeds or experiment.seeds
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("run seeds must be non-empty and unique")
    run_contract = {
        "status": "running",
        "variant": experiment.variant,
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "device": str(device),
        "config_path": str(experiment.config_path),
        "config_sha256": experiment.config_sha256,
        "feature_run": str(experiment.feature_run),
        "feature_manifest_sha256": feature_manifest_sha256,
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "train_count": int((selected["split"] == "train").sum()),
        "validation_count": int((selected["split"] == "validation").sum()),
        "class_count": EXPECTED_CLASS_COUNT,
        "seeds": list(selected_seeds),
        "task_ids": list(population.task_ids if task_ids is None else task_ids),
        "policy": experiment.policy,
        "final_test_access": "none",
    }
    status_path = output_root / "RUN_STATUS.json"
    if status_path.exists() and not resume:
        raise FileExistsError(f"run status already exists: {status_path}")
    atomic_write_json(status_path, run_contract)
    completed: list[dict[str, Any]] = []
    for seed in selected_seeds:
        result = run_seed(
            experiment,
            selected,
            device=device,
            seed=seed,
            feature_manifest_sha256=feature_manifest_sha256,
            population_sha256=population.population_sha256,
            resume=resume,
        )
        completed.append(result)
        atomic_write_json(
            status_path,
            {**run_contract, "completed_seeds": [item["seed"] for item in completed]},
        )
    report = {
        **run_contract,
        "status": "pass",
        "completed_seeds": [item["seed"] for item in completed],
        "seed_summaries": completed,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(output_root / "RUN_COMPLETE.json", report)
    atomic_write_json(status_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="override output only for a bounded smoke"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment = load_experiment(args.config)
    bounded = (
        any(
            value is not None
            for value in (
                args.task_ids,
                args.max_train_rows,
                args.max_validation_rows,
                args.max_epochs,
                args.min_epochs,
                args.patience,
            )
        )
        or args.seeds is not None
        and tuple(args.seeds) != experiment.seeds
    )
    if (
        not bounded
        and not args.preflight_only
        and args.output is not None
        and args.output.resolve() != experiment.output_root
    ):
        raise ValueError("output override is reserved for a bounded smoke")
    if (
        args.max_epochs is not None
        or args.min_epochs is not None
        or args.patience is not None
    ):
        if not bounded:
            raise ValueError("epoch-policy overrides are reserved for bounded smoke")
        experiment = replace(
            experiment,
            max_epochs=args.max_epochs
            if args.max_epochs is not None
            else experiment.max_epochs,
            min_epochs=args.min_epochs
            if args.min_epochs is not None
            else experiment.min_epochs,
            patience=args.patience
            if args.patience is not None
            else experiment.patience,
        )
        if (
            experiment.max_epochs <= 0
            or experiment.min_epochs <= 0
            or experiment.patience <= 0
        ):
            raise ValueError("bounded smoke epoch policy must be positive")
    if args.preflight_only:
        report = preflight(experiment)
        if args.output is not None:
            output = args.output.resolve()
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite preflight output: {output}"
                )
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / "PREFLIGHT_REPORT.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = run_experiment(
        experiment,
        device=device,
        output=args.output,
        task_ids=tuple(args.task_ids) if args.task_ids else None,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        seeds=tuple(args.seeds) if args.seeds else None,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
