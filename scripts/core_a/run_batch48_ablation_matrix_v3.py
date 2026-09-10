"""Run the pre-registered physical-batch-48 Core A ablation matrix.

The runner reuses the accepted v3 token dataset and official attentive-head
construction. It has strict physical-batch-48 and gradient-accumulation-1
contracts, supports only the three registered single-factor ablations, records
train-only class weights, and has no final-test code path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from ee6008.config import CoreAConfig, expand_path, load_config
from ee6008.full_attentive import EarlyStoppingState, extended_classification_metrics
from ee6008.full_attentive_data import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_TOKEN_SHAPE,
    load_v3_population,
)
from run_full_attentive_probe_v3 import (
    autocast_context,
    build_head,
    checkpoint_payload,
    make_loader,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

EXPECTED_BATCH_SIZE = 48
EXPECTED_MAX_EPOCHS = 20
EXPECTED_MIN_EPOCHS = 5
EXPECTED_PATIENCE = 5
EXPECTED_MIN_DELTA = 0.001
EXPECTED_LEARNING_RATE = 0.001
EXPECTED_WEIGHT_DECAY = 0.0001
EXPECTED_NUM_HEADS = 16
EXPECTED_PROBE_DEPTH = 4
EXPECTED_CALIBRATION_BINS = 15
EXPECTED_GRADIENT_ACCUMULATION = 1
EXPECTED_TRAIN = 2757
EXPECTED_VALIDATION = 617
EXPECTED_POPULATION = EXPECTED_TRAIN + EXPECTED_VALIDATION
SMOKE_STEPS = 4
SMOKE_SAMPLES = EXPECTED_BATCH_SIZE * SMOKE_STEPS
SMOKE_MAX_VRAM_GIB = 21.5
SMOKE_SAFETY_MARGIN_GIB = 2.5
DEFAULT_SEEDS = (20260825, 20260826, 20260827)
ALLOWED_ABLATIONS = {"standard", "depth2", "label_smoothing", "class_balanced"}
ALLOWED_CHANGED_VARIABLES = {
    "standard": "physical_batch_size",
    "depth2": "probe_depth",
    "label_smoothing": "label_smoothing",
    "class_balanced": "class_balanced_loss",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
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


def parse_shape(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return [int(item) for item in value]


@dataclass(frozen=True)
class Experiment:
    core: CoreAConfig
    config_path: Path
    config_sha256: str
    raw: dict[str, Any]
    variant: str
    ablation: str
    single_changed_variable: str
    parent_config_path: Path
    feature_run: Path
    output_root: Path
    physical_gpu: int
    max_epochs: int
    min_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_heads: int
    probe_depth: int
    calibration_bins: int
    label_smoothing: float
    loss_name: str
    class_weight_path: Path | None
    class_weight_sha256: str | None
    class_weights: tuple[float, ...] | None
    num_workers: int
    preload_tokens: bool
    use_bfloat16: bool
    seeds: tuple[int, ...]

    @property
    def policy(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "single_changed_variable": self.single_changed_variable,
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "selection_metric": "validation_macro_f1",
            "tie_break_1": "lower_validation_nll",
            "tie_break_2": "earlier_epoch",
            "batch_size": self.batch_size,
            "physical_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "probe_depth": self.probe_depth,
            "num_heads": self.num_heads,
            "calibration_bins": self.calibration_bins,
            "label_smoothing": self.label_smoothing,
            "loss_name": self.loss_name,
            "class_weight_path": str(self.class_weight_path) if self.class_weight_path else None,
            "class_weight_sha256": self.class_weight_sha256,
            "class_weight_formula": "w_c = sqrt(n_max / n_c); normalized by arithmetic mean to 1" if self.class_weights else None,
        }


def load_experiment(path: Path) -> Experiment:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"experiment config must be a mapping: {path}")
    core = load_config(path)
    variant = str(raw.get("variant", ""))
    if variant not in {"dynamic", "static_repeated_first_frame"}:
        raise ValueError(f"unsupported variant: {variant}")
    ablation = str(raw.get("ablation", ""))
    if ablation not in ALLOWED_ABLATIONS:
        raise ValueError(f"unsupported ablation: {ablation}")
    single_changed_variable = str(raw.get("single_changed_variable", ""))
    if single_changed_variable != ALLOWED_CHANGED_VARIABLES[ablation]:
        raise ValueError(f"single_changed_variable mismatch for {ablation}")
    if core.allow_final_test_label_metrics or raw.get("final_test_allowed", False):
        raise ValueError("batch48 runner requires final-test access disabled")
    seeds = tuple(int(seed) for seed in raw.get("seeds", DEFAULT_SEEDS))
    parent_config_path = expand_path(raw["parent_config_path"]).resolve()
    if not parent_config_path.is_file():
        raise FileNotFoundError(parent_config_path)
    label_smoothing = float(raw.get("label_smoothing", 0.0))
    loss_name = str(raw.get("loss_name", "cross_entropy"))
    class_weight_path_value = raw.get("class_weight_path")
    class_weight_path = expand_path(str(class_weight_path_value)).resolve() if class_weight_path_value else None
    class_weight_sha256 = str(raw["class_weight_file_sha256"]) if raw.get("class_weight_file_sha256") else None
    class_weights: tuple[float, ...] | None = None
    if ablation == "class_balanced":
        if class_weight_path is None or class_weight_sha256 is None or not class_weight_path.is_file():
            raise ValueError("class-balanced config must provide an existing weight file and hash")
        if sha256_file(class_weight_path) != class_weight_sha256:
            raise ValueError("class-weight file hash mismatch")
        weight_payload = json.loads(class_weight_path.read_text(encoding="utf-8"))
        if weight_payload.get("status") != "pass" or weight_payload.get("formula") != "w_c = sqrt(n_max / n_c); normalized by arithmetic mean to 1":
            raise ValueError("class-weight provenance/formula mismatch")
        class_weights = tuple(float(value) for value in weight_payload.get("weights", []))
        if len(class_weights) != EXPECTED_CLASS_COUNT or not np.isfinite(class_weights).all() or not np.isclose(np.mean(class_weights), 1.0, atol=1e-6):
            raise ValueError("class weights are not 120 finite values with arithmetic mean 1")
    elif class_weight_path is not None or class_weight_sha256 is not None:
        raise ValueError("non-class-balanced config must not provide class weights")
    experiment = Experiment(
        core=core,
        config_path=path,
        config_sha256=sha256_file(path),
        raw=raw,
        variant=variant,
        ablation=ablation,
        single_changed_variable=single_changed_variable,
        parent_config_path=parent_config_path,
        feature_run=expand_path(raw["feature_run"]).resolve(),
        output_root=expand_path(raw["output_root"]).resolve(),
        physical_gpu=int(raw["physical_gpu"]),
        max_epochs=int(raw.get("max_epochs", EXPECTED_MAX_EPOCHS)),
        min_epochs=int(raw.get("min_epochs", EXPECTED_MIN_EPOCHS)),
        patience=int(raw.get("patience", EXPECTED_PATIENCE)),
        min_delta=float(raw.get("min_delta", EXPECTED_MIN_DELTA)),
        batch_size=int(raw.get("batch_size", EXPECTED_BATCH_SIZE)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        learning_rate=float(raw.get("learning_rate", EXPECTED_LEARNING_RATE)),
        weight_decay=float(raw.get("weight_decay", EXPECTED_WEIGHT_DECAY)),
        num_heads=int(raw.get("num_heads", EXPECTED_NUM_HEADS)),
        probe_depth=int(raw.get("probe_depth", EXPECTED_PROBE_DEPTH)),
        calibration_bins=int(raw.get("calibration_bins", EXPECTED_CALIBRATION_BINS)),
        label_smoothing=label_smoothing,
        loss_name=loss_name,
        class_weight_path=class_weight_path,
        class_weight_sha256=class_weight_sha256,
        class_weights=class_weights,
        num_workers=int(raw.get("num_workers", 0)),
        preload_tokens=bool(raw.get("preload_tokens", True)),
        use_bfloat16=bool(raw.get("use_bfloat16", True)),
        seeds=seeds,
    )
    checks = {
        "max_epochs": (experiment.max_epochs, EXPECTED_MAX_EPOCHS),
        "min_epochs": (experiment.min_epochs, EXPECTED_MIN_EPOCHS),
        "patience": (experiment.patience, EXPECTED_PATIENCE),
        "min_delta": (experiment.min_delta, EXPECTED_MIN_DELTA),
        "batch_size": (experiment.batch_size, EXPECTED_BATCH_SIZE),
        "gradient_accumulation_steps": (
            experiment.gradient_accumulation_steps,
            EXPECTED_GRADIENT_ACCUMULATION,
        ),
        "learning_rate": (experiment.learning_rate, EXPECTED_LEARNING_RATE),
        "weight_decay": (experiment.weight_decay, EXPECTED_WEIGHT_DECAY),
        "num_heads": (experiment.num_heads, EXPECTED_NUM_HEADS),
        "calibration_bins": (experiment.calibration_bins, EXPECTED_CALIBRATION_BINS),
        "num_workers": (experiment.num_workers, 0),
        "preload_tokens": (experiment.preload_tokens, True),
        "use_bfloat16": (experiment.use_bfloat16, True),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"locked batch48 field {name}={observed!r}, expected {expected!r}")
    if not experiment.seeds or not set(experiment.seeds).issubset(DEFAULT_SEEDS) or len(set(experiment.seeds)) != len(experiment.seeds):
        raise ValueError(f"seeds must be a unique non-empty subset of {DEFAULT_SEEDS}")
    if experiment.physical_gpu not in {2, 3}:
        raise ValueError("Batch-48 matrix permits only physical GPU2 or GPU3")
    if experiment.probe_depth not in {2, 4}:
        raise ValueError("probe_depth must be 2 or 4")
    expected_depth = 2 if ablation == "depth2" else 4
    if experiment.probe_depth != expected_depth:
        raise ValueError(f"probe depth does not match ablation {ablation}")
    expected_smoothing = 0.1 if ablation == "label_smoothing" else 0.0
    if not math.isclose(experiment.label_smoothing, expected_smoothing, abs_tol=1e-9):
        raise ValueError(f"label_smoothing does not match ablation {ablation}")
    expected_loss = {
        "standard": "cross_entropy",
        "depth2": "cross_entropy",
        "label_smoothing": "label_smoothing_cross_entropy",
        "class_balanced": "class_balanced_cross_entropy",
    }[ablation]
    if experiment.loss_name != expected_loss:
        raise ValueError(f"loss_name does not match ablation {ablation}")
    if ablation == "standard" and not (experiment.physical_gpu == 2 and set(experiment.seeds) == set(DEFAULT_SEEDS)):
        raise ValueError("standard initial control must use GPU2 with all three seeds")
    if not experiment.feature_run.is_dir():
        raise FileNotFoundError(experiment.feature_run)
    if raw.get("experiment_status") != "preregistered_batch48_matrix":
        raise ValueError("experiment_status is not preregistered_batch48_matrix")
    return experiment


def attach_feature_manifest(
    population_rows: pd.DataFrame, feature_run: Path
) -> tuple[pd.DataFrame, str]:
    manifest_path = feature_run / "feature_manifest.parquet"
    # Deliberately exclude split/task/category columns, including final-test
    # labels, from this read. The v3 population owns the train/validation join.
    manifest = pd.read_parquet(
        manifest_path,
        columns=["episode_index", "feature_path", "feature_sha256", "token_shape", "pooled_shape"],
    )
    if manifest["episode_index"].duplicated().any():
        raise ValueError(f"duplicate episode IDs in {manifest_path}")
    for column, expected in (
        ("token_shape", list(EXPECTED_TOKEN_SHAPE)),
        ("pooled_shape", [768]),
    ):
        if not manifest[column].map(parse_shape).map(lambda value, expected=expected: value == expected).all():
            raise ValueError(f"unexpected {column} in {manifest_path}")
    manifest["resolved_feature_path"] = manifest["feature_path"].map(
        lambda value: str(value) if Path(str(value)).is_absolute() else str(feature_run / str(value))
    )
    if not manifest["resolved_feature_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError(f"feature file missing below {feature_run}")
    joined = population_rows.merge(
        manifest[["episode_index", "feature_path", "feature_sha256", "token_shape", "pooled_shape", "resolved_feature_path"]],
        on="episode_index",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"feature manifest does not cover v3 population: {feature_run}")
    joined = joined.drop(columns=["_merge"])
    return joined.sort_values("episode_index", kind="mergesort").reset_index(drop=True), sha256_file(manifest_path)


def sample_feature_checks(rows: pd.DataFrame) -> list[dict[str, Any]]:
    positions = sorted(set(np.linspace(0, len(rows) - 1, 10, dtype=np.int64).tolist()))
    result = []
    for position in positions:
        row = rows.iloc[int(position)]
        path = Path(str(row.resolved_feature_path))
        record = torch.load(path, map_location="cpu", weights_only=True)
        tokens = record.get("tokens")
        pooled = record.get("pooled")
        if not isinstance(tokens, Tensor) or tuple(tokens.shape) != EXPECTED_TOKEN_SHAPE:
            raise ValueError(f"sample token shape mismatch: {path}")
        if not isinstance(pooled, Tensor) or tuple(pooled.shape) != (768,):
            raise ValueError(f"sample pooled shape mismatch: {path}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
            raise FloatingPointError(f"sample feature is not finite: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != str(row.feature_sha256):
            raise ValueError(f"sample feature hash mismatch: {path}")
        result.append(
            {
                "episode_index": int(row.episode_index),
                "feature_sha256_matches_manifest": True,
                "token_shape": list(tokens.shape),
                "pooled_shape": list(pooled.shape),
            }
        )
    return result


def nvidia_snapshot(physical_gpu: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    for raw_line in completed.stdout.splitlines():
        fields = [item.strip() for item in raw_line.split(",")]
        if fields and int(fields[0]) == physical_gpu:
            return {
                "gpu_index": physical_gpu,
                "gpu_uuid": fields[1],
                "utilization_percent": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "memory_total_mib": int(fields[5]),
            }
    raise RuntimeError(f"nvidia-smi did not report physical GPU {physical_gpu}")


def head_parameter_contract(experiment: Experiment) -> dict[str, Any]:
    head = build_head(experiment, torch.device("cpu"))
    trainable = sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    named_trainable = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    if any(name.startswith("encoder") for name in named_trainable):
        raise AssertionError("attentive head unexpectedly contains trainable encoder parameters")
    batch4_root = experiment.core.project_root / "repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_FULL_ATTENTIVE_DYNAMIC_STATIC_V3_20260828"
    batch4_variant = "dynamic" if experiment.variant == "dynamic" else "static_repeated_first_frame"
    batch4_checkpoint = batch4_root / batch4_variant / "seed_20260825" / "checkpoint_best.pt"
    if not batch4_checkpoint.is_file():
        raise FileNotFoundError(f"batch4 reference checkpoint missing: {batch4_checkpoint}")
    reference = torch.load(batch4_checkpoint, map_location="cpu", weights_only=True)
    reference_state = reference.get("model_state_dict", {})
    reference_count = sum(value.numel() for value in reference_state.values() if isinstance(value, Tensor))
    return {
        "trainable_head_parameter_count": trainable,
        "trainable_backbone_parameter_count": 0,
        "official_depth4_reference_parameter_count": reference_count,
        "batch4_reference_checkpoint": str(batch4_checkpoint),
        "batch4_reference_checkpoint_sha256": sha256_file(batch4_checkpoint),
        "batch4_reference_parameter_count": reference_count,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
    }


def preflight(experiment: Experiment) -> dict[str, Any]:
    population = load_v3_population(experiment.core, full=True)
    selected, manifest_hash = attach_feature_manifest(population.rows, experiment.feature_run)
    split_counts = {key: int(value) for key, value in selected.groupby("split").size().to_dict().items()}
    if split_counts != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}:
        raise ValueError(f"unexpected split counts: {split_counts}")
    if set(selected["local_label_index"].unique()) != set(range(EXPECTED_CLASS_COUNT)):
        raise ValueError("dense label map is not exactly 0..119")
    raw = experiment.raw
    expected_config_hash = str(raw.get("parent_batch32_config_sha256", ""))
    if sha256_file(experiment.parent_config_path) != expected_config_hash:
        raise ValueError("parent Batch-32 config hash mismatch")
    if manifest_hash != str(raw.get("feature_manifest_sha256")):
        raise ValueError("feature manifest hash differs from locked config")
    if population.split_sha256 != str(raw.get("v3_split_sha256")):
        raise ValueError("v3 split hash differs from locked config")
    if population.label_map_sha256 != str(raw.get("label_map_sha256")):
        raise ValueError("label map hash differs from locked config")
    if experiment.class_weight_path is not None:
        weights = json.loads(experiment.class_weight_path.read_text(encoding="utf-8"))
        if (
            weights.get("train_count") != EXPECTED_TRAIN
            or weights.get("class_count") != EXPECTED_CLASS_COUNT
            or weights.get("population_sha256") != population.population_sha256
            or weights.get("split_sha256") != population.split_sha256
            or weights.get("label_map_sha256") != population.label_map_sha256
        ):
            raise ValueError("class-weight provenance does not match v3 train population")
    metadata: dict[str, Any] = {}
    for name in ("model_contract.json", "provenance.json", "resolved_config.json"):
        path = experiment.feature_run / name
        if path.is_file():
            metadata[name] = json.loads(path.read_text(encoding="utf-8"))
    if (
        experiment.variant == "static_repeated_first_frame"
        and metadata.get("model_contract.json", {}).get("input_mode")
        != "repeated_first_frame"
    ):
        raise ValueError("static feature provenance is not repeated_first_frame")
    head_contract = head_parameter_contract(experiment)
    report = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "variant": experiment.variant,
        "ablation": experiment.ablation,
        "single_changed_variable": experiment.single_changed_variable,
        "loss_name": experiment.loss_name,
        "label_smoothing": experiment.label_smoothing,
        "class_weight_path": str(experiment.class_weight_path) if experiment.class_weight_path else None,
        "class_weight_sha256": experiment.class_weight_sha256,
        "config_path": str(experiment.config_path),
        "config_sha256": experiment.config_sha256,
        "feature_run": str(experiment.feature_run),
        "feature_manifest_sha256": manifest_hash,
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "population_rows": len(selected),
        "split_counts": split_counts,
        "class_count": EXPECTED_CLASS_COUNT,
        "dense_label_values": list(range(EXPECTED_CLASS_COUNT)),
        "token_shape": list(EXPECTED_TOKEN_SHAPE),
        "pooled_shape": [768],
        "sample_feature_checks": sample_feature_checks(selected),
        "metadata": metadata,
        "final_test_selected": 0,
        "final_test_access": "none",
        "output_root_empty_before_launch": not experiment.output_root.exists() or not any(experiment.output_root.iterdir()),
        "old_batch4_output_untouched_by_preflight": True,
        "only_changed_variable": "physical_batch_size",
        "physical_batch_size": EXPECTED_BATCH_SIZE,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "head_contract": head_contract,
        "model": {
            "probe_depth": EXPECTED_PROBE_DEPTH,
            "num_heads": EXPECTED_NUM_HEADS,
            "backbone": "frozen V-JEPA 2.1-B/384 ema_encoder final tokens",
        },
        "policy": experiment.policy,
    }
    return report


def finite_gradients(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def build_loss(experiment: Experiment, device: torch.device) -> nn.Module:
    weight = None
    if experiment.class_weights is not None:
        weight = torch.tensor(experiment.class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=experiment.label_smoothing)


def train_epoch_timed(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_bfloat16: bool,
    loss_fn: nn.Module,
) -> dict[str, Any]:
    model.train()
    started = time.monotonic()
    iterator = iter(loader)
    data_wait = 0.0
    total_loss = 0.0
    total = 0
    correct = 0
    top5_correct = 0
    optimizer_steps = 0
    while True:
        wait_started = time.monotonic()
        try:
            tokens, labels, _ = next(iterator)
        except StopIteration:
            break
        data_wait += time.monotonic() - wait_started
        physical_size = int(labels.numel())
        if physical_size <= 0 or physical_size > EXPECTED_BATCH_SIZE:
            raise AssertionError(f"invalid physical batch size {physical_size}")
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_bfloat16):
            logits = model(tokens)
            loss = loss_fn(logits.float(), labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("non-finite training logits/loss")
        loss.backward()
        if not finite_gradients(model):
            raise FloatingPointError("non-finite training gradient")
        optimizer.step()
        optimizer_steps += 1
        total_loss += float(loss.item()) * physical_size
        total += physical_size
        predictions = logits.argmax(dim=1)
        correct += int(predictions.eq(labels).sum().item())
        top5_correct += int(
            logits.topk(5, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().item()
        )
    if total != EXPECTED_TRAIN:
        raise ValueError(f"training exposure count mismatch: {total}")
    elapsed = time.monotonic() - started
    return {
        "loss": total_loss / total,
        "top1": correct / total,
        "top5": top5_correct / total,
        "total": total,
        "optimizer_steps": optimizer_steps,
        "samples_exposed": total,
        "data_loader_wait_seconds": data_wait,
        "wall_time_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0.0,
    }


def evaluate_timed(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, Any]:
    model.eval()
    started = time.monotonic()
    data_wait = 0.0
    logits_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    episode_list: list[Tensor] = []
    with torch.inference_mode():
        iterator = iter(loader)
        while True:
            wait_started = time.monotonic()
            try:
                tokens, labels, episodes = next(iterator)
            except StopIteration:
                break
            data_wait += time.monotonic() - wait_started
            tokens = tokens.to(device, non_blocking=True)
            with autocast_context(device, use_bfloat16):
                logits = model(tokens)
            logits = logits.float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite validation logits")
            logits_list.append(logits)
            labels_list.append(labels.cpu())
            episode_list.append(episodes.cpu())
    if not logits_list:
        raise ValueError("empty validation loader")
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    episodes = torch.cat(episode_list)
    if len(labels) != EXPECTED_VALIDATION or len(torch.unique(episodes)) != EXPECTED_VALIDATION:
        raise ValueError("validation episode count/uniqueness mismatch")
    metrics = extended_classification_metrics(
        logits, labels, EXPECTED_CLASS_COUNT, calibration_bins=EXPECTED_CALIBRATION_BINS
    )
    elapsed = time.monotonic() - started
    return {
        "metrics": metrics,
        "logits": logits.numpy(),
        "labels": labels.numpy(),
        "episode_index": episodes.numpy(),
        "predictions": logits.argmax(dim=1).numpy(),
        "data_loader_wait_seconds": data_wait,
        "wall_time_seconds": elapsed,
    }


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
    "optimizer_steps",
    "samples_exposed",
    "data_loader_wait_seconds",
    "samples_per_second",
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


def write_prediction(seed_dir: Path, name: str, result: dict[str, Any]) -> None:
    atomic_write_npz(
        seed_dir / name,
        episode_index=result["episode_index"],
        labels=result["labels"],
        logits=result["logits"],
        predictions=result["predictions"],
    )


def run_seed(
    experiment: Experiment,
    selected: pd.DataFrame,
    *,
    device: torch.device,
    seed: int,
    feature_manifest_sha256: str,
    population_sha256: str,
) -> dict[str, Any]:
    seed_dir = experiment.output_root / f"seed_{seed}"
    if seed_dir.exists():
        raise FileExistsError(f"refusing to reuse existing seed directory: {seed_dir}")
    seed_dir.mkdir(parents=True)
    seed_contract = {
        "seed": seed,
        "ablation": experiment.ablation,
        "loss_name": experiment.loss_name,
        "label_smoothing": experiment.label_smoothing,
        "config_sha256": experiment.config_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "population_sha256": population_sha256,
        "class_weight_sha256": experiment.class_weight_sha256,
        "policy": experiment.policy,
    }
    atomic_write_json(seed_dir / "seed_contract.json", seed_contract)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_head(experiment, device)
    if any(name.startswith("encoder") and parameter.requires_grad for name, parameter in model.named_parameters()):
        raise AssertionError("backbone parameters are trainable")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=experiment.max_epochs
    )
    loss_fn = build_loss(experiment, device)
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
    last_validation: dict[str, Any] | None = None
    curves: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, experiment.max_epochs + 1):
        epoch_started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_epoch_timed(
            model, train_loader, optimizer, device=device, use_bfloat16=experiment.use_bfloat16, loss_fn=loss_fn
        )
        validation = evaluate_timed(
            model, validation_loader, device=device, use_bfloat16=experiment.use_bfloat16
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
            write_prediction(seed_dir, "predictions_best.npz", validation)
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        epoch_wall = time.monotonic() - epoch_started
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
            "validation_mean_max_softmax_probability": metrics["mean_max_softmax_probability"],
            "validation_correct": metrics["correct"],
            "validation_total": metrics["total"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": train_result["optimizer_steps"],
            "samples_exposed": train_result["samples_exposed"],
            "data_loader_wait_seconds": train_result["data_loader_wait_seconds"] + validation["data_loader_wait_seconds"],
            "samples_per_second": (EXPECTED_TRAIN + EXPECTED_VALIDATION) / epoch_wall if epoch_wall > 0 else 0.0,
            "epoch_wall_time_seconds": epoch_wall,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "best_so_far": improved,
            "bad_epochs": early_state.bad_epochs,
            "host_load_1m": host_load_1m(),
        }
        curves.append(curve)
        write_prediction(seed_dir, f"predictions_epoch_{epoch:03d}.npz", validation)
        atomic_write_csv(seed_dir / "epoch_curves.csv", curves, CURVE_FIELDS)
        atomic_write_torch(
            seed_dir / "checkpoint_current.pt",
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
            f"BATCH48_PROGRESS ablation={experiment.ablation} variant={experiment.variant} seed={seed} epoch={epoch}/{experiment.max_epochs} "
            f"val_macro_f1={float(metrics['macro_f1']):.6f} val_top1={float(metrics['top1']):.6f} "
            f"optimizer_steps={train_result['optimizer_steps']} peak_reserved={peak_reserved}",
            flush=True,
        )
        if epoch >= experiment.min_epochs and early_state.bad_epochs >= experiment.patience:
            early_state.stop_reason = "patience_reached"
            break
    if last_validation is None or best_metrics is None:
        raise RuntimeError("seed did not execute a valid epoch")
    last_epoch = int(curves[-1]["epoch"])
    if early_state.stop_reason is None:
        early_state.stop_reason = "max_epochs_reached"
    write_prediction(seed_dir, "predictions_last.npz", last_validation)
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
    completed = {
        "status": "pass",
        "variant": experiment.variant,
        "ablation": experiment.ablation,
        "single_changed_variable": experiment.single_changed_variable,
        "loss_name": experiment.loss_name,
        "label_smoothing": experiment.label_smoothing,
        "class_weight_sha256": experiment.class_weight_sha256,
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
        "total_optimizer_steps": int(sum(int(row["optimizer_steps"]) for row in curves)),
        "total_samples_exposed": int(sum(int(row["samples_exposed"]) for row in curves)),
        "peak_vram_allocated_bytes": max(int(row["peak_vram_allocated_bytes"]) for row in curves),
        "peak_vram_reserved_bytes": max(int(row["peak_vram_reserved_bytes"]) for row in curves),
        "final_test_access": "none",
        "backbone_trainable_parameter_count": 0,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
    }
    atomic_write_json(seed_dir / "seed_complete.json", completed)
    return completed


def run_experiment(experiment: Experiment, *, device: torch.device) -> dict[str, Any]:
    population = load_v3_population(experiment.core, full=True)
    selected, manifest_hash = attach_feature_manifest(population.rows, experiment.feature_run)
    configured_seeds = tuple(int(seed) for seed in experiment.raw.get("seeds", DEFAULT_SEEDS))
    requested_seeds = tuple(experiment.seeds)
    existing_completed: list[int] = []
    existing_status_path = experiment.output_root / "RUN_STATUS.json"
    if experiment.output_root.exists() and any(experiment.output_root.iterdir()):
        if not existing_status_path.is_file():
            raise FileExistsError(f"non-empty output has no compatible run status: {experiment.output_root}")
        existing_status = json.loads(existing_status_path.read_text(encoding="utf-8"))
        if existing_status.get("config_sha256") != experiment.config_sha256 or existing_status.get("variant") != experiment.variant or existing_status.get("ablation") != experiment.ablation:
            raise ValueError(f"existing output contract mismatch: {experiment.output_root}")
        existing_completed = [int(seed) for seed in existing_status.get("completed_seeds", [])]
        if set(requested_seeds) & set(existing_completed):
            raise FileExistsError(f"requested seed already completed: {set(requested_seeds) & set(existing_completed)}")
    experiment.output_root.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "status": "running",
        "variant": experiment.variant,
        "ablation": experiment.ablation,
        "single_changed_variable": experiment.single_changed_variable,
        "loss_name": experiment.loss_name,
        "label_smoothing": experiment.label_smoothing,
        "class_weight_path": str(experiment.class_weight_path) if experiment.class_weight_path else None,
        "class_weight_sha256": experiment.class_weight_sha256,
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        "cwd": os.getcwd(),
        "device": str(device),
        "physical_gpu": experiment.physical_gpu,
        "gpu_snapshot_at_start": nvidia_snapshot(experiment.physical_gpu),
        "config_path": str(experiment.config_path),
        "config_sha256": experiment.config_sha256,
        "feature_run": str(experiment.feature_run),
        "feature_manifest_sha256": manifest_hash,
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "train_count": EXPECTED_TRAIN,
        "validation_count": EXPECTED_VALIDATION,
        "class_count": EXPECTED_CLASS_COUNT,
        "seeds": list(configured_seeds),
        "requested_seeds": list(requested_seeds),
        "completed_seeds": existing_completed,
        "policy": experiment.policy,
        "final_test_selected": 0,
        "final_test_access": "none",
        "backbone_trainable_parameter_count": 0,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "head_trainable_parameter_count": sum(parameter.numel() for parameter in build_head(experiment, torch.device("cpu")).parameters() if parameter.requires_grad),
    }
    atomic_write_json(experiment.output_root / "RUN_STATUS.json", run_contract)
    completed = []
    all_completed = list(existing_completed)
    for seed in requested_seeds:
        result = run_seed(
            experiment,
            selected,
            device=device,
            seed=seed,
            feature_manifest_sha256=manifest_hash,
            population_sha256=population.population_sha256,
        )
        completed.append(result)
        all_completed.append(int(result["seed"]))
        atomic_write_json(
            experiment.output_root / "RUN_STATUS.json",
            {**run_contract, "status": "running", "completed_seeds": sorted(set(all_completed))},
        )
    all_completed = sorted(set(all_completed))
    if set(all_completed) != set(configured_seeds):
        report = {
            **run_contract,
            "status": "running",
            "completed_seeds": all_completed,
            "requested_seed_summaries": completed,
            "final_test_access": "none",
        }
        atomic_write_json(experiment.output_root / "RUN_STATUS.json", report)
        return report
    seed_summaries = []
    for seed in configured_seeds:
        seed_summary_path = experiment.output_root / f"seed_{seed}" / "seed_complete.json"
        seed_summaries.append(json.loads(seed_summary_path.read_text(encoding="utf-8")))
    report = {
        **run_contract,
        "status": "pass",
        "completed_seeds": all_completed,
        "seed_summaries": seed_summaries,
        "gpu_snapshot_at_end": nvidia_snapshot(experiment.physical_gpu),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(experiment.output_root / "RUN_COMPLETE.json", report)
    atomic_write_json(experiment.output_root / "RUN_STATUS.json", report)
    return report


def run_memory_smoke(
    experiment: Experiment,
    *,
    device: torch.device,
    output: Path,
    smoke_steps: int,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("batch48 memory smoke requires CUDA")
    if smoke_steps <= 0:
        raise ValueError("smoke_steps must be positive")
    population = load_v3_population(experiment.core, full=True)
    selected, manifest_hash = attach_feature_manifest(population.rows, experiment.feature_run)
    smoke_samples = EXPECTED_BATCH_SIZE * smoke_steps
    train_rows = selected[selected["split"].eq("train")].head(smoke_samples)
    if len(train_rows) != smoke_samples:
        raise ValueError(f"smoke does not have {smoke_samples} real train rows")
    shared_cache: dict[str, Tensor] = {}
    loader = make_loader(
        train_rows,
        batch_size=EXPECTED_BATCH_SIZE,
        shuffle=False,
        seed=experiment.seeds[0],
        num_workers=0,
        preload=True,
        shared_cache=shared_cache,
        pin_memory=True,
    )
    model = build_head(experiment, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay
    )
    loss_fn = build_loss(experiment, device)
    torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    step_rows = []
    total_wait = 0.0
    started = time.monotonic()
    for step in range(1, smoke_steps + 1):
        wait_started = time.monotonic()
        tokens, labels, _ = next(iterator)
        wait = time.monotonic() - wait_started
        total_wait += wait
        if int(labels.numel()) != EXPECTED_BATCH_SIZE:
            raise AssertionError(f"smoke physical batch is {labels.numel()}, not 48")
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        with autocast_context(device, experiment.use_bfloat16):
            logits = model(tokens)
            loss = loss_fn(logits.float(), labels)
        finite_logits = bool(torch.isfinite(logits).all())
        finite_loss = bool(torch.isfinite(loss))
        if not finite_logits or not finite_loss:
            raise FloatingPointError("smoke logits/loss is not finite")
        loss.backward()
        finite_grad = finite_gradients(model)
        if not finite_grad:
            raise FloatingPointError("smoke gradient is not finite")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_rows.append(
            {
                "step": step,
                "physical_batch_size": int(labels.numel()),
                "loss": float(loss.item()),
                "finite_logits": finite_logits,
                "finite_loss": finite_loss,
                "finite_gradients": finite_grad,
                "step_time_seconds": time.monotonic() - step_started,
                "data_loader_wait_seconds": wait,
            }
        )
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    gpu = nvidia_snapshot(experiment.physical_gpu)
    peak_reserved_gib = peak_reserved / (1024**3)
    safety_limit = min(SMOKE_MAX_VRAM_GIB, gpu["memory_total_mib"] / 1024 - SMOKE_SAFETY_MARGIN_GIB) * (1024**3)
    passed = bool(
        len(step_rows) == smoke_steps
        and all(row["physical_batch_size"] == EXPECTED_BATCH_SIZE for row in step_rows)
        and all(row["finite_logits"] and row["finite_loss"] and row["finite_gradients"] for row in step_rows)
        and peak_reserved <= safety_limit
        and gpu["memory_used_mib"] <= SMOKE_MAX_VRAM_GIB * 1024
    )
    report = {
        "status": "pass" if passed else "batch48_not_feasible",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "variant": experiment.variant,
        "ablation": experiment.ablation,
        "loss_name": experiment.loss_name,
        "label_smoothing": experiment.label_smoothing,
        "config_path": str(experiment.config_path),
        "config_sha256": experiment.config_sha256,
        "feature_manifest_sha256": manifest_hash,
        "population": {"source": "train_only", "rows": SMOKE_SAMPLES, "final_test_selected": 0},
        "physical_gpu": experiment.physical_gpu,
        "gpu": gpu,
        "device": str(device),
        "torch_version": torch.__version__,
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        "pid": os.getpid(),
        "physical_batch_size": EXPECTED_BATCH_SIZE,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "smoke_steps": smoke_steps,
        "optimizer_steps": len(step_rows),
        "sample_exposures": smoke_samples,
        "step_rows": step_rows,
        "elapsed_seconds": time.monotonic() - started,
        "samples_per_second": smoke_samples / (time.monotonic() - started),
        "data_loader_wait_seconds": total_wait,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "peak_vram_reserved_gib": peak_reserved_gib,
        "safety_limit_bytes": int(safety_limit),
        "cpu_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "head_trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "backbone_trainable_parameter_count": 0,
        "formal_result_saved": False,
        "cuda_error": None,
        "final_test_access": "none",
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    del model, optimizer, loader, shared_cache
    torch.cuda.empty_cache()
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--memory-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    parser.add_argument("--only-seed", type=int)
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    if args.only_seed is not None:
        if args.only_seed not in experiment.seeds:
            raise ValueError(f"--only-seed {args.only_seed} is not present in config seeds")
        experiment = replace(experiment, seeds=(args.only_seed,))
    if experiment.physical_gpu != args.physical_gpu:
        raise ValueError("physical GPU does not match frozen config")
    if args.preflight_only:
        report = preflight(experiment)
        if args.output is not None:
            output = args.output.resolve()
            if output.exists():
                raise FileExistsError(output)
            atomic_write_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    device = torch.device(args.device)
    if args.memory_smoke:
        if args.output is None:
            raise ValueError("--output is required for memory smoke")
        report = run_memory_smoke(
            experiment,
            device=device,
            output=args.output,
            smoke_steps=args.smoke_steps,
        )
        return 0 if report["status"] == "pass" else 3
    if args.output is not None:
        raise ValueError("--output is reserved for preflight or memory smoke")
    if args.smoke_report is None or not args.smoke_report.is_file():
        raise ValueError("a passing --smoke-report is required before full training")
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    if (
        smoke.get("status") != "pass"
        or smoke.get("physical_batch_size") != EXPECTED_BATCH_SIZE
        or smoke.get("config_sha256") != experiment.config_sha256
        or smoke.get("ablation") != experiment.ablation
        or smoke.get("variant") != experiment.variant
    ):
        raise ValueError("smoke report is not a passing matching physical-batch-48 report")
    report = run_experiment(experiment, device=device)
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
