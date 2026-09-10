"""Run the isolated raw-video top-2-block V-JEPA LoRA feasibility pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ee6008.config import CoreAConfig, load_config
from ee6008.data.clips import decode_prefix, preprocess_vjepa
from ee6008.data.selection import benchmark_primary_mask
from ee6008.lora import (
    LoRAConfig,
    adapter_state_dict,
    assert_frozen_boundary,
    base_named_parameters,
    inject_top_block_lora,
    lora_named_parameters,
    parameter_counts,
    trainable_named_parameters,
)
from ee6008.metrics import classification_metrics
from ee6008.models import load_vjepa2_1_base
from ee6008.top_block_finetune import describe_encoder, forward_top_two_lora

DEFAULT_SEEDS = (20260825, 20260826, 20260827)
EXPECTED_PILOT_TASKS = 12
EFFECTIVE_BATCH_SIZE = 4
HEAD_LR = 1e-3
LORA_LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.part")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    np.savez_compressed(temporary, **arrays)
    actual_temporary = temporary.with_suffix(temporary.suffix + ".npz")
    os.replace(actual_temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_digest(named_parameters: Iterator[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finite_tensor(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"unsupported precision: {precision}")


def make_scaler(device: torch.device, precision: str):
    return torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and precision == "fp16"
    )


@dataclass(frozen=True)
class PilotData:
    config: CoreAConfig
    rows: pd.DataFrame
    task_ids: tuple[int, ...]
    local_labels: dict[int, int]
    full_labels: dict[int, int]
    selection: dict[str, Any]
    split_counts: dict[str, int]
    split_sha256: str
    label_map_sha256: str
    selection_sha256: str
    feature_manifest_sha256: str


def _load_full_label_map(split_table: pd.DataFrame) -> dict[int, int]:
    """Read the v3 labels without assuming excluded tasks are renumbered.

    The duplicate-safe v3 manifest retains holes in the historical
    ``contiguous_label_index`` values for excluded/conflicting tasks.  The
    pilot uses its own dense 12-task labels, so preserving those recorded
    values is the auditable choice here.
    """
    primary = split_table[benchmark_primary_mask(split_table)]
    pairs = primary[["original_task_index", "contiguous_label_index"]].drop_duplicates()
    if pairs["original_task_index"].duplicated().any():
        raise RuntimeError("v3 label map has duplicate task entries")
    result = {
        int(row.original_task_index): int(row.contiguous_label_index)
        for row in pairs.itertuples(index=False)
    }
    values = list(result.values())
    if (
        len(result) != 120
        or len(set(values)) != len(values)
        or any(value < 0 for value in values)
    ):
        raise RuntimeError("v3 primary label map is not unique and non-negative")
    return result


def load_pilot_data(config: CoreAConfig, selection_path: Path, feature_run: Path) -> PilotData:
    split_path = config.manifest_dir / "splits.parquet"
    label_map_path = config.manifest_dir / "label_map.json"
    if not split_path.is_file() or not label_map_path.is_file():
        raise FileNotFoundError("v3 split or label map is missing")
    split_table = pd.read_parquet(split_path)
    required = {
        "episode_index",
        "original_task_index",
        "split",
        "video_path",
        "video_sha256_fingerprint",
        "validation_status",
        "v3_benchmark_inclusion",
        "contiguous_label_index",
    }
    missing = sorted(required - set(split_table.columns))
    if missing:
        raise RuntimeError(f"v3 split missing required columns: {missing}")

    primary = split_table[benchmark_primary_mask(split_table)].copy()
    primary_counts = {
        split: int((primary["split"] == split).sum())
        for split in ("train", "validation", "final_test")
    }
    if primary_counts != {"train": 2757, "validation": 617, "final_test": 602}:
        raise RuntimeError(f"unexpected v3 primary counts: {primary_counts}")
    full_labels = _load_full_label_map(split_table)
    if len(full_labels) != 120:
        raise RuntimeError(f"unexpected v3 primary class count: {len(full_labels)}")
    if len(primary) != 3976:
        raise RuntimeError(f"unexpected v3 primary population: {len(primary)}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    task_ids = tuple(int(value) for value in selection.get("pilot_tasks", ()))
    if len(task_ids) != EXPECTED_PILOT_TASKS or len(set(task_ids)) != EXPECTED_PILOT_TASKS:
        raise RuntimeError(f"pilot selection is not exactly 12 unique tasks: {task_ids}")
    if set(task_ids) - set(full_labels):
        raise RuntimeError("pilot selection contains a non-primary task")
    if {84, 85}.intersection(task_ids):
        raise RuntimeError("conflicting duplicate task 84/85 reached pilot selection")
    local_labels = {task_id: index for index, task_id in enumerate(task_ids)}

    rows = primary[
        primary["original_task_index"].isin(task_ids)
        & primary["split"].isin(("train", "validation"))
        & primary["validation_status"].eq("valid")
    ].copy()
    if rows.empty or set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("pilot has no complete train/validation population")
    if rows["episode_index"].duplicated().any():
        raise RuntimeError("pilot rows contain duplicate episode IDs")
    train_ids = set(rows.loc[rows["split"].eq("train"), "episode_index"])
    val_ids = set(rows.loc[rows["split"].eq("validation"), "episode_index"])
    if train_ids.intersection(val_ids):
        raise RuntimeError("pilot train/validation episode overlap")
    train_fingerprints = set(
        rows.loc[rows["split"].eq("train"), "video_sha256_fingerprint"]
    )
    val_fingerprints = set(
        rows.loc[rows["split"].eq("validation"), "video_sha256_fingerprint"]
    )
    if train_fingerprints.intersection(val_fingerprints):
        raise RuntimeError("pilot train/validation video hash overlap")
    if not rows["v3_benchmark_inclusion"].eq("primary").all():
        raise RuntimeError("non-primary row reached pilot")
    if rows["split"].eq("final_test").any():
        raise RuntimeError("final-test row reached pilot")
    rows["local_label"] = rows["original_task_index"].map(local_labels).astype(int)
    rows = rows.sort_values("episode_index", kind="mergesort").reset_index(drop=True)

    feature_manifest = feature_run / "feature_manifest.parquet"
    if not feature_manifest.is_file():
        raise FileNotFoundError(f"feature manifest missing: {feature_manifest}")
    feature_table = pd.read_parquet(feature_manifest)
    if not {"episode_index", "feature_path"}.issubset(feature_table.columns):
        raise RuntimeError("feature manifest lacks episode_index/feature_path")
    feature_rows = feature_table[feature_table["episode_index"].isin(rows["episode_index"])]
    if len(feature_rows) != len(rows) or feature_rows["episode_index"].duplicated().any():
        raise RuntimeError("pilot feature cache does not cover rows one-to-one")

    duplicate_csv = config.project_root / "repo/ee6008-humanoid-vjepa/reports/data_audit/EXACT_DUPLICATE_GROUPS_20260826.csv"
    duplicate_table = pd.read_csv(duplicate_csv)
    if (
        duplicate_table["duplicate_group_id"].nunique() != 40
        or len(duplicate_table) != 80
        or not duplicate_table["original_task_index"].isin((84, 85)).all()
        or not (~duplicate_table["label_consistent"]).all()
    ):
        raise RuntimeError("exact duplicate exclusion evidence is inconsistent")

    return PilotData(
        config=config,
        rows=rows,
        task_ids=task_ids,
        local_labels=local_labels,
        full_labels=full_labels,
        selection=selection,
        split_counts=primary_counts,
        split_sha256=sha256_file(split_path),
        label_map_sha256=sha256_file(label_map_path),
        selection_sha256=sha256_file(selection_path),
        feature_manifest_sha256=sha256_file(feature_manifest),
    )


class RawVideoDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Deterministic 4-second RGB prefix dataset; never selects final-test."""

    def __init__(self, data: PilotData, rows: pd.DataFrame):
        self.config = data.config
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        row = self.rows.iloc[index]
        video_path = self.config.g1_root / str(row.video_path)
        sample = decode_prefix(
            video_path,
            duration_seconds=self.config.observation_seconds,
            sample_fps=self.config.sample_fps,
            expected_source_fps=self.config.source_fps,
        )
        clip = preprocess_vjepa(sample.frames, resolution=self.config.input_resolution)
        if tuple(clip.shape) != (3, 32, 384, 384):
            raise RuntimeError(f"unexpected preprocessed clip shape: {tuple(clip.shape)}")
        return (
            clip,
            torch.tensor(int(row.local_label), dtype=torch.long),
            torch.tensor(int(row.episode_index), dtype=torch.long),
        )


def _worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    data: PilotData,
    rows: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _worker_init,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(RawVideoDataset(data, rows), **kwargs)


def load_head(source_root: Path, num_classes: int, device: torch.device) -> nn.Module:
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from src.models.attentive_pooler import AttentiveClassifier

    head = AttentiveClassifier(
        embed_dim=768,
        num_heads=16,
        depth=4,
        num_classes=num_classes,
        use_activation_checkpointing=True,
    ).to(device)
    return head


def build_lora_pair(
    data: PilotData,
    device: torch.device,
    lora_config: LoRAConfig,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    encoder, encoder_info = load_vjepa2_1_base(
        source_root=data.config.vjepa_source_root,
        checkpoint=data.config.checkpoint,
        device=device,
        num_frames=data.config.frame_count,
        resolution=data.config.input_resolution,
    )
    inserted = inject_top_block_lora(encoder, config=lora_config)
    head = load_head(data.config.vjepa_source_root, len(data.task_ids), device)
    assert_frozen_boundary(encoder, head)
    info = {
        "encoder_info": encoder_info,
        "inserted_modules": inserted,
        "lora_config": lora_config.to_dict(),
        "parameter_counts": parameter_counts(encoder, head),
        "structure": describe_encoder(encoder),
    }
    return encoder, head, info


def optimizer_for(encoder: nn.Module, head: nn.Module) -> torch.optim.Optimizer:
    lora_parameters = [parameter for _, parameter in lora_named_parameters(encoder)]
    head_parameters = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": LORA_LR},
            {"params": head_parameters, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed_ids = {id(parameter) for parameter in lora_parameters + head_parameters}
    if optimizer_ids != allowed_ids:
        raise AssertionError("optimizer membership differs from LoRA+head boundary")
    return optimizer


def move_batch(
    batch: tuple[Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    clips, labels, episodes = batch
    return (
        clips.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        episodes,
    )


def forward_logits(
    encoder: nn.Module,
    head: nn.Module,
    clips: Tensor,
    *,
    checkpoint_top_blocks: bool,
) -> Tensor:
    tokens = forward_top_two_lora(
        encoder, clips, checkpoint_top_blocks=checkpoint_top_blocks
    )
    logits = head(tokens)
    if logits.ndim != 2 or logits.shape[1] != 12:
        raise RuntimeError(f"unexpected classifier output shape: {tuple(logits.shape)}")
    return logits


def validate_logits(logits: Tensor, name: str) -> None:
    if not finite_tensor(logits):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def gradient_norm(parameters: list[nn.Parameter]) -> float:
    values = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not values:
        return 0.0
    if not all(finite_tensor(value) for value in values):
        raise FloatingPointError("gradient contains NaN or Inf")
    return float(torch.linalg.vector_norm(torch.cat([value.float().flatten() for value in values])).item())


def train_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    precision: str,
    accumulation: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    encoder.train()
    head.train()
    trainable = [parameter for _, parameter in trainable_named_parameters(encoder)] + list(head.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss_values: list[float] = []
    logits_values: list[Tensor] = []
    labels_values: list[Tensor] = []
    grad_norms: list[float] = []
    batch_count = 0
    pending = 0
    for batch_index, batch in enumerate(loader, 1):
        if max_steps is not None and batch_index > max_steps * accumulation:
            break
        clips, labels, _ = move_batch(batch, device)
        with autocast_context(device, precision):
            logits = forward_logits(
                encoder, head, clips, checkpoint_top_blocks=True
            )
            loss = F.cross_entropy(logits.float(), labels)
        validate_logits(logits, "training logits")
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("training loss is NaN or Inf")
        scaler.scale(loss / accumulation).backward()
        pending += 1
        batch_count += 1
        loss_values.append(float(loss.detach().item()))
        logits_values.append(logits.detach().float().cpu())
        labels_values.append(labels.detach().cpu())
        should_step = pending == accumulation or batch_index == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item())
            if not math.isfinite(norm):
                raise FloatingPointError("gradient norm is NaN or Inf")
            grad_norms.append(norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    if not loss_values:
        raise RuntimeError("empty training loader")
    train_logits = torch.cat(logits_values)
    train_labels = torch.cat(labels_values)
    metrics = classification_metrics(train_logits, train_labels, num_classes=12)
    return {
        "train_loss": float(np.mean(loss_values)),
        "train_top1": metrics["top1"],
        "train_top5": metrics["top5"],
        "batch_count": batch_count,
        "optimizer_steps": len(grad_norms),
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
    }


@torch.inference_mode()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str,
) -> dict[str, Any]:
    encoder.eval()
    head.eval()
    logits_values: list[Tensor] = []
    labels_values: list[Tensor] = []
    episode_values: list[Tensor] = []
    losses: list[float] = []
    for batch in loader:
        clips, labels, episodes = move_batch(batch, device)
        with autocast_context(device, precision):
            logits = forward_logits(
                encoder, head, clips, checkpoint_top_blocks=False
            )
            loss = F.cross_entropy(logits.float(), labels)
        validate_logits(logits, "validation logits")
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("validation loss is NaN or Inf")
        logits_values.append(logits.float().cpu())
        labels_values.append(labels.cpu())
        episode_values.append(episodes.cpu())
        losses.append(float(loss.item()))
    logits = torch.cat(logits_values)
    labels = torch.cat(labels_values)
    episodes = torch.cat(episode_values)
    if episodes.unique().numel() != episodes.numel():
        raise RuntimeError("validation episode IDs are not unique")
    metrics = classification_metrics(logits, labels, num_classes=12)
    return {
        "loss": float(np.mean(losses)),
        "metrics": metrics,
        "logits": logits.numpy(),
        "labels": labels.numpy(),
        "episode_index": episodes.numpy(),
    }


def host_telemetry() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        result["load_average_1m"] = os.getloadavg()[0]
    except OSError:
        result["load_average_1m"] = None
    try:
        memory = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
        result["mem_available_bytes"] = memory.get("MemAvailable")
        result["mem_total_bytes"] = memory.get("MemTotal")
    except (OSError, ValueError):
        result["mem_available_bytes"] = None
        result["mem_total_bytes"] = None
    return result


def gpu_telemetry() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                "3",
                "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        parts = [part.strip() for part in output.split(",")]
        return {
            "uuid": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_total_mib": int(parts[2]),
            "utilization_percent": int(parts[3]),
        }
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return {"status": "unavailable"}


def compare_tensors(left: Tensor, right: Tensor) -> dict[str, Any]:
    left = left.float()
    right = right.float()
    if left.shape != right.shape:
        return {
            "shape_left": list(left.shape),
            "shape_right": list(right.shape),
            "finite_left": finite_tensor(left),
            "finite_right": finite_tensor(right),
            "shape_equal": False,
        }
    difference = (left - right).abs()
    cosine = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    return {
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "shape_equal": True,
        "finite_left": finite_tensor(left),
        "finite_right": finite_tensor(right),
        "cosine_similarity": float(cosine),
        "mean_absolute_difference": float(difference.mean().item()),
        "max_absolute_difference": float(difference.max().item()),
    }


def cache_token_path(feature_run: Path, episode_index: int) -> Path:
    feature_table = pd.read_parquet(feature_run / "feature_manifest.parquet")
    match = feature_table[feature_table["episode_index"].eq(episode_index)]
    if len(match) != 1:
        raise RuntimeError(f"feature cache has {len(match)} entries for episode {episode_index}")
    path = Path(str(match.iloc[0].feature_path))
    return path if path.is_absolute() else feature_run / path


def preflight(data: PilotData, feature_run: Path, output: Path, device: torch.device) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    sample_rows = data.rows.head(5)
    dataset = RawVideoDataset(data, sample_rows)
    clips = torch.stack([dataset[index][0] for index in range(len(dataset))]).to(device)
    model, head, info = build_lora_pair(data, device, LoRAConfig())
    model.eval()
    head.eval()

    # Build an independent base model and keep its official output before
    # adapter insertion; the adapted model is then checked against it.
    base_model, _ = load_vjepa2_1_base(
        source_root=data.config.vjepa_source_root,
        checkpoint=data.config.checkpoint,
        device=device,
        num_frames=data.config.frame_count,
        resolution=data.config.input_resolution,
    )
    base_model.eval()
    with torch.inference_mode():
        official_tokens = base_model(clips)
        official_logits = head(official_tokens)
        split_tokens = forward_top_two_lora(model, clips, checkpoint_top_blocks=False)
        zero_lora_logits = head(split_tokens)
    validate_logits(official_tokens, "official tokens")
    validate_logits(split_tokens, "zero-init LoRA tokens")
    validate_logits(official_logits, "official logits")
    validate_logits(zero_lora_logits, "zero-init LoRA logits")

    official_vs_split = compare_tensors(official_tokens, split_tokens)
    logits_equivalence = compare_tensors(official_logits, zero_lora_logits)
    cache_comparisons = []
    for row_index, row in sample_rows.reset_index(drop=True).iterrows():
        record = torch.load(
            cache_token_path(feature_run, int(row.episode_index)),
            map_location="cpu",
            weights_only=True,
        )
        cache_tokens = record.get("tokens")
        if not isinstance(cache_tokens, Tensor):
            raise TypeError("cache record has no tensor tokens")
        cache_comparisons.append(
            {
                "episode_index": int(row.episode_index),
                "cache_vs_zero_lora": compare_tensors(split_tokens[row_index].cpu(), cache_tokens),
            }
        )

    base_model_structure = describe_encoder(base_model)
    adapter_structure = describe_encoder(model)
    zero_initialized = all(
        torch.count_nonzero(parameter).item() == 0
        for name, parameter in lora_named_parameters(model)
        if name.endswith("lora_B")
    )
    report = {
        "status": "pass",
        "phase": "preflight",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "device": str(device),
        "data_gate": data_gate_report(data),
        "model_structure_before_adapter": base_model_structure,
        "model_structure_after_adapter": adapter_structure,
        "lora": info,
        "official_vs_split": official_vs_split,
        "official_logits_vs_zero_lora_logits": logits_equivalence,
        "zero_init_B_matrices": zero_initialized,
        "cache_comparisons": cache_comparisons,
        "checkpoint_sha256": sha256_file(data.config.checkpoint),
        "preprocessing_source_sha256": sha256_file(data.config.project_root / "repo/ee6008-humanoid-vjepa/src/ee6008/data/clips.py"),
        "final_test_access": "none",
    }
    gate_pass = (
        official_vs_split.get("shape_equal", False)
        and official_vs_split.get("max_absolute_difference", float("inf")) <= 1e-5
        and logits_equivalence.get("shape_equal", False)
        and logits_equivalence.get("max_absolute_difference", float("inf")) <= 1e-5
        and zero_initialized
        and all(
            item["cache_vs_zero_lora"].get("cosine_similarity", 0.0) >= 0.99999
            for item in cache_comparisons
        )
    )
    report["gate_pass"] = gate_pass
    atomic_write_json(output / "PREFLIGHT_REPORT.json", report)
    atomic_write_text(output / "PREFLIGHT_REPORT.md", preflight_markdown(report))
    del base_model, model, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not gate_pass:
        raise RuntimeError(f"LoRA preflight gate failed; see {output / 'PREFLIGHT_REPORT.json'}")
    return report


def data_gate_report(data: PilotData) -> dict[str, Any]:
    return {
        "v3_primary_split_counts": data.split_counts,
        "v3_primary_class_count": len(data.full_labels),
        "pilot_task_ids": list(data.task_ids),
        "pilot_local_label_map": data.local_labels,
        "pilot_full_label_map": {str(task): data.full_labels[task] for task in data.task_ids},
        "pilot_train_count": int((data.rows["split"] == "train").sum()),
        "pilot_validation_count": int((data.rows["split"] == "validation").sum()),
        "train_plus_validation_count": len(data.rows),
        "split_sha256": data.split_sha256,
        "label_map_sha256": data.label_map_sha256,
        "selection_sha256": data.selection_sha256,
        "feature_manifest_sha256": data.feature_manifest_sha256,
        "final_test_selected": False,
        "final_test_metrics_computed": False,
        "exact_duplicate_groups": 40,
        "exact_duplicate_episodes": 80,
    }


def preflight_markdown(report: dict[str, Any]) -> str:
    official = report["official_vs_split"]
    logits = report["official_logits_vs_zero_lora_logits"]
    lines = [
        "# V-JEPA top-2 LoRA preflight",
        "",
        f"Status: `{report['status']}`; gate_pass=`{report['gate_pass']}`",
        "",
        f"Official full forward vs split forward max abs: `{official.get('max_absolute_difference')}`",
        f"Official logits vs zero-init LoRA logits max abs: `{logits.get('max_absolute_difference')}`",
        f"Zero-init B matrices: `{report['zero_init_B_matrices']}`",
        "",
        "Cache cosine checks:",
        "",
    ]
    for item in report["cache_comparisons"]:
        lines.append(
            f"- episode {item['episode_index']}: "
            f"cosine={item['cache_vs_zero_lora'].get('cosine_similarity')}"
        )
    lines.extend(["", "No final-test rows, labels, predictions, or metrics were accessed."])
    return "\n".join(lines) + "\n"


def memory_smoke(
    data: PilotData,
    output: Path,
    device: torch.device,
    *,
    precision: str,
    num_workers: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    candidates = (4, 2, 1)
    candidate_reports: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    smoke_rows = data.rows[data.rows["split"].eq("train")].head(20)
    for micro_batch in candidates:
        accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
        record: dict[str, Any] = {
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "effective_batch": micro_batch * accumulation,
            "status": "fail",
        }
        try:
            encoder, head, _ = build_lora_pair(data, device, LoRAConfig())
            optimizer = optimizer_for(encoder, head)
            scaler = make_scaler(device, precision)
            loader = make_loader(
                data,
                smoke_rows,
                batch_size=micro_batch,
                shuffle=False,
                seed=20260825,
                num_workers=0,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = train_epoch(
                encoder,
                head,
                loader,
                optimizer,
                scaler,
                device,
                precision=precision,
                accumulation=accumulation,
                max_steps=5,
            )
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            )
            total_memory = (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda"
                else 0
            )
            record.update(
                {
                    "status": "pass",
                    "training": result,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "total_memory_bytes": total_memory,
                    "free_margin_bytes": total_memory - peak_reserved,
                    "telemetry": gpu_telemetry(),
                }
            )
            if (
                selected is None
                and peak_allocated <= 22 * 1024**3
                and total_memory - peak_reserved >= 2 * 1024**3
            ):
                selected = record.copy()
            del encoder, head, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            record["error"] = str(exc)
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            candidate_reports.append(record)

    report = {
        "status": "pass" if selected is not None else "fail",
        "phase": "memory_smoke",
        "precision": precision,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "candidates": candidate_reports,
        "selected": selected,
        "data_gate": data_gate_report(data),
        "final_test_access": "none",
    }
    atomic_write_json(output / "MEMORY_SMOKE.json", report)
    if selected is None:
        raise RuntimeError(f"no safe micro batch passed memory smoke; see {output}")
    return report


def choose_overfit_rows(data: PilotData) -> pd.DataFrame:
    smoke_tasks = tuple(int(value) for value in data.selection["smoke_tasks"])
    parts = []
    for task_id in smoke_tasks:
        task_rows = data.rows[
            data.rows["split"].eq("train")
            & data.rows["original_task_index"].eq(task_id)
        ].head(8)
        parts.append(task_rows)
    result = pd.concat(parts, ignore_index=True)
    if len(result) > 16 or len(result) == 0:
        raise RuntimeError("2-task overfit subset violates the <=16 episode gate")
    return result


def overfit_smoke(
    data: PilotData,
    output: Path,
    device: torch.device,
    *,
    precision: str,
    micro_batch: int,
    accumulation: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    rows = choose_overfit_rows(data)
    encoder, head, info = build_lora_pair(data, device, LoRAConfig())
    optimizer = optimizer_for(encoder, head)
    scaler = make_scaler(device, precision)
    loader = make_loader(
        data,
        rows,
        batch_size=micro_batch,
        shuffle=True,
        seed=20260825,
        num_workers=0,
    )
    initial_eval = evaluate(encoder, head, make_loader(data, rows, batch_size=micro_batch, shuffle=False, seed=20260825, num_workers=0), device, precision=precision)
    initial_loss = initial_eval["loss"]
    curves = []
    optimizer_steps = 0
    while optimizer_steps < 200:
        result = train_epoch(
            encoder,
            head,
            loader,
            optimizer,
            scaler,
            device,
            precision=precision,
            accumulation=accumulation,
            max_steps=min(5, 200 - optimizer_steps),
        )
        optimizer_steps += int(result["optimizer_steps"])
        train_eval = evaluate(encoder, head, make_loader(data, rows, batch_size=micro_batch, shuffle=False, seed=20260825, num_workers=0), device, precision=precision)
        curves.append(
            {
                "optimizer_steps": optimizer_steps,
                "train_loss": train_eval["loss"],
                "train_top1": train_eval["metrics"]["top1"],
                "grad_norm": result["grad_norm"],
            }
        )
        if optimizer_steps >= 200:
            break
    final = curves[-1]
    base_checkpoint_hash = sha256_file(data.config.checkpoint)
    report = {
        "status": "pass" if final["train_top1"] >= 0.95 or final["train_loss"] <= initial_loss * 0.2 else "fail",
        "phase": "2_task_overfit_smoke",
        "task_ids": sorted(int(value) for value in rows["original_task_index"].unique()),
        "episode_count": len(rows),
        "optimizer_steps": optimizer_steps,
        "initial_train_loss": initial_loss,
        "final": final,
        "curves": curves,
        "model": info,
        "base_checkpoint_sha256_before": base_checkpoint_hash,
        "base_checkpoint_sha256_after": sha256_file(data.config.checkpoint),
        "data_gate": data_gate_report(data),
        "final_test_access": "none",
    }
    atomic_write_json(output / "OVERFIT_SMOKE.json", report)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    del encoder, head, optimizer, scaler
    if report["status"] != "pass":
        raise RuntimeError(f"2-task LoRA overfit gate failed; see {output}")
    return report


def write_curves_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def save_periodic_checkpoint(
    path: Path,
    *,
    epoch: int,
    seed: int,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data: PilotData,
    micro_batch: int,
    accumulation: int,
    precision: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "epoch": epoch,
            "seed": seed,
            "adapter_state_dict": adapter_state_dict(encoder),
            "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "task_ids": list(data.task_ids),
                "micro_batch": micro_batch,
                "accumulation": accumulation,
                "effective_batch": micro_batch * accumulation,
                "precision": precision,
                "lora": LoRAConfig().to_dict(),
            },
        },
    )


def pilot(
    data: PilotData,
    output: Path,
    device: torch.device,
    *,
    precision: str,
    micro_batch: int,
    accumulation: int,
    epochs: int,
    seeds: tuple[int, ...],
    num_workers: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pilot output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    if micro_batch * accumulation != EFFECTIVE_BATCH_SIZE:
        raise ValueError("micro_batch * accumulation must equal effective batch size 4")
    if epochs != 20:
        raise ValueError("the fixed exploratory LoRA pilot uses exactly 20 epochs")

    lora_config = LoRAConfig()
    structure_written = False
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        set_seed(seed)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        encoder, head, model_info = build_lora_pair(data, device, lora_config)
        assert_frozen_boundary(encoder, head)
        if not structure_written:
            atomic_write_json(
                output / "model_structure.json",
                {
                    **model_info,
                    "source_commit": data.config.source_commit,
                    "attentive_head": {
                        "implementation": "third_party/vjepa2/src/models/attentive_pooler.py",
                        "depth": 4,
                        "num_heads": 16,
                        "head_lr": HEAD_LR,
                        "lora_lr": LORA_LR,
                    },
                },
            )
            atomic_write_json(
                output / "trainable_parameter_manifest.json",
                {
                    "parameter_counts": parameter_counts(encoder, head),
                    "encoder_lora_parameters": [name for name, _ in lora_named_parameters(encoder)],
                    "encoder_base_trainable_parameters": [name for name, _ in base_named_parameters(encoder) if _.requires_grad],
                    "head_parameters": [name for name, _ in head.named_parameters()],
                },
            )
            structure_written = True

        optimizer = optimizer_for(encoder, head)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = make_scaler(device, precision)
        train_rows = data.rows[data.rows["split"].eq("train")]
        validation_rows = data.rows[data.rows["split"].eq("validation")]
        train_loader = make_loader(
            data,
            train_rows,
            batch_size=micro_batch,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )
        validation_loader = make_loader(
            data,
            validation_rows,
            batch_size=micro_batch,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
        )
        checkpoint_hash_before = sha256_file(data.config.checkpoint)
        base_parameters_before = tensor_digest(base_named_parameters(encoder))
        started = time.monotonic()
        curves: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            epoch_started = time.monotonic()
            train_result = train_epoch(
                encoder,
                head,
                train_loader,
                optimizer,
                scaler,
                device,
                precision=precision,
                accumulation=accumulation,
            )
            validation = evaluate(
                encoder,
                head,
                validation_loader,
                device,
                precision=precision,
            )
            scheduler.step()
            peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            val_metrics = validation["metrics"]
            curve = {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_result["train_loss"],
                "train_top1": train_result["train_top1"],
                "train_top5": train_result["train_top5"],
                "validation_loss": validation["loss"],
                "validation_top1": val_metrics["top1"],
                "validation_top5": val_metrics["top5"],
                "validation_mean_class_recall_at_1": val_metrics["mean_class_recall_at_1"],
                "validation_mean_class_recall_at_5": val_metrics["mean_class_recall_at_5"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "learning_rate_lora": optimizer.param_groups[0]["lr"],
                "learning_rate_head": optimizer.param_groups[1]["lr"],
                "grad_norm": train_result["grad_norm"],
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "epoch_wall_time_seconds": time.monotonic() - epoch_started,
                "host_telemetry": json.dumps(host_telemetry(), sort_keys=True),
                "gpu3_telemetry": json.dumps(gpu_telemetry(), sort_keys=True),
            }
            curves.append(curve)
            write_curves_csv(seed_dir / "epoch_curves.csv", curves)
            print(
                f"LORA_PROGRESS seed={seed} epoch={epoch}/{epochs} "
                f"train_loss={curve['train_loss']:.6f} val_top1={curve['validation_top1']:.6f} "
                f"peak_allocated={peak_allocated}",
                flush=True,
            )
            if epoch % 5 == 0:
                save_periodic_checkpoint(
                    seed_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    epoch=epoch,
                    seed=seed,
                    encoder=encoder,
                    head=head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data=data,
                    micro_batch=micro_batch,
                    accumulation=accumulation,
                    precision=precision,
                )

        final_validation = evaluate(
            encoder,
            head,
            validation_loader,
            device,
            precision=precision,
        )
        checkpoint_hash_after = sha256_file(data.config.checkpoint)
        base_parameters_after = tensor_digest(base_named_parameters(encoder))
        atomic_save_npz(
            seed_dir / f"predictions_seed_{seed}.npz",
            labels=final_validation["labels"],
            logits=final_validation["logits"],
            episode_index=final_validation["episode_index"],
        )
        atomic_torch_save(
            seed_dir / "adapter_final.pt",
            {"seed": seed, "adapter_state_dict": adapter_state_dict(encoder)},
        )
        atomic_torch_save(
            seed_dir / "head_final.pt",
            {"seed": seed, "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()}},
        )
        completed = {
            "seed": seed,
            "epochs_completed": epochs,
            "final_metrics": final_validation["metrics"],
            "best_validation_epoch": int(max(curves, key=lambda row: row["validation_top1"])["epoch"]),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_file_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "base_parameters_digest_before": base_parameters_before,
            "base_parameters_digest_after": base_parameters_after,
            "base_parameters_unchanged": base_parameters_before == base_parameters_after,
            "prediction_sha256": sha256_file(seed_dir / f"predictions_seed_{seed}.npz"),
            "adapter_sha256": sha256_file(seed_dir / "adapter_final.pt"),
            "head_sha256": sha256_file(seed_dir / "head_final.pt"),
            "wall_time_seconds": time.monotonic() - started,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in curves),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in curves),
            "final_test_access": "none",
        }
        atomic_write_json(seed_dir / "seed_complete.json", completed)
        seed_summaries.append(completed)
        del encoder, head, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "status": "pass",
        "phase": "12_task_lora_pilot",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "platform": platform.platform(),
        "device": str(device),
        "precision": precision,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "epochs": epochs,
        "seeds": list(seeds),
        "task_ids": list(data.task_ids),
        "data_gate": data_gate_report(data),
        "lora": LoRAConfig().to_dict(),
        "head_lr": HEAD_LR,
        "lora_lr": LORA_LR,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed_summaries": seed_summaries,
        "final_test_access": "none",
        "backbone_fine_tuning": False,
        "exploratory_only": True,
    }
    atomic_write_json(output / "PILOT_RUN.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "memory-smoke", "overfit", "pilot"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    feature_run = args.feature_run.resolve()
    data = load_pilot_data(config, args.selection.resolve(), feature_run)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported; run a separately recorded FP16 fallback")
    if args.num_workers < 0 or args.num_workers > 4:
        raise ValueError("num_workers must be between 0 and 4")

    if args.phase == "preflight":
        preflight(data, feature_run, args.output, device)
    elif args.phase == "memory-smoke":
        memory_smoke(data, args.output, device, precision=args.precision, num_workers=args.num_workers)
    elif args.phase == "overfit":
        overfit_smoke(
            data,
            args.output,
            device,
            precision=args.precision,
            micro_batch=args.micro_batch,
            accumulation=args.accumulation,
        )
    else:
        pilot(
            data,
            args.output,
            device,
            precision=args.precision,
            micro_batch=args.micro_batch,
            accumulation=args.accumulation,
            epochs=args.epochs,
            seeds=tuple(args.seeds),
            num_workers=args.num_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
