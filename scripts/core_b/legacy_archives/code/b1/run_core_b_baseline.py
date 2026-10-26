"""Run one full eligible Core-B 1.0-second baseline on GPU3.

The baseline is train/dev-only and uses source-native numeric commands.  It
supports pooled MLP lanes and the official depth-4 attentive pooler; no raw
video is decoded by this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

PHYSICAL_BATCH = 8
ACCUMULATION = 4
EFFECTIVE_BATCH = 32
MAX_EPOCHS = 20
MIN_EPOCHS = 5
PATIENCE = 5
MIN_DELTA = 0.0001
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
STEPS = 30
TARGET_DIMENSIONS = 28
ARM_SLICE = slice(14, 28)
HAND_SLICE = slice(0, 14)
SEED = 20260903
INPUTS_FILE = "CORE_B_FULL_INPUTS_1p0s.npz"
TARGET_NORM_FILE = "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
PROTOCOL_ID = "core_b_gpu3_native_v1_1p0s_batch8_accum4"

LANES: dict[str, dict[str, str]] = {
    "B-STATE": {"kind": "state", "input": "state_action_history", "head": "pooled"},
    "B-D-P-F": {"kind": "pooled", "input": "dynamic_pooled", "head": "pooled"},
    "B-S-P-F": {"kind": "pooled", "input": "static_pooled", "head": "pooled"},
    "B-D-A-F": {"kind": "attentive", "input": "dynamic", "head": "attentive"},
    "B-S-A-F": {"kind": "attentive", "input": "static", "head": "attentive"},
    "B-D-FUS-P-F": {
        "kind": "fusion",
        "input": "dynamic_pooled+state_action_history",
        "head": "pooled",
    },
    "B-S-FUS-P-F": {
        "kind": "fusion",
        "input": "static_pooled+state_action_history",
        "head": "pooled",
    },
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_support(fix_root: Path) -> Any:
    expected = Path(__file__).resolve().parent / "public_support"
    support = fix_root.resolve()
    if support != expected.resolve():
        raise ValueError(f"--fix-root must name the public B1 support package: {expected}")
    analysis = support / "analysis_scripts"
    if not analysis.is_dir():
        raise FileNotFoundError(f"public B1 analysis support is missing: {analysis}")

    _reject_import_collision("fresh_process_batch8_d_a_l_fixed", analysis)
    _reject_import_collision("checkpoint_utils", analysis)
    if str(analysis) in sys.path:
        sys.path.remove(str(analysis))
    sys.path.insert(0, str(analysis))
    fixed = importlib.import_module("fresh_process_batch8_d_a_l_fixed")
    checkpoint_utils = importlib.import_module("checkpoint_utils")
    if not _module_is_within(fixed, analysis):
        raise ImportError("B1 helper did not resolve from the public support package")
    if not _module_is_within(checkpoint_utils, analysis):
        raise ImportError("B1 checkpoint utilities did not resolve from public support")
    return fixed


def _module_is_within(module: Any, root: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin:
        try:
            Path(origin).resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False
    locations = getattr(module, "__path__", ())
    if not locations:
        return False
    try:
        return all(
            Path(location).resolve().is_relative_to(root.resolve())
            for location in locations
        )
    except (OSError, ValueError):
        return False


def _reject_import_collision(module_name: str, root: Path) -> None:
    loaded = sys.modules.get(module_name)
    if loaded is not None and not _module_is_within(loaded, root):
        raise ImportError(
            f"{module_name} is already loaded outside the public B1 support package"
        )


def _load_attentive_pooler(source_root: Path) -> type[nn.Module]:
    expected = Path(__file__).resolve().parent / "public_support" / "vjepa2"
    source = source_root.resolve()
    if source != expected.resolve():
        raise ValueError(f"--source-root must name the public B1 V-JEPA2 subset: {expected}")
    pooler_file = source / "src" / "models" / "attentive_pooler.py"
    if not pooler_file.is_file():
        raise FileNotFoundError(f"public V-JEPA2 attentive pooler is missing: {pooler_file}")

    for name, module in tuple(sys.modules.items()):
        if (name == "src" or name.startswith("src.")) and not _module_is_within(
            module, source
        ):
            raise ImportError(f"{name} is already loaded outside the public V-JEPA2 source")

    if str(source) in sys.path:
        sys.path.remove(str(source))
    sys.path.insert(0, str(source))
    module = importlib.import_module("src.models.attentive_pooler")
    for name in (
        "src",
        "src.models",
        "src.models.attentive_pooler",
        "src.models.utils",
        "src.models.utils.modules",
        "src.utils",
        "src.utils.tensors",
    ):
        loaded = sys.modules.get(name)
        if loaded is None or not _module_is_within(loaded, source):
            raise ImportError(f"{name} did not resolve inside the public V-JEPA2 source")
    pooler_type = getattr(module, "AttentivePooler", None)
    if not isinstance(pooler_type, type) or not issubclass(pooler_type, nn.Module):
        raise ImportError("official V-JEPA2 AttentivePooler class is unavailable")
    return pooler_type


class PreloadedTokenDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Keep verified float16 token cache resident for one attentive child."""

    def __init__(
        self,
        ids: np.ndarray,
        tasks: np.ndarray,
        targets: np.ndarray,
        token_paths: dict[str, str],
    ) -> None:
        self.ids = ids
        self.tasks = tasks
        self.targets = targets
        self.tokens: list[Tensor] = []
        for episode_index in ids.tolist():
            record = torch.load(
                token_paths[str(int(episode_index))],
                map_location="cpu",
                weights_only=True,
            )
            tokens = record.get("tokens") if isinstance(record, dict) else None
            if not torch.is_tensor(tokens) or tuple(tokens.shape) != (9216, 768):
                raise ValueError(f"attentive token shape failed: {episode_index}")
            if not torch.isfinite(tokens).all():
                raise ValueError(f"attentive token finite check failed: {episode_index}")
            self.tokens.append(tokens.contiguous())

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.tokens[index],
            torch.from_numpy(self.targets[index]),
            torch.tensor(self.ids[index], dtype=torch.long),
            torch.tensor(self.tasks[index], dtype=torch.long),
        )


class AttentiveRegression(nn.Module):
    def __init__(self, source_root: Path) -> None:
        super().__init__()
        AttentivePooler = _load_attentive_pooler(source_root)
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=768,
            num_heads=16,
            depth=4,
            use_activation_checkpointing=False,
        )
        self.linear = nn.Linear(768, STEPS * TARGET_DIMENSIONS)

    def forward(self, tokens: Tensor) -> Tensor:
        pooled = self.pooler(tokens).squeeze(1)
        return self.linear(pooled).reshape(-1, STEPS, TARGET_DIMENSIONS)


class PooledRegression(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, STEPS * TARGET_DIMENSIONS),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs).reshape(-1, STEPS, TARGET_DIMENSIONS)


def load_data(root: Path, feasibility_root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    with np.load(root / INPUTS_FILE, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {
        "episode_index",
        "split_code",
        "original_task_index",
        "targets",
        "targets_standardized",
        "last_observed_action",
        "dynamic_pooled",
        "static_pooled",
        "state_action_history",
    }
    if required - set(data):
        raise ValueError(f"Core-B input keys missing: {sorted(required - set(data))}")
    split = data["split_code"]
    train_count = int(np.sum(split == 0))
    if not np.all(split[:train_count] == 0) or not np.all(split[train_count:] == 1):
        raise ValueError("train rows are not leading in Core-B input cache")
    dev_count = int(np.sum(split == 1))
    if train_count != 2741 or dev_count != 614:
        raise ValueError(f"unexpected Core-B eligible population: {train_count}/{dev_count}")
    if len(np.unique(data["episode_index"])) != len(data["episode_index"]):
        raise ValueError("Core-B input episode IDs are not unique")
    if not np.isfinite(data["targets_standardized"]).all():
        raise ValueError("Core-B standardized targets are non-finite")
    norm_all = json.loads(
        (feasibility_root / TARGET_NORM_FILE).read_text(encoding="utf-8")
    )
    norm = norm_all["1.0s"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["safe_std"], dtype=np.float32)
    if mean.shape != (STEPS, TARGET_DIMENSIONS) or std.shape != mean.shape:
        raise ValueError("Core-B target normalization shape failed")
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("Core-B target normalization is invalid")
    token_paths = json.loads((root / "CORE_B_TOKEN_PATHS_1p0s.json").read_text(encoding="utf-8"))
    return data, {"mean": mean, "std": std}, token_paths


def lane_inputs(
    data: dict[str, np.ndarray],
    lane: str,
    split_mask: np.ndarray,
    token_paths: dict[str, dict[str, str]],
) -> tuple[np.ndarray | None, dict[str, str] | None]:
    spec = LANES[lane]
    if spec["kind"] == "state":
        return data["state_action_history"][split_mask], None
    if spec["kind"] == "pooled":
        return data[spec["input"]][split_mask], None
    if spec["kind"] == "fusion":
        pieces = [
            data[name][split_mask]
            for name in spec["input"].split("+")
        ]
        return np.concatenate(pieces, axis=1), None
    return None, {
        str(int(episode)): token_paths[spec["input"]][str(int(episode))]
        for episode in data["episode_index"][split_mask].tolist()
    }


def build_loader(
    inputs: np.ndarray | None,
    token_paths: dict[str, str] | None,
    ids: np.ndarray,
    tasks: np.ndarray,
    targets: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if token_paths is not None:
        dataset = PreloadedTokenDataset(ids, tasks, targets, token_paths)
    else:
        if inputs is None:
            raise ValueError("dense inputs are missing")
        dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(ids),
            torch.from_numpy(tasks),
        )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


def forward_model(model: nn.Module, inputs: Tensor, kind: str) -> Tensor:
    if kind == "attentive":
        return model(inputs.float())
    return model(inputs.float())


def train_epoch(
    fixed: Any,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    global_step: int,
) -> tuple[dict[str, object], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_elements = 0
    microbatches = 0
    optimizer_steps = 0
    pending = 0
    cycle_samples = 0
    cycle_started = time.monotonic()
    cycle_loss_sum = 0.0
    step_rows: list[dict[str, object]] = []
    for batch_index, (inputs, targets, _ids, _tasks) in enumerate(loader, start=1):
        batch_size = int(targets.shape[0])
        if batch_size <= 0 or batch_size > PHYSICAL_BATCH:
            raise RuntimeError(f"Core-B physical batch failed: {batch_size}")
        if pending == 0:
            optimizer.zero_grad(set_to_none=True)
            cycle_samples = 0
            cycle_loss_sum = 0.0
            cycle_started = time.monotonic()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with fixed.autocast_context(device):
            prediction = forward_model(model, inputs, "attentive" if inputs.ndim == 3 else "pooled")
            loss_sum = F.mse_loss(prediction.float(), targets.float(), reduction="sum")
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Core-B training loss is non-finite")
        loss_sum.backward()
        total_loss += float(loss_sum.detach().cpu())
        total_elements += int(targets.numel())
        cycle_loss_sum += float(loss_sum.detach().cpu())
        cycle_samples += batch_size
        microbatches += 1
        pending += 1
        should_step = pending == ACCUMULATION or batch_index == len(loader)
        if should_step:
            denominator = cycle_samples * STEPS * TARGET_DIMENSIONS
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(denominator)
            gradients = [
                parameter for parameter in model.parameters() if parameter.grad is not None
            ]
            norm = torch.nn.utils.clip_grad_norm_(gradients, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Core-B gradient is non-finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            global_step += 1
            step_rows.append(
                {
                    "global_optimizer_step": global_step,
                    "microbatches_in_cycle": pending,
                    "cycle_sample_count": cycle_samples,
                    "cycle_loss_mean": cycle_loss_sum / denominator,
                    "partial_cycle": pending != ACCUMULATION,
                    "scheduler_steps": global_step,
                    "cycle_wall_time_seconds": time.monotonic() - cycle_started,
                }
            )
            pending = 0
        del prediction, loss_sum, inputs, targets
    expected_steps = math.ceil(math.ceil(total_elements / TARGET_DIMENSIONS / STEPS) / PHYSICAL_BATCH / ACCUMULATION)
    if total_elements != len(loader.dataset) * STEPS * TARGET_DIMENSIONS:
        raise RuntimeError("Core-B train element coverage failed")
    if pending != 0 or optimizer_steps != expected_steps:
        raise RuntimeError(
            f"Core-B accumulation failed: pending={pending}, steps={optimizer_steps}, expected={expected_steps}"
        )
    return (
        {
            "loss": total_loss / total_elements,
            "microbatches": microbatches,
            "optimizer_steps": optimizer_steps,
            "global_optimizer_step": global_step,
            "partial_final_cycle": bool(step_rows and step_rows[-1]["partial_cycle"]),
            "step_rows": step_rows,
        },
        global_step,
    )


@torch.inference_mode()
def predict(
    fixed: Any,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    dev_native: np.ndarray,
    dev_persistence: np.ndarray,
) -> dict[str, object]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    ids: list[Tensor] = []
    tasks: list[Tensor] = []
    for inputs, target, episode, task in loader:
        inputs = inputs.to(device, non_blocking=True)
        with fixed.autocast_context(device):
            value = forward_model(
                model, inputs, "attentive" if inputs.ndim == 3 else "pooled"
            ).float().cpu()
        predictions.append(value)
        targets.append(target.cpu())
        ids.append(episode.cpu())
        tasks.append(task.cpu())
    prediction_std = torch.cat(predictions).numpy()
    target_values_std = torch.cat(targets).numpy()
    episode_index = torch.cat(ids).numpy()
    task_index = torch.cat(tasks).numpy()
    if prediction_std.shape != target_values_std.shape or prediction_std.shape[0] != len(dev_native):
        raise RuntimeError("Core-B dev prediction shape/coverage failed")
    prediction_native = prediction_std * target_std + target_mean
    target_native = dev_native
    persistence_native = dev_persistence
    error_std = prediction_std - target_values_std
    error_native = prediction_native - target_native
    persistence_std = (persistence_native - target_mean) / target_std

    def metric_summary(error: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "final_step_mae": float(np.abs(error[:, -1, :]).mean()),
            "arm_mae": float(np.abs(error[:, :, ARM_SLICE]).mean()),
            "hand_mae": float(np.abs(error[:, :, HAND_SLICE]).mean()),
        }

    metrics = {
        "standardized": metric_summary(error_std),
        "native_command_value": metric_summary(error_native),
        "persistence_standardized": metric_summary(
            persistence_std - target_values_std
        ),
        "persistence_native_command_value": metric_summary(
            persistence_native - target_native
        ),
        "per_timestep_standardized_mae": np.abs(error_std).mean(axis=(0, 2)).tolist(),
        "per_joint_time_standardized_mae": np.abs(error_std).mean(axis=0).tolist(),
        "all_finite": bool(
            np.isfinite(prediction_std).all() and np.isfinite(prediction_native).all()
        ),
    }
    return {
        "episode_index": episode_index,
        "original_task_index": task_index,
        "targets_standardized": target_std,
        "predictions_standardized": prediction_std,
        "targets_native": target_native,
        "predictions_native": prediction_native,
        "last_observed_action": persistence_native[:, 0, :],
        "metrics": metrics,
    }


def checkpoint_payload(
    fixed: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    data_manifest: dict[str, object],
    identity: dict[str, object],
    early: dict[str, object],
    epoch: int,
    global_step: int,
) -> dict[str, object]:
    from checkpoint_utils import capture_rng_state

    return fixed.prepare_checkpoint_payload(
        protocol_id=PROTOCOL_ID,
        run_identity=identity,
        epoch=epoch,
        global_optimizer_step=global_step,
        accumulation_state={
            "pending_microbatches": 0,
            "at_cycle_boundary": True,
            "optimizer_step": global_step,
            "microbatches_in_cycle": 0,
            "cycle_sample_count": 0,
        },
        model_state_dict=model.state_dict(),
        head_state_dict={},
        lora_adapter_state_dict={},
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=None,
        early_stopping_state=early,
        data_provenance={
            **data_manifest,
            "stage": "core_b_baseline",
            "final_test_count": 0,
        },
        rng_state=capture_rng_state(),
    )


def apply_checkpoint(
    payload: dict[str, object],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
) -> None:
    from checkpoint_utils import restore_rng_state, validate_checkpoint

    validate_checkpoint(payload)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload["head_state_dict"]:
        raise RuntimeError("Core-B checkpoint unexpectedly has a separate head state")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_state(payload["rng_state"])


def run(args: argparse.Namespace) -> int:
    if args.lane not in LANES:
        raise ValueError(f"unknown Core-B lane: {args.lane}")
    if args.physical_gpu != 3:
        raise ValueError("Core-B role is GPU3-only")
    if args.seed != SEED:
        raise ValueError("Core-B baseline seed must be 20260903")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Core-B runner requires CUDA_VISIBLE_DEVICES=3")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU3 CUDA/BF16 is unavailable")
    run_dir = args.run_root.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing existing Core-B run root: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    feasibility_root = args.feasibility_root.resolve()
    fixed = load_support(args.fix_root.resolve())
    data, target_norm, token_paths = load_data(root, feasibility_root)
    set_seed(SEED)
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    dev_ids = data["episode_index"][dev_mask]
    dev_tasks = data["original_task_index"][dev_mask]
    train_ids = data["episode_index"][train_mask]
    train_tasks = data["original_task_index"][train_mask]
    target_mean = target_norm["mean"]
    target_std = target_norm["std"]
    train_targets = data["targets_standardized"][train_mask]
    dev_targets = data["targets_standardized"][dev_mask]
    dev_native = data["targets"][dev_mask]
    dev_persistence = np.repeat(
        data["last_observed_action"][dev_mask, None, :], STEPS, axis=1
    )
    spec = LANES[args.lane]
    train_inputs, train_tokens = lane_inputs(data, args.lane, train_mask, token_paths)
    dev_inputs, dev_tokens = lane_inputs(data, args.lane, dev_mask, token_paths)
    if train_tokens is not None:
        model = AttentiveRegression(args.source_root.resolve()).to(torch.device("cuda:0"))
        preload_mib = 52000
    else:
        if train_inputs is None or dev_inputs is None:
            raise RuntimeError("Core-B pooled inputs are missing")
        model = PooledRegression(int(train_inputs.shape[1])).to(torch.device("cuda:0"))
        preload_mib = 0
    device = torch.device("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_batches = math.ceil(len(train_ids) / PHYSICAL_BATCH)
    steps_per_epoch = math.ceil(train_batches / ACCUMULATION)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS * steps_per_epoch
    )
    train_loader = build_loader(
        train_inputs,
        train_tokens,
        train_ids,
        train_tasks,
        train_targets,
        shuffle=True,
        seed=SEED,
    )
    dev_loader = build_loader(
        dev_inputs,
        dev_tokens,
        dev_ids,
        dev_tasks,
        dev_targets,
        shuffle=False,
        seed=SEED,
    )
    identity = {
        "cell": args.lane,
        "lane": args.lane,
        "input": spec["input"],
        "head": spec["head"],
        "adaptation": "frozen",
        "seed": SEED,
    }
    input_manifest = json.loads(
        (root / "CORE_B_FULL_DATA_MANIFEST.json").read_text(encoding="utf-8")
    )
    config = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "role": "core_b",
        "cell": args.lane,
        "lane_kind": spec["kind"],
        "input": spec["input"],
        "head": spec["head"],
        "adaptation": "frozen",
        "physical_gpu": 3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_batch": PHYSICAL_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "nominal_effective_batch": EFFECTIVE_BATCH,
        "precision": "BF16",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "train_count": int(train_mask.sum()),
        "development_validation_count": int(dev_mask.sum()),
        "steps_per_epoch": steps_per_epoch,
        "target_horizon": "1.0s",
        "target_steps": STEPS,
        "target_dimensions": TARGET_DIMENSIONS,
        "primary_target_subset": "arm[14:28]",
        "target_physical_unit_status": "unresolved",
        "normalization": "train-only per-dimension mean/std",
        "token_cache_preload": train_tokens is not None,
        "preload_rss_mib": preload_mib,
        "raw_video_io": False,
        "task_labels_used_as_model_input": False,
        "final_test_accessed": 0,
        "input_manifest_sha256": sha256_file(root / "CORE_B_FULL_DATA_MANIFEST.json"),
        "input_cache_sha256": sha256_file(root / INPUTS_FILE),
        "target_normalization_sha256": sha256_file(
            feasibility_root / TARGET_NORM_FILE
        ),
    }
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_dir / "MODEL_CONTRACT.json", {
        "model": type(model).__name__,
        "lane": args.lane,
        "official_attentive_depth": 4 if spec["kind"] == "attentive" else None,
        "activation_checkpointing_or_offload_used": False,
    })
    status: dict[str, object] = {
        "status": "running",
        "exit_code": None,
        "run_id": run_dir.name,
        "cell": args.lane,
        "seed": SEED,
        "physical_gpu": 3,
        "started_utc": utc_now(),
        "last_completed_epoch": 0,
        "global_optimizer_step": 0,
        "final_test_accessed": 0,
    }
    atomic_json(run_dir / "RUN_STATUS.json", status)
    early = {
        "best_epoch": None,
        "best_standardized_mae": math.inf,
        "bad_epochs": 0,
        "stop_reason": None,
    }
    best_payload: dict[str, object] | None = None
    curves: list[dict[str, object]] = []
    global_step = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.monotonic()
        train_result, global_step = train_epoch(
            fixed, model, train_loader, device, optimizer, scheduler, global_step
        )
        validation = predict(
            fixed,
            model,
            dev_loader,
            device,
            target_mean,
            target_std,
            dev_native,
            dev_persistence,
        )
        standard_mae = float(validation["metrics"]["standardized"]["mae"])
        standard_rmse = float(validation["metrics"]["standardized"]["rmse"])
        improved = standard_mae < float(early["best_standardized_mae"]) - MIN_DELTA
        if improved:
            early.update(
                {
                    "best_epoch": epoch,
                    "best_standardized_mae": standard_mae,
                    "bad_epochs": 0,
                }
            )
        else:
            early["bad_epochs"] = int(early["bad_epochs"]) + 1
        torch.cuda.synchronize(device)
        curve = {
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "validation_standardized_mae": standard_mae,
            "validation_standardized_rmse": standard_rmse,
            "validation_arm_mae": float(validation["metrics"]["standardized"]["arm_mae"]),
            "validation_hand_mae": float(validation["metrics"]["standardized"]["hand_mae"]),
            "validation_persistence_mae": float(
                validation["metrics"]["persistence_standardized"]["mae"]
            ),
            "improved": improved,
            "bad_epochs": int(early["bad_epochs"]),
            "microbatches": int(train_result["microbatches"]),
            "optimizer_steps": int(train_result["optimizer_steps"]),
            "global_optimizer_step": global_step,
            "partial_final_cycle": bool(train_result["partial_final_cycle"]),
            "last_cycle_sample_count": int(
                train_result["step_rows"][-1]["cycle_sample_count"]
            ),
            "epoch_wall_time_seconds": time.monotonic() - epoch_started,
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / 2**20, 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / 2**20, 3
            ),
        }
        curves.append(curve)
        if improved:
            best_payload = checkpoint_payload(
                fixed,
                model,
                optimizer,
                scheduler,
                input_manifest,
                identity,
                dict(early),
                epoch,
                global_step,
            )
        status.update(
            {
                "last_completed_epoch": epoch,
                "global_optimizer_step": global_step,
                "best_epoch": early["best_epoch"],
                "bad_epochs": early["bad_epochs"],
                "updated_utc": utc_now(),
            }
        )
        atomic_json(run_dir / "RUN_STATUS.json", status)
        print(
            f"CORE_B_PROGRESS run={run_dir.name} epoch={epoch}/{MAX_EPOCHS} "
            f"train_loss={curve['train_loss']:.6f} "
            f"dev_mae={curve['validation_standardized_mae']:.6f} "
            f"epoch_seconds={curve['epoch_wall_time_seconds']:.1f}",
            flush=True,
        )
        if epoch >= MIN_EPOCHS and int(early["bad_epochs"]) >= PATIENCE:
            early["stop_reason"] = "patience_exhausted"
            break
    if best_payload is None:
        raise RuntimeError("Core-B baseline never selected a best checkpoint")
    if early["stop_reason"] is None:
        early["stop_reason"] = (
            "max_epochs_reached" if len(curves) == MAX_EPOCHS else "completed"
        )
    last_payload = checkpoint_payload(
        fixed,
        model,
        optimizer,
        scheduler,
        input_manifest,
        identity,
        dict(early),
        len(curves),
        global_step,
    )
    from checkpoint_utils import load_checkpoint, save_checkpoint

    save_checkpoint(run_dir / "checkpoint_best.pt", best_payload)
    save_checkpoint(run_dir / "checkpoint_last.pt", last_payload)
    loaded_best = load_checkpoint(run_dir / "checkpoint_best.pt", map_location="cpu")
    loaded_last = load_checkpoint(run_dir / "checkpoint_last.pt", map_location="cpu")
    apply_checkpoint(loaded_best, model, optimizer, scheduler)
    best_prediction = predict(
        fixed, model, dev_loader, device, target_mean, target_std, dev_native, dev_persistence
    )
    best_arrays = {key: value for key, value in best_prediction.items() if key != "metrics"}
    atomic_npz(run_dir / "predictions_best.npz", **best_arrays)
    apply_checkpoint(loaded_last, model, optimizer, scheduler)
    last_prediction = predict(
        fixed, model, dev_loader, device, target_mean, target_std, dev_native, dev_persistence
    )
    last_arrays = {key: value for key, value in last_prediction.items() if key != "metrics"}
    atomic_npz(run_dir / "predictions_last.npz", **last_arrays)
    atomic_csv(run_dir / "epoch_curves.csv", curves)
    atomic_json(run_dir / "epoch_curves.json", {"curves": curves})
    metrics = {
        "status": "complete",
        "role": "core_b",
        "cell": args.lane,
        "lane": args.lane,
        "seed": SEED,
        "physical_gpu": 3,
        "best_epoch": int(early["best_epoch"]),
        "last_epoch": len(curves),
        "early_stopped": len(curves) < MAX_EPOCHS,
        "stop_reason": early["stop_reason"],
        "best": best_prediction["metrics"],
        "last": last_prediction["metrics"],
        "train_count": int(train_mask.sum()),
        "development_validation_count": int(dev_mask.sum()),
        "physical_batch": PHYSICAL_BATCH,
        "gradient_accumulation": ACCUMULATION,
        "nominal_effective_batch": EFFECTIVE_BATCH,
        "steps_per_epoch": steps_per_epoch,
        "global_optimizer_steps": global_step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
        "final_test_accessed": 0,
        "target_physical_unit_status": "unresolved",
        "activation_checkpointing_or_offload_used": False,
        "raw_video_io": False,
    }
    atomic_json(run_dir / "RUN_METRICS.json", metrics)
    status.update(
        {
            "status": "complete",
            "exit_code": 0,
            "completed_utc": utc_now(),
            "best_epoch": early["best_epoch"],
            "last_epoch": len(curves),
        }
    )
    atomic_json(run_dir / "RUN_STATUS.json", status)
    artifact_names = [
        "RUN_CONFIG_LOCKED.json",
        "MODEL_CONTRACT.json",
        "RUN_STATUS.json",
        "RUN_METRICS.json",
        "checkpoint_best.pt",
        "checkpoint_last.pt",
        "predictions_best.npz",
        "predictions_last.npz",
        "epoch_curves.csv",
        "epoch_curves.json",
    ]
    atomic_json(
        run_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "exit_code": 0,
            "role": "core_b",
            "cell": args.lane,
            "seed": SEED,
            "physical_gpu": 3,
            "artifact_sha256": {
                name: sha256_file(run_dir / name) for name in artifact_names
            },
            "final_test_accessed": 0,
            "target_physical_unit_status": "unresolved",
            "raw_video_io": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "cell": args.lane,
                "best_epoch": early["best_epoch"],
                "last_epoch": len(curves),
                "physical_gpu": 3,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--fix-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lane", choices=tuple(LANES), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        run_dir = args.run_root.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            run_dir / "RUN_STATUS.json",
            {
                "status": "failed",
                "exit_code": 1,
                "role": "core_b",
                "cell": args.lane,
                "seed": args.seed,
                "physical_gpu": args.physical_gpu,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "final_test_accessed": 0,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
