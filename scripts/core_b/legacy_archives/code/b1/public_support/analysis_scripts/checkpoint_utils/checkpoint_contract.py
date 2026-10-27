"""Safe, complete, and auditable experiment-local checkpoint contract."""

from __future__ import annotations

import os
import pickle
import random
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_identity",
        "physical_batch",
        "gradient_accumulation",
        "nominal_effective_batch",
        "epoch",
        "global_optimizer_step",
        "accumulation_state",
        "model_state_dict",
        "head_state_dict",
        "lora_adapter_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "early_stopping_state",
        "rng_state",
        "data_provenance",
        "final_test_count",
    }
)
SAFE_PRIMITIVES = (str, int, float, bool, type(None))


class CheckpointContractError(ValueError):
    """Raised when a checkpoint is incomplete, unsafe, or inconsistent."""


def cpu_tree(value: Any) -> Any:
    """Deep-copy a safe tree and move every tensor to CPU.

    Only dictionaries, ordered dictionaries, lists, tuples, tensors, and
    primitive metadata are accepted. The input tree is never mutated.
    """

    if torch.is_tensor(value):
        return value.detach().to(device="cpu").clone(memory_format=torch.preserve_format)
    if isinstance(value, OrderedDict):
        return OrderedDict((key, cpu_tree(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, SAFE_PRIMITIVES):
        return value
    raise TypeError(f"unsupported checkpoint object type: {type(value)!r}")


def _walk_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def assert_all_tensors_cpu(value: Any) -> None:
    """Raise unless all tensors in a nested state tree are CPU tensors."""

    devices = {str(tensor.device) for tensor in _walk_tensors(value)}
    if any(device != "cpu" for device in devices):
        raise CheckpointContractError(f"checkpoint contains non-CPU tensors: {sorted(devices)}")


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and visible CUDA RNG state."""

    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(np.asarray(numpy_state[1], dtype=np.uint32)).clone(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in cuda_states],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise CheckpointContractError(f"RNG state missing keys: {sorted(missing)}")
    python_state = state["python"]
    if isinstance(python_state, list):
        python_state = tuple(python_state)
    if not isinstance(python_state, tuple):
        raise CheckpointContractError("Python RNG state is not a tuple")
    random.setstate(python_state)
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointContractError("NumPy RNG state is not a mapping")
    numpy_keys = numpy_state["state"]
    if torch.is_tensor(numpy_keys):
        numpy_keys = numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_keys,
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointContractError(f"{name} must be a mapping")
    return value


def validate_checkpoint(payload: Mapping[str, Any], *, require_cpu: bool = True) -> None:
    """Validate the complete schema without loading arbitrary Python objects."""

    if not isinstance(payload, Mapping):
        raise CheckpointContractError("checkpoint root must be a mapping")
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise CheckpointContractError(f"checkpoint missing required keys: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError(f"unsupported checkpoint schema: {payload['schema_version']!r}")
    if payload["physical_batch"] != 8 or payload["gradient_accumulation"] != 4:
        raise CheckpointContractError("checkpoint batch contract is not physical8/accum4")
    if payload["nominal_effective_batch"] != 32:
        raise CheckpointContractError("checkpoint effective batch is not 32")
    if payload["final_test_count"] != 0:
        raise CheckpointContractError("checkpoint contains final-test access")
    run_identity = _require_mapping(payload["run_identity"], "run_identity")
    for key in ("cell", "input", "head", "adaptation", "seed"):
        if key not in run_identity:
            raise CheckpointContractError(f"run_identity missing {key}")
    accumulation = _require_mapping(payload["accumulation_state"], "accumulation_state")
    if accumulation.get("pending_microbatches") != 0 or accumulation.get("at_cycle_boundary") is not True:
        raise CheckpointContractError("checkpoint is not at an accumulation boundary")
    if accumulation.get("optimizer_step") != payload["global_optimizer_step"]:
        raise CheckpointContractError("accumulation and global optimizer step disagree")
    for key in ("model_state_dict", "head_state_dict", "lora_adapter_state_dict", "optimizer_state_dict", "scheduler_state_dict", "early_stopping_state", "data_provenance"):
        _require_mapping(payload[key], key)
    if payload["scaler_state_dict"] is not None:
        _require_mapping(payload["scaler_state_dict"], "scaler_state_dict")
    rng = _require_mapping(payload["rng_state"], "rng_state")
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if key not in rng:
            raise CheckpointContractError(f"rng_state missing {key}")
    if require_cpu:
        assert_all_tensors_cpu(payload)


def build_checkpoint_payload(
    *,
    protocol_id: str,
    run_identity: Mapping[str, Any],
    epoch: int,
    global_optimizer_step: int,
    accumulation_state: Mapping[str, Any],
    model_state_dict: Mapping[str, Any],
    head_state_dict: Mapping[str, Any],
    lora_adapter_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    scheduler_state_dict: Mapping[str, Any],
    scaler_state_dict: Mapping[str, Any] | None,
    early_stopping_state: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "run_identity": dict(run_identity),
        "physical_batch": 8,
        "gradient_accumulation": 4,
        "nominal_effective_batch": 32,
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "accumulation_state": dict(accumulation_state),
        "model_state_dict": model_state_dict,
        "head_state_dict": head_state_dict,
        "lora_adapter_state_dict": lora_adapter_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "early_stopping_state": dict(early_stopping_state),
        "rng_state": capture_rng_state() if rng_state is None else rng_state,
        "data_provenance": dict(data_provenance),
        "final_test_count": 0,
    }
    safe_payload = cpu_tree(payload)
    validate_checkpoint(safe_payload)
    return safe_payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically save a validated CPU checkpoint without overwriting."""

    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    safe_payload = cpu_tree(payload)
    validate_checkpoint(safe_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        with temporary.open("wb") as handle:
            torch.save(safe_payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load only the safe tensor/primitive schema and validate it."""

    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckpointContractError(f"checkpoint is missing or empty: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise CheckpointContractError(f"checkpoint load failed: {exc}") from exc
    require_cpu = str(map_location) == "cpu" or map_location == "cpu"
    validate_checkpoint(payload, require_cpu=require_cpu)
    return payload
