from __future__ import annotations

import bisect
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V2 = PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
V3 = PROJECT / "reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z"
V4 = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z"
V5 = PROJECT / "reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z"
V6 = PROJECT / "reports/benchmarks/CORE_B_V6_EQUAL_HALF_ENSEMBLE_CONTROLS_V1_20261015T070036Z"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT")
TRAIN_NPZ = V4 / "data/TRAIN_SELECTED_H30.npz"
DEV_NPZ = V4 / "data/DEV_PREFIXES_NO_LABELS.npz"
DEV_TARGET_NPZ = V2 / "data/DEV_TARGETS_SCORER_ONLY.npz"
NORMALIZER = V4 / "normalizers/NORMALIZER_FORMAL.json"
METRIC_LOCK = V4 / "protocol/METRIC_SCALE_LOCK.json"
RESOLVED_PATHS = V4 / "inventory/RESOLVED_PATHS.json"
TRAIN_PREFIX_FIELDS = (
    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
    "cache_index", "anchor_frame", "history_native", "history_mask",
    "relative_delta", "anchor_native", "inner_role",
)
DEV_PREFIX_FIELDS = (
    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
    "cache_index", "anchor_frame", "history_native", "history_mask",
    "relative_delta", "anchor_native",
)
ROW_FIELDS = (
    "row", "anchor_id", "episode_id", "task_id", "task_rank",
    "source_cluster", "anchor_frame",
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_npy(path: Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz_fields(path: Path, fields: tuple[str, ...], *, forbid_target: bool = False) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if forbid_target and "target_native_h30" in archive.files:
            raise RuntimeError(f"{path}: target field is forbidden in this prefix interface")
        missing = sorted(set(fields) - set(archive.files))
        if missing:
            raise RuntimeError(f"{path}: missing required prefix fields {missing}")
        return {field: archive[field].copy() for field in fields}


def load_train_prefix() -> dict[str, np.ndarray]:
    return load_npz_fields(TRAIN_NPZ, TRAIN_PREFIX_FIELDS)


def load_dev_prefix() -> dict[str, np.ndarray]:
    return load_npz_fields(DEV_NPZ, DEV_PREFIX_FIELDS, forbid_target=True)


def load_train_y_after_teacher_seal() -> np.ndarray:
    seal_path = ROOT / "protocol/TEACHER_BANK_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("refusing to read outer-train Y before the teacher bank is sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("status") != "PASS"
        or int(seal.get("source_prediction_bundles_accepted", -1)) != 6
        or int(seal.get("derived_T_bundles_accepted", -1)) != 3
        or len(seal.get("entries", [])) != 9
    ):
        raise RuntimeError("refusing to read outer-train Y: teacher seal is absent, empty, or incomplete")
    with np.load(TRAIN_NPZ, allow_pickle=False) as archive:
        if "target_native_h30" not in archive.files:
            raise RuntimeError("accepted outer-train archive lacks target_native_h30")
        values = archive["target_native_h30"].copy()
    if values.shape != (1280, 30, 28) or not np.isfinite(values).all():
        raise RuntimeError(f"outer-train Y shape/finite failure: {values.shape}")
    return values


def receiver_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    count = len(data["anchor_id"])
    rows: list[dict[str, Any]] = []
    for i in range(count):
        task = int(data["task_id"][i])
        if task not in TASKS or int(data["task_rank"][i]) != TASKS.index(task):
            raise RuntimeError(f"accepted task order mismatch at row {i}")
        rows.append({
            "row": i,
            "anchor_id": str(data["anchor_id"][i]),
            "episode_id": int(data["episode_id"][i]),
            "task_id": task,
            "task_rank": int(data["task_rank"][i]),
            "source_cluster": str(data["source_cluster"][i]),
            "anchor_frame": int(data["anchor_frame"][i]),
        })
    return rows


def write_receiver_rows(path: Path, data: dict[str, np.ndarray]) -> None:
    write_csv(path, list(ROW_FIELDS), receiver_rows(data))


def read_receiver_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty receiver row file: {path}")
    return rows


def ordered_receiver_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["anchor_id"]) for row in rows).encode("utf-8")
    return sha_bytes(payload)


def load_model_api():
    code = str(V4 / "code")
    locked = str(V4 / "code/locked_source")
    if code not in sys.path:
        sys.path.insert(0, code)
    if locked not in sys.path:
        sys.path.insert(0, locked)
    from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash
    return V2TaskConditionedPredictor, parameter_count, state_hash


class Q0ShardedGrid:
    """Read accepted frozen Q0 feature shards; never writes to the cache."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        if manifest.get("status") != "PASS" or int(manifest.get("final_test_access", -1)) != 0:
            raise RuntimeError("Q0 cache manifest is not PASS/final0")
        self.shape = tuple(int(x) for x in manifest["logical_shape"])
        self.shards = sorted(manifest["shards"], key=lambda row: int(row["start_row"]))
        self.starts = [int(row["start_row"]) for row in self.shards]
        self.arrays: dict[int, np.ndarray] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        cursor = 0
        for shard in self.shards:
            start, end = int(shard["start_row"]), int(shard["end_row"])
            if start != cursor or end <= start:
                raise RuntimeError(f"Q0 shards have gap/overlap at {start}:{end}")
            cursor = end
        if cursor != self.shape[0] or self.shape[1:] != (2, 16, 4, 768):
            raise RuntimeError(f"Q0 logical shape/coverage mismatch: {self.shape}, covered={cursor}")

    def row(self, index: int, anchor_id: str, episode_id: str) -> np.ndarray:
        shard_index = bisect.bisect_right(self.starts, int(index)) - 1
        if shard_index < 0:
            raise IndexError(f"Q0 row has no shard: {index}")
        shard = self.shards[shard_index]
        start, end = int(shard["start_row"]), int(shard["end_row"])
        if index >= end:
            raise IndexError(f"Q0 row is outside shard: {index} not in {start}:{end}")
        if shard_index not in self.arrays:
            array_path, meta_path = Path(shard["array"]), Path(shard["meta"])
            if not array_path.is_file() or not meta_path.is_file():
                raise FileNotFoundError(f"accepted Q0 shard missing: {array_path}")
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (end - start, 2, 16, 4, 768) or values.dtype != np.float32:
                raise RuntimeError(f"Q0 shard shape/dtype mismatch: {array_path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") != "PASS" or int(meta.get("final_test_access", -1)) != 0:
                raise RuntimeError(f"Q0 shard metadata not accepted: {meta_path}")
            if int(meta.get("start_row", -1)) != start or int(meta.get("end_row", -1)) != end:
                raise RuntimeError(f"Q0 shard metadata range mismatch: {meta_path}")
            self.arrays[shard_index] = values
            self.metadata[shard_index] = meta
        local = int(index) - start
        meta = self.metadata[shard_index]
        if str(meta["anchor_ids"][local]) != str(anchor_id) or int(meta["episode_index"][local]) != int(episode_id):
            raise RuntimeError(f"Q0 row identity mismatch at {index}: {anchor_id}/{episode_id}")
        return self.arrays[shard_index][local]


def make_grid() -> Q0ShardedGrid:
    resolved = json.loads(RESOLVED_PATHS.read_text(encoding="utf-8"))
    manifest_path = Path(resolved["feature_cache"])
    if sha_file(manifest_path) != resolved["feature_cache_manifest_sha256"]:
        raise RuntimeError("accepted feature cache manifest SHA-256 mismatch")
    return Q0ShardedGrid(json.loads(manifest_path.read_text(encoding="utf-8")))


def validate_prefix_keys(prefix: dict[str, torch.Tensor], video_mode: str) -> None:
    allowed = {"state", "history_mask", "relative_delta", "anchor_command"}
    if video_mode != "STATE-CTRL":
        allowed |= {"dynamic", "static"}
    if set(prefix) != allowed:
        raise RuntimeError(f"unexpected model prefix keys: {sorted(set(prefix) ^ allowed)}")


def prefix_batch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    normalizer: dict[str, Any],
    device: torch.device,
    video_mode: str,
    grid: Q0ShardedGrid | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    state_mean = np.asarray(normalizer["state_mean"], dtype=np.float32)
    state_std = np.asarray(normalizer["state_safe_std"], dtype=np.float32)
    rel_mean = np.float32(normalizer["relative_delta_mean"])
    rel_std = np.float32(normalizer["relative_delta_safe_std"])
    state = (data["history_native"][indices] - state_mean[None, None]) / state_std[None, None]
    relative = (data["relative_delta"][indices] - rel_mean) / rel_std
    prefix: dict[str, torch.Tensor] = {
        "state": torch.as_tensor(state, device=device, dtype=torch.float32),
        "history_mask": torch.as_tensor(data["history_mask"][indices], device=device, dtype=torch.float32),
        "relative_delta": torch.as_tensor(relative, device=device, dtype=torch.float32),
        "anchor_command": torch.as_tensor(data["anchor_native"][indices], device=device, dtype=torch.float32),
    }
    if video_mode != "STATE-CTRL":
        if grid is None:
            raise RuntimeError("fusion prefix requested without the accepted Q0 cache")
        features = np.zeros((len(indices), 2, 16, 4, 768), dtype=np.float32)
        for pos, index in enumerate(indices.tolist()):
            features[pos] = grid.row(
                int(data["cache_index"][index]),
                str(data["anchor_id"][index]),
                str(data["episode_id"][index]),
            )
        v_mean = np.asarray(normalizer["video_mean"], dtype=np.float32)[None, None, None, :]
        v_std = np.asarray(normalizer["video_safe_std"], dtype=np.float32)[None, None, None, :]
        prefix["dynamic"] = torch.as_tensor((features[:, 0] - v_mean) / v_std, device=device, dtype=torch.float32)
        prefix["static"] = torch.as_tensor((features[:, 1] - v_mean) / v_std, device=device, dtype=torch.float32)
    task_rank = np.asarray(data["task_rank"][indices], dtype=np.int64)
    task_ids = np.asarray(data["task_id"][indices], dtype=np.int64)
    for local, task in enumerate(task_ids.tolist()):
        if task not in TASKS or int(task_rank[local]) != TASKS.index(int(task)):
            raise RuntimeError(f"task/rank input mismatch at receiver row {int(indices[local])}")
    anchor = torch.as_tensor(data["anchor_native"][indices], device=device, dtype=torch.float32)
    tasks = torch.as_tensor(task_rank, device=device, dtype=torch.long)
    validate_prefix_keys(prefix, video_mode)
    return prefix, tasks, anchor


def configure_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def position_fd_seam_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    metric_sigma: np.ndarray,
    metric_fd_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    sigma = np.asarray(metric_sigma, dtype=np.float64)
    fd_scale = np.asarray(metric_fd_scale, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[1:] != (30, 28) or target.shape != prediction.shape:
        raise ValueError(f"prediction/target must be [N,30,28], got {prediction.shape}/{target.shape}")
    if anchor.shape != (prediction.shape[0], 28) or sigma.shape != (28,) or fd_scale.shape != (28,):
        raise ValueError("metric anchor/scale shape mismatch")
    pos_error = np.abs(prediction - target) / sigma[None, None, :]
    position = 0.5 * (
        pos_error[:, :, :14].mean(axis=(1, 2)) + pos_error[:, :, 14:].mean(axis=(1, 2))
    )
    pred_fd = np.diff(prediction, axis=1) / fd_scale[None, None, :]
    target_fd = np.diff(target, axis=1) / fd_scale[None, None, :]
    fd_error = np.abs(pred_fd - target_fd)
    fd = 0.5 * (
        fd_error[:, :, :14].mean(axis=(1, 2)) + fd_error[:, :, 14:].mean(axis=(1, 2))
    )
    pred_seam = (prediction[:, 0] - anchor) / fd_scale[None, :]
    target_seam = (target[:, 0] - anchor) / fd_scale[None, :]
    seam_error = np.abs(pred_seam - target_seam)
    seam = 0.5 * (seam_error[:, :14].mean(axis=1) + seam_error[:, 14:].mean(axis=1))
    native_mae = np.abs(prediction - target).mean(axis=(1, 2))
    return position, fd, seam, native_mae


def distance_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    metric_sigma: np.ndarray,
    metric_fd_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[1:] != (30, 28):
        raise ValueError(f"distance inputs must match [N,30,28], got {prediction.shape}/{reference.shape}")
    sigma = np.asarray(metric_sigma, dtype=np.float64)
    fd_scale = np.asarray(metric_fd_scale, dtype=np.float64)
    error = np.abs(prediction - reference) / sigma[None, None, :]
    position = 0.5 * (error[:, :, :14].mean(axis=(1, 2)) + error[:, :, 14:].mean(axis=(1, 2)))
    delta = np.abs(np.diff(prediction - reference, axis=1)) / fd_scale[None, None, :]
    fd = 0.5 * (delta[:, :, :14].mean(axis=(1, 2)) + delta[:, :, 14:].mean(axis=(1, 2)))
    return position, fd


def task_episode_weights(data: dict[str, np.ndarray]) -> np.ndarray:
    n = len(data["anchor_id"])
    tasks = np.asarray(data["task_id"], dtype=np.int64)
    episodes = np.asarray(data["episode_id"], dtype=np.int64)
    unique_tasks = sorted(int(x) for x in np.unique(tasks))
    if tuple(unique_tasks) != tuple(sorted(TASKS)):
        raise RuntimeError(f"train task population mismatch: {unique_tasks}")
    weights = np.empty(n, dtype=np.float64)
    for task in unique_tasks:
        positions = np.flatnonzero(tasks == task)
        _, inverse, counts = np.unique(episodes[positions], return_inverse=True, return_counts=True)
        weights[positions] = n / (5.0 * len(counts) * counts[inverse])
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("task/episode weights are nonfinite or nonpositive")
    if not np.isclose(weights.sum(), n, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"task/episode weights sum {weights.sum()} differs from W={n}")
    return weights


def state_rng_hash(device: torch.device) -> str:
    data = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return sha_bytes(data)


def load_safe_prediction(path: Path, field: str = "prediction_native") -> np.ndarray:
    from safetensors.numpy import load_file
    tensors = load_file(str(path))
    if field not in tensors:
        raise RuntimeError(f"safe prediction bundle lacks field {field}: {path}")
    values = np.asarray(tensors[field])
    if values.ndim != 3 or values.shape[1:] != (30, 28) or not np.isfinite(values).all():
        raise RuntimeError(f"prediction shape/finite failure: {path}, {values.shape}")
    return values


def save_safe_prediction(path: Path, values: np.ndarray) -> None:
    from safetensors.numpy import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    save_file({"prediction_native": np.ascontiguousarray(values)}, str(temporary))
    os.replace(temporary, path)
