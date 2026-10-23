"""Pure CPU scorer adapter for the V8.1 post-result evidence supplement.

This module deliberately has no imports from the historical V7/V8 programs,
no torch/CUDA dependency, no training entry point, and no implicit file reads.
Callers pass only the already-bound metadata, arrays, and locked scales.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
METHODS = ("G", "U", "M", "T")
TARGET_FIELDS = (
    "anchor_id", "episode_id", "task_id", "source_cluster", "target_native_h30"
)
PREFIX_FIELDS = (
    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
    "anchor_frame", "anchor_native",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordered_id_hash(ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(str(x) for x in ids).encode("utf-8"))


def require_ordered_id_hash(ids: Sequence[str], expected_sha256: str, label: str) -> None:
    actual = ordered_id_hash(ids)
    if actual != expected_sha256:
        raise ValueError(f"{label}: ordered anchor ID hash mismatch: {actual} != {expected_sha256}")


def _safe_array(value: Any, name: str, *, numeric: bool = False) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise ValueError(f"{name}: object/pickle dtype is forbidden")
    if numeric and arr.dtype.kind not in "iuf":
        raise ValueError(f"{name}: expected numeric dtype, got {arr.dtype}")
    if numeric and not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values")
    return arr


def _text(value: Any, name: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict")
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise ValueError(f"{name}: expected UTF-8 text, got {type(value).__name__}")


def text_vector(values: Any, name: str) -> list[str]:
    arr = _safe_array(values, name)
    if arr.ndim != 1 or arr.dtype.kind not in "US":
        raise ValueError(f"{name}: expected a 1-D Unicode/bytes vector, got {arr.shape}/{arr.dtype}")
    return [_text(v, name) for v in arr.tolist()]


def read_npz_fields(path: str | Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    """Read exactly the named fields with pickle loading disabled."""
    path = Path(path)
    wanted = tuple(fields)
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("field allowlist must be nonempty and unique")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(wanted) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        for key in wanted:
            arr = np.asarray(archive[key])
            if arr.dtype.hasobject:
                raise ValueError(f"{path}:{key}: object/pickle dtype is forbidden")
            result[key] = np.array(arr, copy=True)
    return result


def validate_metadata(metadata: Mapping[str, Any], *, expected_tasks: Sequence[int] | None = None) -> dict[str, Any]:
    missing = sorted(set(PREFIX_FIELDS) - set(metadata))
    if missing:
        raise ValueError(f"prefix metadata missing fields {missing}")
    ids = text_vector(metadata["anchor_id"], "anchor_id")
    n = len(ids)
    if n == 0 or len(set(ids)) != n:
        raise ValueError("anchor_id must be nonempty and unique")
    episode = _safe_array(metadata["episode_id"], "episode_id", numeric=True)
    task = _safe_array(metadata["task_id"], "task_id", numeric=True)
    rank = _safe_array(metadata["task_rank"], "task_rank", numeric=True)
    frame = _safe_array(metadata["anchor_frame"], "anchor_frame", numeric=True)
    groups = text_vector(metadata["source_cluster"], "source_cluster")
    anchor = _safe_array(metadata["anchor_native"], "anchor_native", numeric=True).astype(np.float64, copy=False)
    for name, arr in (("episode_id", episode), ("task_id", task), ("task_rank", rank), ("anchor_frame", frame)):
        if arr.shape != (n,):
            raise ValueError(f"{name}: expected {(n,)}, got {arr.shape}")
    if len(groups) != n or anchor.shape != (n, 28):
        raise ValueError(f"metadata row/anchor shape mismatch: n={n}, groups={len(groups)}, anchor={anchor.shape}")
    tasks = tuple(sorted(set(int(x) for x in task.tolist())))
    if expected_tasks is not None and tasks != tuple(sorted(int(x) for x in expected_tasks)):
        raise ValueError(f"task support mismatch: expected {tuple(sorted(expected_tasks))}, got {tasks}")
    for i, t in enumerate(task.tolist()):
        if int(t) not in TASKS or int(rank[i]) != TASKS.index(int(t)):
            raise ValueError(f"task rank/order mismatch at row {i}: task={t}, rank={rank[i]}")
    return {
        "anchor_id": ids,
        "episode_id": episode.astype(np.int64, copy=False),
        "task_id": task.astype(np.int64, copy=False),
        "task_rank": rank.astype(np.int64, copy=False),
        "source_cluster": groups,
        "anchor_frame": frame.astype(np.int64, copy=False),
        "anchor_native": anchor,
    }


def validate_scales(sigma: Any, fd_scale: Any) -> tuple[np.ndarray, np.ndarray]:
    sigma = _safe_array(sigma, "target_sigma", numeric=True).astype(np.float64, copy=False)
    fd_scale = _safe_array(fd_scale, "safe_fd_scale", numeric=True).astype(np.float64, copy=False)
    if sigma.shape != (28,) or fd_scale.shape != (28,):
        raise ValueError(f"scales must both be [28], got {sigma.shape}/{fd_scale.shape}")
    if np.any(sigma <= 0) or np.any(fd_scale <= 0):
        raise ValueError("all locked scale values must be strictly positive")
    return sigma, fd_scale


def align_target_to_metadata(target_bundle: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Join target rows to sealed receiver metadata by ID and locked metadata."""
    meta = validate_metadata(metadata)
    target_ids = text_vector(target_bundle["anchor_id"], "target.anchor_id")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target archive contains duplicate anchor IDs")
    episode = _safe_array(target_bundle["episode_id"], "target.episode_id", numeric=True)
    task = _safe_array(target_bundle["task_id"], "target.task_id", numeric=True)
    groups = text_vector(target_bundle["source_cluster"], "target.source_cluster")
    target = _safe_array(target_bundle["target_native_h30"], "target_native_h30", numeric=True)
    n = len(target_ids)
    if target.shape != (n, 30, 28) or episode.shape != (n,) or task.shape != (n,) or len(groups) != n:
        raise ValueError("target archive ID/metadata/target shapes are inconsistent")
    expected = set(meta["anchor_id"])
    if set(target_ids) != expected:
        raise ValueError("target IDs differ from the locked dev receiver IDs")
    lookup = {anchor_id: i for i, anchor_id in enumerate(target_ids)}
    aligned: list[np.ndarray] = []
    for i, anchor_id in enumerate(meta["anchor_id"]):
        j = lookup[anchor_id]
        if (
            int(episode[j]) != int(meta["episode_id"][i])
            or int(task[j]) != int(meta["task_id"][i])
            or groups[j] != meta["source_cluster"][i]
        ):
            raise ValueError(f"target metadata mismatch for anchor {anchor_id}")
        aligned.append(target[j])
    return np.stack(aligned).astype(np.float64, copy=False), target_ids


def align_prediction_by_id(prediction: Any, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> np.ndarray:
    """Validate a prediction row map, then align its values to metadata order."""
    meta = validate_metadata(metadata)
    pred = _safe_array(prediction, "prediction", numeric=True)
    if pred.shape != (len(rows), 30, 28) or not np.isfinite(pred).all():
        raise ValueError(f"prediction must be finite [rows,30,28], got {pred.shape}")
    parsed: dict[str, tuple[tuple[int, int, int, int, str], np.ndarray]] = {}
    for i, row in enumerate(rows):
        anchor_id = _text(row.get("anchor_id"), "prediction_rows.anchor_id")
        if anchor_id in parsed:
            raise ValueError(f"duplicate prediction anchor ID {anchor_id}")
        task = int(row["task_id"])
        rank_value = row.get("task_rank", "")
        rank = int(rank_value) if rank_value not in (None, "") else TASKS.index(task)
        group = row.get("source_cluster", row.get("source_group", ""))
        parsed[anchor_id] = (
            (int(row["episode_id"]), task, int(row["anchor_frame"]), rank, str(group)), pred[i]
        )
    if set(parsed) != set(meta["anchor_id"]):
        missing = sorted(set(meta["anchor_id"]) - set(parsed))[:3]
        extra = sorted(set(parsed) - set(meta["anchor_id"]))[:3]
        raise ValueError(f"prediction ID set mismatch; missing={missing}, extra={extra}")
    aligned: list[np.ndarray] = []
    for i, anchor_id in enumerate(meta["anchor_id"]):
        row_meta, values = parsed[anchor_id]
        expected = (
            int(meta["episode_id"][i]), int(meta["task_id"][i]), int(meta["anchor_frame"][i]),
            int(meta["task_rank"][i]), meta["source_cluster"][i],
        )
        if row_meta != expected:
            raise ValueError(f"prediction receiver metadata mismatch for anchor {anchor_id}")
        aligned.append(values)
    return np.stack(aligned).astype(np.float64, copy=False)


def elementwise_errors(
    prediction: Any,
    target: Any,
    anchor: Any,
    sigma: Any,
    fd_scale: Any,
) -> dict[str, np.ndarray]:
    """Canonical absolute scaled errors: future position, 29 internal FD, seam."""
    pred = _safe_array(prediction, "prediction", numeric=True).astype(np.float64, copy=False)
    truth = _safe_array(target, "target", numeric=True).astype(np.float64, copy=False)
    anchor = _safe_array(anchor, "anchor", numeric=True).astype(np.float64, copy=False)
    sigma, fd_scale = validate_scales(sigma, fd_scale)
    if pred.ndim != 3 or pred.shape[1:] != (30, 28) or truth.shape != pred.shape:
        raise ValueError(f"prediction and target must match [N,30,28], got {pred.shape}/{truth.shape}")
    if anchor.shape != (pred.shape[0], 28):
        raise ValueError(f"anchor must be [N,28], got {anchor.shape}")
    pos = np.abs(pred - truth) / sigma[None, None, :]
    pred_fd = np.diff(pred, axis=1) / fd_scale[None, None, :]
    true_fd = np.diff(truth, axis=1) / fd_scale[None, None, :]
    internal_fd = np.abs(pred_fd - true_fd)
    pred_seam = (pred[:, 0, :] - anchor) / fd_scale[None, :]
    true_seam = (truth[:, 0, :] - anchor) / fd_scale[None, :]
    seam = np.abs(pred_seam - true_seam)
    return {"position": pos, "internal_fd": internal_fd, "seam": seam}


def canonical_per_window(errors: Mapping[str, np.ndarray], prediction: Any, target: Any) -> dict[str, np.ndarray]:
    """V7 two-coordinate-block scorer, with seam separate from internal FD."""
    pred = _safe_array(prediction, "prediction", numeric=True).astype(np.float64, copy=False)
    truth = _safe_array(target, "target", numeric=True).astype(np.float64, copy=False)
    p = errors["position"]
    f = errors["internal_fd"]
    s = errors["seam"]
    if pred.shape != truth.shape or pred.shape != (len(p), 30, 28):
        raise ValueError("canonical per-window input shape mismatch")
    position = 0.5 * (p[:, :, :14].mean(axis=(1, 2)) + p[:, :, 14:].mean(axis=(1, 2)))
    safe_fd = 0.5 * (f[:, :, :14].mean(axis=(1, 2)) + f[:, :, 14:].mean(axis=(1, 2)))
    seam = 0.5 * (s[:, :14].mean(axis=1) + s[:, 14:].mean(axis=1))
    native = np.abs(pred - truth).mean(axis=(1, 2))
    return {"position_mae": position, "safe_fd_l1": safe_fd, "seam_safe_fd_l1": seam, "native_mae": native}


def canonical_per_window_from_errors(errors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Recover the three accepted per-window diagnostics from sufficient errors."""
    p = _safe_array(errors["position"], "position_error", numeric=True).astype(np.float64, copy=False)
    f = _safe_array(errors["internal_fd"], "internal_fd_error", numeric=True).astype(np.float64, copy=False)
    s = _safe_array(errors["seam"], "seam_error", numeric=True).astype(np.float64, copy=False)
    if p.ndim != 3 or p.shape[1:] != (30, 28) or f.shape != (p.shape[0], 29, 28) or s.shape != (p.shape[0], 28):
        raise ValueError(f"error checkpoint shapes invalid: position={p.shape}, fd={f.shape}, seam={s.shape}")
    return {
        "position_mae": 0.5 * (p[:, :, :14].mean(axis=(1,2)) + p[:, :, 14:].mean(axis=(1,2))),
        "safe_fd_l1": 0.5 * (f[:, :, :14].mean(axis=(1,2)) + f[:, :, 14:].mean(axis=(1,2))),
        "seam_safe_fd_l1": 0.5 * (s[:, :14].mean(axis=1) + s[:, 14:].mean(axis=1)),
    }


def _scope_mean(values: np.ndarray, metadata: Mapping[str, Any], scope: int | str) -> tuple[np.ndarray, dict[str, int]]:
    meta = validate_metadata(metadata)
    arr = _safe_array(values, "scope_values", numeric=True).astype(np.float64, copy=False)
    if arr.ndim < 1 or arr.shape[0] != len(meta["anchor_id"]):
        raise ValueError(f"scope values must have leading support {len(meta['anchor_id'])}, got {arr.shape}")
    task_values = [int(scope)] if scope != "ALL" else list(TASKS)
    by_task: list[np.ndarray] = []
    n_anchors = n_episodes = n_groups = 0
    for task_id in task_values:
        idx = np.flatnonzero(meta["task_id"] == task_id)
        if not len(idx):
            raise ValueError(f"no support for task {task_id}")
        episodes = sorted(set(int(x) for x in meta["episode_id"][idx].tolist()))
        episode_means = [arr[idx[meta['episode_id'][idx] == episode]].mean(axis=0) for episode in episodes]
        by_task.append(np.stack(episode_means).mean(axis=0))
        n_anchors += len(idx)
        n_episodes += len(episodes)
        n_groups += len(set(meta["source_cluster"][i] for i in idx.tolist()))
    result = np.stack(by_task).mean(axis=0)
    return result, {"support": n_anchors, "support_episodes": n_episodes, "support_source_groups": n_groups, "support_tasks": len(task_values)}


def aggregate_slot(
    method: str,
    seed: int,
    metadata: Mapping[str, Any],
    errors: Mapping[str, np.ndarray],
    direct: Mapping[str, np.ndarray],
    scale_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create scalar long-format horizon, axis, and reconciliation records."""
    if method not in METHODS or seed not in SEEDS:
        raise ValueError(f"unlocked method/seed: {method}/{seed}")
    meta = validate_metadata(metadata, expected_tasks=TASKS)
    horizon_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    horizon_sources = (
        ("position", 0.5 * (errors["position"][:, :, :14].mean(axis=2) + errors["position"][:, :, 14:].mean(axis=2)), 30),
        ("internal_fd", 0.5 * (errors["internal_fd"][:, :, :14].mean(axis=2) + errors["internal_fd"][:, :, 14:].mean(axis=2)), 29),
    )
    axis_sources = (
        ("position", errors["position"].mean(axis=1)),
        ("internal_fd", errors["internal_fd"].mean(axis=1)),
        ("seam", errors["seam"]),
    )
    scopes: list[int | str] = [*TASKS, "ALL"]
    direct_names = {"position": "position_mae", "internal_fd": "safe_fd_l1", "seam": "seam_safe_fd_l1"}
    for scope in scopes:
        scope_name = "ALL" if scope == "ALL" else str(scope)
        direct_values: dict[str, float] = {}
        for metric_name, direct_name in direct_names.items():
            value, support = _scope_mean(direct[direct_name], meta, scope)
            direct_values[metric_name] = float(value)
        for metric_name, values, length in horizon_sources:
            vector, support = _scope_mean(values, meta, scope)
            direct_value = direct_values[metric_name]
            for i in range(length):
                label = f"{i+1}->{i+2}" if metric_name == "internal_fd" else i + 1
                horizon_rows.append({
                    "method": method, "seed": seed, "task_or_ALL": scope_name,
                    "metric": metric_name, "step_or_transition": label,
                    "value": float(vector[i]), "support": support["support"],
                    "support_episodes": support["support_episodes"],
                    "support_source_groups": support["support_source_groups"],
                    "support_tasks": support["support_tasks"],
                    "scale_hash": scale_hash,
                })
            step_reconstructed = float(np.mean(vector))
            axis_vector, axis_support = _scope_mean(
                errors["position" if metric_name == "position" else "internal_fd"].mean(axis=1), meta, scope
            )
            axis_reconstructed = float(np.mean(axis_vector))
            macro_rows.append({
                "method": method, "seed": seed, "task_or_ALL": scope_name,
                "metric": direct_names[metric_name], "canonical_direct_value": direct_value,
                "horizon_reconstructed_value": step_reconstructed,
                "axis_reconstructed_value": axis_reconstructed,
                "max_internal_abs_diff": max(abs(direct_value-step_reconstructed), abs(direct_value-axis_reconstructed)),
                "support": support["support"], "support_episodes": support["support_episodes"],
                "support_source_groups": support["support_source_groups"],
                "support_tasks": support["support_tasks"],
                "scale_hash": scale_hash,
            })
        for metric_name, values in axis_sources:
            vector, support = _scope_mean(values, meta, scope)
            for d, v in enumerate(vector.tolist()):
                axis_rows.append({
                    "method": method, "seed": seed, "task_or_ALL": scope_name,
                    "metric": metric_name, "axis": d, "value": float(v),
                    "support": support["support"], "support_episodes": support["support_episodes"],
                    "support_source_groups": support["support_source_groups"], "support_tasks": support["support_tasks"],
                    "scale_hash": scale_hash,
                })
        # Seam has no horizon breakdown; reconcile its canonical and 28-axis means.
        seam_axis, seam_support = _scope_mean(errors["seam"], meta, scope)
        macro_rows.append({
            "method": method, "seed": seed, "task_or_ALL": scope_name,
            "metric": "seam_safe_fd_l1", "canonical_direct_value": direct_values["seam"],
            "horizon_reconstructed_value": "",
            "axis_reconstructed_value": float(np.mean(seam_axis)),
            "max_internal_abs_diff": abs(direct_values["seam"] - float(np.mean(seam_axis))),
            "support": seam_support["support"], "support_episodes": seam_support["support_episodes"],
            "support_source_groups": seam_support["support_source_groups"],
            "support_tasks": seam_support["support_tasks"],
            "scale_hash": scale_hash,
        })
    return horizon_rows, axis_rows, macro_rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"cannot infer columns from empty rows: {path}")
        fields = list(rows[0])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            for key, value in normalized.items():
                if isinstance(value, np.ndarray) or isinstance(value, np.generic) and np.asarray(value).ndim != 0:
                    raise TypeError(f"CSV cell {key} must be scalar, got {type(value).__name__}")
            writer.writerow(normalized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_npy(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("xb") as handle:
        np.save(handle, _safe_array(array, path.name), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_sealed_cache(cache_dir: str | Path, y: np.ndarray, meta: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Persist labels and receiver metadata atomically; write the seal last."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    validated = validate_metadata(meta, expected_tasks=TASKS)
    y = _safe_array(y, "Y_dev", numeric=True).astype(np.float64, copy=False)
    if y.shape != (len(validated["anchor_id"]), 30, 28):
        raise ValueError(f"Y_dev cache shape mismatch: {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("Y_dev cache is non-finite")
    ids_path = cache_dir / "ordered_anchor_ids.npy"
    y_path = cache_dir / "Y_dev.npy"
    metadata_path = cache_dir / "anchor_metadata.npz"
    _atomic_npy(ids_path, np.asarray(validated["anchor_id"], dtype=np.str_))
    _atomic_npy(y_path, y)
    temp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.part")
    with temp.open("xb") as handle:
        np.savez_compressed(
            handle,
            episode_id=validated["episode_id"], task_id=validated["task_id"],
            task_rank=validated["task_rank"], source_cluster=np.asarray(validated["source_cluster"], dtype=np.str_),
            anchor_frame=validated["anchor_frame"], anchor_native=validated["anchor_native"],
        )
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, metadata_path)
    file_hashes = {p.name: sha256_file(p) for p in (ids_path, y_path, metadata_path)}
    seal = {
        "schema_version": 1, "status": "PASS", "cache_role": "SCORER_ONLY_EXPOSED_DEV_LABELS",
        "source_role": "dev_target_archive", "source_split": "dev", "source_archive_load_only": True,
        "source_archive_path": provenance["source_archive_path"],
        "source_archive_sha256": provenance["source_archive_sha256"],
        "source_target_field": provenance["source_target_field"],
        "source_target_dtype": provenance["source_target_dtype"],
        "source_target_shape": provenance["source_target_shape"],
        "source_ordered_id_sha256": provenance["source_ordered_id_sha256"],
        "ordered_id_sha256": ordered_id_hash(validated["anchor_id"]),
        "ordered_anchor_ids": validated["anchor_id"],
        "unique_dev_windows": len(validated["anchor_id"]),
        "target_cache_dtype": str(y.dtype), "target_cache_shape": list(y.shape),
        "target_values_sha256": sha256_bytes(np.ascontiguousarray(y).tobytes()),
        "reader_version": provenance["reader_version"],
        "reader_access_event_id": provenance["reader_access_event_id"],
        "reader_allow_pickle": False, "reader_fields": list(TARGET_FIELDS),
        "files": file_hashes,
        "training_or_forward_path": "FORBIDDEN",
    }
    seal_path = cache_dir / "CACHE_SEAL.json"
    atomic_json(seal_path, seal)
    os.chmod(ids_path, 0o600); os.chmod(y_path, 0o600); os.chmod(metadata_path, 0o600); os.chmod(seal_path, 0o600)
    return seal


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_sealed_cache(cache_dir: str | Path, expected_id_hash: str | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    cache_dir = Path(cache_dir)
    seal = json.loads((cache_dir / "CACHE_SEAL.json").read_text(encoding="utf-8"))
    if seal.get("status") != "PASS" or seal.get("cache_role") != "SCORER_ONLY_EXPOSED_DEV_LABELS":
        raise ValueError("cache seal is not an accepted scorer-only dev-label cache")
    ids_path, y_path, metadata_path = (cache_dir / "ordered_anchor_ids.npy", cache_dir / "Y_dev.npy", cache_dir / "anchor_metadata.npz")
    for path in (ids_path, y_path, metadata_path):
        if sha256_file(path) != seal["files"].get(path.name):
            raise ValueError(f"cache file hash mismatch: {path.name}")
    ids = text_vector(np.load(ids_path, allow_pickle=False), "cache.anchor_id")
    if ordered_id_hash(ids) != seal["ordered_id_sha256"] or ids != seal["ordered_anchor_ids"]:
        raise ValueError("cache ordered ID seal mismatch")
    if expected_id_hash and ordered_id_hash(ids) != expected_id_hash:
        raise ValueError("cache IDs differ from the locked accepted dev IDs")
    y = _safe_array(np.load(y_path, allow_pickle=False), "cache.Y_dev", numeric=True).astype(np.float64, copy=False)
    if y.shape != tuple(seal["target_cache_shape"]) or y.shape != (len(ids), 30, 28):
        raise ValueError("cache target shape mismatch")
    if sha256_bytes(np.ascontiguousarray(y).tobytes()) != seal["target_values_sha256"]:
        raise ValueError("cache target values hash mismatch")
    with np.load(metadata_path, allow_pickle=False) as archive:
        md = {key: np.array(archive[key], copy=True) for key in archive.files}
    md["anchor_id"] = np.asarray(ids, dtype=np.str_)
    md = validate_metadata(md, expected_tasks=TASKS)
    return y, md, seal


def write_error_checkpoint(path: str | Path, ids_hash: str, arrays: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("xb") as handle:
        np.savez_compressed(handle, **{k: _safe_array(v, k, numeric=True) for k, v in arrays.items()})
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    record = {
        "status": "PASS", "path": str(path), "sha256": sha256_file(path),
        "ordered_id_sha256": ids_hash,
        "array_shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
        "array_value_sha256": {k: sha256_bytes(np.ascontiguousarray(v).tobytes()) for k, v in arrays.items()},
        **dict(provenance),
    }
    os.chmod(path, 0o600)
    return record


def load_error_checkpoint(path: str | Path, record: Mapping[str, Any], expected_ids_hash: str) -> dict[str, np.ndarray]:
    path = Path(path)
    if sha256_file(path) != record.get("sha256") or record.get("ordered_id_sha256") != expected_ids_hash:
        raise ValueError(f"checkpoint seal mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: _safe_array(archive[key], key, numeric=True).copy() for key in archive.files}
    for key, arr in result.items():
        if list(arr.shape) != record["array_shapes"].get(key):
            raise ValueError(f"checkpoint shape mismatch for {key}")
        if sha256_bytes(np.ascontiguousarray(arr).tobytes()) != record["array_value_sha256"].get(key):
            raise ValueError(f"checkpoint value hash mismatch for {key}")
    return result

