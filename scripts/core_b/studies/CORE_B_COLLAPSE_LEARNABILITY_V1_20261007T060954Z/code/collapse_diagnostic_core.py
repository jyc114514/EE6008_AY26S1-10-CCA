"""Pure CPU helpers for the train-only D3 collapse/learnability diagnosis."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch


class PanelSelectionError(RuntimeError):
    pass


def reconstruct_native(anchor_native: torch.Tensor, residual_standardized: torch.Tensor,
                       target_sigma: torch.Tensor) -> torch.Tensor:
    if residual_standardized.ndim != 3 or residual_standardized.shape[-1] != 28:
        raise ValueError("residual_standardized must be [batch,time,28]")
    if anchor_native.shape != (residual_standardized.shape[0], 28) or target_sigma.shape != (28,):
        raise ValueError("anchor/sigma shape does not match residual")
    if not torch.isfinite(anchor_native).all() or not torch.isfinite(residual_standardized).all():
        raise ValueError("anchor/residual contains non-finite values")
    if not torch.isfinite(target_sigma).all() or torch.any(target_sigma <= 0):
        raise ValueError("target_sigma must be finite and positive")
    return anchor_native[:, None, :] + target_sigma[None, None, :] * residual_standardized


def loss_components(pred_residual: torch.Tensor, target_native: torch.Tensor,
                    anchor_native: torch.Tensor, target_sigma: torch.Tensor,
                    safe_fd_scale: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Return exact D3-EQ per-row position and future-internal FD components."""
    if pred_residual.ndim != 3 or pred_residual.shape[-1] != 28 or pred_residual.shape[1] < 2:
        raise ValueError("pred_residual must be [batch,time>=2,28]")
    if target_native.shape != pred_residual.shape:
        raise ValueError("target/prediction shapes differ")
    if anchor_native.shape != (pred_residual.shape[0], 28):
        raise ValueError("anchor_native must be [batch,28]")
    if target_sigma.shape != (28,) or safe_fd_scale.shape != (28,):
        raise ValueError("normalizers must both be [28]")
    if not torch.isfinite(target_native).all() or not torch.isfinite(pred_residual).all():
        raise ValueError("target/prediction contains non-finite values")
    if not torch.isfinite(safe_fd_scale).all() or torch.any(safe_fd_scale <= 0):
        raise ValueError("safe_fd_scale must be finite and positive")
    target_residual = (target_native - anchor_native[:, None, :]) / target_sigma[None, None, :]
    position_error = torch.abs(pred_residual - target_residual)
    position = 0.5 * (position_error[..., :14].mean(dim=(1, 2)) + position_error[..., 14:].mean(dim=(1, 2)))
    pred_native = reconstruct_native(anchor_native, pred_residual, target_sigma)
    target_delta = torch.diff(target_native, dim=1) / safe_fd_scale[None, None, :]
    pred_delta = torch.diff(pred_native, dim=1) / safe_fd_scale[None, None, :]
    fd_error = torch.abs(pred_delta - target_delta)
    fd = 0.5 * (fd_error[..., :14].mean(dim=(1, 2)) + fd_error[..., 14:].mean(dim=(1, 2)))
    weighted_fd = 0.1 * fd
    return {"position": position, "fd": fd, "weighted_fd": weighted_fd,
            "full": position + weighted_fd, "internal_difference_count": int(target_native.shape[1] - 1)}


def trajectory_decomposition(pred_native: torch.Tensor, anchor_native: torch.Tensor,
                             target_sigma: torch.Tensor | None = None,
                             tolerance: float = 1e-8) -> dict[str, Any]:
    """Separate within-sample temporal variation from between-sample level offsets."""
    if pred_native.ndim != 3 or pred_native.shape[-1] != 28:
        raise ValueError("pred_native must be [batch,time,28]")
    if anchor_native.shape != (pred_native.shape[0], 28):
        raise ValueError("anchor_native must be [batch,28]")
    if tolerance < 0 or not torch.isfinite(pred_native).all():
        raise ValueError("invalid tolerance or non-finite prediction")
    offset = pred_native - anchor_native[:, None, :]
    time_mean_offset = offset.mean(dim=1)
    within_std_by_sample = offset.std(dim=1, unbiased=False).mean(dim=1)
    between_mean_offset = time_mean_offset.std(dim=0, unbiased=False)
    sample_tv = torch.abs(torch.diff(pred_native, dim=1)).mean(dim=(1, 2))
    result = {
        "anchor_copy": bool(torch.max(torch.abs(offset)).item() <= tolerance),
        "anchor_copy_max_abs_offset": float(torch.max(torch.abs(offset)).item()),
        "time_constant_sample_fraction": float((within_std_by_sample <= tolerance).float().mean().item()),
        "within_sample_time_std_mean": float(within_std_by_sample.mean().item()),
        "between_sample_mean_offset_std": float(between_mean_offset.mean().item()),
        "between_sample_mean_offset_abs_std": float(between_mean_offset.mean().item()),
        "predicted_future_internal_tv_mean": float(sample_tv.mean().item()),
        "time_mean_offset_by_joint": time_mean_offset.mean(dim=0).detach().cpu().tolist(),
    }
    if target_sigma is not None:
        if target_sigma.shape != (28,) or not torch.isfinite(target_sigma).all() or torch.any(target_sigma <= 0):
            raise ValueError("target_sigma must be finite, positive and [28]")
        normalized_offset = offset / target_sigma[None, None, :]
        result["within_sample_time_std_normalized_mean"] = float(
            normalized_offset.std(dim=1, unbiased=False).mean().item())
        result["between_sample_mean_offset_normalized_std"] = float(
            normalizedized_std(normalized_offset.mean(dim=1)).item())
    return result


def normalizedized_std(values: torch.Tensor) -> torch.Tensor:
    """Mean marginal standard deviation, named separately to avoid hidden pooling."""
    return values.std(dim=0, unbiased=False).mean()


def stable_id_hash(protocol_id: str, anchor_id: str) -> str:
    return hashlib.sha256(f"{protocol_id}\0{anchor_id}".encode("utf-8")).hexdigest()


def stable_group_hash(protocol_id: str, group_id: str) -> str:
    return hashlib.sha256(f"{protocol_id}\0group\0{group_id}".encode("utf-8")).hexdigest()


def select_stratified_groups(records: list[dict[str, Any]], quotas: dict[str, int],
                             protocol_id: str) -> list[dict[str, Any]]:
    """Choose one window per source group using only stratum and stable hashes."""
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        stratum = str(record["stratum"])
        group = str(record["source_group_id"])
        by_stratum[stratum][group].append(record)
    selected: list[dict[str, Any]] = []
    for stratum in ("low", "mid", "high"):
        quota = int(quotas.get(stratum, 0))
        groups = by_stratum.get(stratum, {})
        if quota > len(groups):
            raise PanelSelectionError(f"stratum {stratum} has {len(groups)} groups, needs {quota}")
        group_order = sorted(groups, key=lambda g: stable_group_hash(protocol_id, g))
        for group in group_order[:quota]:
            row = min(groups[group], key=lambda x: stable_id_hash(protocol_id, str(x["anchor_id"])))
            selected.append({**row, "selection_hash": stable_id_hash(protocol_id, str(row["anchor_id"])),
                             "selection_reason": f"lowest stable ID hash within selected {stratum} group"})
    if len(selected) != sum(int(v) for v in quotas.values()):
        raise PanelSelectionError("stratified fixture count mismatch")
    if len({str(x["source_group_id"]) for x in selected}) != len(selected):
        raise PanelSelectionError("fixture source groups are not unique")
    return selected


def exact_input_signature(fields: dict[str, Any]) -> str:
    """Hash actual model-input arrays with dtype/shape; excludes metadata and targets."""
    h = hashlib.sha256()
    for name in sorted(fields):
        value = fields[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        h.update(name.encode("utf-8"));h.update(b"\0")
        h.update(str(array.dtype).encode("ascii"));h.update(b"\0")
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def choose_nested_high_panel(p1: dict[str, Any], candidates: list[dict[str, Any]],
                             n_total: int, protocol_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily fill P8 with unique high-motion groups, preferring new tasks and excluding exact input collisions."""
    if n_total < 1 or str(p1.get("stratum")) != "high":
        raise PanelSelectionError("P1 must be a high-motion row and panel size positive")
    selected = [{**p1, "selection_hash": stable_id_hash(protocol_id, str(p1["anchor_id"])),
                 "selection_reason": "locked P1; deterministic train high-stratum ID hash"}]
    groups = {str(p1["source_group_id"])}
    tasks = {int(p1["original_task_index"])}
    rejected: list[dict[str, Any]] = []
    pending = [x for x in candidates if str(x["source_group_id"]) not in groups and str(x.get("stratum")) == "high"]
    while len(selected) < n_total:
        feasible = []
        for row in pending:
            if str(row["source_group_id"]) in groups:
                rejected.append({"anchor_id": row["anchor_id"], "source_group_id": row["source_group_id"], "reason": "duplicate_source_group"})
                continue
            collision = False
            for previous in selected:
                for model_key in ("signature_TS11-D", "signature_STATE-CTRL"):
                    if row[model_key] == previous[model_key]:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                rejected.append({"anchor_id": row["anchor_id"], "source_group_id": row["source_group_id"], "reason": "exact_input_collision"})
                continue
            task = int(row["original_task_index"])
            feasible.append((task in tasks, stable_id_hash(protocol_id, str(row["anchor_id"])), row))
        if not feasible:
            raise PanelSelectionError(f"only {len(selected)}/{n_total} collision-free high-motion groups available")
        novel_task, rank, chosen = min(feasible, key=lambda item: (item[0], item[1]))
        selected.append({**chosen, "selection_hash": rank,
                         "selection_reason": "next stable ID hash; prioritized previously unseen task" if not novel_task else "stable ID hash within available task"})
        groups.add(str(chosen["source_group_id"]))
        tasks.add(int(chosen["original_task_index"]))
        pending = [x for x in pending if str(x["anchor_id"]) != str(chosen["anchor_id"])]
    return selected, rejected
