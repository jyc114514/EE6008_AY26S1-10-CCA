#!/usr/bin/env python3
"""One resumable V2 fit. The model forward receives prefix inputs and task ID only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LOCKED = CODE / "locked_source"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(LOCKED))

from d3_eq_contract import episode_equal_weights, per_sample_loss, shuffled_epoch_order  # noqa: E402
from prepare_v2_data import Q0ShardedGrid  # noqa: E402
from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash  # noqa: E402
from v2_scheduler import UpdateDrivenCosineAnnealingLR, write_lr_trace  # noqa: E402


TASK_IDS = (119, 94, 90, 141, 230)
TASK_RANK = {task_id: index for index, task_id in enumerate(TASK_IDS)}
EFFECTIVE_BATCH = 64
TARGET_DIM = 28
HORIZON = 30
CHECKPOINTS_SCREEN = (40, 80, 160)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    safe = {name: value.detach().to(device="cpu").contiguous() for name, value in tensors.items()}
    save_file(safe, str(temporary))
    os.replace(temporary, path)


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def load_npz() -> dict[str, np.ndarray]:
    path = ROOT / "data/TRAIN_SELECTED_H30.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def safe_load_init(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


def scoped_indices(data: dict[str, np.ndarray], job: dict[str, Any], *, tune: bool = False) -> np.ndarray:
    mask = np.ones(len(data["anchor_id"]), dtype=bool)
    if job["phase"] == "screening":
        role = "inner_tune" if tune else "inner_fit"
        mask &= data["inner_role"] == role
    elif tune:
        return np.empty(0, dtype=np.int64)
    if job["scope"] == "expert":
        mask &= data["task_id"] == int(job["task_id"])
    elif job["scope"] == "shared":
        mask &= np.isin(data["task_id"], np.asarray(TASK_IDS, dtype=np.int64))
    else:
        raise ValueError(f"invalid training scope {job['scope']}")
    return np.flatnonzero(mask).astype(np.int64)


def episode_task_weights(data: dict[str, np.ndarray], indices: np.ndarray, scope: str) -> np.ndarray:
    if not len(indices):
        raise ValueError("cannot weight an empty training population")
    episodes = data["episode_id"][indices].astype(np.int64)
    tasks = data["task_id"][indices].astype(np.int64)
    unique_tasks = sorted(int(x) for x in np.unique(tasks))
    k = len(unique_tasks) if scope == "shared" else 1
    if scope == "shared" and k != 5:
        raise RuntimeError(f"shared model does not contain all five tasks: {unique_tasks}")
    weights = np.empty(len(indices), dtype=np.float64)
    for task_id in unique_tasks:
        local_positions = np.flatnonzero(tasks == task_id)
        local_episode = episodes[local_positions]
        unique_episode, inverse, counts = np.unique(local_episode, return_inverse=True, return_counts=True)
        episode_count = len(unique_episode)
        w_total = len(indices)
        weights[local_positions] = w_total / (k * episode_count * counts[inverse])
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("invalid episode/task sample weights")
    # Keep canonical sample weights in float64 for population-sum checks and bookkeeping.
    # Each actual microbatch loss explicitly casts these values to FP32 at the model boundary.
    return weights.astype(np.float64)


def metric_components(
    pred_residual: torch.Tensor,
    target_native: torch.Tensor,
    anchor_native: torch.Tensor,
    target_sigma: torch.Tensor,
    safe_fd_scale: torch.Tensor,
    metric_sigma: torch.Tensor | None = None,
    metric_safe_fd_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_native = anchor_native[:, None] + target_sigma[None, None] * pred_residual
    eval_sigma = target_sigma if metric_sigma is None else metric_sigma
    eval_fd_scale = safe_fd_scale if metric_safe_fd_scale is None else metric_safe_fd_scale
    position_error = torch.abs(pred_native - target_native) / eval_sigma[None, None]
    position = 0.5 * (
        position_error[..., :14].mean(dim=(1, 2)) + position_error[..., 14:].mean(dim=(1, 2))
    )
    target_fd = torch.diff(target_native, dim=1) / eval_fd_scale[None, None]
    pred_fd = torch.diff(pred_native, dim=1) / eval_fd_scale[None, None]
    fd_error = torch.abs(pred_fd - target_fd)
    fd = 0.5 * (fd_error[..., :14].mean(dim=(1, 2)) + fd_error[..., 14:].mean(dim=(1, 2)))
    target_seam = (target_native[:, 0] - anchor_native) / eval_fd_scale[None]
    pred_seam = (pred_native[:, 0] - anchor_native) / eval_fd_scale[None]
    seam_error = torch.abs(pred_seam - target_seam)
    seam = 0.5 * (seam_error[..., :14].mean(dim=1) + seam_error[..., 14:].mean(dim=1))
    pred_native_mae = torch.abs(pred_native - target_native).mean(dim=(1, 2))
    return position, fd, seam, pred_native_mae


def initialize_model(job: dict[str, Any]) -> tuple[V2TaskConditionedPredictor, str, str]:
    seed = int(job["seed"])
    mode = str(job["video_mode"])
    init_path = Path(job["init_path"])
    if not init_path.is_file():
        raise FileNotFoundError(f"accepted seed INIT missing: {init_path}")
    model = V2TaskConditionedPredictor(seed=seed, video_mode=mode, waypoint_mode=None)
    base_state = safe_load_init(init_path)
    missing, unexpected = model.load_state_dict(base_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"V2 accepted INIT strict load mismatch: missing={missing}, unexpected={unexpected}")
    init_hash = state_hash(model)
    expected = str(job.get("expected_init_state_hash", ""))
    if expected and expected != init_hash:
        raise RuntimeError(f"accepted INIT state hash mismatch: expected={expected}, actual={init_hash}")
    return model, str(init_path), init_hash

def input_batch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    normalizer: dict[str, Any],
    grid: Q0ShardedGrid | None,
    video_mode: str,
    device: torch.device,
    donor_rows: np.ndarray | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            raise RuntimeError("fusion mode missing Q0 feature reader")
        source_rows = indices if donor_rows is None else np.asarray(donor_rows, dtype=np.int64)
        if source_rows.shape != indices.shape:
            raise ValueError("donor_rows must align one-to-one with receiver indices")
        features = np.zeros((len(indices), 2, 16, 4, 768), dtype=np.float32)
        valid = source_rows >= 0
        for pos, donor in enumerate(source_rows.tolist()):
            if donor < 0:
                continue
            features[pos] = grid.row(int(data["cache_index"][donor]), str(data["anchor_id"][donor]), str(data["episode_id"][donor]))
        v_mean = np.asarray(normalizer["video_mean"], dtype=np.float32)[None, None, None, :]
        v_std = np.asarray(normalizer["video_safe_std"], dtype=np.float32)[None, None, None, :]
        dynamic = (features[:, 0] - v_mean) / v_std
        static = (features[:, 1] - v_mean) / v_std
        # A missing legal donor is an explicit zero normalized-block fallback.
        dynamic[~valid] = 0.0
        static[~valid] = 0.0
        prefix["dynamic"] = torch.as_tensor(dynamic, device=device, dtype=torch.float32)
        prefix["static"] = torch.as_tensor(static, device=device, dtype=torch.float32)
    task_rank = np.asarray([int(np.flatnonzero(np.asarray(TASK_IDS) == int(data["task_id"][i]))[0]) for i in indices], dtype=np.int64)
    task_id = torch.as_tensor(task_rank, device=device, dtype=torch.long)
    anchor = torch.as_tensor(data["anchor_native"][indices], device=device, dtype=torch.float32)
    target = torch.as_tensor(data["target_native_h30"][indices], device=device, dtype=torch.float32)
    sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)
    safe_fd = torch.as_tensor(np.asarray(normalizer["safe_fd_scale"], dtype=np.float32), device=device)
    return prefix, task_id, anchor, target, sigma, safe_fd

def episode_equal_tune(metrics: list[tuple[int, int, float, float, float, float]]) -> dict[str, Any]:
    by_task_episode: dict[int, dict[int, list[tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for task_id, episode, pos, fd, seam, native_mae in metrics:
        by_task_episode[int(task_id)][int(episode)].append((float(pos), float(fd), float(seam), float(native_mae)))
    task_results: dict[str, dict[str, float]] = {}
    for task_id, episodes in by_task_episode.items():
        values = np.asarray([np.mean(rows, axis=0) for rows in episodes.values()], dtype=np.float64)
        task_results[str(task_id)] = {
            "position_mae": float(values[:, 0].mean()),
            "safe_fd_l1": float(values[:, 1].mean()),
            "seam_safe_fd_l1": float(values[:, 2].mean()),
            "native_mae": float(values[:, 3].mean()),
            "episodes": len(episodes),
        }
    if not task_results:
        return {"tasks": {}, "macro": None}
    macro = {
        key: float(np.mean([value[key] for value in task_results.values()]))
        for key in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1", "native_mae")
    }
    macro["episodes_by_task"] = {key: value["episodes"] for key, value in task_results.items()}
    return {"tasks": task_results, "macro": macro}


def evaluate(model: V2TaskConditionedPredictor, data: dict[str, np.ndarray], indices: np.ndarray,
             normalizer: dict[str, Any], grid: Q0ShardedGrid | None, job: dict[str, Any], device: torch.device,
             metric_sigma: torch.Tensor, metric_safe_fd_scale: torch.Tensor) -> dict[str, Any]:
    model.eval()
    metrics: list[tuple[int, int, float, float, float, float]] = []
    goal_mae_rows: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            selected = indices[start:start + 64]
            prefix, tasks, anchor, target, sigma, fd_scale = input_batch(
                data, selected, normalizer, grid, str(job["video_mode"]), device
            )
            output = model(prefix, tasks)
            position, fd, seam, native_mae = metric_components(output.action_residual, target, anchor, sigma, fd_scale, metric_sigma, metric_safe_fd_scale)
            pos_np = position.cpu().numpy(); fd_np = fd.cpu().numpy(); seam_np = seam.cpu().numpy(); native_np = native_mae.cpu().numpy()
            for j, ix in enumerate(selected.tolist()):
                metrics.append((int(data["task_id"][ix]), int(data["episode_id"][ix]), float(pos_np[j]), float(fd_np[j]), float(seam_np[j]), float(native_np[j])))
            if output.goal_prediction is not None:
                goal_target = (target[:, -1] - anchor) / sigma[None]
                goal_errors = torch.abs(output.goal_prediction - goal_target).mean(dim=1).cpu().numpy()
                for j, ix in enumerate(selected.tolist()):
                    goal_mae_rows.append((int(data["task_id"][ix]), int(data["episode_id"][ix]), float(goal_errors[j])))
    aggregate = episode_equal_tune(metrics)
    if job["scope"] == "shared":
        macro = aggregate["macro"]
        j_value = float(macro["position_mae"] + 0.1 * macro["safe_fd_l1"]) if macro is not None else None
        aggregate["selection_scope"] = "five_task_macro"
        aggregate["j_tune"] = j_value
    else:
        task_id = str(int(job["task_id"]))
        value = aggregate["tasks"].get(task_id)
        aggregate["selection_scope"] = f"task_{task_id}"
        aggregate["j_tune"] = None if value is None else float(value["position_mae"] + 0.1 * value["safe_fd_l1"])
    if goal_mae_rows:
        g_by_ep: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for task_id, episode, value in goal_mae_rows:
            g_by_ep[task_id][episode].append(value)
        aggregate["goal_endpoint_mae_by_task"] = {
            str(task_id): float(np.mean([np.mean(values) for values in episodes.values()]))
            for task_id, episodes in g_by_ep.items()
        }
    return aggregate


def save_resume(job: dict[str, Any], attempt_dir: Path, model: V2TaskConditionedPredictor,
                optimizer: torch.optim.Optimizer, scheduler: UpdateDrivenCosineAnnealingLR,
                epoch: int, update_count: int, fit_key: str, device: torch.device) -> None:
    resume_dir = Path(str(job["job_root"])) / "safe_resume_state"
    resume_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        f"model::{name}": value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    scalar_state: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        for key, value in state.items():
            if torch.is_tensor(value):
                tensors[f"optimizer::{name}::{key}"] = value.detach().cpu().contiguous()
            else:
                scalar_state.setdefault(name, {})[key] = value
    tensors["rng::torch_cpu"] = torch.get_rng_state().cpu().contiguous()
    tensors["rng::torch_cuda"] = torch.cuda.get_rng_state(device).cpu().contiguous()
    resume_path = resume_dir / f"state_epoch{epoch:03d}.safetensors"
    atomic_safetensors(resume_path, tensors)
    metadata = {
        "schema_version": 1, "fit_key": fit_key, "epoch": epoch, "update_count": update_count,
        "scheduler_state": scheduler.state_dict(), "optimizer_scalar_state": scalar_state,
        "microbatch_size": int(job["microbatch_size"]), "accumulation_steps": int(job["accumulation_steps"]),
        "effective_batch": int(job["microbatch_size"]) * int(job["accumulation_steps"]),
        "resume_tensors": str(resume_path), "resume_tensor_sha256": sha_file(resume_path),
        "saved_utc": utc_now(),
    }
    atomic_json(resume_path.with_suffix(".json"), metadata)


def restore_resume(job: dict[str, Any], model: V2TaskConditionedPredictor, optimizer: torch.optim.Optimizer,
                   scheduler: UpdateDrivenCosineAnnealingLR, device: torch.device, fit_key: str) -> tuple[int, int]:
    resume_dir = Path(str(job["job_root"])) / "safe_resume_state"
    candidates = sorted(resume_dir.glob("state_epoch*.json")) if resume_dir.exists() else []
    if not candidates:
        return 1, 0
    meta_path = candidates[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fit_key") != fit_key:
        raise RuntimeError("resume fit key differs; refusing to load unrelated state")
    tensors_path = Path(meta["resume_tensors"])
    tensors = load_file(str(tensors_path), device="cpu")
    model_state = {key.removeprefix("model::"): value for key, value in tensors.items() if key.startswith("model::")}
    model.load_state_dict(model_state, strict=True)
    named = dict(model.named_parameters())
    for name, parameter in named.items():
        state: dict[str, Any] = {}
        prefix = f"optimizer::{name}::"
        for key, value in tensors.items():
            if key.startswith(prefix):
                state_key = key[len(prefix):]
                state[state_key] = value.cpu() if state_key == "step" else value.to(device)
        for key, value in meta.get("optimizer_scalar_state", {}).get(name, {}).items():
            state[key] = value
        if state:
            optimizer.state[parameter] = state
    scheduler.load_state_dict(meta["scheduler_state"])
    torch.set_rng_state(tensors["rng::torch_cpu"])
    torch.cuda.set_rng_state(tensors["rng::torch_cuda"], device=device)
    epoch = int(meta["epoch"])
    updates = int(meta["update_count"])
    if updates != scheduler.step_calls:
        raise RuntimeError("resume optimizer/scheduler update counts disagree")
    return epoch + 1, updates


def evaluate_task_tune_rows(job: dict[str, Any], data: dict[str, np.ndarray], tune_indices: np.ndarray) -> np.ndarray:
    if job["scope"] == "shared":
        return tune_indices
    return tune_indices[data["task_id"][tune_indices] == int(job["task_id"])]


def run(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_root = Path(job["job_root"])
    fallback_path = job_root / "OOM_FALLBACK.json"
    if fallback_path.exists():
        fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
        if fallback.get("job_id") != job.get("job_id") or fallback.get("effective_batch") != EFFECTIVE_BATCH:
            raise RuntimeError("OOM fallback receipt does not match this fit/effective-batch contract")
        job["microbatch_size"] = int(fallback["microbatch_size"])
        job["accumulation_steps"] = int(fallback["accumulation_steps"])
    attempt_dir = Path(job["attempt_dir"])
    attempt_dir.mkdir(parents=True, exist_ok=True)
    complete_path = attempt_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        print(json.dumps({"status": "already_complete", "job_id": job["job_id"], "attempt_dir": str(attempt_dir)}))
        return 0
    run_status = attempt_dir / "RUN_STATUS.json"
    started = utc_now(); start_perf = time.perf_counter()
    fit_key_payload = {k: job.get(k) for k in ("phase", "model_id", "scope", "task_id", "video_mode", "waypoint_mode", "seed", "lr", "epochs", "normalizer_path", "train_role")}
    fit_key = hashlib.sha256(json.dumps(fit_key_payload, sort_keys=True).encode()).hexdigest()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("V2 fit requires a watcher-authorized CUDA device")
        device = torch.device("cuda:0")
        torch.manual_seed(int(job["seed"]))
        torch.cuda.manual_seed_all(int(job["seed"]))
        np.random.seed(int(job["seed"]))
        random.seed(int(job["seed"]))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        data = load_npz()
        normalizer_path = Path(job["normalizer_path"])
        normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
        train_all = scoped_indices(data, job, tune=False)
        tune_all = scoped_indices(data, job, tune=True)
        program_id = str(job.get("program_id", ""))
        if program_id not in {"A_STATE_COMMON", "C_STATE_NEIGHBOR_REASSIGNED"}:
            raise RuntimeError(f"V4 trainer refuses unexpected program_id={program_id!r}")
        donor_candidate = donor_k = None
        donor_archive_sha256 = None
        if program_id == "C_STATE_NEIGHBOR_REASSIGNED":
            donor_path = Path(job["donor_candidate_path"])
            if not donor_path.is_file():
                raise FileNotFoundError(f"V4 train donor candidates missing: {donor_path}")
            with np.load(donor_path, allow_pickle=False) as donor_npz:
                donor_candidate = donor_npz["candidate_row"].copy()
                donor_k = donor_npz["candidate_k"].copy()
                donor_ids = donor_npz["receiver_anchor_id"].astype(str)
            if donor_candidate.shape != (len(data["anchor_id"]), 5) or donor_k.shape != (len(data["anchor_id"]),):
                raise RuntimeError("V4 donor candidate table has unexpected shape")
            if not np.array_equal(donor_ids, data["anchor_id"].astype(str)):
                raise RuntimeError("V4 donor candidate receiver ID order differs from accepted train NPZ")
            donor_archive_sha256 = sha_file(donor_path)
        if not len(train_all):
            raise RuntimeError("fit contains no training anchors")
        effective = int(job["microbatch_size"]) * int(job["accumulation_steps"])
        if effective != EFFECTIVE_BATCH:
            raise RuntimeError(f"effective batch must remain {EFFECTIVE_BATCH}, got {effective}")
        steps_per_epoch = int(np.ceil(len(train_all) / EFFECTIVE_BATCH))
        scheduler_tmax = 160 * steps_per_epoch
        model, init_path, init_state_hash = initialize_model(job)
        model = model.to(device)
        lr = float(job["lr"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=scheduler_tmax, eta_min=1e-5)
        start_epoch, optimizer_steps = restore_resume(job, model, optimizer, scheduler, device, fit_key)
        grid = None
        if str(job["video_mode"]) != "STATE-CTRL":
            resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
            cache_manifest = json.loads(Path(resolved["feature_cache"]).read_text(encoding="utf-8"))
            grid = Q0ShardedGrid(cache_manifest)

        weights = episode_task_weights(data, train_all, str(job["scope"]))
        weight_by_global = {int(index): float(weight) for index, weight in zip(train_all, weights)}
        tune_eval_indices = evaluate_task_tune_rows(job, data, tune_all)
        target_sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)
        safe_fd_scale = torch.as_tensor(np.asarray(normalizer["safe_fd_scale"], dtype=np.float32), device=device)
        metric_lock = json.loads((ROOT / "protocol/METRIC_SCALE_LOCK.json").read_text(encoding="utf-8"))
        if metric_lock.get("status") != "LOCKED" or int(metric_lock.get("final_heldout_payload_access", -1)) != 0:
            raise RuntimeError("canonical Full-28 metric scale lock is absent or invalid")
        metric_sigma = torch.as_tensor(np.asarray(metric_lock["target_sigma"], dtype=np.float32), device=device)
        metric_safe_fd_scale = torch.as_tensor(np.asarray(metric_lock["safe_fd_scale"], dtype=np.float32), device=device)
        epochs = int(job["epochs"])
        checkpoint_epochs = set(int(x) for x in job.get("checkpoint_epochs", []))
        if job["phase"] == "screening" and checkpoint_epochs != set(CHECKPOINTS_SCREEN):
            raise RuntimeError("screening jobs must evaluate exactly epochs 40,80,160")
        output_dir = attempt_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config_copy = dict(job)
        atomic_json(output_dir / "CONFIG.json", config_copy)
        atomic_json(run_status, {
            "status": "running", "created_utc": started, "job_id": job["job_id"], "fit_key": fit_key,
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "device_logical": str(device), "parameter_count_total": parameter_count(model),
            "init_path": init_path, "init_state_hash": init_state_hash, "epoch_start": start_epoch,
            "train_anchors": len(train_all), "tune_anchors": len(tune_eval_indices),
        })

        epoch_fields = ["epoch", "optimizer_steps", "lr_after_epoch", "train_pos", "train_fd", "train_seam_safe_fd", "train_goal", "train_total", "tune_j", "tune_pos_macro", "tune_fd_macro", "elapsed_seconds"]
        candidate_fields = ["job_id", "model_id", "lr", "checkpoint_epoch", "metric_scope", "task_id", "position_mae", "safe_fd_l1", "seam_safe_fd_l1", "native_mae", "j_tune", "tune_episodes", "goal_endpoint_mae"]
        actual_fit_weight_sum = float(weights.sum())
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise RuntimeError("episode/task weights are nonfinite or nonpositive")
        if not np.isclose(actual_fit_weight_sum, float(len(train_all)), rtol=0.0, atol=1e-4):
            raise RuntimeError(f"full-population sample weights do not sum to W: {actual_fit_weight_sum} vs {len(train_all)}")
        optimizer.zero_grad(set_to_none=True)
        active_grad_counts: set[str] = set()
        resume_every = 10
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            scheduler.set_epoch(epoch)
            order = shuffled_epoch_order(len(train_all), int(job["seed"]), epoch)
            epoch_pos_sum = 0.0; epoch_fd_sum = 0.0; epoch_seam_fd_sum = 0.0; epoch_goal_sum = 0.0; epoch_total_sum = 0.0
            epoch_weight_sum = 0.0
            for group_start in range(0, len(order), EFFECTIVE_BATCH):
                group_local = order[group_start:group_start + EFFECTIVE_BATCH]
                group_indices = train_all[group_local]
                group_n = len(group_indices)
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, group_n, int(job["microbatch_size"])):
                    local_sel = group_local[micro_start:micro_start + int(job["microbatch_size"])]
                    batch_indices = train_all[local_sel]
                    donor_rows = None
                    if program_id == "C_STATE_NEIGHBOR_REASSIGNED":
                        donor_rows = np.full(len(batch_indices), -1, dtype=np.int64)
                        for pos, receiver in enumerate(batch_indices.tolist()):
                            count = int(donor_k[receiver])
                            if count > 0:
                                offset = stable_hash(f"{int(job['seed'])}|{data['anchor_id'][receiver]}")
                                donor_rows[pos] = int(donor_candidate[receiver, (offset + epoch - 1) % count])
                    prefix, task_rank, anchor, target, sigma, fdscale = input_batch(
                        data, batch_indices, normalizer, grid, str(job["video_mode"]), device,
                        donor_rows=donor_rows,
                    )
                    out = model(prefix, task_rank)
                    sample_action = per_sample_loss(out.action_residual, target, anchor, sigma, fdscale)
                    pos_vec, fd_vec, seam_vec, native_vec = metric_components(out.action_residual, target, anchor, sigma, fdscale, metric_sigma, metric_safe_fd_scale)
                    sample_total = sample_action
                    goal_vec = None
                    if out.goal_prediction is not None:
                        goal_target = (target[:, -1] - anchor) / sigma[None]
                        goal_vec = torch.abs(out.goal_prediction - goal_target).mean(dim=1)
                        sample_total = sample_total + 0.1 * goal_vec
                    sample_weight = torch.as_tensor(
                        [weight_by_global[int(i)] for i in batch_indices], device=device, dtype=torch.float32
                    )
                    contribution = torch.sum(sample_total * sample_weight) / float(group_n)
                    if not torch.isfinite(contribution):
                        raise FloatingPointError(f"nonfinite weighted loss at epoch {epoch}")
                    contribution.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError(f"nonfinite gradient {name} at epoch {epoch}")
                            active_grad_counts.add(name)
                    weights_np = sample_weight.detach().cpu().numpy().astype(np.float64)
                    epoch_pos_sum += float(np.dot(pos_vec.detach().cpu().numpy(), weights_np))
                    epoch_fd_sum += float(np.dot(fd_vec.detach().cpu().numpy(), weights_np))
                    epoch_seam_fd_sum += float(np.dot(seam_vec.detach().cpu().numpy(), weights_np))
                    if goal_vec is not None:
                        epoch_goal_sum += float(np.dot(goal_vec.detach().cpu().numpy(), weights_np))
                    epoch_total_sum += float(np.dot(sample_total.detach().cpu().numpy(), weights_np))
                    epoch_weight_sum += float(weights_np.sum())
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(f"nonfinite gradient norm at epoch {epoch}")

            if scheduler.step_calls != optimizer_steps:
                raise RuntimeError("optimizer/scheduler update counts diverged")
            if epoch_weight_sum <= 0 or not np.isclose(epoch_weight_sum, len(train_all), rtol=0.0, atol=1e-4):
                raise RuntimeError("epoch did not visit all training rows exactly once")
            train_pos = epoch_pos_sum / len(train_all)
            train_fd = epoch_fd_sum / len(train_all)
            train_seam_fd = epoch_seam_fd_sum / len(train_all)
            train_goal = epoch_goal_sum / len(train_all)
            train_total = epoch_total_sum / len(train_all)
            tune_result = None
            if job["phase"] == "screening" and epoch in checkpoint_epochs:
                tune_result = evaluate(model, data, tune_eval_indices, normalizer, grid, job, device, metric_sigma, metric_safe_fd_scale)
            append_csv(output_dir / "EPOCH_LOG.csv", epoch_fields, {
                "epoch": epoch, "optimizer_steps": optimizer_steps, "lr_after_epoch": scheduler.get_last_lr()[0],
                "train_pos": train_pos, "train_fd": train_fd, "train_seam_safe_fd": train_seam_fd, "train_goal": train_goal, "train_total": train_total,
                "tune_j": "" if tune_result is None else tune_result["j_tune"],
                "tune_pos_macro": "" if tune_result is None or tune_result.get("macro") is None else tune_result["macro"]["position_mae"],
                "tune_fd_macro": "" if tune_result is None or tune_result.get("macro") is None else tune_result["macro"]["safe_fd_l1"],
                "elapsed_seconds": time.perf_counter() - start_perf,
            })
            if tune_result is not None:
                task_metrics = tune_result.get("tasks", {})
                for task_id in (map(str, TASK_IDS) if job["scope"] == "shared" else [str(int(job["task_id"]))]):
                    metric = task_metrics.get(task_id)
                    if metric is None:
                        append_csv(output_dir / "TUNE_CHECKPOINT_METRICS.csv", candidate_fields, {
                            "job_id": job["job_id"], "model_id": job["model_id"], "lr": lr,
                            "checkpoint_epoch": epoch, "metric_scope": f"task_{task_id}", "task_id": task_id,
                            "position_mae": "NA", "safe_fd_l1": "NA", "seam_safe_fd_l1": "NA", "native_mae": "NA", "j_tune": "NA", "tune_episodes": 0,
                        })
                    else:
                        append_csv(output_dir / "TUNE_CHECKPOINT_METRICS.csv", candidate_fields, {
                            "job_id": job["job_id"], "model_id": job["model_id"], "lr": lr,
                            "checkpoint_epoch": epoch, "metric_scope": f"task_{task_id}", "task_id": task_id,
                            "position_mae": metric["position_mae"], "safe_fd_l1": metric["safe_fd_l1"],
                            "seam_safe_fd_l1": metric["seam_safe_fd_l1"], "native_mae": metric["native_mae"],
                            "j_tune": metric["position_mae"] + 0.1 * metric["safe_fd_l1"],
                            "tune_episodes": metric["episodes"],
                        })
                if job["scope"] == "shared" and tune_result.get("macro") is not None:
                    macro = tune_result["macro"]
                    append_csv(output_dir / "TUNE_CHECKPOINT_METRICS.csv", candidate_fields, {
                        "job_id": job["job_id"], "model_id": job["model_id"], "lr": lr,
                        "checkpoint_epoch": epoch, "metric_scope": "macro", "task_id": "macro",
                        "position_mae": macro["position_mae"], "safe_fd_l1": macro["safe_fd_l1"],
                        "seam_safe_fd_l1": macro["seam_safe_fd_l1"], "native_mae": macro["native_mae"], "j_tune": tune_result["j_tune"],
                        "tune_episodes": ";".join(f"{k}:{v}" for k,v in macro["episodes_by_task"].items()),
                    })
                elif job["scope"] == "expert" and tune_result.get("j_tune") is not None:
                    metric = task_metrics[str(int(job["task_id"]))]
                    append_csv(output_dir / "TUNE_CHECKPOINT_METRICS.csv", candidate_fields, {
                        "job_id": job["job_id"], "model_id": job["model_id"], "lr": lr,
                        "checkpoint_epoch": epoch, "metric_scope": "selection", "task_id": int(job["task_id"]),
                        "position_mae": metric["position_mae"], "safe_fd_l1": metric["safe_fd_l1"],
                        "seam_safe_fd_l1": metric["seam_safe_fd_l1"], "native_mae": metric["native_mae"], "j_tune": tune_result["j_tune"],
                        "tune_episodes": metric["episodes"],
                    })

            if epoch in checkpoint_epochs or epoch % resume_every == 0:
                ckpt_path = output_dir / f"model_epoch{epoch:03d}.safetensors"
                atomic_safetensors(ckpt_path, model.state_dict())
                save_resume(job, attempt_dir, model, optimizer, scheduler, epoch, optimizer_steps, fit_key, device)
                atomic_json(output_dir / f"CHECKPOINT_EPOCH{epoch:03d}.json", {
                    "status": "PASS", "epoch": epoch, "optimizer_steps": optimizer_steps,
                    "weights_path": str(ckpt_path), "weights_sha256": sha_file(ckpt_path),
                    "model_state_hash": state_hash(model), "parameter_count": parameter_count(model),
                    "normalizer_sha256": sha_file(normalizer_path), "fit_key": fit_key,
                    "final_heldout_payload_access": 0,
                })
            atomic_json(run_status, {
                "status": "running", "updated_utc": utc_now(), "job_id": job["job_id"], "fit_key": fit_key,
                "epoch": epoch, "epochs_target": epochs, "optimizer_steps": optimizer_steps,
                "steps_per_epoch": steps_per_epoch, "scheduler_tmax": scheduler_tmax,
                "lr_after_epoch": scheduler.get_last_lr()[0], "train_pos": train_pos, "train_fd": train_fd,
                "train_goal": train_goal, "train_total": train_total, "train_seam_safe_fd": train_seam_fd,
                "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
                "active_gradient_parameter_count_last_seen": len(active_grad_counts),
            })

        final_path = output_dir / "model.safetensors"
        atomic_safetensors(final_path, model.state_dict())
        param_total = parameter_count(model)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        atomic_json(output_dir / "ACTIVE_PARAMETERS.json", {
            "total_parameters": param_total,
            "trainable_parameters": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            "parameters_with_gradient_seen": sorted(active_grad_counts),
            "parameters_with_gradient_count": len(active_grad_counts),
            "trainable_parameter_names": trainable_names,
            "waypoint_mode": job.get("waypoint_mode"),
            "note": "AUX_ONLY goal_projection has trainable tensors but no action-path or loss gradient; listed as expected inactive action parameters.",
        })
        if scheduler.step_calls != optimizer_steps:
            raise RuntimeError("final scheduler and optimizer counts differ")
        if scheduler.step_calls > scheduler_tmax:
            raise RuntimeError("scheduler updates exceed locked T_max")
        lr_trace = output_dir / "LR_TRACE_ACTUAL.csv"
        write_lr_trace(lr_trace, scheduler.update_records)
        model_manifest = {
            "schema_version": 1, "status": "PASS", "created_utc": utc_now(),
            "job_id": job["job_id"], "phase": job["phase"], "model_id": job["model_id"],
            "task_ids": job.get("task_ids", [job.get("task_id")]), "video_mode": job["video_mode"],
            "waypoint_mode": job.get("waypoint_mode"), "seed": int(job["seed"]),
            "lr": lr, "epochs": epochs, "effective_batch": EFFECTIVE_BATCH,
            "microbatch_size": int(job["microbatch_size"]), "accumulation_steps": int(job["accumulation_steps"]),
            "steps_per_epoch": steps_per_epoch, "optimizer_steps": optimizer_steps,
            "scheduler": {"name": "CosineAnnealingLR", "T_max": scheduler_tmax, "eta_min": 1e-5, "step_location": "optimizer post-step hook"},
            "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4},
            "dtype": "float32", "amp": False, "compile": False,
            "program_id": program_id, "donor_candidate_path": job.get("donor_candidate_path"),
            "donor_candidate_sha256": donor_archive_sha256,
            "donor_selection": "stable SHA256(seed|receiver_anchor_id) first16hex plus zero-based epoch modulo K; isolated from model/sampler PRNG" if program_id == "C_STATE_NEIGHBOR_REASSIGNED" else "none; V2 STATE-CTRL zero mask",
            "parameter_count": param_total, "init_path": init_path, "init_state_hash": init_state_hash,
            "final_state_hash": state_hash(model), "model_file": str(final_path), "model_file_sha256": sha_file(final_path),
            "normalizer_path": str(normalizer_path), "normalizer_sha256": sha_file(normalizer_path),
            "train_selected_npz_sha256": sha_file(ROOT / "data/TRAIN_SELECTED_H30.npz"),
            "feature_cache_manifest_sha256": json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text())["feature_cache_manifest_sha256"],
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "logical_device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
            "final_heldout_payload_access": 0,
        }
        atomic_json(output_dir / "MODEL_MANIFEST.json", model_manifest)
        tune_payload = None
        if job["phase"] == "screening":
            metric_path = output_dir / "TUNE_CHECKPOINT_METRICS.csv"
            with metric_path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            tune_payload = records
        complete = {
            "schema_version": 1, "status": "PASS", "completed_utc": utc_now(),
            "job_id": job["job_id"], "fit_key": fit_key, "epochs_completed": epochs,
            "optimizer_steps": optimizer_steps, "steps_per_epoch": steps_per_epoch,
            "scheduler_tmax": scheduler_tmax, "weights_sha256": model_manifest["model_file_sha256"],
            "tune_metrics": tune_payload, "elapsed_seconds": time.perf_counter() - start_perf,
            "supervised_fit": True, "final_heldout_payload_access": 0,
        }
        atomic_json(complete_path, complete)
        atomic_json(run_status, {"status": "complete", "completed_utc": utc_now(), "job_id": job["job_id"], "epochs": epochs, "optimizer_steps": optimizer_steps})
        scheduler.close()
        print(json.dumps({"status": "PASS", "job_id": job["job_id"], "epochs": epochs, "optimizer_steps": optimizer_steps, "model_sha256": model_manifest["model_file_sha256"], "elapsed_seconds": complete["elapsed_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:
        error_text = str(exc).lower()
        is_cuda_oom = ("out of memory" in error_text) or type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}
        if is_cuda_oom and int(job.get("microbatch_size", 0)) == 64 and int(job.get("accumulation_steps", 0)) == 1 and not fallback_path.exists():
            registered = job.get("oom_fallback", {})
            if int(registered.get("microbatch_size", 0)) == 32 and int(registered.get("accumulation_steps", 0)) == 2:
                atomic_json(fallback_path, {
                    "schema_version": 1, "status": "RETRY_WITH_EFFECTIVE_BATCH_PRESERVED",
                    "job_id": job.get("job_id"), "trigger": f"{type(exc).__name__}: {exc}",
                    "microbatch_size": 32, "accumulation_steps": 2, "effective_batch": EFFECTIVE_BATCH,
                    "resume_policy": "reload_last_complete_epoch_tensor_only_state; discard incomplete current epoch",
                    "created_utc": utc_now(),
                })
        atomic_json(run_status, {
            "status": "failed", "failed_utc": utc_now(), "job_id": job.get("job_id"),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "fit_key": fit_key,
        })
        print(json.dumps({"status": "FAILED", "job_id": job.get("job_id"), "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), flush=True)
        return 1


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
