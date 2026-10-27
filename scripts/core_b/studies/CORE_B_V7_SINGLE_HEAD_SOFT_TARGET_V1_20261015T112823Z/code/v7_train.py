from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from v7_common import (
    NORMALIZER,
    PROJECT,
    ROOT,
    SEEDS,
    TASKS,
    TRAIN_NPZ,
    V4,
    atomic_json,
    load_model_api,
    load_safe_prediction,
    load_train_prefix,
    make_grid,
    prefix_batch,
    sha_file,
    state_rng_hash,
    task_episode_weights,
)

V4_CODE = V4 / "code"
V4_LOCKED = V4_CODE / "locked_source"
sys.path.insert(0, str(V4_CODE))
sys.path.insert(0, str(V4_LOCKED))
from d3_eq_contract import per_sample_loss, shuffled_epoch_order  # noqa: E402
from v2_scheduler import UpdateDrivenCosineAnnealingLR, write_lr_trace  # noqa: E402

PYTHON_BIN = PROJECT / "envs/jyc/bin/python"
EFFECTIVE_BATCH = 64
EPOCHS = 160
TASKS_ORDER = (119, 94, 90, 141, 230)
CURVE_FIELDS = [
    "epoch", "optimizer_steps", "lr_after_epoch", "weighted_objective_loss",
    "fit_position_mae", "fit_safe_fd_l1", "fit_seam_safe_fd_l1",
    "ground_truth_position_mae", "ground_truth_safe_fd_l1", "ground_truth_seam_safe_fd_l1",
    "ground_truth_native_mae", "student_to_V_position_distance", "student_to_V_safe_fd_distance",
    "student_to_V_seam_distance", "student_to_T_position_distance",
    "student_to_T_safe_fd_distance", "student_to_T_seam_distance",
    "elapsed_seconds",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_safe_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    safe = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
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


def metric_vectors(
    prediction_native: torch.Tensor,
    reference_native: torch.Tensor,
    anchor_native: torch.Tensor,
    metric_sigma: torch.Tensor,
    metric_fd_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    error = torch.abs(prediction_native - reference_native) / metric_sigma[None, None, :]
    position = 0.5 * (
        error[..., :14].mean(dim=(1, 2)) + error[..., 14:].mean(dim=(1, 2))
    )
    pred_fd = torch.diff(prediction_native, dim=1) / metric_fd_scale[None, None, :]
    ref_fd = torch.diff(reference_native, dim=1) / metric_fd_scale[None, None, :]
    fd_error = torch.abs(pred_fd - ref_fd)
    fd = 0.5 * (
        fd_error[..., :14].mean(dim=(1, 2)) + fd_error[..., 14:].mean(dim=(1, 2))
    )
    pred_seam = (prediction_native[:, 0] - anchor_native) / metric_fd_scale[None, :]
    ref_seam = (reference_native[:, 0] - anchor_native) / metric_fd_scale[None, :]
    seam_error = torch.abs(pred_seam - ref_seam)
    seam = 0.5 * (seam_error[:, :14].mean(dim=1) + seam_error[:, 14:].mean(dim=1))
    native_mae = torch.abs(prediction_native - reference_native).mean(dim=(1, 2))
    return position, fd, seam, native_mae


def pair_distance_vectors(
    prediction_native: torch.Tensor,
    reference_native: torch.Tensor,
    anchor_native: torch.Tensor,
    metric_sigma: torch.Tensor,
    metric_fd_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.abs(prediction_native - reference_native) / metric_sigma[None, None, :]
    position = 0.5 * (pos[..., :14].mean(dim=(1, 2)) + pos[..., 14:].mean(dim=(1, 2)))
    internal = torch.abs(torch.diff(prediction_native - reference_native, dim=1)) / metric_fd_scale[None, None, :]
    fd = 0.5 * (internal[..., :14].mean(dim=(1, 2)) + internal[..., 14:].mean(dim=(1, 2)))
    seam = torch.abs(prediction_native[:, 0] - reference_native[:, 0]) / metric_fd_scale[None, :]
    seam = 0.5 * (seam[:, :14].mean(dim=1) + seam[:, 14:].mean(dim=1))
    return position, fd, seam


def load_job_inputs(job: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seal_path = ROOT / "protocol/TEACHER_BANK_SEAL.json"
    target_lock_path = ROOT / "protocol/TARGET_LOCK.json"
    if not seal_path.is_file() or not target_lock_path.is_file():
        raise RuntimeError("student training cannot start before teacher and target seals exist")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    lock = json.loads(target_lock_path.read_text(encoding="utf-8"))
    if seal.get("status") != "PASS" or lock.get("status") != "LOCKED":
        raise RuntimeError("teacher/target seals are not accepted")
    if sha_file(seal_path) != job["teacher_bank_seal_sha256"]:
        raise RuntimeError("student job teacher-bank seal digest mismatch")
    if sha_file(target_lock_path) != job["target_lock_sha256"]:
        raise RuntimeError("student job target-lock digest mismatch")
    target_entries = [row for row in lock.get("target_entries", []) if int(row["seed"]) == int(job["seed"]) and row["program"] == job["program_id"]]
    if len(target_entries) != 1 or target_entries[0]["target_path"] != job["training_target_path"] or target_entries[0]["target_sha256"] != job["training_target_sha256"]:
        raise RuntimeError("student job target path/hash differs from the locked target role")
    if sha_file(Path(job["training_target_path"])) != job["training_target_sha256"]:
        raise RuntimeError("student training target bytes differ from TARGET_LOCK")
    pairing = json.loads((ROOT / "protocol/INIT_SAMPLER_PAIRING.json").read_text(encoding="utf-8"))
    paired_seed = next(row for row in pairing["seeds"] if int(row["seed"]) == int(job["seed"]))
    if (job["init_path"] != paired_seed["init_path"] or job["init_sha256"] != paired_seed["init_sha256"] or job["sampler_orders_path"] != paired_seed["sampler_orders_path"] or job["sampler_orders_sha256"] != paired_seed["sampler_orders_sha256"]):
        raise RuntimeError("student job INIT/sampler differs from locked paired source")
    source_entries = seal.get("entries", [])
    v_entries = [row for row in source_entries if int(row.get("seed", -1)) == int(job["seed"]) and row.get("teacher_role") == "V"]
    t_entries = [row for row in source_entries if int(row.get("seed", -1)) == int(job["seed"]) and row.get("teacher_role") == "T"]
    if len(v_entries) != 1 or len(t_entries) != 1:
        raise RuntimeError("teacher bank lacks exactly one paired V_i/T_i source")
    if sha_file(Path(v_entries[0]["prediction_path"])) != v_entries[0]["prediction_sha256"] or sha_file(Path(t_entries[0]["prediction_path"])) != t_entries[0]["prediction_sha256"]:
        raise RuntimeError("paired V/T train prediction hash differs from the sealed teacher bank")
    if sha_file(TRAIN_NPZ) != job["train_npz_sha256"] or sha_file(NORMALIZER) != job["normalizer_sha256"]:
        raise RuntimeError("training archive or normalizer hash differs from the locked job")

    # Prefix arrays are loaded through the whitelist. Y is read only from the
    # already sealed V7 training target area, after the teacher bank seal.
    data = load_train_prefix()
    y = np.load(ROOT / "training_targets/Y_NATIVE.npy", allow_pickle=False)
    target = np.load(Path(job["training_target_path"]), allow_pickle=False)
    teacher_entries = json.loads((ROOT / "protocol/TEACHER_BANK_SEAL.json").read_text(encoding="utf-8"))["entries"]
    v_entry = next(row for row in teacher_entries if int(row.get("seed", -1)) == int(job["seed"]) and row.get("teacher_role") == "V")
    t_entry = next(row for row in teacher_entries if int(row.get("seed", -1)) == int(job["seed"]) and row.get("teacher_role") == "T")
    v_path = Path(v_entry["prediction_path"])
    t_path = Path(t_entry["prediction_path"])
    v_native = load_safe_prediction(v_path)
    t_native = load_safe_prediction(t_path)
    for name, values in (("Y", y), ("Z", target), ("V", v_native), ("T", t_native)):
        if values.shape != (1280, 30, 28) or not np.isfinite(values).all():
            raise RuntimeError(f"{name} training array has invalid shape/values: {values.shape}")
    return data, y.astype(np.float64, copy=False), target.astype(np.float64, copy=False), v_native, t_native


def run(job_path: Path) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    job_root = Path(job["job_root"])
    attempt_dir = Path(job["attempt_dir"])
    fallback_path = job_root / "OOM_FALLBACK.json"
    if fallback_path.is_file():
        fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
        if fallback.get("job_id") != job.get("job_id") or int(fallback.get("effective_batch", -1)) != EFFECTIVE_BATCH:
            raise RuntimeError("registered OOM fallback does not match this effective-batch contract")
        job["microbatch_size"] = int(fallback["microbatch_size"])
        job["accumulation_steps"] = int(fallback["accumulation_steps"])
    attempt_dir.mkdir(parents=True, exist_ok=True)
    complete_path = attempt_dir / "RUN_COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") == "PASS":
            print(json.dumps({"status": "already_complete", "job_id": job["job_id"]}), flush=True)
            return 0
    run_status = attempt_dir / "RUN_STATUS.json"
    started = utc_now()
    started_perf = time.perf_counter()
    try:
        seed = int(job["seed"])
        program = str(job["program_id"])
        if seed not in (202610131, 202610132, 202610133) or program not in {"GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"}:
            raise ValueError("fit seed/program is outside the locked 9-slot matrix")
        if str(job["video_mode"]) != "TS11-D" or job.get("waypoint_mode") is not None:
            raise ValueError("V7 students must use the original V_i TS11-D head")
        pilot = bool(job.get("pilot", False))
        microbatch = int(job["microbatch_size"])
        accumulation = int(job["accumulation_steps"])
        if microbatch * accumulation != EFFECTIVE_BATCH:
            raise RuntimeError("microbatch x accumulation differs from effective batch 64")
        if microbatch not in {32, 64} or accumulation not in {1, 2}:
            raise RuntimeError("unsupported registered effective-batch realization")
        if not torch.cuda.is_available():
            raise RuntimeError("student fit requires a watcher-authorized CUDA device")

        device = torch.device("cuda:0")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

        data, y_native, target_native, v_native, t_native = load_job_inputs(job)
        count = len(data["anchor_id"])
        if count != 1280:
            raise RuntimeError(f"outer-train count differs from 1280: {count}")
        init_path = Path(job["init_path"])
        if sha_file(init_path) != job["init_sha256"]:
            raise RuntimeError("common fresh INIT file digest mismatch")
        normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
        metric_lock_path = V4 / "protocol/METRIC_SCALE_LOCK.json"
        metric_lock = json.loads(metric_lock_path.read_text(encoding="utf-8"))
        if metric_lock.get("status") != "LOCKED" or int(metric_lock.get("final_heldout_payload_access", -1)) != 0:
            raise RuntimeError("accepted metric scale lock is absent or invalid")

        model_type, parameter_count, state_hash = load_model_api()
        model = model_type(seed=seed, video_mode="TS11-D", waypoint_mode=None)
        init_tensors = load_file(str(init_path), device="cpu")
        model.load_state_dict(init_tensors, strict=True)
        initial_hash = state_hash(model)
        if initial_hash != job["expected_init_state_hash"]:
            raise RuntimeError("strict fresh INIT state hash mismatch")
        if sha_file(Path(job["init_manifest_path"])) != job["init_manifest_sha256"]:
            raise RuntimeError("accepted common INIT provenance manifest hash mismatch")
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or any(not parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("V7 student head trainability differs from the accepted V4 G-V recipe")
        model.to(device)
        model.train()
        start_rng_hash = state_rng_hash(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
        if len(data["anchor_id"]) != EFFECTIVE_BATCH * 20:
            raise RuntimeError("full train population no longer yields 20 effective updates per epoch")
        t_max = EPOCHS * 20
        scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-5)
        if scheduler.step_calls != 0:
            raise RuntimeError("scheduler did not start at update zero")

        order_path = ROOT / "protocol" / f"SAMPLER_ORDERS_seed{seed}.npy"
        orders = np.load(order_path, allow_pickle=False)
        if orders.shape != (EPOCHS, count) or orders.dtype != np.int64:
            raise RuntimeError(f"locked sampler order has invalid shape/dtype: {orders.shape}/{orders.dtype}")
        if sha_file(order_path) != job["sampler_orders_sha256"]:
            raise RuntimeError("locked sampler order digest mismatch")
        for epoch_index, epoch_order in enumerate(orders, start=1):
            if not np.array_equal(np.sort(epoch_order), np.arange(count, dtype=np.int64)):
                raise RuntimeError("locked sampler epoch is not a complete permutation")
            if not np.array_equal(epoch_order, shuffled_epoch_order(count, seed, epoch_index)):
                raise RuntimeError("saved sampler order differs from the accepted V2 sampler implementation")

        weights = task_episode_weights(data)
        if not np.isclose(weights.sum(), count, rtol=0.0, atol=1e-8):
            raise RuntimeError("corrected task/episode weights do not sum to W")
        weight_tensor_all = torch.as_tensor(weights, dtype=torch.float32, device=device)
        grid = make_grid()
        train_sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)
        train_fd = torch.as_tensor(np.asarray(normalizer["safe_fd_scale"], dtype=np.float32), device=device)
        metric_sigma = torch.as_tensor(np.asarray(metric_lock["target_sigma"], dtype=np.float32), device=device)
        metric_fd = torch.as_tensor(np.asarray(metric_lock["safe_fd_scale"], dtype=np.float32), device=device)
        y_cpu = torch.as_tensor(y_native, dtype=torch.float32)
        target_cpu = torch.as_tensor(target_native, dtype=torch.float32)
        v_cpu = torch.as_tensor(v_native, dtype=torch.float32)
        t_cpu = torch.as_tensor(t_native, dtype=torch.float32)

        torch.cuda.reset_peak_memory_stats(device)
        atomic_json(run_status, {
            "status": "running",
            "created_utc": started,
            "job_id": job["job_id"],
            "program_id": program,
            "seed": seed,
            "pilot": pilot,
            "epoch": 0,
            "optimizer_steps": 0,
            "train_anchor_count": count,
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch_size": microbatch,
            "accumulation_steps": accumulation,
            "init_path": str(init_path),
            "init_sha256": sha_file(init_path),
            "init_state_hash": initial_hash,
            "cuda_rng_initial_sha256": start_rng_hash,
            "train_target_path": job["training_target_path"],
            "train_target_sha256": sha_file(Path(job["training_target_path"])),
            "teacher_bank_seal_sha256": job["teacher_bank_seal_sha256"],
            "target_lock_sha256": job["target_lock_sha256"],
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "final_heldout_payload_access": 0,
        })

        update_count = 0
        epochs_to_run = 1 if pilot else EPOCHS
        max_updates = 2 if pilot else EPOCHS * 20
        epoch_fields = list(CURVE_FIELDS)
        if not pilot:
            curve_path = attempt_dir / "TRAINING_CURVES.csv"
            if curve_path.exists():
                raise FileExistsError(f"preserving existing training curve: {curve_path}")
        else:
            curve_path = attempt_dir / "PILOT_CURVE.csv"
        for epoch in range(1, epochs_to_run + 1):
            model.train()
            scheduler.set_epoch(epoch)
            order = orders[epoch - 1]
            epoch_metric_sums = np.zeros(16, dtype=np.float64)
            epoch_weight_sum = 0.0
            updates_this_epoch = 0
            for group_start in range(0, count, EFFECTIVE_BATCH):
                if pilot and update_count >= max_updates:
                    break
                group_local = order[group_start:group_start + EFFECTIVE_BATCH]
                group_indices = group_local.astype(np.int64, copy=False)
                group_n = len(group_indices)
                if group_n != EFFECTIVE_BATCH:
                    raise RuntimeError("accepted full population unexpectedly yielded a partial batch")
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, group_n, microbatch):
                    micro_local = group_local[micro_start:micro_start + microbatch]
                    indices = micro_local.astype(np.int64, copy=False)
                    prefix, task_rank, anchor = prefix_batch(data, indices, normalizer, device, "TS11-D", grid)
                    output = model(prefix, task_rank)
                    pred_native = anchor[:, None, :] + train_sigma[None, None, :] * output.action_residual.float()
                    ix = torch.as_tensor(indices, dtype=torch.long)
                    z = target_cpu[ix].to(device=device, dtype=torch.float32)
                    y = y_cpu[ix].to(device=device, dtype=torch.float32)
                    v = v_cpu[ix].to(device=device, dtype=torch.float32)
                    t = t_cpu[ix].to(device=device, dtype=torch.float32)
                    sample_loss = per_sample_loss(output.action_residual, z, anchor, train_sigma, train_fd)
                    if not torch.isfinite(sample_loss).all():
                        raise FloatingPointError(f"nonfinite objective in epoch {epoch}")
                    sample_weights = weight_tensor_all[torch.as_tensor(indices, dtype=torch.long, device=device)]
                    contribution = torch.sum(sample_loss * sample_weights) / float(group_n)
                    contribution.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(f"nonfinite gradient in {name}, epoch {epoch}")

                    fit_pos, fit_fd, fit_seam, _ = metric_vectors(pred_native, z, anchor, metric_sigma, metric_fd)
                    gt_pos, gt_fd, gt_seam, gt_native = metric_vectors(pred_native, y, anchor, metric_sigma, metric_fd)
                    v_pos, v_fd, v_seam = pair_distance_vectors(pred_native, v, anchor, metric_sigma, metric_fd)
                    t_pos, t_fd, t_seam = pair_distance_vectors(pred_native, t, anchor, metric_sigma, metric_fd)
                    vectors = torch.stack([
                        sample_loss, fit_pos, fit_fd, fit_seam,
                        gt_pos, gt_fd, gt_seam, gt_native,
                        v_pos, v_fd, v_seam, t_pos, t_fd, t_seam,
                    ], dim=1)
                    weighted = torch.sum(vectors * sample_weights[:, None], dim=0).detach().cpu().numpy().astype(np.float64)
                    epoch_metric_sums[:14] += weighted
                    epoch_weight_sum += float(sample_weights.detach().sum().cpu())

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(f"nonfinite clipped gradient norm in epoch {epoch}")
                optimizer.step()
                update_count += 1
                updates_this_epoch += 1
                if scheduler.step_calls != update_count:
                    raise RuntimeError("optimizer and update-driven scheduler counts diverged")

            if pilot:
                if update_count >= 2:
                    break
                continue
            if updates_this_epoch != 20 or not np.isclose(epoch_weight_sum, count, rtol=0.0, atol=1e-4):
                raise RuntimeError(f"epoch {epoch} did not visit all train rows with locked weights")
            means = epoch_metric_sums[:14] / count
            append_csv(curve_path, epoch_fields, {
                "epoch": epoch,
                "optimizer_steps": update_count,
                "lr_after_epoch": scheduler.get_last_lr()[0],
                "weighted_objective_loss": means[0],
                "fit_position_mae": means[1],
                "fit_safe_fd_l1": means[2],
                "fit_seam_safe_fd_l1": means[3],
                "ground_truth_position_mae": means[4],
                "ground_truth_safe_fd_l1": means[5],
                "ground_truth_seam_safe_fd_l1": means[6],
                "ground_truth_native_mae": means[7],
                "student_to_V_position_distance": means[8],
                "student_to_V_safe_fd_distance": means[9],
                "student_to_V_seam_distance": means[10],
                "student_to_T_position_distance": means[11],
                "student_to_T_safe_fd_distance": means[12],
                "student_to_T_seam_distance": means[13],
                "elapsed_seconds": time.perf_counter() - started_perf,
            })
            if epoch % 10 == 0:
                checkpoint = attempt_dir / f"model_epoch{epoch:03d}.safetensors"
                atomic_safe_tensors(checkpoint, model.state_dict())
                atomic_json(attempt_dir / f"CHECKPOINT_EPOCH{epoch:03d}.json", {
                    "status": "PASS",
                    "epoch": epoch,
                    "optimizer_steps": update_count,
                    "weights_path": str(checkpoint),
                    "weights_sha256": sha_file(checkpoint),
                    "model_state_hash": state_hash(model),
                    "target_role": program,
                    "target_sha256": job["training_target_sha256"],
                    "final_heldout_payload_access": 0,
                })
            atomic_json(run_status, {
                "status": "running",
                "updated_utc": utc_now(),
                "job_id": job["job_id"],
                "program_id": program,
                "seed": seed,
                "epoch": epoch,
                "epochs_target": EPOCHS,
                "optimizer_steps": update_count,
                "steps_per_epoch": 20,
                "scheduler_tmax": t_max,
                "lr_after_epoch": scheduler.get_last_lr()[0],
                "current_weighted_objective_loss": means[0],
                "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            })
        if pilot and update_count == 2:
            pilot_means = epoch_metric_sums[:14] / max(epoch_weight_sum, 1.0)
            append_csv(curve_path, epoch_fields, {
                "epoch": 1, "optimizer_steps": update_count, "lr_after_epoch": scheduler.get_last_lr()[0],
                "weighted_objective_loss": pilot_means[0], "fit_position_mae": pilot_means[1],
                "fit_safe_fd_l1": pilot_means[2], "fit_seam_safe_fd_l1": pilot_means[3],
                "ground_truth_position_mae": pilot_means[4], "ground_truth_safe_fd_l1": pilot_means[5],
                "ground_truth_seam_safe_fd_l1": pilot_means[6], "ground_truth_native_mae": pilot_means[7],
                "student_to_V_position_distance": pilot_means[8], "student_to_V_safe_fd_distance": pilot_means[9],
                "student_to_V_seam_distance": pilot_means[10], "student_to_T_position_distance": pilot_means[11],
                "student_to_T_safe_fd_distance": pilot_means[12], "student_to_T_seam_distance": pilot_means[13],
                "elapsed_seconds": time.perf_counter() - started_perf,
            })
        if pilot:
            if update_count != 2 or not (attempt_dir / "PILOT_CURVE.csv").is_file():
                raise RuntimeError(f"Stage0 pilot completed {update_count}/2 updates")
            pilot_state = state_hash(model)
            pilot_weights = attempt_dir / "pilot_model_roundtrip.safetensors"
            atomic_safe_tensors(pilot_weights, model.state_dict())
            reload_receipt_path = attempt_dir / "FRESH_RELOAD_QA.json"
            reload_job_path = attempt_dir / "fresh_reload_job.json"
            reload_job = {
                "model_path": str(pilot_weights), "seed": seed, "video_mode": "TS11-D",
                "expected_state_hash": pilot_state, "receipt_path": str(reload_receipt_path),
            }
            atomic_json(reload_job_path, reload_job)
            reload_env = dict(os.environ)
            reload_env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"})
            subprocess.run(
                [str(PYTHON_BIN), str(ROOT / "code/v7_reload_check.py"), "--job-json", str(reload_job_path)],
                check=True, env=reload_env, cwd=str(PROJECT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            fresh_reload = json.loads(reload_receipt_path.read_text(encoding="utf-8"))
            if fresh_reload.get("status") != "PASS" or fresh_reload.get("fresh_process") is not True:
                raise RuntimeError("Stage0 fresh-process safe reload receipt did not pass")
            weights_sha = sha_file(pilot_weights)
            pilot_weights.unlink()
            receipt = {
                "schema_version": 1,
                "status": "PASS",
                "job_id": job["job_id"],
                "program_id": program,
                "seed": seed,
                "optimizer_updates": update_count,
                "effective_batch": EFFECTIVE_BATCH,
                "microbatch_size": microbatch,
                "accumulation_steps": accumulation,
                "init_state_hash": initial_hash,
                "cuda_rng_initial_sha256": start_rng_hash,
                "sampler_order_epoch1_sha256": job["sampler_epoch1_sha256"],
                "pilot_model_state_hash": pilot_state,
                "temporary_weights_sha256_before_discard": weights_sha,
                "fresh_process_safe_reload": True,
                "fresh_process_reload_receipt_sha256": sha_file(reload_receipt_path),
                "pilot_weights_discarded": not pilot_weights.exists(),
                "dev_target_values_read": 0,
                "final_heldout_payload_access": 0,
                "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
                "logical_device": str(device),
                "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
                "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
                "elapsed_seconds": time.perf_counter() - started_perf,
            }
            atomic_json(attempt_dir / "PILOT_RECEIPT.json", receipt)
            scheduler.close()
            atomic_json(run_status, {"status": "complete", "completed_utc": utc_now(), "job_id": job["job_id"], "optimizer_updates": 2})
            print(json.dumps({"status": "PASS", "job_id": job["job_id"], "pilot_updates": 2}), flush=True)
            return 0

        if update_count != 3200 or scheduler.step_calls != 3200:
            raise RuntimeError(f"formal fit ended at {update_count}/3200 optimizer updates")
        if sha_file(METRIC_LOCK := V4 / "protocol/METRIC_SCALE_LOCK.json") != job["metric_lock_sha256"]:
            raise RuntimeError("metric scale lock digest changed during fit")
        final_path = attempt_dir / "model.safetensors"
        atomic_safe_tensors(final_path, model.state_dict())
        state_digest = state_hash(model)
        model_manifest = {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": utc_now(),
            "job_id": job["job_id"],
            "study": "post_V6_single_fusion_head_supervised_soft_target_development",
            "program_id": program,
            "seed": seed,
            "task_ids": list(TASKS_ORDER),
            "video_mode": "TS11-D",
            "waypoint_mode": None,
            "target_role": program,
            "training_target_path": job["training_target_path"],
            "training_target_sha256": job["training_target_sha256"],
            "teacher_bank_seal_sha256": job["teacher_bank_seal_sha256"],
            "target_lock_sha256": job["target_lock_sha256"],
            "train_archive_path": str(TRAIN_NPZ),
            "train_archive_sha256": sha_file(TRAIN_NPZ),
            "normalizer_path": str(NORMALIZER),
            "normalizer_sha256": sha_file(NORMALIZER),
            "metric_scale_lock_path": str(METRIC_LOCK),
            "metric_scale_lock_sha256": job["metric_lock_sha256"],
            "init_path": str(init_path),
            "init_sha256": sha_file(init_path),
            "init_state_hash": initial_hash,
            "final_state_hash": state_digest,
            "model_file": str(final_path),
            "model_file_sha256": sha_file(final_path),
            "parameter_count": parameter_count(model),
            "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            "trainable_parameter_names": trainable_names,
            "frozen_encoder": "precomputed accepted V-JEPA Q0 cache; no encoder module is instantiated or optimized",
            "encoder_training": 0,
            "encoder_extraction": 0,
            "optimizer": {"name": "AdamW", "lr": 0.0005, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0001},
            "epochs": EPOCHS,
            "effective_batch": EFFECTIVE_BATCH,
            "microbatch_size": microbatch,
            "accumulation_steps": accumulation,
            "steps_per_epoch": 20,
            "optimizer_steps": update_count,
            "scheduler": {"name": "CosineAnnealingLR", "T_max": t_max, "eta_min": 1e-5, "step_location": "optimizer post-step hook"},
            "gradient_clip_norm": 1.0,
            "dropout": 0.1,
            "dtype": "float32",
            "amp": False,
            "compile": False,
            "sampler": "saved shuffled_epoch_order(seed+epoch) shared across all three arms per seed",
            "sampler_orders_path": str(job["sampler_orders_path"]),
            "sampler_orders_sha256": job["sampler_orders_sha256"],
            "cuda_rng_initial_sha256": start_rng_hash,
            "seed_program_order": job["seed_program_order"],
            "outer_train_anchors": count,
            "outer_train_episodes": 220,
            "dev_selection_or_early_stopping": False,
            "checkpoint_rule": "last epoch 160",
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "logical_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
            "elapsed_seconds": time.perf_counter() - started_perf,
            "teacher_parameters_updated": 0,
            "dev_target_values_read": 0,
            "final_heldout_payload_access": 0,
            "old_root_writes": 0,
        }
        atomic_json(attempt_dir / "MODEL_MANIFEST.json", model_manifest)
        scheduler.close()
        lr_trace = attempt_dir / "LR_TRACE_ACTUAL.csv"
        write_lr_trace(lr_trace, scheduler.update_records)
        complete = {
            "schema_version": 1,
            "status": "PASS",
            "completed_utc": utc_now(),
            "job_id": job["job_id"],
            "program_id": program,
            "seed": seed,
            "epochs_completed": EPOCHS,
            "optimizer_steps": update_count,
            "steps_per_epoch": 20,
            "scheduler_tmax": t_max,
            "weights_sha256": model_manifest["model_file_sha256"],
            "model_state_hash": state_digest,
            "lr_trace_sha256": sha_file(lr_trace),
            "training_curves_sha256": sha_file(attempt_dir / "TRAINING_CURVES.csv"),
            "fresh_optimizer": True,
            "fresh_common_init": True,
            "dev_used_for_selection": False,
            "final_heldout_payload_access": 0,
        }
        atomic_json(complete_path, complete)
        atomic_json(run_status, {
            "status": "complete",
            "completed_utc": utc_now(),
            "job_id": job["job_id"],
            "epochs": EPOCHS,
            "optimizer_steps": update_count,
        })
        print(json.dumps({
            "status": "PASS",
            "job_id": job["job_id"],
            "optimizer_steps": update_count,
            "model_sha256": model_manifest["model_file_sha256"],
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        message = str(exc).lower()
        is_oom = ("out of memory" in message) or type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}
        if is_oom and int(job.get("microbatch_size", 0)) == 64 and int(job.get("accumulation_steps", 0)) == 1 and not fallback_path.exists():
            atomic_json(fallback_path, {
                "schema_version": 1,
                "status": "RETRY_WITH_EFFECTIVE_BATCH_PRESERVED",
                "job_id": job.get("job_id"),
                "trigger": f"{type(exc).__name__}: {exc}",
                "microbatch_size": 32,
                "accumulation_steps": 2,
                "effective_batch": EFFECTIVE_BATCH,
                "resume_policy": "fresh attempt from the same common INIT; preserve failed attempt and exact pairing",
                "created_utc": utc_now(),
            })
        atomic_json(run_status, {
            "status": "failed",
            "failed_utc": utc_now(),
            "job_id": job.get("job_id"),
            "program_id": job.get("program_id"),
            "seed": job.get("seed"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "final_heldout_payload_access": 0,
        })
        print(json.dumps({"status": "FAILED", "job_id": job.get("job_id"), "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
