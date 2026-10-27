#!/usr/bin/env python3
"""Run the locked HP-SINGLE or HP-PROTO cell for Route PROTO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from model import (
    ConditionalPrototypeDecoder,
    HandDecoder,
    SelectorHead,
    compose_hand_prediction,
    loss_components,
    set_all_seeds,
    state_hash,
    total_loss,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
DCT_ROOT = REPO / "reports/core_b/HAND_DCT_DUAL_LOSS_FACTORIAL_V1_20260915T050907Z"
MODE_ROOT = ROOT / "mode_audit"
SCALE_LOCK = ROOT / "configs/PROTOTYPE_LOSS_SCALE_LOCKED.json"
SEED = 20260938
FOLDS = 3
STEPS = 30
HAND = 14
CONTEXT = 1336
BASE_BATCH = 8
BASE_ACCUMULATION = 4
BASE_EPOCHS = 20
SELECTOR_BATCH = 32
SELECTOR_EPOCHS = 40
SELECTOR_MIN_DELTA = 1e-5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
K_MODE = 2
CLUSTER_SEED = 20260937
CELLS = ("HP-SINGLE", "HP-PROTO")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(part, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        torch.save(payload, handle)
    os.replace(part, path)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def fold_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761) + np.uint64(1013904223)) % np.uint64(FOLDS)).astype(np.int64)


def selector_calibration_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return (((ids * np.uint64(2246822519) + np.uint64(3266489917)) % np.uint64(10)) == 0)


def calibration_assignment(ids: np.ndarray) -> np.ndarray:
    return np.asarray([int(hashlib.sha256(f"HD-DUAL-LOSS-CALIBRATION:{int(item)}".encode()).hexdigest()[:8], 16) % 4 == 0 for item in ids], dtype=bool)


def load_contract() -> dict[str, Any]:
    data = load_npz(INPUT)
    train = data["split_code"] == 0
    dev = data["split_code"] == 1
    if int(train.sum()) != 2741 or int(dev.sum()) != 614:
        raise RuntimeError("canonical split contract failed")
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    scale = np.asarray(normalizer["safe_std"], dtype=np.float32)
    loss_contract = json.loads((DCT_ROOT / "TEMPORAL_LOSS_CONTRACT.json").read_text(encoding="utf-8"))
    scale_lock = json.loads(SCALE_LOCK.read_text(encoding="utf-8"))
    if scale_lock.get("status") != "LOCKED" or scale_lock.get("mode_count") != 2 or scale_lock.get("dct_K") != 16:
        raise RuntimeError("prototype scale contract failed")
    if loss_contract.get("lambda_fd", 0) <= 0:
        raise RuntimeError("DCT lambda contract failed")
    oof = load_npz(DCT_ROOT / "oof/HD-D1-L1_OOF.npz")
    train_ids = data["episode_index"][train].astype(np.int64)
    if not np.array_equal(oof["episode_index"], train_ids):
        raise RuntimeError("source OOF train IDs failed")
    mode_assignments: list[dict[str, str]] = []
    with (MODE_ROOT / "CLUSTER_ASSIGNMENTS.csv").open(encoding="utf-8", newline="") as handle:
        mode_assignments = list(csv.DictReader(handle))
    if len(mode_assignments) != len(train_ids) or [int(row["episode_index"]) for row in mode_assignments] != train_ids.tolist():
        raise RuntimeError("mode assignment IDs failed")
    mode_labels = np.asarray([int(row["K2"]) for row in mode_assignments], dtype=np.int64)
    centers_payload = load_npz(MODE_ROOT / "PROTOTYPE_CENTERS.npz")
    centers_flat = centers_payload["centers_K2"].astype(np.float64)
    if centers_flat.shape != (2, 16 * 14):
        raise RuntimeError(f"global K2 center shape {centers_flat.shape} failed")
    centers_whitened = centers_flat.reshape(2, 16, 14)
    coefficient_mean = centers_payload["coefficient_mean"].astype(np.float64)
    coefficient_scale = centers_payload["coefficient_scale"].astype(np.float64)
    centers_native = centers_whitened * coefficient_scale[None, :, :] + coefficient_mean[None, :, :]
    dynamic = data["dynamic_pooled"].astype(np.float32)
    state = data["state_action_history"].astype(np.float32)
    context = np.concatenate((dynamic, state), axis=1)
    state_summary = state.mean(axis=1, keepdims=True).astype(np.float32)
    p_std_train = oof["persistence_hand_standardized"].astype(np.float32)
    t_std_train = oof["targets_hand_standardized"].astype(np.float32)
    p_native_train = oof["persistence_hand_native"].astype(np.float32)
    t_native_train = oof["targets_hand_native"].astype(np.float32)
    target_coeff = np.einsum("kt,etj->ekj", dct_basis_np()[:16], t_native_train - p_native_train).astype(np.float32)
    locked = load_npz(LOCKED_ARM)
    dev_ids = data["episode_index"][dev].astype(np.int64)
    dev_tasks = data["original_task_index"][dev].astype(np.int64)
    if not np.array_equal(locked["episode_index"], dev_ids) or not np.array_equal(locked["original_task_index"], dev_tasks):
        raise RuntimeError("locked Arm IDs failed")
    return {
        "data": data,
        "train_mask": train,
        "dev_mask": dev,
        "context": context,
        "state_summary": state_summary,
        "mean": mean,
        "scale": scale,
        "safe_scale": np.asarray(loss_contract["safe_scale_j"], dtype=np.float32),
        "lambda_fd": float(loss_contract["lambda_fd"]),
        "lambda_mode": float(scale_lock["lambda_mode"]),
        "lambda_refine": float(scale_lock["lambda_refine"]),
        "train_ids": train_ids,
        "train_tasks": data["original_task_index"][train].astype(np.int64),
        "dev_ids": dev_ids,
        "dev_tasks": dev_tasks,
        "p_std_train": p_std_train,
        "t_std_train": t_std_train,
        "p_native_train": p_native_train,
        "t_native_train": t_native_train,
        "target_coeff": target_coeff,
        "mode_labels": mode_labels,
        "centers_native": centers_native.astype(np.float32),
        "coefficient_mean": coefficient_mean.astype(np.float32),
        "coefficient_scale": coefficient_scale.astype(np.float32),
        "locked_arm_std": locked["predictions_standardized"][:, :, 14:28].astype(np.float32),
        "locked_arm_native": locked["predictions_native"][:, :, 14:28].astype(np.float32),
        "input_sha256": digest(INPUT),
        "normalizer_sha256": digest(NORMALIZER),
        "locked_arm_sha256": digest(LOCKED_ARM),
        "dct_root_sha256": digest(DCT_ROOT / "qa/CHECKSUMS.sha256"),
    }


def dct_basis_np(length: int = STEPS) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / length * (t + 0.5) * k) * np.sqrt(2.0 / length)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def fit_fold_centers(contract: dict[str, Any], fit_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.einsum("kt,etj->ekj", dct_basis_np()[:16], contract["t_native_train"] - contract["p_native_train"])
    whitened = ((coefficients - contract["coefficient_mean"]) / contract["coefficient_scale"]).reshape(len(coefficients), -1)
    km = KMeans(n_clusters=K_MODE, init="k-means++", n_init=1, max_iter=300, random_state=CLUSTER_SEED, algorithm="lloyd").fit(whitened[fit_positions])
    centers = km.cluster_centers_.reshape(K_MODE, 16, HAND) * contract["coefficient_scale"][None, :, :] + contract["coefficient_mean"][None, :, :]
    labels = km.predict(whitened[fit_positions]).astype(np.int64)
    return centers.astype(np.float32), labels


def predict_single(model: HandDecoder, context: np.ndarray, p_std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(context), SELECTOR_BATCH):
            x = to_device(context[start : start + SELECTOR_BATCH], device)
            p = to_device(p_std[start : start + SELECTOR_BATCH], device)
            result.append((p + model(x)).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def prototype_outputs(model: ConditionalPrototypeDecoder, context: np.ndarray, p_native: np.ndarray, contract: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    soft: list[np.ndarray] = []
    map_values: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    components: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    mean = to_device(contract["mean"][:, :HAND], device)
    scale = to_device(contract["scale"][:, :HAND], device)
    with torch.inference_mode():
        for start in range(0, len(context), SELECTOR_BATCH):
            stop = start + SELECTOR_BATCH
            probs, coeff, residual = model(to_device(context[start:stop], device))
            p = to_device(p_native[start:stop], device)
            native = p.unsqueeze(1) + residual
            comp_std = (native - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
            soft_std = (probs.unsqueeze(-1).unsqueeze(-1) * comp_std).sum(dim=1)
            map_std = comp_std[torch.arange(len(probs), device=device), probs.argmax(dim=1)]
            soft.append(soft_std.cpu().numpy())
            map_values.append(map_std.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            components.append(comp_std.cpu().numpy())
            coefficients.append(coeff.cpu().numpy())
    return {"soft_std": np.concatenate(soft).astype(np.float32), "map_std": np.concatenate(map_values).astype(np.float32), "probabilities": np.concatenate(probabilities).astype(np.float32), "components_std": np.concatenate(components).astype(np.float32), "coefficients": np.concatenate(coefficients).astype(np.float32)}


def selector_predict(model: SelectorHead, context: np.ndarray, summary: np.ndarray, proposal_std: np.ndarray, p_std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    qs: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    magnitude = np.abs(proposal_std - p_std).mean(axis=2, keepdims=True).astype(np.float32)
    with torch.inference_mode():
        for start in range(0, len(context), SELECTOR_BATCH):
            stop = start + SELECTOR_BATCH
            q = model.probability(to_device(context[start:stop], device), to_device(summary[start:stop], device), to_device(magnitude[start:stop], device))
            p = to_device(p_std[start:stop], device)
            n = to_device(proposal_std[start:stop], device)
            predictions.append(compose_hand_prediction(p, n, q).cpu().numpy())
            qs.append(q.cpu().numpy())
    return np.concatenate(qs).astype(np.float32), np.concatenate(predictions).astype(np.float32)


def train_single_base(cell: str, context: np.ndarray, p_std: np.ndarray, t_std: np.ndarray, p_native: np.ndarray, t_native: np.ndarray, contract: dict[str, Any], run_dir: Path, seed: int, device: torch.device) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "final_test_access": 0})
    set_all_seeds(seed)
    torch.set_num_threads(1)
    dataset = TensorDataset(*(to_device(value, torch.device("cpu")) for value in (context, p_std, t_std, p_native, t_native)))
    loader = DataLoader(dataset, batch_size=BASE_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")
    steps_per_epoch = math.ceil(len(loader) / BASE_ACCUMULATION)
    total_steps = BASE_EPOCHS * steps_per_epoch
    model = HandDecoder(decoder="D1", seed=seed, dct_k=16, hidden_dim=160).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    mean = to_device(contract["mean"][:, :HAND], device)
    scale = to_device(contract["scale"][:, :HAND], device)
    safe = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, BASE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader, start=1):
            if optimizer_steps >= total_steps:
                break
            context_b, p_b, t_b, _p_native_b, tn_b = (value.to(device, non_blocking=True) for value in batch)
            prediction = p_b + model(context_b)
            total, position, fd = total_loss("L1", prediction, t_b, tn_b, mean, scale, safe, lambda_fd)
            if not torch.isfinite(total) or not torch.isfinite(position) or not torch.isfinite(fd):
                raise FloatingPointError(f"non-finite single loss: {cell}")
            (total / BASE_ACCUMULATION).backward()
            loss_value += float(total.detach().cpu())
            batches += 1
            if batch_index % BASE_ACCUMULATION == 0 or batch_index == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"non-finite single gradient: {cell}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
        curves.append({"epoch": epoch, "loss": loss_value / max(batches, 1), "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0})
    if optimizer_steps != total_steps:
        raise RuntimeError(f"single steps {optimizer_steps} != {total_steps}")
    payload = {"cell": cell, "role": run_dir.name, "seed": seed, "decoder": "D1", "loss_mode": "L1", "optimizer_steps": optimizer_steps, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "initial_state_hash": initial_hash}
    atomic_torch(run_dir / "checkpoint_best.pt", payload)
    atomic_torch(run_dir / "checkpoint_last.pt", payload)
    reloaded = HandDecoder(decoder="D1", seed=seed, dct_k=16, hidden_dim=160).to(device)
    reloaded.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    reference = predict_single(model, context[:SELECTOR_BATCH], p_std[:SELECTOR_BATCH], device)
    replay = predict_single(reloaded, context[:SELECTOR_BATCH], p_std[:SELECTOR_BATCH], device)
    replay_error = float(np.max(np.abs(reference - replay)))
    if replay_error > 1e-6:
        raise RuntimeError(f"single checkpoint replay failed: {replay_error}")
    predictions = predict_single(reloaded, context, p_std, device)
    atomic_npz(run_dir / "TRAIN_PREDICTIONS.npz", predictions_hand_standardized=predictions, persistence_hand_standardized=p_std, episode_count=np.asarray([len(context)], dtype=np.int64))
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves})
    write_rows(run_dir / "EPOCH_CURVES.csv", curves)
    config = {"status": "LOCKED", "cell": cell, "role": run_dir.name, "seed": seed, "decoder": "D1", "loss_mode": "L1", "dct_K": 16, "hidden_dim": 160, "physical_batch": BASE_BATCH, "gradient_accumulation": BASE_ACCUMULATION, "epochs": BASE_EPOCHS, "optimizer_steps": total_steps, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "input_sha256": contract["input_sha256"], "normalizer_sha256": contract["normalizer_sha256"], "dev_target_used_in_fit": False, "final_test_access": 0}
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    marker = {"status": "complete", "cell": cell, "role": run_dir.name, "seed": seed, "optimizer_steps": optimizer_steps, "prediction_shape": list(predictions.shape), "strict_checkpoint_reload": "PASS", "next_step_replay_max_abs": replay_error, "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "TRAIN_PREDICTIONS.npz"), "final_test_access": 0}
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "complete", "cell": cell, "role": run_dir.name, "seed": seed, "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()})
    return {"model": reloaded, "predictions": predictions, "marker": marker}


def train_proto_base(cell: str, context: np.ndarray, p_native: np.ndarray, t_std: np.ndarray, t_native: np.ndarray, t_coeff: np.ndarray, labels: np.ndarray, centers_native: np.ndarray, contract: dict[str, Any], run_dir: Path, seed: int, device: torch.device) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "final_test_access": 0})
    set_all_seeds(seed)
    torch.set_num_threads(1)
    dataset = TensorDataset(*(to_device(value, torch.device("cpu")) for value in (context, p_native, t_std, t_native, t_coeff)), torch.from_numpy(labels.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=BASE_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")
    steps_per_epoch = math.ceil(len(loader) / BASE_ACCUMULATION)
    total_steps = BASE_EPOCHS * steps_per_epoch
    model = ConditionalPrototypeDecoder(centers=to_device(centers_native, device), seed=seed, hidden_dim=160).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    mean = to_device(contract["mean"][:, :HAND], device)
    scale = to_device(contract["scale"][:, :HAND], device)
    safe = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    lambda_mode = float(contract["lambda_mode"])
    lambda_refine = float(contract["lambda_refine"])
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, BASE_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader, start=1):
            if optimizer_steps >= total_steps:
                break
            context_b, p_native_b, t_std_b, t_native_b, t_coeff_b, label_b = (value.to(device, non_blocking=True) for value in batch)
            probabilities, coefficients, residual = model(context_b)
            native_components = p_native_b.unsqueeze(1) + residual
            components_std = (native_components - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
            soft_std = (probabilities.unsqueeze(-1).unsqueeze(-1) * components_std).sum(dim=1)
            dual, _position, _fd = total_loss("L1", soft_std, t_std_b, t_native_b, mean, scale, safe, lambda_fd)
            mode_loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), label_b)
            selected_coeff = coefficients[torch.arange(len(label_b), device=device), label_b]
            refine_loss = F.smooth_l1_loss(selected_coeff, t_coeff_b)
            total = dual + lambda_mode * mode_loss + lambda_refine * refine_loss
            if not torch.isfinite(total) or not torch.isfinite(dual) or not torch.isfinite(mode_loss) or not torch.isfinite(refine_loss):
                raise FloatingPointError(f"non-finite prototype loss: {cell}")
            (total / BASE_ACCUMULATION).backward()
            loss_value += float(total.detach().cpu())
            batches += 1
            if batch_index % BASE_ACCUMULATION == 0 or batch_index == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"non-finite prototype gradient: {cell}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
        curves.append({"epoch": epoch, "loss": loss_value / max(batches, 1), "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "role": run_dir.name, "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0})
    if optimizer_steps != total_steps:
        raise RuntimeError(f"prototype steps {optimizer_steps} != {total_steps}")
    payload = {"cell": cell, "role": run_dir.name, "seed": seed, "mode_count": K_MODE, "dct_K": 16, "loss_mode": "L1_PLUS_MODE_REFINEMENT", "optimizer_steps": optimizer_steps, "centers_native": np.asarray(centers_native, dtype=np.float32), "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "initial_state_hash": initial_hash}
    atomic_torch(run_dir / "checkpoint_best.pt", payload)
    atomic_torch(run_dir / "checkpoint_last.pt", payload)
    reloaded = ConditionalPrototypeDecoder(centers=to_device(centers_native, device), seed=seed, hidden_dim=160).to(device)
    reloaded.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    reference = prototype_outputs(model, context[:SELECTOR_BATCH], p_native[:SELECTOR_BATCH], contract, device)["soft_std"]
    replay = prototype_outputs(reloaded, context[:SELECTOR_BATCH], p_native[:SELECTOR_BATCH], contract, device)["soft_std"]
    replay_error = float(np.max(np.abs(reference - replay)))
    if replay_error > 1e-6:
        raise RuntimeError(f"prototype checkpoint replay failed: {replay_error}")
    outputs = prototype_outputs(reloaded, context, p_native, contract, device)
    atomic_npz(run_dir / "TRAIN_PREDICTIONS.npz", predictions_hand_standardized=outputs["soft_std"], mode_probabilities=outputs["probabilities"], persistence_hand_native=p_native, episode_count=np.asarray([len(context)], dtype=np.int64))
    atomic_npz(run_dir / "TRAIN_MODE_OUTPUTS.npz", soft_hand_standardized=outputs["soft_std"], map_hand_standardized=outputs["map_std"], mode_probabilities=outputs["probabilities"], components_hand_standardized=outputs["components_std"], coefficients=outputs["coefficients"])
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves})
    write_rows(run_dir / "EPOCH_CURVES.csv", curves)
    config = {"status": "LOCKED", "cell": cell, "role": run_dir.name, "seed": seed, "mode_count": K_MODE, "dct_K": 16, "loss_mode": "L1_PLUS_MODE_REFINEMENT", "lambda_fd": contract["lambda_fd"], "lambda_mode": lambda_mode, "lambda_refine": lambda_refine, "physical_batch": BASE_BATCH, "gradient_accumulation": BASE_ACCUMULATION, "epochs": BASE_EPOCHS, "optimizer_steps": total_steps, "input_sha256": contract["input_sha256"], "normalizer_sha256": contract["normalizer_sha256"], "dev_target_used_in_fit": False, "final_test_access": 0}
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    marker = {"status": "complete", "cell": cell, "role": run_dir.name, "seed": seed, "optimizer_steps": optimizer_steps, "prediction_shape": list(outputs["soft_std"].shape), "component_shape": list(outputs["components_std"].shape), "mode_probability_shape": list(outputs["probabilities"].shape), "probabilities_sum_max_error": float(np.max(np.abs(outputs["probabilities"].sum(axis=1) - 1.0))), "strict_checkpoint_reload": "PASS", "next_step_replay_max_abs": replay_error, "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "TRAIN_PREDICTIONS.npz"), "final_test_access": 0}
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "complete", "cell": cell, "role": run_dir.name, "seed": seed, "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()})
    return {"model": reloaded, "outputs": outputs, "marker": marker}


def train_selector(cell: str, contract: dict[str, Any], oof_proposal: np.ndarray, run_dir: Path, seed: int, device: torch.device) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    train_mask = contract["train_mask"]
    context = contract["context"][train_mask]
    summary = contract["state_summary"][train_mask]
    p_std = contract["p_std_train"]
    t_std = contract["t_std_train"]
    t_native = contract["t_native_train"]
    p_native = contract["p_native_train"]
    calibration = selector_calibration_assignment(contract["train_ids"])
    fit_positions = np.flatnonzero(~calibration)
    calibration_positions = np.flatnonzero(calibration)
    if len(fit_positions) != 2477 or len(calibration_positions) != 264:
        raise RuntimeError("selector calibration split failed")
    magnitude = np.abs(oof_proposal - p_std).mean(axis=2, keepdims=True).astype(np.float32)
    dataset = TensorDataset(*(to_device(value, torch.device("cpu")) for value in (context, summary, magnitude, p_std, oof_proposal, t_std, t_native, p_native)))
    fit_loader = DataLoader(Subset(dataset, fit_positions.tolist()), batch_size=SELECTOR_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0)
    calibration_loader = DataLoader(Subset(dataset, calibration_positions.tolist()), batch_size=SELECTOR_BATCH, shuffle=False, num_workers=0)
    set_all_seeds(seed)
    torch.set_num_threads(1)
    model = SelectorHead(seed=seed).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(fit_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SELECTOR_EPOCHS * steps_per_epoch)
    mean = to_device(contract["mean"][:, :HAND], device)
    scale = to_device(contract["scale"][:, :HAND], device)
    safe = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    best_value = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    curves: list[dict[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, SELECTOR_EPOCHS + 1):
        model.train()
        fit_loss = 0.0
        batches = 0
        for batch in fit_loader:
            context_b, summary_b, magnitude_b, p_b, n_b, t_b, tn_b, _pn_b = (value.to(device, non_blocking=True) for value in batch)
            q = model.probability(context_b, summary_b, magnitude_b)
            hand = compose_hand_prediction(p_b, n_b, q)
            total, _position, _fd = total_loss("L1", hand, t_b, tn_b, mean, scale, safe, lambda_fd)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite selector loss: {cell}")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite selector gradient: {cell}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            fit_loss += float(total.detach().cpu())
            batches += 1
        calibration_values: list[float] = []
        with torch.inference_mode():
            for batch in calibration_loader:
                context_b, summary_b, magnitude_b, p_b, n_b, t_b, tn_b, _pn_b = (value.to(device, non_blocking=True) for value in batch)
                q = model.probability(context_b, summary_b, magnitude_b)
                hand = compose_hand_prediction(p_b, n_b, q)
                total, _position, _fd = total_loss("L1", hand, t_b, tn_b, mean, scale, safe, lambda_fd)
                calibration_values.append(float(total.cpu()))
        value = float(np.mean(calibration_values))
        if value < best_value - SELECTOR_MIN_DELTA:
            best_value = value
            best_epoch = epoch
            best_state = {key: item.detach().cpu().clone() for key, item in model.state_dict().items()}
        curves.append({"epoch": epoch, "fit_loss": fit_loss / max(batches, 1), "calibration_loss": value, "optimizer_steps": optimizer_steps, "best_epoch": best_epoch, "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(run_dir / "RUN_STATUS.json", {"status": "running", "cell": cell, "stage": "selector", "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "best_epoch": best_epoch, "final_test_access": 0})
    if best_state is None or best_epoch is None:
        raise RuntimeError("selector best state missing")
    last_state = {key: item.detach().cpu().clone() for key, item in model.state_dict().items()}
    atomic_torch(run_dir / "checkpoint_last.pt", {"cell": cell, "seed": seed, "model_state_dict": last_state, "optimizer_steps": optimizer_steps})
    atomic_torch(run_dir / "checkpoint_best.pt", {"cell": cell, "seed": seed, "model_state_dict": best_state, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "initial_state_hash": initial_hash})
    reloaded = SelectorHead(seed=seed).to(device)
    reloaded.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    q, prediction = selector_predict(reloaded, contract["context"][contract["dev_mask"]], contract["state_summary"][contract["dev_mask"]], np.zeros((614, 30, HAND), dtype=np.float32), np.zeros((614, 30, HAND), dtype=np.float32), device)
    # The dummy prediction above is a shape/reload smoke only; formal dev
    # prediction is computed by the caller with the actual base proposal.
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves})
    write_rows(run_dir / "EPOCH_CURVES.csv", curves)
    config = {"status": "LOCKED", "cell": cell, "stage": "selector", "seed": seed, "loss_mode": "L1", "selector_input": "past FUSION context + past state summary + base proposal magnitude", "physical_batch": SELECTOR_BATCH, "epochs": SELECTOR_EPOCHS, "optimizer_steps": optimizer_steps, "selector_fit_rows": int(len(fit_positions)), "selector_calibration_rows": int(len(calibration_positions)), "oof_proposal_used": True, "dev_target_used_in_fit": False, "final_test_access": 0}
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "selector_parameters_only": True, "initial_state_hash": initial_hash, "best_state_hash": state_hash(reloaded), "final_test_access": 0})
    marker = {"status": "complete", "cell": cell, "stage": "selector", "seed": seed, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "strict_checkpoint_reload": "PASS", "final_test_access": 0, "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt")}
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", {"status": "complete", "cell": cell, "stage": "selector", "seed": seed, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()})
    return {"model": reloaded, "marker": marker}


def stage0(cell: str, contract: dict[str, Any], device: torch.device) -> dict[str, object]:
    spec_proto = cell == "HP-PROTO"
    positions = np.flatnonzero(contract["train_mask"])[:32]
    context = contract["context"][positions]
    p_std = contract["data"]["persistence_standardized"][positions, :, :HAND].astype(np.float32)
    t_std = contract["data"]["targets_standardized"][positions, :, :HAND].astype(np.float32)
    p_native = contract["data"]["persistence_native"][positions, :, :HAND].astype(np.float32)
    t_native = contract["data"]["targets_native"][positions, :, :HAND].astype(np.float32)
    mean = to_device(contract["mean"][:, :HAND], device)
    scale = to_device(contract["scale"][:, :HAND], device)
    safe = to_device(contract["safe_scale"], device)
    lambda_fd = torch.tensor(contract["lambda_fd"], dtype=torch.float32, device=device)
    run_dir = ROOT / "runs/stage0" / cell
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = SEED + (10000 if spec_proto else 0)
    set_all_seeds(seed)
    if spec_proto:
        model = ConditionalPrototypeDecoder(centers=to_device(contract["centers_native"], device), seed=seed, hidden_dim=160).to(device)
        before = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        labels = to_device(contract["mode_labels"][:32].astype(np.int64), device).long()
        target_coeff = to_device(contract["target_coeff"][:32], device)
        context_t = to_device(context, device)
        p_native_t = to_device(p_native, device)
        target_std_t = to_device(t_std, device)
        target_native_t = to_device(t_native, device)
        for _ in range(4):
            probabilities, coefficients, residual = model(context_t)
            native_components = p_native_t.unsqueeze(1) + residual
            components_std = (native_components - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
            soft = (probabilities.unsqueeze(-1).unsqueeze(-1) * components_std).sum(dim=1)
            dual, _pos, _fd = total_loss("L1", soft, target_std_t, target_native_t, mean, scale, safe, lambda_fd)
            mode_loss = F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), labels)
            selected = coefficients[torch.arange(32, device=device), labels]
            refine = F.smooth_l1_loss(selected, target_coeff)
            total = dual + contract["lambda_mode"] * mode_loss + contract["lambda_refine"] * refine
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(total) or not torch.isfinite(norm):
                raise FloatingPointError(f"Stage0 prototype non-finite: {cell}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        after = state_hash(model)
        with torch.inference_mode():
            probabilities, coefficients, residual = model(context_t)
            native_components = p_native_t.unsqueeze(1) + residual
            components_std = (native_components - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
            soft = (probabilities.unsqueeze(-1).unsqueeze(-1) * components_std).sum(dim=1)
        checkpoint = run_dir / "checkpoint_stage0.pt"
        atomic_torch(checkpoint, {"cell": cell, "seed": seed, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "centers_native": contract["centers_native"]})
        reload_model = ConditionalPrototypeDecoder(centers=to_device(contract["centers_native"], device), seed=seed, hidden_dim=160).to(device)
        reload_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        with torch.inference_mode():
            replay_prob, replay_coeff, replay_residual = reload_model(context_t)
            replay_native = p_native_t.unsqueeze(1) + replay_residual
            replay_std = (replay_native - mean.unsqueeze(0).unsqueeze(0)) / scale.unsqueeze(0).unsqueeze(0)
            replay_soft = (replay_prob.unsqueeze(-1).unsqueeze(-1) * replay_std).sum(dim=1)
        replay_error = float(torch.max(torch.abs(soft - replay_soft)).cpu())
        selector = SelectorHead(seed=seed + 1000).to(device)
        selector_before = state_hash(selector)
        selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        magnitude = torch.mean(torch.abs(soft - to_device(p_std, device)), dim=2, keepdim=True)
        summary = to_device(contract["state_summary"][positions], device)
        for _ in range(4):
            q = selector.probability(context_t, summary, magnitude)
            hand = compose_hand_prediction(to_device(p_std, device), soft, q)
            total, _pos, _fd = total_loss("L1", hand, target_std_t, target_native_t, mean, scale, safe, lambda_fd)
            total.backward()
            selector_optimizer.step()
            selector_optimizer.zero_grad(set_to_none=True)
        selector_after = state_hash(selector)
        with torch.inference_mode():
            q = selector.probability(context_t, summary, magnitude)
        passed = bool(tuple(soft.shape) == (32, 30, 14) and tuple(probabilities.shape) == (32, 2) and torch.isfinite(soft).all() and torch.isfinite(probabilities).all() and torch.max(torch.abs(probabilities.sum(dim=1) - 1)) <= 1e-6 and before != after and selector_before != selector_after and replay_error <= 1e-6)
        marker = {"status": "PASS" if passed else "FAIL", "cell": cell, "seed": SEED, "base_stage0_seed": seed, "selector_stage0_seed": seed + 1000, "optimizer_steps_base": 4, "optimizer_steps_selector": 4, "prediction_hand_shape": [32, 30, 14], "component_shape": [32, 2, 30, 14], "mode_probability_shape": [32, 2], "probability_sum_max_error": float(torch.max(torch.abs(probabilities.sum(dim=1) - 1)).cpu()), "q_shape": list(q.shape), "q_min": float(q.min()), "q_max": float(q.max()), "finite_loss_gradient": True, "base_parameters_changed": before != after, "selector_parameters_changed": selector_before != selector_after, "strict_checkpoint_reload": "PASS", "next_step_replay_max_abs": replay_error, "physical_gpu": int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1")), "final_test_access": 0, "completed_utc": now()}
    else:
        model = HandDecoder(decoder="D1", seed=seed, dct_k=16, hidden_dim=160).to(device)
        before = state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        context_t = to_device(context, device)
        p_t = to_device(p_std, device)
        t_t = to_device(t_std, device)
        tn_t = to_device(t_native, device)
        for _ in range(4):
            prediction = p_t + model(context_t)
            total, _pos, _fd = total_loss("L1", prediction, t_t, tn_t, mean, scale, safe, lambda_fd)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(total) or not torch.isfinite(norm):
                raise FloatingPointError(f"Stage0 single non-finite: {cell}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        after = state_hash(model)
        with torch.inference_mode():
            prediction = p_t + model(context_t)
        checkpoint = run_dir / "checkpoint_stage0.pt"
        atomic_torch(checkpoint, {"cell": cell, "seed": seed, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}})
        reload_model = HandDecoder(decoder="D1", seed=seed, dct_k=16, hidden_dim=160).to(device)
        reload_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
        with torch.inference_mode():
            replay = p_t + reload_model(context_t)
        replay_error = float(torch.max(torch.abs(prediction - replay)).cpu())
        selector = SelectorHead(seed=seed + 1000).to(device)
        selector_before = state_hash(selector)
        selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        magnitude = torch.mean(torch.abs(prediction - p_t), dim=2, keepdim=True)
        summary = to_device(contract["state_summary"][positions], device)
        for _ in range(4):
            q = selector.probability(context_t, summary, magnitude)
            hand = compose_hand_prediction(p_t, prediction, q)
            total, _pos, _fd = total_loss("L1", hand, t_t, tn_t, mean, scale, safe, lambda_fd)
            total.backward()
            selector_optimizer.step()
            selector_optimizer.zero_grad(set_to_none=True)
        selector_after = state_hash(selector)
        with torch.inference_mode():
            q = selector.probability(context_t, summary, magnitude)
        passed = bool(tuple(prediction.shape) == (32, 30, 14) and tuple(q.shape) == (32, 30, 1) and torch.isfinite(prediction).all() and torch.isfinite(q).all() and float(q.min()) >= 0 and float(q.max()) <= 1 and before != after and selector_before != selector_after and replay_error <= 1e-6)
        marker = {"status": "PASS" if passed else "FAIL", "cell": cell, "seed": SEED, "base_stage0_seed": seed, "selector_stage0_seed": seed + 1000, "optimizer_steps_base": 4, "optimizer_steps_selector": 4, "prediction_hand_shape": list(prediction.shape), "q_shape": list(q.shape), "q_min": float(q.min()), "q_max": float(q.max()), "finite_loss_gradient": True, "base_parameters_changed": before != after, "selector_parameters_changed": selector_before != selector_after, "strict_checkpoint_reload": "PASS", "next_step_replay_max_abs": replay_error, "physical_gpu": int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1")), "final_test_access": 0, "completed_utc": now()}
    atomic_json(run_dir / "RUN_COMPLETE.json", marker)
    atomic_json(run_dir / "RUN_STATUS.json", marker)
    return marker


def run_full(cell: str, contract: dict[str, Any], device: torch.device) -> dict[str, object]:
    train = contract["train_mask"]
    dev = contract["dev_mask"]
    context_train = contract["context"][train]
    context_dev = contract["context"][dev]
    summary_train = contract["state_summary"][train]
    summary_dev = contract["state_summary"][dev]
    p_std_train = contract["p_std_train"]
    t_std_train = contract["t_std_train"]
    p_native_train = contract["p_native_train"]
    t_native_train = contract["t_native_train"]
    folds = fold_assignment(contract["train_ids"])
    run_root = ROOT / "runs" / cell
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": cell, "seed": SEED, "final_test_access": 0})
    oof_proposal = np.full((len(contract["train_ids"]), STEPS, HAND), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    centers_by_fold: list[np.ndarray] = []
    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        if set(contract["train_ids"][fit].tolist()) & set(contract["train_ids"][heldout].tolist()):
            raise RuntimeError("prototype OOF overlap")
        if cell == "HP-SINGLE":
            base = train_single_base(cell, context_train[fit], p_std_train[fit], t_std_train[fit], p_native_train[fit], t_native_train[fit], contract, run_root / f"base_fold_{fold}", SEED + fold * 1000, device)
            fold_prediction = predict_single(base["model"], context_train[heldout], p_std_train[heldout], device)
        else:
            centers, fold_labels = fit_fold_centers(contract, fit)
            centers_by_fold.append(centers)
            atomic_npz(run_root / f"base_fold_{fold}/FOLD_CENTERS.npz", centers_native=centers, fit_labels=fold_labels, fit_episode_index=contract["train_ids"][fit])
            base = train_proto_base(cell, context_train[fit], p_native_train[fit], t_std_train[fit], t_native_train[fit], contract["target_coeff"][fit], fold_labels, centers, contract, run_root / f"base_fold_{fold}", SEED + 20000 + fold * 1000, device)
            fold_prediction = prototype_outputs(base["model"], context_train[heldout], p_native_train[heldout], contract, device)["soft_std"]
        oof_proposal[heldout] = fold_prediction
        fold_rows.append({"fold": fold, "fit_rows": int(len(fit)), "heldout_rows": int(len(heldout)), "overlap": 0, "prediction_shape": list(fold_prediction.shape), "status": "PASS"})
    if not np.isfinite(oof_proposal).all():
        raise RuntimeError("prototype OOF coverage incomplete")
    if cell == "HP-SINGLE":
        all_base = train_single_base(cell, context_train, p_std_train, t_std_train, p_native_train, t_native_train, contract, run_root / "base_all_train", SEED + 3000, device)
        dev_base = predict_single(all_base["model"], context_dev, contract["data"]["persistence_standardized"][dev, :, :HAND].astype(np.float32), device)
        atomic_npz(run_root / "base_all_train/DEV_BASE_PREDICTIONS.npz", predictions_hand_standardized=dev_base, episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
    else:
        all_base = train_proto_base(cell, context_train, p_native_train, t_std_train, t_native_train, contract["target_coeff"], contract["mode_labels"], contract["centers_native"], contract, run_root / "base_all_train", SEED + 23000, device)
        base_outputs = prototype_outputs(all_base["model"], context_dev, contract["data"]["persistence_native"][dev, :, :HAND].astype(np.float32), contract, device)
        dev_base = base_outputs["soft_std"]
        atomic_npz(run_root / "base_all_train/DEV_BASE_MODE_OUTPUTS.npz", soft_hand_standardized=base_outputs["soft_std"], map_hand_standardized=base_outputs["map_std"], mode_probabilities=base_outputs["probabilities"], components_hand_standardized=base_outputs["components_std"], coefficients=base_outputs["coefficients"], episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
    atomic_npz(ROOT / f"oof/{cell}_OOF.npz", episode_index=contract["train_ids"], original_task_index=contract["train_tasks"], fold=folds, targets_hand_standardized=t_std_train, targets_hand_native=t_native_train, persistence_hand_standardized=p_std_train, persistence_hand_native=p_native_train, oof_predictions_hand_standardized=oof_proposal)
    atomic_json(ROOT / f"oof/{cell}_OOF_SUMMARY.json", {"status": "PASS", "cell": cell, "coverage": int(np.isfinite(oof_proposal).all(axis=(1, 2)).sum()), "expected_coverage": 2741, "overlap": 0, "fold_rows": fold_rows, "final_test_access": 0})
    selector_seed = SEED + (25000 if cell == "HP-PROTO" else 5000)
    selector = train_selector(cell, contract, oof_proposal, run_root / "selector", selector_seed, device)
    p_dev_std = contract["data"]["persistence_standardized"][dev, :, :HAND].astype(np.float32)
    p_dev_native = contract["data"]["persistence_native"][dev, :, :HAND].astype(np.float32)
    if cell == "HP-SINGLE":
        q, hand_std = selector_predict(selector["model"], context_dev, summary_dev, dev_base, p_dev_std, device)
        output = {"hand_std": hand_std, "hand_native": hand_std.astype(np.float64) * contract["scale"][None, :, :HAND] + contract["mean"][None, :, :HAND], "q": q, "outputs": ["SINGLE"]}
    else:
        q, gated_soft = selector_predict(selector["model"], context_dev, summary_dev, base_outputs["soft_std"], p_dev_std, device)
        _q_map, gated_map = selector_predict(selector["model"], context_dev, summary_dev, base_outputs["map_std"], p_dev_std, device)
        output = {"mean_std": gated_soft, "map_std": gated_map, "mean_native": gated_soft.astype(np.float64) * contract["scale"][None, :, :HAND] + contract["mean"][None, :, :HAND], "map_native": gated_map.astype(np.float64) * contract["scale"][None, :, :HAND] + contract["mean"][None, :, :HAND], "q": q, "mode_probabilities": base_outputs["probabilities"], "components_std": base_outputs["components_std"], "outputs": ["MAP", "MEAN"]}
    arm_std = contract["locked_arm_std"]
    arm_native = contract["locked_arm_native"]
    if cell == "HP-SINGLE":
        full_std = np.concatenate((output["hand_std"], arm_std), axis=2)
        full_native = np.concatenate((output["hand_native"].astype(np.float32), arm_native), axis=2)
        atomic_npz(run_root / "selector/PREDICTIONS_DEV.npz", hand_predictions_standardized=output["hand_std"], hand_predictions_native=output["hand_native"].astype(np.float32), hand_persistence_standardized=p_dev_std, hand_persistence_native=p_dev_native, hand_base_predictions_standardized=dev_base, q=q, full_predictions_standardized=full_std, full_predictions_native=full_native, locked_arm_standardized=arm_std, locked_arm_native=arm_native, episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
        atomic_npz(ROOT / "predictions/HP-SINGLE_PREDICTIONS_DEV.npz", hand_predictions_standardized=output["hand_std"], hand_predictions_native=output["hand_native"].astype(np.float32), q=q, full_predictions_standardized=full_std, full_predictions_native=full_native, locked_arm_standardized=arm_std, locked_arm_native=arm_native, episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
        marker = {"status": "complete", "cell": cell, "seed": SEED, "outputs": ["SINGLE"], "prediction_hand_shape": list(output["hand_std"].shape), "prediction_full_shape": list(full_std.shape), "q_shape": list(q.shape), "oof_coverage": 2741, "oof_overlap": 0, "base_folds": "3/3", "all_train_base": "complete", "selector": "complete", "strict_checkpoint_reload": "PASS", "final_test_access": 0, "completed_utc": now()}
    else:
        map_full_std = np.concatenate((output["map_std"], arm_std), axis=2)
        mean_full_std = np.concatenate((output["mean_std"], arm_std), axis=2)
        map_full_native = np.concatenate((output["map_native"].astype(np.float32), arm_native), axis=2)
        mean_full_native = np.concatenate((output["mean_native"].astype(np.float32), arm_native), axis=2)
        atomic_npz(run_root / "selector/PREDICTIONS_DEV.npz", map_hand_predictions_standardized=output["map_std"], mean_hand_predictions_standardized=output["mean_std"], map_hand_predictions_native=output["map_native"].astype(np.float32), mean_hand_predictions_native=output["mean_native"].astype(np.float32), hand_persistence_standardized=p_dev_std, hand_persistence_native=p_dev_native, q=q, mode_probabilities=output["mode_probabilities"], components_hand_standardized=output["components_std"], map_full_predictions_standardized=map_full_std, mean_full_predictions_standardized=mean_full_std, map_full_predictions_native=map_full_native, mean_full_predictions_native=mean_full_native, locked_arm_standardized=arm_std, locked_arm_native=arm_native, episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
        atomic_npz(ROOT / "predictions/HP-PROTO_PREDICTIONS_DEV.npz", map_hand_predictions_standardized=output["map_std"], mean_hand_predictions_standardized=output["mean_std"], map_hand_predictions_native=output["map_native"].astype(np.float32), mean_hand_predictions_native=output["mean_native"].astype(np.float32), q=q, mode_probabilities=output["mode_probabilities"], components_hand_standardized=output["components_std"], map_full_predictions_standardized=map_full_std, mean_full_predictions_standardized=mean_full_std, map_full_predictions_native=map_full_native, mean_full_predictions_native=mean_full_native, locked_arm_standardized=arm_std, locked_arm_native=arm_native, episode_index=contract["dev_ids"], original_task_index=contract["dev_tasks"])
        marker = {"status": "complete", "cell": cell, "seed": SEED, "outputs": ["MAP", "MEAN"], "prediction_hand_shape": [614, 30, 14], "prediction_full_shape": [614, 30, 28], "q_shape": list(q.shape), "mode_probability_shape": list(output["mode_probabilities"].shape), "component_shape": list(output["components_std"].shape), "probability_sum_max_error": float(np.max(np.abs(output["mode_probabilities"].sum(axis=1) - 1.0))), "oof_coverage": 2741, "oof_overlap": 0, "base_folds": "3/3", "all_train_base": "complete", "selector": "complete", "strict_checkpoint_reload": "PASS", "final_test_access": 0, "completed_utc": now()}
    atomic_json(run_root / "RUN_COMPLETE.json", marker)
    atomic_json(run_root / "RUN_STATUS.json", marker)
    atomic_json(run_root / "OOF_ACCEPTANCE.json", {"status": "PASS", "rows": fold_rows, "coverage": 2741, "overlap": 0, "final_test_access": 0})
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--phase", choices=("stage0", "full"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise RuntimeError("runner root does not match locked experiment root")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA child runtime unavailable")
    if args.phase == "stage0" and not torch.cuda.is_available():
        raise RuntimeError("real GPU Stage0 requested but CUDA is unavailable")
    contract = load_contract()
    device = torch.device(args.device)
    result = stage0(args.cell, contract, device) if args.phase == "stage0" else run_full(args.cell, contract, device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
