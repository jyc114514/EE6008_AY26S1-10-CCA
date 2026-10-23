#!/usr/bin/env python3
"""Train one matched, selector-only Hand expert gate on train OOF data."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from selector_model import SelectorHead, compose_hand_prediction, state_hash


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
OOF_PATH = ROOT / "oof/OOF_ADVANTAGE_TARGETS.npz"
DEV_BASE_PATH = ROOT / "runs/base_all_train/DEV_BASE_PREDICTIONS.npz"
LOCKED_ARM_PATH = REPO_ROOT / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
THRESHOLD_PATH = REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/MOTION_THRESHOLDS_LOCKED.json"
SELECTOR_SEED = 20260932
STEPS = 30
HAND_DIMS = 14
CONTEXT_DIM = 1336
PHYSICAL_BATCH = 32
MAX_EPOCHS = 40
MIN_EPOCHS = 10
PATIENCE = 6
MIN_DELTA = 1e-5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
MODES = ("SEL-U", "SEL-M", "SEL-AH", "SEL-AS")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    with part.open("wb") as handle: np.savez_compressed(handle, **arrays)
    os.replace(part, path)


def atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle: torch.save(payload, handle)
    os.replace(part, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_contract() -> dict[str, Any]:
    with np.load(INPUT_PATH, allow_pickle=False) as archive: data = {key: archive[key].copy() for key in archive.files}
    train_mask = data["split_code"] == 0; dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614: raise RuntimeError("train/dev count contract failed")
    state = data["state_action_history"].astype(np.float32); dynamic = data["dynamic_pooled"].astype(np.float32)
    context = np.concatenate((dynamic, state), axis=1)
    state_summary = state.mean(axis=1, keepdims=True).astype(np.float32)
    with np.load(OOF_PATH, allow_pickle=False) as archive: oof = {key: archive[key].copy() for key in archive.files}
    train_ids = data["episode_index"][train_mask].astype(np.int64); train_tasks = data["original_task_index"][train_mask].astype(np.int64)
    if not np.array_equal(oof["episode_index"], train_ids) or not np.array_equal(oof["original_task_index"], train_tasks): raise RuntimeError("OOF selector IDs do not match canonical train order")
    if oof["oof_predictions_hand_standardized"].shape != (2741, STEPS, HAND_DIMS): raise RuntimeError("OOF Hand proposal shape failed")
    with np.load(DEV_BASE_PATH, allow_pickle=False) as archive: dev_base = {key: archive[key].copy() for key in archive.files}
    dev_ids = data["episode_index"][dev_mask].astype(np.int64); dev_tasks = data["original_task_index"][dev_mask].astype(np.int64)
    if not np.array_equal(dev_base["episode_index"], dev_ids) or dev_base["predictions_hand_standardized"].shape != (614, STEPS, HAND_DIMS): raise RuntimeError("all-train development base contract failed")
    with np.load(LOCKED_ARM_PATH, allow_pickle=False) as archive: locked = {key: archive[key].copy() for key in archive.files}
    if not np.array_equal(locked["episode_index"], dev_ids) or locked["predictions_standardized"].shape != (614, STEPS, 28): raise RuntimeError("locked Arm source contract failed")
    if not np.array_equal(locked["targets_standardized"], data["targets_standardized"][dev_mask]) or not np.array_equal(locked["persistence_standardized"], data["persistence_standardized"][dev_mask]): raise RuntimeError("locked Arm target pairing failed")
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["thresholds"]
    return {"data": data, "train_mask": train_mask, "dev_mask": dev_mask, "context": context, "state_summary": state_summary, "oof": oof, "dev_base": dev_base, "locked": locked, "thresholds": {key: float(value) for key, value in thresholds.items()}}


def selector_calibration_assignment(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.uint64)
    return (((ids * np.uint64(2246822519) + np.uint64(3266489917)) % np.uint64(10)) == 0)


def loss_for_mode(mode: str, q: torch.Tensor, hand_prediction: torch.Tensor, target: torch.Tensor, motion: torch.Tensor, hard: torch.Tensor, soft: torch.Tensor, pos_weights: dict[str, torch.Tensor]) -> torch.Tensor:
    if mode == "SEL-U": return F.l1_loss(hand_prediction, target, reduction="mean")
    if mode == "SEL-M": return F.binary_cross_entropy(q.squeeze(-1), motion, weight=1.0 + (pos_weights[mode] - 1.0) * motion)
    if mode == "SEL-AH": return F.binary_cross_entropy(q.squeeze(-1), hard, weight=1.0 + (pos_weights[mode] - 1.0) * hard)
    if mode == "SEL-AS": return F.binary_cross_entropy(q.squeeze(-1), soft)
    raise ValueError(mode)


def evaluate_objective(model: SelectorHead, mode: str, batch: tuple[torch.Tensor, ...], device: torch.device, pos_weights: dict[str, torch.Tensor]) -> float:
    context, summary, magnitude, persistence, neural, target, motion, hard, soft = (value.to(device) for value in batch)
    with torch.inference_mode():
        q = model.probability(context, summary, magnitude)
        hand_prediction = persistence + q * (neural - persistence)
        value = loss_for_mode(mode, q, hand_prediction, target, motion, hard, soft, pos_weights)
    return float(value.cpu())


def predict_selector(model: SelectorHead, context: np.ndarray, summary: np.ndarray, magnitude: np.ndarray, persistence: np.ndarray, neural: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); q_values = []; hand_values = []
    with torch.inference_mode():
        for start in range(0, len(context), PHYSICAL_BATCH):
            x = torch.from_numpy(context[start:start + PHYSICAL_BATCH]).to(device); s = torch.from_numpy(summary[start:start + PHYSICAL_BATCH]).to(device); m = torch.from_numpy(magnitude[start:start + PHYSICAL_BATCH]).to(device)
            p = torch.from_numpy(persistence[start:start + PHYSICAL_BATCH]).to(device); n = torch.from_numpy(neural[start:start + PHYSICAL_BATCH]).to(device)
            q = model.probability(x, s, m); hand = p + q * (n - p)
            q_values.append(q.cpu().numpy()); hand_values.append(hand.cpu().numpy())
    return np.concatenate(q_values).astype(np.float32), np.concatenate(hand_values).astype(np.float32)


def train_selector(cell_id: str, root: Path, device: torch.device) -> dict[str, Any]:
    if cell_id not in MODES: raise ValueError(cell_id)
    contract = load_contract(); data = contract["data"]; train_mask = contract["train_mask"]; dev_mask = contract["dev_mask"]; oof = contract["oof"]
    context_train = contract["context"][train_mask]; summary_train = contract["state_summary"][train_mask]
    p_train = oof["persistence_hand_standardized"].astype(np.float32); n_train = oof["oof_predictions_hand_standardized"].astype(np.float32); target_train = oof["targets_hand_standardized"].astype(np.float32)
    magnitude_train = np.abs(n_train - p_train).mean(axis=2, keepdims=True).astype(np.float32)
    motion_train = oof["motion_event_label"].astype(np.float32); hard_train = oof["hard_label"].astype(np.float32); soft_train = oof["soft_target"].astype(np.float32)
    ids_train = oof["episode_index"].astype(np.int64); calib_mask = selector_calibration_assignment(ids_train); fit_mask = ~calib_mask
    if int(calib_mask.sum()) < 200 or int(fit_mask.sum()) < 200: raise RuntimeError("selector train/calibration support too small")
    tensors = [torch.from_numpy(value) for value in (context_train, summary_train, magnitude_train, p_train, n_train, target_train, motion_train, hard_train, soft_train)]
    dataset = TensorDataset(*tensors); fit_indices = np.flatnonzero(fit_mask); calib_indices = np.flatnonzero(calib_mask)
    fit_dataset = torch.utils.data.Subset(dataset, fit_indices.tolist()); calib_dataset = torch.utils.data.Subset(dataset, calib_indices.tolist())
    fit_loader = DataLoader(fit_dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(SELECTOR_SEED + 17), num_workers=0); calib_loader = DataLoader(calib_dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0)
    train_prevalences = {"SEL-M": float(motion_train[fit_mask].mean()), "SEL-AH": float(hard_train[fit_mask].mean())}
    pos_weights = {mode: torch.tensor((1.0 - train_prevalences[mode]) / train_prevalences[mode], dtype=torch.float32, device=device) for mode in ("SEL-M", "SEL-AH")}
    set_seed(SELECTOR_SEED); model = SelectorHead(seed=SELECTOR_SEED).to(device); initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * max(1, math.ceil(len(fit_loader))))
    run_dir = root / "runs" / cell_id; run_dir.mkdir(parents=True, exist_ok=True); status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": SELECTOR_SEED, "final_test_access": 0, "optimizer_steps": 0})
    config = {"protocol_id": "core_b_hand_expert_advantage_selector_v1_20260914", "cell_id": cell_id, "selector_mode": cell_id, "seed": SELECTOR_SEED, "input": "past FUSION context + past state summary + OOF/all-train base residual magnitude + learned time embedding", "input_dim": CONTEXT_DIM, "output_shape": [STEPS, 1], "physical_batch": PHYSICAL_BATCH, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "scheduler": "CosineAnnealingLR", "loss": {"SEL-U": "Hand reconstruction MAE", "SEL-M": "class-weighted BCE motion event", "SEL-AH": "class-weighted BCE OOF hard advantage", "SEL-AS": "BCE OOF soft advantage"}[cell_id], "oof_path": str(OOF_PATH), "oof_sha256": digest(OOF_PATH), "input_sha256": digest(INPUT_PATH), "locked_arm_path": str(LOCKED_ARM_PATH), "locked_arm_sha256": digest(LOCKED_ARM_PATH), "selector_fit_rows": int(fit_mask.sum()), "selector_calibration_rows": int(calib_mask.sum()), "selector_calibration_assignment": "episode-ID hash; shared by all four cells", "dev_target_used_in_fit": False, "future_motion_input": False, "future_advantage_input": False, "base_expert_frozen": True, "arm_locked": True, "final_test_access": 0}
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config); atomic_json(run_dir / "MODEL_CONTRACT.json", {"status": "PASS", "model": "selector_only_hand_gate", "selector_parameters_only": True, "initial_state_hash": initial_hash, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "arm_prediction_source_locked": True, "final_test_access": 0})
    best = {"epoch": None, "value": float("inf"), "bad": 0}; best_state = None; curves = []; optimizer_steps = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); total = 0.0; batches = 0; optimizer.zero_grad(set_to_none=True)
        for batch in fit_loader:
            context, summary, magnitude, persistence, neural, target, motion, hard, soft = (value.to(device) for value in batch)
            q = model.probability(context, summary, magnitude); hand_prediction = persistence + q * (neural - persistence); loss = loss_for_mode(cell_id, q, hand_prediction, target, motion, hard, soft, pos_weights)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite selector loss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if not torch.isfinite(norm): raise FloatingPointError("non-finite selector gradient")
            optimizer.step(); optimizer.zero_grad(set_to_none=True); total += float(loss.detach().cpu()); batches += 1; optimizer_steps += 1; scheduler.step()
        calib_values = [evaluate_objective(model, cell_id, batch, device, pos_weights) for batch in calib_loader]; calib_value = float(np.mean(calib_values)); train_value = total / batches
        improved = calib_value < best["value"] - MIN_DELTA
        if improved: best = {"epoch": epoch, "value": calib_value, "bad": 0}; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: best["bad"] += 1
        curves.append({"epoch": epoch, "fit_loss": train_value, "calibration_loss": calib_value, "optimizer_steps": optimizer_steps, "best_epoch": best["epoch"], "bad_epochs": best["bad"], "learning_rate": scheduler.get_last_lr()[0]})
        atomic_json(status_path, {"status": "running", "cell_id": cell_id, "seed": SELECTOR_SEED, "last_completed_epoch": epoch, "optimizer_steps": optimizer_steps, "best_epoch": best["epoch"], "final_test_access": 0})
    if best_state is None: raise RuntimeError("selector best checkpoint missing")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; atomic_torch(run_dir / "checkpoint_last.pt", {"cell_id": cell_id, "seed": SELECTOR_SEED, "model_state_dict": last_state, "epoch": MAX_EPOCHS}); atomic_torch(run_dir / "checkpoint_best.pt", {"cell_id": cell_id, "seed": SELECTOR_SEED, "model_state_dict": best_state, "epoch": best["epoch"], "initial_state_hash": initial_hash})
    reloaded = SelectorHead(seed=SELECTOR_SEED).to(device); payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False); reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    context_dev = contract["context"][dev_mask]; summary_dev = contract["state_summary"][dev_mask]; p_dev = contract["dev_base"]["predictions_hand_standardized"].astype(np.float32) * 0.0 + data["persistence_standardized"][dev_mask, :, :14].astype(np.float32); n_dev = contract["dev_base"]["predictions_hand_standardized"].astype(np.float32); magnitude_dev = np.abs(n_dev - p_dev).mean(axis=2, keepdims=True).astype(np.float32)
    q_dev, hand_dev = predict_selector(reloaded, context_dev, summary_dev, magnitude_dev, p_dev, n_dev, device)
    locked_arm = contract["locked"]["predictions_standardized"][:, :, 14:28].astype(np.float32); full_dev = np.concatenate((hand_dev, locked_arm), axis=2)
    atomic_npz(run_dir / "PREDICTIONS_DEV.npz", hand_predictions_standardized=hand_dev, hand_persistence_standardized=p_dev, hand_base_predictions_standardized=n_dev, q=q_dev, full_predictions_standardized=full_dev, locked_arm_standardized=locked_arm, episode_index=contract["locked"]["episode_index"], original_task_index=contract["locked"]["original_task_index"])
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        target_dev = archive["targets_standardized"][dev_mask].astype(np.float32)
    hand_mae = float(np.abs(hand_dev - target_dev[:, :, :14]).mean()); full_mae = float(np.abs(full_dev - target_dev).mean()); p_hand_mae = float(np.abs(p_dev - target_dev[:, :, :14]).mean()); metrics = {"hand_mae": hand_mae, "full28_mae": full_mae, "persistence_hand_mae": p_hand_mae, "hand_skill_vs_persistence": (p_hand_mae - hand_mae) / p_hand_mae, "q_mean": float(q_dev.mean()), "q_collapse_fraction": float(np.mean((q_dev < .05) | (q_dev > .95))), "q_hash": hashlib.sha256(q_dev.tobytes()).hexdigest(), "prediction_hand_shape": list(hand_dev.shape), "prediction_full_shape": list(full_dev.shape)}
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curves}); write_csv(run_dir / "EPOCH_CURVES.csv", curves); atomic_json(run_dir / "RUN_METRICS.json", {"status": "complete", "cell_id": cell_id, "seed": SELECTOR_SEED, "best_epoch": best["epoch"], "optimizer_steps": optimizer_steps, "fit_rows": int(fit_mask.sum()), "calibration_rows": int(calib_mask.sum()), "metrics": metrics, "initial_state_hash": initial_hash, "best_state_hash": state_hash(reloaded), "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "checkpoint_last_sha256": digest(run_dir / "checkpoint_last.pt"), "prediction_sha256": digest(run_dir / "PREDICTIONS_DEV.npz"), "dev_target_used_in_fit": False, "final_test_access": 0})
    atomic_json(status_path, {"status": "complete", "cell_id": cell_id, "seed": SELECTOR_SEED, "best_epoch": best["epoch"], "optimizer_steps": optimizer_steps, "final_test_access": 0, "completed_utc": now()}); atomic_json(run_dir / "RUN_COMPLETE.json", {"status": "complete", "cell_id": cell_id, "seed": SELECTOR_SEED, "optimizer_steps": optimizer_steps, "prediction_hand_shape": list(hand_dev.shape), "prediction_full_shape": list(full_dev.shape), "strict_checkpoint_reload": "PASS", "final_test_access": 0, "checkpoint_best_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "PREDICTIONS_DEV.npz")})
    return {"cell_id": cell_id, "status": "complete", "best_epoch": best["epoch"], "optimizer_steps": optimizer_steps, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cell", choices=MODES, required=True); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--device", default="cpu"); args = parser.parse_args(); result = train_selector(args.cell, args.root, torch.device(args.device)); print(json.dumps(result, sort_keys=True, default=json_default)); return 0


if __name__ == "__main__": raise SystemExit(main())
