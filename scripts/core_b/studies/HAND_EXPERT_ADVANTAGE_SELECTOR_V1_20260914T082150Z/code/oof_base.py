#!/usr/bin/env python3
"""Train-only cross-fitted Hand base proposals for the selector study."""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
INPUT_PATH = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
NORMALIZER_PATH = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
THRESHOLD_PATH = REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/MOTION_THRESHOLDS_LOCKED.json"
OOF_SEED = 20260932
FOLDS = 3
STEPS = 30
HAND_DIMS = 14
CONTEXT_DIM = 1336
HIDDEN_DIM = 160
PHYSICAL_BATCH = 8
ACCUMULATION = 4
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def reset_linear(linear: nn.Linear, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); linear.reset_parameters()


class HandResidualProposal(nn.Module):
    """The E-F-SP-U-aligned Hand residual branch, with no Arm output."""

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        self.input = nn.Linear(CONTEXT_DIM, HIDDEN_DIM)
        self.activation = nn.GELU()
        self.output = nn.Linear(HIDDEN_DIM, STEPS * HAND_DIMS)
        reset_linear(self.input, seed)
        reset_linear(self.output, seed + 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(context))).reshape(-1, STEPS, HAND_DIMS)


def load_contract() -> dict[str, Any]:
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train_mask = data["split_code"] == 0
    dev_mask = data["split_code"] == 1
    if int(train_mask.sum()) != 2741 or int(dev_mask.sum()) != 614:
        raise RuntimeError("train/dev count contract failed")
    if data["state_action_history"].shape[1] != 568 or data["dynamic_pooled"].shape[1] != 768:
        raise RuntimeError("context dimension contract failed")
    with np.load(INPUT_PATH, allow_pickle=False) as _:
        pass
    normalizer = json.loads(NORMALIZER_PATH.read_text(encoding="utf-8"))["1.0s"]
    thresholds = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["thresholds"]
    return {"data": data, "train_mask": train_mask, "dev_mask": dev_mask, "mean": np.asarray(normalizer["mean"], dtype=np.float32), "std": np.asarray(normalizer["safe_std"], dtype=np.float32), "thresholds": {key: float(value) for key, value in thresholds.items()}}


def fold_assignment(episode_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(episode_ids, dtype=np.uint64)
    return ((ids * np.uint64(2654435761) + np.uint64(1013904223)) % np.uint64(FOLDS)).astype(np.int64)


def hand_motion(target: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    first = np.abs(target[:, :1, :14] - persistence[:, :1, :14]).mean(axis=2)
    transitions = np.abs(np.diff(target[:, :, :14], axis=1)).mean(axis=2)
    return np.concatenate((first, transitions), axis=1)


def predict(model: nn.Module, context: np.ndarray, persistence: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(context), PHYSICAL_BATCH):
            x = torch.from_numpy(context[start:start + PHYSICAL_BATCH]).to(device)
            p = torch.from_numpy(persistence[start:start + PHYSICAL_BATCH]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                residual = model(x)
            output.append((p[:, :, :14] + residual.float()).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_base(train_context: np.ndarray, train_persistence: np.ndarray, train_target: np.ndarray, run_dir: Path, seed: int, role: str, device: torch.device) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "role": role, "seed": seed, "final_test_access": 0, "optimizer_steps": 0})
    set_seed(seed); torch.set_num_threads(1)
    residual = train_target[:, :, :14] - train_persistence[:, :, :14]
    dataset = TensorDataset(torch.from_numpy(train_context), torch.from_numpy(residual), torch.from_numpy(train_persistence[:, :, :14]))
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed + 17), num_workers=0, pin_memory=device.type == "cuda")
    model = HandResidualProposal(seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(loader) / ACCUMULATION)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch)
    config = {"protocol_id": "core_b_hand_expert_advantage_selector_oof_20260914", "role": role, "seed": seed, "input_dim": CONTEXT_DIM, "hidden_dim": HIDDEN_DIM, "output_shape": [STEPS, HAND_DIMS], "target": "Hand target minus Hand persistence", "physical_batch": PHYSICAL_BATCH, "gradient_accumulation": ACCUMULATION, "epochs": EPOCHS, "steps_per_epoch": steps_per_epoch, "optimizer_steps": EPOCHS * steps_per_epoch, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "input_sha256": digest(INPUT_PATH), "normalizer_sha256": digest(NORMALIZER_PATH), "dev_target_used_in_fit": False, "encoder_or_lora_modified": False, "final_test_access": 0}
    if role.startswith("fold_"):
        config["fold_training_rows"] = int(len(train_context)); config["heldout_prediction_after_fit"] = True
    atomic_json(run_dir / "RUN_CONFIG_LOCKED.json", config)
    curve = []; global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total = 0.0; batches = 0
        for batch_index, (context, residual_batch, persistence_batch) in enumerate(loader, start=1):
            context = context.to(device, non_blocking=True); residual_batch = residual_batch.to(device, non_blocking=True); persistence_batch = persistence_batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                residual_pred = model(context); loss = F.mse_loss((persistence_batch + residual_pred).float(), (persistence_batch + residual_batch).float(), reduction="mean")
            if not torch.isfinite(loss): raise FloatingPointError("non-finite OOF base loss")
            (loss / ACCUMULATION).backward(); total += float(loss.detach().cpu()); batches += 1
            if batch_index % ACCUMULATION == 0 or batch_index == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                if not torch.isfinite(norm): raise FloatingPointError("non-finite OOF base gradient")
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
        curve.append({"epoch": epoch, "loss": total / batches, "optimizer_steps": global_step})
        atomic_json(status_path, {"status": "running", "role": role, "seed": seed, "last_completed_epoch": epoch, "optimizer_steps": global_step, "final_test_access": 0})
    if global_step != EPOCHS * steps_per_epoch:
        raise RuntimeError("OOF base optimizer-step contract failed")
    payload = {"role": role, "seed": seed, "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}
    atomic_torch(run_dir / "checkpoint_best.pt", payload); atomic_torch(run_dir / "checkpoint_last.pt", payload)
    reloaded = HandResidualProposal(seed=seed).to(device); reload_payload = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False); reloaded.load_state_dict(reload_payload["model_state_dict"], strict=True)
    pred = predict(reloaded, train_context, train_persistence, device)
    atomic_npz(run_dir / "TRAIN_PREDICTIONS.npz", predictions_hand_standardized=pred, persistence_hand_standardized=train_persistence[:, :, :14], episode_count=np.asarray([len(train_context)], dtype=np.int64))
    atomic_json(run_dir / "EPOCH_CURVES.json", {"status": "PASS", "rows": curve})
    (run_dir / "EPOCH_CURVES.csv").write_text("epoch,loss,optimizer_steps\n" + "\n".join(f"{row['epoch']},{row['loss']},{row['optimizer_steps']}" for row in curve) + "\n", encoding="utf-8")
    atomic_json(run_dir / "RUN_COMPLETE.json", {"status": "complete", "role": role, "seed": seed, "optimizer_steps": global_step, "strict_checkpoint_reload": "PASS", "prediction_shape": list(pred.shape), "final_test_access": 0, "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "TRAIN_PREDICTIONS.npz")})
    atomic_json(status_path, {"status": "complete", "role": role, "seed": seed, "optimizer_steps": global_step, "final_test_access": 0, "completed_utc": now()})
    return {"role": role, "seed": seed, "status": "complete", "optimizer_steps": global_step, "prediction": pred, "checkpoint_sha256": digest(run_dir / "checkpoint_best.pt"), "prediction_sha256": digest(run_dir / "TRAIN_PREDICTIONS.npz")}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); contract = load_contract(); data = contract["data"]; train_mask = contract["train_mask"]; dev_mask = contract["dev_mask"]
    state = data["state_action_history"].astype(np.float32); dynamic = data["dynamic_pooled"].astype(np.float32); context_all = np.concatenate((dynamic, state), axis=1)
    target = data["targets_standardized"].astype(np.float32); persistence = data["persistence_standardized"].astype(np.float32); ids = data["episode_index"].astype(np.int64); tasks = data["original_task_index"].astype(np.int64)
    train_ids = ids[train_mask]; folds = fold_assignment(train_ids)
    if len(np.unique(train_ids)) != len(train_ids) or sorted(np.unique(folds).tolist()) != [0, 1, 2]: raise RuntimeError("fold assignment support contract failed")
    assignment_rows = [{"episode_index": int(episode), "original_task_index": int(task), "fold": int(fold)} for episode, task, fold in zip(train_ids, tasks[train_mask], folds)]
    write_csv(args.root / "oof/OOF_FOLD_ASSIGNMENT.csv", assignment_rows); atomic_json(args.root / "oof/OOF_FOLD_ASSIGNMENT.json", {"status": "PASS", "fold_count": FOLDS, "rows": assignment_rows, "coverage": len(assignment_rows), "unique_train_ids": len(np.unique(train_ids)), "assignment_seed": OOF_SEED, "hash_rule": "(episode_id*2654435761+1013904223)%3"})
    device = torch.device(args.device); oof_pred = np.full((len(train_ids), STEPS, HAND_DIMS), np.nan, dtype=np.float32); acceptance = []
    for fold in range(FOLDS):
        train_positions = np.flatnonzero(folds != fold); heldout_positions = np.flatnonzero(folds == fold)
        if set(train_ids[train_positions].tolist()) & set(train_ids[heldout_positions].tolist()): raise RuntimeError("OOF fold overlap")
        run = train_base(context_all[train_mask][train_positions], persistence[train_mask][train_positions], target[train_mask][train_positions], args.root / "runs" / f"base_fold_{fold}", OOF_SEED + fold * 1000, f"fold_{fold}", device)
        heldout_pred = predict(HandResidualProposal(seed=OOF_SEED + fold * 1000).to(device), context_all[train_mask][heldout_positions], persistence[train_mask][heldout_positions], device) if False else None
        # Reload the accepted fold checkpoint for held-out prediction only.
        model = HandResidualProposal(seed=OOF_SEED + fold * 1000).to(device); payload = torch.load(args.root / "runs" / f"base_fold_{fold}/checkpoint_best.pt", map_location="cpu", weights_only=False); model.load_state_dict(payload["model_state_dict"], strict=True)
        heldout_pred = predict(model, context_all[train_mask][heldout_positions], persistence[train_mask][heldout_positions], device)
        oof_pred[heldout_positions] = heldout_pred
        acceptance.append({"run": f"base_fold_{fold}", "heldout_rows": int(len(heldout_positions)), "fit_rows": int(len(train_positions)), "overlap": 0, "prediction_shape": list(heldout_pred.shape), "strict_reload": "PASS", "status": "PASS", "final_test_access": 0})
    if not np.isfinite(oof_pred).all(): raise RuntimeError("OOF coverage is incomplete")
    all_train = train_mask
    all_run = train_base(context_all[all_train], persistence[all_train], target[all_train], args.root / "runs/base_all_train", OOF_SEED + 3000, "all_train", device)
    all_model = HandResidualProposal(seed=OOF_SEED + 3000).to(device); all_payload = torch.load(args.root / "runs/base_all_train/checkpoint_best.pt", map_location="cpu", weights_only=False); all_model.load_state_dict(all_payload["model_state_dict"], strict=True)
    dev_pred = predict(all_model, context_all[dev_mask], persistence[dev_mask], device)
    atomic_npz(args.root / "runs/base_all_train/DEV_BASE_PREDICTIONS.npz", predictions_hand_standardized=dev_pred, episode_index=ids[dev_mask], original_task_index=tasks[dev_mask])
    acceptance.append({"run": "base_all_train", "fit_rows": int(train_mask.sum()), "heldout_rows": 0, "overlap": 0, "prediction_shape_dev": list(dev_pred.shape), "strict_reload": "PASS", "status": "PASS", "final_test_access": 0})
    p_hand = persistence[train_mask][:, :, :14]; y_hand = target[train_mask][:, :, :14]; e_p = np.abs(p_hand - y_hand); e_n = np.abs(oof_pred - y_hand); advantage = e_p.mean(axis=2) - e_n.mean(axis=2); nonzero = np.abs(advantage[np.abs(advantage) > 0]); tau = float(np.median(nonzero)) if len(nonzero) else 0.0
    if not np.isfinite(tau) or tau <= 0: raise RuntimeError("OOF tau is non-finite or zero")
    hard = (advantage > 0).astype(np.float32); soft = (1.0 / (1.0 + np.exp(-np.clip(advantage / tau, -6, 6)))).astype(np.float32)
    motion = hand_motion(target[train_mask], persistence[train_mask]); motion_event = (motion >= contract["thresholds"]["HAND_STEP_Q75"]).astype(np.float32)
    atomic_npz(args.root / "oof/OOF_ADVANTAGE_TARGETS.npz", episode_index=train_ids, original_task_index=tasks[train_mask], fold=folds, targets_hand_standardized=y_hand, persistence_hand_standardized=p_hand, oof_predictions_hand_standardized=oof_pred, advantage=advantage.astype(np.float32), hard_label=hard, soft_target=soft, motion_event_label=motion_event)
    oof_summary = {"status": "PASS", "train_rows": int(train_mask.sum()), "fold_count": FOLDS, "coverage": int(np.isfinite(oof_pred).all(axis=(1,2)).sum()), "expected_coverage": int(train_mask.sum()), "fold_counts": {str(fold): int((folds == fold).sum()) for fold in range(FOLDS)}, "hard_prevalence": float(hard.mean()), "soft_mean": float(soft.mean()), "motion_event_prevalence": float(motion_event.mean()), "tau": tau, "oof_prediction_shape": list(oof_pred.shape), "dev_base_prediction_shape": list(dev_pred.shape), "dev_targets_used_in_fit": False, "final_test_access": 0, "fold_overlap": 0, "all_train_base_checkpoint_sha256": digest(args.root / "runs/base_all_train/checkpoint_best.pt")}
    atomic_json(args.root / "oof/OOF_SUMMARY.json", oof_summary); (args.root / "oof/OOF_SUMMARY.md").write_text(f"# OOF summary\n\nStatus: `PASS`; 3-fold train-only coverage `{oof_summary['coverage']}/{oof_summary['expected_coverage']}`; fold counts `{oof_summary['fold_counts']}`; tau=`{tau:.9f}`; hard advantage prevalence=`{hard.mean():.6%}`. Development targets were not used in OOF fitting.\n", encoding="utf-8")
    write_csv(args.root / "oof/OOF_ACCEPTANCE.csv", acceptance); atomic_json(args.root / "oof/OOF_ACCEPTANCE.json", {"status": "PASS", "rows": acceptance, "all_rows_accepted": True, "final_test_access": 0})
    print(json.dumps({"status": "PASS", "folds": FOLDS, "oof_coverage": oof_summary["coverage"], "tau": tau, "hard_prevalence": float(hard.mean()), "dev_base_shape": list(dev_pred.shape), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
