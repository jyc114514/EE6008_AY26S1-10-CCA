#!/usr/bin/env python3
"""Run one Core-B B3 raw-video pooled LoRA condition.

This runner is intentionally narrow: one dense selected strategy, one
video/fusion branch, one dynamic/static input and LoRA adaptation.  It uses
the project V-JEPA loader and LoRA seam, never the frozen feature cache, and
keeps canonical inference unmerged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PHYSICAL_BATCH = 8
ACCUMULATION = 4
EFFECTIVE_BATCH = 32
MAX_EPOCHS = 20
MIN_EPOCHS = 5
PATIENCE = 5
MIN_DELTA = 0.0001
HEAD_LR = 0.001
LORA_LR = 0.0001
WEIGHT_DECAY = 0.0001
STEPS = 30
TARGET_DIM = 28
SEED = 20260903
ARM = slice(14, 28)
HAND = slice(0, 14)


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
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(part, **arrays)
    os.replace(part, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(part, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


class RawRegressionDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], data_root: Path, video_input: str):
        self.rows = rows
        self.data_root = data_root
        self.video_input = video_input

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from ee6008.data.clips import decode_prefix, preprocess_vjepa

        row = self.rows[index]
        sample = decode_prefix(self.data_root / row["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        frames = sample.frames
        if self.video_input == "static":
            frames = np.repeat(frames[:1], frames.shape[0], axis=0)
        clip = preprocess_vjepa(frames, resolution=384)
        if tuple(clip.shape) != (3, 32, 384, 384):
            raise RuntimeError(f"unexpected clip shape: {tuple(clip.shape)}")
        return (
            clip,
            torch.from_numpy(row["state_action_history"]),
            torch.from_numpy(row["target"]),
            torch.tensor(row["episode_index"], dtype=torch.long),
        )


class PooledRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, STEPS * TARGET_DIM))

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value).reshape(-1, STEPS, TARGET_DIM)


class B3Model(nn.Module):
    def __init__(self, encoder: nn.Module, head: PooledRegression, fusion: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.fusion = fusion


def make_forward(model: B3Model, clips: Tensor, state: Tensor, fixed: Any, train: bool, feature_mean: Tensor, feature_std: Tensor) -> Tensor:
    from ee6008.top_block_finetune import forward_top_two_lora

    tokens = forward_top_two_lora(model.encoder, clips, checkpoint_top_blocks=False)
    pooled = tokens.mean(dim=1).float()
    # Match the train-only feature standardization used by the frozen pooled
    # B2 head. This is part of the F/L comparison contract.
    pooled = (pooled - feature_mean) / feature_std
    if model.fusion:
        pooled = torch.cat([pooled, state.float()], dim=1)
    return model.head(pooled)


def metric_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "standardized": {
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "arm_mae": float(absolute[:, :, ARM].mean()),
            "hand_mae": float(absolute[:, :, HAND].mean()),
            "final_step_mae": float(absolute[:, -1, :].mean()),
        },
        "all_finite": bool(np.isfinite(prediction).all()),
    }


def evaluate(model: B3Model, loader: DataLoader, device: torch.device, fixed: Any, *, target_mean: np.ndarray, target_norm_std: np.ndarray, feature_mean: Tensor, feature_std: Tensor) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    with torch.inference_mode():
        for clips, state, target, episode in loader:
            clips = clips.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            with autocast_context(device):
                output = make_forward(model, clips, state, fixed, train=False, feature_mean=feature_mean, feature_std=feature_std).float().cpu().numpy()
            predictions.append(output)
            targets.append(target.numpy())
            ids.append(episode.numpy())
    prediction_std = np.concatenate(predictions, axis=0)
    target_values_std = np.concatenate(targets, axis=0)
    episode_ids = np.concatenate(ids, axis=0)
    prediction_native = prediction_std * target_norm_std[None, :, :] + target_mean[None, :, :]
    target_native = target_values_std * target_norm_std[None, :, :] + target_mean[None, :, :]
    return {
        "episode_index": episode_ids,
        "targets_standardized": target_values_std,
        "predictions_standardized": prediction_std,
        "targets_native": target_native,
        "predictions_native": prediction_native,
        "metrics": metric_summary(prediction_std, target_values_std),
    }


def load_rows(input_root: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(input_root / "CORE_B_FULL_INPUTS_1p0s.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    for index in range(len(data["episode_index"])):
        episode = int(data["episode_index"][index])
        rows.append({
            "episode_index": episode,
            "split_code": int(data["split_code"][index]),
            "video_path": manifest[episode]["video_path"],
            "state_action_history": data["state_action_history"][index].astype(np.float32),
            "target": data["targets_standardized"][index].astype(np.float32),
        })
    if sum(row["split_code"] == 0 for row in rows) != 2741 or sum(row["split_code"] == 1 for row in rows) != 614:
        raise ValueError("B3 train/dev population failed")
    return rows, data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fix-root", type=Path, required=True)
    parser.add_argument("--source-experiment-root", type=Path, required=True)
    parser.add_argument("--input-normalization", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--branch", choices=("video_only", "fusion"), required=True)
    parser.add_argument("--video-input", choices=("dynamic", "static"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "cell": args.condition_id, "physical_gpu": args.physical_gpu, "seed": args.seed, "started_utc": utc_now(), "final_test_accessed": 0}
    atomic_json(run_root / "RUN_STATUS.json", status)
    try:
        if args.physical_gpu not in (0, 1, 2, 3) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            raise RuntimeError("B3 GPU contract failed")
        if args.seed != SEED or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("B3 CUDA/BF16/seed contract failed")
        set_seed(SEED)
        device = torch.device("cuda:0")
        rows, data = load_rows(args.input_root.resolve(), args.manifest.resolve())
        target_norm_path = args.repo_root / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
        target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))["1.0s"]
        target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
        target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
        input_norm_path = args.input_normalization.resolve()
        input_norm_all = json.loads(input_norm_path.read_text(encoding="utf-8"))
        input_norm_key = "dynamic_pooled" if args.video_input == "dynamic" else "static_pooled"
        input_norm = input_norm_all[input_norm_key]
        feature_mean = torch.tensor(input_norm["mean"], dtype=torch.float32, device=device)
        feature_std = torch.tensor(input_norm["safe_std"], dtype=torch.float32, device=device)
        if feature_mean.shape != (768,) or feature_std.shape != (768,) or not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all() or torch.any(feature_std <= 0):
            raise ValueError("B3 feature normalization contract failed")
        train_rows = [row for row in rows if row["split_code"] == 0]
        dev_rows = [row for row in rows if row["split_code"] == 1]
        train_loader = DataLoader(RawRegressionDataset(train_rows, args.project_root.resolve() / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d", args.video_input), batch_size=PHYSICAL_BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=1, pin_memory=True)
        dev_loader = DataLoader(RawRegressionDataset(dev_rows, args.project_root.resolve() / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d", args.video_input), batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=1, pin_memory=True)
        sys.path.insert(0, str(args.source_experiment_root.resolve() / "analysis_scripts"))
        import run_full120_lora as legacy
        from ee6008.config import load_config
        from ee6008.lora import LoRAConfig, inject_top_block_lora, lora_named_parameters, base_named_parameters, parameter_counts
        from ee6008.models import load_vjepa2_1_base
        config = load_config(args.repo_root.resolve() / "configs/core_a_benchmark_v3_20260826.yaml")
        encoder, encoder_info = load_vjepa2_1_base(source_root=config.vjepa_source_root, checkpoint=config.checkpoint, device=device, num_frames=32, resolution=384)
        inserted = inject_top_block_lora(encoder, config=LoRAConfig())
        input_dim = 768 + (568 if args.branch == "fusion" else 0)
        head = PooledRegression(input_dim).to(device)
        model = B3Model(encoder, head, args.branch == "fusion").to(device)
        if len(inserted) != 4:
            raise RuntimeError(f"unexpected LoRA insertion count: {len(inserted)}")
        adapter_params = [parameter for _, parameter in lora_named_parameters(encoder)]
        base_digest_before = hashlib.sha256(repr([(name, tuple(value.shape)) for name, value in base_named_parameters(encoder)]).encode()).hexdigest()
        optimizer = torch.optim.AdamW([{"params": adapter_params, "lr": LORA_LR}, {"params": list(head.parameters()), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
        steps_per_epoch = math.ceil(math.ceil(len(train_rows) / PHYSICAL_BATCH) / ACCUMULATION)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS * steps_per_epoch)
        atomic_json(run_root / "RUN_CONFIG_LOCKED.json", {"protocol_id":"core_b_b3_raw_lora_20260908_batch8_accum4", "run_id":args.condition_id, "cell":args.condition_id, "branch":args.branch, "video_input":args.video_input, "adaptation":"lora", "head":"pooled", "physical_gpu":args.physical_gpu, "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_batch":PHYSICAL_BATCH, "gradient_accumulation":ACCUMULATION, "nominal_effective_batch":EFFECTIVE_BATCH, "optimizer":"AdamW", "head_learning_rate":HEAD_LR, "adapter_learning_rate":LORA_LR, "weight_decay":WEIGHT_DECAY, "max_epochs":MAX_EPOCHS, "min_epochs":MIN_EPOCHS, "patience":PATIENCE, "min_delta":MIN_DELTA, "steps_per_epoch":steps_per_epoch, "precision":"BF16", "lora":LoRAConfig().to_dict(), "inserted_modules":list(inserted), "encoder_info":encoder_info, "parameter_counts":parameter_counts(encoder, head), "raw_video_io":True, "canonical_inference":"unmerged_base_plus_adapter", "feature_normalization_key":input_norm_key, "feature_normalization_path":str(input_norm_path), "feature_normalization_sha256":sha256_file(input_norm_path), "feature_normalization_fit":"train rows only", "train_count":2741, "development_validation_count":614, "target_shape":[30,28], "final_test_accessed":0})
        atomic_json(run_root / "MODEL_CONTRACT.json", {"status":"pass", "base_parameters_frozen":True, "adapter_gradient_smoke":"pending_first_backward", "merge_export":"not_used_for_canonical_inference"})
        early = {"best_epoch": None, "best_arm_mae": float("inf"), "best_full_mae": float("inf"), "bad_epochs": 0, "stop_reason": None}
        curves: list[dict[str, object]] = []
        best_payload: dict[str, Any] | None = None
        global_step = 0
        started = time.monotonic()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.encoder.train(); model.head.train(); optimizer.zero_grad(set_to_none=True); pending = 0; cycle_samples = 0; total_loss = 0.0; total_elements = 0; adapter_grad_checked = False
            for batch_index, (clips, state, target, _episode) in enumerate(train_loader, start=1):
                clips = clips.to(device, non_blocking=True); state = state.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                if pending == 0: optimizer.zero_grad(set_to_none=True); cycle_samples = 0
                with autocast_context(device):
                    prediction = make_forward(model, clips, state, None, train=True, feature_mean=feature_mean, feature_std=feature_std)
                    loss_sum = F.mse_loss(prediction.float(), target.float(), reduction="sum")
                if not torch.isfinite(loss_sum): raise FloatingPointError("B3 loss non-finite")
                loss_sum.backward(); total_loss += float(loss_sum.detach().cpu()); total_elements += int(target.numel()); cycle_samples += int(target.shape[0]); pending += 1
                if not adapter_grad_checked:
                    if not adapter_params or any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in adapter_params): raise FloatingPointError("B3 adapter gradient smoke failed")
                    adapter_grad_checked = True
                if pending == ACCUMULATION or batch_index == len(train_loader):
                    denominator = cycle_samples * STEPS * TARGET_DIM
                    for parameter in list(adapter_params) + list(head.parameters()):
                        if parameter.grad is not None: parameter.grad.div_(denominator)
                    trainable = list(adapter_params) + list(head.parameters())
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(norm): raise FloatingPointError("B3 gradient non-finite")
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1; pending = 0
            evaluation = evaluate(model, dev_loader, device, None, target_mean=target_mean, target_norm_std=target_std, feature_mean=feature_mean, feature_std=feature_std)
            arm = float(evaluation["metrics"]["standardized"]["arm_mae"]); full = float(evaluation["metrics"]["standardized"]["mae"]); improved = arm < early["best_arm_mae"] - MIN_DELTA or (abs(arm - early["best_arm_mae"]) <= MIN_DELTA and full < early["best_full_mae"] - MIN_DELTA)
            if improved:
                early.update({"best_epoch":epoch,"best_arm_mae":arm,"best_full_mae":full,"bad_epochs":0}); best_payload={"model_state_dict":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},"epoch":epoch,"global_optimizer_step":global_step,"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"early_stopping_state":dict(early)}
            else: early["bad_epochs"] += 1
            curve={"epoch":epoch,"train_loss":total_loss/total_elements,"validation_standardized_arm_mae":arm,"validation_standardized_full_mae":full,"best_epoch":early["best_epoch"],"bad_epochs":early["bad_epochs"],"global_optimizer_step":global_step,"epoch_seconds":time.monotonic()-started}
            curves.append(curve); atomic_json(run_root/"RUN_STATUS.json", {"status":"running","cell":args.condition_id,"physical_gpu":args.physical_gpu,"seed":SEED,"last_completed_epoch":epoch,"best_epoch":early["best_epoch"],"global_optimizer_step":global_step,"final_test_accessed":0,"updated_utc":utc_now()}); print(f"B3_PROGRESS run={args.condition_id} epoch={epoch}/{MAX_EPOCHS} arm_mae={arm:.6f} full_mae={full:.6f}",flush=True)
            if epoch >= MIN_EPOCHS and early["bad_epochs"] >= PATIENCE: early["stop_reason"]="patience_exhausted"; break
        if best_payload is None: raise RuntimeError("B3 no best checkpoint")
        if early["stop_reason"] is None: early["stop_reason"]="max_epochs_reached" if len(curves)==MAX_EPOCHS else "completed"
        last_payload={"model_state_dict":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},"epoch":len(curves),"global_optimizer_step":global_step,"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"early_stopping_state":dict(early)}
        torch.save(best_payload, run_root/"checkpoint_best.pt"); torch.save(last_payload, run_root/"checkpoint_last.pt")
        model.load_state_dict(best_payload["model_state_dict"], strict=True); best_eval=evaluate(model,dev_loader,device,None,target_mean=target_mean,target_norm_std=target_std,feature_mean=feature_mean,feature_std=feature_std); model.load_state_dict(last_payload["model_state_dict"],strict=True); last_eval=evaluate(model,dev_loader,device,None,target_mean=target_mean,target_norm_std=target_std,feature_mean=feature_mean,feature_std=feature_std)
        for suffix, result in (("best",best_eval),("last",last_eval)): atomic_npz(run_root/f"predictions_{suffix}.npz", **{key:value for key,value in result.items() if key!="metrics"})
        atomic_json(run_root/"RUN_METRICS.json", {"status":"complete","cell":args.condition_id,"branch":args.branch,"video_input":args.video_input,"seed":SEED,"physical_gpu":args.physical_gpu,"best_epoch":early["best_epoch"],"last_epoch":len(curves),"early_stopped":len(curves)<MAX_EPOCHS,"stop_reason":early["stop_reason"],"best":best_eval["metrics"],"last":last_eval["metrics"],"train_count":2741,"development_validation_count":614,"global_optimizer_steps":global_step,"peak_allocated_mib":round(torch.cuda.max_memory_allocated(device)/2**20,3),"peak_reserved_mib":round(torch.cuda.max_memory_reserved(device)/2**20,3),"elapsed_seconds":time.monotonic()-started,"raw_video_io":True,"canonical_inference":"unmerged_base_plus_adapter","final_test_accessed":0,"adapter_gradient_smoke":True,"base_parameters_frozen":True,"base_digest_contract":base_digest_before})
        atomic_csv(run_root/"epoch_curves.csv",curves); atomic_json(run_root/"epoch_curves.json",{"curves":curves}); atomic_json(run_root/"RUN_STATUS.json", {"status":"complete","cell":args.condition_id,"physical_gpu":args.physical_gpu,"seed":SEED,"best_epoch":early["best_epoch"],"last_epoch":len(curves),"global_optimizer_step":global_step,"completed_utc":utc_now(),"exit_code":0,"final_test_accessed":0}); atomic_json(run_root/"RUN_COMPLETE.json", {"status":"complete","cell":args.condition_id,"physical_gpu":args.physical_gpu,"final_test_accessed":0,"canonical_inference":"unmerged_base_plus_adapter"})
        return 0
    except Exception as exc:
        atomic_json(run_root/"RUN_STATUS.json", {"status":"failed","cell":args.condition_id,"physical_gpu":args.physical_gpu,"exit_code":1,"error_type":type(exc).__name__,"error_message":str(exc),"traceback":traceback.format_exc(),"final_test_accessed":0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
