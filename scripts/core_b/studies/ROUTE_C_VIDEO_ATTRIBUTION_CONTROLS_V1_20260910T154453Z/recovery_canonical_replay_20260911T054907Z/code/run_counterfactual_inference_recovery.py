#!/usr/bin/env python3
"""Run one Route C model lane: P1-P4 pixel controls and R1-R3 fusion controls.

P0 is exact-reused from the accepted canonical prediction artifact.  This
runner performs a 16-episode raw P0 identity replay before producing new
conditions.  It is inference-only: no optimizer, gradients, checkpoint write,
or final-test access is possible in this code path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch import Tensor, nn

ROUTE_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
B23 = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
DATA_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
MANIFEST = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.csv"
TARGET_NORM = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
FEATURE_NORM = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"
BASE_CHECKPOINT = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
BASE_SOURCE = PROJECT / "repo/third_party/vjepa2"
INPUT = B23 / "inputs/B2_INPUTS_0125.npz"
PAIRING = ROUTE_ROOT / "ROUTE_C_PAIRING_MANIFEST.csv"
STAGE0 = ROUTE_ROOT / "ROUTE_C_STAGE0_EPISODES.csv"
BATCH_SIZE = 2
STEPS = 30
DIMS = 28
SEED = 20260910
REPLAY_RTOL = 1e-5
REPLAY_ATOL = 1e-5
REPLAY_METRIC_ATOL = 2e-6


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(part, path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(part, **arrays)
    os.replace(part, path)


class PooledRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, STEPS * DIMS))

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value).reshape(-1, STEPS, DIMS)


class B3Model(nn.Module):
    def __init__(self, encoder: nn.Module, head: PooledRegression, fusion: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.fusion = fusion


def load_data() -> tuple[dict[str, np.ndarray], dict[int, dict[str, str]], dict[int, int]]:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    dev = data["split_code"] == 1
    if int(dev.sum()) != 614 or data["targets_standardized"][dev].shape != (614, 30, 28):
        raise ValueError("canonical Route C dev target contract failed")
    with MANIFEST.open(newline="") as handle:
        manifest = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    rows = {int(episode): manifest[int(episode)] for episode in data["episode_index"][dev].tolist()}
    if any(int(rows[int(episode)]["final_test_accessed"]) != 0 for episode in rows):
        raise ValueError("manifest contains final-test access")
    tasks = {int(episode): int(data["original_task_index"][index]) for index, episode in enumerate(data["episode_index"][dev])}
    return data, rows, tasks


def load_models(args: argparse.Namespace, device: torch.device) -> tuple[B3Model, dict[str, Any]]:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(BASE_SOURCE))
    from ee6008.lora import LoRAConfig, inject_top_block_lora
    from ee6008.models import load_vjepa2_1_base
    from ee6008.top_block_finetune import forward_top_two_lora

    encoder, encoder_info = load_vjepa2_1_base(source_root=BASE_SOURCE, checkpoint=BASE_CHECKPOINT, device=device, num_frames=32, resolution=384)
    if args.adaptation == "lora":
        inserted = inject_top_block_lora(encoder, config=LoRAConfig())
        if len(inserted) != 4:
            raise RuntimeError(f"expected four LoRA modules, got {inserted}")
    else:
        inserted = []
    input_dim = 768 + (568 if args.branch == "fusion" else 0)
    head = PooledRegression(input_dim).to(device)
    model = B3Model(encoder, head, args.branch == "fusion").to(device)
    payload = torch.load(Path(args.checkpoint_path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    if args.adaptation == "lora":
        message = model.load_state_dict(state, strict=True)
    else:
        message = model.head.load_state_dict(state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"non-strict checkpoint load: {message}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"encoder_info": encoder_info, "inserted_modules": list(inserted), "checkpoint_payload_keys": list(payload), "forward_source": "ee6008.top_block_finetune.forward_top_two_lora", "use_bf16_autocast": args.adaptation == "lora"}


def state_digest(module: nn.Module, selector: str) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        if selector == "base" and ("lora_A" in name or "lora_B" in name):
            continue
        if selector == "adapter" and "lora_" not in name:
            continue
        if selector == "head" and not name.startswith("head."):
            continue
        if selector == "base" and name.startswith("head."):
            continue
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def autocast_for(use_bf16: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()


def head_forward(model: B3Model, pooled_raw: Tensor, state: Tensor, feature_mean: Tensor, feature_std: Tensor) -> Tensor:
    pooled = (pooled_raw.float() - feature_mean) / feature_std
    value = torch.cat([pooled, state.float()], dim=1) if model.fusion else pooled
    return model.head(value)


def encode(model: B3Model, clips: Tensor, *, use_bf16: bool) -> Tensor:
    sys.path.insert(0, str(REPO / "src"))
    from ee6008.top_block_finetune import forward_top_two_lora
    with autocast_for(use_bf16):
        tokens = forward_top_two_lora(model.encoder, clips, checkpoint_top_blocks=False)
        return tokens.mean(dim=1).float()


def write_condition(condition: str, model: str, ids: np.ndarray, task: np.ndarray, target_std: np.ndarray, target_native: np.ndarray, prediction_std: np.ndarray, prediction_native: np.ndarray, out_root: Path, *, audit: dict[str, Any], status: str = "accepted") -> None:
    condition_root = out_root / condition
    arrays_path = condition_root / "predictions.npz"
    atomic_npz(arrays_path, episode_index=ids.astype(np.int64), original_task_index=task.astype(np.int64), targets_standardized=target_std.astype(np.float32), predictions_standardized=prediction_std.astype(np.float32), targets_native=target_native.astype(np.float32), predictions_native=prediction_native.astype(np.float32))
    atomic_json(condition_root / "RUN_CONFIG_LOCKED.json", {"protocol_id": "core_b_route_c_video_attribution_controls_v1_20260910", "model": model, "condition": condition, "status": status, "optimizer_steps": 0, "checkpoint_weights_modified": False, "final_test_accessed": 0, "target_shape": list(target_std.shape), "prediction_sha256": sha256(arrays_path)})
    atomic_json(condition_root / "INTERVENTION_AUDIT.json", audit)
    atomic_json(condition_root / "RUN_COMPLETE.json", {"status": status, "model": model, "condition": condition, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("B3-FUS-D-F", "B3-FUS-D-L", "B3-V-D-F", "B3-V-D-L"))
    parser.add_argument("--branch", required=True, choices=("fusion", "video_only"))
    parser.add_argument("--adaptation", required=True, choices=("frozen", "lora"))
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--canonical-prediction", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.physical_gpu not in (1, 2, 3) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Route C GPU1/2/3 binding contract failed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in Route C child")
    if args.batch_size <= 0 or args.batch_size > 4:
        raise ValueError("Route C batch size must be 1..4")
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "RUN_STATUS.json", {"status": "running", "model": args.model, "physical_gpu": args.physical_gpu, "optimizer_steps": 0, "final_test_accessed": 0, "started_utc": now()})
    data, manifest, task_map = load_data()
    dev_mask = data["split_code"] == 1
    ids = np.asarray(data["episode_index"][dev_mask], dtype=np.int64)
    tasks = np.asarray(data["original_task_index"][dev_mask], dtype=np.int64)
    target_std = np.asarray(data["targets_standardized"][dev_mask], dtype=np.float64)
    target_native = np.asarray(data["targets"][dev_mask], dtype=np.float64)
    state = np.asarray(data["state_action_history"][dev_mask], dtype=np.float32)
    normalizer = json.loads(TARGET_NORM.read_text())["1.0s"]
    target_mean = np.asarray(normalizer["mean"], dtype=np.float32)
    target_std_scale = np.asarray(normalizer["safe_std"], dtype=np.float32)
    feature_norm = json.loads(FEATURE_NORM.read_text())
    feature_key = "dynamic_pooled"
    feature_mean = torch.tensor(feature_norm[feature_key]["mean"], dtype=torch.float32, device="cuda")
    feature_std = torch.tensor(feature_norm[feature_key]["safe_std"], dtype=torch.float32, device="cuda")
    if feature_mean.shape != (768,) or feature_std.shape != (768,) or not torch.isfinite(feature_std).all() or torch.any(feature_std <= 0):
        raise ValueError("feature normalization contract failed")
    device = torch.device("cuda:0")
    model, model_info = load_models(args, device)
    before_hashes = {selector: state_digest(model, selector) for selector in ("base", "adapter", "head")}
    canonical = np.load(args.canonical_prediction, allow_pickle=False)
    canonical_ids = np.asarray(canonical["episode_index"], dtype=np.int64)
    stage0_ids = {int(row["episode_id"]) for row in csv.DictReader(STAGE0.open(newline=""))}
    replay_indices = [index for index, episode in enumerate(ids.tolist()) if int(episode) in stage0_ids]
    if len(replay_indices) != 16:
        raise RuntimeError(f"stage0 identity replay set is {len(replay_indices)}, expected 16")
    canonical_index = {int(episode): index for index, episode in enumerate(canonical_ids.tolist())}
    replay_diffs: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(replay_indices), args.batch_size):
            batch_indices = replay_indices[start:start + args.batch_size]
            clips = []
            for index in batch_indices:
                from ee6008.data.clips import decode_prefix, preprocess_vjepa
                row = manifest[int(ids[index])]
                video = DATA_ROOT / row["video_path"]
                sample = decode_prefix(video, duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
                clip = preprocess_vjepa(sample.frames, resolution=384)
                if tuple(clip.shape) != (3, 32, 384, 384):
                    raise RuntimeError(f"identity clip shape {clip.shape}")
                clips.append(clip)
            raw = encode(model, torch.stack(clips).to(device), use_bf16=model_info["use_bf16_autocast"])
            prediction = head_forward(model, raw, torch.tensor(state[batch_indices], device=device), feature_mean, feature_std).float().cpu().numpy()
            for local, index in enumerate(batch_indices):
                expected = canonical["predictions_standardized"][canonical_index[int(ids[index])]]
                replay_diffs.append(float(np.max(np.abs(prediction[local] - expected))))
    max_replay = max(replay_diffs)
    if max_replay > REPLAY_ATOL + REPLAY_RTOL * max(1.0, float(np.max(np.abs(canonical["predictions_standardized"])) )):
        raise RuntimeError(f"identity replay failed max_abs={max_replay}")
    np.random.seed(SEED)
    p_predictions = {condition: np.empty((len(ids), STEPS, DIMS), dtype=np.float32) for condition in ("P1", "P2", "P3", "P4")}
    chrono_raw = np.empty((len(ids), 768), dtype=np.float32)
    pixel_audits: dict[str, dict[str, Any]] = {condition: {"condition": condition, "changed_frame_count": 0, "position_maps": {}, "optimizer_steps": 0, "final_test_accessed": 0} for condition in p_predictions}
    from apply_pixel_interventions import apply_pixel_intervention
    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            batch_indices = list(range(start, min(start + args.batch_size, len(ids))))
            base_frames = []
            for index in batch_indices:
                from ee6008.data.clips import decode_prefix, preprocess_vjepa
                row = manifest[int(ids[index])]
                sample = decode_prefix(DATA_ROOT / row["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
                base_frames.append(sample.frames)
            chronological_clips = torch.stack([preprocess_vjepa(frames, resolution=384) for frames in base_frames]).to(device)
            chrono_features = encode(model, chronological_clips, use_bf16=model_info["use_bf16_autocast"])
            chrono_raw[batch_indices] = chrono_features.cpu().numpy()
            state_batch = torch.tensor(state[batch_indices], device=device)
            for condition in p_predictions:
                transformed = []
                for local, index in enumerate(batch_indices):
                    changed, positions = apply_pixel_intervention(base_frames[local], condition, seed=SEED, episode_id=int(ids[index]))
                    transformed.append(preprocess_vjepa(changed, resolution=384))
                    pixel_audits[condition]["position_maps"][str(int(ids[index]))] = positions.tolist()
                    if not np.array_equal(changed, base_frames[local]): pixel_audits[condition]["changed_frame_count"] += 1
                features = encode(model, torch.stack(transformed).to(device), use_bf16=model_info["use_bf16_autocast"])
                p_predictions[condition][batch_indices] = head_forward(model, features, state_batch, feature_mean, feature_std).float().cpu().numpy()
    inverse = lambda prediction: prediction * target_std_scale[None, :, :] + target_mean[None, :, :]
    for condition, predictions in p_predictions.items():
        audit = pixel_audits[condition]
        audit["prediction_max_abs_finite"] = bool(np.isfinite(predictions).all())
        audit["intervention_changed_expected"] = bool(audit["changed_frame_count"] == len(ids))
        prediction_native = inverse(predictions).astype(np.float32)
        write_condition(condition=f"RC-{args.model}-{condition}", model=args.model, ids=ids, task=tasks, target_std=target_std, target_native=target_native, prediction_std=predictions, prediction_native=prediction_native, out_root=args.output_root, audit=audit)
    if args.branch == "fusion":
        pairing = {int(row["recipient_episode"]): row for row in csv.DictReader(PAIRING.open(newline=""))}
        id_index = {int(episode): index for index, episode in enumerate(ids.tolist())}
        norm_features = (chrono_raw - feature_mean.cpu().numpy()) / feature_std.cpu().numpy()
        for condition, donor_key in (("R1", None), ("R2", "same_task_donor"), ("R3", "cross_task_donor")):
            eligible = [int(episode) for episode in ids.tolist() if donor_key is None or pairing[int(episode)][donor_key] != ""]
            prediction = np.empty((len(eligible), STEPS, DIMS), dtype=np.float32)
            state_indices = np.asarray([id_index[episode] for episode in eligible], dtype=np.int64)
            if condition == "R1":
                feature_input = np.zeros((len(eligible), 768), dtype=np.float32)
            else:
                donors = np.asarray([int(pairing[episode][donor_key]) for episode in eligible], dtype=np.int64)
                feature_input = norm_features[np.asarray([id_index[int(donor)] for donor in donors], dtype=np.int64)]
            with torch.inference_mode():
                for start in range(0, len(eligible), 128):
                    part = slice(start, min(start + 128, len(eligible)))
                    raw_or_normalized = torch.tensor(feature_input[part], device=device, dtype=torch.float32)
                    state_part = torch.tensor(state[state_indices[part]], device=device)
                    value = torch.cat([raw_or_normalized, state_part], dim=1)
                    prediction[part] = model.head(value).float().cpu().numpy()
            prediction_native = inverse(prediction).astype(np.float32)
            audit = {"condition": condition, "eligible_N": len(eligible), "recipient_ids": eligible, "state_target_unchanged": True, "donor_future_target_used": False, "optimizer_steps": 0, "final_test_accessed": 0, "video_feature_changed_expected": True}
            write_condition(condition=f"RC-{args.model}-{condition}", model=args.model, ids=ids[state_indices], task=tasks[state_indices], target_std=target_std[state_indices], target_native=target_native[state_indices], prediction_std=prediction, prediction_native=prediction_native, out_root=args.output_root, audit=audit)
    after_hashes = {selector: state_digest(model, selector) for selector in ("base", "adapter", "head")}
    if before_hashes != after_hashes:
        raise RuntimeError(f"checkpoint/model state changed during inference: {before_hashes} != {after_hashes}")
    atomic_json(args.output_root / "CHECKPOINT_IMMUTABILITY.json", {"before": before_hashes, "after": after_hashes, "unchanged": True, "optimizer_steps": 0, "checkpoint_weights_modified": False})
    atomic_json(args.output_root / "IDENTITY_REPLAY.json", {"N": len(replay_indices), "max_prediction_abs_diff": max_replay, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL, "primary_metric_atol": REPLAY_METRIC_ATOL, "status": "PASS"})
    atomic_json(args.output_root / "RUN_STATUS.json", {"status": "complete", "model": args.model, "physical_gpu": args.physical_gpu, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now(), "identity_replay": "PASS"})
    atomic_json(args.output_root / "RUN_COMPLETE.json", {"status": "complete", "model": args.model, "physical_gpu": args.physical_gpu, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})
    print(json.dumps({"status": "complete", "model": args.model, "new_conditions": list(p_predictions) + (["R1", "R2", "R3"] if args.branch == "fusion" else []), "identity_replay_max_abs": max_replay}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
