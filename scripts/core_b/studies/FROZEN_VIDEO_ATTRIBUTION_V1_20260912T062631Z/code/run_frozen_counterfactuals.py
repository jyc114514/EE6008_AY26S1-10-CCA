#!/usr/bin/env python3
"""Run one Frozen-only attribution lane with a locked direct-encoder seam."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[0]
STEPS = 30
DIMS = 28
HEAD_BATCH = 8
ENCODER_BATCH = 1
ATOL = 1e-5
RTOL = 1e-5


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def normalized_uuid(value: str) -> str:
    return value.strip().removeprefix("GPU-")


class PooledRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, STEPS * DIMS))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).reshape(-1, STEPS, DIMS)


class FrozenModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: PooledRegression, fusion: bool):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.fusion = fusion


def load_frozen_model(row: dict[str, str], project_root: Path, device: torch.device) -> tuple[FrozenModel, dict[str, object]]:
    repo = project_root / "repo/ee6008-humanoid-vjepa"
    source = project_root / "repo/third_party/vjepa2"
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(source))
    from ee6008.models import load_vjepa2_1_base

    fusion = row["branch"] == "fusion"
    encoder, info = load_vjepa2_1_base(source_root=source, checkpoint=Path(row["base_checkpoint_path"]), device=device, num_frames=32, resolution=384)
    head = PooledRegression(768 + (568 if fusion else 0)).to(device)
    model = FrozenModel(encoder, head, fusion).to(device)
    payload = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("Frozen checkpoint lacks model_state_dict")
    message = model.head.load_state_dict(state, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"Frozen head load was not strict: {message}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"encoder_info": info, "checkpoint_payload_keys": sorted(payload), "checkpoint_epoch": payload.get("epoch"), "fusion": fusion}


def nvidia_uuid(physical_gpu: int) -> str:
    query = "index,uuid,name,utilization.gpu,memory.used,memory.total"
    result = subprocess.run(["nvidia-smi", "-i", str(physical_gpu), f"--query-gpu={query}", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
    return fields[1]


def pixel_intervention(frames: np.ndarray, condition: str, episode_id: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected frame shape {frames.shape}")
    source = np.arange(32, dtype=np.int64)
    if condition == "P1":
        positions = np.zeros(32, dtype=np.int64)
    elif condition == "P2":
        positions = np.full(32, 31, dtype=np.int64)
    elif condition == "P3":
        positions = source[::-1]
    elif condition == "P4":
        positions = np.random.default_rng(seed + int(episode_id)).permutation(32).astype(np.int64)
        if np.array_equal(positions, source):
            positions = np.roll(positions, 1)
    else:
        raise ValueError(condition)
    if np.array_equal(positions, source):
        raise AssertionError(f"identity pixel intervention for {condition}/{episode_id}")
    return frames[positions].copy(), positions


def head_predict(model: FrozenModel, features: np.ndarray, state: np.ndarray, indices: list[int], feature_mean: torch.Tensor, feature_std: torch.Tensor, device: torch.device) -> np.ndarray:
    output = np.empty((len(indices), STEPS, DIMS), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), HEAD_BATCH):
            chunk = indices[start : start + HEAD_BATCH]
            raw = torch.tensor(features[chunk], dtype=torch.float32, device=device)
            state_part = torch.tensor(state[chunk], dtype=torch.float32, device=device)
            pooled = (raw - feature_mean) / feature_std
            value = torch.cat([pooled, state_part], dim=1) if model.fusion else pooled
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model.head(value).float().cpu().numpy()
            output[start : start + len(chunk)] = prediction
    return output


def encode_single(model: FrozenModel, frames: np.ndarray, preprocess, device: torch.device) -> np.ndarray:
    clip = preprocess(frames, resolution=384)
    if tuple(clip.shape) != (3, 32, 384, 384):
        raise RuntimeError(f"unexpected preprocessed clip shape {tuple(clip.shape)}")
    clip_batch = clip.unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pooled = model.encoder(clip_batch).mean(dim=1).float()
    return pooled[0].detach().cpu().numpy().astype(np.float32)


def write_condition(out_root: Path, cell_id: str, model: str, condition: str, ids: np.ndarray, tasks: np.ndarray, targets_std: np.ndarray, targets_native: np.ndarray, prediction_std: np.ndarray, target_mean: np.ndarray, target_scale: np.ndarray, audit: dict[str, object], source_row: dict[str, str]) -> None:
    condition_root = out_root / cell_id
    condition_root.mkdir(parents=True, exist_ok=True)
    prediction_native = (prediction_std * target_scale[None, :, :] + target_mean[None, :, :]).astype(np.float32)
    array_path = condition_root / "predictions.npz"
    part = condition_root / ".predictions.part.npz"
    np.savez_compressed(part, episode_index=ids.astype(np.int64), original_task_index=tasks.astype(np.int64), targets_standardized=targets_std.astype(np.float32), targets_native=targets_native.astype(np.float32), predictions_standardized=prediction_std.astype(np.float32), predictions_native=prediction_native)
    os.replace(part, array_path)
    atomic_json(condition_root / "RUN_CONFIG_LOCKED.json", {"protocol_id": "core_b_frozen_video_attribution_v1_20260912", "cell_id": cell_id, "model": model, "condition": condition, "status": "accepted", "optimizer_steps": 0, "checkpoint_weights_modified": False, "final_test_accessed": 0, "target_shape": list(targets_std.shape), "prediction_sha256": sha256_file(array_path), "checkpoint_sha256": source_row["checkpoint_sha256"], "config_sha256": source_row["config_sha256"], "feature_normalization_sha256": source_row["feature_normalization_sha256"], "target_normalization_sha256": source_row["target_normalization_sha256"]})
    atomic_json(condition_root / "INTERVENTION_AUDIT.json", audit)
    atomic_json(condition_root / "RUN_COMPLETE.json", {"status": "accepted", "cell_id": cell_id, "model": model, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("B3-V-D-F", "B3-FUS-D-F"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--target-normalization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260910)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES binding mismatch")
    if os.environ.get("GPU_WATCHER_PHYSICAL_GPU") not in {None, str(args.physical_gpu)}:
        raise RuntimeError("watcher physical GPU binding mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen child CUDA contract failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "model": args.model, "physical_gpu": args.physical_gpu, "gpu_uuid": nvidia_uuid(args.physical_gpu), "optimizer_steps": 0, "final_test_accessed": 0, "started_utc": now()})
    try:
        with args.registry.open(newline="", encoding="utf-8") as handle:
            rows = {row["model"]: row for row in csv.DictReader(handle)}
        row = rows[args.model]
        with np.load(args.input, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        dev = data["split_code"] == 1
        ids = np.asarray(data["episode_index"][dev], dtype=np.int64)
        tasks = np.asarray(data["original_task_index"][dev], dtype=np.int64)
        state = np.asarray(data["state_action_history"][dev], dtype=np.float32)
        targets_std = np.asarray(data["targets_standardized"][dev], dtype=np.float32)
        targets_native = np.asarray(data["targets"][dev], dtype=np.float32)
        if targets_std.shape != (614, 30, 28) or len(np.unique(ids)) != 614:
            raise RuntimeError("Frozen population/target contract failed")
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            manifest = {int(item["episode_index"]): item for item in csv.DictReader(handle)}
        if any(int(manifest[int(episode)]["final_test_accessed"]) != 0 for episode in ids.tolist()):
            raise RuntimeError("final-test marker found in development manifest")
        with args.pairing.open(newline="", encoding="utf-8") as handle:
            pairing = {int(item["recipient_episode"]): item for item in csv.DictReader(handle)}
        target_norm = json.loads(args.target_normalization.read_text(encoding="utf-8"))["1.0s"]
        target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
        target_scale = np.asarray(target_norm["safe_std"], dtype=np.float32)
        feature_norm = json.loads(Path(row["feature_normalization_path"]).read_text(encoding="utf-8"))["dynamic_pooled"]
        device = torch.device("cuda:0")
        model, model_info = load_frozen_model(row, args.project_root, device)
        before_digest = state_digest(model)
        from ee6008.data import clips as clips_module
        from ee6008.data.clips import decode_prefix, preprocess_vjepa
        clips_module._DECORD_USABLE = False
        data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
        feature_mean = torch.tensor(feature_norm["mean"], dtype=torch.float32, device=device)
        feature_std = torch.tensor(feature_norm["safe_std"], dtype=torch.float32, device=device)
        canonical = np.load(Path(row["canonical_prediction_path"]), allow_pickle=False)
        canonical_index = {int(value): position for position, value in enumerate(canonical["episode_index"].tolist())}
        id_index = {int(value): position for position, value in enumerate(ids.tolist())}
        stage0_ids = [int(item["episode_id"]) for item in csv.DictReader(args.stage0.open(newline="", encoding="utf-8"))]
        stage0_indices = [id_index[episode] for episode in stage0_ids]
        stage0_features = np.full((len(ids), 768), np.nan, dtype=np.float32)
        for index in stage0_indices:
            sample = decode_prefix(data_root / manifest[int(ids[index])]["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
            stage0_features[index] = encode_single(model, sample.frames, preprocess_vjepa, device)
        stage0_prediction = head_predict(model, stage0_features, state, stage0_indices, feature_mean, feature_std, device)
        stage0_expected = np.stack([canonical["predictions_standardized"][canonical_index[int(ids[index])]] for index in stage0_indices]).astype(np.float32)
        stage0_delta = stage0_prediction - stage0_expected
        tolerance = ATOL + RTOL * max(1.0, float(np.max(np.abs(canonical["predictions_standardized"]))))
        stage0_max_abs = float(np.max(np.abs(stage0_delta)))
        atomic_json(args.output_root / "STAGE0_IDENTITY.json", {"status": "PASS" if stage0_max_abs <= tolerance else "FAIL", "N": len(stage0_indices), "episode_ids": stage0_ids, "max_abs": stage0_max_abs, "locked_tolerance": tolerance, "encoder_batch": ENCODER_BATCH, "head_batch": HEAD_BATCH})
        if stage0_max_abs > tolerance:
            raise RuntimeError(f"Frozen Stage0 identity failed max_abs={stage0_max_abs} tolerance={tolerance}")

        conditions = ("P1", "P2", "P3", "P4")
        transformed_features = {condition: np.empty((len(ids), 768), dtype=np.float32) for condition in conditions}
        position_maps = {condition: np.empty((len(ids), 32), dtype=np.int64) for condition in conditions}
        frame_hash_rows: list[dict[str, object]] = []
        for index, episode in enumerate(ids.tolist()):
            sample = decode_prefix(data_root / manifest[int(episode)]["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
            frames = sample.frames
            if not np.isfinite(frames.astype(np.float32)).all():
                raise RuntimeError(f"non-finite decoded frames for {episode}")
            if not np.isfinite(stage0_features[index]).all():
                stage0_features[index] = encode_single(model, frames, preprocess_vjepa, device)
            base_hash = sha256_array(frames)
            row_hashes: dict[str, object] = {"episode_id": int(episode), "base_sha256": base_hash}
            for condition in conditions:
                transformed, positions = pixel_intervention(frames, condition, int(episode), args.shuffle_seed)
                position_maps[condition][index] = positions
                transformed_features[condition][index] = encode_single(model, transformed, preprocess_vjepa, device)
                row_hashes[f"{condition}_sha256"] = sha256_array(transformed)
                row_hashes[f"{condition}_nonidentity"] = bool(not np.array_equal(positions, np.arange(32)))
            frame_hash_rows.append(row_hashes)
        for condition in conditions:
            cell = ("VF-" if row["branch"] == "video_only" else "FF-") + condition
            prediction = head_predict(model, transformed_features[condition], state, list(range(len(ids))), feature_mean, feature_std, device)
            map_path = args.output_root / cell / "frame_maps.npz"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(map_path, episode_index=ids, source_positions=position_maps[condition])
            audit = {"condition": condition, "level": "raw_pixel", "N": len(ids), "source_positions_sha256": sha256_file(map_path), "frame_hash_rows": frame_hash_rows, "multiset_preserved": condition in {"P3", "P4"}, "resort_guard": "direct indexed frames; no timestamp sorting after intervention", "state_target_id_unchanged": True, "optimizer_steps": 0, "final_test_accessed": 0, "checkpoint_state_unchanged": True}
            write_condition(args.output_root, cell, args.model, condition, ids, tasks, targets_std, targets_native, prediction, target_mean, target_scale, audit, row)

        if row["branch"] == "fusion":
            normalized_chrono = (stage0_features - feature_mean.detach().cpu().numpy()) / feature_std.detach().cpu().numpy()
            feature_conditions = {"R1": np.zeros_like(normalized_chrono), "R2": np.empty_like(normalized_chrono), "R3": np.empty_like(normalized_chrono)}
            same_donors: list[int] = []
            cross_donors: list[int] = []
            for episode in ids.tolist():
                pair = pairing[int(episode)]
                same_donors.append(-1 if pair["same_task_donor"] == "" else id_index[int(pair["same_task_donor"])])
                cross_donors.append(id_index[int(pair["cross_task_donor"])])
            for index in range(len(ids)):
                if same_donors[index] >= 0:
                    feature_conditions["R2"][index] = normalized_chrono[same_donors[index]]
                else:
                    feature_conditions["R2"][index] = normalized_chrono[index]
                feature_conditions["R3"][index] = normalized_chrono[cross_donors[index]]
            for condition, feature_input in feature_conditions.items():
                eligible = [index for index in range(len(ids)) if condition != "R2" or same_donors[index] >= 0]
                raw = torch.tensor(feature_input[eligible], dtype=torch.float32, device=device)
                prediction = np.empty((len(eligible), STEPS, DIMS), dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0, len(eligible), HEAD_BATCH):
                        part = list(range(start, min(start + HEAD_BATCH, len(eligible))))
                        state_part = torch.tensor(state[part], dtype=torch.float32, device=device)
                        value = torch.cat([raw[part], state_part], dim=1)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            prediction[part] = model.head(value).float().cpu().numpy()
                recipient_ids = ids[eligible]
                donor_episode_ids = ids[np.asarray(same_donors)[eligible]] if condition == "R2" else ids[np.asarray(cross_donors)[eligible]] if condition == "R3" else np.asarray([], dtype=np.int64)
                audit = {"condition": condition, "level": "normalized_feature", "N": len(eligible), "recipient_ids": recipient_ids.tolist(), "state_sha256": sha256_array(state[eligible]), "target_std_sha256": sha256_array(targets_std[eligible]), "video_feature_input_sha256": sha256_array(feature_input[eligible]), "chronological_video_feature_sha256": sha256_array(normalized_chrono), "donor_episode_ids": donor_episode_ids.tolist(), "zero_definition": "R1 normalized video feature exactly zero" if condition == "R1" else "not_applicable", "donor_future_target_used": False, "recipient_state_target_id_unchanged": True, "optimizer_steps": 0, "final_test_accessed": 0}
                write_condition(args.output_root, "FF-" + condition, args.model, condition, recipient_ids, tasks[eligible], targets_std[eligible], targets_native[eligible], prediction, target_mean, target_scale, audit, row)
        after_digest = state_digest(model)
        if before_digest != after_digest:
            raise RuntimeError("Frozen model state changed during inference")
        atomic_json(args.output_root / "CHECKPOINT_IMMUTABILITY.json", {"before": before_digest, "after": after_digest, "unchanged": True, "checkpoint_sha256": row["checkpoint_sha256"], "optimizer_steps": 0, "checkpoint_weights_modified": False})
        atomic_json(args.output_root / "RUN_STATUS.json", {"status": "complete", "model": args.model, "physical_gpu": args.physical_gpu, "gpu_uuid": nvidia_uuid(args.physical_gpu), "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now(), "stage0_identity": "PASS", "logical_conditions": 5 if row["branch"] == "video_only" else 8})
        atomic_json(args.output_root / "RUN_COMPLETE.json", {"status": "complete", "model": args.model, "physical_gpu": args.physical_gpu, "gpu_uuid": nvidia_uuid(args.physical_gpu), "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})
        print(json.dumps({"status": "complete", "model": args.model, "stage0_max_abs": stage0_max_abs, "logical_conditions": 5 if row["branch"] == "video_only" else 8}, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(args.output_root / "RUN_FAILED.json", {"status": "failed", "model": args.model, "error": f"{type(exc).__name__}: {exc}", "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})
        atomic_json(status_path, {"status": "failed", "model": args.model, "error": f"{type(exc).__name__}: {exc}", "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
