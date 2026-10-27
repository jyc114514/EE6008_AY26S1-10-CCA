"""GPU dense V3 feature extraction plus sealed-model inference for D2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, atomic_write_json, read_json, sha256_file


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
CHECKPOINT = PROJECT_ROOT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA_SOURCE = PROJECT_ROOT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT_ROOT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"


def stack():
    sys.path.insert(0, str(PUBLIC_SRC)); sys.path.insert(1, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips
    clips._DECORD_USABLE = False
    from ee6008.models import load_vjepa2_1_base
    spec = importlib.util.spec_from_file_location("v3_model_dense", V3 / "code/model.py")
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return clips.preprocess_vjepa, load_vjepa2_1_base, module


def decode(video_path: Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened(): raise RuntimeError(video_path)
    values = {}
    try:
        for index in range(int(wanted[-1]) + 1):
            ok, frame = capture.read()
            if not ok: raise RuntimeError(f"decode failed {video_path} frame {index}")
            if index in set(int(item) for item in wanted.tolist()): values[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally: capture.release()
    return values


def extract_features(data: dict[str, np.ndarray], preprocess, load_base, device: torch.device, resume: bool) -> Path:
    n = len(data["dense_id"]); path = BENCH_ROOT / "d2/D2_DENSE_TEMPORAL_FEATURES.npy"
    if path.exists() and resume:
        values = np.load(path, mmap_mode="r")
        if tuple(values.shape) == (n, 2, 16, 768) and np.isfinite(values).all(): return path
    mem = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 2, 16, 768)); mem[:] = np.nan; mem.flush()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=CHECKPOINT, device=device, num_frames=32, resolution=384)
    encoder.eval()
    groups: dict[str, list[int]] = {}
    for i, value in enumerate(data["video_path"]): groups.setdefault(str(value), []).append(i)
    for group_number, (relative, indices) in enumerate(groups.items(), start=1):
        video_path = (RAW_ROOT / relative).resolve()
        wanted = sorted({int(frame) for index in indices for frame in np.floor(np.linspace(int(data["anchor_frame"][index]) - 119, int(data["anchor_frame"][index]), 32)).astype(np.int64).tolist()})
        frame_map = decode(video_path, np.asarray(wanted, dtype=np.int64))
        for index in indices:
            frame_indices = np.floor(np.linspace(int(data["anchor_frame"][index]) - 119, int(data["anchor_frame"][index]), 32)).astype(np.int64)
            frames = np.stack([frame_map[int(item)] for item in frame_indices])
            static = np.repeat(frames[-1:, ...], 32, axis=0)
            for mode, raw in enumerate((frames, static)):
                clip = preprocess(raw, resolution=384).to(device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = encoder(clip.unsqueeze(0))[0].float().reshape(16, 576, 768).mean(dim=1).cpu().numpy().astype(np.float32)
                mem[index, mode] = tokens
        mem.flush()
        print(json.dumps({"group": group_number, "groups": len(groups), "rows_done": sum(len(v) for v in groups.values()) if group_number == len(groups) else max(indices) + 1}), flush=True)
    del mem
    atomic_write_json(BENCH_ROOT / "d2/D2_DENSE_FEATURE_MANIFEST.json", {"status": "PASS", "rows": n, "shape": [n, 2, 16, 768], "source_backbone_hash": sha256_file(CHECKPOINT), "final_test_access": 0})
    return path


def infer(data: dict[str, np.ndarray], features_path: Path, v3_model, device: torch.device) -> int:
    features = np.load(features_path, mmap_mode="r")
    target_norm = read_json(V3 / "data/V3_NORMALIZERS.json")
    video_norm = read_json(V3 / "data/V3_VIDEO_NORMALIZER.json")
    sigma = torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32); video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    state = torch.tensor(data["history_standardized"], device=device)
    mask = torch.tensor(data["history_mask"], device=device)
    rel = torch.tensor(data["history_relative_delta_standardized"], device=device)
    anchor = torch.tensor(data["anchor_command_native"], device=device)
    use = {"S-HIST": (True, "none"), "F-POOL-D": (True, "pool"), "F-TOK-D": (True, "dynamic"), "F-TOK-S": (True, "static"), "V-TOK-D-A": (False, "dynamic"), "V-TOK-S-A": (False, "static")}
    markers = sorted((V3 / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json"))
    for marker_path in markers:
        marker = read_json(marker_path)
        if marker.get("status") != "PASS": continue
        cell = str(marker["cell"]); horizon = int(marker["horizon_steps"]); seed = int(marker["seed"]); run = marker_path.parent.name
        out = BENCH_ROOT / "predictions_dense" / f"{run}.npz"
        if out.exists(): continue
        payload = torch.load(marker_path.parent.parent.parent / "runs" / run / ("formal_pilot_retry1" if (marker_path.parent.parent.parent / "runs" / run / "formal_pilot_retry1/checkpoint_last.pt").exists() else "formal") / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        model = v3_model.V3Full28Predictor(horizon=horizon, seed=seed).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
        use_state, mode = use[cell]
        pred = np.empty((len(state), horizon, 28), dtype=np.float32)
        for start in range(0, len(state), 64):
            end = min(len(state), start + 64)
            if mode == "none": video = np.zeros((end - start, 16, 768), dtype=np.float32)
            else:
                video = np.asarray(features[start:end, 0 if mode in {"dynamic", "pool"} else 1], dtype=np.float32)
                video = (video - video_mean) / video_std
                if mode == "pool": video = video.mean(axis=1, keepdims=True).repeat(16, axis=1)
            with torch.inference_mode():
                residual = model(state[start:end], mask[start:end], rel[start:end], torch.tensor(video, device=device), anchor[start:end], use_state=use_state, use_video=mode != "none")
                pred[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
        part = out.with_suffix(".part")
        with part.open("wb") as handle:
            np.savez_compressed(handle, dense_id=data["dense_id"], episode_index=data["episode_index"], anchor_frame=data["anchor_frame"], anchor_timestamp=data["anchor_timestamp"], predictions_native=pred, horizon_steps=np.asarray(horizon), cell=np.asarray(cell), seed=np.asarray(seed))
        part.replace(out)
        atomic_write_json(out.with_suffix(".json"), {"status": "PASS", "run_id": run, "cell": cell, "seed": seed, "horizon_steps": horizon, "rows": len(pred), "final_test_access": 0})
    return len(markers)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("dense pipeline requires approved CUDA")
    torch.set_num_threads(1); device = torch.device("cuda:0")
    with np.load(BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz", allow_pickle=False) as archive: data = {key: archive[key] for key in archive.files}
    preprocess, load_base, v3_model = stack()
    features = extract_features(data, preprocess, load_base, device, args.resume)
    bundle_count = infer(data, features, v3_model, device)
    atomic_write_json(BENCH_ROOT / "d2/D2_DENSE_PIPELINE_COMPLETE.json", {"status": "PASS", "rows": len(data["dense_id"]), "bundles_seen": bundle_count, "final_test_access": 0})
    print(json.dumps({"status": "PASS", "rows": len(data["dense_id"]), "bundles_seen": bundle_count}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
