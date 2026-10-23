#!/usr/bin/env python3
"""Frozen V-JEPA D/S temporal-token extraction for V3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
PRETRAINED = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA_SOURCE = PROJECT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
PYTHON_BATCH = 4
TOKEN_SHAPE = (16, 768)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def import_stack():
    if str(PUBLIC_SRC) not in sys.path: sys.path.insert(0, str(PUBLIC_SRC))
    if str(VJEPA_SOURCE) not in sys.path: sys.path.insert(0, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips
    clips._DECORD_USABLE = False
    from ee6008.models import encode_clip, load_vjepa2_1_base
    return clips.preprocess_vjepa, load_vjepa2_1_base, encode_clip


def load_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    offset = 0
    for name in ("V3_MAIN_INPUTS.npz", "V3_SUPPLEMENTARY_1S_INPUTS.npz"):
        with np.load(EXP / "data" / name, allow_pickle=False) as archive:
            n = len(archive["episode_index"])
            rows.extend({"cache_index": offset + i, "episode_index": int(archive["episode_index"][i]), "anchor_id": str(archive["anchor_id"][i]), "anchor_frame": int(archive["anchor_frame"][i]), "video_path": str(archive["video_path"][i]), "split_code": int(archive["split_code"][i]), "population": "primary" if name.startswith("V3_MAIN") else "supplementary_1s"} for i in range(n))
            offset += n
    return rows, rows[:]


def read_frames(video_path: Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"cannot open {video_path}")
    wanted_set = set(int(x) for x in wanted.tolist()); output = {}
    max_frame = int(wanted[-1])
    try:
        frame_index = 0
        while frame_index <= max_frame:
            ok, frame = cap.read()
            if not ok: raise RuntimeError(f"decode failed at frame {frame_index}: {video_path}")
            if frame_index in wanted_set: output[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally: cap.release()
    if len(output) != len(wanted_set): raise RuntimeError(f"missing wanted frames for {video_path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--stage0-rows", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("feature extraction requires approved CUDA")
    torch.cuda.set_device(0); device = torch.device("cuda:0")
    preprocess, load_base, _encode_clip = import_stack()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=PRETRAINED, device=device, num_frames=32, resolution=384)
    encoder.eval()
    rows, _ = load_inputs()
    selected = rows[: args.stage0_rows] if args.stage0 else rows[args.start_row:args.end_row]
    if args.stage0:
        output = EXP / "cache/stage0_features.npz"
    else:
        output = EXP / "cache/V3_TEMPORAL_FEATURES.npy"
        if output.exists() and not args.resume: raise FileExistsError(output)
        if not output.exists():
            mm = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(rows), 2, TOKEN_SHAPE[0], TOKEN_SHAPE[1])); mm[:] = np.nan; mm.flush(); del mm
    # Group by video so each raw file is decoded once for all anchors in it.
    groups: dict[str, list[dict[str, object]]] = {}
    for row in selected: groups.setdefault(str(row["video_path"]), []).append(row)
    stage_values = []
    mm = None if args.stage0 else np.lib.format.open_memmap(output, mode="r+")
    group_total = len(groups)
    for group_number, (relative_path, group) in enumerate(groups.items(), start=1):
        group_started = time.perf_counter()
        video_path = (RAW_ROOT / relative_path).resolve()
        if not video_path.is_relative_to(RAW_ROOT.resolve()) or not video_path.is_file(): raise FileNotFoundError(video_path)
        wanted = sorted({int(a) for row in group for a in np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64).tolist()})
        frame_map = read_frames(video_path, np.asarray(wanted, dtype=np.int64))
        clips_d = []
        metadata = []
        for row in group:
            indices = np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64)
            frames = np.stack([frame_map[int(index)] for index in indices], axis=0)
            static_frames = np.repeat(frames[-1:, ...], 32, axis=0)
            clips_d.append((preprocess(frames, resolution=384), preprocess(static_frames, resolution=384)))
            metadata.append(row)
        for start in range(0, len(clips_d), PYTHON_BATCH):
            batch = clips_d[start:start + PYTHON_BATCH]; batch_meta = metadata[start:start + PYTHON_BATCH]
            # The accepted public bridge API uses one C,T,H,W clip.  Keep the
            # single-clip path here: on this host the direct B=4 forward
            # selected a much slower attention path, while the one-clip path
            # has the measured/accepted bridge behavior.
            d_token_list = []; s_token_list = []
            for d_item, s_item in batch:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    d_token, _ = _encode_clip(encoder, d_item.to(device))
                    s_token, _ = _encode_clip(encoder, s_item.to(device))
                d_token_list.append(d_token); s_token_list.append(s_token)
            d_tokens = torch.stack(d_token_list, dim=0)
            s_tokens = torch.stack(s_token_list, dim=0)
            def spatial_mean(tokens: torch.Tensor) -> np.ndarray:
                values = tokens.float().reshape(tokens.shape[0], 16, 576, 768).mean(dim=2).cpu().numpy().astype(np.float32)
                return values
            d_values = spatial_mean(d_tokens); s_values = spatial_mean(s_tokens)
            for j, row in enumerate(batch_meta):
                if args.stage0: stage_values.append((int(row["cache_index"]), d_values[j], s_values[j]))
                else: mm[int(row["cache_index"]), 0] = d_values[j]; mm[int(row["cache_index"]), 1] = s_values[j]
        print(json.dumps({"group": group_number, "groups_total": group_total, "rows_done": sum(len(value) for value in groups.values()) if args.stage0 else group_number, "anchors_in_group": len(group), "elapsed_seconds": time.perf_counter() - group_started}, sort_keys=True), flush=True)
    if mm is not None: mm.flush(); del mm
    if args.stage0:
        ids = np.asarray([item[0] for item in stage_values], dtype=np.int64); d = np.stack([item[1] for item in stage_values]); s = np.stack([item[2] for item in stage_values]); np.savez_compressed(output, cache_index=ids, dynamic_tokens=d, static_tokens=s)
    manifest = {"status": "PASS", "stage0": args.stage0, "rows_processed": len(selected), "total_rows": len(rows), "feature_output": str(output), "token_shape": list(TOKEN_SHAPE), "raw_encoder_checkpoint": str(PRETRAINED), "raw_encoder_checkpoint_sha256": digest(PRETRAINED), "encoder_info": info, "preprocess": "OpenCV deterministic frame index floor(linspace(anchor-119,anchor,32)), RGB, resize-shortest-side/center-crop 384, ImageNet normalization", "decoder": "OpenCV", "final_test_access": 0, "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown")}
    manifest_name = "cache/STAGE0_FEATURE_MANIFEST.json" if args.stage0 else ("cache/CACHE_RESUME_COMPLETE.json" if args.resume else "cache/CACHE_EXTRACTION_COMPLETE.json")
    (EXP / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage0": args.stage0, "rows_processed": len(selected), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
