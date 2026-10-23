"""Fresh frozen-encoder D/S spatial-grid extraction for D3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, atomic_write_json, sha256_file, utc_now


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
CHECKPOINT = PROJECT_ROOT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA_SOURCE = PROJECT_ROOT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT_ROOT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
GRID_SHAPE = (16, 4, 768)


def import_stack():
    sys.path.insert(0, str(PUBLIC_SRC))
    sys.path.insert(1, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips

    clips._DECORD_USABLE = False
    from ee6008.models import load_vjepa2_1_base

    return clips.preprocess_vjepa, load_vjepa2_1_base


def read_frames(video_path: Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    wanted_set = {int(index) for index in wanted.tolist()}
    frames: dict[int, np.ndarray] = {}
    try:
        for index in range(int(wanted[-1]) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"decode failed at frame {index}: {video_path}")
            if index in wanted_set:
                frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    if len(frames) != len(wanted_set):
        raise RuntimeError(f"missing decoded frames for {video_path}: {len(frames)}/{len(wanted_set)}")
    return frames


def grid_pool(tokens: torch.Tensor) -> tuple[np.ndarray, float]:
    if tuple(tokens.shape) != (16 * 24 * 24, 768):
        raise ValueError(f"raw encoder contract mismatch: {tuple(tokens.shape)}")
    grid = tokens.float().reshape(16, 24, 24, 768)
    bins = []
    for row_start, row_end in ((0, 12), (12, 24)):
        for col_start, col_end in ((0, 12), (12, 24)):
            bins.append(grid[:, row_start:row_end, col_start:col_end].mean(dim=(1, 2)))
    values = torch.stack(bins, dim=1).cpu().numpy().astype(np.float32)
    bridge = float(np.max(np.abs(values.mean(axis=1) - grid.mean(dim=(1, 2)).cpu().numpy())))
    return values, bridge


def load_rows() -> dict[str, np.ndarray]:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def shard_key(ids: np.ndarray, start: int, end: int) -> str:
    digest = hashlib.sha256()
    digest.update(ids.astype("U64").tobytes())
    digest.update(f"{start}:{end}:D3_GRID_V1:(16,4,768):float32".encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--stage0-rows", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D3 extraction requires approved CUDA")
    device = torch.device("cuda:0")
    preprocess, load_base = import_stack()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=CHECKPOINT, device=device, num_frames=32, resolution=384)
    encoder.eval()
    data = load_rows()
    total = len(data["episode_index"])
    start = 0 if args.stage0 else max(0, args.start_row)
    end = min(total, args.stage0_rows if args.stage0 else (args.end_row if args.end_row is not None else total))
    if start >= end:
        raise ValueError(f"empty extraction range {start}:{end}")
    shard_root = BENCH_ROOT / "d3/cache/shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    rows_done = 0
    bridge_max = 0.0
    shard_count = 0
    for chunk_start in range(start, end, args.shard_rows):
        chunk_end = min(end, chunk_start + args.shard_rows)
        out = shard_root / f"grid_{chunk_start:06d}_{chunk_end:06d}.npy"
        meta_path = out.with_suffix(".json")
        if args.resume and out.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bridge_max = max(bridge_max, float(meta.get("bridge_max_abs", 0.0)))
            rows_done += chunk_end - chunk_start
            shard_count += 1
            continue
        values = np.empty((chunk_end - chunk_start, 2, *GRID_SHAPE), dtype=np.float32)
        chunk_rows = [{key: data[key][index] for key in data} for index in range(chunk_start, chunk_end)]
        groups: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {}
        for local, row in enumerate(chunk_rows):
            groups.setdefault(str(row["video_path"]), []).append((local, row))
        started = time.perf_counter()
        for relative_path, group in groups.items():
            video_path = (RAW_ROOT / relative_path).resolve()
            if not video_path.is_relative_to(RAW_ROOT.resolve()) or not video_path.is_file():
                raise FileNotFoundError(video_path)
            wanted = sorted({int(frame) for _, row in group for frame in np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64).tolist()})
            frame_map = read_frames(video_path, np.asarray(wanted, dtype=np.int64))
            for local, row in group:
                indices = np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64)
                frames = np.stack([frame_map[int(index)] for index in indices], axis=0)
                static = np.repeat(frames[-1:, ...], 32, axis=0)
                for mode, raw in enumerate((frames, static)):
                    clip = preprocess(raw, resolution=384).to(device)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tokens = encoder(clip.unsqueeze(0))[0]
                    pooled, bridge = grid_pool(tokens)
                    values[local, mode] = pooled
                    bridge_max = max(bridge_max, bridge)
        part = out.with_name(f".{out.name}.{os.getpid()}.part")
        with part.open("wb") as handle:
            np.save(handle, values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, out)
        meta = {"status": "PASS", "start_row": chunk_start, "end_row": chunk_end, "row_count": chunk_end - chunk_start, "anchor_ids": [str(value) for value in data["anchor_id"][chunk_start:chunk_end]], "episode_index": [int(value) for value in data["episode_index"][chunk_start:chunk_end]], "shape": list(values.shape), "dtype": "float32", "source_backbone_hash": sha256_file(CHECKPOINT), "preprocess_hash": "V3_OPEN_CV_PREFIX_4S_8HZ_RGB_RESIZE_CENTER_CROP_IMAGENET", "grid_layout": "T16_H24_W24_to_T16_2x2_area_mean", "cache_key": shard_key(data["anchor_id"][chunk_start:chunk_end], chunk_start, chunk_end), "bridge_max_abs": bridge_max, "elapsed_seconds": time.perf_counter() - started, "final_test_access": 0}
        atomic_write_json(meta_path, meta)
        rows_done += chunk_end - chunk_start
        shard_count += 1
        print(json.dumps({"status": "shard_complete", "start_row": chunk_start, "end_row": chunk_end, "rows_done": rows_done, "rows_total": end - start, "bridge_max_abs": bridge_max}, sort_keys=True), flush=True)
    manifest = {"status": "PASS", "stage0": bool(args.stage0), "start_row": start, "end_row": end, "rows_processed": rows_done, "total_rows": total, "shard_count": shard_count, "grid_shape_per_mode": list(GRID_SHAPE), "cache_shape_logical": [total, 2, *GRID_SHAPE], "source_backbone_hash": sha256_file(CHECKPOINT), "encoder_info": info, "preprocess_hash": "V3_OPEN_CV_PREFIX_4S_8HZ_RGB_RESIZE_CENTER_CROP_IMAGENET", "grid_layout": "T16 x 2x2 disjoint spatial bins; bins are floor boundaries and area-weighted reconstruction", "bridge_max_abs": bridge_max, "final_test_access": 0, "generated_utc": utc_now()}
    atomic_write_json(BENCH_ROOT / ("d3/cache/D3_STAGE0_MANIFEST.json" if args.stage0 else "d3/cache/D3_EXTRACTION_RUN_MANIFEST.json"), manifest)
    print(json.dumps({"status": "PASS", "rows_processed": rows_done, "stage0": bool(args.stage0), "bridge_max_abs": bridge_max}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
