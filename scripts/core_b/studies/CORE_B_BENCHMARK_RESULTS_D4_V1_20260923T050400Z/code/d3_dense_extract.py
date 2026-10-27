"""Inference-only D3 D/S spatial-grid extraction for the fixed D2 dense panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, V3_ROOT

sys.path.insert(0, str(BENCH_ROOT / "code"))
from d3_extract_grid import CHECKPOINT, RAW_ROOT, VJEPA_SOURCE, grid_pool, import_stack, read_frames  # noqa: E402

DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
DENSE_ROOT = NEXT_ROOT / "d3_extension"
GRID_SHAPE = (16, 4, 768)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def load_rows() -> dict[str, np.ndarray]:
    with np.load(DENSE_INPUTS, allow_pickle=False) as archive:
        keys = ("dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp", "video_path")
        return {key: archive[key].copy() for key in keys}


def selected_rows(data: dict[str, np.ndarray], stage0: bool, count: int) -> np.ndarray:
    if not stage0:
        return np.arange(len(data["dense_id"]), dtype=np.int64)
    by_episode = {}
    for idx, episode in enumerate(data["episode_index"].astype(int)):
        by_episode.setdefault(int(episode), idx)
    ordered = sorted(by_episode.values(), key=lambda idx: hashlib.sha256(f"{data['dense_id'][idx]}|D3_DENSE_STAGE0_V1".encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError(f"stage0 requested {count} distinct episodes, found {len(ordered)}")
    return np.asarray(ordered[:count], dtype=np.int64)


def shard_signature(ids: np.ndarray, start: int, end: int) -> str:
    digest = hashlib.sha256()
    digest.update(ids.astype("U96").tobytes())
    digest.update(f"{start}:{end}:D3_DENSE_GRID_V1:(16,4,768):float32".encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--stage0-rows", type=int, default=16)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D3 dense extraction requires an approved CUDA watcher child")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    preprocess, load_base = import_stack()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=CHECKPOINT, device=device, num_frames=32, resolution=384)
    encoder.eval()
    data = load_rows()
    total = len(data["dense_id"])
    indices = selected_rows(data, args.stage0, args.stage0_rows)
    if args.stage0:
        start, end = 0, len(indices)
        shard_root = DENSE_ROOT / "stage0/shards"
        manifest_path = DENSE_ROOT / "stage0/D3_DENSE_STAGE0_MANIFEST.json"
    else:
        start = max(0, args.start_row)
        end = min(total, args.end_row if args.end_row is not None else total)
        indices = np.arange(start, end, dtype=np.int64)
        shard_root = DENSE_ROOT / "cache/shards"
        manifest_path = DENSE_ROOT / "cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json"
    if start >= end:
        raise ValueError(f"empty D3 dense extraction range: {start}:{end}")
    shard_root.mkdir(parents=True, exist_ok=True)
    metas = []
    bridge_max = 0.0
    processed = 0
    elapsed_sum = 0.0
    for chunk_start in range(start, end, args.shard_rows):
        chunk_end = min(end, chunk_start + args.shard_rows)
        row_indices = indices[chunk_start:chunk_end] if args.stage0 else np.arange(chunk_start, chunk_end, dtype=np.int64)
        ids = data["dense_id"][row_indices].astype("U96")
        out = shard_root / f"grid_{chunk_start:06d}_{chunk_end:06d}.npy"
        meta_path = out.with_suffix(".json")
        if args.resume and out.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "PASS" and meta.get("cache_signature") == shard_signature(ids, chunk_start, chunk_end) and meta.get("dense_ids") == ids.tolist() and meta.get("array_sha256") == sha256(out):
                metas.append({"start_row": chunk_start, "end_row": chunk_end, "array": str(out), "meta": str(meta_path), "bytes": out.stat().st_size, "sha256": sha256(out)})
                bridge_max = max(bridge_max, float(meta.get("bridge_max_abs", 0.0)))
                processed += len(row_indices)
                elapsed_sum += float(meta.get("elapsed_seconds", 0.0))
                continue
            raise ValueError(f"resume shard exists but lacks an accepted matching manifest: {out}")

        values = np.empty((len(row_indices), 2, *GRID_SHAPE), dtype=np.float32)
        by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for local, global_index in enumerate(row_indices):
            by_video[str(data["video_path"][global_index])].append((local, int(global_index)))
        started = time.perf_counter()
        for relative_video, local_rows in by_video.items():
            video_path = (RAW_ROOT / relative_video).resolve()
            if not video_path.is_relative_to(RAW_ROOT.resolve()) or not video_path.is_file():
                raise FileNotFoundError(f"dense video escaped/readiness check failed: {video_path}")
            wanted = sorted({int(index) for _, global_index in local_rows for index in np.floor(np.linspace(int(data["anchor_frame"][global_index]) - 119, int(data["anchor_frame"][global_index]), 32)).astype(np.int64).tolist()})
            frame_map = read_frames(video_path, np.asarray(wanted, dtype=np.int64))
            for local, global_index in local_rows:
                anchor = int(data["anchor_frame"][global_index])
                frame_indices = np.floor(np.linspace(anchor - 119, anchor, 32)).astype(np.int64)
                frames = np.stack([frame_map[int(index)] for index in frame_indices], axis=0)
                static = np.repeat(frames[-1:, ...], 32, axis=0)
                for mode, raw in enumerate((frames, static)):
                    clip = preprocess(raw, resolution=384).to(device)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        tokens = encoder(clip.unsqueeze(0))[0]
                    pooled, bridge = grid_pool(tokens)
                    values[local, mode] = pooled
                    bridge_max = max(bridge_max, bridge)
        elapsed = time.perf_counter() - started
        part = out.with_name(f".{out.name}.{os.getpid()}.part")
        with part.open("wb") as handle:
            np.save(handle, values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, out)
        meta = {"status": "PASS", "start_row": chunk_start, "end_row": chunk_end, "row_count": len(row_indices), "dense_ids": ids.tolist(), "episode_index": data["episode_index"][row_indices].astype(int).tolist(), "shape": list(values.shape), "dtype": "float32", "source_backbone_sha256": sha256(CHECKPOINT), "preprocess_hash": "V3_OPEN_CV_PREFIX_4S_8HZ_RGB_RESIZE_CENTER_CROP_IMAGENET", "grid_layout": "T16x2x2 spatial bins; 24x24 token grid area mean", "cache_signature": shard_signature(ids, chunk_start, chunk_end), "array_sha256": sha256(out), "bridge_max_abs": bridge_max, "elapsed_seconds": elapsed, "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024)), "final_test_access": 0}
        atomic_json(meta_path, meta)
        metas.append({"start_row": chunk_start, "end_row": chunk_end, "array": str(out), "meta": str(meta_path), "bytes": out.stat().st_size, "sha256": sha256(out)})
        bridge_max = max(bridge_max, float(meta["bridge_max_abs"]))
        processed += len(row_indices)
        elapsed_sum += elapsed
        print(json.dumps({"status": "shard_complete", "range": [chunk_start, chunk_end], "rows_done": processed, "rows_target": end - start, "elapsed_seconds": elapsed, "bridge_max_abs": bridge_max}, sort_keys=True), flush=True)

    manifest = {"status": "PASS", "stage0": bool(args.stage0), "start_row": start, "end_row": end, "rows_processed": processed, "total_dense_rows": total, "shard_count": len(metas), "grid_shape_per_mode": list(GRID_SHAPE), "source_backbone_sha256": sha256(CHECKPOINT), "encoder_info": info, "preprocess_hash": "V3_OPEN_CV_PREFIX_4S_8HZ_RGB_RESIZE_CENTER_CROP_IMAGENET", "bridge_max_abs": bridge_max, "child_elapsed_seconds_sum": elapsed_sum, "measured_peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "measured_peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024)), "shards": metas, "final_test_access": 0}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "stage0": bool(args.stage0), "rows_processed": processed, "elapsed_seconds_sum": elapsed_sum, "peak_allocated_mib": manifest["measured_peak_allocated_mib"], "peak_reserved_mib": manifest["measured_peak_reserved_mib"], "final_test_access": 0}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
