#!/usr/bin/env python3
"""CPU-only identity, storage, frame, and preprocessing audit for two frozen train IDs."""
from __future__ import annotations
import os
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
V1_RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
V9 = PROJECT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
D3_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
D3_SOURCE = D3_ROOT / "code/d3_extract_grid.py"
CACHE_MANIFEST = D3_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
MAIN_INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
BRIDGE = V1_RUN / "replay/core_b/core_b_bridge_manifest.csv"
OUTPUT = OUT / "core_b/CORE_B_TRAIN_BRIDGE_CPU_AUDIT.json"
EXPECTED_CACHE_SHA = "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee"
FIXED_IDS = ("e001726_a01_f000149", "e001769_a00_f000119")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return sha_bytes(arr.tobytes())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if sha_file(CACHE_MANIFEST) != EXPECTED_CACHE_SHA:
        raise RuntimeError("accepted Q0 manifest hash changed")
    bridge_rows = list(csv.DictReader(BRIDGE.open(encoding="utf-8-sig", newline="")))
    selected = [r for r in bridge_rows if r["dataset_role"] == "train" and r["anchor_id"] in FIXED_IDS]
    if tuple(r["anchor_id"] for r in selected) != FIXED_IDS:
        by_id = {r["anchor_id"]: r for r in selected}
        if set(by_id) != set(FIXED_IDS):
            raise RuntimeError("the two preselected train bridge IDs are missing")
        selected = [by_id[anchor_id] for anchor_id in FIXED_IDS]

    d3 = load_module(D3_SOURCE, "v2_historical_d3_source")
    preprocess, _ = d3.import_stack()
    clips_module = sys.modules[preprocess.__module__]
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "PASS" or int(cache_manifest.get("final_test_access", -1)) != 0:
        raise RuntimeError("accepted cache manifest status/access marker is invalid")

    # Import the exact accepted Q0 row reader without touching any GPU API.
    q0_module = load_module(V9 / "code/prepare_v2_data.py", "v2_accepted_prepare_v2_data")
    q0_reader = q0_module.Q0ShardedGrid(cache_manifest)
    metadata_keys = ("anchor_id", "episode_index", "original_task_index", "split_code", "anchor_frame", "video_path")
    rows = []
    for bridge_row in selected:
        anchor_id = bridge_row["anchor_id"]
        with np.load(MAIN_INPUTS, allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64", copy=False)
            hit = np.flatnonzero(ids == anchor_id)
            if len(hit) != 1:
                raise RuntimeError(f"V3 main-input metadata has {len(hit)} rows for {anchor_id}")
            ix = int(hit[0])
            values = {key: archive[key][ix].item() for key in metadata_keys}
        expected = {
            "episode_index": int(bridge_row["episode_id"]),
            "original_task_index": int(bridge_row["task_id"]),
            "anchor_frame": int(bridge_row["anchor_frame"]),
            "video_path": bridge_row["video_path"],
        }
        for key, want in expected.items():
            got = values[key]
            if isinstance(got, bytes):
                got = got.decode("utf-8")
            if got != want:
                raise RuntimeError(f"V3/main-input vs bridge mismatch for {anchor_id} {key}: {got!r} != {want!r}")
        cache_index = int(bridge_row["cache_index"])
        if cache_index != ix:
            raise RuntimeError(f"V3 row index and frozen cache_index differ for {anchor_id}: {ix} != {cache_index}")
        shard = next((s for s in cache_manifest["shards"] if int(s["start_row"]) <= cache_index < int(s["end_row"])), None)
        if shard is None:
            raise RuntimeError(f"no Q0 shard covers {anchor_id} at {cache_index}")
        shard_start, shard_end = int(shard["start_row"]), int(shard["end_row"])
        shard_array_path, shard_meta_path = Path(shard["array"]), Path(shard["meta"])
        shard_array = np.load(shard_array_path, mmap_mode="r", allow_pickle=False)
        shard_meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))
        local = cache_index - shard_start
        if tuple(shard_array.shape) != (shard_end - shard_start, 2, 16, 4, 768) or shard_array.dtype != np.float32:
            raise RuntimeError(f"Q0 shard storage contract mismatch: {shard_array.shape}/{shard_array.dtype}")
        if str(shard_meta["anchor_ids"][local]) != anchor_id or int(shard_meta["episode_index"][local]) != int(bridge_row["episode_id"]):
            raise RuntimeError(f"Q0 shard metadata row identity mismatch for {anchor_id}")
        q0_row = q0_reader.row(cache_index, anchor_id, str(bridge_row["episode_id"]))
        direct_row = np.asarray(shard_array[local])
        if q0_row.dtype != np.float32 or tuple(q0_row.shape) != (2, 16, 4, 768) or not np.array_equal(q0_row, direct_row):
            raise RuntimeError(f"Q0 accepted reader changed/altered cache row {anchor_id}")

        frame_indices = np.floor(np.linspace(int(bridge_row["anchor_frame"]) - 119, int(bridge_row["anchor_frame"]), 32)).astype(np.int64)
        raw_path = (d3.RAW_ROOT / str(bridge_row["video_path"])).resolve()
        if not raw_path.is_relative_to(d3.RAW_ROOT.resolve()) or not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open train source clip {raw_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_map = {}
        wanted = set(map(int, frame_indices.tolist()))
        try:
            for frame_index in range(int(frame_indices[-1]) + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"decode failed at {frame_index}: {raw_path}")
                if frame_index in wanted:
                    frame_map[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        if len(frame_map) != len(wanted):
            raise RuntimeError(f"decoded {len(frame_map)}/{len(wanted)} frames for {anchor_id}")
        frames = np.stack([frame_map[int(i)] for i in frame_indices], axis=0)
        static = np.repeat(frames[-1:, ...], 32, axis=0)
        preprocessed = []
        for mode, raw in (("dynamic", frames), ("static", static)):
            tensor = preprocess(raw, resolution=384).detach().cpu().contiguous()
            tensor_array = tensor.numpy()
            preprocessed.append({
                "mode": mode,
                "shape": list(tensor_array.shape),
                "dtype": str(tensor_array.dtype),
                "sha256": hash_array(tensor_array),
                "finite": bool(np.isfinite(tensor_array).all()),
            })
        rows.append({
            "dataset_role": bridge_row["dataset_role"],
            "anchor_id": anchor_id,
            "canonical_id": bridge_row["canonical_id"],
            "source_key_sha256": bridge_row["source_key"],
            "episode_id": int(bridge_row["episode_id"]),
            "task_id": int(bridge_row["task_id"]),
            "v3_main_input_row": ix,
            "cache_index": cache_index,
            "split_code_raw": int(values["split_code"]),
            "anchor_frame": int(bridge_row["anchor_frame"]),
            "video_path": str(raw_path),
            "video_sha256": sha_file(raw_path),
            "video_fps": fps,
            "video_frame_count": frame_count,
            "frame_indices": frame_indices.tolist(),
            "frame_indices_sha256": hash_array(frame_indices.astype("<i8")),
            "unique_frame_count": len(wanted),
            "rgb_frames_sha256": hash_array(frames),
            "rgb_shape": list(frames.shape),
            "preprocessed": preprocessed,
            "preprocess_module_path": str(Path(clips_module.__file__).resolve()),
            "preprocess_module_sha256": sha_file(Path(clips_module.__file__).resolve()),
            "cache_shard_start": shard_start,
            "cache_shard_end": shard_end,
            "cache_shard_array_path": str(shard_array_path),
            "cache_shard_array_sha256": sha_file(shard_array_path),
            "cache_shard_meta_path": str(shard_meta_path),
            "cache_shard_meta_sha256": sha_file(shard_meta_path),
            "cache_shard_row_offset": local,
            "cache_shard_dtype": str(shard_array.dtype),
            "cache_shard_shape": list(shard_array.shape),
            "q0_reader_return_dtype": str(q0_row.dtype),
            "q0_reader_return_shape": list(q0_row.shape),
            "q0_reader_equals_direct_shard_slice": True,
            "q0_row_sha256": hash_array(q0_row),
            "shard_meta_preprocess_hash": shard_meta.get("preprocess_hash"),
            "shard_meta_grid_layout": shard_meta.get("grid_layout"),
            "shard_meta_source_backbone_hash": shard_meta.get("source_backbone_hash"),
            "shard_meta_binds_extractor_script_hash": any("extract" in str(k).lower() and "sha" in str(k).lower() for k in shard_meta),
        })
    report = {
        "schema_version": 1,
        "status": "PASS_CPU_METADATA_FRAME_STORAGE_PREPROCESS_AUDIT",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "exactly two frozen train bridge rows; no dev outcomes, final payload, targets, training, or CUDA access",
        "source_hashes": {
            "d3_extractor_script_sha256": sha_file(D3_SOURCE),
            "v3_main_inputs_sha256": sha_file(MAIN_INPUTS),
            "q0_manifest_sha256": sha_file(CACHE_MANIFEST),
            "v9_prepare_v2_data_sha256": sha_file(V9 / "code/prepare_v2_data.py"),
            "preprocess_module_path": str(Path(clips_module.__file__).resolve()),
            "preprocess_module_sha256": sha_file(Path(clips_module.__file__).resolve()),
        },
        "profile_known_from_source": {
            "d3_autocast": "torch.autocast(cuda, dtype=bfloat16)",
            "d3_encoder_batch_size": 1,
            "d3_module_eval": True,
            "d3_cache_output_dtype": "float32",
            "d3_grid_shape": [16, 4, 768],
            "d3_grid_pool": "raw [16,24,24,768] tokens cast to float32; four disjoint 12x12 area means",
            "d3_row_frame_schedule": "floor(linspace(anchor_frame-119, anchor_frame, 32))",
        },
        "profile_missing_from_historical_receipts": [
            "extractor script hash in cache-run receipt",
            "torch/CUDA runtime for cache generation",
            "TF32/matmul precision flags",
            "cudnn and SDPA/attention backend",
            "batch and tail shapes as run",
            "raw-frame/preprocessed-input/raw-token content hashes",
        ],
        "rows": rows,
        "all_identity_checks_pass": True,
        "all_q0_storage_checks_pass": True,
        "all_preprocessed_inputs_finite": all(p["finite"] for row in rows for p in row["preprocessed"]),
        "final_payload_read": False,
        "future_targets_materialized": False,
        "cuda_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "anchor_ids": [r["anchor_id"] for r in rows], "output": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
