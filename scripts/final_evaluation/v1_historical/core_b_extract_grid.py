"""Extract frozen D3 [2,16,4,768] video grids for bridge or gated final use.

The implementation pins the accepted D3 extractor, encoder checkpoint, frame
selection and spatial pooling contract. Final mode verifies both parent
receipts before parsing its input list or opening any video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
PROTOCOL_ID = "EE6008_FINAL_TEST_V1_20261019T041642Z"
DATASET_REVISION = "6b6b599def75e3558b2736062e79bdbeec6cd224"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
CHECKPOINT = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
CHECKPOINT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
VJEPA_SOURCE = PROJECT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
HISTORICAL_EXTRACTOR = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/code/d3_extract_grid.py"
HISTORICAL_EXTRACTOR_SHA256 = "25a27199f3854564d0760050891c6817333ca56e7ec2c4f6a35189721cdbb127"
GRID_SHAPE = (16, 4, 768)
OUTPUT_SHAPE = (2, *GRID_SHAPE)
TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


@dataclass(frozen=True)
class GridRow:
    dataset_role: str
    canonical_id: str
    source_key: str
    literal_source_key: str
    episode_id: int
    task_id: int
    anchor_id: str
    anchor_frame: int
    cache_index: int
    video_path: str
    parquet_path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_receipts(
    *,
    freeze_receipt_path: Path,
    final_access_receipt_path: Path,
    input_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = Path(freeze_receipt_path).resolve()
    access_path = Path(final_access_receipt_path).resolve()
    if freeze_path != (OUT_ROOT / "protocol/FREEZE_RECEIPT.json").resolve():
        raise ValueError("final extraction requires the canonical FREEZE_RECEIPT.json")
    if access_path != (OUT_ROOT / "protocol/FINAL_ACCESS_OPENED.json").resolve():
        raise ValueError("final extraction requires the canonical FINAL_ACCESS_OPENED.json")
    freeze = _read_json(freeze_path)
    access = _read_json(access_path)
    freeze_sha = sha256_file(freeze_path)
    lock_path = OUT_ROOT / "protocol/FINAL_STUDY_LOCK.json"
    if access.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK":
        raise ValueError("parent final-access receipt is not open")
    if access.get("core_b_access_authorized") is not True:
        raise ValueError("parent access receipt does not authorize Core B final feature access")
    if not access.get("opened_utc") or not access.get("first_final_payload_access_utc") or not access.get("access_stage"):
        raise ValueError("parent final-access receipt lacks access timestamps/stage")
    if access.get("protocol_id") != PROTOCOL_ID or freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol ID mismatch in final receipts")
    if access.get("study_lock_sha256") != sha256_file(lock_path):
        raise ValueError("final-access receipt does not bind the current frozen study lock")
    if access.get("freeze_receipt_sha256") != freeze_sha:
        raise ValueError("final-access receipt does not bind the current freeze receipt")
    if freeze.get("study_lock_sha256") != sha256_file(lock_path):
        raise ValueError("freeze receipt does not bind the current study lock")
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS":
        raise ValueError("freeze receipt is not a completed pre-access freeze")
    input_path = Path(input_manifest_path).resolve()
    expected_input = (OUT_ROOT / "protocol/CORE_B_FINAL_VIDEO_MAP.csv").resolve()
    if input_path != expected_input:
        raise ValueError("final extraction accepts only the locked Core B video map")
    expected_input_sha = freeze.get("input_sha256", {}).get("core_b_final_video_map")
    if not expected_input_sha or sha256_file(input_path) != expected_input_sha:
        raise ValueError("Core B video map hash differs from the freeze receipt")
    return freeze, access


def read_rows(path: Path, *, mode: str) -> list[GridRow]:
    path = Path(path)
    expected_fields = (
        "dataset_role", "canonical_id", "source_key", "literal_source_key",
        "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index",
        "video_path", "parquet_path",
    )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"input manifest columns must be exactly {expected_fields}")
        raw = list(reader)
    if not raw:
        raise ValueError("feature extraction manifest is empty")
    rows: list[GridRow] = []
    for line, item in enumerate(raw, start=2):
        role = str(item["dataset_role"])
        if mode == "bridge" and role not in {"train", "dev"}:
            raise ValueError(f"bridge input includes a non-train/dev role at line {line}")
        if mode == "final" and role != "final_test":
            raise ValueError(f"final input includes a non-final role at line {line}")
        rows.append(GridRow(
            dataset_role=role,
            canonical_id=str(item["canonical_id"]),
            source_key=str(item["source_key"]),
            literal_source_key=str(item["literal_source_key"]),
            episode_id=int(item["episode_id"]),
            task_id=int(item["task_id"]),
            anchor_id=str(item["anchor_id"]),
            anchor_frame=int(item["anchor_frame"]),
            cache_index=int(item["cache_index"]),
            video_path=str(item["video_path"]),
            parquet_path=str(item["parquet_path"]),
        ))
    ids = [row.canonical_id for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("extraction manifest contains empty/duplicate canonical IDs")
    if len({row.cache_index for row in rows}) != len(rows):
        raise ValueError("extraction manifest contains duplicate cache indices")
    return rows


def _raw_video(relative_path: str) -> Path:
    root = RAW_ROOT.resolve()
    path = (root / relative_path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"video path escapes the fixed raw data root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def wanted_frame_indices(anchor_frame: int) -> np.ndarray:
    if anchor_frame < 119:
        raise ValueError(f"anchor frame cannot provide 120-row prefix: {anchor_frame}")
    return np.floor(np.linspace(anchor_frame - 119, anchor_frame, 32)).astype(np.int64)


def read_frames_through(video_path: Path, wanted: np.ndarray, max_frame: int | None = None) -> dict[int, np.ndarray]:
    """Decode only through the largest selected anchor frame, never beyond it."""
    stop = int(wanted[-1]) if max_frame is None else int(max_frame)
    if stop < int(wanted[-1]):
        raise ValueError("max_frame is earlier than a requested frame")
    wanted_set = {int(value) for value in wanted.tolist()}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    try:
        for frame_index in range(stop + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"decode failed before anchor frame {frame_index}: {video_path}")
            if frame_index in wanted_set:
                frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    if set(frames) != wanted_set:
        raise RuntimeError(f"missing requested frame(s): {video_path}")
    return frames


def grid_pool(tokens: torch.Tensor) -> tuple[np.ndarray, float]:
    if tuple(tokens.shape) != (16 * 24 * 24, 768):
        raise ValueError(f"raw encoder token shape mismatch: {tuple(tokens.shape)}")
    grid = tokens.float().reshape(16, 24, 24, 768)
    bins = []
    for row_start, row_end in ((0, 12), (12, 24)):
        for col_start, col_end in ((0, 12), (12, 24)):
            bins.append(grid[:, row_start:row_end, col_start:col_end].mean(dim=(1, 2)))
    values = torch.stack(bins, dim=1).cpu().numpy().astype(np.float32)
    bridge = float(np.max(np.abs(values.mean(axis=1) - grid.mean(dim=(1, 2)).cpu().numpy())))
    return values, bridge


def extract_grid_rows(
    rows: Sequence[GridRow],
    *,
    preprocess: Callable[[np.ndarray, int], torch.Tensor],
    encoder: torch.nn.Module,
    device: torch.device,
    read_fn: Callable[[Path, np.ndarray, int | None], dict[int, np.ndarray]] = read_frames_through,
) -> tuple[np.ndarray, dict[str, int | float]]:
    if not rows:
        raise ValueError("cannot extract an empty row list")
    result = np.empty((len(rows), *OUTPUT_SHAPE), dtype=np.float32)
    by_video: dict[str, list[tuple[int, GridRow]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_video[row.video_path].append((index, row))
    rows_done = 0
    decoded_video_frames = 0
    selected_unique_frames = 0
    bridge_max = 0.0
    for video_rel, group in by_video.items():
        video = _raw_video(video_rel)
        per_row = {index: wanted_frame_indices(row.anchor_frame) for index, row in group}
        wanted = np.asarray(sorted({int(x) for values in per_row.values() for x in values}), dtype=np.int64)
        max_anchor = max(row.anchor_frame for _, row in group)
        frame_map = read_fn(video, wanted, max_anchor)
        decoded_video_frames += max_anchor + 1
        selected_unique_frames += len(frame_map)
        for index, row in group:
            indices = per_row[index]
            clip_rgb = np.stack([frame_map[int(frame)] for frame in indices], axis=0)
            clip_static = np.repeat(clip_rgb[-1:, ...], 32, axis=0)
            for mode_index, frames in enumerate((clip_rgb, clip_static)):
                clip = preprocess(frames, 384).to(device)
                if tuple(clip.shape) != (3, 32, 384, 384):
                    raise ValueError(f"V-JEPA preprocess shape mismatch: {tuple(clip.shape)}")
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    tokens = encoder(clip.unsqueeze(0))[0]
                pooled, bridge_error = grid_pool(tokens)
                if tuple(pooled.shape) != GRID_SHAPE or not np.isfinite(pooled).all():
                    raise FloatingPointError("feature grid is nonfinite or has the wrong shape")
                result[index, mode_index] = pooled
                bridge_max = max(bridge_max, bridge_error)
            rows_done += 1
    if rows_done != len(rows) or not np.isfinite(result).all():
        raise FloatingPointError("feature grid extraction incomplete/nonfinite")
    return result, {
        "rows": rows_done,
        "video_count": len(by_video),
        "decoded_video_frames": decoded_video_frames,
        "selected_unique_frames": selected_unique_frames,
        "encoder_input_frames": rows_done * 2 * 32,
        "bridge_max_abs": bridge_max,
    }


def _load_encoder(device: torch.device) -> tuple[torch.nn.Module, Callable[[np.ndarray, int], torch.Tensor], dict[str, Any]]:
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("V-JEPA checkpoint hash differs from the locked source")
    if sha256_file(HISTORICAL_EXTRACTOR) != HISTORICAL_EXTRACTOR_SHA256:
        raise ValueError("accepted D3 extractor source hash changed")
    sys.path.insert(0, str(PUBLIC_SRC))
    sys.path.insert(1, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips
    clips._DECORD_USABLE = False
    from ee6008.data.clips import preprocess_vjepa
    from ee6008.models import load_vjepa2_1_base

    encoder, info = load_vjepa2_1_base(
        source_root=VJEPA_SOURCE,
        checkpoint=CHECKPOINT,
        device=device,
        num_frames=32,
        resolution=384,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "backbone_sha256": CHECKPOINT_SHA256,
        "historical_extractor_sha256": HISTORICAL_EXTRACTOR_SHA256,
        "encoder_info": info,
        "preprocess": "OpenCV BGR-to-RGB; accepted V-JEPA 2.1 preprocess_vjepa; 32 frames; 384 crop; ImageNet normalization",
        "frame_sampling": "floor(linspace(anchor_frame-119, anchor_frame, 32))",
        "grid_pooling": "T16 x four disjoint H24xW24 quadrants, each 12x12 area mean",
        "feature_shape_per_condition": list(GRID_SHAPE),
    }
    return encoder, preprocess_vjepa, metadata


def run(
    *,
    mode: str,
    input_manifest: Path,
    output_array: Path,
    output_manifest: Path,
    device_name: str,
    freeze_receipt: Path | None = None,
    final_access_receipt: Path | None = None,
    stage0: bool = False,
) -> dict[str, Any]:
    if mode not in {"bridge", "final"}:
        raise ValueError("mode must be bridge or final")
    if stage0 and mode != "bridge":
        raise ValueError("stage0 smoke mode is limited to train/dev bridge rows")
    if device_name != "cuda":
        raise ValueError("V-JEPA grid extraction is authorized only through CUDA watcher")
    input_manifest = Path(input_manifest).resolve()
    output_array = Path(output_array).resolve()
    output_manifest = Path(output_manifest).resolve()
    for value in (output_array, output_manifest):
        if not value.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("grid arrays and manifests must be stored in RUN_ROOT")
        if value.exists():
            raise FileExistsError(f"refusing to overwrite extraction output: {value}")
    parent_receipts: dict[str, Any] = {}
    if mode == "final":
        if freeze_receipt is None or final_access_receipt is None:
            raise ValueError("final grid extraction requires both parent receipts")
        freeze, access = _verify_receipts(
            freeze_receipt_path=freeze_receipt,
            final_access_receipt_path=final_access_receipt,
            input_manifest_path=input_manifest,
        )
        parent_receipts = {
            "freeze_receipt_sha256": sha256_file(freeze_receipt),
            "final_access_opened_sha256": sha256_file(final_access_receipt),
            "study_lock_sha256": freeze["study_lock_sha256"],
            "first_final_payload_access_utc": access.get("first_final_payload_access_utc"),
        }
    elif not input_manifest.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("bridge inputs must remain under RUN_ROOT")

    rows = read_rows(input_manifest, mode=mode)
    if mode == "bridge":
        if stage0:
            if len(rows) != 1 or rows[0].dataset_role not in {"train", "dev"}:
                raise ValueError("Core B stage0 requires one preselected train/dev bridge row")
        else:
            if len(rows) != 64 or {row.dataset_role for row in rows} != {"train", "dev"}:
                raise ValueError("registered Core B bridge must have 32 train and 32 dev rows")
            if sum(row.dataset_role == "train" for row in rows) != 32 or sum(row.dataset_role == "dev" for row in rows) != 32:
                raise ValueError("registered Core B bridge split support is not 32/32")
    else:
        if len(rows) != 162:
            raise ValueError(f"final Core B feature input count must match frozen 162 anchors, got {len(rows)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the watcher-authorized child")
    device = torch.device("cuda:0")
    encoder, preprocess, encoder_metadata = _load_encoder(device)
    started = time.perf_counter()
    features, decode_meta = extract_grid_rows(rows, preprocess=preprocess, encoder=encoder, device=device)
    output_array.parent.mkdir(parents=True, exist_ok=True)
    temporary_array = output_array.with_name(f".{output_array.name}.{os.getpid()}.part.npy")
    with temporary_array.open("wb") as stream:
        np.save(stream, features)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_array, output_array)
    row_meta = output_array.with_suffix(".rows.csv")
    meta_rows = [{
        "row_index": i,
        "dataset_role": row.dataset_role,
        "canonical_id": row.canonical_id,
        "source_key": row.source_key,
        "literal_source_key": row.literal_source_key,
        "episode_id": row.episode_id,
        "task_id": row.task_id,
        "anchor_id": row.anchor_id,
        "anchor_frame": row.anchor_frame,
        "cache_index": row.cache_index,
        "video_path": row.video_path,
        "parquet_path": row.parquet_path,
    } for i, row in enumerate(rows)]
    row_meta.parent.mkdir(parents=True, exist_ok=True)
    temporary_meta = row_meta.with_name(f".{row_meta.name}.{os.getpid()}.part")
    with temporary_meta.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(meta_rows[0]), extrasaction="raise")
        writer.writeheader(); writer.writerows(meta_rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary_meta, row_meta)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_role": mode,
        "stage0_smoke": bool(stage0),
        "final_test_access": 1 if mode == "final" else 0,
        **parent_receipts,
        **encoder_metadata,
        "source_manifest_path": str(input_manifest),
        "source_manifest_sha256": sha256_file(input_manifest),
        "ordered_canonical_ids_sha256": hashlib.sha256("\n".join(row.canonical_id for row in rows).encode()).hexdigest(),
        "shape": list(features.shape),
        "dtype": "float32",
        "array_path": str(output_array),
        "array_sha256": sha256_file(output_array),
        "row_metadata_path": str(row_meta),
        "row_metadata_sha256": sha256_file(row_meta),
        "shards": [{"array": str(output_array), "meta": str(row_meta)}],
        "decode": decode_meta,
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "logical_device": str(device),
        "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
        "elapsed_seconds": time.perf_counter() - started,
        "generated_utc": utc_now(),
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = output_manifest.with_name(f".{output_manifest.name}.{os.getpid()}.part")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, output_manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bridge", "final"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-array", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--freeze-receipt", type=Path)
    parser.add_argument("--final-access-receipt", type=Path)
    parser.add_argument("--stage0", action="store_true")
    args = parser.parse_args()
    result = run(
        mode=args.mode,
        input_manifest=args.input_manifest,
        output_array=args.output_array,
        output_manifest=args.output_manifest,
        device_name=args.device,
        freeze_receipt=args.freeze_receipt,
        final_access_receipt=args.final_access_receipt,
        stage0=args.stage0,
    )
    print(json.dumps({"status": result["status"], "rows": result["shape"][0], "array_path": result["array_path"], "array_sha256": result["array_sha256"], "manifest_sha256": sha256_file(args.output_manifest), "final_test_access": result["final_test_access"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
