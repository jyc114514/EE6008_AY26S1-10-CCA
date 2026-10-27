#!/usr/bin/env python3
"""Target-blind CPU inference for exact D3-EQ old/corrected checkpoints on D4's fixed panel."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ_ROOT = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
BENCH_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
D4_ROOT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
DENSE_ROOT = TASK_ROOT / "dense_extension"
LOCK_PATH = DENSE_ROOT / "DENSE_EXTENSION_PROTOCOL_LOCKED.json"
INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
PANEL = BENCH_ROOT / "d2/D2_DENSE_PANEL.csv"
CACHE_ROOT = D4_ROOT / "d3_extension/cache"
CACHE_MANIFEST = CACHE_ROOT / "D3_DENSE_EXTRACTION_RUN_MANIFEST.json"
CACHE_ACCEPTANCE = D4_ROOT / "d3_extension/qa/D3_DENSE_CACHE_ACCEPTANCE.json"
PAIR_PREFLIGHT = D4_ROOT / "d3_extension/qa/D3_D4_DENSE_TARGET_JOIN_PREFLIGHT.json"
VIDEO_NORMALIZER = BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"
TARGET_NORMALIZER = V3_ROOT / "data/V3_NORMALIZERS.json"
OLD_PROTOCOL = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
STUDY_PROTOCOL = TASK_ROOT / "PROTOCOL_LOCKED.json"
MATRIX = TASK_ROOT / "protocol/MATRIX_12.csv"
FORMAL_ACCEPTANCE = TASK_ROOT / "FORMAL_TRAINING_ACCEPTANCE.json"
PRIMARY_PREDICTION_ACCEPTANCE = TASK_ROOT / "PREDICTION_ACCEPTANCE.json"
MODEL_SOURCE = TASK_ROOT / "code/d3_model.py"
sys.path.insert(0, str(TASK_ROOT / "code"))
from d3_model import D3TimeSpacePredictor  # noqa: E402

TARGET_BLIND_KEYS = (
    "dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp",
    "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native",
)
SEEDS = (20261001, 20261002, 20261003)
CONFIGS = ("TS11-D", "STATE-CTRL")
HORIZONS = (30, 60)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def read_matrix() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(c, s, h) for c in CONFIGS for s in SEEDS for h in HORIZONS}
    found = {(r["config"], int(r["seed"]), int(r["horizon_steps"])) for r in rows}
    if len(rows) != 12 or found != expected or len({r["run_id"] for r in rows}) != 12:
        raise AssertionError("locked corrected-scheduler matrix is not the exact 12-cell set")
    return rows


def resolve_cells(rows: list[dict[str, str]], verify: bool = True) -> list[dict[str, Any]]:
    old_protocol_sha = sha_file(OLD_PROTOCOL)
    study_protocol_sha = sha_file(STUDY_PROTOCOL)
    model_sha = sha_file(MODEL_SOURCE)
    cells = []
    for row in rows:
        config, horizon, seed = row["config"], int(row["horizon_steps"]), int(row["seed"])
        if row["old_protocol_sha256"] != old_protocol_sha or row["old_model_source_sha256"] != model_sha:
            raise AssertionError(f"matched old protocol/model identity differs: {row['run_id']}")
        old_run_id = row["matched_old_run_id"]
        old_checkpoint = EQ_ROOT / "d3/runs" / old_run_id / "checkpoint_last.pt"
        new_run = TASK_ROOT / "runs/formal" / row["run_id"]
        new_checkpoint = new_run / "checkpoint_last.pt"
        complete_path = new_run / "RUN_COMPLETE.json"
        if not old_checkpoint.is_file() or not new_checkpoint.is_file() or not complete_path.is_file():
            raise FileNotFoundError(f"matched checkpoint/marker missing for {row['run_id']}")
        old_hash = sha_file(old_checkpoint)
        new_hash = sha_file(new_checkpoint)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if old_hash != row["old_checkpoint_sha256"]:
            raise AssertionError(f"matched old checkpoint hash differs: {old_run_id}")
        if (complete.get("status") != "PASS" or int(complete.get("epochs", -1)) != 40
                or int(complete.get("optimizer_steps", -1)) != 11120
                or int(complete.get("scheduler_step_calls", -1)) != 11120
                or complete.get("checkpoint_last_sha256") != new_hash
                or complete.get("study_protocol_hash") != study_protocol_sha
                or complete.get("protocol_hash") != old_protocol_sha):
            raise AssertionError(f"corrected checkpoint acceptance/hash binding failed: {row['run_id']}")
        cells.append({
            "config": config, "seed": seed, "horizon_steps": horizon,
            "old_run_id": old_run_id, "old_checkpoint": str(old_checkpoint), "old_checkpoint_sha256": old_hash,
            "corrected_run_id": row["run_id"], "corrected_checkpoint": str(new_checkpoint),
            "corrected_checkpoint_sha256": new_hash,
        })
    return cells


def load_inputs() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    with np.load(INPUTS, allow_pickle=False) as archive:
        keys = list(archive.files)
        if any(any(term in key.lower() for term in ("target", "future")) for key in keys):
            raise AssertionError("dense input archive exposes a target/future field")
        missing = set(TARGET_BLIND_KEYS) - set(keys)
        if missing:
            raise AssertionError(f"dense target-blind input keys missing: {sorted(missing)}")
        data = {key: archive[key].copy() for key in TARGET_BLIND_KEYS}
    ids = data["dense_id"].astype("U96")
    if len(ids) != 3718 or len(set(ids.tolist())) != 3718:
        raise AssertionError("D4 dense panel must have 3718 unique IDs")
    with PANEL.open(newline="", encoding="utf-8") as stream:
        panel = list(csv.DictReader(stream))
    if len(panel) != 3718 or any(row.get("final_test_access") != "0" for row in panel):
        raise AssertionError("D2 dense panel row/final-test contract failed")
    panel_ids = [f"dense_e{int(row['episode_index']):06d}_a{int(row['anchor_frame']):06d}" for row in panel]
    if panel_ids != ids.tolist():
        raise AssertionError("dense input IDs do not exactly match panel order")
    for key in ("episode_index", "anchor_frame"):
        expected = np.asarray([int(row[key]) for row in panel], dtype=np.int64)
        if not np.array_equal(data[key], expected):
            raise AssertionError(f"dense input {key} differs from fixed panel")
    times = np.asarray([float(row["anchor_timestamp"]) for row in panel], dtype=np.float64)
    if not np.allclose(data["anchor_timestamp"], times, atol=1e-6, rtol=0):
        raise AssertionError("dense input timestamps differ from fixed panel")
    if len(set(data["episode_index"].tolist())) != 118:
        raise AssertionError("dense panel must map to 118 episodes")
    expected_shapes = {
        "history_standardized": (3718, 32, 71), "history_mask": (3718, 32),
        "history_relative_delta_standardized": (3718, 32), "anchor_command_native": (3718, 28),
        "original_task_index": (3718,), "anchor_timestamp": (3718,),
    }
    for key, shape in expected_shapes.items():
        if data[key].shape != shape:
            raise AssertionError(f"dense input shape mismatch for {key}: {data[key].shape}")
        if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
            raise AssertionError(f"dense input has non-finite values: {key}")
    return data, ids, {"archive_keys": keys, "rows": 3718, "episodes": 118, "ids_sha256": sha_array(ids)}


def verify_cache(ids: np.ndarray) -> dict[str, Any]:
    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    acceptance = json.loads(CACHE_ACCEPTANCE.read_text(encoding="utf-8"))
    manifest_sha = sha_file(CACHE_MANIFEST)
    if (manifest.get("status") != "PASS" or int(manifest.get("rows_processed", -1)) != 3718
            or acceptance.get("status") != "PASS" or int(acceptance.get("rows", -1)) != 3718
            or int(acceptance.get("unique_ids", -1)) != 3718
            or acceptance.get("manifest_sha256") != manifest_sha):
        raise AssertionError("accepted D4 dense cache manifest/hash/row contract failed")
    rows, cursor = [], 0
    cache_root = CACHE_ROOT.resolve()
    for shard in sorted(manifest["shards"], key=lambda item: int(item["start_row"])):
        start, end = int(shard["start_row"]), int(shard["end_row"])
        array_path = Path(shard["array"])
        meta_path = Path(shard["meta"])
        if start != cursor or end <= start or not array_path.resolve().is_relative_to(cache_root) or not meta_path.resolve().is_relative_to(cache_root):
            raise AssertionError("dense cache shard coverage/path boundary failed")
        if not array_path.is_file() or not meta_path.is_file() or array_path.stat().st_size != int(shard["bytes"]):
            raise AssertionError(f"dense cache shard missing/size mismatch: {array_path.name}")
        array_sha = sha_file(array_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if array_sha != shard["sha256"] or array_sha != meta.get("array_sha256"):
            raise AssertionError(f"dense cache shard hash mismatch: {array_path.name}")
        if meta.get("status") != "PASS" or meta.get("preprocess_hash") != manifest.get("preprocess_hash"):
            raise AssertionError(f"dense cache shard status/preprocess mismatch: {meta_path.name}")
        if meta.get("dense_ids") != ids[start:end].tolist():
            raise AssertionError(f"dense cache shard ID order mismatch: {meta_path.name}")
        if meta.get("source_backbone_sha256") != manifest.get("source_backbone_sha256"):
            raise AssertionError(f"dense cache backbone mismatch: {meta_path.name}")
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (end - start, 2, 16, 4, 768) or values.dtype != np.float32:
            raise AssertionError(f"dense cache feature shape/dtype mismatch: {array_path.name}")
        cursor = end
        rows.append({"start_row": start, "end_row": end, "array": str(array_path), "meta": str(meta_path), "sha256": array_sha})
    if cursor != 3718 or len(rows) != int(manifest.get("shard_count", -1)):
        raise AssertionError("dense cache shards do not cover the full fixed panel")
    return {"manifest_sha256": manifest_sha, "acceptance_sha256": sha_file(CACHE_ACCEPTANCE),
            "rows": cursor, "shards": rows, "source_backbone_sha256": manifest["source_backbone_sha256"],
            "preprocess_hash": manifest["preprocess_hash"]}


def preflight() -> dict[str, Any]:
    rows = read_matrix()
    cells = resolve_cells(rows)
    data, ids, panel = load_inputs()
    cache = verify_cache(ids)
    pair = json.loads(PAIR_PREFLIGHT.read_text(encoding="utf-8"))
    if (pair.get("status") != "PASS" or int(pair.get("dense_ids", -1)) != 3718
            or int(pair.get("source_groups", -1)) != 118 or int(pair.get("pairs_h30", -1)) != 3600
            or int(pair.get("pairs_h60", -1)) != 3600 or pair.get("final_test_access") != 0):
        raise AssertionError("D4 locked common-absolute-target pair contract failed")
    formal = json.loads(FORMAL_ACCEPTANCE.read_text(encoding="utf-8"))
    if formal.get("formal_training_accepted") != "12/12" or formal.get("final_test_access") != 0:
        raise AssertionError("corrected 12-cell training acceptance is not complete")
    prediction_acceptance = json.loads(PRIMARY_PREDICTION_ACCEPTANCE.read_text(encoding="utf-8"))
    if (prediction_acceptance.get("status") != "PASS"
            or prediction_acceptance.get("train_bundles_accepted") != "12/12"
            or prediction_acceptance.get("dev_bundles_accepted") != "12/12"
            or prediction_acceptance.get("final_test_access") != 0):
        raise AssertionError("primary target-blind train/dev prediction acceptance must pass before dense extension")
    return {"status": "PASS", "task_root": str(TASK_ROOT), "matrix_rows": len(rows),
            "dense_cells": len(cells) * 2, "old_cells": len(cells), "corrected_cells": len(cells),
            "input": panel, "cache": cache, "rolling_pair_contract": pair,
            "target_arrays_loaded": False, "final_test_access": 0,
            "training_gpu_hours": float(formal["actual_training_gpu_hours"]),
            "primary_prediction_acceptance_sha256": sha_file(PRIMARY_PREDICTION_ACCEPTANCE),
            "inference_device": "cpu", "cells": cells}


def write_lock(state: dict[str, Any]) -> dict[str, Any]:
    analysis_script = TASK_ROOT / "code/analyze_corrected_dense_rolling.py"
    if not analysis_script.is_file():
        raise FileNotFoundError("dense rolling analysis source must exist before protocol lock")
    if LOCK_PATH.exists():
        raise FileExistsError(f"dense extension protocol lock already exists: {LOCK_PATH}")
    lock = {
        "schema_version": 1, "status": "LOCKED", "study_id": "D3_EQ_CORRECTED_SCHEDULER_DENSE_SECONDARY_V1",
        "scope": "inference and pre-specified D4 same-absolute-target rolling-effect analysis only; no training",
        "parent_study_protocol_sha256": sha_file(STUDY_PROTOCOL),
        "parent_matrix_sha256": sha_file(MATRIX),
        "parent_prediction_acceptance_sha256": state["primary_prediction_acceptance_sha256"],
        "parent_prediction_acceptance_required": "PASS 12/12 train and 12/12 dev",
        "dense_input_path": str(INPUTS), "dense_input_sha256": sha_file(INPUTS),
        "dense_panel_path": str(PANEL), "dense_panel_sha256": sha_file(PANEL),
        "dense_ids_sha256": state["input"]["ids_sha256"], "dense_rows": 3718, "episodes": 118,
        "source_groups": 118, "pairs_per_horizon": 3600,
        "feature_cache_manifest_path": str(CACHE_MANIFEST), "feature_cache_manifest_sha256": state["cache"]["manifest_sha256"],
        "feature_cache_acceptance_sha256": state["cache"]["acceptance_sha256"],
        "feature_cache_shards": state["cache"]["shards"],
        "encoder_checkpoint_sha256": state["cache"]["source_backbone_sha256"],
        "preprocess_hash": state["cache"]["preprocess_hash"],
        "video_normalizer_path": str(VIDEO_NORMALIZER), "video_normalizer_sha256": sha_file(VIDEO_NORMALIZER),
        "target_normalizer_path": str(TARGET_NORMALIZER), "target_normalizer_sha256": sha_file(TARGET_NORMALIZER),
        "old_as_run_protocol_sha256": sha_file(OLD_PROTOCOL), "model_source_sha256": sha_file(MODEL_SOURCE),
        "inference_source_path": str(Path(__file__).resolve()), "inference_source_sha256": sha_file(Path(__file__).resolve()),
        "analysis_source_path": str(analysis_script.resolve()), "analysis_source_sha256": sha_file(analysis_script),
        "d4_pair_manifest_path": str(D4_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"),
        "d4_pair_manifest_sha256": sha_file(D4_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"),
        "d4_pair_time_support_path": str(D4_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv"),
        "d4_pair_time_support_sha256": sha_file(D4_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv"),
        "d4_analysis_source_path": str(D4_ROOT / "code/analyze_d3_d4_dense.py"),
        "d4_analysis_source_sha256": sha_file(D4_ROOT / "code/analyze_d3_d4_dense.py"),
        "d4_metric_source_path": str(D4_ROOT / "code/d4_eval.py"),
        "d4_metric_source_sha256": sha_file(D4_ROOT / "code/d4_eval.py"),
        "cells": state["cells"], "roles": ["old_as_run", "corrected_scheduler"],
        "legacy_d4_dense_predictions_reused": False,
        "legacy_mismatch_reason": "their 12 relevant checkpoint hashes differ from this matrix's matched old checkpoints",
        "execution": {"device": "cpu", "batch_size": 64, "targets_loaded_during_inference": False,
                      "training": False, "final_test_access": 0},
        "estimands": {"pair_method": "D4 locked same absolute target frame/timestamp common support",
                      "U": "D4 rolling update gain at identical common future timestamps",
                      "scheduler_gain": "corrected-scheduler metric minus matched old-as-run metric",
                      "video_specific": "TS11-D scheduler gain minus STATE-CTRL scheduler gain",
                      "bootstrap": "10,000 source-group resamples; average the same three fixed seeds within group before aggregate interval; no p-values"},
        "interpretation_scope": "secondary development-panel diagnostic; not independent confirmation or final/held-out evidence",
        "final_test_access": 0,
    }
    DENSE_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_json(LOCK_PATH, lock)
    return lock


class GridStore:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for out_i, row_i in enumerate(indices.astype(np.int64)):
            shard_i = bisect.bisect_right(self.starts, int(row_i)) - 1
            shard = self.shards[shard_i]
            if not int(shard["start_row"]) <= int(row_i) < int(shard["end_row"]):
                raise IndexError(f"feature cache misses dense input row {row_i}")
            by_shard.setdefault(shard_i, []).append((out_i, int(row_i) - int(shard["start_row"])))
        for shard_i, positions in by_shard.items():
            values = np.load(self.shards[shard_i]["array"], mmap_mode="r", allow_pickle=False)
            for out_i, local_i in positions:
                dynamic[out_i] = values[local_i, 0]
                static[out_i] = values[local_i, 1]
        return dynamic, static


def require_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError("dense extension lock is required before smoke/inference/audit")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("final_test_access") != 0:
        raise AssertionError("dense extension lock is invalid")
    if lock.get("inference_source_sha256") != sha_file(Path(__file__).resolve()):
        raise AssertionError("dense inference source changed after lock")
    if lock.get("analysis_source_sha256") != sha_file(TASK_ROOT / "code/analyze_corrected_dense_rolling.py"):
        raise AssertionError("dense analysis source changed after lock")
    if lock.get("parent_matrix_sha256") != sha_file(MATRIX) or lock.get("parent_study_protocol_sha256") != sha_file(STUDY_PROTOCOL):
        raise AssertionError("parent matrix/protocol changed after dense lock")
    if lock.get("parent_prediction_acceptance_sha256") != sha_file(PRIMARY_PREDICTION_ACCEPTANCE):
        raise AssertionError("primary train/dev prediction acceptance changed after dense lock")
    for key, path in (("dense_input_sha256", INPUTS), ("dense_panel_sha256", PANEL),
                      ("video_normalizer_sha256", VIDEO_NORMALIZER), ("target_normalizer_sha256", TARGET_NORMALIZER),
                      ("d4_pair_manifest_sha256", D4_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"),
                      ("d4_pair_time_support_sha256", D4_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv"),
                      ("d4_analysis_source_sha256", D4_ROOT / "code/analyze_d3_d4_dense.py"),
                      ("d4_metric_source_sha256", D4_ROOT / "code/d4_eval.py")):
        if lock.get(key) != sha_file(path):
            raise AssertionError(f"locked dense source/input changed: {path}")
    return lock


def predict_cell(data: dict[str, np.ndarray], grid: GridStore, video_mean: np.ndarray,
                 video_std: np.ndarray, sigma: torch.Tensor, cell: dict[str, Any],
                 checkpoint_path: Path, row_limit: int) -> np.ndarray:
    horizon, seed, config = int(cell["horizon_steps"]), int(cell["seed"]), str(cell["config"])
    n = min(3718, row_limit)
    state = torch.from_numpy(data["history_standardized"][:n].astype(np.float32, copy=False))
    mask = torch.from_numpy(data["history_mask"][:n].astype(np.float32, copy=False))
    relative = torch.from_numpy(data["history_relative_delta_standardized"][:n].astype(np.float32, copy=False))
    anchor = torch.from_numpy(data["anchor_command_native"][:n].astype(np.float32, copy=False))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    predictions = np.empty((n, horizon, 28), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, 64):
            end = min(n, start + 64)
            indices = np.arange(start, end, dtype=np.int64)
            dynamic, static = grid.get(indices)
            dynamic = ((dynamic - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            static = ((static - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            residual = model(state[start:end], mask[start:end], relative[start:end],
                             torch.from_numpy(dynamic), torch.from_numpy(static), anchor[start:end])
            predictions[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
    del model, payload
    if predictions.shape != (n, horizon, 28) or not np.isfinite(predictions).all():
        raise AssertionError(f"dense prediction shape/finite check failed: {cell['run_id']}")
    return predictions


def run_inference(smoke: bool = False) -> None:
    lock = require_lock()
    rows = read_matrix()
    data, ids, _ = load_inputs()
    cache_state = verify_cache(ids)
    if (cache_state["manifest_sha256"] != lock["feature_cache_manifest_sha256"]
            or cache_state["shards"] != lock["feature_cache_shards"]):
        raise AssertionError("dense cache shards changed after protocol lock")
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if sha_file(CACHE_MANIFEST) != lock["feature_cache_manifest_sha256"]:
        raise AssertionError("dense cache manifest changed after lock")
    video_norm = json.loads(VIDEO_NORMALIZER.read_text(encoding="utf-8"))
    target_norm = json.loads(TARGET_NORMALIZER.read_text(encoding="utf-8"))
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.as_tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32)).view(1, 1, 28)
    grid = GridStore(cache_manifest)
    cells = lock["cells"]
    selected_rows = rows[:1] if smoke else rows
    row_limit = 32 if smoke else 3718
    out_root = DENSE_ROOT / "smoke" if smoke else DENSE_ROOT / "predictions"
    started = time.perf_counter()
    for row in selected_rows:
        config, seed, horizon = row["config"], int(row["seed"]), int(row["horizon_steps"])
        cell = next(c for c in cells if (c["config"], c["seed"], c["horizon_steps"]) == (config, seed, horizon))
        for role, run_id, checkpoint_key in (
            ("old_as_run", cell["old_run_id"], "old_checkpoint"),
            ("corrected_scheduler", cell["corrected_run_id"], "corrected_checkpoint"),
        ):
            checkpoint = Path(cell[checkpoint_key])
            checkpoint_hash_key = "old_checkpoint_sha256" if role == "old_as_run" else "corrected_checkpoint_sha256"
            expected_checkpoint = cell[checkpoint_hash_key]
            if sha_file(checkpoint) != expected_checkpoint:
                raise AssertionError(f"checkpoint changed after lock: {run_id}")
            output_dir = out_root / role
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{run_id}.npz"
            sidecar_path = output_dir / f"{run_id}.json"
            if output_path.exists() or sidecar_path.exists():
                raise FileExistsError(f"dense output exists; no overwrite: {output_path}")
            t0 = time.perf_counter()
            prediction = predict_cell(data, grid, video_mean, video_std, sigma, row, checkpoint, row_limit)
            n = len(prediction)
            temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.part")
            with temp_path.open("wb") as stream:
                np.savez_compressed(stream, dense_id=ids[:n], episode_index=data["episode_index"][:n],
                                    original_task_index=data["original_task_index"][:n], anchor_frame=data["anchor_frame"][:n],
                                    anchor_timestamp=data["anchor_timestamp"][:n], predictions_native=prediction,
                                    horizon_steps=np.asarray(horizon), config=np.asarray(config), seed=np.asarray(seed),
                                    role=np.asarray(role), run_id=np.asarray(run_id))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, output_path)
            sidecar = {
                "status": "SMOKE_ONLY" if smoke else "PASS", "role": role, "run_id": run_id,
                "config": config, "seed": seed, "horizon_steps": horizon, "rows": n,
                "expected_rows": 3718, "dense_ids_sha256": sha_array(ids[:n]),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": expected_checkpoint,
                "prediction_path": str(output_path), "prediction_sha256": sha_file(output_path),
                "dense_extension_protocol_sha256": sha_file(LOCK_PATH),
                "parent_study_protocol_sha256": lock["parent_study_protocol_sha256"],
                "dense_input_sha256": lock["dense_input_sha256"],
                "feature_cache_manifest_sha256": lock["feature_cache_manifest_sha256"],
                "video_normalizer_sha256": lock["video_normalizer_sha256"],
                "target_normalizer_sha256": lock["target_normalizer_sha256"],
                "device": "cpu", "target_arrays_loaded": False, "optimizer_created": False,
                "backward_called": False, "optimizer_steps": 0, "final_test_access": 0,
                "elapsed_seconds": round(time.perf_counter() - t0, 6),
            }
            atomic_json(sidecar_path, sidecar)
            print(json.dumps({"status": sidecar["status"], "role": role, "run_id": run_id,
                              "rows": n, "seconds": sidecar["elapsed_seconds"]}, sort_keys=True), flush=True)
    print(json.dumps({"status": "SMOKE_COMPLETE" if smoke else "INFERENCE_COMPLETE",
                      "cells": len(selected_rows) * 2, "rows_per_bundle": row_limit,
                      "device": "cpu", "wall_seconds": round(time.perf_counter() - started, 3),
                      "final_test_access": 0}, sort_keys=True), flush=True)


def audit_predictions() -> None:
    lock = require_lock()
    rows = read_matrix()
    data, ids, _ = load_inputs()
    expected_ids_sha = sha_array(ids)
    expected_count = 0
    records = []
    for row in rows:
        key = (row["config"], int(row["seed"]), int(row["horizon_steps"]))
        cell = next(c for c in lock["cells"] if (c["config"], c["seed"], c["horizon_steps"]) == key)
        for role, run_id, checkpoint_field in (
            ("old_as_run", cell["old_run_id"], "old_checkpoint_sha256"),
            ("corrected_scheduler", cell["corrected_run_id"], "corrected_checkpoint_sha256"),
        ):
            side_path = DENSE_ROOT / "predictions" / role / f"{run_id}.json"
            pred_path = side_path.with_suffix(".npz")
            side = json.loads(side_path.read_text(encoding="utf-8"))
            if (side.get("status") != "PASS" or side.get("role") != role or side.get("run_id") != run_id
                    or side.get("rows") != 3718 or side.get("dense_ids_sha256") != expected_ids_sha
                    or side.get("checkpoint_sha256") != cell[checkpoint_field]
                    or side.get("prediction_sha256") != sha_file(pred_path)
                    or side.get("dense_extension_protocol_sha256") != sha_file(LOCK_PATH)
                    or side.get("target_arrays_loaded") is not False or side.get("final_test_access") != 0):
                raise AssertionError(f"dense prediction sidecar identity failed: {run_id}/{role}")
            with np.load(pred_path, allow_pickle=False) as bundle:
                pred_ids = bundle["dense_id"].astype("U96")
                prediction = bundle["predictions_native"]
                if (not np.array_equal(pred_ids, ids) or prediction.shape != (3718, int(row["horizon_steps"]), 28)
                        or not np.isfinite(prediction).all()
                        or str(bundle["role"].item()) != role or str(bundle["run_id"].item()) != run_id):
                    raise AssertionError(f"dense prediction bundle ID/shape/finiteness failed: {run_id}/{role}")
                for field in ("episode_index", "original_task_index", "anchor_frame"):
                    if not np.array_equal(bundle[field], data[field]):
                        raise AssertionError(f"dense prediction metadata mismatch {field}: {run_id}/{role}")
                if not np.allclose(bundle["anchor_timestamp"], data["anchor_timestamp"], atol=1e-6, rtol=0):
                    raise AssertionError(f"dense prediction timestamp mismatch: {run_id}/{role}")
            expected_count += 1
            records.append({"role": role, "run_id": run_id, "checkpoint_sha256": cell[checkpoint_field],
                            "prediction_sha256": side["prediction_sha256"], "rows": 3718})
    if expected_count != 24:
        raise AssertionError(f"expected 24 accepted dense predictions, got {expected_count}")
    acceptance = {
        "status": "PASS", "study_id": "D3_EQ_CORRECTED_SCHEDULER_DENSE_SECONDARY_V1",
        "expected_bundles": 24, "accepted_bundles": 24, "old_as_run_bundles": 12,
        "corrected_scheduler_bundles": 12, "dense_rows_per_bundle": 3718,
        "dense_ids_sha256": expected_ids_sha, "source_groups": 118,
        "target_arrays_loaded_during_inference": False, "training": "none",
        "device": "cpu", "optimizer_created": False, "backward_called": False,
        "dense_extension_protocol_sha256": sha_file(LOCK_PATH), "bundles": records,
        "final_test_access": 0,
    }
    atomic_json(DENSE_ROOT / "PREDICTION_ACCEPTANCE.json", acceptance)
    print(json.dumps({"status": "PASS", "accepted_bundles": 24, "rows_per_bundle": 3718,
                      "dense_ids_sha256": expected_ids_sha, "final_test_access": 0}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--write-lock", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--infer", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.preflight:
        result = preflight()
        print(json.dumps({key: value for key, value in result.items() if key != "cells" and key != "cache"},
                         ensure_ascii=False, sort_keys=True))
    elif args.write_lock:
        result = preflight()
        lock = write_lock(result)
        print(json.dumps({"status": "LOCKED", "lock_path": str(LOCK_PATH),
                          "lock_sha256": sha_file(LOCK_PATH), "inference_source_sha256": lock["inference_source_sha256"],
                          "checkpoint_count": len(lock["cells"]) * 2, "final_test_access": 0}, sort_keys=True))
    elif args.smoke:
        run_inference(smoke=True)
    elif args.infer:
        run_inference(smoke=False)
    else:
        audit_predictions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
