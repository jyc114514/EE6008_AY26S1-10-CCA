#!/usr/bin/env python3
"""Restartable, target-blind CPU inference over accepted D3-EQ epoch-40 heads."""
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

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OLD = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
sys.path.insert(0, str(EQ / "code"))
from d3_model import D3TimeSpacePredictor  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class GridStore:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda x: int(x["start_row"]))
        self.starts = [int(x["start_row"]) for x in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for out_i, row_i in enumerate(indices.astype(np.int64)):
            shard_i = bisect.bisect_right(self.starts, int(row_i)) - 1
            shard = self.shards[shard_i]
            if not int(shard["start_row"]) <= int(row_i) < int(shard["end_row"]):
                raise IndexError(f"feature cache misses input row {row_i}")
            by_shard.setdefault(shard_i, []).append((out_i, int(row_i) - int(shard["start_row"])))
        for shard_i, pairs in by_shard.items():
            values = np.load(self.shards[shard_i]["array"], mmap_mode="r", allow_pickle=False)
            for out_i, local_i in pairs:
                dynamic[out_i] = values[local_i, 0]
                static[out_i] = values[local_i, 1]
        return dynamic, static


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=0, help="0 means all 36")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all primary-train rows")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--benchmark-only", action="store_true", help="measure bounded inference without writing prediction bundles")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-output", action="store_true", help="write partial smoke bundles outside accepted train outputs")
    args = ap.parse_args()
    if args.batch_size != 64 or args.threads < 1:
        raise ValueError("locked inference batch size is 64; CPU threads must be positive")
    if args.max_runs < 0 or args.max_rows < 0:
        raise ValueError("limits must be non-negative")
    if args.smoke_output and (args.max_runs == 0 or args.max_rows == 0):
        raise ValueError("smoke output requires explicit --max-runs and --max-rows")
    if args.benchmark_only and (args.max_runs == 0 or args.max_rows == 0):
        raise ValueError("benchmark-only mode requires explicit --max-runs and --max-rows")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    physical_gpu: int | None = None
    if args.device == "cuda":
        physical_gpu = int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1"))
        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if physical_gpu not in {2, 3} or visible_gpu != str(physical_gpu):
            raise RuntimeError("GPU child must be task-whitelisted physical GPU2/3 with exact CUDA_VISIBLE_DEVICES mapping")
        if not torch.cuda.is_available():
            raise RuntimeError("approved watcher child cannot access its assigned CUDA device")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        if os.environ.get("GPU_WATCHER_PHYSICAL_GPU"):
            raise RuntimeError("CPU inference refuses an inherited GPU-watcher device assignment")
        device = torch.device("cpu")

    with (EQ / "protocol/D3_EQ_MATRIX_36.csv").open(newline="", encoding="utf-8") as f:
        matrix = list(csv.DictReader(f))
    if len(matrix) != 36 or len({x["run_id"] for x in matrix}) != 36:
        raise AssertionError("locked matrix must contain exactly 36 unique runs")
    if args.max_runs:
        matrix = matrix[: args.max_runs]

    protocol_path = EQ / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
    protocol_sha = sha_file(protocol_path)
    input_path = V3 / "data/V3_MAIN_INPUTS.npz"
    archive = np.load(input_path, allow_pickle=False)
    # Explicit allowlist: targets_native and every target cache are not opened here.
    split = archive["split_code"].astype(np.int8, copy=True)
    train_idx = np.flatnonzero(split == 0).astype(np.int64)
    input_rows = len(split)
    if input_rows != 20079 or len(train_idx) != 17757:
        raise AssertionError(f"unexpected primary-train input population: total={input_rows}, train={len(train_idx)}")
    if args.max_rows:
        train_idx = train_idx[: args.max_rows]
    keys = ("anchor_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp",
            "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")
    data = {key: archive[key][train_idx].copy() for key in keys}
    ids = data["anchor_id"].astype("U96")
    if len(set(ids.tolist())) != len(ids):
        raise AssertionError("train inference IDs are not unique")

    cache_manifest_path = OLD / "d3/cache/D3_CACHE_MANIFEST.json"
    video_norm_path = OLD / "d3/cache/D3_VIDEO_NORMALIZER.json"
    target_norm_path = V3 / "data/V3_NORMALIZERS.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    video_norm = json.loads(video_norm_path.read_text(encoding="utf-8"))
    target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))
    grid = GridStore(cache_manifest)
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    state = torch.from_numpy(data["history_standardized"].astype(np.float32, copy=False))
    mask = torch.from_numpy(data["history_mask"].astype(np.float32, copy=False))
    relative = torch.from_numpy(data["history_relative_delta_standardized"].astype(np.float32, copy=False))
    anchor = torch.from_numpy(data["anchor_command_native"].astype(np.float32, copy=False))
    out_dir = OUT / ("predictions_smoke_cuda_guard_v2" if args.smoke_output else "predictions/train")
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in matrix:
        run_id = run["run_id"]
        config, horizon, seed = run["config"], int(run["horizon_steps"]), int(run["seed"])
        run_root = EQ / "d3/runs" / run_id
        complete = json.loads((run_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        checkpoint = run_root / "checkpoint_last.pt"
        checkpoint_hash = sha_file(checkpoint)
        if complete.get("status") != "PASS" or int(complete.get("epochs", -1)) != 40 or int(complete.get("optimizer_steps", -1)) != 11120:
            raise AssertionError(f"unaccepted epoch-40 checkpoint: {run_id}")
        if checkpoint_hash != complete.get("checkpoint_last_sha256") or complete.get("protocol_hash") != protocol_sha:
            raise AssertionError(f"checkpoint/protocol binding failure: {run_id}")
        pred_path = out_dir / f"{run_id}.npz"
        sidecar_path = out_dir / f"{run_id}.json"
        expected_ids_hash = sha_array(ids.astype("U96"))
        if args.resume and pred_path.is_file() and sidecar_path.is_file():
            side = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (side.get("status") == "PASS" and int(side.get("rows", -1)) == len(ids)
                    and side.get("ids_sha256") == expected_ids_hash
                    and side.get("checkpoint_sha256") == checkpoint_hash
                    and side.get("prediction_sha256") == sha_file(pred_path)):
                print(json.dumps({"status": "PASS_REUSED", "run_id": run_id, "rows": len(ids)}), flush=True)
                continue
            raise FileExistsError(f"existing output is not a verified complete bundle: {pred_path}")
        if pred_path.exists() or sidecar_path.exists():
            raise FileExistsError(f"output already exists; use --resume only for a verified bundle: {pred_path}")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        if physical_gpu is not None:
            torch.cuda.reset_peak_memory_stats(device)
        predictions = np.empty((len(ids), horizon, 28), dtype=np.float32)
        t0 = time.perf_counter()
        for start in range(0, len(ids), args.batch_size):
            end = min(len(ids), start + args.batch_size)
            dynamic, static = grid.get(train_idx[start:end])
            dynamic = ((dynamic - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            static = ((static - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            with torch.inference_mode():
                anchor_batch = anchor[start:end].to(device)
                residual = model(state[start:end].to(device), mask[start:end].to(device), relative[start:end].to(device),
                                 torch.from_numpy(dynamic).to(device), torch.from_numpy(static).to(device), anchor_batch)
                predictions[start:end] = (anchor_batch[:, None, :] + residual * sigma).detach().cpu().numpy().astype(np.float32)
            if end == min(args.batch_size, len(ids)) or end % 2048 == 0 or end == len(ids):
                print(json.dumps({"status": "inference_progress", "run_id": run_id, "rows_done": end,
                                  "rows_total": len(ids), "elapsed_seconds": round(time.perf_counter() - t0, 3)}), flush=True)
        if predictions.shape != (len(ids), horizon, 28) or not np.isfinite(predictions).all():
            raise AssertionError(f"prediction shape/finite failure: {run_id}")
        if args.benchmark_only:
            print(json.dumps({"status": "BENCHMARK_ONLY_PASS", "run_id": run_id,
                              "rows": len(ids), "seconds": round(time.perf_counter() - t0, 6),
                              "rows_per_second": round(len(ids) / max(time.perf_counter() - t0, 1e-9), 3),
                              "prediction_bundle_written": False, "target_arrays_loaded": False}), flush=True)
            continue
        tmp = pred_path.with_name(f".{pred_path.name}.{os.getpid()}.part")
        with tmp.open("wb") as f:
            np.savez_compressed(f, anchor_id=ids, episode_index=data["episode_index"],
                                original_task_index=data["original_task_index"], anchor_frame=data["anchor_frame"],
                                anchor_timestamp=data["anchor_timestamp"], predictions_native=predictions,
                                horizon_steps=np.asarray(horizon), config=np.asarray(config), seed=np.asarray(seed),
                                split=np.asarray("train"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pred_path)
        full_run = len(ids) == 17757
        sidecar = {"status": "PASS" if full_run else "PARTIAL_SMOKE", "run_id": run_id,
                   "config": config, "horizon_steps": horizon, "seed": seed, "rows": len(ids),
                   "expected_rows": 17757, "ids_sha256": expected_ids_hash,
                   "checkpoint_sha256": checkpoint_hash, "prediction_sha256": sha_file(pred_path),
                   "feature_cache_manifest_sha256": sha_file(cache_manifest_path),
                   "video_normalizer_sha256": sha_file(video_norm_path),
                   "target_normalizer_sha256": sha_file(target_norm_path),
                   "target_arrays_loaded": False, "split_used": "primary_train_only",
                   "device": str(device), "physical_gpu": physical_gpu,
                   "logical_device": "cuda:0" if physical_gpu is not None else "cpu",
                   "optimizer_created": False, "backward_called": False,
                   "optimizer_steps": 0, "final_test_access": 0,
                   "elapsed_seconds": round(time.perf_counter() - t0, 6)}
        if physical_gpu is not None:
            sidecar["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            sidecar["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        atomic_json(sidecar_path, sidecar)
        print(json.dumps({"status": sidecar["status"], "run_id": run_id, "rows": len(ids),
                          "seconds": sidecar["elapsed_seconds"], "prediction_bytes": pred_path.stat().st_size}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
