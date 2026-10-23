#!/usr/bin/env python3
"""Fresh-process V2 inference over label-free train-tune or dev prefix inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "locked_source"))

from prepare_v2_data import Q0ShardedGrid  # noqa: E402
from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_prefix_archive(path: Path, kind: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if kind == "train":
            keys = [
                "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster", "cache_index",
                "anchor_frame", "history_native", "history_mask", "relative_delta", "anchor_native", "inner_role",
            ]
            for key in keys:
                if key not in source.files:
                    raise RuntimeError(f"training prefix archive missing {key}")
            # target_native_h30 is deliberately never requested by inference.
        elif kind == "dev":
            keys = [
                "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster", "cache_index",
                "anchor_frame", "history_native", "history_mask", "relative_delta", "anchor_native",
            ]
            for key in keys:
                if key not in source.files:
                    raise RuntimeError(f"label-free dev prefix archive missing {key}")
            if "target_native_h30" in source.files:
                raise RuntimeError("dev prefix archive must remain target-free")
        else:
            raise ValueError(f"invalid dataset_kind={kind}")
        return {key: source[key].copy() for key in keys}


def select_rows(data: dict[str, np.ndarray], job: dict[str, Any]) -> np.ndarray:
    keep = np.ones(len(data["anchor_id"]), dtype=bool)
    if job["dataset_kind"] == "train":
        role = str(job.get("dataset_role", "inner_tune"))
        keep &= data["inner_role"] == role
    if job.get("scope") == "expert":
        keep &= data["task_id"] == int(job["task_id"])
    if job.get("task_filter") is not None:
        keep &= data["task_id"] == int(job["task_filter"])
    indices = np.flatnonzero(keep)
    order = np.argsort(data["anchor_id"][indices], kind="stable")
    indices = indices[order]
    max_rows = job.get("max_rows")
    if max_rows is not None:
        indices = indices[:int(max_rows)]
    if not len(indices):
        raise RuntimeError("inference selection has zero legal prefix rows")
    return indices.astype(np.int64)


def make_prefix(data: dict[str, np.ndarray], selected: np.ndarray, normalizer: dict[str, Any],
                grid: Q0ShardedGrid | None, video_mode: str, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state_mean = np.asarray(normalizer["state_mean"], dtype=np.float32)
    state_std = np.asarray(normalizer["state_safe_std"], dtype=np.float32)
    rel_mean = np.float32(normalizer["relative_delta_mean"])
    rel_std = np.float32(normalizer["relative_delta_safe_std"])
    state = (data["history_native"][selected] - state_mean[None, None]) / state_std[None, None]
    relative = (data["relative_delta"][selected] - rel_mean) / rel_std
    prefix: dict[str, torch.Tensor] = {
        "state": torch.as_tensor(state, device=device, dtype=torch.float32),
        "history_mask": torch.as_tensor(data["history_mask"][selected], device=device, dtype=torch.float32),
        "relative_delta": torch.as_tensor(relative, device=device, dtype=torch.float32),
        "anchor_command": torch.as_tensor(data["anchor_native"][selected], device=device, dtype=torch.float32),
    }
    if video_mode != "STATE-CTRL":
        if grid is None:
            raise RuntimeError("video fusion inference is missing the accepted Q0 cache reader")
        features = np.stack([
            grid.row(int(data["cache_index"][i]), str(data["anchor_id"][i]), str(data["episode_id"][i]))
            for i in selected
        ]).astype(np.float32, copy=False)
        v_mean = np.asarray(normalizer["video_mean"], dtype=np.float32)[None, None, None, :]
        v_std = np.asarray(normalizer["video_safe_std"], dtype=np.float32)[None, None, None, :]
        prefix["dynamic"] = torch.as_tensor((features[:, 0] - v_mean) / v_std, device=device, dtype=torch.float32)
        prefix["static"] = torch.as_tensor((features[:, 1] - v_mean) / v_std, device=device, dtype=torch.float32)
    task_rank = torch.as_tensor(data["task_rank"][selected].astype(np.int64), device=device, dtype=torch.long)
    return prefix, task_rank


def write_ids(path: Path, data: dict[str, np.ndarray], selected: np.ndarray) -> None:
    fields = ["row", "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster", "anchor_frame"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for out_i, ix in enumerate(selected.tolist()):
            writer.writerow({
                "row": out_i,
                "anchor_id": str(data["anchor_id"][ix]),
                "episode_id": int(data["episode_id"][ix]),
                "task_id": int(data["task_id"][ix]),
                "task_rank": int(data["task_rank"][ix]),
                "source_cluster": str(data["source_cluster"][ix]),
                "anchor_frame": int(data["anchor_frame"][ix]),
            })


def run(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["output_root"])
    complete_path = out_dir / "PREDICTION_COMPLETE.json"
    if complete_path.exists():
        print(json.dumps({"status": "already_complete", "job_id": job["job_id"]}))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    run_status = out_dir / "RUN_STATUS.json"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("V2 inference requires a watcher-authorized CUDA device")
        torch.manual_seed(int(job["seed"]))
        torch.cuda.manual_seed_all(int(job["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        device = torch.device("cuda:0")
        # Each watcher child is a fresh process with a single visible logical cuda:0;
        # leave the allocator peak counter at its fresh-process baseline.
        started = utc_now()
        model_path = Path(job["model_path"])
        model_manifest_path = Path(job["model_manifest_path"])
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if model_manifest.get("status") != "PASS" or sha_file(model_path) != model_manifest.get("model_file_sha256"):
            raise RuntimeError("model file hash/manifest acceptance failed")
        normalizer_path = Path(job["normalizer_path"])
        normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
        if normalizer.get("status") not in {"LOCKED_SCREENING", "LOCKED_FORMAL"}:
            raise RuntimeError("normalizer lock is absent")
        model = V2TaskConditionedPredictor(
            seed=int(job["seed"]),
            video_mode=str(job["video_mode"]),
            waypoint_mode=job.get("waypoint_mode"),
        )
        loaded_state = load_file(str(model_path), device="cpu")
        missing, unexpected = model.load_state_dict(loaded_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"safe model strict reload mismatch: missing={missing}, unexpected={unexpected}")
        reloaded_state_hash = state_hash(model)
        if reloaded_state_hash != model_manifest.get("final_state_hash"):
            raise RuntimeError("reloaded model state hash differs from accepted training manifest")
        model = model.to(device).eval()

        data_path = Path(job["data_path"])
        data = load_prefix_archive(data_path, str(job["dataset_kind"]))
        selected = select_rows(data, job)
        grid = None
        resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
        if str(job["video_mode"]) != "STATE-CTRL":
            feature_manifest_path = Path(resolved["feature_cache"])
            grid = Q0ShardedGrid(json.loads(feature_manifest_path.read_text(encoding="utf-8")))
        prefix, task_rank = make_prefix(data, selected, normalizer, grid, str(job["video_mode"]), device)
        predictions: list[torch.Tensor] = []
        goals: list[torch.Tensor] = []
        anchors = torch.as_tensor(data["anchor_native"][selected], device=device, dtype=torch.float32)
        sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)
        with torch.no_grad():
            for start in range(0, len(selected), int(job.get("batch_size", 32))):
                end = min(start + int(job.get("batch_size", 32)), len(selected))
                batch_prefix = {key: value[start:end] for key, value in prefix.items()}
                output = model(batch_prefix, task_rank[start:end])
                predictions.append(output.action_residual.detach().float().cpu())
                if output.goal_prediction is not None:
                    goals.append(output.goal_prediction.detach().float().cpu())
        residual = torch.cat(predictions, dim=0)
        native = anchors[:, None] + sigma[None, None] * residual.to(device)
        tensors: dict[str, torch.Tensor] = {
            "action_residual": residual.contiguous(),
            "prediction_native": native.detach().cpu().contiguous(),
        }
        if goals:
            goal_norm = torch.cat(goals, dim=0)
            goal_native = anchors + sigma[None] * goal_norm.to(device)
            tensors["goal_prediction_normalized"] = goal_norm.contiguous()
            tensors["goal_prediction_native"] = goal_native.detach().cpu().contiguous()
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise FloatingPointError("nonfinite prediction tensor")
        prediction_path = out_dir / "PREDICTIONS.safetensors"
        tmp = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.part")
        save_file(tensors, str(tmp))
        os.replace(tmp, prediction_path)
        ids_path = out_dir / "PREDICTION_ROWS.csv"
        write_ids(ids_path, data, selected)
        reload_qa = {
            "schema_version": 1,
            "status": "PASS",
            "fresh_process": True,
            "safe_format": "safetensors",
            "strict_state_load": True,
            "trained_model_state_hash": model_manifest["final_state_hash"],
            "reloaded_model_state_hash": reloaded_state_hash,
            "parameter_count": parameter_count(model),
            "prediction_rows": len(selected),
            "prediction_shape": list(tensors["prediction_native"].shape),
            "finite_outputs": True,
            "future_target_arrays_requested": False,
            "final_heldout_payload_access": 0,
        }
        atomic_json(out_dir / "RELOAD_QA.json", reload_qa)
        complete = {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": utc_now(),
            "job_id": job["job_id"],
            "model_id": job["model_id"],
            "phase": job["phase"],
            "seed": int(job["seed"]),
            "dataset_kind": job["dataset_kind"],
            "dataset_role": job.get("dataset_role"),
            "model_path": str(model_path),
            "model_sha256": sha_file(model_path),
            "model_state_hash": reloaded_state_hash,
            "normalizer_path": str(normalizer_path),
            "normalizer_sha256": sha_file(normalizer_path),
            "data_path": str(data_path),
            "data_sha256": sha_file(data_path),
            "feature_cache_manifest_sha256": resolved["feature_cache_manifest_sha256"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha_file(prediction_path),
            "row_ids_path": str(ids_path),
            "row_ids_sha256": sha_file(ids_path),
            "rows": len(selected),
            "prediction_shape": list(tensors["prediction_native"].shape),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "logical_device": str(device),
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
            "inference_script_sha256": sha_file(Path(__file__)),
            "future_target_arrays_requested": False,
            "final_heldout_payload_access": 0,
            "elapsed_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds(),
        }
        atomic_json(out_dir / "RELOAD_QA.json", reload_qa)
        atomic_json(complete_path, complete)
        atomic_json(run_status, {"status": "complete", "completed_utc": utc_now(), "job_id": job["job_id"]})
        print(json.dumps({"status": "PASS", "job_id": job["job_id"], "rows": len(selected), "prediction_sha256": complete["prediction_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(run_status, {
            "status": "failed", "failed_utc": utc_now(), "job_id": job.get("job_id"),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        })
        print(json.dumps({"status": "FAILED", "job_id": job.get("job_id"), "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), flush=True)
        return 1


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
