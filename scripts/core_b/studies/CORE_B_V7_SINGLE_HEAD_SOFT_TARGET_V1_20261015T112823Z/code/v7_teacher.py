from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from v7_common import (
    NORMALIZER,
    ROOT,
    TRAIN_NPZ,
    TASKS,
    atomic_json,
    configure_determinism,
    load_model_api,
    load_train_prefix,
    make_grid,
    prefix_batch,
    receiver_rows,
    save_safe_prediction,
    sha_file,
    write_receiver_rows,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(job_path: Path) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    output = Path(job["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    complete_path = output / "PREDICTION_COMPLETE.json"
    if complete_path.exists():
        prior = json.loads(complete_path.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            print(json.dumps({"status": "already_complete", "job_id": job["job_id"]}), flush=True)
            return 0

    started = time.perf_counter()
    try:
        if sha_file(TRAIN_NPZ) != job["train_npz_sha256"]:
            raise RuntimeError("outer-train archive digest differs from the locked job")
        if sha_file(NORMALIZER) != job["normalizer_sha256"]:
            raise RuntimeError("normalizer digest differs from the locked job")
        if not torch.cuda.is_available():
            raise RuntimeError("teacher replay requires a watcher-authorized CUDA device")
        device = torch.device("cuda:0")
        seed = int(job["seed"])
        role = str(job["teacher_role"])
        if role not in {"S", "V"} or seed not in (202610131, 202610132, 202610133):
            raise ValueError("teacher role/seed is outside the locked 6-bundle bank")

        configure_determinism(seed)
        data = load_train_prefix()
        if len(data["anchor_id"]) != 1280:
            raise RuntimeError(f"train receiver count mismatch: {len(data['anchor_id'])}")
        rows = receiver_rows(data)
        norm = json.loads(NORMALIZER.read_text(encoding="utf-8"))
        source_map_path = ROOT / "protocol/SOURCE_MAP.json"
        if sha_file(source_map_path) != job.get("source_map_sha256"):
            raise RuntimeError("V7 source map digest differs from the teacher job lock")
        source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
        source_matches = [row for row in source_map["models"] if int(row["seed"]) == seed and row["role"] == role]
        if len(source_matches) != 1:
            raise RuntimeError("V7 source map does not contain exactly one accepted teacher identity")
        locked_source = source_matches[0]
        model_manifest_path = Path(job["model_manifest_path"])
        model_path = Path(job["model_path"])
        if str(model_path) != locked_source["model_path"] or str(model_manifest_path) != locked_source["model_manifest_path"]:
            raise RuntimeError("teacher job path differs from the locked accepted source model")
        manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        expected_mode = "STATE-CTRL" if role == "S" else "TS11-D"
        if (
            manifest.get("status") != "PASS"
            or int(manifest.get("epochs", -1)) != 160
            or int(manifest.get("optimizer_steps", -1)) != 3200
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("video_mode") != expected_mode
            or int(manifest.get("final_heldout_payload_access", -1)) != 0
        ):
            raise RuntimeError(f"accepted {role} source model manifest failed the locked identity checks")
        if sha_file(model_path) != manifest.get("model_file_sha256") or manifest.get("model_file_sha256") != job.get("model_sha256_expected") or manifest.get("model_file_sha256") != locked_source["model_sha256"]:
            raise RuntimeError(f"accepted source model weights hash mismatch: {model_path}")

        model_type, _, state_hash = load_model_api()
        model = model_type(seed=seed, video_mode=expected_mode, waypoint_mode=None)
        model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
        before_hash = state_hash(model)
        if before_hash != manifest.get("final_state_hash"):
            raise RuntimeError("source model strict safe reload state hash mismatch")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(device).eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("frozen teacher contains a trainable parameter")
        if role == "V":
            grid = make_grid()
        else:
            grid = None
        sigma = torch.as_tensor(np.asarray(norm["target_sigma"], dtype=np.float32), device=device)
        outputs: list[np.ndarray] = []
        batch_size = int(job.get("batch_size", 32))
        if batch_size != 32:
            raise RuntimeError("teacher inference batch size differs from the locked V4 profile")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for start in range(0, len(data["anchor_id"]), batch_size):
                indices = np.arange(start, min(start + batch_size, len(data["anchor_id"])), dtype=np.int64)
                prefix, task_rank, anchor = prefix_batch(data, indices, norm, device, expected_mode, grid)
                output_value = model(prefix, task_rank)
                native = anchor[:, None, :] + sigma[None, None, :] * output_value.action_residual.float()
                outputs.append(native.detach().cpu().numpy().astype(np.float64, copy=False))
        torch.cuda.synchronize(device)
        prediction = np.concatenate(outputs, axis=0)
        if prediction.shape != (1280, 30, 28) or not np.isfinite(prediction).all():
            raise RuntimeError(f"teacher prediction shape/finite failure: {prediction.shape}")
        after_hash = state_hash(model)
        if before_hash != after_hash:
            raise RuntimeError("teacher parameter/buffer hash changed during inference")

        pred_path = output / "PREDICTIONS.safetensors"
        rows_path = output / "PREDICTION_ROWS.csv"
        save_safe_prediction(pred_path, prediction)
        write_receiver_rows(rows_path, data)
        physical_gpu = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "job_id": job["job_id"],
            "teacher_role": role,
            "seed": seed,
            "source_model_path": str(model_path),
            "source_model_sha256": sha_file(model_path),
            "source_model_manifest_path": str(model_manifest_path),
            "source_model_manifest_sha256": sha_file(model_manifest_path),
            "source_final_state_hash": before_hash,
            "source_epoch": 160,
            "source_optimizer_steps": 3200,
            "teacher_parameters_updated": 0,
            "teacher_requires_grad_all_false": True,
            "teacher_buffers_unchanged": True,
            "student_forward": False,
            "train_npz_path": str(TRAIN_NPZ),
            "train_npz_sha256": sha_file(TRAIN_NPZ),
            "train_fields_requested": sorted(data),
            "future_target_field_requested": False,
            "future_target_values_read": 0,
            "receiver_rows": len(rows),
            "ordered_receiver_ids_sha256": job["ordered_receiver_ids_sha256"],
            "prediction_path": str(pred_path),
            "prediction_sha256": sha_file(pred_path),
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": "float64",
            "prediction_rows_path": str(rows_path),
            "prediction_rows_sha256": sha_file(rows_path),
            "all_finite": True,
            "feature_cache_manifest_sha256": job["feature_cache_manifest_sha256"] if role == "V" else None,
            "logical_device": str(device),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": visible_devices,
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
            "elapsed_seconds": time.perf_counter() - started,
            "final_heldout_payload_access": 0,
            "prediction_role": "outer_train_in_sample_supervision_not_OOF",
        }
        atomic_json(complete_path, receipt)
        atomic_json(status_path, {"status": "complete", "completed_utc": utc_now(), "job_id": job["job_id"]})
        print(json.dumps({"status": "PASS", "job_id": job["job_id"], "rows": 1280, "prediction_sha256": receipt["prediction_sha256"]}), flush=True)
        return 0
    except Exception as exc:
        atomic_json(status_path, {
            "status": "failed",
            "failed_utc": utc_now(),
            "job_id": job.get("job_id"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        })
        print(json.dumps({"status": "FAILED", "job_id": job.get("job_id"), "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
