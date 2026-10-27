from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

V8_ROOT = Path(__file__).resolve().parents[1]
V7_ROOT = V8_ROOT.parent / "CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z"
V7_CODE = V7_ROOT / "code"
if str(V7_CODE) not in sys.path:
    sys.path.insert(0, str(V7_CODE))

from v7_common import (  # noqa: E402
    NORMALIZER,
    TRAIN_NPZ,
    atomic_json,
    load_model_api,
    load_train_prefix,
    make_grid,
    ordered_receiver_hash,
    prefix_batch,
    receiver_rows,
    save_safe_prediction,
    sha_file,
    write_receiver_rows,
)

EXPECTED_PROGRAMS = {"SELF_SOFT", "MIX_SOFT"}
EXPECTED_SEEDS = {202610131, 202610132, 202610133}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(job_path: Path) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    output = Path(job["output_root"]).resolve()
    if V8_ROOT.resolve() not in output.parents:
        raise RuntimeError(f"refusing output outside V8 root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    complete_path = output / "PREDICTION_COMPLETE.json"
    if complete_path.exists():
        raise FileExistsError(f"preserving existing output: {complete_path}")

    started = time.perf_counter()
    try:
        program = str(job["program_id"])
        seed = int(job["seed"])
        if program not in EXPECTED_PROGRAMS or seed not in EXPECTED_SEEDS:
            raise ValueError("job is outside the six authorized SELF_SOFT/MIX_SOFT seeds")
        fit_index_path = V7_ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json"
        fit_index = json.loads(fit_index_path.read_text(encoding="utf-8"))
        accepted = [
            row for row in fit_index["entries"]
            if row.get("program_id") == program and int(row.get("seed", -1)) == seed
        ]
        if len(accepted) != 1 or not accepted[0].get("passed"):
            raise RuntimeError("no unique accepted V7 final model is bound to this job")
        fit = accepted[0]
        for key in ("model_path", "model_sha256", "model_manifest_path", "model_manifest_sha256"):
            if str(job[key]) != str(fit[key]):
                raise RuntimeError(f"job {key} differs from FIT_ACCEPTANCE_INDEX")
        model_path = Path(fit["model_path"])
        manifest_path = Path(fit["model_manifest_path"])
        if sha_file(fit_index_path) != job["fit_acceptance_index_sha256"]:
            raise RuntimeError("V7 FIT_ACCEPTANCE_INDEX digest differs from the job lock")
        if sha_file(manifest_path) != fit["model_manifest_sha256"]:
            raise RuntimeError("accepted model manifest digest mismatch")
        if sha_file(model_path) != fit["model_sha256"]:
            raise RuntimeError("accepted model weights digest mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "PASS"
            or str(manifest.get("program_id")) != program
            or int(manifest.get("seed", -1)) != seed
            or int(manifest.get("epochs", -1)) != 160
            or int(manifest.get("optimizer_steps", -1)) != 3200
            or int(manifest.get("final_heldout_payload_access", -1)) != 0
            or int(manifest.get("encoder_extraction", -1)) != 0
            or manifest.get("normalizer_sha256") != job["normalizer_sha256"]
            or manifest.get("train_archive_sha256") != job["train_npz_sha256"]
        ):
            raise RuntimeError("accepted model manifest failed frozen replay identity checks")
        if sha_file(TRAIN_NPZ) != job["train_npz_sha256"]:
            raise RuntimeError("locked outer-train archive digest mismatch")
        if sha_file(NORMALIZER) != job["normalizer_sha256"]:
            raise RuntimeError("accepted normalizer digest mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("frozen train-prefix replay requires watcher-authorized CUDA")
        device = torch.device("cuda:0")

        data = load_train_prefix()
        if set(data) != {
            "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
            "cache_index", "anchor_frame", "history_native", "history_mask",
            "relative_delta", "anchor_native", "inner_role",
        }:
            raise RuntimeError("train prefix loader returned unexpected keys")
        if "target_native_h30" in data or len(data["anchor_id"]) != 1280:
            raise RuntimeError("frozen replay loaded a target or wrong train row count")
        rows = receiver_rows(data)
        id_hash = ordered_receiver_hash(rows)
        if id_hash != job["ordered_receiver_ids_sha256"]:
            raise RuntimeError("accepted ordered train receiver IDs differ from the replay lock")
        if sha_file(Path(job["feature_cache_manifest_path"])) != job["feature_cache_manifest_sha256"]:
            raise RuntimeError("feature-cache manifest digest mismatch")
        normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
        sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)
        model_type, _, state_hash = load_model_api()
        model = model_type(seed=seed, video_mode="TS11-D", waypoint_mode=None)
        model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
        before_hash = state_hash(model)
        if before_hash != manifest.get("final_state_hash"):
            raise RuntimeError("loaded final model state differs from accepted manifest")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("frozen model has trainable parameters")
        model.to(device)
        model.eval()
        grid = make_grid()
        torch.cuda.reset_peak_memory_stats(device)

        outputs = []
        batch_size = int(job["batch_size"])
        if batch_size != 32 or 1280 % batch_size:
            raise RuntimeError("authorized replay profile requires 40 full batches of 32")
        with torch.inference_mode():
            for start in range(0, 1280, batch_size):
                indices = np.arange(start, start + batch_size, dtype=np.int64)
                prefix, task_rank, anchor = prefix_batch(
                    data, indices, normalizer, device, "TS11-D", grid
                )
                result = model(prefix, task_rank)
                native = anchor[:, None, :] + sigma[None, None, :] * result.action_residual.float()
                outputs.append(native.detach().cpu().numpy().astype(np.float64, copy=False))
        torch.cuda.synchronize(device)
        prediction = np.concatenate(outputs, axis=0)
        if prediction.shape != (1280, 30, 28) or not np.isfinite(prediction).all():
            raise RuntimeError(f"prediction shape/finite failure: {prediction.shape}")
        after_hash = state_hash(model)
        if before_hash != after_hash:
            raise RuntimeError("frozen model parameter/buffer state changed during inference")

        prediction_path = output / "PREDICTIONS.safetensors"
        rows_path = output / "PREDICTION_ROWS.csv"
        save_safe_prediction(prediction_path, prediction)
        write_receiver_rows(rows_path, data)
        receipt = {
            "schema_version": 1,
            "status": "PASS",
            "created_utc": utc_now(),
            "job_id": job["job_id"],
            "program_id": program,
            "seed": seed,
            "split": "outer_train_in_sample_not_OOF",
            "model_path": str(model_path),
            "model_sha256": sha_file(model_path),
            "model_manifest_path": str(manifest_path),
            "model_manifest_sha256": sha_file(manifest_path),
            "model_state_hash_before": before_hash,
            "model_state_hash_after": after_hash,
            "model_epoch": 160,
            "model_parameters_requires_grad": False,
            "model_eval_mode": True,
            "inference_mode": True,
            "optimizer_constructed": False,
            "optimizer_updates": 0,
            "teacher_forward": False,
            "encoder_extraction": 0,
            "train_npz_path": str(TRAIN_NPZ),
            "train_npz_sha256": sha_file(TRAIN_NPZ),
            "train_fields_loaded": sorted(data),
            "future_target_field_requested": False,
            "future_target_values_read": 0,
            "dev_target_values_read": 0,
            "receiver_rows": len(rows),
            "ordered_receiver_ids_sha256": id_hash,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha_file(prediction_path),
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": "float64",
            "prediction_rows_path": str(rows_path),
            "prediction_rows_sha256": sha_file(rows_path),
            "all_finite": True,
            "batch_size": batch_size,
            "batches": 40,
            "video_mode": "TS11-D",
            "target_sigma_sha256": job["target_sigma_sha256"],
            "normalizer_sha256": job["normalizer_sha256"],
            "feature_cache_manifest_sha256": job["feature_cache_manifest_sha256"],
            "logical_device": str(device),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
            "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
            "elapsed_seconds": time.perf_counter() - started,
            "final_heldout_payload_access": 0,
            "old_root_writes": 0,
        }
        atomic_json(complete_path, receipt)
        atomic_json(status_path, {"status": "complete", "completed_utc": utc_now(), "job_id": job["job_id"]})
        print(json.dumps({
            "status": "PASS", "job_id": job["job_id"], "rows": len(rows),
            "prediction_sha256": receipt["prediction_sha256"],
            "physical_gpu": receipt["physical_gpu"],
            "peak_allocated_mib": receipt["peak_memory_allocated_mib"],
            "peak_reserved_mib": receipt["peak_memory_reserved_mib"],
            "elapsed_seconds": receipt["elapsed_seconds"],
        }, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        atomic_json(status_path, {
            "status": "failed",
            "failed_utc": utc_now(),
            "job_id": job.get("job_id"),
            "program_id": job.get("program_id"),
            "seed": job.get("seed"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
            "final_heldout_payload_access": 0,
        })
        print(json.dumps({
            "status": "FAILED", "job_id": job.get("job_id"),
            "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True), flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
