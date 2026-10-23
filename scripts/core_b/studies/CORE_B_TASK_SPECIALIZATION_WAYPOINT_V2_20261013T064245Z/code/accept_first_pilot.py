#!/usr/bin/env python3
"""Accept the two real screening pilot fits after independent fresh-process reload/inference."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    pilot_queue = json.loads((ROOT / "coordination/queues/SCREENING_PILOT.json").read_text())
    infer_queue_path = ROOT / "coordination/queues/PILOT_RELOAD.json"
    infer_queue = json.loads(infer_queue_path.read_text())
    if len(pilot_queue["jobs"]) != 2 or len(infer_queue["jobs"]) != 2:
        raise RuntimeError("pilot training and inference queues must each contain two jobs")
    accepted = []
    model_hashes = {}
    gpu_records = []
    for train_watcher_path in pilot_queue["jobs"]:
        train_outer = json.loads(Path(train_watcher_path).read_text())
        fit_path = Path(train_outer["command"][-1])
        fit = json.loads(fit_path.read_text())
        run_root = Path(fit["job_root"])
        complete_path = run_root / "RUN_COMPLETE.json"
        manifest_path = run_root / "MODEL_MANIFEST.json"
        checkpoint_receipt = run_root / "CHECKPOINT_EPOCH160.json"
        checkpoint_path = run_root / "model_epoch160.safetensors"
        if not all(path.is_file() for path in (complete_path, manifest_path, checkpoint_receipt, checkpoint_path)):
            raise RuntimeError(f"real screening pilot artifacts incomplete: {fit['job_id']}")
        complete = json.loads(complete_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        ckpt = json.loads(checkpoint_receipt.read_text())
        if complete.get("status") != "PASS" or int(complete["epochs_completed"]) != 160:
            raise RuntimeError(f"pilot fit did not finish all 160 epochs: {fit['job_id']}")
        if sha_file(checkpoint_path) != ckpt["weights_sha256"] or ckpt["model_state_hash"] != manifest["final_state_hash"]:
            raise RuntimeError(f"pilot checkpoint hash/state receipt mismatch: {fit['job_id']}")
        model_hashes[fit["model_id"]] = manifest["init_state_hash"]
        gpu_records.append({"job_id": fit["job_id"], "physical_gpu": manifest["physical_gpu"], "elapsed_seconds": complete["elapsed_seconds"], "peak_memory_reserved_mib": manifest["peak_memory_reserved_mib"]})
        accepted.append((fit, manifest, checkpoint_path))

    if set(model_hashes) != {"S-S-119", "S-V-119"}:
        raise RuntimeError(f"pilot model IDs differ: {sorted(model_hashes)}")
    if model_hashes["S-S-119"] != model_hashes["S-V-119"]:
        raise RuntimeError("state-only/fusion pilots did not share exact seed-locked INIT state")

    inference_by_training: dict[str, tuple[dict, Path]] = {}
    for raw in infer_queue["jobs"]:
        outer = json.loads(Path(raw).read_text())
        fit_path = Path(outer["command"][-1])
        inference = json.loads(fit_path.read_text())
        inference_by_training[str(inference["training_job_id"])] = (inference, Path(inference["output_root"]))
    prediction_records = []
    for fit, manifest, checkpoint_path in accepted:
        inference, out_dir = inference_by_training[fit["job_id"]]
        complete_path = out_dir / "PREDICTION_COMPLETE.json"
        reload_path = out_dir / "RELOAD_QA.json"
        predictions_path = out_dir / "PREDICTIONS.safetensors"
        id_path = out_dir / "PREDICTION_ROWS.csv"
        if not all(path.is_file() for path in (complete_path, reload_path, predictions_path, id_path)):
            raise RuntimeError(f"pilot fresh-process reload/prediction is incomplete: {fit['job_id']}")
        complete = json.loads(complete_path.read_text())
        reload_qa = json.loads(reload_path.read_text())
        pred = load_file(str(predictions_path), device="cpu")
        ids = rows(id_path)
        if complete.get("status") != "PASS" or complete.get("model_sha256") != sha_file(checkpoint_path):
            raise RuntimeError(f"pilot inference was not bound to the exact epoch160 model: {fit['job_id']}")
        if reload_qa.get("fresh_process") is not True or reload_qa.get("strict_state_load") is not True or reload_qa.get("reloaded_model_state_hash") != manifest["final_state_hash"]:
            raise RuntimeError(f"pilot fresh-process strict reload QA failed: {fit['job_id']}")
        if list(pred["prediction_native"].shape) != [1, 30, 28] or not torch_is_finite(pred):
            raise RuntimeError(f"pilot output shape/finite check failed: {fit['job_id']}")
        if len(ids) != 1 or int(ids[0]["task_id"]) != 119:
            raise RuntimeError(f"pilot did not score one task-119 inner-tune prefix: {fit['job_id']}")
        if inference["dataset_kind"] != "train" or inference["dataset_role"] != "inner_tune":
            raise RuntimeError("pilot prediction source must be a legal inner-tune prefix")
        prediction_records.append({
            "model_id": fit["model_id"], "path": str(predictions_path),
            "sha256": sha_file(predictions_path), "rows": 1,
        })

    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "pilot_screening_fits": 2,
        "pilot_training_epochs_each": 160,
        "pilot_lr": 1e-3,
        "pilot_seed": 202610130,
        "real_training_and_safe_checkpoint_save": True,
        "fresh_process_strict_safetensors_reload": True,
        "one_legal_inner_tune_prefix_prediction_each": True,
        "shared_state_video_init_hash_equal": True,
        "pilot_gpu_records": gpu_records,
        "prediction_records": prediction_records,
        "dev_targets_read": False,
        "final_heldout_payload_access": 0,
    }
    path = ROOT / "qa/FIRST_REAL_PILOT.json"
    if path.exists():
        raise FileExistsError(f"pilot acceptance already exists; refusing overwrite: {path}")
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def torch_is_finite(tensors: dict) -> bool:
    import torch
    return all(torch.isfinite(value).all().item() for value in tensors.values())


if __name__ == "__main__":
    main()
