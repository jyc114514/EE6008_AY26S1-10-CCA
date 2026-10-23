from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from v7_common import atomic_json, load_model_api, sha_file, validate_prefix_keys


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(job_path: Path) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    model_type, parameter_count, state_hash = load_model_api()
    model = model_type(
        seed=int(job["seed"]),
        video_mode=str(job.get("video_mode", "TS11-D")),
        waypoint_mode=None,
    )
    tensors = load_file(str(job["model_path"]), device="cpu")
    if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("safe model tensor payload is empty or nonfinite")
    model.load_state_dict(tensors, strict=True)
    actual_hash = state_hash(model)
    if actual_hash != job["expected_state_hash"]:
        raise RuntimeError(f"fresh process state hash mismatch: {actual_hash} != {job['expected_state_hash']}")
    replay_result = {"target_blind_eval_replay": "NOT_REQUESTED"}
    if job.get("prefix_fixture_path") and job.get("expected_prediction_path"):
        fixture = load_file(str(job["prefix_fixture_path"]), device="cpu")
        expected = load_file(str(job["expected_prediction_path"]), device="cpu")
        prefix = {key: fixture[key] for key in ("state", "history_mask", "relative_delta", "anchor_command", "dynamic", "static") if key in fixture}
        validate_prefix_keys(prefix, str(job.get("video_mode", "TS11-D")))
        task_rank = fixture["task_rank"].to(dtype=torch.long)
        anchor = fixture["anchor_native"].to(dtype=torch.float32)
        sigma = fixture["target_sigma"].to(dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(prefix, task_rank)
            replay_native = anchor[:, None, :] + sigma[None, None, :] * output.action_residual.float()
        expected_native = expected["prediction_native"].to(dtype=torch.float32)
        if replay_native.shape != expected_native.shape:
            raise RuntimeError("target-blind fresh-process replay shape differs from the fixture output")
        max_abs = float(torch.max(torch.abs(replay_native - expected_native)).item())
        atol = float(job.get("replay_atol", 1e-7))
        rtol = float(job.get("replay_rtol", 1e-6))
        if not torch.allclose(replay_native, expected_native, atol=atol, rtol=rtol):
            raise RuntimeError(f"target-blind fresh-process eval replay mismatch: max_abs={max_abs}")
        replay_result = {
            "target_blind_eval_replay": "PASS",
            "prefix_fixture_path": str(job["prefix_fixture_path"]),
            "prefix_fixture_sha256": sha_file(Path(job["prefix_fixture_path"])),
            "expected_prediction_path": str(job["expected_prediction_path"]),
            "expected_prediction_sha256": sha_file(Path(job["expected_prediction_path"])),
            "max_abs_difference": max_abs,
            "atol": atol,
            "rtol": rtol,
            "outcome_or_teacher_inputs_present": False,
        }
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "model_path": str(job["model_path"]),
        "model_sha256": sha_file(Path(job["model_path"])),
        "model_state_hash": actual_hash,
        "video_mode": str(job.get("video_mode", "TS11-D")),
        "seed": int(job["seed"]),
        "parameter_count": parameter_count(model),
        "strict_safe_tensor_load": True,
        "fresh_process": True,
        "device": "CPU",
        "cuda_device_queries": 0,
        **replay_result,
        "final_heldout_payload_access": 0,
    }
    atomic_json(Path(job["receipt_path"]), receipt)
    print(json.dumps({"status": "PASS", "state_hash": actual_hash}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
