from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from safetensors.torch import load_file

from v7_common import (
    NORMALIZER,
    ROOT,
    SEEDS,
    V2,
    V4,
    atomic_json,
    load_dev_prefix,
    load_model_api,
    make_grid,
    prefix_batch,
    sha_file,
)

SEED = 202610131


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_eval_model(seed: int, mode: str, model_path: Path, expected_hash: str | None, device: torch.device):
    model_type, parameter_count, state_hash = load_model_api()
    model = model_type(seed=seed, video_mode=mode, waypoint_mode=None)
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    digest = state_hash(model)
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(f"timing model state hash mismatch: {model_path}")
    return model.to(device).eval(), parameter_count(model), digest


def run(job_path: Path) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    output = Path(job["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("timing requires a watcher-authorized CUDA device")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    dev = load_dev_prefix()
    indices = np.arange(32, dtype=np.int64)
    normalizer = json.loads(NORMALIZER.read_text(encoding="utf-8"))
    grid = make_grid()
    prefix, task_rank, anchor = prefix_batch(dev, indices, normalizer, device, "TS11-D", grid)
    sigma = torch.as_tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device)

    source_map = json.loads((ROOT / "protocol/SOURCE_MAP.json").read_text(encoding="utf-8"))
    source_rows = {(int(row["seed"]), row["role"]): row for row in source_map["models"]}
    s_entry = source_rows[(SEED, "S")]
    v_entry = source_rows[(SEED, "V")]
    s_model, s_params, s_state = load_eval_model(SEED, "STATE-CTRL", Path(s_entry["model_path"]), s_entry["final_state_hash"], device)
    v_model, v_params, v_state = load_eval_model(SEED, "TS11-D", Path(v_entry["model_path"]), v_entry["final_state_hash"], device)
    fit_index = json.loads((ROOT / "protocol/FIT_ACCEPTANCE_INDEX.json").read_text(encoding="utf-8"))
    if fit_index.get("status") != "PASS":
        raise RuntimeError("timing requires the accepted full 9-fit matrix")
    students: dict[str, tuple[Any, int, str]] = {}
    fit_map = {(row["program_id"], int(row["seed"])): row for row in fit_index["entries"]}
    for program in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"):
        entry = fit_map[(program, SEED)]
        manifest = json.loads(Path(entry["model_manifest_path"]).read_text(encoding="utf-8"))
        students[program], n_params, state_digest = load_eval_model(
            SEED, "TS11-D", Path(entry["model_path"]), manifest["final_state_hash"], device
        )
        students[program] = (students[program], n_params, state_digest)

    def forward(model) -> None:
        with torch.inference_mode():
            output_value = model(prefix, task_rank)
            _native = anchor[:, None, :] + sigma[None, None, :] * output_value.action_residual.float()

    paths: list[tuple[str, list[Callable[[], None]], int, list[str]]] = [
        ("V_SINGLE", [lambda: forward(v_model)], v_params, [v_state]),
        ("S_PLUS_V_TWO_HEAD", [lambda: forward(s_model), lambda: forward(v_model)], s_params + v_params, [s_state, v_state]),
    ]
    for program in ("GT_RETRAIN", "SELF_SOFT", "MIX_SOFT"):
        model, n_params, state_digest = students[program]
        paths.append((f"{program}_SINGLE", [lambda model=model: forward(model)], n_params, [state_digest]))

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    results = []
    for name, operations, parameters, state_hashes in paths:
        for _ in range(5):
            for operation in operations:
                operation()
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(20):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            for operation in operations:
                operation()
            torch.cuda.synchronize(device)
            durations.append((time.perf_counter_ns() - started) / 1e9)
        values = np.asarray(durations, dtype=np.float64)
        results.append({
            "path": name,
            "warmups": 5,
            "repeats": 20,
            "batch_size": 32,
            "dtype": "float32",
            "median_seconds": float(np.median(values)),
            "q1_seconds": float(np.quantile(values, 0.25)),
            "q3_seconds": float(np.quantile(values, 0.75)),
            "iqr_seconds": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "parameter_count": int(parameters),
            "model_state_hashes": state_hashes,
        })
    torch.cuda.synchronize(device)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": utc_now(),
        "job_id": job["job_id"],
        "seed": SEED,
        "fixture": "first 32 ordered V4 label-free exposed-dev prefixes",
        "dev_target_values_read": 0,
        "prefix_and_Q0_features_preloaded_before_timer": True,
        "encoder_or_feature_extraction_timed": False,
        "model_loading_timed": False,
        "synchronization": "torch.cuda.synchronize before and after each timed repeat",
        "warmups": 5,
        "repeats": 20,
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "logical_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        "peak_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
        "results": results,
        "latency_scope": "cached input head forward; not end-to-end latency or a guaranteed 2x reduction",
        "final_heldout_payload_access": 0,
    }
    atomic_json(output / "INFERENCE_TIMING.json", receipt)
    print(json.dumps({"status": "PASS", "job_id": job["job_id"], "paths": len(results)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job_json)


if __name__ == "__main__":
    raise SystemExit(main())
