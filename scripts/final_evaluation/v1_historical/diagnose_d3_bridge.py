"""Compare the exact historical D3 extractor with the new train-only bridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
HISTORICAL = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/code/d3_extract_grid.py"
HISTORICAL_SHA256 = "25a27199f3854564d0760050891c6817333ca56e7ec2c4f6a35189721cdbb127"
Q0_MANIFEST = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
STAGE0_ROOT = RUN_ROOT / "features/stage0/core_b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float32)
    b = np.asarray(expected, dtype=np.float32)
    difference = np.abs(a - b)
    return {
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "p95_abs_difference": float(np.quantile(difference, 0.95)),
        "p99_abs_difference": float(np.quantile(difference, 0.99)),
        "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu", type=int, default=3)
    args = parser.parse_args()
    observed_gpu = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    if observed_gpu != str(args.expected_gpu):
        raise RuntimeError(f"watcher selected physical GPU {observed_gpu!r}; expected {args.expected_gpu}")
    if sha256_file(HISTORICAL) != HISTORICAL_SHA256:
        raise RuntimeError("historical D3 extractor source SHA mismatch")
    manifest_path = RUN_ROOT / "replay/core_b/core_b_bridge_stage0_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["dataset_role"] not in {"train", "dev"}:
        raise ValueError("diagnosis accepts only the registered single train/dev smoke row")
    row = rows[0]
    spec = importlib.util.spec_from_file_location("historical_d3_extract", HISTORICAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import historical D3 extractor")
    sys.path.insert(0, str(HISTORICAL.parent))
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    preprocess, load_base = historical.import_stack()
    device = torch.device("cuda:0")
    encoder, info = load_base(
        source_root=historical.VJEPA_SOURCE,
        checkpoint=historical.CHECKPOINT,
        device=device,
        num_frames=32,
        resolution=384,
    )
    encoder.eval()
    video = (historical.RAW_ROOT / row["video_path"]).resolve()
    wanted = historical.np.floor(np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)).astype(np.int64)
    frame_map = historical.read_frames(video, wanted)
    frames = np.stack([frame_map[int(index)] for index in wanted], axis=0)
    static = np.repeat(frames[-1:, ...], 32, axis=0)
    old_values = np.empty((2, 16, 4, 768), dtype=np.float32)
    with torch.inference_mode():
        for mode_index, raw in enumerate((frames, static)):
            clip = preprocess(raw, resolution=384).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = encoder(clip.unsqueeze(0))[0]
            old_values[mode_index], _ = historical.grid_pool(tokens)
    new_values = np.load(STAGE0_ROOT / "grid.npy", mmap_mode="r", allow_pickle=False)[0]

    sys.path.insert(0, str(OUT_ROOT / "code"))
    import core_b_predict as predictor
    accepted_sources = predictor.load_accepted_sources()
    accepted_manifest = json.loads(Q0_MANIFEST.read_text(encoding="utf-8"))
    q0 = accepted_sources.q0_grid_class(accepted_manifest)
    accepted_values = q0.row(int(row["cache_index"]), row["anchor_id"], row["episode_id"])

    report = {
        "schema_version": 1,
        "status": "COMPLETED_DIAGNOSTIC",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "physical_gpu": int(observed_gpu),
        "scope": "single accepted train/dev anchor; no final data or model outcomes",
        "anchor_id": row["anchor_id"],
        "episode_id": int(row["episode_id"]),
        "task_id": int(row["task_id"]),
        "video_sha256": sha256_file(video),
        "historical_extractor_sha256": HISTORICAL_SHA256,
        "checkpoint_sha256": sha256_file(historical.CHECKPOINT),
        "encoder_info": info,
        "comparisons": {
            "old_exact_source_vs_cached_q0": {
                "dynamic": metrics(old_values[0], accepted_values[0]),
                "static": metrics(old_values[1], accepted_values[1]),
            },
            "new_extractor_vs_old_exact_source": {
                "dynamic": metrics(new_values[0], old_values[0]),
                "static": metrics(new_values[1], old_values[1]),
            },
            "new_extractor_vs_cached_q0": {
                "dynamic": metrics(new_values[0], accepted_values[0]),
                "static": metrics(new_values[1], accepted_values[1]),
            },
        },
        "final_payload_read": False,
    }
    output = RUN_ROOT / "replay/COREB_D3_BRIDGE_DIAGNOSIS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
