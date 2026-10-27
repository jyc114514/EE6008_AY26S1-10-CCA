"""Lock D3 cache/matrix/training contract before new scores are read."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, atomic_write_json, read_json, sha256_json, utc_now, write_csv


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
CONFIGS = ["TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL"]
HORIZONS = [(30, 1.0), (60, 2.0)]


def observed_seeds() -> set[int]:
    observed: set[int] = set()
    candidate_paths = [V3 / "SEED_REGISTRY.json"]
    candidate_paths.extend(path for path in REPO_ROOT.joinpath("reports/core_b").glob("*/MATRIX*.csv") if path.is_file())
    candidate_paths.extend(path for path in REPO_ROOT.joinpath("reports/core_b").glob("*/B_MATRIX.csv") if path.is_file())
    for path in candidate_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        observed.update(int(value) for value in re.findall(r"(?<!\d)2026\d{4}(?!\d)", text))
    return observed


def main() -> int:
    excluded = observed_seeds()
    seeds = []
    candidate = 20261001
    while len(seeds) < 3:
        if candidate not in excluded:
            seeds.append(candidate)
        candidate += 1
    protocol = {
        "protocol_id": "CORE_B_D3_TIME_SPACE_FACTORIAL_V1",
        "generated_utc": utc_now(),
        "configs": CONFIGS,
        "horizons": [steps for steps, _ in HORIZONS],
        "seeds": seeds,
        "matrix_size": len(CONFIGS) * len(HORIZONS) * len(seeds),
        "cache": {"shape": ["N", 2, 16, 4, 768], "dtype": "float32", "dynamic": "past32 real frames", "static": "last legal frame repeated 32 times", "grid": "16x24x24 raw tokens to 16x2x2 area-weighted bins", "normalizer": "D primary train only, episode/window/time/patch-area weighted"},
        "model": {"state": "V3 TCN state branch 71->128, 5 causal residual dilations 1,2,4,8,16", "video": "shared 768->128 with fixed temporal plus row/col 2D encodings", "slots": 64, "decoder": "V3 two-layer TransformerDecoder d_model=128 heads=4 FFN=512", "output": "anchor_native + target_sigma * zero-initialized residual"},
        "training": {"epochs": 40, "checkpoint_rule": "last40", "optimizer": "AdamW lr=1e-3 betas=(.9,.999) eps=1e-8 wd=1e-4", "schedule": "cosine to 1e-5", "loss": "equal Arm/Hand standardized L1 + 0.1 safe-d future FD L1", "effective_batch": 64, "fallback": "batch32 accumulation2 only after measured OOM", "early_stopping": False, "warm_start": False, "encoder_updated": False},
        "controls": {"TS00-D": "time mean + spatial mean repeated to 64", "TS10-D": "time preserved + spatial mean repeated per time slot", "TS01-D": "spatial bins preserved + time mean repeated", "TS11-D": "time and space preserved", "TS11-S": "static clip grid with same slot contract", "STATE-CTRL": "video content zero after normalization"},
        "primary_contrasts": ["TS11-D_vs_TS10-D", "TS11-D_vs_TS01-D", "TS11-D_vs_TS11-S", "TS11-D_vs_STATE-CTRL"],
        "factorial_interaction": "(L11-L10)-(L01-L00)",
        "final_test_access": 0,
    }
    protocol["protocol_hash"] = sha256_json(protocol)
    atomic_write_json(BENCH_ROOT / "protocol/D3_PROTOCOL_LOCKED.json", protocol)
    rows = []
    for seed in seeds:
        for horizon, seconds in HORIZONS:
            for config in CONFIGS:
                rows.append({"run_id": f"D3-{config}-H{horizon}-S{seed}", "config": config, "seed": seed, "horizon_steps": horizon, "horizon_seconds": seconds, "training_status": "LOCKED_NOT_STARTED", "prediction_status": "LOCKED_NOT_STARTED", "checkpoint_rule": "last40", "final_test_access": 0})
    write_csv(BENCH_ROOT / "protocol/D3_MATRIX_36.csv", rows, list(rows[0].keys()))
    atomic_write_json(BENCH_ROOT / "protocol/D3_SEED_REGISTRY.json", {"status": "LOCKED", "seeds": seeds, "excluded_observed_seeds": sorted(excluded), "rule": "smallest three unobserved integers >=20261001", "final_test_access": 0})
    atomic_write_json(BENCH_ROOT / "protocol/D3_RESOURCE_ESTIMATE.json", {"status": "PRE_STAGE0_ESTIMATE", "gpu_hour_ceiling": 96.0, "disk_gib_ceiling": 100.0, "grid_cache_gib_estimate": 3.94, "stage0_peak_mib_registered": 12288, "training_runs": 36, "final_test_access": 0})
    print(json.dumps({"status": "LOCKED", "seeds": seeds, "matrix": len(rows), "protocol_hash": protocol["protocol_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
