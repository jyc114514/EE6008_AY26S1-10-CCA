"""Cross-variant, validation-only preflight for the batch-32 rescue study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_batch32_rescue_v3 import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_POPULATION,
    EXPECTED_TRAIN,
    EXPECTED_VALIDATION,
    atomic_write_json,
    attach_feature_manifest,
    load_experiment,
    load_v3_population,
    preflight,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-config", type=Path, required=True)
    parser.add_argument("--static-config", type=Path, required=True)
    parser.add_argument("--static-repeat-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    dynamic_experiment = load_experiment(args.dynamic_config)
    static_experiment = load_experiment(args.static_config)
    if dynamic_experiment.variant != "dynamic" or static_experiment.variant != "static_repeated_first_frame":
        raise ValueError("dynamic/static config variants are not paired")
    if dynamic_experiment.physical_gpu != 2 or static_experiment.physical_gpu != 3:
        raise ValueError("batch32 study is restricted to physical GPU2/GPU3")
    dynamic_population = load_v3_population(dynamic_experiment.core, full=True)
    static_population = load_v3_population(static_experiment.core, full=True)
    if dynamic_population.population_sha256 != static_population.population_sha256:
        raise ValueError("dynamic/static population hashes differ")
    dynamic_rows, dynamic_manifest_hash = attach_feature_manifest(
        dynamic_population.rows, dynamic_experiment.feature_run
    )
    static_rows, static_manifest_hash = attach_feature_manifest(
        static_population.rows, static_experiment.feature_run
    )
    for column in ("episode_index", "split", "local_label_index"):
        if not np.array_equal(dynamic_rows[column].to_numpy(), static_rows[column].to_numpy()):
            raise ValueError(f"dynamic/static {column} ordering differs")
    repeat_audit = json.loads(args.static_repeat_audit.read_text(encoding="utf-8"))
    raw_repeat = repeat_audit.get("raw_repeat_check", {})
    if raw_repeat.get("all_samples_pass") is not True or raw_repeat.get("sample_count") != 10:
        raise ValueError("historical static repeated-frame raw audit is not 10/10 pass")
    if raw_repeat.get("unique_raw_first_frame_hashes") != 10:
        raise ValueError("historical static raw audit does not have 10 distinct first-frame hashes")
    dynamic_preflight = preflight(dynamic_experiment)
    static_preflight = preflight(static_experiment)
    if dynamic_preflight.get("status") != "pass" or static_preflight.get("status") != "pass":
        raise ValueError("variant preflight failed")
    report: dict[str, Any] = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "scope": "CPU input preflight for batch32 dynamic/static; no final-test labels",
        "population": {
            "rows": EXPECTED_POPULATION,
            "train": EXPECTED_TRAIN,
            "development_validation": EXPECTED_VALIDATION,
            "classes": EXPECTED_CLASS_COUNT,
            "population_sha256": dynamic_population.population_sha256,
            "split_sha256": dynamic_population.split_sha256,
            "label_map_sha256": dynamic_population.label_map_sha256,
            "dense_label_values": list(range(EXPECTED_CLASS_COUNT)),
            "final_test_selected": 0,
        },
        "dynamic": dynamic_preflight,
        "static": static_preflight,
        "dynamic_static_alignment": {
            "episode_index": True,
            "split": True,
            "local_label_index": True,
            "population_order": True,
            "dynamic_manifest_sha256": dynamic_manifest_hash,
            "static_manifest_sha256": static_manifest_hash,
            "feature_rows": len(dynamic_rows),
        },
        "static_repeat_frame_audit": {
            "path": str(args.static_repeat_audit.resolve()),
            "sha256": sha256_file(args.static_repeat_audit.resolve()),
            "sample_count": 10,
            "all_samples_pass": True,
            "unique_raw_first_frame_hashes": 10,
        },
        "guards": {
            "final_test_selected": 0,
            "final_test_access": "none",
            "backbone_trainable_parameter_count": 0,
            "physical_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "old_batch4_results_modified": False,
            "new_cache_extraction": False,
            "new_download": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
