"""Guarded inference-job builder requiring 36/36 training and epoch-coverage QA."""

from __future__ import annotations

import json
from pathlib import Path

import build_d3_eq_inference_jobs as builder


EQ_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    train = json.loads((EQ_ROOT / "qa/D3_EQ_TRAINING_ACCEPTANCE.json").read_text(encoding="utf-8"))
    epoch = json.loads((EQ_ROOT / "qa/D3_EQ_EPOCH_COVERAGE_QA.json").read_text(encoding="utf-8"))
    if train.get("status") != "PASS" or train.get("training_cells") != 36:
        raise PermissionError("all 36 accepted D3-EQ training runs are required before inference")
    if epoch.get("status") != "PASS" or epoch.get("accepted_runs") != 36:
        raise PermissionError("full 40-epoch/unique-train-window coverage QA must pass before inference")
    original = builder.job_record

    def run_isolated_job(config, horizon, seed, *, stage0: bool, peak_mib: int):
        record = original(config, horizon, seed, stage0=stage0, peak_mib=peak_mib)
        # A single study-isolated output directory contains uniquely named
        # bundles keyed by run_id; it is separate from the historical D3 root.
        record["job"]["output_root"] = record["output_root"]
        return record

    builder.job_record = run_isolated_job
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
