"""Write D4/D5 preparation artifacts and real-input coverage metadata."""

from __future__ import annotations

import json
from pathlib import Path

from common import BENCH_ROOT, atomic_write_json, read_csv, utc_now, write_csv


def main() -> int:
    dense = read_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    d4_rows = []
    for row in dense:
        d4_rows.append({
            "episode_index": row["episode_index"], "task": row["task"], "anchor_frame": row["anchor_frame"],
            "anchor_timestamp": row["anchor_timestamp"], "old_forecast_support": "unknown_no_sealed_old_bundle",
            "new_forecast_support": "unknown_no_sealed_new_bundle", "target_intersection": "metadata_only_target_1s_and_2s_supported",
            "fixed_history_window_seconds": 4.0, "stale_fresh_video_control": "required_future_execution",
            "latency_measurement": "required_future_execution", "status": "coverage_audit_only", "final_test_access": 0,
        })
    write_csv(BENCH_ROOT / "prepared/d4/D4_ANCHOR_COVERAGE.csv", d4_rows, list(d4_rows[0].keys()) if d4_rows else ["episode_index"])
    d4_interface = {
        "protocol_id": "CORE_B_D4_ROLLING_UPDATE_V1", "status": "prepared_not_executed", "input": "old/new deterministic forecasts with absolute timestamps and source-native target trajectory", "rules": {"history": "fixed 4s sliding window; append new observations and drop oldest", "alignment": "intersect old/new forecast support with identical future target timestamps", "utility": "old_error - new_error", "stability": "report mean absolute forecast revision separately", "controls": ["stale_vs_fresh_video", "latency", "missing_frame", "timestamp_gap"], "statistics": "episode/source-group paired outcomes; boundary and empty intersections are QA failures/NA"}, "disabled": "no real D4 utility/effect table, perturbation, or rolling-model experiment in this window", "final_test_access": 0,
    }
    atomic_write_json(BENCH_ROOT / "prepared/d4/D4_INTERFACE.json", d4_interface)
    (BENCH_ROOT / "prepared/d4/D4_ROLLING_PROTOCOL.md").write_text("""# D4 rolling prediction protocol\n\nStatus: `prepared_not_executed`.\n\nD4 will compare old/new forecasts only on the intersection of absolute future timestamps and the same target points. The observation history remains a fixed 4-second sliding window; newly observed frames enter while the oldest frames leave. A shorter intersection is reported as coverage, not as an information gain.\n\nPrimary utility is `old_error - new_error`; raw forecast revision/stability is a separate outcome and does not reward an unchanged forecast. Required controls are stale/fresh video, delay, missing frames, timestamp gaps, and actual inference cost. Any asynchronous state/video update is an out-of-distribution/synchrony comparison, not a causal intervention.\n\nThe current dense panel coverage is metadata-only in `D4_ANCHOR_COVERAGE.csv`; no real forecast utility was calculated because V3 prediction bundles are not sealed. Synthetic interface tests are in `test_d4_synthetic.py`.\n""", encoding="utf-8")

    model_rows = read_csv(BENCH_ROOT / "inventory/MODEL_REGISTRY.csv")
    d5_rows = []
    for row in model_rows:
        family = row["family"]
        output_type = "deterministic_trajectory" if family not in {"BASELINES"} else "deterministic_baseline_trajectory"
        if "PROTOTYPE" in family or "MIXTURE" in family:
            output_type = "unknown_requires_source_audit"
        d5_rows.append({"model_id": row["model_id"], "family": family, "seed": row["seed"], "output_type": output_type, "conditional_samples": "not_established", "mixture_weights": "not_established", "multiple_seeds_are_conditional_samples": "false", "calibration_status": "not_fitted", "final_test_access": 0})
    write_csv(BENCH_ROOT / "prepared/d5/D5_MODEL_CAPABILITY.csv", d5_rows, list(d5_rows[0].keys()) if d5_rows else ["model_id"])
    d5_interface = {
        "protocol_id": "CORE_B_D5_PROBABILISTIC_V1", "status": "prepared_not_executed", "sample_shape": "(N,K,H,28)", "required_fields": ["samples", "sample_weights", "seed", "draw_count", "absolute_timestamps", "target_schema", "train_only_scaling"], "scores": {"energy_score": "E||X-y|| - 1/2 E||X-X'|| with finite K rule fixed before results", "fd_distribution": "score future internal differences separately", "coverage": ["marginal", "per_time", "simultaneous_path"], "calibration": ["event probability", "risk coverage", "credible horizon"]}, "constraints": ["deterministic trajectories and multiple seeds are not conditional multimodality", "no in-sample residual called independent calibration", "best-of-K/minADE secondary only"], "disabled": "no real probabilistic training, calibration fitting, or leaderboard in this window", "final_test_access": 0,
    }
    atomic_write_json(BENCH_ROOT / "prepared/d5/D5_INTERFACE.json", d5_interface)
    (BENCH_ROOT / "prepared/d5/D5_PROBABILISTIC_PROTOCOL.md").write_text("""# D5 probabilistic prediction protocol\n\nStatus: `prepared_not_executed`.\n\nThe future interface is `samples=(N,K,H,28)` with fixed draw count, optional sample weights, seed, timestamps, and train-only scaling. Separate deterministic trajectories, mixture/sample outputs, and multiple training seeds; seed variation is not conditional multimodality.\n\nPrimary proper score is a finite-sample Energy Score with an explicit K rule. Report time-dependent/FD distribution scores, marginal/per-time/simultaneous path coverage, width, event-probability calibration, risk-coverage, and credible horizon. `minADE`/best-of-K is secondary and cannot validate a full conditional distribution from one ground-truth trajectory.\n\nTraining/calibration/evaluation source-group splits must be independent or cross-fitted; existing train-seen residuals are not an independent calibration set. Current capability and interface files are preparation only. Synthetic formula tests are in `test_d5_synthetic.py`.\n""", encoding="utf-8")
    atomic_write_json(BENCH_ROOT / "prepared/PREPARED_STATUS.json", {"generated_utc": utc_now(), "D4": "prepared_not_executed", "D5": "prepared_not_executed", "d4_coverage_rows": len(d4_rows), "d5_capability_rows": len(d5_rows), "final_test_access": 0})
    print(json.dumps({"D4": "prepared_not_executed", "D5": "prepared_not_executed", "D4_rows": len(d4_rows), "D5_rows": len(d5_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
