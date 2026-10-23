"""Lock D2_V3 dev matcher and fixed input panel before dense predictions load."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement"
EVALUATOR = OUT / "evaluate_d2_v3_panel.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not EVALUATOR.is_file():
        raise FileNotFoundError("write the final evaluator before locking its code hash")
    with np.load(BENCH / "d2/D2_DENSE_INPUTS.npz", allow_pickle=False) as archive:
        panel_ids = archive["dense_id"].astype("U96")
        episodes = archive["episode_index"].astype(np.int64)
        tasks = archive["original_task_index"].astype(np.int64)
        frames = archive["anchor_frame"].astype(np.int64)
        timestamps = archive["anchor_timestamp"].astype(np.float64)
    if len(panel_ids) != 3718 or len(set(panel_ids.tolist())) != 3718:
        raise AssertionError("locked dense panel IDs are not 3718 unique records")
    if len(set(episodes.tolist())) != 118:
        raise AssertionError("locked dense panel no longer has 118 episodes")

    acceptance = []
    with (RESULT / "d4/D4_BUNDLE_ACCEPTANCE.csv").open(newline="", encoding="utf-8") as handle:
        acceptance = list(csv.DictReader(handle))
    if len(acceptance) != 36 or any(row["status"] != "PASS_ID_frame_timestamp_shape_finite" or row["final_test_access"] != "0" for row in acceptance):
        raise AssertionError("fixed D4 V3 dense bundle acceptance is not 36/36")

    threshold_path = OUT / "D2_V3_PROTOCOL_LOCKED.json"
    threshold = json.loads(threshold_path.read_text())
    if threshold.get("status") != "locked_train_thresholds_ready" or threshold.get("dev_accessed") is not False:
        raise AssertionError("train-only D2_V3 threshold lock is not ready")

    lock = {
        "protocol_id": "CORE_B_D2_EVENT_MATCHER_V3",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_BEFORE_D2_V3_DEV_PREDICTION_READ",
        "threshold_protocol_path": str(threshold_path),
        "threshold_protocol_sha256": sha256_file(threshold_path),
        "evaluator_code_path": str(EVALUATOR),
        "evaluator_code_sha256": sha256_file(EVALUATOR),
        "fixed_panel": {
            "input_path": str(BENCH / "d2/D2_DENSE_INPUTS.npz"),
            "input_sha256": sha256_file(BENCH / "d2/D2_DENSE_INPUTS.npz"),
            "panel_csv_sha256": sha256_file(BENCH / "d2/D2_DENSE_PANEL.csv"),
            "coverage_csv_sha256": sha256_file(RESULT / "results/D2_DENSE_PANEL_TASK_COVERAGE.csv"),
            "queries": len(panel_ids),
            "episodes": len(set(episodes.tolist())),
            "unique_tasks": len(set(tasks.tolist())),
            "unique_dense_ids": len(set(panel_ids.tolist())),
            "prediction_bundles_expected": len(acceptance),
            "bundle_acceptance_sha256": sha256_file(RESULT / "d4/D4_BUNDLE_ACCEPTANCE.csv"),
            "prediction_outcomes_read_before_lock": False,
        },
        "group_order": ["Hand", "Arm"],
        "lead_thresholds_seconds": [0.25, 0.5, 1.0],
        "effective_horizon": "For H nominal forecast steps, an onset is scoreable only if its 3-step confirmation can finish; effective endpoint is the timestamp at H-2 raw steps after query.",
        "query_state": "Compute q/m and state from the legal raw prefix through query. Unknown state queries are excluded; known-but-unarmed queries continue state transitions through predictions and may re-arm after a predicted quiet run.",
        "forecast_event": "Append predicted command trajectory to true legal prefix; compute the same 8-step moving signal; first of 3 consecutive m>theta_on is forecast onset; confirmation is 2 steps later; only onset after query and within effective horizon is eligible.",
        "cluster_rule": "Within source_group/body/model/horizon, process query alerts sorted by query_time, predicted_onset_time, dense_id. Merge into the earliest cluster whose fixed first predicted onset is within 0.2s; keep first query and onset as immutable center; later alerts do not move center.",
        "matching_rule": "Order clusters by first query time, fixed predicted onset, then dense_id. For each cluster, match one unused independent GT event satisfying query_time < onset <= query_time+effective_horizon and abs(predicted_onset-onset)<=0.2s; choose minimum timing difference, then earliest GT timestamp, then stable event ID. Each GT event is used once; unmatched clusters are false alarms.",
        "opportunity_rule": "A complete GT event is in the horizon denominator if at least one state-known dense query precedes it within effective horizon and the raw timeline contains the full nominal forecast; lead-specific denominator additionally requires a qualifying query with that actual lead.",
        "exposure_rule": "Per source-group panel block, last legal query timestamp minus first legal query timestamp; overlapping forecasts do not add exposure.",
        "bootstrap": {"unit": "exact video_sha256_fingerprint source group", "replicates": 10000, "seed": 20260924, "interval": "95th percentile bootstrap interval", "fa_rate": "false clusters / summed actual exposure hours per replicate"},
        "support_gate": {"Hand_min_independent_complete_events_per_horizon": 30, "Hand_min_tasks": 10, "below_gate_label": "low_support; report counts and intervals without efficacy wording"},
        "panel_selection": "Reuse the pre-existing fixed 118-source-group/3718-query task-balanced development panel. No outcome-based expansion or episode replacement.",
        "interpretation": "D2_V3 is an exploratory measurement-definition sensitivity on previously exposed development outcomes; it does not replace D2_V2 or create a candidate gate.",
        "final_test_access": 0,
    }
    lock_path = OUT / "D2_V3_MATCHER_LOCKED.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "panel_queries": len(panel_ids), "panel_episodes": len(set(episodes.tolist())), "bundles": len(acceptance), "threshold_hash": lock["threshold_protocol_sha256"], "evaluator_code_hash": lock["evaluator_code_sha256"], "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
