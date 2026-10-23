"""Independent output invariants and deterministic boundary/cluster fixtures."""

from __future__ import annotations
import os

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_d2_v3_panel import cluster_alerts, state_trace


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads((ROOT / "D2_V3_MATCHER_LOCKED_ATTEMPT_003.json").read_text())
    summary = json.loads((ROOT / "D2_V3_EVALUATION_SUMMARY.json").read_text())
    results = read_csv(ROOT / "D2_V3_RESULTS.csv")
    audit = read_csv(ROOT / "D2_V3_FORECAST_CLUSTER_MATCH_AUDIT.csv")
    assert protocol["support_gate"]["Hand_min_independent_complete_events_per_horizon"] == 30
    assert summary["accepted_dense_predictions"] == 36
    assert len(results) == 144
    assert all(row["final_test_access"] == "0" for row in results)

    # State machine emits the first of three highs after a completed six-quiet
    # run, with confirmation two raw steps later.
    signal = np.asarray([0.0] * 6 + [2.0] * 3, dtype=np.float64)
    trace = state_trace(signal, first_endpoint_frame=8, frame_count=17, on=1.0, off=0.1)
    assert len(trace["events"]) == 1
    assert trace["events"][0]["onset_frame"] == 14
    assert trace["events"][0]["confirmation_frame"] == 16
    unknown_trace = state_trace(np.asarray([2.0] * 9), first_endpoint_frame=8, frame_count=17, on=1.0, off=0.1)
    assert not bool(unknown_trace["known"][-1]) and not unknown_trace["events"]

    # The clustering center is immutable and grouping is not transitive.
    alerts = [
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q0", "query_time": 0.0, "predicted_onset_time": 1.00},
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q1", "query_time": 0.1, "predicted_onset_time": 1.15},
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q2", "query_time": 0.2, "predicted_onset_time": 1.25},
    ]
    clusters = cluster_alerts(alerts)
    assert len(clusters) == 2
    assert clusters[0]["center_predicted_onset"] == 1.0 and clusters[0]["report_count"] == 2
    assert clusters[1]["center_predicted_onset"] == 1.25 and clusters[1]["report_count"] == 1

    by_cell_group = defaultdict(list)
    for row in audit:
        key = (row["model"], int(row["seed"]), int(row["horizon_steps"]), row["body"], row["source_group_id"])
        by_cell_group[key].append(row)
        if row["match_status"] in {"hit", "false_alarm"}:
            assert int(row["report_count"]) >= 1
        if row["match_status"] == "hit":
            assert float(row["lead_seconds"]) > 0
            assert abs(float(row["timing_signed_error_seconds"])) <= 0.2 + 1e-8
            max_effective_h = (int(row["horizon_steps"]) - 2) / 30.0 + 2e-5
            assert float(row["lead_seconds"]) <= max_effective_h
        if row["match_status"] == "missed_gt":
            assert int(row["report_count"]) == 0
    for key, rows in by_cell_group.items():
        hits = [row["matched_event_id"] for row in rows if row["match_status"] == "hit"]
        assert len(hits) == len(set(hits)), f"one-to-one GT matching failed for {key}"
        clusters_found = [row["cluster_id"] for row in rows if row["cluster_id"]]
        assert len(clusters_found) == len(set(clusters_found)), f"cluster IDs duplicated for {key}"

    result_map = {(r["model"], int(r["seed"]), int(r["horizon_steps"]), r["event_group"]): r for r in results}
    for key, row in result_map.items():
        cell_audit = [
            record for record in audit
            if (record["model"], int(record["seed"]), int(record["horizon_steps"]), record["body"]) == key
        ]
        assert int(row["hits"]) == sum(record["match_status"] == "hit" for record in cell_audit)
        assert int(row["false_alarms"]) == sum(record["match_status"] == "false_alarm" for record in cell_audit)
        assert int(row["misses"]) == sum(record["match_status"] == "missed_gt" for record in cell_audit)
        assert float(row["exposure_hours"]) > 0
        assert float(row["recall_ci95_low"]) <= float(row["recall_ci95_high"])

    hand = [row for row in results if row["event_group"] == "Hand"]
    assert all(int(row["full_opportunity_events"]) >= 30 and int(row["tasks_with_complete_opportunity"]) >= 10 for row in hand)
    qa = {
        "status": "PASS",
        "dense_prediction_cells": 36,
        "result_rows": len(results),
        "cluster_audit_rows": len(audit),
        "state_machine_boundary_fixture": "PASS: onset backdated to first high; confirmation after third high",
        "unknown_state_fixture": "PASS",
        "cluster_nontransitivity_fixture": "PASS: 1.00 and 1.15 merge; 1.25 starts new cluster against fixed 1.00 center",
        "one_to_one_ground_truth_matches": "PASS for every model/seed/horizon/body/source group",
        "lead_time_and_timing_constraints": "PASS",
        "result_counts_reconcile_to_cluster_audit": "PASS",
        "hand_support_gate": "PASS for every horizon/model: >=30 events and >=10 tasks",
        "final_test_access": 0,
    }
    (ROOT / "D2_V3_EVENT_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, sort_keys=True))


if __name__ == "__main__":
    main()
