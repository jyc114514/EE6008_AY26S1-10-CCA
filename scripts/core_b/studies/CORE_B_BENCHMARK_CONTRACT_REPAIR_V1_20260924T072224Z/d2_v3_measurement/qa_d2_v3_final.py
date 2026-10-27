"""Final D2_V3 event, clustering and acceptance QA with locked denominators."""

from __future__ import annotations
import os

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_d2_v3_panel import cluster_alerts, state_trace


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    matcher = json.loads((ROOT / "D2_V3_MATCHER_LOCKED_ATTEMPT_003.json").read_text())
    summary = json.loads((ROOT / "D2_V3_EVALUATION_SUMMARY.json").read_text())
    results = read_csv(ROOT / "D2_V3_RESULTS.csv")
    audit = read_csv(ROOT / "D2_V3_FORECAST_CLUSTER_MATCH_AUDIT.csv")
    if matcher.get("transform_changes_estimand_or_match_rule") is not False:
        raise AssertionError("matcher code repair changed the locked estimand")
    if summary.get("accepted_dense_predictions") != 36 or len(results) != 72:
        raise AssertionError("D2_V3 matrix denominator must be 36 cells × 2 bodies = 72 rows")
    if any(row["final_test_access"] != "0" for row in results):
        raise AssertionError("final-test access marker not zero in all result rows")

    # Boundary fixture: six quiet moving windows arm; three highs emit onset at
    # first high, confirmed exactly two raw steps later.
    toy_signal = np.asarray([0.0] * 6 + [2.0] * 3)
    state = state_trace(toy_signal, first_endpoint_frame=8, frame_count=17, on=1.0, off=0.1)
    assert len(state["events"]) == 1
    assert state["events"][0]["onset_frame"] == 14
    assert state["events"][0]["confirmation_frame"] == 16
    unknown = state_trace(np.asarray([2.0] * 9), first_endpoint_frame=8, frame_count=17, on=1.0, off=0.1)
    assert not bool(unknown["known"][-1]) and not unknown["events"]

    # Cluster rule is fixed-center, not transitive chaining.
    alerts = [
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q0", "query_time": 0.0, "predicted_onset_time": 1.0},
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q1", "query_time": 0.1, "predicted_onset_time": 1.15},
        {"source_group_id": "g", "body": "Hand", "episode_index": 1, "task": 0, "dense_id": "q2", "query_time": 0.2, "predicted_onset_time": 1.25},
    ]
    clusters = cluster_alerts(alerts)
    assert len(clusters) == 2
    assert clusters[0]["center_predicted_onset"] == 1.0 and clusters[0]["report_count"] == 2
    assert clusters[1]["center_predicted_onset"] == 1.25 and clusters[1]["report_count"] == 1

    keys = Counter((row["model"], row["seed"], row["horizon_steps"], row["event_group"]) for row in results)
    assert len(keys) == 72 and all(count == 1 for count in keys.values())
    grouped = {}
    for row in audit:
        cell_group = (row["model"], row["seed"], row["horizon_steps"], row["body"], row["source_group_id"])
        grouped.setdefault(cell_group, []).append(row)
        if row["match_status"] in {"hit", "false_alarm"}:
            assert int(row["report_count"]) >= 1
        if row["match_status"] == "hit":
            assert float(row["lead_seconds"]) > 0
            assert abs(float(row["timing_signed_error_seconds"])) <= 0.2 + 1e-8
            assert float(row["lead_seconds"]) <= (int(row["horizon_steps"]) - 2) / 30.0 + 2e-5
        if row["match_status"] == "missed_gt":
            assert int(row["report_count"]) == 0
    for cell_group, entries in grouped.items():
        hit_ids = [entry["matched_event_id"] for entry in entries if entry["match_status"] == "hit"]
        assert len(hit_ids) == len(set(hit_ids)), f"one GT event matched more than once: {cell_group}"
        cluster_ids = [entry["cluster_id"] for entry in entries if entry["cluster_id"]]
        assert len(cluster_ids) == len(set(cluster_ids)), f"duplicate forecast cluster id: {cell_group}"

    for row in results:
        cell_rows = [entry for entry in audit if (entry["model"], entry["seed"], entry["horizon_steps"], entry["body"]) == (row["model"], row["seed"], row["horizon_steps"], row["event_group"])]
        assert int(row["hits"]) == sum(entry["match_status"] == "hit" for entry in cell_rows)
        assert int(row["false_alarms"]) == sum(entry["match_status"] == "false_alarm" for entry in cell_rows)
        assert int(row["misses"]) == sum(entry["match_status"] == "missed_gt" for entry in cell_rows)
        assert float(row["exposure_hours"]) > 0
        if row["recall_ci95_low"] and row["recall_ci95_high"]:
            assert float(row["recall_ci95_low"]) <= float(row["recall_ci95_high"])

    hand = [row for row in results if row["event_group"] == "Hand"]
    assert all(int(row["full_opportunity_events"]) >= 30 and int(row["tasks_with_complete_opportunity"]) >= 10 for row in hand)
    qa = {
        "status": "PASS",
        "prediction_cells": 36,
        "result_rows": len(results),
        "cluster_audit_rows": len(audit),
        "state_machine_onset_backdating_fixture": "PASS",
        "unknown_state_fixture": "PASS",
        "fixed_center_cluster_nontransitivity_fixture": "PASS",
        "one_to_one_gt_matching_invariants": "PASS for every cell/body/source group",
        "effective_horizon_and_timing_constraints": "PASS",
        "metrics_reconcile_with_cluster_audit": "PASS",
        "hand_support_gate": "PASS; every model/horizon has >=30 events and >=10 tasks",
        "final_test_access": 0,
    }
    (ROOT / "D2_V3_EVENT_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, sort_keys=True))


if __name__ == "__main__":
    main()
