"""Smoke tests for D2 event reanalysis query-state and denominator contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import reanalyze_d2_v2 as d2
from event_v2 import blank_state


def test_cached_forecast_returns_query_state_not_postforecast_state() -> None:
    action = np.zeros((32, 28), dtype=np.float64)
    timestamps = np.arange(32, dtype=np.float64) / 30.0
    pred = np.ones((6, 28), dtype=np.float64)
    initial = blank_state()
    for state in initial:
        state.update({"known": True, "armed": True, "quiet_run": 0, "high_run": 0})
    thresholds = np.asarray([[0.01, 0.0], [0.01, 0.0]])
    _, qstate = d2.cached_forecast(action, timestamps, 20, pred, thresholds, initial)
    assert all(item["known"] and item["armed"] for item in qstate)


def test_one_real_dense_episode_has_consistent_group_sums() -> None:
    root = d2.BENCH_ROOT
    panel = d2.read_csv(root / "d2/D2_DENSE_PANEL.csv")
    episode = int(panel[0]["episode_index"])
    rows = [row for row in panel if int(row["episode_index"]) == episode]
    rows = [{**row, "dense_id": f"dense_e{episode:06d}_a{int(row['anchor_frame']):06d}"} for row in rows]
    anchors = d2.read_csv(d2.V3_ROOT / "ANCHOR_MANIFEST.csv")
    meta = next(row for row in anchors if int(row["episode_index"]) == episode and row["split"] == "dev" and row["population"] == "primary")
    timestamps, action = d2.load_episode(meta)
    threshold_doc = json.loads((d2.NEXT_ROOT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
    thresholds = np.column_stack((threshold_doc["theta_on"], threshold_doc["theta_off"]))
    episode_data = {episode: (timestamps, action)}
    state_cache = d2.make_query_state_cache(rows, episode_data, thresholds)
    truth = {episode: d2.full_truth_events(action, timestamps, d2.SIGMA, thresholds)}
    score = d2.dense_episode_score(episode, rows, {}, "PERSISTENCE", 30, episode_data, truth, state_cache, thresholds)
    assert score["exposure_seconds"] > 0
    assert score["All"]["opp_full"] == score["Hand"]["opp_full"] + score["Arm"]["opp_full"]
    assert score["All"]["false_alarms"] == score["Hand"]["false_alarms"] + score["Arm"]["false_alarms"]
    assert all(score[group]["hits_full"] <= score[group]["opp_full"] for group in ("Hand", "Arm", "All"))
