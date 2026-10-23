"""Small deterministic tests for D2/D4 event-state semantics."""

from __future__ import annotations

import numpy as np

from event_v2 import GROUPS, blank_state, cluster_alarms, max_cardinality_match, replay_q


def test_six_quiet_then_three_high_emits_group_specific_onset() -> None:
    q = np.zeros((9, 2), dtype=np.float64)
    q[6:, 0] = 0.2
    timestamps = np.arange(20, dtype=np.float64) / 30.0
    events, state, _ = replay_q(q, np.arange(1, 10), timestamps, np.asarray([[0.1, 0.01], [0.1, 0.01]]))
    assert len(events) == 1
    assert events[0]["group"] == GROUPS[0]
    assert events[0]["onset_frame"] == 7
    assert events[0]["confirm_frame"] == 9
    assert state[0]["known"] is True and state[0]["armed"] is False
    assert state[1]["known"] is True and state[1]["armed"] is True


def test_middle_band_breaks_both_runs_but_preserves_armed_state() -> None:
    q = np.zeros((13, 2), dtype=np.float64)
    q[6:8, 0] = 0.2
    q[8, 0] = 0.05  # middle band resets the high streak but does not disarm
    q[9:12, 0] = 0.2
    ts = np.arange(20, dtype=np.float64) / 30.0
    events, state, _ = replay_q(q, np.arange(1, 14), ts, np.asarray([[0.1, 0.01], [0.1, 0.01]]))
    assert len(events) == 1 and events[0]["onset_frame"] == 10
    assert state[0]["known"] is True and state[0]["armed"] is False


def test_unknown_state_is_not_implicitly_armed() -> None:
    q = np.full((4, 2), 0.2, dtype=np.float64)
    ts = np.arange(8, dtype=np.float64) / 30.0
    events, state, _ = replay_q(q, np.arange(1, 5), ts, np.asarray([[0.1, 0.01], [0.1, 0.01]]))
    assert events == []
    assert all(x["known"] is False and x["armed"] is False for x in state)
    assert blank_state()[0]["known"] is False


def test_global_matching_maximizes_hits_then_minimizes_error() -> None:
    predicted = [{"onset_timestamp": 0.00}, {"onset_timestamp": 0.19}]
    truth = [{"onset_timestamp": 0.10}, {"onset_timestamp": -0.19}]
    matches = max_cardinality_match(predicted, truth, 0.2)
    assert len(matches) == 2


def test_dense_alarm_cluster_uses_first_onset_and_keeps_queries() -> None:
    alarms = [
        {"onset_timestamp": 1.02, "query_timestamp": 0.3},
        {"onset_timestamp": 1.08, "query_timestamp": 0.5},
        {"onset_timestamp": 1.40, "query_timestamp": 0.9},
    ]
    clusters = cluster_alarms(alarms, 0.2)
    assert len(clusters) == 2
    assert clusters[0]["onset_timestamp"] == 1.02
    assert len(clusters[0]["alarms"]) == 2
    assert clusters[0]["query_timestamp"] == 0.3
