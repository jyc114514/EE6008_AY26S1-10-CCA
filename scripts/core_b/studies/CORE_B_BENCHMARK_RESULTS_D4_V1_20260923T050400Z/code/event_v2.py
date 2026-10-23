"""Group-aware threshold event state machine for the D2/D4 v2 reanalysis."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

GROUPS = ("Hand", "Arm")


def q_signal(action: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    delta = np.abs(np.diff(np.asarray(action, dtype=np.float64), axis=0)) / np.asarray(sigma, dtype=np.float64)[None, :]
    return np.stack((delta[:, :14].mean(axis=1), delta[:, 14:].mean(axis=1)), axis=1)


def blank_state() -> list[dict[str, object]]:
    return [{"known": False, "armed": False, "quiet_run": 0, "high_run": 0} for _ in GROUPS]


def replay_q(
    q_values: np.ndarray,
    end_frames: np.ndarray,
    timestamps: np.ndarray,
    thresholds: np.ndarray,
    initial: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Replay Hand/Arm q sequences; middle-band values break streaks, not state."""
    states = deepcopy(initial) if initial is not None else blank_state()
    events: list[dict[str, object]] = []
    for row, frame in enumerate(np.asarray(end_frames, dtype=np.int64)):
        for group_index, group_name in enumerate(GROUPS):
            value = float(q_values[row, group_index])
            on, off = map(float, thresholds[group_index])
            state = states[group_index]
            if value <= off:
                state["quiet_run"] = int(state["quiet_run"]) + 1
                state["high_run"] = 0
                if int(state["quiet_run"]) >= 6:
                    state["armed"] = True
                    state["known"] = True
            elif value > on:
                state["quiet_run"] = 0
                if bool(state["armed"]):
                    state["high_run"] = int(state["high_run"]) + 1
                    if int(state["high_run"]) >= 3:
                        onset_frame = int(frame) - 2
                        events.append({"group": group_name, "onset_frame": onset_frame, "confirm_frame": int(frame), "onset_timestamp": float(timestamps[onset_frame]), "confirm_timestamp": float(timestamps[int(frame)]), "signal_at_onset": float(q_values[row - 2, group_index])})
                        state["armed"] = False
                        state["high_run"] = 0
                else:
                    state["high_run"] = 0
            else:
                state["quiet_run"] = 0
                state["high_run"] = 0
    events.sort(key=lambda item: (float(item["onset_timestamp"]), str(item["group"])))
    return events, states, []


def observed_state(
    action: np.ndarray,
    timestamps: np.ndarray,
    start_frame: int,
    anchor_frame: int,
    sigma: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if start_frame < 0 or anchor_frame < start_frame or anchor_frame >= len(action):
        raise ValueError("invalid observed prefix frame range")
    if anchor_frame == start_frame:
        return blank_state(), []
    local_action = action[start_frame : anchor_frame + 1]
    q = q_signal(local_action, sigma)
    endpoints = np.arange(start_frame + 1, anchor_frame + 1, dtype=np.int64)
    events, state, _ = replay_q(q, endpoints, timestamps, thresholds)
    return state, events


def full_truth_events(action: np.ndarray, timestamps: np.ndarray, sigma: np.ndarray, thresholds: np.ndarray) -> list[dict[str, object]]:
    q = q_signal(action, sigma)
    endpoints = np.arange(1, len(action), dtype=np.int64)
    events, _, _ = replay_q(q, endpoints, timestamps, thresholds)
    return events


def forecast_events(
    action: np.ndarray,
    timestamps: np.ndarray,
    anchor_frame: int,
    prediction: np.ndarray,
    sigma: np.ndarray,
    thresholds: np.ndarray,
    prefix_frames: int = 120,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    horizon = len(prediction)
    if anchor_frame + horizon >= len(action):
        raise ValueError("ground-truth timestamp support is shorter than forecast horizon")
    start = max(0, anchor_frame - prefix_frames + 1)
    state, _ = observed_state(action, timestamps, start, anchor_frame, sigma, thresholds)
    state_at_query = deepcopy(state)
    combined = np.concatenate((action[anchor_frame : anchor_frame + 1], np.asarray(prediction, dtype=np.float64)), axis=0)
    q = q_signal(combined, sigma)
    endpoints = np.arange(anchor_frame + 1, anchor_frame + horizon + 1, dtype=np.int64)
    events, _, _ = replay_q(q, endpoints, timestamps, thresholds, initial=state)
    for event in events:
        event["query_frame"] = int(anchor_frame)
        event["query_timestamp"] = float(timestamps[anchor_frame])
    return events, state_at_query


def max_cardinality_match(
    predicted: list[dict[str, object]],
    truth: list[dict[str, object]],
    tolerance: float,
    eligible=None,
) -> list[tuple[int, int, float]]:
    """Maximum hit count, then minimum absolute timing error, deterministic ties."""
    if not predicted or not truth:
        return []
    from scipy.optimize import linear_sum_assignment

    n_pred, n_truth = len(predicted), len(truth)
    dummy_cost = tolerance * (n_pred + 1) + 1.0
    invalid_cost = dummy_cost * (n_pred + 2)
    cost = np.full((n_pred, n_truth + n_pred), dummy_cost, dtype=np.float64)
    cost[:, :n_truth] = invalid_cost
    valid = np.zeros((n_pred, n_truth), dtype=bool)
    for i, pred in enumerate(predicted):
        for j, target in enumerate(truth):
            error = abs(float(pred["onset_timestamp"]) - float(target["onset_timestamp"]))
            allowed = error <= tolerance and (eligible is None or eligible(i, j, pred, target))
            if allowed:
                # Tiny row/column term makes exact ties deterministic without changing error ordering.
                cost[i, j] = error + (i * n_truth + j) * 1e-12
                valid[i, j] = True
    rows, cols = linear_sum_assignment(cost)
    result = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if j < n_truth and valid[i, j]:
            result.append((i, j, float(predicted[i]["onset_timestamp"]) - float(truth[j]["onset_timestamp"])))
    return sorted(result, key=lambda item: (item[0], item[1]))


def cluster_alarms(alarms: list[dict[str, object]], tolerance: float = 0.2) -> list[dict[str, object]]:
    ordered = sorted(alarms, key=lambda item: (float(item["onset_timestamp"]), float(item["query_timestamp"])))
    clusters: list[dict[str, object]] = []
    for alarm in ordered:
        onset = float(alarm["onset_timestamp"])
        if not clusters or onset - float(clusters[-1]["onset_timestamp"]) > tolerance:
            clusters.append({"onset_timestamp": onset, "alarms": [alarm]})
        else:
            clusters[-1]["alarms"].append(alarm)
    for cluster in clusters:
        cluster["query_timestamp"] = min(float(item["query_timestamp"]) for item in cluster["alarms"])
    return clusters
