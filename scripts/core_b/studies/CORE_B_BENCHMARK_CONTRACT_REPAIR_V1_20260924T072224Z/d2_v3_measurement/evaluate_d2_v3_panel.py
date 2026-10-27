"""Apply the locked D2_V3 measurement/matcher to the fixed dense dev panel."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement"
LOCK_PATH = OUT / "D2_V3_MATCHER_LOCKED.json"
RAW_ROOT = Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["raw_root"]).resolve()
TOLERANCE = 0.2
LEAD_THRESHOLDS = (0.25, 0.5, 1.0)
HORIZONS = (30, 60)
GROUPS = ("Hand", "Arm")
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260924


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_split_groups(split: str, episode_ids: set[int]) -> dict[int, dict[str, object]]:
    split_path = Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["split_manifest"])
    table = pq.read_table(
        split_path,
        columns=["episode_index", "original_task_index", "split", "video_sha256_fingerprint", "v3_benchmark_inclusion"],
        filters=[("split", "=", split), ("v3_benchmark_inclusion", "=", "primary")],
    )
    output = {}
    for row in table.to_pylist():
        episode = int(row["episode_index"])
        if episode in episode_ids:
            output[episode] = {
                "task": int(row["original_task_index"]),
                "group": str(row["video_sha256_fingerprint"]),
            }
    missing = episode_ids - set(output)
    if missing:
        raise AssertionError(f"{split} source group/task identity missing for {len(missing)} episodes")
    return output


def load_episode(row: dict[str, str], source: dict[str, object], sigma: np.ndarray) -> dict[str, object]:
    raw_root = RAW_ROOT.resolve(strict=True)
    path = (raw_root / row["parquet_path"]).resolve(strict=True)
    if not path.is_relative_to(raw_root):
        raise AssertionError("dev Parquet escaped the approved raw root")
    table = pq.read_table(path, columns=["timestamp", "action"])
    timestamp = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all() or not np.isfinite(timestamp).all():
        raise AssertionError(f"dev raw timeline/action contract failed for episode {row['episode_index']}")
    dt = np.diff(timestamp)
    if not len(dt) or np.any(dt <= 0) or np.max(np.abs(dt - 1.0 / 30.0)) > 1e-5:
        raise AssertionError(f"D2_V3 definition not applicable to dev episode {row['episode_index']}: raw data not 30Hz")
    delta = np.abs(np.diff(action, axis=0)) / sigma[None, :]
    q = {"Hand": np.mean(delta[:, :14], axis=1), "Arm": np.mean(delta[:, 14:], axis=1)}
    moving = {
        body: np.convolve(value, np.ones(8, dtype=np.float64) / 8.0, mode="valid")
        for body, value in q.items()
    }
    return {
        "timestamp": timestamp,
        "action": action,
        "q": q,
        "m": moving,
        "task": int(source["task"]),
        "group": str(source["group"]),
    }


def state_trace(signal: np.ndarray, first_endpoint_frame: int, frame_count: int, on: float, off: float):
    known = np.zeros(frame_count, dtype=bool)
    armed = np.zeros(frame_count, dtype=bool)
    quiet_at = np.zeros(frame_count, dtype=np.int16)
    high_at = np.zeros(frame_count, dtype=np.int16)
    events = []
    is_known = False
    is_armed = False
    quiet = 0
    high = 0
    signal_index = 0
    for frame in range(frame_count):
        if frame >= first_endpoint_frame and signal_index < len(signal):
            value = float(signal[signal_index])
            signal_index += 1
            if value <= off:
                quiet += 1
                high = 0
                if quiet >= 6:
                    is_armed = True
                    is_known = True
            elif value > on:
                quiet = 0
                if is_armed:
                    high += 1
                    if high >= 3:
                        onset_frame = frame - 2
                        events.append({"onset_frame": onset_frame, "onset_timestamp_index": onset_frame, "confirmation_frame": frame})
                        is_armed = False
                        high = 0
                else:
                    high = 0
            else:
                quiet = 0
                high = 0
        known[frame] = is_known
        armed[frame] = is_armed
        quiet_at[frame] = quiet
        high_at[frame] = high
    return {"known": known, "armed": armed, "quiet": quiet_at, "high": high_at, "events": events}


def clustered_state_and_events(signal: np.ndarray, frame: int, state: dict[str, object], on: float, off: float):
    known = bool(state["known"][frame])
    if not known or frame < 7:
        return known, bool(state["armed"][frame]), []
    queue = list(np.asarray(signal["q"] if isinstance(signal, dict) else signal)[frame - 7 : frame])
    return known, bool(state["armed"][frame]), queue


def predict_events(
    prediction: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    q: np.ndarray,
    frame: int,
    snapshot: dict[str, object],
    on: float,
    off: float,
) -> tuple[bool, bool, list[dict[str, object]]]:
    known = bool(snapshot["known"][frame])
    armed = bool(snapshot["armed"][frame])
    quiet = int(snapshot["quiet"][frame])
    high = int(snapshot["high"][frame])
    if not known or frame < 7:
        return known, armed, []
    queue = list(np.asarray(q[frame - 7 : frame], dtype=np.float64))
    previous = action[frame]
    forecasts = []
    for step, command in enumerate(prediction, start=1):
        q_pred = float(np.mean(np.abs(command - previous) / SIGMA))
        queue.append(q_pred)
        if len(queue) > 8:
            queue.pop(0)
        m_pred = float(np.mean(queue))
        end_frame = frame + step
        if m_pred <= off:
            quiet += 1
            high = 0
            if quiet >= 6:
                armed = True
        elif m_pred > on:
            quiet = 0
            if armed:
                high += 1
                if high >= 3:
                    onset_frame = end_frame - 2
                    if frame < onset_frame < len(timestamp):
                        forecasts.append({
                            "predicted_onset_frame": onset_frame,
                            "predicted_onset_time": float(timestamp[onset_frame]),
                        })
                    armed = False
                    high = 0
            else:
                high = 0
        else:
            quiet = 0
            high = 0
        previous = command
    return known, bool(snapshot["armed"][frame]), forecasts


def bootstrap_ci(numerator: np.ndarray, denominator: np.ndarray, boot_indices: np.ndarray, *, scale: float = 1.0):
    boot_num = numerator[boot_indices].sum(axis=1)
    boot_den = denominator[boot_indices].sum(axis=1)
    values = np.full(len(boot_num), np.nan, dtype=np.float64)
    valid = boot_den > 0
    values[valid] = boot_num[valid] / boot_den[valid] * scale
    if not np.isfinite(values).any():
        return None, None
    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def cluster_alerts(alerts: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(alerts, key=lambda item: (item["query_time"], item["predicted_onset_time"], item["dense_id"]))
    clusters: list[dict[str, object]] = []
    for alert in ordered:
        matches = [
            cluster for cluster in clusters
            if abs(float(alert["predicted_onset_time"]) - float(cluster["center_predicted_onset"])) <= TOLERANCE
        ]
        if matches:
            cluster = min(matches, key=lambda item: (item["first_query_time"], item["center_predicted_onset"], item["cluster_id"]))
            cluster["report_count"] += 1
            cluster["last_query_time"] = float(alert["query_time"])
            continue
        index = len(clusters)
        clusters.append({
            "cluster_id": f"{alert['source_group_id']}|{alert['body']}|{index:04d}",
            "source_group_id": alert["source_group_id"],
            "episode_index": alert["episode_index"],
            "task": alert["task"],
            "body": alert["body"],
            "first_dense_id": alert["dense_id"],
            "first_query_time": float(alert["query_time"]),
            "last_query_time": float(alert["query_time"]),
            "center_predicted_onset": float(alert["predicted_onset_time"]),
            "report_count": 1,
        })
    return clusters


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text())
    if lock.get("status") != "LOCKED_BEFORE_D2_V3_DEV_PREDICTION_READ":
        raise AssertionError("D2_V3 matcher protocol is not locked")
    if lock.get("evaluator_code_sha256") != sha256_file(Path(__file__)):
        raise AssertionError("evaluator source hash differs from the pre-evaluation matcher lock")
    threshold_path = OUT / "D2_V3_PROTOCOL_LOCKED.json"
    if lock.get("threshold_protocol_sha256") != sha256_file(threshold_path):
        raise AssertionError("D2_V3 threshold lock hash mismatch")
    if lock.get("fixed_panel", {}).get("input_sha256") != sha256_file(BENCH / "d2/D2_DENSE_INPUTS.npz"):
        raise AssertionError("fixed dense panel input changed after matcher lock")
    threshold_lock = json.loads(threshold_path.read_text())
    if threshold_lock.get("status") != "locked_train_thresholds_ready":
        raise AssertionError("train-only D2_V3 threshold lock is not applicable")
    on = {body: float(threshold_lock["thresholds"][body]["theta_on"]) for body in GROUPS}
    off = {body: float(threshold_lock["thresholds"][body]["theta_off"]) for body in GROUPS}

    dev_rows = {}
    with (V3 / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0":
                dev_rows.setdefault(int(row["episode_index"]), row)
    dev_sources = load_split_groups("dev", set(dev_rows))
    d1_sources = {row["episode_index"]: row["source_group_id"] for row in read_csv(RESULT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv")}
    group_mismatch_count = sum(
        str(d1_sources.get(str(episode), d1_sources.get(episode, ""))) != str(source["group"])
        for episode, source in dev_sources.items()
    )
    if group_mismatch_count:
        raise AssertionError(f"D1 source group fingerprint mismatches canonical video hash: {group_mismatch_count}")

    with np.load(BENCH / "d2/D2_DENSE_INPUTS.npz", allow_pickle=False) as archive:
        panel = {key: archive[key] for key in (
            "dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp", "anchor_command_native",
        )}
    panel_count = len(panel["dense_id"])
    if panel_count != 3718 or len(set(panel["dense_id"].astype(str).tolist())) != panel_count:
        raise AssertionError("dense panel ID count/uniqueness changed")
    panel_episodes = set(int(x) for x in panel["episode_index"])
    panel_groups = sorted({str(dev_sources[int(ep)]["group"]) for ep in panel_episodes})
    panel_tasks = sorted({int(x) for x in panel["original_task_index"]})
    if len(panel_episodes) != 118 or len(panel_groups) != 118:
        raise AssertionError("fixed panel no longer maps to 118 independent source groups")

    dev_data = {}
    v3_truth_events: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    v2_truth_events: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for episode, row in sorted(dev_rows.items()):
        raw = load_episode(row, dev_sources[episode], SIGMA)
        data_row = {**raw, "episode_index": episode}
        dev_data[episode] = data_row
        for body in GROUPS:
            state3 = state_trace(raw["m"][body], 8, len(raw["timestamp"]), on[body], off[body])
            state2 = state_trace(raw["q"][body], 1, len(raw["timestamp"]), float(D2_V2_ON[body]), float(D2_V2_OFF[body]))
            data_row.setdefault("states_v3", {})[body] = state3
            data_row.setdefault("states_v2", {})[body] = state2
            for event in state3["events"]:
                event_id = f"{raw['group']}|{body}|{event['onset_frame']}"
                v3_truth_events[(raw["group"], body)].setdefault(event_id, {
                    "event_id": event_id,
                    "source_group_id": raw["group"],
                    "episode_index": episode,
                    "task": raw["task"],
                    "body": body,
                    "onset_frame": int(event["onset_frame"]),
                    "onset_time": float(raw["timestamp"][int(event["onset_frame"])]),
                    "confirmation_frame": int(event["confirmation_frame"]),
                })
            # Reconstruct old V2 events for direct label-record vs independent-event audit.
            for event in state2["events"]:
                event_id = f"{raw['group']}|{body}|{event['onset_frame']}"
                v2_truth_events[(raw["group"], body)].setdefault(event_id, {
                    "event_id": event_id,
                    "source_group_id": raw["group"],
                    "episode_index": episode,
                    "task": raw["task"],
                    "body": body,
                    "onset_frame": int(event["onset_frame"]),
                    "onset_time": float(raw["timestamp"][int(event["onset_frame"])]),
                    "confirmation_frame": int(event["confirmation_frame"]),
                })

    # Query states and opportunity denominators are built entirely from raw
    # dev ground truth and the fixed panel, before model forecasts are read.
    queries_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    query_rows: list[dict[str, object]] = []
    for index in range(panel_count):
        dense_id = str(panel["dense_id"][index])
        episode = int(panel["episode_index"][index])
        frame = int(panel["anchor_frame"][index])
        data_row = dev_data[episode]
        ts = data_row["timestamp"]
        if frame < 8 or frame + 60 >= len(ts):
            raise AssertionError(f"fixed panel query has an invalid history/future boundary: {dense_id}")
        if abs(float(ts[frame]) - float(panel["anchor_timestamp"][index])) > 1e-5:
            raise AssertionError(f"dense input timestamp mismatch at {dense_id}")
        if not np.allclose(data_row["action"][frame], panel["anchor_command_native"][index], atol=1e-5, rtol=0):
            raise AssertionError(f"dense input anchor command mismatch at {dense_id}")
        query = {
            "dense_id": dense_id,
            "episode_index": episode,
            "task": int(panel["original_task_index"][index]),
            "source_group_id": data_row["group"],
            "anchor_frame": frame,
            "query_time": float(ts[frame]),
        }
        for body in GROUPS:
            snap3 = data_row["states_v3"][body]
            snap2 = data_row["states_v2"][body]
            query[f"v3_known_{body}"] = bool(snap3["known"][frame])
            query[f"v3_armed_{body}"] = bool(snap3["armed"][frame])
            query[f"v2_known_{body}"] = bool(snap2["known"][frame])
            query[f"v2_armed_{body}"] = bool(snap2["armed"][frame])
        for horizon in HORIZONS:
            endpoint = frame + horizon - 2
            query[f"full_h{horizon}"] = frame + horizon < len(ts)
            query[f"effective_end_h{horizon}"] = float(ts[endpoint])
        query_rows.append(query)
        queries_by_group[str(data_row["group"])].append(query)

    # Ground-truth opportunities by independent event and effective forecast horizon.
    opportunity_by_h_body_group: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    lead_opportunity_by_h_body_group: dict[tuple[int, str, str, float], set[str]] = defaultdict(set)
    panel_truth_by_body_group: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    panel_groups_set = set(panel_groups)
    for (group, body), events in v3_truth_events.items():
        if group in panel_groups_set:
            panel_truth_by_body_group[(group, body)].update(events)
    for group in panel_groups:
        for body in GROUPS:
            events = panel_truth_by_body_group[(group, body)]
            queries = queries_by_group[group]
            for event_id, event in events.items():
                for horizon in HORIZONS:
                    candidates = []
                    for query in queries:
                        if not query[f"v3_known_{body}"] or not query[f"full_h{horizon}"]:
                            continue
                        lead = float(event["onset_time"]) - float(query["query_time"])
                        if 0 < lead <= float(query[f"effective_end_h{horizon}"]) - float(query["query_time"]) + 1e-8:
                            candidates.append(lead)
                    if candidates:
                        opportunity_by_h_body_group[(horizon, body, group)].add(event_id)
                        for lead_threshold in LEAD_THRESHOLDS:
                            if any(lead >= lead_threshold for lead in candidates):
                                lead_opportunity_by_h_body_group[(horizon, body, group, lead_threshold)].add(event_id)

    # Open the already locked and accepted V3 forecast bundles only now.
    acceptance_rows = read_csv(RESULT / "d4/D4_BUNDLE_ACCEPTANCE.csv")
    if len(acceptance_rows) != 36:
        raise AssertionError("locked dense prediction acceptance manifest is not 36 rows")
    group_task = {str(source["group"]): int(source["task"]) for source in dev_sources.values() if str(source["group"]) in panel_groups_set}
    panel_query_counts = Counter(str(row["source_group_id"]) for row in query_rows)
    exposure_by_group = {}
    for group in panel_groups:
        group_queries = queries_by_group[group]
        exposure_by_group[group] = max(float(row["query_time"]) for row in group_queries) - min(float(row["query_time"]) for row in group_queries)
    if not np.isclose(sum(exposure_by_group.values()) / 3600.0, 0.2496111027399699, atol=1e-5):
        raise AssertionError("fixed panel exposure differs from the locked D2_V2 dense panel")

    # Query support audit includes the historical exact-quiet gate and V3 state.
    state_support_rows = []
    panel_episode_ids = set(int(x) for x in panel["episode_index"])
    old_threshold_support = {}
    for body in GROUPS:
        q_all = np.concatenate([dev_data[episode]["q"][body] for episode in sorted(dev_data)])
        q_panel = np.concatenate([dev_data[episode]["q"][body] for episode in sorted(panel_episode_ids)])
        old_threshold_support[body] = {
            "dev_exact_q_zero_fraction": float(np.mean(q_all == 0)),
            "panel_episode_exact_q_zero_fraction": float(np.mean(q_panel == 0)),
            "dev_raw_steps": len(q_all),
            "panel_episode_raw_steps": len(q_panel),
        }
        for definition in ("D2_V2", "D2_V3"):
            for horizon in HORIZONS:
                qset = [row for row in query_rows if row[f"full_h{horizon}"]]
                known_key = f"{definition.lower()}_known_{body}"
                armed_key = f"{definition.lower()}_armed_{body}"
                unknown = sum(not row[known_key] for row in qset)
                known_unarmed = sum(row[known_key] and not row[armed_key] for row in qset)
                armed_count = sum(row[armed_key] for row in qset)
                incomplete = sum(not row[f"full_h{horizon}"] for row in query_rows)
                state_support_rows.append({
                    "definition": definition,
                    "body": body,
                    "horizon_steps": horizon,
                    "queries": len(query_rows),
                    "complete_future_queries": len(qset),
                    "unknown_state_queries": unknown,
                    "known_unarmed_or_ongoing_queries": known_unarmed,
                    "armed_at_anchor_queries": armed_count,
                    "boundary_incomplete_queries": incomplete,
                    "exact_q_zero_fraction_dev": old_threshold_support[body]["dev_exact_q_zero_fraction"],
                    "exact_q_zero_fraction_panel_episodes": old_threshold_support[body]["panel_episode_exact_q_zero_fraction"],
                    "theta_off": float(D2_V2_OFF[body]) if definition == "D2_V2" else off[body],
                    "final_test_access": 0,
                })

    # Evaluate every fixed-panel model/seed/horizon and retain cluster audit rows.
    group_result_rows: list[dict[str, object]] = []
    cluster_audit_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    v3_panel_event_summary = {}
    for body in GROUPS:
        panel_events = {group: panel_truth_by_body_group[(group, body)] for group in panel_groups}
        v3_panel_event_summary[body] = {
            "event_records": sum(len(events) for events in panel_events.values()),
            "independent_source_group_events": len({key for events in panel_events.values() for key in events}),
            "source_groups_with_event": sum(bool(events) for events in panel_events.values()),
            "tasks_with_event": len({group_task[group] for group, events in panel_events.items() if events}),
        }
        for horizon in HORIZONS:
            for group in panel_groups:
                support_rows.append({
                    "scope": "panel_opportunities",
                    "body": body,
                    "horizon_steps": horizon,
                    "source_group_id": group,
                    "task": group_task[group],
                    "event_records": len(panel_events[group]),
                    "complete_opportunity_events": len(opportunity_by_h_body_group[(horizon, body, group)]),
                    **{f"opportunity_events_lead_ge_{lead:g}": len(lead_opportunity_by_h_body_group[(horizon, body, group, lead)]) for lead in LEAD_THRESHOLDS},
                    "final_test_access": 0,
                })

    accepted_cells = set()
    for accepted in acceptance_rows:
        bundle_path = Path(accepted["prediction_bundle"]).resolve(strict=True)
        dense_root = (BENCH / "predictions_dense").resolve(strict=True)
        if not bundle_path.is_relative_to(dense_root):
            raise AssertionError("dense forecast bundle escaped the locked predictions_dense root")
        if sha256_file(bundle_path) != accepted["prediction_sha256"]:
            raise AssertionError(f"dense prediction hash changed: {bundle_path.name}")
        sidecar = json.loads(bundle_path.with_suffix(".json").read_text())
        if sidecar.get("status") != "PASS" or sidecar.get("final_test_access") != 0:
            raise AssertionError(f"dense forecast sidecar failed: {bundle_path.name}")
        with np.load(bundle_path, allow_pickle=False) as archive:
            dense_id = archive["dense_id"].astype("U96")
            episode_index = archive["episode_index"].astype(np.int64)
            anchor_frame = archive["anchor_frame"].astype(np.int64)
            anchor_timestamp = archive["anchor_timestamp"].astype(np.float64)
            predictions = archive["predictions_native"].astype(np.float32)
            horizon = int(archive["horizon_steps"])
            model = str(archive["cell"])
            seed = int(archive["seed"])
        if accepted["model"] != model or int(accepted["seed"]) != seed or int(accepted["horizon_steps"]) != horizon:
            raise AssertionError(f"dense model/seed/horizon mismatch: {bundle_path.name}")
        if len(dense_id) != panel_count or set(dense_id.tolist()) != set(panel["dense_id"].astype(str).tolist()) or predictions.shape != (panel_count, horizon, 28):
            raise AssertionError(f"dense prediction IDs/shape mismatch: {bundle_path.name}")
        if not np.isfinite(predictions).all():
            raise AssertionError(f"nonfinite dense predictions: {bundle_path.name}")
        prediction_order = {str(value): int(i) for i, value in enumerate(dense_id)}
        panel_order = np.asarray([prediction_order[str(value)] for value in panel["dense_id"]], dtype=np.int64)
        predictions = predictions[panel_order]
        if not (
            np.array_equal(episode_index[panel_order], panel["episode_index"])
            and np.array_equal(anchor_frame[panel_order], panel["anchor_frame"])
            and np.allclose(anchor_timestamp[panel_order], panel["anchor_timestamp"], atol=1e-5, rtol=0)
        ):
            raise AssertionError(f"dense forecast row metadata mismatch: {bundle_path.name}")
        cell_key = (model, seed, horizon)
        if cell_key in accepted_cells:
            raise AssertionError(f"duplicate dense model cell: {cell_key}")
        accepted_cells.add(cell_key)

        for body in GROUPS:
            alerts_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
            query_counts_by_group = Counter()
            for index, query in enumerate(query_rows):
                episode = int(query["episode_index"])
                frame = int(query["anchor_frame"])
                data_row = dev_data[episode]
                known, armed_at_anchor, forecasts = predict_events(
                    predictions[index], data_row["action"], data_row["timestamp"], data_row["q"][body], frame,
                    data_row["states_v3"][body], on[body], off[body],
                )
                query_counts_by_group[str(query["source_group_id"])] += 1
                if not known or not query[f"full_h{horizon}"]:
                    continue
                effective_end = float(query[f"effective_end_h{horizon}"])
                for forecast in forecasts:
                    predicted_time = float(forecast["predicted_onset_time"])
                    if not (float(query["query_time"]) < predicted_time <= effective_end + 1e-8):
                        continue
                    alerts_by_group[str(query["source_group_id"])].append({
                        "source_group_id": query["source_group_id"],
                        "episode_index": episode,
                        "task": query["task"],
                        "body": body,
                        "dense_id": query["dense_id"],
                        "query_time": query["query_time"],
                        "predicted_onset_time": predicted_time,
                        "predicted_onset_frame": forecast["predicted_onset_frame"],
                        "known_at_anchor": known,
                        "armed_at_anchor": armed_at_anchor,
                    })

            for group in panel_groups:
                clusters = cluster_alerts(alerts_by_group[group])
                opportunity_events = opportunity_by_h_body_group[(horizon, body, group)]
                all_events = panel_events[group]
                unmatched = set(opportunity_events)
                hits = 0
                timing_abs = []
                timing_signed = []
                for cluster in sorted(clusters, key=lambda item: (item["first_query_time"], item["center_predicted_onset"], item["first_dense_id"])):
                    candidates = []
                    for event_id in unmatched:
                        event = all_events[event_id]
                        lead = float(event["onset_time"]) - float(cluster["first_query_time"])
                        time_delta = abs(float(cluster["center_predicted_onset"]) - float(event["onset_time"]))
                        query = cluster["first_query_time"]
                        eff_h = next(
                            float(row[f"effective_end_h{horizon}"]) - float(row["query_time"])
                            for row in queries_by_group[group] if row["dense_id"] == cluster["first_dense_id"]
                        )
                        if 0 < lead <= eff_h + 1e-8 and time_delta <= TOLERANCE:
                            candidates.append((time_delta, float(event["onset_time"]), event_id, event, lead))
                    if candidates:
                        _, _, event_id, event, lead = min(candidates, key=lambda item: (item[0], item[1], item[2]))
                        unmatched.remove(event_id)
                        hits += 1
                        error = float(cluster["center_predicted_onset"]) - float(event["onset_time"])
                        timing_abs.append(abs(error))
                        timing_signed.append(error)
                        cluster_audit_rows.append({
                            "model": model, "seed": seed, "horizon_steps": horizon, "body": body,
                            "cluster_id": cluster["cluster_id"], "source_group_id": group, "task": group_task[group],
                            "first_dense_id": cluster["first_dense_id"], "first_query_time": cluster["first_query_time"],
                            "fixed_predicted_onset_time": cluster["center_predicted_onset"], "report_count": cluster["report_count"],
                            "match_status": "hit", "matched_event_id": event_id, "ground_truth_onset_time": event["onset_time"],
                            "lead_seconds": lead, "timing_signed_error_seconds": error,
                            "final_test_access": 0,
                        })
                    else:
                        cluster_audit_rows.append({
                            "model": model, "seed": seed, "horizon_steps": horizon, "body": body,
                            "cluster_id": cluster["cluster_id"], "source_group_id": group, "task": group_task[group],
                            "first_dense_id": cluster["first_dense_id"], "first_query_time": cluster["first_query_time"],
                            "fixed_predicted_onset_time": cluster["center_predicted_onset"], "report_count": cluster["report_count"],
                            "match_status": "false_alarm", "matched_event_id": "", "ground_truth_onset_time": "",
                            "lead_seconds": "", "timing_signed_error_seconds": "",
                            "final_test_access": 0,
                        })
                for event_id in sorted(unmatched):
                    event = all_events[event_id]
                    cluster_audit_rows.append({
                        "model": model, "seed": seed, "horizon_steps": horizon, "body": body,
                        "cluster_id": "", "source_group_id": group, "task": group_task[group],
                        "first_dense_id": "", "first_query_time": "", "fixed_predicted_onset_time": "", "report_count": 0,
                        "match_status": "missed_gt", "matched_event_id": event_id, "ground_truth_onset_time": event["onset_time"],
                        "lead_seconds": "", "timing_signed_error_seconds": "",
                        "final_test_access": 0,
                    })

                lead_hits = {}
                for lead_threshold in LEAD_THRESHOLDS:
                    lead_hit_count = 0
                    for event_row in cluster_audit_rows:
                        if event_row.get("model") == model and int(event_row.get("seed", -1)) == seed and int(event_row.get("horizon_steps", -1)) == horizon and event_row.get("body") == body and event_row.get("source_group_id") == group and event_row.get("match_status") == "hit" and float(event_row.get("lead_seconds", 0)) >= lead_threshold:
                            lead_hit_count += 1
                    lead_hits[lead_threshold] = lead_hit_count

                exposure = exposure_by_group[group]
                group_result_rows.append({
                    "model": model, "seed": seed, "horizon_steps": horizon, "body": body,
                    "source_group_id": group, "task": group_task[group], "panel_queries": query_counts_by_group[group],
                    "opportunities": len(opportunity_events), "hits": hits, "misses": len(opportunity_events) - hits,
                    "prediction_clusters": len(clusters), "false_alarms": len(clusters) - hits,
                    "exposure_seconds": exposure, "timing_abs_sum": sum(timing_abs), "timing_signed_sum": sum(timing_signed), "timing_count": len(timing_abs),
                    **{f"lead_opportunities_{lead:g}": len(lead_opportunity_by_h_body_group[(horizon, body, group, lead)]) for lead in LEAD_THRESHOLDS},
                    **{f"lead_hits_{lead:g}": lead_hits[lead] for lead in LEAD_THRESHOLDS},
                    "eligible_known_queries": sum(bool(row[f"v3_known_{body}"]) for row in queries_by_group[group]),
                    "armed_at_anchor_queries": sum(bool(row[f"v3_armed_{body}"]) for row in queries_by_group[group]),
                    "final_test_access": 0,
                })

    if len(accepted_cells) != 36:
        raise AssertionError(f"expected 36 dense V3 prediction cells, got {len(accepted_cells)}")

    # Bootstrap all group-level outcomes together, preserving source clusters.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_indices = rng.integers(0, len(panel_groups), size=(BOOTSTRAPS, len(panel_groups)), dtype=np.int16)
    metrics_rows = []
    for model, seed, horizon in sorted(accepted_cells):
        for body in GROUPS:
            by_group = {row["source_group_id"]: row for row in group_result_rows if row["model"] == model and row["seed"] == seed and row["horizon_steps"] == horizon and row["body"] == body}
            rows = [by_group[group] for group in panel_groups]
            opp = np.asarray([row["opportunities"] for row in rows], dtype=np.float64)
            hits = np.asarray([row["hits"] for row in rows], dtype=np.float64)
            false = np.asarray([row["false_alarms"] for row in rows], dtype=np.float64)
            exposure = np.asarray([row["exposure_seconds"] for row in rows], dtype=np.float64)
            timing_abs = np.asarray([row["timing_abs_sum"] for row in rows], dtype=np.float64)
            timing_signed = np.asarray([row["timing_signed_sum"] for row in rows], dtype=np.float64)
            timing_n = np.asarray([row["timing_count"] for row in rows], dtype=np.float64)
            recall = float(hits.sum() / opp.sum()) if opp.sum() else None
            recall_ci = bootstrap_ci(hits, opp, boot_indices)
            fa_rate = float(false.sum() / exposure.sum() * 3600.0) if exposure.sum() else None
            fa_ci = bootstrap_ci(false, exposure, boot_indices, scale=3600.0)
            timing_abs_mean = float(timing_abs.sum() / timing_n.sum()) if timing_n.sum() else None
            timing_signed_mean = float(timing_signed.sum() / timing_n.sum()) if timing_n.sum() else None
            timing_abs_ci = bootstrap_ci(timing_abs, timing_n, boot_indices)
            row_out: dict[str, object] = {
                "model": model, "seed": seed, "horizon_steps": horizon, "event_group": body,
                "episodes": 118, "source_groups": 118, "queries": panel_count,
                "eligible_queries": int(sum(row["eligible_known_queries"] for row in rows)),
                "truth_event_records_in_panel": v3_panel_event_summary[body]["event_records"],
                "panel_independent_events": v3_panel_event_summary[body]["independent_source_group_events"],
                "full_opportunity_events": int(opp.sum()), "hits": int(hits.sum()), "misses": int(opp.sum() - hits.sum()),
                "predicted_forecast_clusters": int(sum(row["prediction_clusters"] for row in rows)),
                "false_alarms": int(false.sum()), "recall": recall,
                "recall_ci95_low": recall_ci[0], "recall_ci95_high": recall_ci[1],
                "false_alarms_per_hour": fa_rate, "false_alarms_per_hour_ci95_low": fa_ci[0], "false_alarms_per_hour_ci95_high": fa_ci[1],
                "exposure_seconds": float(exposure.sum()), "exposure_hours": float(exposure.sum() / 3600.0),
                "timing_abs_error_seconds": timing_abs_mean, "timing_abs_error_ci95_low": timing_abs_ci[0], "timing_abs_error_ci95_high": timing_abs_ci[1],
                "timing_signed_error_seconds": timing_signed_mean,
                "tasks_with_complete_opportunity": len({group_task[row["source_group_id"]] for row in rows if row["opportunities"] > 0}),
                "support_status": "low_support" if body == "Hand" and (opp.sum() < 30 or len({group_task[row["source_group_id"]] for row in rows if row["opportunities"] > 0}) < 10) else "support_rule_pass",
                "bootstrap_unit": "exact video_sha256_fingerprint source group",
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "interval": "95% source-group percentile bootstrap; development exploratory",
                "final_test_access": 0,
            }
            for lead in LEAD_THRESHOLDS:
                lead_opp = np.asarray([row[f"lead_opportunities_{lead:g}"] for row in rows], dtype=np.float64)
                lead_hits = np.asarray([row[f"lead_hits_{lead:g}"] for row in rows], dtype=np.float64)
                ci = bootstrap_ci(lead_hits, lead_opp, boot_indices)
                row_out[f"opportunities_lead_ge_{lead:g}"] = int(lead_opp.sum())
                row_out[f"hits_lead_ge_{lead:g}"] = int(lead_hits.sum())
                row_out[f"recall_lead_ge_{lead:g}"] = float(lead_hits.sum() / lead_opp.sum()) if lead_opp.sum() else None
                row_out[f"recall_lead_ge_{lead:g}_ci95_low"] = ci[0]
                row_out[f"recall_lead_ge_{lead:g}_ci95_high"] = ci[1]
            metrics_rows.append(row_out)

    # Historical D2_V2 event registry and its exact-quiet gate are audited in
    # parallel, never overwritten. This distinguishes rows from independent IDs.
    old_registry = read_csv(RESULT / "results/D2_EVENT_REGISTRY_V2.csv")
    old_event_records = len(old_registry)
    old_event_keys = {
        (row["source_group_id"], row["group"], row["onset_frame"])
        for row in old_registry
    }
    old_dense_rows = read_csv(RESULT / "results/D2_DENSE_RESULTS_V2.csv")
    old_hand_support = [
        row for row in old_dense_rows
        if row["model"] == "F-TOK-D" and row["event_group"] == "Hand"
    ]
    old_v2_summary = {
        "D2_V2_event_registry_records": old_event_records,
        "D2_V2_independent_group_body_frame_events": len(old_event_keys),
        "D2_V2_source_groups_with_event": len({key[0] for key in old_event_keys}),
        "D2_V2_theta_off_hand": float(D2_V2_OFF["Hand"]),
        "D2_V2_theta_off_arm": float(D2_V2_OFF["Arm"]),
        "dev_exact_q_zero_fraction": old_threshold_support["Hand"]["dev_exact_q_zero_fraction"],
        "panel_episode_exact_q_zero_fraction": old_threshold_support["Hand"]["panel_episode_exact_q_zero_fraction"],
        "V2_F_TOK_D_hand_complete_opportunities_by_horizon": {row["horizon_steps"]: int(row["full_opportunity_events"]) for row in old_hand_support},
        "panel_queries": panel_count,
        "panel_episodes": len(panel_episodes),
        "panel_source_groups": len(panel_groups),
        "panel_exposure_hours": float(sum(exposure_by_group.values()) / 3600.0),
        "final_test_access": 0,
    }

    # Development label support. The panel line is held constant, and panel
    # selection is independent of model errors.
    support_summary_rows = []
    for split_name, event_map, episode_set in (
        ("train", None, set()),
        ("dev", v3_truth_events, set(dev_data)),
        ("panel", panel_truth_by_body_group, panel_episode_ids),
    ):
        if split_name == "train":
            train_summary = json.loads((OUT / "D2_V3_TRAIN_SUPPORT_SUMMARY.json").read_text())
            for body in GROUPS:
                block = train_summary["independent_events"][body]
                support_summary_rows.append({"split": "train", "body": body, "horizon_steps": "NA", "episodes": 2712, "source_groups": 2712, "event_records": block["event_records"], "independent_events": block["independent_source_group_events"], "source_groups_with_event": block["source_groups_with_event"], "tasks_with_event": block["tasks_with_event"], "final_test_access": 0})
            continue
        unique_groups = {str(dev_data[episode]["group"]) for episode in episode_set}
        for body in GROUPS:
            record_count = sum(len(data_row["states_v3"][body]["events"]) for episode, data_row in dev_data.items() if episode in episode_set)
            unique_events = set()
            group_with_event = set()
            task_with_event = set()
            for (group, event_body), events in event_map.items():
                if event_body == body and group in unique_groups:
                    for event_id, event in events.items():
                        unique_events.add(event_id)
                        group_with_event.add(group)
                        task_with_event.add(int(event["task"]))
            support_summary_rows.append({"split": split_name, "body": body, "horizon_steps": "NA", "episodes": len(episode_set), "source_groups": len(unique_groups), "event_records": record_count, "independent_events": len(unique_events), "source_groups_with_event": len(group_with_event), "tasks_with_event": len(task_with_event), "final_test_access": 0})
    for row in metrics_rows:
        support_summary_rows.append({
            "split": "panel_opportunities", "body": row["event_group"], "horizon_steps": row["horizon_steps"],
            "episodes": row["episodes"], "source_groups": row["source_groups"], "event_records": row["truth_event_records_in_panel"],
            "independent_events": row["panel_independent_events"], "full_opportunity_events": row["full_opportunity_events"],
            "tasks_with_event": row["tasks_with_complete_opportunity"], "support_status": row["support_status"],
            "model": row["model"], "seed": row["seed"], "final_test_access": 0,
        })

    overall_status = "completed_exploratory"
    panel_hand_support = [row for row in metrics_rows if row["event_group"] == "Hand"]
    hand_low = any(row["support_status"] == "low_support" for row in panel_hand_support)
    if hand_low:
        overall_status = "completed_exploratory_low_hand_support"
    summary = {
        "status": overall_status,
        "D2_V3_STATUS": "completed_exploratory",
        "threshold_lock_sha256": sha256_file(threshold_path),
        "matcher_lock_sha256": sha256_file(LOCK_PATH),
        "evaluator_code_sha256": sha256_file(Path(__file__)),
        "train_support": json.loads((OUT / "D2_V3_TRAIN_SUPPORT_SUMMARY.json").read_text()),
        "dev_support": [row for row in support_summary_rows if row["split"] == "dev"],
        "panel_query_support_by_state": state_support_rows,
        "panel_label_opportunities": [row for row in support_summary_rows if row["split"] == "panel_opportunities" and row["model"] == "F-TOK-D"],
        "panel_query_count": panel_count,
        "panel_episode_count": len(panel_episodes),
        "panel_source_group_count": len(panel_groups),
        "panel_task_count": len(panel_tasks),
        "panel_exposure_hours": float(sum(exposure_by_group.values()) / 3600.0),
        "accepted_dense_predictions": len(accepted_cells),
        "old_D2_V2_audit": old_v2_summary,
        "hand_panel_low_support_any_horizon": hand_low,
        "final_test_access": 0,
    }
    write_csv(OUT / "D2_V3_RESULTS.csv", metrics_rows)
    write_csv(OUT / "D2_V3_GROUP_METRICS.csv", group_result_rows)
    write_csv(OUT / "D2_V3_FORECAST_CLUSTER_MATCH_AUDIT.csv", cluster_audit_rows)
    write_csv(OUT / "D2_V3_STATE_SUPPORT.csv", state_support_rows)
    write_csv(OUT / "D2_V3_SUPPORT_SUMMARY.csv", support_summary_rows)
    (OUT / "D2_V2_V3_SUPPORT_AUDIT.json").write_text(json.dumps(old_v2_summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "D2_V3_EVALUATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# D2_V3 exploratory event measurement sensitivity",
        "",
        f"Status: `{overall_status}`. This is a prelocked measurement-definition sensitivity on repeatedly exposed development outcomes; it does not replace D2_V2 or select a new candidate.",
        "",
        "## Frozen definition and support",
        "",
        f"- Train-only thresholds: Hand `theta_on={on['Hand']:.6f}`, `theta_off={off['Hand']:.6f}`; Arm `theta_on={on['Arm']:.6f}`, `theta_off={off['Arm']:.6f}`. The 30 Hz raw timeline check passed; no threshold was fit to dev.",
        f"- The fixed panel is {panel_count} queries, {len(panel_episodes)} episodes, {len(panel_groups)} exact source groups across {len(panel_tasks)} tasks; total observed exposure is {sum(exposure_by_group.values())/3600:.6f} hours.",
        f"- Train independent events: Hand {threshold_lock['train_independent_events']['Hand']['independent_source_group_events']} across {threshold_lock['train_independent_events']['Hand']['source_groups_with_event']} groups / {threshold_lock['train_independent_events']['Hand']['tasks_with_event']} tasks; Arm {threshold_lock['train_independent_events']['Arm']['independent_source_group_events']} across {threshold_lock['train_independent_events']['Arm']['source_groups_with_event']} groups / {threshold_lock['train_independent_events']['Arm']['tasks_with_event']} tasks.",
        "- Labels are smoothed command-change onsets, not physical velocity or contact; onset is backdated to the first of three confirming high steps.",
        "",
        "## D2_V2 comparison and query-state audit",
        "",
        f"- D2_V2 has {old_event_records} event rows but {len(old_event_keys)} unique source-group/body/onset-frame keys. The old Hand `theta_off=0` makes quiet exactly `q==0`; raw exact-zero fractions are reported in `D2_V2_V3_SUPPORT_AUDIT.json`.",
        f"- On the old task-balanced panel, F-TOK-D Hand complete opportunities were {old_v2_summary['V2_F_TOK_D_hand_complete_opportunities_by_horizon']}; 0/5 is not evidence of universal incapacity. Unknown, known-but-unarmed, armed-at-anchor, and boundary counts under both definitions are in `D2_V3_STATE_SUPPORT.csv`.",
        "",
        "## D2_V3 forecast metrics",
        "",
        "The complete 36-cell × Hand/Arm table is `D2_V3_RESULTS.csv`; it includes raw event counts, one-to-one hits/misses, false clusters, exposure hours, FA/hour and source-group bootstrap intervals, plus lead-specific denominators and recall. `D2_V3_FORECAST_CLUSTER_MATCH_AUDIT.csv` preserves every matched cluster, false alarm, and missed GT event.",
        "",
        "Alarm clustering uses the immutable first predicted-onset center; matching is one-to-one and constrained by the actual first query time and effective H−2 onset horizon. CIs resample exact video-hash source groups (10,000 draws, fixed seed). Rates on the short panel are exploratory, not deployment rates.",
        "",
        "## Support decision",
        "",
        f"Hand low-support gate triggered: `{hand_low}`. A gate triggers if any horizon/cell has fewer than 30 independent complete opportunity events or fewer than 10 supported tasks. See `D2_V3_SUPPORT_SUMMARY.csv`.",
    ]
    if hand_low:
        report += [
            "",
            "A support expansion is proposed separately in `D2_SUPPORT_EXPANSION_PROPOSAL.md`; the current panel is not replaced and no episode was selected using model error.",
        ]
    else:
        report += ["", "No expansion proposal is required by the locked low-support rule; retain the fixed panel and stop widening it to improve model scores."]
    report += ["", "Final/held-out access: `0`. D2_V2 historical files were read only and not overwritten."]
    (OUT / "D2_V3_EVENT_MEASUREMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": overall_status,
        "accepted_dense_predictions": len(accepted_cells),
        "panel_groups": len(panel_groups),
        "panel_queries": panel_count,
        "Hand_opportunities_H30": next(row["full_opportunity_events"] for row in metrics_rows if row["model"] == "F-TOK-D" and row["seed"] == 20260950 and row["horizon_steps"] == 30 and row["event_group"] == "Hand"),
        "Hand_opportunities_H60": next(row["full_opportunity_events"] for row in metrics_rows if row["model"] == "F-TOK-D" and row["seed"] == 20260950 and row["horizon_steps"] == 60 and row["event_group"] == "Hand"),
        "hand_low_support_any_horizon": hand_low,
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    # Load legacy V2 thresholds before any dev labels are built.
    legacy = json.loads((RESULT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
    D2_V2_ON = {"Hand": float(legacy["theta_on"][0]), "Arm": float(legacy["theta_on"][1])}
    D2_V2_OFF = {"Hand": float(legacy["theta_off"][0]), "Arm": float(legacy["theta_off"][1])}
    main()
