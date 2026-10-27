"""Fixed-threshold D2_V3 event diagnostic on old/new dense predictions; no refit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
PREV_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EVENT_CODE = PREV_ROOT / "d2_v3_measurement/evaluate_d2_v3_panel.py"
EVENT_PROTOCOL = PREV_ROOT / "d2_v3_measurement/D2_V3_PROTOCOL_LOCKED.json"
OLD_PRED_ROOT = RESULT_ROOT / "d3_extension/predictions"
NEW_PRED_ROOT = EQ_ROOT / "d3/dense_predictions"
INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
OUT = EQ_ROOT / "d3/analysis"
SEEDS = (20261001, 20261002, 20261003)
HORIZONS = (30, 60)
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260924


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty event secondary CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prediction_grid(root: Path, ids: np.ndarray) -> dict[tuple[str, int, int], np.ndarray]:
    out = {}
    for sidecar in sorted(root.glob("*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        path = sidecar.with_suffix(".npz")
        if not path.is_file() or sha256_file(path) != record.get("prediction_sha256"):
            raise AssertionError(f"event-secondary prediction hash mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            local_ids = data["dense_id"].astype("U96")
            prediction = data["predictions_native"].astype(np.float32, copy=True)
            key = (str(data["config"].item()), int(data["seed"].item()), int(data["horizon_steps"]))
        if len(set(local_ids.tolist())) != 3718 or set(local_ids.tolist()) != set(ids.tolist()):
            raise AssertionError(f"event-secondary dense ID mismatch: {path.name}")
        if prediction.shape != (3718, key[2], 28) or not np.isfinite(prediction).all():
            raise AssertionError(f"event-secondary prediction shape/finite check failed: {path.name}")
        index = {str(value): i for i, value in enumerate(local_ids)}
        out[key] = prediction[np.asarray([index[str(value)] for value in ids], dtype=np.int64)]
    if len(out) != 36:
        raise AssertionError(f"event-secondary prediction grid incomplete: {len(out)}/36")
    return out


def bootstrap_rate(numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray, scale: float = 1.0) -> tuple[float | None, float | None, float | None]:
    point_den = float(denominator.sum())
    if point_den <= 0:
        return None, None, None
    point = float(numerator.sum() / point_den * scale)
    boot_num = numerator[draws].sum(axis=1)
    boot_den = denominator[draws].sum(axis=1)
    valid = boot_den > 0
    if not np.any(valid):
        return point, None, None
    ratios = boot_num[valid] / boot_den[valid] * scale
    low, high = np.quantile(ratios, [0.025, 0.975])
    return point, float(low), float(high)


def event_protocol_contract_matches(
    d3_lock: dict[str, Any],
    d2v3: dict[str, Any],
    protocol_path: Path,
    evaluator_path: Path,
) -> bool:
    """Check the exact D3-bound D2_V3 protocol and evaluator, not a generic label."""
    contract = d3_lock.get("event_secondary", {})
    if d3_lock.get("status") != "LOCKED":
        return False
    if contract.get("status") != "secondary fixed-threshold diagnostic only":
        return False
    if contract.get("dev_threshold_refit") is not False or contract.get("final_test_access") != 0:
        return False
    if d2v3.get("status") != "locked_train_thresholds_ready":
        return False
    if d2v3.get("protocol_id") != "CORE_B_D2_EVENT_MEASUREMENT_SENSITIVITY_V3":
        return False
    if d2v3.get("final_test_access") != 0:
        return False
    if d2v3.get("smoothing_window_raw_steps") != contract.get("smoothing_window_raw_steps"):
        return False
    if d2v3.get("threshold_algorithm") != contract.get("threshold_algorithm"):
        return False
    if d2v3.get("thresholds") != contract.get("thresholds"):
        return False
    if sha256_file(protocol_path) != contract.get("protocol_sha256"):
        return False
    if sha256_file(evaluator_path) != contract.get("evaluator_source_sha256"):
        return False
    return True


def main() -> int:
    protocol = json.loads((EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    d2v3 = json.loads(EVENT_PROTOCOL.read_text(encoding="utf-8"))
    if not event_protocol_contract_matches(protocol, d2v3, EVENT_PROTOCOL, EVENT_CODE):
        raise AssertionError("D3-EQ/D2_V3 event protocol, thresholds, or evaluator hash mismatch")
    spec = importlib.util.spec_from_file_location("locked_d2_v3_event_evaluator", EVENT_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(EVENT_CODE)
    event = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(event)

    with np.load(INPUTS, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ("dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp")}
    dense_ids = data["dense_id"].astype("U96")
    episodes = data["episode_index"].astype(np.int64)
    frames = data["anchor_frame"].astype(np.int64)
    tasks = data["original_task_index"].astype(np.int64)
    query_times = data["anchor_timestamp"].astype(np.float64)
    if len(dense_ids) != 3718 or len(set(dense_ids.tolist())) != 3718:
        raise AssertionError("event secondary requires the fixed 3,718-query panel")

    dev_rows = [row for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8")) if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0"]
    meta_by_episode = {}
    for row in dev_rows:
        meta_by_episode.setdefault(int(row["episode_index"]), row)
    split_path = Path(json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))["split_manifest"])
    split_table = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(split_path, columns=["episode_index", "original_task_index", "split", "video_sha256_fingerprint"], filters=[("split", "=", "validation"), ("v3_benchmark_inclusion", "=", "primary")])
    source_info = {int(row["episode_index"]): {"task": int(row["original_task_index"]), "group": str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}")} for row in split_table.to_pylist()}
    panel_episodes = sorted(set(episodes.tolist()))
    if len(panel_episodes) != 118 or any(episode not in source_info or episode not in meta_by_episode for episode in panel_episodes):
        raise AssertionError("dense panel does not join to the locked 118 primary-dev source episodes")

    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float64)
    thresholds = np.asarray([[float(d2v3["thresholds"][body]["theta_on"]), float(d2v3["thresholds"][body]["theta_off"])] for body in event.GROUPS], dtype=np.float64)
    on = {body: float(thresholds[i, 0]) for i, body in enumerate(event.GROUPS)}
    off = {body: float(thresholds[i, 1]) for i, body in enumerate(event.GROUPS)}

    episode_data = {}
    truth_by_group_body = defaultdict(dict)
    states_by_episode_body = {}
    source_group_tasks = {}
    for episode in panel_episodes:
        raw = event.load_episode(meta_by_episode[episode], source_info[episode], sigma)
        episode_data[episode] = raw
        source_group_tasks[raw["group"]] = raw["task"]
        for body in event.GROUPS:
            trace = event.state_trace(raw["m"][body], 8, len(raw["timestamp"]), on[body], off[body])
            states_by_episode_body[(episode, body)] = trace
            for item in trace["events"]:
                onset = int(item["onset_frame"])
                event_id = f"{raw['group']}|{body}|{onset}"
                truth_by_group_body[(raw["group"], body)].setdefault(event_id, {
                    "event_id": event_id,
                    "source_group_id": raw["group"],
                    "episode_index": episode,
                    "task": raw["task"],
                    "body": body,
                    "onset_frame": onset,
                    "onset_time": float(raw["timestamp"][onset]),
                    "confirmation_frame": int(item["confirmation_frame"]),
                })

    query_records = []
    queries_by_group = defaultdict(list)
    opportunity = defaultdict(set)
    lead_opportunity = defaultdict(lambda: defaultdict(set))
    for index, (dense_id, episode, frame, task, query_time) in enumerate(zip(dense_ids, episodes, frames, tasks, query_times)):
        episode = int(episode)
        frame = int(frame)
        raw = episode_data[episode]
        if frame < 8 or frame + 60 >= len(raw["timestamp"]):
            raise AssertionError(f"D2_V3 event panel query has invalid H60 support: {dense_id}")
        if abs(float(raw["timestamp"][frame]) - float(query_time)) > 1e-5:
            raise AssertionError(f"event panel anchor time mismatch: {dense_id}")
        group = str(raw["group"])
        query = {
            "dense_id": str(dense_id), "episode_index": episode, "task": int(task),
            "source_group_id": group, "frame": frame, "query_time": float(query_time),
            "raw_index": index,
        }
        for body in event.GROUPS:
            state = states_by_episode_body[(episode, body)]
            query[f"known_{body}"] = bool(state["known"][frame])
            query[f"armed_{body}"] = bool(state["armed"][frame])
            for horizon in HORIZONS:
                query[f"full_h{horizon}"] = frame + horizon < len(raw["timestamp"])
                end_frame = frame + horizon - 2
                query[f"effective_end_h{horizon}"] = float(raw["timestamp"][end_frame])
        query_records.append(query)
        queries_by_group[group].append(query)
        for body in event.GROUPS:
            for horizon in HORIZONS:
                if not query[f"known_{body}"] or not query[f"full_h{horizon}"]:
                    continue
                end_time = float(query[f"effective_end_h{horizon}"])
                for event_id, truth in truth_by_group_body[(group, body)].items():
                    lead = float(truth["onset_time"]) - float(query_time)
                    if 0 < lead <= end_time - float(query_time) + 1e-8:
                        opportunity[(horizon, body, group)].add(event_id)
                        for threshold in (0.25, 0.5, 1.0):
                            if lead >= threshold:
                                lead_opportunity[(horizon, body, group)][threshold].add(event_id)

    group_ids = sorted(set(source_group_tasks))
    if len(group_ids) != 118:
        raise AssertionError("event panel source-group count differs from 118")
    old_predictions = load_prediction_grid(OLD_PRED_ROOT, dense_ids)
    new_predictions = load_prediction_grid(NEW_PRED_ROOT, dense_ids)
    expected = {(config, seed, horizon) for config in CONFIGS for seed in SEEDS for horizon in HORIZONS}
    if set(old_predictions) != expected or set(new_predictions) != expected:
        raise AssertionError("event-secondary old/new prediction grid must be 36/36")

    panel_queries = defaultdict(list)
    for query in query_records:
        panel_queries[query["source_group_id"]].append(query)
    exposure_seconds = {}
    for group, queries in panel_queries.items():
        exposure_seconds[group] = max(q["query_time"] for q in queries) - min(q["query_time"] for q in queries)
    if abs(sum(exposure_seconds.values()) / 3600.0 - 0.2496111027399699) > 1e-5:
        raise AssertionError("fixed panel exposure differs from the locked 0.249611h support")

    index_by_group = {group: i for i, group in enumerate(group_ids)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(group_ids), size=(BOOTSTRAPS, len(group_ids)), dtype=np.int32)
    group_rows = []
    summary_rows = []
    for study, prediction_grid in (("OLD_D3_ACTUAL", old_predictions), ("D3_EQ_V1", new_predictions)):
        for (config, seed, horizon), prediction in sorted(prediction_grid.items()):
            for body in event.GROUPS:
                alerts_by_group = defaultdict(list)
                known_queries = defaultdict(int)
                armed_queries = defaultdict(int)
                for query in query_records:
                    group = query["source_group_id"]
                    known = bool(query[f"known_{body}"])
                    horizon_complete = bool(query[f"full_h{horizon}"])
                    known_queries[group] += int(known and horizon_complete)
                    armed_queries[group] += int(bool(query[f"armed_{body}"]) and horizon_complete)
                    if not known or not horizon_complete:
                        continue
                    episode = int(query["episode_index"])
                    raw = episode_data[episode]
                    state = states_by_episode_body[(episode, body)]
                    forecasts, state_known_armed = event.predict_events(
                        prediction[int(query["raw_index"])], raw["action"], raw["timestamp"], raw["q"][body],
                        int(query["frame"]), state, on[body], off[body],
                    )
                    if not bool(state_known_armed[0]):
                        continue
                    end_time = float(query[f"effective_end_h{horizon}"])
                    for forecast in forecasts:
                        onset_time = float(forecast["predicted_onset_time"])
                        if float(query["query_time"]) < onset_time <= end_time + 1e-8:
                            alerts_by_group[group].append({
                                "source_group_id": group,
                                "episode_index": episode,
                                "task": int(query["task"]),
                                "body": body,
                                "dense_id": query["dense_id"],
                                "query_time": float(query["query_time"]),
                                "predicted_onset_time": onset_time,
                                "predicted_onset_frame": int(forecast["predicted_onset_frame"]),
                            })

                for group in group_ids:
                    clusters = event.cluster_alerts(alerts_by_group[group])
                    opportunities = opportunity[(horizon, body, group)]
                    truth_events = truth_by_group_body[(group, body)]
                    unmatched = set(opportunities)
                    hits = 0
                    timing_abs_sum = 0.0
                    timing_signed_sum = 0.0
                    timing_count = 0
                    for cluster in sorted(clusters, key=lambda x: (float(x["first_query_time"]), float(x["center_predicted_onset"]), str(x["first_dense_id"]))):
                        first_query = next(q for q in panel_queries[group] if q["dense_id"] == cluster["first_dense_id"])
                        end_time = float(first_query[f"effective_end_h{horizon}"])
                        candidates = []
                        for event_id in unmatched:
                            truth = truth_events[event_id]
                            lead = float(truth["onset_time"]) - float(cluster["first_query_time"])
                            error = abs(float(cluster["center_predicted_onset"]) - float(truth["onset_time"]))
                            if 0 < lead <= end_time - float(cluster["first_query_time"]) + 1e-8 and error <= float(event.TOLERANCE):
                                candidates.append((error, float(truth["onset_time"]), event_id, truth))
                        if candidates:
                            error_abs, _, event_id, truth = min(candidates, key=lambda x: (x[0], x[1], x[2]))
                            unmatched.remove(event_id)
                            hits += 1
                            signed = float(cluster["center_predicted_onset"]) - float(truth["onset_time"])
                            timing_abs_sum += abs(signed)
                            timing_signed_sum += signed
                            timing_count += 1
                    false_alarms = len(clusters) - hits
                    opp = len(opportunities)
                    exposure = float(exposure_seconds[group])
                    group_rows.append({
                        "study": study, "config": config, "seed": seed, "horizon_steps": horizon,
                        "body": body, "source_group_id": group,
                        "truth_event_records": len(truth_events),
                        "complete_opportunities": opp, "hits": hits, "misses": opp - hits,
                        "predicted_alarm_clusters": len(clusters), "false_alarms": false_alarms,
                        "exposure_seconds": exposure,
                        "timing_abs_sum": timing_abs_sum, "timing_signed_sum": timing_signed_sum,
                        "timing_count": timing_count,
                        "known_complete_queries": known_queries[group],
                        "armed_complete_queries": armed_queries[group],
                        "opportunities_lead_ge_025": len(lead_opportunity[(horizon, body, group)][0.25]),
                        "opportunities_lead_ge_05": len(lead_opportunity[(horizon, body, group)][0.5]),
                        "opportunities_lead_ge_1": len(lead_opportunity[(horizon, body, group)][1.0]),
                        "final_test_access": 0,
                    })

                rows = [r for r in group_rows if r["study"] == study and r["config"] == config and r["seed"] == seed and r["horizon_steps"] == horizon and r["body"] == body]
                opp = np.asarray([r["complete_opportunities"] for r in rows], dtype=np.float64)
                hits = np.asarray([r["hits"] for r in rows], dtype=np.float64)
                false = np.asarray([r["false_alarms"] for r in rows], dtype=np.float64)
                exposure = np.asarray([r["exposure_seconds"] for r in rows], dtype=np.float64)
                abs_sum = np.asarray([r["timing_abs_sum"] for r in rows], dtype=np.float64)
                signed_sum = np.asarray([r["timing_signed_sum"] for r in rows], dtype=np.float64)
                timing_n = np.asarray([r["timing_count"] for r in rows], dtype=np.float64)
                rec, rec_low, rec_high = bootstrap_rate(hits, opp, draws)
                fa, fa_low, fa_high = bootstrap_rate(false, exposure, draws, scale=3600.0)
                timing_abs, timing_abs_low, timing_abs_high = bootstrap_rate(abs_sum, timing_n, draws)
                timing_signed, timing_signed_low, timing_signed_high = bootstrap_rate(signed_sum, timing_n, draws)
                summary_rows.append({
                    "study": study, "config": config, "seed": seed, "horizon_steps": horizon, "body": body,
                    "source_groups": len(group_ids), "queries": len(dense_ids),
                    "truth_event_records": int(sum(r["truth_event_records"] for r in rows)),
                    "complete_opportunities": int(opp.sum()), "hits": int(hits.sum()), "misses": int(opp.sum() - hits.sum()),
                    "predicted_alarm_clusters": int(sum(r["predicted_alarm_clusters"] for r in rows)),
                    "false_alarms": int(false.sum()), "recall": rec, "recall_ci95_low": rec_low, "recall_ci95_high": rec_high,
                    "false_alarms_per_hour": fa, "false_alarms_per_hour_ci95_low": fa_low, "false_alarms_per_hour_ci95_high": fa_high,
                    "exposure_hours": float(exposure.sum() / 3600.0),
                    "timing_abs_error_seconds": timing_abs, "timing_abs_ci95_low": timing_abs_low, "timing_abs_ci95_high": timing_abs_high,
                    "timing_signed_error_seconds": timing_signed, "timing_signed_ci95_low": timing_signed_low, "timing_signed_ci95_high": timing_signed_high,
                    "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video_sha256_fingerprint source group",
                    "threshold_source": "locked D2_V3 primary-train thresholds; no dev refit",
                    "deployment_interpretation": "secondary development diagnostic on short 0.25h dense panel, not deployment alarm rate",
                    "final_test_access": 0,
                })

    write_csv(OUT / "D3_EQ_D2V3_EVENT_GROUP_AUDIT.csv", group_rows)
    write_csv(OUT / "D3_EQ_D2V3_EVENT_SECONDARY.csv", summary_rows)
    qa = {
        "status": "PASS",
        "d2_v3_protocol_sha256": sha256_file(EVENT_PROTOCOL),
        "d2_v3_event_code_sha256": sha256_file(EVENT_CODE),
        "old_dense_prediction_grid": 36,
        "d3_eq_dense_prediction_grid": 36,
        "queries": len(dense_ids), "source_groups": len(group_ids),
        "thresholds_refit": False,
        "target_labels_rederived_with_locked_definition": True,
        "event_state_machine": "D2_V3 8-step smoothed command-change, 6-quiet-step arming, 3-high-step onset backdating",
        "event_match": "one-to-one onset match within locked tolerance after first-query/effective-horizon eligibility; overlapping alarms clustered at fixed center",
        "exposure_hours": float(sum(exposure_seconds.values()) / 3600.0),
        "dev_targets_loaded_after_training": True,
        "final_test_access": 0,
    }
    atomic_json(OUT / "D3_EQ_D2V3_EVENT_SECONDARY_QA.json", qa)
    print(json.dumps(qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
