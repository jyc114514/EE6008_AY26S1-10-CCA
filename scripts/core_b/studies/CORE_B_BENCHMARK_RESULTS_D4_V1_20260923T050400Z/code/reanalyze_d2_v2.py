"""Corrected, group-aware sparse/dense event reanalysis on development data."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from event_v2 import GROUPS, cluster_alarms, full_truth_events, max_cardinality_match, observed_state, q_signal, replay_q
from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, REPO_ROOT, V3_ROOT

RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
SIGMA = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
LEADS = (0.25, 0.5, 1.0)
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260924


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_groups() -> dict[int, str]:
    table = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint"])
    return {int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}") for row in table.to_pylist() if row["split"] in {"train", "validation"}}


def load_episode(meta: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = (RAW_ROOT / meta["parquet_path"]).resolve()
    if not parquet_path.is_relative_to(RAW_ROOT.resolve()):
        raise ValueError(f"raw path escaped approved root: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["timestamp", "action"])
    timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 28 or len(timestamps) != len(action) or not np.isfinite(action).all() or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"raw action/timestamp contract failed: {parquet_path}")
    return timestamps, action


def make_query_state_cache(rows: list[dict[str, str]], episode_data: dict[int, tuple[np.ndarray, np.ndarray]], thresholds: np.ndarray):
    cache = {}
    for row in rows:
        ep, frame = int(row["episode_index"]), int(row["anchor_frame"])
        if (ep, frame) in cache:
            continue
        timestamps, action = episode_data[ep]
        start = max(0, frame - 119)
        state, _ = observed_state(action, timestamps, start, frame, SIGMA, thresholds)
        cache[(ep, frame)] = state
    return cache


def cached_forecast(action, timestamps, frame, pred, thresholds, initial_state):
    horizon = len(pred)
    combined = np.concatenate((action[frame : frame + 1], np.asarray(pred, dtype=np.float64)), axis=0)
    q = q_signal(combined, SIGMA)
    endpoints = np.arange(frame + 1, frame + horizon + 1, dtype=np.int64)
    state_at_query = deepcopy(initial_state)
    events, _, _ = replay_q(q, endpoints, timestamps, thresholds, initial=deepcopy(state_at_query))
    for event in events:
        event["query_frame"] = int(frame)
        event["query_timestamp"] = float(timestamps[frame])
    return events, state_at_query


def baseline_prediction(action: np.ndarray, frame: int, horizon: int, model: str) -> np.ndarray:
    if model == "PERSISTENCE":
        return np.repeat(action[frame : frame + 1], horizon, axis=0)
    if model == "COMMAND_SLOPE":
        slope = action[frame] - action[max(0, frame - 1)]
        return action[frame][None, :] + np.arange(1, horizon + 1, dtype=np.float64)[:, None] * slope[None, :]
    raise ValueError(model)


def group_state_is_eligible(state: list[dict[str, object]], group_index: int) -> bool:
    item = state[group_index]
    return bool(item["known"] and item["armed"])


def effective_end_frame(anchor: int, horizon: int) -> int:
    # Three consecutive q>theta_on steps confirm an onset; onset itself is two
    # command steps before the final confirming q endpoint.
    return anchor + horizon - 2


def eligible_queries_for_event(event, query_rows, group_index, horizon, timestamps, state_cache):
    out = []
    for row in query_rows:
        frame = int(row["anchor_frame"])
        state = state_cache[(int(row["episode_index"]), frame)]
        if not group_state_is_eligible(state, group_index):
            continue
        end = effective_end_frame(frame, horizon)
        if frame < int(event["onset_frame"]) <= end and int(event["confirm_frame"]) <= frame + horizon:
            out.append({"frame": frame, "query_timestamp": float(timestamps[frame]), "effective_end_timestamp": float(timestamps[end]), "lead_seconds": float(event["onset_timestamp"]) - float(timestamps[frame])})
    return out


def count_template():
    return {"opp_full": 0, "hits_full": 0, "false_alarms": 0, "predicted_clusters": 0, "matched_boundary": 0, "truth_in_block": 0, "timing_abs_sum": 0.0, "timing_signed_sum": 0.0, "timing_n": 0, "eligible_queries": 0, "opportunities_by_lead": {lead: 0 for lead in LEADS}, "hits_by_lead": {lead: 0 for lead in LEADS}}


def dense_episode_score(episode, panel_rows, predictions_by_id, pred_model, horizon, episode_data, truth_by_ep, state_cache, thresholds):
    timestamps, action = episode_data[episode]
    query_rows = sorted(panel_rows, key=lambda r: int(r["query_index"]))
    first_frame, last_frame = int(query_rows[0]["anchor_frame"]), int(query_rows[-1]["anchor_frame"])
    first_time, last_time = float(timestamps[first_frame]), float(timestamps[last_frame])
    horizon_duration = float(timestamps[first_frame + horizon] - timestamps[first_frame])
    exposure = max(last_time - first_time, 0.0)
    result = {group: count_template() for group in GROUPS}
    result["All"] = count_template()
    result["exposure_seconds"] = exposure
    event_slices = []
    alarms_by_group = {group: [] for group in GROUPS}

    for row in query_rows:
        frame = int(row["anchor_frame"])
        dense_id = row["dense_id"]
        if pred_model in {"PERSISTENCE", "COMMAND_SLOPE"}:
            pred = baseline_prediction(action, frame, horizon, pred_model)
        else:
            pred = predictions_by_id[dense_id]
        events, qstate = cached_forecast(action, timestamps, frame, pred, thresholds, state_cache[(episode, frame)])
        for group_index, group in enumerate(GROUPS):
            if not group_state_is_eligible(qstate, group_index):
                continue
            result[group]["eligible_queries"] += 1
            for event in events:
                if event["group"] == group and int(event["onset_frame"]) > frame:
                    event["effective_end_frame"] = effective_end_frame(frame, horizon)
                    event["query_frame"] = frame
                    alarms_by_group[group].append(event)

    per_episode_matches = {group: [] for group in GROUPS}
    for group_index, group in enumerate(GROUPS):
        truth = [event for event in truth_by_ep[episode] if event["group"] == group]
        in_block = [event for event in truth if float(event["onset_timestamp"]) < last_time]
        result[group]["truth_in_block"] = len(in_block)
        result["All"]["truth_in_block"] += len(in_block)
        opportunity = {}
        for event in in_block:
            q_windows = eligible_queries_for_event(event, query_rows, group_index, horizon, timestamps, state_cache)
            if not q_windows:
                continue
            # Main denominator requires H seconds of legal dense history inside
            # this fixed block plus one eligible query that can confirm onset.
            if float(event["onset_timestamp"]) - first_time >= horizon_duration:
                opportunity[int(event["onset_frame"])] = q_windows
        for event_id, q_windows in opportunity.items():
            result[group]["opp_full"] += 1
            for lead in LEADS:
                if any(q["lead_seconds"] >= lead for q in q_windows):
                    result[group]["opportunities_by_lead"][lead] += 1
        clusters = cluster_alarms(alarms_by_group[group], 0.2)
        result[group]["predicted_clusters"] = len(clusters)
        all_matchable = [event for event in truth if eligible_queries_for_event(event, query_rows, group_index, horizon, timestamps, state_cache)]

        def pair_eligible(i, j, cluster, event):
            return bool(eligible_queries_for_event(event, cluster["alarms"], group_index, horizon, timestamps, state_cache))

        # Candidate event eligibility uses the alarm cluster's own query set.
        def pair_eligible(i, j, cluster, event):
            valid = []
            for alarm in cluster["alarms"]:
                q = int(alarm["query_frame"])
                if q < int(event["onset_frame"]) <= int(alarm["effective_end_frame"]) and int(event["confirm_frame"]) <= q + horizon:
                    valid.append(q)
            return bool(valid)

        matches = max_cardinality_match(clusters, all_matchable, 0.2, eligible=pair_eligible)
        matched_cluster_indices = set()
        for pi, ti, timing_error in matches:
            matched_cluster_indices.add(pi)
            event = all_matchable[ti]
            candidate_queries = [int(alarm["query_frame"]) for alarm in clusters[pi]["alarms"] if int(alarm["query_frame"]) < int(event["onset_frame"]) <= int(alarm["effective_end_frame"]) and int(event["confirm_frame"]) <= int(alarm["query_frame"]) + horizon]
            if not candidate_queries:
                continue
            query_frame = min(candidate_queries)
            lead_seconds = float(event["onset_timestamp"]) - float(timestamps[query_frame])
            if int(event["onset_frame"]) in opportunity:
                result[group]["hits_full"] += 1
                result[group]["timing_abs_sum"] += abs(timing_error)
                result[group]["timing_signed_sum"] += timing_error
                result[group]["timing_n"] += 1
                for lead in LEADS:
                    if lead_seconds >= lead and any(q["lead_seconds"] >= lead for q in opportunity[int(event["onset_frame"])]):
                        result[group]["hits_by_lead"][lead] += 1
                per_episode_matches[group].append({"event": event, "lead": lead_seconds, "timing_error": timing_error})
            else:
                result[group]["matched_boundary"] += 1
        result[group]["false_alarms"] = len(clusters) - len(matches)
        for key in ("opp_full", "hits_full", "false_alarms", "predicted_clusters", "matched_boundary", "truth_in_block", "timing_abs_sum", "timing_signed_sum", "timing_n", "eligible_queries"):
            result["All"][key] += result[group][key]
        for lead in LEADS:
            result["All"]["opportunities_by_lead"][lead] += result[group]["opportunities_by_lead"][lead]
            result["All"]["hits_by_lead"][lead] += result[group]["hits_by_lead"][lead]
    result["All"]["false_alarms_per_hour"] = None
    return result


def make_bootstrap_counts(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAPS, n), dtype=np.int16)
    counts = np.zeros((BOOTSTRAPS, n), dtype=np.uint16)
    row_index = np.broadcast_to(np.arange(BOOTSTRAPS, dtype=np.int32)[:, None], draws.shape)
    np.add.at(counts, (row_index, draws), 1)
    return counts


def bootstrap_ratio(numer: np.ndarray, denom: np.ndarray, counts: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    total_n = float(numer.sum())
    total_d = float(denom.sum())
    estimate = total_n / total_d if total_d > 0 else float("nan")
    bn = counts @ numer.astype(np.float64)
    bd = counts @ denom.astype(np.float64)
    reps = np.divide(bn, bd, out=np.full(len(counts), np.nan), where=bd > 0)
    finite = reps[np.isfinite(reps)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return estimate, float(lo), float(hi), reps


def bootstrap_rate(numer: np.ndarray, exposure: np.ndarray, counts: np.ndarray, scale: float) -> tuple[float, float, float, np.ndarray]:
    total_e = float(exposure.sum())
    estimate = float(numer.sum() / total_e * scale) if total_e > 0 else float("nan")
    bn = counts @ numer.astype(np.float64)
    be = counts @ exposure.astype(np.float64)
    reps = np.divide(bn * scale, be, out=np.full(len(counts), np.nan), where=be > 0)
    finite = reps[np.isfinite(reps)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return estimate, float(lo), float(hi), reps


def main() -> None:
    thresholds_doc = json.loads((NEXT_ROOT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
    thresholds = np.column_stack((thresholds_doc["theta_on"], thresholds_doc["theta_off"]))
    source_map = source_groups()
    anchor_manifest = read_csv(V3_ROOT / "ANCHOR_MANIFEST.csv")
    meta_by_episode = {}
    for row in anchor_manifest:
        if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0":
            meta_by_episode.setdefault(int(row["episode_index"]), row)
    dev_npz = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    dev_indices = np.flatnonzero(dev_npz["split_code"] == 1)
    anchor_to_row = {str(dev_npz["anchor_id"][i]): int(i) for i in dev_indices}
    dev_episodes = sorted({int(dev_npz["episode_index"][i]) for i in dev_indices})
    episode_data = {ep: load_episode(meta_by_episode[ep]) for ep in dev_episodes}
    truth_by_ep = {ep: full_truth_events(episode_data[ep][1], episode_data[ep][0], SIGMA, thresholds) for ep in dev_episodes}
    truth_registry = []
    for ep in dev_episodes:
        for event in truth_by_ep[ep]:
            truth_registry.append({"episode_index": ep, "source_group_id": source_map.get(ep, f"episode:{ep}"), "group": event["group"], "onset_frame": event["onset_frame"], "confirm_frame": event["confirm_frame"], "onset_timestamp": event["onset_timestamp"], "confirm_timestamp": event["confirm_timestamp"], "signal_at_onset": event["signal_at_onset"], "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D2_EVENT_REGISTRY_V2.csv", truth_registry)

    dense_panel = read_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    dense_rows = [{**row, "dense_id": f"dense_e{int(row['episode_index']):06d}_a{int(row['anchor_frame']):06d}"} for row in dense_panel]
    dense_by_episode = defaultdict(list)
    for row in dense_rows:
        dense_by_episode[int(row["episode_index"])].append(row)
    dense_state_cache = make_query_state_cache(dense_rows, episode_data, thresholds)
    sparse_rows = [row for row in anchor_manifest if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0"]
    sparse_state_cache = make_query_state_cache(sparse_rows, episode_data, thresholds)

    # Fixed development panel audit: task-balanced by design, not natural prevalence.
    task_ep = defaultdict(set)
    task_queries = defaultdict(int)
    for row in dense_rows:
        ep, task = int(row["episode_index"]), int(row["task"])
        task_ep[task].add(ep)
        task_queries[task] += 1
    coverage_rows = [{"task": task, "episodes": len(task_ep[task]), "queries": task_queries[task], "source_groups": len({source_map.get(ep, f"episode:{ep}") for ep in task_ep[task]}), "panel_population": "task_balanced_dense_development", "final_test_access": 0} for task in sorted(task_ep)]
    write_csv(NEXT_ROOT / "results/D2_DENSE_PANEL_TASK_COVERAGE.csv", coverage_rows)

    dense_models = []
    for path in sorted((BENCH_ROOT / "predictions_dense").glob("V3-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["dense_id"].astype("U96")
            predictions = archive["predictions_native"]
            horizon = int(archive["horizon_steps"])
            cell = str(archive["cell"].item())
            seed = int(archive["seed"].item())
        if len(ids) != len(dense_rows) or set(ids.tolist()) != {row["dense_id"] for row in dense_rows}:
            raise ValueError(f"dense prediction ID mismatch: {path.name}")
        lookup = {str(value): i for i, value in enumerate(ids)}
        pred_map = {row["dense_id"]: predictions[lookup[row["dense_id"]]] for row in dense_rows}
        dense_models.append((cell, seed, horizon, pred_map, path.name))
    if len(dense_models) != 36:
        raise ValueError(f"expected 36 accepted dense V3 bundles, found {len(dense_models)}")

    dense_stats: dict[tuple[str, int, int], dict[int, dict[str, dict]]] = {}
    for baseline in ("PERSISTENCE", "COMMAND_SLOPE"):
        for horizon in (30, 60):
            key = (baseline, 0, horizon)
            dense_stats[key] = {}
            for ep, rows in dense_by_episode.items():
                dense_stats[key][ep] = dense_episode_score(ep, rows, {}, baseline, horizon, episode_data, truth_by_ep, dense_state_cache, thresholds)
    for cell, seed, horizon, pred_map, _ in dense_models:
        key = (cell, seed, horizon)
        dense_stats[key] = {}
        for ep, rows in dense_by_episode.items():
            dense_stats[key][ep] = dense_episode_score(ep, rows, pred_map, cell, horizon, episode_data, truth_by_ep, dense_state_cache, thresholds)

    dense_summary_rows = []
    source_group_ids = sorted({source_map.get(ep, f"episode:{ep}") for ep in dense_by_episode})
    group_to_idx = {g: i for i, g in enumerate(source_group_ids)}
    counts = make_bootstrap_counts(len(source_group_ids))
    metric_reps = {}
    for key, ep_stats in dense_stats.items():
        model, seed, horizon = key
        for group in (*GROUPS, "All"):
            per_group = {g: count_template() for g in source_group_ids}
            for ep, stats in ep_stats.items():
                gid = source_map.get(ep, f"episode:{ep}")
                dest = per_group[gid]
                source = stats[group]
                for field in ("opp_full", "hits_full", "false_alarms", "predicted_clusters", "matched_boundary", "truth_in_block", "timing_abs_sum", "timing_signed_sum", "timing_n", "eligible_queries"):
                    dest[field] += source[field]
                for lead in LEADS:
                    dest["opportunities_by_lead"][lead] += source["opportunities_by_lead"][lead]
                    dest["hits_by_lead"][lead] += source["hits_by_lead"][lead]
            n_opp = np.asarray([per_group[g]["opp_full"] for g in source_group_ids], dtype=np.float64)
            hits = np.asarray([per_group[g]["hits_full"] for g in source_group_ids], dtype=np.float64)
            fa = np.asarray([per_group[g]["false_alarms"] for g in source_group_ids], dtype=np.float64)
            exposure = np.asarray([sum(ep_stats[ep]["exposure_seconds"] for ep in ep_stats if source_map.get(ep, f"episode:{ep}") == g) for g in source_group_ids], dtype=np.float64)
            recall, recall_lo, recall_hi, recall_reps = bootstrap_ratio(hits, n_opp, counts)
            fa_rate, fa_lo, fa_hi, fa_reps = bootstrap_rate(fa, exposure, counts, 3600.0)
            row = {"model": model, "seed": seed if seed else "NOT_APPLICABLE", "horizon_steps": horizon, "event_group": group, "episodes": len(ep_stats), "source_groups": len(source_group_ids), "queries": sum(len(rows) for rows in dense_by_episode.values()), "eligible_queries": int(sum(ep_stats[ep][group]["eligible_queries"] for ep in ep_stats)), "truth_events_in_block": int(sum(ep_stats[ep][group]["truth_in_block"] for ep in ep_stats)), "full_opportunity_events": int(n_opp.sum()), "hits": int(hits.sum()), "misses": int(n_opp.sum() - hits.sum()), "boundary_matches_excluded": int(sum(ep_stats[ep][group]["matched_boundary"] for ep in ep_stats)), "predicted_forecast_clusters": int(sum(ep_stats[ep][group]["predicted_clusters"] for ep in ep_stats)), "false_alarms": int(fa.sum()), "recall": recall, "recall_ci95_low": recall_lo, "recall_ci95_high": recall_hi, "false_alarms_per_hour": fa_rate, "false_alarms_per_hour_ci95_low": fa_lo, "false_alarms_per_hour_ci95_high": fa_hi, "exposure_hours": float(exposure.sum() / 3600.0), "timing_abs_error_seconds": (sum(ep_stats[ep][group]["timing_abs_sum"] for ep in ep_stats) / sum(ep_stats[ep][group]["timing_n"] for ep in ep_stats)) if sum(ep_stats[ep][group]["timing_n"] for ep in ep_stats) else None, "timing_signed_error_seconds": (sum(ep_stats[ep][group]["timing_signed_sum"] for ep in ep_stats) / sum(ep_stats[ep][group]["timing_n"] for ep in ep_stats)) if sum(ep_stats[ep][group]["timing_n"] for ep in ep_stats) else None, "panel_population": "task_balanced_dense_development_not_natural_prevalence", "bootstrap_unit": "video_sha256_fingerprint source group", "final_test_access": 0}
            for lead in LEADS:
                lead_hits = np.asarray([per_group[g]["hits_by_lead"][lead] for g in source_group_ids], dtype=np.float64)
                lead_opp = np.asarray([per_group[g]["opportunities_by_lead"][lead] for g in source_group_ids], dtype=np.float64)
                value, lo, hi, reps = bootstrap_ratio(lead_hits, lead_opp, counts)
                suffix = str(lead).replace(".", "p")
                row[f"opportunity_events_lead_ge_{suffix}"] = int(lead_opp.sum())
                row[f"hits_lead_ge_{suffix}"] = int(lead_hits.sum())
                row[f"recall_lead_ge_{suffix}"] = value
                row[f"recall_lead_ge_{suffix}_ci95_low"] = lo
                row[f"recall_lead_ge_{suffix}_ci95_high"] = hi
            dense_summary_rows.append(row)
            metric_reps[(key, group)] = {"recall": recall_reps, "fa_rate": fa_reps}
    write_csv(NEXT_ROOT / "results/D2_DENSE_RESULTS_V2.csv", dense_summary_rows)

    paired_rows = []
    for cell, seed, horizon, _, _ in dense_models:
        for baseline in ("PERSISTENCE", "COMMAND_SLOPE", "S-HIST"):
            base_key = (baseline, 0, horizon) if baseline != "S-HIST" else (baseline, seed, horizon)
            model_key = (cell, seed, horizon)
            for group in (*GROUPS, "All"):
                model_row = next(row for row in dense_summary_rows if row["model"] == cell and row["seed"] == seed and row["horizon_steps"] == horizon and row["event_group"] == group)
                base_row = next(row for row in dense_summary_rows if row["model"] == baseline and row["seed"] == ("NOT_APPLICABLE" if baseline != "S-HIST" else seed) and row["horizon_steps"] == horizon and row["event_group"] == group)
                rep = metric_reps[(model_key, group)]
                bref = metric_reps[(base_key, group)]
                recall_diff = rep["recall"] - bref["recall"]
                fa_diff = rep["fa_rate"] - bref["fa_rate"]
                finite_r = recall_diff[np.isfinite(recall_diff)]
                finite_f = fa_diff[np.isfinite(fa_diff)]
                paired_rows.append({"model": cell, "seed": seed, "baseline": baseline, "horizon_steps": horizon, "event_group": group, "recall_difference": model_row["recall"] - base_row["recall"] if model_row["recall"] is not None and base_row["recall"] is not None else None, "recall_diff_ci95_low": float(np.quantile(finite_r, .025)) if len(finite_r) else None, "recall_diff_ci95_high": float(np.quantile(finite_r, .975)) if len(finite_r) else None, "false_alarms_per_hour_difference": model_row["false_alarms_per_hour"] - base_row["false_alarms_per_hour"], "false_alarm_diff_ci95_low": float(np.quantile(finite_f, .025)) if len(finite_f) else None, "false_alarm_diff_ci95_high": float(np.quantile(finite_f, .975)) if len(finite_f) else None, "paired_source_groups": len(source_group_ids), "final_test_access": 0})
    write_csv(NEXT_ROOT / "results/D2_DENSE_PAIRED_EFFECTS_V2.csv", paired_rows)

    # Sparse primary anchors: report query-event opportunities, not unique event
    # prevalence and not overlapping-window false alarms/hour.
    sparse_output = []
    sparse_runs = []
    for marker_path in sorted((V3_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS":
            continue
        cell, seed = str(marker["cell"]), int(marker["seed"])
        with np.load(marker_path.parent / "primary.npz", allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64")
            predictions = archive["predictions_native"]
            horizon = int(archive["horizon_steps"])
        if len(ids) != len(anchor_to_row) or set(ids.tolist()) != set(anchor_to_row):
            raise ValueError(f"sparse prediction IDs differ from primary dev anchors: {marker_path.parent.name}")
        pred_lookup = {str(value): i for i, value in enumerate(ids)}
        sparse_runs.append((cell, seed, horizon, marker_path.parent.name, predictions, pred_lookup))
    if len(sparse_runs) != 36:
        raise ValueError(f"expected 36 accepted sparse V3 prediction bundles, found {len(sparse_runs)}")

    sparse_query_by_ep = defaultdict(list)
    for row in sparse_rows:
        sparse_query_by_ep[int(row["episode_index"])].append(row)
    sparse_support_by_horizon = {}
    for h in (30, 60):
        support_episodes = sorted(ep for ep, rows in sparse_query_by_ep.items() if any(int(row["anchor_frame"]) + h < len(episode_data[ep][1]) for row in rows))
        groups = sorted({source_map.get(ep, f"episode:{ep}") for ep in support_episodes})
        sparse_support_by_horizon[h] = {"episodes": support_episodes, "groups": groups, "group_index": {g: i for i, g in enumerate(groups)}, "bootstrap_counts": make_bootstrap_counts(len(groups))}
    for cell, seed, horizon, run_name, predictions, pred_lookup in sparse_runs:
        ep_counts = defaultdict(lambda: {group: count_template() for group in (*GROUPS, "All")})
        sparse_support = sparse_support_by_horizon[horizon]
        for ep in sparse_support["episodes"]:
            ep_counts[ep]
        for anchor_id, source_index in anchor_to_row.items():
            ep, frame = int(dev_npz["episode_index"][source_index]), int(dev_npz["anchor_frame"][source_index])
            timestamps, action = episode_data[ep]
            if frame + horizon >= len(action):
                continue
            pred = predictions[pred_lookup[anchor_id]]
            events, qstate = cached_forecast(action, timestamps, frame, pred, thresholds, sparse_state_cache[(ep, frame)])
            truth = [e for e in truth_by_ep[ep] if int(e["onset_frame"]) > frame and int(e["confirm_frame"]) <= frame + horizon]
            for gi, group in enumerate(GROUPS):
                if not group_state_is_eligible(qstate, gi):
                    continue
                truth_g = [e for e in truth if e["group"] == group]
                pred_g = [e for e in events if e["group"] == group and int(e["onset_frame"]) > frame]
                dest = ep_counts[ep][group]
                dest["eligible_queries"] += 1
                dest["opp_full"] += len(truth_g)
                matches = max_cardinality_match(pred_g, truth_g, 0.2)
                dest["hits_full"] += len(matches)
                dest["false_alarms"] += len(pred_g) - len(matches)
                dest["predicted_clusters"] += len(pred_g)
                dest["timing_abs_sum"] += sum(abs(x[2]) for x in matches)
                dest["timing_signed_sum"] += sum(x[2] for x in matches)
                dest["timing_n"] += len(matches)
                for lead in LEADS:
                    opp = [e for e in truth_g if float(e["onset_timestamp"]) - float(timestamps[frame]) >= lead]
                    dest["opportunities_by_lead"][lead] += len(opp)
                    matched_truth_idx = {ti for pi, ti, err in matches if float(truth_g[ti]["onset_timestamp"]) - float(timestamps[frame]) >= lead}
                    dest["hits_by_lead"][lead] += len(matched_truth_idx)
        for ep in sorted(ep_counts):
            all_dest = ep_counts[ep]["All"]
            for group in GROUPS:
                source = ep_counts[ep][group]
                for field in ("opp_full", "hits_full", "false_alarms", "predicted_clusters", "eligible_queries", "timing_abs_sum", "timing_signed_sum", "timing_n"):
                    all_dest[field] += source[field]
                for lead in LEADS:
                    all_dest["opportunities_by_lead"][lead] += source["opportunities_by_lead"][lead]
                    all_dest["hits_by_lead"][lead] += source["hits_by_lead"][lead]
        # Add per-episode rows; dense blocks are not the sparse evaluation unit.
        ep_ids = sorted(ep_counts)
        sgroups = sparse_support["groups"]
        if {source_map.get(ep, f"episode:{ep}") for ep in ep_ids} != set(sgroups):
            raise ValueError("sparse V3 run did not cover the fixed legal horizon-support source groups")
        gindex = sparse_support["group_index"]
        bcounts = sparse_support["bootstrap_counts"]
        for group in (*GROUPS, "All"):
            nopp = np.zeros(len(sgroups)); hits = np.zeros(len(sgroups)); fap = np.zeros(len(sgroups))
            by_g = {g: count_template() for g in sgroups}
            for ep in ep_ids:
                g = source_map.get(ep, f"episode:{ep}")
                src = ep_counts[ep][group]
                for field in ("opp_full", "hits_full", "false_alarms", "predicted_clusters", "eligible_queries", "timing_abs_sum", "timing_signed_sum", "timing_n"):
                    by_g[g][field] += src[field]
                for lead in LEADS:
                    by_g[g]["opportunities_by_lead"][lead] += src["opportunities_by_lead"][lead]
                    by_g[g]["hits_by_lead"][lead] += src["hits_by_lead"][lead]
            for g, i in gindex.items():
                nopp[i], hits[i], fap[i] = by_g[g]["opp_full"], by_g[g]["hits_full"], by_g[g]["false_alarms"]
            recall, lo, hi, _ = bootstrap_ratio(hits, nopp, bcounts)
            row = {"model": cell, "seed": seed, "horizon_steps": horizon, "event_group": group, "episodes": len(ep_ids), "source_groups": len(sgroups), "query_event_opportunities": int(nopp.sum()), "hits": int(hits.sum()), "misses": int(nopp.sum() - hits.sum()), "false_alarms_per_overlapping_query_sum": int(fap.sum()), "recall_query_event": recall, "recall_ci95_low": lo, "recall_ci95_high": hi, "false_alarms_per_hour": None, "timing_abs_error_seconds": (sum(by_g[g]["timing_abs_sum"] for g in sgroups) / sum(by_g[g]["timing_n"] for g in sgroups)) if sum(by_g[g]["timing_n"] for g in sgroups) else None, "denominator_note": "query-event opportunities repeat events across overlapping sparse anchors; no false-alarms/hour claimed", "final_test_access": 0}
            for lead in LEADS:
                n = np.asarray([by_g[g]["opportunities_by_lead"][lead] for g in sgroups], dtype=np.float64)
                h = np.asarray([by_g[g]["hits_by_lead"][lead] for g in sgroups], dtype=np.float64)
                value, lci, hci, _ = bootstrap_ratio(h, n, bcounts)
                suffix = str(lead).replace(".", "p")
                row[f"opportunities_lead_ge_{suffix}"] = int(n.sum())
                row[f"hits_lead_ge_{suffix}"] = int(h.sum())
                row[f"recall_lead_ge_{suffix}"] = value
                row[f"recall_lead_ge_{suffix}_ci95_low"] = lci
                row[f"recall_lead_ge_{suffix}_ci95_high"] = hci
            sparse_output.append(row)
    write_csv(NEXT_ROOT / "results/D2_SPARSE_RESULTS_V2.csv", sparse_output)

    summary = {"status": "D2_EVENT_REANALYSIS_V2_COMPLETE", "protocol_hash": thresholds_doc["protocol_hash"], "thresholds_train_only": {"theta_on_hand_arm_order": thresholds_doc["theta_on"], "theta_off_hand_arm_order": thresholds_doc["theta_off"], "train_episodes": thresholds_doc["train_episodes"], "timestamp_dt_max_abs_error_seconds": thresholds_doc["timestamp_check_max_abs_dt_error_seconds"]}, "dev_episodes_for_truth_registry": len(dev_episodes), "corrected_truth_events_group_specific": len(truth_registry), "sparse_prediction_bundles": len(sparse_runs), "sparse_dev_rows": len(sparse_rows), "dense_prediction_bundles": len(dense_models), "dense_panel_episodes": len(dense_by_episode), "dense_panel_queries": len(dense_rows), "dense_task_count": len(task_ep), "dense_panel_population": "task_balanced_dense_development_not_natural_prevalence", "dense_source_groups": len(source_group_ids), "dense_results_rows": len(dense_summary_rows), "paired_effect_rows": len(paired_rows), "sparse_results_rows": len(sparse_output), "lead_thresholds_seconds": list(LEADS), "cluster_bootstraps": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED, "final_test_access": 0}
    (NEXT_ROOT / "results/D2_EVENT_REANALYSIS_V2_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
