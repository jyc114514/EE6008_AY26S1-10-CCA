"""D4 rolling-update evaluation on accepted V3 dense development predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from event_v2 import GROUPS, full_truth_events, observed_state
from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, REPO_ROOT, V3_ROOT

RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
SIGMA = np.asarray(json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())["target_sigma"], dtype=np.float64)
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260926
CONTRASTS = {
    "video": ("F-TOK-D", "S-HIST"),
    "temporal": ("F-TOK-D", "F-POOL-D"),
    "dynamic": ("F-TOK-D", "F-TOK-S"),
}
BODIES = {"Hand": slice(0, 14), "Arm": slice(14, 28), "Full28": slice(0, 28)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v) for k, v in row.items()} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_groups() -> dict[int, str]:
    table = pq.read_table(SPLIT, columns=["episode_index", "split", "video_sha256_fingerprint"])
    return {int(r["episode_index"]): str(r["video_sha256_fingerprint"] or f"episode:{r['episode_index']}") for r in table.to_pylist() if r["split"] in {"train", "validation"}}


def load_episode(meta: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = (RAW_ROOT / meta["parquet_path"]).resolve()
    if not parquet_path.is_relative_to(RAW_ROOT.resolve()):
        raise ValueError(f"raw action path escapes allowlisted root: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["timestamp", "action"])
    timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 28 or len(actions) != len(timestamps) or not np.isfinite(actions).all() or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"raw action/timestamp contract failed: {parquet_path}")
    return timestamps, actions


def valid_common_frames(old_frame: int, new_frame: int, horizon: int, n_frames: int) -> np.ndarray:
    if new_frame <= old_frame:
        raise ValueError("rolling pair must have new_frame > old_frame")
    if new_frame - old_frame not in {7, 8}:
        raise ValueError("D4 panel anchor interval must retain its measured 7/8-frame gap")
    start, end = new_frame + 1, min(old_frame + horizon, n_frames - 1)
    if end < start:
        return np.asarray([], dtype=np.int64)
    frames = np.arange(start, end + 1, dtype=np.int64)
    if np.any(frames - old_frame - 1 >= horizon) or np.any(frames - new_frame - 1 >= horizon):
        raise ValueError("common target frame mapped outside native forecast horizon")
    return frames


def validate_query_panel(rows: list[dict[str, str]]) -> None:
    seen = set()
    by_block = defaultdict(list)
    for row in rows:
        key = (int(row["episode_index"]), row["block_id"], int(row["query_index"]))
        if key in seen:
            raise ValueError(f"duplicate dense query identity: {key}")
        seen.add(key)
        by_block[(key[0], key[1])].append((key[2], int(row["anchor_frame"])))
    for block, values in by_block.items():
        ordered = sorted(values)
        if [query for query, _ in ordered] != list(range(len(ordered))):
            raise ValueError(f"dense query indices are not contiguous within block: {block}")
        gaps = [right[1] - left[1] for left, right in zip(ordered, ordered[1:])]
        if any(gap not in {7, 8} for gap in gaps):
            raise ValueError(f"dense anchor gap is not an observed 7/8-frame interval: {block}:{gaps[:8]}")


def pair_metrics(old_pred: np.ndarray, new_pred: np.ndarray, truth: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    if old_pred.shape != new_pred.shape or old_pred.shape != truth.shape or old_pred.ndim != 2 or old_pred.shape[1] != 28:
        raise ValueError("old/new/GT common-support arrays must share shape (N,28)")
    if not len(old_pred):
        raise ValueError("D4 pair has empty common future support")
    result = {}
    for body, sl in BODIES.items():
        o, n, y, scale = old_pred[:, sl], new_pred[:, sl], truth[:, sl], sigma[sl]
        e_old = np.abs(o - y) / scale[None, :]
        e_new = np.abs(n - y) / scale[None, :]
        result[f"{body}|L_old"] = float(e_old.mean())
        result[f"{body}|L_new"] = float(e_new.mean())
        result[f"{body}|U"] = float(e_old.mean() - e_new.mean())
        result[f"{body}|R"] = float(np.mean(np.abs(o - n) / scale[None, :]))
        result[f"{body}|native_MAE_old"] = float(np.mean(np.abs(o - y)))
        result[f"{body}|native_MAE_new"] = float(np.mean(np.abs(n - y)))
        if len(o) > 1:
            result[f"{body}|FD_old"] = float(np.mean(np.abs(np.diff(o, axis=0) - np.diff(y, axis=0))))
            result[f"{body}|FD_new"] = float(np.mean(np.abs(np.diff(n, axis=0) - np.diff(y, axis=0))))
        else:
            result[f"{body}|FD_old"] = float("nan")
            result[f"{body}|FD_new"] = float("nan")
        result[f"{body}|endpoint_old"] = float(np.mean(np.abs(o[-1] - y[-1]) / scale))
        result[f"{body}|endpoint_new"] = float(np.mean(np.abs(n[-1] - y[-1]) / scale))
        lower = e_new < e_old - 1e-12
        higher = e_new > e_old + 1e-12
        result[f"{body}|coord_error_lower_share"] = float(lower.mean())
        result[f"{body}|coord_error_higher_share"] = float(higher.mean())
        result[f"{body}|coord_error_tie_share"] = float((~lower & ~higher).mean())
    return result


def bootstrap_counts(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAPS, n), dtype=np.int32)
    out = np.zeros((BOOTSTRAPS, n), dtype=np.uint16)
    row_ids = np.broadcast_to(np.arange(BOOTSTRAPS, dtype=np.int32)[:, None], draws.shape)
    np.add.at(out, (row_ids, draws), 1)
    return out


def group_mean(pair_values: np.ndarray, pair_groups: np.ndarray, ordered_groups: list[str]) -> np.ndarray:
    result = np.full(len(ordered_groups), np.nan, dtype=np.float64)
    for index, group in enumerate(ordered_groups):
        local = pair_groups == group
        if local.any():
            result[index] = float(np.mean(pair_values[local]))
    return result


def summarize_groups(group_values: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(group_values)
    if not valid.any():
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(group_values[valid]))
    numerator = counts @ np.nan_to_num(group_values, nan=0.0)
    denominator = counts @ valid.astype(np.uint16)
    reps = np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)
    finite = reps[np.isfinite(reps)]
    if not len(finite):
        return point, float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return point, float(low), float(high)


def metric_row(metric, values, pair_groups, ordered_groups, counts, metadata):
    by_group = group_mean(values, pair_groups, ordered_groups)
    estimate, low, high = summarize_groups(by_group, counts)
    return {**metadata, "metric": metric, "estimate": estimate, "ci95_low": low, "ci95_high": high, "N_source_groups": int(np.isfinite(by_group).sum()), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_unit": "video_sha256_fingerprint source group"}, by_group


def metric_keys() -> list[str]:
    keys = []
    for body in BODIES:
        keys.extend(f"{body}|{metric}" for metric in ("L_old", "L_new", "U", "R", "native_MAE_old", "native_MAE_new", "FD_old", "FD_new", "endpoint_old", "endpoint_new", "coord_error_lower_share", "coord_error_higher_share", "coord_error_tie_share"))
    return keys


def main() -> None:
    start_wall = time.time()
    protocol = json.loads((NEXT_ROOT / "protocol/D4_PROTOCOL_LOCKED.json").read_text())
    thresholds_doc = json.loads((NEXT_ROOT / "protocol/D2_EVENT_REANALYSIS_V2.json").read_text())
    thresholds = np.column_stack((thresholds_doc["theta_on"], thresholds_doc["theta_off"]))
    input_manifest_path = NEXT_ROOT / "d4/D4_PAIR_MANIFEST_INPUT_ONLY.csv"
    locked_pairs = read_csv(input_manifest_path)
    if len(locked_pairs) != 7200:
        raise ValueError(f"locked D4 input-only manifest must contain 7200 rows, found {len(locked_pairs)}")
    panel = read_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    validate_query_panel(panel)
    panel_by_id = {f"dense_e{int(r['episode_index']):06d}_a{int(r['anchor_frame']):06d}": r for r in panel}
    base_pairs = {}
    for row in locked_pairs:
        key = (row["pair_id"], int(row["horizon_steps"]))
        base_pairs[key] = row
    pairs_by_h = {h: [r for r in locked_pairs if int(r["horizon_steps"]) == h and int(r["common_target_steps"]) > 0] for h in (30, 60)}
    if len(pairs_by_h[30]) != len(pairs_by_h[60]) or not pairs_by_h[30]:
        raise ValueError("H30/H60 D4 pair population mismatch or empty pair set")
    if len(base_pairs) != len(locked_pairs):
        raise ValueError("locked D4 input-only manifest contains duplicate pair/horizon keys")
    if len(pairs_by_h[30]) != 3600 or len(pairs_by_h[60]) != 3600:
        raise ValueError(f"expected 3600 adjacent pairs per horizon, got H30={len(pairs_by_h[30])}, H60={len(pairs_by_h[60])}")
    # Lock an outcome-blind representative pair before opening any prediction values.
    h60_pairs = [r for r in pairs_by_h[60]]
    representative = min(h60_pairs, key=lambda r: hashlib.sha256(f"{r['pair_id']}|CORE_B_D4_REP_V2".encode()).hexdigest())
    selection_lock = {"protocol_id": protocol["protocol_id"], "representative_pair_id": representative["pair_id"], "representative_selection_rule": protocol["execution_details"]["representative_pair_selection"], "input_pair_manifest_sha256": sha256_file(input_manifest_path), "prediction_outcomes_opened": False, "final_test_access": 0}
    (NEXT_ROOT / "d4/D4_SELECTION_LOCK.json").write_text(json.dumps(selection_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_map = load_source_groups()
    anchor_rows = [r for r in read_csv(V3_ROOT / "ANCHOR_MANIFEST.csv") if r["split"] == "dev" and r["population"] == "primary" and r["final_test_access"] == "0"]
    meta_by_ep = {}
    for row in anchor_rows:
        meta_by_ep.setdefault(int(row["episode_index"]), row)
    episode_ids = sorted({int(r["episode_index"]) for r in panel})
    episode_data = {ep: load_episode(meta_by_ep[ep]) for ep in episode_ids}
    truth_events = {ep: full_truth_events(episode_data[ep][1], episode_data[ep][0], SIGMA, thresholds) for ep in episode_ids}
    query_state = {}
    for row in panel:
        ep, frame = int(row["episode_index"]), int(row["anchor_frame"])
        if (ep, frame) not in query_state:
            timestamps, action = episode_data[ep]
            query_state[(ep, frame)] = observed_state(action, timestamps, max(0, frame - 119), frame, SIGMA, thresholds)[0]

    # GT-defined event bins and state-eligibility exclusions are computed before D4 errors.
    event_slice = {}
    pair_for_event = {}
    for row in locked_pairs:
        h, ep = int(row["horizon_steps"]), int(row["episode_index"])
        a, b = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
        lo, hi = int(row["common_start_frame"] or 0), int(row["common_end_frame"] or -1)
        ts, _ = episode_data[ep]
        for gi, group in enumerate(GROUPS):
            old_state = query_state[(ep, a)][gi]
            new_state = query_state[(ep, b)][gi]
            if not bool(old_state["known"] and old_state["armed"] and new_state["known"] and new_state["armed"]):
                reason = "old_query_unknown_or_disarmed" if not bool(old_state["known"] and old_state["armed"]) else "new_query_unknown_or_disarmed"
                event_slice[(row["pair_id"], h, group)] = {"event_bin": "NA_STATE_INELIGIBLE", "exclusion_reason": reason, "independent_event_id": "", "extra_events_in_T": 0, "lead_seconds": ""}
                continue
            in_t = [e for e in truth_events[ep] if e["group"] == group and lo <= int(e["onset_frame"]) <= hi and int(e["confirm_frame"]) <= hi]
            after_new = [e for e in truth_events[ep] if e["group"] == group and int(e["onset_frame"]) > b]
            in_t.sort(key=lambda e: int(e["onset_frame"]))
            after_new.sort(key=lambda e: int(e["onset_frame"]))
            if in_t:
                first = in_t[0]
                lead = float(first["onset_timestamp"]) - float(ts[b])
                if lead <= 0.5:
                    bin_name = "(0,0.5s]"
                elif lead <= 1.0:
                    bin_name = "(0.5s,1s]"
                elif lead <= 2.0:
                    bin_name = "(1s,2s]"
                else:
                    bin_name = "NA_BEYOND_REGISTERED_EVENT_BINS"
                event_slice[(row["pair_id"], h, group)] = {"event_bin": bin_name, "exclusion_reason": "", "independent_event_id": f"e{ep:06d}_{group}_f{int(first['onset_frame']):06d}", "extra_events_in_T": max(0, len(in_t) - 1), "lead_seconds": lead}
            elif after_new and int(after_new[0]["confirm_frame"]) > b + h:
                event_slice[(row["pair_id"], h, group)] = {"event_bin": "NA_BEYOND_NATIVE_HORIZON", "exclusion_reason": "", "independent_event_id": "", "extra_events_in_T": 0, "lead_seconds": ""}
            else:
                event_slice[(row["pair_id"], h, group)] = {"event_bin": "none_within_T", "exclusion_reason": "", "independent_event_id": "", "extra_events_in_T": 0, "lead_seconds": ""}

    target_time_rows = []
    for row in locked_pairs:
        ep, h = int(row["episode_index"]), int(row["horizon_steps"])
        frames = valid_common_frames(int(row["old_anchor_frame"]), int(row["new_anchor_frame"]), h, len(episode_data[ep][1]))
        ts = episode_data[ep][0]
        target_time_rows.append({"pair_id": row["pair_id"], "episode_index": ep, "horizon_steps": h, "common_target_frames": json.dumps(frames.tolist()), "common_target_timestamps": json.dumps([float(x) for x in ts[frames].tolist()]), "raw_target_parquet": meta_by_ep[ep]["parquet_path"], "source_group_id": source_map[ep], "final_test_access": 0})
    write_csv(NEXT_ROOT / "d4/D4_PAIR_TIME_SUPPORT.csv", target_time_rows)
    # Dense inputs contain no future targets; verify ID/frame/time metadata only.
    dense_input = np.load(BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz", allow_pickle=False)
    dense_ids = dense_input["dense_id"].astype("U96")
    dense_id_to_index = {str(value): i for i, value in enumerate(dense_ids)}
    if set(dense_id_to_index) != set(panel_by_id):
        raise ValueError("D4 fixed-panel input ID set mismatch")
    dense_meta = {}
    for dense_id, row in panel_by_id.items():
        idx = dense_id_to_index[dense_id]
        ep, frame = int(row["episode_index"]), int(row["anchor_frame"])
        if int(dense_input["episode_index"][idx]) != ep or int(dense_input["anchor_frame"][idx]) != frame:
            raise ValueError(f"D4 dense input ID/frame mismatch: {dense_id}")
        dense_meta[dense_id] = {"episode": ep, "frame": frame, "timestamp": float(dense_input["anchor_timestamp"][idx]), "panel_timestamp": float(row["anchor_timestamp"])}
        actual_ts = episode_data[ep][0][frame]
        if abs(dense_meta[dense_id]["timestamp"] - actual_ts) > 1e-6 or abs(dense_meta[dense_id]["panel_timestamp"] - actual_ts) > 1e-6:
            raise ValueError(f"D4 anchor timestamp mismatch to Parquet: {dense_id}")
    if any("target" in key.lower() or "future" in key.lower() for key in dense_input.files):
        raise ValueError("D4 dense inference input unexpectedly contains target/future fields")

    prediction_bundles = {}
    bundle_rows = []
    for path in sorted((BENCH_ROOT / "predictions_dense").glob("V3-*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            ids = bundle["dense_id"].astype("U96")
            pred = bundle["predictions_native"]
            episode = bundle["episode_index"].astype(np.int64)
            frame = bundle["anchor_frame"].astype(np.int64)
            timestamp = bundle["anchor_timestamp"].astype(np.float64)
            horizon = int(bundle["horizon_steps"])
            cell = str(bundle["cell"].item())
            seed = int(bundle["seed"].item())
        if len(ids) != len(panel) or set(ids.tolist()) != set(panel_by_id) or pred.shape != (len(panel), horizon, 28) or not np.isfinite(pred).all():
            raise ValueError(f"D4 dense prediction ID/shape/finiteness mismatch: {path.name}")
        lookup = {str(value): i for i, value in enumerate(ids)}
        aligned = np.empty_like(pred)
        for row_index, dense_id in enumerate(panel_by_id):
            src = lookup[dense_id]
            expected = dense_meta[dense_id]
            if int(episode[src]) != expected["episode"] or int(frame[src]) != expected["frame"] or abs(float(timestamp[src]) - expected["timestamp"]) > 1e-6:
                raise ValueError(f"D4 dense prediction metadata mismatch: {path.name}:{dense_id}")
            aligned[row_index] = pred[src]
        prediction_bundles[(cell, seed, horizon)] = {"ids": list(panel_by_id), "pred": aligned, "id_index": {dense_id: i for i, dense_id in enumerate(panel_by_id)}, "path": path}
        bundle_rows.append({"model": cell, "seed": seed, "horizon_steps": horizon, "prediction_bundle": str(path), "prediction_sha256": sha256_file(path), "dense_ids": len(ids), "accepted_ids": len(panel_by_id), "status": "PASS_ID_frame_timestamp_shape_finite", "final_test_access": 0})
    expected_bundles = {(cell, seed, h) for cell in ("S-HIST", "F-POOL-D", "F-TOK-D", "F-TOK-S", "V-TOK-D-A", "V-TOK-S-A") for seed in (20260950, 20260951, 20260952) for h in (30, 60)}
    if set(prediction_bundles) != expected_bundles:
        raise ValueError(f"D4 V3 dense accepted bundle grid={len(prediction_bundles)}/36 differs from expected")
    write_csv(NEXT_ROOT / "d4/D4_BUNDLE_ACCEPTANCE.csv", bundle_rows)

    all_pair_rows = []
    pair_metrics_by_model = {}
    pair_metric_rows = []
    model_groups = {}
    pair_rows_by_h = {}
    ordered_source_groups = sorted({source_map[ep] for ep in episode_ids})
    if len(ordered_source_groups) != len(episode_ids):
        raise ValueError("D4 dense panel episodes are not source-group-distinct; nested aggregation review required")
    group_counts = bootstrap_counts(len(ordered_source_groups))
    for horizon in (30, 60):
        base = pairs_by_h[horizon]
        pair_rows_by_h[horizon] = base
        pair_groups = np.asarray([source_map[int(row["episode_index"])] for row in base], dtype="U64")
        for key, bundle in sorted(prediction_bundles.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
            cell, seed, model_horizon = key
            if model_horizon != horizon:
                continue
            pred = bundle["pred"]
            idx = bundle["id_index"]
            output = {metric: np.empty(len(base), dtype=np.float64) for metric in metric_keys()}
            for pair_index, row in enumerate(base):
                ep = int(row["episode_index"])
                a, b = int(row["old_anchor_frame"]), int(row["new_anchor_frame"])
                old_id, new_id = row["old_dense_id"], row["new_dense_id"]
                frames = valid_common_frames(a, b, horizon, len(episode_data[ep][1]))
                expected_frames = np.arange(int(row["common_start_frame"]), int(row["common_end_frame"]) + 1, dtype=np.int64)
                if not np.array_equal(frames, expected_frames) or len(frames) != int(row["common_target_steps"]):
                    raise ValueError(f"D4 common target frame mismatch: {row['pair_id']}")
                timestamps, actions = episode_data[ep]
                for frame in frames:
                    if not np.isfinite(timestamps[frame]) or (frame > 0 and timestamps[frame] <= timestamps[frame - 1]):
                        raise ValueError(f"D4 common target timestamp invalid: episode={ep},frame={frame}")
                old_index, new_index = idx[old_id], idx[new_id]
                old_forecast = pred[old_index, frames - a - 1]
                new_forecast = pred[new_index, frames - b - 1]
                truth = actions[frames]
                measured = pair_metrics(old_forecast, new_forecast, truth, SIGMA)
                for name in output:
                    output[name][pair_index] = measured[name]
                for body in BODIES:
                    fields = {name.split("|", 1)[1]: float(output[name][pair_index]) for name in output if name.startswith(f"{body}|")}
                    pair_metric_rows.append({"pair_id": row["pair_id"], "model": cell, "seed": seed, "horizon_steps": horizon, "body": body, "episode_index": ep, "source_group_id": source_map[ep], "old_dense_id": old_id, "new_dense_id": new_id, "common_start_frame": row["common_start_frame"], "common_end_frame": row["common_end_frame"], "common_target_steps": row["common_target_steps"], **fields, "final_test_access": 0})
            pair_metrics_by_model[key] = output
            model_groups[key] = pair_groups

    write_csv(NEXT_ROOT / "d4/D4_PAIR_METRICS.csv", pair_metric_rows)
    # Per-cell level outcomes, with pairs averaged within episode/source group.
    primary_rows, bootstrap_rows = [], []
    model_keys = sorted(pair_metrics_by_model, key=lambda x: (x[2], x[1], x[0]))
    metric_names = [key for key in metric_keys() if not (key.endswith("|FD_old") or key.endswith("|FD_new"))]
    metric_names += [f"{body}|FD_{side}" for body in BODIES for side in ("old", "new")]
    for key in model_keys:
        cell, seed, horizon = key
        values = pair_metrics_by_model[key]
        groups = model_groups[key]
        meta = {"model": cell, "seed": seed, "horizon_steps": horizon, "pair_count": len(groups), "episode_count": len(set(int(r["episode_index"]) for r in pairs_by_h[horizon])), "source_group_count": len(ordered_source_groups)}
        for metric in metric_names:
            row, _ = metric_row(metric, values[metric], groups, ordered_source_groups, group_counts, meta)
            primary_rows.append({**row, "final_test_access": 0})
            bootstrap_rows.append({"analysis": "per_model", **meta, **row, "final_test_access": 0})
    write_csv(NEXT_ROOT / "d4/D4_PRIMARY_RESULTS_BY_SEED.csv", primary_rows)

    # Shared-pair paired differences in rolling utility and absolute new-time quality.
    effect_rows = []
    contrast_pair_delta = {}
    for horizon in (30, 60):
        base = pairs_by_h[horizon]
        pair_groups = np.asarray([source_map[int(row["episode_index"])] for row in base], dtype="U64")
        for seed in (20260950, 20260951, 20260952):
            for contrast, (candidate, control) in CONTRASTS.items():
                cand_key, ctrl_key = (candidate, seed, horizon), (control, seed, horizon)
                for body in BODIES:
                    for metric in ("U", "L_new", "FD_new", "R", "endpoint_new", "coord_error_lower_share", "coord_error_higher_share"):
                        key_metric = f"{body}|{metric}"
                        delta = pair_metrics_by_model[cand_key][key_metric] - pair_metrics_by_model[ctrl_key][key_metric]
                        group_delta = group_mean(delta, pair_groups, ordered_source_groups)
                        estimate, low, high = summarize_groups(group_delta, group_counts)
                        row = {"contrast": contrast, "candidate": candidate, "control": control, "seed": seed, "horizon_steps": horizon, "body": body, "metric": f"delta_{metric}", "estimand": "candidate_minus_control on identical pairs/GT; for U positive favors candidate; for loss/FD negative favors candidate", "estimate": estimate, "ci95_low": low, "ci95_high": high, "pair_count": len(base), "episode_count": len(episode_ids), "source_group_count": len(ordered_source_groups), "bootstrap_replicates": BOOTSTRAPS, "final_test_access": 0}
                        effect_rows.append(row)
                        bootstrap_rows.append({"analysis": "paired_effect", **row})
                        if metric == "U":
                            contrast_pair_delta[(contrast, seed, horizon, body)] = delta
                # Also report raw U level by candidate/control without interpreting it as contrast.
    write_csv(NEXT_ROOT / "d4/D4_PAIRED_EFFECTS.csv", effect_rows)
    write_csv(NEXT_ROOT / "d4/D4_BOOTSTRAP.csv", bootstrap_rows)

    # Expanded, contrast-specific common-pair manifest: one row per registered
    # seed x horizon x contrast x accepted adjacent pair.
    expanded_pairs = []
    pair_id_to_event_rows = {}
    for horizon in (30, 60):
        for row in pairs_by_h[horizon]:
            ep = int(row["episode_index"])
            for group in GROUPS:
                event_info = event_slice[(row["pair_id"], horizon, group)]
                pair_id_to_event_rows[(row["pair_id"], horizon, group)] = event_info
            for seed in (20260950, 20260951, 20260952):
                for contrast, (candidate, control) in CONTRASTS.items():
                    expanded_pairs.append({**row, "contrast": contrast, "seed": seed, "candidate_model": candidate, "candidate_run_id": f"V3-{candidate}-H{horizon}-S{seed}", "control_model": control, "control_run_id": f"V3-{control}-H{horizon}-S{seed}", "required_bundle_count": 4, "all_required_bundles_accepted": True, "source_group_id": source_map[ep], "timestamp_frame_join_status": "PASS_parquet_exact_frames", "event_hand_slice": event_slice[(row["pair_id"], horizon, "Hand")]["event_bin"], "event_arm_slice": event_slice[(row["pair_id"], horizon, "Arm")]["event_bin"], "final_test_access": 0})
    for pair in expanded_pairs:
        ep, h = int(pair["episode_index"]), int(pair["horizon_steps"])
        pair["candidate_prediction_bundle"] = str(prediction_bundles[(pair["candidate_model"], int(pair["seed"]), h)]["path"])
        pair["control_prediction_bundle"] = str(prediction_bundles[(pair["control_model"], int(pair["seed"]), h)]["path"])
        pair["raw_target_parquet"] = meta_by_ep[ep]["parquet_path"]
        pair["target_time_support_pair_id"] = pair["pair_id"]
    write_csv(NEXT_ROOT / "d4/D4_PAIR_MANIFEST.csv", expanded_pairs)

    # Event ambiguity summary for same-group utility contrasts.
    event_rows = []
    bins = ("(0,0.5s]", "(0.5s,1s]", "(1s,2s]", "none_within_T")
    for horizon in (30, 60):
        base = pairs_by_h[horizon]
        pair_groups = np.asarray([source_map[int(row["episode_index"])] for row in base], dtype="U64")
        pair_lookup = {row["pair_id"]: i for i, row in enumerate(base)}
        for seed in (20260950, 20260951, 20260952):
            for contrast, (candidate, control) in CONTRASTS.items():
                for group in GROUPS:
                    body = group
                    delta_u = contrast_pair_delta[(contrast, seed, horizon, body)]
                    delta_new = pair_metrics_by_model[(candidate, seed, horizon)][f"{body}|L_new"] - pair_metrics_by_model[(control, seed, horizon)][f"{body}|L_new"]
                    for event_bin in bins:
                        mask = np.zeros(len(base), dtype=bool)
                        independent_events = set()
                        extra_event_count = 0
                        for i, pair in enumerate(base):
                            info = event_slice[(pair["pair_id"], horizon, group)]
                            if info["event_bin"] == event_bin:
                                mask[i] = True
                                if info["independent_event_id"]:
                                    independent_events.add(info["independent_event_id"])
                                    extra_event_count += int(info["extra_events_in_T"])
                        selected_groups = group_mean(np.ones(len(base)), pair_groups, ordered_source_groups)
                        values_u = np.full(len(ordered_source_groups), np.nan)
                        values_l = np.full(len(ordered_source_groups), np.nan)
                        for gi, gid in enumerate(ordered_source_groups):
                            local = mask & (pair_groups == gid)
                            if local.any():
                                values_u[gi] = float(delta_u[local].mean())
                                values_l[gi] = float(delta_new[local].mean())
                        u, ulo, uhi = summarize_groups(values_u, group_counts)
                        le, llo, lhi = summarize_groups(values_l, group_counts)
                        event_rows.append({"contrast": contrast, "candidate": candidate, "control": control, "seed": seed, "horizon_steps": horizon, "event_group": group, "event_bin": event_bin, "delta_u": u, "delta_u_ci95_low": ulo, "delta_u_ci95_high": uhi, "delta_l_new": le, "delta_l_new_ci95_low": llo, "delta_l_new_ci95_high": lhi, "pair_count": int(mask.sum()), "episode_count": len(set(int(base[i]["episode_index"]) for i in np.flatnonzero(mask))), "source_group_count": int(np.isfinite(values_u).sum()), "independent_events": len(independent_events), "additional_events_in_T": extra_event_count, "excluded_state_ineligible_pairs": sum(event_slice[(p["pair_id"], horizon, group)]["event_bin"] == "NA_STATE_INELIGIBLE" for p in base), "final_test_access": 0})
    for row in event_rows:
        row["support_status"] = "measured" if int(row["pair_count"]) > 0 else "NA_no_eligible_pairs"
        for metric in ("delta_u", "delta_u_ci95_low", "delta_u_ci95_high", "delta_l_new", "delta_l_new_ci95_low", "delta_l_new_ci95_high"):
            if row[metric] is not None and not np.isfinite(float(row[metric])):
                row[metric] = None
    write_csv(NEXT_ROOT / "d4/D4_EVENT_AMBIGUITY_SLICES.csv", event_rows)

    # Deterministic representative and post-hoc worst-regression trajectory.
    rep_row = next(row for row in h60_pairs if row["pair_id"] == representative["pair_id"])
    failure_seed, failure_horizon, failure_body = 20260950, 60, "Full28"
    failure_delta = contrast_pair_delta[("video", failure_seed, failure_horizon, failure_body)]
    failure_index = int(np.argmin(failure_delta))
    failure_row = pairs_by_h[failure_horizon][failure_index]
    save_trajectory(rep_row, 20260950, "representative", prediction_bundles, episode_data)
    save_trajectory(failure_row, failure_seed, "worst_negative_delta_u_video_diagnostic", prediction_bundles, episode_data)

    summary = {"status": "D4_V3_DENSE_CORE_COMPLETE", "protocol_id": protocol["protocol_id"], "v3_dense_prediction_bundles_accepted": len(prediction_bundles), "expected_dense_prediction_bundles": 36, "input_panel_episodes": len(episode_ids), "input_panel_queries": len(panel), "adjacent_pairs_per_horizon": len(pairs_by_h[30]), "pair_manifest_rows": len(expanded_pairs), "event_slice_rows": len(event_rows), "source_groups": len(ordered_source_groups), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED, "representative_pair_id": representative["pair_id"], "failure_example_pair_id": failure_row["pair_id"], "elapsed_cpu_seconds": time.time() - start_wall, "gpu_hours": 0, "controls_deferred": ["stale_fresh_video", "latency", "missing_frame", "timestamp_gap"], "final_test_access": 0}
    (NEXT_ROOT / "d4/D4_STATUS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def save_trajectory(pair: dict[str, str], seed: int, suffix: str, bundles: dict, episode_data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell, horizon = "F-TOK-D", int(pair["horizon_steps"])
    episode = int(pair["episode_index"])
    a, b = int(pair["old_anchor_frame"]), int(pair["new_anchor_frame"])
    frames = valid_common_frames(a, b, horizon, len(episode_data[episode][1]))
    timestamps, actions = episode_data[episode]
    old_id, new_id = pair["old_dense_id"], pair["new_dense_id"]
    pred = bundles[(cell, seed, horizon)]["pred"]
    lookup = bundles[(cell, seed, horizon)]["id_index"]
    old = pred[lookup[old_id], frames - a - 1]
    new = pred[lookup[new_id], frames - b - 1]
    truth = actions[frames]
    x = timestamps[frames] - timestamps[b]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, dim, label in ((axes[0], 0, "Hand coordinate 0"), (axes[1], 14, "Arm coordinate 14")):
        ax.plot(x, truth[:, dim], color="black", lw=1.8, label="GT")
        ax.plot(x, old[:, dim], color="#dc6b19", lw=1.4, label=f"old @ frame {a}")
        ax.plot(x, new[:, dim], color="#2878b5", lw=1.4, label=f"new @ frame {b}")
        ax.axvline(0, color="0.5", lw=0.8, ls="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("seconds relative to new observation")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"D4 {suffix}: episode {episode}, H={horizon}, seed={seed}, common targets={len(frames)}")
    trajectory_data = NEXT_ROOT / "d4/trajectories" / f"D4_TRAJECTORY_DATA_{suffix}.npz"
    trajectory_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory_data, pair_id=np.asarray(pair["pair_id"]), episode_index=np.asarray(episode), model=np.asarray(cell), seed=np.asarray(seed), horizon_steps=np.asarray(horizon), old_anchor_frame=np.asarray(a), new_anchor_frame=np.asarray(b), common_target_frames=frames, common_target_timestamps=timestamps[frames], gt_native=truth, old_prediction_native=old, new_prediction_native=new, final_test_access=np.asarray(0))
    out = NEXT_ROOT / "d4/figures" / f"D4_TRAJECTORY_{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
