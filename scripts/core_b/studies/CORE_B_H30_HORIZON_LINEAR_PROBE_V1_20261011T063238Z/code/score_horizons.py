from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as _scipy_eigh_preload
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
QUERY = PROJECT / "reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
TRAIN_CACHE = EQ / "d3/train_targets_trainonly.npz"
HORIZONS = (1, 3, 6, 10, 15, 30)
THREAD_LIMIT = min(8, os.cpu_count() or 1)
EPISODE_FIELDS = [
    "split", "system", "family", "query", "seed", "horizon",
    "episode_index", "source_group_id", "anchors",
    "actual_target_delta_ms_mean", "actual_target_delta_ms_min", "actual_target_delta_ms_max",
    "endpoint_position_standardized_mae", "prefix_position_standardized_mae",
    "endpoint_position_native_mae", "prefix_position_native_mae",
    "future_internal_fd_native_mae", "future_internal_fd_safe_normalized",
    "internal_difference_count", "seam_error_native", "seam_error_safe_normalized",
    "seam_pred_step_abs_native", "seam_gt_step_abs_native",
    "tv_pred", "tv_gt", "zero_gt_tv_windows",
    "direction_accuracy_active", "positive_recall", "negative_recall", "balanced_direction_accuracy",
    "active_components", "positive_components", "negative_components",
    "final_test_access", "heldout_access",
]
SUMMARY_FIELDS = [
    "split", "system", "family", "query", "seed", "horizon",
    "nominal_target_delta_ms", "actual_target_delta_ms_mean", "actual_target_delta_ms_min",
    "actual_target_delta_ms_max", "anchors", "episodes", "source_groups",
    "endpoint_position_standardized_mae", "prefix_position_standardized_mae",
    "endpoint_position_native_mae", "prefix_position_native_mae",
    "position_endpoint_skill_vs_persistence", "position_prefix_skill_vs_persistence",
    "future_internal_fd_native_mae", "future_internal_fd_safe_normalized",
    "future_internal_fd_native_skill_vs_persistence", "future_internal_fd_safe_skill_vs_persistence",
    "internal_difference_count", "seam_error_native", "seam_error_safe_normalized",
    "seam_pred_step_abs_native", "seam_gt_step_abs_native",
    "tv_pred_episode_equal_mean", "tv_gt_episode_equal_mean", "tv_ratio_aggregate",
    "zero_gt_tv_windows", "zero_gt_tv_episodes",
    "direction_accuracy_active_episode_equal", "positive_recall_episode_equal",
    "negative_recall_episode_equal", "balanced_direction_accuracy_episode_equal",
    "active_components", "positive_components", "negative_components",
    "direction_episodes_active", "direction_episodes_both_signs",
    "final_test_access", "heldout_access",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(content, encoding="utf-8")


def write_rows(path: Path, fieldnames: list[str], rows) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_npz_predictions(path: Path, expected_ids: np.ndarray, expected_sha: str | None = None) -> np.ndarray:
    if expected_sha is not None and sha(path) != expected_sha:
        raise AssertionError(f"prediction source hash changed: {path}")
    with np.load(path, allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U64", copy=True)
        pred = z["predictions_native"].astype(np.float32, copy=True)
    if pred.shape != (len(ids), 30, 28) or not np.isfinite(pred).all():
        raise AssertionError(f"invalid prediction shape/finite values: {path} {pred.shape}")
    if len(np.unique(ids)) != len(ids) or set(ids.tolist()) != set(expected_ids.tolist()):
        raise AssertionError(f"prediction IDs do not match locked split: {path}")
    order = {str(anchor): i for i, anchor in enumerate(ids.tolist())}
    return pred[np.asarray([order[str(anchor)] for anchor in expected_ids.tolist()], dtype=np.int64)]


def load_accepted_targets_and_inputs():
    with np.load(INPUTS, allow_pickle=False) as z:
        ids = z["anchor_id"].astype("U64", copy=True)
        episodes = z["episode_index"].astype(np.int64, copy=True)
        split_code = z["split_code"].astype(np.int8, copy=True)
        anchor = z["anchor_command_native"].astype(np.float32, copy=True)
        frames = z["anchor_frame"].astype(np.int64, copy=True)
        anchor_ts = z["anchor_timestamp"].astype(np.float64, copy=True)
        # This archive stores H60 labels; only the locked H30 slice is retained/scored.
        target_h30_all = z["targets_native"][:, :30, :].astype(np.float32, copy=True)
    if target_h30_all.shape != (20079, 30, 28):
        raise AssertionError(f"unexpected H30 target slice shape {target_h30_all.shape}")
    split = np.where(split_code == 0, "train", "dev")
    if set(np.unique(split).tolist()) != {"train", "dev"}:
        raise AssertionError("non-primary split code in the accepted main input archive")
    if int(np.sum(split == "train")) != 17757 or int(np.sum(split == "dev")) != 2322:
        raise AssertionError("train/dev counts changed after model lock")
    return ids, episodes, split, anchor, frames, anchor_ts, target_h30_all


def read_manifest_maps(ids: np.ndarray, episodes: np.ndarray, split: np.ndarray):
    qrows = read_csv(QUERY / "inventory/ANCHOR_MANIFEST.csv")
    v3rows = read_csv(V3 / "ANCHOR_MANIFEST.csv")
    qmap = {r["anchor_id"]: r for r in qrows}
    vmap = {r["anchor_id"]: r for r in v3rows if r.get("population") == "primary"}
    if len(qmap) != len(qrows) or len(vmap) != 20079:
        raise AssertionError("duplicate/missing locked primary manifest rows")
    groups = np.empty(len(ids), dtype="U128")
    parquet_path = np.empty(len(ids), dtype="U256")
    for i, aid in enumerate(ids.tolist()):
        q = qmap.get(aid)
        v = vmap.get(aid)
        if q is None or v is None:
            raise AssertionError(f"anchor missing from locked source manifest: {aid}")
        if q["split"] != split[i] or int(q["episode_index"]) != int(episodes[i]):
            raise AssertionError(f"source manifest identity mismatch: {aid}")
        if int(q.get("final_test_access", "0")) != 0 or int(v.get("final_test_access", "0")) != 0:
            raise AssertionError(f"final/heldout row present: {aid}")
        if v["split"] != split[i] or int(v["episode_index"]) != int(episodes[i]):
            raise AssertionError(f"V3 manifest identity mismatch: {aid}")
        groups[i] = q["source_group_id"]
        parquet_path[i] = v["parquet_path"]
    if len(set(groups[split == "train"].tolist())) != 2712 or len(set(groups[split == "dev"].tolist())) != 603:
        raise AssertionError("source-group counts differ from source acceptance")
    if set(groups[split == "train"].tolist()) & set(groups[split == "dev"].tolist()):
        raise AssertionError("train/dev group overlap")
    for ep in np.unique(episodes):
        if len(set(groups[episodes == ep].tolist())) != 1:
            raise AssertionError(f"episode maps to multiple source groups: {ep}")
    return groups, parquet_path, qmap, vmap


def actual_horizon_times(ids, episodes, split, frames, anchor_ts, parquet_paths):
    import pyarrow.parquet as pq
    ep_paths = {}
    for ep, rel in zip(episodes.tolist(), parquet_paths.tolist()):
        if ep in ep_paths and ep_paths[ep] != rel:
            raise AssertionError(f"episode has multiple canonical parquet paths: {ep}")
        ep_paths[ep] = rel
    timestamp_by_episode = {}
    for ep, rel in ep_paths.items():
        path = RAW_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=["frame_index", "timestamp"], use_threads=False)
        fr = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        if not np.array_equal(fr, np.arange(len(fr), dtype=np.int64)) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise AssertionError(f"raw timestamp metadata validation failed for episode {ep}")
        timestamp_by_episode[int(ep)] = ts
    rows = []
    per_horizon = {h: np.empty(len(ids), dtype=np.float64) for h in HORIZONS}
    for i, (aid, ep, split_i, frame, ts0) in enumerate(zip(ids.tolist(), episodes.tolist(), split.tolist(), frames.tolist(), anchor_ts.tolist())):
        timeline = timestamp_by_episode[int(ep)]
        if frame + 30 >= len(timeline):
            raise AssertionError(f"insufficient raw timestamp support at H30: {aid}")
        observed_anchor = float(timeline[frame])
        if abs(observed_anchor - float(ts0)) > 1e-5:
            raise AssertionError(f"anchor timestamp mismatch against raw metadata: {aid}")
        for h in HORIZONS:
            dt = float(timeline[frame + h] - timeline[frame])
            if dt <= 0 or not np.isfinite(dt):
                raise AssertionError(f"invalid actual target time for {aid}, H={h}")
            per_horizon[h][i] = dt
            rows.append({
                "anchor_id": aid, "episode_index": int(ep), "split": split_i,
                "horizon": h, "anchor_frame": int(frame), "target_frame": int(frame + h),
                "anchor_timestamp_raw": observed_anchor,
                "target_timestamp_raw": float(timeline[frame + h]),
                "actual_target_delta_seconds": dt,
                "actual_target_delta_ms": dt * 1000.0,
                "timestamp_source": "read-only raw Parquet frame_index/timestamp columns; action column not read",
                "final_test_access": 0,
            })
    return per_horizon, rows, len(timestamp_by_episode)


def episode_metrics_for_model(
    pred, target, anchor, sigma, safe, threshold, episodes, groups, split, system, family, query, seed, horizon, time_delta
):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ridge_core", ROOT / "code/ridge_core.py")
    core = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(core)
    m = core.horizon_window_metrics(pred, target, anchor, sigma, safe, threshold, horizon)
    dirm = core.episode_direction_metrics(m, episodes)
    out_rows = []
    per_ep: dict[str, dict[str, float]] = {}
    for ep in np.unique(episodes):
        idx = episodes == ep
        ep_groups = np.unique(groups[idx])
        if len(ep_groups) != 1:
            raise AssertionError(f"group identity not unique within episode {ep}")
        gid = str(ep_groups[0])
        d = dirm[int(ep)]
        def mean_metric(name):
            values = np.asarray(m[name][idx], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else float("nan")
        row = {
            "split": split, "system": system, "family": family, "query": query,
            "seed": seed if seed is not None else "NA", "horizon": horizon,
            "episode_index": int(ep), "source_group_id": gid, "anchors": int(idx.sum()),
            "actual_target_delta_ms_mean": float(np.mean(time_delta[idx]) * 1000.0),
            "actual_target_delta_ms_min": float(np.min(time_delta[idx]) * 1000.0),
            "actual_target_delta_ms_max": float(np.max(time_delta[idx]) * 1000.0),
            "endpoint_position_standardized_mae": mean_metric("endpoint_position_standardized_mae"),
            "prefix_position_standardized_mae": mean_metric("prefix_position_standardized_mae"),
            "endpoint_position_native_mae": mean_metric("endpoint_position_native_mae"),
            "prefix_position_native_mae": mean_metric("prefix_position_native_mae"),
            "future_internal_fd_native_mae": mean_metric("future_internal_fd_native_mae"),
            "future_internal_fd_safe_normalized": mean_metric("future_internal_fd_safe_normalized"),
            "internal_difference_count": horizon - 1,
            "seam_error_native": mean_metric("seam_error_native"),
            "seam_error_safe_normalized": mean_metric("seam_error_safe_normalized"),
            "seam_pred_step_abs_native": mean_metric("seam_pred_step_abs_native"),
            "seam_gt_step_abs_native": mean_metric("seam_gt_step_abs_native"),
            "tv_pred": mean_metric("tv_pred"), "tv_gt": mean_metric("tv_gt"),
            "zero_gt_tv_windows": int(np.sum(m["zero_gt_tv_windows"][idx])),
            "direction_accuracy_active": d["direction_accuracy_active"],
            "positive_recall": d["positive_recall"], "negative_recall": d["negative_recall"],
            "balanced_direction_accuracy": d["balanced_direction_accuracy"],
            "active_components": int(d["active_components"]),
            "positive_components": int(d["positive_components"]),
            "negative_components": int(d["negative_components"]),
            "final_test_access": 0, "heldout_access": 0,
        }
        out_rows.append(row)
        per_ep[gid] = row
    return out_rows, per_ep


def mean_finite(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def make_summary(split, system, family, query, seed, horizon, rows, persistence_rows, time_delta):
    def vals(key):
        return [float(r[key]) for r in rows]
    def pvals(key):
        return [float(r[key]) for r in persistence_rows]
    def skill(key):
        base = mean_finite(pvals(key))
        model = mean_finite(vals(key))
        return 1.0 - model / base if np.isfinite(model) and np.isfinite(base) and base != 0 else float("nan")
    tvp = mean_finite(vals("tv_pred"))
    tvg = mean_finite(vals("tv_gt"))
    active_eps = [r for r in rows if int(r["active_components"]) > 0]
    balanced_eps = [r for r in rows if int(r["positive_components"]) > 0 and int(r["negative_components"]) > 0]
    all_rows = rows
    return {
        "split": split, "system": system, "family": family, "query": query,
        "seed": seed if seed is not None else "NA", "horizon": horizon,
        "nominal_target_delta_ms": horizon / 30.0 * 1000.0,
        "actual_target_delta_ms_mean": float(np.mean(time_delta) * 1000.0),
        "actual_target_delta_ms_min": float(np.min(time_delta) * 1000.0),
        "actual_target_delta_ms_max": float(np.max(time_delta) * 1000.0),
        "anchors": int(sum(int(r["anchors"]) for r in rows)),
        "episodes": len(rows), "source_groups": len(set(r["source_group_id"] for r in rows)),
        "endpoint_position_standardized_mae": mean_finite(vals("endpoint_position_standardized_mae")),
        "prefix_position_standardized_mae": mean_finite(vals("prefix_position_standardized_mae")),
        "endpoint_position_native_mae": mean_finite(vals("endpoint_position_native_mae")),
        "prefix_position_native_mae": mean_finite(vals("prefix_position_native_mae")),
        "position_endpoint_skill_vs_persistence": skill("endpoint_position_standardized_mae"),
        "position_prefix_skill_vs_persistence": skill("prefix_position_standardized_mae"),
        "future_internal_fd_native_mae": mean_finite(vals("future_internal_fd_native_mae")),
        "future_internal_fd_safe_normalized": mean_finite(vals("future_internal_fd_safe_normalized")),
        "future_internal_fd_native_skill_vs_persistence": skill("future_internal_fd_native_mae"),
        "future_internal_fd_safe_skill_vs_persistence": skill("future_internal_fd_safe_normalized"),
        "internal_difference_count": horizon - 1,
        "seam_error_native": mean_finite(vals("seam_error_native")),
        "seam_error_safe_normalized": mean_finite(vals("seam_error_safe_normalized")),
        "seam_pred_step_abs_native": mean_finite(vals("seam_pred_step_abs_native")),
        "seam_gt_step_abs_native": mean_finite(vals("seam_gt_step_abs_native")),
        "tv_pred_episode_equal_mean": tvp, "tv_gt_episode_equal_mean": tvg,
        "tv_ratio_aggregate": tvp / tvg if np.isfinite(tvp) and np.isfinite(tvg) and tvg > 0 else float("nan"),
        "zero_gt_tv_windows": int(sum(int(r["zero_gt_tv_windows"]) for r in rows)),
        "zero_gt_tv_episodes": int(sum(float(r["tv_gt"]) == 0.0 for r in rows if np.isfinite(float(r["tv_gt"])))),
        "direction_accuracy_active_episode_equal": mean_finite([r["direction_accuracy_active"] for r in rows]),
        "positive_recall_episode_equal": mean_finite([r["positive_recall"] for r in rows]),
        "negative_recall_episode_equal": mean_finite([r["negative_recall"] for r in rows]),
        "balanced_direction_accuracy_episode_equal": mean_finite([r["balanced_direction_accuracy"] for r in rows]),
        "active_components": int(sum(int(r["active_components"]) for r in rows)),
        "positive_components": int(sum(int(r["positive_components"]) for r in rows)),
        "negative_components": int(sum(int(r["negative_components"]) for r in rows)),
        "direction_episodes_active": len(active_eps),
        "direction_episodes_both_signs": len(balanced_eps),
        "final_test_access": 0, "heldout_access": 0,
    }


def load_all_prediction_records(lock):
    refs = lock["bundles"]
    if len(refs) != 24:
        raise AssertionError("locked reference model inventory is not 24 bundles")
    return refs


def main():
    import importlib.util
    pools = threadpool_info()
    if any(p.get("user_api") == "blas" and int(p.get("num_threads", 0)) > THREAD_LIMIT for p in pools):
        raise RuntimeError(f"scorer BLAS thread cap check failed: {pools}")
    spec = importlib.util.spec_from_file_location("ridge_core", ROOT / "code/ridge_core.py")
    core = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(core)

    model_lock_path = ROOT / "models/RIDGE_MODEL_LOCK.json"
    prediction_accept_path = ROOT / "qa/RIDGE_PREDICTION_ACCEPTANCE.json"
    if not model_lock_path.is_file() or not prediction_accept_path.is_file():
        raise AssertionError("model and target-free prediction lock are absent")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    pred_accept = json.loads(prediction_accept_path.read_text(encoding="utf-8"))
    if model_lock.get("status") != "PASS_MODEL_AND_PREDICTIONS_LOCKED_BEFORE_DEV_SCORING":
        raise AssertionError("Ridge model lock does not pass")
    if pred_accept.get("status") != "PASS_RIDGE_TRAIN_DEV_PREDICTIONS" or pred_accept.get("dev_targets_read_before_lock") is not False:
        raise AssertionError("Ridge prediction acceptance does not pass")
    if sha(model_lock_path) != pred_accept["model_lock_sha256"]:
        raise AssertionError("model lock hash differs from Ridge prediction acceptance")
    if model_lock.get("dev_targets_read") is not False or model_lock.get("final_test_access") != 0 or model_lock.get("heldout_access") != 0:
        raise AssertionError("Ridge lock crossed outcome boundary")
    for rel, expected in [
        (model_lock["model_sha256"], ROOT / "models/ridge_full_train.npz"),
        (model_lock["train_prediction_sha256"], ROOT / "predictions/train/RIDGE_FULL_TRAIN.npz"),
        (model_lock["dev_prediction_sha256"], ROOT / "predictions/dev/RIDGE_FULL_DEV.npz"),
    ]:
        if sha(expected) != rel:
            raise AssertionError(f"locked Ridge artifact hash changed: {expected}")
    if (ROOT / "results/HORIZON_EPISODE_METRICS.csv").exists():
        raise FileExistsError("scored horizon outputs already exist; preserve them and resume explicitly")

    protocol = json.loads((ROOT / "protocol/PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    model_reuse = json.loads((ROOT / "inventory/MODEL_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    if model_reuse["status"] != "NO_EXACT_ACCEPTED_MATCH_NEW_CPU_RIDGE_REQUIRED":
        raise AssertionError("baseline reuse gate is not the locked new Ridge path")
    write_json(ROOT / "protocol/OUTCOME_SCORE_GATE.json", {
        "status": "PASS_MODEL_LOCK_AND_PREDICTIONS_VERIFIED_BEFORE_TARGET_READ",
        "verified_utc": utc_now(), "model_lock_sha256": sha(model_lock_path),
        "ridge_prediction_acceptance_sha256": sha(prediction_accept_path),
        "dev_targets_read_before_gate": False, "final_test_access": 0, "heldout_access": 0,
    })

    score_started = utc_now()
    input_hash = sha(INPUTS)
    if input_hash != protocol["input"]["source_sha256"]:
        raise AssertionError("main input source hash changed after lock")
    ids, episodes, split, anchor, frames, anchor_ts, targets = load_accepted_targets_and_inputs()
    groups, parquet_paths, qmap, vmap = read_manifest_maps(ids, episodes, split)
    if targets.shape != (len(ids), 30, 28):
        raise AssertionError("scoring target shape mismatch")
    if not np.isfinite(targets).all() or not np.isfinite(anchor).all():
        raise AssertionError("scoring target/anchor finite gate failed")
    if set(np.unique(split).tolist()) != {"train", "dev"}:
        raise AssertionError("final/heldout split appears in score source")

    norms = json.loads((V3 / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(norms["target_sigma"], dtype=np.float64)
    safe = np.asarray(norms["safe_fd_scale"], dtype=np.float64)
    threshold_doc = json.loads((PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/MOTION_ACTIVITY_THRESHOLD.json").read_text(encoding="utf-8"))
    threshold = np.asarray(threshold_doc["direction_threshold_vector"], dtype=np.float64)
    if sigma.shape != (28,) or safe.shape != (28,) or threshold.shape != (28,):
        raise AssertionError("normalizer or frozen direction threshold width mismatch")
    time_delta_by_h, time_rows, raw_episode_files = actual_horizon_times(ids, episodes, split, frames, anchor_ts, parquet_paths)

    # Target access begins only after the accepted Ridge lock above; only primary train/dev H30 is scored.
    targets_by_split = {}
    anchors_by_split = {}
    episodes_by_split = {}
    groups_by_split = {}
    ids_by_split = {}
    times_by_split = {}
    for spl in ("train", "dev"):
        ix = np.flatnonzero(split == spl)
        targets_by_split[spl] = targets[ix]
        anchors_by_split[spl] = anchor[ix].astype(np.float64)
        episodes_by_split[spl] = episodes[ix]
        groups_by_split[spl] = groups[ix]
        ids_by_split[spl] = ids[ix]
        times_by_split[spl] = {h: time_delta_by_h[h][ix] for h in HORIZONS}

    # Verify train-only target cache exactly matches the accepted H30 target source.
    with np.load(TRAIN_CACHE, allow_pickle=False) as z:
        train_cache_ids = z["anchor_id"].astype("U64", copy=True)
        train_cache_eps = z["episode_index"].astype(np.int64, copy=True)
        train_cache_anchor = z["anchor_command_native"].astype(np.float32, copy=True)
        train_cache_h30 = z["targets_native_60"][:, :30, :].astype(np.float32, copy=True)
    train_ix = np.flatnonzero(split == "train")
    if not np.array_equal(train_cache_ids, ids[train_ix]) or not np.array_equal(train_cache_eps, episodes[train_ix]):
        raise AssertionError("accepted train target cache IDs differ from locked train")
    if not np.allclose(train_cache_anchor, anchor[train_ix], rtol=0, atol=1e-6):
        raise AssertionError("accepted train target cache anchors differ")
    train_target_delta = float(np.max(np.abs(train_cache_h30 - targets[train_ix])))
    if train_target_delta > 1e-6:
        raise AssertionError(f"H30 train target cache differs from V3 input archive by {train_target_delta}")

    refs = load_all_prediction_records(json.loads((ROOT / "protocol/REFERENCE_MODELS.json").read_text(encoding="utf-8")))
    systems_by_split: dict[str, list[dict[str, object]]] = {"train": [], "dev": []}
    reference_hashes_checked = []
    for rec in refs:
        spl = str(rec["split"])
        if spl not in ("train", "dev"):
            raise AssertionError("non-primary split in accepted prediction references")
        pred = load_npz_predictions(Path(str(rec["prediction_path"])), ids_by_split[spl], str(rec["prediction_sha256"]))
        systems_by_split[spl].append({
            "system": f'{rec["query"]}_{rec["system"]}', "family": str(rec["system"]),
            "query": str(rec["query"]), "seed": int(rec["seed"]), "pred": pred,
            "source": "accepted_prior_prediction", "prediction_path": str(rec["prediction_path"]),
            "prediction_sha256": str(rec["prediction_sha256"]),
        })
        reference_hashes_checked.append({"path": rec["prediction_path"], "sha256": rec["prediction_sha256"]})
    for spl in ("train", "dev"):
        rid = load_npz_predictions(ROOT / f"predictions/{spl}/RIDGE_FULL_{spl.upper()}.npz", ids_by_split[spl],
                                   model_lock["train_prediction_sha256"] if spl == "train" else model_lock["dev_prediction_sha256"])
        systems_by_split[spl].append({"system": "Ridge-CPU", "family": "Ridge-CPU", "query": "none", "seed": None,
                                      "pred": rid, "source": "new_cpu_ridge", "prediction_path": str(ROOT / f"predictions/{spl}/RIDGE_FULL_{spl.upper()}.npz"),
                                      "prediction_sha256": model_lock["train_prediction_sha256"] if spl == "train" else model_lock["dev_prediction_sha256"]})
        anc = anchors_by_split[spl]
        systems_by_split[spl].append({"system": "Persistence", "family": "Persistence", "query": "none", "seed": None,
                                      "pred": np.repeat(anc[:, None, :], 30, axis=1).astype(np.float32),
                                      "source": "deterministic_anchor_hold", "prediction_path": "computed_from_locked_anchor", "prediction_sha256": "not_applicable"})

    # Every fixed H30 case is taken from the old lock; no new-score case selection.
    case_rows = read_csv(QUERY / "qualitative/CASE_MANIFEST.csv")
    if len(case_rows) != 12 or any(int(r["final_test_access"]) != 0 for r in case_rows):
        raise AssertionError("fixed pre-existing case selection does not pass")
    case_ids = [r["anchor_id"] for r in case_rows]
    if len(set(case_ids)) != 12 or not set(case_ids).issubset(set(ids_by_split["dev"].tolist())):
        raise AssertionError("locked case IDs differ from accepted dev inputs")
    case_groups = {r["anchor_id"]: r["source_group_id"] for r in case_rows}

    # Score all accepted 12 old model predictions plus Ridge/Persistence separately; never average predictions.
    episode_final = ROOT / "results/HORIZON_EPISODE_METRICS.csv"
    summary_rows = []
    group_maps: dict[tuple[str, str, int | None, int], dict[str, dict[str, float]]] = {}
    tmp_episode = ROOT / "results/HORIZON_EPISODE_METRICS.csv.part"
    if tmp_episode.exists():
        raise FileExistsError("partial episode metrics exist; preserve them for explicit recovery")
    with tmp_episode.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS, extrasaction="raise")
        writer.writeheader()
        for spl in ("train", "dev"):
            y = targets_by_split[spl].astype(np.float32, copy=False)
            anc = anchors_by_split[spl]
            eps = episodes_by_split[spl]
            gr = groups_by_split[spl]
            for system in systems_by_split[spl]:
                for h in HORIZONS:
                    ep_rows, ep_map = episode_metrics_for_model(
                        system["pred"], y, anc, sigma, safe, threshold, eps, gr,
                        spl, str(system["system"]), str(system["family"]), str(system["query"]),
                        system["seed"], h, times_by_split[spl][h],
                    )
                    writer.writerows(ep_rows)
                    persistence_map = None
                    for candidate in systems_by_split[spl]:
                        if candidate["system"] == "Persistence":
                            if candidate is system:
                                persistence_map = ep_map
                            else:
                                p_rows, persistence_map = episode_metrics_for_model(
                                    candidate["pred"], y, anc, sigma, safe, threshold, eps, gr,
                                    spl, "Persistence", "Persistence", "none", None, h, times_by_split[spl][h],
                                )
                            break
                    if persistence_map is None:
                        raise AssertionError("Persistence score map missing")
                    summary_rows.append(make_summary(
                        spl, str(system["system"]), str(system["family"]), str(system["query"]),
                        system["seed"], h, ep_rows, list(persistence_map.values()), times_by_split[spl][h],
                    ))
                    group_maps[(spl, str(system["system"]), system["seed"], h)] = ep_map
    os.replace(tmp_episode, episode_final)
    write_rows(ROOT / "results/HORIZON_METRICS_BY_SEED.csv", SUMMARY_FIELDS, summary_rows)
    write_rows(ROOT / "results/ACTUAL_HORIZON_TIMESTAMPS.csv", list(time_rows[0].keys()), time_rows)

    # Verify all prior model H30 primary metrics reproduce their accepted tables.
    accepted_rows = read_csv(QUERY / "results/TRAIN_DEV_BY_SEED.csv")
    accepted_lookup = {}
    for r in accepted_rows:
        key = (r["split"], r["config"], r["query_variant"], int(r["seed"]))
        accepted_lookup[key] = r
    reproduced_rows = []
    for row in summary_rows:
        if int(row["horizon"]) != 30 or row["family"] in ("Ridge-CPU", "Persistence"):
            continue
        variant = "old" if row["query"] == "Q0" else "had"
        key = (row["split"], row["family"], variant, int(row["seed"]))
        a = accepted_lookup.get(key)
        if a is None:
            raise AssertionError(f"accepted old H30 result row missing: {key}")
        for old_name, new_name in [
            ("position_standardized_mae", "prefix_position_standardized_mae"),
            ("safe_fd_normalized", "future_internal_fd_safe_normalized"),
        ]:
            ref = float(a[old_name])
            got = float(row[new_name])
            delta = got - ref
            reproduced_rows.append({
                "split": row["split"], "family": row["family"], "query": row["query"],
                "seed": int(row["seed"]), "metric": old_name,
                "accepted_value": ref, "recomputed_value": got, "delta": delta,
                "absolute_tolerance": 1e-5, "status": "PASS" if abs(delta) <= 1e-5 else "FAIL",
            })
    if len(reproduced_rows) != 48 or any(r["status"] != "PASS" for r in reproduced_rows):
        raise AssertionError("old accepted H30 position/safe-FD reproduction failed")
    write_rows(ROOT / "results/H30_REFERENCE_REPRODUCTION.csv", list(reproduced_rows[0].keys()), reproduced_rows)
    write_json(ROOT / "qa/H30_REFERENCE_REPRODUCTION.json", {
        "status": "PASS_OLD_12_MODEL_H30_POSITION_AND_SAFE_FD_REPRODUCED",
        "comparisons": len(reproduced_rows), "absolute_tolerance": 1e-5,
        "max_abs_delta": max(abs(float(r["delta"])) for r in reproduced_rows),
        "accepted_reference": str(QUERY / "results/TRAIN_DEV_BY_SEED.csv"),
        "final_test_access": 0, "heldout_access": 0,
    })

    query_gains_ref = read_csv(QUERY / "results/QUERY_GAINS.csv")
    gain_rows = []
    dev_seed_rows = {(r["system"], int(r["seed"]), r["query"]): r for r in summary_rows if r["split"] == "dev" and int(r["horizon"]) == 30 and r["family"] in ("TS11-D", "STATE-CTRL") and r["seed"] != "NA"}
    for family in ("TS11-D", "STATE-CTRL"):
        for seed in (20261001, 20261002, 20261003):
            for metric, key in (("position_standardized_mae", "prefix_position_standardized_mae"),
                                ("safe_fd_normalized", "future_internal_fd_safe_normalized")):
                old = dev_seed_rows[(f"Q0_{family}", seed, "Q0")][key]
                had = dev_seed_rows[(f"Q1_{family}", seed, "Q1")][key]
                gain = float(old) - float(had)
                match = [r for r in query_gains_ref if r["split"] == "dev" and r["model"] == family and int(r["seed"]) == seed and r["metric"] == metric]
                if len(match) != 1:
                    raise AssertionError(f"source query gain missing or duplicated: {family}/{seed}/{metric}")
                reference = float(match[0]["G_Q_old_minus_had"])
                delta = gain - reference
                gain_rows.append({"family": family, "seed": seed, "metric": metric, "recomputed_old_minus_had": gain,
                                  "accepted_old_minus_had": reference, "delta": delta, "absolute_tolerance": 1e-5,
                                  "status": "PASS" if abs(delta) <= 1e-5 else "FAIL"})
    if any(r["status"] != "PASS" for r in gain_rows):
        raise AssertionError("accepted old H30 per-seed query gain reproduction failed")
    write_rows(ROOT / "results/H30_QUERY_GAIN_REPRODUCTION.csv", list(gain_rows[0].keys()), gain_rows)

    # Paired H30 dev comparison; source-group/episode values are the resampling units.
    state_q0 = [group_maps[("dev", f"Q0_STATE-CTRL", s, 30)] for s in (20261001, 20261002, 20261003)]
    ridge_map = group_maps[("dev", "Ridge-CPU", None, 30)]
    group_ids = sorted(ridge_map)
    paired = []
    pos_state, pos_ridge, fd_state, fd_ridge = {}, {}, {}, {}
    for gid in group_ids:
        sv = [m[gid] for m in state_q0]
        pos_state[gid] = float(np.mean([float(x["prefix_position_standardized_mae"]) for x in sv]))
        fd_state[gid] = float(np.mean([float(x["future_internal_fd_safe_normalized"]) for x in sv]))
        pos_ridge[gid] = float(ridge_map[gid]["prefix_position_standardized_mae"])
        fd_ridge[gid] = float(ridge_map[gid]["future_internal_fd_safe_normalized"])
        paired.append({"source_group_id": gid, "state_q0_mean_seed_position": pos_state[gid],
                       "ridge_position": pos_ridge[gid], "state_minus_ridge_position": pos_state[gid] - pos_ridge[gid],
                       "state_q0_mean_seed_safe_fd": fd_state[gid], "ridge_safe_fd": fd_ridge[gid],
                       "state_minus_ridge_safe_fd": fd_state[gid] - fd_ridge[gid],
                       "final_test_access": 0, "heldout_access": 0})
    if len(group_ids) != 603:
        raise AssertionError(f"H30 paired dev bootstrap has {len(group_ids)} groups, expected 603")
    write_rows(ROOT / "results/PAIRED_H30_RIDGE_VS_STATE_Q0_GROUP.csv", list(paired[0].keys()), paired)
    ci_pos = core.bootstrap_group_differences(pos_state, pos_ridge, group_ids, seed=20261011, replicates=10000)
    ci_fd = core.bootstrap_group_differences(fd_state, fd_ridge, group_ids, seed=20261011, replicates=10000)
    ci_rows = [
        {"split": "dev", "contrast": "L_STATE_Q0_mean_3seed - L_Ridge", "metric": "H30 prefix Full28 position standardized MAE", **ci_pos,
         "interpretation": "positive favors Ridge; conditional on fixed models and repeatedly exposed dev; no p-value"},
        {"split": "dev", "contrast": "L_STATE_Q0_mean_3seed - L_Ridge", "metric": "H30 future-internal safe-FD normalized MAE", **ci_fd,
         "interpretation": "positive favors Ridge; conditional on fixed models and repeatedly exposed dev; no p-value"},
    ]
    write_rows(ROOT / "results/PAIRED_H30_RIDGE_VS_STATE_Q0_CI.csv", list(ci_rows[0].keys()), ci_rows)

    write_rows(ROOT / "figure_data/HORIZON_SUMMARY.csv", SUMMARY_FIELDS, summary_rows)
    write_rows(ROOT / "figure_data/QUERY_GAIN_POINTS.csv", list(gain_rows[0].keys()), gain_rows)
    write_rows(ROOT / "figure_data/PAIRED_H30_RIDGE_VS_STATE_Q0.csv", list(paired[0].keys()), paired)

    # Preserve the 12 originally locked case IDs for a later fixed-path figure; no case is selected by Ridge.
    case_index = {str(a): i for i, a in enumerate(ids.tolist())}
    case_out = []
    for c in case_rows:
        i = case_index[c["anchor_id"]]
        for spl in ("dev",):
            if split[i] != spl:
                raise AssertionError(f"locked case is not dev: {c['anchor_id']}")
        case_out.append({
            "case_id": c["case_id"], "stratum": c["stratum"], "stratum_rank": c["stratum_rank"],
            "source_group_id": c["source_group_id"], "episode_index": int(c["episode_index"]),
            "anchor_id": c["anchor_id"], "dev_index": i, "target_first_frame": int(c["target_first_frame"]),
            "target_last_frame": int(c["target_last_frame"]), "selection_hash": c["selection_hash"],
            "selection_source": str(QUERY / "qualitative/CASE_MANIFEST.csv"), "ridge_score_used_for_selection": False,
            "final_test_access": 0,
        })
    write_rows(ROOT / "figure_data/LOCKED_CASE_IDS.csv", list(case_out[0].keys()), case_out)

    score_finished = utc_now()
    write_json(ROOT / "protocol/OUTCOME_SCORE_ACCESS.json", {
        "status": "PASS_DEV_TARGETS_OPENED_AFTER_RIDGE_MODEL_LOCK",
        "score_started_utc": score_started, "score_finished_utc": score_finished,
        "ridge_model_lock_sha256": sha(model_lock_path),
        "target_source": str(INPUTS), "target_source_sha256": input_hash,
        "target_horizon_used": "targets_native[:,0:30,:] only; future steps 31:60 not scored",
        "target_archive_note": "source NPZ member stores 60 steps; this scorer retains and scores only the prelocked H30 slice",
        "split_rows_scored": {"train": 17757, "dev": 2322},
        "groups": {"train": 2712, "dev": 603},
        "raw_timestamp_metadata": {"episodes": raw_episode_files, "source_columns": ["frame_index", "timestamp"], "action_column_read": False},
        "dev_scope": "previously exposed exploratory development",
        "final_test_access": 0, "heldout_access": 0,
    })
    write_json(ROOT / "qa/HORIZON_ANALYSIS_ACCEPTANCE.json", {
        "status": "PASS_FIXED_H30_HORIZON_ANALYSIS_AND_RIDGE_COMPARISON",
        "horizons": list(HORIZONS), "systems": "12 accepted fixed models plus one CPU Ridge and Persistence; seeds preserved",
        "old_h30_reference_comparisons": len(reproduced_rows), "old_h30_query_gain_comparisons": len(gain_rows),
        "ridge_h30_paired_groups": len(group_ids), "bootstrap_replicates": 10000, "bootstrap_seed": 20261011,
        "p_values": False, "dev_previously_exposed": True, "independent_confirmation": False,
        "final_test_access": 0, "heldout_access": 0, "anchor_relative_position_error_is_duplicate": True,
        "future_internal_fd_h1": "NA", "future_internal_fd_h30_steps": 29,
        "actual_timestamp_metadata_episodes": raw_episode_files,
        "action_target_slice_scored": "0:30 only",
    })
    result = {
        "status": "PASS",
        "score_started_utc": score_started, "score_finished_utc": score_finished,
        "horizon_rows": len(summary_rows), "episode_metric_rows": sum(1 for _ in episode_final.open(encoding="utf-8")) - 1,
        "locked_case_count": len(case_out), "scorer_threadpool_info": pools,
        "reference_prediction_hashes_checked": len(reference_hashes_checked),
        "max_reference_abs_delta": max(abs(float(r["delta"])) for r in reproduced_rows),
        "max_query_gain_abs_delta": max(abs(float(r["delta"])) for r in gain_rows),
        "ridge_lambda": model_lock["selected_global_lambda"],
        "ridge_h30_dev": {
            "position_prefix_standardized_mae": next(r["prefix_position_standardized_mae"] for r in summary_rows if r["split"]=="dev" and r["system"]=="Ridge-CPU" and int(r["horizon"])==30),
            "safe_fd_normalized": next(r["future_internal_fd_safe_normalized"] for r in summary_rows if r["split"]=="dev" and r["system"]=="Ridge-CPU" and int(r["horizon"])==30),
            "position_prefix_skill_vs_persistence": next(r["position_prefix_skill_vs_persistence"] for r in summary_rows if r["split"]=="dev" and r["system"]=="Ridge-CPU" and int(r["horizon"])==30),
            "safe_fd_skill_vs_persistence": next(r["future_internal_fd_safe_skill_vs_persistence"] for r in summary_rows if r["split"]=="dev" and r["system"]=="Ridge-CPU" and int(r["horizon"])==30),
        },
        "ridge_vs_state_q0_h30": ci_rows,
        "final_test_access": 0, "heldout_access": 0,
    }
    write_json(ROOT / "results/HORIZON_SCORE_SUMMARY.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=THREAD_LIMIT):
        raise SystemExit(main())

