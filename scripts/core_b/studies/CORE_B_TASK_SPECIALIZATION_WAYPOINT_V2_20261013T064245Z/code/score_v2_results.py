#!/usr/bin/env python3
"""Score sealed V2 dev predictions, paired contrasts and conditional group bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1", "native_mae")
PHASE_A_CONTRASTS = ("D_S", "D_V", "V_G", "V_S", "I")
WAYPOINT_CONTRASTS = ("B_GAIN", "AUX_VS_SV", "GOAL_VS_SV")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_dev_prefixes() -> dict[str, np.ndarray]:
    path = ROOT / "data/DEV_PREFIXES_NO_LABELS.npz"
    with np.load(path, allow_pickle=False) as archive:
        keys = ("anchor_id", "episode_id", "task_id", "source_cluster", "anchor_frame", "parquet_path", "anchor_native")
        return {key: archive[key].copy() for key in keys}


def read_labels_after_seal(prefix: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("prediction batch has not been sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") not in {"SEALED_BEFORE_TARGET_ACCESS", "SEALED_WITH_MISSING_PREDICTIONS_BEFORE_TARGET_ACCESS"}:
        raise RuntimeError("invalid prediction seal state")
    if seal.get("dev_target_labels_read_before_seal") is not False:
        raise RuntimeError("seal indicates dev labels were accessed too early")
    target_path = ROOT / "data/DEV_TARGETS_SCORER_ONLY.npz"
    if target_path.exists():
        with np.load(target_path, allow_pickle=False) as archive:
            existing = {key: archive[key].copy() for key in archive.files}
        exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
        exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
        qa_path = ROOT / "qa/DEV_TARGET_SCORING_QA.json"
        if exposure.get("dev_target_labels_read_after_prediction_seal") is not True or not qa_path.exists():
            backup = exposure_path.with_name(f"OUTCOME_EXPOSURE.json.pre-dev-score-resume-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak")
            if not backup.exists():
                import shutil
                shutil.copy2(exposure_path, backup)
            seal_sha = sha_file(seal_path)
            exposure.update({
                "dev_target_labels_read": True,
                "dev_target_labels_read_after_prediction_seal": True,
                "dev_target_label_rows_selected": int(len(existing["anchor_id"]) * 30),
                "dev_target_artifact": str(target_path),
                "dev_target_artifact_sha256": sha_file(target_path),
                "prediction_seal_sha256": seal_sha,
                "current_split_target_values_loaded_after_prediction_seal": True,
                "current_split_raw_parquet_target_rows_read": 0,
                "current_split_reused_historical_scorer_only_target_artifact": True,
                "prior_split_dev_target_access_preserved": True,
                "prior_split_dev_scores_used_for_corrected_selection": False,
                "final_heldout_payload_access": 0,
                "new_final_aggregate_counts_read_or_printed": False,
            })
            atomic_json(exposure_path, exposure)
            atomic_json(qa_path, {
                "schema_version": 1, "created_utc": utc_now(), "status": "PASS",
                "prediction_seal_path": str(seal_path), "prediction_seal_sha256": seal_sha,
                "selected_dev_windows": len(existing["anchor_id"]),
                "future_action_rows_read": int(len(existing["anchor_id"]) * 30),
                "raw_parquet_target_rows_read_in_current_scoring": 0,
                "target_artifact_reused_from_superseded_split": True,
                "loaded_after_current_split_prediction_seal": True,
                "target_frame_rule": "a+1 through a+30 only; scorer-only artifact created after superseded split seal and reused only after corrected prediction seal",
                "labels_entered_model_forward": False, "final_heldout_payload_access": 0,
            })
        return existing

    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    raw_root = Path(resolved["raw_dataset_root"]).resolve()
    rows_by_parquet: dict[str, list[int]] = defaultdict(list)
    for i, raw in enumerate(prefix["parquet_path"].tolist()):
        rows_by_parquet[str(raw)].append(i)
    targets = np.empty((len(prefix["anchor_id"]), 30, 28), dtype=np.float32)
    read_rows = 0
    for relative, indices in rows_by_parquet.items():
        path = (raw_root / relative).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise FileNotFoundError(f"dev scorer source parquet missing/outside accepted raw root: {relative}")
        requested: set[int] = set()
        for ix in indices:
            anchor = int(prefix["anchor_frame"][ix])
            requested.update(range(anchor + 1, anchor + 31))
        sorted_requested = sorted(requested)
        table = ds.dataset(str(path), format="parquet").to_table(
            columns=["frame_index", "action"],
            filter=ds.field("frame_index").isin(sorted_requested),
        )
        frames = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if actions.shape != (len(frames), 28) or not np.isfinite(actions).all():
            raise RuntimeError(f"dev scorer target action shape/finite failure: {relative}")
        if len(frames) != len(sorted_requested) or set(frames.tolist()) != set(sorted_requested):
            raise RuntimeError(f"dev scorer target row filter mismatch: {relative}")
        mapping = {int(frame): actions[j].astype(np.float32) for j, frame in enumerate(frames.tolist())}
        for ix in indices:
            anchor = int(prefix["anchor_frame"][ix])
            targets[ix] = np.stack([mapping[a] for a in range(anchor + 1, anchor + 31)])
            read_rows += 30
    if not np.isfinite(targets).all():
        raise RuntimeError("scorer-only dev target artifact has nonfinite labels")
    arrays = {
        "anchor_id": prefix["anchor_id"],
        "episode_id": prefix["episode_id"],
        "task_id": prefix["task_id"],
        "source_cluster": prefix["source_cluster"],
        "target_native_h30": targets,
    }
    tmp = target_path.with_name(f".{target_path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target_path)
    exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    backup = exposure_path.with_name(f"OUTCOME_EXPOSURE.json.pre-dev-score-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(exposure_path, backup)
    exposure.update({
        "dev_target_labels_read": True,
        "dev_target_labels_read_after_prediction_seal": True,
        "current_split_dev_target_artifact_pending_after_new_prediction_seal": False,
        "current_split_target_values_loaded_after_prediction_seal": True,
        "current_split_raw_parquet_target_rows_read": int(read_rows),
        "dev_target_label_rows_selected": int(read_rows),
        "dev_target_artifact": str(target_path),
        "dev_target_artifact_sha256": sha_file(target_path),
        "prediction_seal_sha256": sha_file(seal_path),
        "current_split_target_values_loaded_after_prediction_seal": True,
        "current_split_raw_parquet_target_rows_read": int(read_rows),
        "current_split_reused_historical_scorer_only_target_artifact": False,
        "prior_split_dev_target_access_preserved": bool(exposure.get("historical_superseded_split_dev_target_access", {}).get("read_after_prediction_seal", False)),
        "prior_split_dev_scores_used_for_corrected_selection": False,
        "final_heldout_payload_access": 0,
        "new_final_aggregate_counts_read_or_printed": False,
    })
    atomic_json(exposure_path, exposure)
    atomic_json(ROOT / "qa/DEV_TARGET_SCORING_QA.json", {
        "schema_version": 1, "created_utc": utc_now(), "status": "PASS",
        "prediction_seal_path": str(seal_path), "prediction_seal_sha256": sha_file(seal_path),
        "selected_dev_windows": len(prefix["anchor_id"]), "future_action_rows_read": int(read_rows),
        "target_frame_rule": "a+1 through a+30 only, selected by frame_index filter after prediction seal",
        "labels_entered_model_forward": False, "final_heldout_payload_access": 0,
    })
    with np.load(target_path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_predictions(prefix: dict[str, np.ndarray]) -> tuple[dict[tuple[str, int], dict[str, np.ndarray]], dict[tuple[str, int], dict[str, np.ndarray]], dict[str, dict[str, Any]], list[str]]:
    queue = json.loads((ROOT / "coordination/queues/FORMAL_DEV.json").read_text(encoding="utf-8"))
    seal = json.loads((ROOT / "protocol/PREDICTION_BATCH_SEAL.json").read_text(encoding="utf-8"))
    sealed_entries = {str(row["job_id"]): row for row in seal.get("prediction_entries", [])}
    sealed_missing = set(str(x) for x in seal.get("missing_prediction_jobs", []))
    pred_map: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    goal_map: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    prefix_index = {str(anchor): i for i, anchor in enumerate(prefix["anchor_id"].tolist())}
    if len(prefix_index) != len(prefix["anchor_id"]):
        raise RuntimeError("dev prefix anchor IDs are not unique")
    for raw in queue["jobs"]:
        watcher = json.loads(Path(raw).read_text(encoding="utf-8"))
        config_path = Path(watcher["command"][-1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        job_id = str(config["job_id"])
        if job_id in sealed_missing:
            missing.append(job_id)
            continue
        seal_entry = sealed_entries.get(job_id)
        if seal_entry is None:
            raise RuntimeError(f"prediction job was not present in pre-score seal: {job_id}")
        out_dir = Path(config["output_root"])
        complete_path = out_dir / "PREDICTION_COMPLETE.json"
        pred_path = out_dir / "PREDICTIONS.safetensors"
        ids_path = out_dir / "PREDICTION_ROWS.csv"
        if not all(path.is_file() for path in (complete_path, pred_path, ids_path)):
            missing.append(str(config["job_id"]))
            continue
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS" or sha_file(pred_path) != complete.get("prediction_sha256"):
            raise RuntimeError(f"sealed prediction completion/hash invalid: {job_id}")
        if sha_file(pred_path) != seal_entry.get("prediction_sha256") or sha_file(ids_path) != seal_entry.get("rows_sha256"):
            raise RuntimeError(f"sealed prediction artifact changed after seal: {job_id}")
        if complete.get("future_target_arrays_requested") is not False:
            raise RuntimeError(f"inference receipt records target arrays: {config['job_id']}")
        tensors = load_file(str(pred_path), device="cpu")
        pred_native = tensors["prediction_native"].numpy().astype(np.float32, copy=False)
        goal_tensor = tensors.get("goal_prediction_native")
        goal_native = None if goal_tensor is None else goal_tensor.numpy().astype(np.float32, copy=False)
        with ids_path.open(newline="", encoding="utf-8") as handle:
            ids = list(csv.DictReader(handle))
        if pred_native.shape != (len(ids), 30, 28):
            raise RuntimeError(f"prediction tensor/ID shape mismatch: {config['job_id']}")
        model_id = str(config["model_id"])
        seed = int(config["seed"])
        values: dict[str, np.ndarray] = {}
        goal_values: dict[str, np.ndarray] = {}
        seen: set[str] = set()
        expected_mask = np.ones(len(prefix["anchor_id"]), dtype=bool)
        if config.get("scope") == "expert":
            expected_mask &= prefix["task_id"] == int(config["task_id"])
        expected_order = np.flatnonzero(expected_mask)
        expected_order = expected_order[np.argsort(prefix["anchor_id"][expected_order], kind="stable")]
        expected_ids = [str(prefix["anchor_id"][ix]) for ix in expected_order.tolist()]
        actual_ids = [str(row["anchor_id"]) for row in ids]
        if actual_ids != expected_ids:
            raise RuntimeError(f"prediction support/order differs from locked dev prefix selection: {job_id}")
        for j, row in enumerate(ids):
            anchor_id = str(row["anchor_id"])
            if anchor_id not in prefix_index or anchor_id in seen:
                raise RuntimeError(f"prediction row is absent/duplicated in V2 dev prefixes: {config['job_id']} {anchor_id}")
            ix = prefix_index[anchor_id]
            if int(row["task_id"]) != int(prefix["task_id"][ix]) or int(row["episode_id"]) != int(prefix["episode_id"][ix]):
                raise RuntimeError(f"prediction sample identity mismatch: {config['job_id']} {anchor_id}")
            if config.get("scope") == "expert" and int(row["task_id"]) != int(config["task_id"]):
                raise RuntimeError(f"expert prediction contains another task: {config['job_id']}")
            values[anchor_id] = pred_native[j]
            if goal_native is not None:
                if goal_native.shape != (len(ids), 28):
                    raise RuntimeError(f"waypoint goal prediction shape mismatch: {job_id}")
                goal_values[anchor_id] = goal_native[j]
            seen.add(anchor_id)
        key = (model_id, seed)
        if key in pred_map:
            raise RuntimeError(f"duplicate model/seed prediction set: {key}")
        pred_map[key] = values
        goal_map[key] = goal_values
        meta_map[config["job_id"]] = {**config, "complete_path": str(complete_path), "prediction_path": str(pred_path), "prediction_sha256": complete["prediction_sha256"]}
    if set(missing) != sealed_missing:
        raise RuntimeError("inference missing-job set changed after the pre-score seal")
    if set(meta_map) != set(sealed_entries):
        raise RuntimeError("sealed prediction job set differs from loaded prediction files")
    return pred_map, goal_map, meta_map, missing


def window_metrics(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray,
                   metric_sigma: np.ndarray, metric_fd: np.ndarray) -> dict[str, float]:
    error = np.abs(pred - target) / metric_sigma[None]
    position = float(error.mean())
    fd_error = np.abs(np.diff(pred, axis=0) - np.diff(target, axis=0)) / metric_fd[None]
    fd = float(fd_error.mean())
    seam = float((np.abs((pred[0] - anchor) - (target[0] - anchor)) / metric_fd).mean())
    native = float(np.abs(pred - target).mean())
    return {"position_mae": position, "safe_fd_l1": fd, "seam_safe_fd_l1": seam, "native_mae": native}


def episode_equal(values: list[dict[str, float]]) -> dict[str, float]:
    return {metric: float(np.mean([row[metric] for row in values])) for metric in METRICS}


def contrast_values(task: int, seed: int, metric: str, get: Any) -> dict[str, float | None]:
    gs = get("G-S", seed, task, metric)
    gv = get("G-V", seed, task, metric)
    ss = get(f"S-S-{task}", seed, task, metric)
    sv = get(f"S-V-{task}", seed, task, metric)
    if any(value is None for value in (gs, gv, ss, sv)):
        return {name: None for name in PHASE_A_CONTRASTS}
    ds = gs - ss
    dv = gv - sv
    vg = gs - gv
    vs = ss - sv
    interaction = dv - ds
    if abs(interaction - (vs - vg)) > 1e-10:
        raise RuntimeError(f"D/V/I algebra identity failed task={task} seed={seed} metric={metric}")
    return {"D_S": ds, "D_V": dv, "V_G": vg, "V_S": vs, "I": interaction}


def main() -> None:
    summary_path = ROOT / "results/RESULTS_SUMMARY.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") in {"PASS", "PARTIAL"}:
        print(json.dumps({"status": "already_scored", "summary": str(summary_path)}))
        return
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("scorer refuses to run before prediction seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    prefix = load_dev_prefixes()
    # Recheck all sealed prediction artifacts and their full dev support before opening dev targets.
    pred_map, goal_pred_map, pred_meta, missing_prediction_jobs = load_predictions(prefix)
    if not pred_map or sum(len(rows) for rows in pred_map.values()) == 0:
        raise RuntimeError("scorer refuses to open dev targets when the sealed batch contains zero legal predictions")
    label_archive = read_labels_after_seal(prefix)
    if not np.array_equal(label_archive["anchor_id"], prefix["anchor_id"]):
        raise RuntimeError("sealed dev labels are not aligned to label-free dev prefix IDs")
    targets = label_archive["target_native_h30"]
    if targets.shape != (len(prefix["anchor_id"]), 30, 28):
        raise RuntimeError("scorer-only dev target shape mismatch")
    metric_lock = json.loads((ROOT / "protocol/METRIC_SCALE_LOCK.json").read_text(encoding="utf-8"))
    metric_sigma = np.asarray(metric_lock["target_sigma"], dtype=np.float64)
    metric_fd = np.asarray(metric_lock["safe_fd_scale"], dtype=np.float64)
    prefix_index = {str(anchor): i for i, anchor in enumerate(prefix["anchor_id"].tolist())}
    waypoint_episode_values: dict[tuple[str, int, int], dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (model_id, seed), goals in goal_pred_map.items():
        if model_id not in {"B-AUX_ONLY", "B-GOAL_COND"}:
            continue
        for anchor_id, goal_native in goals.items():
            ix = prefix_index[anchor_id]
            target_endpoint = targets[ix, -1].astype(np.float64)
            error = np.abs(goal_native.astype(np.float64) - target_endpoint)
            waypoint_episode_values[(model_id, seed, int(prefix["task_id"][ix]))][int(prefix["episode_id"][ix])].append(
                (float(error.mean()), float((error / metric_sigma).mean()))
            )
    waypoint_endpoint_rows: list[dict[str, Any]] = []
    waypoint_endpoint_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for (model_id, seed, task), episodes in waypoint_episode_values.items():
        native = [float(np.mean([value[0] for value in windows])) for windows in episodes.values()]
        normalized = [float(np.mean([value[1] for value in windows])) for windows in episodes.values()]
        group_count = len({
            str(prefix["source_cluster"][prefix_index[anchor]])
            for anchor in goal_pred_map[(model_id, seed)]
        })
        row = {
            "model_id": model_id, "seed": seed, "task_id": task,
            "dev_episodes": len(episodes), "dev_source_groups": group_count,
            "goal_endpoint_mae_native": float(np.mean(native)) if native else "NA",
            "goal_endpoint_mae_normalized": float(np.mean(normalized)) if normalized else "NA",
            "support_note": "small_support" if len(episodes) < 20 else "",
        }
        waypoint_endpoint_rows.append(row)
        waypoint_endpoint_lookup[(model_id, seed)] = row

    per_anchor: list[dict[str, Any]] = []
    episode_values: dict[tuple[str, int, int, int], dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    episode_cluster: dict[tuple[int, int], str] = {}
    for i, raw_anchor in enumerate(prefix["anchor_id"].tolist()):
        anchor_id = str(raw_anchor)
        task = int(prefix["task_id"][i])
        episode = int(prefix["episode_id"][i])
        cluster = str(prefix["source_cluster"][i])
        prior = episode_cluster.setdefault((task, episode), cluster)
        if prior != cluster:
            raise RuntimeError(f"dev source cluster changes within episode {episode}")
        anchor = prefix["anchor_native"][i].astype(np.float64)
        target = targets[i].astype(np.float64)
        persistence = np.broadcast_to(anchor[None], (30, 28)).copy()
        metrics = window_metrics(persistence, target, anchor, metric_sigma, metric_fd)
        per_anchor.append({
            "model_id": "Persistence", "seed": "shared", "task_id": task, "episode_id": episode,
            "source_cluster": cluster, "anchor_id": anchor_id, "anchor_frame": int(prefix["anchor_frame"][i]),
            **metrics, "prediction_available": True,
        })
        episode_values[("Persistence", 0, task, episode)][cluster].append(metrics)

    for (model_id, seed), predictions in pred_map.items():
        for anchor_id, pred in predictions.items():
            i = prefix_index[anchor_id]
            task = int(prefix["task_id"][i])
            episode = int(prefix["episode_id"][i])
            cluster = str(prefix["source_cluster"][i])
            metrics = window_metrics(pred.astype(np.float64), targets[i].astype(np.float64),
                                     prefix["anchor_native"][i].astype(np.float64), metric_sigma, metric_fd)
            per_anchor.append({
                "model_id": model_id, "seed": seed, "task_id": task, "episode_id": episode,
                "source_cluster": cluster, "anchor_id": anchor_id, "anchor_frame": int(prefix["anchor_frame"][i]),
                **metrics, "prediction_available": True,
            })
            episode_values[(model_id, seed, task, episode)][cluster].append(metrics)

    episode_metric: dict[tuple[str, int, int, int, str], float] = {}
    by_model_task_seed: dict[tuple[str, int, int], list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for (model_id, seed, task, episode), cluster_rows in episode_values.items():
        flat = [row for rows in cluster_rows.values() for row in rows]
        if not flat:
            continue
        value = episode_equal(flat)
        by_model_task_seed[(model_id, seed, task)].append((episode, value))
        for metric in METRICS:
            episode_metric[(model_id, seed, task, episode, metric)] = value[metric]

    abs_rows: list[dict[str, Any]] = []
    abs_lookup: dict[tuple[str, int, int, str], float | None] = {}
    learned_models_by_task: dict[int, list[str]] = {}
    for task in TASK_IDS:
        models = ["G-S", "G-V", f"S-S-{task}", f"S-V-{task}"]
        if task == 119:
            models.extend(["B-AUX_ONLY", "B-GOAL_COND"])
        learned_models_by_task[task] = models
        for model_id in models:
            for seed in SEEDS:
                episodes = by_model_task_seed.get((model_id, seed, task), [])
                values = [metric_row for _, metric_row in episodes]
                support_eps = len(values)
                support_groups = len({episode_cluster[(task, episode)] for episode, _ in episodes})
                row = {
                    "model_id": model_id, "seed": seed, "task_id": task,
                    "dev_anchors": sum(1 for x in per_anchor if x["model_id"] == model_id and x["seed"] == seed and int(x["task_id"]) == task),
                    "dev_episodes": support_eps, "dev_source_groups": support_groups,
                    "status": "PASS" if support_eps else "NA",
                    "support_note": "small_support" if support_eps < 20 else "",
                }
                for metric in METRICS:
                    value = float(np.mean([item[metric] for item in values])) if values else None
                    row[metric] = "NA" if value is None else value
                    abs_lookup[(model_id, seed, task, metric)] = value
                row["selection_J"] = "NA" if not values else float(np.mean([item["position_mae"] + 0.1 * item["safe_fd_l1"] for item in values]))
                abs_rows.append(row)
    for task in TASK_IDS:
        episodes = by_model_task_seed.get(("Persistence", 0, task), [])
        values = [value for _, value in episodes]
        row = {
            "model_id": "Persistence", "seed": "shared", "task_id": task,
            "dev_anchors": sum(1 for x in per_anchor if x["model_id"] == "Persistence" and int(x["task_id"]) == task),
            "dev_episodes": len(values),
            "dev_source_groups": len({episode_cluster[(task, episode)] for episode, _ in episodes}),
            "status": "PASS" if values else "NA",
            "support_note": "small_support" if len(values) < 20 else "",
        }
        for metric in METRICS:
            row[metric] = float(np.mean([value[metric] for value in values])) if values else "NA"
        row["selection_J"] = float(np.mean([value["position_mae"] + 0.1 * value["safe_fd_l1"] for value in values])) if values else "NA"
        abs_rows.append(row)

    contrast_rows: list[dict[str, Any]] = []
    contrast_lookup: dict[tuple[int, int, str, str], float | None] = {}
    for task in TASK_IDS:
        for seed in SEEDS:
            for metric in METRICS:
                values = contrast_values(task, seed, metric, lambda m, s, t, met: abs_lookup.get((m, s, t, met)))
                for contrast, value in values.items():
                    contrast_lookup[(task, seed, contrast, metric)] = value
                    contrast_rows.append({
                        "task_id": task, "seed": seed, "metric": metric, "contrast": contrast,
                        "value": "NA" if value is None else value, "positive_means_benefit": True,
                        "available_task_count": int(value is not None),
                    })
                sv = abs_lookup.get((f"S-V-{task}", seed, task, metric))
                aux = abs_lookup.get(("B-AUX_ONLY", seed, task, metric)) if task == 119 else None
                goal = abs_lookup.get(("B-GOAL_COND", seed, task, metric)) if task == 119 else None
                if task == 119:
                    for name, val in {
                        "B_GAIN": None if aux is None or goal is None else aux - goal,
                        "AUX_VS_SV": None if aux is None or sv is None else sv - aux,
                        "GOAL_VS_SV": None if goal is None or sv is None else sv - goal,
                    }.items():
                        contrast_rows.append({"task_id": task, "seed": seed, "metric": metric, "contrast": name,
                                             "value": "NA" if val is None else val, "positive_means_benefit": True,
                                             "available_task_count": int(val is not None)})
                        contrast_lookup[(task, seed, name, metric)] = val

    loo_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for metric in METRICS:
            for contrast in PHASE_A_CONTRASTS:
                task_vals = {task: contrast_lookup.get((task, seed, contrast, metric)) for task in TASK_IDS}
                for omitted in TASK_IDS:
                    remaining = [value for task, value in task_vals.items() if task != omitted and value is not None]
                    loo_rows.append({
                        "seed": seed, "metric": metric, "contrast": contrast, "omitted_task_id": omitted,
                        "loo_mean": float(np.mean(remaining)) if remaining else "NA",
                        "included_tasks": len(remaining), "planned_tasks": 4,
                    })

    summary_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for contrast in (*PHASE_A_CONTRASTS, *WAYPOINT_CONTRASTS):
            if contrast in WAYPOINT_CONTRASTS:
                task_scope = (119,)
            else:
                task_scope = TASK_IDS
            seed_values: list[float] = []
            for seed in SEEDS:
                per_task = [contrast_lookup.get((task, seed, contrast, metric)) for task in task_scope]
                available = [value for value in per_task if value is not None]
                if available:
                    seed_values.append(float(np.mean(available)))
            summary_scope = "task_119" if contrast in WAYPOINT_CONTRASTS else ("five_task_macro" if len(seed_values) == len(SEEDS) and all(
                all(contrast_lookup.get((task, seed, contrast, metric)) is not None for task in TASK_IDS) for seed in SEEDS
            ) else "available_task_macro")
            summary_rows.append({
                "scope": summary_scope, "task_id": "119" if contrast in WAYPOINT_CONTRASTS else "ALL5_OR_AVAILABLE",
                "contrast": contrast, "metric": metric,
                "mean_across_seeds": float(np.mean(seed_values)) if seed_values else "NA",
                "sample_sd_across_seeds": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else "NA",
                "median_across_seeds": float(np.median(seed_values)) if seed_values else "NA",
                "n_seeds": len(seed_values), "n_tasks_per_seed": len(task_scope),
                "session_independence_verified": False,
            })
            if contrast in PHASE_A_CONTRASTS:
                for task in TASK_IDS:
                    task_seed_values = [
                        contrast_lookup[(task, seed, contrast, metric)]
                        for seed in SEEDS
                        if contrast_lookup.get((task, seed, contrast, metric)) is not None
                    ]
                    summary_rows.append({
                        "scope": "task", "task_id": task, "contrast": contrast, "metric": metric,
                        "mean_across_seeds": float(np.mean(task_seed_values)) if task_seed_values else "NA",
                        "sample_sd_across_seeds": float(np.std(task_seed_values, ddof=1)) if len(task_seed_values) > 1 else "NA",
                        "median_across_seeds": float(np.median(task_seed_values)) if task_seed_values else "NA",
                        "n_seeds": len(task_seed_values), "n_tasks_per_seed": 1,
                        "session_independence_verified": False,
                    })

    # Paired source-group bootstrap; one set of source draws is shared by every method, seed and metric.
    bootstrap_replicates = 5000
    bootstrap_seed = 202610134
    cluster_to_episodes: dict[int, dict[str, list[int]]] = {task: defaultdict(list) for task in TASK_IDS}
    for (task, episode), cluster in episode_cluster.items():
        cluster_to_episodes[task][cluster].append(episode)
    ordered_clusters = {task: sorted(groups) for task, groups in cluster_to_episodes.items()}
    rng = np.random.default_rng(bootstrap_seed)
    draws: list[dict[int, list[str]]] = []
    for _ in range(bootstrap_replicates):
        draw: dict[int, list[str]] = {}
        for task in TASK_IDS:
            clusters = ordered_clusters[task]
            if clusters:
                selected = rng.choice(clusters, size=len(clusters), replace=True).tolist()
                draw[task] = [str(value) for value in selected]
            else:
                draw[task] = []
        draws.append(draw)

    def sampled_model_metric(model: str, seed: int, task: int, metric: str, selected_clusters: list[str]) -> float | None:
        sampled_eps: list[int] = []
        for cluster in selected_clusters:
            sampled_eps.extend(cluster_to_episodes[task].get(cluster, []))
        values = [episode_metric.get((model, seed, task, episode, metric)) for episode in sampled_eps]
        values = [float(value) for value in values if value is not None]
        return float(np.mean(values)) if values else None

    boot_task: dict[tuple[int, int, str, str], np.ndarray] = {}
    boot_waypoint: dict[tuple[int, int, str, str], np.ndarray] = {}
    for task in TASK_IDS:
        for seed in SEEDS:
            for metric in METRICS:
                array = np.full(bootstrap_replicates, np.nan, dtype=np.float64)
                for b, draw in enumerate(draws):
                    values = contrast_values(
                        task, seed, metric,
                        lambda model, sd, tid, met: sampled_model_metric(model, sd, tid, met, draw.get(tid, [])),
                    )
                    for contrast in PHASE_A_CONTRASTS:
                        if values[contrast] is not None:
                            boot_task[(task, seed, contrast, metric)] = boot_task.get((task, seed, contrast, metric), array.copy())
                            boot_task[(task, seed, contrast, metric)][b] = float(values[contrast])
                    if task == 119:
                        aux = sampled_model_metric("B-AUX_ONLY", seed, task, metric, draw[task])
                        goal = sampled_model_metric("B-GOAL_COND", seed, task, metric, draw[task])
                        sv = sampled_model_metric("S-V-119", seed, task, metric, draw[task])
                        for name, val in {
                            "B_GAIN": None if aux is None or goal is None else aux - goal,
                            "AUX_VS_SV": None if aux is None or sv is None else sv - aux,
                            "GOAL_VS_SV": None if goal is None or sv is None else sv - goal,
                        }.items():
                            key = (task, seed, name, metric)
                            if val is not None:
                                boot_waypoint.setdefault(key, np.full(bootstrap_replicates, np.nan, dtype=np.float64))[b] = float(val)

    ci_rows: list[dict[str, Any]] = []
    for key, values in boot_task.items():
        task, seed, contrast, metric = key
        finite = values[np.isfinite(values)]
        support_n = len(ordered_clusters[task])
        ci_rows.append({
            "scope": "task", "task_id": task, "seed": seed, "contrast": contrast, "metric": metric,
            "ci_low_2_5": float(np.percentile(finite, 2.5)) if support_n >= 2 and len(finite) else "NA",
            "ci_high_97_5": float(np.percentile(finite, 97.5)) if support_n >= 2 and len(finite) else "NA",
            "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
            "source_groups": support_n, "status": "CONDITIONAL_DIAGNOSTIC" if support_n >= 2 and len(finite) else "NA_LT2_GROUPS",
            "dependency_note": "session/trial independence unverified; interval is conditional on source-group partition",
        })
    for key, values in boot_waypoint.items():
        task, seed, contrast, metric = key
        finite = values[np.isfinite(values)]
        support_n = len(ordered_clusters[task])
        ci_rows.append({
            "scope": "task_119", "task_id": task, "seed": seed, "contrast": contrast, "metric": metric,
            "ci_low_2_5": float(np.percentile(finite, 2.5)) if support_n >= 2 and len(finite) else "NA",
            "ci_high_97_5": float(np.percentile(finite, 97.5)) if support_n >= 2 and len(finite) else "NA",
            "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
            "source_groups": support_n, "status": "CONDITIONAL_DIAGNOSTIC" if support_n >= 2 and len(finite) else "NA_LT2_GROUPS",
            "dependency_note": "session/trial independence unverified; interval is conditional on source-group partition",
        })
    for seed in SEEDS:
        for metric in METRICS:
            for contrast in PHASE_A_CONTRASTS:
                task_arrays = [boot_task.get((task, seed, contrast, metric)) for task in TASK_IDS]
                if not all(array is not None for array in task_arrays):
                    continue
                matrix = np.stack(task_arrays, axis=0)
                full_finite = np.isfinite(matrix).all(axis=0)
                macro = matrix[:, full_finite].mean(axis=0) if full_finite.any() else np.asarray([])
                ci_rows.append({
                    "scope": "five_task_macro", "task_id": "ALL5", "seed": seed, "contrast": contrast, "metric": metric,
                    "ci_low_2_5": float(np.percentile(macro, 2.5)) if len(macro) else "NA",
                    "ci_high_97_5": float(np.percentile(macro, 97.5)) if len(macro) else "NA",
                    "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed,
                    "source_groups": ";".join(f"{task}:{len(ordered_clusters[task])}" for task in TASK_IDS),
                    "status": "CONDITIONAL_DIAGNOSTIC" if len(macro) else "NA_MISSING_TASK_CONTRAST",
                    "dependency_note": "same within-task source draws shared across methods/seeds/metrics; sessions unverified",
                })

    write_csv(ROOT / "results/ABSOLUTE_LOSSES.csv", list(abs_rows[0]), abs_rows)
    write_csv(ROOT / "results/ALL_SEED_D_V_I.csv", list(contrast_rows[0]), contrast_rows)
    write_csv(ROOT / "results/ALL_SEED_LOO.csv", list(loo_rows[0]), loo_rows)
    write_csv(ROOT / "results/CONDITIONAL_CI.csv", list(ci_rows[0]) if ci_rows else ["scope", "task_id", "seed", "contrast", "metric", "status"], ci_rows)
    write_csv(ROOT / "results/SUMMARY_STATISTICS.csv", list(summary_rows[0]), summary_rows)

    waypoint_rows = [row for row in contrast_rows if row["contrast"] in WAYPOINT_CONTRASTS]
    selected = json.loads((ROOT / "protocol/LR_EPOCH_SELECTION.json").read_text(encoding="utf-8"))
    sv_choice = next((row for row in selected.get("logical_model_choices", []) if row["model_id"] == "S-V-119"), {})
    for seed in SEEDS:
        aux = waypoint_endpoint_lookup.get(("B-AUX_ONLY", seed), {}).get("goal_endpoint_mae_normalized")
        goal = waypoint_endpoint_lookup.get(("B-GOAL_COND", seed), {}).get("goal_endpoint_mae_normalized")
        gain = None if aux in (None, "NA") or goal in (None, "NA") else float(aux) - float(goal)
        waypoint_rows.append({
            "task_id": 119, "seed": seed, "metric": "goal_endpoint_mae_normalized",
            "contrast": "B_GOAL_HEAD_GAIN", "value": "NA" if gain is None else gain,
            "positive_means_benefit": True, "available_task_count": int(gain is not None),
        })
    for row in waypoint_rows:
        row["selected_sv_lr"] = sv_choice.get("selected_lr", "NA")
        row["selected_sv_epoch"] = sv_choice.get("selected_epoch", "NA")
    write_csv(ROOT / "results/WAYPOINT.csv", list(waypoint_rows[0]) if waypoint_rows else ["task_id", "seed", "metric", "contrast", "value"], waypoint_rows)
    endpoint_fields = list(waypoint_endpoint_rows[0]) if waypoint_endpoint_rows else [
        "model_id", "seed", "task_id", "dev_episodes", "dev_source_groups",
        "goal_endpoint_mae_native", "goal_endpoint_mae_normalized", "support_note",
    ]
    write_csv(ROOT / "results/WAYPOINT_ENDPOINT.csv", endpoint_fields, waypoint_endpoint_rows)

    persistence_rows = []
    for row in abs_rows:
        model = row["model_id"]
        if model == "Persistence" or model.startswith("B-"):
            continue
        seed = int(row["seed"]); task = int(row["task_id"])
        base = next((x for x in abs_rows if x["model_id"] == "Persistence" and int(x["task_id"]) == task), None)
        if base is None:
            continue
        for metric in METRICS:
            model_v = row[metric]
            base_v = base[metric]
            persistence_rows.append({
                "model_id": model, "seed": seed, "task_id": task, "metric": metric,
                "persistence_minus_model": "NA" if model_v == "NA" or base_v == "NA" else float(base_v) - float(model_v),
                "positive_means_model_better_than_persistence": True,
            })
    write_csv(ROOT / "results/PERSISTENCE.csv", list(persistence_rows[0]) if persistence_rows else ["model_id", "seed", "task_id", "metric"], persistence_rows)

    figure_rows = []
    for row in per_anchor:
        figure_rows.append(row)
    write_csv(ROOT / "figure_data/DEV_ANCHOR_METRICS.csv", list(figure_rows[0]) if figure_rows else ["model_id", "seed", "task_id"], figure_rows)

    resource_rows = []
    for phase in ("screening", "formal", "waypoint"):
        for manifest_path in (ROOT / "model_only" / phase).glob("*/MODEL_MANIFEST.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete_path = manifest_path.parent / "RUN_COMPLETE.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else {}
            resource_rows.append({
                "phase": phase, "job_id": manifest["job_id"], "model_id": manifest["model_id"],
                "task_ids": ";".join(str(x) for x in manifest.get("task_ids", [])),
                "seed": manifest["seed"], "lr": manifest["lr"], "epochs": manifest["epochs"],
                "optimizer_steps": manifest["optimizer_steps"], "elapsed_seconds": complete.get("elapsed_seconds", ""),
                "physical_gpu": manifest["physical_gpu"], "peak_memory_allocated_mib": manifest["peak_memory_allocated_mib"],
                "peak_memory_reserved_mib": manifest["peak_memory_reserved_mib"], "model_file_sha256": manifest["model_file_sha256"],
            })
    resource_rows.sort(key=lambda row: (row["phase"], row["job_id"]))
    if resource_rows:
        write_csv(ROOT / "results/TRAINING_RESOURCE_LEDGER.csv", list(resource_rows[0]), resource_rows)
    else:
        write_csv(ROOT / "results/TRAINING_RESOURCE_LEDGER.csv", ["phase", "job_id", "model_id", "seed", "lr", "epochs", "optimizer_steps", "physical_gpu"], [])

    dev_support = []
    with (ROOT / "inventory/DEV_SUPPORT_V2.csv").open(newline="", encoding="utf-8") as handle:
        dev_support = list(csv.DictReader(handle))
    formal_status_path = ROOT / "coordination/PHASE_STATUS_FORMAL.json"
    waypoint_status_path = ROOT / "coordination/PHASE_STATUS_WAYPOINT.json"
    screening_lock = json.loads((ROOT / "protocol/LR_EPOCH_SELECTION.json").read_text())
    train_task_rows = read_csv(ROOT / "inventory/TRAIN_TASKS_V2.csv")
    train_episode_count = sum(int(row["train_episodes"]) for row in train_task_rows)
    train_anchor_count = sum(int(row["train_anchors"]) for row in train_task_rows)
    inner_fit_anchor_count = sum(int(row["fit_anchors"]) for row in train_task_rows)
    inner_tune_anchor_count = sum(int(row["tune_anchors"]) for row in train_task_rows)
    inner_fit_episode_count = sum(int(row["fit_episodes"]) for row in train_task_rows)
    inner_tune_episode_count = sum(int(row["tune_episodes"]) for row in train_task_rows)
    result_summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "PASS" if not missing_prediction_jobs and not seal.get("missing_prediction_jobs") else "PARTIAL",
        "protocol": "V2_existing_five_tasks_development",
        "fixed_task_ids": list(TASK_IDS),
        "train_anchors": train_anchor_count,
        "train_episodes": train_episode_count,
        "inner_split": {
            "fit_anchors": inner_fit_anchor_count, "tune_anchors": inner_tune_anchor_count,
            "fit_episodes": inner_fit_episode_count, "tune_episodes": inner_tune_episode_count,
        },
        "dev_support": dev_support,
        "screening_fits_accepted": screening_lock.get("accepted_fits"),
        "screening_fits_planned": 36,
        "formal_fits_accepted": json.loads(formal_status_path.read_text()).get("accepted_fits", 0) if formal_status_path.exists() else 0,
        "formal_fits_planned": 36,
        "waypoint_fits_accepted": json.loads(waypoint_status_path.read_text()).get("accepted_fits", 0) if waypoint_status_path.exists() else 0,
        "waypoint_fits_planned": 6,
        "fresh_process_prediction_jobs_sealed": seal.get("sealed_prediction_jobs"),
        "fresh_process_prediction_jobs_missing": missing_prediction_jobs,
        "selected_lr_epoch": screening_lock.get("logical_model_choices"),
        "metric_scales_source_sha256": metric_lock.get("source_sha256"),
        "conditional_bootstrap_replicates": bootstrap_replicates,
        "conditional_bootstrap_seed": bootstrap_seed,
        "collection_independence": "unverified_episode_source_only",
        "final_heldout_payload_access": 0,
        "new_final_aggregate_counts_read_or_printed": False,
        "parent_aggregate_metadata_exposure_preserved": True,
        "independent_confirmation": False,
        "human_review_status": "pending",
        "artifacts": {
            "absolute_losses": str(ROOT / "results/ABSOLUTE_LOSSES.csv"),
            "d_v_i": str(ROOT / "results/ALL_SEED_D_V_I.csv"),
            "loo": str(ROOT / "results/ALL_SEED_LOO.csv"),
            "conditional_ci": str(ROOT / "results/CONDITIONAL_CI.csv"),
            "waypoint": str(ROOT / "results/WAYPOINT.csv"),
            "waypoint_endpoint": str(ROOT / "results/WAYPOINT_ENDPOINT.csv"),
            "persistence": str(ROOT / "results/PERSISTENCE.csv"),
            "training_resource_ledger": str(ROOT / "results/TRAINING_RESOURCE_LEDGER.csv"),
            "figure_data": str(ROOT / "figure_data/DEV_ANCHOR_METRICS.csv"),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(summary_path, result_summary)
    exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    prior_exposure_backup = exposure_path.with_name(
        f"OUTCOME_EXPOSURE.json.pre-corrected-score-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
    )
    if not prior_exposure_backup.exists():
        import shutil
        shutil.copy2(exposure_path, prior_exposure_backup)
    exposure.update({
        "dev_predictions_or_errors_read": True,
        "current_split_dev_metrics_written": True,
        "current_split_prediction_seal_sha256": sha_file(seal_path),
        "current_split_result_summary_sha256": sha_file(summary_path),
        "current_split_lr_epoch_selection_sha256": sha_file(ROOT / "protocol/LR_EPOCH_SELECTION.json"),
        "prior_superseded_split_scores_used_for_current_selection": False,
        "final_heldout_payload_access": 0,
        "new_final_aggregate_counts_read_or_printed": False,
    })
    atomic_json(exposure_path, exposure)
    print(json.dumps({
        "status": result_summary["status"], "dev_targets_windows": len(targets),
        "sealed_prediction_jobs": seal.get("sealed_prediction_jobs"),
        "missing_prediction_jobs": len(missing_prediction_jobs),
        "metrics_rows": len(abs_rows), "contrast_rows": len(contrast_rows),
        "bootstrap_rows": len(ci_rows), "final_heldout_payload_access": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
