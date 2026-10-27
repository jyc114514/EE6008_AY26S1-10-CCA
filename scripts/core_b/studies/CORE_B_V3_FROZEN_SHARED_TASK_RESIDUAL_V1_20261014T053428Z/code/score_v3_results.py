#!/usr/bin/env python3
"""Post-seal CPU scoring for V3. Refuses empty, missing, or altered prediction batches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 202610134


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def load_prefixes() -> dict[str, np.ndarray]:
    paths = read_json(ROOT / "inventory/V3_PATHS.json")
    path = Path(paths["dev_prefix_npz"])
    with np.load(path, allow_pickle=False) as archive:
        required = ("anchor_id", "episode_id", "task_id", "source_cluster", "anchor_frame", "anchor_native")
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise RuntimeError(f"label-free dev prefix archive lacks fields: {missing}")
        return {key: archive[key].copy() for key in required}


def prefix_rows(prefix: dict[str, np.ndarray]) -> tuple[list[str], dict[str, int]]:
    ids = [as_text(x) for x in prefix["anchor_id"].tolist()]
    index = {anchor_id: i for i, anchor_id in enumerate(ids)}
    if len(index) != len(ids):
        raise RuntimeError("label-free dev prefix anchor IDs are not unique")
    if len(ids) != 185:
        raise RuntimeError(f"expected locked 185 dev anchors, observed {len(ids)}")
    return ids, index


def expected_ids_for_model(prefix: dict[str, np.ndarray], model_id: str) -> list[str]:
    mask = np.ones(len(prefix["anchor_id"]), dtype=bool)
    if model_id == "S-V-90":
        mask &= prefix["task_id"] == 90
    selected = np.flatnonzero(mask)
    selected = selected[np.argsort(np.asarray([as_text(x) for x in prefix["anchor_id"]])[selected], kind="stable")]
    return [as_text(prefix["anchor_id"][i]) for i in selected.tolist()]


def validate_rows(rows: list[dict[str, str]], prefix: dict[str, np.ndarray], model_id: str,
                  prediction: np.ndarray, expected_path: Path) -> dict[str, dict[str, Any]]:
    want = expected_ids_for_model(prefix, model_id)
    actual = [str(row["anchor_id"]) for row in rows]
    if actual != want:
        raise RuntimeError(f"prediction ID support/order mismatch for {expected_path}: expected {len(want)} IDs, got {len(actual)}")
    if prediction.shape != (len(rows), 30, 28):
        raise RuntimeError(f"prediction shape mismatch for {expected_path}: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"nonfinite prediction array: {expected_path}")
    pref_index = {as_text(x): i for i, x in enumerate(prefix["anchor_id"].tolist())}
    result: dict[str, dict[str, Any]] = {}
    for j, row in enumerate(rows):
        aid = str(row["anchor_id"])
        ix = pref_index[aid]
        for field in ("episode_id", "task_id"):
            if field in row and int(row[field]) != int(prefix[field][ix]):
                raise RuntimeError(f"prediction {field} differs from label-free prefix for anchor {aid}")
        if "source_cluster" in row and str(row["source_cluster"]) != as_text(prefix["source_cluster"][ix]):
            raise RuntimeError(f"prediction source cluster differs from prefix for anchor {aid}")
        result[aid] = {"prediction": prediction[j].astype(np.float64, copy=False), "prefix_index": ix,
                       "row": row}
    return result


def load_sealed_predictions(prefix: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    seal = read_json(seal_path)
    if seal.get("status") != "SEALED_BEFORE_TARGET_ACCESS" or seal.get("dev_target_reader_called") is not False or seal.get("dev_target_rows_read") != 0:
        raise RuntimeError("V3 scorer refuses to run without a full pre-target, zero-read seal")
    if seal.get("final_heldout_payload_access") != 0 or seal.get("future_labels_used_in_forward") is not False:
        raise RuntimeError("seal violates final-access or input-label separation lock")
    if len(seal.get("new_prediction_entries", [])) != 30 or seal.get("missing_new_prediction_jobs"):
        raise RuntimeError("V3 scorer refuses an incomplete or empty 30-job sealed prediction batch")
    if len(seal.get("reused_v2_prediction_references", [])) != 9:
        raise RuntimeError("expected the six main and three S-V-90 accepted V2 prediction references")

    ids, _ = prefix_rows(prefix)
    jobs: list[dict[str, Any]] = []
    new_seen: set[str] = set()
    main_count = probe_count = 0
    for entry in seal["new_prediction_entries"]:
        job_id = str(entry["job_id"])
        if job_id in new_seen:
            raise RuntimeError(f"duplicate job ID in seal: {job_id}")
        new_seen.add(job_id)
        cfg = Path(entry["job_config_path"])
        if not cfg.is_file() or sha(cfg) != entry["job_config_sha256"]:
            raise RuntimeError(f"sealed job config changed or missing: {job_id}")
        out = Path(entry["output_root"])
        pred_path, rows_path, complete_path = out / "PREDICTIONS.safetensors", out / "PREDICTION_ROWS.csv", out / "PREDICTION_COMPLETE.json"
        if not all(p.is_file() for p in (pred_path, rows_path, complete_path)):
            raise RuntimeError(f"sealed prediction artifact disappeared: {job_id}")
        if sha(pred_path) != entry["prediction_sha256"] or sha(rows_path) != entry["prediction_rows_sha256"]:
            raise RuntimeError(f"sealed prediction hash changed: {job_id}")
        completion = read_json(complete_path)
        if completion.get("status") != "PASS" or sha(complete_path) != entry["completion_sha256"]:
            raise RuntimeError(f"prediction completion marker changed or failed: {job_id}")
        tensor = load_file(str(pred_path), device="cpu")
        pred = tensor["prediction_native"].numpy()
        cfg_data = read_json(cfg)
        if cfg_data.get("job_id") != job_id or cfg_data.get("output_root") != str(out):
            raise RuntimeError(f"sealed job metadata mismatch: {job_id}")
        rows = read_csv(rows_path)
        mapped = validate_rows(rows, prefix, str(entry["model_id"]), pred, pred_path)
        kind = str(entry["method_kind"])
        if kind == "rgb_probe":
            probe_count += 1
            method = f"RGB_PROBE_{entry['model_id']}_{entry['condition']}"
        else:
            main_count += 1
            method = str(entry["method_id"])
        jobs.append({"job_id": job_id, "method": method, "model_id": str(entry["model_id"]),
                     "seed": int(entry["seed"]), "condition": str(entry.get("condition", "MATCHED")),
                     "kind": kind, "new": True, "path": str(pred_path), "rows_path": str(rows_path),
                     "output_root": str(out), "prediction_sha256": sha(pred_path), "mapped": mapped,
                     "row_count": len(rows), "config_path": str(cfg)})

    if main_count != 18 or probe_count != 12:
        raise RuntimeError(f"sealed manifest mix changed: main={main_count}, RGB={probe_count}")
    reused_seen: set[tuple[str, int]] = set()
    for entry in seal["reused_v2_prediction_references"]:
        model_id, seed = str(entry["model_id"]), int(entry["seed"])
        pair = (model_id, seed)
        if pair in reused_seen:
            raise RuntimeError(f"duplicate V2 reference {pair}")
        reused_seen.add(pair)
        pred_path, rows_path = Path(entry["prediction_path"]), Path(entry["prediction_rows_path"])
        if not pred_path.is_file() or not rows_path.is_file():
            raise RuntimeError(f"accepted V2 prediction reference missing: {pair}")
        if sha(pred_path) != entry["prediction_sha256"] or sha(rows_path) != entry["prediction_rows_sha256"]:
            raise RuntimeError(f"accepted V2 prediction reference changed: {pair}")
        pred = load_file(str(pred_path), device="cpu")["prediction_native"].numpy()
        rows = read_csv(rows_path)
        mapped = validate_rows(rows, prefix, model_id, pred, pred_path)
        method = {"G-S": "BASE_STATE", "G-V": "BASE_VIDEO", "S-V-90": "SV90_MATCHED"}[model_id]
        jobs.append({"job_id": f"V2REF_{model_id}_{seed}", "method": method, "model_id": model_id,
                     "seed": seed, "condition": "MATCHED", "kind": "v2_reference", "new": False,
                     "path": str(pred_path), "rows_path": str(rows_path), "output_root": str(pred_path.parent),
                     "prediction_sha256": sha(pred_path), "mapped": mapped, "row_count": len(rows),
                     "config_path": "accepted V2 seal reference"})
    if len(reused_seen) != 9:
        raise RuntimeError("V2 reference set must contain 3 G-S, 3 G-V, and 3 S-V-90 seeds")
    return jobs, seal


def metric_arrays(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray,
                  target_sigma: np.ndarray, fd_scale: np.ndarray) -> dict[str, np.ndarray]:
    pos = (np.abs(pred - target) / target_sigma[None, None, :]).mean(axis=(1, 2))
    fd = (np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) / fd_scale[None, None, :]).mean(axis=(1, 2))
    seam = (np.abs((pred[:, 0, :] - anchor) - (target[:, 0, :] - anchor)) / fd_scale[None, :]).mean(axis=1)
    return {"position_mae": pos, "safe_fd_l1": fd, "seam_safe_fd_l1": seam}


def load_targets_after_seal(prefix: dict[str, np.ndarray], seal: dict[str, Any]) -> tuple[np.ndarray, Path, int]:
    if seal.get("status") != "SEALED_BEFORE_TARGET_ACCESS" or not seal.get("new_prediction_entries"):
        raise RuntimeError("target reader refuses absent/empty pre-target seal")
    paths = read_json(ROOT / "inventory/V3_PATHS.json")
    target_path = Path(paths["scorer_only_dev_target_path"])
    if not target_path.is_file():
        raise FileNotFoundError(f"accepted scorer-only current dev target artifact missing: {target_path}")
    with np.load(target_path, allow_pickle=False) as archive:
        required = ("anchor_id", "episode_id", "task_id", "source_cluster", "target_native_h30")
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise RuntimeError(f"scorer-only dev target artifact lacks required arrays: {missing}")
        target_ids = [as_text(x) for x in archive["anchor_id"].tolist()]
        target = archive["target_native_h30"].astype(np.float64, copy=True)
        for field in ("episode_id", "task_id", "source_cluster"):
            a = [as_text(x) for x in archive[field].tolist()]
            b = [as_text(x) for x in prefix[field].tolist()]
            if a != b:
                raise RuntimeError(f"scorer-only target {field} mapping differs from locked prefix metadata")
    prefix_ids, _ = prefix_rows(prefix)
    if target_ids != prefix_ids or target.shape != (len(prefix_ids), 30, 28) or not np.isfinite(target).all():
        raise RuntimeError("scorer-only target IDs, shape, or finite-value check failed")
    # The artifact contains all 30 future action rows for each of 185 dev windows.
    return target, target_path, int(target.shape[0] * target.shape[1])


def build_contrast_definitions() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for modality in ("STATE", "VIDEO"):
        out.extend([
            {"contrast": f"G_R_shared_{modality}", "family": "G_R_shared", "lhs": f"BASE_{modality}", "rhs": f"SHARED_RES_{modality}"},
            {"contrast": f"G_R_task_{modality}", "family": "G_R_task", "lhs": f"BASE_{modality}", "rhs": f"TASK_RES_{modality}"},
            {"contrast": f"T_R_{modality}", "family": "T_R", "lhs": f"SHARED_RES_{modality}", "rhs": f"TASK_RES_{modality}"},
            {"contrast": f"G_BIAS_{modality}", "family": "G_BIAS", "lhs": f"BASE_{modality}", "rhs": f"BIAS_{modality}"},
            {"contrast": f"R_vs_BIAS_shared_{modality}", "family": "R_vs_BIAS", "lhs": f"BIAS_{modality}", "rhs": f"SHARED_RES_{modality}"},
            {"contrast": f"R_vs_BIAS_task_{modality}", "family": "R_vs_BIAS", "lhs": f"BIAS_{modality}", "rhs": f"TASK_RES_{modality}"},
        ])
    for program, state, video in [
        ("BASE", "BASE_STATE", "BASE_VIDEO"), ("BIAS", "BIAS_STATE", "BIAS_VIDEO"),
        ("SHARED_RES", "SHARED_RES_STATE", "SHARED_RES_VIDEO"), ("TASK_RES", "TASK_RES_STATE", "TASK_RES_VIDEO"),
    ]:
        out.append({"contrast": f"V_program_{program}", "family": "V_program", "lhs": state, "rhs": video})
    return out


def update_exposure(row_reads: int, target_path: Path, target_sha: str, seal_path: Path,
                    seal_sha: str, attempt: str) -> Path:
    path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"OUTCOME_EXPOSURE.json.pre-v3-score-{stamp}.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    exposure = read_json(path)
    attempt_reads: dict[str, int] = {}
    ledger = ROOT / "qa/FAILED_ATTEMPTS.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("phase") == "post_seal_scoring" and item.get("dev_target_reader_called"):
                attempt_reads[str(item.get("attempt"))] = int(item.get("dev_target_rows_read", 0))
    attempt_reads[attempt] = int(row_reads)
    cumulative_v3 = sum(attempt_reads.values())
    exposure.update({
        "v3_target_reader_called": True,
        "v3_current_dev_target_rows_read": int(cumulative_v3),
        "v3_successful_scoring_attempt_rows_read": int(row_reads),
        "v3_target_rows_read_by_attempt": attempt_reads,
        "v3_current_dev_target_windows_read": 185,
        "v3_target_read_attempt": attempt,
        "v3_target_read_after_prediction_seal": True,
        "v3_target_artifact": str(target_path),
        "v3_target_artifact_sha256": target_sha,
        "v3_prediction_seal_sha256": seal_sha,
        "parent_v2_cumulative_dev_target_row_reads": 16650,
        "cumulative_parent_plus_v3_dev_target_row_reads": int(16650 + cumulative_v3),
        "v3_final_heldout_payload_access": 0,
        "v3_future_labels_used_in_forward": False,
    })
    atomic_json(path, exposure)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="validate seal/predictions without opening dev target labels")
    parser.add_argument("--attempt", default="attempt_01")
    args = parser.parse_args()

    prefix = load_prefixes()
    jobs, seal = load_sealed_predictions(prefix)
    summary = {"status": "PASS", "preflight_only": True, "new_jobs": sum(j["new"] for j in jobs),
               "reused_v2_jobs": sum(not j["new"] for j in jobs), "job_count": len(jobs),
               "new_main_jobs": sum(j["new"] and j["kind"] != "rgb_probe" for j in jobs),
               "new_rgb_probe_jobs": sum(j["new"] and j["kind"] == "rgb_probe" for j in jobs),
               "dev_target_reader_called": False, "dev_target_rows_read": 0,
               "final_heldout_payload_access": 0, "prediction_seal_sha256": sha(ROOT / "protocol/PREDICTION_BATCH_SEAL.json")}
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return

    out = ROOT / "results/scoring" / args.attempt
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"preserving existing scoring attempt; choose a new --attempt: {out}")
    out.mkdir(parents=True, exist_ok=True)
    target_rows_read = 0
    target_path: Path | None = None
    try:
        # All prediction files and the complete nonempty seal have been revalidated above.
        target, target_path, target_rows_read = load_targets_after_seal(prefix, seal)
        metric_lock_path = V2 / "protocol/METRIC_SCALE_LOCK.json"
        metric_lock = read_json(metric_lock_path)
        target_sigma = np.asarray(metric_lock["target_sigma"], dtype=np.float64)
        fd_scale = np.asarray(metric_lock["safe_fd_scale"], dtype=np.float64)
        if target_sigma.shape != (28,) or fd_scale.shape != (28,) or np.any(target_sigma <= 0) or np.any(fd_scale <= 0):
            raise RuntimeError("accepted V2 metric-scale lock is invalid")
        prefix_ids, prefix_index = prefix_rows(prefix)
        anchors = np.asarray(prefix["anchor_native"], dtype=np.float64)
        task_ids = np.asarray(prefix["task_id"], dtype=np.int64)
        episode_ids = np.asarray(prefix["episode_id"], dtype=np.int64)
        clusters = [as_text(x) for x in prefix["source_cluster"].tolist()]

        per_window: list[dict[str, Any]] = []
        episode_scores: dict[tuple[str, int, int, int, str], list[float]] = defaultdict(list)
        row_metric_lookup: dict[tuple[str, int, str], dict[str, float]] = {}
        rgb_job_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
        for job in jobs:
            method, seed = str(job["method"]), int(job["seed"])
            ids_in_job = list(job["mapped"])
            pred = np.stack([job["mapped"][aid]["prediction"] for aid in ids_in_job]).astype(np.float64)
            indexes = np.asarray([job["mapped"][aid]["prefix_index"] for aid in ids_in_job], dtype=np.int64)
            target_job = target[indexes]
            anchor_job = anchors[indexes]
            metric = metric_arrays(pred, target_job, anchor_job, target_sigma, fd_scale)
            job_rows = []
            for local, aid in enumerate(ids_in_job):
                ix = int(indexes[local]); task = int(task_ids[ix]); ep = int(episode_ids[ix]); cluster = clusters[ix]
                record = {"job_id": job["job_id"], "method": method, "model_id": job["model_id"],
                          "seed": seed, "condition": job["condition"], "kind": job["kind"],
                          "is_new_prediction": job["new"], "anchor_id": aid, "task_id": task,
                          "episode_id": ep, "source_cluster": cluster}
                for m in METRICS:
                    val = float(metric[m][local]); record[m] = val
                    row_metric_lookup[(method, seed, aid)] = row_metric_lookup.get((method, seed, aid), {})
                    row_metric_lookup[(method, seed, aid)][m] = val
                    episode_scores[(method, seed, task, ep, m)].append(val)
                job_rows.append(record)
            per_window.extend(job_rows)
            if job["kind"] == "rgb_probe":
                rgb_job_by_key[(job["model_id"], seed, job["condition"])] = job

        # Accepted persistence is the deterministic anchor-hold baseline. Score it on the same sealed dev windows.
        persistence = np.repeat(anchors[:, None, :], 30, axis=1)
        pmetric = metric_arrays(persistence, target, anchors, target_sigma, fd_scale)
        persistence_window_values: dict[tuple[int, str, str], float] = {}
        for i, aid in enumerate(prefix_ids):
            t, ep, cluster = int(task_ids[i]), int(episode_ids[i]), clusters[i]
            for metric_name in METRICS:
                value = float(pmetric[metric_name][i])
                persistence_window_values[(t, ep, aid)] = value
                episode_scores[("PERSISTENCE", SEEDS[0], t, ep, metric_name)].append(value)
            for seed in SEEDS[1:]:
                for metric_name in METRICS:
                    episode_scores[("PERSISTENCE", seed, t, ep, metric_name)].append(float(pmetric[metric_name][i]))
            for seed in SEEDS:
                rec = {"job_id": "V2_ACCEPTED_PERSISTENCE", "method": "PERSISTENCE", "model_id": "PERSISTENCE",
                       "seed": seed, "condition": "ANCHOR_HOLD", "kind": "v2_accepted_baseline",
                       "is_new_prediction": False, "anchor_id": aid, "task_id": t, "episode_id": ep,
                       "source_cluster": cluster}
                for metric_name in METRICS: rec[metric_name] = float(pmetric[metric_name][i])
                per_window.append(rec)

        # Episode-equal task means are the analysis units; windows are retained only for paired scoring and figures.
        methods = ["PERSISTENCE", "BASE_STATE", "BASE_VIDEO", "BIAS_STATE", "BIAS_VIDEO",
                   "SHARED_RES_STATE", "TASK_RES_STATE", "SHARED_RES_VIDEO", "TASK_RES_VIDEO"]
        absolute: dict[tuple[str, int, int, str], float] = {}
        episode_value: dict[tuple[str, int, int, int, str], float] = {}
        episode_cluster: dict[tuple[int, int], str] = {}
        for i, aid in enumerate(prefix_ids):
            key = (int(task_ids[i]), int(episode_ids[i])); c = clusters[i]
            previous = episode_cluster.setdefault(key, c)
            if previous != c:
                raise RuntimeError(f"episode maps to multiple source clusters: {key}")
        for method in methods:
            for seed in SEEDS:
                for task in TASKS:
                    eps = sorted({ep for (m, s, t, ep, metric_name) in episode_scores if m == method and s == seed and t == task})
                    if not eps:
                        raise RuntimeError(f"missing scored support for {method}/{seed}/{task}")
                    for metric_name in METRICS:
                        vals = []
                        for ep in eps:
                            arr = episode_scores[(method, seed, task, ep, metric_name)]
                            ep_value = float(np.mean(arr))
                            episode_value[(method, seed, task, ep, metric_name)] = ep_value
                            vals.append(ep_value)
                        absolute[(method, seed, task, metric_name)] = float(np.mean(vals))

        contrast_defs = build_contrast_definitions()
        contrasts: dict[tuple[str, int, int, str], float] = {}
        for d in contrast_defs:
            for seed in SEEDS:
                for task in TASKS:
                    for metric_name in METRICS:
                        contrasts[(d["contrast"], seed, task, metric_name)] = (
                            absolute[(d["lhs"], seed, task, metric_name)] - absolute[(d["rhs"], seed, task, metric_name)])

        # Reconcile the recomputed accepted V2 base losses and persistence deltas with the saved V2 scorer tables.
        v2_absolute_path = V2 / "results/ABSOLUTE_LOSSES.csv"
        v2_persist_path = V2 / "results/PERSISTENCE.csv"
        v2_absolute_rows = read_csv(v2_absolute_path)
        v2_abs_map = {(r["model_id"], int(r["seed"]), int(r["task_id"])): r
                      for r in v2_absolute_rows
                      if r["status"] == "PASS" and r["model_id"] in {"G-S", "G-V"} and r["seed"] in {str(s) for s in SEEDS}}
        v2_persistence_abs = {int(r["task_id"]): r for r in v2_absolute_rows
                              if r["status"] == "PASS" and r["model_id"].lower() == "persistence" and r["seed"] == "shared"}
        v2_persist_rows = read_csv(v2_persist_path)
        max_v2_base_diff = 0.0
        for method, model_id in (("BASE_STATE", "G-S"), ("BASE_VIDEO", "G-V")):
            for seed in SEEDS:
                for task in TASKS:
                    src = v2_abs_map[(model_id, seed, task)]
                    for metric_name in METRICS:
                        delta = abs(absolute[(method, seed, task, metric_name)] - float(src[metric_name]))
                        max_v2_base_diff = max(max_v2_base_diff, delta)
        max_persistence_diff = 0.0
        for task, row in v2_persistence_abs.items():
            for metric_name in METRICS:
                delta = abs(absolute[("PERSISTENCE", SEEDS[0], task, metric_name)] - float(row[metric_name]))
                max_persistence_diff = max(max_persistence_diff, delta)
        for r in v2_persist_rows:
            if r["model_id"] not in {"G-S", "G-V"} or r["metric"] not in {"position_mae", "safe_fd_l1"}:
                continue
            method = "BASE_STATE" if r["model_id"] == "G-S" else "BASE_VIDEO"
            seed, task, metric_name = int(r["seed"]), int(r["task_id"]), r["metric"]
            if seed not in SEEDS or task not in TASKS:
                continue
            expected = absolute[(method, seed, task, metric_name)] + float(r["persistence_minus_model"])
            delta = abs(absolute[("PERSISTENCE", seed, task, metric_name)] - expected)
            max_persistence_diff = max(max_persistence_diff, delta)

        # Reusable paired source-group draws; exact same draws are used for all methods, seeds, and metrics.
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draw_indices: dict[int, np.ndarray] = {}
        group_ids: dict[int, list[str]] = {}
        for task in TASKS:
            groups = sorted({episode_cluster[(task, ep)] for (t, ep) in episode_cluster if t == task})
            if not groups:
                raise RuntimeError(f"no source groups for task {task}")
            group_ids[task] = groups
            draw_indices[task] = rng.integers(0, len(groups), size=(BOOTSTRAP_N, len(groups)), dtype=np.int16)
        draw_npz = ROOT / "protocol/SOURCE_GROUP_DRAWS.npz"
        if draw_npz.exists():
            raise FileExistsError(draw_npz)
        np.savez_compressed(draw_npz, **{f"task_{t}": draw_indices[t] for t in TASKS})
        draw_json = ROOT / "protocol/SOURCE_GROUP_DRAWS.json"
        if draw_json.exists():
            raise FileExistsError(draw_json)
        atomic_json(draw_json, {"schema_version": 1, "status": "LOCKED_AFTER_SEAL_BEFORE_ANALYSIS_OUTPUT",
                                "method": "paired source-group bootstrap; sample clusters with replacement within task; retain all episodes in each sampled cluster",
                                "generator": "NumPy PCG64", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_N,
                                "draws_shared_across": "all main methods, formal seeds, metrics, and contrasts",
                                "task_order": list(TASKS), "cluster_ids_by_task": {str(t): group_ids[t] for t in TASKS},
                                "draw_array_shapes": {str(t): list(draw_indices[t].shape) for t in TASKS},
                                "npz_path": str(draw_npz), "npz_sha256": sha(draw_npz),
                                "conditional_on_trained_seeds_and_tuning": True,
                                "session_trial_independence": "unverified"})

        method_draws: dict[tuple[str, int, int, str], np.ndarray] = {}
        for task in TASKS:
            groups = group_ids[task]
            group_eps = [[ep for (t, ep), c in episode_cluster.items() if t == task and c == gid] for gid in groups]
            draws = draw_indices[task]
            for method in methods:
                for seed in SEEDS:
                    for metric_name in METRICS:
                        sums = np.asarray([sum(episode_value[(method, seed, task, ep, metric_name)] for ep in eps) for eps in group_eps], dtype=np.float64)
                        counts = np.asarray([len(eps) for eps in group_eps], dtype=np.float64)
                        numerator = sums[draws].sum(axis=1)
                        denominator = counts[draws].sum(axis=1)
                        method_draws[(method, seed, task, metric_name)] = numerator / denominator
        contrast_draws: dict[tuple[str, int, int, str], np.ndarray] = {}
        for d in contrast_defs:
            name = d["contrast"]
            for seed in SEEDS:
                for task in TASKS:
                    for metric_name in METRICS:
                        contrast_draws[(name, seed, task, metric_name)] = (
                            method_draws[(d["lhs"], seed, task, metric_name)] - method_draws[(d["rhs"], seed, task, metric_name)])

        # Primary result tables.
        metric_rows: list[dict[str, Any]] = []
        for method in methods:
            for seed in SEEDS:
                for task in TASKS:
                    base = {"method": method, "seed": seed, "task_id": task,
                            "dev_episodes": len({ep for (t, ep) in episode_cluster if t == task}),
                            "dev_source_groups": len(group_ids[task])}
                    for metric_name in METRICS:
                        metric_rows.append({**base, "metric": metric_name, "value": absolute[(method, seed, task, metric_name)]})
        write_csv(out / "METHOD_METRICS_BY_TASK_SEED.csv", metric_rows)

        contrast_rows: list[dict[str, Any]] = []
        for d in contrast_defs:
            for seed in SEEDS:
                for task in TASKS:
                    for metric_name in METRICS:
                        contrast_rows.append({"contrast": d["contrast"], "family": d["family"], "lhs": d["lhs"],
                                              "rhs": d["rhs"], "seed": seed, "task_id": task,
                                              "metric": metric_name, "value": contrasts[(d["contrast"], seed, task, metric_name)],
                                              "positive_means_rhs_error_lower": True})
        write_csv(out / "CONTRASTS_BY_TASK_SEED.csv", contrast_rows)

        summary_rows: list[dict[str, Any]] = []
        outcomes: list[tuple[str, str, str]] = [("method", m, m) for m in methods]
        outcomes += [("contrast", d["contrast"], d["family"]) for d in contrast_defs]
        for kind, name, family in outcomes:
            for metric_name in METRICS:
                for task in TASKS:
                    vals = [absolute[(name, s, task, metric_name)] for s in SEEDS] if kind == "method" else [contrasts[(name, s, task, metric_name)] for s in SEEDS]
                    summary_rows.append({"scope": "task", "task_id": task, "result_type": kind, "name": name,
                                         "family": family, "metric": metric_name, "mean_across_seeds": float(np.mean(vals)),
                                         "sample_sd_across_seeds": float(np.std(vals, ddof=1)), "median_across_seeds": float(np.median(vals)),
                                         "n_seeds": len(vals), "session_trial_independence_verified": False})
                vals = []
                for seed in SEEDS:
                    by_task = [absolute[(name, seed, t, metric_name)] for t in TASKS] if kind == "method" else [contrasts[(name, seed, t, metric_name)] for t in TASKS]
                    vals.append(float(np.mean(by_task)))
                summary_rows.append({"scope": "five_task_macro", "task_id": "ALL5", "result_type": kind, "name": name,
                                     "family": family, "metric": metric_name, "mean_across_seeds": float(np.mean(vals)),
                                     "sample_sd_across_seeds": float(np.std(vals, ddof=1)), "median_across_seeds": float(np.median(vals)),
                                     "n_seeds": len(vals), "session_trial_independence_verified": False})
        write_csv(out / "SUMMARY_STATISTICS.csv", summary_rows)

        # True leave-one-seed-out: retain two seeds and all five tasks.
        seed_loo_rows: list[dict[str, Any]] = []
        for kind, name, family in outcomes:
            for metric_name in METRICS:
                for omitted in SEEDS:
                    kept = [s for s in SEEDS if s != omitted]
                    for task in TASKS:
                        vals = [absolute[(name, s, task, metric_name)] for s in kept] if kind == "method" else [contrasts[(name, s, task, metric_name)] for s in kept]
                        seed_loo_rows.append({"scope": "task", "result_type": kind, "name": name, "family": family,
                                              "metric": metric_name, "omitted_seed": omitted, "included_seeds": ";".join(map(str, kept)),
                                              "task_id": task, "included_tasks": ";".join(map(str, TASKS)), "n_seeds_retained": 2,
                                              "loo_mean": float(np.mean(vals))})
                    vals = []
                    for s in kept:
                        taskvals = [absolute[(name, s, t, metric_name)] for t in TASKS] if kind == "method" else [contrasts[(name, s, t, metric_name)] for t in TASKS]
                        vals.append(float(np.mean(taskvals)))
                    seed_loo_rows.append({"scope": "five_task_macro", "result_type": kind, "name": name, "family": family,
                                          "metric": metric_name, "omitted_seed": omitted, "included_seeds": ";".join(map(str, kept)),
                                          "task_id": "ALL5", "included_tasks": ";".join(map(str, TASKS)), "n_seeds_retained": 2,
                                          "loo_mean": float(np.mean(vals))})
        write_csv(out / "LEAVE_ONE_SEED_OUT.csv", seed_loo_rows)

        # True leave-one-task-out: retain four tasks separately within each seed.
        task_loo_rows: list[dict[str, Any]] = []
        for kind, name, family in outcomes:
            for metric_name in METRICS:
                for omitted_task in TASKS:
                    kept_tasks = [t for t in TASKS if t != omitted_task]
                    per_seed_vals = []
                    for seed in SEEDS:
                        vals = [absolute[(name, seed, t, metric_name)] for t in kept_tasks] if kind == "method" else [contrasts[(name, seed, t, metric_name)] for t in kept_tasks]
                        loo = float(np.mean(vals)); per_seed_vals.append(loo)
                        task_loo_rows.append({"scope": "per_seed", "result_type": kind, "name": name, "family": family,
                                              "metric": metric_name, "omitted_task_id": omitted_task, "seed": seed,
                                              "included_tasks": ";".join(map(str, kept_tasks)), "n_tasks_retained": 4, "loo_mean": loo})
                    task_loo_rows.append({"scope": "all_seeds_mean", "result_type": kind, "name": name, "family": family,
                                          "metric": metric_name, "omitted_task_id": omitted_task, "seed": "ALL3",
                                          "included_tasks": ";".join(map(str, kept_tasks)), "n_tasks_retained": 4,
                                          "loo_mean": float(np.mean(per_seed_vals))})
        write_csv(out / "LEAVE_ONE_TASK_OUT.csv", task_loo_rows)

        # Paired conditional source-group percentile intervals, conditional on trained seeds/tuning.
        ci_rows: list[dict[str, Any]] = []
        for d in contrast_defs:
            name = d["contrast"]
            for metric_name in METRICS:
                for task in TASKS:
                    ng = len(group_ids[task])
                    for seed in SEEDS:
                        vals = contrast_draws[(name, seed, task, metric_name)]
                        lo, hi = (np.quantile(vals, [0.025, 0.975]).tolist() if ng > 1 else (None, None))
                        ci_rows.append({"scope": "task_per_seed", "task_id": task, "seed": seed, "contrast": name,
                                        "metric": metric_name, "ci_low_2_5": lo, "ci_high_97_5": hi,
                                        "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
                                        "source_groups": ng, "source_episodes": len({ep for (t, ep) in episode_cluster if t == task}),
                                        "status": "CONDITIONAL_DIAGNOSTIC" if ng > 1 else "NA_SINGLE_GROUP",
                                        "dependency_note": "paired source-group resampling; conditional on trained seeds/tuning; session/trial independence unverified"})
                    vals_allseed = np.mean(np.stack([contrast_draws[(name, s, task, metric_name)] for s in SEEDS]), axis=0)
                    lo, hi = (np.quantile(vals_allseed, [0.025, 0.975]).tolist() if ng > 1 else (None, None))
                    ci_rows.append({"scope": "task_all_seeds", "task_id": task, "seed": "ALL3", "contrast": name,
                                    "metric": metric_name, "ci_low_2_5": lo, "ci_high_97_5": hi,
                                    "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
                                    "source_groups": ng, "source_episodes": len({ep for (t, ep) in episode_cluster if t == task}),
                                    "status": "CONDITIONAL_DIAGNOSTIC" if ng > 1 else "NA_SINGLE_GROUP",
                                    "dependency_note": "paired source-group resampling; averaged over the three fixed seeds"})
                for seed in SEEDS:
                    vals = np.mean(np.stack([contrast_draws[(name, seed, t, metric_name)] for t in TASKS]), axis=0)
                    ng_min = min(len(group_ids[t]) for t in TASKS)
                    lo, hi = (np.quantile(vals, [0.025, 0.975]).tolist() if ng_min > 1 else (None, None))
                    ci_rows.append({"scope": "five_task_macro_per_seed", "task_id": "ALL5", "seed": seed, "contrast": name,
                                    "metric": metric_name, "ci_low_2_5": lo, "ci_high_97_5": hi,
                                    "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
                                    "source_groups": ";".join(str(len(group_ids[t])) for t in TASKS),
                                    "source_episodes": ";".join(str(len({ep for (tt, ep) in episode_cluster if tt == t})) for t in TASKS),
                                    "status": "CONDITIONAL_DIAGNOSTIC" if ng_min > 1 else "NA_SINGLE_GROUP",
                                    "dependency_note": "equal five-task macro inside each paired draw; conditional on trained seed"})
                vals_all = np.mean(np.stack([contrast_draws[(name, s, t, metric_name)] for s in SEEDS for t in TASKS]), axis=0)
                ng_min = min(len(group_ids[t]) for t in TASKS)
                lo, hi = (np.quantile(vals_all, [0.025, 0.975]).tolist() if ng_min > 1 else (None, None))
                ci_rows.append({"scope": "five_task_macro_all_seeds", "task_id": "ALL5", "seed": "ALL3", "contrast": name,
                                "metric": metric_name, "ci_low_2_5": lo, "ci_high_97_5": hi,
                                "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
                                "source_groups": ";".join(str(len(group_ids[t])) for t in TASKS),
                                "source_episodes": ";".join(str(len({ep for (tt, ep) in episode_cluster if tt == t})) for t in TASKS),
                                "status": "CONDITIONAL_DIAGNOSTIC" if ng_min > 1 else "NA_SINGLE_GROUP",
                                "dependency_note": "paired source-group draws shared across all tasks, methods, seeds, and metrics; conditional on three trained seeds/tuning"})
        write_csv(out / "PAIRED_SOURCE_GROUP_CI.csv", ci_rows)

        # RGB perturbation effects are paired to the same checkpoint's accepted matched prediction.
        rgb_rows: list[dict[str, Any]] = []
        rgb_diag_rows: list[dict[str, Any]] = []
        for job in jobs:
            if job["kind"] != "rgb_probe":
                continue
            model, seed, cond = job["model_id"], job["seed"], job["condition"]
            matched_method = "BASE_VIDEO" if model == "G-V" else "SV90_MATCHED"
            probe_key = job["method"]
            per_task_metrics: dict[tuple[int, str], list[float]] = defaultdict(list)
            by_ep_diag: dict[tuple[int, int, str], list[float]] = defaultdict(list)
            diag_path = Path(job["output_root"]) / "PROBE_DIAGNOSTICS.csv"
            diag = read_csv(diag_path)
            if len(diag) != job["row_count"] or {r["anchor_id"] for r in diag} != set(job["mapped"]):
                raise RuntimeError(f"RGB diagnostics ID/support mismatch: {job['job_id']}")
            ix_map = {aid: item["prefix_index"] for aid, item in job["mapped"].items()}
            for r in diag:
                ix = int(ix_map[r["anchor_id"]]); t = int(task_ids[ix]); ep = int(episode_ids[ix])
                for field in ("h_mean_abs_change", "h_relative_l2_change", "base_residual_mean_abs_change", "base_residual_max_abs_change"):
                    by_ep_diag[(t, ep, field)].append(float(r[field]))
            tasks_here = sorted({int(task_ids[item["prefix_index"]]) for item in job["mapped"].values()})
            for task in tasks_here:
                task_eps = sorted({int(episode_ids[item["prefix_index"]]) for item in job["mapped"].values()
                                   if int(task_ids[item["prefix_index"]]) == task})
                rec = {"model_id": model, "seed": seed, "condition": cond, "task_id": task,
                       "matched_reference_method": matched_method, "task_episodes": len(task_eps),
                       "condition": cond, "rgb_intervention_is_diagnostic_only": True}
                for metric_name in METRICS:
                    vals = []
                    for aid, item in job["mapped"].items():
                        ix = int(item["prefix_index"])
                        if int(task_ids[ix]) != task:
                            continue
                        new_val = row_metric_lookup[(probe_key, seed, aid)][metric_name]
                        match_val = row_metric_lookup[(matched_method, seed, aid)][metric_name]
                        vals.append((int(episode_ids[ix]), new_val, match_val))
                    ep_new: dict[int, list[float]] = defaultdict(list); ep_match: dict[int, list[float]] = defaultdict(list)
                    for ep, nval, mval in vals:
                        ep_new[ep].append(nval); ep_match[ep].append(mval)
                    new_mean = float(np.mean([np.mean(ep_new[e]) for e in task_eps]))
                    match_mean = float(np.mean([np.mean(ep_match[e]) for e in task_eps]))
                    rec[f"{metric_name}_matched"] = match_mean
                    rec[f"{metric_name}_condition"] = new_mean
                    rec[f"{metric_name}_condition_minus_matched"] = new_mean - match_mean
                for field in ("h_mean_abs_change", "h_relative_l2_change", "base_residual_mean_abs_change", "base_residual_max_abs_change"):
                    epvals = [float(np.mean(by_ep_diag[(task, ep, field)])) for ep in task_eps]
                    rec[field] = float(np.mean(epvals))
                rgb_rows.append(rec)
                rgb_diag_rows.append(rec.copy())
        if len(rgb_rows) != 36:
            raise RuntimeError(f"RGB probe score table must contain 36 task-condition-seed views, got {len(rgb_rows)}")

        # Check the declared paired contrasts and metric identities; no hypothesis tests or p-values are produced.
        if set(methods) - {m for m, _, _, _ in absolute}:
            raise RuntimeError("one or more primary method scores are missing")
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "PER_WINDOW_LOSSES.csv", per_window)
        write_csv(out / "RGB_PROBE_RESULTS.csv", rgb_rows)

        # Metric discrepancy with accepted V2 tables is an audit value, not a selection gate.
        qa_status = "PASS" if max_v2_base_diff <= 1e-6 and max_persistence_diff <= 1e-6 else "PARTIAL_RECONCILIATION_DIFFERENCE"
        pred_seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
        scoring_qa = {"schema_version": 1, "status": qa_status, "created_utc": utc_now(),
                      "prediction_seal_path": str(pred_seal_path), "prediction_seal_sha256": sha(pred_seal_path),
                      "scorer_script_sha256": sha(Path(__file__)), "target_artifact": str(target_path),
                      "target_artifact_sha256": sha(target_path), "target_archive_sha256_read_after_seal": True,
                      "dev_target_rows_read": target_rows_read, "dev_target_windows_read": 185,
                      "dev_future_action_rows_read": target_rows_read,
                      "prior_failed_v3_target_rows_read": sum(int(json.loads(line).get("dev_target_rows_read", 0))
                          for line in (ROOT / "qa/FAILED_ATTEMPTS.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip() and json.loads(line).get("phase") == "post_seal_scoring"
                          and json.loads(line).get("dev_target_reader_called")),
                      "scored_prediction_jobs_new": 30,
                      "main_new_prediction_jobs": 18, "main_new_task_method_seed_views": 90,
                      "reused_v2_main_base_jobs": 6, "reused_v2_main_task_views": 30,
                      "rgb_probe_new_jobs": 12, "rgb_probe_new_task_condition_views": 36,
                      "rgb_matched_v2_reference_jobs": 6, "rgb_matched_v2_task_views": 18,
                      "all_sealed_new_predictions_revalidated_before_target_read": True,
                      "empty_or_incomplete_prediction_batch_can_reach_target_reader": False,
                      "future_labels_used_in_model_forward": False, "final_heldout_payload_access": 0,
                      "metric_lock_path": str(metric_lock_path), "metric_lock_sha256": sha(metric_lock_path),
                      "max_abs_v2_base_metric_reconciliation_difference": max_v2_base_diff,
                      "max_abs_v2_persistence_metric_reconciliation_difference": max_persistence_diff,
                      "reconciliation_tolerance": 1e-6,
                      "aggregation": "windows mean within episode; episode-equal task mean; equal five-task macro",
                      "bootstrap": {"method": "paired source-group resampling within task; all episodes in each sampled cluster retained",
                                    "replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                                    "conditional_on_fixed_trained_seeds_and_tuning": True,
                                    "session_trial_independence": "unverified", "p_values": "not computed by protocol"},
                      "outputs": sorted(p.name for p in out.iterdir() if p.is_file())}
        atomic_json(out / "DEV_TARGET_SCORING_QA.json", scoring_qa)
        exposure_backup = update_exposure(target_rows_read, target_path, sha(target_path), pred_seal_path,
                                          sha(pred_seal_path), args.attempt)
        scoring_qa["outcome_exposure_backup"] = str(exposure_backup)
        atomic_json(out / "DEV_TARGET_SCORING_QA.json", scoring_qa)
        print(json.dumps({"status": qa_status, "attempt": args.attempt, "new_jobs": 30,
                          "reused_v2_jobs": 9, "new_dev_future_action_rows_read": target_rows_read,
                          "parent_v2_cumulative_rows_preserved": 16650,
                          "parent_plus_v3_rows": 16650 + target_rows_read,
                          "max_v2_base_metric_difference": max_v2_base_diff,
                          "max_persistence_reconciliation_difference": max_persistence_diff,
                          "output_dir": str(out), "final_heldout_payload_access": 0}, indent=2))
    except Exception as exc:
        failed = {"schema_version": 1, "status": "FAILED", "created_utc": utc_now(), "attempt": args.attempt,
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                  "dev_target_reader_called": target_path is not None,
                  "dev_target_rows_read": target_rows_read, "final_heldout_payload_access": 0}
        atomic_json(out / "SCORING_FAILURE.json", failed)
        ledger = ROOT / "qa/FAILED_ATTEMPTS.jsonl"
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": utc_now(), "phase": "post_seal_scoring", "attempt": args.attempt,
                                "status": "failed", "error": failed["error"],
                                "dev_target_reader_called": failed["dev_target_reader_called"],
                                "dev_target_rows_read": target_rows_read, "final_heldout_payload_access": 0,
                                "gpu_started": False}, ensure_ascii=False, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
