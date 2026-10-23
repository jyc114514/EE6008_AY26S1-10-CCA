#!/usr/bin/env python3
"""Score sealed Core B V2 dev/final predictions with the frozen V9 metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V9_DEV_TARGETS = PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz"
V9_DEV_TARGETS_SHA = "233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27"
V9_DEV_TARGETS_BYTES = 435083
METRIC_LOCK = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/protocol/METRIC_SCALE_LOCK.json"
METRIC_LOCK_SHA = "dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32"
NORMALIZER = PROJECT / "reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/normalizers/NORMALIZER_FORMAL.json"
NORMALIZER_SHA = "22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d"
FINAL_POP = V1_OUT / "protocol/CORE_B_FINAL_POPULATION_METADATA.csv"
FINAL_VIDEO_MAP = V1_OUT / "protocol/CORE_B_FINAL_VIDEO_MAP.csv"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
CURRENT_LOCK = OUT / "core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
FINAL_FREEZE = OUT / "core_b/CORE_B_V2_FINAL_FREEZE.json"
FINAL_ACCESS = OUT / "protocol/FINAL_ACCESS_OPENED.json"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("A0", "AD", "V0", "VD", "S_OPT", "MIX0", "MIXD", "PERSISTENCE")
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")
COMPARISONS = (("A0", "V0"), ("PERSISTENCE", "V0"), ("V0", "VD"),
               ("A0", "AD"), ("AD", "VD"), ("MIX0", "MIXD"), ("V0", "MIX0"))

sys.path.insert(0, str(V1_OUT / "code"))
from analysis_metrics import core_b_per_anchor_metrics, core_b_task_macro, sample_sd  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty score table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_prefix(split: str) -> dict[str, np.ndarray]:
    path = RUN / "replay/core_b" / ("DEV_PREFIXES_CURRENT_PROFILE_V2.npz" if split == "dev" else "FINAL_PREFIXES_CURRENT_PROFILE_V2.npz")
    with np.load(path, allow_pickle=False) as archive:
        forbidden = [key for key in archive.files if any(term in key.lower() for term in ("target", "targets", "future", "label", "ground_truth"))]
        if forbidden:
            raise ValueError(f"prediction prefix contains forbidden keys: {forbidden}")
        needed = {"canonical_id", "anchor_id", "episode_id", "task_id", "source_cluster", "anchor_native"}
        if split == "final_test":
            needed |= {"parquet_path", "anchor_frame", "source_key"}
        if not needed.issubset(archive.files):
            raise ValueError(f"prediction prefix missing fields: {sorted(needed-set(archive.files))}")
        return {k: archive[k].copy() for k in needed}


def verify_seal(split: str, lock_sha: str, freeze_sha: str | None, access_sha: str | None) -> tuple[dict, dict[tuple[str, str], dict[str, Any]], Path]:
    path = OUT / "core_b" / ("CURRENT_PROFILE_DEV_PREDICTION_SEAL.json" if split == "dev" else "PREDICTION_MANIFEST_AND_SEAL.json")
    seal = json.loads(path.read_text(encoding="utf-8"))
    expected_status = "SEALED_BEFORE_DEV_TARGET_SCORING" if split == "dev" else "SEALED_BEFORE_FINAL_TARGET_ACCESS"
    if seal.get("status") != expected_status or seal.get("split") != split:
        raise ValueError(f"{split} prediction seal status mismatch")
    if seal.get("candidate_profile_lock_sha256") != lock_sha:
        raise ValueError("prediction seal does not bind current profile lock")
    if split == "final_test" and (seal.get("final_freeze_sha256") != freeze_sha or seal.get("final_access_receipt_sha256") != access_sha):
        raise ValueError("final prediction seal does not bind final freeze/access")
    keys: dict[tuple[str, str], dict[str, Any]] = {}
    for row in seal.get("prediction_records", []):
        key = (str(row["program"]), "REFERENCE" if row.get("seed") is None else str(int(row["seed"])))
        if key in keys:
            raise ValueError(f"duplicate prediction seal key: {key}")
        if sha(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"prediction bundle hash mismatch: {key}")
        keys[key] = row
    expected = {(p, str(seed)) for p in PROGRAMS if p != "PERSISTENCE" for seed in SEEDS}
    expected.add(("PERSISTENCE", "REFERENCE"))
    if set(keys) != expected or len(keys) != 22:
        raise ValueError(f"prediction seal matrix mismatch: missing={sorted(expected-set(keys))}, extra={sorted(set(keys)-expected)}")
    return seal, keys, path


def make_exposure(path: Path, split: str, seal_path: Path, seal: dict, target_source: str) -> dict:
    latch = {
        "schema_version": 1, "status": "READ_STARTED", "split": split, "started_utc": utc(),
        "prediction_seal_path": str(seal_path), "prediction_seal_sha256": sha(seal_path),
        "candidate_profile_lock_sha256": seal["candidate_profile_lock_sha256"],
        "final_freeze_sha256": seal.get("final_freeze_sha256"),
        "final_access_receipt_sha256": seal.get("final_access_receipt_sha256"),
        "target_source": target_source, "target_values_materialized_before_seal": False,
        "target_rows_materialized_current": 0, "target_values_materialized_current": 0,
        "raw_parquet_action_rows_read_unique": 0, "final_payload_reads": 0,
    }
    if path.exists():
        raise FileExistsError(f"refusing to overwrite target exposure latch: {path}")
    atomic_json(path, latch)
    return latch


def read_dev_targets(prefix: dict[str, np.ndarray], latch: dict, exposure_path: Path) -> np.ndarray:
    if V9_DEV_TARGETS.stat().st_size != V9_DEV_TARGETS_BYTES:
        raise ValueError("accepted dev target artifact size mismatch")
    latch["target_source_sha256_expected"] = V9_DEV_TARGETS_SHA
    latch["target_source_size_bytes_expected"] = V9_DEV_TARGETS_BYTES
    latch["target_archive_read_started_utc"] = utc()
    atomic_json(exposure_path, latch)
    blob = V9_DEV_TARGETS.read_bytes()
    actual_sha = hashlib.sha256(blob).hexdigest()
    latch["target_archive_content_reads"] = 1
    latch["target_source_sha256_actual"] = actual_sha
    if len(blob) != V9_DEV_TARGETS_BYTES or actual_sha != V9_DEV_TARGETS_SHA:
        latch.update({"status": "FAILED", "failure_utc": utc(), "target_values_materialized_current": 0})
        atomic_json(exposure_path, latch)
        raise ValueError("accepted scorer-only dev target archive hash/size mismatch")
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        required = {"anchor_id", "episode_id", "task_id", "source_cluster", "target_native_h30"}
        if not required.issubset(archive.files):
            raise ValueError("dev target archive schema mismatch")
        target_ids = archive["anchor_id"].astype(str)
        target_episodes = archive["episode_id"].astype(np.int64)
        target_tasks = archive["task_id"].astype(np.int64)
        target_groups = archive["source_cluster"].astype(str)
        target_raw = np.asarray(archive["target_native_h30"], dtype=np.float64).copy()
    anchors = prefix["anchor_id"].astype(str)
    if len(target_ids) != 185 or set(target_ids.tolist()) != set(anchors.tolist()):
        raise ValueError("dev target population differs from sealed current-profile predictions")
    lookup = {anchor: i for i, anchor in enumerate(target_ids.tolist())}
    order = np.asarray([lookup[a] for a in anchors.tolist()], dtype=np.int64)
    target = target_raw[order]
    for i, j in enumerate(order.tolist()):
        if (int(target_episodes[j]), int(target_tasks[j]), str(target_groups[j])) != (
            int(prefix["episode_id"][i]), int(prefix["task_id"][i]), str(prefix["source_cluster"][i])
        ):
            raise ValueError(f"dev target identity mismatch at {anchors[i]}")
    if target.shape != (185, 30, 28) or not np.isfinite(target).all():
        raise ValueError("dev H30 target shape/finite check failed")
    latch.update({
        "status": "TARGETS_MATERIALIZED_AFTER_SEAL",
        "target_archive_content_reads": 1,
        "target_rows_materialized_current": 185,
        "target_values_materialized_current": int(target.size),
        "target_materialized_utc": utc(),
    })
    atomic_json(exposure_path, latch)
    return target


def read_final_targets(prefix: dict[str, np.ndarray], lock: dict, latch: dict, exposure_path: Path) -> np.ndarray:
    rows = len(prefix["anchor_id"])
    if rows != 162:
        raise ValueError("final Core B target reader expects 162 sealed anchors")
    with FINAL_POP.open(encoding="utf-8-sig", newline="") as stream:
        population = list(csv.DictReader(stream))
    if len(population) != 162 or sha(FINAL_POP) != lock["final_inputs"]["final_population_metadata_sha256"]:
        raise ValueError("final metadata differs from pre-access lock")
    if sha(FINAL_VIDEO_MAP) != lock["final_inputs"]["final_video_map_sha256"]:
        raise ValueError("final video map differs from pre-access lock")
    pop_by_id = {r["anchor_id"]: r for r in population}
    if len(pop_by_id) != rows or set(pop_by_id) != set(prefix["anchor_id"].astype(str).tolist()):
        raise ValueError("final population IDs differ from sealed predictions")
    for i, anchor in enumerate(prefix["anchor_id"].astype(str).tolist()):
        p = pop_by_id[anchor]
        if (int(p["episode_id"]), int(p["task_id"]), int(p["anchor_frame"]), str(p["literal_source_key"])) != (
            int(prefix["episode_id"][i]), int(prefix["task_id"][i]), int(prefix["anchor_frame"][i]), str(prefix["parquet_path"][i])
        ):
            raise ValueError(f"final prediction/prefix metadata mismatch: {anchor}")
        if int(p["target_h30_row_start_inclusive"]) != int(p["anchor_frame"]) + 1:
            raise ValueError(f"locked H30 start is not anchor+1: {anchor}")
        if int(p["target_h30_row_end_inclusive"]) != int(p["anchor_frame"]) + 30:
            raise ValueError(f"locked H30 end is not anchor+30: {anchor}")

    rows_by_parquet: dict[str, list[int]] = defaultdict(list)
    for i, rel in enumerate(prefix["parquet_path"].astype(str).tolist()):
        rows_by_parquet[rel].append(i)
    target = np.empty((rows, 30, 28), dtype=np.float32)
    unique_raw_rows = 0
    parquet_receipts = []
    for relative, indices in sorted(rows_by_parquet.items()):
        parquet = (RAW_ROOT / relative).resolve()
        if not parquet.is_relative_to(RAW_ROOT.resolve()) or not parquet.is_file():
            raise FileNotFoundError(f"final target parquet is absent or escapes the fixed raw root: {relative}")
        requested: set[int] = set()
        for ix in indices:
            anchor = int(prefix["anchor_frame"][ix])
            requested.update(range(anchor + 1, anchor + 31))
        frames_requested = sorted(requested)
        latch["current_parquet_path"] = relative
        latch["current_parquet_requested_unique_action_rows"] = len(frames_requested)
        latch["current_parquet_read_started_utc"] = utc()
        atomic_json(exposure_path, latch)
        source_sha = sha(parquet)
        table = ds.dataset(str(parquet), format="parquet").to_table(
            columns=["frame_index", "action"], filter=ds.field("frame_index").isin(frames_requested)
        )
        frame_index = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        unique_raw_rows += len(frame_index)
        latch["raw_parquet_action_rows_read_unique"] = unique_raw_rows
        latch["raw_parquet_files_read"] = len(parquet_receipts) + 1
        latch["current_raw_action_rows_materialized"] = len(frame_index)
        atomic_json(exposure_path, latch)
        if action.shape != (len(frame_index), 28) or not np.isfinite(action).all():
            raise ValueError(f"final target action schema/finite check failed: {relative}")
        if len(frame_index) != len(frames_requested) or set(frame_index.tolist()) != set(frames_requested):
            raise ValueError(f"final target row filter mismatch: {relative}")
        action_by_frame = {int(frame): action[i].astype(np.float32) for i, frame in enumerate(frame_index.tolist())}
        for ix in indices:
            anchor = int(prefix["anchor_frame"][ix])
            target[ix] = np.stack([action_by_frame[frame] for frame in range(anchor + 1, anchor + 31)])
            latch["target_rows_materialized_current"] += 30
            latch["target_values_materialized_current"] += 30 * 28
            atomic_json(exposure_path, latch)
        parquet_receipts.append({
            "parquet_path": relative, "parquet_sha256_after_seal": source_sha,
            "unique_action_rows_selected": len(frame_index), "target_anchor_rows_materialized": 30 * len(indices),
        })
    if target.shape != (162, 30, 28) or not np.isfinite(target).all():
        raise ValueError("final H30 targets failed shape/finite validation")

    target_path = RUN / "scores/core_b/FINAL_TARGETS_AFTER_PREDICTION_SEAL.npz"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        raise FileExistsError(f"refusing to overwrite final target artifact: {target_path}")
    temp = target_path.with_name(f".{target_path.name}.{os.getpid()}.part")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, anchor_id=prefix["anchor_id"], episode_id=prefix["episode_id"],
                            task_id=prefix["task_id"], source_cluster=prefix["source_cluster"],
                            target_native_h30=target)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target_path)
    latch.update({
        "status": "TARGETS_MATERIALIZED_AFTER_SEAL",
        "target_archive_path": str(target_path), "target_archive_sha256": sha(target_path),
        "target_archive_content_reads": 0,
        "target_rows_materialized_current": int(rows * 30),
        "target_values_materialized_current": int(target.size),
        "raw_parquet_action_rows_read_unique": unique_raw_rows,
        "raw_parquet_files_read": len(parquet_receipts),
        "parquet_source_receipts": parquet_receipts,
        "target_materialized_utc": utc(),
        "final_payload_reads": len(parquet_receipts),
    })
    atomic_json(exposure_path, latch)
    return target


def load_bundles(records: dict[tuple[str, str], dict[str, Any]], canonical: np.ndarray) -> dict[tuple[str, int | None], np.ndarray]:
    output = {}
    for (program, seed_text), row in records.items():
        with np.load(Path(row["path"]), allow_pickle=False) as archive:
            ids = archive["canonical_ids"].astype(str)
            pred = np.asarray(archive["prediction_native"], dtype=np.float64)
        if not np.array_equal(ids, canonical):
            raise ValueError(f"sealed prediction canonical order mismatch: {program}/{seed_text}")
        seed = None if seed_text == "REFERENCE" else int(seed_text)
        output[(program, seed)] = pred
    return output


def score_all(prefix: dict[str, np.ndarray], target: np.ndarray, bundles: dict[tuple[str, int | None], np.ndarray],
              metric_scale: dict, draw_npz: Path) -> dict[str, list[dict[str, Any]]]:
    canonical = prefix["canonical_id"].astype(str)
    anchors = prefix["anchor_id"].astype(str)
    episodes = prefix["episode_id"].astype(np.int64)
    tasks = prefix["task_id"].astype(np.int64)
    groups = prefix["source_cluster"].astype(str)
    anchor_native = np.asarray(prefix["anchor_native"], dtype=np.float64)
    sigma = np.asarray(metric_scale["target_sigma"], dtype=np.float64)
    fdscale = np.asarray(metric_scale["safe_fd_scale"], dtype=np.float64)
    with np.load(draw_npz, allow_pickle=False) as draw_file:
        draw_arrays = {int(key.removeprefix("task_")): draw_file[key].astype(np.int32) for key in draw_file.files}
    if set(draw_arrays) != set(TASKS) or any(arr.shape[0] != 5000 for arr in draw_arrays.values()):
        raise ValueError("locked Core B bootstrap draw arrays differ from protocol")

    per_anchor_rows, task_rows, episode_rows, seed_rows = [], [], [], []
    point = {}
    bootstrap = {}
    for program in PROGRAMS:
        seeds: tuple[int | None, ...] = (None,) if program == "PERSISTENCE" else SEEDS
        for seed in seeds:
            pred = bundles[(program, seed)]
            per = core_b_per_anchor_metrics(pred, target, anchor_native, sigma, fdscale)
            macro, task_summary, episode_summary = core_b_task_macro(per, episodes, tasks, fixed_tasks=TASKS)
            if macro["status"] != "estimable" or set(macro["observed_tasks"]) != set(TASKS):
                raise ValueError(f"fixed five-task macro is not estimable: {program}/{seed}")
            point[(program, seed)] = {m: float(macro[m]) for m in METRICS}
            for row in task_summary:
                task_rows.append({"program": program, "seed": "REFERENCE" if seed is None else seed, **row})
            for row in episode_summary:
                episode_rows.append({"program": program, "seed": "REFERENCE" if seed is None else seed, **row})
            for i in range(len(anchors)):
                per_anchor_rows.append({
                    "program": program, "seed": "REFERENCE" if seed is None else seed,
                    "canonical_id": canonical[i], "episode_id": int(episodes[i]), "task_id": int(tasks[i]),
                    "anchor_id": anchors[i], "source_cluster": groups[i],
                    **{m: float(per[m][i]) for m in METRICS},
                })
            seed_rows.append({
                "program": program, "seed": "REFERENCE" if seed is None else seed,
                "episode_support": int(len(set(episodes.tolist())),),
                "anchor_support": len(anchors), **point[(program, seed)],
            })
            # Episode-equal values within each task, resampled with the pre-registered draw indices.
            for metric in METRICS:
                task_boot = []
                for task in TASKS:
                    mask_task = tasks == task
                    clusters = sorted(set(groups[mask_task].tolist()))
                    if len(clusters) != draw_arrays[task].shape[1]:
                        raise ValueError(f"bootstrap cluster support mismatch for task {task}")
                    cluster_values = np.asarray([np.mean(per[metric][mask_task & (groups == c)]) for c in clusters], dtype=np.float64)
                    sampled = cluster_values[draw_arrays[task]]
                    task_boot.append(sampled.mean(axis=1))
                bootstrap[(program, seed, metric)] = np.mean(np.stack(task_boot, axis=1), axis=1)

    summary_rows = []
    for program in PROGRAMS:
        active_seeds: tuple[int | None, ...] = (None,) if program == "PERSISTENCE" else SEEDS
        row = {"program": program, "training_seed_count": 0 if program == "PERSISTENCE" else len(SEEDS),
               "episode_support": int(len(set(episodes.tolist()))), "anchor_support": len(anchors), "scope": "CURRENT_PROFILE_V2"}
        for metric in METRICS:
            values = [point[(program, seed)][metric] for seed in active_seeds]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_sd"] = sample_sd(values) if len(values) >= 2 else None
            boot_values = np.mean(np.stack([bootstrap[(program, seed, metric)] for seed in active_seeds], axis=1), axis=1)
            lo, hi = np.quantile(boot_values, [0.025, 0.975])
            row[f"{metric}_conditional_bootstrap_95_low"] = float(lo)
            row[f"{metric}_conditional_bootstrap_95_high"] = float(hi)
        summary_rows.append(row)

    contrast_rows, bootstrap_rows = [], []
    for reference, candidate in COMPARISONS:
        paired_seeds = SEEDS
        for metric in METRICS:
            ref_seed_values = [point[(reference, None if reference == "PERSISTENCE" else seed)][metric] for seed in paired_seeds]
            cand_seed_values = [point[(candidate, seed)][metric] for seed in paired_seeds]
            gains = np.asarray(ref_seed_values, dtype=np.float64) - np.asarray(cand_seed_values, dtype=np.float64)
            if reference == "PERSISTENCE":
                boot_gain = np.mean(np.stack([
                    bootstrap[(reference, None, metric)] - bootstrap[(candidate, seed, metric)]
                    for seed in paired_seeds
                ], axis=1), axis=1)
            else:
                boot_gain = np.mean(np.stack([
                    bootstrap[(reference, seed, metric)] - bootstrap[(candidate, seed, metric)]
                    for seed in paired_seeds
                ], axis=1), axis=1)
            lo, hi = np.quantile(boot_gain, [0.025, 0.975])
            contrast_rows.append({
                "contrast": f"{reference}_minus_{candidate}", "reference": reference, "candidate": candidate,
                "metric": metric, "grain": "three_seed_task_macro",
                "reference_minus_candidate_mean": float(gains.mean()),
                "sample_sd_gain_across_training_seeds": sample_sd(gains),
                "seed_count": 3, "episode_support": len(set(episodes.tolist())),
                "anchor_support": len(anchors), "task_support": len(TASKS),
                "direction": "positive favors candidate lower error",
            })
            bootstrap_rows.append({
                "contrast": f"{reference}_minus_{candidate}", "reference": reference, "candidate": candidate,
                "metric": metric, "draws": 5000, "seed": 2026101902,
                "interval_type": "paired_within_task_episode_cluster_percentile_95_conditional_on_fixed_models",
                "gain_mean": float(boot_gain.mean()), "ci_low": float(lo), "ci_high": float(hi),
                "ci_crosses_zero": bool(lo <= 0 <= hi),
            })
            for held_out in SEEDS:
                keep = [seed for seed in SEEDS if seed != held_out]
                loo_gains = [
                    point[(reference, None if reference == "PERSISTENCE" else seed)][metric]
                    - point[(candidate, seed)][metric]
                    for seed in keep
                ]
                contrast_rows.append({
                    "contrast": f"{reference}_minus_{candidate}", "reference": reference, "candidate": candidate,
                    "metric": metric, "grain": "seed_leave_one_out",
                    "held_out_seed": held_out, "remaining_seed_count": 2,
                    "reference_minus_candidate_mean": float(np.mean(loo_gains)),
                    "sample_sd_gain_across_training_seeds": sample_sd(loo_gains),
                    "episode_support": len(set(episodes.tolist())), "anchor_support": len(anchors),
                    "task_support": len(TASKS), "direction": "positive favors candidate lower error",
                })
    results = {
        "per_anchor": per_anchor_rows, "task": task_rows, "episode": episode_rows,
        "seed": seed_rows, "summary": summary_rows, "contrast": contrast_rows,
        "bootstrap": bootstrap_rows,
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "final_test"), required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-final-freeze-sha256")
    parser.add_argument("--expected-final-access-sha256")
    args = parser.parse_args()
    if sha(CURRENT_LOCK) != args.expected_lock_sha256:
        raise ValueError("CURRENT_PROFILE_V2 lock hash mismatch")
    lock = json.loads(CURRENT_LOCK.read_text(encoding="utf-8"))
    if sha(METRIC_LOCK) != METRIC_LOCK_SHA or sha(NORMALIZER) != NORMALIZER_SHA:
        raise ValueError("accepted metric scale or residual normalizer hash changed")
    if args.split == "final_test":
        if not args.expected_final_freeze_sha256 or not args.expected_final_access_sha256:
            raise ValueError("final scoring requires final freeze/access receipt")
        if sha(FINAL_FREEZE) != args.expected_final_freeze_sha256:
            raise ValueError("V2 final freeze hash mismatch")
        from core_b_predict_v2 import verify_final_access_receipt
        verify_final_access_receipt(FINAL_ACCESS, args.expected_final_access_sha256)

    split = args.split
    score_dir = OUT / "core_b" / ("current_profile_dev_scores" if split == "dev" else "final_scores")
    exposure_path = OUT / "core_b" / ("CURRENT_PROFILE_DEV_TARGET_EXPOSURE.json" if split == "dev" else "FINAL_TARGET_EXPOSURE.json")
    draw_npz = OUT / "core_b" / f"CORE_B_{'DEV' if split == 'dev' else 'FINAL'}_BOOTSTRAP_DRAWS_V2.npz"
    draw_meta = OUT / "core_b" / f"CORE_B_{'DEV' if split == 'dev' else 'FINAL'}_BOOTSTRAP_DRAWS_V2.json"
    draw_receipt = OUT / "core_b" / f"CORE_B_{'DEV' if split == 'dev' else 'FINAL'}_BOOTSTRAP_DRAWS_V2_RECEIPT.json"
    if score_dir.exists() or exposure_path.exists():
        raise FileExistsError(f"refusing to overwrite {split} score/exposure outputs")
    if sha(draw_npz) != json.loads(draw_receipt.read_text())["draw_npz_sha256"] or sha(draw_meta) != json.loads(draw_receipt.read_text())["draw_metadata_sha256"]:
        raise ValueError("fixed bootstrap draw manifest/receipt hash mismatch")
    draw_metadata = json.loads(draw_meta.read_text(encoding="utf-8"))
    expected_draw_status = "LOCKED_BEFORE_CURRENT_PROFILE_DEV_SCORE" if split == "dev" else "LOCKED_BEFORE_FINAL_SCORE"
    if draw_metadata.get("status") != expected_draw_status or draw_metadata.get("draws") != 5000 or draw_metadata.get("seed") != 2026101902:
        raise ValueError("Core B paired bootstrap protocol is not in its fixed pre-score state")
    metric_scale = json.loads(METRIC_LOCK.read_text(encoding="utf-8"))
    if metric_scale.get("status") != "LOCKED" or metric_scale.get("metric_contract") != "Full-28 absolute position MAE and future-internal 29-difference safe-FD L1; seam reported separately":
        raise ValueError("accepted metric-scale contract differs from V9")
    if split == "dev":
        old_q0_summary = V1_OUT / "replay/dev_replay/attempt_06_score_retry_02/CORE_B_DEV_REPLAY_SUMMARY.csv"
        if not old_q0_summary.is_file():
            raise FileNotFoundError("accepted cached-Q0 dev summary is missing")
        if (OUT / "core_b/CURRENT_PROFILE_DEV_RESULTS.csv").exists() or (OUT / "core_b/CURRENT_PROFILE_VS_Q0_DEV.csv").exists() or (OUT / "core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json").exists():
            raise FileExistsError("current-profile dev result artifacts already exist")
    else:
        final_outputs = [
            OUT / "core_b/FINAL_TASK_SEED_RESULTS.csv", OUT / "core_b/FINAL_SUMMARY_AND_CONTRASTS.csv",
            OUT / "core_b/FINAL_PAIRED_BOOTSTRAP_INTERVALS.csv", OUT / "core_b/CORE_B_V2_FINAL_SCORE_RECEIPT.json",
            RUN / "scores/core_b/FINAL_TARGETS_AFTER_PREDICTION_SEAL.npz",
        ]
        if any(path.exists() for path in final_outputs):
            raise FileExistsError("final score/target artifacts already exist; refusing a second target read")

    seal_path = OUT / "core_b" / ("CURRENT_PROFILE_DEV_PREDICTION_SEAL.json" if split == "dev" else "PREDICTION_MANIFEST_AND_SEAL.json")
    seal, prediction_records, _ = verify_seal(split, args.expected_lock_sha256, args.expected_final_freeze_sha256, args.expected_final_access_sha256)
    prefix = load_prefix(split)
    target_source = str(V9_DEV_TARGETS) if split == "dev" else "fixed final V3 parquet action rows selected after prediction seal"
    latch = make_exposure(exposure_path, split, seal_path, seal, target_source)
    if split == "dev":
        target = read_dev_targets(prefix, latch, exposure_path)
    else:
        target = read_final_targets(prefix, lock, latch, exposure_path)
    prefix["canonical_id"] = prefix["canonical_id"].astype(str)
    prefix["anchor_id"] = prefix["anchor_id"].astype(str)
    predictions = load_bundles(prediction_records, prefix["canonical_id"])
    scored = score_all(prefix, target, predictions, metric_scale, draw_npz)

    score_dir.mkdir(parents=True)
    atomic_csv(score_dir / "PER_ANCHOR.csv", scored["per_anchor"])
    atomic_csv(score_dir / "PER_TASK_SEED.csv", scored["task"])
    atomic_csv(score_dir / "PER_EPISODE_SEED.csv", scored["episode"])
    atomic_csv(score_dir / "PER_SEED.csv", scored["seed"])
    atomic_csv(score_dir / "SUMMARY.csv", scored["summary"])
    atomic_csv(score_dir / "CONTRASTS_AND_SEED_LOO.csv", scored["contrast"])
    atomic_csv(score_dir / "PAIRED_BOOTSTRAP_INTERVALS.csv", scored["bootstrap"])

    if split == "dev":
        # Keep a side-by-side comparison with the previously accepted cached-Q0 replay.
        old_path = V1_OUT / "replay/dev_replay/attempt_06_score_retry_02/CORE_B_DEV_REPLAY_SUMMARY.csv"
        old = {row["program"]: row for row in csv.DictReader(old_path.open(encoding="utf-8-sig", newline=""))}
        q0_rows = []
        for row in scored["summary"]:
            ref = old[row["program"]]
            q0_rows.append({
                "profile": "CURRENT_PROFILE_V2", "program": row["program"],
                "anchor_support": row["anchor_support"], "episode_support": row["episode_support"],
                **{f"current_{m}": row[f"{m}_mean"] for m in METRICS},
                **{f"cached_q0_{m}": float(ref[f"{m}_mean"]) for m in METRICS},
                **{f"current_minus_cached_q0_{m}": row[f"{m}_mean"] - float(ref[f"{m}_mean"]) for m in METRICS},
                "cached_q0_source": str(old_path), "cached_q0_profile": "accepted Q0 dev replay",
            })
        atomic_csv(OUT / "core_b/CURRENT_PROFILE_VS_Q0_DEV.csv", q0_rows)
        atomic_csv(OUT / "core_b/CURRENT_PROFILE_DEV_RESULTS.csv", scored["summary"])
        acceptance = {
            "schema_version": 1, "status": "PASS_CURRENT_PROFILE_V2_DEV_TECHNICAL_GATE",
            "created_utc": utc(), "route": "R2",
            "candidate_profile_lock_sha256": args.expected_lock_sha256,
            "feature_manifest_sha256": sha(OUT / "core_b/CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json"),
            "prediction_seal_sha256": sha(seal_path), "target_exposure_sha256": sha(exposure_path),
            "rows": 185, "episode_support": len(set(prefix["episode_id"].tolist())),
            "task_support": {str(t): int(np.sum(prefix["task_id"] == t)) for t in TASKS},
            "prediction_bundles": 22, "target_rows_materialized_after_seal": 185,
            "target_values_materialized_after_seal": int(target.size),
            "metrics_path": str(OUT / "core_b/CURRENT_PROFILE_DEV_RESULTS.csv"),
            "cached_q0_comparison_path": str(OUT / "core_b/CURRENT_PROFILE_VS_Q0_DEV.csv"),
            "old_q0_bridge_status": "FAIL_UNRESOLVED",
            "current_profile_results_used_for_profile_selection": False,
            "final_payload_read": False, "new_training_runs": 0,
            "technical_gates": {
                "source_identity": "PASS", "prefix_end_at_anchor": "PASS",
                "feature_repeatability": "PASS", "prediction_matrix_complete_finite_aligned": "PASS",
                "metric_scale_and_scoring": "PASS",
            },
            "score_interpretation": "descriptive current-profile development evaluation; no effect-based profile selection or training",
        }
        atomic_json(OUT / "core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json", acceptance)
        latch["status"] = "SCORED_CURRENT_PROFILE_V2_DEV"
        latch["score_receipt_path"] = str(OUT / "core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json")
        latch["score_receipt_sha256"] = sha(OUT / "core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json")
    else:
        atomic_csv(OUT / "core_b/FINAL_TASK_SEED_RESULTS.csv", scored["task"])
        atomic_csv(OUT / "core_b/FINAL_SUMMARY_AND_CONTRASTS.csv", scored["summary"] + scored["contrast"])
        atomic_csv(OUT / "core_b/FINAL_PAIRED_BOOTSTRAP_INTERVALS.csv", scored["bootstrap"])
        receipt = {
            "schema_version": 1, "status": "PASS_CURRENT_PROFILE_V2_FINAL_SCORED",
            "created_utc": utc(), "candidate_profile_lock_sha256": args.expected_lock_sha256,
            "final_freeze_sha256": args.expected_final_freeze_sha256,
            "final_access_receipt_sha256": args.expected_final_access_sha256,
            "prediction_seal_sha256": sha(seal_path), "target_exposure_sha256": sha(exposure_path),
            "rows": 162, "episode_support": len(set(prefix["episode_id"].tolist())),
            "task_support": {str(t): int(np.sum(prefix["task_id"] == t)) for t in TASKS},
            "prediction_bundles": 22, "target_rows_materialized_after_seal": int(len(target)),
            "target_values_materialized_after_seal": int(target.size),
            "unique_raw_parquet_action_rows_read": int(latch["raw_parquet_action_rows_read_unique"]),
            "raw_parquet_files_read": int(latch["raw_parquet_files_read"]),
            "new_training_runs": 0, "final_payload_reads": int(latch["final_payload_reads"]),
            "metrics": str(OUT / "core_b/FINAL_TASK_SEED_RESULTS.csv"),
            "summary": str(OUT / "core_b/FINAL_SUMMARY_AND_CONTRASTS.csv"),
            "status_by_route": {"original_q0_bridge": "FAIL_UNRESOLVED", "current_profile_route": "R2_COMPLETED"},
        }
        atomic_json(OUT / "core_b/CORE_B_V2_FINAL_SCORE_RECEIPT.json", receipt)
        latch["status"] = "SCORED_CURRENT_PROFILE_V2_FINAL"
        latch["score_receipt_path"] = str(OUT / "core_b/CORE_B_V2_FINAL_SCORE_RECEIPT.json")
        latch["score_receipt_sha256"] = sha(OUT / "core_b/CORE_B_V2_FINAL_SCORE_RECEIPT.json")
    latch["scoring_completed_utc"] = utc()
    atomic_json(exposure_path, latch)
    print(json.dumps({"status": latch["status"], "split": split, "rows": len(target),
                      "score_dir": str(score_dir), "summary": scored["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
