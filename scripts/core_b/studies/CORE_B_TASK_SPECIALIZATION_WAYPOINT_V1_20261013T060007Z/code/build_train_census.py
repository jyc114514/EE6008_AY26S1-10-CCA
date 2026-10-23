#!/usr/bin/env python3
"""Build the train-only task census and apply the predeclared support stop gate."""

from __future__ import annotations
import os

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"
N2 = PROJECT / "reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z"
D3 = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V1_20261013T060007Z"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample_indices(anchor_frame: int) -> list[int]:
    return np.floor(np.linspace(anchor_frame - 119, anchor_frame, 32)).astype(np.int64).tolist()


def main() -> None:
    created = utc_now()
    source_registry = load_json(V3 / "SOURCE_REGISTRY.json")
    split_manifest_path = Path(source_registry["split_manifest"])
    dataset_manifest_component = next((part for part in split_manifest_path.parts if part.startswith("g1_")), "")
    if not dataset_manifest_component:
        raise RuntimeError("accepted split manifest path lacks its dataset content identifier")
    dataset_content_id = dataset_manifest_component.removeprefix("g1_")
    if len(dataset_content_id) != 40 or not dataset_content_id.isalnum():
        raise RuntimeError("dataset content identifier in the accepted manifest path is malformed")
    if not Path(source_registry["raw_root"]).name.endswith(dataset_content_id[:8]):
        raise RuntimeError("raw dataset root suffix does not match accepted dataset content identifier")
    cache_acceptance = load_json(V3 / "cache/CACHE_ACCEPTED.json")
    cache_manifest = load_json(V3 / "cache/CACHE_MANIFEST.json")
    extraction = load_json(V3 / "cache/CACHE_EXTRACTION_COMPLETE.json")
    token_contract = load_json(V3 / "TOKEN_LAYOUT_CONTRACT.json")
    target_manifest = load_json(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json")
    training_protocol = load_json(D3 / "PROTOCOL_LOCKED.json")
    q0_ts11 = load_json(N2 / "safe_models/TS11-D__Q0__S20261001__H30/EXPORT_MANIFEST.json")
    q0_state = load_json(N2 / "safe_models/STATE-CTRL__Q0__S20261001__H30/EXPORT_MANIFEST.json")
    q0_acceptance = load_json(N2 / "qa/SAFE_EXPORT_ACCEPTANCE.json")

    required_passes = {
        "source_registry": source_registry.get("status") == "LOCKED" and source_registry.get("final_test_access") == 0,
        "split_protocol": dataset_content_id == "6b6b599def75e3558b2736062e79bdbeec6cd224",
        "feature_cache": cache_acceptance.get("status") == "PASS" and cache_acceptance.get("final_test_access") == 0,
        "feature_extraction": extraction.get("status") == "PASS" and extraction.get("final_test_access") == 0,
        "token_contract": token_contract.get("status") == "PASS" and token_contract.get("final_test_access") == 0,
        "target_cache": target_manifest.get("status") == "PASS" and target_manifest.get("scope") == "primary_train_only" and target_manifest.get("final_test_access") == 0,
        "q0_safe_exports": q0_acceptance.get("status", "").startswith("PASS_") and q0_acceptance.get("final_test_access") == 0,
        "q0_ts11_export": q0_ts11.get("status") == "EXPORTED_PENDING_FRESH_PROCESS_ACCEPTANCE" and q0_ts11.get("final_test_access") == 0,
        "q0_state_export": q0_state.get("status") == "EXPORTED_PENDING_FRESH_PROCESS_ACCEPTANCE" and q0_state.get("final_test_access") == 0,
    }
    if not all(required_passes.values()):
        raise RuntimeError(f"accepted-source gate failed: {required_passes}")

    catalogue_rows = read_csv(SPLIT / "tasks.csv")
    task_names = {str(row["original_task_index"]): row["task_name_raw"] for row in catalogue_rows}
    allowlist_rows = read_csv(V3 / "DATA_ALLOWLIST.csv")
    train_allowlist = [row for row in allowlist_rows if row.get("split") == "train"]
    if any(row.get("final_test_access") != "0" for row in train_allowlist):
        raise RuntimeError("train allowlist contains a nonzero final-test access marker")
    allow_by_episode = {str(row["episode_index"]): row for row in train_allowlist}
    if len(allow_by_episode) != len(train_allowlist):
        raise RuntimeError("duplicate episode rows in canonical train allowlist")

    # The accepted anchor manifest is train/dev metadata only; retain only train primary rows.
    all_anchor_metadata = read_csv(V3 / "ANCHOR_MANIFEST.csv")
    train_primary = [
        row for row in all_anchor_metadata
        if row.get("split") == "train" and row.get("population") == "primary"
    ]
    if any(row.get("final_test_access") != "0" for row in train_primary):
        raise RuntimeError("train anchor metadata contains a nonzero final-test access marker")

    input_path = V3 / "data/V3_MAIN_INPUTS.npz"
    with np.load(input_path, allow_pickle=False) as archive:
        split_code = archive["split_code"]
        split_codes_observed = sorted(int(x) for x in np.unique(split_code).tolist())
        if split_codes_observed != [0, 1]:
            raise RuntimeError(f"accepted main input has unexpected split codes: {split_codes_observed}")
        train_indices = np.flatnonzero(split_code == 0).astype(np.int64)
        metadata = {
            "anchor_id": archive["anchor_id"][train_indices].astype("U64", copy=True),
            "episode_index": archive["episode_index"][train_indices].astype(np.int64, copy=True),
            "original_task_index": archive["original_task_index"][train_indices].astype(np.int64, copy=True),
            "anchor_frame": archive["anchor_frame"][train_indices].astype(np.int64, copy=True),
            "video_path": archive["video_path"][train_indices].astype("U512", copy=True),
        }
        # No target, state, RGB, dev metadata, or heldout arrays are requested from the archive.

    feature_path = Path(cache_manifest["feature_cache"])
    if not feature_path.is_file():
        raise RuntimeError(f"accepted frozen feature cache is missing: {feature_path}")
    feature_header = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    feature_shape = tuple(int(x) for x in feature_header.shape)
    feature_dtype = str(feature_header.dtype)
    del feature_header
    if feature_shape != tuple(int(x) for x in cache_manifest["shape"]):
        raise RuntimeError(f"feature cache header differs from accepted manifest: {feature_shape}")
    if len(train_indices) and int(train_indices.max()) >= feature_shape[0]:
        raise RuntimeError("one or more train cache indices fall outside the accepted frozen feature cache")

    anchor_by_id = {str(row["anchor_id"]): row for row in train_primary}
    if len(anchor_by_id) != len(train_primary):
        raise RuntimeError("duplicate anchor IDs in train primary manifest")
    if len(train_indices) != len(train_primary):
        raise RuntimeError("train input rows do not match train primary anchor metadata rows")

    episode_meta: dict[str, dict[str, Any]] = {}
    task_anchor_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cache_index, index in enumerate(train_indices.tolist()):
        row_i = cache_index
        anchor_id = str(metadata["anchor_id"][row_i])
        episode_id = str(int(metadata["episode_index"][row_i]))
        task_id = str(int(metadata["original_task_index"][row_i]))
        anchor_frame = int(metadata["anchor_frame"][row_i])
        video_path = str(metadata["video_path"][row_i])
        anchor = anchor_by_id.get(anchor_id)
        if anchor is None:
            raise RuntimeError(f"train input anchor missing from accepted manifest: {anchor_id}")
        for key, expected in (
            ("episode_index", episode_id),
            ("original_task_index", task_id),
            ("anchor_frame", str(anchor_frame)),
            ("video_path", video_path),
        ):
            if str(anchor[key]) != expected:
                raise RuntimeError(f"train input/anchor metadata mismatch at {anchor_id}: {key}")
        if episode_id not in allow_by_episode:
            raise RuntimeError(f"train primary episode missing from canonical train allowlist: {episode_id}")
        identity = allow_by_episode[episode_id]
        assignment_key = str(identity["split_assignment_key"])
        frames = sample_indices(anchor_frame)
        h30_start = int(anchor["target_1s_start_frame"])
        h30_end = int(anchor["target_1s_end_frame"])
        h30_legal = (
            h30_start == anchor_frame + 1
            and h30_end - h30_start + 1 == 30
            and len(frames) == 32
            and len(set(frames)) == 32
            and min(frames) == anchor_frame - 119
            and max(frames) == anchor_frame
            and min(frames) >= 0
            and max(frames) <= anchor_frame
        )
        if not h30_legal:
            raise RuntimeError(f"nonlegal H30 or prefix window in accepted primary train row: {anchor_id}")
        row = {
            "anchor_id": anchor_id,
            "episode_id": episode_id,
            "task_id": task_id,
            "cluster_id": assignment_key,
            "source_record_id": str(anchor["parquet_path"]),
            "video_source_id": video_path,
            "camera_view": Path(video_path).parts[2] if len(Path(video_path).parts) > 2 else "unknown",
            "cache_index": int(index),
            "anchor_frame": anchor_frame,
            "prefix_frame_first": min(frames),
            "prefix_frame_last": max(frames),
            "prefix_sampled_frame_count": len(frames),
            "target_h30_first_frame": h30_start,
            "target_h30_last_frame": h30_end,
            "target_h30_frame_count": h30_end - h30_start + 1,
        }
        task_anchor_rows[task_id].append(row)
        if episode_id in episode_meta:
            prior = episode_meta[episode_id]
            if (prior["task_id"], prior["cluster_id"], prior["source_record_id"], prior["video_source_id"]) != (
                task_id, assignment_key, row["source_record_id"], video_path
            ):
                raise RuntimeError(f"episode identity maps to conflicting task/source metadata: {episode_id}")
            prior["anchors"].append(row)
        else:
            episode_meta[episode_id] = {
                "task_id": task_id,
                "cluster_id": assignment_key,
                "source_record_id": row["source_record_id"],
                "video_source_id": video_path,
                "anchors": [row],
            }

    if len(task_anchor_rows) != 120:
        raise RuntimeError(f"unexpected train task support count: {len(task_anchor_rows)}")
    if len(episode_meta) != len(set(episode_meta)):
        raise RuntimeError("train episode deduplication failed")
    cluster_keys = [value["cluster_id"] for value in episode_meta.values()]
    if len(cluster_keys) != len(set(cluster_keys)):
        raise RuntimeError("accepted split assignment keys connect multiple train episodes; inspect before ranking")

    census: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for task_id in task_names:
        anchors = task_anchor_rows.get(task_id, [])
        episodes = [
            (episode_id, value)
            for episode_id, value in episode_meta.items()
            if value["task_id"] == task_id
        ]
        clusters = {value["cluster_id"] for _, value in episodes}
        records = {value["source_record_id"] for _, value in episodes}
        videos = {value["video_source_id"] for _, value in episodes}
        cameras = sorted({str(row["camera_view"]) for row in anchors})
        frames = [int(row["anchor_frame"]) for row in anchors]
        for episode_id, value in episodes:
            episode_anchors = value["anchors"]
            evidence.append({
                "task_id": task_id,
                "task_name_raw": task_names[task_id],
                "episode_id": episode_id,
                "cluster_id": value["cluster_id"],
                "cluster_identity_source": "canonical train split_assignment_key",
                "source_record_id": value["source_record_id"],
                "video_source_id": value["video_source_id"],
                "alias_link": "not separately represented in accepted train metadata",
                "session_or_trial_id": "UNKNOWN",
                "train_h30_anchor_count": len(episode_anchors),
                "camera_views": ";".join(sorted({str(row["camera_view"]) for row in episode_anchors})),
                "evidence_level": "accepted_episode_source_only",
                "collection_independence": "unverified",
            })
        census.append({
            "task_id": task_id,
            "task_name_raw": task_names[task_id],
            "robot": "G1",
            "source_revision": dataset_content_id,
            "role": "canonical_primary_train",
            "valid_train_episodes": len(episodes),
            "observed_source_clusters": len(clusters),
            "source_record_count": len(records),
            "video_source_count": len(videos),
            "h30_legal_anchors": len(anchors),
            "prefix_cache_anchors": len(anchors),
            "cache_complete": "true" if len(anchors) == sum(len(value["anchors"]) for _, value in episodes) else "false",
            "camera_views": ";".join(cameras),
            "prefix_source_span_frames": 120 if anchors else 0,
            "prefix_sampled_frames": 32 if anchors else 0,
            "anchor_frame_min": min(frames) if frames else "",
            "anchor_frame_max": max(frames) if frames else "",
            "target_h30_frames_per_anchor": 30 if anchors else 0,
            "session_or_trial_ids": "UNKNOWN",
            "collection_independence": "unverified_episode_source_only",
            "minimum_50_train_episodes_pass": str(len(clusters) >= 50).lower(),
            "exclusion_reason": "" if len(clusters) >= 50 else "fewer_than_50_valid_cached_primary_train_source_clusters",
        })

    census.sort(key=lambda row: (-int(row["observed_source_clusters"]), str(row["task_id"])))
    for rank, row in enumerate(census, start=1):
        row["train_metadata_rank"] = rank
    top_five = census[:5]
    pass_50 = [row for row in census if int(row["observed_source_clusters"]) >= 50]
    if len(pass_50) >= 5:
        raise RuntimeError("support gate unexpectedly passes; this script is only the insufficient-support receipt path")

    write_csv(OUT / "inventory/TRAIN_TASK_CENSUS.csv", [
        "train_metadata_rank", "task_id", "task_name_raw", "robot", "source_revision", "role",
        "valid_train_episodes", "observed_source_clusters", "source_record_count", "video_source_count",
        "h30_legal_anchors", "prefix_cache_anchors", "cache_complete", "camera_views",
        "prefix_source_span_frames", "prefix_sampled_frames", "anchor_frame_min", "anchor_frame_max",
        "target_h30_frames_per_anchor", "session_or_trial_ids", "collection_independence",
        "minimum_50_train_episodes_pass", "exclusion_reason",
    ], census)
    write_csv(OUT / "inventory/CLUSTER_IDENTITY_EVIDENCE.csv", [
        "task_id", "task_name_raw", "episode_id", "cluster_id", "cluster_identity_source",
        "source_record_id", "video_source_id", "alias_link", "session_or_trial_id",
        "train_h30_anchor_count", "camera_views", "evidence_level", "collection_independence",
    ], evidence)
    exclusions = [row for row in census if int(row["observed_source_clusters"]) < 50]
    write_csv(OUT / "inventory/TASK_EXCLUSIONS.csv", [
        "train_metadata_rank", "task_id", "task_name_raw", "valid_train_episodes",
        "observed_source_clusters", "h30_legal_anchors", "exclusion_reason",
    ], exclusions)

    source_files = {
        "v3_source_registry": V3 / "SOURCE_REGISTRY.json",
        "v3_data_allowlist": V3 / "DATA_ALLOWLIST.csv",
        "v3_anchor_manifest": V3 / "ANCHOR_MANIFEST.csv",
        "v3_task_catalogue": SPLIT / "tasks.csv",
        "v3_cache_acceptance": V3 / "cache/CACHE_ACCEPTED.json",
        "v3_cache_manifest": V3 / "cache/CACHE_MANIFEST.json",
        "v3_extraction_receipt": V3 / "cache/CACHE_EXTRACTION_COMPLETE.json",
        "v3_token_layout": V3 / "TOKEN_LAYOUT_CONTRACT.json",
        "target_cache_manifest": EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json",
        "accepted_q0_training_protocol": D3 / "PROTOCOL_LOCKED.json",
        "accepted_q0_model_code": D3 / "code/d3_model.py",
        "accepted_q0_training_code": D3 / "code/d3_train_corrected.py",
        "n2_safe_export_acceptance": N2 / "qa/SAFE_EXPORT_ACCEPTANCE.json",
        "n2_ts11_q0_export": N2 / "safe_models/TS11-D__Q0__S20261001__H30/EXPORT_MANIFEST.json",
        "n2_state_q0_export": N2 / "safe_models/STATE-CTRL__Q0__S20261001__H30/EXPORT_MANIFEST.json",
    }
    file_records = {
        key: {"path": str(path), "sha256": digest_file(path), "bytes": path.stat().st_size}
        for key, path in source_files.items()
    }
    resolved = {
        "schema_version": 1,
        "created_utc": created,
        "study_type": "task_frequency_selected_development_experiment",
        "run_root": str(OUT),
        "repo_root": str(REPO),
        "git_metadata": "unavailable: git executable and .git directory absent at designated repo root",
        "dataset_content_id": dataset_content_id,
        "dataset_content_id_source_field": "identifier embedded in accepted SOURCE_REGISTRY.json:split_manifest path; raw_root suffix independently matches its first 8 characters",
        "raw_dataset_root": source_registry["raw_root"],
        "canonical_split_manifest": source_registry["split_manifest"],
        "task_catalogue": str(SPLIT / "tasks.csv"),
        "train_allowlist": str(V3 / "DATA_ALLOWLIST.csv"),
        "train_anchor_manifest": str(V3 / "ANCHOR_MANIFEST.csv"),
        "main_input_archive": str(input_path),
        "main_input_sha256_from_accepted_q0": q0_ts11["input_sha256"],
        "target_cache_manifest": str(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"),
        "target_cache_sha256": target_manifest["target_cache_sha256"],
        "feature_cache": str(feature_path),
        "feature_cache_sha256_from_accepted_manifest": cache_manifest["feature_cache_sha256"],
        "feature_cache_sha256_recomputed_this_run": False,
        "feature_cache_shape_header": list(feature_shape),
        "feature_cache_dtype_header": feature_dtype,
        "feature_cache_row_count": int(cache_manifest["row_count"]),
        "prefix_provenance": {
            "encoder_checkpoint": extraction["raw_encoder_checkpoint"],
            "encoder_checkpoint_sha256": extraction["raw_encoder_checkpoint_sha256"],
            "preprocessing": extraction["preprocess"],
            "extractor_source": str(V3 / "code/extract_v3_features.py"),
            "extractor_source_sha256_at_audit": digest_file(V3 / "code/extract_v3_features.py"),
            "per_row_frame_rule": "floor(linspace(anchor_frame-119, anchor_frame, 32)); all selected frames <= anchor",
            "frame_identity_source": "V3_MAIN_INPUTS metadata anchor_id/video_path/anchor_frame checked against accepted ANCHOR_MANIFEST",
            "feature_cache_row_index_rule": "V3 main input row index; accepted extractor concatenates main rows before supplementary rows",
        },
        "accepted_q0_recipe": {
            "learning_rate_ref": 1e-3,
            "optimizer": training_protocol["training"]["optimizer"],
            "epochs": training_protocol["training"]["epochs"],
            "dtype": training_protocol["training"]["dtype"],
            "gradient_clip_norm": training_protocol["training"]["gradient_clip_norm"],
            "scheduler": training_protocol["only_scientific_intervention"],
            "source_protocol_sha256": training_protocol["source_protocol_sha256"],
        },
        "source_files": file_records,
    }
    write_json(OUT / "inventory/RESOLVED_PATHS.json", resolved)

    top_candidates = [{
        "train_metadata_rank": int(row["train_metadata_rank"]),
        "task_id": str(row["task_id"]),
        "original_task_name": str(row["task_name_raw"]),
        "valid_train_episodes": int(row["valid_train_episodes"]),
        "observed_source_clusters": int(row["observed_source_clusters"]),
        "h30_legal_anchors": int(row["h30_legal_anchors"]),
        "minimum_50_train_episodes_pass": row["minimum_50_train_episodes_pass"] == "true",
    } for row in top_five]
    lock = {
        "schema_version": 1,
        "created_utc": created,
        "status": "insufficient_task_support",
        "study_type": "task_frequency_selected_development_experiment",
        "ranking_population": "canonical primary train only; unique accepted source assignment clusters with a complete H30 prefix cache row",
        "ranking_key": "valid_train_source_clusters descending, task_id canonical string ascending",
        "train_metadata_ranking_used_dev_support": False,
        "train_metadata_ranking_used_dev_outcomes": False,
        "minimum_train_episodes_per_task": 50,
        "minimum_inner_fit_episodes_and_clusters": {"episodes": 30, "clusters": 10},
        "minimum_inner_tune_episodes_and_clusters": {"episodes": 10, "clusters": 5},
        "base_threshold_pass_task_count": len(pass_50),
        "required_selected_task_count": 5,
        "selected_tasks": [],
        "top_five_train_only_candidates": top_candidates,
        "inner_fit_tune_hash_split": "not computed because fewer than five tasks pass the mandatory 50-episode gate",
        "dev_support_by_task_read": False,
        "dev_outcomes_read_this_run": False,
        "formal_fits_admitted": 0,
        "screening_fits_admitted": 0,
        "phase_b_fits_admitted": 0,
        "stop_reason": f"Only {len(pass_50)} task(s) have at least 50 unique valid, cached primary-train source clusters; the fifth-ranked task has {top_candidates[4]['valid_train_episodes']} episodes. The fixed protocol requires five eligible tasks; thresholds were not reduced.",
        "collection_independence": "unverified_episode_source_only",
        "cluster_identity_limit": "accepted split assignment keys and source paths are available; no session/trial/alias IDs are present in the inspected train metadata",
        "final_heldout_payload_access": 0,
        "final_heldout_metadata": "no record IDs, labels, predictions, or metrics read; see OUTCOME_EXPOSURE.json for aggregate split-summary metadata observed during preliminary manifest inspection",
        "evidence": {
            "train_task_census": "inventory/TRAIN_TASK_CENSUS.csv",
            "task_exclusions": "inventory/TASK_EXCLUSIONS.csv",
            "cluster_identity_evidence": "inventory/CLUSTER_IDENTITY_EVIDENCE.csv",
            "resolved_paths": "inventory/RESOLVED_PATHS.json",
            "support_gate": {
                "total_valid_primary_train_episodes": len(episode_meta),
                "total_h30_legal_primary_train_anchors": len(train_primary),
                "train_tasks_with_support": len(task_anchor_rows),
                "tasks_passing_at_least_50_source_clusters": len(pass_50),
                "max_valid_train_episodes": int(top_candidates[0]["valid_train_episodes"]),
                "fifth_ranked_valid_train_episodes": int(top_candidates[4]["valid_train_episodes"]),
            },
        },
    }
    write_json(OUT / "protocol/TASK_SELECTION_LOCK.json", lock)
    write_json(OUT / "protocol/PROTOCOL_LOCK.json", {
        "schema_version": 1,
        "created_utc": created,
        "status": "insufficient_task_support",
        "study_type": "task_frequency_selected_development_experiment",
        "scope": "H30 / Full-28 / accepted Q0 architecture reference / frozen V-JEPA prefix features",
        "encoder_training": 0,
        "encoder_extraction": 0,
        "final_heldout_payload_access": 0,
        "new_dev_outcome_access": 0,
        "stop_stage": "metadata_admitted_before_tasks_locked",
        "stop_reason": lock["stop_reason"],
        "task_selection_lock_sha256": digest_file(OUT / "protocol/TASK_SELECTION_LOCK.json"),
    })
    write_json(OUT / "inventory/DEV_SUPPORT.json", {
        "status": "not_read_due_to_train_support_gate",
        "dev_support_by_task_read": False,
        "dev_outcomes_read": False,
        "reason": "the required five-task train eligibility gate failed before task lock",
        "final_heldout_payload_access": 0,
    })
    write_json(OUT / "inventory/OUTCOME_EXPOSURE.json", {
        "schema_version": 1,
        "created_utc": created,
        "task_selection_used_dev_outcomes": False,
        "new_dev_outcomes_read": False,
        "dev_support_by_task_read": False,
        "dev_metadata_scope": "global split-code and accepted aggregate counts only; no per-task dev support rows or dev target values were inspected",
        "final_heldout_payload_access": 0,
        "final_heldout_metadata_scope": "During preliminary manifest inspection, the accepted v3 split_summary.json was opened and its aggregate split_counts/per_primary_task_split_counts fields were emitted, including final_test counts. No final-test record IDs, labels, videos, targets, predictions, or metrics were read. This is disclosed as aggregate metadata exposure.",
        "nonselection_fields_observed": [
            "accepted CACHE_MANIFEST.json cache-level D_S_mean_abs_difference field; not used for task ranking",
            "accepted v3 split summary aggregate train/validation/final_test counts by task; not used for task ranking"
        ],
        "ranking_inputs": [
            "canonical primary train anchor metadata",
            "train-only split_assignment_key/source paths",
            "accepted frozen prefix cache coverage metadata",
            "accepted task catalogue IDs and raw names"
        ],
        "development_or_final_metrics_used_for_selection": False,
        "old_roots_written": False
    })
    write_json(OUT / "qa/METADATA_GATE_ACCEPTANCE.json", {
        "status": "PASS_STOP_RULE_APPLIED",
        "created_utc": created,
        "source_receipts_pass": required_passes,
        "train_input_rows": int(len(train_indices)),
        "train_primary_anchor_rows": len(train_primary),
        "train_valid_episodes_with_h30_cache": len(episode_meta),
        "train_task_count": len(task_anchor_rows),
        "min_50_task_count": len(pass_50),
        "selected_task_count": 0,
        "cache_rows_checked_by_header_only": int(feature_shape[0]),
        "train_anchor_identity_joins_pass": True,
        "train_h30_frame_bounds_pass": True,
        "prefix_frame_max_le_anchor_pass": True,
        "target_arrays_read": False,
        "dev_targets_or_outcomes_read": False,
        "heldout_payload_read": False,
        "note": "The predeclared insufficiency stop gate is an accepted outcome of the metadata audit; no model, score, prediction, or GPU training was started.",
    })

    print(json.dumps({
        "status": lock["status"],
        "train_valid_episodes": len(episode_meta),
        "train_h30_anchors": len(train_primary),
        "supported_tasks": len(task_anchor_rows),
        "tasks_passing_minimum_50": len(pass_50),
        "top_five": top_candidates,
        "dev_support_read": False,
        "dev_outcomes_read": False,
        "heldout_payload_read": False,
        "run_root": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
