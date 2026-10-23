#!/usr/bin/env python3
"""Preserve split-salted V2 outputs before rebuilding the literal locked source-key split."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/INNER_SPLIT_DATASET_SALTED_V1_20261013T083000Z"
TASK_IDS = (119, 94, 90, 141, 230)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def move_rel(rel: str, moved: list[str]) -> None:
    source = ROOT / rel
    if not source.exists():
        return
    target = ARCHIVE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"archive destination already exists: {target}")
    shutil.move(str(source), str(target))
    moved.append(rel)


def main() -> None:
    if ARCHIVE.exists():
        raise FileExistsError(f"split correction archive already exists: {ARCHIVE}")
    owner_lock_path = ROOT / "coordination/OWNER.lock"
    with owner_lock_path.open("a+") as owner_lock:
        try:
            fcntl.flock(owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("V2 owner is still active; refuse split correction archive") from exc

        watcher_status_dir = ROOT / "coordination/watcher_status"
        active = []
        for path in watcher_status_dir.glob("*.json"):
            if path.name.startswith("dryrun_") or ".attempt_" in path.name:
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if state.get("status") == "running":
                active.append(path.name)
        if active:
            raise RuntimeError(f"watcher statuses still report active children: {active}")

        resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
        dataset_id = str(resolved["dataset_content_id"])
        split_rows = read_csv(ROOT / "protocol/INNER_SPLIT.csv")
        current_roles = {(int(row["task_id"]), str(row["cluster_id"])): str(row["role"]) for row in split_rows}

        train_npz_path = ROOT / "data/TRAIN_SELECTED_H30.npz"
        with np.load(train_npz_path, allow_pickle=False) as archive:
            episode_id = archive["episode_id"].astype(np.int64, copy=True)
            task_id = archive["task_id"].astype(np.int64, copy=True)
            source_cluster = archive["source_cluster"].astype("U128", copy=True)
            old_role = archive["inner_role"].astype("U16", copy=True)

        allow_path = Path(resolved["v3_root"]) / "DATA_ALLOWLIST.csv"
        allow = [
            row for row in read_csv(allow_path)
            if int(row["original_task_index"]) in TASK_IDS and row.get("split") == "train"
        ]
        split_key_by_episode = {str(row["episode_index"]): str(row["split_assignment_key"]) for row in allow}
        if len(split_key_by_episode) != 220:
            raise RuntimeError(f"accepted train split-key map has {len(split_key_by_episode)} episodes, expected 220")
        parent_identity = read_csv(Path(resolved["parent_root"]) / "inventory/CLUSTER_IDENTITY_EVIDENCE.csv")
        cluster_by_episode = {
            str(row["episode_id"]): str(row["cluster_id"])
            for row in parent_identity if int(row["task_id"]) in TASK_IDS
        }
        joins = sorted(set(split_key_by_episode) & set(cluster_by_episode))
        if len(joins) != 220:
            raise RuntimeError(f"split-key/parent identity join covers {len(joins)} episodes, expected 220")
        same_key = sum(split_key_by_episode[ep] == cluster_by_episode[ep] for ep in joins)
        if same_key != 220:
            raise RuntimeError(f"canonical split key differs from parent group for {220-same_key} episodes; stop for review")

        new_roles: dict[tuple[int, str], str] = {}
        comparison: dict[str, Any] = {}
        for tid in TASK_IDS:
            group_keys = sorted({str(source_cluster[i]) for i in range(len(task_id)) if int(task_id[i]) == tid})
            tune_n = math.ceil(0.2 * len(group_keys)) if len(group_keys) >= 2 else (1 if group_keys else 0)
            salted = sorted(group_keys, key=lambda key: hashlib.sha256(
                f"v2split20261013|{dataset_id}|{key}".encode("utf-8")
            ).hexdigest())[:tune_n]
            literal = sorted(group_keys, key=lambda key: hashlib.sha256(
                f"v2split20261013|{key}".encode("utf-8")
            ).hexdigest())[:tune_n]
            current_tune = {key for (task, key), role in current_roles.items() if task == tid and role == "inner_tune"}
            if current_tune != set(salted):
                raise RuntimeError(f"current role assignment does not match recorded dataset-salted algorithm for task {tid}")
            tune_set = set(literal)
            for key in group_keys:
                new_roles[(tid, key)] = "inner_tune" if key in tune_set else "inner_fit"
            old_anchor_tune = int(sum(1 for i in range(len(task_id)) if int(task_id[i]) == tid and old_role[i] == "inner_tune"))
            new_anchor_tune = int(sum(1 for i in range(len(task_id)) if int(task_id[i]) == tid and new_roles[(tid, str(source_cluster[i]))] == "inner_tune"))
            comparison[str(tid)] = {
                "source_groups": len(group_keys), "tune_groups": tune_n,
                "current_dataset_salted_matches_record": True,
                "current_tune_groups": len(current_tune),
                "literal_v2_tune_groups": len(tune_set),
                "changed_tune_groups": len(current_tune.symmetric_difference(tune_set)),
                "old_tune_anchors": old_anchor_tune, "literal_v2_tune_anchors": new_anchor_tune,
            }

        role_changes = sum(old_role[i] != new_roles[(int(task_id[i]), str(source_cluster[i]))] for i in range(len(task_id)))
        if not role_changes:
            raise RuntimeError("literal split unexpectedly matches existing assignment; no archive needed")

        old_exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
        old_exposure = json.loads(old_exposure_path.read_text(encoding="utf-8"))
        dev_target_path = ROOT / "data/DEV_TARGETS_SCORER_ONLY.npz"
        dev_target_sha = sha(dev_target_path) if dev_target_path.is_file() else None
        old_seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
        old_seal_sha = sha(old_seal_path) if old_seal_path.is_file() else None
        completed_counts = {}
        for phase in ("screening", "formal", "waypoint"):
            outdir = ROOT / "model_only" / phase
            completed_counts[phase] = sum(
                1 for path in outdir.glob("*/RUN_COMPLETE.json")
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
            )
        failed_attempt_count = sum(
            1 for phase in ("screening", "formal", "waypoint")
            for _ in (ROOT / "model_only" / phase).glob("*/RUN_STATUS.failed-attempt-*.json")
        )

        ARCHIVE.mkdir(parents=True)
        moved: list[str] = []
        for rel in (
            "data/TRAIN_SELECTED_H30.npz",
            "protocol/INNER_SPLIT.csv", "protocol/INNER_SPLIT_LOCK.json",
            "protocol/LR_EPOCH_SELECTION.json", "protocol/FORMAL_MATRIX.json",
            "protocol/PREDICTION_BATCH_SEAL.json",
            "inventory/TRAIN_TASKS_V2.csv", "inventory/OUTCOME_EXPOSURE.json",
            "qa/INPUT_AND_CACHE_QA.json", "qa/NORMALIZER_SCREENING_QA.json", "qa/NORMALIZER_FORMAL_QA.json",
            "qa/SAMPLE_WEIGHT_PRECISION_QA.json", "qa/FIRST_REAL_PILOT.json",
            "qa/DEV_TARGET_SCORING_QA.json", "qa/FIGURE_BUILD_QA.json", "qa/FINAL_ACCEPTANCE.json",
            "RETURN_TO_CHATGPT.zh-CN.md", "RETURN_TO_CHATGPT.json",
            "coordination/JOB_LEDGER.json", "results", "screening", "normalizers",
            "model_only/screening", "model_only/formal", "model_only/waypoint",
            "predictions", "figures", "figure_data",
            "configs/fits/formal", "configs/watcher/formal",
            "configs/fits/waypoint", "configs/watcher/waypoint",
            "configs/inference/formal_dev", "configs/watcher/formal_dev",
            "coordination/queues/FORMAL.json", "coordination/queues/WAYPOINT.json",
            "coordination/queues/FORMAL_DEV.json",
        ):
            move_rel(rel, moved)

        for path in sorted((ROOT / "coordination").glob("PHASE_STATUS_*.json")):
            move_rel(path.relative_to(ROOT).as_posix(), moved)
        for path in sorted((ROOT / "coordination").glob("OWNER_STATE*.json")):
            move_rel(path.relative_to(ROOT).as_posix(), moved)
        for path in sorted((ROOT / "inventory").glob("OUTCOME_EXPOSURE.json.*.bak")):
            move_rel(path.relative_to(ROOT).as_posix(), moved)

        queue_paths = [
            ROOT / "coordination/queues/SCREENING_ALL.json",
            ROOT / "coordination/queues/PILOT_RELOAD.json",
            ROOT / "coordination/queues/SCREENING_PILOT.json",
            ROOT / "coordination/queues/SCREENING_REMAINING.json",
            ARCHIVE / "coordination/queues/FORMAL.json",
            ARCHIVE / "coordination/queues/WAYPOINT.json",
            ARCHIVE / "coordination/queues/FORMAL_DEV.json",
        ]
        job_ids: set[str] = set()
        for queue_path in queue_paths:
            if not queue_path.is_file():
                continue
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            for raw in queue.get("jobs", []):
                path = Path(raw)
                if not path.is_file():
                    continue
                watcher = json.loads(path.read_text(encoding="utf-8"))
                job_ids.add(str(watcher["job_id"]))
        for base in (
            ROOT / "coordination/watcher_status",
            ROOT / "coordination/watcher_status/console",
            ROOT / "logs/watcher_children",
        ):
            if not base.is_dir():
                continue
            for path in sorted(base.iterdir()):
                if any(path.name == job_id + ".json" or path.name == job_id + ".log" or path.name.startswith(job_id + ".attempt_") for job_id in job_ids):
                    move_rel(path.relative_to(ROOT).as_posix(), moved)

        current_train = ROOT / "data/TRAIN_SELECTED_H30.npz"
        new_roles_array = np.asarray([
            new_roles[(int(tid), str(cluster))]
            for tid, cluster in zip(task_id.tolist(), source_cluster.tolist())
        ], dtype="U16")
        tmp = current_train.with_name(f".{current_train.name}.part.npz")
        with np.load(ARCHIVE / "data/TRAIN_SELECTED_H30.npz", allow_pickle=False) as old:
            arrays = {key: old[key].copy() for key in old.files}
        arrays["inner_role"] = new_roles_array
        np.savez_compressed(tmp, **arrays)
        tmp.replace(current_train)

        summary = {
            "schema_version": 1,
            "status": "SUPERSEDED_PROTOCOL_VERSION_ARCHIVED",
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "reason": "V2 specified sha256('v2split20261013|' + canonical_source_key); prior implementation added dataset_content_id as an extra salt, producing different tune groups in every task.",
            "source_key_validation": {
                "train_episodes_joined": len(joins), "split_assignment_key_equals_parent_cluster": same_key,
                "key_used_for_literal_v2_split": "parent cluster_id, verified equal to accepted split_assignment_key for all selected train episodes",
            },
            "split_assignment_comparison": comparison,
            "train_anchor_roles_changed": int(role_changes),
            "prior_completed_fit_receipts_archived": completed_counts,
            "prior_failed_gpu_attempt_receipts_archived": failed_attempt_count,
            "prior_dev_target_labels_were_read_after_old_prediction_seal": old_exposure.get("dev_target_labels_read_after_prediction_seal") is True,
            "prior_dev_target_artifact_sha256_retained": dev_target_sha,
            "prior_prediction_seal_sha256": old_seal_sha,
            "prior_dev_scores_used_for_corrected_lr_epoch_selection": False,
            "new_train_npz_sha256": sha(current_train),
            "archive_root": str(ARCHIVE),
            "moved_artifact_count": len(moved),
            "moved_artifact_paths": moved,
            "final_heldout_payload_access": 0,
        }
        (ARCHIVE / "ARCHIVE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (ROOT / "protocol/INNER_SPLIT_CORRECTION.json").write_text(json.dumps({
            "schema_version": 1, "status": "APPLIED",
            "created_utc": summary["created_utc"],
            "specified_rule": "sort sha256('v2split20261013|' + canonical_source_key); first ceil(0.2*n_groups) groups per task are inner_tune",
            "canonical_source_key": "accepted split_assignment_key; parent cluster identity equals this key for all 220 selected train episodes",
            "superseded_rule": "sha256('v2split20261013|' + dataset_content_id + '|' + cluster_id)",
            "superseded_output_root": str(ARCHIVE),
            "changed_anchor_roles": int(role_changes),
            "prior_dev_label_access_preserved": old_exposure.get("dev_target_labels_read_after_prediction_seal") is True,
            "prior_dev_scores_used_for_corrected_selection": False,
            "corrected_matrix_logical_fit_slots": {"screening": 36, "formal": 36, "waypoint": 6},
            "final_heldout_payload_access": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "PASS",
            "role_assignments_changed": int(role_changes),
            "prior_completed_fits_archived": completed_counts,
            "prior_failed_gpu_attempts_archived": failed_attempt_count,
            "prior_dev_labels_read": old_exposure.get("dev_target_labels_read_after_prediction_seal"),
            "new_train_npz_sha256": sha(current_train),
            "archive_root": str(ARCHIVE),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
