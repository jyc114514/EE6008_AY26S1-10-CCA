#!/usr/bin/env python3
"""CPU-only audit that locked inner groups use the literal V2 canonical-source hash."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    split_lock = json.loads((ROOT / "protocol/INNER_SPLIT_LOCK.json").read_text(encoding="utf-8"))
    if split_lock.get("status") != "LOCKED" or "canonical_source_key" not in split_lock.get("algorithm", ""):
        raise RuntimeError("literal inner split lock is absent")
    expected_algorithm = "sort sha256('v2split20261013|' + canonical_source_key); first ceil(0.2*n_clusters) per task is inner_tune"
    if split_lock.get("algorithm") != expected_algorithm:
        raise RuntimeError("split-lock algorithm text does not exactly match V2")
    rows = read_csv(ROOT / "protocol/INNER_SPLIT.csv")
    roles = {(int(row["task_id"]), str(row["cluster_id"])): str(row["role"]) for row in rows}
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    v3 = Path(resolved["v3_root"])
    allow = {
        str(row["episode_index"]): str(row["split_assignment_key"])
        for row in read_csv(v3 / "DATA_ALLOWLIST.csv")
        if int(row["original_task_index"]) in TASK_IDS and row.get("split") == "train"
    }
    parent = {
        str(row["episode_id"]): str(row["cluster_id"])
        for row in read_csv(Path(resolved["parent_root"]) / "inventory/CLUSTER_IDENTITY_EVIDENCE.csv")
        if int(row["task_id"]) in TASK_IDS
    }
    join = sorted(set(allow) & set(parent))
    if len(join) != 220 or any(allow[episode] != parent[episode] for episode in join):
        raise RuntimeError("canonical source key does not match accepted parent source group for all train episodes")
    with np.load(ROOT / "data/TRAIN_SELECTED_H30.npz", allow_pickle=False) as data:
        episode = data["episode_id"].copy()
        task = data["task_id"].copy()
        source = data["source_cluster"].astype("U128", copy=True)
        inner_role = data["inner_role"].astype("U16", copy=True)
        anchor_ids = data["anchor_id"].astype("U64", copy=True)
    if len(set(anchor_ids.tolist())) != len(anchor_ids):
        raise RuntimeError("train anchor IDs are not unique")
    task_results = {}
    for task_id in TASK_IDS:
        keys = sorted({str(source[i]) for i in range(len(task)) if int(task[i]) == task_id})
        tune_n = math.ceil(0.2 * len(keys)) if len(keys) >= 2 else (1 if keys else 0)
        expected_tune = set(sorted(
            keys,
            key=lambda key: hashlib.sha256(f"v2split20261013|{key}".encode("utf-8")).hexdigest(),
        )[:tune_n])
        actual_tune = {key for (tid, key), role in roles.items() if tid == task_id and role == "inner_tune"}
        if actual_tune != expected_tune:
            raise RuntimeError(f"literal hash assignment mismatch for task {task_id}")
        fit_anchors = sum(int(task[i]) == task_id and inner_role[i] == "inner_fit" for i in range(len(task)))
        tune_anchors = sum(int(task[i]) == task_id and inner_role[i] == "inner_tune" for i in range(len(task)))
        task_results[str(task_id)] = {
            "source_groups": len(keys), "tune_groups": tune_n,
            "fit_anchors": int(fit_anchors), "tune_anchors": int(tune_anchors),
            "actual_tune_group_set_matches_literal_sha": True,
        }
    old_data = ROOT / "archive/INNER_SPLIT_DATASET_SALTED_V1_20261013T083000Z/data/TRAIN_SELECTED_H30.npz"
    with np.load(old_data, allow_pickle=False) as old:
        old_anchor = old["anchor_id"].astype("U64", copy=True)
        old_role = old["inner_role"].astype("U16", copy=True)
    old_role_by_anchor = {str(anchor): str(role) for anchor, role in zip(old_anchor.tolist(), old_role.tolist())}
    if set(old_role_by_anchor) != set(anchor_ids.tolist()):
        raise RuntimeError("old and corrected train anchor identity sets differ")
    changed = sum(old_role_by_anchor[str(anchor)] != str(role) for anchor, role in zip(anchor_ids.tolist(), inner_role.tolist()))
    if changed != 367:
        raise RuntimeError(f"expected 367 archived role changes, observed {changed}")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "hash_rule": "sha256('v2split20261013|' + canonical_source_key)",
        "source_key_joined_episodes": len(join),
        "source_key_equals_parent_cluster_for_every_train_episode": True,
        "all_train_tune_groups_disjoint": split_lock.get("all_train_tune_groups_disjoint") is True,
        "fixed_task_ids": list(TASK_IDS),
        "task_counts": task_results,
        "train_rows": len(task),
        "changed_anchor_roles_vs_superseded_salted_split": changed,
        "dev_targets_read": False,
        "final_heldout_payload_access": 0,
    }
    (ROOT / "qa/INNER_SPLIT_LITERAL_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
