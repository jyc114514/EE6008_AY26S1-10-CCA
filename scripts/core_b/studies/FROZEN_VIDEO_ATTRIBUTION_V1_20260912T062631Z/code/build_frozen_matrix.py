#!/usr/bin/env python3
"""Build the locked Frozen-only matrix, checkpoint registry, and donor pairs."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
B23 = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
ROUTE = REPO / "reports/core_b/ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z"
REGISTRY = ROUTE / "ROUTE_C_CHECKPOINT_REGISTRY.csv"
INPUT = B23 / "inputs/B2_INPUTS_0125.npz"
OUT = ROOT.parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def build_registry() -> list[dict[str, str]]:
    with REGISTRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {row["model"]: row for row in rows if row["model"] in {"B3-V-D-F", "B3-FUS-D-F"}}
    if set(wanted) != {"B3-V-D-F", "B3-FUS-D-F"}:
        raise RuntimeError("Frozen checkpoint registry rows are incomplete")
    for model, row in wanted.items():
        path_keys = {"input_archive_path": "input_archive", "data_manifest_path": "data_manifest", "frame_map_path": "frame_map"}
        for key in ("checkpoint_path", "config_path", "canonical_prediction_path", "base_checkpoint_path", "feature_normalization_path", "target_normalization_path", "input_archive_path", "data_manifest_path", "frame_map_path"):
            source_key = path_keys.get(key, key)
            path = Path(row[source_key])
            if not path.exists():
                raise FileNotFoundError(f"{model} {key}: {path}")
        hash_keys = {"input_archive_sha256": "input_archive", "data_manifest_sha256": "data_manifest", "frame_map_sha256": "frame_map"}
        for key, path_key in (("checkpoint_sha256", "checkpoint_path"), ("config_sha256", "config_path"), ("canonical_prediction_sha256", "canonical_prediction_path"), ("base_checkpoint_sha256", "base_checkpoint_path"), ("feature_normalization_sha256", "feature_normalization_path"), ("target_normalization_sha256", "target_normalization_path"), ("input_archive_sha256", "input_archive_path"), ("data_manifest_sha256", "data_manifest_path"), ("frame_map_sha256", "frame_map_path")):
            source_key = path_keys.get(path_key, path_key)
            observed = digest(Path(row[source_key]))
            if observed != row[key]:
                raise RuntimeError(f"{model} {key} mismatch: {observed} != {row[key]}")
        row = dict(row)
        for old, new in (("input_archive", "input_archive_path"), ("data_manifest", "data_manifest_path"), ("frame_map", "frame_map_path")):
            row[new] = row[old]
        row["identity_status"] = "verified_614_full_max_abs_0"
        wanted[model] = row
    fields = ["model", "source_cell", "branch", "adaptation", "checkpoint_path", "checkpoint_sha256", "config_path", "config_sha256", "canonical_prediction_path", "canonical_prediction_sha256", "selected_epoch", "base_checkpoint_path", "base_checkpoint_sha256", "feature_normalization_path", "feature_normalization_sha256", "target_normalization_path", "target_normalization_sha256", "input_archive_path", "input_archive_sha256", "data_manifest_path", "data_manifest_sha256", "frame_map_path", "frame_map_sha256", "final_test_access", "canonical_inference", "identity_status"]
    selected = [wanted["B3-V-D-F"], wanted["B3-FUS-D-F"]]
    with (OUT / "FROZEN_CHECKPOINT_REGISTRY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    write_json(OUT / "FROZEN_CHECKPOINT_REGISTRY.json", {"rows": selected, "status": "PASS", "frozen_identity": {row["model"]: "614/614 max_abs=0" for row in selected}})
    return selected


def build_matrix() -> None:
    rows = [
        ("VF-P0", "B3-V-D-F", "video_only", "chronological_matched", "pixel", "none", "exact_reuse"),
        ("VF-P1", "B3-V-D-F", "video_only", "repeated_first", "pixel", "raw_video", "new_inference"),
        ("VF-P2", "B3-V-D-F", "video_only", "repeated_last", "pixel", "raw_video", "new_inference"),
        ("VF-P3", "B3-V-D-F", "video_only", "reversed_order", "pixel", "raw_video", "new_inference"),
        ("VF-P4", "B3-V-D-F", "video_only", "seeded_shuffle", "pixel", "raw_video", "new_inference"),
        ("FF-P0", "B3-FUS-D-F", "fusion", "chronological_matched", "pixel", "none", "exact_reuse"),
        ("FF-P1", "B3-FUS-D-F", "fusion", "repeated_first", "pixel", "raw_video", "new_inference"),
        ("FF-P2", "B3-FUS-D-F", "fusion", "repeated_last", "pixel", "raw_video", "new_inference"),
        ("FF-P3", "B3-FUS-D-F", "fusion", "reversed_order", "pixel", "raw_video", "new_inference"),
        ("FF-P4", "B3-FUS-D-F", "fusion", "seeded_shuffle", "pixel", "raw_video", "new_inference"),
        ("FF-R1", "B3-FUS-D-F", "fusion", "zero_video_feature", "feature", "normalized_video", "new_inference"),
        ("FF-R2", "B3-FUS-D-F", "fusion", "same_task_mismatch", "feature", "donor_video", "new_inference"),
        ("FF-R3", "B3-FUS-D-F", "fusion", "cross_task_mismatch", "feature", "donor_video", "new_inference"),
    ]
    fields = ["cell_id", "model", "branch", "condition", "condition_family", "intervention_level", "execution", "status", "final_test_access"]
    payload = [{**dict(zip(fields[:7], row)), "status": "locked", "final_test_access": 0} for row in rows]
    with (OUT / "FROZEN_ATTR_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload)
    write_json(OUT / "FROZEN_ATTR_MATRIX.json", {"status": "locked", "logical_cells": 13, "exact_reuse": [row["cell_id"] for row in payload if row["execution"] == "exact_reuse"], "new_inference": [row["cell_id"] for row in payload if row["execution"] == "new_inference"], "rows": payload})


def derange(values: list[int], rng: np.random.Generator) -> dict[int, int]:
    if len(values) <= 1:
        return {value: -1 for value in values}
    base = np.asarray(values, dtype=np.int64)
    for _ in range(100):
        donor = base[rng.permutation(len(base))]
        if not np.any(donor == base):
            return {int(recipient): int(source) for recipient, source in zip(base.tolist(), donor.tolist())}
    donor = np.roll(base, 1)
    return {int(recipient): int(source) for recipient, source in zip(base.tolist(), donor.tolist())}


def build_pairs() -> None:
    with np.load(INPUT, allow_pickle=False) as archive:
        ids_all = np.asarray(archive["episode_index"], dtype=np.int64)
        split = np.asarray(archive["split_code"], dtype=np.int8)
        tasks_all = np.asarray(archive["original_task_index"], dtype=np.int64)
    dev = split == 1
    ids = ids_all[dev]
    tasks = tasks_all[dev]
    if len(ids) != 614 or len(np.unique(ids)) != 614:
        raise RuntimeError("pairing population contract failed")
    rng_same = np.random.default_rng(20260911)
    by_task: dict[int, list[int]] = {}
    for position, task in enumerate(tasks.tolist()):
        by_task.setdefault(int(task), []).append(position)
    same_by_recipient: dict[int, int] = {}
    for positions in by_task.values():
        mapping = derange(positions, rng_same)
        for recipient_position, donor_position in mapping.items():
            same_by_recipient[int(ids[recipient_position])] = -1 if donor_position < 0 else int(ids[donor_position])
    rng_cross = np.random.default_rng(20260912)
    order = rng_cross.permutation(len(ids)).tolist()
    cross_by_recipient: dict[int, int] = {}
    for recipient_position in range(len(ids)):
        donor_position = -1
        for offset in range(len(ids)):
            candidate = int(order[(recipient_position + offset + 1) % len(ids)])
            if int(tasks[candidate]) != int(tasks[recipient_position]):
                donor_position = candidate
                break
        if donor_position < 0:
            raise RuntimeError(f"no cross-task donor for {ids[recipient_position]}")
        cross_by_recipient[int(ids[recipient_position])] = int(ids[donor_position])
    rows = []
    for position, (episode, task) in enumerate(zip(ids.tolist(), tasks.tolist())):
        same = same_by_recipient[int(episode)]
        cross = cross_by_recipient[int(episode)]
        rows.append({"recipient_episode": int(episode), "recipient_task": int(task), "recipient_dev_position": position, "same_task_donor": same if same >= 0 else "", "cross_task_donor": cross, "same_task_eligible": same >= 0, "cross_task_eligible": True, "same_task_fixed_point": bool(same == int(episode)), "cross_task_same_task": bool(cross == int(episode) or tasks[np.where(ids == cross)[0][0]] == task), "selection_basis": "episode_id_and_task_only; target_not_read_for_selection", "same_task_seed": 20260911, "cross_task_seed": 20260912})
    if any(row["same_task_fixed_point"] for row in rows) or any(row["cross_task_same_task"] for row in rows):
        raise RuntimeError("pairing invariant failed")
    fields = list(rows[0])
    with (OUT / "PAIRING_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    donor_counts: dict[int, int] = {}
    for row in rows:
        donor_counts[int(row["cross_task_donor"])] = donor_counts.get(int(row["cross_task_donor"]), 0) + 1
    write_json(OUT / "PAIRING_MANIFEST.json", {"status": "PASS", "N": len(rows), "same_task_seed": 20260911, "cross_task_seed": 20260912, "same_task_eligible_N": sum(bool(row["same_task_eligible"]) for row in rows), "cross_task_eligible_N": len(rows), "max_cross_task_donor_reuse": max(donor_counts.values()), "selection_does_not_read_targets": True, "rows": rows})


if __name__ == "__main__":
    build_registry()
    build_matrix()
    build_pairs()
    print(json.dumps({"status": "PASS", "logical_cells": 13, "pairing_N": 614}, sort_keys=True))
