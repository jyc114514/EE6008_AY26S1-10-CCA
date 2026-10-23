"""Build fixed, label-free development and bridge manifests for EE6008 replay.

This command reads only whitelisted split/cache metadata and already accepted
train/dev prefix arrays. It does not open video, final Parquet payload columns,
feature tensors, labels, or future targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
DATASET_REVISION = "6b6b599def75e3558b2736062e79bdbeec6cd224"
TASK_IDS = (119, 94, 90, 141, 230)
TASK_BRIDGE_QUOTAS = {119: 7, 94: 7, 90: 6, 141: 6, 230: 6}

SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
CORE_A_DYNAMIC_FEATURES = PROJECT / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet"
CORE_A_STATIC_FEATURES = PROJECT / "features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet"
CORE_B_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
CORE_B_ALLOWLIST = CORE_B_ROOT / "DATA_ALLOWLIST.csv"
CORE_B_ANCHORS = CORE_B_ROOT / "ANCHOR_MANIFEST.csv"
V9_ROOT = PROJECT / "reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
V9_TRAIN_PREFIX = PROJECT / "reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/data/TRAIN_PREFIXES_NO_LABELS.npz"
V9_DEV_PREFIX = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/data/DEV_PREFIXES_NO_LABELS.npz"
CORE_A_MATRIX = OUT_ROOT / "protocol/FINAL_MODEL_MATRIX.csv"
CORE_B_FINAL_POPULATION = OUT_ROOT / "protocol/CORE_B_FINAL_POPULATION_METADATA.csv"

CORE_B_PREFIX_FIELDS = (
    "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
    "cache_index", "anchor_frame", "history_native", "history_mask",
    "relative_delta", "anchor_native",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_core_a_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Explicit projection: no task names, labels, categories, or label maps.
    columns = [
        "episode_index", "split", "video_path", "parquet_path",
        "classification_inclusion", "split_assignment_key",
        "v3_benchmark_inclusion",
    ]
    table = pd.read_parquet(SPLIT_MANIFEST, columns=columns)
    primary = table[
        table["split"].isin(["train", "validation"])
        & table["classification_inclusion"].astype(bool)
        & (table["v3_benchmark_inclusion"].astype(str) == "primary")
    ].copy()
    train = primary[primary["split"] == "train"].copy()
    dev = primary[primary["split"] == "validation"].copy()
    if len(train) != 2757 or len(dev) != 617:
        raise ValueError(f"Core A primary train/dev counts differ from source lock: {len(train)}/{len(dev)}")
    if primary["episode_index"].duplicated().any():
        raise ValueError("Core A primary metadata has duplicate episode IDs")
    return train, dev


def _feature_map(path: Path) -> dict[int, str]:
    # Historical cache split labels predate the current v3 assignment. Project
    # only episode id and feature path, then join to current v3 membership.
    # Label-bearing columns and cache payloads are excluded.
    frame = pd.read_parquet(path, columns=["episode_index", "feature_path"])
    frame["episode_index"] = frame["episode_index"].astype(np.int64)
    if frame["episode_index"].duplicated().any():
        raise ValueError(f"duplicate train/dev episode IDs in feature manifest: {path}")
    return {int(row.episode_index): str(row.feature_path) for row in frame.itertuples(index=False)}


def _accepted_dev_order(program: str, seed: int) -> list[int]:
    with CORE_A_MATRIX.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    match = [
        row for row in rows
        if row["core"] == "A" and row["program"] == program and int(row["seed"]) == seed
    ]
    if len(match) != 1:
        raise ValueError(f"expected one accepted Core A matrix row for {program}/{seed}")
    path = Path(match[0]["checkpoint_path"]).with_name("predictions_best.npz")
    with np.load(path, allow_pickle=False) as archive:
        if "episode_index" not in archive.files:
            raise ValueError(f"accepted Core A dev predictions lack episode_index: {path}")
        ids = archive["episode_index"].astype(np.int64).tolist()
    if len(ids) != 617 or len(set(ids)) != 617:
        raise ValueError(f"accepted Core A dev prediction IDs invalid: {path}")
    return ids


def write_core_a_manifests(run_root: Path) -> dict[str, Any]:
    train, dev = read_core_a_population()
    dynamic = _feature_map(CORE_A_DYNAMIC_FEATURES)
    static = _feature_map(CORE_A_STATIC_FEATURES)
    expected_dev = set(dev["episode_index"].astype(int).tolist())
    order = _accepted_dev_order("A_DYNAMIC", 20260825)
    if set(order) != expected_dev:
        raise ValueError("accepted Core A dev prediction IDs differ from the frozen v3 primary dev population")
    if order != _accepted_dev_order("A_STATIC", 20260825):
        raise ValueError("accepted dynamic/static Core A dev row order differs")
    for name, mapping in (("dynamic", dynamic), ("static", static)):
        missing = expected_dev - mapping.keys()
        if missing:
            raise ValueError(f"{name} feature cache misses {len(missing)} primary dev IDs")
    replay_root = run_root / "replay/core_a"
    outputs: dict[str, Any] = {}
    for condition, mapping in (("dynamic", dynamic), ("static", static)):
        manifest = replay_root / f"{condition}_dev_manifest.csv"
        rows = [{"episode_index": episode, "feature_path": mapping[episode]} for episode in order]
        atomic_csv(manifest, ["episode_index", "feature_path"], rows)
        outputs[condition] = {
            "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
            "rows": len(rows), "episode_index_order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest(),
            "feature_manifest_source": str(CORE_A_DYNAMIC_FEATURES if condition == "dynamic" else CORE_A_STATIC_FEATURES),
        }
    # Fixed 32/32 train/validation bridge, ranked before any fresh feature or score read.
    bridge_rows: list[dict[str, Any]] = []
    for split, population, count in (("train", train, 32), ("validation", dev, 32)):
        values = []
        for row in population.to_dict(orient="records"):
            episode = int(row["episode_index"])
            identity = f"{DATASET_REVISION}|{row['split_assignment_key']}|{row['parquet_path']}|{episode}|{split}"
            values.append((key_hash(identity), identity, row))
        for digest, identity, row in sorted(values)[:count]:
            bridge_rows.append({
                "episode_index": int(row["episode_index"]),
                "video_path": str(row["video_path"]),
            })
    bridge = replay_root / "core_a_bridge_manifest.csv"
    atomic_csv(bridge, ["episode_index", "video_path"], bridge_rows)
    outputs["bridge"] = {
        "path": str(bridge), "sha256": sha256_file(bridge), "rows": len(bridge_rows),
        "selection": "first 32 by SHA-256 composite identity within train and validation, fixed before fresh feature extraction",
    }
    dev_by_episode = {int(row["episode_index"]): row for row in dev.to_dict(orient="records")}
    identity_rows = []
    for episode in order:
        row = dev_by_episode[int(episode)]
        source_key = str(row["split_assignment_key"])
        literal_source_key = str(row["parquet_path"])
        canonical_id = "|".join((DATASET_REVISION, source_key, literal_source_key, str(int(episode)), "dev"))
        identity_rows.append({
            "dataset_revision": DATASET_REVISION,
            "split": "dev",
            "source_key": source_key,
            "literal_source_key": literal_source_key,
            "episode_id": str(int(episode)),
            "video_path": str(row["video_path"]),
            "canonical_id": canonical_id,
        })
    identity_path = replay_root / "core_a_dev_identity.csv"
    atomic_csv(identity_path, list(identity_rows[0]), identity_rows)
    outputs["identity"] = {"path": str(identity_path), "sha256": sha256_file(identity_path), "rows": len(identity_rows)}
    return outputs


def _read_prefix(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(CORE_B_PREFIX_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"accepted V9 prefix missing required fields: {sorted(missing)}")
        # Read only the accepted label-free prefix; never read target archives.
        return {key: archive[key].copy() for key in CORE_B_PREFIX_FIELDS}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _build_core_b_identity_maps() -> tuple[dict[str, dict[str, str]], dict[int, dict[str, str]]]:
    # These V3 files contain split/source/anchor metadata only; no target values.
    allowlist = _read_csv_dicts(CORE_B_ALLOWLIST)
    allow_by_episode = {
        int(row["episode_index"]): row for row in allowlist
        if row["split"] in {"train", "dev"}
    }
    anchors = _read_csv_dicts(CORE_B_ANCHORS)
    anchor_by_id = {
        row["anchor_id"]: row for row in anchors
        if row["split"] in {"train", "dev"}
    }
    if not allow_by_episode or not anchor_by_id:
        raise ValueError("V3 train/dev identity metadata is empty")
    return anchor_by_id, allow_by_episode


def _canonical_id(episode: int, task: int, anchor_id: str, split: str, allow: dict[str, str], anchor: dict[str, str]) -> str:
    return "|".join((
        DATASET_REVISION,
        allow["split_assignment_key"],
        anchor["parquet_path"],
        str(episode),
        str(task),
        anchor_id,
        split,
    ))


def _enrich_prefix(path: Path, split: str, anchor_by_id: dict[str, dict[str, str]], allow_by_episode: dict[int, dict[str, str]]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    arrays = _read_prefix(path)
    ids = arrays["anchor_id"].astype(str)
    episodes = arrays["episode_id"].astype(np.int64)
    tasks = arrays["task_id"].astype(np.int64)
    roles = np.full(len(ids), split, dtype="U8")
    literal: list[str] = []
    source: list[str] = []
    canonical: list[str] = []
    meta_rows: list[dict[str, str]] = []
    for i, anchor_id in enumerate(ids.tolist()):
        episode = int(episodes[i]); task = int(tasks[i])
        if anchor_id not in anchor_by_id or episode not in allow_by_episode:
            raise ValueError(f"V3 train/dev metadata missing anchor identity: {anchor_id}")
        anchor = anchor_by_id[anchor_id]
        allow = allow_by_episode[episode]
        if int(anchor["episode_index"]) != episode or int(anchor["original_task_index"]) != task:
            raise ValueError(f"V3 anchor episode/task mismatch: {anchor_id}")
        if str(anchor["split"]) != allow["split"]:
            raise ValueError(f"V3 anchor/data split mismatch: {anchor_id}")
        if split == "train" and allow["split"] != "train":
            raise ValueError(f"non-train source in accepted train prefix: {anchor_id}")
        if split == "dev" and allow["split"] != "dev":
            raise ValueError(f"non-validation source in accepted dev prefix: {anchor_id}")
        if int(anchor["anchor_frame"]) != int(arrays["anchor_frame"][i]):
            raise ValueError(f"V3 anchor frame mismatch: {anchor_id}")
        canonical_id = _canonical_id(episode, task, anchor_id, split, allow, anchor)
        source.append(str(allow["split_assignment_key"]))
        literal.append(str(anchor["parquet_path"]))
        canonical.append(canonical_id)
        meta_rows.append({
            "dataset_role": split,
            "canonical_id": canonical_id,
            "source_key": str(allow["split_assignment_key"]),
            "literal_source_key": str(anchor["parquet_path"]),
            "episode_id": str(episode),
            "task_id": str(task),
            "anchor_id": anchor_id,
            "anchor_frame": str(int(anchor["anchor_frame"])),
            "cache_index": str(int(arrays["cache_index"][i])),
            "video_path": str(anchor["video_path"]),
            "parquet_path": str(anchor["parquet_path"]),
        })
    arrays.update({
        "dataset_revision": np.full(len(ids), DATASET_REVISION, dtype="U40"),
        "source_key": np.asarray(source, dtype="U128"),
        "literal_source_key": np.asarray(literal, dtype="U256"),
        "split": roles,
        "canonical_id": np.asarray(canonical, dtype="U768"),
    })
    return arrays, meta_rows


def _bridge_select(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for task in TASK_IDS:
        candidates = [row for row in rows if int(row["task_id"]) == task]
        quota = TASK_BRIDGE_QUOTAS[task]
        ordered = sorted(candidates, key=lambda row: key_hash(row["canonical_id"]))
        if len(ordered) < quota:
            raise ValueError(f"Core B {split} task {task} has only {len(ordered)} bridge candidates")
        chosen.extend(ordered[:quota])
    return chosen


def write_core_b_inputs(run_root: Path) -> dict[str, Any]:
    anchor_by_id, allow_by_episode = _build_core_b_identity_maps()
    replay_root = run_root / "replay/core_b"
    train_data, train_rows = _enrich_prefix(V9_TRAIN_PREFIX, "train", anchor_by_id, allow_by_episode)
    dev_data, dev_rows = _enrich_prefix(V9_DEV_PREFIX, "dev", anchor_by_id, allow_by_episode)
    train_path = replay_root / "TRAIN_PREFIXES_CANONICAL_NO_LABELS.npz"
    dev_path = replay_root / "DEV_PREFIXES_CANONICAL_NO_LABELS.npz"
    train_safe_path = replay_root / "TRAIN_PREFIXES_NO_LABELS.npz"
    dev_safe_path = replay_root / "DEV_PREFIXES_NO_LABELS.npz"
    atomic_npz(train_path, train_data)
    atomic_npz(dev_path, dev_data)
    # The predictor's path safety allowlist recognizes this exact suffix.
    atomic_npz(train_safe_path, train_data)
    atomic_npz(dev_safe_path, dev_data)
    if len(dev_rows) != 185 or len(train_rows) != 1280:
        raise ValueError(f"accepted V9 train/dev prefix support changed: {len(train_rows)}/{len(dev_rows)}")
    bridge_rows = _bridge_select(train_rows, "train") + _bridge_select(dev_rows, "dev")
    bridge_path = replay_root / "core_b_bridge_manifest.csv"
    atomic_csv(bridge_path, [
        "dataset_role", "canonical_id", "source_key", "literal_source_key",
        "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index",
        "video_path", "parquet_path",
    ], bridge_rows)
    dev_map_path = replay_root / "core_b_dev_identity.csv"
    atomic_csv(dev_map_path, list(dev_rows[0]), dev_rows)

    # Join final Core B anchor metadata to V3 path metadata with explicit keys.
    # This writes path/index metadata only; it does not open Parquet payload columns.
    final_rows = pd.read_csv(CORE_B_FINAL_POPULATION)
    split_metadata = pd.read_parquet(
        SPLIT_MANIFEST,
        columns=["episode_index", "split", "parquet_path", "video_path", "split_assignment_key"],
    )
    final_split = split_metadata[split_metadata["split"] == "final_test"].copy()
    final_split["episode_index"] = final_split["episode_index"].astype(np.int64)
    final_rows["episode_id"] = final_rows["episode_id"].astype(np.int64)
    merged = final_rows.merge(
        final_split,
        left_on=["episode_id", "literal_source_key", "source_key"],
        right_on=["episode_index", "parquet_path", "split_assignment_key"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(merged) != len(final_rows) or not (merged["_merge"] == "both").all():
        raise ValueError("Core B final video metadata failed exact source/episode join")
    map_rows = []
    for cache_index, row in enumerate(merged.to_dict(orient="records")):
        episode_id = int(row["episode_id"])
        task_id = int(row["task_id"])
        anchor_id = str(row["anchor_id"])
        source_key = str(row["source_key"])
        literal_source_key = str(row["literal_source_key"])
        map_rows.append({
            "dataset_role": "final_test",
            "canonical_id": "|".join((DATASET_REVISION, source_key, literal_source_key, str(episode_id), str(task_id), anchor_id, "final_test")),
            "source_key": source_key,
            "literal_source_key": literal_source_key,
            "episode_id": str(episode_id),
            "task_id": str(task_id),
            "anchor_id": anchor_id,
            "anchor_frame": str(int(row["anchor_frame"])),
            "cache_index": str(cache_index),
            "video_path": str(row["video_path"]),
            "parquet_path": literal_source_key,
        })
    final_video_map = OUT_ROOT / "protocol/CORE_B_FINAL_VIDEO_MAP.csv"
    atomic_csv(final_video_map, list(map_rows[0]), map_rows)

    return {
        "core_b_train_prefix": {"path": str(train_path), "sha256": sha256_file(train_path), "rows": len(train_rows)},
        "core_b_dev_prefix": {"path": str(dev_path), "sha256": sha256_file(dev_path), "rows": len(dev_rows)},
        "core_b_train_prefix_safe_basename": {"path": str(train_safe_path), "sha256": sha256_file(train_safe_path), "rows": len(train_rows)},
        "core_b_dev_prefix_safe_basename": {"path": str(dev_safe_path), "sha256": sha256_file(dev_safe_path), "rows": len(dev_rows)},
        "core_b_bridge": {"path": str(bridge_path), "sha256": sha256_file(bridge_path), "rows": len(bridge_rows), "task_quotas_per_split": TASK_BRIDGE_QUOTAS},
        "core_b_dev_identity": {"path": str(dev_map_path), "sha256": sha256_file(dev_map_path), "rows": len(dev_rows)},
        "core_b_final_video_map": {"path": str(final_video_map), "sha256": sha256_file(final_video_map), "rows": len(map_rows), "final_payload_read": False},
    }


def write_combined_final_population() -> dict[str, Any]:
    """Write the required 602+162 label-free final population index."""
    core_a = pd.read_csv(OUT_ROOT / "protocol/CORE_A_FINAL_POPULATION_METADATA.csv")
    core_b = pd.read_csv(CORE_B_FINAL_POPULATION)
    video_map = pd.read_csv(OUT_ROOT / "protocol/CORE_B_FINAL_VIDEO_MAP.csv")
    core_a_rows: list[dict[str, str]] = []
    for row in core_a.to_dict(orient="records"):
        episode = int(row["episode_index"])
        source_key = str(row["source_key"])
        literal = str(row["literal_source_key"])
        canonical = "|".join((str(row["dataset_revision"]), source_key, literal, str(episode), "final_test"))
        core_a_rows.append({
            "core": "A",
            "dataset_revision": str(row["dataset_revision"]),
            "split": "final_test",
            "dataset_role": "final_test",
            "source_key": source_key,
            "literal_source_key": literal,
            "episode_id": str(episode),
            "task_id": "",
            "anchor_id": "",
            "anchor_frame": "",
            "cache_index": "",
            "video_path": str(row["video_path"]),
            "parquet_path": str(row["parquet_path"]),
            "canonical_id": canonical,
            "video_sha256_fingerprint": str(row["video_sha256_fingerprint"]),
            "metadata_length": str(row["metadata_length"]),
            "eligible_4s": str(row["eligible_4s"]),
            "v3_benchmark_inclusion": str(row["v3_benchmark_inclusion"]),
        })
    merged = core_b.merge(
        video_map,
        on=["source_key", "literal_source_key", "episode_id", "task_id", "anchor_id", "anchor_frame"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_video"),
        indicator=True,
    )
    if len(merged) != len(core_b) or not (merged["_merge"] == "both").all():
        raise ValueError("Core B final population does not match the locked video map")
    core_b_rows: list[dict[str, str]] = []
    core_b_fields = [
        "dataset_revision", "split", "source_key", "literal_source_key", "episode_id", "task_id",
        "anchor_id", "anchor_frame", "history_frame_start", "history_frame_end",
        "target_h30_row_start_inclusive", "target_h30_row_end_inclusive", "structure_metadata_length",
        "v3_candidate_2s_anchor_count", "v3_primary_dev_cap", "history_video_frame_rule",
        "history_action_query_rule", "future_input_to_predictor", "eligibility_basis", "physical_action_clock_status",
    ]
    for row in merged.to_dict(orient="records"):
        record = {key: str(row[key]) for key in core_b_fields}
        record.update({
            "core": "B",
            "dataset_role": "final_test",
            "cache_index": str(row["cache_index"]),
            "video_path": str(row["video_path_video"] if "video_path_video" in row else row["video_path"]),
            "parquet_path": str(row["literal_source_key"]),
            "canonical_id": str(row["canonical_id"]),
        })
        core_b_rows.append(record)
    all_fields = [
        "core", "dataset_revision", "split", "dataset_role", "source_key", "literal_source_key",
        "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index", "video_path",
        "parquet_path", "canonical_id", "video_sha256_fingerprint", "metadata_length", "eligible_4s",
        "v3_benchmark_inclusion", "history_frame_start", "history_frame_end",
        "target_h30_row_start_inclusive", "target_h30_row_end_inclusive", "structure_metadata_length",
        "v3_candidate_2s_anchor_count", "v3_primary_dev_cap", "history_video_frame_rule",
        "history_action_query_rule", "future_input_to_predictor", "eligibility_basis", "physical_action_clock_status",
    ]
    combined = core_a_rows + core_b_rows
    if len(core_a_rows) != 602 or len(core_b_rows) != 162 or len({row["canonical_id"] for row in combined}) != 764:
        raise ValueError("combined final population counts or canonical IDs differ from the lock")
    path = OUT_ROOT / "protocol/FINAL_POPULATION_METADATA.csv"
    atomic_csv(path, all_fields, combined)
    core_a_identity_path = OUT_ROOT / "protocol/CORE_A_FINAL_IDENTITY.csv"
    core_a_identity = [{
        "dataset_revision": row["dataset_revision"], "split": "final_test", "source_key": row["source_key"],
        "literal_source_key": row["literal_source_key"], "episode_id": row["episode_id"],
        "video_path": row["video_path"], "canonical_id": row["canonical_id"],
    } for row in core_a_rows]
    atomic_csv(core_a_identity_path, list(core_a_identity[0]), core_a_identity)
    video_index_path = OUT_ROOT / "protocol/CORE_A_FINAL_VIDEO_INDEX.csv"
    core_a_video_index = [{"episode_index": int(row["episode_id"]), "video_path": row["video_path"]} for row in core_a_rows]
    atomic_csv(video_index_path, ["episode_index", "video_path"], core_a_video_index)
    return {
        "path": str(path), "sha256": sha256_file(path), "rows": len(combined),
        "core_a_rows": len(core_a_rows), "core_b_anchors": len(core_b_rows),
        "core_a_identity_path": str(core_a_identity_path), "core_a_identity_sha256": sha256_file(core_a_identity_path),
        "core_a_final_video_index_path": str(video_index_path), "core_a_final_video_index_sha256": sha256_file(video_index_path),
        "labels_or_targets_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z'))
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_absolute() or not str(run_root).startswith(str(PROJECT.resolve())):
        raise ValueError("run-root must remain inside the verified project")
    result = {
        "schema_version": 1,
        "status": "DEV_REPLAY_INPUTS_PREPARED",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_revision": DATASET_REVISION,
        "source_metadata_hashes": {
            "split_manifest": sha256_file(SPLIT_MANIFEST),
            "core_a_dynamic_feature_manifest": sha256_file(CORE_A_DYNAMIC_FEATURES),
            "core_a_static_feature_manifest": sha256_file(CORE_A_STATIC_FEATURES),
            "core_b_anchor_manifest": sha256_file(CORE_B_ANCHORS),
            "core_b_data_allowlist": sha256_file(CORE_B_ALLOWLIST),
            "v9_train_prefix": sha256_file(V9_TRAIN_PREFIX),
            "v9_dev_prefix": sha256_file(V9_DEV_PREFIX),
        },
        "core_a": write_core_a_manifests(run_root),
        "core_b": write_core_b_inputs(run_root),
    }
    result["final_population"] = write_combined_final_population()
    receipt = run_root / "replay/DEV_REPLAY_INPUTS_RECEIPT.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps({"status": result["status"], "receipt": str(receipt), "core_a": result["core_a"], "core_b": result["core_b"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
