#!/usr/bin/env python3
"""Build the fixed label-free 185-row Core B dev inputs for CURRENT_PROFILE_V2.

This reads only the already accepted dev prefix arrays and source/anchor
metadata. It deliberately never indexes ``targets_native`` from V3_MAIN_INPUTS.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
V9_DEV_PREFIX = PROJECT / "reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/data/DEV_PREFIXES_NO_LABELS.npz"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
ALLOWLIST = V3 / "DATA_ALLOWLIST.csv"
ANCHORS = V3 / "ANCHOR_MANIFEST.csv"
Q0_MANIFEST = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
DATASET_REVISION = "6b6b599def75e3558b2736062e79bdbeec6cd224"
TASKS = (119, 94, 90, 141, 230)
CSV_FIELDS = (
    "dataset_role", "canonical_id", "source_key", "literal_source_key",
    "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index",
    "video_path", "parquet_path",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty dev feature manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main() -> int:
    if not all(path.is_file() for path in (V9_DEV_PREFIX, ALLOWLIST, ANCHORS, Q0_MANIFEST)):
        raise FileNotFoundError("one or more accepted dev identity sources are missing")
    prefix_sha = sha(V9_DEV_PREFIX)
    expected_prefix_sha = "8cc74fef6d794320b33bbb6d43c0a93230b21373af1c2b910cb74d1057d09090"
    if prefix_sha != expected_prefix_sha:
        raise ValueError(f"accepted no-label dev prefix hash changed: {prefix_sha}")
    q0_sha = sha(Q0_MANIFEST)
    expected_q0_sha = "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee"
    if q0_sha != expected_q0_sha:
        raise ValueError(f"accepted Q0 cache manifest hash changed: {q0_sha}")

    allow_rows = list(csv.DictReader(ALLOWLIST.open(encoding="utf-8-sig", newline="")))
    allow = {int(r["episode_index"]): r for r in allow_rows if r["split"] == "dev"}
    if len(allow) != len([r for r in allow_rows if r["split"] == "dev"]):
        raise ValueError("duplicate dev episode in V3 allowlist")
    anchor_rows = list(csv.DictReader(ANCHORS.open(encoding="utf-8-sig", newline="")))
    anchor_by_id = {r["anchor_id"]: r for r in anchor_rows if r["split"] == "dev"}
    if len(anchor_by_id) != len([r for r in anchor_rows if r["split"] == "dev"]):
        raise ValueError("duplicate dev anchor in V3 anchor manifest")

    with np.load(V9_DEV_PREFIX, allow_pickle=False) as src:
        required = {
            "anchor_id", "episode_id", "task_id", "task_rank", "source_cluster",
            "cache_index", "anchor_frame", "history_native", "history_mask",
            "relative_delta", "anchor_native",
        }
        if not required.issubset(src.files):
            raise ValueError(f"accepted prefix schema missing {sorted(required-set(src.files))}")
        arrays = {key: src[key].copy() for key in sorted(required)}

    ids = arrays["anchor_id"].astype(str)
    episodes = arrays["episode_id"].astype(np.int64)
    tasks = arrays["task_id"].astype(np.int64)
    cache_idx = arrays["cache_index"].astype(np.int64)
    frames = arrays["anchor_frame"].astype(np.int64)
    n = len(ids)
    if n != 185 or len(set(ids.tolist())) != n or len(set(cache_idx.tolist())) != n:
        raise ValueError("accepted Core B dev IDs/cache indices must be 185 unique rows")
    if arrays["history_native"].shape != (n, 32, 71) or arrays["history_native"].dtype != np.float32:
        raise ValueError("dev state/history tensor contract changed")
    if arrays["history_mask"].shape != (n, 32) or arrays["history_mask"].dtype != np.float32:
        raise ValueError("dev history-mask tensor contract changed")
    if arrays["relative_delta"].shape != (n, 32) or arrays["relative_delta"].dtype != np.float32:
        raise ValueError("dev relative-delta tensor contract changed")
    if arrays["anchor_native"].shape != (n, 28) or arrays["anchor_native"].dtype != np.float32:
        raise ValueError("dev anchor command contract changed")
    if not all(np.isfinite(arrays[k]).all() for k in ("history_native", "history_mask", "relative_delta", "anchor_native")):
        raise ValueError("dev no-label prefix has nonfinite values")

    # Resolve cache indexes using identity-only Q0 shard sidecars. This avoids
    # reading the mixed-target V3_MAIN_INPUTS NPZ.
    import json as _json
    q0 = _json.loads(Q0_MANIFEST.read_text(encoding="utf-8"))
    if q0.get("status") != "PASS" or int(q0.get("final_test_access", -1)) != 0:
        raise ValueError("accepted Q0 manifest is not a development-only PASS")
    cache_identity: dict[int, tuple[str, int]] = {}
    for shard in q0.get("shards", []):
        meta_path = Path(shard["meta"])
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        start, end = int(meta["start_row"]), int(meta["end_row"])
        shard_ids = [str(v) for v in meta["anchor_ids"]]
        shard_episodes = [int(v) for v in meta["episode_index"]]
        if end - start != len(shard_ids) or len(shard_ids) != len(shard_episodes):
            raise ValueError(f"Q0 identity sidecar length mismatch: {meta_path}")
        for offset, (sid, episode) in enumerate(zip(shard_ids, shard_episodes, strict=True)):
            index = start + offset
            if index in cache_identity:
                raise ValueError(f"duplicate Q0 cache identity index: {index}")
            cache_identity[index] = (sid, episode)

    manifest_rows: list[dict[str, str]] = []
    canonical_ids: list[str] = []
    for i, anchor_id in enumerate(ids.tolist()):
        episode, task, row_index = int(episodes[i]), int(tasks[i]), int(cache_idx[i])
        if row_index not in cache_identity or cache_identity[row_index] != (anchor_id, episode):
            raise ValueError(f"Q0 cache index/anchor identity mismatch: {anchor_id}/{row_index}")
        if episode not in allow or anchor_id not in anchor_by_id:
            raise ValueError(f"dev allowlist/anchor row missing: {anchor_id}")
        arow, erow = anchor_by_id[anchor_id], allow[episode]
        if int(arow["episode_index"]) != episode or int(arow["original_task_index"]) != task:
            raise ValueError(f"V3 anchor metadata identity mismatch: {anchor_id}")
        if int(arow["anchor_frame"]) != int(frames[i]) or int(arow["anchor_frame"]) < 119:
            raise ValueError(f"V3 anchor frame violates prefix contract: {anchor_id}")
        if int(erow["original_task_index"]) != task:
            raise ValueError(f"V3 allowlist task mismatch: {anchor_id}")
        video_path = str(arow["video_path"])
        raw_root = RAW_ROOT.resolve()
        absolute_video = (raw_root / video_path).resolve()
        if not absolute_video.is_relative_to(raw_root) or not absolute_video.is_file():
            raise FileNotFoundError(f"dev video path is absent or escapes the G1 root: {anchor_id}")
        source_key = str(erow["split_assignment_key"])
        literal_source_key = str(arow["parquet_path"])
        canonical_id = "|".join((DATASET_REVISION, source_key, literal_source_key,
                                 str(episode), str(task), anchor_id, "dev"))
        canonical_ids.append(canonical_id)
        manifest_rows.append({
            "dataset_role": "dev", "canonical_id": canonical_id,
            "source_key": source_key, "literal_source_key": literal_source_key,
            "episode_id": str(episode), "task_id": str(task), "anchor_id": anchor_id,
            "anchor_frame": str(int(frames[i])), "cache_index": str(row_index),
            "video_path": video_path, "parquet_path": literal_source_key,
        })

    if set(tasks.tolist()) != set(TASKS) or len(set(int(x) for x in episodes)) != 47:
        raise ValueError("dev support differs from the frozen 185-anchor / 47-episode / five-task population")
    if len(set(canonical_ids)) != n:
        raise ValueError("canonical dev identities are not unique")

    prefix_out = RUN / "replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
    manifest_out = OUT / "core_b/DEV_CURRENT_PROFILE_INPUTS.csv"
    receipt_out = OUT / "core_b/DEV_CURRENT_PROFILE_INPUTS_RECEIPT.json"
    if prefix_out.exists() or manifest_out.exists() or receipt_out.exists():
        raise FileExistsError("current-profile dev input outputs already exist; refusing overwrite")
    arrays["anchor_id"] = ids.astype("U64")
    arrays["episode_id"] = episodes
    arrays["task_id"] = tasks
    arrays["cache_index"] = cache_idx
    arrays["anchor_frame"] = frames
    arrays["canonical_id"] = np.asarray(canonical_ids, dtype="U768")
    arrays["dataset_revision"] = np.full(n, DATASET_REVISION, dtype="U64")
    arrays["source_key"] = np.asarray([r["source_key"] for r in manifest_rows], dtype="U256")
    arrays["literal_source_key"] = np.asarray([r["literal_source_key"] for r in manifest_rows], dtype="U512")
    arrays["split"] = np.full(n, "dev", dtype="U8")
    prefix_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = prefix_out.with_name(f".{prefix_out.name}.{os.getpid()}.part")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, prefix_out)
    atomic_csv(manifest_out, manifest_rows)
    receipt = {
        "schema_version": 1,
        "status": "PASS_LABEL_FREE_CURRENT_PROFILE_DEV_INPUTS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": n,
        "episode_support": len(set(int(x) for x in episodes)),
        "task_support": {str(t): int(np.sum(tasks == t)) for t in TASKS},
        "canonical_id_order_sha256": hashlib.sha256("\n".join(canonical_ids).encode()).hexdigest(),
        "anchor_id_order_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "source_prefix_path": str(V9_DEV_PREFIX),
        "source_prefix_sha256": prefix_sha,
        "v3_identity_sources": {
            "allowlist_path": str(ALLOWLIST), "allowlist_sha256": sha(ALLOWLIST),
            "anchor_manifest_path": str(ANCHORS), "anchor_manifest_sha256": sha(ANCHORS),
            "q0_manifest_path": str(Q0_MANIFEST), "q0_manifest_sha256": q0_sha,
            "q0_identity_sidecars_loaded": len(q0["shards"]),
            "q0_feature_arrays_loaded": False,
        },
        "outputs": {
            "prefix_path": str(prefix_out), "prefix_sha256": sha(prefix_out),
            "input_manifest_path": str(manifest_out), "input_manifest_sha256": sha(manifest_out),
        },
        "future_targets_materialized": False,
        "final_payload_read": False,
    }
    atomic_json(receipt_out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
