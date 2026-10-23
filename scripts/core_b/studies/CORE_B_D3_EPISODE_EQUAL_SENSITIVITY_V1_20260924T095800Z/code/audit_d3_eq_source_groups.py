"""Verify source/duplicate-group identities for primary and dense dev panels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
SPLIT_MANIFEST = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
DENSE_INPUTS = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_hash(pairs: list[tuple[int, str]]) -> str:
    raw = "\n".join(f"{episode}\t{fingerprint}" for episode, fingerprint in sorted(pairs)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    split_table = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"])
    source_group_by_episode = {}
    for row in split_table.to_pylist():
        if row["split"] not in {"train", "validation"}:
            continue
        episode = int(row["episode_index"])
        fingerprint = str(row["video_sha256_fingerprint"] or f"episode:{episode}")
        previous = source_group_by_episode.get(episode)
        if previous is not None and previous != fingerprint:
            raise AssertionError(f"split manifest assigns multiple source groups to episode {episode}")
        source_group_by_episode[episode] = fingerprint

    anchor_rows = [
        row
        for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8"))
        if row["split"] == "dev" and row["population"] == "primary" and row["final_test_access"] == "0"
    ]
    primary_episodes = sorted({int(row["episode_index"]) for row in anchor_rows})
    primary_pairs = [(episode, source_group_by_episode[episode]) for episode in primary_episodes]
    dense = np.load(DENSE_INPUTS, allow_pickle=False)
    dense_episodes = sorted(set(dense["episode_index"].astype(np.int64).tolist()))
    dense_pairs = [(episode, source_group_by_episode[episode]) for episode in dense_episodes]
    primary_groups = {fingerprint for _, fingerprint in primary_pairs}
    dense_groups = {fingerprint for _, fingerprint in dense_pairs}
    if len(primary_episodes) != 603 or len(primary_groups) != 603:
        raise AssertionError("primary dev must map one-to-one to 603 source fingerprints")
    if len(dense_episodes) != 118 or len(dense_groups) != 118:
        raise AssertionError("dense panel must map one-to-one to 118 source fingerprints")
    if any(ep not in source_group_by_episode for ep in primary_episodes + dense_episodes):
        raise AssertionError("a dev episode lacks a locked source-group fingerprint")

    report = {
        "status": "PASS",
        "source_group_key": "video_sha256_fingerprint from duplicate-safe V3 split manifest; fallback episode key only if fingerprint absent",
        "primary_dev": {
            "episodes": len(primary_episodes),
            "source_fingerprint_groups": len(primary_groups),
            "multi_episode_source_groups": len(primary_episodes) - len(primary_groups),
            "episode_to_source_group_bijection": True,
            "episode_group_mapping_sha256": mapping_hash(primary_pairs),
        },
        "dense_panel": {
            "queries": 3718,
            "episodes": len(dense_episodes),
            "source_fingerprint_groups": len(dense_groups),
            "multi_episode_source_groups": len(dense_episodes) - len(dense_groups),
            "episode_to_source_group_bijection": True,
            "episode_group_mapping_sha256": mapping_hash(dense_pairs),
        },
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "anchor_manifest_sha256": sha256_file(V3_ROOT / "ANCHOR_MANIFEST.csv"),
        "dense_inputs_sha256": sha256_file(DENSE_INPUTS),
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_SOURCE_GROUP_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
