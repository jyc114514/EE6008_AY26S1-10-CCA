"""Hash only the primary train/dev action files and their exact SHA groups."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement"
RAW_ROOT = Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["raw_root"]).resolve(strict=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest_rows = {}
    with (V3 / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in {"train", "dev"} and row["population"] == "primary" and row["final_test_access"] == "0":
                manifest_rows[(row["split"], int(row["episode_index"]))] = row
    train_ids = {episode for split, episode in manifest_rows if split == "train"}
    train_source = pq.read_table(
        Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["split_manifest"]),
        columns=["episode_index", "split", "video_sha256_fingerprint", "v3_benchmark_inclusion"],
        filters=[("split", "=", "train"), ("v3_benchmark_inclusion", "=", "primary")],
    ).to_pylist()
    group_by_episode = {int(row["episode_index"]): str(row["video_sha256_fingerprint"]) for row in train_source if int(row["episode_index"]) in train_ids}
    if train_ids - set(group_by_episode):
        raise AssertionError("train source hashes do not cover primary train episodes")
    with (RESULT / "results/D1_SOURCE_GROUP_QUERY_TABLE.csv").open(newline="", encoding="utf-8") as handle:
        dev_source = {int(row["episode_index"]): str(row["source_group_id"]) for row in csv.DictReader(handle)}
    dev_ids = {episode for split, episode in manifest_rows if split == "dev"}
    if dev_ids - set(dev_source):
        raise AssertionError("D1 exact-video SHA table does not cover V3 primary dev episodes")

    output = []
    for (split, episode), row in sorted(manifest_rows.items(), key=lambda item: (item[0][0], item[0][1])):
        path = (RAW_ROOT / row["parquet_path"]).resolve(strict=True)
        if not path.is_relative_to(RAW_ROOT):
            raise AssertionError("raw action path escaped the immutable dataset root")
        group = group_by_episode[episode] if split == "train" else dev_source[episode]
        stat = path.stat()
        output.append({
            "split": split,
            "episode_index": episode,
            "task": int(row["original_task_index"]),
            "source_group_id": group,
            "parquet_relative_path": row["parquet_path"],
            "bytes": stat.st_size,
            "sha256": digest(path),
            "final_test_access": 0,
        })
    counts = Counter(row["split"] for row in output)
    out_path = OUT / "D2_V3_RAW_ACTION_INPUT_HASHES.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "status": "PASS",
        "primary_train_episodes_hashed": counts["train"],
        "primary_dev_episodes_hashed": counts["dev"],
        "raw_action_files_hashed": len(output),
        "unique_source_groups_train": len({row["source_group_id"] for row in output if row["split"] == "train"}),
        "unique_source_groups_dev": len({row["source_group_id"] for row in output if row["split"] == "dev"}),
        "heldout_or_final_paths_included": False,
        "raw_root_writes": 0,
        "final_test_access": 0,
    }
    (OUT / "D2_V3_RAW_ACTION_HASH_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
