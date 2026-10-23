"""Hash exactly the primary train/dev action Parquet files used for D2_V3."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement"
RAW_ROOT = Path(json.loads((V3 / "SOURCE_REGISTRY.json").read_text())["raw_root"]).resolve(strict=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = {}
    with (V3 / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in {"train", "dev"} or row["population"] != "primary" or row["final_test_access"] != "0":
                continue
            key = (row["split"], row["episode_index"])
            rows[key] = row
    output = []
    for (split, episode), row in sorted(rows.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        path = (RAW_ROOT / row["parquet_path"]).resolve(strict=True)
        if not path.is_relative_to(RAW_ROOT):
            raise AssertionError("raw action path escaped the immutable dataset root")
        stat = path.stat()
        output.append({
            "split": split,
            "episode_index": int(episode),
            "source_group_id": "",
            "parquet_relative_path": row["parquet_path"],
            "bytes": stat.st_size,
            "sha256": digest(path),
            "final_test_access": 0,
        })
    counts = Counter(row["split"] for row in output)
    output_path = OUT / "D2_V3_RAW_ACTION_INPUT_HASHES.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "status": "PASS",
        "primary_train_episodes_hashed": counts["train"],
        "primary_dev_episodes_hashed": counts["dev"],
        "raw_action_files_hashed": len(output),
        "final_test_or_heldout_paths_included": False,
        "output_csv": str(output_path),
    }
    (OUT / "D2_V3_RAW_ACTION_HASH_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
