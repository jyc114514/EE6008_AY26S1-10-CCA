"""Audit the fixed D4 pair table's source-fingerprint IDs read-only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
PAIR_PATH = EQ_ROOT / "protocol/D3_EQ_D4_PAIR_MANIFEST_INPUT_ONLY.csv"
SPLIT_MANIFEST = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    group_rows = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"]).to_pylist()
    source_group = {
        int(row["episode_index"]): str(row["video_sha256_fingerprint"] or f"episode:{row['episode_index']}")
        for row in group_rows if row["split"] in {"train", "validation"}
    }
    pairs = list(csv.DictReader(PAIR_PATH.open(newline="", encoding="utf-8")))
    mismatch = [row["pair_id"] for row in pairs if source_group.get(int(row["episode_index"])) != row["source_group_id"]]
    horizon_counts = {str(h): sum(int(row["horizon_steps"]) == h for row in pairs) for h in (30, 60)}
    groups = {row["source_group_id"] for row in pairs}
    if mismatch or len(pairs) != 7200 or horizon_counts != {"30": 3600, "60": 3600} or len(groups) != 118:
        raise AssertionError(f"D4 pair source-group audit failed: mismatches={len(mismatch)}")
    report = {
        "status": "PASS",
        "pair_rows": len(pairs),
        "horizon_counts": horizon_counts,
        "source_group_count": len(groups),
        "source_group_id_mismatches": len(mismatch),
        "source_group_key": "video_sha256_fingerprint from duplicate-safe split manifest",
        "pair_manifest_sha256": sha256_file(PAIR_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_D4_SOURCE_GROUP_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
