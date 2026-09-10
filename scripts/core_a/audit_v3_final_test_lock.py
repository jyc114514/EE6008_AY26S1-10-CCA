"""Audit the v3 final-test handoff without reading protected label values."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch

from ee6008.data.selection import benchmark_primary_mask

LABEL_COLUMNS = {
    "original_task_index",
    "contiguous_label_index",
    "classification_inclusion",
    "task_name_raw",
    "task_name_normalized",
    "category_raw",
    "category_normalized",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--label-free-index", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    split = pd.read_parquet(args.split_manifest)
    index = pd.read_parquet(args.label_free_index)
    feature_manifest_path = args.feature_run / "feature_manifest.parquet"
    feature_columns = parquet_columns(feature_manifest_path)
    label_manifest_columns = parquet_columns(args.label_manifest)
    index_label_columns = sorted(LABEL_COLUMNS.intersection(index.columns))
    feature_label_columns = sorted(LABEL_COLUMNS.intersection(feature_columns))
    final_split = split[split["split"].eq("final_test")]
    primary_final = final_split[benchmark_primary_mask(final_split)]
    primary_episode_ids = set(primary_final["episode_index"].astype(int))
    primary_index = index[index["episode_index"].isin(primary_episode_ids)].copy()
    primary_index_path = args.output_dir / "final_test_index_label_free_primary.parquet"

    sample_metadata_keys: list[str] = []
    feature_rows = pd.read_parquet(
        feature_manifest_path, columns=["episode_index", "feature_path"]
    )
    final_feature_rows = feature_rows[
        feature_rows["episode_index"].isin(index["episode_index"])
    ]
    if not final_feature_rows.empty:
        sample_path = Path(str(final_feature_rows.iloc[0]["feature_path"]))
        record = torch.load(sample_path, map_location="cpu", weights_only=True)
        sample_metadata_keys = sorted(record.get("metadata", {}).keys())
    historical_metadata_label_keys = sorted(
        set(sample_metadata_keys).intersection(LABEL_COLUMNS)
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    primary_index.to_parquet(primary_index_path, index=False)
    report: dict[str, Any] = {
        "status": (
            "protocol_deviation"
            if index_label_columns
            or feature_label_columns
            or historical_metadata_label_keys
            else "pass"
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "label_free_index": str(args.label_free_index),
        "label_free_index_sha256": sha256_file(args.label_free_index),
        "label_free_index_columns": list(index.columns),
        "label_free_index_label_columns": index_label_columns,
        "all_final_test_rows": len(final_split),
        "primary_final_test_rows": len(primary_final),
        "excluded_final_test_rows": int(len(final_split) - len(primary_final)),
        "primary_index_path": str(primary_index_path),
        "primary_index_sha256": sha256_file(primary_index_path),
        "protected_label_manifest": str(args.label_manifest),
        "protected_label_manifest_sha256": sha256_file(args.label_manifest),
        "protected_label_manifest_columns": label_manifest_columns,
        "feature_manifest": str(feature_manifest_path),
        "feature_manifest_columns": feature_columns,
        "feature_manifest_label_columns": feature_label_columns,
        "sample_historical_feature_metadata_keys": sample_metadata_keys,
        "historical_metadata_label_keys": historical_metadata_label_keys,
        "final_test_metric_access": "none",
        "final_test_label_values_read": False,
        "interpretation": (
            "The v3 label-free index and primary-only label-free index contain no label columns. "
            "The protected label manifest was inspected only at schema/hash level. The inherited "
            "feature cache is flagged if its manifest or sampled metadata remains label-bearing; "
            "this audit never reads, prints, aggregates or scores protected label values."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# V3 final-test handoff audit",
        "",
        f"Status: {report['status']}",
        f"All v3 final-test rows: {report['all_final_test_rows']}",
        f"Primary final-test rows: {report['primary_final_test_rows']}",
        f"Excluded rows: {report['excluded_final_test_rows']}",
        "",
        "The label-free index and primary-only index contain no task/category/label columns.",
        "The protected label manifest was checked only by schema and SHA-256; no label values were read.",
        f"Historical feature manifest label columns: {feature_label_columns}",
        f"Historical feature metadata label keys: {historical_metadata_label_keys}",
        "",
        "No final-test metrics, predictions, confusion matrix, class geometry or model-selection score was computed.",
    ]
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
