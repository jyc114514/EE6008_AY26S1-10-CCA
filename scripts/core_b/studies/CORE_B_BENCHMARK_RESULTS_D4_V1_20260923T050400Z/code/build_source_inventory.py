"""Reconcile sealed source artifacts, V3_STATE files, caches, and failure logs."""

from __future__ import annotations

import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from pathlibs import BENCH_ROOT, NEXT_ROOT, PROJECT_ROOT, STATE_ROOT

SOURCE_INV = BENCH_ROOT / "inventory/ARTIFACT_INVENTORY.csv"
STATE_SUMS = STATE_ROOT / "qa/CHECKSUMS.sha256"
STATE_REL = str(STATE_ROOT.relative_to(PROJECT_ROOT))
FIELDS = ["artifact_path", "source_root", "artifact_kind", "bytes", "sha256", "hash_status", "snapshot_or_failure_marker", "final_test_access"]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def classify(path: str) -> str:
    p = path.lower()
    name = Path(p).name
    if "failed_attempt" in p or "failure" in p or "oom" in name or "interrupt" in name or "retry" in name:
        return "failure_or_retry_record"
    if "/predictions/" in p or name.endswith(".npz") and "predict" in p:
        return "prediction_or_forecast"
    if "/cache/" in p or "/feature" in p or name.endswith(".npy"):
        return "feature_or_cache"
    if "/checkpoint" in p or name.endswith((".pt", ".pth", ".ckpt")):
        return "checkpoint_or_weight"
    if "/logs/" in p or name.endswith((".log", ".stdout.log", ".stderr.log")):
        return "failure_or_execution_log"
    if name.endswith((".csv", ".json", ".md", ".ini", ".yaml", ".yml", ".sh", ".py")):
        return "contract_or_metadata"
    return "other"


def main():
    original = [{**row, "hash_status": "BENCH_ROOT_artifact_inventory_record"} for row in read_csv(SOURCE_INV)]
    seen = {row["artifact_path"] for row in original}
    state_rows = []
    listed_relatives = set()
    unmanifested_rows = []
    checksum_lines = STATE_SUMS.read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        relative = relative.removeprefix("./")
        listed_relatives.add(relative)
        path = STATE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"verified V3_STATE manifest path missing: {relative}")
        state_rows.append({"artifact_path": f"{STATE_REL}/{relative}", "source_root": STATE_REL, "artifact_kind": classify(relative), "bytes": path.stat().st_size, "sha256": digest, "hash_status": "listed_in_verified_V3_STATE_CHECKSUMS", "snapshot_or_failure_marker": int("failed_attempt" in relative.lower() or "failed_attempts/" in relative.lower()), "final_test_access": 0})
    actual_relatives = {str(p.relative_to(STATE_ROOT)) for p in STATE_ROOT.rglob("*") if p.is_file() and p != STATE_SUMS}
    unmanifested_paths = sorted(actual_relatives - listed_relatives)
    for relative in unmanifested_paths:
        path = STATE_ROOT / relative
        digest = file_sha256(path)
        row = {"artifact_path": f"{STATE_REL}/{relative}", "source_root": STATE_REL, "artifact_kind": classify(relative), "bytes": path.stat().st_size, "sha256": digest, "hash_status": "unmanifested_hash_computed_in_NEXT_ROOT", "snapshot_or_failure_marker": 0, "final_test_access": 0}
        state_rows.append(row)
        unmanifested_rows.append({"artifact_path": row["artifact_path"], "bytes": row["bytes"], "sha256": digest, "hash_status": row["hash_status"], "artifact_kind": row["artifact_kind"], "final_test_access": 0})
    if len(state_rows) != len(actual_relatives):
        raise ValueError(f"V3_STATE reconciled artifact rows={len(state_rows)} vs actual files excluding checksum manifest={len(actual_relatives)}")
    write_csv(NEXT_ROOT / "inventory/V3_STATE_CHECKSUM_GAPS.csv", unmanifested_rows, ["artifact_path", "bytes", "sha256", "hash_status", "artifact_kind", "final_test_access"])
    for row in state_rows:
        if row["artifact_path"] in seen:
            raise ValueError(f"duplicate source artifact path in reconciled inventory: {row['artifact_path']}")
    combined = original + state_rows
    write_csv(NEXT_ROOT / "inventory/ARTIFACT_INVENTORY_RECONCILED.csv", combined, FIELDS)
    write_csv(NEXT_ROOT / "inventory/V3_STATE_ARTIFACT_INVENTORY.csv", state_rows, FIELDS)

    failures = []
    for row in original:
        marker = str(row["snapshot_or_failure_marker"]) == "1"
        artifact_kind = row["artifact_kind"]
        path = row["artifact_path"]
        if marker or artifact_kind == "failure_or_execution_log" or classify(path) == "failure_or_retry_record":
            if marker:
                category = "explicit_failure_or_snapshot_marker"
            elif classify(path) == "failure_or_retry_record":
                category = "retry_or_failure_named_artifact_status_requires_source_marker"
            else:
                category = "execution_log_not_assumed_failed"
            failures.append({"artifact_path": path, "source_root": row["source_root"], "artifact_kind": artifact_kind, "bytes": row["bytes"], "sha256": row["sha256"], "snapshot_or_failure_marker": row["snapshot_or_failure_marker"], "classification": category, "final_test_access": row["final_test_access"]})
    for row in state_rows:
        if row["artifact_kind"] in {"failure_or_execution_log", "failure_or_retry_record"}:
            failures.append({"artifact_path": row["artifact_path"], "source_root": row["source_root"], "artifact_kind": row["artifact_kind"], "bytes": row["bytes"], "sha256": row["sha256"], "snapshot_or_failure_marker": row["snapshot_or_failure_marker"], "classification": "V3_STATE_execution_or_retry_record_status_not_inferred_from_filename", "final_test_access": 0})
    write_csv(NEXT_ROOT / "inventory/RUN_FAILURE_REGISTER.csv", failures, ["artifact_path", "source_root", "artifact_kind", "bytes", "sha256", "snapshot_or_failure_marker", "classification", "final_test_access"])

    summary = []
    grouped = defaultdict(list)
    for row in combined:
        grouped[(row["source_root"], row["artifact_kind"])].append(row)
    for (root, kind), values in sorted(grouped.items()):
        summary.append({"source_root": root, "artifact_kind": kind, "artifact_count": len(values), "total_bytes": sum(int(row["bytes"]) for row in values), "explicit_marker_count": sum(str(row["snapshot_or_failure_marker"]) == "1" for row in values), "final_test_access_max": max(int(row["final_test_access"]) for row in values)})
    write_csv(NEXT_ROOT / "inventory/ARTIFACT_KIND_SUMMARY.csv", summary, ["source_root", "artifact_kind", "artifact_count", "total_bytes", "explicit_marker_count", "final_test_access_max"])
    metadata = {"bench_root_artifact_rows": len(original), "bench_root_artifact_kinds": dict(Counter(row["artifact_kind"] for row in original)), "v3_state_checksum_rows": len(state_rows), "v3_state_checksum_manifest": str(STATE_SUMS), "v3_state_checksum_verified_before_inventory": True, "combined_rows": len(combined), "failure_and_execution_log_register_rows": len(failures), "classification_note": "failure_or_execution_log entries are retained; a log filename does not prove failure unless an explicit marker/status says so", "final_test_access": 0}
    metadata["v3_state_checksum_manifest_entries"] = len(listed_relatives)
    metadata["v3_state_listed_checksum_entries_verified"] = True
    metadata["v3_state_checksum_manifest_complete_for_all_files"] = len(unmanifested_rows) == 0
    metadata["v3_state_unmanifested_paths_are_generated_python_bytecode"] = all("__pycache__" in row["artifact_path"] for row in unmanifested_rows)
    metadata["v3_state_actual_files_excluding_manifest"] = len(actual_relatives)
    metadata["v3_state_unmanifested_count"] = len(unmanifested_rows)
    metadata["v3_state_unmanifested_paths"] = [row["artifact_path"] for row in unmanifested_rows]
    metadata["v3_state_manifest_complete"] = len(unmanifested_rows) == 0
    (NEXT_ROOT / "inventory/ARTIFACT_INVENTORY_SUMMARY.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
