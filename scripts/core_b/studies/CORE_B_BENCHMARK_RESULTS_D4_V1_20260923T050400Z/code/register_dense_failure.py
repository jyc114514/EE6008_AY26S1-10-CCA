"""Register the recovered optional dense-inference failure without losing evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / "inventory/RUN_FAILURE_REGISTER.csv"
SUMMARY = ROOT / "inventory/ARTIFACT_INVENTORY_SUMMARY.json"
ATTEMPT = ROOT / "d3_extension/invalid_runs/D3_SIV_MISSING_FEATURES_ATTEMPT_001/FAILURE_RECORD.json"
LOG = ROOT / "d3_extension/invalid_runs/D3_SIV_MISSING_FEATURES_ATTEMPT_001/child.log"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with REGISTER.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "artifact_path" not in fields or "classification" not in fields:
        raise ValueError("RUN_FAILURE_REGISTER schema mismatch")
    prefix = "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/d3_extension/invalid_runs/D3_SIV_MISSING_FEATURES_ATTEMPT_001"
    additions = []
    for path, kind, classification in (
        (ATTEMPT, "failure_or_retry_record", "optional dense inference failed before prediction bundle creation; recovered without data or model changes"),
        (LOG, "failure_or_execution_log", "V3_STATE dense child traceback preserved; no output bundle was written"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = f"{prefix}/{path.name}"
        if any(row["artifact_path"] == rel for row in rows):
            continue
        additions.append({
            "artifact_path": rel, "source_root": "NEXT_ROOT_optional_dense_extension",
            "artifact_kind": kind, "bytes": path.stat().st_size, "sha256": sha(path),
            "snapshot_or_failure_marker": 1, "classification": classification, "final_test_access": 0,
        })
    if additions:
        backup = REGISTER.with_name(f"RUN_FAILURE_REGISTER.csv.bak_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(REGISTER, backup)
        with REGISTER.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerows(additions)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["next_root_dense_extension_failure_entries"] = sum(
        row["source_root"] == "NEXT_ROOT_optional_dense_extension" for row in rows + additions
    )
    summary["failure_and_execution_log_register_rows"] = len(rows) + len(additions)
    summary["dense_extension_failure_note"] = (
        "The V3_STATE dense feature-key mismatch is retained under invalid_runs; its corrected retry completed with 54/54 predictions and rolling diagnostics."
    )
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qa = {
        "status": "PASS", "new_rows": len(additions),
        "registered_optional_dense_failure_entries": summary["next_root_dense_extension_failure_entries"],
        "register_rows": len(rows) + len(additions), "final_test_access": 0,
    }
    (ROOT / "qa/RUN_FAILURE_REGISTER_UPDATE_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
