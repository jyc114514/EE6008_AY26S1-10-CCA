#!/usr/bin/env python3
"""Seal all available dev predictions before the scorer reads any dev targets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    if seal_path.exists():
        raise FileExistsError(f"prediction seal already exists; refusing overwrite: {seal_path}")
    if (ROOT / "data/DEV_TARGETS_SCORER_ONLY.npz").exists():
        raise RuntimeError("dev scorer-only target artifact exists before prediction seal")
    selection_path = ROOT / "protocol/LR_EPOCH_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") not in {"LOCKED", "LOCKED_WITH_PARTIAL_SEARCH"}:
        raise RuntimeError("inner-tune LR/epoch selection is not locked")
    queue_path = ROOT / "coordination/queues/FORMAL_DEV.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in queue["jobs"]:
        watcher_path = Path(raw)
        watcher = json.loads(watcher_path.read_text(encoding="utf-8"))
        output_root = Path(watcher["output_root"])
        complete_path = output_root / "PREDICTION_COMPLETE.json"
        prediction_path = output_root / "PREDICTIONS.safetensors"
        ids_path = output_root / "PREDICTION_ROWS.csv"
        if not all(path.is_file() for path in (complete_path, prediction_path, ids_path)):
            missing.append(str(watcher["job_id"]))
            continue
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS":
            missing.append(str(watcher["job_id"]))
            continue
        if complete.get("future_target_arrays_requested") is not False or int(complete.get("final_heldout_payload_access", -1)) != 0:
            raise RuntimeError(f"prediction job violated target/final lock: {watcher['job_id']}")
        if sha_file(prediction_path) != complete.get("prediction_sha256") or sha_file(ids_path) != complete.get("row_ids_sha256"):
            raise RuntimeError(f"prediction file checksum mismatch: {watcher['job_id']}")
        entries.append({
            "job_id": watcher["job_id"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": complete["prediction_sha256"],
            "rows_path": str(ids_path),
            "rows_sha256": complete["row_ids_sha256"],
            "rows": int(complete["rows"]),
            "model_id": complete["model_id"],
            "seed": int(complete["seed"]),
            "model_sha256": complete["model_sha256"],
            "model_state_hash": complete["model_state_hash"],
        })
    if not entries:
        raise RuntimeError("refusing to seal a zero-prediction batch; scorer must not open dev targets")
    status = "SEALED_BEFORE_TARGET_ACCESS" if not missing else "SEALED_WITH_MISSING_PREDICTIONS_BEFORE_TARGET_ACCESS"
    seal = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "protocol_id": "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013",
        "selection_lock_sha256": sha_file(selection_path),
        "expected_inference_jobs": len(queue["jobs"]),
        "sealed_prediction_jobs": len(entries),
        "missing_prediction_jobs": missing,
        "prediction_entries": sorted(entries, key=lambda row: row["job_id"]),
        "dev_target_labels_read_before_seal": False,
        "final_heldout_payload_access": 0,
    }
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: seal[k] for k in ("status", "expected_inference_jobs", "sealed_prediction_jobs", "missing_prediction_jobs", "dev_target_labels_read_before_seal", "final_heldout_payload_access")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
