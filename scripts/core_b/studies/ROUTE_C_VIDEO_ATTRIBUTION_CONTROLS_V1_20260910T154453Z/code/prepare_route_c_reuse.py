#!/usr/bin/env python3
"""Materialize the four accepted chronological P0 cells as exact reuse."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
FORMAL = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T070136Z_FORMAL_B3_8OF8"
INPUT = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/inputs/B2_INPUTS_0125.npz"
OUT = ROOT / "predictions"
MAPPING = {"B3-FUS-D-F": "RC-FUS-F-P0", "B3-FUS-D-L": "RC-FUS-L-P0", "B3-V-D-F": "RC-V-F-P0", "B3-V-D-L": "RC-V-L-P0"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(part, path)


def main() -> None:
    inventory = json.loads((FORMAL / "CORE_B_LOGICAL_CELL_INVENTORY.json").read_text())["rows"]
    sources = {row["cell"]: Path(row["prediction_path"]) for row in inventory if row.get("stage") in {"B1", "B2", "B3"} and str(row.get("status", "")).startswith("accepted")}
    with np.load(INPUT, allow_pickle=False) as archive:
        dev = archive["split_code"] == 1
        canonical_ids = np.asarray(archive["episode_index"][dev], dtype=np.int64)
        canonical_tasks = np.asarray(archive["original_task_index"][dev], dtype=np.int64)
    task_lookup = {int(episode): int(task) for episode, task in zip(canonical_ids, canonical_tasks)}
    records = []
    for model, condition in MAPPING.items():
        source_cell = {"B3-FUS-D-F": "B2-FUS-0125", "B3-FUS-D-L": "B3-FUS-D-L", "B3-V-D-F": "B2-V-0125", "B3-V-D-L": "B3-V-D-L"}[model]
        source_path = sources[source_cell]
        with np.load(source_path, allow_pickle=False) as source:
            ids = np.asarray(source["episode_index"], dtype=np.int64)
            order = np.asarray([int(np.flatnonzero(canonical_ids == episode)[0]) for episode in ids], dtype=np.int64)
            if len(ids) != 614 or len(np.unique(ids)) != 614:
                raise ValueError(f"invalid P0 source {model}")
            arrays = {key: source[key] for key in source.files if key in {"episode_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native"}}
            arrays["original_task_index"] = canonical_tasks[order]
        target = OUT / condition
        target.mkdir(parents=True, exist_ok=True)
        prediction_path = target / "predictions.npz"
        if prediction_path.exists():
            raise FileExistsError(prediction_path)
        part = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.part.npz")
        np.savez_compressed(part, **arrays); os.replace(part, prediction_path)
        atomic_json(target / "RUN_CONFIG_LOCKED.json", {"protocol_id": "core_b_route_c_video_attribution_controls_v1_20260910", "model": model, "condition": "P0", "status": "reused", "source_cell": source_cell, "source_prediction_path": str(source_path), "source_prediction_sha256": sha256(source_path), "prediction_sha256": sha256(prediction_path), "optimizer_steps": 0, "final_test_accessed": 0})
        atomic_json(target / "INTERVENTION_AUDIT.json", {"condition": "P0", "exact_reuse": True, "source_cell": source_cell, "pixel_transformation": "identity", "optimizer_steps": 0, "final_test_accessed": 0, "recipient_ids": ids.tolist()})
        atomic_json(target / "RUN_COMPLETE.json", {"status": "reused", "model": model, "condition": "P0", "source_cell": source_cell, "optimizer_steps": 0, "final_test_accessed": 0})
        records.append({"model": model, "condition": condition, "source_cell": source_cell, "source_path": str(source_path), "prediction_path": str(prediction_path), "source_sha256": sha256(source_path), "prediction_sha256": sha256(prediction_path), "N": len(ids), "status": "reused"})
    (ROOT / "ROUTE_C_P0_REUSE_RECORD.json").write_text(json.dumps({"rows": records, "count": len(records), "status": "PASS"}, indent=2) + "\n")
    with (ROOT / "ROUTE_C_RUN_ACCEPTANCE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "condition", "source_cell", "N", "status", "optimizer_steps", "final_test_access"]); writer.writeheader(); writer.writerows({"model": row["model"], "condition": "P0", "source_cell": row["source_cell"], "N": row["N"], "status": "reused", "optimizer_steps": 0, "final_test_access": 0} for row in records)
    print(json.dumps({"status": "PASS", "reused": len(records), "N_each": 614}))


if __name__ == "__main__":
    main()
