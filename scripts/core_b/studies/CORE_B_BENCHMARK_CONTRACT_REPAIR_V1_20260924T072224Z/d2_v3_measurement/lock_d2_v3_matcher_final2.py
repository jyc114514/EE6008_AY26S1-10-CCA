"""Lock the body-specific lookup correction without changing matcher rules."""

from __future__ import annotations
import os

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))
CORE = OUT / "evaluate_d2_v3_panel.py"
BASE = OUT / "D2_V3_MATCHER_LOCKED.json"
SUPERSEDED = OUT / "D2_V3_MATCHER_LOCKED_ATTEMPT_002.json"
LOCK = OUT / "D2_V3_MATCHER_LOCKED_ATTEMPT_003.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transformed(source: str) -> str:
    replacements = (
        ("from collections import defaultdict", "from collections import Counter, defaultdict"),
        ('known_key = f"{definition.lower()}_known_{body}"', 'known_key = f"{definition.lower().replace(\'d2_\', \'\')}_known_{body}"'),
        ('armed_key = f"{definition.lower()}_armed_{body}"', 'armed_key = f"{definition.lower().replace(\'d2_\', \'\')}_armed_{body}"'),
        ("all_events = panel_events[group]", "all_events = panel_truth_by_body_group[(group, body)]"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise AssertionError(f"compatibility patch site count mismatch: {old}")
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    source = CORE.read_text(encoding="utf-8")
    compiled = transformed(source)
    ast.parse(compiled)
    base = json.loads(BASE.read_text())
    superseded_hash = file_hash(SUPERSEDED) if SUPERSEDED.exists() else file_hash(BASE)
    result = dict(base)
    result.update({
        "status": "LOCKED_BEFORE_D2_V3_DEV_PREDICTION_READ",
        "lock_attempt": "003",
        "supersedes_matcher_lock_sha256": superseded_hash,
        "evaluator_code_path": str(CORE) + " plus deterministic in-memory compatibility transform",
        "evaluator_code_sha256": file_hash(CORE),
        "runtime_transformed_source_sha256": hashlib.sha256(compiled.encode("utf-8")).hexdigest(),
        "runtime_transformations": [
            "Import collections.Counter for query support aggregation.",
            "Map D2_V2/D2_V3 query-state keys to existing v2_/v3_ fields.",
            "Select GT event map with the current body key during one-to-one matching.",
        ],
        "transform_changes_estimand_or_match_rule": False,
        "previous_dev_forecast_values_exposed_in_failed_attempt": True,
        "previous_failed_attempt_completed_model_cells": 0,
        "outcome_exposure": "Development forecasts were read in attempt 005, but no cell summary/artifact was completed; the fix only corrects body-key lookup and leaves thresholds, clustering, matching, denominators and bootstrap unchanged.",
        "prediction_values_read_during_lock": False,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
    })
    LOCK.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_sha256": result["runtime_transformed_source_sha256"], "method_changed": False, "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
