"""Pin two non-semantic evaluator compatibility fixes before prediction read."""

from __future__ import annotations
import os

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))
CORE = OUT / "evaluate_d2_v3_panel.py"
BASE_LOCK = OUT / "D2_V3_MATCHER_LOCKED.json"
LOCK = OUT / "D2_V3_MATCHER_LOCKED_ATTEMPT_002.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(source: str) -> str:
    changes = (
        ("from collections import defaultdict", "from collections import Counter, defaultdict"),
        ('known_key = f"{definition.lower()}_known_{body}"', 'known_key = f"{definition.lower().replace(\'d2_\', \'\')}_known_{body}"'),
        ('armed_key = f"{definition.lower()}_armed_{body}"', 'armed_key = f"{definition.lower().replace(\'d2_\', \'\')}_armed_{body}"'),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise AssertionError(f"expected exactly one compatibility site: {old}")
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    original = CORE.read_text(encoding="utf-8")
    transformed = transform(original)
    import ast
    ast.parse(transformed)
    base = json.loads(BASE_LOCK.read_text())
    result = dict(base)
    result.update({
        "protocol_id": "CORE_B_D2_EVENT_MATCHER_V3",
        "status": "LOCKED_BEFORE_D2_V3_DEV_PREDICTION_READ_ATTEMPT_002",
        "supersedes_matcher_lock_sha256": sha256_file(BASE_LOCK),
        "evaluator_code_path": str(CORE) + " plus deterministic in-memory compatibility transform",
        "evaluator_code_sha256": sha256_file(CORE),
        "runtime_transformed_source_sha256": sha256_bytes(transformed.encode("utf-8")),
        "runtime_transformations": [
            "Import collections.Counter for the already-defined query aggregation code.",
            "Map D2_V2/D2_V3 prefixes to the existing v2_/v3_ query-state fields.",
        ],
        "transform_changes_estimand_or_match_rule": False,
        "prediction_values_read_during_lock": False,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
    })
    LOCK.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "core_hash": result["evaluator_code_sha256"],
        "transformed_hash": result["runtime_transformed_source_sha256"],
        "rules_changed": result["transform_changes_estimand_or_match_rule"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
