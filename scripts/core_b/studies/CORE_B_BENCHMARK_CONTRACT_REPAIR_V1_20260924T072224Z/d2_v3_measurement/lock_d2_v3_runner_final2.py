"""Lock attempt-two dev metadata join/runner, preserving attempt one's lock."""

from __future__ import annotations
import os

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))
RUNNER = OUT / "run_d2_v3_locked_evaluation_final2.py"
MATCHER = OUT / "D2_V3_MATCHER_LOCKED.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    matcher = json.loads(MATCHER.read_text())
    lock = {
        "status": "LOCKED_BEFORE_D2_V3_PREDICTION_EVALUATION",
        "attempt_id": "D2_V3_EVAL_ATTEMPT_002",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "matcher_protocol_sha256": sha256_file(MATCHER),
        "threshold_protocol_sha256": matcher["threshold_protocol_sha256"],
        "core_evaluator_sha256": matcher["evaluator_code_sha256"],
        "runner_path": str(RUNNER),
        "runner_code_sha256": sha256_file(RUNNER),
        "dev_source_group_join": "dev-only canonical split rows joined against the sealed D1 exact video SHA source-group table; IDs must agree 603/603",
        "csv_schema_adapter": "union all output row keys; blank missing cells; no metric transformations",
        "prediction_values_read_during_lock": False,
        "final_test_access": 0,
    }
    (OUT / "D2_V3_RUNNER_LOCKED_ATTEMPT_002.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, sort_keys=True))


if __name__ == "__main__":
    main()
