"""Pin the tiny compatibility runner and the already locked D2_V3 protocol."""

from __future__ import annotations
import os

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement'))
RUNNER = OUT / "run_d2_v3_locked_evaluation.py"
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
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "matcher_protocol_sha256": sha256_file(MATCHER),
        "threshold_protocol_sha256": matcher["threshold_protocol_sha256"],
        "core_evaluator_sha256": matcher["evaluator_code_sha256"],
        "runner_path": str(RUNNER),
        "runner_code_sha256": sha256_file(RUNNER),
        "reads_prediction_values": False,
        "final_test_access": 0,
    }
    (OUT / "D2_V3_RUNNER_LOCKED.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, sort_keys=True))


if __name__ == "__main__":
    main()
