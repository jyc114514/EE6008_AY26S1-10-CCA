"""CPU verification entrypoint supplying retained shared Stage0 lease helpers."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
sys.path.insert(0, str(CODE_ROOT))


def lease_hours(status_path: Path) -> float:
    if not status_path.is_file():
        return 0.0
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or int(data.get("exit_code", -1)) != 0:
        return 0.0
    start = datetime.fromisoformat(data["lease_acquired_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(data["lease_released_utc"].replace("Z", "+00:00"))
    return max((end - start).total_seconds() / 3600.0, 0.0)


def main() -> int:
    import d3_train_eq
    sys.modules["d3_eq_train"] = d3_train_eq
    import verify_d3_eq_stage0 as verifier
    verifier.lease_hours = lease_hours
    return verifier.main()


if __name__ == "__main__":
    raise SystemExit(main())
