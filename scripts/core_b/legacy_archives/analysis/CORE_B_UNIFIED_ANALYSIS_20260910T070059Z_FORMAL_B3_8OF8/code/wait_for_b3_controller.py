#!/usr/bin/env python3
"""Read-only bounded heartbeat monitor for the pre-existing B3 controller."""

from __future__ import annotations
import os

import json
import time
from datetime import datetime, timezone
from pathlib import Path


STATUS = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/B3_CONTROLLER_STATUS.json'))


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    last = None
    while True:
        data = json.loads(STATUS.read_text(encoding="utf-8"))
        signature = (data.get("status"), tuple(sorted((x.get("condition"), x.get("gpu")) for x in data.get("active_wave", []))), len(data.get("completed", [])))
        if signature != last:
            print(json.dumps({"observed_utc": utc(), "status": data.get("status"), "active_wave": data.get("active_wave", []), "completed_count": len(data.get("completed", [])), "final_test_access": data.get("final_test_access", 0)}), flush=True)
            last = signature
        if data.get("status") in {"complete", "failed", "blocked", "terminal"}:
            return 0 if data.get("status") == "complete" else 2
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
