#!/usr/bin/env python3
"""CPU/read-only LoRA sidecar audit; never imports torch or launches a GPU."""

from __future__ import annotations
import os

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[0].parent
SIDE = ROOT / "lora_sidecar"
DIAG = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z/recovery_canonical_replay_20260911T054907Z/lora_canonical_identity_20260912T053528Z/diagnostics/CAB_8ROW_SUMMARY.json'))


def main() -> int:
    evidence = json.loads(DIAG.read_text(encoding="utf-8"))
    assert evidence["models"]["B3-FUS-D-L"]["classification"] == "A_EQUALS_B_NOT_C"
    assert evidence["models"]["B3-V-D-L"]["classification"] == "A_EQUALS_B_NOT_C"
    result = {"status": "PASS_READ_ONLY", "source": str(DIAG), "gpu_replay": False, "models": evidence["models"], "canonical_generation_state": "unknown", "mainline_blocked": False}
    (SIDE / "LORA_READONLY_AUDIT_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
