#!/usr/bin/env python3
"""Read-only summary of the exploratory lag diagnostics."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    data = json.loads((ROOT / "V2_TEMPORAL_LAG_DIAGNOSTICS.json").read_text())
    print(json.dumps({"lags": data["lags"], "synthetic_validation": data["synthetic_validation"]}, indent=2))
