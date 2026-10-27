#!/usr/bin/env python3
"""Read-only source-gate summary for the completed v2 package."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(json.dumps(json.loads((ROOT / "V2_SOURCE_ACCEPTANCE.json").read_text()), indent=2))
