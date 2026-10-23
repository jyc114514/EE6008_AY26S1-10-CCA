#!/usr/bin/env python3
"""Read-only viewer for train-derived motion strata."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(json.dumps(json.loads((ROOT / "MOTION_STRATA_MANIFEST.json").read_text()), indent=2))
