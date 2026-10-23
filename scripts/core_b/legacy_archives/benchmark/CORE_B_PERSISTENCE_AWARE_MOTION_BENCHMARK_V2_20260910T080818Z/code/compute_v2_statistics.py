#!/usr/bin/env python3
"""Read-only summary of locked v2 paired effects."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    with (ROOT / "V2_PAIRED_EFFECTS.csv").open(newline="") as handle:
        print(f"paired_effect_rows={sum(1 for _ in csv.DictReader(handle))}")
