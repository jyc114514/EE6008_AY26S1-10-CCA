import csv
from pathlib import Path


def test_selection_manifest_has_one_fixed_set():
    root = Path(__file__).parents[1]
    with (root / "V2_TRAJECTORY_SELECTION_MANIFEST.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["motion_stratum"] for row in rows} == {"low", "mid", "high"}
