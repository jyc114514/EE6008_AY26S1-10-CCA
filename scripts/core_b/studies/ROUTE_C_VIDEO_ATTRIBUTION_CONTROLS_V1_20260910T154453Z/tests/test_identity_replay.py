import json
from pathlib import Path


def test_reuse_records_exist_for_four_models():
    root = Path(__file__).parents[1]
    data = json.loads((root / "ROUTE_C_P0_REUSE_RECORD.json").read_text())
    assert data["status"] == "PASS"
    assert data["count"] == 4
