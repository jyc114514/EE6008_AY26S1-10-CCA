import json
from pathlib import Path


def test_protocol_disallows_optimizer_steps():
    root = Path(__file__).parents[1]
    protocol = json.loads((root / "ROUTE_C_PROTOCOL_LOCKED.json").read_text())
    assert protocol["execution"]["optimizer_steps_required"] == 0
    assert protocol["execution"]["checkpoint_weights_modified"] is False
