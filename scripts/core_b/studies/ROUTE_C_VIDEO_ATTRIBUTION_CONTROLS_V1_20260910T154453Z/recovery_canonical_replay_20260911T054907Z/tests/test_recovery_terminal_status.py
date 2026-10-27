import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_recovery_runner_has_terminal_failure_and_success_markers():
    source_path = ROOT / "code/replay_raw_identity.py"
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(source_path))
    assert "RUN_FAILED.json" in source
    assert '"status": "failed"' in source
    assert '"status": "complete"' in source
    assert '"final_test_accessed": 0' in source


def test_recovery_protocol_keeps_locked_tolerance():
    protocol = (ROOT / "RECOVERY_PROTOCOL_LOCKED.json").read_text(encoding="utf-8")
    assert '"prediction_atol": 1e-05' in protocol
    assert '"prediction_rtol": 1e-05' in protocol
