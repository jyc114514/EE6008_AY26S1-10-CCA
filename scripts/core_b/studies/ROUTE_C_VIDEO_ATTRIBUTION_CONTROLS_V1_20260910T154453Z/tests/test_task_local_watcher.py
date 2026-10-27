import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from task_local_gpu_watcher import Snapshot, ALLOWED


def test_task_local_gpu_override_is_scoped():
    assert ALLOWED == (1, 2, 3)
    assert Snapshot(1, "uuid", "3090", 0, 24576, 0, "now").idle()
    assert not Snapshot(1, "uuid", "3090", 7862, 24576, 100, "now").idle()
