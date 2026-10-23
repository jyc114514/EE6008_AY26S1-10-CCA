"""CPU-only tests: GPU1 is never selected and an occupied GPU2 falls to GPU3."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import gpu23_launcher as adapter


def load_shared_watcher():
    spec = importlib.util.spec_from_file_location("task_test_shared_watcher", adapter.WATCHER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def snapshot(watcher, index: int, utilization: int = 1):
    return watcher.GpuSnapshot(index, f"GPU-{index}", "mock", 100, 24000, utilization, "2026-10-06T00:00:00Z")


def job(root: Path):
    return {"job_id": "cpu-mock", "protocol_id": "D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1",
            "output_root": str(adapter.OUT / "test-mock-output"), "command": ["mock-child"],
            "resource_estimate": {"measured_peak_mib": 512}}


def main() -> int:
    watcher = load_shared_watcher()
    adapter._load_watcher = lambda: watcher
    # Only physical GPU1 is visible/idle: filtered pool is empty, bounded wait
    # expires, and launch_child must remain untouched.
    with tempfile.TemporaryDirectory(prefix="d3eq-gpu23-test-") as temp:
        root = Path(temp)
        launch = patch.object(watcher, "launch_child", side_effect=AssertionError("GPU1 fallback launched"))
        with launch:
            rc = adapter.run_registered_job_gpu23(
                job(root), lease_dir=root / "lease1", status_path=root / "status1.json",
                visible_snapshots_fn=lambda: [snapshot(watcher, 1)],
                snapshot_fn=lambda i: snapshot(watcher, i), poll_seconds=0.01, max_wait_seconds=0.03)
        state = json.loads((root / "status1.json").read_text())
        assert rc == 125 and state["status"] == "capacity_timeout"
        assert state["task_allowed_physical_gpus"] == [2, 3]
        assert state["task_visible_pool_mode"] == "fixed_allowlist_2_3"

    # GPU2 and GPU3 visible, but a held GPU2 lease forces the watcher to choose
    # GPU3; GPU1 may be present and maximally idle but is not exposed to ranking.
    with tempfile.TemporaryDirectory(prefix="d3eq-gpu23-test-") as temp:
        root = Path(temp)
        held = watcher.LocalGpuLease(root / "leases/gpu_2.lease", 2)
        assert held.acquire()
        chosen: list[int] = []
        def mock_launch(selected_job, status_path):
            chosen.append(int(selected_job["physical_gpu"]))
            return 0
        with patch.object(watcher, "launch_child", side_effect=mock_launch):
            rc = adapter.run_registered_job_gpu23(
                job(root), lease_dir=root / "leases", status_path=root / "status2.json",
                visible_snapshots_fn=lambda: [snapshot(watcher, 1, 0), snapshot(watcher, 2, 5), snapshot(watcher, 3, 10)],
                snapshot_fn=lambda i: snapshot(watcher, i), poll_seconds=0.01, max_wait_seconds=0.1)
        held.release()
        state = json.loads((root / "status2.json").read_text())
        assert rc == 0 and chosen == [3], (rc, chosen)
        assert state["selected_gpu"] == 3
        assert state["task_allowed_physical_gpus"] == [2, 3]

    print('{"status":"PASS","gpu1_only_pool":"no_launch_timeout","gpu2_lease_held":"selected_gpu3","gpu_query_performed":false,"child_process_started":false}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
