import ast
import fcntl
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import task_local_gpu_watcher as watcher


def _release(handle: object) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _snapshot(gpu: int, *, utilization: int = 0, memory: int = 15) -> watcher.Snapshot:
    return watcher.Snapshot(
        gpu,
        f"GPU-test-{gpu}",
        "NVIDIA GeForce RTX 3090",
        memory,
        24576,
        utilization,
        watcher.now(),
    )


def test_task_local_pool_accepts_gpu1_2_3_and_rejects_gpu0(monkeypatch, tmp_path):
    assert watcher.ALLOWED == (1, 2, 3)
    real_snapshot = watcher.snapshot

    calls: list[int] = []

    def fake_snapshot(gpu: int) -> watcher.Snapshot:
        calls.append(gpu)
        return _snapshot(gpu)

    monkeypatch.setattr(watcher, "snapshot", fake_snapshot)
    monkeypatch.setattr(watcher.time, "sleep", lambda _seconds: None)

    for gpu in watcher.ALLOWED:
        samples, final = watcher.wait_idle(
            gpu,
            poll_seconds=0,
            status_path=tmp_path / f"gpu_{gpu}.json",
        )
        assert len(samples) == watcher.IDLE_SAMPLES
        assert final.index == gpu
        assert all(sample.index == gpu for sample in samples)

    monkeypatch.setattr(watcher, "snapshot", real_snapshot)
    with pytest.raises(ValueError, match="allowlist"):
        watcher.snapshot(0)

    assert calls == [gpu for gpu in watcher.ALLOWED for _ in range(watcher.IDLE_SAMPLES + 1)]


def test_two_child_slots_are_the_maximum_and_release_is_reusable(tmp_path):
    lease_dir = tmp_path / "leases"
    handles = [watcher.acquire(lease_dir / f"child_slot_{index}.lease") for index in range(2)]
    assert all(handle is not None for handle in handles)

    assert watcher.acquire(lease_dir / "child_slot_0.lease") is None

    for handle in handles:
        _release(handle)

    reusable = watcher.acquire(lease_dir / "child_slot_0.lease")
    assert reusable is not None
    _release(reusable)


def test_launch_records_exact_index_uuid_binding_and_heartbeat(tmp_path):
    env_path = tmp_path / "child_env.json"
    status_path = tmp_path / "status.json"
    log_path = tmp_path / "child.log"
    uuid = "GPU-test-1"
    child_code = (
        "import json, os, pathlib, time; "
        f"pathlib.Path({str(env_path)!r}).write_text(json.dumps({{key: os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'GPU_WATCHER_PHYSICAL_GPU', 'GPU_WATCHER_GPU_UUID', 'ROUTE_C_TASK_LOCAL_GPU_UUID']}})); "
        "time.sleep(0.08)"
    )
    job = {
        "job_id": "route-c-test-gpu1",
        "protocol_id": "core_b_route_c_test",
        "physical_gpu": 1,
        "output_root": str(tmp_path / "output"),
        "log_path": str(log_path),
        "command": [sys.executable, "-c", child_code],
    }

    result = watcher.launch(
        job,
        status_path,
        cwd=tmp_path,
        gpu_uuid=uuid,
        heartbeat_seconds=0.01,
    )

    assert result == 0
    assert json.loads(env_path.read_text()) == {
        "CUDA_VISIBLE_DEVICES": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "GPU_WATCHER_PHYSICAL_GPU": "1",
        "GPU_WATCHER_GPU_UUID": uuid,
        "ROUTE_C_TASK_LOCAL_GPU_UUID": uuid,
    }
    status = json.loads(status_path.read_text())
    assert status["physical_gpu"] == 1
    assert status["gpu_uuid"] == uuid
    assert status["heartbeat_count"] >= 1
    assert status["non_child_process_signaled"] is False


def test_main_releases_gpu_and_child_leases(monkeypatch, tmp_path):
    job_path = tmp_path / "job.json"
    status_path = tmp_path / "status.json"
    lease_dir = tmp_path / "leases"
    job_path.write_text(
        json.dumps(
            {
                "job_id": "route-c-release-test",
                "protocol_id": "core_b_route_c_test",
                "physical_gpu": 1,
                "output_root": str(tmp_path / "output"),
                "command": ["unused"],
            }
        )
    )
    stable = _snapshot(1)

    monkeypatch.setattr(
        watcher,
        "wait_idle",
        lambda _gpu, *, poll_seconds, status_path: ([stable] * watcher.IDLE_SAMPLES, stable),
    )
    monkeypatch.setattr(watcher, "snapshot", lambda _gpu: stable)

    def fake_launch(job, status_path, cwd, *, gpu_uuid):
        status_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "physical_gpu": job["physical_gpu"],
                    "gpu_uuid": gpu_uuid,
                    "non_child_process_signaled": False,
                }
            )
        )
        return 0

    monkeypatch.setattr(watcher, "launch", fake_launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_local_gpu_watcher.py",
            "--job-json",
            str(job_path),
            "--status-path",
            str(status_path),
            "--lease-dir",
            str(lease_dir),
        ],
    )

    assert watcher.main() == 0
    status = json.loads(status_path.read_text())
    assert status["lease_released"] is True
    assert status["non_child_process_signaled"] is False

    for path in (lease_dir / "gpu_1.lease", lease_dir / "child_slot_0.lease", lease_dir / "child_slot_1.lease"):
        handle = watcher.acquire(path)
        assert handle is not None
        _release(handle)


def test_signal_surface_only_forwards_to_owned_child():
    source = inspect.getsource(watcher)
    tree = ast.parse(source)
    prohibited = {"kill", "killpg", "terminate", "pause", "renice", "reset"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        assert node.func.attr not in prohibited
        if node.func.attr == "send_signal":
            assert isinstance(node.func.value, ast.Name)
            assert node.func.value.id == "child"

    assert "non_child_process_signaled" in source
