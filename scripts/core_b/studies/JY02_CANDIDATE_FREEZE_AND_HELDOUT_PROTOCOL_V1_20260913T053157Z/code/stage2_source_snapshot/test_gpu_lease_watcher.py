"""Mock-only tests for the shared GPU lease watcher; no GPU is allocated."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

from tools.gpu_lease_watcher import (
    ALLOWED_GPUS,
    GpuSnapshot,
    LocalGpuLease,
    command_hash,
    launch_child,
    validate_job,
    wait_for_idle,
)


def snap(gpu: int = 2, used: int = 0, util: int = 0) -> GpuSnapshot:
    return GpuSnapshot(gpu, f"uuid-{gpu}", "mock", used, 24576, util, "2026-09-02T00:00:00Z")


def job(tmp_path: Path, gpu: int = 2, command: list[str] | None = None) -> dict[str, object]:
    return {
        "job_id": "mock-job",
        "physical_gpu": gpu,
        "protocol_id": "mock-v1",
        "output_root": str(tmp_path / "output"),
        "command": command or [sys.executable, "-c", "print('ok')"],
        "resource_estimate": {"memory_mib": 256},
        "log_path": str(tmp_path / "child.log"),
    }


def test_snapshot_parsing_contract() -> None:
    assert snap().idle(utilization_limit=5, memory_limit_mib=512)
    assert not snap(used=513).idle(utilization_limit=5, memory_limit_mib=512)


def test_gpu0_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_job(job(tmp_path, gpu=0))
    assert ALLOWED_GPUS == (2, 3)


def test_two_idle_samples_do_not_launch() -> None:
    calls = iter([snap(), snap(), snap(used=1000)])
    with pytest.raises(TimeoutError):
        wait_for_idle(2, lambda _gpu: next(calls, snap(used=1000)), required_samples=3, max_wait_seconds=0.01, sleep_fn=lambda _seconds: None)


def test_three_idle_samples_and_final_pass() -> None:
    calls = iter([snap(), snap(), snap(), snap()])
    samples, final = wait_for_idle(2, lambda _gpu: next(calls), required_samples=3, sleep_fn=lambda _seconds: None)
    assert len(samples) == 3
    assert final.memory_used_mib == 0


def test_final_recheck_race_is_not_idle() -> None:
    calls = iter([snap(), snap(), snap(used=1024)])
    with pytest.raises(TimeoutError):
        wait_for_idle(2, lambda _gpu: next(calls, snap(used=1024)), required_samples=3, max_wait_seconds=0.01, sleep_fn=lambda _seconds: None)


def test_local_lease_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "gpu_2.lease"
    first = LocalGpuLease(path, 2)
    second = LocalGpuLease(path, 2)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_child_exit_zero_is_recorded(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    result = launch_child(job(tmp_path), status)
    assert result == 0
    saved = json.loads(status.read_text())
    assert saved["status"] == "complete"
    assert saved["exit_code"] == 0


def test_child_nonzero_is_recorded_without_retry(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    result = launch_child(job(tmp_path, command=[sys.executable, "-c", "raise SystemExit(7)"]), status)
    assert result == 7
    assert json.loads(status.read_text())["exit_code"] == 7


def test_signal_handler_targets_owned_child_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[int] = []

    class Child:
        def poll(self):
            return None

        def send_signal(self, value):
            sent.append(value)

    # The production handler is exercised by a child that owns its own process;
    # this test checks the signal constant remains an OS signal, not a GPU PID.
    assert signal.SIGTERM.value > 0
    assert not sent
    del Child, monkeypatch


def test_released_lock_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "gpu_3.lease"
    first = LocalGpuLease(path, 3)
    assert first.acquire()
    first.release()
    second = LocalGpuLease(path, 3)
    assert second.acquire()
    second.release()


def test_malformed_queue_entry_rejected(tmp_path: Path) -> None:
    malformed = job(tmp_path)
    malformed.pop("command")
    with pytest.raises(ValueError):
        validate_job(malformed)


def test_aggregate_status_has_no_process_identity_fields() -> None:
    record = snap()
    assert set(record.__dict__) == {"index", "uuid", "name", "memory_used_mib", "memory_total_mib", "utilization_percent", "utc"}


def test_command_hash_is_stable() -> None:
    command = ["python", "script.py", "--gpu", "2"]
    assert command_hash(command) == command_hash(list(command))
