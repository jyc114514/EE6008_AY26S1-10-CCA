from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SHARED_TOOLS = PROJECT / "repo/ee6008-humanoid-vjepa/tools"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(SHARED_TOOLS))
import gpu_lease_watcher as watcher  # noqa: E402
import run_orchestrator  # noqa: E402


def _snapshot(index: int = 2, free: int = 24000) -> watcher.GpuSnapshot:
    total = 24576
    return watcher.GpuSnapshot(index, f"GPU-TEST-{index}", "TEST GPU", total-free, total, 0, "test")


def _job(tmp_path: Path, job_id: str, command: list[str]) -> dict:
    return {
        "job_id": job_id,
        "protocol_id": "TEST_SHARED_WATCHER_POLICY",
        "output_root": str(tmp_path / "output" / job_id),
        "command": command,
        "resource_estimate": {"measured_peak_mib": 256},
        "log_path": str(tmp_path / f"{job_id}.log"),
        "max_oom_requeues": 0,
    }


def test_controller_uses_shared_watcher_shared_lease_and_two_slots() -> None:
    assert run_orchestrator.MAX_ACTIVE == 2
    assert run_orchestrator.WATCHER == PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
    assert run_orchestrator.LEASE_DIR == PROJECT / "repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease"
    assert (ROOT / "receipts/REJECTED_WATCHER_COPY_20261008.snapshot").is_file()
    assert not (CODE / "gpu_lease_watcher_dynamic.py").exists()


def test_two_project_slots_block_third_and_gpu_lease_is_exclusive(tmp_path: Path) -> None:
    first = watcher.ProjectChildSlot(tmp_path, slot_count=2)
    second = watcher.ProjectChildSlot(tmp_path, slot_count=2)
    third = watcher.ProjectChildSlot(tmp_path, slot_count=2)
    assert first.acquire(job_id="one")
    assert second.acquire(job_id="two")
    assert not third.acquire(job_id="three")
    first.release()
    second.release()
    assert third.acquire(job_id="three")
    third.release()

    gpu_a = watcher.LocalGpuLease(tmp_path / "gpu_2.lease", 2)
    gpu_b = watcher.LocalGpuLease(tmp_path / "gpu_2.lease", 2)
    assert gpu_a.acquire()
    assert not gpu_b.acquire()
    gpu_a.release()
    assert gpu_b.acquire()
    gpu_b.release()


def test_dynamic_candidate_pool_falls_back_when_preference_lacks_capacity() -> None:
    candidates = [
        watcher.GpuSnapshot(0, "GPU-0", "GPU0", 1000, 24576, 98, "test"),
        watcher.GpuSnapshot(2, "GPU-2", "GPU2", 24000, 24576, 0, "test"),
    ]
    picked = watcher.choose_gpu(candidates, 1536, preferred=2)
    assert picked.index == 0
    assert picked.uuid == "GPU-0"


def test_launch_receipt_matches_selected_uuid_and_child_physical_index(tmp_path: Path) -> None:
    snap = _snapshot(index=2)
    out = tmp_path / "child.json"
    command = [sys.executable, "-c", "import json,os; print(json.dumps({'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'physical_gpu':os.environ['GPU_WATCHER_PHYSICAL_GPU']}))"]
    job = _job(tmp_path, "uuid-bind", command)
    status_path = tmp_path / "status.json"
    rc = watcher.run_registered_job(
        job, lease_dir=tmp_path / "leases", status_path=status_path,
        visible_snapshots_fn=lambda: [snap], snapshot_fn=lambda idx: snap,
        poll_seconds=0.01, max_wait_seconds=0, max_oom_requeues=0,
    )
    assert rc == 0
    receipt = json.loads(status_path.read_text())
    child = json.loads(Path(job["log_path"]).read_text().strip())
    assert receipt["final_recheck"]["uuid"] == snap.uuid
    assert receipt["physical_gpu"] == snap.index
    assert receipt["lease_path"].endswith("gpu_2.lease")
    assert child["cuda_visible_devices"] == str(snap.index)
    assert child["physical_gpu"] == str(snap.index)
    assert receipt["lease_released"] is True
    assert receipt["project_child_slot_released"] is True


def test_child_failure_releases_its_gpu_and_project_slots(tmp_path: Path) -> None:
    snap = _snapshot(index=1)
    job = _job(tmp_path, "failure-release", [sys.executable, "-c", "raise SystemExit(7)"])
    status_path = tmp_path / "failed-status.json"
    rc = watcher.run_registered_job(
        job, lease_dir=tmp_path / "leases", status_path=status_path,
        visible_snapshots_fn=lambda: [snap], snapshot_fn=lambda idx: snap,
        poll_seconds=0.01, max_wait_seconds=0, max_oom_requeues=0,
    )
    assert rc == 7
    receipt = json.loads(status_path.read_text())
    assert receipt["lease_released"] is True
    assert receipt["project_child_slot_released"] is True
    slot = watcher.ProjectChildSlot(tmp_path / "leases", slot_count=2)
    gpu = watcher.LocalGpuLease(tmp_path / "leases/gpu_1.lease", 1)
    assert slot.acquire(job_id="after-failure")
    assert gpu.acquire()
    gpu.release()
    slot.release()


def test_capacity_shortfall_waits_then_reports_timeout_without_gpu_query(tmp_path: Path) -> None:
    low = _snapshot(index=2, free=700)
    job = _job(tmp_path, "capacity-wait", [sys.executable, "-c", "raise SystemExit(0)"])
    status_path = tmp_path / "capacity-status.json"
    rc = watcher.run_registered_job(
        job, lease_dir=tmp_path / "leases", status_path=status_path,
        visible_snapshots_fn=lambda: [low], snapshot_fn=lambda idx: low,
        poll_seconds=0.01, max_wait_seconds=0.03, max_oom_requeues=0,
    )
    assert rc == 125
    receipt = json.loads(status_path.read_text())
    assert receipt["status"] == "capacity_timeout"
    assert not Path(job["log_path"]).exists()


def test_queue_loader_rejects_duplicate_job_keys(tmp_path: Path) -> None:
    row = _job(tmp_path, "duplicate", [sys.executable, "-c", "pass"])
    row.update({"phase": "formal", "run_id": "same"})
    queue = tmp_path / "jobs.jsonl"
    queue.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate job IDs"):
        run_orchestrator.load_jobs(queue)


def test_controller_lock_enforces_one_owner_per_run_root(tmp_path: Path) -> None:
    original_root = run_orchestrator.ROOT
    run_orchestrator.ROOT = tmp_path
    first = None
    try:
        first, owner = run_orchestrator.acquire_controller_lock("test-session", "coordination/jobs/STAGE0.jsonl")
        assert owner["pid"] > 0
        assert owner["tmux_session"] == "test-session"
        with pytest.raises(RuntimeError, match="another controller"):
            run_orchestrator.acquire_controller_lock("duplicate", "coordination/jobs/STAGE0.jsonl")
        first.close()
        first = None
        second, _ = run_orchestrator.acquire_controller_lock("resume", "coordination/jobs/STAGE0.jsonl")
        second.close()
    finally:
        if first is not None:
            first.close()
        run_orchestrator.ROOT = original_root
