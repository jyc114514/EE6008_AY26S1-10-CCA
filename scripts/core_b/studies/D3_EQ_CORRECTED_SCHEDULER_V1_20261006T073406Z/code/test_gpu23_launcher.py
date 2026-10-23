"""CPU-only whitelist and shared-watcher lease-selection tests."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "code"))

from gpu23_filter import filter_snapshots  # noqa: E402

import torch  # noqa: E402
import gpu23_launcher  # noqa: E402
import physical_gpu_guard  # noqa: E402
import run_queue  # noqa: E402
WATCHER_PATH = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
SPEC = importlib.util.spec_from_file_location("task_test_shared_gpu_watcher", WATCHER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load shared watcher for CPU tests")
WATCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WATCHER
SPEC.loader.exec_module(WATCHER)

UUIDS = {
    2: "GPU-test-physical-2",
    3: "GPU-test-physical-3",
}


def snapshot(index: int, *, free_mib: int = 4096, utilization: int = 0):
    return WATCHER.GpuSnapshot(
        index=index,
        uuid=UUIDS.get(index, f"GPU-test-physical-{index}"),
        name="Mock RTX",
        memory_used_mib=24576 - free_mib,
        memory_total_mib=24576,
        utilization_percent=utilization,
        utc="2026-10-06T00:00:00Z",
    )


def job(root: Path) -> dict[str, object]:
    return {
        "job_id": "MOCK-D3EQ-GPU23",
        "protocol_id": "CORRECTED_SCHEDULER_MATCHED_12",
        "output_root": str(root / "out"),
        "command": ["/bin/true"],
        "resource_estimate": {"measured_peak_mib": 246},
        "preferred_gpu": 2,
        "environment": {},
    }


class TestTaskGpuWhitelist(unittest.TestCase):
    def test_gpu1_only_never_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launches: list[int] = []
            original = WATCHER.launch_child
            WATCHER.launch_child = lambda selected, status_path, cwd=None: launches.append(int(selected["physical_gpu"])) or 0
            try:
                allowed = filter_snapshots([snapshot(1)], UUIDS)
                self.assertEqual(allowed, [])
                code = WATCHER.run_registered_job(
                    job(root),
                    lease_dir=root / "leases",
                    status_path=root / "status.json",
                    visible_snapshots_fn=lambda: allowed,
                    snapshot_fn=lambda index: snapshot(index),
                    poll_seconds=0.01,
                    max_wait_seconds=0.05,
                    max_oom_requeues=0,
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(launches, [])
            finally:
                WATCHER.launch_child = original

    def test_gpu3_only_never_launches_for_gpu2_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launches: list[int] = []
            original = WATCHER.launch_child
            WATCHER.launch_child = lambda selected, status_path, cwd=None: launches.append(int(selected["physical_gpu"])) or 0
            visible = filter_snapshots([snapshot(3)], {2: UUIDS[2]})
            try:
                code = WATCHER.run_registered_job(
                    job(root), lease_dir=root / "leases", status_path=root / "status.json",
                    visible_snapshots_fn=lambda: visible,
                    snapshot_fn=lambda index: snapshot(index), poll_seconds=0.01,
                    max_wait_seconds=0.05, max_oom_requeues=0,
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(visible, [])
                self.assertEqual(launches, [])
            finally:
                WATCHER.launch_child = original

    def test_gpu2_lease_waits_without_fallback_to_gpu3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lease_dir = root / "leases"
            lease_dir.mkdir()
            locked_gpu2 = (lease_dir / "gpu_2.lease").open("a+", encoding="utf-8")
            fcntl.flock(locked_gpu2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            launches: list[int] = []
            original = WATCHER.launch_child
            WATCHER.launch_child = lambda selected, status_path, cwd=None: launches.append(int(selected["physical_gpu"])) or 0
            visible = filter_snapshots([snapshot(2), snapshot(3)], {2: UUIDS[2]})
            try:
                self.assertEqual([item.index for item in visible], [2])
                code = WATCHER.run_registered_job(
                    job(root), lease_dir=lease_dir, status_path=root / "status.json",
                    visible_snapshots_fn=lambda: visible,
                    snapshot_fn=lambda index: snapshot(index), poll_seconds=0.01,
                    max_wait_seconds=0.05, max_oom_requeues=0,
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(launches, [])
            finally:
                WATCHER.launch_child = original
                fcntl.flock(locked_gpu2.fileno(), fcntl.LOCK_UN)
                locked_gpu2.close()

    def test_gpu2_capacity_spawns_only_gpu2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launches: list[int] = []
            original = WATCHER.launch_child
            WATCHER.launch_child = lambda selected, status_path, cwd=None: launches.append(int(selected["physical_gpu"])) or 0
            visible = filter_snapshots([snapshot(1), snapshot(2), snapshot(3)], {2: UUIDS[2]})
            try:
                code = WATCHER.run_registered_job(
                    job(root), lease_dir=root / "leases", status_path=root / "status.json",
                    visible_snapshots_fn=lambda: visible,
                    snapshot_fn=lambda index: snapshot(index), poll_seconds=0.01,
                    max_wait_seconds=0.05, max_oom_requeues=0,
                )
                self.assertEqual(code, 0)
                self.assertEqual(launches, [2])
            finally:
                WATCHER.launch_child = original

    def test_high_utilization_gpu2_is_filtered_and_waited_on(self) -> None:
        launches: list[int] = []
        original = WATCHER.launch_child
        WATCHER.launch_child = lambda selected, status_path, cwd=None: launches.append(int(selected["physical_gpu"])) or 0
        hot = snapshot(2, utilization=100)
        visible = filter_snapshots([hot], {2: UUIDS[2]})
        try:
            self.assertEqual(visible, [])
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                code = WATCHER.run_registered_job(
                    job(root), lease_dir=root / "leases", status_path=root / "status.json",
                    visible_snapshots_fn=lambda: visible,
                    snapshot_fn=lambda index: hot, poll_seconds=0.01,
                    max_wait_seconds=0.05, max_oom_requeues=0,
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(launches, [])
        finally:
            WATCHER.launch_child = original
        wait_error = {"status": "telemetry_error", "error": "RuntimeError: D3_RESOURCE_WAIT_HIGH_UTIL"}
        self.assertTrue(run_queue.is_resource_wait_attempt(125, wait_error))
        with self.assertRaisesRegex(RuntimeError, "D3_RESOURCE_WAIT_HIGH_UTIL"):
            from gpu23_filter import validate_live_identity
            validate_live_identity(2, UUIDS[2], {2: UUIDS[2]}, utilization_percent=100)

    def test_single_gpu2_candidate_means_serial_queue(self) -> None:
        self.assertEqual(run_queue.effective_queue_concurrency([2]), 1)
        with self.assertRaises(ValueError):
            run_queue.effective_queue_concurrency([3])

    def test_child_attempt_retry_is_bounded_per_job_id(self) -> None:
        self.assertTrue(run_queue.can_retry_failed_job("CELL-ATTEMPT1"))
        self.assertFalse(run_queue.can_retry_failed_job("CELL-ATTEMPT2"))
        self.assertFalse(run_queue.can_retry_failed_job("CELL-ATTEMPT3"))
        self.assertEqual(run_queue.job_attempt_number("CELL-ATTEMPT3"), 3)

    def test_resource_timeout_is_not_a_science_failure(self) -> None:
        waiting = {"status": "capacity_timeout"}
        self.assertTrue(run_queue.is_resource_wait_attempt(125, waiting))
        launched = {"status": "capacity_timeout", "lease_acquired_utc": "2026-10-06T00:00:00Z"}
        self.assertFalse(run_queue.is_resource_wait_attempt(125, launched))
        self.assertFalse(run_queue.is_resource_wait_attempt(0, waiting))


    def test_child_guard_checks_physical_uuid_and_logical_cuda_zero(self) -> None:
        uuid_map = json.dumps({"2": UUIDS[2]})
        name_map = json.dumps({"2": "Mock RTX"})
        environment = {
            "GPU_WATCHER_PHYSICAL_GPU": "2",
            "CUDA_VISIBLE_DEVICES": "2",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "D3_TASK_GPU_ALLOWLIST": "2",
            "D3_GPU_UUID_MAP": uuid_map,
            "D3_GPU_NAME_MAP": name_map,
            "GPU_WATCHER_JOB_ID": "MOCK-GUARD",
        }
        result = SimpleNamespace(stdout=f"{UUIDS[2]}, Mock RTX" + chr(10))
        properties = SimpleNamespace(name="Mock RTX", uuid=UUIDS[2].removeprefix("GPU-"))
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("subprocess.run", return_value=result) as run,
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(torch.cuda, "set_device") as set_device,
            mock.patch.object(torch.cuda, "get_device_properties", return_value=properties),
        ):
            physical, uuid, name = physical_gpu_guard.verify_assigned_device("mock-nvidia-smi")
            self.assertEqual((physical, uuid, name), (2, UUIDS[2], "Mock RTX"))
            self.assertEqual(os.environ["D3_EXPECTED_GPU_UUID"], UUIDS[2])
            self.assertEqual(run.call_args.args[0][1:3], ["-i", "2"])
            set_device.assert_called_once_with(0)

    def test_torch_uuid_representation_normalizes_prefix_and_bytes(self) -> None:
        suffix = UUIDS[2].removeprefix("GPU-")
        self.assertEqual(physical_gpu_guard.normalize_gpu_uuid(UUIDS[2]),
                         physical_gpu_guard.normalize_gpu_uuid(suffix))
        self.assertEqual(physical_gpu_guard.normalize_gpu_uuid(suffix.encode("ascii")),
                         physical_gpu_guard.normalize_gpu_uuid(UUIDS[2]))

    def test_child_guard_fails_closed_on_uuid_mismatch(self) -> None:
        environment = {
            "GPU_WATCHER_PHYSICAL_GPU": "2",
            "CUDA_VISIBLE_DEVICES": "2",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "D3_TASK_GPU_ALLOWLIST": "2",
            "D3_GPU_UUID_MAP": json.dumps({"2": UUIDS[2]}),
        }
        result = SimpleNamespace(stdout="GPU-unapproved, Mock RTX" + chr(10))
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("subprocess.run", return_value=result),
        ):
            with self.assertRaises(RuntimeError):
                physical_gpu_guard.verify_assigned_device("mock-nvidia-smi")
    def test_child_guard_rejects_inactive_gpu3_and_device_mapping_mismatch(self) -> None:
        inactive = {
            "GPU_WATCHER_PHYSICAL_GPU": "3",
            "CUDA_VISIBLE_DEVICES": "3",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "D3_TASK_GPU_ALLOWLIST": "2",
            "D3_GPU_UUID_MAP": json.dumps({"2": UUIDS[2]}),
        }
        with mock.patch.dict(os.environ, inactive, clear=True):
            with self.assertRaises(RuntimeError):
                physical_gpu_guard.verify_assigned_device("mock-nvidia-smi")
        mismatch = {
            "GPU_WATCHER_PHYSICAL_GPU": "2",
            "CUDA_VISIBLE_DEVICES": "3",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "D3_TASK_GPU_ALLOWLIST": "2",
            "D3_GPU_UUID_MAP": json.dumps({"2": UUIDS[2]}),
        }
        with mock.patch.dict(os.environ, mismatch, clear=True):
            with self.assertRaises(RuntimeError):
                physical_gpu_guard.verify_assigned_device("mock-nvidia-smi")

    def test_launcher_dry_run_validates_schema_without_gpu_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_data = {
                "job_id": "MOCK-D3EQ-DRY-RUN",
                "protocol_id": "CORRECTED_SCHEDULER_MATCHED_12",
                "output_root": str(TASK / "runs/dry-run-probe"),
                "command": ["/bin/true"],
                "resource_estimate": {"measured_peak_mib": 246},
                "preferred_gpu": 2,
                "environment": {},
            }
            job_path = root / "job.json"
            job_path.write_text(json.dumps(job_data), encoding="utf-8")
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps({
                "status": "LOCKED",
                "study_id": "CORRECTED_SCHEDULER_MATCHED_12",
            }), encoding="utf-8")
            with (
                mock.patch.object(gpu23_launcher, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(gpu23_launcher, "load_allowed_uuids", return_value={2: UUIDS[2]}),
                mock.patch.object(sys, "argv", [
                    "gpu23_launcher", "--job-json", str(job_path),
                    "--lease-dir", str(root / "leases"), "--status-path", str(root / "status.json"),
                    "--dry-run",
                ]),
            ):
                self.assertEqual(gpu23_launcher.main(), 0)

if __name__ == "__main__":
    unittest.main()
