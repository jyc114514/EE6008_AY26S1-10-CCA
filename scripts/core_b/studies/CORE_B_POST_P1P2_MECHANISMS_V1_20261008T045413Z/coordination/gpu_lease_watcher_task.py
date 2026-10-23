"""Dynamic, capacity-aware GPU leasing and child-process launching.

The task-scoped watcher copy enumerates all GPUs visible to ``nvidia-smi`` and uses aggregate
memory/utilization telemetry only. Utilization is a ranking signal, never an
idle admission gate. Per-GPU and project-child ``flock`` semantics are preserved;
this task sets the child-slot count to the visible GPU count at admission. It never
inspects or signals an external process.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import random
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Self, TextIO

GPU_POOL_MODE = "dynamic_all_visible"
ONE_CHILD_PER_GPU = True
DEFAULT_SAFETY_MARGIN_MIB = 512
SAFETY_MARGIN_FRACTION = 0.10
DEFAULT_MAX_OOM_REQUEUES = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    utc: str

    @property
    def free_memory_mib(self) -> int:
        return max(self.memory_total_mib - self.memory_used_mib, 0)

    def capacity_ok(self, required_memory_mib: int) -> bool:
        return self.free_memory_mib >= required_memory_mib


def _parse_nvidia_rows(stdout: str) -> list[GpuSnapshot]:
    rows: list[GpuSnapshot] = []
    for raw in csv.reader(io.StringIO(stdout)):
        fields = [field.strip() for field in raw]
        if not fields:
            continue
        if len(fields) != 6:
            raise RuntimeError("aggregate nvidia-smi row has an unexpected schema")
        rows.append(
            GpuSnapshot(
                index=int(fields[0]),
                uuid=fields[1],
                name=fields[2],
                memory_used_mib=int(fields[3]),
                memory_total_mib=int(fields[4]),
                utilization_percent=int(fields[5]),
                utc=utc_now(),
            )
        )
    if not rows:
        raise RuntimeError("nvidia-smi did not report any visible GPU")
    return sorted(rows, key=lambda item: item.index)


def snapshot_visible_gpus(*, nvidia_smi: str = "nvidia-smi") -> list[GpuSnapshot]:
    """Return aggregate snapshots for every currently visible GPU."""

    query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(
        [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _parse_nvidia_rows(result.stdout)


def enumerate_visible_gpus(*, nvidia_smi: str = "nvidia-smi") -> tuple[int, ...]:
    return tuple(snapshot.index for snapshot in snapshot_visible_gpus(nvidia_smi=nvidia_smi))


def snapshot_gpu(gpu: int, *, nvidia_smi: str = "nvidia-smi") -> GpuSnapshot:
    """Compatibility helper that finds one GPU after dynamic enumeration."""

    for snapshot in snapshot_visible_gpus(nvidia_smi=nvidia_smi):
        if snapshot.index == gpu:
            return snapshot
    raise ValueError(f"GPU {gpu} is not currently visible to nvidia-smi")


def snapshot_record(snapshot: GpuSnapshot) -> dict[str, object]:
    record = asdict(snapshot)
    record["free_memory_mib"] = snapshot.free_memory_mib
    return record


def _positive_integral(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    integer = int(value)
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def resource_peak_memory_mib(job: dict[str, object]) -> int:
    resource = job.get("resource_estimate")
    if not isinstance(resource, dict):
        raise TypeError("resource_estimate must be a mapping")
    # measured_peak_mib is preferred. The other names preserve compatibility
    # with older project queue entries while applying the same contract.
    for key in ("measured_peak_mib", "peak_memory_mib", "gpu_memory_mib", "memory_mib"):
        if key in resource:
            return _positive_integral(resource[key], name=f"resource_estimate.{key}")
    raise ValueError("resource_estimate needs measured_peak_mib or a compatible peak-memory field")


def safety_margin_mib(measured_peak_mib: int) -> int:
    peak = _positive_integral(measured_peak_mib, name="measured_peak_mib")
    return max(DEFAULT_SAFETY_MARGIN_MIB, math.ceil(SAFETY_MARGIN_FRACTION * peak))


def admission_required_mib(measured_peak_mib: int) -> int:
    return measured_peak_mib + safety_margin_mib(measured_peak_mib)


def validate_job(job: dict[str, object]) -> None:
    required = {"job_id", "protocol_id", "output_root", "command", "resource_estimate"}
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"queue entry missing required fields: {missing}")
    for key in ("physical_gpu", "preferred_gpu"):
        if key in job and (not isinstance(job[key], int) or isinstance(job[key], bool) or int(job[key]) < 0):
            raise ValueError(f"queue {key} must be a non-negative integer when supplied")
    if "physical_gpu" in job and "preferred_gpu" in job and job["physical_gpu"] != job["preferred_gpu"]:
        raise ValueError("physical_gpu and preferred_gpu disagree")
    command = job["command"]
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("queue command must be a non-empty argv list")
    output_root = job["output_root"]
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise ValueError("queue output_root must be an absolute path")
    for key in ("job_id", "protocol_id"):
        if not isinstance(job[key], str) or not job[key]:
            raise ValueError(f"queue {key} must be a non-empty string")
    resource_peak_memory_mib(job)
    max_requeues = job.get("max_oom_requeues", DEFAULT_MAX_OOM_REQUEUES)
    if not isinstance(max_requeues, int) or isinstance(max_requeues, bool) or max_requeues < 0:
        raise ValueError("max_oom_requeues must be a non-negative integer")
    if "environment" in job and (
        not isinstance(job["environment"], dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in job["environment"].items())
    ):
        raise TypeError("environment must be a string-to-string mapping")


def preferred_gpu(job: dict[str, object]) -> int | None:
    value = job.get("preferred_gpu", job.get("physical_gpu"))
    return int(value) if value is not None else None


class CapacityUnavailable(RuntimeError):
    pass


def rank_gpus(snapshots: Iterable[GpuSnapshot], required_memory_mib: int, *, preferred: int | None = None) -> list[GpuSnapshot]:
    candidates = [snapshot for snapshot in snapshots if snapshot.capacity_ok(required_memory_mib)]
    if not candidates:
        raise CapacityUnavailable(f"no visible GPU satisfies required free memory {required_memory_mib} MiB")
    return sorted(
        candidates,
        key=lambda snapshot: (
            0 if preferred is not None and snapshot.index == preferred else 1,
            -snapshot.free_memory_mib,
            snapshot.utilization_percent,
            snapshot.index,
        ),
    )


def choose_gpu(snapshots: Iterable[GpuSnapshot], required_memory_mib: int, *, preferred: int | None = None) -> GpuSnapshot:
    return rank_gpus(snapshots, required_memory_mib, preferred=preferred)[0]


SnapshotFn = Callable[[int], GpuSnapshot]
VisibleSnapshotsFn = Callable[[], list[GpuSnapshot]]
SleepFn = Callable[[float], None]


def wait_for_capacity(
    snapshots_fn: VisibleSnapshotsFn,
    measured_peak_mib: int,
    *,
    preferred: int | None = None,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 0.0,
    sleep_fn: SleepFn = time.sleep,
    rng: random.Random | None = None,
) -> tuple[list[GpuSnapshot], GpuSnapshot]:
    """Wait for any visible GPU to meet the memory contract.

    Utilization is intentionally absent from admission. It is only used by
    ``choose_gpu`` after the free-memory contract has passed.
    """

    peak = _positive_integral(measured_peak_mib, name="measured_peak_mib")
    if poll_seconds < 0 or max_wait_seconds < 0:
        raise ValueError("poll_seconds and max_wait_seconds must be non-negative")
    required = admission_required_mib(peak)
    randomizer = rng or random.Random()
    started = time.monotonic()
    while True:
        snapshots = list(snapshots_fn())
        try:
            selected = choose_gpu(snapshots, required, preferred=preferred)
            return snapshots, selected
        except CapacityUnavailable:
            if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
                raise TimeoutError(f"no visible GPU met capacity contract within {max_wait_seconds}s")
            jitter = randomizer.uniform(0.0, min(max(poll_seconds, 0.01) * 0.1, 5.0))
            sleep_fn(max(poll_seconds, 0.01) + jitter)


def command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


class LocalGpuLease:
    """A process-scoped advisory lock for this Unix account's GPU lease."""

    def __init__(self, path: Path, gpu: int):
        self.path = path
        self.gpu = gpu
        self.handle: TextIO | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            return False
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "gpu": self.gpu, "acquired_utc": utc_now()}) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self) -> Self:
        if not self.acquire():
            raise BlockingIOError(f"GPU lease is held: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class ProjectChildSlot:
    """Account-local slots; the task launcher supplies a live visible-GPU count."""

    def __init__(self, lease_dir: Path, slot_count: int):
        if not isinstance(slot_count, int) or slot_count <= 0:
            raise ValueError("slot_count must equal a positive live visible-GPU count")
        self.lease_dir = lease_dir
        self.slot_count = slot_count
        self.handle: TextIO | None = None
        self.slot_index: int | None = None

    def acquire(self, *, job_id: str) -> bool:
        if self.handle is not None:
            return False
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        for index in range(self.slot_count):
            path = self.lease_dir / f"project_child_{index}.lease"
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "job_id": job_id, "slot": index, "acquired_utc": utc_now()}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            self.handle = handle
            self.slot_index = index
            return True
        return False

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None
        self.slot_index = None


def _signal_number(returncode: int | None) -> int | None:
    return -returncode if returncode is not None and returncode < 0 else None


def launch_child(job: dict[str, object], status_path: Path, *, cwd: Path | None = None) -> int:
    validate_job(job)
    if "physical_gpu" not in job:
        raise ValueError("launch_child requires a selected physical_gpu")
    gpu = int(job["physical_gpu"])
    command = list(job["command"])
    log_path = Path(str(job.get("log_path", Path(str(job["output_root"])) / "watcher_child.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in job.get("environment", {}).items()})
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["GPU_WATCHER_JOB_ID"] = str(job["job_id"])
    environment["GPU_WATCHER_PHYSICAL_GPU"] = str(gpu)
    inherited_status: dict[str, object] = {}
    if status_path.exists():
        inherited_status = json.loads(status_path.read_text(encoding="utf-8"))
    status = {
        **inherited_status,
        "status": "launching",
        "job_id": job["job_id"],
        "protocol_id": job["protocol_id"],
        "physical_gpu": gpu,
        "visible_gpu_pool": GPU_POOL_MODE,
        "command_hash": command_hash(command),
        "output_root": job["output_root"],
        "log_path": str(log_path),
        "started_utc": utc_now(),
        "watcher_pid": os.getpid(),
        "non_child_process_signaled": False,
    }
    atomic_write_json(status_path, status)
    child: subprocess.Popen[str] | None = None
    original_handler = signal.getsignal(signal.SIGTERM)
    returncode: int | None = None

    def forward_to_owned_child(signum: int, _frame) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_to_owned_child)
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            child = subprocess.Popen(command, cwd=str(cwd) if cwd else None, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            status.update({"status": "running", "child_pid": child.pid, "launch_utc": utc_now()})
            atomic_write_json(status_path, status)
            returncode = child.wait()
    finally:
        signal.signal(signal.SIGTERM, original_handler)
    if returncode is None:
        raise RuntimeError("owned child returned without an exit code")
    status.update({"status": "complete" if returncode == 0 else "failed", "exit_code": returncode if returncode >= 0 else None, "signal": _signal_number(returncode), "completed_utc": utc_now()})
    atomic_write_json(status_path, status)
    return returncode


def child_oom_detected(returncode: int, log_path: Path) -> bool:
    if returncode == 137:
        return True
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _sleep_or_timeout(started: float, max_wait_seconds: float, poll_seconds: float, sleep_fn: SleepFn) -> bool:
    if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
        return True
    sleep_fn(max(poll_seconds, 0.01))
    return False


def run_registered_job(
    job: dict[str, object],
    *,
    lease_dir: Path,
    status_path: Path,
    visible_snapshots_fn: VisibleSnapshotsFn | None = None,
    snapshot_fn: SnapshotFn | None = None,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 0.0,
    nvidia_smi: str = "nvidia-smi",
    max_oom_requeues: int | None = None,
) -> int:
    validate_job(job)
    if poll_seconds < 0 or max_wait_seconds < 0:
        raise ValueError("poll_seconds and max_wait_seconds must be non-negative")
    peak = resource_peak_memory_mib(job)
    required = admission_required_mib(peak)
    requeue_limit = job.get("max_oom_requeues", DEFAULT_MAX_OOM_REQUEUES) if max_oom_requeues is None else max_oom_requeues
    if not isinstance(requeue_limit, int) or isinstance(requeue_limit, bool) or requeue_limit < 0:
        raise ValueError("max_oom_requeues must be a non-negative integer")
    provider = visible_snapshots_fn or (lambda: snapshot_visible_gpus(nvidia_smi=nvidia_smi))
    started = time.monotonic()
    oom_requeues = 0
    status_base: dict[str, object] = {
        "status": "waiting",
        "job_id": job["job_id"],
        "protocol_id": job["protocol_id"],
        "preferred_gpu": preferred_gpu(job),
        "visible_gpu_pool": GPU_POOL_MODE,
        "one_child_per_gpu": ONE_CHILD_PER_GPU,
        "max_concurrent_project_gpu_children": "dynamic_visible_gpu_count_at_admission",
        "measured_peak_mib": peak,
        "safety_margin_mib": safety_margin_mib(peak),
        "required_free_memory_mib": required,
        "non_child_process_signaled": False,
        "external_process_identity_recorded": False,
        "oom_requeue_count": 0,
        "wait_started_utc": utc_now(),
    }
    atomic_write_json(status_path, status_base)
    while True:
        try:
            snapshots = list(provider())
            status_base["max_concurrent_project_gpu_children"] = len(snapshots)
            candidates = rank_gpus(snapshots, required, preferred=preferred_gpu(job))
        except CapacityUnavailable as exc:
            status_base.update({"status": "waiting_for_memory_capacity", "capacity_error": str(exc), "visible_gpu_count": len(snapshots) if "snapshots" in locals() else 0})
            atomic_write_json(status_path, status_base)
            if _sleep_or_timeout(started, max_wait_seconds, poll_seconds, time.sleep):
                status_base.update({"status": "capacity_timeout", "completed_utc": utc_now()})
                atomic_write_json(status_path, status_base)
                return 125
            continue
        except Exception as exc:
            status_base.update({"status": "telemetry_error", "error": f"{type(exc).__name__}: {exc}", "completed_utc": utc_now()})
            atomic_write_json(status_path, status_base)
            return 125

        slot = ProjectChildSlot(lease_dir, slot_count=len(snapshots))
        if not slot.acquire(job_id=str(job["job_id"])):
            status_base.update({"status": "waiting_for_project_child_slot", "selected_gpu": candidates[0].index, "visible_snapshots": [snapshot_record(item) for item in snapshots]})
            atomic_write_json(status_path, status_base)
            if _sleep_or_timeout(started, max_wait_seconds, poll_seconds, time.sleep):
                status_base.update({"status": "project_child_slot_timeout", "completed_utc": utc_now()})
                atomic_write_json(status_path, status_base)
                return 125
            continue

        try:
            last_wait_status = "waiting_for_local_gpu_lease"
            for selected in candidates:
                gpu_lease = LocalGpuLease(lease_dir / f"gpu_{selected.index}.lease", selected.index)
                if not gpu_lease.acquire():
                    continue
                try:
                    try:
                        final_recheck = snapshot_fn(selected.index) if snapshot_fn is not None else snapshot_gpu(selected.index, nvidia_smi=nvidia_smi)
                    except Exception as exc:
                        status_base.update({"status": "telemetry_error", "error": f"{type(exc).__name__}: {exc}", "completed_utc": utc_now()})
                        atomic_write_json(status_path, status_base)
                        return 125
                    if not final_recheck.capacity_ok(required):
                        last_wait_status = "waiting_final_memory_recheck_race"
                        status_base.update({"status": last_wait_status, "selected_gpu": selected.index, "final_recheck": snapshot_record(final_recheck)})
                        atomic_write_json(status_path, status_base)
                        continue

                    launch_job = dict(job)
                    launch_job.pop("preferred_gpu", None)  # ranking hint is resolved; child receives only the assigned physical GPU
                    launch_job["physical_gpu"] = selected.index
                    status_base.update({"status": "launching", "selected_gpu": selected.index, "visible_snapshots": [snapshot_record(item) for item in snapshots], "final_recheck": snapshot_record(final_recheck), "lease_path": str(gpu_lease.path), "lease_acquired_utc": utc_now(), "project_child_slot": slot.slot_index})
                    atomic_write_json(status_path, status_base)
                    returncode = launch_child(launch_job, status_path)
                    log_path = Path(str(launch_job.get("log_path", Path(str(launch_job["output_root"])) / "watcher_child.log")))
                    if child_oom_detected(returncode, log_path):
                        if oom_requeues < requeue_limit:
                            oom_requeues += 1
                            status_base.update({"status": "oom_requeued", "oom_requeue_count": oom_requeues, "last_oom_physical_gpu": selected.index})
                            atomic_write_json(status_path, status_base)
                            continue
                        status_base.update({"status": "oom_exhausted", "oom_requeue_count": oom_requeues, "completed_utc": utc_now()})
                        atomic_write_json(status_path, status_base)
                    return returncode
                finally:
                    gpu_lease.release()
            status_base.update({"status": last_wait_status, "selected_gpu_candidates": [item.index for item in candidates]})
            atomic_write_json(status_path, status_base)
            if _sleep_or_timeout(started, max_wait_seconds, poll_seconds, time.sleep):
                status_base.update({"status": "local_gpu_lease_or_recheck_timeout", "completed_utc": utc_now()})
                atomic_write_json(status_path, status_base)
                return 125
            continue
        finally:
            slot.release()
            current = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            current["lease_released"] = True
            current["project_child_slot_released"] = True
            current["lease_released_utc"] = utc_now()
            atomic_write_json(status_path, current)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--lease-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--max-oom-requeues", type=int, default=DEFAULT_MAX_OOM_REQUEUES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    validate_job(job)
    if args.dry_run:
        peak = resource_peak_memory_mib(job)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "job_id": job["job_id"],
                    "preferred_gpu": preferred_gpu(job),
                    "visible_gpu_pool": GPU_POOL_MODE,
                    "measured_peak_mib": peak,
                    "safety_margin_mib": safety_margin_mib(peak),
                    "required_free_memory_mib": admission_required_mib(peak),
                    "max_concurrent_project_gpu_children": "dynamic_visible_gpu_count_at_admission",
                    "command_hash": command_hash(job["command"]),
                },
                sort_keys=True,
            )
        )
        return 0
    return run_registered_job(
        job,
        lease_dir=args.lease_dir,
        status_path=args.status_path,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        nvidia_smi=args.nvidia_smi,
        max_oom_requeues=args.max_oom_requeues,
    )


if __name__ == "__main__":
    raise SystemExit(main())
