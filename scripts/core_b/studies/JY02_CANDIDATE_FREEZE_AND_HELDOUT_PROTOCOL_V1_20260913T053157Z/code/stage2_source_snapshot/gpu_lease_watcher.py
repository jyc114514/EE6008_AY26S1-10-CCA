"""Safe first-come GPU2/3 leasing and child-process launching.

The watcher observes aggregate ``nvidia-smi`` telemetry only.  It never queries
process tables, sends signals to non-child processes, or treats a local lease
as ownership of another user's device.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Self, TextIO

ALLOWED_GPUS = (2, 3)
DEFAULT_IDLE_UTILIZATION_PERCENT = 5
DEFAULT_IDLE_MEMORY_MIB = 512
DEFAULT_IDLE_SAMPLES = 3


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

    def idle(self, *, utilization_limit: int, memory_limit_mib: int) -> bool:
        return self.utilization_percent <= utilization_limit and self.memory_used_mib <= memory_limit_mib


def snapshot_gpu(gpu: int, *, nvidia_smi: str = "nvidia-smi") -> GpuSnapshot:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {gpu} is outside allowlist {ALLOWED_GPUS}")
    query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(
        [nvidia_smi, "-i", str(gpu), f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("aggregate nvidia-smi query did not return exactly one row")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("aggregate nvidia-smi row has an unexpected schema")
    return GpuSnapshot(
        index=int(fields[0]),
        uuid=fields[1],
        name=fields[2],
        memory_used_mib=int(fields[3]),
        memory_total_mib=int(fields[4]),
        utilization_percent=int(fields[5]),
        utc=utc_now(),
    )


def validate_job(job: dict[str, object]) -> None:
    required = {"job_id", "physical_gpu", "protocol_id", "output_root", "command", "resource_estimate"}
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"queue entry missing required fields: {missing}")
    gpu = job["physical_gpu"]
    if not isinstance(gpu, int) or gpu not in ALLOWED_GPUS:
        raise ValueError(f"queue entry GPU must be one of {ALLOWED_GPUS}")
    command = job["command"]
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("queue command must be a non-empty argv list")
    output_root = job["output_root"]
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise ValueError("queue output_root must be an absolute path")
    for key in ("job_id", "protocol_id"):
        if not isinstance(job[key], str) or not job[key]:
            raise ValueError(f"queue {key} must be a non-empty string")
    if not isinstance(job["resource_estimate"], dict):
        raise TypeError("resource_estimate must be a mapping")
    if "environment" in job and (not isinstance(job["environment"], dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in job["environment"].items())):
        raise TypeError("environment must be a string-to-string mapping")


def command_hash(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


class LocalGpuLease:
    """A process-scoped advisory lock for this Unix account's watchers."""

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


SnapshotFn = Callable[[int], GpuSnapshot]
SleepFn = Callable[[float], None]


def wait_for_idle(
    gpu: int,
    snapshot_fn: SnapshotFn,
    *,
    required_samples: int = DEFAULT_IDLE_SAMPLES,
    utilization_limit: int = DEFAULT_IDLE_UTILIZATION_PERCENT,
    memory_limit_mib: int = DEFAULT_IDLE_MEMORY_MIB,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 0.0,
    sleep_fn: SleepFn = time.sleep,
    rng: random.Random | None = None,
) -> tuple[list[GpuSnapshot], GpuSnapshot]:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {gpu} is outside allowlist {ALLOWED_GPUS}")
    if required_samples <= 0 or poll_seconds < 0 or max_wait_seconds < 0:
        raise ValueError("idle samples and wait intervals must be non-negative/positive")
    randomizer = rng or random.Random()
    started = time.monotonic()
    consecutive: list[GpuSnapshot] = []
    backoff = max(poll_seconds, 0.01)
    while True:
        snapshot = snapshot_fn(gpu)
        if snapshot.idle(utilization_limit=utilization_limit, memory_limit_mib=memory_limit_mib):
            consecutive.append(snapshot)
            if len(consecutive) >= required_samples:
                return consecutive, snapshot
        else:
            consecutive = []
        if max_wait_seconds and time.monotonic() - started >= max_wait_seconds:
            raise TimeoutError(f"GPU {gpu} did not pass idle gate within {max_wait_seconds}s")
        jitter = randomizer.uniform(0.0, min(backoff * 0.1, 5.0))
        sleep_fn(backoff + jitter)
        backoff = min(max(backoff * 2.0, poll_seconds), 300.0)


def _signal_number(returncode: int | None) -> int | None:
    return -returncode if returncode is not None and returncode < 0 else None


def launch_child(job: dict[str, object], status_path: Path, *, cwd: Path | None = None) -> int:
    validate_job(job)
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
    status = {
        "status": "launching",
        "job_id": job["job_id"],
        "protocol_id": job["protocol_id"],
        "physical_gpu": gpu,
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
    status.update({"status": "complete" if returncode == 0 else "failed", "exit_code": returncode if returncode >= 0 else None, "signal": _signal_number(returncode), "completed_utc": utc_now()})
    atomic_write_json(status_path, status)
    return returncode


def run_registered_job(
    job: dict[str, object],
    *,
    lease_dir: Path,
    status_path: Path,
    snapshot_fn: SnapshotFn = snapshot_gpu,
    required_samples: int = DEFAULT_IDLE_SAMPLES,
    utilization_limit: int = DEFAULT_IDLE_UTILIZATION_PERCENT,
    memory_limit_mib: int = DEFAULT_IDLE_MEMORY_MIB,
    poll_seconds: float = 30.0,
    max_wait_seconds: float = 0.0,
    nvidia_smi: str = "nvidia-smi",
) -> int:
    validate_job(job)
    gpu = int(job["physical_gpu"])
    effective_snapshot = snapshot_fn if snapshot_fn is not snapshot_gpu else lambda device: snapshot_gpu(device, nvidia_smi=nvidia_smi)
    status_base = {"status": "waiting", "job_id": job["job_id"], "physical_gpu": gpu, "protocol_id": job["protocol_id"], "allowlist": list(ALLOWED_GPUS), "non_child_process_signaled": False, "other_user_process_inspected": False, "wait_started_utc": utc_now()}
    atomic_write_json(status_path, status_base)
    while True:
        try:
            samples, _final_before_lease = wait_for_idle(
                gpu,
                effective_snapshot,
                required_samples=required_samples,
                utilization_limit=utilization_limit,
                memory_limit_mib=memory_limit_mib,
                poll_seconds=poll_seconds,
                max_wait_seconds=max_wait_seconds,
            )
        except TimeoutError as exc:
            status_base.update({"status": "waiting_timeout", "error": str(exc), "completed_utc": utc_now()})
            atomic_write_json(status_path, status_base)
            return 125
        lease = LocalGpuLease(lease_dir / f"gpu_{gpu}.lease", gpu)
        if not lease.acquire():
            status_base.update({"status": "waiting_for_local_lease", "idle_samples": [asdict(item) for item in samples]})
            atomic_write_json(status_path, status_base)
            if poll_seconds:
                time.sleep(poll_seconds)
            continue
        try:
            final_recheck = effective_snapshot(gpu)
            if not final_recheck.idle(utilization_limit=utilization_limit, memory_limit_mib=memory_limit_mib):
                status_base.update({"status": "waiting_final_recheck_race", "idle_samples": [asdict(item) for item in samples], "final_recheck": asdict(final_recheck)})
                atomic_write_json(status_path, status_base)
                if poll_seconds:
                    time.sleep(poll_seconds)
                continue
            status_base.update({"status": "launching", "idle_samples": [asdict(item) for item in samples], "final_recheck": asdict(final_recheck), "lease_path": str(lease.path), "lease_acquired_utc": utc_now()})
            atomic_write_json(status_path, status_base)
            return launch_child(job, status_path)
        finally:
            lease.release()
            final_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            final_status["lease_released"] = True
            final_status["lease_released_utc"] = utc_now()
            atomic_write_json(status_path, final_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--lease-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=0.0)
    parser.add_argument("--idle-utilization-percent", type=int, default=DEFAULT_IDLE_UTILIZATION_PERCENT)
    parser.add_argument("--idle-memory-mib", type=int, default=DEFAULT_IDLE_MEMORY_MIB)
    parser.add_argument("--idle-samples", type=int, default=DEFAULT_IDLE_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    validate_job(job)
    if args.dry_run:
        print(json.dumps({"status": "valid", "job_id": job["job_id"], "physical_gpu": job["physical_gpu"], "command_hash": command_hash(job["command"])}, sort_keys=True))
        return 0
    return run_registered_job(
        job,
        lease_dir=args.lease_dir,
        status_path=args.status_path,
        required_samples=args.idle_samples,
        utilization_limit=args.idle_utilization_percent,
        memory_limit_mib=args.idle_memory_mib,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        nvidia_smi=args.nvidia_smi,
    )


if __name__ == "__main__":
    raise SystemExit(main())
