#!/usr/bin/env python3
"""Task-local GPU1/2/3 watcher for Route C.

This is intentionally isolated under EXP_ROOT.  It preserves the shared
watcher's non-preemption behavior while adding the explicitly authorized GPU1
for this task only; the global watcher and project policy are untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = (1, 2, 3)
IDLE_UTIL = 5
IDLE_MEM = 512
IDLE_SAMPLES = 3
HEARTBEAT_SECONDS = 30.0


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(part, path)


@dataclass(frozen=True)
class Snapshot:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    utc: str

    def idle(self) -> bool:
        return self.utilization_percent <= IDLE_UTIL and self.memory_used_mib <= IDLE_MEM


def snapshot(gpu: int) -> Snapshot:
    if gpu not in ALLOWED:
        raise ValueError(f"GPU {gpu} outside task-local allowlist {ALLOWED}")
    query = "index,uuid,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("unexpected nvidia-smi aggregate schema")
    return Snapshot(int(fields[0]), fields[1], fields[2], int(fields[3]), int(fields[4]), int(fields[5]), now())


def acquire(path: Path) -> object | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid": os.getpid(), "acquired_utc": now()}) + "\n"); handle.flush(); os.fsync(handle.fileno())
    return handle


def wait_idle(gpu: int, *, poll_seconds: float, status_path: Path) -> tuple[list[Snapshot], Snapshot]:
    samples: list[Snapshot] = []
    backoff = max(poll_seconds, 1.0)
    while True:
        current = snapshot(gpu)
        status = {"status": "waiting", "physical_gpu": gpu, "updated_utc": now(), "idle_samples": [asdict(item) for item in samples], "latest": asdict(current), "allowlist": list(ALLOWED), "non_child_process_signaled": False}
        atomic_json(status_path, status)
        if current.idle():
            samples.append(current)
            if len(samples) == IDLE_SAMPLES:
                final = snapshot(gpu)
                if final.idle():
                    return samples, final
                samples = []
        else:
            samples = []
        time.sleep(backoff + random.uniform(0, min(0.1 * backoff, 5.0)))
        backoff = min(backoff * 1.25, 300.0)


def launch(
    job: dict[str, object],
    status_path: Path,
    cwd: Path | None,
    *,
    gpu_uuid: str,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> int:
    gpu = int(job["physical_gpu"])
    if gpu not in ALLOWED or not gpu_uuid:
        raise ValueError("invalid task-local GPU binding")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat interval must be positive")
    command = [str(item) for item in job["command"]]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in job.get("environment", {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["GPU_WATCHER_PHYSICAL_GPU"] = str(gpu)
    env["GPU_WATCHER_GPU_UUID"] = gpu_uuid
    env["ROUTE_C_TASK_LOCAL_GPU"] = str(gpu)
    env["ROUTE_C_TASK_LOCAL_GPU_UUID"] = gpu_uuid
    log_path = Path(str(job["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child = None
    original = signal.getsignal(signal.SIGTERM)
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    heartbeat_count = 0

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    status = {
        "status": "launching",
        "job_id": job["job_id"],
        "protocol_id": job["protocol_id"],
        "physical_gpu": gpu,
        "gpu_uuid": gpu_uuid,
        "command": command,
        "output_root": job["output_root"],
        "log_path": str(log_path),
        "launch_utc": now(),
        "heartbeat_seconds": heartbeat_seconds,
        "heartbeat_count": heartbeat_count,
        "non_child_process_signaled": False,
    }
    atomic_json(status_path, status)

    def heartbeat() -> None:
        nonlocal heartbeat_count
        while not heartbeat_stop.wait(heartbeat_seconds):
            heartbeat_count += 1
            status["heartbeat_count"] = heartbeat_count
            status["heartbeat_utc"] = now()
            atomic_json(status_path, dict(status))

    try:
        with log_path.open("w") as handle:
            child = subprocess.Popen(command, cwd=str(cwd) if cwd else None, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            status.update({"status": "running", "child_pid": child.pid, "heartbeat_utc": now()})
            atomic_json(status_path, status)
            heartbeat_thread = threading.Thread(target=heartbeat, name="route-c-watcher-heartbeat", daemon=True)
            heartbeat_thread.start()
            code = child.wait()
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join()
        signal.signal(signal.SIGTERM, original)
    status.update({"status": "complete" if code == 0 else "failed", "exit_code": code, "completed_utc": now(), "heartbeat_count": heartbeat_count, "non_child_process_signaled": False})
    atomic_json(status_path, status)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--job-json", type=Path, required=True); parser.add_argument("--status-path", type=Path, required=True); parser.add_argument("--lease-dir", type=Path, required=True); parser.add_argument("--poll-seconds", type=float, default=30.0); parser.add_argument("--cwd", type=Path, default=None); args = parser.parse_args()
    job = json.loads(args.job_json.read_text())
    gpu = int(job["physical_gpu"])
    if gpu not in ALLOWED or not Path(str(job["output_root"])).is_absolute() or not isinstance(job.get("command"), list) or not job.get("protocol_id"):
        raise ValueError("invalid task-local Route C job")
    samples, final = wait_idle(gpu, poll_seconds=args.poll_seconds, status_path=args.status_path)
    gpu_lock = acquire(args.lease_dir / f"gpu_{gpu}.lease")
    slot = None
    if gpu_lock is not None:
        for index in range(2):
            slot = acquire(args.lease_dir / f"child_slot_{index}.lease")
            if slot is not None:
                break
    if gpu_lock is None or slot is None:
        if gpu_lock is not None: gpu_lock.close()
        if slot is not None: slot.close()
        atomic_json(args.status_path, {"status": "waiting_local_lease", "physical_gpu": gpu, "updated_utc": now(), "non_child_process_signaled": False})
        return 125
    try:
        final = snapshot(gpu)
        if not final.idle():
            atomic_json(args.status_path, {"status": "waiting_final_recheck_race", "physical_gpu": gpu, "final": asdict(final), "non_child_process_signaled": False})
            return 125
        atomic_json(args.status_path, {"status": "launching", "physical_gpu": gpu, "gpu_uuid": final.uuid, "idle_samples": [asdict(item) for item in samples], "final_recheck": asdict(final), "non_child_process_signaled": False})
        return launch(job, args.status_path, args.cwd, gpu_uuid=final.uuid)
    finally:
        fcntl.flock(slot.fileno(), fcntl.LOCK_UN); slot.close(); fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN); gpu_lock.close()
        current = json.loads(args.status_path.read_text()) if args.status_path.exists() else {}
        current.update({"lease_released": True, "lease_released_utc": now(), "non_child_process_signaled": False}); atomic_json(args.status_path, current)


if __name__ == "__main__":
    raise SystemExit(main())
