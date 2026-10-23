"""Task-local GPU2/3 whitelist adapter around the shared lease watcher.

The adapter never invokes a GPU query itself. The shared watcher receives the
filtered nvidia-smi executable plus filtered/mockable inventory callbacks, so
its ranking and lease code can never see or select a physical GPU outside
{2, 3}.
"""
from __future__ import annotations
import os

import importlib.util
from pathlib import Path
import sys
from typing import Callable, Iterable

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
WATCHER_PATH = PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"
FILTER_PATH = OUT / "code/gpu23_filter.py"
ALLOWED_PHYSICAL_GPUS = frozenset({2, 3})


def _load_watcher():
    spec = importlib.util.spec_from_file_location("d3eq_task_gpu_watcher", WATCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared watcher: {WATCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def allowed_snapshots(values: Iterable[object]) -> list[object]:
    result = []
    seen: set[int] = set()
    for item in values:
        index = int(getattr(item, "index"))
        if index in seen:
            raise ValueError(f"duplicate physical GPU index in inventory: {index}")
        seen.add(index)
        if index in ALLOWED_PHYSICAL_GPUS:
            result.append(item)
    return sorted(result, key=lambda item: int(getattr(item, "index")))


def run_registered_job_gpu23(
    job: dict[str, object],
    *,
    lease_dir: Path,
    status_path: Path,
    visible_snapshots_fn: Callable[[], Iterable[object]],
    snapshot_fn: Callable[[int], object],
    poll_seconds: float = 1.0,
    max_wait_seconds: float = 0.0,
) -> int:
    """Run one watcher job, exposing only allowlisted physical indices."""
    watcher = _load_watcher()
    provider = lambda: allowed_snapshots(visible_snapshots_fn())

    def strict_snapshot(index: int):
        if index not in ALLOWED_PHYSICAL_GPUS:
            raise ValueError(f"watcher attempted disallowed physical GPU {index}")
        value = snapshot_fn(index)
        if int(getattr(value, "index")) != index:
            raise ValueError("final-recheck GPU index mismatch")
        return value

    if not str(job.get("output_root", "")).startswith(str(OUT)):
        raise ValueError("task GPU job output_root must be inside this task's OUT_ROOT")
    if job.get("protocol_id") != "D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1":
        raise ValueError("unexpected protocol id for task-local GPU lease")
    result = watcher.run_registered_job(
        job,
        lease_dir=lease_dir,
        status_path=status_path,
        visible_snapshots_fn=provider,
        snapshot_fn=strict_snapshot,
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
        nvidia_smi=str(FILTER_PATH),
        max_oom_requeues=0,
    )
    if status_path.is_file():
        import json

        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["task_allowed_physical_gpus"] = sorted(ALLOWED_PHYSICAL_GPUS)
        status["task_visible_pool_mode"] = "fixed_allowlist_2_3"
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(result)
