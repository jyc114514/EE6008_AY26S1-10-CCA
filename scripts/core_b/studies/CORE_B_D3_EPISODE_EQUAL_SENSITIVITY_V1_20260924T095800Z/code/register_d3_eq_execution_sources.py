"""Bind the guarded D3-EQ launcher, job builder and their pre-run configs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = EQ_ROOT / "protocol/D3_EQ_SOURCE_HASHES.json"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
AUDIT_PATH = EQ_ROOT / "protocol/D3_EQ_LOCK_FINALIZATION_AUDIT.json"
STATUS_PATH = EQ_ROOT / "STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.register.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("execution source registry can only be extended before any GPU run")
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    for key, path in (
        ("numerical_edge_case_test", EQ_ROOT / "code/test_d3_eq_numerical_edges.py"),
        ("numerical_edge_case_result", EQ_ROOT / "qa/D3_EQ_NUMERICAL_EDGE_TESTS.json"),
        ("execution_job_builder", EQ_ROOT / "code/build_d3_eq_jobs.py"),
        ("watcher_queue_runner", EQ_ROOT / "code/run_d3_eq_queue.py"),
        ("guarded_tmux_queue_entry", EQ_ROOT / "code/run_d3_eq_queue_entry.py"),
        ("execution_source_registrar", Path(__file__).resolve()),
        ("pre_stage0_resource_estimate_csv", EQ_ROOT / "RESOURCE_ESTIMATE.csv"),
        ("stage0_job_manifest", EQ_ROOT / "configs/D3_EQ_STAGE0_JOBS.json"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        sources["sources"][key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    sources["generated_utc"] = utc_now()
    atomic_json(SOURCE_PATH, sources)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["source_hashes_sha256"] = sha256_file(SOURCE_PATH)
    lock["canonical_protocol_sha256"] = sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"})
    atomic_json(LOCK_PATH, lock)

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit.update({
        "execution_sources_registered_utc": utc_now(),
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(SOURCE_PATH),
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"}),
        "final_lock_sha256": sha256_file(LOCK_PATH),
        "training_started": False,
    })
    atomic_json(AUDIT_PATH, audit)
    status.update({
        "updated_utc": utc_now(),
        "source_hashes_complete": True,
        "stage0_job_specs": "12_GENERATED_DRY_RUN_PENDING",
        "gpu_state_query": "not_run_requires_explicit_host_approval",
        "gpu_training_started": False,
        "protocol_file_sha256": sha256_file(LOCK_PATH),
    })
    atomic_json(STATUS_PATH, status)
    print(json.dumps({"status": "PASS", "source_count": len(sources["sources"]), "source_hashes_sha256": lock["source_hashes_sha256"], "protocol_sha256": sha256_file(LOCK_PATH)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
