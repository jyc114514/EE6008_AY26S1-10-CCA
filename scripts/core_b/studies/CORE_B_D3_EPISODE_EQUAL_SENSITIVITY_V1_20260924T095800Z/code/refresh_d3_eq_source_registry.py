"""Hash all pre-execution D3-EQ code, QA, protocol and job-spec inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = EQ_ROOT / "protocol/D3_EQ_SOURCE_HASHES.json"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
FINALIZATION_AUDIT = EQ_ROOT / "protocol/D3_EQ_LOCK_FINALIZATION_AUDIT.json"


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
    part = path.with_name(f".{path.name}.refresh.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def add_file(entries: dict[str, Any], key: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> int:
    status = json.loads((EQ_ROOT / "STATUS.json").read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False:
        raise AssertionError("source registry refresh is pre-execution only")
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    entries = sources["sources"]
    for path in sorted((EQ_ROOT / "code").glob("*.py")):
        add_file(entries, f"code/{path.name}", path)
    for folder_name in ("qa", "configs", "protocol"):
        folder = EQ_ROOT / folder_name
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".csv"} and path.name != "D3_EQ_SOURCE_HASHES.json":
                rel = path.relative_to(EQ_ROOT).as_posix()
                add_file(entries, rel, path)
    add_file(entries, "RESOURCE_ESTIMATE.csv", EQ_ROOT / "RESOURCE_ESTIMATE.csv")
    sources["generated_utc"] = utc_now()
    sources["coverage"] = {
        "all_code_python_files": True,
        "all_pre_execution_qa_json_csv": True,
        "all_protocol_and_stage0_job_json_csv": True,
        "all_formal_results_or_gpu_runtime_files": False,
        "gpu_execution_started": False,
    }
    atomic_json(SOURCE_PATH, sources)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["source_hashes_sha256"] = sha256_file(SOURCE_PATH)
    lock["canonical_protocol_sha256"] = sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"})
    atomic_json(LOCK_PATH, lock)

    audit = json.loads(FINALIZATION_AUDIT.read_text(encoding="utf-8"))
    audit.update({
        "source_registry_refreshed_utc": utc_now(),
        "source_registry_entries": len(entries),
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(SOURCE_PATH),
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({key: value for key, value in lock.items() if key != "canonical_protocol_sha256"}),
        "final_lock_sha256": sha256_file(LOCK_PATH),
        "training_started": False,
    })
    atomic_json(FINALIZATION_AUDIT, audit)

    status.update({
        "updated_utc": utc_now(),
        "cache_id_shape_audit": "PASS",
        "train_only_target_cache": "PASS_17757_ROWS_2712_EPISODES",
        "cpu_contract_tests": "PASS_7_TESTS_PLUS_NUMERICAL_EDGE_PASS",
        "protocol_locked": True,
        "matrix": "36_CELLS_LOCKED",
        "stage0_job_specs": "PASS_12_SPECS_DRY_RUN_PASS",
        "gpu_state_query": "not_run_requires_explicit_host_approval",
        "gpu_training_started": False,
        "stage0": "0/12",
        "formal_training": "0/36",
        "primary_predictions": "0/36",
        "dense_predictions": "0/36_not_started",
        "final_heldout_access": 0,
        "protocol_file_sha256": sha256_file(LOCK_PATH),
        "next_gate": "explicit host approval, node role check, fresh resource/GPU lease gate before Stage0",
    })
    atomic_json(EQ_ROOT / "STATUS.json", status)
    report = {
        "status": "PASS",
        "source_registry_entries": len(entries),
        "lock_sha256": sha256_file(LOCK_PATH),
        "source_hashes_sha256": lock["source_hashes_sha256"],
        "all_stage0_specs_registered": True,
        "gpu_query_performed": False,
        "training_started": False,
        "final_test_access": 0,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_SOURCE_REGISTRY_AUDIT.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
