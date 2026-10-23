"""Finalize and self-audit the new D3-EQ lock before any run is launched."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
SOURCES_PATH = EQ_ROOT / "protocol/D3_EQ_SOURCE_HASHES.json"
SEEDS_PATH = EQ_ROOT / "protocol/D3_EQ_SEED_REGISTRY.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.finalize.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    previous_lock_sha = sha256_file(LOCK_PATH)
    if lock.get("status") != "LOCKED" or lock.get("final_test_access") != 0:
        raise AssertionError("new lock is not in the pre-execution locked state")
    status = json.loads((EQ_ROOT / "STATUS.json").read_text(encoding="utf-8"))
    if status.get("gpu_training_started") is not False or status.get("stage0") != "0/12":
        raise AssertionError("cannot finalize a lock after execution has begun")

    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    fit_ids_sha = normalizers.get("fit_ids_sha256")
    if not isinstance(fit_ids_sha, str) or len(fit_ids_sha) != 64:
        raise AssertionError("V3 normalizer fit-ID hash is missing")
    # The generator's first draft wrote a boolean audit flag into this field;
    # replace it with the actual locked fit-ID digest before any run starts.
    prior_value = lock["target_cache"]["normalizer_fit_ids_sha256"]
    lock["target_cache"]["normalizer_fit_ids_sha256"] = fit_ids_sha

    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    source_items = sources["sources"]
    for key, path in (
        ("provenance_note", EQ_ROOT / "ORIGINAL_TASK_PROVENANCE.md"),
        ("methodology_diff", EQ_ROOT / "METHODOLOGY_DIFF.md"),
        ("protocol_lock_generator", EQ_ROOT / "code/lock_d3_eq_protocol.py"),
        ("protocol_lock_finalizer", Path(__file__).resolve()),
        ("locked_seed_registry", SEEDS_PATH),
        ("locked_matrix", EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv"),
    ):
        source_items[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    sources["generated_utc"] = utc_now()
    atomic_json(SOURCES_PATH, sources)

    lock["seed_registry_path"] = "D3_EQ_SEED_REGISTRY.json"
    lock["seed_registry_sha256"] = sha256_file(SEEDS_PATH)
    lock["source_hashes_sha256"] = sha256_file(SOURCES_PATH)
    lock["canonical_protocol_sha256"] = sha256_json({k: v for k, v in lock.items() if k != "canonical_protocol_sha256"})
    atomic_json(LOCK_PATH, lock)
    final_lock_sha = sha256_file(LOCK_PATH)
    audit = {
        "status": "PASS",
        "finalized_utc": utc_now(),
        "new_lock_only": True,
        "pre_finalization_lock_sha256": previous_lock_sha,
        "final_lock_sha256": final_lock_sha,
        "normalizer_fit_ids_sha256_before": prior_value,
        "normalizer_fit_ids_sha256_corrected_to": fit_ids_sha,
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({k: v for k, v in lock.items() if k != "canonical_protocol_sha256"}),
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(SOURCES_PATH),
        "old_lock_amended": False,
        "training_started": False,
        "final_test_access": 0,
    }
    audit_path = EQ_ROOT / "protocol/D3_EQ_LOCK_FINALIZATION_AUDIT.json"
    atomic_json(audit_path, audit)
    status.update({
        "updated_utc": utc_now(),
        "protocol_lock_finalization": "PASS",
        "protocol_file_sha256": final_lock_sha,
        "source_hashes_complete": True,
        "next_gate": "explicit host approval, node role check, current resource/GPU lease gate before Stage0",
    })
    atomic_json(EQ_ROOT / "STATUS.json", status)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
