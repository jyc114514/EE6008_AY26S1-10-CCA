"""Add late pre-execution QA source artifacts to the new D3-EQ lock hashes."""

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
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    part = path.with_name(f".{path.name}.augment.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    sources = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    for key, path in (
        ("numerical_edge_case_test", EQ_ROOT / "code/test_d3_eq_numerical_edges.py"),
        ("source_hash_augmenter", Path(__file__).resolve()),
    ):
        sources["sources"][key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    sources["generated_utc"] = utc_now()
    atomic_json(SOURCE_PATH, sources)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock["source_hashes_sha256"] = sha256_file(SOURCE_PATH)
    lock["canonical_protocol_sha256"] = sha256_json({k: v for k, v in lock.items() if k != "canonical_protocol_sha256"})
    atomic_json(LOCK_PATH, lock)

    audit = json.loads(FINALIZATION_AUDIT.read_text(encoding="utf-8"))
    audit.update({
        "source_hashes_augmented_utc": utc_now(),
        "source_hashes_sha256_matches": lock["source_hashes_sha256"] == sha256_file(SOURCE_PATH),
        "final_lock_sha256": sha256_file(LOCK_PATH),
        "canonical_protocol_sha256_valid": lock["canonical_protocol_sha256"] == sha256_json({k: v for k, v in lock.items() if k != "canonical_protocol_sha256"}),
        "numerical_edge_case_source_registered": True,
        "training_started": False,
    })
    atomic_json(FINALIZATION_AUDIT, audit)
    print(json.dumps({"status": "PASS", "lock_sha256": sha256_file(LOCK_PATH), "source_hashes_sha256": lock["source_hashes_sha256"], "canonical_protocol_sha256_valid": audit["canonical_protocol_sha256_valid"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
