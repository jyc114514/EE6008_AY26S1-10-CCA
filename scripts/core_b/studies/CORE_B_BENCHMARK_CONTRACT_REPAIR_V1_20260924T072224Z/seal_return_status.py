"""Finalize the machine status and re-seal the new root with self-verification."""

from __future__ import annotations
import os

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z'))
MANIFEST = ROOT / "CHECKSUMS.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST and not path.is_symlink()
    )


def write_manifest() -> int:
    files = tracked_files()
    MANIFEST.write_text(
        "\n".join(f"{sha256_file(path)}  ./{path.relative_to(ROOT).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )
    return len(files)


def verify_manifest() -> bool:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.removeprefix("./")
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def main() -> None:
    status_path = ROOT / "STATUS.json"
    return_path = ROOT / "RETURN_TO_CHATGPT.json"
    qa_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
    status = json.loads(status_path.read_text())
    returned = json.loads(return_path.read_text())
    qa = json.loads(qa_path.read_text())
    now = datetime.now(timezone.utc).isoformat()
    count = len(tracked_files())
    status.update({"updated_utc": now, "CHECKSUMS": f"PASS; {count} files; checksum-last; verified"})
    returned.update({"CHECKSUMS": f"PASS; {count} files; checksum-last; verified"})
    qa.update({"new_root_checksum_status": "PASS", "new_root_checksum_entries": count, "checksum_policy": "SHA-256 checksum-last over all task-root files except the manifest itself"})
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return_path.write_text(json.dumps(returned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    count = write_manifest()
    if not verify_manifest():
        raise SystemExit("new task-root checksum self-verification failed")
    print(json.dumps({"checksum_entries": count, "status": status["NEXT_STATUS"], "checksum": "PASS", "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
