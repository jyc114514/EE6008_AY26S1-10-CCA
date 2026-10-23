#!/usr/bin/env python3
"""Finalize N5 status, create CHECKSUMS.sha256 last, then verify read-only."""
import os

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_N5_STATISTICAL_CLOSEOUT_READINESS_V1_20261013T053044Z'))
MANIFEST = ROOT / "CHECKSUMS.sha256"
PASS = "PASS_CHECKSUMS_GENERATED_LAST_AND_VERIFIED_READ_ONLY"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    files = []
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in N5 root: {path}")
        if path.is_file() and path != MANIFEST:
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def create_manifest():
    files = inventory()
    content = "".join(f"{sha(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files)
    MANIFEST.write_text(content, encoding="utf-8")
    return files


def verify_manifest(expected_files):
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
    expected_paths = {p.relative_to(ROOT).as_posix() for p in expected_files}
    if set(listed) != expected_paths:
        raise ValueError("checksum manifest file set does not match N5 root inventory")
    for rel, digest in listed.items():
        actual = sha(ROOT / rel)
        if actual != digest:
            raise ValueError(f"checksum verification failed: {rel}")
    return {"file_count": len(listed), "manifest_sha256": sha(MANIFEST), "status": "PASS"}


def write_json(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def update_resource_snapshot():
    path = ROOT / "coordination/RESOURCE_LOG.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    retained = []
    snapshot = None
    for row in rows:
        if row.get("stage") == "pre_checksum_final_root_snapshot":
            if snapshot is None:
                snapshot = row
            continue
        retained.append(row)
    if snapshot is None:
        raise ValueError("pre-checksum resource snapshot row is missing")
    retained.append(snapshot)
    free_bytes = shutil.disk_usage(ROOT).free
    for _ in range(4):
        root_bytes = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST)
        snapshot["output_bytes_delta"] = str(root_bytes)
        snapshot["notes"] = f"final pre-manifest content bytes={root_bytes}; filesystem free bytes={free_bytes}; quota CLI unavailable"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
        measured = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST)
        if measured == root_bytes:
            return
    raise ValueError("resource snapshot byte count did not stabilize")


def update_completion_status():
    acceptance_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance["checksum_acceptance"] = PASS
    acceptance["checksum_manifest_path"] = str(MANIFEST)
    write_json(acceptance_path, acceptance)

    ini = ROOT / "FINAL_STATUS.ini"
    lines = [x for x in ini.read_text(encoding="utf-8").splitlines() if not x.startswith("CHECKSUM_ACCEPTANCE=")]
    lines.append(f"CHECKSUM_ACCEPTANCE={PASS}")
    ini.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status_json = ROOT / "FINAL_STATUS.ini.json"
    status = json.loads(status_json.read_text(encoding="utf-8"))
    status["checksum_acceptance"] = PASS
    status["checksum_manifest_path"] = str(MANIFEST)
    write_json(status_json, status)

    return_json = ROOT / "RETURN_TO_CHATGPT.zh-CN.json"
    ret = json.loads(return_json.read_text(encoding="utf-8"))
    ret["checksum_acceptance"] = PASS
    ret["checksum_manifest_path"] = str(MANIFEST)
    write_json(return_json, ret)

    return_md = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
    s = return_md.read_text(encoding="utf-8")
    s = s.replace(
        "校验清单将在最后生成 `CHECKSUMS.sha256` 并只读复核。",
        "`CHECKSUMS.sha256` 已最后生成并逐文件只读复核。",
    )
    return_md.write_text(s, encoding="utf-8")

    owner_path = ROOT / "coordination/OWNER_STATE.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner["checksum_acceptance"] = PASS
    owner["writer_active"] = False
    write_json(owner_path, owner)


def main():
    # First pass checks the complete content set. Completion receipts are then updated,
    # followed by a fresh final manifest and a second read-only verification.
    files = create_manifest()
    first = verify_manifest(files)
    if first["status"] != "PASS":
        raise SystemExit("first checksum verification failed")
    update_completion_status()
    update_resource_snapshot()
    files = create_manifest()
    final = verify_manifest(files)
    if final["status"] != "PASS":
        raise SystemExit("final checksum verification failed")
    print(json.dumps({
        "status": PASS,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": final["manifest_sha256"],
        "verified_files_excluding_manifest": final["file_count"],
        "old_roots_written": 0,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
