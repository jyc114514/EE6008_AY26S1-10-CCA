#!/usr/bin/env python3
"""Finalize owner/handoff metadata and write the V2 checksum manifest last."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def inventory() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.name != "CHECKSUMS.sha256" and ".part" not in path.name and not path.is_symlink():
            result.append(path)
    return sorted(result, key=lambda value: value.relative_to(ROOT).as_posix())


def main() -> None:
    owner_lock_path = ROOT / "coordination/OWNER.lock"
    owner_lock_handle = owner_lock_path.open("a+")
    try:
        fcntl.flock(owner_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("V2 owner is still active; refuse checksum-last sealing") from exc
    checksum_path = ROOT / "CHECKSUMS.sha256"
    if checksum_path.exists():
        raise FileExistsError("CHECKSUMS.sha256 already exists; checksum-last seal is immutable")
    acceptance_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
    if not acceptance_path.is_file():
        raise RuntimeError("final acceptance receipt is missing")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") not in {"PASS", "COMPLETE_WITH_PARTIAL_SEARCH", "PARTIAL"}:
        raise RuntimeError(f"unexpected final status: {acceptance.get('status')}")

    status = acceptance["status"]
    owner_status = {
        "PASS": "complete",
        "COMPLETE_WITH_PARTIAL_SEARCH": "complete_with_partial_search",
        "PARTIAL": "partial",
    }[status]
    owner_path = ROOT / "coordination/OWNER_STATE.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    owner.update({
        "updated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": owner_status, "phase": owner_status, "owner_lock_acquired": False,
        "active_jobs": 0, "next_gate": "human figure review pending" if status != "PARTIAL" else "resume listed failed or incomplete fits",
        "final_heldout_payload_access": 0,
    })
    atomic_json(owner_path, owner)

    resume_path = ROOT / "coordination/RESUME_COMMANDS.json"
    if resume_path.exists():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        resume["status"] = owner_status
        resume["checksum_manifest"] = str(checksum_path)
        atomic_json(resume_path, resume)

    acceptance["checksum_acceptance"] = {
        "status": "PASS",
        "manifest": str(checksum_path),
        "self_entry_excluded": True,
        "temporary_parts_excluded": True,
    }
    acceptance["checksum_sealed_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    atomic_json(acceptance_path, acceptance)

    return_json_path = ROOT / "RETURN_TO_CHATGPT.json"
    return_json = json.loads(return_json_path.read_text(encoding="utf-8"))
    return_json["checksum_manifest"] = str(checksum_path)
    return_json["checksum_status"] = "PASS"
    return_json["status"] = status
    atomic_json(return_json_path, return_json)
    return_md = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
    body = return_md.read_text(encoding="utf-8")
    if "CHECKSUMS.sha256" not in body:
        body += f"\n\nChecksum manifest: {checksum_path}\n"
        return_md.write_text(body, encoding="utf-8")

    files = inventory()
    lines = [f"{sha(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    content = "\n".join(lines) + "\n"
    checksum_path.write_text(content, encoding="utf-8")
    written = checksum_path.read_text(encoding="utf-8")
    if written != content:
        raise RuntimeError("checksum manifest readback differs from planned bytes")
    for line in written.splitlines():
        expected, relative = line.split("  ", 1)
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"checksum verification failed: {relative}")
    print(json.dumps({
        "status": "PASS", "final_status": status,
        "checksum_file_count": len(files), "checksum_manifest": str(checksum_path),
        "checksum_manifest_sha256": sha(checksum_path),
        "final_heldout_payload_access": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
