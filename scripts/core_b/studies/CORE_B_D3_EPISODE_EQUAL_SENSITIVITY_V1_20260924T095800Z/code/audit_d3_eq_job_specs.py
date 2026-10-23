"""Audit Stage0 watcher job JSONs and pin their individual command hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_hash(command: list[str]) -> str:
    data = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest_path = EQ_ROOT / "configs/D3_EQ_STAGE0_JOBS.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for item in manifest["jobs"]:
        path = Path(item["job_path"])
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("job_id") != item["job_id"] or job.get("protocol_id") != "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1":
            raise AssertionError("Stage0 job identity/protocol mismatch")
        if job.get("max_oom_requeues") != 0 or int(job["resource_estimate"]["measured_peak_mib"]) != 12288:
            raise AssertionError("Stage0 OOM/admission policy changed")
        if not Path(job["output_root"]).is_relative_to(EQ_ROOT):
            raise AssertionError("Stage0 output root escaped the D3-EQ root")
        if "--stage0" not in job["command"]:
            raise AssertionError("Stage0 command does not include --stage0")
        entries.append({
            "job_id": job["job_id"],
            "job_json_path": str(path),
            "job_json_sha256": sha256_file(path),
            "command_sha256": command_hash(job["command"]),
            "output_root": job["output_root"],
            "measured_peak_mib": job["resource_estimate"]["measured_peak_mib"],
        })
    if len(entries) != 12:
        raise AssertionError(f"expected 12 Stage0 specs, found {len(entries)}")
    report = {
        "status": "PASS",
        "stage0_job_count": len(entries),
        "stage0_queue_manifest_sha256": sha256_file(manifest_path),
        "jobs": entries,
        "gpu_query_performed": False,
        "gpu_child_launched": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_STAGE0_JOB_SPEC_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage0_job_count": len(entries), "queue_manifest_sha256": report["stage0_queue_manifest_sha256"], "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
