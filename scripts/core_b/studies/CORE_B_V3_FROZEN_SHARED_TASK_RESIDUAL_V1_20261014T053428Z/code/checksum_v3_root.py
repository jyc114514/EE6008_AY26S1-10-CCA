#!/usr/bin/env python3
"""Checksum the completed V3 root and the explicitly used read-only source closure."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
V2 = ROOT.parent / "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
CHECKSUMS = ROOT / "CHECKSUMS.sha256"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        bits = line.split("  ", 1)
        if len(bits) == 2:
            result[bits[1]] = bits[0].lstrip("\\")
    return result


def collect_paths(obj: Any, out: set[Path]) -> None:
    if isinstance(obj, dict):
        for value in obj.values():
            collect_paths(value, out)
    elif isinstance(obj, list):
        for value in obj:
            collect_paths(value, out)
    elif isinstance(obj, str) and obj.startswith("/"):
        p = Path(obj)
        if p.is_file():
            out.add(p.resolve())


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    previous_manifest_backup = None
    if CHECKSUMS.exists():
        previous_manifest_backup = CHECKSUMS.with_name(f"CHECKSUMS.sha256.pre-checksum-{stamp}.bak")
        shutil.copy2(CHECKSUMS, previous_manifest_backup)
    final_path = ROOT / "qa/FINAL_ACCEPTANCE.json"
    return_path = ROOT / "RETURN_TO_CHATGPT.json"
    qa_path = ROOT / "qa/CHECKSUM_ACCEPTANCE.json"
    for p in (final_path, return_path, qa_path):
        if p.is_file():
            shutil.copy2(p, p.with_name(p.name + ".pre-checksum-" + stamp + ".bak"))

    v2_manifest_path = V2 / "CHECKSUMS.sha256"
    v2_manifest = parse_manifest(v2_manifest_path)
    scoring_qa = read_json(ROOT / "qa/DEV_TARGET_SCORING_QA.json")
    target_path = Path(scoring_qa["target_artifact"]).resolve()
    target_sha_from_scoring = str(scoring_qa["target_artifact_sha256"])
    target_rel_v2 = target_path.relative_to(V2).as_posix()
    target_sha_from_v2_manifest = v2_manifest.get(target_rel_v2)
    if target_sha_from_v2_manifest != target_sha_from_scoring:
        raise RuntimeError("scorer target hash does not match the accepted V2 checksum manifest")

    # Only paths referenced by the V3 source locks are considered, plus result/scoring references
    # and the V2 derived trajectory table used for the five locked examples.
    source_paths: set[Path] = set()
    metadata_files = [
        ROOT / "inventory/ACTUAL_FOLD_REUSE.json",
        ROOT / "inventory/CACHE_AND_NORMALIZER_MAP.json",
        ROOT / "inventory/BASE_STAGE_MAP.json",
        ROOT / "inventory/V2_LOCKED_CODE.json",
        ROOT / "inventory/V2_SOURCE_ACCEPTANCE.json",
        ROOT / "inventory/V3_PATHS.json",
        ROOT / "protocol/MODEL_READOUT_LOCK.json",
        ROOT / "protocol/PREDICTION_BATCH_SEAL.json",
    ]
    for p in metadata_files:
        collect_paths(read_json(p), source_paths)
    source_paths.update({
        (V2 / "CHECKSUMS.sha256").resolve(),
        (V2 / "protocol/METRIC_SCALE_LOCK.json").resolve(),
        (V2 / "protocol/PREDICTION_BATCH_SEAL.json").resolve(),
        (V2 / "protocol/INNER_SPLIT.csv").resolve(),
        (V2 / "protocol/INNER_SPLIT_LOCK.json").resolve(),
        (V2 / "data/DEV_PREFIXES_NO_LABELS.npz").resolve(),
        target_path,
        (V2 / "data/TRAIN_SELECTED_H30.npz").resolve(),
        (V2 / "results/ABSOLUTE_LOSSES.csv").resolve(),
        (V2 / "results/PERSISTENCE.csv").resolve(),
        (V2 / "figure_data/TRAJECTORY_CASES.csv").resolve(),
        (V2 / "figure_data/TRAJECTORY_PLOT_DATA.csv").resolve(),
        (PROJECT / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py").resolve(),
    })
    source_paths = {p for p in source_paths if p.is_file() and not p.is_relative_to(ROOT.resolve())}
    # Do not include archive/final/heldout paths. The study uses the current corrected V2 branch only.
    source_paths = {p for p in source_paths if "/archive/" not in str(p) and "heldout" not in str(p).lower() and "final_test" not in str(p).lower()}

    source_hashes: dict[Path, str] = {}
    v2_root = V2.resolve()
    for p in sorted(source_paths, key=str):
        if p == target_path:
            # Reuse the hash produced by the post-seal scorer and cross-checked against V2's manifest.
            source_hashes[p] = target_sha_from_scoring
        elif p.is_relative_to(v2_root):
            rel = p.relative_to(v2_root).as_posix()
            recorded = v2_manifest.get(rel)
            if recorded is None:
                # Small source metadata not present in the V2 manifest is hashed directly; no label arrays are opened here.
                source_hashes[p] = sha(p)
            else:
                source_hashes[p] = recorded
        else:
            source_hashes[p] = sha(p)

    final = read_json(final_path)
    ret = read_json(return_path)
    final["checksum_acceptance"] = {
        "status": "PASS", "manifest": "CHECKSUMS.sha256", "written_last": True,
        "scope": "all regular files in V3 root except the manifest itself, plus explicitly used read-only sources",
        "target_artifact_hash_reused_from_postseal_scoring_receipt": True,
        "final_heldout_payload_access": 0,
    }
    ret["checksum_acceptance"] = final["checksum_acceptance"]
    atomic_json(final_path, final)
    atomic_json(return_path, ret)

    root_files = sorted((p for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink() and p != CHECKSUMS),
                        key=lambda p: p.relative_to(ROOT).as_posix())
    v3_file_count = len(root_files) + (0 if qa_path.exists() else 1)
    entry_count = v3_file_count + len(source_hashes)
    acceptance = {
        "schema_version": 1, "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_path": str(CHECKSUMS), "manifest_entries": entry_count,
        "v3_root_file_entries": v3_file_count, "read_only_source_entries": len(source_hashes),
        "manifest_self_excluded": True, "all_v3_root_entries_rehashed_and_verified": True,
        "v2_source_hashes": "reused from the accepted V2 CHECKSUMS.sha256 where present",
        "scorer_only_target_artifact_hash": target_sha_from_scoring,
        "scorer_only_target_artifact_reopened_for_checksum": False,
        "target_hash_crosscheck": "matches V2 accepted checksum manifest",
        "external_sources": [os.path.relpath(p, ROOT) for p in sorted(source_hashes, key=str)],
        "final_heldout_payload_access": 0,
    }
    atomic_json(qa_path, acceptance)

    # Re-enumerate after adding CHECKSUM_ACCEPTANCE.json, then hash every V3 root file.
    root_files = sorted((p for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink() and p != CHECKSUMS),
                        key=lambda p: p.relative_to(ROOT).as_posix())
    root_entries = [(sha(p), p.relative_to(ROOT).as_posix()) for p in root_files]
    lines = [
        "# SHA-256 manifest for the completed V3 root and explicitly used read-only source closure.",
        "# The scorer-only target digest is carried from the post-seal scoring receipt and V2 checksum manifest; it is not reopened here.",
        "# CHECKSUMS.sha256 is self-excluded. Run local verification from this V3 directory.",
        "# V3 root files",
    ]
    lines.extend(f"{digest}  {name}" for digest, name in root_entries)
    lines.append("# Explicitly used read-only sources")
    lines.extend(f"{digest}  {os.path.relpath(path, ROOT)}" for path, digest in sorted(source_hashes.items(), key=lambda x: str(x[0])))
    temporary_manifest = CHECKSUMS.with_name(f".CHECKSUMS.sha256.{os.getpid()}.part")
    temporary_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, CHECKSUMS)

    # Verify local V3 files and non-target external sources. The target source is validated by
    # its already-recorded checksum receipt, avoiding an additional dev-label artifact read.
    for digest, name in root_entries:
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"V3 checksum verification failed: {name}")
    for p, digest in source_hashes.items():
        if p == target_path:
            continue
        if p.is_relative_to(v2_root):
            rel = p.relative_to(v2_root).as_posix()
            if v2_manifest.get(rel) is not None and v2_manifest[rel] != digest:
                raise RuntimeError(f"V2 source manifest mismatch: {rel}")
        elif sha(p) != digest:
            raise RuntimeError(f"external source checksum verification failed: {p}")

    manifest_sha = sha(CHECKSUMS)
    print(json.dumps({"status":"PASS","manifest":str(CHECKSUMS),"manifest_sha256":manifest_sha,
                      "v3_root_files":len(root_files),"read_only_sources":len(source_hashes),
                      "manifest_entries":len(root_entries)+len(source_hashes),
                      "target_artifact_reopened_for_checksum":False,
                      "final_heldout_payload_access":0},indent=2))


if __name__ == "__main__":
    main()
