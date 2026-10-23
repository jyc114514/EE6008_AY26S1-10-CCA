#!/usr/bin/env python3
"""Create a bounded, read-only inventory for Route C replay recovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--b23-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    route = args.route_root.resolve()
    b23 = args.b23_root.resolve()
    feature = args.feature_root.resolve()
    recovery = args.recovery_root.resolve()

    explicit_paths = [
        project / "AGENTS.md",
        project / "TEAM_HANDOFF_DOCS_20260901/DEVELOPMENT_GUIDE.zh-CN.md",
        route / "ROUTE_C_PROTOCOL_LOCKED.json",
        route / "ROUTE_C_MATRIX.json",
        route / "ROUTE_C_CHECKPOINT_REGISTRY.json",
        route / "ROUTE_C_DISPATCH_STATUS.json",
        route / "ROUTE_C_RUN_ACCEPTANCE.csv",
        route / "logs/B3-FUS-D-F_gpu1.log",
        route / "logs/B3-FUS-D-F_watcher.json",
        route / "predictions/B3-FUS-D-F/RUN_STATUS.json",
        route / "code/run_counterfactual_inference.py",
        route / "code/dispatch_route_c.py",
        route / "code/task_local_gpu_watcher.py",
        route / "tests/test_task_local_watcher_addendum.py",
        b23 / "RESOLVED_PROTOCOL.json",
        b23 / "inputs/B2_INPUTS_0125.npz",
        b23 / "inputs/B2_FRAME_MAPS.json",
        b23 / "runs/B2-FUS-0125/RUN_CONFIG_LOCKED.json",
        b23 / "runs/B2-FUS-0125/predictions_best.npz",
        b23 / "runs/corrected/B3-FUS-D-L/RUN_CONFIG_LOCKED.json",
        project / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        feature / "provenance.json",
        feature / "resolved_config.json",
        feature / "model_contract.json",
        feature / "feature_manifest.parquet",
        project / "repo/ee6008-humanoid-vjepa/src/ee6008/data/clips.py",
        project / "repo/ee6008-humanoid-vjepa/src/ee6008/models.py",
        project / "repo/third_party/vjepa2/app/vjepa_2_1/models/vision_transformer.py",
    ]

    known_hashes = {
        str((project / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt").resolve()): "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
        str((b23 / "inputs/B2_INPUTS_0125.npz").resolve()): "0af89451e8d5585a5231695662e916a0de71976fa2c98a20f666e90af440377c",
        str((b23 / "runs/B2-FUS-0125/predictions_best.npz").resolve()): "79c19751cd36f4befb353129a2752bad8f791644c261816350c06a35943fcda5",
    }

    rows: list[dict[str, object]] = []
    hash_rows: list[tuple[str, str]] = []
    for path in explicit_paths:
        resolved = path.resolve()
        item: dict[str, object] = {"path": str(resolved), "exists": resolved.exists()}
        if resolved.exists():
            stat = resolved.stat()
            item.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            if str(resolved) in known_hashes:
                item.update({"sha256": known_hashes[str(resolved)], "hash_source": "pre-existing registry evidence"})
                hash_rows.append((known_hashes[str(resolved)], str(resolved)))
            elif stat.st_size <= 100_000_000:
                value = sha256(resolved)
                item.update({"sha256": value, "hash_source": "computed before recovery"})
                hash_rows.append((value, str(resolved)))
            else:
                item.update({"sha256": None, "hash_source": "not recomputed due size bound"})
        rows.append(item)

    recovery.mkdir(parents=True, exist_ok=True)
    inventory = {
        "created_utc": utc_now(),
        "recovery_root": str(recovery),
        "route_root": str(route),
        "b23_root": str(b23),
        "feature_root": str(feature),
        "status": "pre_recovery_inventory",
        "items": rows,
        "old_artifacts_unchanged": True,
        "final_test_access": 0,
    }
    (recovery / "PRE_RECOVERY_INVENTORY.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (recovery / "PRE_RECOVERY_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "exists", "size", "mtime_ns", "sha256", "hash_source"])
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    with (recovery / "PRE_RECOVERY_HASHES.sha256").open("w", encoding="utf-8") as handle:
        for value, path in sorted(hash_rows, key=lambda pair: pair[1]):
            handle.write(f"{value}  {path}\n")
    print(json.dumps({"status": "created", "recovery_root": str(recovery), "inventory_items": len(rows), "hash_entries": len(hash_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
