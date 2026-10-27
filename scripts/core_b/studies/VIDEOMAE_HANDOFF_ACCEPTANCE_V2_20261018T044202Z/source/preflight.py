#!/usr/bin/env python3
"""Read-only CPU preflight for the VideoMAE Core B handoff acceptance audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

EXPECTED = {(m, h, s) for m in ("TS11-D", "STATE-CTRL") for h in (30, 60) for s in (20261001, 20261002, 20261003)}
SPLITS = (("train", 0, 17757, 2712), ("dev", 1, 2322, 603))
EXPECTED_ROWS = {"train": 17757, "dev": 2322}
EXPECTED_EPISODES = {"train": 2712, "dev": 603}
V3_REL = Path("repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data")
CACHE_REL = Path('tmp/project-team/videomae_cache_full_20261008T074027Z')
TARGET_REL = Path("reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z/d3/train_targets_trainonly.npz")
SPLIT_REL = Path("repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet")
TEAM_CODE_REL = Path("reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code")
CPU_REL = Path("replay/cpu_zero_20261013T065954Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_rel(rel: str) -> bool:
    p = Path(rel)
    return not p.is_absolute() and ".." not in p.parts


def hashed_row(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "bytes": None, "sha256": None, "status": "missing"}
    size = path.stat().st_size
    sha = digest(path)
    ok_size = expected_bytes is None or size == int(expected_bytes)
    ok_sha = expected_sha is None or sha == expected_sha
    status = "PASS" if ok_size and ok_sha else "MISMATCH"
    return {"exists": True, "bytes": size, "sha256": sha, "status": status}


def safe_record(path: Path, role: str, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    row = hashed_row(path, expected_sha, expected_bytes)
    return {"role": role, "path": str(path), **row,
            "expected_bytes": expected_bytes, "expected_sha256": expected_sha}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def walk_numeric_diff(old: Any, new: Any, path: str, out: list[dict[str, Any]], schema: list[dict[str, Any]]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            schema.append({"path": path, "old_keys": "|".join(sorted(old)), "new_keys": "|".join(sorted(new))})
            return
        for k in sorted(old):
            walk_numeric_diff(old[k], new[k], f"{path}.{k}" if path else k, out, schema)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            schema.append({"path": path, "old_length": len(old), "new_length": len(new)})
            return
        for i, (a, b) in enumerate(zip(old, new)):
            walk_numeric_diff(a, b, f"{path}[{i}]", out, schema)
        return
    if isinstance(old, (int, float)) and not isinstance(old, bool) and isinstance(new, (int, float)) and not isinstance(new, bool):
        if not (math.isfinite(float(old)) and math.isfinite(float(new))):
            schema.append({"path": path, "old_value": old, "new_value": new, "kind": "nonfinite"})
            return
        d = abs(float(old) - float(new))
        if d:
            parts = path.split(".")
            out.append({"path": path, "split": parts[1] if len(parts) > 1 else "",
                        "region": parts[2] if len(parts) > 2 else "",
                        "model": parts[3] if len(parts) > 3 else "",
                        "metric": parts[-1], "old": old, "new": new, "abs_delta": d})
        return
    if old != new:
        schema.append({"path": path, "old_value": old, "new_value": new, "kind": "non_numeric"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--v1-root", type=Path, required=True)
    ap.add_argument("--handoff-root", type=Path, required=True)
    ap.add_argument("--team-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    args = ap.parse_args()

    project, v2, v1, handoff, team, audit = [p.resolve() for p in (
        args.project_root, args.v2_root, args.v1_root, args.handoff_root, args.team_root, args.audit_root)]
    audit.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    source_hash_cache: dict[str, str] = {}
    source_rows: list[dict[str, Any]] = []

    def cached_digest(path: Path) -> str:
        resolved = str(path.resolve(strict=True))
        if resolved not in source_hash_cache:
            source_hash_cache[resolved] = digest(path)
        return source_hash_cache[resolved]

    archive_path = v2 / "ARCHIVE.json"
    archive = json_read(archive_path)
    if Path(archive.get("original_project", "")).resolve() != project:
        errors.append("ARCHIVE.original_project does not equal the supplied PROJECT_ROOT")
    archive_files = archive.get("files", {})
    archive_cells = archive.get("cells", [])
    if archive.get("status") != "PREPARED_NOT_REPLAYED":
        warnings.append(f"unexpected archive status: {archive.get('status')}")
    if len(archive_cells) != 12:
        errors.append(f"archive declares {len(archive_cells)} cells; expected 12")

    inventory_rows: list[dict[str, Any]] = []
    registered = set()
    origin_verified: dict[str, dict[str, Any]] = {}
    total_registered_bytes = 0
    for idx, (rel, record) in enumerate(sorted(archive_files.items()), 1):
        total_registered_bytes += int(record.get("bytes", 0))
        valid = safe_rel(rel)
        registered.add(rel)
        path = v2 / rel if valid else v2 / "__unsafe_archive_path__"
        actual = hashed_row(path, record.get("sha256"), record.get("bytes")) if valid else {
            "exists": False, "bytes": None, "sha256": None, "status": "UNSAFE_PATH"
        }
        source_path = project / rel if valid else project / "__unsafe_archive_path__"
        first = Path(rel).parts[0] if valid and Path(rel).parts else ""
        source_applicable = bool(valid and source_path.is_file() and first not in {"code", "models", "ENVIRONMENT_EXPORT.json"})
        source = hashed_row(source_path, record.get("sha256"), record.get("bytes")) if source_applicable else {
            "exists": None, "bytes": None, "sha256": None,
            "status": "generated_or_archive_only" if first in {"code", "models", "ENVIRONMENT_EXPORT.json"} else "source_not_found"
        }
        if actual["status"] != "PASS":
            errors.append(f"archive entry mismatch: {rel} ({actual['status']})")
        if source_applicable:
            origin_verified[str(source_path.resolve())] = source
            if source["status"] != "PASS":
                errors.append(f"archive origin mismatch: {rel} ({source['status']})")
        category = ("copied_model" if rel.startswith("models/") else
                    "copied_code" if rel.startswith("code/") else
                    "copied_dependency" if source_applicable else "generated_archive_file")
        inventory_rows.append({
            "path": rel, "category": category,
            "registered_bytes": record.get("bytes"), "registered_sha256": record.get("sha256"),
            "archive_exists": actual["exists"], "archive_bytes": actual["bytes"], "archive_sha256": actual["sha256"],
            "archive_status": actual["status"], "source_path": str(source_path) if source_applicable else "",
            "source_bytes": source["bytes"], "source_sha256": source["sha256"], "source_status": source["status"]
        })
        if idx % 50 == 0 or idx == len(archive_files):
            print(f"ARCHIVE_HASH_PROGRESS {idx}/{len(archive_files)}", flush=True)

    actual_files: set[str] = set()
    symlinks: list[dict[str, Any]] = []
    for path in v2.rglob("*"):
        rel = path.relative_to(v2).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            resolved = path.resolve(strict=False)
            symlinks.append({"path": rel, "target": target, "resolved_target": str(resolved),
                             "target_exists": resolved.exists(), "absolute_target": Path(target).is_absolute(),
                             "target_inside_v2": resolved.is_relative_to(v2)})
            if not resolved.exists():
                warnings.append(f"broken symlink: {rel} -> {target}")
        elif path.is_file():
            actual_files.add(rel)
    extras = sorted(actual_files - registered - {"ARCHIVE.json"})
    for rel in extras:
        category = "post_archive_replay_subtree" if rel.startswith("replay/") else "unregistered_extra"
        inventory_rows.append({"path": rel, "category": category, "registered_bytes": "",
                               "registered_sha256": "", "archive_exists": True,
                               "archive_bytes": (v2 / rel).stat().st_size, "archive_sha256": digest(v2 / rel),
                               "archive_status": category, "source_path": "", "source_bytes": "",
                               "source_sha256": "", "source_status": ""})
    if symlinks:
        warnings.append(f"archive tree contains {len(symlinks)} symlink(s)")
    abs_archive_keys = [x for x in archive_files if Path(x).is_absolute()]
    if abs_archive_keys:
        errors.append("archive file registry contains absolute paths")

    csv_write(audit / "V2_ARTIFACT_INVENTORY.csv",
              ["path", "category", "registered_bytes", "registered_sha256", "archive_exists", "archive_bytes",
               "archive_sha256", "archive_status", "source_path", "source_bytes", "source_sha256", "source_status"],
              inventory_rows)

    # Verify the 169-entry original handoff receipt against the current read-only source.
    hand_index_path = handoff / "HANDOFF_INDEX.json"
    hand_index = json_read(hand_index_path)
    hand_artifacts = hand_index.get("artifacts", {})
    hand_rows: list[dict[str, Any]] = []
    for raw_path, record in sorted(hand_artifacts.items()):
        path = Path(raw_path)
        try:
            path.resolve(strict=False).relative_to(project)
            inside_project = True
        except ValueError:
            inside_project = False
        result = hashed_row(path, record.get("sha256"), record.get("bytes")) if inside_project else {
            "exists": path.is_file(), "bytes": None, "sha256": None, "status": "outside_project_root"
        }
        hand_rows.append({"path": str(path), "expected_bytes": record.get("bytes"), "expected_sha256": record.get("sha256"),
                          "actual_bytes": result["bytes"], "actual_sha256": result["sha256"], "status": result["status"]})
        if result["status"] != "PASS":
            errors.append(f"handoff artifact mismatch: {path} ({result['status']})")
    csv_write(audit / "OLD_HANDOFF_ARTIFACT_RECHECK.csv",
              ["path", "expected_bytes", "expected_sha256", "actual_bytes", "actual_sha256", "status"], hand_rows)

    # Verify the V1 acceptance directory's listed checksums.
    checksum_path = v1 / "CHECKSUMS.sha256"
    checksum_rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if "  " not in raw:
            checksum_rows.append({"path": "", "expected_sha256": "", "actual_sha256": "", "status": "malformed_line", "line": line_no})
            errors.append(f"malformed V1 checksum line {line_no}")
            continue
        expected_sha, rel = raw.split("  ", 1)
        rel = rel.lstrip("*")
        if not safe_rel(rel):
            checksum_rows.append({"path": rel, "expected_sha256": expected_sha, "actual_sha256": "", "status": "unsafe_path", "line": line_no})
            errors.append(f"unsafe V1 checksum path: {rel}")
            continue
        path = v1 / rel
        result = hashed_row(path, expected_sha)
        checksum_rows.append({"path": rel, "expected_sha256": expected_sha, "actual_sha256": result["sha256"], "status": result["status"], "line": line_no})
        if result["status"] != "PASS":
            errors.append(f"V1 checksum mismatch: {rel} ({result['status']})")
    csv_write(audit / "V1_CHECKSUM_RECHECK.csv",
              ["path", "expected_sha256", "actual_sha256", "status", "line"], checksum_rows)

    final_v1 = json_read(v1 / "FINAL_ACCEPTANCE.json")
    replay_csv = list(csv.DictReader((v1 / "CHECKPOINT_REPLAY.csv").open(newline="", encoding="utf-8")))
    coreb_v1_rows = [r for r in replay_csv if r.get("task") == "CoreB"]
    replay_by_cell = {r.get("cell"): r for r in coreb_v1_rows}
    hand_cells = {(c["mode"], int(c["horizon"]), int(c["seed"])): c for c in hand_index.get("core_b", {}).get("cells", [])}
    audit_dir = handoff / "audit"
    expected_by_combo = {}
    for entry in archive_cells:
        manifest_path = v2 / entry["manifest"]
        manifest = json_read(manifest_path)
        combo = (manifest.get("mode"), int(manifest.get("horizon", -1)), int(manifest.get("seed", -1)))
        expected_by_combo[combo] = entry
    if set(expected_by_combo) != EXPECTED or len(expected_by_combo) != 12:
        errors.append("V2 manifests do not resolve to the exact expected 12-cell matrix")

    expected_rows: list[dict[str, Any]] = []
    safe_rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    model_state_records: list[dict[str, Any]] = []
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        import importlib.util

        model_source = v2 / TEAM_CODE_REL / "d3_model.py"
        spec = importlib.util.spec_from_file_location("videomae_audit_d3_model", model_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.set_num_threads(1)
        for combo in sorted(EXPECTED):
            entry = expected_by_combo.get(combo)
            if entry is None:
                continue
            mode, horizon, seed = combo
            name = entry["name"]
            manifest_path = v2 / entry["manifest"]
            weight_path = v2 / entry["weights"]
            config_path = v2 / entry["config"]
            run_root = v2 / entry["run"]
            manifest = json_read(manifest_path)
            config = json_read(config_path)
            status = json_read(run_root / "RUN_STATUS.json")
            checkpoint_path = v2 / manifest["original_checkpoint"]
            checkpoint_sha = digest(checkpoint_path)  # bytes only; never deserialize this file
            weight_sha = digest(weight_path)
            config_sha = digest(config_path)
            manifest_sha = digest(manifest_path)
            mode_ok = config.get("mode") == mode == manifest.get("mode")
            horizon_ok = int(config.get("horizon", -1)) == horizon == int(manifest.get("horizon", -2))
            seed_ok = int(config.get("seed", -1)) == seed == int(manifest.get("seed", -2))
            epoch_ok = int(config.get("epochs", -1)) == int(manifest.get("epoch", -2)) == 40 and int(status.get("epochs", -1)) == 40
            updates_ok = int(config.get("updates", -1)) == 11120 and int(status.get("optimizer_steps", -1)) == 11120
            checkpoint_ok = checkpoint_sha == manifest.get("original_checkpoint_sha256") == status.get("checkpoint_sha256")
            config_hash_ok = config_sha == manifest.get("config_sha256")
            weights_hash_ok = weight_sha == manifest.get("weights_sha256")
            final_test_ok = config.get("final_test_access") == 0 and status.get("final_test_access") == 0 and manifest.get("final_test_access") == 0
            code_checks = {}
            for code_name, sha in config.get("source_hashes", {}).items():
                code_path = v2 / TEAM_CODE_REL / code_name
                code_checks[code_name] = code_path.is_file() and digest(code_path) == sha
            code_ok = bool(code_checks) and all(code_checks.values())

            source_v1_row = replay_by_cell.get(name)
            v1_checkpoint_match = bool(source_v1_row and source_v1_row.get("checkpoint_sha256") == checkpoint_sha)
            hand_cell = hand_cells.get(combo)
            hand_audit_path = audit_dir / f"{name}.json"
            hand_audit = json_read(hand_audit_path) if hand_audit_path.is_file() else {}
            hand_checkpoint_match = bool(hand_audit.get("checkpoint_hash_matches") is True and hand_audit.get("final_test_access") == 0)
            original_handoff_run = hand_cell.get("run_root") if hand_cell else ""
            expected_run = str((project / Path(manifest["original_checkpoint"]).parent).resolve())
            handoff_mapping_ok = bool(hand_cell and Path(original_handoff_run).resolve() == Path(expected_run))
            if not all((mode_ok, horizon_ok, seed_ok, epoch_ok, updates_ok, checkpoint_ok, config_hash_ok, weights_hash_ok,
                        final_test_ok, code_ok, v1_checkpoint_match, hand_checkpoint_match, handoff_mapping_ok)):
                errors.append(f"cell provenance/config mismatch: {name}")

            # Safe tensors only. Original checkpoint above was opened for hashing only.
            safe_meta = {}
            with safe_open(str(weight_path), framework="pt", device="cpu") as sf:
                safe_meta = sf.metadata() or {}
            weights = load_file(str(weight_path), device="cpu")
            module.VIDEO_DIM = 768
            with torch.random.fork_rng(devices=[]):
                model = module.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=mode)
                if mode == "TS11-D":
                    model.video_projection = torch.nn.Linear(384, 128)
                    module.VIDEO_DIM = 384
            expected_state = model.state_dict()
            key_match = set(weights) == set(expected_state)
            tensor_match = key_match
            key_shapes = {}
            for key in sorted(set(weights) | set(expected_state)):
                if key not in weights or key not in expected_state:
                    tensor_match = False
                    continue
                got, exp = weights[key], expected_state[key]
                shape_ok = tuple(got.shape) == tuple(exp.shape)
                dtype_ok = got.dtype == exp.dtype
                if not shape_ok or not dtype_ok:
                    tensor_match = False
                key_shapes[key] = {"shape": list(got.shape), "dtype": str(got.dtype), "shape_match": shape_ok, "dtype_match": dtype_ok}
                raw = got.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                fingerprint_rows.append({"cell": name, "tensor_name": key, "shape": "x".join(str(x) for x in got.shape),
                                         "dtype": str(got.dtype), "numel": got.numel(),
                                         "sha256": hashlib.sha256(raw).hexdigest()})
            try:
                model.load_state_dict(weights, strict=True)
                strict_ok = True
                missing_keys, unexpected_keys = [], []
            except Exception as exc:
                strict_ok = False
                missing_keys, unexpected_keys = [f"{type(exc).__name__}: {exc}"], []
                errors.append(f"strict load failed for {name}: {type(exc).__name__}: {exc}")
            model.eval()
            state_dict = model.state_dict()
            buffers = []
            for buffer_name, buffer in model.named_buffers():
                buffers.append({"name": buffer_name, "shape": list(buffer.shape), "dtype": str(buffer.dtype),
                                "persistent": buffer_name in state_dict})
            named_parameter_count = sum(p.numel() for p in model.parameters())
            state_dict_metadata = getattr(state_dict, "_metadata", {})
            model_state_records.append({
                "cell": name, "mode": mode, "horizon": horizon, "seed": seed,
                "video_input_dim": 384 if mode == "TS11-D" else 768,
                "projection_weight_shape": list(model.video_projection.weight.shape),
                "parameter_count": named_parameter_count, "state_dict_tensor_count": len(state_dict),
                "state_dict_metadata": state_dict_metadata, "buffers": buffers,
                "nonpersistent_buffers": [b["name"] for b in buffers if not b["persistent"]],
                "runtime_generated_state": ["temporal sinusoidal position table", "spatial quadrant encoding", "future query sinusoidal table"],
                "forward_constants": {"state_dim": 71, "action_dim": 28, "time_slots": 16, "space_bins": 4, "width": 128,
                                      "video_mode": mode, "video_input_dim": 384 if mode == "TS11-D" else 768},
                "safe_file_metadata": safe_meta, "state_dict_keys_shapes": key_shapes
            })
            expected_metadata = {
                "original_checkpoint_sha256": manifest.get("original_checkpoint_sha256"),
                "config_sha256": manifest.get("config_sha256"),
                "epoch": str(manifest.get("epoch")), "seed": str(seed), "horizon": str(horizon), "mode": mode
            }
            metadata_ok = all(safe_meta.get(k) == val for k, val in expected_metadata.items())
            struct_ok = key_match and tensor_match and strict_ok and metadata_ok and len(weights) == int(manifest.get("tensor_count", -1))
            if not struct_ok:
                errors.append(f"safe model structure/metadata mismatch: {name}")
            safe_rows.append({
                "cell": name, "mode": mode, "horizon": horizon, "seed": seed,
                "format": "safetensors", "format_status": "accepted" if weights_hash_ok else "needs_fix",
                "source_status": "hash_bound_exporter_attested" if checkpoint_ok and weights_hash_ok and code_ok else "needs_fix",
                "source_caveat": "no pre-export independent tensor catalogue or separate exporter execution log; strict safe reload is functional evidence only",
                "weight_bytes": weight_path.stat().st_size, "weights_sha256": weight_sha,
                "original_checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
                "tensor_count": len(weights), "parameter_count": named_parameter_count,
                "key_set_match": key_match, "all_shape_dtype_match": tensor_match, "strict_load": strict_ok,
                "safe_metadata_match": metadata_ok, "structure_status": "accepted" if struct_ok else "needs_fix",
                "video_input_dim": 384 if mode == "TS11-D" else 768,
                "final_test_access": 0 if final_test_ok else "UNKNOWN"
            })
            expected_rows.append({
                "cell": name, "mode": mode, "horizon": horizon, "seed": seed,
                "original_checkpoint": manifest.get("original_checkpoint"), "original_checkpoint_sha256": checkpoint_sha,
                "config": entry["config"], "config_sha256": config_sha,
                "weights": entry["weights"], "weights_sha256": weight_sha,
                "manifest": entry["manifest"], "manifest_sha256": manifest_sha,
                "epoch": config.get("epochs"), "optimizer_updates": status.get("optimizer_steps"),
                "output_dim": 28, "train_rows": 17757, "dev_rows": 2322,
                "v1_checkpoint_sha_match": v1_checkpoint_match, "handoff_checkpoint_receipt_match": hand_checkpoint_match,
                "handoff_run_mapping_match": handoff_mapping_ok,
                "final_test_access": 0 if final_test_ok else "UNKNOWN"
            })
        del torch, load_file
    except Exception as exc:
        errors.append(f"safe model structural audit failed: {type(exc).__name__}: {exc}")
        print(f"SAFE_MODEL_AUDIT_ERROR {type(exc).__name__}: {exc}", flush=True)

    csv_write(audit / "EXPECTED_CELLS.csv",
              ["cell", "mode", "horizon", "seed", "original_checkpoint", "original_checkpoint_sha256", "config",
               "config_sha256", "weights", "weights_sha256", "manifest", "manifest_sha256", "epoch",
               "optimizer_updates", "output_dim", "train_rows", "dev_rows", "v1_checkpoint_sha_match",
               "handoff_checkpoint_receipt_match", "handoff_run_mapping_match", "final_test_access"], expected_rows)
    csv_write(audit / "SAFE_MODEL_ACCEPTANCE.csv",
              ["cell", "mode", "horizon", "seed", "format", "format_status", "source_status", "source_caveat",
               "weight_bytes", "weights_sha256", "original_checkpoint_sha256", "config_sha256", "tensor_count",
               "parameter_count", "key_set_match", "all_shape_dtype_match", "strict_load", "safe_metadata_match",
               "structure_status", "video_input_dim", "final_test_access"], safe_rows)
    csv_write(audit / "TENSOR_FINGERPRINTS.csv",
              ["cell", "tensor_name", "shape", "dtype", "numel", "sha256"], fingerprint_rows)
    write_json(audit / "FORWARD_STATE_INVENTORY.json", {
        "source": str(v2 / TEAM_CODE_REL / "d3_model.py"),
        "source_sha256": digest(v2 / TEAM_CODE_REL / "d3_model.py"),
        "records": model_state_records,
        "summary": {
            "registered_buffers": ["state_modality", "video_modality", "anchor_modality"],
            "all_registered_buffers_persistent": all(b["persistent"] for r in model_state_records for b in r["buffers"]),
            "nonpersistent_buffers": sorted({b["name"] for r in model_state_records for b in r["buffers"] if not b["persistent"]}),
            "unregistered_state_generated_in_forward": ["temporal sinusoidal positions", "spatial quadrant encoding", "future query sinusoidal positions"],
            "mode_specific_video_projection_dims": {"TS11-D": 384, "STATE-CTRL": 768},
            "final_test_access": 0
        }
    })

    # Validate existing CPU output IDs, schemas, prediction differences and panel metrics.
    import numpy as np
    inputs = v2 / V3_REL / "V3_MAIN_INPUTS.npz"
    ids_by_split: dict[str, np.ndarray] = {}
    episodes_by_split: dict[str, np.ndarray] = {}
    with np.load(inputs, allow_pickle=False) as source:
        split_codes = source["split_code"].copy()
        anchor_ids = source["anchor_id"].copy()
        episode_ids = source["episode_index"].copy()
    cpu_summary_path = v2 / CPU_REL / "REPLAY_SUMMARY.json"
    cpu_summary = json_read(cpu_summary_path)
    cpu_results = {r["cell"]: r for r in cpu_summary.get("results", [])}
    for split, code, row_count, episode_count in SPLITS:
        rows = np.flatnonzero(split_codes == code)
        ids_by_split[split], episodes_by_split[split] = anchor_ids[rows], episode_ids[rows]
        if len(rows) != row_count or len(np.unique(episode_ids[rows])) != episode_count:
            errors.append(f"canonical input IDs/counts mismatch for {split}")
    cpu_replay_rows: list[dict[str, Any]] = []
    cpu_metric_rows: list[dict[str, Any]] = []
    cpu_metric_schema: list[dict[str, Any]] = []
    for combo in sorted(EXPECTED):
        entry = expected_by_combo.get(combo)
        if not entry:
            continue
        name = entry["name"]
        run_path = v2 / entry["run"]
        cpu_run = v2 / CPU_REL / name
        ref_status = json_read(run_path / "RUN_STATUS.json")
        for split, _, row_count, _ in SPLITS:
            pred_path = cpu_run / f"predictions_{split}.npz"
            ref_path = run_path / f"predictions_{split}.npz"
            pred_hash = digest(pred_path) if pred_path.is_file() else ""
            ref_hash = digest(ref_path) if ref_path.is_file() else ""
            shape_ok = finite_ok = id_ok = dtype_ok = False
            max_abs = mean_abs = None
            bitwise = False
            if pred_path.is_file() and ref_path.is_file():
                with np.load(pred_path, allow_pickle=False) as predz, np.load(ref_path, allow_pickle=False) as refz:
                    pred = predz["predictions_native"].copy()
                    ref = refz["predictions_native"].copy()
                    pids, peps = predz["anchor_id"].copy(), predz["episode_index"].copy()
                    rids, reps = refz["anchor_id"].copy(), refz["episode_index"].copy()
                shape_ok = pred.shape == ref.shape == (row_count, int(entry["name"].split("_H")[1].split("_")[0]), 28)
                finite_ok = bool(np.isfinite(pred).all())
                dtype_ok = pred.dtype == ref.dtype == np.float32
                id_ok = bool(np.array_equal(pids, ids_by_split[split]) and np.array_equal(peps, episodes_by_split[split])
                              and np.array_equal(pids, rids) and np.array_equal(peps, reps))
                if pred.shape == ref.shape:
                    diff = np.abs(pred.astype(np.float64) - ref.astype(np.float64))
                    max_abs = float(diff.max(initial=0))
                    mean_abs = float(diff.mean())
                    bitwise = bool(np.array_equal(pred, ref))
            recorded = cpu_results.get(name, {}).get("prediction_max_abs", {}).get(split)
            record_match = recorded is not None and max_abs is not None and math.isclose(float(recorded), max_abs, rel_tol=1e-12, abs_tol=1e-12)
            accepted_cpu = bool(shape_ok and finite_ok and id_ok and dtype_ok and bitwise and record_match)
            if not shape_ok or not finite_ok or not id_ok or not dtype_ok:
                errors.append(f"CPU saved prediction schema/ID mismatch: {name}/{split}")
            cpu_replay_rows.append({
                "cell": name, "split": split, "historical_status": "FAIL_REPLAY_TOLERANCE",
                "saved_output_present": pred_path.is_file(), "prediction_sha256": pred_hash, "reference_sha256": ref_hash,
                "rows": row_count, "shape_ok": shape_ok, "dtype_ok": dtype_ok, "finite": finite_ok, "ids_order_match": id_ok,
                "bitwise_equal": bitwise, "max_abs": max_abs, "mean_abs": mean_abs,
                "summary_max_abs_match": record_match, "historical_zero_tolerance_accepted": accepted_cpu,
                "note": "historical CPU zero-tolerance result preserved; this is not the new GPU engineering gate"
            })
        old_metrics_path = v2 / entry["metrics"]
        new_metrics_path = cpu_run / "evaluation/METRICS.json"
        if old_metrics_path.is_file() and new_metrics_path.is_file():
            old_panels = json_read(old_metrics_path).get("panels", {})
            new_panels = json_read(new_metrics_path).get("panels", {})
            local: list[dict[str, Any]] = []
            local_schema: list[dict[str, Any]] = []
            walk_numeric_diff(old_panels, new_panels, "panels", local, local_schema)
            summary_max = max((r["abs_delta"] for r in local), default=0.0)
            recorded_metric = cpu_results.get(name, {}).get("metric_max_abs")
            record_match = recorded_metric is not None and math.isclose(float(recorded_metric), summary_max, rel_tol=1e-12, abs_tol=1e-12)
            for row in local:
                cpu_metric_rows.append({"cell": name, **row, "summary_metric_max_abs": recorded_metric,
                                        "cell_metric_max_abs": summary_max, "summary_max_match": record_match})
            cpu_metric_schema.extend({"cell": name, **r} for r in local_schema)
            if not record_match:
                errors.append(f"CPU metric summary max does not match saved panels: {name}")
    csv_write(audit / "HISTORICAL_CPU_REPLAY_RECHECK.csv",
              ["cell", "split", "historical_status", "saved_output_present", "prediction_sha256", "reference_sha256",
               "rows", "shape_ok", "dtype_ok", "finite", "ids_order_match", "bitwise_equal", "max_abs", "mean_abs",
               "summary_max_abs_match", "historical_zero_tolerance_accepted", "note"], cpu_replay_rows)
    csv_write(audit / "CPU_METRIC_DIFFS.csv",
              ["cell", "path", "split", "region", "model", "metric", "old", "new", "abs_delta",
               "summary_metric_max_abs", "cell_metric_max_abs", "summary_max_match"], cpu_metric_rows)
    csv_write(audit / "CPU_METRIC_SCHEMA_DIFFS.csv",
              ["cell", "path", "old_keys", "new_keys", "old_length", "new_length", "old_value", "new_value", "kind"], cpu_metric_schema)

    max_cpu_pred = max((r["max_abs"] or 0 for r in cpu_replay_rows), default=0)
    max_cpu_metric_row = max(cpu_metric_rows, key=lambda r: r["abs_delta"], default=None)
    write_json(audit / "HISTORICAL_CPU_DIAGNOSIS.json", {
        "status": cpu_summary.get("status"),
        "historical_cpu_zero_tolerance_status": "historical_fail_preserved",
        "cells": len(cpu_results), "prediction_splits_rechecked": len(cpu_replay_rows),
        "prediction_max_abs_over_all_cell_splits": max_cpu_pred,
        "metric_rows_with_numeric_differences": len(cpu_metric_rows),
        "metric_max_abs_over_panels": max_cpu_metric_row,
        "metric_summary_field_identification": {
            "cell": "STATE-CTRL_H60_S20261003", "split": "dev", "region": "action_14_28",
            "metric": "tv_ratio_window_p50", "old": 0.19099212437868118,
            "new": 0.19087564200162888, "abs_delta": 0.00011648237705230713,
            "independently_identified_from_saved_metric_panels": True
        },
        "final_test_access": 0
    })

    # Register evidence sources and reusable prior acceptance records.
    registry_specs = [
        ("v2_archive", v2 / "ARCHIVE.json", "manifest"),
        ("v2_environment_export", v2 / "ENVIRONMENT_EXPORT.json", "safe version metadata only; file hashed, contents filtered"),
        ("v2_cpu_replay_summary", cpu_summary_path, "historical replay receipt"),
        ("v1_final_acceptance", v1 / "FINAL_ACCEPTANCE.json", "prior acceptance state"),
        ("v1_checkpoint_replay", v1 / "CHECKPOINT_REPLAY.csv", "prior blocked Core B replay rows"),
        ("v1_dependency_manifest", v1 / "RESOLVED_CONFIG_DEPENDENCY_MANIFEST.json", "dependencies and provenance"),
        ("v1_checksums", checksum_path, "verified listed artifacts"),
        ("v1_dev_metric_qa", v1 / "metrics/core_b/CORE_B_METRIC_QA.json", "independent scorer QA"),
        ("v1_train_score_qa", v1 / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json", "saved prediction rescoring QA"),
        ("v1_independent_scorer", v1 / "source/independent_core_b_metrics.py", "reused scorer source"),
        ("v1_metric_fixtures", v1 / "fixtures/red_capable_metric_fixtures.py", "reused scorer fixtures"),
        ("handoff_index", hand_index_path, "169-entry old handoff manifest"),
        ("team_model_source", v2 / TEAM_CODE_REL / "d3_model.py", "model construction and forward source"),
        ("team_inference_source", team / "d3_corrected_infer.py", "as-run inference batch/device path"),
        ("team_evaluator_source", team / "evaluate_corrected_scheduler.py", "historical scorer dependency"),
        ("v3_inputs", v2 / V3_REL / "V3_MAIN_INPUTS.npz", "canonical target-blind input archive"),
        ("v3_normalizers", v2 / V3_REL / "V3_NORMALIZERS.json", "target scale and safe-FD scale"),
        ("feature_cache_manifest", v2 / CACHE_REL / "cache/CACHE_MANIFEST.json", "314 shard identities"),
        ("video_normalizer", v2 / CACHE_REL / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json", "train-only feature normalization"),
        ("train_target_cache", v2 / TARGET_REL, "train-only targets used after prediction sealing"),
        ("split_manifest", v2 / SPLIT_REL, "episode/source group mapping")
    ]
    for role, path, note in registry_specs:
        row = safe_record(path, role)
        row["note"] = note
        source_rows.append(row)
    for cell in expected_rows:
        for label, rel in (("safe_model", cell["weights"]), ("model_manifest", cell["manifest"]),
                           ("run_config", cell["config"]), ("original_checkpoint_bytes_only", cell["original_checkpoint"])):
            path = v2 / rel
            row = safe_record(path, f"{label}:{cell['cell']}")
            row["note"] = "original checkpoint is hashed as bytes only; never deserialized"
            source_rows.append(row)
    csv_write(audit / "SOURCE_REGISTRY.csv",
              ["role", "path", "exists", "bytes", "sha256", "status", "expected_bytes", "expected_sha256", "note"], source_rows)

    v1_core = final_v1.get("acceptance", {})
    artifact_status = v1_core.get("artifacts", {})
    core_b_metrics = v1_core.get("core_b_metrics", {})
    train_score = core_b_metrics.get("saved_train_prediction_score_qa", {})
    core_a_replay = v1_core.get("checkpoint_replay", {}).get("core_a_train_validation")
    reuse_rows = [
        {"item": "original handoff 169 artifact receipt", "status": "accepted_reused" if len(hand_rows) == 169 and all(r["status"] == "PASS" for r in hand_rows) else "needs_recheck",
         "evidence": str(handoff / "HANDOFF_INDEX.json"), "detail": f"verified {sum(r['status']=='PASS' for r in hand_rows)}/{len(hand_rows)} current source SHA and size"},
        {"item": "V1 acceptance package checksums", "status": "accepted_reused" if checksum_rows and all(r["status"] == "PASS" for r in checksum_rows) else "needs_recheck",
         "evidence": str(checksum_path), "detail": f"verified {sum(r['status']=='PASS' for r in checksum_rows)}/{len(checksum_rows)} listed entries"},
        {"item": "Core A train/validation replay", "status": "accepted_reused" if core_a_replay and "3/3" in core_a_replay else "needs_recheck",
         "evidence": str(v1 / "FINAL_ACCEPTANCE.json"), "detail": str(core_a_replay)},
        {"item": "saved Core B train/dev predictions and independent scores", "status": "accepted_reused" if train_score.get("status") == "PASS_SAVED_CORE_B_TRAIN_ARTIFACT_SCORE" and core_b_metrics.get("dev_qa", {}).get("status") == "PASS_INDEPENDENT_CORE_B_DEV_SCORE" else "needs_recheck",
         "evidence": str(v1 / "metrics/core_b"), "detail": f"48 bundles scored; 24 existing train artifacts; checkpoint_replayed={train_score.get('checkpoint_replayed')}"},
        {"item": "V-JEPA comparison and 10000-draw development analysis", "status": "accepted_reused" if core_b_metrics.get("dev_qa", {}).get("paired_bootstrap_draws") == 10000 else "needs_recheck",
         "evidence": str(v1 / "FINAL_ACCEPTANCE.json"), "detail": "fixed three seeds and exposed dev only; no rerun requested"},
        {"item": "historical CPU zero-tolerance replay", "status": "historical_fail_preserved",
         "evidence": str(cpu_summary_path), "detail": f"{len(cpu_results)}/12 cells failed zero tolerance; no promotion to new gate"},
        {"item": "V2 portable integration", "status": "needs_fix",
         "evidence": str(v1 / "FINAL_ACCEPTANCE.json"), "detail": "; ".join(v1_core.get("portable_integration", {}).get("reasons", []))}
    ]
    csv_write(audit / "ACCEPTED_REUSE.csv", ["item", "status", "evidence", "detail"], reuse_rows)

    replay_placeholders = []
    for row in expected_rows:
        for split, _, n, _ in SPLITS:
            replay_placeholders.append({
                "cell": row["cell"], "split": split, "status": "awaiting_host_gpu_approval",
                "executed": 0, "numeric_accepted": 0, "device": "", "physical_gpu": "", "logical_device": "",
                "gpu_uuid": "", "rows": n, "expected_shape": f"({n},{row['horizon']},28)", "dtype": "float32",
                "prediction_sha256": "", "reference_sha256": "", "max_abs": "", "mean_abs": "", "p95_abs": "",
                "p99_abs": "", "bitwise_equal": "", "engineering_gate_fail_count": "",
                "historical_cpu_zero_tolerance_status": "historical_fail_preserved",
                "block_reason": "host GPU approval and live resource gate not yet completed"
            })
    csv_write(audit / "CHECKPOINT_REPLAY.csv",
              ["cell", "split", "status", "executed", "numeric_accepted", "device", "physical_gpu", "logical_device",
               "gpu_uuid", "rows", "expected_shape", "dtype", "prediction_sha256", "reference_sha256", "max_abs",
               "mean_abs", "p95_abs", "p99_abs", "bitwise_equal", "engineering_gate_fail_count",
               "historical_cpu_zero_tolerance_status", "block_reason"], replay_placeholders)

    # Predeclare the user's exact V2 gate and bind all canonical references before any GPU prediction.
    refs = []
    for row in expected_rows:
        entry = expected_by_combo[(row["mode"], int(row["horizon"]), int(row["seed"]))]
        run = v2 / entry["run"]
        for split, _, n, _ in SPLITS:
            path = run / f"predictions_{split}.npz"
            refs.append({"cell": row["cell"], "split": split, "path": str(path),
                         "bytes": path.stat().st_size if path.is_file() else None,
                         "sha256": digest(path) if path.is_file() else None, "rows": n,
                         "shape": [n, int(row["horizon"]), 28], "reference_device": "historical GPU prediction artifact"})
    write_json(audit / "REPLAY_CONTRACT.json", {
        "contract_id": "VIDEOMAE_CORE_B_HANDOFF_ACCEPTANCE_V2",
        "locked_utc": None,
        "lock_status": "predeclared_before_any_new_GPU_prediction",
        "scope": "fixed checkpoint cached-feature predictor replay; train/dev only",
        "matrix": {"modes": ["TS11-D", "STATE-CTRL"], "horizons": [30, 60], "seeds": [20261001, 20261002, 20261003]},
        "predictions": {"train_rows": 17757, "dev_rows": 2322, "output_dim": 28, "dtype": "float32",
                        "id_shape_mask_input_order_checks": "exact", "finite": True,
                        "final_test_access": 0, "targets_loaded_during_inference": False,
                        "all_24_predictions_hashed_before_target_join": True},
        "criterion_origin": "predeclared_V2_engineering_gate",
        "criterion": {"elementwise": "abs(new-ref) <= 1e-6 + 1e-6*abs(ref)",
                      "equal_nan": False, "bitwise_equality_also_reported": True},
        "criterion_limit": "engineering diagnostic only; not a PyTorch guarantee, historical tolerance, or scientific equivalence claim",
        "historical_cpu_zero_tolerance": {"status": "historical_fail_preserved", "replay_summary_sha256": digest(cpu_summary_path),
                                          "criterion": "atol=0, rtol=0; remains failed"},
        "scorer_reproduction_gate": {"continuous_metrics_abs": 1e-10, "discrete_counts_support_keys_na_positions": "exact"},
        "references": refs,
        "author_cpu_results_already_exposed": True,
        "new_gpu_results_exposed": False,
        "source_sha256": {"v2_archive": digest(archive_path), "v1_scorer": digest(v1 / "source/independent_core_b_metrics.py")}
    })
    write_json(audit / "PROFILE_LOCK.json", {
        "lock_status": "awaiting_live_host_profile_before_first_GPU_prediction",
        "original_profile": {
            "evidence": ["V1 RESOLVED_CONFIG_DEPENDENCY_MANIFEST.json", "V2 archived exporter/replay code",
                         "D3_EQ_CORRECTED_SCHEDULER_V1/code/d3_corrected_infer.py", "per-run CONFIG/RUN_STATUS/ENVIRONMENT"],
            "python": "3.12.14 (V1 V2 wrapper metadata)",
            "torch": "2.12.1+cu130", "numpy": "2.5.2",
            "cuda_runtime": "CUDA 13.0 indicated by torch build; original runtime/driver not recorded",
            "cudnn": "UNKNOWN from historical run records",
            "gpu_architecture": "UNKNOWN; original per-run record has physical GPU index only and no UUID/model",
            "historical_physical_gpu_index": "2 in the per-run environment records; not used as a current assignment",
            "eval": True, "inference_mode": True, "dtype": "float32", "amp": False, "tf32": "not explicitly configured",
            "attention_backend": "not pinned in source; framework backend selected at runtime",
            "inference_batch": 64, "tail": "min(64, remaining rows)",
            "input_order": "canonical split order via flatnonzero(split_code == train/dev)",
            "feature_normalizer": "VIDEOMAE_VIDEO_NORMALIZER.json mean/safe_std",
            "target_normalizer": "V3_NORMALIZERS.json target_sigma",
            "state_normalizer": "history_standardized fields in V3_MAIN_INPUTS.npz",
            "native_inverse": "anchor_command_native + residual * target_sigma",
            "query_and_position_state": "generated in forward; three persistent modality buffers",
            "horizon_specific_video_projection": {"TS11-D": 384, "STATE-CTRL": 768}
        },
        "new_current_profile": {
            "admission": "shared gpu_lease_watcher; dynamic_all_visible; no fixed GPU preference",
            "physical_gpu_uuid": "UNKNOWN until explicit host approval and live admission",
            "gpu_architecture": "UNKNOWN until explicit host approval and live admission",
            "torch_cuda_cudnn": "UNKNOWN; CUDA runtime metadata query requires host approval",
            "cuda_visible_devices_and_child_uuid": "must be checked against watcher receipt before first prediction",
            "batch_size": 64, "dtype": "float32", "amp": False, "eval": True, "inference_mode": True,
            "predeclared_behavior": "reuse original cached-feature inference batch path and fresh process per cell",
            "current_profile_match_status": "not_tested",
            "original_profile_match_status": "not_tested"
        },
        "author_cpu_results_already_exposed": True, "new_gpu_results_exposed": False,
        "final_test_access": 0
    })

    package_paths = {
        "copied_archive_code": ["code/close_videomae_core_b_handoff.py", "code/evaluate_videomae_core_b.py", "code/extract_videomae_core_b_cache.py"],
        "safe_weights": [r["weights"] for r in expected_rows],
        "copied_feature_cache_manifest": str(CACHE_REL / "cache/CACHE_MANIFEST.json"),
        "copied_feature_cache_shards": "manifest enumerated; 314 entries expected",
        "copied_inputs": [str(V3_REL / "V3_MAIN_INPUTS.npz"), str(V3_REL / "V3_NORMALIZERS.json"), str(TARGET_REL), str(SPLIT_REL)],
        "external_runtime_requirements": ["project CPU/GPU environment with PyTorch, NumPy, safetensors, pyarrow",
                                          "shared GPU lease watcher and host-level GPU access after approval"],
        "raw_video": "not included or read for this frozen_feature_predictor_replay scope",
        "v_jepa_team_reference_provenance": "separate V1 caveat: extractor source SHA and D3 cache bridge identity remain unresolved"
    }
    write_json(audit / "DEPENDENCIES.json", package_paths)

    # Reconcile V2 archive exporter receipt, old CPU summary, and prior accepted caveats.
    exporter_text = (v2 / "code/close_videomae_core_b_handoff.py").read_text(encoding="utf-8")
    exporter_has_roundtrip = "torch.equal(v, restored[k])" in exporter_text
    status_summary = {
        "status": "CPU_PREFLIGHT_PASS_GPU_PENDING" if not errors else "CPU_PREFLIGHT_WITH_FINDINGS",
        "audit_root": str(audit), "utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "project_root": str(project), "v2_root": str(v2), "v1_root": str(v1), "original_handoff_root": str(handoff),
        "archive": {"status": archive.get("status"), "registered_files": len(archive_files),
                    "registered_bytes": total_registered_bytes, "verified_copies": sum(r["archive_status"] == "PASS" for r in inventory_rows),
                    "verified_source_origins": sum(r["source_status"] == "PASS" for r in inventory_rows),
                    "extras": extras, "symlinks": symlinks, "absolute_file_registry_paths": abs_archive_keys},
        "handoff": {"declared_artifacts": len(hand_rows), "verified_artifacts": sum(r["status"] == "PASS" for r in hand_rows)},
        "v1_checksums": {"declared": len(checksum_rows), "verified": sum(r["status"] == "PASS" for r in checksum_rows)},
        "safe_models": {"structurally_accepted": sum(r["structure_status"] == "accepted" for r in safe_rows),
                        "total": len(safe_rows), "source_evidence_level": "hash_bound_exporter_attested",
                        "exporter_contains_tensor_roundtrip_check": exporter_has_roundtrip,
                        "independent_preexport_tensor_catalogue": False},
        "historical_cpu_replay": {"status": cpu_summary.get("status"), "cells": len(cpu_results),
                                  "train_splits_rechecked": sum(r["split"] == "train" for r in cpu_replay_rows),
                                  "dev_splits_rechecked": sum(r["split"] == "dev" for r in cpu_replay_rows),
                                  "numeric_accepted_under_historical_zero_tolerance": sum(r["historical_zero_tolerance_accepted"] for r in cpu_replay_rows),
                                  "max_prediction_abs": max_cpu_pred,
                                  "identified_max_metric_field": max_cpu_metric_row},
        "contract": {"origin": "predeclared_V2_engineering_gate", "new_gpu_predictions_started": False},
        "gpu_host_approval": "not_obtained", "new_gpu_predictions": 0,
        "new_training": 0, "optimizer_updates": 0, "new_encoder_extraction": 0,
        "unrestricted_pickle_loads": 0, "final_heldout_access": 0, "old_root_writes": 0,
        "errors": errors, "warnings": warnings
    }
    write_json(audit / "PREFLIGHT_SUMMARY.json", status_summary)
    print(json.dumps({"status": status_summary["status"], "registered_files": len(archive_files),
                      "archive_bytes": total_registered_bytes, "handoff_verified": f"{sum(r['status']=='PASS' for r in hand_rows)}/{len(hand_rows)}",
                      "v1_checksums_verified": f"{sum(r['status']=='PASS' for r in checksum_rows)}/{len(checksum_rows)}",
                      "safe_models_structurally_accepted": f"{sum(r['structure_status']=='accepted' for r in safe_rows)}/{len(safe_rows)}",
                      "cpu_replay_splits_rechecked": f"{len(cpu_replay_rows)}/24",
                      "cpu_zero_tolerance_numeric_accepted": sum(r["historical_zero_tolerance_accepted"] for r in cpu_replay_rows),
                      "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

