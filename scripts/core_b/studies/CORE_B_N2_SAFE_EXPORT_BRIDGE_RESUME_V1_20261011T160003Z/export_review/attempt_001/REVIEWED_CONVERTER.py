#!/usr/bin/env python3
"""Restricted CPU-only extraction of model tensors from the 12 pinned Core B checkpoints.

This file is intended to run only through the platform's require_escalated review
route after the accompanying PLAN.json and source hashes have been reviewed.
It never uses unrestricted pickle loading and never writes a full checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickletools
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "export_review" / "PLAN.json"
REQUEST_PATH = ROOT / "MODEL_EXPORT_REQUEST.csv"
SCHEMA_PATH = ROOT / "export_review" / "EXPECTED_STATE_SCHEMA.json"
SYMBOLS_PATH = ROOT / "export_review" / "EXACT_SYMBOLS.json"
NEGATIVE_PATH = ROOT / "export_review" / "NEGATIVE_TESTS.json"
RUNTIME_EVIDENCE_PATH = ROOT / "export_review" / "RUNTIME_API_EVIDENCE.md"
SAFE_ROOT = ROOT / "safe_models"
TMP_ROOT = ROOT / "tmp"
EXPECTED_BASELINE_GLOBALS = {
    "torch.nested._internal.nested_tensor.NestedTensor",
    "torch.nested._internal.nested_tensor._rebuild_njt",
    "torch.distributed._mesh_layout._MeshLayout",
}
DTYPE_LITERAL_RE = re.compile(r"[biufc][1248]\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    return f"{module}.{name}"


def table_sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def pickle_dtype_literals(checkpoint: Path) -> list[str]:
    with zipfile.ZipFile(checkpoint, "r") as archive:
        names = [name for name in archive.namelist() if name == "data.pkl" or name.endswith("/data.pkl")]
        if len(names) != 1:
            raise RuntimeError(f"expected one data.pkl member, found {len(names)}: {checkpoint}")
        operations = pickletools.genops(archive.read(names[0]))
        return sorted({arg for opcode, arg, _ in operations
                       if isinstance(arg, str) and DTYPE_LITERAL_RE.fullmatch(arg)})


def load_plan(expected_plan_sha256: str) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    if sha_file(PLAN_PATH) != expected_plan_sha256:
        raise RuntimeError("PLAN SHA differs from command-pinned platform-reviewed SHA")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("new_root") != str(ROOT):
        raise RuntimeError("PLAN new_root does not equal this converter's parent root")
    if sha_file(Path(__file__).resolve()) != plan.get("converter_sha256"):
        raise RuntimeError("converter SHA differs from the platform-reviewed PLAN")
    if sha_file(REQUEST_PATH) != plan.get("request_csv_sha256"):
        raise RuntimeError("MODEL_EXPORT_REQUEST.csv SHA differs from the platform-reviewed PLAN")
    if sha_file(SCHEMA_PATH) != plan.get("expected_state_schema_sha256"):
        raise RuntimeError("expected state schema SHA differs from the platform-reviewed PLAN")
    if sha_file(SYMBOLS_PATH) != plan.get("exact_symbols_sha256"):
        raise RuntimeError("exact symbol list SHA differs from the platform-reviewed PLAN")
    if sha_file(NEGATIVE_PATH) != plan.get("negative_tests_sha256"):
        raise RuntimeError("negative-test plan SHA differs from the platform-reviewed PLAN")
    if sha_file(RUNTIME_EVIDENCE_PATH) != plan.get("runtime_api_evidence_sha256"):
        raise RuntimeError("runtime API evidence SHA differs from the platform-reviewed PLAN")
    if not ROOT.is_dir() or not SAFE_ROOT.is_dir() or not TMP_ROOT.is_dir():
        raise RuntimeError("required task-owned root directories are absent")
    if SAFE_ROOT.is_symlink() or TMP_ROOT.is_symlink():
        raise RuntimeError("symlinked output or temporary root is forbidden")
    with REQUEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12 or len({row["model_id"] for row in rows}) != 12:
        raise RuntimeError("request must contain the exact 12 unique fixed H30 models")
    planned = plan.get("source_checkpoints", [])
    if len(planned) != 12:
        raise RuntimeError("PLAN must pin exactly 12 source checkpoints")
    by_id = {item["model_id"]: item for item in planned}
    if set(by_id) != {row["model_id"] for row in rows}:
        raise RuntimeError("request model ids differ from the platform-reviewed source list")
    for row in rows:
        pin = by_id[row["model_id"]]
        if row["checkpoint_path"] != pin["path"] or row["checkpoint_sha256"] != pin["sha256"]:
            raise RuntimeError(f"request source path/hash differs from PLAN for {row['model_id']}")
        source = Path(pin["path"])
        if not source.is_file() or sha_file(source) != pin["sha256"]:
            raise RuntimeError(f"pinned source checkpoint missing or changed: {source}")
        if row["model_code_sha256"] != pin["model_code_sha256"]:
            raise RuntimeError(f"model source code identity changed: {row['model_id']}")
        if row["training_protocol_lock_sha256"] != pin["training_protocol_lock_sha256"]:
            raise RuntimeError(f"training protocol lock identity changed: {row['model_id']}")
        if row["final_test_access"] != "0":
            raise RuntimeError(f"final-test access field is not zero: {row['model_id']}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return plan, rows, schema


def static_preflight(plan: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    initial_globals = sorted(qualified_name(x) for x in torch.serialization.get_safe_globals())
    if set(initial_globals) != EXPECTED_BASELINE_GLOBALS:
        raise RuntimeError(f"unexpected process-start safe globals: {initial_globals}")
    # These three PyTorch package-start entries are cleared in this isolated process.
    # Only the four exact NumPy symbols below are active inside the temporary scope.
    torch.serialization.clear_safe_globals()
    if torch.serialization.get_safe_globals():
        raise RuntimeError("could not clear process-local inherited user safe globals")
    union: set[str] = set()
    per_file: list[dict[str, Any]] = []
    expected_globals = set(plan["static_unsafe_globals"])
    expected_dtypes = set(plan["numpy_dtype_literals"])
    for row in rows:
        path = Path(row["checkpoint_path"])
        globals_here = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
        dtypes_here = pickle_dtype_literals(path)
        if set(globals_here) != expected_globals:
            raise RuntimeError(f"unreviewed GLOBAL set for {row['model_id']}: {globals_here}")
        if set(dtypes_here) != expected_dtypes:
            raise RuntimeError(f"unreviewed NumPy dtype literal for {row['model_id']}: {dtypes_here}")
        union.update(globals_here)
        per_file.append({"model_id": row["model_id"], "static_globals": globals_here,
                         "numpy_dtype_literals": dtypes_here})
    if sorted(union) != sorted(expected_globals):
        raise RuntimeError("static GLOBAL union changed")
    if os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is not None:
        raise RuntimeError("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD must be unset")
    return {"startup_safe_globals_cleared": initial_globals,
            "static_globals_union": sorted(union), "per_file": per_file}


def exact_numpy_symbols(plan: dict[str, Any]) -> list[Any]:
    reconstruct = np._core.multiarray._reconstruct
    dynamic_dtype = type(np.dtype("u4"))
    symbols = [reconstruct, np.ndarray, np.dtype, dynamic_dtype]
    actual = sorted(qualified_name(x) for x in symbols)
    if actual != sorted(plan["temporary_safe_globals"]):
        raise RuntimeError(f"runtime symbols differ from exact reviewed set: {actual}")
    if qualified_name(dynamic_dtype) != plan["dynamic_dtype_class"]:
        raise RuntimeError("dynamic dtype class differs from the reviewed NumPy runtime")
    return symbols


def expected_model(row: dict[str, str], schema: dict[str, Any], query_table: np.ndarray):
    source_dir = ROOT / "source"
    if sha_file(source_dir / "d3_model.py") != row["model_code_sha256"]:
        raise RuntimeError(f"copied model code hash mismatch for {row['model_id']}")
    sys.path.insert(0, str(source_dir))
    from d3_model import D3TimeSpacePredictor
    if row["query"] == "Q0":
        return D3TimeSpacePredictor(horizon=30, seed=int(row["seed"]), video_mode=row["model"])
    if sha_file(source_dir / "fixed_future_query_model.py") != row["query_wrapper_code_sha256"]:
        raise RuntimeError(f"copied Q1 wrapper code hash mismatch for {row['model_id']}")
    from fixed_future_query_model import FixedFutureQueryOverride
    base = D3TimeSpacePredictor(horizon=30, seed=int(row["seed"]), video_mode=row["model"])
    return FixedFutureQueryOverride(base, torch.from_numpy(query_table))


def tensor_fingerprint(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return sha_bytes(array.tobytes(order="C"))


def validate_and_copy_state(row: dict[str, str], payload: dict[str, Any],
                            schema: dict[str, Any], query_table: np.ndarray) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = Path(row["checkpoint_path"])
    complete_path = Path(row["run_complete_path"])
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    expected_meta = {
        "status": "PASS", "epochs": 40, "optimizer_steps": 11120,
        "seed": int(row["seed"]), "horizon": 30, "config": row["model"],
        "checkpoint_last_sha256": row["checkpoint_sha256"],
    }
    for key, expected in expected_meta.items():
        if complete.get(key) != expected:
            raise RuntimeError(f"RUN_COMPLETE {key} mismatch for {row['model_id']}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint top level is not a primitive mapping: {row['model_id']}")
    payload_meta = {
        "epoch": 40, "optimizer_steps": 11120, "seed": int(row["seed"]),
        "horizon": 30, "config": row["model"], "study_protocol_hash": row["training_protocol_lock_sha256"],
        "final_test_access": 0,
    }
    for key, expected in payload_meta.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"checkpoint payload {key} mismatch for {row['model_id']}")
    if row["query"] == "Q1":
        expected_query = row["accepted_query_table_sha256"]
        if payload.get("query_table_sha256") != expected_query or payload.get("query_table_buffer_sha256") != expected_query:
            raise RuntimeError(f"Q1 payload query identity mismatch for {row['model_id']}")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"payload['model'] is not a mapping: {row['model_id']}")
    expected_state = schema[row["query"]]["state_dict"]
    if set(state) != set(expected_state):
        raise RuntimeError(f"state key set differs from the exact model schema: {row['model_id']}")
    clean: dict[str, torch.Tensor] = {}
    catalogue: list[dict[str, Any]] = []
    for name in sorted(expected_state):
        value = state[name]
        if type(value) is not torch.Tensor:
            raise RuntimeError(f"state value is not a plain torch.Tensor: {row['model_id']}::{name}")
        spec = expected_state[name]
        if list(value.shape) != spec["shape"] or str(value.dtype).removeprefix("torch.") != spec["dtype"]:
            raise RuntimeError(f"tensor shape/dtype mismatch: {row['model_id']}::{name}")
        if value.dtype != torch.float32 or value.layout != torch.strided or value.is_sparse:
            raise RuntimeError(f"tensor dtype/layout is outside the reviewed schema: {row['model_id']}::{name}")
        copied = value.detach().to(device="cpu").contiguous().clone()
        clean[name] = copied
        catalogue.append({"name": name, "shape": list(copied.shape), "dtype": "float32",
                          "numel": int(copied.numel()), "source_tensor_sha256": tensor_fingerprint(copied)})
    if row["query"] == "Q0":
        if "future_query_table" in clean:
            raise RuntimeError("Q0 query state unexpectedly contains the Q1 buffer")
        if json.loads(row["required_auxiliary_names"]) != []:
            raise RuntimeError("Q0 required_auxiliary_names must be explicitly []")
    else:
        table_tensor = clean.get("future_query_table")
        if table_tensor is None or list(table_tensor.shape) != [30, 128]:
            raise RuntimeError("Q1 persistent future_query_table is missing or has wrong shape")
        if table_sha(table_tensor.numpy()) != row["accepted_query_table_sha256"]:
            raise RuntimeError("Q1 stored query table does not match accepted table hash")
        if json.loads(row["required_auxiliary_names"]) != []:
            raise RuntimeError("Q1 required_auxiliary_names must be explicitly []")
    model = expected_model(row, schema, query_table)
    expected_param_count = int(schema["expected_parameter_count"])
    from d3_model import parameter_count
    if parameter_count(model) != expected_param_count:
        raise RuntimeError(f"unexpected parameter count for {row['model_id']}")
    result = {"model_id": row["model_id"], "state_tensor_count": len(catalogue),
              "parameter_count": expected_param_count, "tensor_catalogue": catalogue,
              "source_checkpoint_path": str(checkpoint), "source_checkpoint_sha256": row["checkpoint_sha256"],
              "query": row["query"], "query_table_sha256": row["accepted_query_table_sha256"]}
    del model
    return clean, result


def execute(plan: dict[str, Any], rows: list[dict[str, str]], schema: dict[str, Any]) -> int:
    initial = static_preflight(plan, rows)
    symbols = exact_numpy_symbols(plan)
    with np.load(Path(plan["query_archive_path"]), allow_pickle=False) as archive:
        query_table = np.asarray(archive["candidate_h30"], dtype=np.float32).copy()
    if query_table.shape != (30, 128) or table_sha(query_table) != plan["query_table_sha256"]:
        raise RuntimeError("accepted Q1 query archive/table hash mismatch")
    final_dirs = [SAFE_ROOT / row["model_id"] for row in rows]
    if any(path.exists() for path in final_dirs):
        raise FileExistsError("safe model destination already exists; converter never overwrites")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = TMP_ROOT / f"safe_export_staging_{stamp}_{os.getpid()}"
    staging.mkdir()
    states: dict[str, dict[str, torch.Tensor]] = {}
    catalogues: dict[str, dict[str, Any]] = {}
    pilot_ids = set(plan["pilot_model_ids"])
    by_id = {row["model_id"]: row for row in rows}
    if len(pilot_ids) != 2 or not pilot_ids.issubset(by_id):
        raise RuntimeError("PLAN must specify exactly one representative Q0 and one Q1 pilot")
    pilot_results = []
    # This is the sole temporary allowlist scope. It contains only the four reviewed NumPy symbols.
    with torch.serialization.safe_globals(symbols):
        if sorted(qualified_name(x) for x in torch.serialization.get_safe_globals()) != sorted(plan["temporary_safe_globals"]):
            raise RuntimeError("active safe-global scope differs from exact PLAN set")
        ordered = [by_id[mid] for mid in plan["pilot_model_ids"]] + [row for row in rows if row["model_id"] not in pilot_ids]
        for index, row in enumerate(ordered):
            payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=True)
            clean, catalog = validate_and_copy_state(row, payload, schema, query_table)
            states[row["model_id"]] = clean
            catalogues[row["model_id"]] = catalog
            if row["model_id"] in pilot_ids:
                pilot_results.append({"model_id": row["model_id"], "status": "PASS",
                                      "source_checkpoint_sha256": row["checkpoint_sha256"],
                                      "state_tensor_count": catalog["state_tensor_count"],
                                      "parameter_count": catalog["parameter_count"]})
            del payload
            gc.collect()
            if len(pilot_results) == 2:
                write_json(ROOT / "export_review" / "PILOT_ACCEPTANCE.json",
                           {"schema_version": 1, "status": "PASS_REPRESENTATIVE_Q0_Q1",
                            "created_utc": utc_now(), "static_preflight": initial,
                            "pilot_models": pilot_results,
                            "note": "CPU-only weights_only=True; exact temporary NumPy safe_globals; no output model file created by pilot stage."})
        if set(states) != set(by_id):
            raise RuntimeError("not all 12 fixed checkpoints were validated")
        for row in rows:
            model_id = row["model_id"]
            model_stage = staging / model_id
            model_stage.mkdir()
            tensor_map = states[model_id]
            model_file = model_stage / "model.safetensors"
            save_file(tensor_map, str(model_file))
            catalogue = catalogues[model_id]
            catalogue_obj = {"schema_version": 1, "model_id": model_id,
                             "source_checkpoint_sha256": row["checkpoint_sha256"],
                             "tensor_count": catalogue["state_tensor_count"],
                             "tensors": catalogue["tensor_catalogue"]}
            write_json(model_stage / "TENSOR_CATALOGUE.json", catalogue_obj)
            query_obj = {
                "schema_version": 1, "query": row["query"], "query_variant": row["query_variant"],
                "horizon_steps": 30, "query_table_sha256": row["accepted_query_table_sha256"],
                "query_archive_path": row["query_archive_path"], "query_archive_sha256": row["query_archive_sha256"],
                "q0_forward_formula": "torch.arange(1, horizon+1, dtype=float32)/30.0 -> sinusoidal(times,128)" if row["query"] == "Q0" else None,
                "q1_buffer_key": "future_query_table" if row["query"] == "Q1" else None,
                "q1_table_shape": [30, 128] if row["query"] == "Q1" else None,
                "required_auxiliary_names": json.loads(row["required_auxiliary_names"]),
            }
            write_json(model_stage / "QUERY_SPEC.json", query_obj)
            buffer_obj = {"schema_version": 1, "query": row["query"],
                          "buffers": schema[row["query"]]["buffers"],
                          "required_auxiliary_names": json.loads(row["required_auxiliary_names"])}
            write_json(model_stage / "BUFFER_INVENTORY.json", buffer_obj)
            manifest = {
                "schema_version": 1, "status": "EXPORTED_PENDING_FRESH_PROCESS_ACCEPTANCE",
                "model_id": model_id, "model": row["model"], "query": row["query"],
                "seed": int(row["seed"]), "horizon_steps": 30, "epoch": 40, "optimizer_steps": 11120,
                "source_route": "restricted_weights_only_recovery_via_platform_review",
                "source_checkpoint_path": row["checkpoint_path"],
                "source_checkpoint_sha256": row["checkpoint_sha256"],
                "source_checkpoint_epoch40_sha256": row["checkpoint_epoch40_sha256"],
                "source_run_complete_path": row["run_complete_path"],
                "source_run_complete_sha256": row["run_complete_sha256"],
                "model_code_path": row["model_code_path"], "model_code_sha256": row["model_code_sha256"],
                "query_wrapper_code_path": row["query_wrapper_code_path"],
                "query_wrapper_code_sha256": row["query_wrapper_code_sha256"],
                "training_protocol_lock_sha256": row["training_protocol_lock_sha256"],
                "state_key_path": "payload['model']",
                "source_tensor_count": catalogue["state_tensor_count"],
                "parameter_count": catalogue["parameter_count"],
                "model_file": model_file.name, "model_file_sha256": sha_file(model_file),
                "tensor_catalogue_sha256": sha_file(model_stage / "TENSOR_CATALOGUE.json"),
                "query_spec_sha256": sha_file(model_stage / "QUERY_SPEC.json"),
                "buffer_inventory_sha256": sha_file(model_stage / "BUFFER_INVENTORY.json"),
                "accepted_profile_sidecar_path": row["accepted_profile_sidecar_path"],
                "accepted_profile_sidecar_sha256": row["accepted_profile_sidecar_sha256"],
                "accepted_n1_dev_prediction_path": row["accepted_n1_dev_prediction_path"],
                "accepted_n1_dev_prediction_sha256": row["accepted_n1_dev_prediction_sha256"],
                "input_sha256": row["input_sha256"],
                "feature_cache_manifest_sha256": row["feature_cache_manifest_sha256"],
                "video_normalizer_sha256": row["video_normalizer_sha256"],
                "target_normalizer_sha256": row["target_normalizer_sha256"],
                "query_table_sha256": row["accepted_query_table_sha256"],
                "load_method": "torch.load(map_location='cpu', weights_only=True)",
                "temporary_safe_globals": plan["temporary_safe_globals"],
                "optimizer_scheduler_rng_saved": False, "full_payload_saved": False,
                "final_test_access": 0,
            }
            write_json(model_stage / "EXPORT_MANIFEST.json", manifest)
        if torch.serialization.get_safe_globals():
            raise RuntimeError("temporary safe_globals did not clear at scope exit")
    # Only task-owned staging directories are moved into the guarded safe_models root.
    for row in rows:
        source_dir = staging / row["model_id"]
        target_dir = SAFE_ROOT / row["model_id"]
        if target_dir.exists() or target_dir.is_symlink():
            raise FileExistsError(f"refuse to overwrite {target_dir}")
        os.replace(source_dir, target_dir)
    write_json(staging / "STAGING_COMPLETE.json",
               {"status": "PASS", "created_utc": utc_now(), "model_ids": [row["model_id"] for row in rows]})
    write_json(ROOT / "export_review" / "RESTRICTED_CONVERTER_EXECUTION.json",
               {"status": "PASS_EXPORTS_WRITTEN_FRESH_RELOAD_PENDING",
                "created_utc": utc_now(), "converter_sha256": sha_file(Path(__file__).resolve()),
                "plan_sha256": sha_file(PLAN_PATH), "models": len(rows),
                "platform_decision_record": "See APPROVAL_DECISION_RECORD.json from actual require_escalated tool result."})
    print(json.dumps({"status": "PASS_EXPORTS_WRITTEN_FRESH_RELOAD_PENDING",
                      "models": len(rows), "safe_models_root": str(SAFE_ROOT),
                      "staging_root": str(staging)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    plan, rows, schema = load_plan(args.expected_plan_sha256)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS_NO_DESERIALIZATION_NO_MODEL_OUTPUT",
                          "models": len(rows), "source_files_hash_verified": len(rows),
                          "converter_sha256": sha_file(Path(__file__).resolve()),
                          "plan_sha256": sha_file(PLAN_PATH)}, sort_keys=True))
        return 0
    if args.execute:
        return execute(plan, rows, schema)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
