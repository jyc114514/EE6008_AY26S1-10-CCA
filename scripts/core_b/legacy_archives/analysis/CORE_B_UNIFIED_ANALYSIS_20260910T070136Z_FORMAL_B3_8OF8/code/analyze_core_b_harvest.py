#!/usr/bin/env python3
"""Independent Core B B1/B2/B3 harvest and statistical analysis.

The program is fail-closed: it does not calculate formal B3 results until the
pre-existing B3 controller is terminal and all eight logical cells have
complete, contract-compatible artifacts.  It never starts training, never
accesses final-test data, and keeps the old contextual-token B2 proxy outside
the formal pixel-sampling matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ARCHIVE = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z"
EXP = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
OUT = Path(os.environ.get("CORE_B_ANALYSIS_OUT", str(ARCHIVE / "core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z")))
INPUT_NPZ = EXP / "inputs/B2_INPUTS_0125.npz"
TARGET_NORM = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
B1_ACCEPTANCE = PROJECT / "repo/ee6008-humanoid-vjepa/reports/coordination/CORE_A_CORE_B_DUAL_GPU_20260903T064316Z/recovery_gpu0_contractfix_20260904T073316Z/core_b/b1_harvest_20260904T082132Z/B1_RUN_ACCEPTANCE.json"
B1_INPUTS = PROJECT / "repo/ee6008-humanoid-vjepa/reports/coordination/CORE_A_CORE_B_DUAL_GPU_20260903T064316Z/recovery_gpu0_contractfix_20260904T073316Z/core_b/B1_INFERENCE_INPUTS.json"
B1_CANONICAL_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/coordination/CORE_A_CORE_B_DUAL_GPU_20260903T064316Z/recovery_gpu0_contractfix_20260904T073316Z/core_b/b1_inference_20260904T073316Z"
B3_CONTROLLER = EXP / "B3_CONTROLLER_STATUS.json"
B2_ACCEPTANCE = EXP / "TRUE_B2_ACCEPTANCE.json"
B2_SELECTION = EXP / "B2_VALID_SELECTION.json"
B2_FIDELITY = EXP / "B2_PROTOCOL_FIDELITY_AUDIT.json"
B23_PROTOCOL = EXP / "RESOLVED_PROTOCOL.json"
B23_MATRIX = EXP / "B2_B3_MATRIX.json"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260909
PERM_REPS = 20_000
PERM_SEED = 20260910
STEPS = 30
DIMS = 28
ARM = slice(14, 28)
HAND = slice(0, 14)
FULL = slice(0, 28)


class HarvestBlocked(RuntimeError):
    """Raised when the existing matrix is not safe to harvest yet."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, label: str) -> None:
    if not path.is_file():
        raise HarvestBlocked(f"missing {label}: {path}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path, "prediction/input NPZ")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def id_reindex(ids: np.ndarray, expected_ids: np.ndarray) -> np.ndarray:
    positions = {int(value): index for index, value in enumerate(ids.tolist())}
    try:
        return np.asarray([positions[int(value)] for value in expected_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise HarvestBlocked(f"episode ID set mismatch; missing {exc.args[0]}") from exc


def validate_arrays(
    path: Path,
    expected_ids: np.ndarray,
    expected_target_std: np.ndarray,
    expected_target_native: np.ndarray,
    expected_last_action: np.ndarray,
    required_keys: set[str],
) -> dict[str, np.ndarray]:
    arrays = load_npz(path)
    missing = required_keys.difference(arrays)
    if missing:
        raise HarvestBlocked(f"{path} missing keys {sorted(missing)}")
    ids = arrays["episode_index"]
    if ids.shape != (len(expected_ids),) or len(np.unique(ids)) != len(ids):
        raise HarvestBlocked(f"{path} episode_index schema/uniqueness failed")
    order = id_reindex(ids, expected_ids)
    for name, shape in {
        "targets_standardized": expected_target_std.shape,
        "predictions_standardized": expected_target_std.shape,
        "targets_native": expected_target_native.shape,
        "predictions_native": expected_target_native.shape,
    }.items():
        if arrays[name].shape != shape:
            raise HarvestBlocked(f"{path} {name} shape {arrays[name].shape} != {shape}")
        if not np.isfinite(arrays[name]).all():
            raise HarvestBlocked(f"{path} {name} contains non-finite values")
    target_std = arrays["targets_standardized"][order]
    target_native = arrays["targets_native"][order]
    if not np.array_equal(target_std, expected_target_std):
        raise HarvestBlocked(f"{path} targets_standardized do not match canonical targets by episode ID")
    native_roundtrip_abs = float(np.max(np.abs(target_native.astype(np.float64) - expected_target_native.astype(np.float64))))
    if not np.allclose(target_native, expected_target_native, rtol=1e-6, atol=5e-7):
        raise HarvestBlocked(f"{path} targets_native round-trip mismatch max_abs={native_roundtrip_abs}")
    if "last_observed_action" in arrays:
        if arrays["last_observed_action"].shape != expected_last_action.shape:
            raise HarvestBlocked(f"{path} last_observed_action shape failed")
        last_action = arrays["last_observed_action"][order]
        if not np.array_equal(last_action, expected_last_action):
            raise HarvestBlocked(f"{path} last_observed_action is not canonical")
    else:
        last_action = expected_last_action.copy()
    result = {key: arrays[key][order] for key in arrays}
    result["episode_index"] = expected_ids.copy()
    result["last_observed_action"] = last_action
    result["_target_native_exact"] = np.asarray(np.array_equal(target_native, expected_target_native))
    result["_target_native_roundtrip_max_abs"] = np.asarray(native_roundtrip_abs)
    return result


def load_canonical_inputs() -> dict[str, np.ndarray]:
    arrays = load_npz(INPUT_NPZ)
    required = {"episode_index", "split_code", "original_task_index", "targets", "targets_standardized", "last_observed_action"}
    if not required.issubset(arrays):
        raise HarvestBlocked(f"canonical inputs missing {sorted(required.difference(arrays))}")
    if arrays["targets"].shape != (3355, STEPS, DIMS) or arrays["targets_standardized"].shape != (3355, STEPS, DIMS):
        raise HarvestBlocked("canonical input target shape failed")
    train = arrays["split_code"] == 0
    dev = arrays["split_code"] == 1
    if int(train.sum()) != 2741 or int(dev.sum()) != 614:
        raise HarvestBlocked(f"canonical split count failed: train={train.sum()} dev={dev.sum()}")
    return {
        "all": arrays,
        "train_mask": train,
        "dev_mask": dev,
        "ids": arrays["episode_index"][dev],
        "tasks": arrays["original_task_index"][dev],
        "target_native": arrays["targets"][dev].astype(np.float32),
        "target_std": arrays["targets_standardized"][dev].astype(np.float32),
        "last_action": arrays["last_observed_action"][dev].astype(np.float32),
    }


def read_run_meta(run_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("RUN_STATUS.json", "RUN_CONFIG_LOCKED.json", "RUN_METRICS.json", "RUN_COMPLETE.json"):
        path = run_dir / name
        if path.is_file():
            meta[name[:-5].lower()] = read_json(path)
    return meta


def check_complete_meta(meta: dict[str, Any], label: str) -> None:
    status = meta.get("run_status", {})
    metrics = meta.get("run_metrics", {})
    if status.get("status") != "complete" or status.get("exit_code") not in (0, None):
        raise HarvestBlocked(f"{label} status is not complete: {status.get('status')} exit={status.get('exit_code')}")
    if metrics and metrics.get("status") not in (None, "complete", "schema_only"):
        raise HarvestBlocked(f"{label} RUN_METRICS status is not complete")
    if status.get("final_test_accessed", 0) != 0 or metrics.get("final_test_accessed", 0) != 0:
        raise HarvestBlocked(f"{label} final-test access is nonzero")


def record_from_prediction(
    *,
    cell: str,
    stage: str,
    branch: str,
    video_input: str,
    adaptation: str,
    prediction_path: Path,
    run_dir: Path,
    inputs: dict[str, np.ndarray],
    source_kind: str,
    canonical_status: str,
    required_keys: set[str],
    raw_lora: bool = False,
    reused_from: str = "",
) -> dict[str, Any]:
    arrays = validate_arrays(prediction_path, inputs["ids"], inputs["target_std"], inputs["target_native"], inputs["last_action"], required_keys)
    meta = read_run_meta(run_dir)
    check_complete_meta(meta, cell)
    config = meta.get("run_config_locked", {})
    if config.get("final_test_accessed", 0) != 0:
        raise HarvestBlocked(f"{cell} config final-test access failed")
    if config.get("development_validation_count", 614) != 614:
        raise HarvestBlocked(f"{cell} development count failed")
    if raw_lora:
        if config.get("feature_normalization_fit") != "train rows only" or config.get("feature_normalization_key") not in {"dynamic_pooled", "static_pooled"} or not config.get("feature_normalization_sha256"):
            raise HarvestBlocked(f"{cell} corrected raw-LoRA feature normalization contract failed")
        if config.get("canonical_inference") != "unmerged_base_plus_adapter":
            raise HarvestBlocked(f"{cell} raw-LoRA canonical inference is not unmerged")
        expected_modules = {"blocks.10.attn.qkv", "blocks.10.attn.proj", "blocks.11.attn.qkv", "blocks.11.attn.proj"}
        if set(config.get("inserted_modules", [])) != expected_modules:
            raise HarvestBlocked(f"{cell} LoRA inserted module set failed")
        metrics = meta.get("run_metrics", {})
        if metrics.get("base_parameters_frozen") is not True or metrics.get("adapter_gradient_smoke") is not True:
            raise HarvestBlocked(f"{cell} base-freeze/adapter-gradient evidence failed")
    return {
        "cell": cell,
        "stage": stage,
        "branch": branch,
        "video_input": video_input,
        "adaptation": adaptation,
        "prediction_path": prediction_path,
        "run_dir": run_dir,
        "source_kind": source_kind,
        "canonical_status": canonical_status,
        "raw_lora": raw_lora,
        "reused_from": reused_from,
        "arrays": arrays,
        "meta": meta,
        "config": config,
    }


def load_b1_records(inputs: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    acceptance = read_json(B1_ACCEPTANCE)
    if acceptance.get("accepted") != 7 or acceptance.get("required") != 7 or acceptance.get("final_test_access", 0) != 0 or acceptance.get("core_b_b1_retrained") is not False:
        raise HarvestBlocked("B1 canonical acceptance is not 7/7 non-retrained")
    input_manifest = read_json(B1_INPUTS)
    source_runs = {row["lane"]: Path(row["run_dir"]) for row in input_manifest.get("rows", [])}
    records = []
    acceptance_rows = []
    mapping = {
        "B-STATE": ("state_only", "none", "frozen"),
        "B-D-P-F": ("video_only", "dynamic", "frozen"),
        "B-S-P-F": ("video_only", "static", "frozen"),
        "B-D-A-F": ("video_only", "dynamic", "frozen"),
        "B-S-A-F": ("video_only", "static", "frozen"),
        "B-D-FUS-P-F": ("fusion", "dynamic", "frozen"),
        "B-S-FUS-P-F": ("fusion", "static", "frozen"),
    }
    for row in acceptance.get("rows", []):
        lane = row["lane"]
        if row.get("status") != "pass" or row.get("best_status") != "pass" or row.get("last_status") != "pass":
            raise HarvestBlocked(f"B1 {lane} acceptance row failed")
        best = Path(row["best_prediction_bundle"])
        last = Path(row["last_prediction_bundle"])
        source_run = source_runs.get(lane)
        if source_run is None:
            raise HarvestBlocked(f"B1 {lane} source run not found")
        if not best.is_file() or not last.is_file():
            raise HarvestBlocked(f"B1 {lane} canonical best/last prediction missing")
        for path in (best.parent / "RUN_STATUS.json", best.parent / "RUN_CONFIG_LOCKED.json", best.parent / "RUN_METRICS.json", best.parent / "RUN_COMPLETE.json"):
            require(path, f"B1 {lane} canonical marker")
        source_checkpoint = source_run / "checkpoint_best.pt"
        require(source_checkpoint, f"B1 {lane} source checkpoint")
        if sha256(source_checkpoint) != row.get("best_checkpoint_sha256"):
            raise HarvestBlocked(f"B1 {lane} source checkpoint hash mismatch")
        branch, video, adaptation = mapping[lane]
        rec = record_from_prediction(
            cell=lane,
            stage="B1",
            branch=branch,
            video_input=video,
            adaptation=adaptation,
            prediction_path=best,
            run_dir=best.parent,
            inputs=inputs,
            source_kind="canonical_b1_inference_bundle",
            canonical_status="accepted",
            required_keys={"episode_index", "original_task_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native", "last_observed_action"},
        )
        rec["source_training_run"] = source_run
        rec["last_prediction_path"] = last
        records.append(rec)
        acceptance_rows.append({**row, "independent_status": "accepted", "source_training_run": source_run, "source_checkpoint_sha256_verified": True})
    if {r["cell"] for r in records} != set(mapping):
        raise HarvestBlocked("B1 logical cell set mismatch")
    return records, acceptance_rows


def load_b2_records(inputs: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    acceptance = read_json(B2_ACCEPTANCE)
    if acceptance.get("accepted_count") != 6 or acceptance.get("logical_coverage") != "6/6" or acceptance.get("status") != "pass" or acceptance.get("final_test_access", 0) != 0 or acceptance.get("pixel_observation_protocol_verified") is not True:
        raise HarvestBlocked("true-pixel B2 acceptance is not 6/6")
    selection = read_json(B2_SELECTION)
    if selection.get("status") != "pass" or selection.get("selected_interval_seconds") != 0.125 or selection.get("selected_unique_frames") != 32:
        raise HarvestBlocked("B2 selection is not the locked dense fallback")
    records = []
    rows = []
    formal_sources = {
        "B2-V-0125": EXP / "runs/B2-V-0125",
        "B2-FUS-0125": EXP / "runs/B2-FUS-0125",
        "B2-V-0250": EXP / "runs/TRUE-B2-V-0250",
        "B2-FUS-0250": EXP / "runs/TRUE-B2-FUS-0250",
        "B2-V-0500": EXP / "runs/TRUE-B2-V-0500",
        "B2-FUS-0500": EXP / "runs/TRUE-B2-FUS-0500",
    }
    spec = {
        "B2-V-0125": ("video_only", "dynamic", 0.125, 32, "valid_dense32_reference"),
        "B2-FUS-0125": ("fusion", "dynamic", 0.125, 32, "valid_dense32_reference"),
        "B2-V-0250": ("video_only", "dynamic", 0.25, 16, "true_pixel"),
        "B2-FUS-0250": ("fusion", "dynamic", 0.25, 16, "true_pixel"),
        "B2-V-0500": ("video_only", "dynamic", 0.5, 8, "true_pixel"),
        "B2-FUS-0500": ("fusion", "dynamic", 0.5, 8, "true_pixel"),
    }
    acceptance_by_condition = {r["condition"]: r for r in acceptance.get("rows", [])}
    for cell, run_dir in formal_sources.items():
        source_id = cell if cell.endswith("0125") else "TRUE-" + cell
        a = acceptance_by_condition[source_id]
        if not a.get("complete_ok") or not a.get("contract_ok") or not a.get("ids_ok") or not a.get("targets_ok"):
            raise HarvestBlocked(f"B2 {cell} acceptance subgate failed")
        prediction = run_dir / "predictions_best.npz"
        rec = record_from_prediction(
            cell=cell,
            stage="B2",
            branch=spec[cell][0],
            video_input=spec[cell][1],
            adaptation="frozen",
            prediction_path=prediction,
            run_dir=run_dir,
            inputs=inputs,
            source_kind="true_pixel_b2" if spec[cell][4] == "true_pixel" else "reused_dense_b2_control",
            canonical_status="accepted",
            required_keys={"episode_index", "original_task_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native", "last_observed_action"},
        )
        rec["interval_seconds"] = spec[cell][2]
        rec["unique_observation_frames"] = spec[cell][3]
        rec["encoder_slots"] = 32
        records.append(rec)
        rows.append({**a, "logical_cell": cell, "independent_status": "accepted", "source_kind": rec["source_kind"], "prediction_sha256": sha256(prediction)})
    proxy_rows = []
    proxy_sources = {
        "PROXY-B2-V-0250": (EXP / "runs/B2-V-0250", "video_only", 0.25, 16),
        "PROXY-B2-FUS-0250": (EXP / "runs/B2-FUS-0250", "fusion", 0.25, 16),
        "PROXY-B2-V-0500": (EXP / "runs/B2-V-0500", "video_only", 0.5, 8),
        "PROXY-B2-FUS-0500": (EXP / "runs/B2-FUS-0500", "fusion", 0.5, 8),
    }
    for cell, (run_dir, branch, interval, frames) in proxy_sources.items():
        prediction = run_dir / "predictions_best.npz"
        if not prediction.is_file():
            continue
        rec = record_from_prediction(
            cell=cell,
            stage="B2_PROXY",
            branch=branch,
            video_input="dynamic",
            adaptation="frozen",
            prediction_path=prediction,
            run_dir=run_dir,
            inputs=inputs,
            source_kind="auxiliary_token_aggregation_proxy",
            canonical_status="proxy_auxiliary",
            required_keys={"episode_index", "original_task_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native", "last_observed_action"},
        )
        rec["interval_seconds"] = interval
        rec["unique_observation_frames"] = frames
        records.append(rec)
        proxy_rows.append({"cell": cell, "run_dir": run_dir, "status": "proxy_auxiliary", "reason": "selected/duplicated dense contextual tokens after the full encoder", "prediction_sha256": sha256(prediction)})
    return records, rows, proxy_rows


def controller_terminal() -> dict[str, Any]:
    controller = read_json(B3_CONTROLLER)
    status = controller.get("status")
    if status not in {"complete", "complete_pending_analysis", "terminal"}:
        raise HarvestBlocked(f"B3 controller is not terminal: {status}; active_wave={controller.get('active_wave')}")
    active_wave = controller.get("active_wave", [])
    if active_wave:
        # complete_pending_analysis can be written before the controller
        # clears the just-finished wave. Reconcile from the cell's own terminal
        # markers without mutating the controller source file.
        if status == "complete_pending_analysis":
            unresolved = []
            for item in active_wave:
                condition = item.get("condition", "")
                candidates = [EXP / "runs/corrected" / condition / "RUN_STATUS.json", EXP / "runs" / condition / "RUN_STATUS.json"]
                cell_status = next((read_json(path) for path in candidates if path.is_file()), {})
                if cell_status.get("status") != "complete" or cell_status.get("exit_code") != 0:
                    unresolved.append({"condition": condition, "status": cell_status.get("status"), "exit_code": cell_status.get("exit_code")})
            if unresolved:
                raise HarvestBlocked(f"B3 controller active wave has unresolved cells: {unresolved}")
            controller = dict(controller)
            controller["active_wave_reconciled_as_completed"] = active_wave
            controller["active_wave"] = []
        else:
            raise HarvestBlocked(f"B3 controller still has active wave: {active_wave}")
    if controller.get("final_test_access", 0) != 0:
        raise HarvestBlocked("B3 controller final-test access is nonzero")
    return controller


def load_b3_records(inputs: dict[str, np.ndarray], b2_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    controller = controller_terminal()
    b2_by_cell = {r["cell"]: r for r in b2_records if r["stage"] == "B2" and r["source_kind"] != "auxiliary_token_aggregation_proxy"}
    mapping = [
        ("B3-V-D-F", b2_by_cell["B2-V-0125"]["run_dir"], b2_by_cell["B2-V-0125"]["prediction_path"], "video_only", "dynamic", "frozen", False, "B2-V-0125"),
        ("B3-V-D-L", EXP / "runs/corrected/B3-V-D-L", EXP / "runs/corrected/B3-V-D-L/predictions_best.npz", "video_only", "dynamic", "lora", True, ""),
        ("B3-V-S-F", EXP / "runs/B3-V-S-F", EXP / "runs/B3-V-S-F/predictions_best.npz", "video_only", "static", "frozen", False, ""),
        ("B3-V-S-L", EXP / "runs/corrected/B3-V-S-L", EXP / "runs/corrected/B3-V-S-L/predictions_best.npz", "video_only", "static", "lora", True, ""),
        ("B3-FUS-D-F", b2_by_cell["B2-FUS-0125"]["run_dir"], b2_by_cell["B2-FUS-0125"]["prediction_path"], "fusion", "dynamic", "frozen", False, "B2-FUS-0125"),
        ("B3-FUS-D-L", EXP / "runs/corrected/B3-FUS-D-L", EXP / "runs/corrected/B3-FUS-D-L/predictions_best.npz", "fusion", "dynamic", "lora", True, ""),
        ("B3-FUS-S-F", EXP / "runs/B3-FUS-S-F", EXP / "runs/B3-FUS-S-F/predictions_best.npz", "fusion", "static", "frozen", False, ""),
        ("B3-FUS-S-L", EXP / "runs/corrected/B3-FUS-S-L", EXP / "runs/corrected/B3-FUS-S-L/predictions_best.npz", "fusion", "static", "lora", True, ""),
    ]
    records = []
    rows = []
    for cell, run_dir, prediction, branch, video, adaptation, raw, reused_from in mapping:
        if reused_from:
            # The B2 run has already passed the strict B2 contract.  Revalidate
            # its files here and make the logical B3 reuse explicit.
            source = b2_by_cell[reused_from]
            rec = record_from_prediction(
                cell=cell,
                stage="B3",
                branch=branch,
                video_input=video,
                adaptation=adaptation,
                prediction_path=source["prediction_path"],
                run_dir=source["run_dir"],
                inputs=inputs,
                source_kind="exact_reuse_of_selected_B2_dense_control",
                canonical_status="accepted_reused",
                required_keys={"episode_index", "original_task_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native", "last_observed_action"},
                reused_from=reused_from,
            )
        else:
            rec = record_from_prediction(
                cell=cell,
                stage="B3",
                branch=branch,
                video_input=video,
                adaptation=adaptation,
                prediction_path=prediction,
                run_dir=run_dir,
                inputs=inputs,
                source_kind="corrected_raw_lora_or_pooled_static_runner" if raw or adaptation == "frozen" else "b3_runner",
                canonical_status="accepted",
                required_keys={"episode_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native"},
                raw_lora=raw,
            )
        rec["strategy"] = "dense_0.125s_32_pixel_frames"
        rec["encoder_slots"] = 32
        records.append(rec)
        rows.append({"cell": cell, "run_dir": run_dir, "prediction_path": prediction, "reused_from": reused_from, "source_kind": rec["source_kind"], "status": rec["canonical_status"], "prediction_sha256": sha256(prediction)})
    if {r["cell"] for r in records} != {"B3-V-D-F", "B3-V-D-L", "B3-V-S-F", "B3-V-S-L", "B3-FUS-D-F", "B3-FUS-D-L", "B3-FUS-S-F", "B3-FUS-S-L"}:
        raise HarvestBlocked("B3 logical cell set is not 8/8")
    return records, controller, rows


def add_controls(inputs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    all_arrays = inputs["all"]
    train_mask = inputs["train_mask"]
    train_native = all_arrays["targets"][train_mask].astype(np.float32)
    global_mean = np.broadcast_to(train_native.mean(axis=(0, 1)), (614, STEPS, DIMS)).copy()
    per_time = np.broadcast_to(train_native.mean(axis=0), (614, STEPS, DIMS)).copy()
    zero = np.zeros((614, STEPS, DIMS), dtype=np.float32)
    persistence = np.repeat(inputs["last_action"][:, None, :], STEPS, axis=1)
    return [
        {"cell": "ZERO-ACTION", "stage": "CONTROL", "branch": "zero", "video_input": "none", "adaptation": "fixed", "source_kind": "zero_action_control", "canonical_status": "reference_control", "arrays": {"episode_index": inputs["ids"], "targets_native": inputs["target_native"], "targets_standardized": inputs["target_std"], "predictions_native": zero, "last_observed_action": inputs["last_action"]}},
        {"cell": "TRAIN-GLOBAL-MEAN", "stage": "CONTROL", "branch": "train_global_mean", "video_input": "none", "adaptation": "fixed", "source_kind": "train_only_global_mean_control", "canonical_status": "reference_control", "arrays": {"episode_index": inputs["ids"], "targets_native": inputs["target_native"], "targets_standardized": inputs["target_std"], "predictions_native": global_mean, "last_observed_action": inputs["last_action"]}},
        {"cell": "TRAIN-PER-TIME-MEAN", "stage": "CONTROL", "branch": "train_per_timestep_mean", "video_input": "none", "adaptation": "fixed", "source_kind": "train_only_per_timestep_mean_control", "canonical_status": "reference_control", "arrays": {"episode_index": inputs["ids"], "targets_native": inputs["target_native"], "targets_standardized": inputs["target_std"], "predictions_native": per_time, "last_observed_action": inputs["last_action"]}},
        {"cell": "PERSISTENCE", "stage": "CONTROL", "branch": "persistence", "video_input": "none", "adaptation": "fixed", "source_kind": "last_observed_action_persistence", "canonical_status": "reference_control", "arrays": {"episode_index": inputs["ids"], "targets_native": inputs["target_native"], "targets_standardized": inputs["target_std"], "predictions_native": persistence, "last_observed_action": inputs["last_action"]}},
    ]


def metrics_for(pred_std: np.ndarray, target_std: np.ndarray, pred_native: np.ndarray, target_native: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for space, pred, target in (("standardized", pred_std, target_std), ("native", pred_native, target_native)):
        error = pred.astype(np.float64) - target.astype(np.float64)
        absolute = np.abs(error)
        delta = np.diff(pred.astype(np.float64), axis=1) - np.diff(target.astype(np.float64), axis=1)
        abs_delta = np.abs(delta)
        space_result: dict[str, Any] = {}
        for name, sl in (("arm", ARM), ("hand", HAND), ("full28", FULL)):
            episode = absolute[:, :, sl].mean(axis=(1, 2))
            final = absolute[:, -1, sl].mean(axis=1)
            trend_episode = abs_delta[:, :, sl].mean(axis=(1, 2))
            space_result[f"{name}_mae"] = float(episode.mean())
            space_result[f"{name}_final_step_mae"] = float(final.mean())
            space_result[f"{name}_first_diff_mae"] = float(trend_episode.mean())
            space_result[f"episode_{name}_mae"] = episode
            space_result[f"episode_{name}_final_step_mae"] = final
            space_result[f"episode_{name}_first_diff_mae"] = trend_episode
            space_result[f"per_timestep_{name}_mae"] = absolute[:, :, sl].mean(axis=(0, 2))
            space_result[f"per_joint_time_{name}_mae"] = absolute[:, :, sl].mean(axis=0)
        rmse_episode = np.sqrt(np.square(error).mean(axis=(1, 2)))
        space_result["rmse"] = float(rmse_episode.mean())
        space_result["episode_rmse"] = rmse_episode
        space_result["per_timestep_rmse"] = np.sqrt(np.square(error).mean(axis=(0, 2)))
        space_result["per_joint_time_mae"] = absolute.mean(axis=0)
        space_result["mae"] = float(absolute.mean())
        space_result["final_step_mae"] = float(absolute[:, -1, :].mean())
        space_result["first_diff_mae"] = float(abs_delta.mean())
        result[space] = space_result
    return result


def attach_metrics(records: list[dict[str, Any]], inputs: dict[str, np.ndarray]) -> None:
    # Controls already contain native predictions; standardize them with the
    # same locked train-only target statistics below.
    norm = read_json(TARGET_NORM)["1.0s"]
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["safe_std"], dtype=np.float64)
    for record in records:
        arrays = record["arrays"]
        if "predictions_native" in arrays:
            pred_native = arrays["predictions_native"].astype(np.float32)
            pred_std = ((pred_native.astype(np.float64) - mean) / std).astype(np.float32)
        else:
            pred_std = arrays["predictions_standardized"].astype(np.float32)
            pred_native = (pred_std.astype(np.float64) * std + mean).astype(np.float32)
        target_std = arrays["targets_standardized"].astype(np.float32)
        target_native = arrays["targets_native"].astype(np.float32)
        arrays["predictions_standardized"] = pred_std
        arrays["predictions_native"] = pred_native
        record["metrics"] = metrics_for(pred_std, target_std, pred_native, target_native)
        record["selected_epoch"] = record.get("meta", {}).get("run_metrics", {}).get("best_epoch", "")
        record["final_test_accessed"] = 0


def metric_row(record: dict[str, Any]) -> dict[str, Any]:
    m = record["metrics"]
    row = {
        "stage": record["stage"],
        "cell": record["cell"],
        "branch": record.get("branch", ""),
        "video_input": record.get("video_input", ""),
        "adaptation": record.get("adaptation", ""),
        "source_kind": record.get("source_kind", ""),
        "canonical_status": record.get("canonical_status", ""),
        "reused_from": record.get("reused_from", ""),
        "strategy": record.get("strategy", ""),
        "interval_seconds": record.get("interval_seconds", ""),
        "unique_observation_frames": record.get("unique_observation_frames", ""),
        "encoder_slots": record.get("encoder_slots", ""),
        "selected_epoch": record.get("selected_epoch", ""),
        "prediction_path": record.get("prediction_path", record.get("source_kind", "")),
        "final_test_accessed": 0,
    }
    for space in ("standardized", "native"):
        sm = m[space]
        prefix = "std" if space == "standardized" else "native"
        for name in ("arm", "hand", "full28"):
            row[f"{prefix}_{name}_mae"] = sm[f"{name}_mae"]
            row[f"{prefix}_{name}_final_step_mae"] = sm[f"{name}_final_step_mae"]
            row[f"{prefix}_{name}_first_diff_mae"] = sm[f"{name}_first_diff_mae"]
        row[f"{prefix}_rmse"] = sm["rmse"]
        row[f"{prefix}_mae"] = sm["mae"]
        row[f"{prefix}_final_step_mae"] = sm["final_step_mae"]
        row[f"{prefix}_first_diff_mae"] = sm["first_diff_mae"]
    return row


def per_episode_rows(records: list[dict[str, Any]], inputs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        m = record["metrics"]
        for i, episode_id in enumerate(inputs["ids"]):
            rows.append({
                "stage": record["stage"],
                "cell": record["cell"],
                "episode_index": int(episode_id),
                "original_task_index": int(inputs["tasks"][i]),
                "std_arm_mae": float(m["standardized"]["episode_arm_mae"][i]),
                "std_hand_mae": float(m["standardized"]["episode_hand_mae"][i]),
                "std_full28_mae": float(m["standardized"]["episode_full28_mae"][i]),
                "std_arm_final_step_mae": float(m["standardized"]["episode_arm_final_step_mae"][i]),
                "std_arm_first_diff_mae": float(m["standardized"]["episode_arm_first_diff_mae"][i]),
                "std_rmse": float(m["standardized"]["episode_rmse"][i]),
                "native_arm_mae": float(m["native"]["episode_arm_mae"][i]),
                "native_hand_mae": float(m["native"]["episode_hand_mae"][i]),
                "native_full28_mae": float(m["native"]["episode_full28_mae"][i]),
                "native_arm_final_step_mae": float(m["native"]["episode_arm_final_step_mae"][i]),
                "native_arm_first_diff_mae": float(m["native"]["episode_arm_first_diff_mae"][i]),
                "native_rmse": float(m["native"]["episode_rmse"][i]),
            })
    return rows


def per_timestep_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for space in ("standardized", "native"):
            sm = record["metrics"][space]
            prefix = "std" if space == "standardized" else "native"
            for step in range(STEPS):
                rows.append({"stage": record["stage"], "cell": record["cell"], "space": space, "time_step": step, "arm_mae": float(sm["per_timestep_arm_mae"][step]) if step < STEPS else "", "hand_mae": float(sm["per_timestep_hand_mae"][step]) if step < STEPS else "", "full28_mae": float(sm["per_timestep_full28_mae"][step]) if step < STEPS else "", "rmse": float(sm["per_timestep_rmse"][step])})
    return rows


def per_joint_time_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for space in ("standardized", "native"):
            matrix = record["metrics"][space]["per_joint_time_mae"]
            for step in range(STEPS):
                for joint in range(DIMS):
                    rows.append({"stage": record["stage"], "cell": record["cell"], "space": space, "time_step": step, "joint": joint, "mae": float(matrix[step, joint])})
    return rows


def per_task_rows(records: list[dict[str, Any]], inputs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    tasks = inputs["tasks"]
    for record in records:
        errors = record["metrics"]["standardized"]["episode_arm_mae"]
        native = record["metrics"]["native"]["episode_arm_mae"]
        for task in sorted(np.unique(tasks).tolist()):
            mask = tasks == task
            rows.append({"stage": record["stage"], "cell": record["cell"], "original_task_index": int(task), "episode_count": int(mask.sum()), "std_arm_mae": float(errors[mask].mean()), "native_arm_mae": float(native[mask].mean())})
    return rows


def bootstrap_samples(diff: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(diff, dtype=np.float64)[indices].mean(axis=1)


def signflip_p(diff: np.ndarray, signs: np.ndarray) -> float:
    observed = abs(float(np.mean(diff)))
    null = np.abs((signs * diff[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def holm_adjust(rows: list[dict[str, Any]], p_key: str = "raw_p_value") -> None:
    if not rows:
        return
    ordered = sorted(range(len(rows)), key=lambda i: float(rows[i][p_key]))
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * float(rows[index][p_key])))
        rows[index]["holm_adjusted_p"] = running


def pairwise_row(candidate: dict[str, Any], reference: dict[str, Any], family: str, indices: np.ndarray, signs: np.ndarray, comparison: str) -> dict[str, Any]:
    cand = candidate["metrics"]["standardized"]["episode_arm_mae"]
    ref = reference["metrics"]["standardized"]["episode_arm_mae"]
    diff = ref - cand  # positive means candidate has lower error
    samples = bootstrap_samples(diff, indices)
    low, high = np.quantile(samples, [0.025, 0.975])
    ref_means = ref[indices].mean(axis=1)
    cand_means = cand[indices].mean(axis=1)
    relative_samples = 1.0 - cand_means / np.maximum(ref_means, 1e-12)
    relative_low, relative_high = np.quantile(relative_samples, [0.025, 0.975])
    return {
        "comparison": comparison,
        "family": family,
        "candidate": candidate["cell"],
        "reference": reference["cell"],
        "metric": "standardized_arm14_trajectory_mae",
        "delta_definition": "reference MAE minus candidate MAE; positive means candidate is lower-error",
        "candidate_point": float(cand.mean()),
        "reference_point": float(ref.mean()),
        "improvement_point": float(diff.mean()),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "relative_improvement_point": float(1.0 - cand.mean() / max(ref.mean(), 1e-12)),
        "relative_ci95_lower": float(relative_low),
        "relative_ci95_upper": float(relative_high),
        "candidate_hand_mae": float(candidate["metrics"]["standardized"]["hand_mae"]),
        "reference_hand_mae": float(reference["metrics"]["standardized"]["hand_mae"]),
        "candidate_full28_mae": float(candidate["metrics"]["standardized"]["full28_mae"]),
        "reference_full28_mae": float(reference["metrics"]["standardized"]["full28_mae"]),
        "candidate_native_arm_mae": float(candidate["metrics"]["native"]["arm_mae"]),
        "reference_native_arm_mae": float(reference["metrics"]["native"]["arm_mae"]),
        "raw_p_value": signflip_p(diff, signs),
        "bootstrap_n": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_n": PERM_REPS,
        "permutation_seed": PERM_SEED,
        "episode_candidate_lower_error": int(np.sum(cand < ref)),
        "episode_reference_lower_error": int(np.sum(ref < cand)),
        "final_test_access": 0,
    }


def make_comparisons(records: list[dict[str, Any]], indices: np.ndarray, signs: np.ndarray) -> list[dict[str, Any]]:
    by = {r["cell"]: r for r in records}
    specs = [
        ("B1_learned_vs_persistence", "B1", c, "PERSISTENCE") for c in ["B-STATE", "B-D-P-F", "B-S-P-F", "B-D-A-F", "B-S-A-F", "B-D-FUS-P-F", "B-S-FUS-P-F"]
    ]
    specs += [
        ("B1_dynamic_static", "B1", "B-D-P-F", "B-S-P-F"),
        ("B1_dynamic_static", "B1", "B-D-A-F", "B-S-A-F"),
        ("B1_dynamic_static", "B1", "B-D-FUS-P-F", "B-S-FUS-P-F"),
        ("B1_pooled_attentive", "B1", "B-D-P-F", "B-D-A-F"),
        ("B1_pooled_attentive", "B1", "B-S-P-F", "B-S-A-F"),
        ("B1_fusion_vs_state", "B1", "B-D-FUS-P-F", "B-STATE"),
        ("B1_fusion_vs_state", "B1", "B-S-FUS-P-F", "B-STATE"),
        ("B2_sparse_vs_dense", "B2", "B2-V-0250", "B2-V-0125"),
        ("B2_sparse_vs_dense", "B2", "B2-V-0500", "B2-V-0125"),
        ("B2_sparse_vs_dense", "B2", "B2-FUS-0250", "B2-FUS-0125"),
        ("B2_sparse_vs_dense", "B2", "B2-FUS-0500", "B2-FUS-0125"),
        ("B2_fusion_vs_video", "B2", "B2-FUS-0125", "B2-V-0125"),
        ("B2_fusion_vs_video", "B2", "B2-FUS-0250", "B2-V-0250"),
        ("B2_fusion_vs_video", "B2", "B2-FUS-0500", "B2-V-0500"),
        ("B2_fusion_vs_state", "B2", "B2-FUS-0125", "B-STATE"),
        ("B2_fusion_vs_persistence", "B2", "B2-FUS-0125", "PERSISTENCE"),
        ("B3_lora_matched", "B3", "B3-V-D-L", "B3-V-D-F"),
        ("B3_lora_matched", "B3", "B3-V-S-L", "B3-V-S-F"),
        ("B3_lora_matched", "B3", "B3-FUS-D-L", "B3-FUS-D-F"),
        ("B3_lora_matched", "B3", "B3-FUS-S-L", "B3-FUS-S-F"),
        ("B3_dynamic_static", "B3", "B3-V-D-F", "B3-V-S-F"),
        ("B3_dynamic_static", "B3", "B3-V-D-L", "B3-V-S-L"),
        ("B3_dynamic_static", "B3", "B3-FUS-D-F", "B3-FUS-S-F"),
        ("B3_dynamic_static", "B3", "B3-FUS-D-L", "B3-FUS-S-L"),
        ("B3_fusion_vs_state", "B3", "B3-FUS-D-F", "B-STATE"),
        ("B3_fusion_vs_state", "B3", "B3-FUS-D-L", "B-STATE"),
        ("B3_vs_persistence", "B3", "B3-V-D-F", "PERSISTENCE"),
        ("B3_vs_persistence", "B3", "B3-V-D-L", "PERSISTENCE"),
        ("B3_vs_persistence", "B3", "B3-FUS-D-F", "PERSISTENCE"),
        ("B3_vs_persistence", "B3", "B3-FUS-D-L", "PERSISTENCE"),
    ]
    rows = []
    for family, stage, cand, ref in specs:
        if cand not in by or ref not in by:
            raise HarvestBlocked(f"comparison source missing: {cand} or {ref}")
        rows.append(pairwise_row(by[cand], by[ref], family, indices, signs, f"{cand}_vs_{ref}"))
        rows[-1]["stage"] = stage
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        holm_adjust(subset)
    holm_adjust(rows)
    for row in rows:
        row["holm_global_p"] = row.pop("holm_adjusted_p")
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        holm_adjust(subset)
        for row in subset:
            row["holm_family_p"] = row.pop("holm_adjusted_p")
    return rows


def factorial_effects(records: list[dict[str, Any]], indices: np.ndarray) -> list[dict[str, Any]]:
    by = {r["cell"]: r for r in records}
    rows = []
    for label, d_f, d_l, s_f, s_l in [
        ("video_only", "B3-V-D-F", "B3-V-D-L", "B3-V-S-F", "B3-V-S-L"),
        ("fusion", "B3-FUS-D-F", "B3-FUS-D-L", "B3-FUS-S-F", "B3-FUS-S-L"),
    ]:
        d_fa = by[d_f]["metrics"]["standardized"]["episode_arm_mae"]
        d_la = by[d_l]["metrics"]["standardized"]["episode_arm_mae"]
        s_fa = by[s_f]["metrics"]["standardized"]["episode_arm_mae"]
        s_la = by[s_l]["metrics"]["standardized"]["episode_arm_mae"]
        gd = d_fa - d_la
        gs = s_fa - s_la
        interaction = gd - gs
        gd_s = gd[indices].mean(axis=1)
        gs_s = gs[indices].mean(axis=1)
        int_s = interaction[indices].mean(axis=1)
        gd_low, gd_high = np.quantile(gd_s, [0.025, 0.975])
        gs_low, gs_high = np.quantile(gs_s, [0.025, 0.975])
        int_low, int_high = np.quantile(int_s, [0.025, 0.975])
        rows.append({
            "branch": label,
            "dynamic_frozen": d_f,
            "dynamic_lora": d_l,
            "static_frozen": s_f,
            "static_lora": s_l,
            "G_D_point": float(gd.mean()),
            "G_D_ci95_lower": float(gd_low),
            "G_D_ci95_upper": float(gd_high),
            "G_D_relative_point": float(1 - d_la.mean() / d_fa.mean()),
            "G_S_point": float(gs.mean()),
            "G_S_ci95_lower": float(gs_low),
            "G_S_ci95_upper": float(gs_high),
            "G_S_relative_point": float(1 - s_la.mean() / s_fa.mean()),
            "interaction_point": float(interaction.mean()),
            "interaction_ci95_lower": float(int_low),
            "interaction_ci95_upper": float(int_high),
            "T_F_point": float(s_fa.mean() - d_fa.mean()),
            "T_L_point": float(s_la.mean() - d_la.mean()),
            "final_test_access": 0,
        })
    return rows


def b3_decision(records: list[dict[str, Any]], effects: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["cell"]: r for r in records}
    comp = {r["comparison"]: r for r in comparisons}
    gates = []
    for branch, pfx, ext in [("video_only", "V", "PERSISTENCE"), ("fusion", "FUS", "STATE")]:
        d_f, d_l = by[f"B3-{pfx}-D-F"], by[f"B3-{pfx}-D-L"]
        s_f, s_l = by[f"B3-{pfx}-S-F"], by[f"B3-{pfx}-S-L"]
        effect = next(row for row in effects if row["branch"] == branch)
        matched = comp[f"B3-{pfx}-D-L_vs_B3-{pfx}-D-F"]
        external_reference = "B-STATE" if ext == "STATE" else ext
        external = comp[f"B3-{pfx}-D-L_vs_{external_reference}"]
        lora_hand_degradation = d_l["metrics"]["standardized"]["hand_mae"] / d_f["metrics"]["standardized"]["hand_mae"] - 1
        lora_full_degradation = d_l["metrics"]["standardized"]["full28_mae"] / d_f["metrics"]["standardized"]["full28_mae"] - 1
        checks = {
            "dynamic_relative_arm_gain_at_least_2pct": effect["G_D_relative_point"] >= 0.02,
            "dynamic_arm_bootstrap_ci_lower_gt_zero": matched["ci95_lower"] > 0,
            "dynamic_arm_holm_family_p_lt_0_05": matched["holm_family_p"] < 0.05,
            "hand_degradation_at_most_1pct": lora_hand_degradation <= 0.01,
            "full28_degradation_at_most_1pct": lora_full_degradation <= 0.01,
            "dynamic_gain_at_least_static_gain": effect["interaction_point"] >= 0,
            "external_gate_ci_lower_gt_zero": external["ci95_lower"] > 0,
            "external_gate_holm_family_p_lt_0_05": external["holm_family_p"] < 0.05,
        }
        gates.append({
            "branch": branch,
            "dynamic_frozen": d_f["cell"],
            "dynamic_lora": d_l["cell"],
            "static_frozen": s_f["cell"],
            "static_lora": s_l["cell"],
            "external_reference": ext,
            "dynamic_relative_arm_gain": effect["G_D_relative_point"],
            "hand_degradation": lora_hand_degradation,
            "full28_degradation": lora_full_degradation,
            "checks": checks,
            "candidate_gate": bool(all(checks.values())),
        })
    candidates = [g for g in gates if g["candidate_gate"]]
    return {
        "status": "pass",
        "formal_matrix": "8/8",
        "strategy": "dense_0.125s_32_pixel_frames",
        "candidate_count": len(candidates),
        "selected_candidate": ("fusion" if any(g["branch"] == "fusion" for g in candidates) else "video_only" if candidates else None),
        "recommendation_for_new_multi_seed": "ready_only_if_human_authorizes_and_candidate_gate_passes" if candidates else "not_ready_b3_gate_not_passed",
        "stage_is_single_seed_exploratory": True,
        "final_test_access": 0,
        "gates": gates,
    }


def write_lineage(records: list[dict[str, Any]], inputs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    paths = [INPUT_NPZ, TARGET_NORM, B1_ACCEPTANCE, B1_INPUTS, B2_ACCEPTANCE, B2_SELECTION, B2_FIDELITY, B23_PROTOCOL, B23_MATRIX, B3_CONTROLLER]
    for path in paths:
        if path.is_file():
            rows.append({"role": "protocol_or_canonical_input", "cell": "ALL", "path": path, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    for record in records:
        path = Path(record.get("prediction_path", "")) if record.get("prediction_path") else None
        if path and path.is_file():
            rows.append({"role": "prediction", "cell": record["cell"], "path": path, "sha256": sha256(path), "size_bytes": path.stat().st_size})
        run_dir = record.get("run_dir")
        if run_dir:
            for name in ("RUN_STATUS.json", "RUN_CONFIG_LOCKED.json", "RUN_METRICS.json", "RUN_COMPLETE.json"):
                file = Path(run_dir) / name
                if file.is_file():
                    rows.append({"role": name, "cell": record["cell"], "path": file, "sha256": sha256(file), "size_bytes": file.stat().st_size})
            checkpoint = Path(run_dir) / "checkpoint_best.pt"
            if checkpoint.is_file():
                rows.append({"role": "checkpoint_best", "cell": record["cell"], "path": checkpoint, "sha256": sha256(checkpoint), "size_bytes": checkpoint.stat().st_size})
    return rows


def write_inventory(b1: list[dict[str, Any]], b2: list[dict[str, Any]], b3: list[dict[str, Any]], proxy: list[dict[str, Any]], controller: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in b1 + b2 + b3:
        rows.append({"stage": record["stage"], "cell": record["cell"], "research_question": "Core B trajectory prediction at the locked 1.0s horizon", "branch": record.get("branch", ""), "video_input": record.get("video_input", ""), "adaptation": record.get("adaptation", ""), "protocol_id": record.get("config", {}).get("protocol_id", ""), "seed": record.get("meta", {}).get("run_status", {}).get("seed", 20260903), "population": "train=2741;dev=614", "target_shape": "614x30x28", "source_kind": record.get("source_kind", ""), "status": record.get("canonical_status", ""), "reused_from": record.get("reused_from", ""), "prediction_path": record.get("prediction_path", ""), "final_test_access": 0})
    for record in proxy:
        rows.append({"stage": "B2_PROXY", "cell": record["cell"], "research_question": "auxiliary token aggregation proxy only", "branch": record.get("branch", ""), "video_input": "dynamic", "adaptation": "frozen", "protocol_id": "proxy_contextual_tokens", "seed": 20260903, "population": "train=2741;dev=614", "target_shape": "614x30x28", "source_kind": "auxiliary_token_aggregation_proxy", "status": "proxy_auxiliary_excluded_from_selection", "reused_from": "", "prediction_path": record.get("prediction_path", ""), "final_test_access": 0})
    rows.append({"stage": "B3_CONTROLLER", "cell": "ALL", "research_question": "controller provenance", "branch": "", "video_input": "", "adaptation": "", "protocol_id": "", "seed": 20260903, "population": "", "target_shape": "", "source_kind": "controller", "status": controller.get("status"), "reused_from": "", "prediction_path": B3_CONTROLLER, "final_test_access": controller.get("final_test_access", 0)})
    return rows


def representative_episodes(records: list[dict[str, Any]], inputs: dict[str, np.ndarray]) -> tuple[list[int], list[dict[str, Any]]]:
    persistence = next(r for r in records if r["cell"] == "PERSISTENCE")
    errors = persistence["metrics"]["standardized"]["episode_arm_mae"]
    chosen = []
    for quantile in (0.25, 0.50, 0.75):
        target = float(np.quantile(errors, quantile))
        index = int(np.argmin(np.abs(errors - target)))
        if index not in chosen:
            chosen.append(index)
    rows = [{"rank": i + 1, "quantile": q, "episode_index": int(inputs["ids"][idx]), "persistence_std_arm_mae": float(errors[idx])} for i, (q, idx) in enumerate(zip((0.25, 0.50, 0.75), chosen))]
    return [int(inputs["ids"][idx]) for idx in chosen], rows


def trajectory_rows(records: list[dict[str, Any]], inputs: dict[str, np.ndarray], episode_ids: list[int], model: str = "B3-FUS-D-L") -> list[dict[str, Any]]:
    by = {r["cell"]: r for r in records}
    selected = by[model]
    id_to_index = {int(value): i for i, value in enumerate(inputs["ids"].tolist())}
    rows = []
    persistence = by["PERSISTENCE"]
    for episode_id in episode_ids:
        i = id_to_index[episode_id]
        for step in range(STEPS):
            for joint in range(14):
                joint_index = 14 + joint
                rows.append({"episode_index": episode_id, "time_step": step, "arm_joint": joint, "target": float(inputs["target_native"][i, step, joint_index]), "persistence": float(persistence["arrays"]["predictions_native"][i, step, joint_index]), "model": float(selected["arrays"]["predictions_native"][i, step, joint_index]), "model_cell": model})
    return rows


def write_conclusions(records: list[dict[str, Any]], b2_selection: dict[str, Any], b3_decision: dict[str, Any], factorial: list[dict[str, Any]], comparisons: list[dict[str, Any]], proxy_rows: list[dict[str, Any]]) -> None:
    by = {r["cell"]: r for r in records}
    learned_b1 = [r for r in records if r["stage"] == "B1"]
    best_b1 = min(learned_b1, key=lambda r: r["metrics"]["standardized"]["arm_mae"])
    b2_formal = [r for r in records if r["stage"] == "B2"]
    best_b2 = min(b2_formal, key=lambda r: r["metrics"]["standardized"]["arm_mae"])
    b3_formal = [r for r in records if r["stage"] == "B3"]
    best_b3 = min(b3_formal, key=lambda r: r["metrics"]["standardized"]["arm_mae"])
    fus_state = next(r for r in comparisons if r["comparison"] == "B3-FUS-D-L_vs_B-STATE")
    fus_persist = next(r for r in comparisons if r["comparison"] == "B3-FUS-D-L_vs_PERSISTENCE")
    lines = [
        "# Core B 完整收菜与结论（中文）",
        "",
        f"分析版本：`CORE_B_UNIFIED_ANALYSIS_20260909T163929Z`；生成时间：{utc_now()}。正式分析范围为 G1 train=2741、development=614、target=(30,28)，final-test access=0。",
        "",
        "## 验收状态",
        "",
        "- B1 canonical inference：7/7 accepted，未重训；",
        "- true-pixel B2：6/6 accepted；旧 contextual-token sparse B2 单独标为 proxy，不进入正式选择；",
        "- corrected B3：8/8 accepted，统一使用 dense 0.125s/32 pixel observations、pooled head、seed 20260903 和 corrected train-only feature normalization；",
        "- Core B 的 native command-value 单位仍未被物理语义/URDF 审计确认，因此 native 数值只作为 source-native command-value error 报告。",
        "",
        "## 主要结果",
        "",
        f"- B1 最低 standardized arm14 MAE：{best_b1['cell']} = {best_b1['metrics']['standardized']['arm_mae']:.6f}；persistence = {by['PERSISTENCE']['metrics']['standardized']['arm_mae']:.6f}。",
        f"- true-pixel B2 最低 standardized arm14 MAE：{best_b2['cell']} = {best_b2['metrics']['standardized']['arm_mae']:.6f}；正式选择仍为 dense 0.125s/32，稀疏组没有形成合格效率/精度候选。",
        f"- B3 八格中最低 standardized arm14 MAE：{best_b3['cell']} = {best_b3['metrics']['standardized']['arm_mae']:.6f}。",
        f"- B3-FUS-D-L 相对 STATE 的 improvement（state MAE−fusion LoRA MAE）为 {fus_state['improvement_point']:.6f}，95% CI [{fus_state['ci95_lower']:.6f}, {fus_state['ci95_upper']:.6f}]；相对 persistence 为 {fus_persist['improvement_point']:.6f}，95% CI [{fus_persist['ci95_lower']:.6f}, {fus_persist['ci95_upper']:.6f}]。",
        "",
        "## 解释边界",
        "",
        "B1/B2/B3 的 learned trajectory error 必须与 persistence、STATE、video-only 和 fusion 同时阅读。若 learned 误差仍高于 persistence，只能说明当前 head/target parameterization 没有超过强 naive control；不能据此否定视觉 encoder 的表示能力。",
        "",
        "B2 的 sparse 结论只来自真正像素级采样；旧 proxy 只说明 dense contextual representation 的下游抽稀/聚合行为，不能说明 encoder 只观察了 16/8 帧。",
        "",
        "B3 是 single-seed exploratory matrix；episode bootstrap 不是 multi-seed confirmation。B3 candidate gate、若未通过，不启动 fresh-seed confirmation。",
        "",
        "## 失败与工程保留",
        "",
        "早期 manifest、module-count、缺 feature normalization 和纯 writer 错误均保留在 failed-attempt evidence，不计入 formal matrix。LoRA canonical inference 使用 unmerged base+adapter；没有把工程成功、GPU 利用率或单个 child 完成当作科学完成证明。",
        "",
        "详细的 B1/B2/B3 指标、per-time/per-joint/per-task、paired bootstrap、Holm、轨迹和图表底层数据均位于本版本目录。",
    ]
    (OUT / "conclusions/CORE_B_EXECUTIVE_SUMMARY.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "conclusions/CORE_B_COMPLETE_EXPERIMENT_REVIEW.zh-CN.md").write_text(
        "# Core B complete experiment review\n\n"
        "B1 tests whether frozen state/video/attentive/fusion predictors can forecast the locked 1.0s, 30x28 target. B2 tests true pixel-level observation sparsity while holding the latest observation and target start fixed. B3 tests frozen versus raw-encoder LoRA under the selected B2 strategy.\n\n"
        "B1/B2/B3 are reported separately; proxy B2 is auxiliary only. See `CORE_B_UNIFIED_RESULTS.csv`, `CORE_B_PAIRWISE_COMPARISONS.csv`, `CORE_B_FACTORIAL_EFFECTS.csv`, and the stage-specific findings files.\n",
        encoding="utf-8",
    )
    stage_docs = {
        "CORE_B_B1_FINDINGS.zh-CN.md": "B1 7/7 canonical inference accepted without retraining. It supplies state, video, attentive and fusion controls and is evaluated against train-only means and last-observed-action persistence.",
        "CORE_B_B2_FINDINGS.zh-CN.md": "True-pixel B2 is 6/6 accepted. Dense 0.125s/32 is selected by the locked fallback rule; sparse conditions are retained as valid negative/inconclusive results. The old token-aggregation proxy is excluded.",
        "CORE_B_B3_FINDINGS.zh-CN.md": f"Corrected B3 is 8/8 accepted. The current formal candidate decision is `{b3_decision['recommendation_for_new_multi_seed']}`. It remains single-seed exploratory.",
        "CORE_B_WHAT_WE_LEARNED.zh-CN.md": "The central evidence is the relationship between video, state/history, fusion and persistence. Feature normalization and unmerged LoRA provenance are required for B3 validity; changing the observation budget did not produce a sparse formal winner in the current screen.",
        "CORE_B_LIMITATIONS_AND_REMAINING.zh-CN.md": "Action units and physical semantics remain unresolved; there is one training seed; final-test is locked; B2 sparse and B3 candidate claims are bounded by the current development population and registered gates.",
        "CORE_B_PROFESSOR_UPDATE.zh-CN.md": "Core B formal harvest is complete for B1 7/7, true-pixel B2 6/6 and corrected B3 8/8. No new direction, multi-seed confirmation or final-test was started. The next decision depends on the formal B3 gate and the physical command-semantic audit.",
    }
    for name, text in stage_docs.items():
        (OUT / "conclusions" / name).write_text(f"# {name}\n\n{text}\n\nAll claims are bounded to the versioned tables and final-test access=0.\n", encoding="utf-8")


def main() -> int:
    for directory in (OUT, OUT / "data", OUT / "code", OUT / "figures", OUT / "figure_data", OUT / "conclusions", OUT / "evidence", OUT / "qa", OUT / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = load_canonical_inputs()
    b1_records, b1_acceptance_rows = load_b1_records(inputs)
    b2_records, b2_acceptance_rows, proxy_rows = load_b2_records(inputs)
    b3_records, controller, b3_rows = load_b3_records(inputs, b2_records)
    controls = add_controls(inputs)
    b2_formal_records = [record for record in b2_records if record["stage"] == "B2"]
    proxy_records = [record for record in b2_records if record["stage"] == "B2_PROXY"]
    formal = b1_records + b2_formal_records + b3_records
    all_records = formal + controls
    attach_metrics(all_records + proxy_records, inputs)
    # Proxy rows are useful in the dedicated proxy-vs-true display but never
    # enter formal comparisons, candidate gates, or selected strategy logic.
    formal_results = [metric_row(r) for r in formal + controls]
    write_csv(OUT / "CORE_B_UNIFIED_RESULTS.csv", formal_results)
    write_json(OUT / "CORE_B_UNIFIED_RESULTS.json", {"status": "complete", "formal_rows": formal_results, "proxy_excluded": True})
    write_csv(OUT / "B1_RESULTS_INDEPENDENT.csv", [metric_row(r) for r in b1_records + controls])
    write_csv(OUT / "B2_RESULTS_INDEPENDENT.csv", [metric_row(r) for r in b2_records if r["stage"] == "B2"] + [metric_row(r) for r in controls if r["cell"] in {"PERSISTENCE", "B-STATE"}])
    write_csv(OUT / "B3_RESULTS_INDEPENDENT.csv", [metric_row(r) for r in b3_records] + [metric_row(r) for r in controls if r["cell"] in {"PERSISTENCE", "B-STATE"}])
    write_csv(OUT / "CORE_B_PER_EPISODE_METRICS.csv", per_episode_rows(formal + controls, inputs))
    write_csv(OUT / "CORE_B_PER_TIMESTEP_METRICS.csv", per_timestep_rows(formal + controls))
    write_csv(OUT / "CORE_B_PER_JOINT_TIME_METRICS.csv", per_joint_time_rows(formal + controls))
    write_csv(OUT / "CORE_B_PER_TASK_METRICS.csv", per_task_rows(formal + controls, inputs))
    write_json(OUT / "B1_REACCEPTANCE.json", {"status": "pass", "logical_coverage": "7/7", "accepted_count": 7, "core_b_b1_retrained": False, "final_test_access": 0, "rows": b1_acceptance_rows})
    write_csv(OUT / "B1_REACCEPTANCE.csv", b1_acceptance_rows)
    write_json(OUT / "B2_REACCEPTANCE.json", {"status": "pass", "logical_coverage": "6/6", "pixel_observation_protocol_verified": True, "selected_strategy": "dense_0.125s_32_pixel_frames", "final_test_access": 0, "rows": b2_acceptance_rows})
    write_csv(OUT / "B2_REACCEPTANCE.csv", b2_acceptance_rows)
    write_json(OUT / "B3_REACCEPTANCE.json", {"status": "pass", "logical_coverage": "8/8", "controller_status": controller, "final_test_access": 0, "rows": b3_rows})
    write_csv(OUT / "B3_REACCEPTANCE.csv", b3_rows)
    proxy_metric_rows = [metric_row(r) for r in proxy_records]
    write_csv(OUT / "B2_PROXY_RESULTS.csv", proxy_metric_rows)
    write_json(OUT / "B2_PROXY_RECLASSIFICATION.json", {"status": "auxiliary_token_aggregation_proxy", "formal_selection_included": False, "old_sparse_cells": [r["cell"] for r in proxy_records], "reason": "dense contextual tokens were selected/aggregated after the full spatiotemporal encoder", "final_test_access": 0})
    write_csv(OUT / "B2_PROXY_INVENTORY.csv", proxy_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 614, size=(BOOTSTRAP_REPS, 614), dtype=np.uint16)
    np.save(OUT / "data/paired_episode_bootstrap_indices.npy", indices, allow_pickle=False)
    signs_rng = np.random.default_rng(PERM_SEED)
    signs = np.where(signs_rng.integers(0, 2, size=(PERM_REPS, 614), dtype=np.int8) == 0, -1.0, 1.0)
    np.save(OUT / "data/paired_signflip_signs.npy", signs.astype(np.int8), allow_pickle=False)
    write_json(OUT / "BOOTSTRAP_RULE.json", {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "episode_count": 614, "unit": "episode", "same_resamples_for_all_formal_comparisons": True, "ci": "percentile 2.5%/97.5%", "permutation_replicates": PERM_REPS, "permutation_seed": PERM_SEED, "final_test_access": 0})
    comparisons = make_comparisons(formal + controls, indices, signs)
    write_csv(OUT / "CORE_B_PAIRWISE_COMPARISONS.csv", comparisons)
    write_json(OUT / "CORE_B_PAIRWISE_COMPARISONS.json", {"status": "complete", "rows": comparisons, "final_test_access": 0})
    effects = factorial_effects(formal + controls, indices)
    write_csv(OUT / "CORE_B_FACTORIAL_EFFECTS.csv", effects)
    write_json(OUT / "CORE_B_FACTORIAL_EFFECTS.json", {"status": "complete", "rows": effects, "final_test_access": 0})
    decision = b3_decision(b3_records + controls, effects, comparisons)
    write_json(OUT / "B3_CANDIDATE_DECISION.json", decision)
    (OUT / "B3_CANDIDATE_DECISION.md").write_text("# B3 candidate decision\n\n" + f"- recommendation: `{decision['recommendation_for_new_multi_seed']}`\n- candidate_count: `{decision['candidate_count']}`\n- formal matrix: `8/8`\n- final-test access: `0`\n", encoding="utf-8")
    write_json(OUT / "B2_SELECTION_INDEPENDENT.json", {"status": "pass", "formal_coverage": "6/6", "selected_strategy": "dense_0.125s_32_pixel_frames", "source_locked_selection": read_json(B2_SELECTION), "proxy_excluded": True, "final_test_access": 0})
    write_json(OUT / "B3_PIPELINE_AND_NORMALIZATION_AUDIT.json", {"status": "pass", "formal_coverage": "8/8", "strategy": "dense_0.125s_32_pixel_frames", "raw_lora_feature_normalization": "corrected train rows only", "canonical_inference": "unmerged_base_plus_adapter", "final_test_access": 0})
    (OUT / "B3_PIPELINE_AND_NORMALIZATION_AUDIT.md").write_text("# B3 pipeline and normalization audit\n\nAll corrected raw-LoRA cells passed the train-only dynamic/static feature-normalization, raw encoder forward, base-freeze, adapter-gradient and unmerged-inference checks.\n", encoding="utf-8")
    write_json(OUT / "B3_STRATEGY_PROVENANCE.json", {"status": "pass", "strategy": "dense_0.125s_32_pixel_frames", "source": str(B2_SELECTION), "final_test_access": 0})
    lineage = write_lineage(formal + controls, inputs)
    write_csv(OUT / "SOURCE_MANIFEST.csv", lineage)
    write_json(OUT / "SOURCE_MANIFEST.json", lineage)
    inventory = write_inventory(b1_records, [r for r in b2_records if r["stage"] == "B2"], b3_records, proxy_records, controller)
    write_csv(OUT / "CORE_B_LOGICAL_CELL_INVENTORY.csv", inventory)
    write_json(OUT / "CORE_B_LOGICAL_CELL_INVENTORY.json", {"status": "complete", "rows": inventory, "final_test_access": 0})
    write_json(OUT / "LIVE_STATE_RECONCILIATION.json", {"observed_utc": utc_now(), "controller_path": B3_CONTROLLER, "controller_status": controller, "b1_acceptance": "7/7", "true_pixel_b2_acceptance": "6/6", "corrected_b3_acceptance": "8/8", "gpu_query": "not used by CPU-only harvest", "new_gpu_jobs_started": False, "core_a_new_training": False, "final_test_access": 0, "old_proxy_reclassified": True})
    (OUT / "LIVE_STATE_RECONCILIATION.md").write_text("# Core B live-state reconciliation\n\nThe pre-existing B3 controller reached a terminal state before this harvest. B1=7/7, true-pixel B2=6/6 and corrected B3=8/8 were independently revalidated. This analysis started no GPU job and did not access final-test.\n", encoding="utf-8")
    representative_ids, representative_meta = representative_episodes(formal + controls, inputs)
    write_json(OUT / "REPRESENTATIVE_EPISODES.json", {"rule": "nearest episode to persistence standardized arm14 error quantiles 0.25/0.50/0.75", "episodes": representative_meta, "model_for_trajectory_plot": "B3-FUS-D-L", "final_test_access": 0})
    write_csv(OUT / "TRAJECTORY_REPRESENTATIVE_EPISODES.csv", trajectory_rows(formal + controls, inputs, representative_ids))
    write_conclusions(formal + controls, read_json(B2_SELECTION), decision, effects, comparisons, proxy_rows)
    protocol = read_json(B23_PROTOCOL)
    analysis_protocol = {
        "analysis_id": "CORE_B_UNIFIED_ANALYSIS_20260909T163929Z",
        "created_utc": utc_now(),
        "mode": "CPU-only independent harvest",
        "protocol_id": protocol.get("protocol_id"),
        "population": {"train": 2741, "development_validation": 614, "target_steps": 30, "target_dimensions": 28, "final_test_access": 0},
        "formal_stages": {"B1": "7/7", "B2_true_pixel": "6/6", "B3_corrected": "8/8"},
        "proxy_boundary": "old sparse B2 contextual-token aggregation is auxiliary and excluded from formal selection",
        "metrics": {"primary": "standardized arm14 trajectory MAE", "secondary": ["standardized hand14/full28/final-step/RMSE", "native command-value MAE/RMSE", "per-time", "per-joint-time", "per-task", "first-difference"]},
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "episode", "same_indices_all_comparisons": True},
        "paired_p_value": {"method": "two-sided sign-flip permutation", "replicates": PERM_REPS, "seed": PERM_SEED, "holm": "within-family and global retained"},
        "b3_gate": "registered relative dynamic arm improvement >=2%, paired CI lower >0, Holm p<.05, hand/full degradation <=1%, interaction >=0, and external gate for fusion",
        "single_seed": True,
        "units_status": "unresolved physical semantics; native values are source-native command-value metrics",
        "no_new_training": True,
        "gpu_jobs_started_by_analysis": False,
        "final_test_access": 0,
        "python": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    write_json(OUT / "ANALYSIS_PROTOCOL_LOCKED.json", analysis_protocol)
    write_json(OUT / "ANALYSIS_STATUS.json", {"status": "complete", "analysis_id": analysis_protocol["analysis_id"], "b1": "7/7", "b2_true_pixel": "6/6", "b3_corrected": "8/8", "formal_result_rows": len(formal_results), "proxy_rows": len(proxy_records), "final_test_access": 0, "b3_candidate_decision": decision["recommendation_for_new_multi_seed"]})
    print(json.dumps({"status": "complete", "b1": "7/7", "b2_true_pixel": "6/6", "b3": "8/8", "out": str(OUT), "candidate_count": decision["candidate_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
