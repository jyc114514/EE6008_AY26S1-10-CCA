from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
SCHED_CODE = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
if str(SCHED_CODE) not in sys.path:
    sys.path.insert(0, str(SCHED_CODE))
if str(LEARN / "code") not in sys.path:
    sys.path.insert(0, str(LEARN / "code"))
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))
from d3_eq_contract import per_sample_loss
from d3_model import D3TimeSpacePredictor, parameter_count, state_hash
from collapse_diagnostic_core import loss_components
from future_query_model import FixedFutureQueryOverride
from orthogonal_query import hash_float32_array
import run_tiny_fit as old_runner

STAGE0_ROOT = ROOT / "runs/stage0"
FORMAL_ROOT = ROOT / "runs/formal"
LOCK_PATH = ROOT / "protocol/PROTOCOL_LOCKED.json"
MATRIX_PATH = ROOT / "protocol/MATRIX_4.csv"
MATCHED_PATH = ROOT / "protocol/MATCHED_BASELINE_4.csv"
QUERY_PATH = ROOT / "protocol/QUERY_TABLES_AND_HASHES.json"
QUERY_NPZ = ROOT / "protocol/HADAMARD_QUERY_TABLES.npz"
STAGE0_ACCEPT = ROOT / "protocol/B_STAGE0_ACCEPTANCE.json"
RESOURCE_STAGE0 = ROOT / "protocol/B_RESOURCE_GATE_STAGE0.json"
RESOURCE_PRE = ROOT / "protocol/RESOURCE_GATE_PRE_STAGE0.json"

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name("." + path.name + f".{os.getpid()}.part")
    torch.save(payload, part)
    with part.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(part, path)

def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def verify_locks() -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("CPU query-fit runner requires CUDA_VISIBLE_DEVICES empty and will not probe CUDA")
    lock = read_json(LOCK_PATH)
    if lock.get("status") not in ("LOCKED_BEFORE_STAGE0", "LOCKED_BEFORE_FORMAL"):
        raise AssertionError("B protocol is not in a runnable locked state")
    if lock.get("device") != "cpu" or lock.get("torch_threads") != 1 or lock.get("torch_interop_threads") != 1:
        raise AssertionError("CPU runtime lock mismatch")
    budget = lock.get("resource_budget", {})
    if int(budget.get("max_cpu_wall_seconds", 0)) != 3600:
        raise AssertionError("CPU wall-time budget must be locked at 60 minutes")
    if int(budget.get("max_new_output_bytes", 0)) != 5 * 1024**3:
        raise AssertionError("new-output budget must be locked at 5 GiB")
    if not RESOURCE_PRE.is_file() or read_json(RESOURCE_PRE).get("status") != "PASS":
        raise AssertionError("pre-Stage0 CPU/disk cost gate missing or not PASS")
    usage = shutil.disk_usage(ROOT)
    if usage.free / usage.total < float(budget["minimum_disk_free_fraction"]):
        raise OSError("current filesystem free fraction is below the locked minimum")
    if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > int(budget["max_new_output_bytes"]):
        raise OSError("current new-root output exceeds the locked output cap")
    for path, expected in lock["source_hashes"].items():
        actual = sha(Path(path))
        if actual != expected:
            raise AssertionError(f"locked source changed: {path}")
    old_runner.set_cpu_runtime(1)
    old_source_lock_hash = old_runner.verify_source_lock()
    if old_source_lock_hash != lock["historical_source_lock_sha256"]:
        raise AssertionError("historical tiny-fit source lock hash mismatch")
    query_meta = read_json(QUERY_PATH)
    if query_meta.get("status") != "PASS_QUERY_TABLES_LOCKED":
        raise AssertionError("query table metadata is not accepted")
    if sha(QUERY_PATH) != lock["query_tables_json_sha256"]:
        raise AssertionError("query table lock metadata hash mismatch")
    if sha(QUERY_NPZ) != lock["query_tables_npz_sha256"]:
        raise AssertionError("query table NPZ hash mismatch")
    matrix_sha = sha(MATRIX_PATH)
    if matrix_sha != lock["matrix_sha256"]:
        raise AssertionError("MATRIX_4 hash mismatch")
    baseline_sha = sha(MATCHED_PATH)
    if baseline_sha != lock["matched_baseline_sha256"]:
        raise AssertionError("MATCHED_BASELINE_4 hash mismatch")
    matrix = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
    if len(matrix) != 4 or len({r["run_id"] for r in matrix}) != 4:
        raise AssertionError("new query matrix must contain exactly four unique cells")
    return lock, matrix, query_meta

def query_table() -> torch.Tensor:
    with np.load(QUERY_NPZ, allow_pickle=False) as z:
        q = z["candidate_h30"].astype(np.float32, copy=True)
    if q.shape != (30, 128):
        raise AssertionError(f"locked H30 query table shape mismatch: {q.shape}")
    if hash_float32_array(q) != read_json(QUERY_PATH)["candidate_table_hashes"]["H30"]:
        raise AssertionError("runtime candidate table hash mismatch")
    return torch.from_numpy(q)

def build_model_for_row(row: dict[str, str], init_state: dict[str, torch.Tensor], query: torch.Tensor,
                        protocol: dict[str, Any]) -> tuple[D3TimeSpacePredictor, FixedFutureQueryOverride]:
    config = row["config"]
    seed = int(row["seed"])
    base = old_runner.build_model(config, seed, init_state)
    if state_hash(base) != protocol["init_state_hash"]:
        raise AssertionError(f"base INIT state hash mismatch for {config}")
    if parameter_count(base) != int(protocol["parameter_count"]):
        raise AssertionError("parameter count differs from frozen INIT")
    wrapper = FixedFutureQueryOverride(base, query)
    wrapper.eval()
    if parameter_count(wrapper) != parameter_count(base):
        raise AssertionError("query wrapper added trainable parameters")
    if wrapper.future_query_table.requires_grad:
        raise AssertionError("candidate query table must be a fixed buffer")
    return base, wrapper

def checkpoint_payload(row: dict[str, str], base: D3TimeSpacePredictor,
                       wrapper: FixedFutureQueryOverride, optimizer: torch.optim.Optimizer | None,
                       updates: int, protocol: dict[str, Any], history: dict[str, Any],
                       include_optimizer: bool) -> dict[str, Any]:
    payload = {
        "schema_version": 1, "run_id": row["run_id"], "config": row["config"],
        "loss": row["loss"], "panel": row["panel"], "horizon": 30,
        "seed": int(row["seed"]), "optimizer_updates": updates,
        "wrapper_state": wrapper.state_dict(), "base_state_hash": state_hash(base),
        "base_parameter_hash": parameter_hash(base), "parameter_count": parameter_count(base),
        "query_table_sha256": hash_float32_array(wrapper.future_query_table),
        "init_state_file_sha256": protocol["init_state_file_sha256"],
        "init_state_hash": protocol["init_state_hash"],
        "panel_lock_sha256": protocol["panel_lock_sha256"],
        "source_lock_sha256": protocol["historical_source_lock_sha256"],
        "protocol_sha256": sha(LOCK_PATH), "matrix_sha256": sha(MATRIX_PATH),
        "query_tables_json_sha256": sha(QUERY_PATH), "device": "cpu",
        "dtype": "float32", "dev_targets_opened": False,
        "final_or_heldout_access": 0, "new_benchmark_candidate": False,
        "history": history,
    }
    if include_optimizer and optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload

def save_prediction(path: Path, panel: str, metadata: dict[str, np.ndarray],
                    tensors: dict[str, torch.Tensor], prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(
            f,
            anchor_id=metadata[panel]["anchor_id"].astype("U96"),
            target_native=tensors["target"].cpu().numpy().astype(np.float32),
            anchor_native=tensors["anchor"].cpu().numpy().astype(np.float32),
            prediction_native=prediction.astype(np.float32),
        )

def run_one(row: dict[str, str], panel_tensors: dict[str, torch.Tensor],
            metadata: dict[str, np.ndarray], init_state: dict[str, torch.Tensor],
            query: torch.Tensor, protocol: dict[str, Any], updates: int,
            run_root: Path, stage0: bool, wall_deadline: float,
            max_output_bytes: int, min_disk_fraction: float) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"run root already exists; preserve it, do not overwrite: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    base, wrapper = build_model_for_row(row, init_state, query, protocol)
    panel = row["panel"]
    t = panel_tensors[panel]
    optimizer = torch.optim.AdamW(
        base.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    history: dict[str, list[dict[str, Any]]] = {"summary": [], "samples": []}
    update_seconds: list[float] = []
    grad_norms: list[float] = []
    start = time.perf_counter()
    for step in range(1, updates + 1):
        if time.perf_counter() >= wall_deadline:
            raise TimeoutError("locked total CPU wall-time budget reached")
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual = wrapper(t["state"], t["mask"], t["relative"], t["dynamic"], t["static"], t["anchor"])
        parts = loss_components(residual, t["target"], t["anchor"], t["sigma"], t["safe_fd"])
        locked = per_sample_loss(residual, t["target"], t["anchor"], t["sigma"], t["safe_fd"])
        if not torch.allclose(locked, parts["full"], atol=1e-7, rtol=0):
            raise AssertionError(f"objective contract mismatch: {row['run_id']} step={step}")
        objective = parts["full"].mean()
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError(f"non-finite objective: {row['run_id']} step={step}")
        objective.backward()
        for name, parameter in base.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"non-finite gradient {name}: {row['run_id']} step={step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=1.0)
        grad_norms.append(float(grad_norm.detach().item()))
        optimizer.step()
        base.zero_grad(set_to_none=True)
        update_seconds.append(time.perf_counter() - tick)
        evaluate_now = (step % 100 == 0) or (stage0 and step == updates)
        if evaluate_now:
            summary, sample_rows, prediction = old_runner.evaluate(
                wrapper, panel_tensors[panel], metadata[panel], step, row["run_id"], "FULL_LOSS"
            )
            history["summary"] = [x for x in history["summary"] if int(x["step"]) != step] + [summary]
            history["samples"] = [x for x in history["samples"] if int(x["step"]) != step] + sample_rows
            write_csv(run_root / "METRICS.csv", history["summary"])
            write_csv(run_root / "SAMPLE_METRICS.csv", history["samples"])
            pred_name = f"PREDICTIONS_STEP_{step:04d}.npz"
            save_prediction(run_root / pred_name, panel, metadata, panel_tensors[panel], prediction)
        if step % 100 == 0:
            usage = shutil.disk_usage(ROOT)
            if usage.free / usage.total < min_disk_fraction:
                raise OSError("filesystem free fraction fell below the locked minimum")
            if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > max_output_bytes:
                raise OSError("new-root output exceeded the locked byte cap")
        if not stage0 and step % 100 == 0:
            payload = checkpoint_payload(row, base, wrapper, optimizer, step, protocol, history, True)
            atomic_torch_save(run_root / "checkpoint_latest.pt", payload)
            write_json(run_root / "RUN_STATUS.json", {
                "status": "RUNNING", "run_id": row["run_id"], "optimizer_updates": step,
                "last_checkpoint_sha256": sha(run_root / "checkpoint_latest.pt"),
                "protocol_sha256": sha(LOCK_PATH), "device": "cpu",
                "dev_targets_opened": False, "final_or_heldout_access": 0,
            })
    wall = time.perf_counter() - start
    if int(history["summary"][-1]["step"]) != updates:
        summary, sample_rows, prediction = old_runner.evaluate(
            wrapper, panel_tensors[panel], metadata[panel], updates, row["run_id"], "FULL_LOSS"
        )
        history["summary"] = [x for x in history["summary"] if int(x["step"]) != updates] + [summary]
        history["samples"] = [x for x in history["samples"] if int(x["step"]) != updates] + sample_rows
        write_csv(run_root / "METRICS.csv", history["summary"])
        write_csv(run_root / "SAMPLE_METRICS.csv", history["samples"])
        save_prediction(run_root / f"PREDICTIONS_STEP_{updates:04d}.npz", panel, metadata, panel_tensors[panel], prediction)
    final_summary = history["summary"][-1]
    if stage0:
        final_prediction_path = run_root / f"PREDICTIONS_STEP_{updates:04d}.npz"
        checkpoint_path = run_root / "checkpoint_stage0.pt"
    else:
        final_prediction_path = run_root / "PREDICTIONS_FINAL.npz"
        with (run_root / "PREDICTIONS_STEP_2000.npz").open("rb") as src:
            with np.load(src, allow_pickle=False) as z:
                prediction = z["prediction_native"].copy()
                target = z["target_native"].copy()
                anchor = z["anchor_native"].copy()
                ids = z["anchor_id"].copy()
        with final_prediction_path.open("wb") as f:
            np.savez_compressed(f, anchor_id=ids, target_native=target, anchor_native=anchor,
                                prediction_native=prediction.astype(np.float32))
        checkpoint_path = run_root / "checkpoint_query_model.pt"
    payload = checkpoint_payload(row, base, wrapper, optimizer if stage0 else None, updates, protocol, history, False)
    atomic_torch_save(checkpoint_path, payload)
    reload_command = [
        sys.executable, str(ROOT / "code/verify_query_checkpoint.py"),
        "--run-dir", str(run_root),
    ]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    verify = subprocess.run(reload_command, cwd=str(ROOT), env=env, capture_output=True, text=True, check=False)
    (run_root / "FRESH_PROCESS_RELOAD_COMMAND.json").write_text(
        json.dumps({"argv": reload_command, "cwd": str(ROOT), "returncode": verify.returncode,
                    "stdout": verify.stdout, "stderr": verify.stderr}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if time.perf_counter() >= wall_deadline:
        raise TimeoutError("locked total CPU wall-time budget reached during fresh-process reload")
    if verify.returncode != 0:
        raise RuntimeError(f"fresh-process reload failed for {row['run_id']}: {verify.stdout} {verify.stderr}")
    if hash_float32_array(wrapper.future_query_table) != protocol["query_table_sha256_h30"]:
        raise AssertionError("candidate query buffer changed during optimizer updates")
    reload_result = read_json(run_root / "FRESH_PROCESS_RELOAD.json")
    if reload_result.get("status") != "PASS" or not reload_result.get("prediction_bitwise_equal"):
        raise AssertionError(f"fresh-process check not accepted: {row['run_id']}")
    complete = {
        "status": "PASS", "run_id": row["run_id"], "config": row["config"],
        "loss": row["loss"], "panel": panel, "seed": int(row["seed"]),
        "horizon_steps": 30, "optimizer_updates": updates,
        "protocol_sha256": sha(LOCK_PATH), "matrix_sha256": sha(MATRIX_PATH),
        "query_tables_json_sha256": sha(QUERY_PATH), "query_table_sha256": hash_float32_array(wrapper.future_query_table),
        "source_registry_sha256": protocol["source_registry_sha256"],
        "historical_source_lock_sha256": protocol["historical_source_lock_sha256"],
        "init_state_file_sha256": protocol["init_state_file_sha256"],
        "init_state_hash": protocol["init_state_hash"],
        "panel_lock_sha256": protocol["panel_lock_sha256"],
        "model_state_hash": state_hash(base), "parameter_hash": parameter_hash(base),
        "parameter_count": parameter_count(base),
        "candidate_trainable_parameter_count": 0,
        "device": "cpu", "dtype": "float32",
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "elapsed_seconds": wall,
        "mean_update_seconds": float(np.mean(update_seconds)) if update_seconds else 0.0,
        "p95_update_seconds": float(np.quantile(update_seconds, 0.95)) if update_seconds else 0.0,
        "max_grad_norm_before_clip": max(grad_norms) if grad_norms else 0.0,
        "final_position_error": final_summary["position_error_normalized_mean"],
        "final_fd_error": final_summary["fd_error_safe_scale_normalized_mean"],
        "final_tv_ratio": final_summary["tv_ratio_mean"],
        "strong_fit_samples": final_summary["per_sample_strong_fit_count"],
        "panel_rows": final_summary["panel_rows"],
        "stage0_only": stage0, "new_benchmark_candidate": False,
        "dev_targets_opened": False, "final_or_heldout_access": 0,
        "fresh_process_reload": True, "fresh_process_prediction_bitwise_equal": True,
        "final_checkpoint_sha256": sha(checkpoint_path),
        "prediction_sha256": sha(final_prediction_path),
        "prediction_path": str(final_prediction_path),
    }
    write_json(run_root / ("STAGE0_COMPLETE.json" if stage0 else "RUN_COMPLETE.json"), complete)
    return complete

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage0", action="store_true")
    group.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    lock, matrix, query_meta = verify_locks()
    budget = lock["resource_budget"]
    max_wall = float(budget["max_cpu_wall_seconds"])
    prior_seconds = 0.0
    if args.formal:
        if not STAGE0_ACCEPT.is_file():
            raise FileNotFoundError("Stage0 acceptance missing; formal fits prohibited")
        stage0 = read_json(STAGE0_ACCEPT)
        if stage0.get("status") != "PASS" or stage0.get("accepted_cells") != 4:
            raise AssertionError("Stage0 4/4 acceptance gate failed")
        stage0_resource = read_json(RESOURCE_STAGE0)
        if stage0_resource.get("status") != "PASS":
            raise AssertionError("Stage0 resource/cost gate did not pass")
        prior_seconds = float(stage0_resource["stage0_elapsed_seconds_sum"])
        projected_total = float(stage0_resource["formal_projection"]["projected_total_cpu_wall_seconds"])
        if projected_total > max_wall:
            raise TimeoutError("projected Stage0 plus formal work exceeds the locked 60-minute total")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise AssertionError("CPU threads were not set by the verified preflight lock")
    reserve_seconds = (float(stage0_resource["formal_projection"]["startup_and_input_reserve_seconds"]) +
                       float(stage0_resource["formal_projection"]["post_run_analysis_and_figure_seconds"])) if args.formal else float(budget.get("post_run_overhead_seconds", 0.0))
    wall_deadline = time.perf_counter() + max(0.0, max_wall - prior_seconds - reserve_seconds)
    panels, metadata, panel_lock = old_runner.load_panels()
    init_state = old_runner.load_init(read_json(LEARN / "protocol/B_PROTOCOL_LOCKED.json"))
    query = query_table()
    results = []
    failures = []
    run_root = STAGE0_ROOT if args.stage0 else FORMAL_ROOT
    for row in matrix:
        folder = run_root / row["run_id"]
        try:
            result = run_one(row, panels, metadata, init_state, query, lock,
                             4 if args.stage0 else 2000, folder, bool(args.stage0), wall_deadline,
                             int(budget["max_new_output_bytes"]), float(budget["minimum_disk_free_fraction"]))
            results.append(result)
        except Exception as exc:
            failure = {
                "status": "FAILED", "run_id": row["run_id"], "phase": "stage0" if args.stage0 else "formal",
                "utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc),
                "run_root": str(folder), "optimizer_updates_may_be_partial": folder.exists(),
                "protocol_sha256": sha(LOCK_PATH), "device": "cpu",
                "dev_targets_opened": False, "final_or_heldout_access": 0,
            }
            write_json(folder / ("STAGE0_FAILURE.json" if args.stage0 else "RUN_FAILURE.json"), failure)
            failures.append(failure)
    if args.stage0:
        accepted = len(results) == 4 and not failures and all(x["fresh_process_prediction_bitwise_equal"] for x in results)
        stage0_result = {
            "status": "PASS" if accepted else "FAIL",
            "accepted_cells": len(results), "all_four_cells_accepted": accepted,
            "attempted_cells": len(matrix), "updates_each": 4,
            "protocol_sha256": sha(LOCK_PATH), "matrix_sha256": sha(MATRIX_PATH),
            "query_tables_json_sha256": sha(QUERY_PATH),
            "elapsed_seconds_sum": sum(x["elapsed_seconds"] for x in results),
            "device": "cpu", "torch_threads": 1, "gpu_queries": 0, "gpu_use": 0, "gpu_lease": 0,
            "dev_targets_opened": False, "final_or_heldout_access": 0,
            "results": results, "failures": failures,
            "status_note": "Stage0 weights are discarded; all four formal runs start from the original INIT.",
        }
        write_json(STAGE0_ACCEPT, stage0_result)
        print(json.dumps(stage0_result, ensure_ascii=False, sort_keys=True, default=str))
        return 0 if accepted else 2
    result = {
        "status": "PASS" if len(results) == 4 and not failures else "PARTIAL",
        "accepted_cells": len(results), "attempted_cells": len(matrix),
        "protocol_sha256": sha(LOCK_PATH), "matrix_sha256": sha(MATRIX_PATH),
        "elapsed_seconds_sum": sum(x["elapsed_seconds"] for x in results),
        "device": "cpu", "torch_threads": 1,
        "dev_targets_opened": False, "final_or_heldout_access": 0,
        "results": results, "failures": failures,
    }
    write_json(ROOT / "protocol/B_FORMAL_ACCEPTANCE.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result["status"] == "PASS" else 2

if __name__ == "__main__":
    raise SystemExit(main())

