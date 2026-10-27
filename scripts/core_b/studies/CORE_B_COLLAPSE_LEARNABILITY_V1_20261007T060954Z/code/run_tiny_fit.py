#!/usr/bin/env python3
"""Bounded, restartable train-only D3 tiny-fit diagnostic (CPU runner)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
sys.path.insert(0, str(SCHED / "code"))
from d3_eq_contract import per_sample_loss  # noqa: E402
from d3_model import D3TimeSpacePredictor, state_hash  # noqa: E402
from d3_train_corrected import load_train_bundle, sha256_file  # noqa: E402
sys.path.insert(0, str(ROOT / "code"))
from collapse_diagnostic_core import exact_input_signature, loss_components  # noqa: E402

PROTO_PATH = ROOT / "protocol/B_PROTOCOL_LOCKED.json"
PANEL_PATH = ROOT / "protocol/B_PANEL_SELECTION_LOCK.json"
MATRIX_PATH = ROOT / "protocol/B_MATRIX_8.csv"
SOURCE_LOCK_PATH = ROOT / "protocol/B_EXECUTION_SOURCE_LOCK.json"
EXECUTION_LOCK_PATH = ROOT / "protocol/B_EXECUTION_LOCK.json"
RESOURCE_GATE_PATH = ROOT / "protocol/B_RESOURCE_GATE.json"
RUNS = ROOT / "runs"
SMOKE_ROOT = RUNS / "smoke"
STAGE0_ROOT = RUNS / "stage0"
FORMAL_ROOT = RUNS / "formal"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "no_rows"}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def verify_source_lock() -> str:
    lock = read_json(SOURCE_LOCK_PATH)
    for raw_path, expected in lock["file_hashes"].items():
        path = Path(raw_path)
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"B source changed after lock: {path}")
    return sha256_file(SOURCE_LOCK_PATH)


def set_cpu_runtime(threads: int = 1) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("CPU runner requires CUDA_VISIBLE_DEVICES='' and will not probe CUDA")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def disable_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            raise AssertionError("ordinary dropout remains enabled")
        if isinstance(module, nn.MultiheadAttention) and module.dropout != 0.0:
            raise AssertionError("attention dropout remains enabled")


def load_panels() -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, np.ndarray], dict[str, Any]]:
    panel_lock = read_json(PANEL_PATH)
    if panel_lock.get("status") != "LOCKED_BEFORE_STAGE0_OR_TRAINING":
        raise AssertionError("panel lock is not accepted")
    bundle = load_train_bundle()
    source_hashes = panel_lock["source_hashes"]
    for key, path in (
        ("train_target_cache", EQ / "d3/train_targets_trainonly.npz"),
        ("target_cache_manifest", EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"),
        ("feature_cache_manifest", BENCH / "d3/cache/D3_CACHE_MANIFEST.json"),
        ("video_normalizer", BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json"),
        ("target_normalizer", V3 / "data/V3_NORMALIZERS.json"),
    ):
        if sha256_file(path) != source_hashes[key]:
            raise AssertionError(f"B panel input changed: {key}")
    by_panel: dict[str, list[dict[str, Any]]] = {"P1": [], "P8": []}
    for row in panel_lock["rows"]:
        by_panel[row["panel"]].append(row)
    result: dict[str, dict[str, torch.Tensor]] = {}
    metadata: dict[str, np.ndarray] = {}
    for panel, rows in by_panel.items():
        rows = sorted(rows, key=lambda x: int(x["panel_order"]))
        positions = np.asarray([int(x["train_position"]) for x in rows], dtype=np.int64)
        ids = bundle["train_ids"][positions].astype("U96")
        if ids.tolist() != [str(x["anchor_id"]) for x in rows]:
            raise AssertionError(f"{panel} IDs differ from locked panel order")
        if len(set(x["source_group_id"] for x in rows)) != len(rows):
            raise AssertionError(f"{panel} includes repeated source groups")
        source_rows = bundle["train_indices"][positions]
        dynamic, static = bundle["grid_store"].get(source_rows)
        dynamic_norm = ((dynamic - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
        static_norm = ((static - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
        state = bundle["inputs"]["history_standardized"][positions].astype(np.float32, copy=False)
        mask = bundle["inputs"]["history_mask"][positions].astype(np.float32, copy=False)
        relative = bundle["inputs"]["history_relative_delta_standardized"][positions].astype(np.float32, copy=False)
        anchor = bundle["anchors"][positions].astype(np.float32, copy=False)
        target = bundle["targets60"][positions, :30].astype(np.float32, copy=False)
        for i, row in enumerate(rows):
            core = {"state": state[i], "history_mask": mask[i], "history_relative_delta": relative[i], "anchor_command": anchor[i]}
            if exact_input_signature(core) != row["state_input_sha256"]:
                raise AssertionError(f"state-control effective input hash mismatch: {row['anchor_id']}")
            if exact_input_signature({**core, "dynamic_grid_normalized": dynamic_norm[i]}) != row["TS11-D_effective_input_sha256"]:
                raise AssertionError(f"TS11-D effective input hash mismatch: {row['anchor_id']}")
            target_hash = hashlib.sha256(np.ascontiguousarray(target[i]).tobytes()).hexdigest()
            if target_hash != row["target_signature"]:
                raise AssertionError(f"train target hash mismatch: {row['anchor_id']}")
        result[panel] = {
            "state": torch.from_numpy(state), "mask": torch.from_numpy(mask),
            "relative": torch.from_numpy(relative), "dynamic": torch.from_numpy(dynamic_norm),
            "static": torch.from_numpy(static_norm), "anchor": torch.from_numpy(anchor),
            "target": torch.from_numpy(target), "sigma": bundle["target_sigma"].float(),
            "safe_fd": bundle["safe_fd_scale"].float(),
        }
        metadata[panel] = {
            "anchor_id": ids, "episode_id": bundle["episode_ids"][positions],
            "task_id": bundle["task_ids"][positions],
            "source_group_id": np.asarray([x["source_group_id"] for x in rows], dtype="U128"),
            "stratum": np.asarray([x["stratum"] for x in rows], dtype="U8"),
            "persistence_position": np.asarray([x["persistence_position_error_normalized"] for x in rows], dtype=np.float64),
            "persistence_fd": np.asarray([x["persistence_fd_error_safe_scale_normalized"] for x in rows], dtype=np.float64),
            "direction_threshold": np.asarray(read_json(SCHED / "MOTION_ACTIVITY_THRESHOLD.json")["direction_threshold_vector"], dtype=np.float32),
            "panel_hash": sha256_file(PANEL_PATH),
        }
    return result, metadata, panel_lock


def build_model(config: str, seed: int, init_state: dict[str, torch.Tensor]) -> nn.Module:
    model = D3TimeSpacePredictor(horizon=30, seed=seed, video_mode=config)
    model.load_state_dict(init_state, strict=True)
    disable_dropout(model)
    return model


def forward(model: nn.Module, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    residual = model(tensors["state"], tensors["mask"], tensors["relative"],
                     tensors["dynamic"], tensors["static"], tensors["anchor"])
    native = tensors["anchor"][:, None, :] + tensors["sigma"][None, None, :] * residual
    return residual, native


def evaluate(model: nn.Module, tensors: dict[str, torch.Tensor], meta: dict[str, np.ndarray],
             step: int, run_id: str, loss_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    with torch.no_grad():
        residual, native = forward(model, tensors)
        parts = loss_components(residual, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
        locked = per_sample_loss(residual, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
        if not torch.allclose(locked, parts["full"], atol=1e-7, rtol=0):
            raise AssertionError("evaluation loss differs from locked D3-EQ contract")
        pred_np = native.cpu().numpy().astype(np.float32, copy=True)
        target = tensors["target"].cpu().numpy()
        anchor = tensors["anchor"].cpu().numpy()
        sigma = tensors["sigma"].cpu().numpy()
        safe = tensors["safe_fd"].cpu().numpy()
        diff_pred = np.diff(pred_np, axis=1)
        diff_target = np.diff(target, axis=1)
        pred_tv = (np.abs(diff_pred) / safe[None, None, :]).mean(axis=(1, 2))
        target_tv = (np.abs(diff_target) / safe[None, None, :]).mean(axis=(1, 2))
        tv_ratio = np.divide(pred_tv, target_tv, out=np.full_like(pred_tv, np.nan), where=target_tv > 0)
        active = np.abs(diff_target) > meta["direction_threshold"][None, None, :]
        direction_ok = (np.sign(diff_pred) == np.sign(diff_target)) & active
        active_count = active.sum(axis=(1, 2))
        direction = np.divide(direction_ok.sum(axis=(1, 2)), active_count,
                              out=np.full(len(active_count), np.nan, dtype=np.float64), where=active_count > 0)
        time_std = pred_np.std(axis=1, ddof=0).mean(axis=1)
        time_std_norm = ((pred_np - anchor[:, None, :]) / sigma[None, None, :]).std(axis=1, ddof=0).mean(axis=1)
        offset = pred_np - anchor[:, None, :]
        time_mean_offset = offset.mean(axis=1)
        cross_sample_offset_std = float(time_mean_offset.std(axis=0, ddof=0).mean()) if len(time_mean_offset) > 1 else 0.0
        pos = parts["position"].cpu().numpy()
        fd = parts["fd"].cpu().numpy()
        weighted_fd = parts["weighted_fd"].cpu().numpy()
        full = parts["full"].cpu().numpy()
        # The optimizer objective is per-row full loss or position-only, reduced equally over this panel.
        selected_obj = full if loss_name == "FULL_LOSS" else pos
        sample_rows = []
        for i, anchor_id in enumerate(meta["anchor_id"].tolist()):
            pos_limit = 0.10 * meta["persistence_position"][i]
            fd_limit = 0.20 * meta["persistence_fd"][i]
            checks = {
                "position_pass": bool(pos[i] <= pos_limit),
                "fd_pass": bool(fd[i] <= fd_limit),
                "tv_pass": bool(0.8 <= tv_ratio[i] <= 1.2) if np.isfinite(tv_ratio[i]) else False,
            }
            sample_rows.append({
                "run_id": run_id, "step": step, "anchor_id": anchor_id,
                "source_group_id": str(meta["source_group_id"][i]), "original_task_index": int(meta["task_id"][i]),
                "position_error_normalized": float(pos[i]),
                "persistence_position_error_normalized": float(meta["persistence_position"][i]),
                "position_error_limit_10pct_persistence": float(pos_limit),
                "position_gain_vs_persistence": float(1.0-pos[i]/meta["persistence_position"][i]) if meta["persistence_position"][i] > 0 else "NA",
                "fd_error_safe_scale_normalized": float(fd[i]),
                "persistence_fd_error_safe_scale_normalized": float(meta["persistence_fd"][i]),
                "fd_error_limit_20pct_persistence": float(fd_limit),
                "fd_gain_vs_persistence": float(1.0-fd[i]/meta["persistence_fd"][i]) if meta["persistence_fd"][i] > 0 else "NA",
                "prediction_tv_safe_scale": float(pred_tv[i]), "target_tv_safe_scale": float(target_tv[i]),
                "tv_ratio": float(tv_ratio[i]) if np.isfinite(tv_ratio[i]) else "NA",
                "direction_accuracy_active": float(direction[i]) if np.isfinite(direction[i]) else "NA",
                "direction_active_steps": int(active_count[i]),
                "within_sample_time_std_native": float(time_std[i]),
                "within_sample_time_std_normalized": float(time_std_norm[i]),
                "prediction_minus_anchor_max_abs": float(np.abs(offset[i]).max()),
                "prediction_native_min": float(pred_np[i].min()), "prediction_native_max": float(pred_np[i].max()),
                "strong_position_pass": checks["position_pass"], "strong_fd_pass": checks["fd_pass"],
                "strong_tv_pass": checks["tv_pass"],
                "strong_all_metrics_pass": all(checks.values()),
            })
        summary = {
            "run_id": run_id, "step": step, "loss_name": loss_name,
            "objective_mean": float(selected_obj.mean()),
            "position_error_normalized_mean": float(pos.mean()),
            "weighted_0p1_fd_loss_mean": float(weighted_fd.mean()),
            "fd_error_safe_scale_normalized_mean": float(fd.mean()),
            "persistence_position_error_normalized_mean": float(meta["persistence_position"].mean()),
            "persistence_fd_error_safe_scale_normalized_mean": float(meta["persistence_fd"].mean()),
            "position_gain_vs_persistence_mean": float(1.0-pos.mean()/meta["persistence_position"].mean()),
            "fd_gain_vs_persistence_mean": float(1.0-fd.mean()/meta["persistence_fd"].mean()),
            "prediction_tv_safe_scale_mean": float(pred_tv.mean()),
            "target_tv_safe_scale_mean": float(target_tv.mean()),
            "tv_ratio_mean": float(np.nanmean(tv_ratio)),
            "direction_accuracy_active_mean": float(np.nanmean(direction)) if np.isfinite(direction).any() else "NA",
            "direction_active_steps_total": int(active_count.sum()),
            "within_sample_time_std_native_mean": float(time_std.mean()),
            "within_sample_time_std_normalized_mean": float(time_std_norm.mean()),
            "cross_sample_time_mean_offset_std_native": cross_sample_offset_std,
            "all_samples_strong_fit": all(row["strong_all_metrics_pass"] for row in sample_rows),
            "per_sample_strong_fit_count": sum(bool(row["strong_all_metrics_pass"]) for row in sample_rows),
            "panel_rows": len(sample_rows),
        }
    return summary, sample_rows, pred_np


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(payload, part)
    with part.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(part, path)


def checkpoint_payload(model: nn.Module, optimizer: torch.optim.Optimizer, step: int,
                       history: dict[str, Any], run_id: str, config: str, loss_name: str,
                       panel: str, protocol_hash: str, panel_hash: str, source_lock_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "status": "RESUMABLE_TRAIN_ONLY_DIAGNOSTIC",
        "run_id": run_id, "config": config, "horizon": 30, "seed": 20261002,
        "loss_name": loss_name, "panel": panel, "optimizer_updates": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "history": history, "protocol_sha256": protocol_hash, "panel_lock_sha256": panel_hash,
        "source_lock_sha256": source_lock_hash, "model_state_hash": state_hash(model),
        "device": "cpu", "final_test_access": 0,
    }


def load_init(protocol: dict[str, Any]) -> dict[str, torch.Tensor]:
    path = BENCH / f"d3/runs/INIT_S{protocol['seed']}.pt"
    if sha256_file(path) != protocol["init_state_file_sha256"]:
        raise AssertionError("original INIT file hash changed")
    init = torch.load(path, map_location="cpu", weights_only=True)
    expected_state_hash = read_json(EQ / "protocol/D3_EQ_SEED_REGISTRY.json")["seeds"][str(protocol["seed"])]["init_state_hash"]
    test_model = D3TimeSpacePredictor(horizon=30, seed=protocol["seed"], video_mode="TS11-D")
    test_model.load_state_dict(init, strict=True)
    if state_hash(test_model) != expected_state_hash:
        raise AssertionError("original INIT state_dict hash mismatch")
    return init


def train_updates(run_id: str, config: str, loss_name: str, panel: str, tensors: dict[str, torch.Tensor],
                  meta: dict[str, np.ndarray], steps: int, run_root: Path, protocol_hash: str,
                  panel_hash: str, source_lock_hash: str, init_state: dict[str, torch.Tensor],
                  formal: bool, resume: bool = False) -> dict[str, Any]:
    protocol = read_json(PROTO_PATH)
    run_root.mkdir(parents=True, exist_ok=True)
    complete_path = run_root / ("RUN_COMPLETE.json" if formal else "STAGE0_OR_SMOKE_COMPLETE.json")
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete.get("status") == "PASS" and complete.get("protocol_sha256") == protocol_hash:
            return {"status": "PASS_REUSED", "run_id": run_id, "run_root": str(run_root)}
        raise AssertionError(f"existing run completion conflicts with locked protocol: {run_id}")
    if not resume and any(run_root.iterdir()):
        raise FileExistsError(f"partial run directory exists; preserve it and resume explicitly: {run_root}")
    model = build_model(config, int(protocol["seed"]), init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    history: dict[str, Any] = {"summary": [], "samples": []}
    start_step = 0
    latest_path = run_root / "checkpoint_latest.pt"
    if resume and not latest_path.is_file():
        raise FileNotFoundError(f"cannot resume without checkpoint_latest.pt: {run_id}")
    if resume and latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        for key, expected in (("run_id", run_id), ("config", config), ("loss_name", loss_name),
                              ("panel", panel), ("protocol_sha256", protocol_hash),
                              ("panel_lock_sha256", panel_hash), ("source_lock_sha256", source_lock_hash)):
            if saved.get(key) != expected:
                raise AssertionError(f"resume checkpoint {key} mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start_step = int(saved["optimizer_updates"])
        if state_hash(model) != saved["model_state_hash"]:
            raise AssertionError("resume checkpoint model hash mismatch")
    elif latest_path.exists() and not resume:
        raise FileExistsError(f"resumable checkpoint exists; use --resume {run_id}")
    start_time = time.perf_counter()
    step_times = []
    for step in range(start_step + 1, steps + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        residual, _ = forward(model, tensors)
        parts = loss_components(residual, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
        locked = per_sample_loss(residual, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
        if not torch.allclose(locked, parts["full"], atol=1e-7, rtol=0):
            raise AssertionError("training objective components differ from locked contract")
        objective = parts["full"].mean() if loss_name == "FULL_LOSS" else parts["position"].mean()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        step_times.append(time.perf_counter() - t0)
        if step % 100 == 0 or (not formal and step == steps):
            summary, sample_rows, prediction = evaluate(model, tensors, meta, step, run_id, loss_name)
            history["summary"] = [row for row in history["summary"] if int(row["step"]) != step] + [summary]
            history["samples"] = [row for row in history["samples"] if int(row["step"]) != step] + sample_rows
            write_csv(run_root / "METRICS.csv", history["summary"])
            write_csv(run_root / "SAMPLE_METRICS.csv", history["samples"])
            with (run_root / f"PREDICTIONS_STEP_{step:04d}.npz").open("wb") as f:
                np.savez_compressed(f, anchor_id=meta["anchor_id"].astype("U96"),
                                    target_native=tensors["target"].numpy(),
                                    anchor_native=tensors["anchor"].numpy(),
                                    prediction_native=prediction.astype(np.float32))
        if formal and step % 100 == 0:
            atomic_checkpoint(latest_path, checkpoint_payload(model, optimizer, step, history, run_id,
                              config, loss_name, panel, protocol_hash, panel_hash, source_lock_hash))
        if not formal and step == steps:
            atomic_checkpoint(latest_path, checkpoint_payload(model, optimizer, step, history, run_id,
                              config, loss_name, panel, protocol_hash, panel_hash, source_lock_hash))
    wall = time.perf_counter() - start_time
    final_step = int(steps)
    if formal:
        summary, sample_rows, prediction = evaluate(model, tensors, meta, final_step, run_id, loss_name)
        history["summary"] = [row for row in history["summary"] if int(row["step"]) != final_step] + [summary]
        history["samples"] = [row for row in history["samples"] if int(row["step"]) != final_step] + sample_rows
        write_csv(run_root / "METRICS.csv", history["summary"])
        write_csv(run_root / "SAMPLE_METRICS.csv", history["samples"])
        with (run_root / "PREDICTIONS_FINAL.npz").open("wb") as f:
            np.savez_compressed(f, anchor_id=meta["anchor_id"].astype("U96"),
                                target_native=tensors["target"].numpy(),
                                anchor_native=tensors["anchor"].numpy(), prediction_native=prediction.astype(np.float32))
        atomic_checkpoint(run_root / "checkpoint_final.pt", {"model": model.state_dict(),
                         "model_state_hash": state_hash(model), "run_id": run_id,
                         "protocol_sha256": protocol_hash, "panel_lock_sha256": panel_hash,
                         "optimizer_updates": steps, "init_state_file_sha256": protocol["init_state_file_sha256"],
                         "device": "cpu", "final_test_access": 0})
        if latest_path.exists():
            latest_path.unlink()
    else:
        summary = history["summary"][-1]
        with (run_root / "PREDICTIONS_STEP_0004.npz").open("rb") as f:
            with np.load(f, allow_pickle=False) as z:
                prediction = z["prediction_native"].copy()
    complete = {
        "status": "PASS", "run_id": run_id, "config": config, "loss": loss_name,
        "panel": panel, "seed": int(protocol["seed"]), "horizon_steps": 30,
        "optimizer_updates": steps, "protocol_sha256": protocol_hash,
        "panel_lock_sha256": panel_hash, "source_lock_sha256": source_lock_hash,
        "init_state_file_sha256": protocol["init_state_file_sha256"],
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": wall, "mean_update_seconds": float(np.mean(step_times)) if step_times else 0.0,
        "p95_update_seconds": float(np.quantile(step_times, 0.95)) if step_times else 0.0,
        "strong_training_fit": bool(summary["all_samples_strong_fit"]),
        "strong_fit_samples": int(summary["per_sample_strong_fit_count"]),
        "panel_rows": int(summary["panel_rows"]),
        "dev_targets_opened": False, "final_or_heldout_access": 0,
        "new_benchmark_candidate": False,
        "final_checkpoint_sha256": sha256_file(run_root / "checkpoint_final.pt") if formal else sha256_file(latest_path),
        "prediction_sha256": sha256_file(run_root / "PREDICTIONS_FINAL.npz") if formal else sha256_file(run_root / "PREDICTIONS_STEP_0004.npz"),
    }
    write_json(complete_path if formal else run_root / "STAGE0_OR_SMOKE_COMPLETE.json", complete)
    return complete


def execute_smoke() -> dict[str, Any]:
    if verify_source_lock() != sha256_file(SOURCE_LOCK_PATH):
        raise AssertionError("B source lock mismatch")
    protocol = read_json(PROTO_PATH)
    panel_hash = sha256_file(PANEL_PATH)
    protocol_hash = sha256_file(PROTO_PATH)
    matrix = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
    if len(matrix) != 8 or any(row["protocol_sha256"] != protocol_hash or row["panel_lock_sha256"] != panel_hash for row in matrix):
        raise AssertionError("B matrix is not bound to B protocol and panel lock")
    set_cpu_runtime(1)
    panels, metadata, _panel = load_panels()
    init_state = load_init(protocol)
    records = []
    for row in matrix:
        panel, config, loss_name, run_id = row["panel"], row["config"], row["loss"], row["run_id"]
        is_stage0 = panel == "P1"
        root = (STAGE0_ROOT if is_stage0 else SMOKE_ROOT) / run_id
        expected_class = "stage0_p1" if is_stage0 else "cost_probe_p8"
        result = train_updates(run_id, config, loss_name, panel, panels[panel], metadata[panel], 4,
                               root, protocol_hash, panel_hash, sha256_file(SOURCE_LOCK_PATH),
                               init_state, formal=False, resume=False)
        result["probe_class"] = expected_class
        records.append(result)
        print(json.dumps({"status": result["status"], "run_id": run_id, "panel": panel,
                          "elapsed_seconds": result["elapsed_seconds"],
                          "mean_update_seconds": result["mean_update_seconds"]}), flush=True)
    write_csv(ROOT / "results/FOUR_UPDATE_SMOKE_TIMING.csv", records)
    manifest = {
        "status": "PASS", "runs": len(records), "p1_stage0_cells": sum(x["probe_class"] == "stage0_p1" for x in records),
        "p8_cost_probe_cells": sum(x["probe_class"] == "cost_probe_p8" for x in records),
        "all_updates": 4, "device": "cpu", "torch_threads": torch.get_num_threads(),
        "protocol_sha256": protocol_hash, "panel_lock_sha256": panel_hash,
        "timing_csv_sha256": sha256_file(ROOT / "results/FOUR_UPDATE_SMOKE_TIMING.csv"),
        "dev_targets_opened": False, "final_or_heldout_access": 0,
    }
    write_json(ROOT / "protocol/B_STAGE0_SMOKE_ACCEPTANCE.json", manifest)
    return manifest


def verify_stage0() -> dict[str, Any]:
    source_lock_hash = verify_source_lock()
    protocol = read_json(PROTO_PATH)
    protocol_hash = sha256_file(PROTO_PATH)
    panel_hash = sha256_file(PANEL_PATH)
    panel_lock = read_json(PANEL_PATH)
    p1_row = next(x for x in panel_lock["rows"] if x["panel"] == "P1")
    set_cpu_runtime(1)
    panels, metadata, _ = load_panels()
    results = []
    for config in ("TS11-D", "STATE-CTRL"):
        for loss_name in ("FULL_LOSS", "POS_ONLY"):
            run_id = f"TINYFIT-{config}-H30-{loss_name}-P1-S20261002"
            root = STAGE0_ROOT / run_id
            status = read_json(root / "STAGE0_OR_SMOKE_COMPLETE.json")
            if status.get("optimizer_updates") != 4 or status.get("panel") != "P1" or status.get("protocol_sha256") != protocol_hash:
                raise AssertionError(f"Stage0 status mismatch: {run_id}")
            ckpt = torch.load(root / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
            model = build_model(config, int(protocol["seed"]), load_init(protocol))
            model.load_state_dict(ckpt["model"], strict=True)
            loaded_hash = state_hash(model)
            if loaded_hash != ckpt["model_state_hash"]:
                raise AssertionError(f"Stage0 fresh-process model hash mismatch: {run_id}")
            if sha256_file(root / "checkpoint_latest.pt") != status["final_checkpoint_sha256"]:
                raise AssertionError(f"Stage0 checkpoint file hash mismatch: {run_id}")
            with np.load(root / "PREDICTIONS_STEP_0004.npz", allow_pickle=False) as saved:
                ids = saved["anchor_id"].astype("U96")
                saved_prediction = saved["prediction_native"].copy()
            if ids.tolist() != metadata["P1"]["anchor_id"].tolist():
                raise AssertionError(f"Stage0 prediction IDs differ: {run_id}")
            with torch.no_grad():
                _, prediction = forward(model, panels["P1"])
            current = prediction.numpy().astype(np.float32)
            if not np.array_equal(saved_prediction, current):
                raise AssertionError(f"Stage0 save/reload predictions differ bitwise: {run_id}")
            results.append({"run_id": run_id, "config": config, "loss": loss_name,
                            "fresh_process_model_hash_equal": True, "prediction_bitwise_equal": True,
                            "updates": 4, "P1_anchor_id": str(p1_row["anchor_id"])})
    result = {"status": "PASS", "cells": 4, "fresh_process_reload": True,
              "prediction_replay_bitwise": True, "source_lock_sha256": source_lock_hash,
              "protocol_sha256": protocol_hash, "panel_lock_sha256": panel_hash,
              "dev_targets_opened": False, "final_or_heldout_access": 0, "results": results}
    write_json(ROOT / "protocol/B_STAGE0_RELOAD_ACCEPTANCE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-all", action="store_true", help="4-update CPU probe for all 8 matrix cells; P1 cells are Stage0")
    group.add_argument("--verify-stage0", action="store_true", help="fresh-process reload verification of the four P1 Stage0 cells")
    group.add_argument("--run", metavar="RUN_ID", help="execute one locked 2000-update formal run")
    group.add_argument("--resume", metavar="RUN_ID", help="resume one locked run from checkpoint_latest.pt")
    args = parser.parse_args()
    if args.smoke_all:
        result = execute_smoke()
    elif args.verify_stage0:
        result = verify_stage0()
    else:
        run_id = args.run or args.resume
        verify_source_lock()
        if not EXECUTION_LOCK_PATH.is_file() or not RESOURCE_GATE_PATH.is_file():
            raise FileNotFoundError("B execution lock/resource gate missing; formal optimization prohibited")
        execution = read_json(EXECUTION_LOCK_PATH)
        gate = read_json(RESOURCE_GATE_PATH)
        if execution.get("status") != "LOCKED_BEFORE_FULL_FITS" or gate.get("status") != "PASS":
            raise AssertionError("CPU/resource gate not accepted")
        if execution.get("device") != "cpu":
            raise AssertionError("this runner is CPU-only; GPU execution requires separate approved UUID wrapper")
        protocol = read_json(PROTO_PATH)
        panel_hash = sha256_file(PANEL_PATH); protocol_hash = sha256_file(PROTO_PATH)
        matrix = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
        row = next((x for x in matrix if x["run_id"] == run_id), None)
        if row is None:
            raise KeyError(f"run is not in locked 8-cell matrix: {run_id}")
        if execution.get("matrix_sha256") != sha256_file(MATRIX_PATH) or execution.get("protocol_sha256") != protocol_hash or execution.get("panel_lock_sha256") != panel_hash:
            raise AssertionError("B execution lock hashes differ from current locked matrix/protocol/panel")
        set_cpu_runtime(int(execution["torch_threads"]))
        panels, metadata, _ = load_panels()
        init_state = load_init(protocol)
        result = train_updates(run_id, row["config"], row["loss"], row["panel"], panels[row["panel"]],
                               metadata[row["panel"]], 2000, FORMAL_ROOT / run_id,
                               protocol_hash, panel_hash, sha256_file(SOURCE_LOCK_PATH), init_state,
                               formal=True, resume=bool(args.resume))
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
