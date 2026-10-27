#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-015; source SHA-256: 73b8e815e3c2fb1dffc93f44ca39bb4e09ce226bf9aea9c487f11e049a728109
"""Train-only, CPU replay and local-gradient forensics for D3-EQ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("EE6008_COLLAPSE_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = Path(os.environ.get("EE6008_D3_EQ_STUDY_ROOT", PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z")).expanduser().resolve()
DIAG = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"
PUBLIC_REPO_ROOT = Path(__file__).resolve().parents[5]
PUBLIC_D3_CODE = PUBLIC_REPO_ROOT / "scripts/core_b/studies/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
sys.path.insert(0, str(PUBLIC_D3_CODE))
from d3_eq_contract import per_sample_loss  # noqa: E402
from d3_model import D3TimeSpacePredictor, sinusoidal, state_hash  # noqa: E402
from d3_train_corrected_source_342cf4f3 import load_train_bundle, sha256_file  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collapse_diagnostic_core import (  # noqa: E402
    exact_input_signature,
    loss_components,
    select_stratified_groups,
    trajectory_decomposition,
)

PROTOCOL_ID = "CORE_B_COLLAPSE_LEARNABILITY_DIAGNOSTIC_V1"
LOCK_PATH = ROOT / "protocol/PHASE_A_EXECUTION_LOCK.json"
AMENDMENT_PATH = ROOT / "protocol/PHASE_A_AMENDMENT_01.json"
SOURCE_LOCK_PATH = ROOT / "protocol/PHASE_A_SOURCE_LOCK.json"
FIXTURE_PATH = ROOT / "TRAIN_FORENSIC_FIXTURE.json"
OUT = ROOT / "forensics"


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_array(array: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(array).tobytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def train_group_map() -> dict[int, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(SPLIT_MANIFEST, columns=["episode_index", "split", "video_sha256_fingerprint"])
    episodes = table["episode_index"].to_numpy(zero_copy_only=False)
    splits = table["split"].to_pylist()
    fingerprints = table["video_sha256_fingerprint"].to_pylist()
    mapping: dict[int, tuple[str, str]] = {}
    for episode, split, fingerprint in zip(episodes, splits, fingerprints):
        if fingerprint is None or not str(fingerprint).strip():
            raise AssertionError(f"missing source fingerprint for episode {episode}")
        normalized_split = "dev" if str(split) in {"validation", "val"} else str(split)
        value = (normalized_split, str(fingerprint))
        key = int(episode)
        if key in mapping and mapping[key] != value:
            raise AssertionError(f"conflicting split/source rows for episode {key}")
        mapping[key] = value
    result: dict[int, str] = {}
    for episode, (split, group) in mapping.items():
        if split == "train":
            result[episode] = group
    return result


def source_hashes() -> dict[str, str]:
    task_prompt_value = os.environ.get("EE6008_TASK_PROMPT_PATH")
    if not task_prompt_value:
        raise RuntimeError("EE6008_TASK_PROMPT_PATH must point to the local task attachment")
    task_prompt = Path(task_prompt_value).expanduser().resolve()
    if not task_prompt.is_file():
        raise FileNotFoundError(task_prompt)
    return {
        "task_prompt": sha256_file(task_prompt),
        "phase_a_lock": sha256_file(LOCK_PATH),
        "phase_a_amendment": sha256_file(AMENDMENT_PATH),
        "phase_a_source_lock": sha256_file(SOURCE_LOCK_PATH),
        "source_acceptance": sha256_file(ROOT / "SOURCE_ACCEPTANCE.json"),
        "study_protocol": sha256_file(ROOT / "PROTOCOL_LOCKED.json"),
        "scheduler_matrix": sha256_file(SCHED / "protocol/MATRIX_12.csv"),
        "scheduler_model_source": sha256_file(SCHED / "code/d3_model.py"),
        "locked_model_source": sha256_file(EQ / "code/d3_model.py"),
        "v3_inputs": sha256_file(V3 / "data/V3_MAIN_INPUTS.npz"),
        "target_normalizer": sha256_file(V3 / "data/V3_NORMALIZERS.json"),
        "target_cache": sha256_file(EQ / "d3/train_targets_trainonly.npz"),
        "target_cache_manifest": sha256_file(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"),
        "feature_cache_manifest": sha256_file(BENCH / "d3/cache/D3_CACHE_MANIFEST.json"),
        "video_normalizer": sha256_file(BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json"),
        "motion_threshold": sha256_file(SCHED / "MOTION_ACTIVITY_THRESHOLD.json"),
        "split_manifest": sha256_file(SPLIT_MANIFEST),
    }


def verify_phase_a_source_lock() -> str:
    lock = json_read(SOURCE_LOCK_PATH)
    for raw_path, expected in lock["file_hashes"].items():
        path = Path(raw_path)
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"Phase A source changed after lock: {path}")
    return sha256_file(SOURCE_LOCK_PATH)


def load_train_metadata(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        split = archive["split_code"].astype(np.int8, copy=False)
        indices = bundle["train_indices"]
        if np.any(split[indices] != 0):
            raise AssertionError("non-train row entered train-only replay")
        required = ("anchor_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp")
        meta = {key: np.asarray(archive[key][indices]).copy() for key in required}
    if not np.array_equal(meta["anchor_id"].astype("U64"), bundle["train_ids"]):
        raise AssertionError("train input metadata IDs differ from train bundle")
    if not np.array_equal(meta["episode_index"].astype(np.int64), bundle["episode_ids"]):
        raise AssertionError("train input metadata episodes differ from target cache")
    if not np.array_equal(meta["original_task_index"].astype(np.int64), bundle["task_ids"]):
        raise AssertionError("train input metadata tasks differ from train bundle")
    if not np.isfinite(meta["anchor_timestamp"].astype(np.float64)).all():
        raise AssertionError("non-finite train anchor timestamp")
    return meta


def score_motion_strata(bundle: dict[str, Any]) -> tuple[np.ndarray, dict[str, str], dict[str, float]]:
    targets = bundle["targets60"][:, :30]
    sigma = bundle["target_sigma"].numpy()
    per_window = (np.abs(np.diff(targets, axis=1)) / sigma[None, None, :]).mean(axis=(1, 2))
    episode_ids = bundle["episode_ids"].astype(np.int64, copy=False)
    groups_by_episode = train_group_map()
    row_groups = np.asarray([groups_by_episode[int(e)] for e in episode_ids], dtype="U128")
    unique_episodes = np.unique(episode_ids)
    episode_group: dict[int, str] = {}
    episode_score: dict[int, float] = {}
    for episode in unique_episodes:
        rows = episode_ids == episode
        groups = np.unique(row_groups[rows])
        if len(groups) != 1:
            raise AssertionError(f"episode {episode} maps to multiple source groups")
        episode_group[int(episode)] = str(groups[0])
        episode_score[int(episode)] = float(per_window[rows].mean())
    by_group: dict[str, list[float]] = defaultdict(list)
    for episode, group in episode_group.items():
        by_group[group].append(episode_score[episode])
    scores = {group: float(np.mean(values)) for group, values in by_group.items()}
    threshold_doc = json_read(SCHED / "MOTION_ACTIVITY_THRESHOLD.json")
    threshold = threshold_doc["strata_thresholds"]["30"]
    q25, q75 = float(threshold["q25"]), float(threshold["q75"])
    labels = {
        group: ("low" if score <= q25 else "high" if score >= q75 else "mid")
        for group, score in scores.items()
    }
    if len(scores) != 2712 or set(labels.values()) != {"low", "mid", "high"}:
        raise AssertionError("train source-group activity support differs from frozen contract")
    return per_window, labels, {"q25": q25, "q75": q75}


def lock_fixture() -> dict[str, Any]:
    source_lock_hash = verify_phase_a_source_lock()
    lock = json_read(LOCK_PATH)
    if lock.get("locked_before_any_phase_a_forward") is not True:
        raise AssertionError("Phase A execution lock is absent")
    amendment = json_read(AMENDMENT_PATH)
    if amendment.get("status") != "LOCKED_BEFORE_CORRECTED_REPLAY" or amendment.get("selection_changed") is not False:
        raise AssertionError("Phase A horizon amendment is absent or changes sample selection")
    bundle = load_train_bundle()
    metadata = load_train_metadata(bundle)
    per_window, labels, thresholds = score_motion_strata(bundle)
    groups_by_episode = train_group_map()
    records = []
    for pos, (anchor_id, episode, task, frame, timestamp) in enumerate(zip(
        metadata["anchor_id"], metadata["episode_index"], metadata["original_task_index"],
        metadata["anchor_frame"], metadata["anchor_timestamp"],
    )):
        group = groups_by_episode[int(episode)]
        records.append({
            "anchor_id": str(anchor_id), "train_position": pos,
            "source_row_index": int(bundle["train_indices"][pos]),
            "episode_id": int(episode), "source_group_id": group,
            "stratum": labels[group], "original_task_index": int(task),
            "anchor_frame": int(frame), "anchor_timestamp": float(timestamp),
        })
    selected = select_stratified_groups(records, {"low": 5, "mid": 5, "high": 6}, PROTOCOL_ID)
    positions = np.asarray([int(row["train_position"]) for row in selected], dtype=np.int64)
    source_rows = bundle["train_indices"][positions]
    dynamic, static = bundle["grid_store"].get(source_rows)
    video_mean, video_std = bundle["video_mean"], bundle["video_std"]
    dynamic_norm = ((dynamic - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
    static_norm = ((static - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
    state = bundle["inputs"]["history_standardized"][positions]
    masks = bundle["inputs"]["history_mask"][positions]
    rel = bundle["inputs"]["history_relative_delta_standardized"][positions]
    anchors = bundle["anchors"][positions]
    targets = bundle["targets60"][positions, :30]
    rows = []
    for i, base in enumerate(selected):
        core = {
            "state": state[i], "history_mask": masks[i],
            "history_relative_delta": rel[i], "anchor_command": anchors[i],
        }
        state_sig = exact_input_signature(core)
        ts11_sig = exact_input_signature({**core, "dynamic_grid_normalized": dynamic_norm[i]})
        row = {
            **base,
            "state_input_sha256": state_sig,
            "TS11-D_effective_input_sha256": ts11_sig,
            "dynamic_grid_raw_sha256": sha_array(dynamic[i]),
            "dynamic_grid_normalized_sha256": sha_array(dynamic_norm[i]),
            "static_grid_raw_sha256": sha_array(static[i]),
            "static_grid_normalized_sha256": sha_array(static_norm[i]),
            "target_h30_sha256": sha_array(targets[i]),
            "target_h60_sha256": sha_array(bundle["targets60"][positions[i]]),
        }
        rows.append(row)
    if len({x["source_group_id"] for x in rows}) != 16:
        raise AssertionError("fixture source groups are not unique")
    if not np.isfinite(dynamic_norm).all() or not np.isfinite(targets).all() or not np.isfinite(bundle["targets60"][positions]).all():
        raise AssertionError("fixture contains non-finite train data")
    result = {
        "schema_version": 1, "status": "LOCKED_BEFORE_FORWARD",
        "protocol_id": PROTOCOL_ID, "phase_a_lock_sha256": sha256_file(LOCK_PATH),
        "phase_a_amendment_sha256": sha256_file(AMENDMENT_PATH),
        "phase_a_source_lock_sha256": source_lock_hash,
        "selected_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "target_scope": "primary_train_only", "rows": 16,
        "strata_thresholds": thresholds, "strata_quotas": {"low": 5, "mid": 5, "high": 6},
        "selected_strata_counts": {key: sum(row["stratum"] == key for row in rows) for key in ("low", "mid", "high")},
        "source_hashes": source_hashes(),
        "row_contract": {
            "train_windows": len(bundle["train_ids"]), "train_episodes": len(np.unique(bundle["episode_ids"])),
            "target_cache_scope": "primary_train_only", "dev_targets_opened": False,
            "normalizer_target_sigma_sha256": bundle["normalizer_hash"],
            "safe_fd_scale_sha256": sha_array(bundle["safe_fd_scale"].numpy()),
            "target_cache_sha256": bundle["target_cache_hash"],
            "feature_cache_manifest_sha256": bundle["cache_manifest_hash"],
            "video_normalizer_sha256": bundle["video_normalizer_hash"],
        },
        "rows_selected_by_locked_rule": rows,
        "forward_executed_before_lock": False,
    }
    write_json(FIXTURE_PATH, result)
    return result


def selected_cell_index() -> dict[tuple[str, int, int], dict[str, Any]]:
    source = json_read(ROOT / "SOURCE_ACCEPTANCE.json")
    return {(x["config"], int(x["horizon_steps"]), int(x["seed"])): x
            for x in source["selected_train_only_cells"]}


def checkpoint_refs() -> list[dict[str, Any]]:
    cells = selected_cell_index()
    refs: list[dict[str, Any]] = []
    def add_epoch(config: str, horizon: int, seed: int, epoch: int) -> None:
        cell = cells[(config, horizon, seed)]
        checkpoint = next(x for x in cell["epoch_checkpoints"] if x["family"] == "corrected" and int(x["epoch"]) == epoch)
        refs.append({
            "label": f"corrected_{config}_H{horizon}_S{seed}_E{epoch}", "family": "corrected",
            "config": config, "horizon": horizon, "seed": seed, "epoch": epoch,
            "run_id": cell["corrected_run_id"],
            "checkpoint": Path(checkpoint["path"]), "checkpoint_sha256": checkpoint["sha256"],
            "prediction_root": SCHED / "predictions/train", "prediction_run_id": cell["corrected_run_id"],
            "prediction_sidecar_sha256": cell["corrected_train_bundle_sidecar_sha256"],
            "accepted_checkpoint_sha256": cell["corrected_checkpoint_sha256"],
            "accepted_checkpoint": SCHED / "runs/formal" / cell["corrected_run_id"] / "checkpoint_last.pt",
            "prediction_sha256": None,
            "accepted_compare": epoch == 40,
        })
    # The old as-run matched anomaly is loaded from the accepted last checkpoint.
    anomaly = cells[("TS11-D", 30, 20261002)]
    refs.append({
        "label": "old_as_run_TS11-D_H30_S20261002_E40", "family": "old_as_run",
        "config": "TS11-D", "horizon": 30, "seed": 20261002, "epoch": 40,
        "run_id": anomaly["old_run_id"],
        "checkpoint": EQ / "d3/runs" / anomaly["old_run_id"] / "checkpoint_last.pt",
        "checkpoint_sha256": anomaly["old_checkpoint_sha256"],
        "prediction_root": DIAG / "predictions/train", "prediction_run_id": anomaly["old_run_id"],
        "prediction_sidecar_sha256": anomaly["old_train_bundle_sidecar_sha256"],
        "accepted_checkpoint_sha256": anomaly["old_checkpoint_sha256"],
        "accepted_checkpoint": EQ / "d3/runs" / anomaly["old_run_id"] / "checkpoint_last.pt",
        "prediction_sha256": None, "accepted_compare": True,
    })
    for epoch in (10, 20, 30, 40):
        add_epoch("TS11-D", 30, 20261002, epoch)
    for config, horizon, seed in (("STATE-CTRL", 30, 20261002), ("TS11-D", 30, 20261001),
                                  ("TS11-D", 30, 20261003), ("TS11-D", 60, 20261002)):
        add_epoch(config, horizon, seed, 40)
    return refs


def verify_run_identity(ref: dict[str, Any]) -> dict[str, Any]:
    path: Path = ref["checkpoint"]
    actual_hash = sha256_file(path)
    if actual_hash != ref["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint hash mismatch: {ref['label']} {actual_hash}")
    # Local accepted checkpoints contain NumPy RNG state; safe loading therefore uses
    # weights_only=False after the file's SHA-256 has matched its accepted manifest.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("config", ref["config"]), ("horizon", ref["horizon"]), ("seed", ref["seed"])):
        if payload.get(key) != expected:
            raise AssertionError(f"checkpoint {key} identity mismatch for {ref['label']}")
    if ref["epoch"] in (10, 20, 30, 40) and payload.get("epoch") != ref["epoch"]:
        raise AssertionError(f"checkpoint epoch identity mismatch for {ref['label']}")
    accepted_checkpoint = Path(ref["accepted_checkpoint"])
    accepted_checkpoint_hash = sha256_file(accepted_checkpoint)
    if accepted_checkpoint_hash != ref["accepted_checkpoint_sha256"]:
        raise AssertionError(f"accepted last checkpoint hash mismatch: {ref['label']}")
    if ref["accepted_compare"]:
        accepted_payload = torch.load(accepted_checkpoint, map_location="cpu", weights_only=False)
        def hash_state_dict(state: dict[str, torch.Tensor]) -> str:
            digest = hashlib.sha256()
            for name, value in sorted(state.items()):
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().numpy().tobytes())
            return digest.hexdigest()
        if hash_state_dict(payload["model"]) != hash_state_dict(accepted_payload["model"]):
            raise AssertionError(f"selected checkpoint weights differ from accepted last checkpoint: {ref['label']}")
    cell = selected_cell_index()[(ref["config"], ref["horizon"], ref["seed"])]
    # Old H30/H60 and corrected final weights must remain tied to the frozen matrix row.
    if ref["family"] == "old_as_run":
        matrix_rows = list(csv.DictReader((SCHED / "protocol/MATRIX_12.csv").open(newline="", encoding="utf-8")))
        row = next(x for x in matrix_rows if x["run_id"] == f"D3EQ-CS-{ref['config']}-H{ref['horizon']}-S{ref['seed']}")
        if row["matched_old_run_id"] != ref["run_id"] or row["old_checkpoint_sha256"] != actual_hash:
            raise AssertionError("old checkpoint does not match corrected-scheduler matrix binding")
    sidecar_path = ref["prediction_root"] / f"{ref['prediction_run_id']}.json"
    pred_path = ref["prediction_root"] / f"{ref['prediction_run_id']}.npz"
    if sha256_file(sidecar_path) != ref["prediction_sidecar_sha256"]:
        raise AssertionError(f"accepted prediction sidecar hash mismatch: {ref['label']}")
    sidecar = json_read(sidecar_path)
    if (sidecar.get("status") != "PASS" or int(sidecar.get("rows", -1)) != 17757
            or sidecar.get("split_used") != "primary_train_only" or sidecar.get("target_arrays_loaded") is not False
            or sidecar.get("checkpoint_sha256") != ref["accepted_checkpoint_sha256"]):
        raise AssertionError(f"accepted train prediction sidecar identity/status failure: {ref['label']}")
    pred_hash = sha256_file(pred_path)
    if sidecar.get("prediction_sha256") != pred_hash:
        raise AssertionError(f"accepted prediction bundle hash mismatch: {ref['label']}")
    if ref["accepted_compare"]:
        ref["prediction_sha256"] = pred_hash
    return {"payload": payload, "sidecar": sidecar, "prediction_path": pred_path, "prediction_sha256": pred_hash}


def disable_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            raise AssertionError("dropout module did not disable")
        if isinstance(module, nn.MultiheadAttention) and module.dropout != 0.0:
            raise AssertionError("attention dropout did not disable")


def module_group(name: str) -> str:
    if name.startswith("output."):
        return "output_head"
    if name.startswith("decoder."):
        return "decoder"
    if name.startswith("tcn_blocks."):
        return "state_tcn"
    if name.startswith("video_projection."):
        return "video_projection"
    if name.startswith("state_projection."):
        return "state_projection"
    if name.startswith("anchor_projection."):
        return "anchor_projection"
    return "other"


def parameter_trace(model: nn.Module, init_state: dict[str, torch.Tensor], ref: dict[str, Any]) -> list[dict[str, Any]]:
    group_sums: dict[str, list[float]] = defaultdict(list)
    init_sums: dict[str, list[float]] = defaultdict(list)
    current_sums: dict[str, list[float]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu().float()
        initial = init_state[name].detach().cpu().float()
        group = module_group(name)
        group_sums[group].append(float(torch.sum((current - initial) ** 2).item()))
        init_sums[group].append(float(torch.sum(initial ** 2).item()))
        current_sums[group].append(float(torch.sum(current ** 2).item()))
    rows = []
    for group in sorted(group_sums):
        delta = math.sqrt(sum(group_sums[group]))
        initial = math.sqrt(sum(init_sums[group]))
        current = math.sqrt(sum(current_sums[group]))
        rows.append({
            "row_type": "parameter", "model": ref["label"], "module": group,
            "parameter_delta_l2_from_INIT": delta, "INIT_parameter_l2": initial,
            "current_parameter_l2": current,
            "relative_delta_l2_from_INIT": delta / initial if initial else "NA",
            "output_weight_l2": float(model.output.weight.detach().norm().item()),
            "output_bias_l2": float(model.output.bias.detach().norm().item()),
        })
    return rows


def query_trace(horizon: int, model_label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    times = torch.arange(1, horizon + 1, dtype=torch.float32) / 30.0
    query = sinusoidal(times, 128).detach().cpu().numpy().astype(np.float32)
    q64 = query.astype(np.float64)
    norms = np.linalg.norm(q64, axis=1)
    distances = np.linalg.norm(np.diff(q64, axis=0), axis=1)
    cosine = np.sum(q64[:-1] * q64[1:], axis=1) / (norms[:-1] * norms[1:])
    singular = np.linalg.svd(q64, compute_uv=False)
    rank_cut = float(singular[0] * np.finfo(np.float32).eps * max(q64.shape))
    effective_rank = int(np.sum(singular > rank_cut))
    rows = [{
        "row_type": "query", "model": model_label, "query_index_1based": i + 1,
        "time_seconds": float(times[i]), "query_norm": float(norms[i]),
        "adjacent_l2": "NA" if i == 0 else float(distances[i - 1]),
        "adjacent_cosine": "NA" if i == 0 else float(cosine[i - 1]),
        "spectrum_index": "NA", "singular_value": "NA",
        "decoder_tgt_mask": "None (default causal mask disabled)",
        "query_sha256": sha_array(query), "effective_rank_fp32_scale": effective_rank,
    } for i in range(horizon)]
    spectrum = {
        "horizon": horizon, "query_shape": list(q64.shape),
        "unique_query_rows": int(np.unique(query, axis=0).shape[0]),
        "adjacent_l2_min": float(distances.min()), "adjacent_l2_max": float(distances.max()),
        "adjacent_cosine_min": float(cosine.min()), "adjacent_cosine_max": float(cosine.max()),
        "singular_values_float64_svd_of_fp32_query": singular.tolist(),
        "fp32_scale_rank_threshold": rank_cut, "fp32_scale_effective_rank": effective_rank,
        "decoder_tgt_mask": None,
    }
    return rows, spectrum


def gradient_probe(model: nn.Module, tensors: dict[str, torch.Tensor], ref: dict[str, Any]) -> list[dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    before = state_hash(model)
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("unexpected accumulated parameter.grad before local probe")
    pred = model(tensors["state"], tensors["mask"], tensors["relative"], tensors["dynamic"], tensors["static"], tensors["anchor"])
    components = loss_components(pred, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
    locked = per_sample_loss(pred, tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
    if not torch.allclose(components["full"], locked, atol=1e-7, rtol=0):
        raise AssertionError("local loss decomposition differs from locked D3-EQ contract")
    objectives = [components["position"].mean(), components["weighted_fd"].mean(), components["full"].mean()]
    params = tuple(p for p in model.parameters() if p.requires_grad)
    grads = []
    for i, objective in enumerate(objectives):
        grads.append(torch.autograd.grad(objective, params, retain_graph=i < len(objectives) - 1,
                                         allow_unused=True, create_graph=False))
    if state_hash(model) != before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("autograd.grad changed model state or accumulated .grad")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    by_objective: list[dict[str, dict[str, torch.Tensor]]] = []
    for grad_tuple in grads:
        mapped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for name, grad in zip(names, grad_tuple):
            mapped[module_group(name)][name] = torch.zeros_like(dict(model.named_parameters())[name]) if grad is None else grad.detach()
        by_objective.append(mapped)
    rows: list[dict[str, Any]] = []
    modules = sorted(set().union(*(x.keys() for x in by_objective)))
    for group in ["all", *modules]:
        per = []
        for mapped in by_objective:
            selected = [g for m, values in mapped.items() if group == "all" or m == group for g in values.values()]
            per.append(selected)
        vectors = [torch.cat([g.reshape(-1) for g in values]) if values else torch.zeros(0) for values in per]
        norms = [float(torch.linalg.vector_norm(v).item()) for v in vectors]
        pnorm, fdnorm, tnorm = norms
        denom = pnorm * fdnorm
        cosine = float(torch.dot(vectors[0], vectors[1]).item() / denom) if denom > 0 else "NA"
        rows.append({
            "model": ref["label"], "module": group,
            "L_pos": float(objectives[0].detach().item()),
            "weighted_0p1_L_delta": float(objectives[1].detach().item()),
            "L_total": float(objectives[2].detach().item()),
            "g_pos_l2": pnorm, "g_0p1fd_l2": fdnorm, "g_total_l2": tnorm,
            "g_fd_over_g_pos": fdnorm / pnorm if pnorm > 0 else "NA",
            "g_pos_fd_cosine": cosine,
            "gradient_none_count": int(sum(g is None for tup in grads[:2] for g in tup)),
            "optimizer_created": False, "backward_called": False,
            "parameter_hash_before_after_equal": True,
        })
    return rows


def fixture_tensors(bundle: dict[str, Any], fixture: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    positions = np.asarray([int(row["train_position"]) for row in fixture["rows_selected_by_locked_rule"]], dtype=np.int64)
    rows = bundle["train_indices"][positions]
    dynamic, static = bundle["grid_store"].get(rows)
    dynamic = ((dynamic - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
    static = ((static - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
    tensors = {
        "state": torch.from_numpy(bundle["inputs"]["history_standardized"][positions].astype(np.float32, copy=False)),
        "mask": torch.from_numpy(bundle["inputs"]["history_mask"][positions].astype(np.float32, copy=False)),
        "relative": torch.from_numpy(bundle["inputs"]["history_relative_delta_standardized"][positions].astype(np.float32, copy=False)),
        "dynamic": torch.from_numpy(dynamic), "static": torch.from_numpy(static),
        "anchor": torch.from_numpy(bundle["anchors"][positions].astype(np.float32, copy=False)),
        "target": torch.from_numpy(bundle["targets60"][positions, :30].astype(np.float32, copy=False)),
        "target60": torch.from_numpy(bundle["targets60"][positions].astype(np.float32, copy=False)),
        "sigma": bundle["target_sigma"].float(), "safe_fd": bundle["safe_fd_scale"].float(),
    }
    metadata = {"positions": positions, "source_rows": rows,
                "ids": bundle["train_ids"][positions], "targets": bundle["targets60"][positions, :30],
                "targets60": bundle["targets60"][positions], "anchors": bundle["anchors"][positions]}
    return tensors, metadata


def accepted_rows(ref: dict[str, Any], path: Path, ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        bundle_ids = archive["anchor_id"].astype("U96")
        if "split" not in archive.files or str(archive["split"].item()) != "train":
            raise AssertionError(f"accepted prediction is not train-only: {ref['label']}")
        values = archive["predictions_native"]
        if len(bundle_ids) != 17757 or len(np.unique(bundle_ids)) != 17757:
            raise AssertionError(f"accepted train prediction row count/ID uniqueness failure: {ref['label']}")
        lookup = {value: i for i, value in enumerate(bundle_ids.tolist())}
        if not set(ids.tolist()).issubset(lookup):
            raise AssertionError(f"fixture ID absent from accepted bundle: {ref['label']}")
        return np.asarray(values[[lookup[str(x)] for x in ids]], dtype=np.float32).copy()


def run_replay(replicate: str) -> Path:
    source_lock_hash = verify_phase_a_source_lock()
    if not FIXTURE_PATH.is_file():
        raise FileNotFoundError("TRAIN_FORENSIC_FIXTURE.json must be locked before first forward")
    fixture = json_read(FIXTURE_PATH)
    if fixture.get("status") != "LOCKED_BEFORE_FORWARD" or fixture.get("forward_executed_before_lock") is not False:
        raise AssertionError("fixture lock state invalid")
    if fixture.get("phase_a_lock_sha256") != sha256_file(LOCK_PATH):
        raise AssertionError("Phase A execution lock changed since fixture selection")
    if fixture.get("phase_a_source_lock_sha256") != source_lock_hash:
        raise AssertionError("Phase A implementation lock changed since fixture selection")
    if fixture.get("phase_a_amendment_sha256") != sha256_file(AMENDMENT_PATH):
        raise AssertionError("Phase A horizon amendment changed since fixture selection")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bundle = load_train_bundle()
    tensors, metadata = fixture_tensors(bundle, fixture)
    out = OUT / replicate
    out.mkdir(parents=True, exist_ok=False)
    source = source_hashes()
    init_path = BENCH / "d3/runs/INIT_S20261002.pt"
    if sha256_file(init_path) != json_read(ROOT / "SOURCE_ACCEPTANCE.json")["seed_20261002_init"]["sha256"]:
        raise AssertionError("seed-20261002 INIT file hash mismatch")
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    model_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    query_rows_all: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    query_spectra: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    raw_records: list[dict[str, Any]] = []
    locked_tol = json_read(LOCK_PATH)["accepted_prediction_replay_tolerance"]
    refs = checkpoint_refs()
    for ref in refs:
        identity = verify_run_identity(ref)
        payload = identity["payload"]
        model = D3TimeSpacePredictor(horizon=ref["horizon"], seed=ref["seed"], video_mode=ref["config"])
        model.load_state_dict(payload["model"], strict=True)
        disable_dropout(model)
        before_hash = state_hash(model)
        model_tensors = tensors if ref["horizon"] == 30 else {**tensors, "target": tensors["target60"]}
        with torch.inference_mode():
            pred_residual = model(tensors["state"], tensors["mask"], tensors["relative"],
                                  tensors["dynamic"], tensors["static"], tensors["anchor"])
            pred_native = tensors["anchor"][:, None, :] + pred_residual * tensors["sigma"][None, None, :]
        pred_native_np = pred_native.detach().cpu().numpy().astype(np.float32, copy=True)
        residual_np = pred_residual.detach().cpu().numpy().astype(np.float32, copy=True)
        decomposition = trajectory_decomposition(pred_native, tensors["anchor"], tensors["sigma"])
        comp = loss_components(pred_residual, model_tensors["target"], tensors["anchor"], tensors["sigma"], tensors["safe_fd"])
        pos_by_row = comp["position"].detach().cpu().numpy()
        fd_by_row = comp["fd"].detach().cpu().numpy()
        for i, anchor_id in enumerate(metadata["ids"].tolist()):
            offset = pred_native_np[i] - metadata["anchors"][i][None, :]
            target = metadata["targets60"][i, :ref["horizon"]]
            per_pos_native = np.abs(pred_native_np[i] - target).mean()
            per_fd = np.abs(np.diff(pred_native_np[i], axis=0) - np.diff(target, axis=0)).mean()
            record = {
                "model": ref["label"], "anchor_id": anchor_id,
                "config": ref["config"], "horizon": ref["horizon"], "seed": ref["seed"],
                "family": ref["family"], "checkpoint_epoch": ref["epoch"],
                "position_error_normalized": float(pos_by_row[i]),
                "fd_error_safe_scale_normalized": float(fd_by_row[i]),
                "native_position_mae": float(per_pos_native), "native_internal_fd_mae": float(per_fd),
                "pred_future_tv_native": float(np.abs(np.diff(pred_native_np[i], axis=0)).mean()),
                "pred_future_tv_normalized": float(np.abs(np.diff(pred_native_np[i], axis=0) / bundle["safe_fd_scale"].numpy()[None, :]).mean()),
                "offset_time_std_native": float(offset.std(axis=0).mean()),
                "offset_time_range_native": float((offset.max(axis=0) - offset.min(axis=0)).mean()),
                "offset_time_mean_native_abs": float(np.abs(offset.mean(axis=0)).mean()),
                "anchor_copy_max_abs": float(np.abs(offset).max()),
            }
            raw_records.append(record)
        accepted_match = None
        if ref["accepted_compare"]:
            accepted_match = accepted_rows(ref, identity["prediction_path"], metadata["ids"])
            delta = np.abs(pred_native_np - accepted_match)
            within = np.allclose(pred_native_np, accepted_match, atol=float(locked_tol["atol"]),
                                 rtol=float(locked_tol["rtol"]))
            if not within:
                raise AssertionError(f"raw forward differs from accepted train bundle: {ref['label']} max_abs={float(delta.max())}")
            ref["accepted_max_abs_difference"] = float(delta.max())
            ref["accepted_mean_abs_difference"] = float(delta.mean())
            ref["accepted_within_locked_tolerance"] = True
        outputs[ref["label"]] = pred_native_np
        model_rows.append({
            **{k: ref[k] for k in ("label", "family", "config", "horizon", "seed", "epoch", "run_id")},
            "checkpoint_sha256": ref["checkpoint_sha256"],
            "prediction_sidecar_sha256": ref["prediction_sidecar_sha256"],
            "prediction_sha256": identity["prediction_sha256"],
            "accepted_forward_compared": ref["accepted_compare"],
            "accepted_max_abs_difference": ref.get("accepted_max_abs_difference", "NA"),
            "accepted_mean_abs_difference": ref.get("accepted_mean_abs_difference", "NA"),
            "accepted_within_locked_tolerance": ref.get("accepted_within_locked_tolerance", "NA"),
            **decomposition,
            "position_error_normalized_mean": float(np.mean(pos_by_row)),
            "fd_error_safe_scale_normalized_mean": float(np.mean(fd_by_row)),
            "prediction_native_sha256": sha_array(pred_native_np),
            "prediction_residual_sha256": sha_array(residual_np),
            "checkpoint_state_hash_after_forward": state_hash(model),
            "state_hash_unchanged_by_forward": before_hash == state_hash(model),
        })
        if state_hash(model) != before_hash:
            raise AssertionError(f"forward mutated weights: {ref['label']}")
        parameter_rows.extend(parameter_trace(model, init_state, ref))
        qrows, qs = query_trace(ref["horizon"], ref["label"])
        query_rows_all.extend(qrows)
        query_spectra[ref["label"]] = qs
        gradient_rows.extend(gradient_probe(model, model_tensors, ref))
        with (out / f"PREDICTION__{ref['label']}.npz").open("wb") as model_output:
            np.savez_compressed(model_output, anchor_id=metadata["ids"].astype("U96"),
                                target_native=metadata["targets60"][:, :ref["horizon"]].astype(np.float32),
                                anchor_native=metadata["anchors"].astype(np.float32),
                                prediction_native=pred_native_np, residual_standardized=residual_np)
        write_csv(out / "MODEL_REPLAY_PARTIAL.csv", model_rows)
        write_csv(out / "SAMPLE_METRICS_PARTIAL.csv", raw_records)
        write_csv(out / "PARAMETER_AND_QUERY_TRACE_PARTIAL.csv", parameter_rows + query_rows_all)
        write_csv(out / "LOSS_GRADIENT_COMPONENTS_PARTIAL.csv", gradient_rows)
        write_json(out / "QUERY_SPECTRA_PARTIAL.json", query_spectra)
        write_json(out / "PROGRESS.json", {"status": "RUNNING", "replicate": replicate,
                                            "models_completed": len(model_rows),
                                            "last_completed_model": ref["label"],
                                            "optimizer_updates": 0, "dev_targets_opened": False,
                                            "final_or_heldout_access": 0})
        model.zero_grad(set_to_none=True)
        del model, payload, identity

    with (out / "PREDICTIONS_FIXTURE.npz").open("wb") as handle:
        np.savez_compressed(
            handle, anchor_id=metadata["ids"].astype("U96"),
            target_native_h30=metadata["targets"].astype(np.float32),
            target_native_h60=metadata["targets60"].astype(np.float32),
            anchor_native=metadata["anchors"].astype(np.float32),
            **{f"prediction_native__{name}": value for name, value in outputs.items()},
        )
    write_csv(out / "SAMPLE_METRICS.csv", raw_records)
    trace_rows = parameter_rows + query_rows_all
    for label, spectrum in query_spectra.items():
        for i, value in enumerate(spectrum["singular_values_float64_svd_of_fp32_query"]):
            trace_rows.append({"row_type": "query_singular_value", "model": label,
                               "spectrum_index": i + 1, "singular_value": value,
                               "effective_rank_fp32_scale": spectrum["fp32_scale_effective_rank"]})
    write_csv(out / "PARAMETER_AND_QUERY_TRACE.csv", trace_rows)
    write_csv(out / "LOSS_GRADIENT_COMPONENTS.csv", gradient_rows)
    write_csv(out / "MODEL_REPLAY.csv", model_rows)
    report = {
        "schema_version": 1, "status": "PASS", "replicate": replicate,
        "phase": "A_read_only_train_only", "device": "cpu", "torch_threads": 1,
        "optimizer_created": False, "optimizer_updates": 0, "backward_called": False,
        "dev_targets_opened": False, "final_or_heldout_access": 0,
        "fixture_sha256": sha256_file(FIXTURE_PATH), "phase_a_lock_sha256": sha256_file(LOCK_PATH),
        "source_hashes": source,
        "models": model_rows, "query_spectra": query_spectra,
        "prediction_fixture_sha256": sha256_file(out / "PREDICTIONS_FIXTURE.npz"),
        "sample_metrics_sha256": sha256_file(out / "SAMPLE_METRICS.csv"),
        "parameter_query_trace_sha256": sha256_file(out / "PARAMETER_AND_QUERY_TRACE.csv"),
        "gradient_components_sha256": sha256_file(out / "LOSS_GRADIENT_COMPONENTS.csv"),
    }
    write_json(out / "A_REPLAY.json", report)
    return out


def compare_replicates(first: str, second: str) -> dict[str, Any]:
    path_a, path_b = OUT / first, OUT / second
    a, b = json_read(path_a / "A_REPLAY.json"), json_read(path_b / "A_REPLAY.json")
    with np.load(path_a / "PREDICTIONS_FIXTURE.npz", allow_pickle=False) as za, np.load(path_b / "PREDICTIONS_FIXTURE.npz", allow_pickle=False) as zb:
        common = sorted(set(za.files) & set(zb.files))
        if set(za.files) != set(zb.files):
            raise AssertionError("replay output array key sets differ")
        array_equal = all(np.array_equal(za[key], zb[key]) for key in common)
        max_abs = max(float(np.max(np.abs(za[key] - zb[key]))) for key in common if np.issubdtype(za[key].dtype, np.number))
    tables_equal = all((path_a / name).read_bytes() == (path_b / name).read_bytes()
                       for name in ("SAMPLE_METRICS.csv", "PARAMETER_AND_QUERY_TRACE.csv", "LOSS_GRADIENT_COMPONENTS.csv", "MODEL_REPLAY.csv"))
    if not array_equal or not tables_equal:
        raise AssertionError(f"replicate replay is not bitwise stable (array_equal={array_equal}, tables_equal={tables_equal}, max_abs={max_abs})")
    result = {
        "schema_version": 1, "status": "PASS", "replicates": [first, second],
        "prediction_arrays_bitwise_equal": array_equal, "metric_and_trace_csv_bitwise_equal": tables_equal,
        "max_abs_difference": max_abs, "fixture_sha256": a["fixture_sha256"],
        "accepted_prediction_replay_tolerance": json_read(LOCK_PATH)["accepted_prediction_replay_tolerance"],
        "repeat_tolerance": json_read(LOCK_PATH)["repeat_run_tolerance"],
    }
    write_json(ROOT / "forensics/REPLAY_REPEATABILITY.json", result)
    return result


def main() -> int:
    required = ("EE6008_PROJECT_ROOT", "EE6008_COLLAPSE_STUDY_ROOT", "EE6008_D3_EQ_STUDY_ROOT", "EE6008_D3_SCHEDULER_STUDY_ROOT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("set explicit server-artifact roots: " + ", ".join(missing))
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock-fixture", action="store_true")
    group.add_argument("--replay", metavar="REPLICATE")
    group.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    if args.lock_fixture:
        result = lock_fixture()
        print(json.dumps({"status": result["status"], "rows": result["rows"],
                          "strata": result["selected_strata_counts"], "fixture_sha256": sha256_file(FIXTURE_PATH)}))
    elif args.replay:
        path = run_replay(args.replay)
        print(json.dumps({"status": "PASS", "replicate": args.replay, "out": str(path),
                          "report_sha256": sha256_file(path / "A_REPLAY.json")}))
    else:
        print(json.dumps(compare_replicates(*args.compare), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
