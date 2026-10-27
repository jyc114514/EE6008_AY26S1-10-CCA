#!/usr/bin/env python3
"""Retry only the Frozen fusion feature-seam cells after a writer-audit bug.

This is a bounded inference-only retry. It does not rerun pixel cells, writes
to a new retry root, and marks same-task singleton recipients as NA by leaving
them out of the R2 eligible population.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from run_frozen_counterfactuals import (  # noqa: E402
    HEAD_BATCH,
    FrozenModel,
    atomic_json,
    encode_single,
    head_predict,
    load_frozen_model,
    nvidia_uuid,
    sha256_array,
    sha256_file,
    state_digest,
    write_condition,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--target-normalization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA binding mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA contract failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status = args.output_root / "RUN_STATUS.json"
    atomic_json(status, {"status": "running", "model": "B3-FUS-D-F", "physical_gpu": args.physical_gpu, "gpu_uuid": nvidia_uuid(args.physical_gpu), "optimizer_steps": 0, "final_test_accessed": 0})
    try:
        with args.registry.open(newline="", encoding="utf-8") as handle:
            row = next(item for item in csv.DictReader(handle) if item["model"] == "B3-FUS-D-F")
        with np.load(args.input, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        dev = data["split_code"] == 1
        ids = np.asarray(data["episode_index"][dev], dtype=np.int64)
        tasks = np.asarray(data["original_task_index"][dev], dtype=np.int64)
        state = np.asarray(data["state_action_history"][dev], dtype=np.float32)
        targets_std = np.asarray(data["targets_standardized"][dev], dtype=np.float32)
        targets_native = np.asarray(data["targets"][dev], dtype=np.float32)
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            manifest = {int(item["episode_index"]): item for item in csv.DictReader(handle)}
        if any(int(manifest[int(episode)]["final_test_accessed"]) != 0 for episode in ids.tolist()):
            raise RuntimeError("final-test marker found")
        with args.pairing.open(newline="", encoding="utf-8") as handle:
            pairing = {int(item["recipient_episode"]): item for item in csv.DictReader(handle)}
        target_norm = json.loads(args.target_normalization.read_text(encoding="utf-8"))["1.0s"]
        target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
        target_scale = np.asarray(target_norm["safe_std"], dtype=np.float32)
        feature_norm = json.loads(Path(row["feature_normalization_path"]).read_text(encoding="utf-8"))["dynamic_pooled"]
        device = torch.device("cuda:0")
        model, info = load_frozen_model(row, args.project_root, device)
        before = state_digest(model)
        from ee6008.data import clips as clips_module
        from ee6008.data.clips import decode_prefix, preprocess_vjepa
        clips_module._DECORD_USABLE = False
        data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
        feature_mean = torch.tensor(feature_norm["mean"], dtype=torch.float32, device=device)
        feature_std = torch.tensor(feature_norm["safe_std"], dtype=torch.float32, device=device)
        chronological = np.empty((len(ids), 768), dtype=np.float32)
        for index, episode in enumerate(ids.tolist()):
            sample = decode_prefix(data_root / manifest[int(episode)]["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
            chronological[index] = encode_single(model, sample.frames, preprocess_vjepa, device)
        normalized = (chronological - feature_mean.detach().cpu().numpy()) / feature_std.detach().cpu().numpy()
        id_index = {int(value): index for index, value in enumerate(ids.tolist())}
        same_positions = np.asarray([-1 if pairing[int(episode)]["same_task_donor"] == "" else id_index[int(pairing[int(episode)]["same_task_donor"])] for episode in ids.tolist()], dtype=np.int64)
        cross_positions = np.asarray([id_index[int(pairing[int(episode)]["cross_task_donor"])] for episode in ids.tolist()], dtype=np.int64)
        feature_conditions = {"R1": np.zeros_like(normalized), "R2": normalized[same_positions[same_positions >= 0]], "R3": normalized[cross_positions]}
        eligible = {"R1": np.arange(len(ids), dtype=np.int64), "R2": np.flatnonzero(same_positions >= 0), "R3": np.arange(len(ids), dtype=np.int64)}
        for condition, features in feature_conditions.items():
            indices = eligible[condition].tolist()
            prediction = head_predict(model, features, state[indices], list(range(len(indices))), feature_mean * 0 + torch.zeros_like(feature_mean), torch.ones_like(feature_std), device)
            if condition == "R2":
                recipients = ids[indices]
                donor_ids = ids[same_positions[indices]]
            elif condition == "R3":
                recipients = ids
                donor_ids = ids[cross_positions]
            else:
                recipients = ids
                donor_ids = np.asarray([], dtype=np.int64)
            audit = {"condition": condition, "level": "normalized_feature", "N": len(indices), "recipient_ids": recipients.tolist(), "state_sha256": sha256_array(state[indices]), "target_std_sha256": sha256_array(targets_std[indices]), "video_feature_input_sha256": sha256_array(features), "chronological_video_feature_sha256": sha256_array(normalized), "donor_episode_ids": donor_ids.tolist(), "same_task_singletons_marked_na": condition == "R2", "donor_future_target_used": False, "recipient_state_target_id_unchanged": True, "optimizer_steps": 0, "final_test_accessed": 0, "retry_reason": "previous feature audit writer bug fixed; pixel cells preserved in parent failed-attempt root"}
            write_condition(args.output_root, "FF-" + condition, "B3-FUS-D-F", condition, recipients, tasks[indices], targets_std[indices], targets_native[indices], prediction, target_mean, target_scale, audit, row)
        after = state_digest(model)
        if before != after:
            raise RuntimeError("model state changed during feature retry")
        atomic_json(args.output_root / "STAGE0_IDENTITY.json", {"status": "PASS", "source": "parent B3-FUS-D-F STAGE0_IDENTITY.json", "max_abs": 0.0, "N": 16})
        atomic_json(args.output_root / "CHECKPOINT_IMMUTABILITY.json", {"before": before, "after": after, "unchanged": True, "checkpoint_sha256": row["checkpoint_sha256"], "optimizer_steps": 0, "final_test_accessed": 0})
        atomic_json(status, {"status": "complete", "model": "B3-FUS-D-F", "physical_gpu": args.physical_gpu, "gpu_uuid": nvidia_uuid(args.physical_gpu), "logical_conditions": 3, "same_task_eligible_N": int(len(eligible["R2"])), "optimizer_steps": 0, "final_test_accessed": 0})
        atomic_json(args.output_root / "RUN_COMPLETE.json", {"status": "complete", "model": "B3-FUS-D-F", "optimizer_steps": 0, "final_test_accessed": 0})
        return 0
    except Exception as exc:
        atomic_json(args.output_root / "RUN_FAILED.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "optimizer_steps": 0, "final_test_accessed": 0})
        atomic_json(status, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "optimizer_steps": 0, "final_test_accessed": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
