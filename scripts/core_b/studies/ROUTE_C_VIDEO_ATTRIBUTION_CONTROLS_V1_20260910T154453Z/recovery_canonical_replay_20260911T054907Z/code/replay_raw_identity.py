#!/usr/bin/env python3
"""Recovery-only raw chronological identity replay with canonical precision/batching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{__import__('os').getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("route_runner_for_recovery", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_indices(ids: np.ndarray, stage0: Path, gate: str, one_id: int) -> list[int]:
    if gate == "one":
        wanted = [one_id]
    elif gate == "sixteen":
        with stage0.open(newline="", encoding="utf-8") as handle:
            wanted = [int(row["episode_id"]) for row in csv.DictReader(handle)]
    else:
        wanted = [int(value) for value in ids.tolist()]
    index = {int(value): position for position, value in enumerate(ids.tolist())}
    missing = [value for value in wanted if value not in index]
    if missing:
        raise ValueError(f"selected IDs missing from development population: {missing}")
    return [index[value] for value in wanted]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("B3-FUS-D-F", "B3-FUS-D-L", "B3-V-D-F", "B3-V-D-L"), required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--route-runner", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate", choices=("one", "sixteen", "full"), required=True)
    parser.add_argument("--one-episode-id", type=int, default=376)
    parser.add_argument("--encoder-batch-size", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "running", "gate": args.gate, "model": args.model, "optimizer_steps": 0, "final_test_accessed": 0, "started_utc": utc_now()})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if args.encoder_batch_size <= 0 or args.head_batch_size <= 0:
            raise ValueError("batch sizes must be positive")

        runner = load_module(args.route_runner.resolve())
        data, manifest, _tasks = runner.load_data()
        dev = data["split_code"] == 1
        ids = np.asarray(data["episode_index"][dev], dtype=np.int64)
        state = np.asarray(data["state_action_history"][dev], dtype=np.float32)
        target_ids = np.asarray(ids, dtype=np.int64)
        indices = selected_indices(ids, args.route_root / "ROUTE_C_STAGE0_EPISODES.csv", args.gate, args.one_episode_id)
        context_indices: set[int] = set()
        for selected_index in indices:
            batch_start = (selected_index // args.head_batch_size) * args.head_batch_size
            batch_end = min(batch_start + args.head_batch_size, len(ids))
            context_indices.update(range(batch_start, batch_end))
        compute_indices = sorted(context_indices)

        branch = "fusion" if args.model.startswith("B3-FUS") else "video_only"
        adaptation = "lora" if args.model.endswith("-L") else "frozen"
        registry = next(row for row in csv.DictReader((args.route_root / "ROUTE_C_CHECKPOINT_REGISTRY.csv").open(newline="", encoding="utf-8")) if row["model"] == args.model)
        model_args = SimpleNamespace(adaptation=adaptation, branch=branch, checkpoint_path=Path(registry["checkpoint_path"]))
        device = torch.device("cuda:0")
        model, model_info = runner.load_models(model_args, device)
        model_info["use_bf16_autocast"] = True
        before = {selector: runner.state_digest(model, selector) for selector in ("base", "adapter", "head")}

        from ee6008.data import clips as clips_module
        from ee6008.data.clips import decode_prefix, preprocess_vjepa
        clips_module._DECORD_USABLE = False
        feature_norm = json.loads(runner.FEATURE_NORM.read_text(encoding="utf-8"))["dynamic_pooled"]
        feature_mean = torch.tensor(feature_norm["mean"], dtype=torch.float32, device=device)
        feature_std = torch.tensor(feature_norm["safe_std"], dtype=torch.float32, device=device)
        raw_features = np.empty((len(compute_indices), 768), dtype=np.float32)

        for start in range(0, len(compute_indices), args.encoder_batch_size):
            chunk = compute_indices[start : start + args.encoder_batch_size]
            clips = []
            for index in chunk:
                row = manifest[int(ids[index])]
                sample = decode_prefix(runner.DATA_ROOT / row["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
                clip = preprocess_vjepa(sample.frames, resolution=384)
                if tuple(clip.shape) != (3, 32, 384, 384):
                    raise RuntimeError(f"unexpected clip shape for {ids[index]}: {tuple(clip.shape)}")
                clips.append(clip)
            clip_batch = torch.stack(clips).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if adaptation == "frozen":
                    raw = model.encoder(clip_batch).mean(dim=1).float()
                else:
                    raw = runner.encode(model, clip_batch, use_bf16=True)
            raw_features[start : start + len(chunk)] = raw.float().detach().cpu().numpy()

        prediction_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(compute_indices), args.head_batch_size):
                chunk = compute_indices[start : start + args.head_batch_size]
                feature_part = torch.tensor(raw_features[start : start + len(chunk)], device=device)
                state_part = torch.tensor(state[chunk], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    value = runner.head_forward(model, feature_part, state_part, feature_mean, feature_std).float().cpu().numpy()
                prediction_parts.append(value)
        candidate = np.concatenate(prediction_parts, axis=0)

        raw_feature_ledger = None
        if adaptation == "frozen":
            feature_manifest = pd.read_parquet(args.feature_root / "feature_manifest.parquet")
            feature_rows = {int(row.episode_index): Path(str(row.feature_path)) for row in feature_manifest.itertuples(index=False)}
            raw_feature_ledger = []
            for position, index in enumerate(compute_indices):
                episode = int(ids[index])
                record = torch.load(feature_rows[episode], map_location="cpu", weights_only=True)
                stored = record["pooled"].float().numpy()
                delta = raw_features[position] - stored
                raw_feature_ledger.append({"episode_id": episode, "max_abs": float(np.max(np.abs(delta))), "mean_abs": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(np.square(delta))))})

        canonical = np.load(Path(registry["canonical_prediction_path"]), allow_pickle=False)
        canonical_index = {int(value): position for position, value in enumerate(canonical["episode_index"].tolist())}
        compute_position = {index: position for position, index in enumerate(compute_indices)}
        per_episode = []
        max_abs = 0.0
        all_abs: list[np.ndarray] = []
        for index in indices:
            local = compute_position[index]
            episode = int(ids[index])
            expected = np.asarray(canonical["predictions_standardized"][canonical_index[episode]], dtype=np.float32)
            delta = candidate[local] - expected
            absolute = np.abs(delta)
            max_abs = max(max_abs, float(np.max(absolute)))
            all_abs.append(absolute.reshape(-1))
            per_episode.append({"episode_id": episode, "max_abs": float(np.max(absolute)), "mean_abs": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(np.square(delta))))})
        flattened = np.concatenate(all_abs)
        after = {selector: runner.state_digest(model, selector) for selector in ("base", "adapter", "head")}
        if before != after:
            raise RuntimeError("model/checkpoint state changed during recovery identity")
        tolerance = 1e-5 + 1e-5 * max(1.0, float(np.max(np.abs(canonical["predictions_standardized"]))))
        result = {
            "status": "pass" if max_abs <= tolerance else "fail",
            "gate": args.gate,
            "model": args.model,
            "branch": branch,
            "adaptation": adaptation,
            "selected_N": len(indices),
            "selected_episode_ids": [int(ids[index]) for index in indices],
            "compute_context_N": len(compute_indices),
            "compute_context_episode_ids": [int(ids[index]) for index in compute_indices],
            "raw_feature_vs_cache": raw_feature_ledger if raw_feature_ledger is not None else "not_applicable_for_lora_encoder",
            "max_abs": max_abs,
            "mean_abs": float(np.mean(flattened)),
            "rmse": float(np.sqrt(np.mean(np.square(flattened)))),
            "locked_atol": 1e-5,
            "locked_rtol": 1e-5,
            "effective_tolerance_bound": tolerance,
            "encoder_batch_size": args.encoder_batch_size,
            "head_batch_size": args.head_batch_size,
            "decoder": "opencv_forced",
            "encoder_forward": "direct_encoder_mean" if adaptation == "frozen" else "unmerged_lora_forward_top_two",
            "precision": "CUDA BF16 autocast encoder and head",
            "checkpoint_sha256": registry["checkpoint_sha256"],
            "canonical_prediction_sha256": registry["canonical_prediction_sha256"],
            "optimizer_steps": 0,
            "final_test_accessed": 0,
            "checkpoint_state_unchanged": True,
            "per_episode": per_episode,
        }
        atomic_json(args.output_root / "IDENTITY_RESULT.json", result)
        atomic_json(status_path, {"status": "complete" if result["status"] == "pass" else "failed", "gate": args.gate, "model": args.model, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": utc_now(), "max_abs": max_abs})
        if result["status"] != "pass":
            atomic_json(args.output_root / "RUN_FAILED.json", {"status": "failed", "reason": "identity_replay", "completed_utc": utc_now(), "max_abs": max_abs, "gate": args.gate})
            print(json.dumps(result, sort_keys=True))
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"status": "failed", "gate": args.gate, "model": args.model, "error": f"{type(exc).__name__}: {exc}", "completed_utc": utc_now(), "optimizer_steps": 0, "final_test_accessed": 0}
        atomic_json(args.output_root / "RUN_FAILED.json", failure)
        atomic_json(status_path, {"status": "failed", "gate": args.gate, "model": args.model, "optimizer_steps": 0, "final_test_accessed": 0, "completed_utc": utc_now(), "error": failure["error"]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
