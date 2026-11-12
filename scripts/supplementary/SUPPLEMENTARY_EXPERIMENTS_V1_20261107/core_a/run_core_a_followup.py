"""Core A fourth-second repeated-clip experiment using the accepted Batch-48 code.

Only the visual observation changes: original sampled positions 24:32 are tiled
four times.  Source code, source datasets, and accepted baseline outputs are
read-only.  All fitted seeds and optimizer/selection settings come from the
accepted A_STATIC configs named in FINAL_MODEL_MATRIX.csv.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

OLD_ROOT = os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
REPO_REL = "repo/ee6008-humanoid-vjepa"
MATRIX_REL = "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol/FINAL_MODEL_MATRIX.csv"
VARIANT = "fourth_second_8_frames_repeated_4_times"
SEEDS = (20260825, 20260826, 20260827)
COUNTS = {"train": 2757, "validation": 617, "final_test": 602}
CHECKPOINT_SHA = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
TRAIN_FIELDS = (
    "max_epochs", "min_epochs", "patience", "min_delta", "batch_size",
    "gradient_accumulation_steps", "learning_rate", "weight_decay", "num_heads",
    "probe_depth", "calibration_bins", "label_smoothing", "loss_name",
    "num_workers", "preload_tokens", "use_bfloat16",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def immutable_json(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value, default=str)):
            raise ValueError(f"Existing experiment contract differs: {path}")
    else:
        write_json(path, value)


def relocate(path: str, source: Path) -> Path:
    if not OLD_ROOT:
        raise ValueError("Set EE6008_LEGACY_SOURCE_PREFIX to the exact absolute prefix recorded in the authorized source manifests")
    if path == OLD_ROOT:
        return source
    if not path.startswith(OLD_ROOT + "/"):
        raise ValueError(f"Source path is outside recorded project root: {path}")
    return source / path[len(OLD_ROOT) + 1:]


def bootstrap(source: Path):
    for path in (source / REPO_REL / "src", source / REPO_REL / "scripts"):
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import run_batch48_ablation_matrix_v3 as batch
    from ee6008.config import load_config
    return np, pd, torch, yaml, batch, load_config


def read_baselines(source: Path, yaml):
    matrix = source / MATRIX_REL
    with matrix.open(newline="", encoding="utf-8-sig") as f:
        refs = [r for r in csv.DictReader(f) if r["core"] == "A" and r["program"] == "A_STATIC"]
    if sorted(int(r["seed"]) for r in refs) != list(SEEDS):
        raise ValueError("Accepted A_STATIC matrix must contain exactly three expected seeds")
    result = {}
    for ref in refs:
        path = relocate(ref["config_path"], source)
        if not path.is_file():
            raise FileNotFoundError(f"Missing accepted source config: {path}")
        if sha(path) != ref["config_sha256"]:
            raise ValueError(f"Accepted source config hash mismatch: {path}")
        raw = yaml.safe_load(path.read_text())
        missing = [k for k in TRAIN_FIELDS if k not in raw]
        if missing:
            raise ValueError(f"Source config fields require explicit review (no guesses): {missing}: {path}")
        result[int(ref["seed"])] = (path, raw, ref)
    reference = {k: result[SEEDS[0]][1][k] for k in TRAIN_FIELDS}
    for seed, (_, raw, _) in result.items():
        differing = {k: raw[k] for k in TRAIN_FIELDS if raw[k] != reference[k]}
        if differing:
            raise ValueError(f"Accepted configs differ in training settings at seed {seed}: {differing}")
    if reference["batch_size"] != 48 or reference["gradient_accumulation_steps"] != 1:
        raise ValueError("Expected original physical batch 48 / accumulation 1")
    if reference["probe_depth"] != 4 or reference["num_heads"] != 16:
        raise ValueError("Expected original depth-4 / 16-head attentive classifier")
    if reference["loss_name"] != "cross_entropy" or reference["label_smoothing"] != 0:
        raise ValueError("Expected accepted standard cross-entropy setup")
    return result


def core_config(args, refs, load_config):
    original = load_config(refs[SEEDS[0]][0])
    return dataclasses.replace(
        original,
        project_root=args.source_root,
        g1_root=relocate(str(original.g1_root), args.source_root),
        manifest_dir=relocate(str(original.manifest_dir), args.source_root),
        checkpoint=relocate(str(original.checkpoint), args.source_root),
        vjepa_source_root=relocate(str(original.vjepa_source_root), args.source_root),
        feature_root=args.feature_root,
    )


def all_population(core, pd):
    from ee6008.full_attentive_data import load_v3_population
    from ee6008.data.selection import benchmark_primary_mask
    population = load_v3_population(core, full=True)
    # Final labels are not materialized during extraction/training/prediction.
    table = pd.read_parquet(core.manifest_dir / "splits.parquet", columns=[
        "episode_index", "split", "v3_benchmark_inclusion", "validation_status", "video_path"])
    rows = table[benchmark_primary_mask(table) & table.validation_status.eq("valid")].copy()
    rows = rows[rows.split.isin(COUNTS)].sort_values("episode_index", kind="mergesort").reset_index(drop=True)
    counts = rows.groupby("split").size().to_dict()
    if counts != COUNTS or rows.episode_index.duplicated().any():
        raise ValueError(f"Primary population mismatch: {counts}")
    rows = rows.merge(population.rows[["episode_index", "local_label_index"]],
                      on="episode_index", how="left", validate="one_to_one")
    rows["local_label_index"] = rows.local_label_index.fillna(-1).astype(int)
    return population, rows


def environment(torch):
    return {
        "hostname": socket.gethostname(), "user": os.environ.get("USER"),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def prepare(args, refs, core, population, rows, batch, torch):
    for path in (core.checkpoint, core.manifest_dir / "splits.parquet", core.manifest_dir / "label_map.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha(core.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Encoder checkpoint SHA256 mismatch")
    for raw in (v[1] for v in refs.values()):
        if raw["v3_split_sha256"] != population.split_sha256 or raw["label_map_sha256"] != population.label_map_sha256:
            raise ValueError("Original split/label map hashes differ from accepted training config")
    missing = [str(core.g1_root / str(v)) for v in rows.video_path if not (core.g1_root / str(v)).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} raw video paths missing, first: {missing[:3]}")
    contract = {
        "protocol": "core_a_fourth_second_repeated_clip_v1", "variant": VARIANT,
        "sampled_positions": list(range(24, 32)) * 4,
        "sampled_timestamps_seconds": [i / 8 for i in range(24, 32)] * 4,
        "repeat_count_is_clip_repetition_not_training_repeats": True,
        "source_root": str(args.source_root), "feature_root": str(args.feature_root),
        "output_root": str(args.output_root), "seeds": list(SEEDS),
        "training_policy": {k: refs[SEEDS[0]][1][k] for k in TRAIN_FIELDS},
        "accepted_config_hashes": {str(s): sha(v[0]) for s, v in refs.items()},
        "original_run_seed_source": str(args.source_root / REPO_REL / "scripts/run_batch48_ablation_matrix_v3.py"),
        "original_run_seed_source_sha256": sha(args.source_root / REPO_REL / "scripts/run_batch48_ablation_matrix_v3.py"),
        "original_encoder_source_sha256": sha(args.source_root / REPO_REL / "src/ee6008/models.py"),
        "original_preprocess_sha256": sha(args.source_root / REPO_REL / "src/ee6008/data/clips.py"),
        "runner_sha256": sha(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "population_sha256": population.population_sha256,
        "split_counts": COUNTS,
        "encoder_numeric_policy": "original run_core_a CUDA BF16 autocast; FP16 saved tokens; no forced SDPA/backend change",
        "feature_token_shape": [9216, 768], "token_storage_dtype": "float16",
        "token_storage_bytes_lower_bound": len(rows) * 9216 * 768 * 2,
        "original_train_dev_token_ram_bytes_lower_bound": (2757 + 617) * 9216 * 768 * 2,
    }
    immutable_json(args.output_root / "EXPERIMENT_CONTRACT.json", contract)
    mapping = [{"column_index": index, "canonical_task_id": task} for task, index in population.local_label_map.items()]
    immutable_json(args.output_root / "CLASS_MAP.json", sorted(mapping, key=lambda x: x["column_index"]))
    write_json(args.output_root / f"ENVIRONMENT_{args.stage}_seed_{args.seed or 'all'}_gpu_{args.physical_gpu}.json", environment(torch))
    print(json.dumps({"stage": args.stage, "counts": COUNTS, "token_gib": contract["token_storage_bytes_lower_bound"] / 2**30,
                      "train_dev_token_ram_gib": contract["original_train_dev_token_ram_bytes_lower_bound"] / 2**30}), flush=True)


def repeat_fourth_second(frames, np):
    if len(frames) != 32:
        raise ValueError("Expected the original 32 sampled frames before repetition")
    return np.concatenate([frames[24:32]] * 4, axis=0)


def extract(args, core, rows, np, pd, torch, batch):
    from ee6008.cache import FeatureCache
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.models import load_vjepa2_1_base, encode_clip
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production encoder extraction requires CUDA")
    selected = rows[rows.split.isin(args.splits)].copy()
    if args.limit is not None:
        selected = selected.head(args.limit)
    cache_cfg = {
        **core.to_dict(), "variant": VARIANT, "sampled_positions": list(range(24, 32)) * 4,
        "checkpoint_sha256": CHECKPOINT_SHA, "encoder_autocast": "bfloat16", "stored_tokens": "float16",
        "preprocess_sha256": sha(args.source_root / REPO_REL / "src/ee6008/data/clips.py"),
        "model_loader_sha256": sha(args.source_root / REPO_REL / "src/ee6008/models.py"),
    }
    args.feature_root.mkdir(parents=True, exist_ok=True)
    immutable_json(args.feature_root / "OBSERVATION_CONTRACT.json", cache_cfg)
    cache = FeatureCache(args.feature_root / "entries", cache_cfg)
    model, model_info = load_vjepa2_1_base(source_root=core.vjepa_source_root,
        checkpoint=core.checkpoint, device=device, num_frames=32, resolution=384)
    write_json(args.feature_root / "model_contract.json", {**model_info, "input_mode": VARIANT})
    manifest_path = args.feature_root / "feature_manifest.parquet"
    records = {}
    if manifest_path.exists():
        records = {int(r["episode_index"]): r for r in pd.read_parquet(manifest_path).to_dict("records")}
    started = time.monotonic()
    newly_encoded = 0
    for position, row in enumerate(selected.itertuples(index=False), 1):
        episode = int(row.episode_index)
        path = cache.path_for(episode)
        if not cache.has_valid(episode):
            sample = decode_prefix(core.g1_root / str(row.video_path), duration_seconds=4.0,
                                   sample_fps=8, expected_source_fps=core.source_fps)
            # Repeat the complete one-second sequence, preserving within-second order.
            frames = repeat_fourth_second(sample.frames, np)
            if frames.shape[0] != 32:
                raise ValueError("Fourth-second extraction did not produce 8 x 4 frames")
            clip = preprocess_vjepa(frames, 384).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens, pooled = encode_clip(model, clip)
            if tuple(tokens.shape) != (9216, 768):
                raise ValueError(f"Unexpected encoder token layout: {tuple(tokens.shape)}")
            metadata = {
                "split": str(row.split), "video_path": str(row.video_path), "input_mode": VARIANT,
                "sampled_positions": list(range(24, 32)) * 4,
                "frame_indices": np.tile(sample.frame_indices[24:32], 4).tolist(),
                "timestamps_seconds": np.tile(sample.timestamps_seconds[24:32], 4).tolist(),
                "max_source_frame_index": int(sample.frame_indices[31]),
                "max_sample_timestamp_seconds": float(sample.timestamps_seconds[31]),
                "source_fps_observed": sample.source_fps, "decoder": sample.decoder,
            }
            cache.write(episode_index=episode, tokens=tokens.half(), pooled=pooled.float(), metadata=metadata)
            newly_encoded += 1
            del tokens, pooled, clip
        record = cache.load(episode)
        metadata = record["metadata"]
        if metadata.get("input_mode") != VARIANT or tuple(record["tokens"].shape) != (9216, 768):
            raise ValueError(f"Unexpected reused cache entry {path}")
        records[episode] = {
            "episode_index": episode, "split": str(row.split),
            "feature_path": str(path.relative_to(args.feature_root)), "feature_sha256": sha(path),
            "feature_bytes": path.stat().st_size, "token_shape": "[9216, 768]", "pooled_shape": "[768]",
            "video_path": str(row.video_path), "max_source_frame_index": metadata["max_source_frame_index"],
            "max_sample_timestamp_seconds": metadata["max_sample_timestamp_seconds"],
            "source_fps_observed": metadata["source_fps_observed"], "input_mode": VARIANT,
        }
        del record
        if position % 10 == 0 or position == len(selected):
            tmp = manifest_path.with_name(manifest_path.name + ".part")
            pd.DataFrame(list(records.values())).sort_values("episode_index").to_parquet(tmp, index=False)
            os.replace(tmp, manifest_path)
            elapsed = time.monotonic() - started
            print(json.dumps({"stage": "extract", "progress": position, "selected": len(selected),
                "newly_encoded": newly_encoded, "elapsed_seconds": elapsed,
                "seconds_per_new_episode_including_io": elapsed / newly_encoded if newly_encoded else None}), flush=True)
    full = set(rows.episode_index.astype(int)) == set(records)
    write_json(args.feature_root / "EXTRACTION_STATUS.json", {
        "status": "complete" if full else "partial", "encoded_rows": len(records), "expected_rows": len(rows),
        "newly_encoded": newly_encoded, "elapsed_seconds": time.monotonic() - started,
        "feature_manifest_sha256": sha(manifest_path), "environment": environment(torch),
    })


def experiment(args, refs, core, batch, yaml, seed):
    original_path, raw, ref = refs[seed]
    resolved = {**raw, **core.to_dict(), "variant": VARIANT,
                "single_changed_variable": "video_observation_fourth_second_repeated_four_times",
                "output_root": str(args.output_root / "models"), "feature_run": str(args.feature_root),
                "physical_gpu": None, "physical_gpu_assigned_and_recorded_at_launch": True,
                "original_config_path": str(original_path), "original_config_sha256": sha(original_path)}
    path = args.output_root / "configs" / f"resolved_seed_{seed}.json"
    immutable_json(path, resolved)
    parent_path = relocate(raw["parent_config_path"], args.source_root)
    if not parent_path.is_file() or sha(parent_path) != raw["parent_batch32_config_sha256"]:
        raise ValueError(f"Parent Batch-32 config missing or mismatched: {parent_path}")
    return batch.Experiment(
        core=core, config_path=path, config_sha256=sha(path), raw=resolved, variant=VARIANT,
        ablation="standard", single_changed_variable="video_observation_fourth_second_repeated_four_times",
        parent_config_path=parent_path, feature_run=args.feature_root,
        output_root=args.output_root / "models", physical_gpu=args.physical_gpu,
        **{k: raw[k] for k in TRAIN_FIELDS}, class_weight_path=None, class_weight_sha256=None,
        class_weights=None, seeds=SEEDS,
    )


def smoke(args, refs, core, population, np, pd, torch, batch, yaml):
    exp = experiment(args, refs, core, batch, yaml, args.seed or SEEDS[0])
    manifest = pd.read_parquet(args.feature_root / "feature_manifest.parquet")
    ids = set(manifest.episode_index.astype(int))
    smoke_rows = population.rows[population.rows.split.eq("train") & population.rows.episode_index.isin(ids)].head(48)
    if len(smoke_rows) != 48:
        raise ValueError("Smoke requires at least 48 extracted training episodes")
    selected, _ = batch.attach_feature_manifest(smoke_rows, args.feature_root)
    device = torch.device(args.device)
    random.seed(SEEDS[0]); np.random.seed(SEEDS[0]); torch.manual_seed(SEEDS[0])
    model = batch.build_head(exp, device)
    opt = torch.optim.AdamW(model.parameters(), lr=exp.learning_rate, weight_decay=exp.weight_decay)
    loss_fn = batch.build_loss(exp, device)
    loader = batch.make_loader(selected, batch_size=48, shuffle=False, seed=SEEDS[0],
        num_workers=0, preload=True, shared_cache={}, pin_memory=device.type == "cuda")
    tokens, labels, _ = next(iter(loader))
    tokens = tokens.to(device); labels = labels.to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    train_times = []
    for _ in range(4):
        start = time.monotonic(); model.train(); opt.zero_grad(set_to_none=True)
        with batch.autocast_context(device, exp.use_bfloat16):
            logits = model(tokens); loss = loss_fn(logits.float(), labels)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite smoke loss")
        loss.backward()
        if not batch.finite_gradients(model): raise FloatingPointError("Nonfinite smoke gradient")
        opt.step()
        if device.type == "cuda": torch.cuda.synchronize()
        train_times.append(time.monotonic() - start)
    model.eval(); start = time.monotonic()
    with torch.inference_mode(), batch.autocast_context(device, exp.use_bfloat16):
        logits = model(tokens)
    if device.type == "cuda": torch.cuda.synchronize()
    eval_seconds = time.monotonic() - start
    report = {"status": "pass", "physical_batch_size": 48, "gradient_accumulation": 1,
        "smoke_steps": 4, "repeated_smoke_examples": 48, "train_step_seconds": train_times,
        "eval_step_seconds": eval_seconds,
        "compute_only_estimated_epoch_seconds": float(np.mean(train_times[1:]) * 58 + eval_seconds * 13),
        "estimate_excludes_preload_checkpoint_io_and_other_jobs": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "environment": environment(torch), "production_seed_state_not_reused": True}
    write_json(args.output_root / "SMOKE.json", report)
    print(json.dumps(report), flush=True)


def train(args, refs, core, population, torch, batch, yaml):
    selected, feature_sha = training_features(args, population, batch)
    if selected.groupby("split").size().to_dict() != {"train": 2757, "validation": 617}:
        raise ValueError("Training feature coverage mismatch")
    smoke_path = args.output_root / "SMOKE.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text())["status"] != "pass":
        raise ValueError("Run successful --stage smoke before production training")
    device = torch.device(args.device)
    for seed in ([args.seed] if args.seed else SEEDS):
        exp = experiment(args, refs, core, batch, yaml, seed)
        complete = exp.output_root / f"seed_{seed}" / "seed_complete.json"
        if complete.exists():
            result = json.loads(complete.read_text())
            expected = result["seed_contract"]
            if expected["config_sha256"] != exp.config_sha256 or expected["feature_manifest_sha256"] != feature_sha:
                raise ValueError(f"Completed model has different inputs: {complete}")
            continue
        exp.output_root.mkdir(parents=True, exist_ok=True)
        # Original training loop, optimizer, loader order, precision and selection.
        batch.run_seed(exp, selected, device=device, seed=seed,
                       feature_manifest_sha256=feature_sha, population_sha256=population.population_sha256)
    statuses = {str(s): (args.output_root / "models" / f"seed_{s}" / "seed_complete.json").exists() for s in SEEDS}
    write_json(args.output_root / "TRAIN_STATUS.json", {"status": "complete" if all(statuses.values()) else "partial", "seeds": statuses})


def training_features(args, population, batch):
    """Keep the fitted-model identity stable when final features are appended."""
    import pandas as pd
    selected, _ = batch.attach_feature_manifest(population.rows, args.feature_root)
    fields = ["episode_index", "feature_path", "feature_sha256", "token_shape", "pooled_shape"]
    snapshot = selected[fields].sort_values("episode_index", kind="mergesort").reset_index(drop=True)
    path = args.output_root / "manifests" / "TRAIN_FEATURE_MANIFEST.parquet"
    if path.exists():
        if not pd.read_parquet(path).equals(snapshot):
            raise ValueError("Train/development features changed after manifest was frozen")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        snapshot.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    return selected, sha(path)


def predict(args, refs, core, population, rows, np, pd, torch, batch, yaml):
    _, feature_sha = training_features(args, population, batch)
    selected, complete_feature_sha = batch.attach_feature_manifest(rows[rows.split.isin(args.splits)], args.feature_root)
    features = pd.read_parquet(args.feature_root / "feature_manifest.parquet")
    timing = features.set_index("episode_index").to_dict("index")
    task_by_column = np.asarray(population.task_ids, dtype=np.int64)
    device = torch.device(args.device)
    for seed in ([args.seed] if args.seed else SEEDS):
        exp = experiment(args, refs, core, batch, yaml, seed)
        checkpoint = exp.output_root / f"seed_{seed}" / "checkpoint_best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["feature_manifest_sha256"] != feature_sha or payload["experiment_config_sha256"] != exp.config_sha256:
            raise ValueError("Prediction checkpoint does not match this experiment")
        model = batch.build_head(exp, device)
        model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
        for split in args.splits:
            subset = selected[selected.split.eq(split)]
            loader = batch.make_loader(subset, batch_size=48, shuffle=False, seed=seed,
                num_workers=0, preload=False, shared_cache={}, pin_memory=device.type == "cuda")
            logits_list, episode_list = [], []
            with torch.inference_mode():
                for tokens, _, episodes in loader:
                    with batch.autocast_context(device, exp.use_bfloat16):
                        logits = model(tokens.to(device, non_blocking=True)).float().cpu()
                    if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite prediction")
                    logits_list.append(logits.numpy()); episode_list.append(episodes.numpy())
            logits = np.concatenate(logits_list); episodes = np.concatenate(episode_list)
            if len(episodes) != COUNTS[split] or len(set(episodes.tolist())) != COUNTS[split]:
                raise ValueError("Prediction episode support mismatch")
            ranking = np.argsort(-logits, axis=1, kind="stable")[:, :5]
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            out = args.output_root / "predictions" / f"seed_{seed}"
            batch.atomic_write_npz(out / f"{split}.npz", episode_index=episodes, logits=logits,
                canonical_task_ids=task_by_column, predicted_task_id=task_by_column[ranking[:, 0]])
            records = []
            for i, eid in enumerate(episodes):
                meta = timing[int(eid)]
                records.append({"episode_index": int(eid), "split": split, "seed": seed,
                    "predicted_task_id": int(task_by_column[ranking[i, 0]]),
                    "top5_task_ids_json": json.dumps(task_by_column[ranking[i]].tolist()),
                    "top1_probability": float(probabilities[i, ranking[i, 0]]),
                    "max_source_frame_index": int(meta["max_source_frame_index"]),
                    "max_sample_timestamp_seconds": float(meta["max_sample_timestamp_seconds"]),
                    "source_fps_observed": float(meta["source_fps_observed"]),
                    "video_path": meta["video_path"], "input_mode": VARIANT,
                    "prediction_scope": "in_sample" if split == "train" else "held_out_from_training"})
            batch.atomic_write_csv(out / f"{split}.csv", records, list(records[0]))
            write_json(out / f"{split}.receipt.json", {"checkpoint_sha256": sha(checkpoint),
                "training_feature_manifest_sha256": feature_sha, "complete_feature_manifest_sha256": complete_feature_sha,
                "row_count": len(records),
                "csv_sha256": sha(out / f"{split}.csv"), "npz_sha256": sha(out / f"{split}.npz"),
                "truth_labels_used_for_prediction": False, "output_column_map": "CLASS_MAP.json"})
            del loader, logits_list, episode_list
        del model
    print("Core A per-seed per-split canonical predictions completed", flush=True)


def score(args, population, rows, np, pd, torch, batch):
    from sklearn.metrics import accuracy_score, f1_score
    # Read final truth only after all selected prediction files exist.
    for seed in SEEDS:
        for split in args.splits:
            if not (args.output_root / "predictions" / f"seed_{seed}" / f"{split}.npz").is_file():
                raise FileNotFoundError(f"Missing sealed prediction for seed {seed} split {split}")
    core_manifest = args.source_root / REPO_REL / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
    truth_table = pd.read_parquet(core_manifest, columns=["episode_index", "original_task_index"]).set_index("episode_index")
    metrics = []
    for seed in SEEDS:
        for split in args.splits:
            path = args.output_root / "predictions" / f"seed_{seed}" / f"{split}.npz"
            with np.load(path, allow_pickle=False) as p:
                episode = p["episode_index"]; logits = p["logits"]; mapping = p["canonical_task_ids"]
            truth = truth_table.loc[episode].original_task_index.to_numpy(dtype=np.int64)
            top = mapping[np.argsort(-logits, axis=1, kind="stable")[:, :5]]
            metrics.append({"seed": seed, "split": split, "episodes": len(truth),
                "top1_accuracy": accuracy_score(truth, top[:, 0]),
                "top5_accuracy": np.mean(np.any(top == truth[:, None], axis=1)),
                "macro_f1_120": f1_score(truth, top[:, 0], labels=sorted(population.task_ids), average="macro", zero_division=0)})
    batch.atomic_write_csv(args.output_root / "results" / "metrics_per_seed.csv", metrics, list(metrics[0]))
    table = pd.DataFrame(metrics)
    summary = []
    for split in args.splits:
        group = table[table.split.eq(split)]
        for metric in ("top1_accuracy", "top5_accuracy", "macro_f1_120"):
            summary.append({"split": split, "metric": metric, "mean": group[metric].mean(),
                            "sample_sd": group[metric].std(ddof=1), "seeds": 3, "episodes": COUNTS[split]})
    batch.atomic_write_csv(args.output_root / "results" / "metrics_summary.csv", summary, list(summary[0]))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "extract", "smoke", "train", "predict", "score"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--splits", nargs="+", choices=tuple(COUNTS))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not OLD_ROOT:
        parser.error("Set EE6008_LEGACY_SOURCE_PREFIX before reading external assets")
    if args.splits is None:
        args.splits = ["validation", "final_test"] if args.stage in {"predict", "score"} else list(COUNTS)
    args.source_root = args.source_root.resolve(); args.output_root = args.output_root.resolve(); args.feature_root = args.feature_root.resolve()
    if args.source_root == args.output_root or args.source_root in args.output_root.parents or args.source_root in args.feature_root.parents:
        raise ValueError("Outputs/features must be outside the read-only source tree")
    np, pd, torch, yaml, batch, load_config = bootstrap(args.source_root)
    refs = read_baselines(args.source_root, yaml)
    core = core_config(args, refs, load_config)
    population, rows = all_population(core, pd)
    prepare(args, refs, core, population, rows, batch, torch)
    if args.stage == "extract": extract(args, core, rows, np, pd, torch, batch)
    elif args.stage == "smoke": smoke(args, refs, core, population, np, pd, torch, batch, yaml)
    elif args.stage == "train": train(args, refs, core, population, torch, batch, yaml)
    elif args.stage == "predict": predict(args, refs, core, population, rows, np, pd, torch, batch, yaml)
    elif args.stage == "score": score(args, population, rows, np, pd, torch, batch)


if __name__ == "__main__":
    main()
