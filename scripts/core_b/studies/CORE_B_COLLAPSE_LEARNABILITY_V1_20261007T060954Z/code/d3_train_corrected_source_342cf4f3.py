# Repaired public copy of EXCL-CODE-015-DEPENDENCY; source SHA-256: 342cf4f3423f3e5f0a4701661f53bf9e1498d2f3f7daa1351f83902eb2029e6a
"""D3-EQ trainer: historical model/config with samplewise episode weighting."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d3_eq_contract import effective_update_groups, episode_equal_weights, per_sample_loss, shuffled_epoch_order
from d3_model import D3TimeSpacePredictor, parameter_count, state_hash
from corrected_scheduler import UpdateDrivenCosineAnnealingLR, write_lr_trace


TASK_ROOT = Path(os.environ.get("EE6008_D3_SCHEDULER_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT_ROOT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
EQ_ROOT = Path(os.environ.get("EE6008_D3_EQ_STUDY_ROOT", PROJECT_ROOT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z")).expanduser().resolve()
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
TARGET_CACHE = EQ_ROOT / "d3/train_targets_trainonly.npz"
TARGET_CACHE_MANIFEST = EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"
CACHE_MANIFEST = BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
VIDEO_NORMALIZER = BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"
PROTOCOL_LOCK = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
STUDY_PROTOCOL_LOCK = TASK_ROOT / "PROTOCOL_LOCKED.json"
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(value, part)
    os.replace(part, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

class GridStore:
    """Read-only grid cache reader, preserving the historical D3 row identity."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            if int(source_index) >= int(shard["end_row"]):
                raise IndexError(f"cache row {source_index} is not covered by its shard")
            local = int(source_index) - int(shard["start_row"])
            grouped.setdefault(shard_index, []).append((output_index, local))
        for shard_index, pairs in grouped.items():
            shard = self.shards[shard_index]
            values = np.load(shard["array"], mmap_mode="r", allow_pickle=False)
            for output_index, local in pairs:
                dynamic[output_index] = values[local, 0]
                static[output_index] = values[local, 1]
        return dynamic, static


def load_train_bundle() -> dict[str, Any]:
    """Load train inputs only; future targets come from train-only Parquets."""
    input_archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = input_archive["split_code"].astype(np.int8, copy=True)
    if set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError("D3-EQ input must contain only train/dev codes")
    train_indices = np.flatnonzero(split == 0).astype(np.int64)
    train_ids = input_archive["anchor_id"][train_indices].astype("U64", copy=True)
    episodes = input_archive["episode_index"][train_indices].astype(np.int64, copy=True)
    tasks = input_archive["original_task_index"][train_indices].astype(np.int64, copy=True)
    input_keys = (
        "history_standardized",
        "history_mask",
        "history_relative_delta_standardized",
        "anchor_command_native",
    )
    x = {key: np.asarray(input_archive[key][train_indices]).copy() for key in input_keys}
    # Deliberately do not access input_archive["targets_native"].

    cache_archive = np.load(TARGET_CACHE, allow_pickle=False)
    target_ids = cache_archive["anchor_id"].astype("U64", copy=True)
    target_episodes = cache_archive["episode_index"].astype(np.int64, copy=True)
    target_anchors = cache_archive["anchor_command_native"].astype(np.float32, copy=True)
    targets60 = cache_archive["targets_native_60"].astype(np.float32, copy=True)
    if not np.array_equal(target_ids, train_ids) or not np.array_equal(target_episodes, episodes):
        raise AssertionError("train-only target cache IDs/order differ from V3 train rows")
    if targets60.shape != (len(train_indices), 60, 28) or target_anchors.shape != (len(train_indices), 28):
        raise AssertionError("train-only target cache shape mismatch")
    anchor_delta = float(np.max(np.abs(target_anchors - x["anchor_command_native"])))
    if anchor_delta > 1e-6:
        raise AssertionError(f"train raw anchor join mismatch: {anchor_delta}")

    target_norm = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    video_norm = json.loads(VIDEO_NORMALIZER.read_text(encoding="utf-8"))
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    target_cache_manifest = json.loads(TARGET_CACHE_MANIFEST.read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "PASS" or video_norm.get("status") != "LOCKED":
        raise AssertionError("D3 feature cache or video normalizer is not accepted")
    if target_cache_manifest.get("status") != "PASS" or target_cache_manifest.get("scope") != "primary_train_only":
        raise AssertionError("D3-EQ target cache is not train-only/accepted")
    if target_cache_manifest.get("target_cache_sha256") != sha256_file(TARGET_CACHE):
        raise AssertionError("D3-EQ target-cache checksum mismatch")
    if len(train_indices) != 17757 or len(np.unique(episodes)) != 2712:
        raise AssertionError("D3-EQ train population differs from locked 17,757/2,712 counts")

    protocol = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED" or protocol.get("final_test_access") != 0:
        raise AssertionError("D3-EQ protocol is absent or not locked")
    if protocol.get("train_windows") != len(train_indices) or protocol.get("train_episodes") != len(np.unique(episodes)):
        raise AssertionError("D3-EQ protocol population mismatch")

    return {
        "train_indices": train_indices,
        "train_ids": train_ids,
        "episode_ids": episodes,
        "task_ids": tasks,
        "inputs": x,
        "anchors": target_anchors,
        "targets60": targets60,
        "target_sigma": torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32)),
        "safe_fd_scale": torch.tensor(np.asarray(target_norm["safe_fd_scale"], dtype=np.float32)),
        "video_mean": np.asarray(video_norm["mean"], dtype=np.float32),
        "video_std": np.asarray(video_norm["safe_std"], dtype=np.float32),
        "grid_store": GridStore(cache_manifest),
        "weights": episode_equal_weights(episodes),
        "protocol_hash": sha256_file(PROTOCOL_LOCK),
        "target_cache_hash": sha256_file(TARGET_CACHE),
        "cache_manifest_hash": sha256_file(CACHE_MANIFEST),
        "normalizer_hash": sha256_file(V3_ROOT / "data/V3_NORMALIZERS.json"),
        "video_normalizer_hash": sha256_file(VIDEO_NORMALIZER),
        "target_cache_manifest_hash": sha256_file(TARGET_CACHE_MANIFEST),
        "anchor_join_max_abs_error": anchor_delta,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def latest_checkpoint(run_root: Path) -> Path | None:
    checkpoints = sorted(run_root.glob("checkpoint_epoch*.pt"), key=lambda p: int(p.stem.removeprefix("checkpoint_epoch")))
    return checkpoints[-1] if checkpoints else None


def make_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int,
    optimizer_steps: int, seed: int, config: str, horizon: int,
    init_hash: str, bundle: dict[str, Any], study_protocol_hash: str, last_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "last_report": last_report,
        "next_epoch": epoch + 1,
        "optimizer_steps": optimizer_steps,
        "seed": seed,
        "config": config,
        "horizon": horizon,
        "study_protocol_hash": study_protocol_hash,
        "init_hash": init_hash,
        "protocol_hash": bundle["protocol_hash"],
        "target_cache_hash": bundle["target_cache_hash"],
        "cache_manifest_hash": bundle["cache_manifest_hash"],
        "normalizer_hash": bundle["normalizer_hash"],
        "video_normalizer_hash": bundle["video_normalizer_hash"],
        "target_cache_manifest_hash": bundle["target_cache_manifest_hash"],
        "sampler_rule": "np.random.default_rng(seed+epoch).shuffle(train_positions)",
        "rng_state": rng_state(),
        "final_test_access": 0,
    }


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def append_epoch_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def train(args: argparse.Namespace) -> int:
    if args.config not in CONFIGS or args.horizon not in (30, 60):
        raise ValueError("invalid locked D3-EQ cell")
    if args.batch_size not in (32, 64) or args.accumulation not in (1, 2):
        raise ValueError("only effective batch64 or batch32/accum2 is allowed")
    if args.batch_size * args.accumulation != 64:
        raise ValueError("effective batch must remain 64")
    if not torch.cuda.is_available():
        raise RuntimeError("D3-EQ training requires the separately approved CUDA host")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    seed_everything(args.seed)
    bundle = load_train_bundle()
    run_root = args.run_root.resolve()
    if not run_root.is_relative_to((TASK_ROOT / "runs").resolve()):
        raise ValueError("run_root must be inside this task's isolated runs/ tree")
    run_root.mkdir(parents=True, exist_ok=True)
    complete_path = run_root / "RUN_COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") == "PASS" and complete.get("protocol_hash") == bundle["protocol_hash"] and complete.get("study_protocol_hash") == sha256_file(STUDY_PROTOCOL_LOCK):
            print(json.dumps({"status": "PASS_REUSED", "run_root": str(run_root)}, sort_keys=True))
            return 0
        raise AssertionError("existing completion marker does not match the locked protocol")

    study_protocol = json.loads(STUDY_PROTOCOL_LOCK.read_text(encoding="utf-8"))
    study_protocol_hash = sha256_file(STUDY_PROTOCOL_LOCK)
    if study_protocol.get("status") != "LOCKED":
        raise AssertionError("corrected-scheduler task protocol is not LOCKED")
    code_hashes = study_protocol.get("task_code_hashes", {})
    if code_hashes.get("code/d3_train_corrected.py") != sha256_file(Path(__file__).resolve()):
        raise AssertionError("trainer code hash differs from the prelocked task protocol")
    if code_hashes.get("code/corrected_scheduler.py") != sha256_file(TASK_ROOT / "code/corrected_scheduler.py"):
        raise AssertionError("scheduler code hash differs from the prelocked task protocol")
    if study_protocol.get("matrix_sha256") != sha256_file(TASK_ROOT / "protocol/MATRIX_12.csv"):
        raise AssertionError("12-cell matrix hash differs from the prelocked task protocol")
    with (TASK_ROOT / "protocol/MATRIX_12.csv").open(newline="", encoding="utf-8") as matrix_handle:
        matrix_rows = list(csv.DictReader(matrix_handle))
    locked_cells = {(row["config"], int(row["seed"]), int(row["horizon_steps"])) for row in matrix_rows}
    if len(matrix_rows) != 12 or (args.config, args.seed, args.horizon) not in locked_cells:
        raise AssertionError("requested cell is absent from the locked 12-cell matrix")
    init_path = BENCH_ROOT / f"d3/runs/INIT_S{args.seed}.pt"
    if not init_path.is_file():
        raise FileNotFoundError(f"historical initialization state missing for seed {args.seed}")
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    model = D3TimeSpacePredictor(horizon=args.horizon, seed=args.seed, video_mode=args.config)
    model.load_state_dict(init_state, strict=True)
    init_hash = state_hash(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    n_train = len(bundle["train_indices"])
    steps_per_epoch = (n_train + 63) // 64
    if steps_per_epoch != 278 or 40 * steps_per_epoch != 11120:
        raise AssertionError(f"locked population implies 278*40 updates, observed {steps_per_epoch}*40")
    scheduler = UpdateDrivenCosineAnnealingLR(optimizer, T_max=40 * steps_per_epoch, eta_min=1e-5)
    epochs = 1 if args.stage0 else 40
    stage0_rows = min(256, n_train)
    start_epoch = 1
    optimizer_steps = 0
    last_report: dict[str, Any] = {}
    parent_checkpoint = latest_checkpoint(run_root) if not args.stage0 else None
    if parent_checkpoint:
        payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        for key, value in (
            ("protocol_hash", bundle["protocol_hash"]),
            ("study_protocol_hash", study_protocol_hash),
            ("target_cache_hash", bundle["target_cache_hash"]),
            ("cache_manifest_hash", bundle["cache_manifest_hash"]),
            ("normalizer_hash", bundle["normalizer_hash"]),
            ("video_normalizer_hash", bundle["video_normalizer_hash"]),
            ("target_cache_manifest_hash", bundle["target_cache_manifest_hash"]),
        ):
            if payload.get(key) != value:
                raise AssertionError(f"resume checkpoint {key} mismatch")
        if (payload.get("seed"), payload.get("config"), payload.get("horizon"), payload.get("init_hash")) != (args.seed, args.config, args.horizon, init_hash):
            raise AssertionError("resume checkpoint cell/initialization mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["next_epoch"])
        optimizer_steps = int(payload["optimizer_steps"])
        if optimizer_steps != scheduler.step_calls:
            raise AssertionError("resume optimizer_updates and scheduler calls disagree")
        if not np.isclose(optimizer.param_groups[0]["lr"], scheduler.get_last_lr()[0], rtol=0.0, atol=1e-12):
            raise AssertionError("resume optimizer LR and scheduler last_lr disagree")
        restore_rng_state(payload["rng_state"])
        last_report = payload.get("last_report", {})

    expected_init = json.loads((EQ_ROOT / "protocol/D3_EQ_SEED_REGISTRY.json").read_text(encoding="utf-8"))["seeds"][str(args.seed)]["init_state_hash"]
    expected_init_file_hash = json.loads((EQ_ROOT / "protocol/D3_EQ_SEED_REGISTRY.json").read_text(encoding="utf-8"))["seeds"][str(args.seed)]["init_state_file_sha256"]
    if expected_init_file_hash != sha256_file(init_path):
        raise AssertionError("initialization file checksum differs from the prelocked seed registry")
    if expected_init != init_hash:
        raise AssertionError("historical initialization hash differs from the prelocked seed registry")

    run_metadata = {
        "protocol_id": "CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1",
        "config": args.config,
        "horizon": args.horizon,
        "seed": args.seed,
        "effective_batch": 64,
        "physical_batch_size": args.batch_size,
        "accumulation_steps": args.accumulation,
        "epochs": 1 if args.stage0 else 40,
        "checkpoint_rule": "four-step Stage0" if args.stage0 else "every completed epoch; final=epoch40",
        "init_state_hash": init_hash,
        "init_state_path": str(init_path),
        "init_state_file_sha256": sha256_file(init_path),
        "study_protocol_hash": study_protocol_hash,
        "loss_contract_source_sha256": sha256_file(Path(__file__).with_name("d3_eq_contract.py")),
        "scheduler_contract": {"name": "CosineAnnealingLR", "T_max": 11120, "eta_min": 1e-5,
                               "step_location": "immediately_after_successful_optimizer_step"},
        "protocol_hash": bundle["protocol_hash"],
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "model_source_sha256": sha256_file(Path(__file__).with_name("d3_model.py")),
        "target_cache_sha256": bundle["target_cache_hash"],
        "cache_manifest_sha256": bundle["cache_manifest_hash"],
        "normalizer_sha256": bundle["normalizer_hash"],
        "video_normalizer_sha256": bundle["video_normalizer_hash"],
        "target_cache_manifest_sha256": bundle["target_cache_manifest_hash"],
        "source_target_scope": "primary_train_parquet_only; no dev target array loaded",
        "train_windows": n_train,
        "train_episodes": int(len(np.unique(bundle["episode_ids"]))),
        "anchor_join_max_abs_error": bundle["anchor_join_max_abs_error"],
        "final_test_access": 0,
    }
    atomic_json(run_root / "RUN_ENVIRONMENT.json", {
        **run_metadata,
        "started_utc": utc_now(),
        "hostname": socket.gethostname(),
        "logical_device": "cuda:0",
        "physical_gpu_index": os.environ.get("GPU_WATCHER_PHYSICAL_GPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu_uuid": os.environ.get("D3_EXPECTED_GPU_UUID"),
        "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID"),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "command_argv": sys.argv,
        "cwd": str(Path.cwd()),
    })
    append_jsonl(run_root / "ATTEMPTS.jsonl", {
        "attempt_id": f"{utc_now()}-{os.getpid()}",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "watcher_job_id": os.environ.get("GPU_WATCHER_JOB_ID"),
        "hostname": socket.gethostname(),
        "physical_gpu_index": os.environ.get("GPU_WATCHER_PHYSICAL_GPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu_uuid": os.environ.get("D3_EXPECTED_GPU_UUID"),
        "start_epoch": start_epoch,
        "resume_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "command_argv": sys.argv,
        "final_test_access": 0,
    })
    atomic_json(run_root / "RUN_STATUS.json", {
        "status": "running",
        "started_utc": utc_now(),
        "start_epoch": start_epoch,
        "seed": args.seed,
        "config": args.config,
        "horizon": args.horizon,
        "resume_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "final_test_access": 0,
    })
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        for epoch in range(start_epoch, epochs + 1):
            scheduler.set_epoch(epoch)
            model.train()
            order = shuffled_epoch_order(n_train, args.seed, epoch)
            if args.stage0:
                order = order[:stage0_rows]
            optimizer.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            unweighted_loss_sum = 0.0
            seen_rows = 0
            epoch_steps = 0
            for group_start, group_end, micros in effective_update_groups(
                len(order), microbatch_size=args.batch_size, accumulation_steps=args.accumulation
            ):
                group_n = group_end - group_start
                for micro_start, micro_end in micros:
                    positions = order[micro_start:micro_end]
                    source_indices = bundle["train_indices"][positions]
                    dynamic, static = bundle["grid_store"].get(source_indices)
                    dynamic = ((dynamic - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
                    static = ((static - bundle["video_mean"][None, None, None, :]) / bundle["video_std"][None, None, None, :]).astype(np.float32)
                    inputs = bundle["inputs"]
                    state = torch.as_tensor(inputs["history_standardized"][positions], device=device)
                    mask = torch.as_tensor(inputs["history_mask"][positions], device=device)
                    relative = torch.as_tensor(inputs["history_relative_delta_standardized"][positions], device=device)
                    anchor = torch.as_tensor(bundle["anchors"][positions], device=device)
                    target = torch.as_tensor(bundle["targets60"][positions, : args.horizon], device=device)
                    weight = torch.as_tensor(bundle["weights"][positions], dtype=torch.float32, device=device)
                    dynamic_t = torch.as_tensor(dynamic, device=device)
                    static_t = torch.as_tensor(static, device=device)
                    pred = model(state, mask, relative, dynamic_t, static_t, anchor)
                    sample_loss = per_sample_loss(pred, target, anchor, bundle["target_sigma"].to(device), bundle["safe_fd_scale"].to(device))
                    weighted_contribution = torch.sum(weight * sample_loss) / group_n
                    weighted_contribution.backward()
                    weighted_loss_sum += float(torch.sum(weight * sample_loss.detach()).cpu())
                    unweighted_loss_sum += float(sample_loss.detach().sum().cpu())
                    seen_rows += len(positions)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                epoch_steps += 1
                if scheduler.step_calls != optimizer_steps:
                    raise AssertionError("each successful optimizer update must advance the scheduler exactly once")
                if args.stage0 and epoch_steps >= 4:
                    break
            # Preserve the historical call frequency and T_max; this isolates
            # the episode-weighted objective from the old scheduler behavior.
            scheduler.step()
            if scheduler.step_calls != optimizer_steps or scheduler.legacy_epoch_step_calls != epoch:
                raise AssertionError("scheduler/optimizer update counts diverged")
            epoch_records = [record for record in scheduler.update_records if record["epoch"] == epoch]
            if len(epoch_records) != epoch_steps:
                raise AssertionError("scheduler trace does not contain one row per optimizer update")
            last_report = {
                "epoch": epoch,
                "epoch_optimizer_steps": epoch_steps,
                "optimizer_steps_total": optimizer_steps,
                "scheduler_step_calls_total": scheduler.step_calls,
                "rows_seen": seen_rows,
                "weighted_loss_sum_over_seen_rows": weighted_loss_sum / max(seen_rows, 1),
                "unweighted_mean_loss": unweighted_loss_sum / max(seen_rows, 1),
                "lr": float(scheduler.get_last_lr()[0]),
                "lr_first_update_used": epoch_records[0]["lr_used_for_update"],
                "lr_last_update_used": epoch_records[-1]["lr_used_for_update"],
                "lr_after_epoch": float(scheduler.get_last_lr()[0]),
                "duration_seconds": time.monotonic() - started,
            }
            append_epoch_log(run_root / "EPOCH_LOG.csv", {
                "utc": utc_now(), "epoch": epoch, "optimizer_steps_total": optimizer_steps,
                "rows_seen": seen_rows, "weighted_loss_sum_over_seen_rows": last_report["weighted_loss_sum_over_seen_rows"],
                "unweighted_mean_loss": last_report["unweighted_mean_loss"], "lr": last_report["lr"],
                "stage0": int(args.stage0), "final_test_access": 0,
            })
            if args.stage0:
                if epoch_steps != 4:
                    raise AssertionError(f"Stage0 must perform exactly four optimizer steps, observed {epoch_steps}")
                if scheduler.step_calls != 4 or scheduler.legacy_epoch_step_calls != 1:
                    raise AssertionError("Stage0 requires four post-update hooks and one inert epoch-end call")
                payload = make_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, optimizer_steps=optimizer_steps, seed=args.seed, config=args.config, horizon=args.horizon, init_hash=init_hash, bundle=bundle, study_protocol_hash=study_protocol_hash, last_report=last_report)
                atomic_torch_save(run_root / "checkpoint_stage0.pt", payload)
                write_lr_trace(run_root / "LR_TRACE_ACTUAL.csv", scheduler.update_records)
                peak = int(torch.cuda.max_memory_allocated(device))
                report = {
                    "status": "PASS",
                    "config": args.config,
                    "horizon": args.horizon,
                    "seed": args.seed,
                    "optimizer_steps": optimizer_steps,
                    "stage0_rows": seen_rows,
                    "scheduler_step_calls": scheduler.step_calls,
                    "legacy_epoch_step_calls": scheduler.legacy_epoch_step_calls,
                    "lr_first_update_used": scheduler.update_records[0]["lr_used_for_update"],
                    "lr_after_four_updates": scheduler.get_last_lr()[0],
                    "init_hash": init_hash,
                    "peak_allocated_bytes": peak,
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "checkpoint_sha256": sha256_file(run_root / "checkpoint_stage0.pt"),
                    "parameter_count": parameter_count(model),
                    "final_test_access": 0,
                }
                atomic_json(run_root / "STAGE0_REPORT.json", report)
                atomic_json(run_root / "RUN_STATUS.json", {"status": "stage0_complete", **report, "completed_utc": utc_now()})
                print(json.dumps(report, sort_keys=True))
                return 0
            if epoch >= 1:
                payload = make_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, optimizer_steps=optimizer_steps, seed=args.seed, config=args.config, horizon=args.horizon, init_hash=init_hash, bundle=bundle, study_protocol_hash=study_protocol_hash, last_report=last_report)
                atomic_torch_save(run_root / f"checkpoint_epoch{epoch}.pt", payload)
                write_lr_trace(run_root / "LR_TRACE_ACTUAL.csv", scheduler.update_records)
                atomic_json(run_root / "RUN_STATUS.json", {
                    "status": "running", "epoch": epoch, "optimizer_steps": optimizer_steps,
                    "last_report": last_report, "updated_utc": utc_now(), "final_test_access": 0,
                })
        expected_steps = 40 * steps_per_epoch
        if optimizer_steps != expected_steps:
            raise AssertionError(f"optimizer step count {optimizer_steps} != expected {expected_steps}")
        if scheduler.step_calls != expected_steps or scheduler.legacy_epoch_step_calls != 40:
            raise AssertionError("formal run must have 11,120 post-update calls and 40 inert epoch callbacks")
        if not np.isclose(scheduler.get_last_lr()[0], 1e-5, rtol=0.0, atol=2e-11):
            raise AssertionError("final scheduler LR must equal eta_min=1e-5")
        if last_report.get("rows_seen") != n_train:
            raise AssertionError("final epoch did not visit all train windows exactly once")
        last_payload = make_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=40, optimizer_steps=optimizer_steps, seed=args.seed, config=args.config, horizon=args.horizon, init_hash=init_hash, bundle=bundle, study_protocol_hash=study_protocol_hash, last_report=last_report)
        atomic_torch_save(run_root / "checkpoint_last.pt", last_payload)
        write_lr_trace(run_root / "LR_TRACE_ACTUAL.csv", scheduler.update_records)
        peak = int(torch.cuda.max_memory_allocated(device))
        complete = {
            "status": "PASS",
            "config": args.config,
            "horizon": args.horizon,
            "seed": args.seed,
            "epochs": 40,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
            "study_protocol_hash": study_protocol_hash,
            "scheduler_step_calls": scheduler.step_calls,
            "legacy_epoch_step_calls": scheduler.legacy_epoch_step_calls,
            "scheduler_lr_after_last_update": scheduler.get_last_lr()[0],
            "init_hash": init_hash,
            "protocol_hash": bundle["protocol_hash"],
            "checkpoint_last_sha256": sha256_file(run_root / "checkpoint_last.pt"),
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "parameter_count": parameter_count(model),
            "last_report": last_report,
            "encoder_updated": False,
            "final_test_access": 0,
        }
        atomic_json(complete_path, complete)
        atomic_json(run_root / "RUN_STATUS.json", {"status": "complete", "completed_utc": utc_now(), **complete})
        print(json.dumps(complete, sort_keys=True))
    except BaseException as exc:
        atomic_json(run_root / "RUN_STATUS.json", {
            "status": "failed_or_interrupted",
            "utc": utc_now(),
            "epoch_last_started": last_report.get("epoch"),
            "optimizer_steps": optimizer_steps,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "latest_checkpoint": str(latest_checkpoint(run_root)) if latest_checkpoint(run_root) else None,
            "final_test_access": 0,
        })
        raise
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--horizon", required=True, type=int, choices=(30, 60))
    parser.add_argument("--seed", required=True, type=int, choices=(20261001, 20261002, 20261003))
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
