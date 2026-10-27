"""Run the bounded Core B native-space GPU smoke without touching final-test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TRAIN_COUNT = 256
DEV_COUNT = 128
INPUT_DIMS = {"video_only": 768, "state_action_history_only": 568, "gated_fusion": 768 + 568}
HORIZONS = {"1p0s": ("1.0s", 30), "0p5s": ("0.5s", 15)}
SEED = 20260902
PHYSICAL_BATCH = 64
MAX_EPOCHS = 10
MIN_EPOCHS = 3
PATIENCE = 3
MIN_DELTA = 0.0001
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, steps: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, steps * 28),
        )
        self.steps = steps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).reshape(inputs.shape[0], self.steps, 28)


def load_inputs(root: Path, horizon: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = root / f"NATIVE_SMOKE_INPUTS_{horizon}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if data["episode_index"].shape != (TRAIN_COUNT + DEV_COUNT,):
        raise ValueError("unexpected smoke sample count")
    if not np.array_equal(data["split_code"][:TRAIN_COUNT], np.zeros(TRAIN_COUNT, dtype=np.int8)):
        raise ValueError("train split contract failed")
    if not np.array_equal(data["split_code"][TRAIN_COUNT:], np.ones(DEV_COUNT, dtype=np.int8)):
        raise ValueError("dev split contract failed")
    if len(np.unique(data["episode_index"])) != TRAIN_COUNT + DEV_COUNT:
        raise ValueError("duplicate smoke episode IDs")
    if not np.isfinite(data["targets"]).all() or not np.isfinite(data["video_pooled"]).all() or not np.isfinite(data["state_action_history"]).all():
        raise ValueError("non-finite smoke input")
    manifest = json.loads((root / "NATIVE_SMOKE_SAMPLE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("final_test_accessed") != 0 or manifest.get("task_labels_used_as_model_input") is not False:
        raise ValueError("sample manifest final-test/label contract failed")
    return data, manifest


def load_contracts(root: Path, horizon: str, data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    input_norm = json.loads((root / "NATIVE_SMOKE_INPUT_NORMALIZATION.json").read_text(encoding="utf-8"))
    norm_path = root.parent / "CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
    target_norm_all = json.loads(norm_path.read_text(encoding="utf-8"))
    source_name, steps = HORIZONS[horizon]
    target_norm = target_norm_all[source_name]
    target_mean = np.asarray(target_norm["mean"], dtype=np.float32)
    target_std = np.asarray(target_norm["safe_std"], dtype=np.float32)
    if target_mean.shape != (steps, 28) or target_std.shape != (steps, 28):
        raise ValueError(f"target normalization shape failed for {horizon}")
    input_arrays = {}
    for name, source in (("video_pooled", data["video_pooled"]), ("state_action_history", data["state_action_history"])):
        mean = np.asarray(input_norm[name]["mean"], dtype=np.float32)
        std = np.asarray(input_norm[name]["safe_std"], dtype=np.float32)
        if mean.shape != (source.shape[1],) or std.shape != (source.shape[1],) or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError(f"input normalization shape/finite contract failed for {name}")
        input_arrays[name] = (mean, std)
    return input_arrays, {"mean": target_mean, "std": target_std}, {"source": str(norm_path), "sha256": sha256_file(norm_path), "horizon_key": source_name}


def make_input(data: dict[str, np.ndarray], input_arrays: dict[str, tuple[np.ndarray, np.ndarray]], lane: str) -> np.ndarray:
    pieces = []
    if lane in {"video_only", "gated_fusion"}:
        mean, std = input_arrays["video_pooled"]
        pieces.append((data["video_pooled"] - mean) / std)
    if lane in {"state_action_history_only", "gated_fusion"}:
        mean, std = input_arrays["state_action_history"]
        pieces.append((data["state_action_history"] - mean) / std)
    if not pieces:
        raise ValueError(f"unknown lane {lane}")
    result = np.concatenate(pieces, axis=1)
    if result.shape[1] != INPUT_DIMS[lane] or not np.isfinite(result).all():
        raise ValueError(f"input contract failed for {lane}")
    return result.astype(np.float32, copy=False)


def target_arrays(data: dict[str, np.ndarray], target_norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    native = data["targets"].astype(np.float32, copy=False)
    standardized = (native - target_norm["mean"]) / target_norm["std"]
    if not np.isfinite(standardized).all():
        raise ValueError("non-finite standardized target")
    return native, standardized.astype(np.float32, copy=False)


def model_predict(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(inputs.to(device, non_blocking=True)).float()


def metrics(pred: np.ndarray, target: np.ndarray, target_norm: dict[str, np.ndarray], train_native: np.ndarray, state_history: np.ndarray) -> dict[str, object]:
    error = pred - target
    native_pred = pred * target_norm["std"] + target_norm["mean"]
    native_target = target * target_norm["std"] + target_norm["mean"]
    persistence_native = np.repeat(state_history[:, -28:][:, None, :], target.shape[1], axis=1)
    train_mean_native = train_native.mean(axis=0)
    global_mean_native = np.broadcast_to(train_mean_native, target.shape).copy()
    persistence_std = (persistence_native - target_norm["mean"]) / target_norm["std"]
    global_mean_std = (global_mean_native - target_norm["mean"]) / target_norm["std"]
    def summary(values: np.ndarray) -> dict[str, float]:
        abs_values = np.abs(values)
        return {"mae": float(abs_values.mean()), "rmse": float(np.sqrt(np.mean(values**2))), "final_step_mae": float(abs_values[:, -1, :].mean())}
    return {
        "standardized": summary(error),
        "native_command_value": summary(native_pred - native_target),
        "global_mean_baseline_standardized": summary(global_mean_std - target),
        "persistence_baseline_standardized": summary(persistence_std - target),
        "global_mean_baseline_native_command_value": summary(global_mean_native - native_target),
        "persistence_baseline_native_command_value": summary(persistence_native - native_target),
        "per_timestep_standardized_mae": np.abs(error).mean(axis=(0, 2)).astype(float).tolist(),
        "per_dimension_standardized_mae": np.abs(error).mean(axis=(0, 1)).astype(float).tolist(),
        "all_finite": bool(np.isfinite(pred).all() and np.isfinite(target).all() and np.isfinite(native_pred).all()),
    }


def train_one(root: Path, lane: str, horizon: str, physical_gpu: int) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(f"refusing GPU mapping: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, expected {physical_gpu!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    data, _ = load_inputs(root, horizon)
    input_norm, target_norm, target_source = load_contracts(root, horizon, data)
    inputs = make_input(data, input_norm, lane)
    native_targets, standardized_targets = target_arrays(data, target_norm)
    train_x, dev_x = inputs[:TRAIN_COUNT], inputs[TRAIN_COUNT:]
    train_y, dev_y = standardized_targets[:TRAIN_COUNT], standardized_targets[TRAIN_COUNT:]
    train_native = native_targets[:TRAIN_COUNT]
    dev_native = native_targets[TRAIN_COUNT:]
    output_dir = root / "runs" / f"{lane}_{horizon}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "status": "locked_before_training",
        "created_utc": utc_now(),
        "lane": lane,
        "horizon": horizon,
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": SEED,
        "physical_batch_size": PHYSICAL_BATCH,
        "gradient_accumulation_steps": 1,
        "train_count": TRAIN_COUNT,
        "dev_count": DEV_COUNT,
        "model": "bounded_mlp_input_linear256_gelu_output",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "precision": "BF16 autocast",
        "max_epochs": MAX_EPOCHS,
        "min_epochs": MIN_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "selection_metric": "development_standardized_mae",
        "target_normalization": target_source,
        "target_physical_unit_status": "unresolved",
        "native_unit_conversion": False,
        "task_labels_used_as_model_input": False,
        "final_test_accessed": 0,
        "full_benchmark_started": False,
        "sample_manifest_sha256": sha256_file(root / "NATIVE_SMOKE_SAMPLE_MANIFEST.json"),
        "input_cache_sha256": sha256_file(root / f"NATIVE_SMOKE_INPUTS_{horizon}.npz"),
    }
    config_path = root / "smoke_configs" / f"{lane}_{horizon}_LOCKED.json"
    if config_path.exists():
        raise RuntimeError(f"refusing to overwrite existing locked config: {config_path}")
    write_json(config_path, config)
    set_seed(SEED)
    device = torch.device("cuda:0")
    model = RegressionHead(inputs.shape[1], standardized_targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_dataset, batch_size=PHYSICAL_BATCH, shuffle=True, generator=loader_generator, drop_last=False, num_workers=0, pin_memory=True)
    dev_inputs_tensor = torch.from_numpy(dev_x)
    curves = []
    best_value = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_inputs, batch_targets in loader:
            if batch_inputs.shape[0] != PHYSICAL_BATCH:
                raise RuntimeError(f"physical batch contract failed: {batch_inputs.shape[0]}")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(batch_inputs.to(device, non_blocking=True))
                loss = loss_fn(predictions, batch_targets.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dev_predictions = model(dev_inputs_tensor.to(device, non_blocking=True)).float().cpu().numpy()
        dev_error = dev_predictions - dev_y
        dev_mae = float(np.abs(dev_error).mean())
        dev_rmse = float(np.sqrt(np.mean(dev_error**2)))
        improved = dev_mae < best_value - MIN_DELTA
        if improved:
            best_value = dev_mae
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        curves.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "dev_standardized_mae": dev_mae, "dev_standardized_rmse": dev_rmse, "improved": improved, "stale_epochs": stale_epochs})
        if epoch >= MIN_EPOCHS and stale_epochs >= PATIENCE:
            break
    elapsed = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("no best checkpoint selected")
    last_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_pred = model_predict(model, dev_inputs_tensor, device).cpu().numpy()
    model.load_state_dict(last_state)
    last_pred = model_predict(model, dev_inputs_tensor, device).cpu().numpy()
    recovered_standardized = ((best_pred * target_norm["std"] + target_norm["mean"]) - target_norm["mean"]) / target_norm["std"]
    round_trip = recovered_standardized - best_pred
    if float(np.max(np.abs(round_trip))) > 1e-5:
        raise RuntimeError("native inverse-transform round trip failed")
    torch.save({"model_state_dict": best_state, "config": config, "best_epoch": best_epoch}, output_dir / "checkpoint_best.pt")
    torch.save({"model_state_dict": last_state, "config": config, "last_epoch": curves[-1]["epoch"]}, output_dir / "checkpoint_last.pt")
    np.savez_compressed(output_dir / "predictions_best.npz", episode_index=data["episode_index"][TRAIN_COUNT:], split_code=data["split_code"][TRAIN_COUNT:], predictions_standardized=best_pred.astype(np.float32), targets_standardized=dev_y.astype(np.float32), predictions_native=(best_pred * target_norm["std"] + target_norm["mean"]).astype(np.float32), targets_native=dev_native.astype(np.float32))
    np.savez_compressed(output_dir / "predictions_last.npz", episode_index=data["episode_index"][TRAIN_COUNT:], split_code=data["split_code"][TRAIN_COUNT:], predictions_standardized=last_pred.astype(np.float32), targets_standardized=dev_y.astype(np.float32), predictions_native=(last_pred * target_norm["std"] + target_norm["mean"]).astype(np.float32), targets_native=dev_native.astype(np.float32))
    with (output_dir / "epoch_curves.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,dev_standardized_mae,dev_standardized_rmse,improved,stale_epochs\n")
        for row in curves:
            handle.write(f"{row['epoch']},{row['train_loss']:.10g},{row['dev_standardized_mae']:.10g},{row['dev_standardized_rmse']:.10g},{int(row['improved'])},{row['stale_epochs']}\n")
    best_metrics = metrics(best_pred, dev_y, target_norm, train_native, data["state_action_history"][TRAIN_COUNT:])
    global_mean = float(best_metrics["global_mean_baseline_standardized"]["mae"])
    outcome = "ENGINEERING_PASS_SIGNAL_PRESENT" if best_value < global_mean else "ENGINEERING_PASS_NO_CLEAR_SIGNAL"
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    run_metrics = {"lane": lane, "horizon": horizon, "seed": SEED, "best_epoch": best_epoch, "last_epoch": curves[-1]["epoch"], "early_stopped": curves[-1]["epoch"] < MAX_EPOCHS, "best_standardized_mae": best_value, "global_mean_standardized_mae": global_mean, "signal_present": bool(best_value < global_mean), "outcome": outcome, "best": best_metrics, "last": metrics(last_pred, dev_y, target_norm, train_native, data["state_action_history"][TRAIN_COUNT:]), "elapsed_seconds": elapsed, "samples_per_second": float(TRAIN_COUNT * curves[-1]["epoch"] / max(elapsed, 1e-9)), "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_name": torch.cuda.get_device_name(device), "final_test_accessed": 0, "task_labels_used_as_model_input": False}
    write_json(output_dir / "epoch_curves.json", {"curves": curves})
    write_json(output_dir / "RUN_METRICS.json", run_metrics)
    status = {"status": "complete", "exit_code": 0, "lane": lane, "horizon": horizon, "artifacts_written": ["checkpoint_best.pt", "checkpoint_last.pt", "predictions_best.npz", "predictions_last.npz", "epoch_curves.csv", "epoch_curves.json", "RUN_METRICS.json"], "completed_utc": utc_now()}
    write_json(output_dir / "RUN_STATUS.json", status)
    artifact_hashes = {name: sha256_file(output_dir / name) for name in status["artifacts_written"]}
    complete = {"status": "complete", "exit_code": 0, "lane": lane, "horizon": horizon, "seed": SEED, "best_epoch": best_epoch, "last_epoch": curves[-1]["epoch"], "artifact_sha256": artifact_hashes, "config_sha256": sha256_file(config_path), "sample_manifest_sha256": config["sample_manifest_sha256"], "final_test_accessed": 0, "completed_utc": utc_now()}
    write_json(output_dir / "RUN_COMPLETE.json", complete)
    print(json.dumps({"status": "complete", "lane": lane, "horizon": horizon, "best_epoch": best_epoch, "best_standardized_mae": best_value, "outcome": outcome, "elapsed_seconds": elapsed}, sort_keys=True))
    return 0


def contract_smoke(root: Path, physical_gpu: int) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(f"refusing GPU mapping: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, expected {physical_gpu!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    rows = []
    for horizon in HORIZONS:
        data, _ = load_inputs(root, horizon)
        input_norm, target_norm, _ = load_contracts(root, horizon, data)
        _, standardized = target_arrays(data, target_norm)
        for lane in ("video_only", "state_action_history_only", "gated_fusion"):
            inputs = make_input(data, input_norm, lane)
            set_seed(SEED)
            model = RegressionHead(inputs.shape[1], standardized.shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for step in range(4):
                batch_x = torch.from_numpy(inputs[(step * PHYSICAL_BATCH) : ((step + 1) * PHYSICAL_BATCH)]).to(device, non_blocking=True)
                batch_y = torch.from_numpy(standardized[(step * PHYSICAL_BATCH) : ((step + 1) * PHYSICAL_BATCH)]).to(device, non_blocking=True)
                if batch_x.shape[0] != PHYSICAL_BATCH:
                    raise RuntimeError("contract smoke physical batch is not 64")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = nn.functional.mse_loss(model(batch_x), batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError("contract smoke non-finite loss")
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            rows.append({"horizon": horizon, "lane": lane, "physical_batch_size": PHYSICAL_BATCH, "steps": 4, "elapsed_seconds": time.perf_counter() - start, "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_name": torch.cuda.get_device_name(device), "status": "pass"})
            del model, optimizer
            torch.cuda.empty_cache()
    result = {"status": "pass", "physical_gpu": physical_gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "physical_batch_size": PHYSICAL_BATCH, "gradient_accumulation_steps": 1, "rows": rows, "final_test_accessed": 0, "full_benchmark_started": False, "completed_utc": utc_now()}
    write_json(root / "BATCH64_CONTRACT_SMOKE.json", result)
    (root / "BATCH64_CONTRACT_SMOKE.md").write_text("# Batch-64 physical contract smoke\n\nStatus: `pass`. Six lane/horizon combinations completed four real optimization steps with physical batch 64, no gradient accumulation, and BF16 CUDA autocast.\n\nFinal-test targets accessed: `0`.\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(rows), "physical_batch_size": PHYSICAL_BATCH}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--lane", choices=tuple(INPUT_DIMS), default=None)
    parser.add_argument("--horizon", choices=tuple(HORIZONS), default=None)
    args = parser.parse_args()
    if args.contract_only:
        return contract_smoke(args.root, args.physical_gpu)
    if args.lane is None or args.horizon is None:
        parser.error("--lane and --horizon are required unless --contract-only is used")
    return train_one(args.root, args.lane, args.horizon, args.physical_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
