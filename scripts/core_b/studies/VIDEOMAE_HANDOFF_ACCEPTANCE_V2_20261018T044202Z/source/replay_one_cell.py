#!/usr/bin/env python3
"""Safe tensor-only frozen-feature VideoMAE replay child for one Core B cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

V3_REL = Path("repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data")
CACHE_REL = Path('tmp/project-team/videomae_cache_full_20261008T074027Z')
TEAM_CODE_REL = Path("reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def array_sha(values) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()




def write_prediction_bundles(output_root: Path, temp_dir: Path, predictions_by_split: dict,
                             rows_by_split: dict, ids, episodes, task_index, anchor_frame,
                             anchor_timestamp, horizon: int, config: str, seed: int,
                             input_shas: dict) -> list[dict]:
    """Serialize predictions and split-aligned canonical metadata through one shared path."""
    import numpy as np

    if not output_root.is_dir():
        raise FileNotFoundError(f"prediction output root must be a pre-created empty directory: {output_root}")
    if not temp_dir.is_dir():
        raise FileNotFoundError(f"prediction temporary directory does not exist: {temp_dir}")
    records = []
    for split, prediction in predictions_by_split.items():
        if split not in rows_by_split or split not in input_shas:
            raise AssertionError(f"missing per-split row or input-hash record: {split}")
        output_rows = np.asarray(rows_by_split[split], dtype=np.int64)
        n = len(output_rows)
        if prediction.ndim != 3 or prediction.shape[0] != n:
            raise AssertionError(f"prediction/row count mismatch for {split}: {prediction.shape}/{n}")
        aligned = {
            "anchor_id": ids[output_rows].astype("U96"),
            "episode_index": episodes[output_rows],
            "original_task_index": task_index[output_rows],
            "anchor_frame": anchor_frame[output_rows],
            "anchor_timestamp": anchor_timestamp[output_rows],
        }
        if any(len(value) != n for value in aligned.values()):
            raise AssertionError(f"metadata/row count mismatch for {split}")
        bundle_path = output_root / f"predictions_{split}.npz"
        if bundle_path.exists():
            raise FileExistsError(f"refusing to overwrite prediction bundle: {bundle_path}")
        with tempfile.NamedTemporaryFile(prefix=".prediction.", suffix=".npz", dir=str(temp_dir), delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            with temp_path.open("wb") as f:
                np.savez_compressed(
                    f, **aligned, predictions_native=prediction,
                    horizon_steps=np.asarray(horizon), config=np.asarray(config),
                    seed=np.asarray(seed), split=np.asarray(split)
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, bundle_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        records.append({
            "split": split, "path": str(bundle_path), "bytes": bundle_path.stat().st_size,
            "sha256": digest(bundle_path), "shape": list(prediction.shape),
            "dtype": str(prediction.dtype), "rows": n,
            "ids_sha256": input_shas[split]["ids_sha256"],
            "episode_ids_sha256": input_shas[split]["episode_ids_sha256"],
            "row_indices_sha256": input_shas[split]["row_indices_sha256"],
            "conditioning_input_hashes": input_shas[split],
            "all_finite": bool(np.isfinite(prediction).all())
        })
    if set(predictions_by_split) != set(rows_by_split):
        raise AssertionError("prediction splits and row-index splits differ")
    return records

def tensor_state_sha(model, torch) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        h.update(name.encode("utf-8") + b"\0")
        h.update(value)
    return h.hexdigest()


def watcher_device_receipt() -> dict:
    physical_text = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical_text.isdecimal() or cvd != physical_text:
        raise RuntimeError(f"watcher physical GPU/CUDA_VISIBLE_DEVICES mismatch: {physical_text!r}/{cvd!r}")
    physical = int(physical_text)
    status_path_text = os.environ.get("AUDIT_GPU_WATCHER_STATUS_PATH", "")
    if not status_path_text:
        raise RuntimeError("AUDIT_GPU_WATCHER_STATUS_PATH is required")
    status_path = Path(status_path_text)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("selected_gpu", -1)) != physical:
        raise RuntimeError("child physical GPU differs from watcher selected_gpu")
    final = status.get("final_recheck") or {}
    expected_uuid = final.get("uuid")
    snapshots = status.get("visible_snapshots") or []
    snapshot = next((row for row in snapshots if int(row.get("index", -1)) == physical), None)
    if not expected_uuid or snapshot is None or snapshot.get("uuid") != expected_uuid:
        raise RuntimeError("watcher snapshot/final recheck UUID mismatch")
    query = subprocess.run(
        ["nvidia-smi", f"--id={physical}",
         "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True
    )
    fields = [item.strip() for item in query.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError("child nvidia-smi aggregate identity query returned unexpected schema")
    actual_index, actual_uuid, actual_name, memory_total = fields
    if int(actual_index) != physical or actual_uuid != expected_uuid:
        raise RuntimeError("child nvidia-smi UUID does not match watcher final recheck UUID")
    return {
        "physical_gpu": physical, "cuda_visible_devices": cvd, "logical_device": "cuda:0",
        "watcher_status_path": str(status_path), "watcher_status_sha256": digest(status_path),
        "watcher_selected_gpu": status.get("selected_gpu"),
        "watcher_visible_pool": status.get("visible_gpu_pool"),
        "watcher_uuid": expected_uuid, "watcher_snapshot_uuid": snapshot.get("uuid"),
        "child_nvidia_smi_index": int(actual_index), "child_nvidia_smi_uuid": actual_uuid,
        "child_nvidia_smi_name": actual_name, "child_total_memory_mib": int(float(memory_total))
    }


class GridStore:
    def __init__(self, manifest: dict, v2_root: Path, original_project: Path):
        self.v2_root = v2_root
        self.original_project = original_project
        self.shards = sorted(manifest["shards"], key=lambda x: int(x["start_row"]))
        self.starts = [int(x["start_row"]) for x in self.shards]
        self.maps = {}

    def _array(self, shard_index: int):
        import numpy as np
        if shard_index not in self.maps:
            source_path = Path(self.shards[shard_index]["array"])
            if not source_path.is_absolute():
                source_path = self.original_project / source_path
            try:
                rel = source_path.resolve(strict=False).relative_to(self.original_project)
            except ValueError as exc:
                raise RuntimeError(f"feature shard is outside declared original project: {source_path}") from exc
            path = self.v2_root / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            self.maps[shard_index] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self.maps[shard_index]

    def get(self, indices):
        import bisect
        import numpy as np
        dynamic = np.empty((len(indices), 16, 4, 384), dtype=np.float32)
        static = np.empty_like(dynamic)
        by_shard = {}
        for out_i, row_i in enumerate(indices.astype(np.int64)):
            shard_i = bisect.bisect_right(self.starts, int(row_i)) - 1
            shard = self.shards[shard_i]
            if not int(shard["start_row"]) <= int(row_i) < int(shard["end_row"]):
                raise IndexError(f"feature cache misses V3 input row {row_i}")
            by_shard.setdefault(shard_i, []).append((out_i, int(row_i) - int(shard["start_row"])))
        for shard_i, pairs in by_shard.items():
            values = self._array(shard_i)
            if values.ndim != 5 or values.shape[1:] != (2, 16, 4, 384) or values.dtype != np.float32:
                raise AssertionError(f"feature shard schema mismatch: shard {shard_i} shape={values.shape} dtype={values.dtype}")
            for out_i, local_i in pairs:
                dynamic[out_i] = values[local_i, 0]
                static[out_i] = values[local_i, 1]
        return dynamic, static


def select_rows(ids, split_codes, split: str, count: int | None):
    import numpy as np
    code = 0 if split == "train" else 1
    rows = np.flatnonzero(split_codes == code).astype(np.int64)
    if count is not None:
        rows = np.asarray(sorted(rows.tolist(), key=lambda i: hashlib.sha256(str(ids[i]).encode("utf-8")).hexdigest())[:count], dtype=np.int64)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--run-kind", choices=("cpu-fixture", "gpu-stage0", "gpu-full"), required=True)
    ap.add_argument("--fixture-rows", type=int, default=64)
    args = ap.parse_args()

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    project, v2, audit = [p.resolve() for p in (args.project_root, args.v2_root, args.audit_root)]
    if args.device == "cuda" and args.run_kind == "cpu-fixture":
        raise ValueError("CPU fixture cannot request CUDA")
    if args.device == "cpu" and args.run_kind != "cpu-fixture":
        raise ValueError("stage0/formal replay must use the approved GPU child")
    if args.device == "cuda":
        contract = json.loads((audit / "REPLAY_CONTRACT.json").read_text(encoding="utf-8"))
        profile = json.loads((audit / "PROFILE_LOCK.json").read_text(encoding="utf-8"))
        if contract.get("locked_utc") is None:
            raise RuntimeError("replay contract is not locked before first GPU prediction")
        if args.run_kind != "gpu-stage0" and profile.get("lock_status") != "locked_before_first_GPU_prediction":
            raise RuntimeError("profile lock is not finalized before a formal GPU prediction")
        if args.run_kind == "gpu-stage0" and profile.get("lock_status") not in {
                "awaiting_live_host_profile_before_first_GPU_prediction", "locked_before_first_GPU_prediction"}:
            raise RuntimeError("profile lock has an unexpected state before Stage0")
        gpu_receipt = watcher_device_receipt()
        if not torch.cuda.is_available():
            raise RuntimeError("watcher-assigned CUDA device is unavailable")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        gpu_receipt = None
        device = torch.device("cpu")

    expected_path = audit / "EXPECTED_CELLS.csv"
    expected_rows = list(csv.DictReader(expected_path.open(newline="", encoding="utf-8")))
    cellrow = next((r for r in expected_rows if r["cell"] == args.cell), None)
    if cellrow is None:
        raise ValueError(f"cell is not one of the 12 locked expected cells: {args.cell}")
    mode, horizon, seed = cellrow["mode"], int(cellrow["horizon"]), int(cellrow["seed"])
    archive = json.loads((v2 / "ARCHIVE.json").read_text(encoding="utf-8"))
    cell = next((c for c in archive["cells"] if c["name"] == args.cell), None)
    if cell is None:
        raise ValueError(f"cell is not registered in V2 archive: {args.cell}")
    manifest_path, weights_path, config_path = [v2 / cell[k] for k in ("manifest", "weights", "config")]
    manifest, config = [json.loads(p.read_text(encoding="utf-8")) for p in (manifest_path, config_path)]
    if (digest(manifest_path) != cellrow["manifest_sha256"] or digest(config_path) != cellrow["config_sha256"]
            or digest(weights_path) != cellrow["weights_sha256"]):
        raise AssertionError("locked model/config/weight hash changed")
    if (manifest["mode"], int(manifest["horizon"]), int(manifest["seed"])) != (mode, horizon, seed):
        raise AssertionError("model manifest identity differs from expected cell")
    run_root = v2 / cell["run"]
    run_status = json.loads((run_root / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if (run_status.get("status") != "PASS_TRAIN_AND_EXPORT" or int(run_status.get("epochs", -1)) != 40
            or int(run_status.get("optimizer_steps", -1)) != 11120
            or run_status.get("final_test_access") != 0
            or int(config.get("epochs", -1)) != 40 or int(config.get("updates", -1)) != 11120
            or config.get("final_test_access") != 0 or bool(config.get("resume_check"))):
        raise AssertionError("fixed epoch-40 model run identity/scope mismatch")
    checkpoint_path = v2 / manifest["original_checkpoint"]
    if digest(checkpoint_path) != manifest["original_checkpoint_sha256"]:
        raise AssertionError("original full checkpoint byte hash differs; file was not deserialized")
    for name, sha in config["source_hashes"].items():
        if digest(v2 / TEAM_CODE_REL / name) != sha:
            raise AssertionError(f"model source hash mismatch: {name}")

    input_path = v2 / V3_REL / "V3_MAIN_INPUTS.npz"
    cache_manifest_path = v2 / CACHE_REL / "cache/CACHE_MANIFEST.json"
    feature_norm_path = v2 / CACHE_REL / "normalization/VIDEOMAE_VIDEO_NORMALIZER.json"
    target_norm_path = v2 / V3_REL / "V3_NORMALIZERS.json"
    input_sha, cache_manifest_sha = digest(input_path), digest(cache_manifest_path)
    feature_norm_sha, target_norm_sha = digest(feature_norm_path), digest(target_norm_path)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("input_sha256") != input_sha:
        raise AssertionError("feature-cache manifest is not bound to the canonical V3 input")
    expected_config_fields = {
        "cache_manifest_sha256": cache_manifest_sha, "normalizer_sha256": feature_norm_sha,
        "target_normalizer_sha256": target_norm_sha
    }
    for key, value in expected_config_fields.items():
        if config.get(key) != value:
            raise AssertionError(f"model config {key} identity mismatch")

    with np.load(input_path, allow_pickle=False) as source:
        split_codes = source["split_code"].copy()
        ids = source["anchor_id"].astype("U96", copy=True)
        episodes = source["episode_index"].astype(np.int64, copy=True)
        task_index = source["original_task_index"].astype(np.int64, copy=True)
        anchor_frame = source["anchor_frame"].astype(np.int64, copy=True)
        anchor_timestamp = source["anchor_timestamp"].astype(np.float64, copy=True)
        state_all = source["history_standardized"].astype(np.float32, copy=True)
        mask_all = source["history_mask"].astype(np.float32, copy=True)
        rel_all = source["history_relative_delta_standardized"].astype(np.float32, copy=True)
        anchor_all = source["anchor_command_native"].astype(np.float32, copy=True)
    if (state_all.shape != (20079, 32, 71) or mask_all.shape != (20079, 32)
            or rel_all.shape != (20079, 32) or anchor_all.shape != (20079, 28)):
        raise AssertionError("canonical conditioning input shape mismatch")
    if not all(np.isfinite(x).all() for x in (state_all, mask_all, rel_all, anchor_all)):
        raise AssertionError("nonfinite canonical conditioning inputs")
    if not np.isin(mask_all, (0.0, 1.0)).all():
        raise AssertionError("history mask contains values outside the locked binary mask")
    if len(ids) != 20079 or len(np.unique(ids)) != len(ids) or set(np.unique(split_codes).tolist()) != {0, 1}:
        raise AssertionError("V3 input anchor IDs or train/dev split codes are invalid")
    split_rows = {name: select_rows(ids, split_codes, name, None) for name in ("train", "dev")}
    if len(split_rows["train"]) != 17757 or len(split_rows["dev"]) != 2322:
        raise AssertionError("locked train/dev input count mismatch")

    if args.run_kind in {"cpu-fixture", "gpu-stage0"}:
        if args.fixture_rows != 64:
            raise ValueError("this audit's predeclared integration/resource fixture is exactly 64 train IDs")
        selected = select_rows(ids, split_codes, "train", args.fixture_rows)
        splits = [("train_fixture", selected)]
    else:
        splits = [("train", split_rows["train"]), ("dev", split_rows["dev"])]

    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    video_norm = json.loads(feature_norm_path.read_text(encoding="utf-8"))
    target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))
    if target_norm.get("fit_scope") != "primary train windows only; episode equal" or target_norm.get("final_test_access") != 0:
        raise AssertionError("target normalizer is not the locked train-only scale")
    sigma = torch.as_tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    if video_mean.shape != (384,) or video_std.shape != (384,) or not np.isfinite(video_mean).all() or not np.isfinite(video_std).all() or np.any(video_std <= 0):
        raise AssertionError("VideoMAE feature normalizer schema mismatch")

    module_spec = importlib.util.spec_from_file_location("audit_d3_model", v2 / TEAM_CODE_REL / "d3_model.py")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    module.VIDEO_DIM = 768
    with torch.random.fork_rng(devices=[]):
        model = module.D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=mode)
        if mode == "TS11-D":
            model.video_projection = torch.nn.Linear(384, 128)
            module.VIDEO_DIM = 384
    with safe_open(str(weights_path), framework="pt", device="cpu") as sf:
        safe_meta = sf.metadata() or {}
    expected_safe_meta = {
        "original_checkpoint_sha256": manifest["original_checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"], "epoch": str(manifest["epoch"]),
        "seed": str(seed), "horizon": str(horizon), "mode": mode
    }
    if any(safe_meta.get(key) != value for key, value in expected_safe_meta.items()):
        raise AssertionError("safetensors metadata does not match the locked cell manifest")
    with torch.no_grad():
        weights = load_file(str(weights_path), device="cpu")
        model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device).eval()
    if model.training:
        raise AssertionError("model is not in eval mode")
    before_state_sha = tensor_state_sha(model, torch)

    grid = GridStore(cache_manifest, v2, Path(archive["original_project"]).resolve())
    predictions_by_split = {}
    rows_by_split = {}
    ids_by_split = {}
    episodes_by_split = {}
    input_shas = {}
    if args.device == "cuda":
        props_before_prediction = torch.cuda.get_device_properties(0)
        profile_path = audit / "PROFILE_LOCK.json"
        profile_lock = json.loads(profile_path.read_text(encoding="utf-8"))
        current_profile = profile_lock.get("new_current_profile", {})
        locked_uuid = current_profile.get("gpu_uuid")
        if locked_uuid is None:
            if args.run_kind != "gpu-stage0":
                raise RuntimeError("first GPU prediction must be the Stage0 profile-locking child")
            host_receipt_path = audit / "GPU_HOST_PREFLIGHT.json"
            if not host_receipt_path.is_file():
                raise RuntimeError("GPU_HOST_PREFLIGHT.json is required before the first GPU prediction")
            host_receipt = json.loads(host_receipt_path.read_text(encoding="utf-8"))
            profile_lock["new_current_profile"] = {
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "torch_cuda_build": torch.version.cuda, "cudnn_build": torch.backends.cudnn.version(),
                "physical_gpu": gpu_receipt["physical_gpu"], "gpu_uuid": gpu_receipt["watcher_uuid"],
                "cuda_visible_devices": gpu_receipt["cuda_visible_devices"], "logical_device": "cuda:0",
                "gpu_name": props_before_prediction.name,
                "gpu_compute_capability": [int(props_before_prediction.major), int(props_before_prediction.minor)],
                "gpu_total_memory_bytes": int(props_before_prediction.total_memory),
                "driver_version": next((g.get("driver_version") for g in host_receipt.get("gpus", [])
                                        if g.get("uuid") == gpu_receipt["watcher_uuid"]), None),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "attention_backend": "PyTorch nn.TransformerDecoder runtime-selected backend; no explicit backend pin",
                "batch_size": 64, "tail": "min(64, remaining rows)", "dtype": "float32",
                "amp": False, "eval": True, "inference_mode": True,
                "feature_manifest_sha256": cache_manifest_sha, "feature_normalizer_sha256": feature_norm_sha,
                "target_normalizer_sha256": target_norm_sha, "input_sha256": input_sha,
                "host_preflight_path": str(host_receipt_path), "host_preflight_sha256": digest(host_receipt_path),
                "admission": "shared gpu_lease_watcher dynamic_all_visible; first current UUID chosen at runtime",
                "historical_gpu_identity": "not comparable: old run records physical index 2 but no UUID/name/architecture"
            }
            profile_lock["lock_status"] = "locked_before_first_GPU_prediction"
            profile_lock["finalized_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z")
            profile_lock["new_gpu_results_exposed_at_lock"] = False
            profile_lock["original_profile_match_status"] = "not_verifiable_historical_gpu_uuid_name_architecture_and_cudnn_missing"
            profile_lock["current_profile_match_status"] = "locked_for_this_audit"
            write_json(profile_path, profile_lock)
        elif locked_uuid != gpu_receipt["watcher_uuid"]:
            raise RuntimeError(f"GPU UUID differs from PROFILE_LOCK: locked={locked_uuid} child={gpu_receipt['watcher_uuid']}")
        torch.cuda.reset_peak_memory_stats(device)
    started = __import__("time").perf_counter()
    with torch.inference_mode():
        for output_split, rows in splits:
            n = len(rows)
            pred = np.empty((n, horizon, 28), dtype=np.float32)
            for start in range(0, n, 64):
                end = min(n, start + 64)
                selected_rows = rows[start:end]
                state = torch.as_tensor(state_all[selected_rows], dtype=torch.float32, device=device)
                mask = torch.as_tensor(mask_all[selected_rows], dtype=torch.float32, device=device)
                rel = torch.as_tensor(rel_all[selected_rows], dtype=torch.float32, device=device)
                anchor = torch.as_tensor(anchor_all[selected_rows], dtype=torch.float32, device=device)
                if mode == "STATE-CTRL":
                    dynamic = torch.zeros((len(selected_rows), 16, 4, 768), dtype=torch.float32, device=device)
                    static = torch.zeros_like(dynamic)
                else:
                    dynamic_np, static_np = grid.get(selected_rows)
                    dynamic_np = ((dynamic_np - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
                    static_np = ((static_np - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
                    dynamic = torch.as_tensor(dynamic_np, dtype=torch.float32, device=device)
                    static = torch.as_tensor(static_np, dtype=torch.float32, device=device)
                residual = model(state, mask, rel, dynamic, static, anchor)
                batch_pred = anchor[:, None, :] + residual * sigma
                if batch_pred.shape != (len(selected_rows), horizon, 28) or not torch.isfinite(batch_pred).all():
                    raise AssertionError(f"nonfinite or wrong-shaped native predictions: {output_split} rows {start}:{end}")
                pred[start:end] = batch_pred.detach().cpu().numpy().astype(np.float32, copy=False)
            if pred.shape != (n, horizon, 28) or pred.dtype != np.float32 or not np.isfinite(pred).all():
                raise AssertionError(f"prediction schema/finite failure: {output_split}")
            predictions_by_split[output_split] = pred
            rows_by_split[output_split] = rows.copy()
            ids_by_split[output_split] = ids[rows]
            episodes_by_split[output_split] = episodes[rows]
            input_shas[output_split] = {
                "ids_sha256": array_sha(ids[rows].astype("U96")),
                "episode_ids_sha256": array_sha(episodes[rows]),
                "row_indices_sha256": array_sha(rows),
                "state_sha256": array_sha(state_all[rows]),
                "history_mask_sha256": array_sha(mask_all[rows]),
                "relative_delta_sha256": array_sha(rel_all[rows]),
                "anchor_command_sha256": array_sha(anchor_all[rows]),
                "original_task_index_sha256": array_sha(task_index[rows]),
                "anchor_frame_sha256": array_sha(anchor_frame[rows]),
                "anchor_timestamp_sha256": array_sha(anchor_timestamp[rows])
            }
    if args.device == "cuda":
        torch.cuda.synchronize(device)
    after_state_sha = tensor_state_sha(model, torch)
    if before_state_sha != after_state_sha:
        raise AssertionError("parameters or buffers changed during inference")
    elapsed = __import__("time").perf_counter() - started

    runtime = {
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_build": torch.backends.cudnn.version() if args.device == "cuda" else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if args.device == "cuda" else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if args.device == "cuda" else None,
        "amp": False, "dtype": "float32", "eval": True, "inference_mode": True,
        "attention_backend": "PyTorch nn.TransformerDecoder runtime-selected backend; no explicit backend pin"
    }
    gpu_name, gpu_capability, gpu_total_memory = None, None, None
    peak_allocated_mib, peak_reserved_mib = None, None
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_capability = [int(props.major), int(props.minor)]
        gpu_total_memory = int(props.total_memory)
        peak_allocated_mib = math.ceil(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        peak_reserved_mib = math.ceil(torch.cuda.max_memory_reserved(device) / (1024 * 1024))
    runtime.update({"device": str(device), "physical_gpu_uuid_receipt": gpu_receipt,
                    "gpu_name_from_torch": gpu_name, "gpu_compute_capability": gpu_capability,
                    "gpu_total_memory_bytes": gpu_total_memory,
                    "peak_allocated_mib": peak_allocated_mib, "peak_reserved_mib": peak_reserved_mib})

    if args.run_kind == "cpu-fixture":
        output_root = audit / "replay/cpu_archive_fixtures_v2" / args.cell
    elif args.run_kind == "gpu-stage0":
        output_root = audit / "replay/gpu_stage0" / args.cell
    else:
        output_root = audit / "replay/gpu" / args.cell
    if not output_root.resolve().is_relative_to(audit):
        raise ValueError("output path escapes the new audit root")
    output_root.mkdir(parents=True, exist_ok=False)
    prediction_records = write_prediction_bundles(
        output_root, audit / "tmp", predictions_by_split, rows_by_split, ids, episodes,
        task_index, anchor_frame, anchor_timestamp, horizon, mode, seed, input_shas
    )
    sidecar = {
        "status": "PASS_CPU_ARCHIVE_FIXTURE" if args.run_kind == "cpu-fixture" else
                  "PASS_GPU_STAGE0" if args.run_kind == "gpu-stage0" else "PASS_GPU_FROZEN_REPLAY",
        "cell": args.cell, "mode": mode, "horizon": horizon, "seed": seed,
        "run_kind": args.run_kind, "device": str(device), "runtime": runtime,
        "safe_file_metadata": safe_meta, "safe_reload_strict": True,
        "original_checkpoint_sha256_bytes_only": digest(checkpoint_path),
        "weights_sha256": digest(weights_path), "config_sha256": digest(config_path),
        "input_sha256": input_sha, "feature_manifest_sha256": cache_manifest_sha,
        "feature_normalizer_sha256": feature_norm_sha, "target_normalizer_sha256": target_norm_sha,
        "target_arrays_loaded": False, "optimizer_created": False, "backward_called": False,
        "optimizer_steps": 0, "encoder_extraction": False, "final_test_access": 0,
        "model_state_sha256_before": before_state_sha, "model_state_sha256_after": after_state_sha,
        "parameters_and_buffers_unchanged": before_state_sha == after_state_sha,
        "cwd": os.getcwd(), "elapsed_seconds": elapsed, "batch_size": 64, "outputs": prediction_records
    }
    write_json(output_root / "REPLAY_RECEIPT.json", sidecar)
    print(json.dumps({"status": sidecar["status"], "cell": args.cell,
                      "splits": [r["split"] for r in prediction_records],
                      "rows": [r["rows"] for r in prediction_records],
                      "peak_allocated_mib": peak_allocated_mib, "peak_reserved_mib": peak_reserved_mib,
                      "gpu_uuid": gpu_receipt["watcher_uuid"] if gpu_receipt else None,
                      "seconds": round(elapsed, 3)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

