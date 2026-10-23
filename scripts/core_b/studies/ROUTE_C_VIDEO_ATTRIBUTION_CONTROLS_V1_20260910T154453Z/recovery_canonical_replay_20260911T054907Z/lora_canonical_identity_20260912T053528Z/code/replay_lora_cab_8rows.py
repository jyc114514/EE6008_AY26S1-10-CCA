#!/usr/bin/env python3
"""Fresh-process C/A/B LoRA canonical identity replay on eight fixed rows.

The script intentionally uses the original B3 raw-LoRA protocol's model
construction and forward seam.  Each invocation is one fresh A or B replay;
the stored canonical array is C.  It only reads source artifacts and writes
new diagnostic artifacts below the experiment-local repair root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


HEAD_BATCH = 8
LOCKED_ATOL = 1e-5
LOCKED_RTOL = 1e-5


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def namespace(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError as exc:
        return f"unavailable:{exc}"


def normalized_uuid(value: str) -> str:
    return value.strip().removeprefix("GPU-")


def metric(delta: np.ndarray) -> dict[str, float]:
    absolute = np.abs(delta)
    return {
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("B3-FUS-D-L", "B3-V-D-L"), required=True)
    parser.add_argument("--replica", choices=("A", "B"), required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--b23-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match physical GPU argument")
    if os.environ.get("GPU_WATCHER_PHYSICAL_GPU") != str(args.physical_gpu):
        raise RuntimeError("watcher physical GPU binding missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in replay child")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")

    physical = args.physical_gpu
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical),
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    smi_fields = [item.strip() for item in query.stdout.strip().splitlines()[0].split(",")]
    smi_uuid = smi_fields[1]
    watcher_uuid = os.environ.get("GPU_WATCHER_GPU_UUID", "")
    if not watcher_uuid or smi_uuid != watcher_uuid:
        raise RuntimeError(f"watcher/nvidia-smi UUID mismatch: {watcher_uuid!r} vs {smi_uuid!r}")
    properties = torch.cuda.get_device_properties(0)
    torch_uuid = str(getattr(properties, "uuid", ""))
    if normalized_uuid(torch_uuid) != normalized_uuid(watcher_uuid):
        raise RuntimeError(f"watcher/PyTorch UUID mismatch: {watcher_uuid!r} vs {torch_uuid!r}")

    protocol_path = args.b23_root / "protocol/run_b3_raw_lora.py"
    protocol = load_module(protocol_path, f"canonical_b3_raw_lora_protocol_{args.model.replace('-', '_')}")
    registry_path = args.route_root / "ROUTE_C_CHECKPOINT_REGISTRY.csv"
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = next(row for row in csv.DictReader(handle) if row["model"] == args.model)

    input_archive = args.b23_root / "inputs/B2_INPUTS_0125.npz"
    with np.load(input_archive, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    dev_mask = data["split_code"] == 1
    ids = np.asarray(data["episode_index"][dev_mask], dtype=np.int64)
    state = np.asarray(data["state_action_history"][dev_mask], dtype=np.float32)
    if len(ids) != 614:
        raise RuntimeError(f"expected 614 canonical dev rows, got {len(ids)}")

    identity = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
    selected_ids = [int(item["episode_id"]) for item in identity["episodes"][:8]]
    if len(selected_ids) != 8 or len(set(selected_ids)) != 8:
        raise RuntimeError("fixed identity manifest does not provide eight unique rows")
    positions = {int(value): index for index, value in enumerate(ids.tolist())}
    missing = [episode for episode in selected_ids if episode not in positions]
    if missing:
        raise RuntimeError(f"fixed identity rows missing from canonical dev order: {missing}")
    selected_indices = [positions[episode] for episode in selected_ids]
    context_set: set[int] = set()
    for selected in selected_indices:
        start = (selected // HEAD_BATCH) * HEAD_BATCH
        context_set.update(range(start, min(start + HEAD_BATCH, len(ids))))
    context_indices = sorted(context_set)

    repo_src = (args.project_root / "repo/ee6008-humanoid-vjepa/src").resolve()
    third_party = (args.project_root / "repo/third_party/vjepa2").resolve()
    sys.path.insert(0, str(repo_src))
    sys.path.insert(0, str(third_party))
    from ee6008.data import clips as clips_module
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.lora import LoRAConfig, inject_top_block_lora
    from ee6008.models import load_vjepa2_1_base

    device = torch.device("cuda:0")
    clips_module._DECORD_USABLE = False
    encoder, encoder_info = load_vjepa2_1_base(
        source_root=third_party,
        checkpoint=Path(registry["base_checkpoint_path"]),
        device=device,
        num_frames=32,
        resolution=384,
    )
    inserted = inject_top_block_lora(encoder, config=LoRAConfig())
    fusion = args.model.startswith("B3-FUS")
    head = protocol.PooledRegression(768 + (568 if fusion else 0))
    model = protocol.B3Model(encoder, head, fusion).to(device)
    payload = torch.load(Path(registry["checkpoint_path"]), map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint has no model_state_dict")
    message = model.load_state_dict(state_dict, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"non-strict checkpoint load: {message}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_digest = state_digest(model)

    feature_norm = json.loads(Path(registry["feature_normalization_path"]).read_text(encoding="utf-8"))["dynamic_pooled"]
    feature_mean = torch.tensor(feature_norm["mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(feature_norm["safe_std"], dtype=torch.float32, device=device)
    with Path(registry["data_manifest"]).open(newline="", encoding="utf-8") as handle:
        manifest = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"

    predictions_by_index: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(context_indices), HEAD_BATCH):
            chunk = context_indices[start : start + HEAD_BATCH]
            clips = []
            for index in chunk:
                episode = int(ids[index])
                sample = decode_prefix(
                    data_root / manifest[episode]["video_path"],
                    duration_seconds=4.0,
                    sample_fps=8,
                    expected_source_fps=30.0,
                )
                clip = preprocess_vjepa(sample.frames, resolution=384)
                if tuple(clip.shape) != (3, 32, 384, 384):
                    raise RuntimeError(f"unexpected clip shape for episode {episode}: {tuple(clip.shape)}")
                clips.append(clip)
            clip_batch = torch.stack(clips).to(device)
            state_batch = torch.tensor(state[chunk], dtype=torch.float32, device=device)
            with protocol.autocast_context(device):
                output = protocol.make_forward(
                    model,
                    clip_batch,
                    state_batch,
                    None,
                    train=False,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                ).float().cpu().numpy()
            for local, index in enumerate(chunk):
                predictions_by_index[index] = output[local]
    torch.cuda.synchronize()
    after_digest = state_digest(model)
    if before_digest != after_digest:
        raise RuntimeError("model state changed during inference")

    candidate = np.stack([predictions_by_index[index] for index in selected_indices]).astype(np.float32)
    canonical_path = Path(registry["canonical_prediction_path"])
    with np.load(canonical_path, allow_pickle=False) as canonical_archive:
        canonical_ids = np.asarray(canonical_archive["episode_index"], dtype=np.int64)
        canonical_prediction = np.asarray(canonical_archive["predictions_standardized"], dtype=np.float32)
    canonical_positions = {int(value): index for index, value in enumerate(canonical_ids.tolist())}
    canonical = np.stack([canonical_prediction[canonical_positions[episode]] for episode in selected_ids]).astype(np.float32)
    delta = candidate - canonical
    effective_tolerance = LOCKED_ATOL + LOCKED_RTOL * max(1.0, float(np.max(np.abs(canonical_prediction))))

    result = {
        "status": "complete",
        "identity_status": "pass" if float(np.max(np.abs(delta))) <= effective_tolerance else "fail",
        "model": args.model,
        "replica": args.replica,
        "selected_N": 8,
        "selected_episode_ids": selected_ids,
        "selected_dev_positions": selected_indices,
        "compute_context_N": len(context_indices),
        "compute_context_episode_ids": [int(ids[index]) for index in context_indices],
        "head_batch_size": HEAD_BATCH,
        "comparison_to_canonical": metric(delta),
        "locked_atol": LOCKED_ATOL,
        "locked_rtol": LOCKED_RTOL,
        "effective_tolerance_bound": effective_tolerance,
        "checkpoint_path": registry["checkpoint_path"],
        "checkpoint_sha256_registry": registry["checkpoint_sha256"],
        "checkpoint_sha256_observed": sha256_file(Path(registry["checkpoint_path"])),
        "checkpoint_payload_keys": sorted(payload.keys()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_optimizer_step": payload.get("global_optimizer_step"),
        "canonical_prediction_path": registry["canonical_prediction_path"],
        "canonical_prediction_sha256_registry": registry["canonical_prediction_sha256"],
        "canonical_prediction_sha256_observed": sha256_file(canonical_path),
        "base_checkpoint_sha256_registry": registry["base_checkpoint_sha256"],
        "inserted_lora_modules": list(inserted),
        "encoder_info": encoder_info,
        "forward_source": "run_b3_raw_lora.make_forward -> forward_top_two_lora",
        "decoder": "opencv_forced",
        "precision": "CUDA BF16 autocast",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": physical,
        "watcher_gpu_uuid": watcher_uuid,
        "nvidia_smi_gpu_uuid": smi_uuid,
        "torch_device": "cuda:0",
        "torch_uuid": torch_uuid,
        "mount_namespace": namespace("/proc/self/ns/mnt"),
        "pid_namespace": namespace("/proc/self/ns/pid"),
        "model_state_digest_before": before_digest,
        "model_state_digest_after": after_digest,
        "model_state_unchanged": True,
        "optimizer_steps": 0,
        "final_test_accessed": 0,
        "checkpoint_weights_modified": False,
        "non_child_process_signaled": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_npz, episode_index=np.asarray(selected_ids, dtype=np.int64), canonical=canonical, candidate=candidate, delta=delta)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
