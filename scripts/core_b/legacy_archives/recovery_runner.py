#!/usr/bin/env python3
"""Recovery-local checkpoint audit, replay and fixed-last inference.

This module deliberately has no training entry point.  It loads the existing
formal checkpoints, performs explicit eval-mode inference, and writes only
under the recovery root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
SOURCE_ROOT = REPO / "reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z"
A_ROOT = REPO / "reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z"
HISTORY_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
BENCHMARK = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
INPUT = HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
PREPROCESS_STATS = HISTORY_ROOT / "PREPROCESS_STATS.json"
TARGET_NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
VIDEO_NORM = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"
A_HEAD_CHECKPOINT = A_ROOT / "phase_a/runs/HI-F-HIST/seed20260940/formal/checkpoint_last.pt"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT_MANIFEST = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
PRETRAINED = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA_SOURCE = PROJECT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
CELLS = {
    "FT-D-FRZ": ("dynamic", "FRZ"),
    "FT-S-FRZ": ("static", "FRZ"),
    "FT-D-LORA4": ("dynamic", "LORA4"),
    "FT-S-LORA4": ("static", "LORA4"),
    "FT-D-DENSE4": ("dynamic", "DENSE4"),
    "FT-S-DENSE4": ("static", "DENSE4"),
}
ADAPTATIONS = ("FRZ", "LORA4", "DENSE4")
SCREEN_SEED = 20260944
INIT_SEED = 20260940
HAND = 14
STEPS = 30
HISTORY_STEPS = 32
STATE_DIM = 71
VIDEO_DIM = 768
PHYSICAL_BATCH = 2
FRAME_MAP = np.asarray([0, 4, 8, 11, 15, 19, 22, 26, 30, 34, 38, 41, 45, 49, 52, 56, 60, 64, 68, 71, 75, 79, 82, 86, 90, 94, 98, 101, 105, 109, 112, 116], dtype=np.int64)
REPLAY_TOLERANCES = {
    "eval_repeat_native_max_abs": 1e-6,
    "a_b_pooled_max_abs": 2e-4,
    "a_b_scaled_max_abs": 2e-4,
    "a_b_native_max_abs": 2e-4,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    np.savez_compressed(part, **values)
    actual = part if part.exists() else part.with_suffix(part.suffix + ".npz")
    os.replace(actual, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def load_input(required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(INPUT, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in required}
    return result


def input_metadata() -> dict[str, object]:
    with np.load(INPUT, allow_pickle=False) as archive:
        shapes = {key: list(archive[key].shape) for key in archive.files}
        keys = list(archive.files)
    return {"path": str(INPUT), "sha256": digest(INPUT), "keys": keys, "shapes": shapes}


def load_stats() -> dict[str, np.ndarray | float]:
    normalizer = json.loads(TARGET_NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    prep = json.loads(PREPROCESS_STATS.read_text(encoding="utf-8"))
    return {
        "target_mean": np.asarray(normalizer["mean"], dtype=np.float32),
        "target_std": np.asarray(normalizer["safe_std"], dtype=np.float32),
        "target_mean_hand": np.asarray(normalizer["mean"], dtype=np.float32)[:, :HAND],
        "target_std_hand": np.asarray(normalizer["safe_std"], dtype=np.float32)[:, :HAND],
        "residual_sigma": np.asarray(prep["residual_sigma"], dtype=np.float32),
        "fd_scale": np.asarray(prep["fd_scale"], dtype=np.float32),
        "lambda_fd": float(prep["lambda_fd"]),
    }


def import_head_model() -> Any:
    path = A_ROOT / "phase_a/code/model.py"
    spec = importlib.util.spec_from_file_location("recovery_a_head_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import locked A head model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_video_modules() -> tuple[Any, Any, Any, Any]:
    if str(PUBLIC_SRC) not in sys.path:
        sys.path.insert(0, str(PUBLIC_SRC))
    if str(VJEPA_SOURCE) not in sys.path:
        sys.path.insert(0, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips_module

    clips_module._DECORD_USABLE = False
    from ee6008.models import encode_clip, load_vjepa2_1_base

    return clips_module.decode_prefix, clips_module.preprocess_vjepa, load_vjepa2_1_base, encode_clip


class RecoveryQVLoRAQKV(nn.Module):
    """Recovery-local equivalent of the locked Q/V fused-qkv adapter."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("fused qkv dimension is not divisible by three")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_probability = dropout
        self.dim = base.out_features // 3
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_A_q = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B_q = nn.Parameter(torch.zeros(self.dim, rank, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_A_v = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B_v = nn.Parameter(torch.zeros(self.dim, rank, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5.0))

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        dropped = F.dropout(x, p=self.dropout_probability, training=self.training)
        q = F.linear(F.linear(dropped, self.lora_A_q), self.lora_B_q) * self.scaling
        v = F.linear(F.linear(dropped, self.lora_A_v), self.lora_B_v) * self.scaling
        return torch.cat((output[..., :self.dim] + q, output[..., self.dim:2 * self.dim], output[..., 2 * self.dim:] + v), dim=-1)


def build_recovery_encoder(device: torch.device, adaptation: str) -> tuple[nn.Module, dict[str, Any]]:
    _decode, _preprocess, load_base, _encode_clip = import_video_modules()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=PRETRAINED, device=device, num_frames=32, resolution=384)
    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 12:
        raise RuntimeError("expected 12 V-JEPA blocks")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if adaptation == "LORA4":
        for index in range(8, 12):
            blocks[index].attn.qkv = RecoveryQVLoRAQKV(blocks[index].attn.qkv)
    elif adaptation == "DENSE4":
        for index in range(8, 12):
            for parameter in blocks[index].parameters():
                parameter.requires_grad_(True)
        norms = getattr(encoder, "norms_block", None)
        if isinstance(norms, nn.ModuleList):
            for parameter in norms[-1].parameters():
                parameter.requires_grad_(False)
    elif adaptation != "FRZ":
        raise ValueError(adaptation)
    return encoder, {**info, "adaptation": adaptation, "top4_indices": [8, 9, 10, 11], "recovery_builder": True}


def load_recovery_head(device: torch.device) -> nn.Module:
    model = import_head_model()
    head = model.HistoryVideoTCNDCT(seed=INIT_SEED)
    payload = torch.load(A_HEAD_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("seed") != INIT_SEED or payload.get("cell") != "HI-F-HIST" or payload.get("epoch") != 40 or payload.get("optimizer_steps") != 1720:
        raise RuntimeError("A head initialization identity failed")
    head.load_state_dict(payload["model_state_dict"], strict=True)
    return head.to(device)


def load_source_runner() -> Any:
    path = SOURCE_ROOT / "phase_b/code/run_b_encoder_adaptation.py"
    spec = importlib.util.spec_from_file_location("source_b_runner_recovery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import source B runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(cell: str) -> Path:
    path = SOURCE_ROOT / f"phase_b/runs/{cell}/formal/checkpoint_last.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_loaded_model(cell: str, device: torch.device, builder: str) -> tuple[nn.Module, nn.Module, dict[str, Any], dict[str, Any]]:
    adaptation = CELLS[cell][1]
    if builder == "source":
        source = load_source_runner()
        encoder, info = source.load_encoder(device, adaptation)
        head = source.load_head(device)
    elif builder == "recovery":
        encoder, info = build_recovery_encoder(device, adaptation)
        head = load_recovery_head(device)
    else:
        raise ValueError(builder)
    checkpoint = checkpoint_path(cell)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {"epoch": 10, "global_step": 430, "seed": SCREEN_SEED}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint identity failed for {cell}: {expected}")
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    encoder.eval()
    head.eval()
    return encoder, head, payload, info


def state_digest(module: nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode("utf-8"))
        value.update(str(tensor.dtype).encode("utf-8"))
        value.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return value.hexdigest()


def module_training_flags(module: nn.Module) -> dict[str, bool]:
    return {name or "<root>": bool(child.training) for name, child in module.named_modules()}


def forward_top4(encoder: nn.Module, clips: Tensor) -> Tensor:
    blocks = encoder.blocks
    _, _, raw_t, height, width = clips.shape
    t_value = raw_t if encoder.check_temporal_dim(clips.shape) else raw_t // encoder.tubelet_size
    h_patches = height // encoder.patch_size
    w_patches = width // encoder.patch_size
    T = t_value if encoder.handle_nonsquare_inputs else None
    H = h_patches if encoder.handle_nonsquare_inputs else None
    W = w_patches if encoder.handle_nonsquare_inputs else None
    mode = "img" if encoder.check_temporal_dim(clips.shape) else "video"
    values = encoder.patch_embed_img(clips) if mode == "img" else encoder.patch_embed(clips)
    if encoder.modality_embedding:
        values = values + (encoder.img_mod_embed if mode == "img" else encoder.video_mod_embed).repeat(values.shape[0], 1, 1)
    for index, block in enumerate(blocks):
        result = block(values, mask=None, T=T, H_patches=H, W_patches=W, return_attn=False, mode=mode)
        values = result[0] if isinstance(result, tuple) else result
    norms = getattr(encoder, "norms_block", None)
    return norms[-1](values) if isinstance(norms, nn.ModuleList) else encoder.norm(values)


def encoder_pooled(encoder: nn.Module, clips: Tensor, adaptation: str) -> Tensor:
    if adaptation == "FRZ":
        tokens = encoder(clips)
    else:
        tokens = forward_top4(encoder, clips)
    return tokens.mean(dim=1).float()


def eval_autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def load_video_rows() -> dict[int, dict[str, Any]]:
    split = pd.read_parquet(SPLIT_MANIFEST, columns=["episode_index", "split", "v3_benchmark_inclusion", "original_task_index", "video_path"])
    split = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")]
    result: dict[int, dict[str, Any]] = {}
    for row in split.itertuples(index=False):
        result[int(row.episode_index)] = {"split": str(row.split), "task": int(row.original_task_index), "video_path": str(row.video_path)}
    return result


def load_input_rows() -> dict[str, np.ndarray]:
    return load_input(("episode_index", "split_code", "original_task_index", "history_standardized", "history_mask", "history_relative_delta_standardized", "persistence_native"))


class RawInputDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, rows: np.ndarray, data: dict[str, np.ndarray], video_rows: dict[int, dict[str, Any]], video_mode: str, decode_prefix: Any, preprocess: Any):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.data = data
        self.video_rows = video_rows
        self.video_mode = video_mode
        self.decode_prefix = decode_prefix
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        row_index = int(self.rows[index])
        episode = int(self.data["episode_index"][row_index])
        video_path = (RAW_ROOT / Path(self.video_rows[episode]["video_path"])).resolve()
        if not video_path.is_relative_to(RAW_ROOT.resolve()) or not video_path.is_file():
            raise FileNotFoundError(video_path)
        sample = self.decode_prefix(video_path, duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        if not np.array_equal(sample.frame_indices, FRAME_MAP) or np.any(sample.frame_indices >= 120):
            raise RuntimeError(f"input prefix contract failed for episode {episode}")
        frames = sample.frames
        if self.video_mode == "static":
            frames = np.repeat(frames[-1:, ...], frames.shape[0], axis=0)
        clip = self.preprocess(frames, resolution=384)
        if tuple(clip.shape) != (3, 32, 384, 384):
            raise RuntimeError(f"clip shape failed: {tuple(clip.shape)}")
        return (
            clip,
            torch.from_numpy(self.data["history_standardized"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["history_mask"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["history_relative_delta_standardized"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["persistence_native"][row_index, 0, :HAND].astype(np.float32)),
            torch.tensor(episode, dtype=torch.long),
            torch.tensor(row_index, dtype=torch.long),
        )


def make_fixture() -> int:
    data = load_input(("episode_index", "split_code", "history_standardized", "history_mask", "history_relative_delta_standardized", "persistence_native"))
    rows = np.flatnonzero(data["split_code"] == 0)[:8].astype(np.int64)
    fixture = {
        "status": "LOCKED",
        "selection": "first eight canonical train rows in input order",
        "rows": rows.tolist(),
        "episode_index": data["episode_index"][rows].astype(np.int64).tolist(),
        "history_hash": array_digest(data["history_standardized"][rows]),
        "mask_hash": array_digest(data["history_mask"][rows]),
        "delta_hash": array_digest(data["history_relative_delta_standardized"][rows]),
        "anchor_hash": array_digest(data["persistence_native"][rows, 0, :HAND]),
        "input_sha256": digest(INPUT),
        "final_test_access": 0,
    }
    atomic_json(RECOVERY / "data/REPLAY_FIXTURE.json", fixture)
    return 0


def loaded_stats_tensors(device: torch.device) -> dict[str, Tensor]:
    stats = load_stats()
    return {key: torch.from_numpy(value).to(device) if isinstance(value, np.ndarray) else torch.tensor(value, device=device) for key, value in stats.items()}


def normalizers(device: torch.device, video_mode: str) -> tuple[Tensor, Tensor]:
    values = json.loads(VIDEO_NORM.read_text(encoding="utf-8"))["dynamic_pooled" if video_mode == "dynamic" else "static_pooled"]
    return torch.tensor(values["mean"], dtype=torch.float32, device=device), torch.tensor(values["safe_std"], dtype=torch.float32, device=device)


def predict_batch(encoder: nn.Module, head: nn.Module, adaptation: str, video_mode: str, batch: tuple[Tensor, ...], device: torch.device) -> dict[str, Tensor]:
    # This context is required for both accepted eval and the disposable
    # train/eval mismatch diagnostic.  no_grad alone would still leave
    # stochastic modules active in train mode; inference_mode removes the
    # graph that previously caused DENSE4 replay memory accumulation.
    with torch.inference_mode():
        clips, state, mask, delta, anchor, episode, rows = batch
        clips = clips.to(device, non_blocking=True)
        state = state.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        delta = delta.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        video_mean, video_std = normalizers(device, video_mode)
        stats = loaded_stats_tensors(device)
        with eval_autocast(device):
            pooled = encoder_pooled(encoder, clips, adaptation)
        scaled = head(state, mask, delta, (pooled - video_mean) / video_std)
        native = anchor.unsqueeze(1) + scaled * stats["residual_sigma"].view(1, 1, HAND)
        standardized = (native - stats["target_mean_hand"].view(1, STEPS, HAND)) / stats["target_std_hand"].view(1, STEPS, HAND)
        return {"episode_index": episode, "row_index": rows, "pooled": pooled, "scaled": scaled, "native": native, "standardized": standardized}


def set_legacy_train_mode(encoder: nn.Module, head: nn.Module, adaptation: str) -> None:
    encoder.eval()
    head.train()
    if adaptation != "FRZ":
        for index in range(8, 12):
            encoder.blocks[index].train()


def selected_rows_for_replay() -> np.ndarray:
    fixture = json.loads((RECOVERY / "data/REPLAY_FIXTURE.json").read_text(encoding="utf-8"))
    return np.asarray(fixture["rows"], dtype=np.int64)


def build_loader(rows: np.ndarray, video_mode: str) -> DataLoader:
    data = load_input_rows()
    video_rows = load_video_rows()
    decode_prefix, preprocess, _load_base, _encode = import_video_modules()
    dataset = RawInputDataset(rows, data, video_rows, video_mode, decode_prefix, preprocess)
    return DataLoader(dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0, pin_memory=False)


def concat_outputs(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    keys = tuple(items[0])
    return {key: torch.cat([item[key] for item in items], dim=0) for key in keys}


def run_forward(encoder: nn.Module, head: nn.Module, cell: str, rows: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    outputs: list[dict[str, Tensor]] = []
    video_mode, adaptation = CELLS[cell]
    for batch in build_loader(rows, video_mode):
        outputs.append(predict_batch(encoder, head, adaptation, video_mode, batch, device))
    return concat_outputs(outputs)


def run_replay(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("replay requires approved CUDA host execution")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    rows = selected_rows_for_replay()
    cell = args.cell
    builder = "source" if args.path == "a" else "recovery"
    encoder, head, payload, info = build_loaded_model(cell, device, builder)
    before_digest = state_digest(nn.ModuleList([encoder, head]))
    encoder.eval(); head.eval()
    torch.manual_seed(9100)
    first = run_forward(encoder, head, cell, rows, device)
    second = run_forward(encoder, head, cell, rows, device)
    eval_repeat_diff = float(torch.max(torch.abs(first["native"] - second["native"])).detach().cpu())
    eval_flags = {"encoder": module_training_flags(encoder), "head": module_training_flags(head)}
    set_legacy_train_mode(encoder, head, CELLS[cell][1])
    torch.manual_seed(9101)
    legacy_first = run_forward(encoder, head, cell, rows, device)
    torch.manual_seed(9102)
    legacy_second = run_forward(encoder, head, cell, rows, device)
    legacy_repeat_diff = float(torch.max(torch.abs(legacy_first["native"] - legacy_second["native"])).detach().cpu())
    legacy_to_eval_diff = float(torch.max(torch.abs(legacy_first["native"] - first["native"])).detach().cpu())
    encoder.eval(); head.eval()
    after_digest = state_digest(nn.ModuleList([encoder, head]))
    output_dir = RECOVERY / f"replay/{cell}/{args.path}"
    atomic_npz(output_dir / "eval_outputs.npz", pooled=first["pooled"].detach().cpu().numpy().astype(np.float32), scaled=first["scaled"].detach().cpu().numpy().astype(np.float32), native=first["native"].detach().cpu().numpy().astype(np.float32), standardized=first["standardized"].detach().cpu().numpy().astype(np.float32), episode_index=first["episode_index"].detach().cpu().numpy(), row_index=first["row_index"].detach().cpu().numpy())
    result = {
        "status": "PASS",
        "cell": cell,
        "path": args.path,
        "builder": builder,
        "checkpoint": str(checkpoint_path(cell)),
        "checkpoint_sha256": digest(checkpoint_path(cell)),
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_global_step": payload["global_step"],
        "fixture_rows": rows.tolist(),
        "fixture_episode_index": first["episode_index"].detach().cpu().numpy().astype(np.int64).tolist(),
        "eval_repeat_native_max_abs": eval_repeat_diff,
        "legacy_train_mode_repeat_native_max_abs": legacy_repeat_diff,
        "legacy_to_eval_native_max_abs": legacy_to_eval_diff,
        "eval_training_flags": eval_flags,
        "post_replay_state_digest_equal": before_digest == after_digest,
        "state_digest_before": before_digest,
        "state_digest_after": after_digest,
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "dtype_contract": "encoder BF16 autocast; head/native FP32",
        "historical_reference_C": "not_saved",
        "final_test_access": 0,
    }
    atomic_json(output_dir / "REPLAY_RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("cell", "path", "eval_repeat_native_max_abs", "legacy_train_mode_repeat_native_max_abs", "legacy_to_eval_native_max_abs", "post_replay_state_digest_equal")}, sort_keys=True))
    return 0


def locked_arm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(LOCKED_ARM, allow_pickle=False) as archive:
        ids = archive["episode_index"].astype(np.int64)
        native = archive["predictions_native"][:, :, 14:].astype(np.float32)
        standardized = archive["predictions_standardized"][:, :, 14:].astype(np.float32)
    return ids, native, standardized


def run_infer(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("inference requires approved CUDA host execution")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    cell = args.cell
    encoder, head, payload, info = build_loaded_model(cell, device, "recovery")
    encoder.eval(); head.eval()
    before_digest = state_digest(nn.ModuleList([encoder, head]))
    data = load_input_rows()
    video_rows = load_video_rows()
    train_rows = np.flatnonzero(data["split_code"] == 0).astype(np.int64)
    dev_rows = np.flatnonzero(data["split_code"] == 1).astype(np.int64)
    if len(train_rows) != 2741 or len(dev_rows) != 614:
        raise RuntimeError("train/dev input contract failed")
    # Required exporter smoke tests use train inputs but never targets.
    smoke8 = run_forward(encoder, head, cell, train_rows[:8], device)
    smoke32 = run_forward(encoder, head, cell, train_rows[:32], device)
    if smoke8["native"].shape != (8, STEPS, HAND) or smoke32["native"].shape != (32, STEPS, HAND):
        raise RuntimeError("exporter train smoke shape failed")
    dev_output = run_forward(encoder, head, cell, dev_rows, device)
    dev_ids = dev_output["episode_index"].detach().cpu().numpy().astype(np.int64)
    input_ids = data["episode_index"][dev_rows].astype(np.int64)
    if not np.array_equal(dev_ids, input_ids) or len(np.unique(dev_ids)) != 614:
        raise RuntimeError("canonical dev ID export contract failed")
    arm_ids, arm_native, arm_std = locked_arm()
    if not np.array_equal(arm_ids, dev_ids):
        raise RuntimeError("locked Arm/dev ID seam failed")
    hand_native = dev_output["native"].detach().cpu().numpy().astype(np.float32)
    hand_std = dev_output["standardized"].detach().cpu().numpy().astype(np.float32)
    full_native = np.concatenate((hand_native, arm_native), axis=2)
    full_std = np.concatenate((hand_std, arm_std), axis=2)
    output_dir = RECOVERY / f"predictions/{cell}"
    atomic_npz(output_dir / "shard_000.npz", episode_index=dev_ids, original_task_index=data["original_task_index"][dev_rows].astype(np.int64), predictions_hand_native=hand_native, predictions_hand_standardized=hand_std, predictions_full_native=full_native, predictions_full_standardized=full_std, locked_arm_native=arm_native, locked_arm_standardized=arm_std)
    after_digest = state_digest(nn.ModuleList([encoder, head]))
    metadata = {
        "status": "PASS",
        "cell": cell,
        "checkpoint": str(checkpoint_path(cell)),
        "checkpoint_sha256": digest(checkpoint_path(cell)),
        "epoch": payload["epoch"],
        "optimizer_steps": payload["global_step"],
        "seed": payload["seed"],
        "dev_rows": 614,
        "dev_ids_sha256": array_digest(dev_ids),
        "input_sha256": digest(INPUT),
        "arm_source": str(LOCKED_ARM),
        "arm_source_sha256": digest(LOCKED_ARM),
        "arm_bitwise_equal_locked": bool(np.array_equal(full_native[:, :, 14:], arm_native) and np.array_equal(full_std[:, :, 14:], arm_std)),
        "prediction_shapes": {"hand_native": list(hand_native.shape), "hand_standardized": list(hand_std.shape), "full_native": list(full_native.shape), "full_standardized": list(full_std.shape)},
        "dtype": "float32",
        "batch_size": PHYSICAL_BATCH,
        "device": str(device),
        "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "unknown"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "state_digest_before": before_digest,
        "state_digest_after": after_digest,
        "parameter_and_buffer_hash_unchanged": before_digest == after_digest,
        "train_smoke_rows": [8, 32],
        "final_test_access": 0,
    }
    atomic_json(output_dir / "DEV_PREDICTION_ACCEPTANCE.json", metadata)
    atomic_json(output_dir / "PREDICTION_EXPORT_COMPLETE.json", {**metadata, "shards": ["shard_000.npz"]})
    print(json.dumps({"status": "PASS", "cell": cell, "dev_rows": 614, "output": str(output_dir), "physical_gpu": metadata["physical_gpu"]}, sort_keys=True))
    return 0


def run_inventory() -> int:
    # CPU-only coverage audit.  Model construction is repeated once per
    # adaptation and then compared against both video variants.
    data = load_input(("episode_index", "split_code", "targets_native", "history_standardized"))
    rows: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    expected_models: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]] = {}
    for adaptation in ADAPTATIONS:
        encoder, info = build_recovery_encoder(torch.device("cpu"), adaptation)
        head = load_recovery_head(torch.device("cpu"))
        expected_models[adaptation] = (encoder.state_dict(), head.state_dict(), sum(p.numel() for p in encoder.parameters() if p.requires_grad), sum(p.numel() for p in head.parameters()))
    for cell, (_video_mode, adaptation) in CELLS.items():
        path = checkpoint_path(cell)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_encoder, expected_head, trainable_encoder_count, head_count = expected_models[adaptation]
        for group, expected, actual in (("encoder", expected_encoder, payload["encoder_state_dict"]), ("head", expected_head, payload["head_state_dict"])):
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            shape_bad = [key for key in set(expected).intersection(actual) if tuple(expected[key].shape) != tuple(actual[key].shape)]
            dtype_bad = [key for key in set(expected).intersection(actual) if expected[key].dtype != actual[key].dtype]
            coverage.append({"cell": cell, "adaptation": adaptation, "group": group, "expected_keys": len(expected), "actual_keys": len(actual), "missing_count": len(missing), "unexpected_count": len(unexpected), "shape_mismatch_count": len(shape_bad), "dtype_mismatch_count": len(dtype_bad), "strict_load_contract": not (missing or unexpected or shape_bad or dtype_bad), "missing_keys": missing[:20], "unexpected_keys": unexpected[:20]})
        rows.append({"cell": cell, "adaptation": adaptation, "video_mode": _video_mode, "checkpoint": str(path), "checkpoint_sha256": digest(path), "payload_keys": sorted(payload.keys()), "epoch": payload.get("epoch"), "global_step": payload.get("global_step"), "seed": payload.get("seed"), "encoder_state_keys": len(payload["encoder_state_dict"]), "head_state_keys": len(payload["head_state_dict"]), "trainable_encoder_parameters_expected": trainable_encoder_count, "head_parameters_expected": head_count, "optimizer_state_entries": len(payload["optimizer_state_dict"].get("state", {})), "final_test_access": 0})
    write_csv(RECOVERY / "CHECKPOINT_INVENTORY.csv", rows)
    write_csv(RECOVERY / "PARAMETER_COVERAGE.csv", coverage)
    atomic_json(RECOVERY / "data/INPUT_METADATA.json", {**input_metadata(), "train_rows": int(np.sum(data["split_code"] == 0)), "dev_rows": int(np.sum(data["split_code"] == 1)), "target_shape": list(data["targets_native"].shape), "history_shape": list(data["history_standardized"].shape), "final_test_access": 0})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "inventory", "replay", "infer"), required=True)
    parser.add_argument("--cell", choices=sorted(CELLS))
    parser.add_argument("--path", choices=("a", "b"), help="source runner A or recovery exporter B")
    args = parser.parse_args()
    if args.mode == "fixture":
        return make_fixture()
    if args.mode == "inventory":
        return run_inventory()
    if args.cell is None:
        raise ValueError("--cell is required")
    if args.mode == "replay":
        if args.path is None:
            raise ValueError("--path is required for replay")
        return run_replay(args)
    return run_infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
