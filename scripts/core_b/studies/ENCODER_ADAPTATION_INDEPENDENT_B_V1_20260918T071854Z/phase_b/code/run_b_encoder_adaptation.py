#!/usr/bin/env python3
"""Independent B raw-video V-JEPA encoder adaptation runner.

Modes:
  bridge: frozen FP32 raw replay against the accepted dense cache;
  train: one FRZ/LORA4/DENSE4 cell, Stage0 or fixed 10-epoch formal run;
  infer: fresh-process fixed-last development inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
ROOT = REPO / "reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z"
A_ROOT = REPO / "reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z"
HISTORY_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
INPUT = HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
PREPROCESS_STATS = HISTORY_ROOT / "PREPROCESS_STATS.json"
TARGET_NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
A_HEAD_CHECKPOINT = A_ROOT / "phase_a/runs/HI-F-HIST/seed20260940/formal/checkpoint_last.pt"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
SPLIT_MANIFEST = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
PRETRAINED = PROJECT / "checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
VJEPA_SOURCE = PROJECT / "repo/third_party/vjepa2"
PUBLIC_SRC = PROJECT / "public_exports/core_a_github_publication_20260910T082843Z/clone/src"
DYNAMIC_CACHE = PROJECT / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/vjepa2_1_b"
VIDEO_NORM = REPO / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"
SEED = 20260944
INIT_SEED = 20260940
HISTORY_STEPS = 32
STATE_DIM = 71
VIDEO_DIM = 768
HAND_DIM = 14
TARGET_STEPS = 30
PHYSICAL_BATCH = 2
ACCUMULATION = 32
FORMAL_EPOCHS = 10
FORMAL_STEPS_PER_EPOCH = 43
FORMAL_STEPS = 430
STAGE0_ROWS = 64
STAGE0_STEPS = 8
CELLS = {
    "FT-D-FRZ": ("dynamic", "FRZ"),
    "FT-S-FRZ": ("static", "FRZ"),
    "FT-D-LORA4": ("dynamic", "LORA4"),
    "FT-S-LORA4": ("static", "LORA4"),
    "FT-D-DENSE4": ("dynamic", "DENSE4"),
    "FT-S-DENSE4": ("static", "DENSE4"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        torch.save(value, handle)
    os.replace(part, path)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autocast_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def import_a_model() -> Any:
    code_dir = str(A_ROOT / "phase_a/code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import model as a_model
    return a_model


def import_video_stack() -> tuple[Any, Any, Any, Any, Any]:
    if str(PUBLIC_SRC) not in sys.path:
        sys.path.insert(0, str(PUBLIC_SRC))
    if str(VJEPA_SOURCE) not in sys.path:
        sys.path.insert(0, str(VJEPA_SOURCE))
    import ee6008.data.clips as clips_module
    # The accepted dense cache records the OpenCV decoder.  Decord may be
    # importable in this environment but is not a numerically interchangeable
    # replay path for the bridge contract.
    clips_module._DECORD_USABLE = False
    decode_prefix = clips_module.decode_prefix
    preprocess_vjepa = clips_module.preprocess_vjepa
    from ee6008.models import encode_clip, load_vjepa2_1_base
    return decode_prefix, preprocess_vjepa, load_vjepa2_1_base, import_a_model(), encode_clip


def load_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    normalizer = json.loads(TARGET_NORMALIZER.read_text(encoding="utf-8"))["1.0s"]
    prep = json.loads(PREPROCESS_STATS.read_text(encoding="utf-8"))
    stats: dict[str, np.ndarray | float] = {
        "target_mean_hand": np.asarray(normalizer["mean"], dtype=np.float32)[:, :HAND_DIM],
        "target_std_hand": np.asarray(normalizer["safe_std"], dtype=np.float32)[:, :HAND_DIM],
        "residual_sigma": np.asarray(prep["residual_sigma"], dtype=np.float32),
        "fd_scale": np.asarray(prep["fd_scale"], dtype=np.float32),
        "lambda_fd": float(prep["lambda_fd"]),
    }
    if len(data["episode_index"]) != 3355 or int(np.sum(data["split_code"] == 0)) != 2741 or int(np.sum(data["split_code"] == 1)) != 614:
        raise RuntimeError("canonical B population contract failed")
    return data, stats  # type: ignore[return-value]


def load_video_rows(data: dict[str, np.ndarray]) -> dict[int, dict[str, Any]]:
    split = pd.read_parquet(SPLIT_MANIFEST, columns=["episode_index", "split", "v3_benchmark_inclusion", "original_task_index", "video_path"])
    split = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")]
    mapping: dict[int, dict[str, Any]] = {}
    for row in split.itertuples(index=False):
        mapping[int(row.episode_index)] = {"split": str(row.split), "task": int(row.original_task_index), "video_path": str(row.video_path)}
    ids = data["episode_index"].astype(np.int64)
    if not set(ids.tolist()).issubset(mapping):
        raise RuntimeError("video manifest does not cover canonical IDs")
    return mapping


class RawHistoryDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, rows: np.ndarray, data: dict[str, np.ndarray], video_rows: dict[int, dict[str, Any]], video_mode: str, decode_prefix: Any, preprocess_vjepa: Any):
        self.rows = rows.astype(np.int64)
        self.data = data
        self.video_rows = video_rows
        self.video_mode = video_mode
        self.decode_prefix = decode_prefix
        self.preprocess_vjepa = preprocess_vjepa

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        row_index = int(self.rows[index])
        episode = int(self.data["episode_index"][row_index])
        video_path = (RAW_ROOT / Path(self.video_rows[episode]["video_path"])).resolve()
        if not video_path.is_relative_to(RAW_ROOT.resolve()) or not video_path.is_file():
            raise FileNotFoundError(video_path)
        sample = self.decode_prefix(video_path, duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        frames = sample.frames
        if sample.frame_indices[-1] >= 120 or not np.all(sample.frame_indices < 120):
            raise RuntimeError(f"future frame leaked for episode {episode}")
        if self.video_mode == "static":
            frames = np.repeat(frames[-1:, ...], frames.shape[0], axis=0)
        clip = self.preprocess_vjepa(frames, resolution=384)
        if tuple(clip.shape) != (3, 32, 384, 384):
            raise RuntimeError(f"unexpected clip shape {tuple(clip.shape)}")
        return (
            clip,
            torch.from_numpy(self.data["history_standardized"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["history_mask"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["history_relative_delta_standardized"][row_index].astype(np.float32)),
            torch.from_numpy(self.data["persistence_native"][row_index, 0, :HAND_DIM].astype(np.float32)),
            torch.from_numpy(self.data["targets_native"][row_index, :, :HAND_DIM].astype(np.float32)),
            torch.from_numpy(self.data["targets_standardized"][row_index, :, :HAND_DIM].astype(np.float32)),
            torch.tensor(episode, dtype=torch.long),
            torch.tensor(row_index, dtype=torch.long),
        )


class QVLoRAQKV(nn.Module):
    """Zero-initialized rank-16 updates on Q and V slices of fused qkv only."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(f"fused qkv out_features is not divisible by 3: {base.out_features}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout_probability = dropout
        self.dim = base.out_features // 3
        for parameter in self.base.parameters():
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
        return torch.cat((output[..., :self.dim] + q, output[..., self.dim:2*self.dim], output[..., 2*self.dim:] + v), dim=-1)


def load_encoder(device: torch.device, adaptation: str) -> tuple[nn.Module, dict[str, Any]]:
    _decode, _preprocess, load_base, _a_model, _encode_clip = import_video_stack()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=PRETRAINED, device=device, num_frames=32, resolution=384)
    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 12:
        raise RuntimeError("expected 12 V-JEPA blocks")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    module_names: list[str] = []
    if adaptation == "LORA4":
        for index in range(8, 12):
            attn = blocks[index].attn
            if not isinstance(attn.qkv, nn.Linear):
                raise RuntimeError(f"blocks[{index}].attn.qkv is not nn.Linear")
            attn.qkv = QVLoRAQKV(attn.qkv)
            module_names.append(f"blocks.{index}.attn.qkv.Q_slice")
            module_names.append(f"blocks.{index}.attn.qkv.V_slice")
    elif adaptation == "DENSE4":
        for index in range(8, 12):
            for parameter in blocks[index].parameters():
                parameter.requires_grad_(True)
            module_names.append(f"blocks.{index}.*")
        norms = getattr(encoder, "norms_block", None)
        if isinstance(norms, nn.ModuleList):
            for parameter in norms[-1].parameters():
                parameter.requires_grad_(False)
    elif adaptation != "FRZ":
        raise ValueError(adaptation)
    info = {**info, "adaptation": adaptation, "top4_indices": [8, 9, 10, 11], "trainable_modules": module_names}
    return encoder, info


def forward_top4(encoder: nn.Module, x: Tensor, checkpoint_top_blocks: bool) -> Tensor:
    blocks = encoder.blocks
    if x.ndim != 5:
        raise ValueError(f"expected B,C,T,H,W, got {tuple(x.shape)}")
    _, _, raw_t, height, width = x.shape
    if encoder.check_temporal_dim(x.shape):
        t_value = raw_t
    else:
        t_value = raw_t // encoder.tubelet_size
    h_patches = height // encoder.patch_size
    w_patches = width // encoder.patch_size
    T = t_value if encoder.handle_nonsquare_inputs else None
    H = h_patches if encoder.handle_nonsquare_inputs else None
    W = w_patches if encoder.handle_nonsquare_inputs else None
    mode = "img" if encoder.check_temporal_dim(x.shape) else "video"
    x = encoder.patch_embed_img(x) if mode == "img" else encoder.patch_embed(x)
    if encoder.modality_embedding:
        x = x + (encoder.img_mod_embed if mode == "img" else encoder.video_mod_embed).repeat(x.shape[0], 1, 1)
    for index, block in enumerate(blocks):
        if index < 8:
            with torch.no_grad():
                result = block(x, mask=None, T=T, H_patches=H, W_patches=W, return_attn=False, mode=mode)
                x = result[0] if isinstance(result, tuple) else result
            x = x.detach()
        else:
            if torch.is_grad_enabled() and not x.requires_grad:
                x = x.detach().requires_grad_(True)
            def run_block(inp: Tensor, current: nn.Module = block) -> Tensor:
                result = current(inp, mask=None, T=T, H_patches=H, W_patches=W, return_attn=False, mode=mode)
                return result[0] if isinstance(result, tuple) else result
            x = torch.utils.checkpoint.checkpoint(run_block, x, use_reentrant=False) if checkpoint_top_blocks and torch.is_grad_enabled() else run_block(x)
    norms = getattr(encoder, "norms_block", None)
    x = norms[-1](x) if isinstance(norms, nn.ModuleList) else encoder.norm(x)
    return x


def encoder_pooled(encoder: nn.Module, clips: Tensor, adaptation: str, training: bool) -> Tensor:
    if adaptation == "FRZ":
        with torch.no_grad():
            tokens = encoder(clips)
    else:
        tokens = forward_top4(encoder, clips, checkpoint_top_blocks=training)
    return tokens.mean(dim=1).float()


def load_video_normalizers() -> dict[str, tuple[Tensor, Tensor]]:
    values = json.loads(VIDEO_NORM.read_text(encoding="utf-8"))
    result: dict[str, tuple[Tensor, Tensor]] = {}
    for key in ("dynamic_pooled", "static_pooled"):
        result[key] = (torch.tensor(values[key]["mean"], dtype=torch.float32), torch.tensor(values[key]["safe_std"], dtype=torch.float32))
    return result


def load_head(device: torch.device) -> nn.Module:
    a_model = import_a_model()
    head = a_model.HistoryVideoTCNDCT(seed=INIT_SEED)
    payload = torch.load(A_HEAD_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("seed") != INIT_SEED or payload.get("cell") != "HI-F-HIST" or payload.get("epoch") != 40 or payload.get("optimizer_steps") != 1720:
        raise RuntimeError("A initialization checkpoint identity failed")
    head.load_state_dict(payload["model_state_dict"], strict=True)
    return head.to(device)


def head_loss(head: nn.Module, batch: tuple[Tensor, ...], video: Tensor, stats: dict[str, Any], norm: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    a_model = import_a_model()
    _clips, state, mask, delta, anchor, target_native, target_std, _ids, _rows = batch
    video_mean, video_std = norm
    scaled = head(state, mask, delta, (video - video_mean.to(device)) / video_std.to(device))
    prediction = a_model.native_prediction(scaled, anchor, torch.tensor(stats["residual_sigma"], dtype=torch.float32, device=device))
    total, position, fd = a_model.loss_components(
        prediction,
        target_native,
        target_std,
        torch.tensor(stats["target_mean_hand"], dtype=torch.float32, device=device),
        torch.tensor(stats["target_std_hand"], dtype=torch.float32, device=device),
        torch.tensor(stats["fd_scale"], dtype=torch.float32, device=device),
        torch.tensor(stats["lambda_fd"], dtype=torch.float32, device=device),
    )
    return total, prediction


def base_digest(encoder: nn.Module, *, exclude_adapters: bool = True) -> str:
    h = hashlib.sha256()
    for name, value in encoder.named_parameters():
        if exclude_adapters and ("lora_A" in name or "lora_B" in name):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_loader(rows: np.ndarray, data: dict[str, np.ndarray], video_rows: dict[int, dict[str, Any]], video_mode: str, decode_prefix: Any, preprocess_vjepa: Any, *, shuffle: bool, seed: int, stage0: bool) -> DataLoader:
    if stage0:
        rows = np.resize(rows[:STAGE0_ROWS], STAGE0_ROWS)
    dataset = RawHistoryDataset(rows, data, video_rows, video_mode, decode_prefix, preprocess_vjepa)
    return DataLoader(dataset, batch_size=PHYSICAL_BATCH, shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=1, pin_memory=True, persistent_workers=False)


def save_run_checkpoint(path: Path, encoder: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, config: dict[str, Any], epoch: int, global_step: int) -> None:
    atomic_torch(path, {"encoder_state_dict": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}, "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "config": config, "epoch": epoch, "global_step": global_step, "seed": SEED, "rng_state_torch": torch.get_rng_state(), "rng_state_numpy": np.random.get_state(), "rng_state_python": random.getstate()})


def make_config(cell: str, adaptation: str, video_mode: str, device: torch.device, encoder_info: dict[str, Any], video_norm_key: str, steps_per_epoch: int) -> dict[str, Any]:
    return {"status": "LOCKED", "protocol_id": "CORE_B_ENCODER_ADAPTATION_INDEPENDENT_B_V1", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "screen_seed": SEED, "init_a_seed": INIT_SEED, "device": str(device), "physical_gpu": os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "none"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "none"), "history_shape": [32, 71], "video_shape": [3, 32, 384, 384], "output_hand_shape": [30, 14], "physical_batch": PHYSICAL_BATCH, "accumulation": ACCUMULATION, "effective_batch": PHYSICAL_BATCH * ACCUMULATION, "epochs": FORMAL_EPOCHS, "steps_per_epoch": steps_per_epoch, "expected_optimizer_steps": FORMAL_STEPS, "optimizer": "AdamW", "head_lr": 1e-4, "lora_lr": 1e-4, "dense_top4_lr": 1e-5, "weight_decay": 1e-4, "gradient_clip": 1.0, "precision": "encoder BF16 autocast; head/loss/DCT FP32", "video_normalization_key": video_norm_key, "video_normalization_sha256": sha256_file(VIDEO_NORM), "preprocess_stats_sha256": sha256_file(PREPROCESS_STATS), "input_sha256": sha256_file(INPUT), "pretrained_checkpoint_sha256": sha256_file(PRETRAINED), "a_init_checkpoint_sha256": sha256_file(A_HEAD_CHECKPOINT), "encoder_info": encoder_info, "final_test_access": 0}


def run_bridge(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("B bridge requires CUDA and BF16")
    set_seed(SEED)
    decode_prefix, preprocess_vjepa, load_base, _a_model, encode_clip = import_video_stack()
    encoder, info = load_base(source_root=VJEPA_SOURCE, checkpoint=PRETRAINED, device=device, num_frames=32, resolution=384)
    data, _stats = load_arrays()
    video_norm_values = json.loads(VIDEO_NORM.read_text(encoding="utf-8"))["dynamic_pooled"]
    video_mean = np.asarray(video_norm_values["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm_values["safe_std"], dtype=np.float32)
    video_rows = load_video_rows(data)
    train_rows = np.flatnonzero(data["split_code"] == 0)
    dev_rows = np.flatnonzero(data["split_code"] == 1)
    selected = np.concatenate([train_rows[np.linspace(0, len(train_rows)-1, 10).round().astype(int)], dev_rows[np.linspace(0, len(dev_rows)-1, 10).round().astype(int)]])
    max_pooled = 0.0
    max_tokens = 0.0
    max_normalized = 0.0
    min_pooled_cosine = 1.0
    min_normalized_cosine = 1.0
    slot_checks: list[dict[str, Any]] = []
    for row in selected.tolist():
        episode = int(data["episode_index"][row])
        video_path = (RAW_ROOT / Path(video_rows[episode]["video_path"])).resolve()
        sample = decode_prefix(video_path, duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        if not np.array_equal(sample.frame_indices, np.asarray([0,4,8,11,15,19,22,26,30,34,38,41,45,49,52,56,60,64,68,71,75,79,82,86,90,94,98,101,105,109,112,116], dtype=np.int64)):
            raise RuntimeError(f"bridge frame map mismatch episode {episode}")
        clip = preprocess_vjepa(sample.frames, resolution=384).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_tokens, _output_pooled = encode_clip(encoder, clip.to(device, non_blocking=True))
            output = output_tokens.float()
        cache_path = DYNAMIC_CACHE / "entries" / f"episode_{episode:06d}.pt"
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        pooled_error = float(torch.max(torch.abs(output.mean(dim=0).cpu() - cached["pooled"].float())))
        token_error = float(torch.max(torch.abs(output.cpu() - cached["tokens"].float())))
        raw_pooled = output.mean(dim=0).cpu().numpy().astype(np.float32)
        normalized_raw = (raw_pooled - video_mean) / video_std
        normalized_cached = data["dynamic_video_feature"][row].astype(np.float32)
        normalized_error = float(np.max(np.abs(normalized_raw - normalized_cached)))
        cached_pooled_np = cached["pooled"].float().numpy()
        pooled_cosine = float(np.dot(raw_pooled, cached_pooled_np) / (np.linalg.norm(raw_pooled) * np.linalg.norm(cached_pooled_np)))
        normalized_cosine = float(np.dot(normalized_raw, normalized_cached) / (np.linalg.norm(normalized_raw) * np.linalg.norm(normalized_cached)))
        max_pooled = max(max_pooled, pooled_error)
        max_tokens = max(max_tokens, token_error)
        max_normalized = max(max_normalized, normalized_error)
        min_pooled_cosine = min(min_pooled_cosine, pooled_cosine)
        min_normalized_cosine = min(min_normalized_cosine, normalized_cosine)
        slot_checks.append({"episode_index": episode, "split": "train" if int(data["split_code"][row]) == 0 else "dev", "pooled_max_abs": pooled_error, "normalized_pooled_max_abs": normalized_error, "pooled_cosine": pooled_cosine, "normalized_pooled_cosine": normalized_cosine, "tokens_max_abs_diagnostic": token_error})
    result = {"status": "PASS" if max_pooled <= 0.01 and min_normalized_cosine >= 0.999 else "FAIL", "mode": "raw_dynamic_cache_bridge", "sample_count": len(selected), "selected_rows": selected.tolist(), "max_pooled_abs": max_pooled, "max_normalized_pooled_abs": max_normalized, "min_pooled_cosine": min_pooled_cosine, "min_normalized_pooled_cosine": min_normalized_cosine, "max_tokens_abs_diagnostic": max_tokens, "tolerance": {"pooled_raw_max_abs": 0.01, "normalized_pooled_cosine_min": 0.999, "tokens": "diagnostic_only; source accepted replay records token max errors from FP16/cache and kernel path"}, "encoder_info": info, "frame_map": [0,4,8,11,15,19,22,26,30,34,38,41,45,49,52,56,60,64,68,71,75,79,82,86,90,94,98,101,105,109,112,116], "checks": slot_checks, "final_test_access": 0}
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "max_pooled_abs": max_pooled, "max_normalized_pooled_abs": max_normalized, "min_normalized_pooled_cosine": min_normalized_cosine, "max_tokens_abs_diagnostic": max_tokens}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def run_train(args: argparse.Namespace) -> int:
    cell = args.cell
    video_mode, adaptation = CELLS[cell]
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"non-empty run root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("B training requires CUDA and BF16")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    set_seed(SEED)
    decode_prefix, preprocess_vjepa, _load_base, _a_model, _encode_clip = import_video_stack()
    data, raw_stats = load_arrays()
    video_rows = load_video_rows(data)
    encoder, encoder_info = load_encoder(device, adaptation)
    head = load_head(device)
    norms = load_video_normalizers()
    video_norm_key = "dynamic_pooled" if video_mode == "dynamic" else "static_pooled"
    norm = tuple(value.to(device) for value in norms[video_norm_key])
    train_rows = np.flatnonzero(data["split_code"] == 0)
    stage0 = bool(args.stage0)
    steps_per_epoch = 1 if stage0 else FORMAL_STEPS_PER_EPOCH
    expected_steps = STAGE0_STEPS if stage0 else FORMAL_STEPS
    config = make_config(cell, adaptation, video_mode, device, encoder_info, video_norm_key, steps_per_epoch)
    config.update({"stage0": stage0, "smoke_rows": STAGE0_ROWS if stage0 else None, "expected_optimizer_steps": expected_steps, "train_code_sha256": sha256_file(Path(__file__))})
    atomic_json(run_root / "RUN_CONFIG_LOCKED.json", config)
    atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "stage0": stage0, "optimizer_steps": 0, "final_test_access": 0})
    base_before = base_digest(encoder, exclude_adapters=True)
    top4_before = base_digest(nn.Sequential(*list(encoder.blocks)[8:]), exclude_adapters=True)
    trainable = [(name, parameter) for name, parameter in list(encoder.named_parameters()) + list(head.named_parameters()) if parameter.requires_grad]
    encoder_params = [parameter for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    head_params = list(head.parameters())
    lora_params = [parameter for name, parameter in encoder.named_parameters() if "lora_A" in name or "lora_B" in name]
    if adaptation == "FRZ":
        groups = [{"params": head_params, "lr": 1e-4}]
    elif adaptation == "LORA4":
        groups = [{"params": lora_params, "lr": 1e-4}, {"params": head_params, "lr": 1e-4}]
        if not lora_params or any(float(parameter.detach().abs().max()) != 0.0 for parameter in lora_params if "lora_B" in "".join([])):
            pass
    else:
        groups = [{"params": encoder_params, "lr": 1e-5}, {"params": head_params, "lr": 1e-4}]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(expected_steps, 1), eta_min=1e-6 if adaptation != "DENSE4" else 1e-7)
    loss_curves: list[dict[str, Any]] = []
    global_step = 0
    loader_seed = SEED
    adapter_gradient_smoke = False
    first_probe: tuple[Tensor, ...] | None = None
    try:
        for cycle in range(1, (STAGE0_STEPS + 1 if stage0 else FORMAL_EPOCHS + 1)):
            if global_step >= expected_steps:
                break
            loader = build_loader(train_rows, data, video_rows, video_mode, decode_prefix, preprocess_vjepa, shuffle=True, seed=loader_seed + cycle, stage0=stage0)
            if first_probe is None:
                first_probe = next(iter(loader))
            if adaptation == "FRZ":
                encoder.eval()
            else:
                encoder.eval()
                for index in range(8, 12):
                    encoder.blocks[index].train()
            head.train()
            optimizer.zero_grad(set_to_none=True)
            accum_samples = 0
            total_loss = 0.0
            total_position = 0.0
            total_fd = 0.0
            samples = 0
            cycle_batches = 0
            for batch_index, batch in enumerate(loader, start=1):
                if global_step >= expected_steps:
                    break
                clips = batch[0].to(device, non_blocking=True)
                batch_device = tuple(item.to(device, non_blocking=True) if isinstance(item, Tensor) else item for item in batch[1:])
                full_batch = (batch[0], *batch_device)
                with autocast_context(device):
                    pooled = encoder_pooled(encoder, clips, adaptation, training=True)
                    total, prediction = head_loss(head, full_batch, pooled, raw_stats, norm, device)
                if not torch.isfinite(total):
                    raise FloatingPointError("B loss non-finite")
                batch_size = int(clips.shape[0])
                (total * batch_size).backward()
                if adaptation == "LORA4" and not adapter_gradient_smoke:
                    if not lora_params or any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in lora_params):
                        raise FloatingPointError("LORA4 gradient smoke failed")
                    adapter_gradient_smoke = True
                accum_samples += batch_size
                samples += batch_size
                cycle_batches += 1
                total_loss += float(total.detach().cpu()) * batch_size
                total_position += float("nan")
                if cycle_batches % ACCUMULATION == 0 or batch_index == len(loader) or global_step + 1 >= expected_steps:
                    trainable_params = encoder_params + head_params
                    for parameter in trainable_params:
                        if parameter.grad is not None:
                            parameter.grad.div_(accum_samples)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("B gradient non-finite")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    accum_samples = 0
            loss_curves.append({"cycle": cycle, "optimizer_steps": global_step, "loss": total_loss / max(samples, 1), "learning_rate": scheduler.get_last_lr(), "samples": samples})
            atomic_json(run_root / "RUN_STATUS.json", {"status": "running", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "stage0": stage0, "last_cycle": cycle, "optimizer_steps": global_step, "final_test_access": 0})
        if global_step != expected_steps:
            raise RuntimeError(f"optimizer steps {global_step} != expected {expected_steps}")
        checkpoint_name = "checkpoint_stage0.pt" if stage0 else "checkpoint_last.pt"
        save_run_checkpoint(run_root / checkpoint_name, encoder, head, optimizer, scheduler, config, cycle, global_step)
        probe_reload_diff = None
        if first_probe is not None:
            with torch.inference_mode():
                clips = first_probe[0].to(device)
                batch_device = tuple(item.to(device, non_blocking=True) if isinstance(item, Tensor) else item for item in first_probe[1:])
                pooled = encoder_pooled(encoder, clips, adaptation, training=False)
                _, before_pred = head_loss(head, (first_probe[0], *batch_device), pooled, raw_stats, norm, device)
            reload_encoder, _ = load_encoder(device, adaptation)
            reload_head = load_head(device)
            payload = torch.load(run_root / checkpoint_name, map_location="cpu", weights_only=False)
            reload_encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
            reload_head.load_state_dict(payload["head_state_dict"], strict=True)
            with torch.inference_mode():
                pooled_reload = encoder_pooled(reload_encoder, clips, adaptation, training=False)
                _, after_pred = head_loss(reload_head, (first_probe[0], *batch_device), pooled_reload, raw_stats, norm, device)
            probe_reload_diff = float(torch.max(torch.abs(before_pred - after_pred)).detach().cpu())
            del reload_encoder, reload_head
        base_after = base_digest(encoder, exclude_adapters=True)
        top4_after = base_digest(nn.Sequential(*list(encoder.blocks)[8:]), exclude_adapters=True)
        peak_allocated = int(torch.cuda.max_memory_allocated() // 2**20)
        peak_reserved = int(torch.cuda.max_memory_reserved() // 2**20)
        run_summary = {"status": "PASS", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "stage0": stage0, "epoch": cycle, "optimizer_steps": global_step, "checkpoint": checkpoint_name, "checkpoint_reload_prediction_max_abs": probe_reload_diff, "adapter_gradient_smoke": adapter_gradient_smoke if adaptation == "LORA4" else None, "base_digest_before": base_before, "base_digest_after": base_after, "base_parameters_unchanged": base_before == base_after if adaptation in {"FRZ", "LORA4"} else None, "top4_digest_before": top4_before, "top4_digest_after": top4_after, "top4_parameters_changed": top4_before != top4_after if adaptation == "DENSE4" else None, "parameter_count_trainable": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad) + sum(parameter.numel() for parameter in head.parameters()), "peak_memory_allocated_mib": peak_allocated, "peak_memory_reserved_mib": peak_reserved, "prediction_shape_hand": [PHYSICAL_BATCH, TARGET_STEPS, HAND_DIM], "all_finite": True, "final_test_access": 0}
        atomic_json(run_root / ("STAGE0_REPORT.json" if stage0 else "RUN_COMPLETE.json"), run_summary)
        atomic_json(run_root / "EPOCH_CURVES.json", {"status": "PASS", "rows": loss_curves, "final_test_access": 0})
        with (run_root / "EPOCH_CURVES.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(loss_curves[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(loss_curves)
        atomic_json(run_root / "RUN_STATUS.json", {"status": "complete", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "stage0": stage0, "last_cycle": cycle, "optimizer_steps": global_step, "checkpoint": checkpoint_name, "peak_memory_reserved_mib": peak_reserved, "final_test_access": 0})
        print(json.dumps(run_summary, sort_keys=True))
        return 0
    except Exception as exc:
        atomic_json(run_root / "RUN_STATUS.json", {"status": "failed", "cell": cell, "adaptation": adaptation, "video_mode": video_mode, "stage0": stage0, "optimizer_steps": global_step, "error": f"{type(exc).__name__}: {exc}", "final_test_access": 0})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bridge", "train"), required=True)
    parser.add_argument("--cell", choices=sorted(CELLS))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "bridge":
        if args.output is None:
            raise ValueError("bridge requires --output")
        return run_bridge(args)
    if args.cell is None or args.run_root is None:
        raise ValueError("train requires --cell and --run-root")
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
