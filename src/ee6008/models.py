"""Explicit local-checkpoint model construction; never invoke pretrained downloads."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def load_vjepa2_1_base(
    *,
    source_root: Path,
    checkpoint: Path,
    device: torch.device,
    num_frames: int = 32,
    resolution: int = 384,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the locked V-JEPA 2.1-B encoder and load ema_encoder locally."""
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from app.vjepa_2_1.models import vision_transformer as vit

    model = vit.vit_base(
        img_size=(resolution, resolution),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "ema_encoder" not in checkpoint_obj:
        raise KeyError("checkpoint has no ema_encoder key")
    state_dict = _clean_state_dict(checkpoint_obj["ema_encoder"])
    message = model.load_state_dict(state_dict, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"checkpoint load was not exact: {message}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "checkpoint_key": "ema_encoder",
        "embed_dim": int(model.embed_dim),
        "num_frames": num_frames,
        "resolution": resolution,
        "output_contract": "B, (T/2)*(H/16)*(W/16), 768",
    }


@torch.inference_mode()
def encode_clip(
    model: torch.nn.Module, clip: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one C,T,H,W clip and return tokens plus mean-pooled features."""
    if clip.ndim != 4:
        raise ValueError(f"expected C,T,H,W clip, got {tuple(clip.shape)}")
    output = model(clip.unsqueeze(0))
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"unexpected encoder output shape {tuple(output.shape)}")
    tokens = output[0]
    pooled = tokens.mean(dim=0)
    if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
        raise FloatingPointError("encoder output contains NaN or Inf")
    return tokens, pooled
