"""Official-forward-compatible top-block V-JEPA LoRA execution helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _block_without_attention_output(
    block: nn.Module,
    x: Tensor,
    *,
    masks: list[Tensor] | None,
    T: int | None,
    H_patches: int | None,
    W_patches: int | None,
    mode: str,
) -> Tensor:
    result = block(
        x,
        mask=masks,
        T=T,
        H_patches=H_patches,
        W_patches=W_patches,
        return_attn=False,
        mode=mode,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


def forward_top_two_lora(
    encoder: nn.Module,
    x: Tensor,
    *,
    checkpoint_top_blocks: bool = True,
) -> Tensor:
    """Run the official V-JEPA path while detaching only below block 10.

    This mirrors ``app/vjepa_2_1/models/vision_transformer.py``: patch
    embedding, modality embedding, RoPE position handling, all 12 blocks and
    the final hierarchical norm are retained.  Lower blocks run under
    ``no_grad``; the activation entering the final two blocks is detached but
    remains a differentiable input to those blocks.
    """

    if x.ndim not in (4, 5):
        raise ValueError(f"expected image/video tensor, got {tuple(x.shape)}")
    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
        raise ValueError("encoder must expose .blocks with at least two blocks")

    if x.ndim == 4:
        _, _, height, width = x.shape
        T = 1
    else:
        _, _, raw_t, height, width = x.shape
        if encoder.check_temporal_dim(x.shape):
            T = raw_t
        else:
            T = raw_t // encoder.tubelet_size
    H_patches = height // encoder.patch_size
    W_patches = width // encoder.patch_size
    if not encoder.handle_nonsquare_inputs:
        T = H_patches = W_patches = None

    masks = None
    if not encoder.use_rope:
        pos_embed = encoder.interpolate_pos_encoding(x, encoder.pos_embed)

    if encoder.check_temporal_dim(x.shape):
        if encoder.patch_embed_img is None:
            raise RuntimeError("model requested image temporal mode without patch_embed_img")
        x = encoder.patch_embed_img(x)
        mode = "img"
        if encoder.modality_embedding:
            x = x + encoder.img_mod_embed.repeat(x.shape[0], 1, 1)
    else:
        x = encoder.patch_embed(x)
        mode = "video"
        if encoder.modality_embedding:
            x = x + encoder.video_mod_embed.repeat(x.shape[0], 1, 1)

    if not encoder.use_rope:
        x = x + pos_embed

    top_start = len(blocks) - 2
    for index, block in enumerate(blocks):
        if index < top_start:
            with torch.no_grad():
                x = _block_without_attention_output(
                    block,
                    x,
                    masks=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    mode=mode,
                )
            x = x.detach()
            continue

        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_(True)

        def run_block(inp: Tensor, current: nn.Module = block) -> Tensor:
            return _block_without_attention_output(
                current,
                inp,
                masks=masks,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                mode=mode,
            )

        if checkpoint_top_blocks and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(
                run_block, x, use_reentrant=False
            )
        else:
            x = run_block(x)

    norms = getattr(encoder, "norms_block", None)
    if isinstance(norms, nn.ModuleList):
        x = norms[-1](x)
    else:
        norm = getattr(encoder, "norm", None)
        if norm is None:
            raise RuntimeError("encoder exposes neither norms_block nor norm")
        x = norm(x)
    if x.ndim != 3:
        raise RuntimeError(f"unexpected top-block output shape: {tuple(x.shape)}")
    return x


def describe_encoder(encoder: nn.Module) -> dict[str, object]:
    """Enumerate the concrete model structure used by the pilot."""

    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("encoder has no ModuleList named blocks")
    block_records = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        block_records.append(
            {
                "index": index,
                "block_class": type(block).__name__,
                "attention_class": type(attention).__name__ if attention else None,
                "qkv_module": f"blocks[{index}].attn.qkv",
                "qkv_class": type(getattr(attention, "qkv", None)).__name__
                if attention
                else None,
                "proj_module": f"blocks[{index}].attn.proj",
                "proj_class": type(getattr(attention, "proj", None)).__name__
                if attention
                else None,
                "has_norm1": hasattr(block, "norm1"),
                "has_norm2": hasattr(block, "norm2"),
                "has_mlp": hasattr(block, "mlp"),
            }
        )
    return {
        "encoder_class": type(encoder).__name__,
        "block_container": "blocks",
        "block_count": len(blocks),
        "embed_dim": int(encoder.embed_dim),
        "num_heads": int(encoder.num_heads),
        "patch_embed_class": type(encoder.patch_embed).__name__,
        "patch_embed_img_class": type(encoder.patch_embed_img).__name__
        if getattr(encoder, "patch_embed_img", None) is not None
        else None,
        "patch_size": int(encoder.patch_size),
        "tubelet_size": int(encoder.tubelet_size),
        "num_frames": int(encoder.num_frames),
        "use_rope": bool(encoder.use_rope),
        "modality_embedding": bool(encoder.modality_embedding),
        "final_norm": "norms_block[-1]"
        if hasattr(encoder, "norms_block")
        else "norm",
        "block_records": block_records,
        "top_two_indices": [len(blocks) - 2, len(blocks) - 1],
        "official_forward_source": "app/vjepa_2_1/models/vision_transformer.py",
    }
