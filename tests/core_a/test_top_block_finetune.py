from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ee6008.lora import LoRAConfig, inject_top_block_lora
from ee6008.top_block_finetune import describe_encoder, forward_top_two_lora


class TinyAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, **_: object) -> Tensor:
        return self.proj(self.qkv(x)[..., : self.proj.in_features])


class TinyBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TinyAttention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, x: Tensor, **_: object) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(
            3, embed_dim, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class TinyEncoder(nn.Module):
    def __init__(self, block_count: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 2
        self.patch_size = 2
        self.tubelet_size = 2
        self.num_frames = 4
        self.use_rope = True
        self.handle_nonsquare_inputs = True
        self.modality_embedding = False
        self.patch_embed = TinyPatchEmbed(embed_dim)
        self.patch_embed_img = None
        self.blocks = nn.ModuleList(
            [TinyBlock(embed_dim) for _ in range(block_count)]
        )
        self.norms_block = nn.ModuleList([nn.LayerNorm(embed_dim)])

    def check_temporal_dim(self, shape: torch.Size) -> bool:
        del shape
        return False


def _full_forward(encoder: TinyEncoder, x: Tensor) -> Tensor:
    x = encoder.patch_embed(x)
    for block in encoder.blocks:
        x = block(x, T=2, H_patches=2, W_patches=2, mode="video")
    return encoder.norms_block[-1](x)


def test_top_block_forward_matches_full_forward_at_zero_init() -> None:
    torch.manual_seed(11)
    base = TinyEncoder()
    adapted = copy.deepcopy(base)
    inject_top_block_lora(adapted, config=LoRAConfig())
    clips = torch.randn(2, 3, 4, 4, 4)

    expected = _full_forward(base, clips)
    actual = forward_top_two_lora(
        adapted, clips, checkpoint_top_blocks=False
    )

    assert actual.shape == (2, 8, 8)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_encoder_description_records_actual_top_two_indices() -> None:
    description = describe_encoder(TinyEncoder())

    assert description["block_count"] == 4
    assert description["top_two_indices"] == [2, 3]
    assert description["final_norm"] == "norms_block[-1]"
