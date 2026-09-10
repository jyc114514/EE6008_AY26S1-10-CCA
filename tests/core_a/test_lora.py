from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ee6008.lora import (
    LoRAConfig,
    LoRALinear,
    assert_frozen_boundary,
    base_named_parameters,
    inject_top_block_lora,
    lora_named_parameters,
)


class TinyAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        values = self.qkv(x)[..., : self.proj.in_features]
        return self.proj(values)


class TinyBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TinyAttention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: Tensor, **_: object) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyEncoder(nn.Module):
    def __init__(self, block_count: int = 4, embed_dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TinyBlock(embed_dim) for _ in range(block_count)]
        )


def _full_forward(encoder: TinyEncoder, x: Tensor) -> Tensor:
    for block in encoder.blocks:
        x = block(x)
    return x


def test_injects_only_final_two_qkv_and_proj_with_zero_b() -> None:
    torch.manual_seed(7)
    encoder = TinyEncoder()
    before = copy.deepcopy(encoder)
    sample = torch.randn(2, 5, 8)

    inserted = inject_top_block_lora(encoder, config=LoRAConfig())

    assert inserted == (
        "blocks.2.attn.qkv",
        "blocks.2.attn.proj",
        "blocks.3.attn.qkv",
        "blocks.3.attn.proj",
    )
    for index, block in enumerate(encoder.blocks):
        for name in ("qkv", "proj"):
            module = getattr(block.attn, name)
            if index < 2:
                assert isinstance(module, nn.Linear)
            else:
                assert isinstance(module, LoRALinear)
                assert torch.count_nonzero(module.lora_B) == 0

    assert torch.allclose(_full_forward(before, sample), _full_forward(encoder, sample))
    assert all(not parameter.requires_grad for _, parameter in base_named_parameters(encoder))
    assert all(parameter.requires_grad for _, parameter in lora_named_parameters(encoder))


def test_frozen_boundary_allows_explicit_head_only() -> None:
    encoder = TinyEncoder()
    inject_top_block_lora(encoder, config=LoRAConfig(rank=4, alpha=8))
    head = nn.Linear(8, 3)

    assert_frozen_boundary(encoder, head)
