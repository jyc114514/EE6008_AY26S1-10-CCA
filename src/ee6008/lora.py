"""Minimal, auditable LoRA insertion for the local V-JEPA encoder."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    """Fixed LoRA hyperparameters for the exploratory pilot."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "scaling": self.scaling}

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """A frozen Linear plus a zero-initialized low-rank residual.

    The wrapped base module is retained verbatim.  ``lora_B`` is initialized to
    zero, so the module is numerically equivalent to the original Linear at
    step zero (up to the surrounding dtype/autocast contract).
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base)!r}")
        config.validate()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = config.rank
        self.alpha = config.alpha
        self.dropout_probability = config.dropout
        self.scaling = config.scaling
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        self._merged = False
        self.register_buffer(
            "_merged_state", torch.zeros((), dtype=torch.bool), persistent=True
        )
        self._weight_before_merge: Tensor | None = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def merged(self) -> bool:
        """Whether the low-rank update is currently fused into ``base``."""

        return self._merged

    def delta_weight(self) -> Tensor:
        """Return the standard ``alpha / rank * B @ A`` weight update."""

        if self.lora_A.shape != (self.rank, self.in_features):
            raise RuntimeError("LoRA A shape does not match the wrapped Linear")
        if self.lora_B.shape != (self.out_features, self.rank):
            raise RuntimeError("LoRA B shape does not match the wrapped Linear")
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge(self) -> None:
        """Fuse the adapter into the base weight exactly once.

        Adapter parameters remain intact while merged so that a checkpoint can
        preserve the complete adapter state.  ``forward`` skips the residual
        path while the fused state is active.
        """

        if self.merged:
            raise RuntimeError("LoRALinear is already merged")
        with torch.no_grad():
            self._weight_before_merge = self.base.weight.detach().clone()
            delta = self.delta_weight().to(
                device=self.base.weight.device, dtype=self.base.weight.dtype
            )
            self.base.weight.add_(delta)
            self._merged_state.fill_(True)
            self._merged = True

    def unmerge(self) -> None:
        """Restore the pre-merge base weight exactly once."""

        if not self.merged:
            raise RuntimeError("LoRALinear is not merged")
        with torch.no_grad():
            if self._weight_before_merge is not None:
                self.base.weight.copy_(
                    self._weight_before_merge.to(
                        device=self.base.weight.device, dtype=self.base.weight.dtype
                    )
                )
            else:
                delta = self.delta_weight().to(
                    device=self.base.weight.device, dtype=self.base.weight.dtype
                )
                self.base.weight.sub_(delta)
            self._weight_before_merge = None
            self._merged_state.fill_(False)
            self._merged = False

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Checkpoints created before the state flag was introduced remain
        # loadable and are unmerged by default.
        state_key = prefix + "_merged_state"
        if state_key not in state_dict:
            state_dict[state_key] = self._merged_state.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._merged = bool(self._merged_state.item())
        self._weight_before_merge = None

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base(x)
        if self.merged:
            return base_output
        dropped = F.dropout(
            x, p=self.dropout_probability, training=self.training
        )
        update = F.linear(F.linear(dropped, self.lora_A), self.lora_B)
        return base_output + self.scaling * update


def merge_lora_modules(module: nn.Module) -> tuple[LoRALinear, ...]:
    """Merge all LoRA modules in a model with a transactional exact-once guard."""

    adapters = tuple(item for item in module.modules() if isinstance(item, LoRALinear))
    if not adapters:
        raise ValueError("module contains no LoRA adapters")
    if any(adapter.merged for adapter in adapters):
        raise RuntimeError("one or more LoRALinear modules are already merged")
    merged: list[LoRALinear] = []
    try:
        for adapter in adapters:
            adapter.merge()
            merged.append(adapter)
    except Exception:
        for adapter in reversed(merged):
            adapter.unmerge()
        raise
    return tuple(merged)


def unmerge_lora_modules(adapters: Iterable[LoRALinear]) -> None:
    """Unmerge a previously returned adapter set with an exact-once guard."""

    selected = tuple(adapters)
    if not selected:
        raise ValueError("no LoRA adapters were supplied for unmerge")
    if len({id(adapter) for adapter in selected}) != len(selected):
        raise ValueError("duplicate LoRA adapter supplied for unmerge")
    if any(not adapter.merged for adapter in selected):
        raise RuntimeError("one or more LoRALinear modules are not merged")
    for adapter in reversed(selected):
        adapter.unmerge()


def freeze_module_parameters(module: nn.Module) -> None:
    """Freeze every parameter in a module before adding explicit adapters."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)


def inject_top_block_lora(
    encoder: nn.Module,
    *,
    config: LoRAConfig,
    block_indices: tuple[int, int] | None = None,
    target_modules: tuple[str, str] = ("qkv", "proj"),
) -> tuple[str, ...]:
    """Insert LoRA into qkv/proj of the final two actual encoder blocks.

    The function discovers the block container and validates module types; it
    does not rely on a guessed block count or silently adapt another location.
    """

    config.validate()
    blocks = getattr(encoder, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
        raise ValueError("encoder must expose at least two blocks in .blocks")
    expected = (len(blocks) - 2, len(blocks) - 1)
    indices = expected if block_indices is None else tuple(block_indices)
    if indices != expected:
        raise ValueError(f"only final two blocks are allowed, expected {expected}")
    if len(target_modules) != 2 or set(target_modules) != {"qkv", "proj"}:
        raise ValueError("the pilot adapts exactly attention qkv and proj")

    freeze_module_parameters(encoder)
    inserted: list[str] = []
    for index in indices:
        block = blocks[index]
        attention = getattr(block, "attn", None)
        if attention is None:
            raise ValueError(f"block {index} has no attention module")
        for name in target_modules:
            current = getattr(attention, name, None)
            if isinstance(current, LoRALinear):
                raise TypeError(
                    f"LoRA already inserted at blocks[{index}].attn.{name}"
                )
            if not isinstance(current, nn.Linear):
                raise TypeError(
                    f"blocks[{index}].attn.{name} is {type(current)!r}, not nn.Linear"
                )
            setattr(attention, name, LoRALinear(current, config))
            inserted.append(f"blocks.{index}.attn.{name}")
    return tuple(inserted)


def lora_named_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Yield only adapter parameters, preserving stable names."""

    for name, parameter in module.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield name, parameter


def trainable_named_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Yield trainable parameters for optimizer-membership audits."""

    return ((name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad)


def base_named_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Yield all non-LoRA parameters, including wrapped base weights."""

    return (
        (name, parameter)
        for name, parameter in module.named_parameters()
        if ".lora_A" not in name and ".lora_B" not in name
    )


def parameter_counts(encoder: nn.Module, head: nn.Module) -> dict[str, int | float]:
    """Return base/adapter/head counts for a provenance report."""

    base = sum(parameter.numel() for _, parameter in base_named_parameters(encoder))
    lora = sum(parameter.numel() for _, parameter in lora_named_parameters(encoder))
    head_count = sum(parameter.numel() for parameter in head.parameters())
    trainable = lora + head_count
    total = base + lora + head_count
    return {
        "total_base_parameters": base,
        "trainable_lora_parameters": lora,
        "trainable_head_parameters": head_count,
        "total_trainable_parameters": trainable,
        "total_model_and_head_parameters": total,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
    }


def adapter_state_dict(encoder: nn.Module) -> dict[str, Tensor]:
    """Return a CPU adapter-only state dictionary."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in lora_named_parameters(encoder)
    }


def assert_frozen_boundary(encoder: nn.Module, head: nn.Module) -> None:
    """Raise if anything except LoRA and the explicit head is trainable."""

    bad_encoder = [name for name, p in base_named_parameters(encoder) if p.requires_grad]
    if bad_encoder:
        raise AssertionError(f"base encoder parameters are trainable: {bad_encoder[:8]}")
    missing_lora = [name for name, p in lora_named_parameters(encoder) if not p.requires_grad]
    if missing_lora:
        raise AssertionError(f"LoRA parameters are frozen: {missing_lora[:8]}")
    bad_head = [name for name, p in head.named_parameters() if not p.requires_grad]
    if bad_head:
        raise AssertionError(f"head parameters are frozen: {bad_head[:8]}")
