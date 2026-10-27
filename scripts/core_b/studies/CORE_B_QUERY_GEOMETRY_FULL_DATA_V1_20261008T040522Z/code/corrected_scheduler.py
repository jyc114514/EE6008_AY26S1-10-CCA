"""Corrected per-optimizer-update cosine schedule with auditable call counts."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import torch


def closed_form_lr(update_count: int, *, initial_lr: float = 1e-3, eta_min: float = 1e-5,
                   t_max: int = 11120) -> float:
    if not 0 <= update_count <= t_max:
        raise ValueError(f"update_count must be in [0,{t_max}]")
    return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * update_count / t_max))


class UpdateDrivenCosineAnnealingLR:
    """Step the wrapped scheduler once in the optimizer's successful-step hook.

    The legacy trainer calls scheduler.step() once at epoch end. That call is
    recorded but intentionally does not advance the wrapped scheduler.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, T_max: int, eta_min: float):
        if T_max != 11120 or eta_min != 1e-5:
            raise ValueError("this study locks T_max=11120 and eta_min=1e-5")
        if not hasattr(optimizer, "register_step_post_hook"):
            raise RuntimeError("installed PyTorch lacks optimizer post-step hooks")
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.inner = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
        self.step_calls = 0
        self.legacy_epoch_step_calls = 0
        self.current_epoch = 0
        self.update_records: list[dict[str, Any]] = []
        self._hook_handle = optimizer.register_step_post_hook(self._after_optimizer_step)
        self._assert_lr_matches(0)

    def _assert_lr_matches(self, update_count: int) -> None:
        actual = float(self.optimizer.param_groups[0]["lr"])
        expected = closed_form_lr(update_count, eta_min=self.eta_min, t_max=self.T_max)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-11):
            raise AssertionError(
                f"optimizer/scheduler LR mismatch at update {update_count}: {actual} != {expected}"
            )

    def _after_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...],
                              kwargs: dict[str, Any]) -> None:
        if self.step_calls >= self.T_max:
            raise AssertionError("scheduler received more optimizer updates than T_max")
        before = float(optimizer.param_groups[0]["lr"])
        count_before = self.step_calls
        self.inner.step()
        self.step_calls += 1
        after = float(optimizer.param_groups[0]["lr"])
        self._assert_lr_matches(self.step_calls)
        self.update_records.append({
            "epoch": self.current_epoch,
            "update_count_before": count_before,
            "update_count_after": self.step_calls,
            "lr_used_for_update": before,
            "lr_after_update": after,
        })

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def step(self, *args: Any, **kwargs: Any) -> None:
        # Compatibility with the old trainer's one call after each epoch.
        self.legacy_epoch_step_calls += 1

    def get_last_lr(self) -> list[float]:
        return self.inner.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "inner_state": self.inner.state_dict(),
            "step_calls": self.step_calls,
            "legacy_epoch_step_calls": self.legacy_epoch_step_calls,
            "current_epoch": self.current_epoch,
            "update_records": self.update_records,
            "T_max": self.T_max,
            "eta_min": self.eta_min,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("T_max"), state.get("eta_min")) != (self.T_max, self.eta_min):
            raise AssertionError("resume scheduler contract mismatch")
        self.inner.load_state_dict(state["inner_state"])
        self.step_calls = int(state["step_calls"])
        self.legacy_epoch_step_calls = int(state["legacy_epoch_step_calls"])
        self.current_epoch = int(state.get("current_epoch", 0))
        self.update_records = list(state["update_records"])
        if int(self.inner.last_epoch) != self.step_calls:
            raise AssertionError("resume scheduler.last_epoch differs from optimizer update count")
        self._assert_lr_matches(self.step_calls)

    def close(self) -> None:
        self._hook_handle.remove()


def write_lr_trace(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    fields = ["epoch", "update_count_before", "update_count_after",
              "lr_used_for_update", "lr_after_update"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
