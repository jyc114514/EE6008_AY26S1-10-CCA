"""V2 update-driven cosine scheduler derived from the accepted Q0 wrapper.

Only T_max is generalized from the accepted epoch-40 value to the V2 locked
160-epoch reference cycle. The optimizer post-step hook and trace contract stay
the same; the shared watcher and its sources are untouched.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import torch


def closed_form_lr(
    update_count: int,
    *,
    initial_lr: float,
    eta_min: float = 1e-5,
    t_max: int,
) -> float:
    if t_max <= 0 or not 0 <= update_count <= t_max:
        raise ValueError(f"update_count must be in [0,{t_max}] and t_max positive")
    return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * update_count / t_max))


class UpdateDrivenCosineAnnealingLR:
    """Advance cosine LR exactly once after each successful optimizer update."""

    def __init__(self, optimizer: torch.optim.Optimizer, *, T_max: int, eta_min: float = 1e-5):
        if isinstance(T_max, bool) or int(T_max) != T_max or T_max <= 0:
            raise ValueError("T_max must be a positive integer")
        if not math.isclose(float(eta_min), 1e-5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("V2 locks eta_min=1e-5")
        if not hasattr(optimizer, "register_step_post_hook"):
            raise RuntimeError("installed PyTorch lacks optimizer post-step hooks")
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.inner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        self.step_calls = 0
        self.legacy_epoch_step_calls = 0
        self.current_epoch = 0
        self.update_records: list[dict[str, Any]] = []
        self._hook_handle = optimizer.register_step_post_hook(self._after_optimizer_step)
        self._assert_lr_matches(0)

    def _assert_lr_matches(self, update_count: int) -> None:
        actual = float(self.optimizer.param_groups[0]["lr"])
        expected = closed_form_lr(
            update_count, initial_lr=self.base_lr, eta_min=self.eta_min, t_max=self.T_max
        )
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-11):
            raise AssertionError(f"optimizer/scheduler LR mismatch at update {update_count}: {actual} != {expected}")

    def _after_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
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

    def get_last_lr(self) -> list[float]:
        """Match the standard scheduler query used by the shared training loop."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "inner_state": self.inner.state_dict(),
            "step_calls": self.step_calls,
            "legacy_epoch_step_calls": self.legacy_epoch_step_calls,
            "current_epoch": self.current_epoch,
            "update_records": self.update_records,
            "T_max": self.T_max,
            "eta_min": self.eta_min,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("T_max"), state.get("eta_min"), state.get("base_lr")) != (self.T_max, self.eta_min, self.base_lr):
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
    fields = ["epoch", "update_count_before", "update_count_after", "lr_used_for_update", "lr_after_update"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
