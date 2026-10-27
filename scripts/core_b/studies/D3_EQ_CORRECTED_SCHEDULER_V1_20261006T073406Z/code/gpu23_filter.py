"""Task-local physical GPU whitelist helpers (no device access)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

ALLOWED_PHYSICAL_IDS = frozenset({2, 3})
HIGH_ACTIVITY_THRESHOLD_PERCENT = 90


def validate_allowlist(allowed_uuids: Mapping[int, str]) -> dict[int, str]:
    normalized = {int(index): str(value) for index, value in allowed_uuids.items()}
    if not normalized or not set(normalized) <= ALLOWED_PHYSICAL_IDS:
        raise ValueError("active GPU candidates must be a nonempty subset of physical IDs {2,3}")
    if any(not value.startswith("GPU-") for value in normalized.values()):
        raise ValueError("allowlist entries must be NVIDIA GPU UUIDs")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("physical GPU UUIDs must be unique")
    return normalized


def filter_snapshots(snapshots: Iterable[Any], allowed_uuids: Mapping[int, str]) -> list[Any]:
    """Keep only exact (physical index, UUID) matches for GPUs 2 and 3."""
    allow = validate_allowlist(allowed_uuids)
    accepted: list[Any] = []
    seen: set[int] = set()
    for snapshot in snapshots:
        index = int(snapshot.index)
        if index not in ALLOWED_PHYSICAL_IDS or index not in allow:
            continue
        if str(snapshot.uuid) != allow[index]:
            continue
        if int(getattr(snapshot, "utilization_percent", 0)) >= HIGH_ACTIVITY_THRESHOLD_PERCENT:
            continue
        if index in seen:
            raise ValueError(f"duplicate aggregate snapshot for physical GPU {index}")
        seen.add(index)
        accepted.append(snapshot)
    return sorted(accepted, key=lambda item: int(item.index))


def validate_live_identity(index: int, uuid: str, allowed_uuids: Mapping[int, str], *,
                           utilization_percent: int | None = None) -> None:
    allow = validate_allowlist(allowed_uuids)
    physical = int(index)
    if physical not in ALLOWED_PHYSICAL_IDS or str(uuid) != allow.get(physical):
        raise RuntimeError(f"physical GPU index/UUID is not task-authorized: {physical}/{uuid}")
    if (utilization_percent is not None
            and int(utilization_percent) >= HIGH_ACTIVITY_THRESHOLD_PERCENT):
        raise RuntimeError(
            f"D3_RESOURCE_WAIT_HIGH_UTIL: physical GPU{physical} utilization "
            f"{utilization_percent}% is at or above {HIGH_ACTIVITY_THRESHOLD_PERCENT}%"
        )
