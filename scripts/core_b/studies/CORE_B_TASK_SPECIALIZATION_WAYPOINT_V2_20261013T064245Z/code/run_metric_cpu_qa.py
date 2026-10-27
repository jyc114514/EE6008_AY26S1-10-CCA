#!/usr/bin/env python3
"""Small deterministic CPU checks for V2 canonical metrics and episode aggregation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from v2_train_job import episode_equal_tune, metric_components

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = torch.full((2, 30, 28), 2.0)
    anchor = torch.zeros((2, 28))
    pred = torch.zeros((2, 30, 28))
    fit_sigma = torch.ones(28)
    fit_fd = torch.ones(28)
    metric_sigma = torch.full((28,), 2.0)
    metric_fd = torch.full((28,), 2.0)
    pos, fd, seam, native = metric_components(pred, target, anchor, fit_sigma, fit_fd, metric_sigma, metric_fd)
    assert torch.allclose(pos, torch.ones(2))
    assert torch.allclose(fd, torch.zeros(2))
    assert torch.allclose(seam, torch.ones(2))
    assert torch.allclose(native, torch.full((2,), 2.0))

    aggregate = episode_equal_tune([
        (119, 10, 1.0, 0.5, 2.0, 2.0),
        (119, 10, 3.0, 0.5, 4.0, 4.0),
        (119, 20, 5.0, 0.5, 6.0, 6.0),
    ])
    metric = aggregate["tasks"]["119"]
    assert metric["episodes"] == 2
    assert abs(metric["position_mae"] - 3.5) < 1e-12
    assert abs(metric["safe_fd_l1"] - 0.5) < 1e-12
    assert abs(metric["seam_safe_fd_l1"] - 4.5) < 1e-12
    assert abs(metric["native_mae"] - 4.5) < 1e-12

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS",
        "canonical_metric_scale_inverse_native_checked": True,
        "future_internal_29diff_separate_from_anchor_to_first_seam": True,
        "metric_scales": {"position": "accepted canonical-train target_sigma", "safe_fd": "accepted canonical-train safe_fd_scale"},
        "two_stage_native_inverse_checked": True,
        "window_to_episode_equal_aggregation_checked": True,
        "episode_equal_metrics": metric,
        "final_heldout_payload_access": 0,
    }
    path = ROOT / "qa/METRIC_CONTRACT_QA.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.pop("created_utc", None)
        compare = dict(payload)
        compare.pop("created_utc", None)
        if previous != compare:
            raise RuntimeError("existing metric QA receipt differs")
    else:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
