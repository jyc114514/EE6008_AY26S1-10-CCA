"""CPU edge checks for constant/near-constant joints and safe FD scales."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from d3_eq_contract import per_sample_loss


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    torch.set_num_threads(1)
    norm = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = torch.tensor(np.asarray(norm["target_sigma"], dtype=np.float32))
    safe_fd = torch.tensor(np.asarray(norm["safe_fd_scale"], dtype=np.float32))
    if not torch.isfinite(sigma).all() or not torch.isfinite(safe_fd).all() or torch.any(sigma <= 0) or torch.any(safe_fd <= 0):
        raise AssertionError("locked train-only target/safe-FD scales must be finite and positive")

    target = torch.zeros((2, 60, 28), dtype=torch.float32)
    # Two exactly constant joints plus tiny but nonzero variation in two more.
    target[:, :, 0] = 0.0
    target[:, :, 14] = 0.0
    target[:, :, 1] = torch.linspace(0.0, 1e-8, 60)
    target[:, :, 15] = torch.linspace(0.0, -1e-8, 60)
    anchor = torch.zeros((2, 28), dtype=torch.float32)
    pred = torch.zeros((2, 60, 28), dtype=torch.float32, requires_grad=True)
    loss = per_sample_loss(pred, target, anchor, sigma, safe_fd)
    loss.sum().backward()
    if not torch.isfinite(loss).all() or not torch.isfinite(pred.grad).all():
        raise AssertionError("constant/near-constant joint objective or gradient is non-finite")
    result = {
        "status": "PASS",
        "locked_target_sigma_min": float(sigma.min()),
        "locked_safe_fd_scale_min": float(safe_fd.min()),
        "constant_joint_count": 2,
        "near_constant_joint_count": 2,
        "loss_finite": True,
        "gradient_finite": True,
        "uses_existing_safe_scale_without_refit": True,
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_NUMERICAL_EDGE_TESTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
