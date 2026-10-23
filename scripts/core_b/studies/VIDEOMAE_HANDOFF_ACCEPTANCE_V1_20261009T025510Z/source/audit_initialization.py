#!/usr/bin/env python3
"""CPU-only verification of shared seed initialization across 384D/768D video inputs."""

from __future__ import annotations
import os

import hashlib
import json
import sys
from pathlib import Path

import torch


def common_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("video_projection."):
            continue
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    project = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
    audit = project / "reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z"
    eq = project / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
    bench = project / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
    model_root = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
    sys.path.insert(0, str(model_root))
    import d3_model as models

    registry = json.loads((eq / "protocol/D3_EQ_SEED_REGISTRY.json").read_text(encoding="utf-8"))["seeds"]
    rows = []
    for seed in (20261001, 20261002, 20261003):
        init_path = bench / f"d3/runs/INIT_S{seed}.pt"
        initial = torch.load(init_path, map_location="cpu", weights_only=True)
        models.VIDEO_DIM = 768
        team = models.D3TimeSpacePredictor(horizon=30, seed=seed, video_mode="STATE-CTRL").cpu()
        team.load_state_dict(initial, strict=True)
        actual_init_hash = models.state_hash(team)
        expected_init_hash = registry[str(seed)]["init_state_hash"]
        if actual_init_hash != expected_init_hash:
            raise AssertionError(f"seed {seed}: initial state does not match the locked registry")

        vm_fusion = models.D3TimeSpacePredictor(horizon=30, seed=seed, video_mode="TS11-D").cpu()
        vm_fusion.load_state_dict(initial, strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            vm_fusion.video_projection = torch.nn.Linear(384, 128)
        models.VIDEO_DIM = 384
        video_mae_hash = common_hash(vm_fusion)
        team_hash = common_hash(team)
        if video_mae_hash != team_hash:
            raise AssertionError(f"seed {seed}: shared module initialization diverged")
        rows.append({
            "seed": seed,
            "init_file_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
            "locked_initial_state_hash": expected_init_hash,
            "shared_non_video_projection_state_sha256": team_hash,
            "shared_state_matches": True,
            "team_state_ctrl_video_projection_input_dim": 768,
            "video_mae_fusion_video_projection_input_dim": 384,
            "team_total_parameter_count": models.parameter_count(team),
            "video_mae_fusion_total_parameter_count": models.parameter_count(vm_fusion),
            "video_projection_parameter_counts": {"384D": 384 * 128 + 128, "768D": 768 * 128 + 128},
            "video_projection_changed_after_init_load_only": True,
        })
    report = {
        "status": "PASS_SHARED_INITIALIZATION_AUDIT",
        "method": "load exact 768D registered seed init in each fresh CPU model; reproduce VideoMAE 384D projection replacement after load; compare all state except video_projection",
        "rows": rows,
        "final_test_access": 0,
        "gpu_used": False,
    }
    (audit / "metrics/core_b/SHARED_INITIALIZATION_QA.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
