"""CPU-only native-layout head interface check; no scientific training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


from path_config import scratch_root
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--feature-file", required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch

    torch.set_num_threads(2)
    project = args.project_root.resolve()
    feature = args.feature_file.resolve()
    if not feature.is_relative_to(scratch_root(project)):
        raise ValueError("Expected a smoke feature inside configured scratch")
    sidecar = json.loads(feature.with_name("SMOKE_REPORT.json").read_text())
    if sidecar.get("status") != "PASS_CORE_B_FEATURE_SMOKE_CPU" or digest(feature) != sidecar["feature_sha256"]:
        raise AssertionError("Feature smoke bundle is not accepted")
    code = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/d3_model.py"
    expected_hash = "6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589"
    if digest(code) != expected_hash:
        raise AssertionError("Team model source changed; re-audit before using it")
    spec = importlib.util.spec_from_file_location("native_videomae_head_fixture", code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(feature, allow_pickle=False) as archive:
        rows = archive["cache_index"].copy()
        ids = archive["anchor_id"].copy()
        dynamic = torch.from_numpy(archive["dynamic"].copy())
        static = torch.from_numpy(archive["static"].copy())
    input_file = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    if digest(input_file) != sidecar["input_sha256"]:
        raise AssertionError("Conditioning archive changed since feature extraction")
    with np.load(input_file, allow_pickle=False) as archive:
        if not np.array_equal(ids, archive["anchor_id"][rows]):
            raise AssertionError("Feature IDs do not align with conditioning inputs")
        state, mask, relative, anchor = [torch.from_numpy(archive[key][rows].copy()) for key in (
            "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native"
        )]
    results = []
    # Construct the original model first, preserving its common head initialization.
    for horizon in (30, 60):
        module.TIME_SLOTS, module.VIDEO_SLOTS, module.VIDEO_DIM = 16, 64, 768
        model = module.D3TimeSpacePredictor(horizon=horizon, seed=20261001, video_mode="TS11-D")
        common = {name: value.clone() for name, value in model.state_dict().items()
                  if not name.startswith("video_projection.")}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261001)
            model.video_projection = torch.nn.Linear(384, 128)
        module.TIME_SLOTS, module.VIDEO_SLOTS, module.VIDEO_DIM = 8, 32, 384
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in common.items()):
            raise AssertionError("Adapter changed common head initialization")
        model.eval()
        with torch.inference_mode():
            residual = model(state, mask, relative, dynamic, static, anchor)
        if tuple(residual.shape) != (4, horizon, 28) or not torch.isfinite(residual).all():
            raise AssertionError("Head output shape/finite failure")
        persistence_error = float(residual.abs().max())
        if persistence_error != 0:
            raise AssertionError("Zero-initialized output must produce exact zero residual")
        # Synthetic loss only checks gradient connectivity; real future targets are absent.
        model.zero_grad(set_to_none=True)
        output = model(state, mask, relative, dynamic, static, anchor)
        torch.nn.functional.mse_loss(output, torch.full_like(output, 0.1)).backward()
        gradient = model.output.weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
            raise AssertionError("Output-head gradient fixture failed")
        results.append({"horizon": horizon, "output_shape": list(residual.shape),
                        "initial_residual_max_abs": persistence_error,
                        "synthetic_gradient_finite_nonzero": True})
    print(json.dumps({
        "status": "PASS_NATIVE_CORE_B_HEAD_CPU_INTERFACE",
        "model_source_sha256": expected_hash,
        "feature_sha256": sidecar["feature_sha256"],
        "results": results,
        "native_video_layout": [8, 4, 384],
        "common_head_initialization_preserved": True,
        "normalization": "raw features for interface smoke only; formal train-only fit pending",
        "targets_loaded": False, "optimizer_steps": 0, "gpu_used": False,
        "final_test_access": 0,
        "formal_comparison_ready": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
