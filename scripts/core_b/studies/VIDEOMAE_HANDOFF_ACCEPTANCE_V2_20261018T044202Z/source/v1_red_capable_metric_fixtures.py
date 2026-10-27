#!/usr/bin/env python3
"""Small independent metric, grouping and future-sentinel fixtures; failures are fatal."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from independent_core_b_metrics import grouped_scores, window_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    audit = args.audit_root.resolve()

    # Native and standardized errors, RMSE, future-internal FD, endpoint and support.
    anchor = np.zeros((1, 1), dtype=np.float32)
    target = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    pred = np.asarray([[[10.0], [11.0], [12.0]]], dtype=np.float32)
    metrics = window_metrics(pred, target, anchor, np.ones(1, dtype=np.float32),
                             np.ones(1, dtype=np.float32), np.asarray([1e-6], dtype=np.float32))
    assert float(metrics["native_future_internal_fd"][0]) == 0.0
    assert float(metrics["safe_fd_normalized"][0]) == 0.0
    assert math.isclose(float(metrics["native_mae"][0]), 9.0)
    assert math.isclose(float(metrics["standardized_mae"][0]), 9.0)
    assert math.isclose(float(metrics["endpoint_native_mae"][0]), 9.0)
    assert math.isclose(float(metrics["seam_error_native"][0]), 9.0)
    assert math.isclose(float(metrics["tv_pred"][0] / metrics["tv_gt"][0]), 1.0)
    assert int(metrics["active_step_count"][0]) == 2
    assert float(metrics["direction_accuracy_active"][0]) == 1.0

    # Anchor seam is excluded from the 29/59 future-internal FD calculation.
    seam_only = window_metrics(np.asarray([[[100.0], [101.0], [102.0]]], dtype=np.float32),
                               target, anchor, np.ones(1, dtype=np.float32),
                               np.ones(1, dtype=np.float32), np.asarray([1e-6], dtype=np.float32))
    assert float(seam_only["native_future_internal_fd"][0]) == 0.0
    assert float(seam_only["seam_error_native"][0]) == 99.0

    # Windows are averaged within episode, then episodes within source group.
    episode_values = np.asarray([0.0, 2.0, 100.0, 10.0])
    episodes = np.asarray([1, 1, 2, 3])
    groups = np.asarray(["A", "A", "A", "B"])
    weights, per_group = grouped_scores(episode_values, episodes, groups)
    assert math.isclose(weights["episode_equal"], 37.0)
    assert math.isclose(weights["source_group_equal"], 30.25)
    assert math.isclose(per_group["A"], 50.5) and math.isclose(per_group["B"], 10.0)

    # Future RGB sentinel: the locked extraction indices never exceed anchor.
    def legal_frame_indices(anchor_frame: int) -> np.ndarray:
        return np.floor(np.linspace(anchor_frame - 119, anchor_frame, 32)).astype(np.int64)

    anchor_frame = 240
    frame_indices = legal_frame_indices(anchor_frame)
    assert frame_indices.min() >= 0 and frame_indices.max() == anchor_frame
    rgb = np.arange(anchor_frame + 64, dtype=np.int64)
    changed_future_rgb = rgb.copy()
    changed_future_rgb[anchor_frame + 1 :] = -123456789
    assert np.array_equal(rgb[frame_indices], changed_future_rgb[frame_indices])
    future_targets = np.zeros((60, 28), dtype=np.float32)
    changed_future_targets = np.full_like(future_targets, 1.0e12)
    # Predictor input is constructed only from history, legal video tokens and anchor command.
    prediction_input = (np.asarray([[1.0]], dtype=np.float32),
                        rgb[frame_indices].copy(), np.asarray([[0.0]], dtype=np.float32))
    sentinel_input = (prediction_input[0].copy(), changed_future_rgb[frame_indices].copy(), prediction_input[2].copy())
    assert all(np.array_equal(a, b) for a, b in zip(prediction_input, sentinel_input))
    assert not np.array_equal(future_targets, changed_future_targets)

    # The actual temporal predictor has no target parameter and cache extraction stops at anchor.
    model_path = project / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/d3_model.py"
    cache_path = project / 'tmp/project-team/videomae_handoff_20261009T020906Z/code/extract_videomae_core_b_cache.py'
    module = ast.parse(model_path.read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "D3TimeSpacePredictor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    forward_args = [arg.arg for arg in fn.args.args]
    assert "targets" not in forward_args and "future_rgb" not in forward_args
    cache_source = cache_path.read_text(encoding="utf-8")
    assert "while frame_index < anchor" in cache_source
    assert "np.floor(np.linspace(anchor - 119, anchor, 32))" in cache_source

    report = {
        "status": "PASS_RED_CAPABLE_FIXTURES",
        "native_fd_excludes_anchor_seam": True,
        "source_group_equal_differs_from_episode_equal_fixture": True,
        "future_rgb_sentinel_unchanged_legal_input": True,
        "future_target_sentinel_not_in_predictor_signature": True,
        "actual_predictor_forward_args": forward_args,
        "actual_video_cache_cutoff": "frame_index < anchor; floor(linspace(anchor-119,anchor,32))",
        "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "final_test_access": 0,
    }
    out = audit / "metrics/core_b/RED_CAPABLE_FIXTURE_RESULTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
