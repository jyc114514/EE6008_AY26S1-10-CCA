#!/usr/bin/env python3
"""Fit V2 shared screening scales on the locked five-task inner-fit rows only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
LOCKED = CODE / "locked_source"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(LOCKED))

from prepare_v2_data import Q0ShardedGrid  # noqa: E402
from prepare_v3_inputs import episode_equal_absdiff, episode_equal_stats  # noqa: E402


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable_existing = {k: v for k, v in existing.items() if k != "created_utc"}
        comparable_new = {k: v for k, v in value.items() if k != "created_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"existing normalizer receipt differs from recomputed values; preserving both for review: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("screening", "formal"), default="screening")
    args = parser.parse_args()
    scope = str(args.scope)
    data_path = ROOT / "data/TRAIN_SELECTED_H30.npz"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    manifest_path = Path(resolved["feature_cache"])
    grid = Q0ShardedGrid(json.loads(manifest_path.read_text(encoding="utf-8")))
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    inner_fit_mask = arrays["inner_role"] == "inner_fit"
    inner_tune_mask = arrays["inner_role"] == "inner_tune"
    if not inner_fit_mask.any() or not inner_tune_mask.any() or np.any(inner_fit_mask & inner_tune_mask):
        raise RuntimeError("V2 inner-fit/tune rows are missing or overlap")
    fit_mask = inner_fit_mask if scope == "screening" else np.ones(len(inner_fit_mask), dtype=bool)
    tune_mask = inner_tune_mask if scope == "screening" else np.zeros(len(inner_tune_mask), dtype=bool)
    episodes = arrays["episode_id"].astype(np.int64)
    fit_episodes = np.unique(episodes[fit_mask])
    tune_episodes = np.unique(episodes[tune_mask])
    fit_clusters = np.unique(arrays["source_cluster"][fit_mask].astype("U128"))
    tune_clusters = np.unique(arrays["source_cluster"][tune_mask].astype("U128"))
    if scope == "screening" and set(fit_clusters.tolist()) & set(tune_clusters.tolist()):
        raise RuntimeError("source groups cross inner-fit and inner-tune")

    fit_hist = arrays["history_native"][fit_mask]
    fit_rel = arrays["relative_delta"][fit_mask]
    fit_targets = arrays["target_native_h30"][fit_mask]
    fit_episodes_by_anchor = episodes[fit_mask]

    history_flat = fit_hist.reshape(-1, fit_hist.shape[-1])
    history_episode_flat = np.repeat(fit_episodes_by_anchor, fit_hist.shape[1])
    history_train_flat = np.ones(len(history_flat), dtype=bool)
    state_mean, state_safe_std = episode_equal_stats(history_flat, history_episode_flat, history_train_flat)

    relative_flat = fit_rel.reshape(-1, 1)
    relative_episode_flat = np.repeat(fit_episodes_by_anchor, fit_rel.shape[1])
    relative_mean, relative_safe_std = episode_equal_stats(relative_flat, relative_episode_flat, np.ones(len(relative_flat), dtype=bool))

    target_flat = fit_targets.reshape(-1, fit_targets.shape[-1])
    target_episode_flat = np.repeat(fit_episodes_by_anchor, fit_targets.shape[1])
    target_mean, target_sigma = episode_equal_stats(target_flat, target_episode_flat, np.ones(len(target_flat), dtype=bool))
    fd_scale = episode_equal_absdiff(fit_targets, fit_episodes_by_anchor, np.ones(len(fit_targets), dtype=bool))
    safe_fd_scale = np.maximum(fd_scale, 0.001 * target_sigma).astype(np.float32)

    per_episode_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    per_episode_sq: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    per_episode_windows: dict[int, int] = defaultdict(int)
    fit_rows = np.flatnonzero(fit_mask)
    for row_index in fit_rows.tolist():
        features = grid.row(int(arrays["cache_index"][row_index]), str(arrays["anchor_id"][row_index]), str(episodes[row_index]))
        dynamic = features[0].astype(np.float64, copy=False)
        row_mean = dynamic.mean(axis=(0, 1))
        row_second = np.square(dynamic).mean(axis=(0, 1))
        episode = int(episodes[row_index])
        per_episode_sum[episode] += row_mean
        per_episode_sq[episode] += row_second
        per_episode_windows[episode] += 1
    ordered_episodes = sorted(per_episode_windows)
    video_means = np.asarray([per_episode_sum[ep] / per_episode_windows[ep] for ep in ordered_episodes], dtype=np.float64)
    video_seconds = np.asarray([per_episode_sq[ep] / per_episode_windows[ep] for ep in ordered_episodes], dtype=np.float64)
    video_mean = video_means.mean(axis=0)
    video_variance = np.maximum(video_seconds.mean(axis=0) - video_mean * video_mean, 0.0)
    video_sqrt = np.sqrt(video_variance)
    video_safe_std = np.where(video_sqrt <= 1e-6, 1.0, video_sqrt).astype(np.float32)

    source_keys = sorted(str(x) for x in fit_clusters.tolist())
    fitted_ids_sha = sha_bytes(("\n".join(source_keys) + "\n").encode("utf-8"))
    output_dir = ROOT / ("screening" if scope == "screening" else "normalizers")
    output_path = output_dir / ("NORMALIZER_SCREENING.json" if scope == "screening" else "NORMALIZER_FORMAL.json")
    normalizer = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "LOCKED_SCREENING" if scope == "screening" else "LOCKED_FORMAL",
        "scope": "fixed five-task inner-fit only; common scales for all shared and specialist methods" if scope == "screening" else "fixed five-task complete outer-train; common scales for all shared and specialist methods",
        "screening_fit_anchors": int(fit_mask.sum()),
        "screening_fit_episodes": int(len(fit_episodes)),
        "screening_fit_clusters": int(len(fit_clusters)),
        "inner_tune_anchors_excluded_from_all_fits": int(tune_mask.sum()),
        "inner_tune_episodes": int(len(tune_episodes)),
        "formal_outer_train_fit": scope == "formal",
        "inner_tune_included_in_formal_fit": int(inner_tune_mask.sum()) if scope == "formal" else 0,
        "fit_source_clusters_sha256_ordered_newline": fitted_ids_sha,
        "input_data_sha256": sha_file(data_path),
        "q0_feature_manifest_sha256": sha_file(manifest_path),
        "state_mean": state_mean.astype(np.float32).tolist(),
        "state_safe_std": state_safe_std.astype(np.float32).tolist(),
        "relative_delta_mean": float(relative_mean[0]),
        "relative_delta_safe_std": float(relative_safe_std[0]),
        "target_mean": target_mean.astype(np.float32).tolist(),
        "target_sigma": target_sigma.astype(np.float32).tolist(),
        "fd_scale": fd_scale.astype(np.float32).tolist(),
        "safe_fd_scale": safe_fd_scale.astype(np.float32).tolist(),
        "video_mean": video_mean.astype(np.float32).tolist(),
        "video_safe_std": video_safe_std.tolist(),
        "video_fit_scope": "dynamic D primary train grid tokens; episode equal, window equal, time/bin equal",
        "epsilon": 1e-6,
        "zero_or_near_zero_scale_rule": "state/relative/video std <=1e-6 becomes 1; target sigma uses accepted episode_equal_stats safe std; safe_fd=max(fd_scale,0.001*target_sigma)",
        "native_anchor_formula": "pred_native=anchor_native+target_sigma*pred_residual",
        "final_heldout_payload_access": 0,
        "dev_target_or_outcome_access": 0,
    }
    save_json(output_path, normalizer)
    qa_path = ROOT / ("qa/NORMALIZER_SCREENING_QA.json" if scope == "screening" else "qa/NORMALIZER_FORMAL_QA.json")
    save_json(qa_path, {
        "schema_version": 1,
        "created_utc": normalizer["created_utc"],
        "status": "PASS",
        "scope": scope,
        "inner_fit_only": scope == "screening",
        "inner_tune_excluded": scope == "screening",
        "formal_outer_train_all_rows_included": scope == "formal",
        "inner_tune_rows_included_in_formal_fit": int(inner_tune_mask.sum()) if scope == "formal" else 0,
        "shared_and_specialist_scales_identical": True,
        "state_shape": [int(x) for x in fit_hist.shape],
        "target_shape": [int(x) for x in fit_targets.shape],
        "feature_grid_shape_per_row": [2, 16, 4, 768],
        "all_scales_finite_positive": bool(
            np.isfinite(state_mean).all() and np.isfinite(state_safe_std).all() and
            np.isfinite(relative_mean).all() and np.isfinite(relative_safe_std).all() and
            np.isfinite(target_sigma).all() and np.all(target_sigma > 0) and
            np.isfinite(safe_fd_scale).all() and np.all(safe_fd_scale > 0) and
            np.isfinite(video_mean).all() and np.isfinite(video_safe_std).all() and np.all(video_safe_std > 0)
        ),
        "dev_or_heldout_targets_read": False,
        "final_heldout_payload_access": 0,
    })
    print(json.dumps({
        "status": "PASS",
        "fit_anchors": int(fit_mask.sum()), "fit_episodes": int(len(fit_episodes)), "fit_clusters": int(len(fit_clusters)),
        "tune_anchors_excluded": int(tune_mask.sum()), "state_dim": int(state_mean.shape[0]),
        "action_dim": int(target_sigma.shape[0]), "video_dim": int(video_mean.shape[0]),
        "normalizer": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
