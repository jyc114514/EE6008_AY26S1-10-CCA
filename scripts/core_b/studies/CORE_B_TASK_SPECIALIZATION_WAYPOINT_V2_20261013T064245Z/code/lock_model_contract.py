#!/usr/bin/env python3
"""Write the fixed V2 model, loss, scale and optimizer contract."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads((ROOT / "inventory/SOURCE_ACCEPTANCE.json").read_text(encoding="utf-8"))
    resolved = json.loads((ROOT / "inventory/RESOLVED_PATHS.json").read_text(encoding="utf-8"))
    metric_path = ROOT / "protocol/METRIC_SCALE_LOCK.json"
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    model_path = ROOT / "code/v2_model.py"
    lock = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "LOCKED_FOR_REMAINING_SCREENING_AND_FORMAL",
        "created_after_preregistered_task119_pilot": True,
        "pilot_fits_in_matrix_already_complete": 2,
        "contract_semantics_changed_after_pilot": False,
        "fixed_task_ids": [119, 94, 90, 141, 230],
        "logical_models": {
            "shared": ["G-S", "G-V"],
            "experts": [f"S-{mode}-{task}" for task in (119, 94, 90, 141, 230) for mode in ("S", "V")],
            "count": 12,
        },
        "architecture": {
            "backbone": "accepted Q0 D3TimeSpacePredictor, horizon30, width128; task-conditioned one-hot projection token",
            "input_api": "forward(prefix_inputs, task_id)",
            "action_output": "[B,30,28] standardized residual",
            "state_only_video_mask": "STATE-CTRL constructs fixed zero dynamic/static grid and performs no RGB cache fetch",
            "fusion_video_mode": "TS11-D accepted prefix grid [2,16,4,768]",
            "waypoint_modes": {
                "AUX_ONLY": "prefix goal head plus 0.1 endpoint auxiliary L1; action decoder receives zero token",
                "GOAL_COND": "same prefix goal head and auxiliary L1; predicted goal projection enters decoder memory",
                "future_goal_target_in_forward": False,
                "per_seed_init_equal": True,
            },
            "parameter_count_expected": 891548,
        },
        "feature_encoder": {"frozen": True, "new_extraction": 0, "cache_manifest_sha256": resolved["feature_cache_manifest_sha256"]},
        "loss": {
            "action": "accepted normalized position L1 + 0.1 future-internal safe-FD L1 over 29 differences",
            "waypoint_auxiliary": "0.1 mean absolute endpoint residual error",
            "episode_task_weight": "w_i=W_current_train/(K*E_task*n_windows_in_episode)",
            "batch_reduction": "sum(w_i*loss_i)/B_actual",
            "shared_K": 5,
            "expert_K": 1,
            "inner_fit_weights_exclude_tune": True,
            "future_label_forward_arguments": 0,
        },
        "normalization": {
            "screening": "common scales fit on the 1021 inner-fit anchors only",
            "formal": "common scales fit on all 1280 selected outer-train anchors",
            "tune_and_final_metrics": "accepted canonical primary-train Full-28 target_sigma and safe_fd_scale after native inverse",
            "metric_scale_lock_sha256": sha(metric_path),
            "zero_scale_rule": "accepted episode-equal source estimator; std<=1e-6 becomes1; safe_fd=max(fd,0.001*target_sigma)",
        },
        "optimizer": {
            "name": "AdamW", "lr_candidates": [0.0005, 0.001, 0.002],
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4,
            "dtype": "float32", "amp": False, "compile": False,
            "effective_batch": 64, "initial_microbatch": 64, "oom_fallback": {"microbatch": 32, "accumulation": 2},
            "scheduler": "update-driven cosine, eta_min=1e-5, T_max=160*optimizer_updates_per_epoch",
            "screening_seed": 202610130, "formal_seeds": [202610131, 202610132, 202610133],
        },
        "selection": {
            "source": "inner_tune only",
            "metric": "episode-equal normalized position MAE+0.1 safe-FD L1 using canonical primary-train scales",
            "per_model_candidates": 9,
            "checkpoints": [40, 80, 160],
            "tie_rule": "within 1e-12 choose shorter epoch, then LR 1e-3, 5e-4, 2e-3",
            "dev_targets_used_for_selection": False,
        },
        "code_hashes": {
            "accepted_q0_model_source_sha256": source["locked_source_copies"]["D3_model.py"]["source_sha256"],
            "v2_model_sha256": sha(model_path),
            "v2_training_job_sha256": sha(ROOT / "code/v2_train_job.py"),
            "v2_scheduler_sha256": sha(ROOT / "code/v2_scheduler.py"),
        },
        "final_heldout_payload_access": 0,
    }
    path = ROOT / "protocol/MODEL_AND_LOSS.json"
    if path.exists():
        raise FileExistsError(f"model contract already exists: {path}")
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "protocol_path": str(path), "final_heldout_payload_access": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
