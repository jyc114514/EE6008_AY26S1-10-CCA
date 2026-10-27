#!/usr/bin/env python3
"""Write the locked History×Video matrix contracts and watcher job records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from model import HistoryVideoTCNDCT, parameter_count


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
PYTHON_BIN = PROJECT / "envs/jyc/bin/python"
INPUT_ROOT = ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
PREPROCESS_STATS = ROOT / "PREPROCESS_STATS.json"
CANONICAL_INPUT = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/inputs/JY02_TARGETS.npz"
CANONICAL_NORMALIZER = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
WATCHER = REPO / "tools/gpu_lease_watcher.py"
SEED = 20260939
CELLS = (
    ("HI-S-LAST", "LAST", "S"),
    ("HI-S-HIST", "HIST", "S"),
    ("HI-F-LAST", "LAST", "F"),
    ("HI-F-HIST", "HIST", "F"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


def _default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    data = {key: value.copy() for key, value in np.load(INPUT_ROOT, allow_pickle=False).items()}
    stats = json.loads(PREPROCESS_STATS.read_text(encoding="utf-8"))
    model = HistoryVideoTCNDCT(seed=SEED)
    rows: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    for cell, state_input, video_input in CELLS:
        rows.append({
            "cell": cell,
            "state_input": state_input,
            "video_input": video_input,
            "state_values_source": "history_standardized" if state_input == "HIST" else "last_history_value_repeated_32",
            "video_values_source": "dynamic_video_feature" if video_input == "F" else "zeros_same_768D_shape",
            "model": "causal_TCN_5_blocks_plus_fixed_DCT16_residual_decoder",
            "seed": SEED,
            "epochs": 40,
            "checkpoint_selection": "last_epoch_only",
            "final_test_access": 0,
        })
        run_root = ROOT / "runs" / cell / "stage0"
        log_path = ROOT / "logs" / f"child_{cell}_stage0.log"
        jobs.append({
            "job_id": f"HVI-{cell}-STAGE0",
            "protocol_id": "HISTORY_VIDEO_INFORMATION_MATRIX_V1_STAGE0",
            "output_root": str(run_root),
            "log_path": str(log_path),
            "command": [
                str(PYTHON_BIN),
                str(ROOT / "code/run_history_video_cell.py"),
                "--cell", cell,
                "--run-root", str(run_root),
                "--input-root", str(INPUT_ROOT),
                "--stage0",
                "--max-optimizer-steps", "8",
                "--smoke-rows", "128",
                "--batch-size", "64",
                "--accumulation", "1",
                "--device", "cuda",
            ],
            "environment": {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "resource_estimate": {"measured_peak_mib": 2048, "estimate_type": "stage0_pre_registered_upper_bound"},
            "max_oom_requeues": 0,
            "final_test_access": 0,
        })

    write_csv(ROOT / "MATRIX.csv", rows)
    write_json(ROOT / "MATRIX.json", {"status": "LOCKED", "seed": SEED, "rows": rows, "cell_count": len(rows), "final_test_access": 0})
    write_json(ROOT / "configs/STAGE0_JOBS.json", {"status": "LOCKED", "jobs": jobs, "lease_dir": str(ROOT / "locks/gpu_lease"), "max_concurrent_project_gpu_children": 2, "final_test_access": 0})
    for job in jobs:
        write_json(ROOT / "configs/jobs" / f"{job['job_id']}.json", job)

    contract = {
        "status": "LOCKED",
        "study": "History×Video information 2×2 direct training",
        "seed": SEED,
        "cells": [row["cell"] for row in rows],
        "population": {"train": 2741, "development": 614, "target_shape": [30, 28], "hand_slice": [0, 14], "arm_slice": [14, 28]},
        "input": {
            "history_source": ["observation.arm_joints", "observation.hand_joints", "observation.leg_joints", "action"],
            "history_state_dimension": 71,
            "history_steps": 32,
            "history_query_interval_seconds": 0.125,
            "anchor": "last legal prefix row frame 119; timestamp t0",
            "history_metadata": ["binary validity mask", "relative timestamp delta to t0"],
            "video_dimension": 768,
            "video_source": "accepted fixed B2 dense dynamic_pooled feature, train-only standardized cache",
            "S_video": "all zeros with same shape as F video feature",
            "future_inputs": False,
            "task_id_input": False,
        },
        "model": {
            "state_projection": "Linear(73,64)",
            "tcn": {"blocks": 5, "channels": 64, "kernel": 3, "dilations": [1, 2, 4, 8, 16], "padding": "left_only_causal", "activation": "ReLU", "dropout": 0.1, "residual": True},
            "video_projection": "Linear(768,128)->ReLU",
            "fusion": "Linear(192,128)->ReLU->Dropout(0.1)",
            "output": "Linear(128,16*14)->inverse_orthonormal_DCT_II",
            "dct_K": 16,
            "parameter_count": parameter_count(model),
            "output_zero_initialization": True,
            "selector": False,
            "prototype": False,
            "stacking": False,
        },
        "target_parameterization": {
            "native_prediction": "last_command_native + residual_sigma_train * residual_scaled",
            "residual_sigma_source": "canonical train future Hand residual nonzero absolute median per joint",
            "fd_scale_source": "canonical train future-to-future Hand delta nonzero absolute median per joint",
            "epsilon": "1e-6*max(1, train Hand future absolute-value 95th percentile)",
            "lambda_fd": stats["lambda_fd"],
            "loss": "canonical standardized Hand L1 position + lambda_fd * native FD L1 scaled by train fd_scale",
            "native_fd_denominator": "29 future-to-future differences x 14 Hand joints",
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "final_learning_rate": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "epochs": 40,
            "physical_batch": 64,
            "effective_batch": 64,
            "fallback": "physical batch 32 + accumulation 2 only after an OOM; same effective batch and denominator",
            "dev_early_stopping": False,
            "formal_checkpoint": "checkpoint_last from epoch 40",
        },
        "normalization": {
            "input_fit_scope": "canonical train only",
            "target_normalizer": str(CANONICAL_NORMALIZER),
            "target_normalizer_sha256": digest(CANONICAL_NORMALIZER),
            "preprocess_stats_sha256": digest(PREPROCESS_STATS),
            "fit_ids_file": str(ROOT / "PREPROCESS_FIT_IDS.json"),
        },
        "source_hashes": {"canonical_input": digest(CANONICAL_INPUT), "locked_arm": digest(LOCKED_ARM), "model_code": digest(ROOT / "code/model.py"), "train_code": digest(ROOT / "code/run_history_video_cell.py") if (ROOT / "code/run_history_video_cell.py").exists() else "pending"},
        "final_test_access": 0,
    }
    write_json(ROOT / "MODEL_AND_LOSS_CONTRACT.json", contract)
    write_json(ROOT / "RESOURCE_CONTRACT.json", {
        "status": "LOCKED",
        "visible_gpu_pool": "dynamic_all_visible",
        "gpu_id_allowlist": None,
        "co_location_allowed": True,
        "utilization": "ranking_only",
        "memory_rule": "free_vram >= measured_peak_mib + max(512 MiB, 10% measured_peak_mib)",
        "max_concurrent_project_gpu_children": 2,
        "max_physical_gpus_in_use": 2,
        "one_project_child_per_gpu": True,
        "non_child_process_signal": False,
        "stage0_required": True,
        "stage0_pre_registered_peak_mib": 2048,
        "formal_peak_registration": "after Stage0, max measured peak plus normal variation is written before formal jobs",
        "oom_policy": "own-child-only; one pre-registered batch32/accumulation2 fallback; no other process signal",
        "final_test_access": 0,
    })
    write_json(ROOT / "PROTOCOL_LOCKED.json", {
        "status": "LOCKED",
        "protocol_id": "HISTORY_VIDEO_INFORMATION_MATRIX_V1",
        "seed": SEED,
        "matrix": [row["cell"] for row in rows],
        "train_epochs": 40,
        "selection": "fixed last epoch; no dev early stopping",
        "single_stage_supervision": True,
        "selector_or_prototype_training": False,
        "base_oof_required": False,
        "train_only_preprocessing": True,
        "final_test_access": 0,
        "professor_contact_started": False,
    })
    (ROOT / "PROTOCOL_LOCKED.md").write_text("""# History × Video information matrix protocol\n\nThis is a four-cell, single-stage, one-seed development experiment. It compares LAST versus HIST past state values and S versus F fixed video values with one matched causal TCN + DCT-16 residual decoder. All fit operations use canonical train rows only; dev is used once after all four epoch-40 checkpoints are accepted.\n\nThe formal checkpoint is always epoch 40 (`checkpoint_last.pt`); dev scores cannot influence training or checkpoint selection. No selector, prototype, oracle, stacking, LoRA, encoder update, final-test access, or automatic follow-up is allowed.\n\nResource selection is dynamic across all visible GPUs. Co-location is allowed subject to the measured-peak memory contract; utilization only ranks candidates. The project may use at most two GPU children and never signals a non-child process.\n""", encoding="utf-8")
    (ROOT / "CONTEXT_AND_INPUT_INVENTORY.md").write_text("""# Context and input inventory\n\n- Canonical population: `JY02_TARGETS.npz`, 3355 rows (`2741` train, `614` dev), targets `(N,30,28)`.\n- State/action source: raw G1 Parquet columns `observation.arm_joints` (14), `observation.hand_joints` (14), `observation.leg_joints` (15), and `action` (28), concatenated in that order to 71 values per query.\n- Anchor: frame 119, the final legal prefix row. Target starts at frame 120; raw seam checks are recorded in `DATA_TIMELINE_MANIFEST.json`.\n- History: 32 query offsets from `t0-31*0.125s` through `t0`; each row is the latest observed Parquet row at or before the query.\n- Video: accepted B2 dense frozen `dynamic_pooled` feature, 768 dimensions, reused exactly; no raw video is read in this study.\n- Full-28 composite: new Hand prediction `[0:14]` plus the locked E-F-SP-G Arm prediction `[14:28]`; Arm remains fixed and is not retrained.\n- Physical units, URDF and FK semantics remain unresolved; results are command trajectories.\n""", encoding="utf-8")
    (ROOT / "EVIDENCE_CORRECTION_ADDENDUM.zh-CN.md").write_text("""# 证据修正附录\n\n上一轮 router/prototype 诊断中的两个判断需要保留修正：旧 `ARI=1` 来自没有在 bootstrap 中重新拟合 centers 的实现，不能再表述为稳定两模式；base OOF 完整也不等于最终 selector OOF 完整，且 fold whitening 复用了 all-train 统计量，因此不能声称严格隔离的最终 OOF。\n\n这些修正不撤销已经生成并测量的 development predictions，也不阻断本轮新的直接监督模型。它们只限制历史结果的证据范围。本轮不使用 selector、prototype、oracle 或预测堆叠；每个模型直接以 train future targets 监督训练。\n\n旧 `+1% FD`、`80% retention` 等是项目操作阈值，不是物理定律。没有证据的物理单位、session 泛化和条件多模态仍记为 unknown。\n""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": len(rows), "parameter_count": parameter_count(model), "stage0_jobs": len(jobs), "seed": SEED}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
